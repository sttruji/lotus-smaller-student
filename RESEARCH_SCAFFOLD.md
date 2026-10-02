# Smaller recurrent student: research and compute-node guide

Repository: [sttruji/lotus-smaller-student](https://github.com/sttruji/lotus-smaller-student).
Current branch: `main`. Documentation updated 2026-10-02.
Upstream: [yingfan-bot/lotus](https://github.com/yingfan-bot/lotus), base commit
`eb77e2f7909c5006f58ff0ad7cd6629b942caa9e`.

The experiment distills an explicit 3B CoT reasoner into a smaller 1B
student whose transformer weights are reused across latent recurrence passes.
This is full fine-tuning of pretrained models, not training a language model from
scratch. It currently reuses the whole smaller transformer at each recurrence;
it does not yet introduce a separate shallow recurrent core or layer pruning.

Companion guides: [training batches](docs/TRAINING_BATCHES.md),
[A100 planning](docs/A100_PLANNING.md), [evaluation](docs/EVALUATION.md),
[audit status](docs/AUDIT_STATUS.md) and [upstream LOTUS](docs/UPSTREAM_LOTUS.md).

## What is implemented

- Independently loaded, frozen teacher architecture, with tokenizer-ID validation
  and fatal errors for missing teacher checkpoints. The teacher is kept outside
  the student's registered modules and replicated in full on each GPU.
  FSDP shards the trainable student/projections on multi-GPU jobs.
- Relative-depth block mapping and separate student-to-teacher width projections;
  embeddings are excluded. Mapping uses deterministic half-up rounding of
  `i * teacher_depth / student_depth`, with one projection per pair. The alternate
  config aligns only the final block.
- Normalized L1 distillation at the position **before the first answer output
  token**. LOTUS supervises the `###` delimiter as part of the answer, so the
  boundary precedes that delimiter. The teacher sees the complete restored CoT
  and receives no answer tokens. Remaining visible rationale during the curriculum
  is included when finding both boundaries.
- Answer-only CE plus a separate explicit student CoT-and-answer CE forward;
  the latter masks question tokens. Each loss has its own sum/count for distributed
  normalization. Initial weights are `1.0` for KD and `0.1` for auxiliary CoT,
  which are pilot choices to tune, not established optimal values.
- At the final stage, all explicit rationale is removed from the student's main
  input, including examples with more reasoning steps than recurrence passes.
  Intermediate per-step LOTUS supervision is disabled in this experiment.
- BF16/FSDP training with configurable microbatch size, gradient accumulation,
  checkpointed activations, CLI overrides, optional W&B logging, and a Slurm
  template for one node.
- Training checkpoints retain projections. Export strips them and keeps only the
  smaller student backbone and tokenizer, with separate LOTUS inference settings.
- Run manifests record resolved configuration, seed, data hashes, model configs,
  available model revision metadata, parameter counts, layer pairs, software
  versions, GPU and code commit. `--save_metrics` adds structured evaluation output.
- Resume-safe best-checkpoint selection, with post-validation periodic-state
  updates and recovery from existing eligible final metadata.
- Strict evaluator backbone loading, explicit error/termination accounting,
  synchronized full-generation timing and local export settings integration.

Two upstream training issues were fixed: recurrent embedding updates are now
out of place so earlier activations remain valid for backward; checkpointed
decoder calls receive no mutable KV cache. Cache-free suffix recomputation also
replays the last loop input, preserving the cached student objective and gradients.
`checkpoint_final` selection for the research experiment starts at the fully
latent stage so an earlier CoT-assisted checkpoint cannot masquerade as the final
latent student. Periodic checkpoints still cover every epoch.

For the default 16-block student and 28-block teacher, relative-depth pairs are:

```text
Student:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
Teacher:  2  4  5  7  9 11 12 14 16 18 19 21 23 25 26 28
```

Each pair has a separate `Linear(2048, 3072, bias=False)` projection, for
100,663,296 training-only parameters. Distillation compares final suffix states
at the pre-answer boundary; the pairs do not correspond to separate recurrence
passes. The final-layer ablation uses only `(16, 28)`. See
[distillation.py](scripts/distillation.py) for mapping, reconstruction and loss.

## Experiment configs

| Config under `args/research/` | Purpose |
| --- | --- |
| `distill_3b_to_looped_1b.yaml` | Main frozen 3B → looped 1B experiment |
| `distill_final_layer.yaml` | Same experiment, final-layer alignment only |
| `looped_1b_no_distillation.yaml` | Remove KD, retaining the same auxiliary CoT loss |
| `cot_sft_student_1b.yaml` | Explicit CoT-SFT baseline at student size |
| `cot_sft_teacher_3b.yaml` | Explicit CoT-SFT baseline at teacher size |

The CoT baselines fine-tune on rationale plus answer and generate written
reasoning at inference. They share data paths, seed and ten-epoch pilot defaults.
This does not equate their compute budgets with recurrent distillation; log total
training time separately. Original `args/gsm8k_*` configs remain available for
published-method comparisons.

The main config starts from LOTUS's published `yingfanbot/gsm-cot-llama1b` and
`yingfanbot/gsm-cot-llama3b` checkpoints for a quick pilot. For the controlled
comparison, first train the two CoT-SFT configs on the same data, then use **those
exact checkpoints** for the student initialization and frozen teacher. Treat
published-initialization pilots as a separate comparison because their upstream
training recipes differ from these shared pilot settings.

The unlooped smaller CODI control, adaptive halting, truncated recurrent
backpropagation, shallow-core architectures, multi-node jobs and batched latency
benchmarks are not implemented in this scaffold.

## Environment and data

Clone the current repository on the compute node:

```bash
git clone --branch main https://github.com/sttruji/lotus-smaller-student.git
cd lotus-smaller-student
```

The original handoff bundle contains the scaffold commit and upstream history.
The published `main` branch includes the later checkpoint and evaluation fixes.

Use the upstream CUDA/PyTorch environment (`nvcr.io/nvidia/pytorch:25.03-py3`,
or `environment.yml`). For Conda:

```bash
conda env create -f environment.yml
conda activate lotus
```

For the prepared container, install runtime dependencies with
`pip install -r requirements.txt`. The requirements file excludes PyTorch;
the chosen environment must provide it. Prepare data and optional CPU checks:

```bash
# Tests additionally use pytest.
pip install pytest==8.3.5
bash preprocessing/gsm_icot.bash
```

The preprocessing script pins the source augmented GSM8K data to
`e06a32ee5e4cd117171daeb4755d2a97ece62761`. It creates
`data/gsm_{train,valid,test}.json`, which are not distributed in this branch.
Make the selected checkpoints accessible through the node's Hugging Face cache
or authentication; the launcher does not request credentials. Requirements pin
Transformers 4.46.2 and Datasets 3.1.0; some other dependencies remain unpinned.

Check data and model/tokenizer configurations before training:

```bash
python scripts/preflight_distillation.py args/research/distill_3b_to_looped_1b.yaml --check-models
```

This reads configs/tokenizers but does not load full model weights. For config
validation without datasets or network access, add `--config-only` and omit
`--check-models`.

## Download-free correctness check

```bash
python scripts/smoke_distillation.py --steps 3
python -m pytest -q tests
```

The smoke test trains tiny random Llamas (student width/depth 16/2, teacher 24/3),
checks projection gradients and a frozen teacher, and verifies a checkpoint
round trip. It writes to `outputs/cpu-smoke/`. It is a correctness check, not a
GSM8K result or a GPU memory benchmark.

## Launch continued training

From the repository root on an allocated CUDA node, the recorded initial
single-GPU option is **microbatch 2, accumulation 64, effective batch 128**:

```bash
NPROC_PER_NODE=1 RUN_NAME=gsm-distill-b128-mb2-seed0 bash launch_distillation.sh \
  --set batch_size_training=2 --set gradient_accumulation_steps=64
```

The second recorded option is **microbatch 8, accumulation 16**:

```bash
NPROC_PER_NODE=1 RUN_NAME=gsm-distill-b128-mb8-seed0 bash launch_distillation.sh \
  --set batch_size_training=8 --set gradient_accumulation_steps=16
```

Profile 8/16 at the fully latent stage before a full run. Both options need
actual A100 memory and throughput measurements. The shared YAML defaults remain
BF16, microbatch 2 and accumulation 16: batch 32 on one GPU or 128 on four.
Effective batch is microbatch per GPU × accumulation × GPU count; see
[training batches](docs/TRAINING_BATCHES.md) for the recorded plan and paper settings.
Accumulation averages globally normalized microbatch means, including a partial
final window; it is
not a token-weighted mean over the entire accumulation window. FSDP synchronizes
each microbatch to avoid accumulating full unsharded gradients. Different
microbatch splits can weight CE differently when valid-token counts differ.

The ten-epoch pilot advances one stage per epoch to six recurrences and 150 latent
positions. LOTUS's `n_looped_iters=6` means **six additional passes after an initial
latent pass**, seven latent-region passes total. With checkpointing, the suffix
forward also recomputes the cached prefix. The same learned smaller weights are
reused; increasing passes does not increase unique backbone parameters.

To change the seed while retaining the initial batch-128 option:

```bash
NPROC_PER_NODE=1 RUN_NAME=gsm-distill-b128-mb2-seed2 bash launch_distillation.sh \
  --set seed=2 --set batch_size_training=2 --set gradient_accumulation_steps=64
```

For a launcher/config check that starts no training:

```bash
DRY_RUN=1 NPROC_PER_NODE=1 bash launch_distillation.sh \
  --set batch_size_training=2 --set gradient_accumulation_steps=64
```

For Slurm, activate the prepared environment and submit from the repository root;
replace account/partition values with the cluster's actual choices:

```bash
NPROC_PER_NODE=1 RUN_NAME=gsm-distill-b128-mb2-seed0 \
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  cluster/train_distillation.sbatch \
  --set batch_size_training=2 --set gradient_accumulation_steps=64
# Four GPUs, retaining effective batch 128 with the default microbatch/accumulation:
NPROC_PER_NODE=4 sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  --gpus-per-node=4 cluster/train_distillation.sbatch
```

The template requests 128 GB host memory, one GPU and a 24-hour walltime by
default. Adjust the walltime for a full run or plan epoch-level resume; the
current runtime estimate is multiple days. GPU memory and throughput for the
real 3B/1B models must be measured on the chosen node; no CUDA
training or distributed FSDP execution was available in the development workspace.
Run a short node pilot before committing to a multi-seed campaign.

For a small **real-model node smoke run** that reaches a fully latent stage,
uses 64 training examples and eight validation examples, and saves checkpoints:

```bash
RUN_NAME=gsm-distill-node-smoke NPROC_PER_NODE=1 bash launch_distillation.sh \
  --set train_max_examples=64 --set val_max_examples=8 \
  --set num_epochs=2 --set max_latent_stage=1 --set c_thought=2 --set max_c_thought=2
```

This changes the latent budget for the smoke run and is not an accuracy result
for the default six-recurrence experiment. Example limits are recorded in the
manifest; they default to the full datasets for research runs.
For a bounded run that reaches the full 150-position stage-six workload, see
[the profiling command](docs/TRAINING_BATCHES.md#bounded-profiling-run).

## Train the controlled baselines, then distill

```bash
CONFIG=args/research/cot_sft_student_1b.yaml NPROC_PER_NODE=1 bash launch_distillation.sh \
  --set batch_size_training=2 --set gradient_accumulation_steps=64
CONFIG=args/research/cot_sft_teacher_3b.yaml NPROC_PER_NODE=1 bash launch_distillation.sh \
  --set batch_size_training=2 --set gradient_accumulation_steps=64
RUN_NAME=gsm-distill-controlled-seed0 NPROC_PER_NODE=1 bash launch_distillation.sh \
  --set batch_size_training=2 --set gradient_accumulation_steps=64 \
  --set load_model_path=./outputs/gsm-cot-sft-student-1b-seed0/checkpoint_final \
  --set teacher_model_path=./outputs/gsm-cot-sft-teacher-3b-seed0/checkpoint_final
```

The local teacher is loaded strictly into its own `teacher_model_id` architecture.
CoT weights, teacher weights and projection shapes are not silently adapted
across incompatible models. Pin `model_revision`, `student_revision`,
`teacher_revision` and `teacher_tokenizer_revision` to Hub commits when using
remote checkpoints for a finalized study.

The final-layer and no-KD comparisons use `CONFIG` in the same way, with the same
batch overrides and controlled student initialization. Supply the controlled
teacher for the final-layer KD comparison. The no-KD config loads no teacher.
Different model sizes may need different physical microbatches; preserve the
effective batch and document the split. Record epoch count, learning rate,
objective settings and total training time for every comparison.

## Resume and export

Periodic directories contain model weights, optimizer and training state. Resume
with the same model, alignment and objective configuration:

```bash
RUN_NAME=gsm-distill-controlled-seed0 NPROC_PER_NODE=1 bash launch_distillation.sh \
  --set batch_size_training=2 --set gradient_accumulation_steps=64 \
  --set resume=7 \
  --set load_model_path=./outputs/gsm-distill-controlled-seed0/checkpoint_7 \
  --set teacher_model_path=./outputs/gsm-cot-sft-teacher-3b-seed0/checkpoint_final
```

Upstream retention keeps only the latest periodic checkpoint. Substitute the
checkpoint that actually exists; resuming restores epoch-level progress and
optimizer/scheduler state, not an interrupted mid-epoch dataloader position.
Periodic checkpoint selection scores are updated after validation. Resume also
reconciles the score with the existing final checkpoint metadata, protecting a
better final model when resuming an older checkpoint written before that update.
For this experiment, only fully latent final checkpoint metadata is eligible.
Retain the original batch split and GPU count on resume; the example assumes
the 2/64 single-GPU controlled run above.

Export the best fully latent student:

```bash
python scripts/export_student.py \
  --checkpoint ./outputs/gsm-distill-controlled-seed0/checkpoint_final \
  --manifest ./outputs/gsm-distill-controlled-seed0/run_manifest.json \
  --output-dir ./outputs/student-inference
```

The export is an HF-format backbone plus tokenizer and `lotus_config.json`.
**Use the LOTUS wrapper for recurrence**; plain HF `generate()` on these weights
does not execute latent loops. The export needs neither teacher nor projections.

## Accuracy and resource measurements

See [the evaluation guide](docs/EVALUATION.md) for the current CLI, metrics
schema, split selection, failure policy and resource accounting.

Evaluate the exported student and both explicit-CoT baselines on the same frozen
test JSON, with greedy decoding and the same output-token cap:

```bash
python scripts/eval.py --model_id ./outputs/student-inference --datasets gsm8k \
  --max_new_tokens 512 --save_preds ./outputs/student-predictions.json \
  --save_metrics ./outputs/student-metrics.json
python scripts/eval.py --model_id meta-llama/Llama-3.2-1B-Instruct \
  --checkpoint ./outputs/gsm-cot-sft-student-1b-seed0/checkpoint_final \
  --datasets gsm8k --cot --max_new_tokens 512 --save_metrics ./outputs/cot-1b-metrics.json
python scripts/eval.py --model_id meta-llama/Llama-3.2-3B-Instruct \
  --checkpoint ./outputs/gsm-cot-sft-teacher-3b-seed0/checkpoint_final \
  --datasets gsm8k --cot --max_new_tokens 512 --save_metrics ./outputs/cot-3b-metrics.json
```

Local HF exports supply their mode, loop count, block width and latent injection
mode through `lotus_config.json`. Explicit CLI settings override those defaults
and are recorded with the exported settings. Training checkpoint loads require a
complete backbone with `strict=True`; known projection/auxiliary heads are omitted
and the shared embedding alias is checked. Incomplete HF exports are rejected too.
Missing latent embeddings are rejected unless `--allow_untrained_latent_tokens`
explicitly requests an untrained baseline; that initialization is recorded.

Metrics use schema version 2. Each attempted example has a prediction or error
record, generated text/token IDs, EOS/token-limit status, and timings. Generation
stops at the first EOS, and generated-token counts include EOS for both model
paths. `truncated` means the token cap was reached without EOS; EOS exactly at the
cap is counted as normal termination. Accuracy uses exact decimal comparison
after numeric answer extraction. A trailing explanation after `### 42` no longer
makes that answer incorrect; the generated text is retained for scoring audits.

Errors stop the run by default after saving any requested reports. Use
`--continue_on_error` to finish the population and record all failures. In either
mode, any errors produce a nonzero exit status and `valid_for_accuracy=false`.
The primary denominator is the entire dataset; failed attempts count as incorrect
when the population is completed. Incomplete and empty datasets have null
accuracy. Successful-only accuracy is a labeled diagnostic. Token and timing
averages use successful examples, with failed and unattempted counts reported
separately. An ordered question/answer SHA-256 and the actual dataset split are
saved. SVAMP defaults to the inherited train+test population; use
`--svamp_split test` for its test split alone.

Use `inference_time` for comparisons: it times the synchronized full generation
call, including prefix, loops, suffix forward and decoding, with tokenization,
text decoding and scoring excluded. Phase totals remain diagnostic; the
`prefill_other_time` bucket includes the previously omitted LOTUS suffix forward
and forward bookkeeping, and `unattributed_inference_time` reports residual time
outside those phase timers. Delimiter detection is a token-ID heuristic. Peak GPU
allocation/reservation and synchronization target the selected device; memory is
reported in GiB and includes the resident model. CPU runs report null GPU memory.

There is no warmup exclusion, latency distribution, batched throughput test or
FLOP accounting yet. Check truncation rates and revisit the cap if full CoT traces
regularly hit it. Latent-position and loop-count sweeps should be reported with
their actual settings; training only one setting does not establish robustness
to other loop counts. CUDA memory/latency and real-model accuracy need a GPU pilot.

## Validation status

The last recorded suite run on 2026-10-01 passed **54 CPU tests** against
implementation commit `0e556f2`, with Hub access disabled. The checkpoint fix was
published as `7405492`; evaluation accounting was published as `0e556f2`.

The CPU checks cover unequal widths/depths, complete CoT reconstruction with
padding and curriculum rationale, absence of answer-token leakage, student and
projection gradients, a frozen teacher, checkpoint serialization, strict
independent teacher loading, inference export/reload, fully latent examples with
more gold steps than loops, stage zero, generation without answer labels,
checkpointed/cached loss-and-gradient agreement, and partial accumulation windows.
They also cover resumed checkpoint selection, atomic selection-state updates,
strict evaluator loading for raw/HF weights, evaluator failure artifacts,
exact numeric scoring, EOS/truncation counts, saved architecture defaults,
stage-zero timing and complete CPU evaluator runs for CoT and LOTUS.
The development smoke run performed three optimizer updates and restored its
checkpoint. CUDA memory fit, large-checkpoint loading, Slurm and multi-GPU FSDP
remain compute-node validation tasks.

On multi-GPU training jobs, validation generation uses a separate unwrapped
student copy built from collectively gathered full weights. This avoids calling
generation with sharded embedding/head parameters and removes forward collectives
from autoregressive decoding. It temporarily adds a full student copy per GPU
and corresponding host memory; allow for that when measuring training memory.
Standalone resource evaluation loads only the exported student.
Outstanding implementation and validation work is tracked in
[audit status](docs/AUDIT_STATUS.md).
