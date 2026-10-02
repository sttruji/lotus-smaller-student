# Smaller recurrent student: 3B teacher → looped 1B student

Research fork of [LOTUS](https://github.com/yingfan-bot/lotus) for distilling an
explicit 3B chain-of-thought teacher into a smaller 1B student that reuses its
transformer weights across latent reasoning passes.

The student is fully fine-tuned with answer CE, hidden-state distillation and an
auxiliary explicit CoT objective. The teacher is frozen. At the final curriculum
stage, the main student input contains the question and latent region with all
written rationale removed.

The current code is on **main** in
[sttruji/lotus-smaller-student](https://github.com/sttruji/lotus-smaller-student).
The original handoff bundle and scaffold branch predate the checkpoint and
evaluation fixes.

## Documentation

| Guide | Contents |
| --- | --- |
| [Research scaffold](RESEARCH_SCAFFOLD.md) | Architecture, objectives, environment, controlled baselines, training, resume and export |
| [Training batches](docs/TRAINING_BATCHES.md) | Recorded 2/64 and 8/16 options, effective batch 128, paper settings and accumulation behavior |
| [A100 planning](docs/A100_PLANNING.md) | Data counts, runtime assumptions, training memory estimates and GPU profiling work |
| [Evaluation](docs/EVALUATION.md) | Strict loading, generation/scoring, failures, truncation, timing and metrics |
| [Audit status](docs/AUDIT_STATUS.md) | Completed fixes, existing CPU evidence and remaining validation |
| [Upstream LOTUS](docs/UPSTREAM_LOTUS.md) | Original method configs, launch commands, published models and citations |

## Current experiment

| Setting | Main research configuration |
| --- | --- |
| Student initialization | yingfanbot/gsm-cot-llama1b |
| Frozen teacher | yingfanbot/gsm-cot-llama3b |
| Student / teacher blocks | 16 / 28 |
| Student / teacher hidden width | 2,048 / 3,072 |
| Alignment | Relative-depth mapping with one bias-free student-to-teacher projection per pair |
| Training-only projections | 100,663,296 parameters |
| Objective | Answer CE + normalized L1 KD at weight 1.0 + explicit student CoT CE at weight 0.1 |
| Curriculum | Ten epochs; stages 0, 1, 2, 3, 4, 5, 6, 6, 6, 6 |
| Saturated latent region | 150 positions; initial pass plus six additional passes |
| Training stack | BF16, activation checkpointing, PyTorch AdamW and native FSDP |
| Export | Student backbone, tokenizer and lotus_config.json; recurrence uses the LOTUS wrapper |

The published initialization is convenient for a pilot. A controlled study first
trains the 1B and 3B CoT baselines on the same prepared splits and then uses those
exact checkpoints for all distillation comparisons. See
[controlled training](RESEARCH_SCAFFOLD.md#train-the-controlled-baselines-then-distill).

## Setup on a CUDA node

Clone the current repository:

~~~bash
git clone https://github.com/sttruji/lotus-smaller-student.git
cd lotus-smaller-student
conda env create -f environment.yml
conda activate lotus
bash preprocessing/gsm_icot.bash
~~~

The environment specifies Python 3.12 and PyTorch 2.7.0 with CUDA 12.8.
[requirements.txt](requirements.txt) provides the remaining runtime dependencies;
it does not install PyTorch. An alternative is the prepared
nvcr.io/nvidia/pytorch:25.03-py3 container plus those requirements. See the
[environment guide](RESEARCH_SCAFFOLD.md#environment-and-data).

Preprocessing creates data/gsm_train.json, data/gsm_valid.json and
data/gsm_test.json from the pinned augmented GSM8K source. Data, checkpoints and
run outputs are git-ignored. Authenticate with huggingface-cli login if the
selected models require Hub access. The research launcher disables W&B by default.

## Recorded single-GPU batch options

Both options target **128 examples per optimizer update**:

| Use | GPU microbatch | Accumulation steps | Status |
| --- | ---: | ---: | --- |
| Initial configuration | 2 | 64 | Start here; actual A100 memory is unmeasured |
| Throughput candidate | 8 | 16 | Profile at the fully latent stage before a full run |

The shared YAML pilot defaults remain microbatch 2 and accumulation 16, which
produce effective batch 32 on one GPU. The commands below explicitly request 128.

~~~bash
# Initial 2/64 option, using the published CoT initialization.
NPROC_PER_NODE=1 RUN_NAME=gsm-distill-b128-mb2-seed0 \
bash launch_distillation.sh \
  --set batch_size_training=2 --set gradient_accumulation_steps=64

# Candidate 8/16 option, after measuring fully latent memory.
NPROC_PER_NODE=1 RUN_NAME=gsm-distill-b128-mb8-seed0 \
bash launch_distillation.sh \
  --set batch_size_training=8 --set gradient_accumulation_steps=16
~~~

Effective batch is microbatch per GPU × accumulation × GPU count. When using
more GPUs, adjust accumulation to retain 128. For a bounded GPU pilot and Slurm
commands, see [training batches](docs/TRAINING_BATCHES.md).

## Export and evaluate

For a controlled run named gsm-distill-controlled-seed0:

~~~bash
python scripts/export_student.py \
  --checkpoint ./outputs/gsm-distill-controlled-seed0/checkpoint_final \
  --manifest ./outputs/gsm-distill-controlled-seed0/run_manifest.json \
  --output-dir ./outputs/student-inference

python scripts/eval.py \
  --model_id ./outputs/student-inference --datasets gsm8k \
  --max_new_tokens 512 \
  --save_preds ./outputs/student-predictions.json \
  --save_metrics ./outputs/student-metrics.json
~~~

Local exports supply inference defaults through lotus_config.json. Raw
checkpoints require the matching backbone and loop settings. The evaluator
rejects incomplete weights, records every attempted example, reports truncation
and uses a synchronized full-generation timer for inference comparisons.
Generation errors produce a nonzero exit status. Read the
[evaluation guide](docs/EVALUATION.md) for denominators, split selection and timing scope.

## Validation status

As of 2026-10-01, the existing suite passed **54 CPU tests** with tiny local
models, including cross-size gradients, frozen-teacher behavior, checkpoint
selection, export/reload and complete evaluator runs. The CPU smoke completed
three optimizer updates.

Real-model accuracy, A100 memory/throughput, CUDA execution, Slurm and multi-GPU
FSDP still require compute-node validation. The runtime and VRAM figures in
[A100 planning](docs/A100_PLANNING.md) are estimates. The standalone evaluator
currently runs one example at a time with no warmup exclusion or batched
throughput benchmark.

## Repository layout

~~~text
args/research/           Controlled experiment and ablation configs
args/gsm8k_*.yaml        Retained upstream method configs
scripts/run.py          Custom training and validation loop
scripts/lotus.py        Recurrent student and generation
scripts/distillation.py Teacher loading and boundary alignment
scripts/selection_state.py  Resume-safe checkpoint selection
scripts/export_student.py  Student-only HF export
scripts/eval.py         Standalone evaluator
scripts/eval_accounting.py  Scoring and evaluation accounting
scripts/preflight_distillation.py  Configuration/data/model checks
scripts/run_metadata.py Run provenance manifests
launch_distillation.sh  Research launcher; CLI overrides and one GPU by default
launch_train.sh         Original method launcher; four GPUs by default
cluster/                Slurm template
preprocessing/          Dataset preparation
tests/                  CPU correctness coverage
docs/                   Batching, evaluation, resource planning and audit guides
~~~

## Attribution and license

Built on [LOTUS](https://arxiv.org/abs/2606.31779), with
[CODI-style hidden-state distillation](https://arxiv.org/abs/2502.21074).
The original training, evaluation and preprocessing code is adapted from
[Coconut](https://github.com/facebookresearch/coconut). Published upstream models
and results are described in [the upstream guide](docs/UPSTREAM_LOTUS.md).

Released under the [MIT License](LICENSE).
