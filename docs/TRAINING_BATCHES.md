# Training batch plan

Recorded on 2026-10-01. The target for the continued-training comparison is
**128 examples per optimizer update**. Use the same frozen data splits for
student CoT-SFT, teacher CoT-SFT, recurrent distillation and the ablations.

## Two recorded single-A100 options

| Use | GPU microbatch | Accumulation steps | GPUs | Effective batch |
| --- | ---: | ---: | ---: | ---: |
| Initial configuration | 2 | 64 | 1 | 128 |
| Throughput candidate | 8 | 16 | 1 | 128 |

The 2/64 option keeps the current physical microbatch and increases the number
of batches accumulated before each optimizer update. The 8/16 option increases
the physical batch to investigate better GPU utilization. Neither has a measured
A100 memory trace or throughput result. Profile the complete student/teacher
workload at stage six, with representative long examples, before selecting 8/16.

The shared pilot defaults in [common.yaml](../args/research/common.yaml) are still
2/16: batch 32 on one GPU and 128 on four. Request the recorded options explicitly:

~~~bash
NPROC_PER_NODE=1 RUN_NAME=gsm-distill-b128-mb2-seed0 \
bash launch_distillation.sh \
  --set batch_size_training=2 --set gradient_accumulation_steps=64

NPROC_PER_NODE=1 RUN_NAME=gsm-distill-b128-mb8-seed0 \
bash launch_distillation.sh \
  --set batch_size_training=8 --set gradient_accumulation_steps=16
~~~

Use a distinct run name for each configuration. The run manifest records the
resolved microbatch, accumulation, world size and effective batch.

## Accumulation and GPU count

~~~text
effective batch = microbatch per GPU × accumulation steps × number of GPUs
~~~

| GPUs | Microbatch per GPU | Accumulation | Effective batch |
| ---: | ---: | ---: | ---: |
| 1 | 2 | 64 | 128 |
| 1 | 8 | 16 | 128 |
| 4 | 2 | 16 | 128 |
| 4 | 8 | 4 | 128 |

At fixed microbatch 2, moving from accumulation 16 to 64 leaves approximately
the same forward/backward work per epoch and makes optimizer updates four times
less frequent. Accumulation reuses gradient buffers; it does not retain the
activation graphs of all 64 microbatches.

The current objective averages globally normalized microbatch means. CE uses
each microbatch's own valid-token count; accumulation does not form one
token-weighted mean over all 128 examples. With unequal sequence lengths, 2/64
and 8/16 can therefore weight CE terms differently. Keep the chosen split
consistent across comparisons and document any change. A partial final
accumulation window uses its actual number of microbatches and can contain fewer
than 128 examples. FSDP synchronizes each microbatch.

## Bounded profiling run

This command exercises all curriculum stages through stage six on the first
128 training and eight validation examples. It uses the full 150-position
latent budget and writes checkpoints, but the restricted population is for
profiling and pipeline checks:

~~~bash
NPROC_PER_NODE=1 RUN_NAME=gsm-distill-profile-b128-mb2 \
bash launch_distillation.sh \
  --set batch_size_training=2 --set gradient_accumulation_steps=64 \
  --set train_max_examples=128 --set val_max_examples=8 --set num_epochs=7
~~~

For the 8/16 candidate, use a new run name and replace the two batch overrides
with batch_size_training=8 and gradient_accumulation_steps=16. This bounded
subset does not establish capacity on long examples from the full dataset.
Measure actual peak allocation/reservation, stage-six examples per second,
optimizer-step duration and validation peaks. See
[A100 planning](A100_PLANNING.md) for the current memory/runtime assumptions.

A launcher configuration check can be requested with DRY_RUN=1. It starts no
training and does not establish memory fit.

For Slurm, submit from the repository root in the prepared CUDA environment:

~~~bash
NPROC_PER_NODE=1 RUN_NAME=gsm-distill-b128-mb2-seed0 \
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  cluster/train_distillation.sbatch \
  --set batch_size_training=2 --set gradient_accumulation_steps=64
~~~

The template requests one GPU, 128 GB host RAM and a 24-hour walltime. Choose the
GPU type and a walltime that match the cluster and planned job. A ten-epoch full
run may exceed 24 hours; resume is epoch-level. Keep the same run name,
microbatch and accumulation settings on resume.

## Paper reference settings

| Source | Model | GPU microbatch | Accumulation | Effective training batch |
| --- | --- | ---: | ---: | ---: |
| [CODI Llama training script](https://github.com/zhenyi4/codi/blob/main/scripts/train_llama1b_gsm8k-aug.sh) | Llama 3.2 1B | 32 | 4 | 128 on one GPU |
| [CODI GPT-2 training script](https://github.com/zhenyi4/codi/blob/main/scripts/train_gpt2_gsm8k-aug.sh) | GPT-2 | 64 | 2 | 128 on one GPU |
| [LOTUS, Appendix D](https://arxiv.org/html/2606.31779v1) | Reported training experiments | Varies | Varies | 128 overall |

The [original CODI paper](https://arxiv.org/html/2502.21074v1) reports a single
A100 80GB, BF16 LoRA with rank 128/alpha 32, and one shared backbone for teacher
and student tasks. Its released Llama preprocessing uses a 200-token filter for
question+rationale+answer; see [train.py](https://github.com/zhenyi4/codi/blob/main/train.py).
Our workload has a separate frozen 3B teacher, a fully trainable 1B student and
projections, and a larger recurrent latent region. CODI's physical batch size
is therefore not a measured capacity result for this implementation.

Matching batch 128 aligns the number of examples per update. The pilot remains
a distinct experiment: objective weights, optimizer schedule, architecture,
data filtering and training duration differ from the original papers.
