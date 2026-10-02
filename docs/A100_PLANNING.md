# A100 training planning

Updated 2026-10-02. This guide covers one ten-epoch frozen-3B-teacher → recurrent
1B-student run, followed by export and evaluation. It starts from the two
published CoT checkpoints in the main research config. Controlled CoT baseline
training, ablations and extra seeds are additional work.

**These are workload estimates. No A100 training or memory trace has been run.**
The recorded single-GPU options are [2/64 and 8/16](TRAINING_BATCHES.md), both
targeting effective batch 128. The calculations below began with physical
microbatch 2; estimates for 8/16 require new padding and throughput measurements.

## Data and schedule

Preprocessing pins the augmented GSM8K source to
[e06a32ee5e4cd117171daeb4755d2a97ece62761](https://github.com/da03/Internalize_CoT_Step_by_Step/tree/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k).

| Split | Source examples |
| --- | ---: |
| Training | 385,620 |
| Validation | 500 |
| Test | 1,319 |

The counted raw training file is 87,805,358 bytes, with SHA-256
0a3909a9e7d8d2f7ad6b8c7b5608aa744988d835f9bf874d4cf06ca77df6bf8c.
Training manifests separately hash the actual prepared JSON files used.

The ten epochs use stages 0, 1, 2, 3, 4, 5, 6, 6, 6, 6. Stage six has 150 latent
positions and an initial latent pass plus six additional passes through the
same student backbone.

Assuming all source training examples remain in each epoch, a complete final
accumulation window is flushed at every epoch boundary:

| Single-GPU configuration | Microbatches over ten epochs | Optimizer updates over ten epochs |
| --- | ---: | ---: |
| YAML pilot default, 2/16 | 1,928,100 | 120,510 |
| Recorded initial option, 2/64 | 1,928,100 | 30,130 |
| Recorded throughput candidate, 8/16 | 482,030 | 30,130 |

The final update in each epoch has fewer than 128 examples. The 8/16 option
also has a smaller final physical batch. Example limits and any future filtering
change these counts.

## Sampled token workload

The initial estimate sampled 20,000 examples with seed 20261001 and the public
student tokenizer, using the training question/step/answer format. Pair padding
was modeled for physical microbatch 2.

| Sequence | Mean | 95th percentile | 99th percentile |
| --- | ---: | ---: | ---: |
| Individual question | 45 tokens | 70 | 88 |
| Individual full rationale | 20 tokens | 40 | 55 |
| Individual explicit question+rationale+answer | 70 tokens | 109 | 134 |
| Stage-six recurrent input, padded in pairs | 209 tokens | 234 | 252 |

Stage-six student body token visits average approximately 1,715 per example,
including recurrent/main suffix work and the auxiliary explicit student forward.
Checkpointing recomputes prefix work during recurrent training. Increasing the
physical batch can increase padding while improving GPU utilization.

The archived [initial estimate snapshot](evidence/a100-estimate-20261001.json)
contains the sample statistics, parameter counts and compute arithmetic.
Its optimizer-update field describes the historical 2/16 setting; the table
above records the new batch-128 options.

## Runtime assumptions

The dense matrix compute proxy is approximately **38.2 exaFLOPs** for ten
epochs at microbatch 2. It includes the trainable body, backward, checkpoint
recomputation, repeated student passes, vocabulary heads, the frozen teacher
body and pair padding.

It excludes quadratic attention work, normalization, CE bandwidth, launch
overhead, optimizer operations, dataset transforms, logging, downloads,
validation generation, checkpoint I/O and final evaluation. Increasing
accumulation from 16 to 64 preserves this main forward/backward workload while
reducing optimizer updates. The proxy cannot predict a fourfold runtime speedup.

The [NVIDIA A100 datasheet](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf)
lists 312 TFLOP/s peak dense BF16 throughput. Sparse peak throughput is not used
in this calculation.

| Assumed sustained dense matrix rate | Compute proxy time |
| --- | ---: |
| 312 TFLOP/s, hardware peak | 34 hours |
| 150 TFLOP/s, optimistic assumption | 71 hours |
| 100 TFLOP/s | 106 hours |
| 75 TFLOP/s | 142 hours |
| 50 TFLOP/s | 212 hours |

The initial planning allowance is **3–4 days in an optimistic setup** and
**5–9 days for budgeting the current implementation**. The 34-hour peak-rate
calculation is neither a measured end-to-end time nor an achievable forecast.
Actual runtime can exceed the allowance. The 8/16 candidate needs measured
examples per second and its own padding estimate.

Use a pilot that reaches stage six; early-stage throughput understates the
saturated recurrence cost. Validation/test output caps are 512 in the comparison
commands, so models that generate long traces can add appreciable time.

## Persistent GPU allocations

Meta-device parameter arithmetic for the configured architectures and three
added student tokens:

| Component | Parameters |
| --- | ---: |
| Student backbone | 1,235,820,544 |
| Trainable alignment projections | 100,663,296 |
| Frozen teacher | 3,212,749,824 |

The implementation casts trainable student/projection weights to BF16.
PyTorch AdamW moment tensors inherit the parameter dtype; the configured path
has no separate FP32 master copy. On one GPU, FSDP does not spread allocations
across other devices.

| Persistent allocation | Estimated GiB |
| --- | ---: |
| Student and projection weights, BF16 | 2.49 |
| Student and projection gradients, BF16 | 2.49 |
| Two Adam moments, BF16 | 4.98 |
| Frozen teacher weights, BF16 | 5.98 |
| **Persistent subtotal** | **15.94** |

The teacher has no gradients or optimizer state and runs without a KV cache.
Student activation checkpointing disables persistent training KV caching, so
prefixes are recomputed. Training still needs activations, vocabulary logits,
temporary tensors, optimizer workspaces, FSDP buffers, CUDA context and
allocator reservation.

For **physical microbatch 2**, the initial planning allowance is **22–30 GiB**
of training VRAM. That is an engineering estimate, not a measured peak or a
guaranteed capacity. Accumulation 64 does not store 64 activation graphs.
The 8/16 peak is unknown and should not be extrapolated by multiplying the whole
22–30 GiB allowance by four: persistent weights/state are shared, while variable
allocations depend on the batch and padding.

On multiple GPUs, the frozen teacher is replicated. Validation temporarily
constructs a full unwrapped student copy on each GPU, adding GPU and host
memory. Standalone exported-student evaluation omits the teacher and projections.
Changing optimizer-state precision changes the persistent subtotal.

## Profiling and host resources

Start with the [bounded stage-six 2/64 run](TRAINING_BATCHES.md#bounded-profiling-run),
then measure 8/16 with a separate run name. Include long examples, optimizer
updates, validation and checkpoint serialization in the capacity check.
The short two-epoch, two-latent-position smoke command only exercises the pipeline.

Dataset preparation and storage use host RAM/disk; only the current batch is
transferred to the GPU. The Slurm template requests 128 GB host RAM and a
24-hour walltime. Configure the requested GPU model and job walltime for the
actual cluster and plan; a ten-epoch job may need epoch-level resume.

Checkpoint-selection and evaluation-accounting fixes are on main. Existing CPU
evidence is summarized in [audit status](AUDIT_STATUS.md). Real-model loading,
BF16 training stability, A100 capacity, CUDA timing and distributed execution
remain validation work.
