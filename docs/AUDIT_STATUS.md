# Audit and validation status

Updated 2026-10-02. The initial review covered scaffold commit
6f58ae74999c83a359ca99a94b7f8bfd65b097d5, based on LOTUS upstream commit
eb77e2f7909c5006f58ff0ad7cd6629b942caa9e. The implementation fixes below
are published on main.

## Completed fixes

| Finding | Current behavior | Implementation |
| --- | --- | --- |
| Resuming stale best accuracy could overwrite a better final checkpoint | Periodic selection state is updated atomically after validation; resume reconciles eligible final metadata | [7405492](https://github.com/sttruji/lotus-smaller-student/commit/7405492), [selection_state.py](../scripts/selection_state.py) |
| Initial EOS did not stop generation | First and subsequent EOS terminate immediately; returned/timed token sequences agree and include EOS | [0e556f2](https://github.com/sttruji/lotus-smaller-student/commit/0e556f2), [lotus.py](../scripts/lotus.py) |
| Evaluator could continue after an incomplete model load | Complete backbone weights are required for raw checkpoints and HF exports; known training heads are excluded explicitly | [eval.py](../scripts/eval.py), [eval_accounting.py](../scripts/eval_accounting.py) |
| Failed generation disappeared from prediction rows | Every attempt has a prediction/error record; failures return nonzero; counts and denominators are explicit | [Evaluation guide](EVALUATION.md) |
| Phase sums omitted the suffix forward | Full synchronized generation time is authoritative; forward overhead and residual time are reported | [Evaluation timing](EVALUATION.md#time-and-memory) |
| Export settings were ignored and instrumentation could target the wrong GPU | Local export settings supply defaults; active overrides are logged; synchronization/memory calls select the requested device | [eval.py](../scripts/eval.py) |
| Trailing prose made numeric answers incorrect | Numeric extraction and exact Decimal comparison; raw text and truncation/termination are retained | [eval_accounting.py](../scripts/eval_accounting.py) |
| Runtime caches could become registered model children; phase state could leak between calls | Cache attributes bypass module registration; stage-zero timers reset and generate-mode state is restored after failed forwards | [lotus.py](../scripts/lotus.py) |

Earlier scaffold fixes cover out-of-place recurrent embedding updates,
cache-free activation-checkpointed decoder calls and objective-preserving suffix
recomputation. Final checkpoint selection begins only at the fully latent stage
for the main research configuration.

## Existing CPU evidence

The last recorded suite run on 2026-10-01 passed **54 tests in 2.55 seconds**
with Hub access disabled. These checks use tiny randomly initialized/local
Llamas, not the published billion-parameter checkpoints. The environment was
Python 3.11.0 on macOS arm64, PyTorch 2.7.0 and Transformers 4.46.2; the intended
CUDA environment specifies Python 3.12.

Coverage includes:

- Unequal-width/depth alignment, projection/student gradients, frozen teacher
  and complete gold-rationale reconstruction without answer leakage.
- Cached/checkpointed loss and gradient agreement, stage zero and fully latent
  examples with more gold steps than loops.
- Checkpoint round trips, atomic selection-state failure handling and recovery
  from stale validation selection state.
- Student-only export and strict raw/HF evaluator loading, including rejection
  of incomplete weights before test data is evaluated.
- Complete CPU evaluator runs for CoT and LOTUS, saved architecture settings,
  failure reports and exit status, exact scoring, EOS/token-cap accounting,
  empty/incomplete populations and dataset-split reporting.

The earlier CPU smoke ran three optimizer updates with losses
5.42756 → 5.33601 → 5.25511, preserved the frozen teacher, produced projection
gradients and restored its checkpoint. Local smoke reports live in ignored
outputs/cpu-smoke; they are development artifacts, not model accuracy results.

## Remaining implementation work

| Item | Current limit / next work |
| --- | --- |
| Vocabulary-head and diagnostic-logit overhead | Latent passes still compute large vocabulary logits, and generation retains diagnostic logits even without intermediate output. Optimize when objectives/analysis do not require them. |
| Training versus standalone scoring | Training validation uses a separate string parser; align policies before treating its score as identical to standalone numeric rescoring. |
| CoT export CLI guard | Local CoT exports can record c_thought=0, which the evaluator's positivity guard rejects. Pass --cot --c_thought 1, or use the raw CoT checkpoint commands, until the guard is scoped to looped inference. Found by source review during this documentation pass. |
| Latency methodology | No warmup exclusion, latency distribution, batched throughput or FLOP accounting. Current phase assignment is heuristic. |
| Reproducibility | Pin model/tokenizer revisions for final runs and capture a complete environment lock; some dependencies remain unpinned. |
| Additional controls | Unlooped smaller CODI control, adaptive halting, shallow recurrent cores and truncated recurrence backpropagation are not implemented. |

## Compute-node validation still needed

1. Load the actual 1B student and independent 3B teacher, verify token IDs,
   provenance and BF16 loss/gradient behavior.
2. Profile both recorded batch-128 options: 2/64 and 8/16. Reach the full
   stage-six workload, include long examples, optimizer and validation peaks,
   and record examples per second and allocated/reserved memory.
3. Exercise checkpoint/export reload and resume across a validation improvement
   through the real CUDA training entry point.
4. Run a two-GPU FSDP pilot and check objective/gradient normalization, collective
   state gathering and validation-generation memory.
5. Train controlled CoT baselines and ablations with recorded initialization,
   data hashes, effective batch, seed and total training compute.
6. Evaluate exported students and baselines on matching test populations with
   equal output caps; report full inference time, truncation and failures.

The [A100 guide](A100_PLANNING.md) contains planning arithmetic, and the
[batch guide](TRAINING_BATCHES.md) records launch options. No real-model accuracy
or A100 performance result is established by the existing CPU evidence.
