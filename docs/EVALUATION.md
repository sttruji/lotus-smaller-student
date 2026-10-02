# Evaluation and metrics

Current implementation: [eval.py](../scripts/eval.py),
[eval_accounting.py](../scripts/eval_accounting.py) and
[Lotus.generate](../scripts/lotus.py). Metrics use schema version 2.

The evaluator runs greedy decoding on one example at a time. It supports the
recurrent student and plain CoT baselines without loading a distillation teacher
or training projections.

## Exported student

~~~bash
python scripts/eval.py \
  --model_id ./outputs/student-inference --datasets gsm8k \
  --max_new_tokens 512 --device cuda:0 \
  --save_preds ./outputs/student-predictions.json \
  --save_metrics ./outputs/student-metrics.json
~~~

Local HF exports supply CoT/looped mode, loop count, block width and injection
mode through lotus_config.json. Explicit CLI options override defaults and
are saved alongside the exported architecture. Relevant overrides are --cot,
--looped, --n_looped_iters, --c_thought, --latent_injection_mode and
--n_latent_override. Remote HF repositories require explicit matching settings;
the evaluator reads lotus_config.json from a local model directory.

For a local CoT export that records c_thought=0, currently pass --c_thought 1
alongside --cot. The CLI requires positive c_thought before choosing the
generation path, even though CoT inference uses no latent positions. Raw CoT
checkpoint commands below avoid this export-default edge case. The parser guard
is tracked in [audit status](AUDIT_STATUS.md).

## Raw checkpoints and explicit baselines

~~~bash
python scripts/eval.py \
  --model_id meta-llama/Llama-3.2-1B-Instruct \
  --checkpoint ./outputs/gsm-distill-controlled-seed0/checkpoint_final \
  --n_looped_iters 6 --c_thought 25 --latent_injection_mode add \
  --datasets gsm8k --max_new_tokens 512 \
  --save_metrics ./outputs/student-raw-metrics.json

python scripts/eval.py \
  --model_id meta-llama/Llama-3.2-1B-Instruct \
  --checkpoint ./outputs/gsm-cot-sft-student-1b-seed0/checkpoint_final \
  --datasets gsm8k --cot --max_new_tokens 512 \
  --save_metrics ./outputs/cot-1b-metrics.json

python scripts/eval.py \
  --model_id meta-llama/Llama-3.2-3B-Instruct \
  --checkpoint ./outputs/gsm-cot-sft-teacher-3b-seed0/checkpoint_final \
  --datasets gsm8k --cot --max_new_tokens 512 \
  --save_metrics ./outputs/cot-3b-metrics.json
~~~

Use the same test population and output cap for all models. The CLI's output cap
defaults to 64; these comparison commands explicitly use 512 so explicit CoT has
room to finish. Report truncation rates and the chosen cap.

Raw weights load with strict backbone validation. Known training-only
projection/auxiliary heads are excluded, and the shared embedding alias is
checked. Missing, unexpected or incompatible backbone weights abort evaluation;
incomplete HF exports are also rejected. Loading/configuration failures occur
before prediction reports are produced.

A bare HF model with newly added latent tokens requires
--allow_untrained_latent_tokens. Newly assigned or initialized latent embeddings
are flagged in metadata. This option is for an intentional untrained baseline.

## Population and scoring

| Dataset | Evaluated population |
| --- | --- |
| gsm8k | Local data/gsm_test.json |
| gsm-hard | reasoning-machines/gsm-hard, train split used as an external benchmark |
| multi-arith | ChilleD/MultiArith, test split |
| svamp | ChilleD/SVAMP train+test by default; --svamp_split test selects only test |
| Local JSON path | Ordered list with question and answer fields |

The ordered question/answer SHA-256 and actual source/split are recorded.
A different SVAMP split changes the population and requires separate reporting.

The answer extractor takes the first numeric span after the last ### delimiter,
or the last number when there is no delimiter. Numeric comparison uses exact
Decimal values, including comma normalization, and rejects nonfinite numeric
answers. Generated text is retained so extraction can be audited. Training-time
validation still uses its own string parser; its accuracy is not guaranteed to
match standalone rescoring.

## Errors, denominators and termination

Generation stops at the first EOS. Every attempted example receives an ok or
error row. Successful rows include generated text and token IDs, extracted
answer, correctness, token counts, stop reason and timing. Error rows include
the example index, question, gold answer, exception type/message and elapsed time.

| Field or policy | Meaning |
| --- | --- |
| total | Entire selected population |
| attempted / successful / failed / unattempted | Explicit execution counts |
| accuracy | correct/total when the population is completed; null when incomplete or empty |
| accuracy_successful | Diagnostic accuracy over successful attempts |
| valid_for_accuracy | True only for a completed, nonempty population with no generation errors |
| generated_tokens | Includes terminal EOS for both CoT and LOTUS |
| content_tokens | Excludes terminal EOS |
| hit_token_limit | Token count equals the configured output cap |
| truncated | Cap reached without EOS; EOS exactly at the cap is normal termination |
| avg_gen_tokens / avg_inference_time | Averages over successful examples |

Generation errors stop the run by default after writing any requested reports.
--continue_on_error records failures and continues through the requested
populations. Completed populations count failures as incorrect. Either mode
returns a nonzero exit status when generation fails, and run_status is failed.
A truncated but otherwise successful generation is still scored under the
configured token budget.

## Time and memory

Use inference_time and avg_inference_time for comparisons. They measure the
synchronized full generation call, covering prefix, loops, suffix forward and
decoding. Tokenization, text decoding and scoring are excluded. Dataset time
and total_elapsed include broader evaluator work.

Per-phase totals remain diagnostics. prefill_other_time includes the LOTUS
suffix forward and forward bookkeeping outside the prefix/loop timers;
unattributed_inference_time reports residual time outside the phase timers.
Delimiter phase detection uses a token-ID heuristic. Phase sums are not the
authoritative full-generation measurement.

Peak GPU allocation/reservation include the resident model and generation,
are reported in GiB, and target the selected --device. Synchronization also
targets that device. CPU evaluation uses --device cpu --fp32 and reports null
GPU memory; CPU timings do not establish GPU performance.

There is no warmup exclusion, latency distribution, batched throughput or
FLOP accounting. Retained diagnostic logits still contribute implementation
overhead. See [audit status](AUDIT_STATUS.md) for remaining work.
