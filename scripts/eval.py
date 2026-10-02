"""Standalone evaluation with explicit accuracy and resource accounting."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
from lotus import Lotus
from eval_accounting import (checkpoint_backbone, evaluate_examples,
                             extract_answer, generation_record)


def load_ood_dataset(name, *, svamp_split="all"):
    """Return the ordered question/answer population evaluated by this run."""
    if name == "gsm-hard":
        ds = load_dataset("reasoning-machines/gsm-hard", split="train")
        data = [(ex["input"], ex["target"]) for ex in ds]
    elif name == "multi-arith":
        ds = load_dataset("ChilleD/MultiArith", split="test")
        data = [(ex["question"], ex["final_ans"]) for ex in ds]
    elif name == "svamp":
        ds = load_dataset("ChilleD/SVAMP")
        population = (concatenate_datasets([ds["train"], ds["test"]])
                      if svamp_split == "all" else ds["test"])
        data = [(ex["question_concat"], ex["Answer"]) for ex in population]
    else:
        path = Path(_REPO_ROOT) / "data/gsm_test.json" if name == "gsm8k" else Path(name)
        with path.open() as stream:
            raw = json.load(stream)
        data = [(ex["question"], ex["answer"]) for ex in raw]
    return [(str(question).strip(), str(answer).replace(",", "").strip())
            for question, answer in data]


def dataset_source(name, svamp_split):
    if name == "gsm-hard":
        return {"dataset": "reasoning-machines/gsm-hard", "split": "train"}
    if name == "multi-arith":
        return {"dataset": "ChilleD/MultiArith", "split": "test"}
    if name == "svamp":
        return {"dataset": "ChilleD/SVAMP", "split": "train+test" if svamp_split == "all" else "test"}
    path = Path(_REPO_ROOT) / "data/gsm_test.json" if name == "gsm8k" else Path(name)
    return {"path": str(path.resolve()), "split": "local_json"}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def phase_times(token_times, marker_ids, *, cot):
    """Heuristic delimiter timing; the full generation timer is authoritative."""
    thought = marker = answer = 0.0
    phase = "thought" if cot else "marker"
    boundary_ids, all_marker_ids = marker_ids
    for token_id, elapsed in token_times:
        if phase == "thought" and token_id in boundary_ids:
            phase = "marker"
        if phase == "thought":
            thought += elapsed
        elif phase == "marker" and token_id in all_marker_ids:
            marker += elapsed
        else:
            phase = "answer"
            answer += elapsed
    return thought, marker, answer


def predict_example(model, tokenizer, question, args, special_ids, marker_ids):
    device = torch.device(args.device)
    question_ids = tokenizer.encode(question + "\n", add_special_tokens=True)
    if args.cot:
        prefix = question_ids
    else:
        start_id, end_id, latent_id = special_ids
        count = (args.n_latent_override if args.n_latent_override is not None
                 else args.n_looped_iters * args.c_thought)
        prefix = question_ids + [start_id] + [latent_id] * count + [end_id]
    input_ids = torch.tensor([prefix], device=device)
    attention_mask = torch.ones_like(input_ids)

    synchronize(device)
    inference_start = time.perf_counter()
    if args.cot:
        prefill_start = time.perf_counter()
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        synchronize(device)
        prefill_time = time.perf_counter() - prefill_start
        cache = output.past_key_values
        first_start = time.perf_counter()
        next_token = torch.argmax(output.logits[0, -1]).item()
        token_times = [(next_token, time.perf_counter() - first_start)]
        generated_ids = [next_token]
        del output
        seq_len = len(prefix)
        for _ in range(args.max_new_tokens - 1):
            if next_token == tokenizer.eos_token_id:
                break
            token_start = time.perf_counter()
            output = model(
                input_ids=torch.tensor([[next_token]], device=device),
                past_key_values=cache,
                position_ids=torch.tensor([[seq_len]], device=device),
                use_cache=True,
            )
            cache = output.past_key_values
            seq_len += 1
            next_token = torch.argmax(output.logits[0, -1]).item()
            synchronize(device)
            token_times.append((next_token, time.perf_counter() - token_start))
            generated_ids.append(next_token)
        thought, marker, answer = phase_times(token_times, marker_ids, cot=True)
        phases = dict(query_prefill_time=prefill_time, thought_time=thought,
                      prefill_other_time=0.0, eot_marker_time=marker, answer_time=answer)
    else:
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens, n_looped_iters=args.n_looped_iters,
        )
        generated_ids = output[0, len(prefix):].tolist()
        timed_ids = [token_id for token_id, _ in model._last_decode_token_times]
        if generated_ids != timed_ids:
            raise ValueError("Returned generation and timed token sequence disagree")
        _, marker, answer = phase_times(model._last_decode_token_times, marker_ids, cot=False)
        phases = dict(query_prefill_time=model._last_timing["query_prefill"],
                      thought_time=model._last_timing["thought"],
                      prefill_other_time=model._last_timing["prefill_other"],
                      eot_marker_time=marker, answer_time=answer)
    synchronize(device)
    inference_time = time.perf_counter() - inference_start
    record = generation_record(generated_ids, tokenizer.eos_token_id, args.max_new_tokens)
    record.update(phases)
    record.update(inference_time=inference_time, input_tokens=len(prefix),
                  generated_text=tokenizer.decode(generated_ids, skip_special_tokens=True))
    return record


def write_reports(args, metadata, results):
    run_metadata = dict(metadata)
    run_metadata["run_status"] = (
        "failed" if any(not result["valid_for_accuracy"] for result in results.values())
        else "complete" if len(results) == len(args.datasets) else "in_progress")
    if args.save_preds:
        path = Path(args.save_preds)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({name: result["per_example"] for name, result in results.items()},
                                   indent=2, allow_nan=False) + "\n")
    if args.save_metrics:
        path = Path(args.save_metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metadata": run_metadata, "datasets": results},
                                   indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="Training state dict, or a directory containing model.pt. Loads strictly.")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--datasets", nargs="+", default=["gsm-hard", "multi-arith", "svamp"])
    parser.add_argument("--n_looped_iters", type=int, default=None)
    parser.add_argument("--c_thought", type=int, default=None)
    parser.add_argument("--latent_injection_mode", choices=["add", "replace"], default=None)
    parser.add_argument("--n_latent_override", type=int, default=None,
                        help="Exact latent-prefix position count; otherwise loops*c_thought.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp32", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cot", dest="cot", action="store_true", default=None)
    mode.add_argument("--looped", dest="cot", action="store_false")
    parser.add_argument("--save_preds", default=None)
    parser.add_argument("--save_metrics", default=None)
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Record failures and continue. Failures count as incorrect; exit status remains nonzero.")
    parser.add_argument("--svamp_split", choices=["all", "test"], default="all",
                        help="all preserves the inherited train+test population; test selects only the test split.")
    parser.add_argument("--allow_untrained_latent_tokens", action="store_true",
                        help="Allow a bare HF baseline to initialize missing latent embeddings; recorded in metadata.")
    args = parser.parse_args()

    # Exported architecture settings are defaults; explicit CLI settings are logged overrides.
    architecture_path = Path(args.model_id) / "lotus_config.json"
    architecture = json.loads(architecture_path.read_text()) if architecture_path.is_file() else {}
    args.cot = args.cot if args.cot is not None else architecture.get("cot", False)
    for field, default in (("n_looped_iters", 6), ("c_thought", 25), ("latent_injection_mode", "add")):
        if getattr(args, field) is None:
            setattr(args, field, architecture.get(field, default))
    if args.max_new_tokens < 1 or args.n_looped_iters < 0 or args.c_thought < 1:
        parser.error("max_new_tokens and c_thought must be positive; n_looped_iters must be nonnegative")
    if args.n_latent_override is not None and args.n_latent_override < 0:
        parser.error("n_latent_override must be nonnegative")
    if args.latent_injection_mode not in ("add", "replace"):
        parser.error("Unsupported latent_injection_mode in exported architecture")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Dataset names must be unique")

    device = torch.device(args.device)
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    print(f"Loading tokenizer and base model from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Evaluation requires an EOS token")
    special_ids = None
    added_latent_tokens = 0
    if not args.cot:
        tokens = ("<|start-latent|>", "<|end-latent|>", "<|latent|>")
        added_latent_tokens = tokenizer.add_tokens(list(tokens))
        special_ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in tokens)
        if added_latent_tokens and not args.checkpoint and not args.allow_untrained_latent_tokens:
            raise ValueError("HF tokenizer has no trained latent tokens. Use a trained checkpoint/export, "
                             "or --allow_untrained_latent_tokens for an intentional untrained baseline.")
    base_model, loading_info = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, output_loading_info=True)
    hf_loaded_completely = not any(loading_info.get(field) for field in
                                  ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    if not args.checkpoint and not hf_loaded_completely:
        raise ValueError(f"HF model weights did not load completely: {loading_info}")

    checkpoint_loaded_strictly = False
    if args.checkpoint:
        path = Path(args.checkpoint)
        path = path / "model.pt" if path.is_dir() else path
        print(f"Loading complete checkpoint from {path}...")
        backbone = checkpoint_backbone(torch.load(path, map_location="cpu", weights_only=True), cot=args.cot)
        embedding_keys = [key for key in backbone
                          if key.endswith(("embed_tokens.weight", "wte.weight"))]
        if len(embedding_keys) != 1:
            raise ValueError("Checkpoint must contain exactly one backbone input embedding table")
        vocabulary = backbone[embedding_keys[0]].shape[0]
        if vocabulary < len(tokenizer):
            raise ValueError("Checkpoint embedding table does not cover the evaluation tokenizer")
        if vocabulary != base_model.get_input_embeddings().num_embeddings:
            base_model.resize_token_embeddings(vocabulary)
        base_model.load_state_dict(backbone, strict=True)
        checkpoint_loaded_strictly = True
        del backbone

    initialized_latent_tokens = False
    if base_model.get_input_embeddings().num_embeddings < len(tokenizer):
        if args.checkpoint or args.cot or not args.allow_untrained_latent_tokens:
            raise ValueError("Model has no weights for the latent tokens. Use a trained checkpoint/export, "
                             "or --allow_untrained_latent_tokens for an intentional untrained baseline.")
        base_model.resize_token_embeddings(len(tokenizer))
        initialized_latent_tokens = True
    if args.cot:
        model = base_model
    else:
        start_id, end_id, latent_id = special_ids
        model = Lotus(base_model, latent_token_id=latent_id, start_latent_id=start_id,
                      end_latent_id=end_id, eos_token_id=tokenizer.eos_token_id,
                      pad_token_id=tokenizer.pad_token_id, c_thought=args.c_thought,
                      latent_injection_mode=args.latent_injection_mode)
    model.to(device).to(dtype).eval()

    boundary_ids = set(tokenizer.encode("###", add_special_tokens=False))
    boundary_ids.update(tokenizer.encode("#", add_special_tokens=False))
    marker_ids = (boundary_ids, boundary_ids | set(tokenizer.encode(" ", add_special_tokens=False)))
    import transformers
    metadata = {
        "schema_version": 2, "model_id": args.model_id, "checkpoint": args.checkpoint,
        "checkpoint_loaded_strictly": checkpoint_loaded_strictly,
        "hf_loaded_completely": hf_loaded_completely,
        "weights_source": "checkpoint" if args.checkpoint else "hf_model",
        "initialized_latent_tokens": initialized_latent_tokens,
        "untrained_latent_tokens": bool(initialized_latent_tokens or (added_latent_tokens and not args.checkpoint)),
        "exported_architecture": architecture,
        "cot": args.cot, "n_looped_iters": args.n_looped_iters if not args.cot else 0,
        "c_thought": args.c_thought if not args.cot else None,
        "n_latent_override": args.n_latent_override,
        "latent_positions": (0 if args.cot else args.n_latent_override
                             if args.n_latent_override is not None
                             else args.n_looped_iters * args.c_thought),
        "latent_injection_mode": args.latent_injection_mode if not args.cot else None,
        "max_new_tokens": args.max_new_tokens, "dtype": str(dtype), "device": str(device),
        "unique_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "batch_size": 1, "warmup_examples": 0, "decoding": "greedy",
        "continue_on_error": args.continue_on_error, "requested_datasets": args.datasets,
        "accuracy_policy": "correct/entire dataset; errors count as incorrect; null if incomplete or empty",
        "answer_extraction": "first number after last ###; otherwise last number; exact Decimal comparison",
        "resource_average_denominator": "successful examples",
        "generated_token_policy": "includes EOS consistently in both CoT and LOTUS",
        "timing_scope": "synchronized full generation call; tokenization, scoring and text decode excluded; no warmup exclusion",
        "phase_detection": "heuristic token IDs for ###, # and space",
        "prefill_other_scope": "LOTUS suffix forward and forward bookkeeping outside prefix/latent timers",
        "memory_scope": "peak allocation/reservation including resident model and generation; selected device; GiB",
    }
    results = {}
    start = time.perf_counter()
    for dataset in args.datasets:
        print(f"\nEvaluating {dataset}...")
        data = load_ood_dataset(dataset, svamp_split=args.svamp_split)
        print(f"Loaded {len(data)} examples; source={dataset_source(dataset, args.svamp_split)}")
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with tqdm(total=len(data), desc=dataset) as progress_bar:
            attempted = correct = 0

            def progress(record):
                nonlocal attempted, correct
                attempted += 1
                correct += int(record["correct"])
                if record["status"] == "error":
                    print(f"\nError on {dataset} example {record['idx']}: "
                          f"{record['error_type']}: {record['error']}", file=sys.stderr)
                elif record["idx"] < 3:
                    print(f"\nPred: {record['pred']!r} | GT: {record['answer']!r} "
                          f"| stop={record['stop_reason']}")
                progress_bar.update(1)
                progress_bar.set_description(f"{dataset} attempted_acc={correct/attempted:.3f}")

            with torch.no_grad():
                result = evaluate_examples(
                    data, lambda question: predict_example(model, tokenizer, question, args, special_ids, marker_ids),
                    continue_on_error=args.continue_on_error, progress=progress,
                )
        synchronize(device)
        result["source"] = dataset_source(dataset, args.svamp_split)
        result["peak_mem_alloc_gb"] = (torch.cuda.max_memory_allocated(device) / 1024**3
                                       if device.type == "cuda" else None)
        result["peak_mem_reserved_gb"] = (torch.cuda.max_memory_reserved(device) / 1024**3
                                          if device.type == "cuda" else None)
        results[dataset] = result
        write_reports(args, metadata, results)
        accuracy = f"{result['accuracy']:.2%}" if result["accuracy"] is not None else "unavailable"
        print(f"{dataset}: status={result['status']} accuracy={accuracy} "
              f"correct={result['correct']}/{result['total']} successful={result['successful']} "
              f"failed={result['failed']} unattempted={result['unattempted']} truncated={result['truncated']}")
        print(f"  full generation={result['inference_time']:.3f}s; "
              f"phase sum={result['measured_phase_time']:.3f}s; "
              f"unattributed={result['unattributed_inference_time']:.3f}s")
        if not result["valid_for_accuracy"] and not args.continue_on_error:
            break

    metadata["total_elapsed"] = time.perf_counter() - start
    write_reports(args, metadata, results)
    for path in (args.save_preds, args.save_metrics):
        if path:
            print(f"Saved {path}")
    return int(len(results) != len(args.datasets)
               or any(not result["valid_for_accuracy"] for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
