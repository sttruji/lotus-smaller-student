"""Scoring, checkpoint validation, and accounting shared by the evaluator."""
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
import time


PHASE_FIELDS = ("query_prefill_time", "thought_time", "prefill_other_time",
                "eot_marker_time", "answer_time")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def extract_answer(text):
    """Prefer the first answer after the last ###; otherwise use the last number."""
    if "###" in text:
        answer = text.rsplit("###", 1)[1].replace(",", "").strip()
        number = re.search(_NUMBER, answer)
        return number.group() if number else answer
    numbers = re.findall(_NUMBER, text.replace(",", ""))
    return numbers[-1] if numbers else text.strip()


def answers_equal(prediction, answer):
    """Exact decimal comparison avoids float rounding and rejects nonfinite answers."""
    prediction = str(prediction).replace(",", "").strip()
    answer = str(answer).replace(",", "").strip()
    if not prediction or not answer:
        return False
    try:
        pred_number, gold_number = Decimal(prediction), Decimal(answer)
    except InvalidOperation:
        return prediction == answer
    return (pred_number.is_finite() and gold_number.is_finite()
            and pred_number == gold_number)


def checkpoint_backbone(state, *, cot=False):
    """Accept a bare backbone or a LOTUS state with known training-only extras.

    The caller must load the returned backbone with strict=True. Unknown keys
    are errors; distillation/auxiliary heads and the shared embedding alias are
    the only wrapper keys intentionally excluded.
    """
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty model state dictionary")
    wrapped = any(key.startswith("base_causallm.") for key in state)
    if cot and wrapped:
        raise ValueError("--cot cannot load a looped checkpoint")
    if not wrapped:
        return state
    extras = [key for key in state
              if not key.startswith(("base_causallm.", "distill_alignment.", "ia_decoder."))
              and key != "embedding.weight"]
    if extras:
        raise ValueError(f"Unknown checkpoint keys: {extras}")
    backbone = {key.removeprefix("base_causallm."): value for key, value in state.items()
                if key.startswith("base_causallm.")}
    if "embedding.weight" in state:
        import torch
        embedding_keys = [key for key in backbone
                          if key.endswith(("embed_tokens.weight", "wte.weight"))]
        if len(embedding_keys) != 1 or not torch.equal(state["embedding.weight"], backbone[embedding_keys[0]]):
            raise ValueError("Shared embedding alias disagrees with the backbone embedding")
    return backbone


def generation_record(token_ids, eos_id, max_new_tokens):
    """Count all generated tokens, including the EOS token that stops decoding."""
    if not 0 < len(token_ids) <= max_new_tokens:
        raise ValueError("Generation returned an invalid token count")
    if eos_id in token_ids[:-1]:
        raise ValueError("Generation continued after EOS")
    ended_on_eos = token_ids[-1] == eos_id
    if not ended_on_eos and len(token_ids) != max_new_tokens:
        raise ValueError("Generation stopped without EOS before the token limit")
    return {
        "generated_token_ids": token_ids,
        "generated_tokens": len(token_ids),
        "content_tokens": len(token_ids) - int(ended_on_eos),
        "ended_on_eos": ended_on_eos,
        "hit_token_limit": len(token_ids) == max_new_tokens,
        "truncated": not ended_on_eos and len(token_ids) == max_new_tokens,
        "stop_reason": "eos" if ended_on_eos else "token_limit",
    }


def summarize_records(records, expected_total, elapsed):
    successful = [record for record in records if record["status"] == "ok"]
    failed = [record for record in records if record["status"] == "error"]
    correct = sum(record["correct"] for record in successful)
    complete = len(records) == expected_total
    status = ("empty" if expected_total == 0 else
              "failed" if not complete else
              "completed_with_errors" if failed else "complete")
    result = {
        "status": status, "valid_for_accuracy": complete and not failed and expected_total > 0,
        "accuracy": correct / expected_total if complete and expected_total else None,
        "accuracy_successful": correct / len(successful) if successful else None,
        "correct": correct, "total": expected_total, "attempted": len(records),
        "successful": len(successful), "failed": len(failed),
        "unattempted": expected_total - len(records),
        "total_gen_tokens": sum(record["generated_tokens"] for record in successful),
        "avg_gen_tokens": (sum(record["generated_tokens"] for record in successful)
                           / len(successful) if successful else None),
        "ended_on_eos": sum(record["ended_on_eos"] for record in successful),
        "hit_token_limit": sum(record["hit_token_limit"] for record in successful),
        "truncated": sum(record["truncated"] for record in successful),
        "truncation_rate": (sum(record["truncated"] for record in successful)
                            / len(successful) if successful else None),
        "time": elapsed,
        "inference_time": sum(record["inference_time"] for record in successful),
        "failed_example_time": sum(record["elapsed"] for record in failed),
        "per_example": records,
    }
    for field in PHASE_FIELDS:
        result[field] = sum(record[field] for record in successful)
    result["avg_inference_time"] = result["inference_time"] / len(successful) if successful else None
    result["measured_phase_time"] = sum(result[field] for field in PHASE_FIELDS)
    result["unattributed_inference_time"] = sum(
        max(0.0, record["inference_time"] - sum(record[field] for field in PHASE_FIELDS))
        for record in successful)
    return result


def evaluate_examples(data, predict, *, continue_on_error=False, progress=None):
    """Record every attempt; stop on the first error unless continuation is explicit."""
    records = []
    start = time.perf_counter()
    for index, (question, answer) in enumerate(data):
        record = {"idx": index, "question": question, "answer": str(answer)}
        example_start = time.perf_counter()
        try:
            prediction = predict(question)
            pred = extract_answer(prediction["generated_text"])
            record.update(prediction)
            record.update(status="ok", pred=pred, correct=answers_equal(pred, answer))
        except Exception as error:
            # Failed attempts never contribute partially collected tokens or phase times.
            record.update(status="error", pred=None, correct=False,
                          error_type=type(error).__name__, error=str(error))
        record["elapsed"] = time.perf_counter() - example_start
        records.append(record)
        if progress is not None:
            progress(record)
        if record["status"] == "error" and not continue_on_error:
            break
    result = summarize_records(records, len(data), time.perf_counter() - start)
    result["dataset_sha256"] = hashlib.sha256(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return result
