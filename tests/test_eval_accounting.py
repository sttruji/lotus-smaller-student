import json
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

import eval as evaluator
from eval_accounting import (PHASE_FIELDS, answers_equal, checkpoint_backbone,
                             evaluate_examples, extract_answer, generation_record)
from smoke_distillation import tiny_problem


def prediction(text="### 42", tokens=None, cap=2):
    result = generation_record(tokens or [5, 1], 1, cap)
    result.update({field: 0.1 for field in PHASE_FIELDS})
    result.update(generated_text=text, inference_time=0.7, input_tokens=3)
    return result


@pytest.mark.parametrize("text,expected", [
    ("### 42\nThe answer is 43.", "42"),
    ("### $1,234.50 dollars", "1234.50"),
    ("Reasoning 9; ### -.5", "-.5"),
    ("### 1 ### 4e2", "4e2"),
    ("Reasoning 9 then 42", "42"),
    ("###", ""),
])
def test_extracts_answer_without_trailing_prose(text, expected):
    assert extract_answer(text) == expected


def test_decimal_scoring_does_not_round_large_integers_or_accept_nonfinite_answers():
    assert answers_equal("1,234.50", "1234.5")
    assert not answers_equal("9007199254740992", "9007199254740993")
    assert not answers_equal("nan", "nan")
    assert not answers_equal("inf", "inf")
    assert not answers_equal("", "")


def test_failure_records_and_resource_denominators_are_explicit():
    def predict(question):
        if question == "fail":
            raise RuntimeError("intentional failed decode")
        return prediction()

    data = [("ok", "42"), ("fail", "42"), ("ok", "42")]
    result = evaluate_examples(data, predict, continue_on_error=True)
    assert result["accuracy"] == 2 / 3
    assert result["accuracy_successful"] == 1
    assert (result["attempted"], result["successful"], result["failed"], result["unattempted"]) == (3, 2, 1, 0)
    assert result["total_gen_tokens"] == 4 and result["avg_gen_tokens"] == 2
    assert result["inference_time"] == 1.4
    assert result["avg_inference_time"] == 0.7
    assert result["status"] == "completed_with_errors" and not result["valid_for_accuracy"]
    assert [record["idx"] for record in result["per_example"]] == [0, 1, 2]
    failure = result["per_example"][1]
    assert failure["status"] == "error" and failure["error_type"] == "RuntimeError"
    assert "intentional" in failure["error"] and "generated_tokens" not in failure
    assert result["unattributed_inference_time"] == pytest.approx(0.4)
    assert len(result["dataset_sha256"]) == 64
    assert evaluate_examples(data, predict, continue_on_error=True)["dataset_sha256"] == result["dataset_sha256"]


def test_default_stops_at_first_error_and_incomplete_accuracy_is_null():
    result = evaluate_examples([("a", "42"), ("b", "42")],
                               lambda question: (_ for _ in ()).throw(ValueError("broken")))
    assert result["status"] == "failed" and result["accuracy"] is None
    assert result["total"] == 2 and result["attempted"] == 1 and result["unattempted"] == 1
    assert result["inference_time"] == 0 and result["avg_gen_tokens"] is None
    assert len(result["per_example"]) == 1


def test_empty_dataset_has_no_fabricated_accuracy_or_division_by_zero():
    result = evaluate_examples([], lambda question: prediction())
    assert result["status"] == "empty" and result["accuracy"] is None
    assert not result["valid_for_accuracy"] and result["avg_gen_tokens"] is None


def test_eos_at_token_cap_is_not_truncation_and_eos_is_counted():
    eos = generation_record([5, 1], 1, 2)
    assert eos["generated_tokens"] == 2 and eos["content_tokens"] == 1
    assert eos["hit_token_limit"] and eos["ended_on_eos"] and not eos["truncated"]
    capped = generation_record([5, 6], 1, 2)
    assert capped["truncated"] and not capped["ended_on_eos"]
    result = evaluate_examples([("a", "42"), ("b", "42")],
                               lambda question: prediction(tokens=[5, 6]) if question == "b" else prediction())
    assert result["truncated"] == 1 and result["truncation_rate"] == 0.5


@pytest.mark.parametrize("tokens,cap", [([], 2), ([1, 5], 2), ([5], 2), ([5, 6, 7], 2)])
def test_invalid_generation_sequences_are_rejected(tokens, cap):
    with pytest.raises(ValueError):
        generation_record(tokens, 1, cap)


def test_complete_training_checkpoint_loads_strictly_without_training_heads():
    trained, _ = tiny_problem()
    state = trained.state_dict()
    backbone = checkpoint_backbone(state)
    bare = LlamaForCausalLM(trained.base_causallm.config)
    bare.load_state_dict(backbone, strict=True)
    assert not any(key.startswith(("distill_alignment.", "embedding.")) for key in backbone)
    for key, value in bare.state_dict().items():
        torch.testing.assert_close(value, trained.base_causallm.state_dict()[key])


def test_partial_or_misnamed_checkpoint_cannot_produce_benchmark_scores():
    trained, _ = tiny_problem()
    state = dict(trained.state_dict())
    del state["base_causallm.model.layers.0.self_attn.q_proj.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        trained.base_causallm.load_state_dict(checkpoint_backbone(state), strict=True)
    state["typo.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="Unknown checkpoint"):
        checkpoint_backbone(state)
    with pytest.raises(ValueError, match="looped checkpoint"):
        checkpoint_backbone(trained.state_dict(), cot=True)


def test_inconsistent_shared_embedding_alias_is_rejected():
    trained, _ = tiny_problem()
    state = dict(trained.state_dict())
    state["embedding.weight"] = torch.zeros_like(state["embedding.weight"])
    with pytest.raises(ValueError, match="alias disagrees"):
        checkpoint_backbone(state)


@pytest.mark.parametrize("first_eos", [True, False])
def test_lotus_stops_on_first_eos_and_returns_all_timed_tokens(monkeypatch, first_eos):
    model, batch = tiny_problem()
    model.eval()
    prefix = batch["input_ids"][:1, :6]
    calls = []

    def first_forward(input_ids, *args, **kwargs):
        logits = torch.zeros(1, input_ids.shape[1], model.base_causallm.config.vocab_size)
        logits[0, -1, 1 if first_eos else 5] = 10
        model._last_kv_cache = None
        return SimpleNamespace(inputs_embeds=model.embedding(input_ids), logits=logits, intermediate_logits=[])

    def decode_forward(**kwargs):
        calls.append(kwargs)
        logits = torch.zeros(1, 1, model.base_causallm.config.vocab_size)
        logits[0, -1, 1] = 10
        return SimpleNamespace(logits=logits, past_key_values=None)

    monkeypatch.setattr(model, "forward", first_forward)
    monkeypatch.setattr(model.base_causallm, "forward", decode_forward)
    with torch.no_grad():
        output = model.generate(prefix, torch.ones_like(prefix), n_looped_iters=1, max_new_tokens=4)
    expected = [1] if first_eos else [5, 1]
    assert output[0, prefix.shape[1]:].tolist() == expected
    assert [token for token, _ in model._last_decode_token_times] == expected
    assert len(calls) == int(not first_eos)
    assert not model._in_generate


def test_stage_zero_generation_resets_previous_latent_phase_timing():
    model, batch = tiny_problem()
    model.eval()
    prefix = batch["input_ids"][:1, :6]
    with torch.no_grad():
        model.generate(prefix, torch.ones_like(prefix), n_looped_iters=1, max_new_tokens=1)
        assert model._last_timing["thought"] > 0
        model.generate(prefix, torch.ones_like(prefix), n_looped_iters=0, max_new_tokens=1)
    assert model._last_timing["thought"] == 0
    assert model._last_timing["query_prefill"] > 0
    assert model._last_timing["prefill_other"] == 0
    assert "_last_kv_cache" not in model._modules


def test_generation_failure_restores_wrapper_mode(monkeypatch):
    model, batch = tiny_problem()
    monkeypatch.setattr(model, "forward", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("failed")))
    with pytest.raises(ValueError):
        model.generate(batch["input_ids"][:1], batch["attention_mask"][:1])
    assert not model._in_generate


def test_gpu_synchronization_targets_the_selected_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", calls.append)
    evaluator.synchronize(torch.device("cpu"))
    evaluator.synchronize(torch.device("cuda:1"))
    assert calls == [torch.device("cuda:1")]


def local_artifacts(tmp_path, *, exported=False):
    model, _ = tiny_problem()
    base = tmp_path / "model"
    model.base_causallm.save_pretrained(base)
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "###": 4, "42": 5}
    vocab.update({f"token{i}": i for i in range(6, 61)})
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>",
                                       bos_token="<bos>", unk_token="<unk>")
    if exported:
        tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
        (base / "lotus_config.json").write_text(json.dumps(
            {"cot": False, "n_looped_iters": 1, "c_thought": 2, "latent_injection_mode": "replace"}))
    tokenizer.save_pretrained(base)
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([{"question": str(i), "answer": "42"} for i in range(3)]))
    metrics, preds = tmp_path / "metrics.json", tmp_path / "preds.json"
    argv = ["eval.py", "--model_id", str(base), "--datasets", str(dataset),
            "--device", "cpu", "--fp32", "--max_new_tokens", "2",
            "--save_metrics", str(metrics), "--save_preds", str(preds)]
    return model, dataset, metrics, preds, argv


def test_cli_evaluates_export_with_saved_architecture_settings(tmp_path, monkeypatch):
    _, dataset, metrics, preds, argv = local_artifacts(tmp_path, exported=True)
    monkeypatch.setattr(sys, "argv", argv)
    assert evaluator.main() == 0
    saved = json.loads(metrics.read_text())
    assert saved["metadata"]["latent_injection_mode"] == "replace"
    assert saved["metadata"]["n_looped_iters"] == 1 and saved["metadata"]["latent_positions"] == 2
    assert saved["metadata"]["run_status"] == "complete"
    result = saved["datasets"][str(dataset)]
    assert result["successful"] == 3 and result["failed"] == 0
    assert result["peak_mem_alloc_gb"] is None and result["inference_time"] > 0
    assert result["prefill_other_time"] > 0
    assert json.loads(preds.read_text())[str(dataset)] == result["per_example"]


@pytest.mark.parametrize("cot", [False, True])
def test_cli_loads_complete_checkpoint_and_counts_actual_generated_tokens(tmp_path, monkeypatch, cot):
    model, dataset, metrics, _, argv = local_artifacts(tmp_path)
    checkpoint = tmp_path / "state.pt"
    torch.save(model.base_causallm.state_dict() if cot else model.state_dict(), checkpoint)
    argv.extend(["--checkpoint", str(checkpoint), "--n_looped_iters", "1", "--c_thought", "2"])
    if cot:
        argv.append("--cot")
    monkeypatch.setattr(sys, "argv", argv)
    assert evaluator.main() == 0
    saved = json.loads(metrics.read_text())
    assert saved["metadata"]["checkpoint_loaded_strictly"]
    result = saved["datasets"][str(dataset)]
    assert result["total_gen_tokens"] == sum(len(record["generated_token_ids"]) for record in result["per_example"])
    assert result["total"] == result["successful"] == 3


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_cli_writes_failure_artifacts_and_returns_nonzero(tmp_path, monkeypatch, continue_on_error):
    _, dataset, metrics, preds, argv = local_artifacts(tmp_path, exported=True)
    def predict(model, tokenizer, question, *args):
        if question == "1":
            raise RuntimeError("decode failed")
        return prediction()
    monkeypatch.setattr(evaluator, "predict_example", predict)
    if continue_on_error:
        argv.append("--continue_on_error")
    monkeypatch.setattr(sys, "argv", argv)
    assert evaluator.main() == 1
    saved = json.loads(metrics.read_text())
    result = saved["datasets"][str(dataset)]
    assert saved["metadata"]["run_status"] == "failed"
    assert result["failed"] == 1 and not result["valid_for_accuracy"]
    assert result["attempted"] == (3 if continue_on_error else 2)
    assert result["accuracy"] == (2 / 3 if continue_on_error else None)
    assert len(json.loads(preds.read_text())[str(dataset)]) == result["attempted"]


def test_cli_rejects_partial_checkpoint_before_loading_test_data(tmp_path, monkeypatch):
    model, _, metrics, _, argv = local_artifacts(tmp_path)
    state = dict(model.state_dict())
    del state["base_causallm.model.layers.0.self_attn.q_proj.weight"]
    checkpoint = tmp_path / "partial.pt"
    torch.save(state, checkpoint)
    argv.extend(["--checkpoint", str(checkpoint)])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(evaluator, "load_ood_dataset", lambda *args, **kwargs: pytest.fail("Must not evaluate partial weights"))
    with pytest.raises(RuntimeError, match="Missing key"):
        evaluator.main()
    assert not metrics.exists()


def test_cli_rejects_incomplete_hf_export_before_loading_test_data(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file
    _, _, metrics, _, argv = local_artifacts(tmp_path, exported=True)
    weights_path = tmp_path / "model/model.safetensors"
    state = load_file(weights_path)
    del state["model.layers.0.self_attn.q_proj.weight"]
    save_file(state, weights_path, metadata={"format": "pt"})
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(evaluator, "load_ood_dataset", lambda *args, **kwargs: pytest.fail("Must not evaluate incomplete HF weights"))
    with pytest.raises(ValueError, match="did not load completely"):
        evaluator.main()
    assert not metrics.exists()


def test_bare_hf_latent_baseline_requires_explicit_opt_in_even_with_padded_vocabulary(tmp_path, monkeypatch):
    _, dataset, metrics, _, argv = local_artifacts(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="no trained latent tokens"):
        evaluator.main()
    argv.extend(["--allow_untrained_latent_tokens", "--n_looped_iters", "1", "--c_thought", "2"])
    assert evaluator.main() == 0
    saved = json.loads(metrics.read_text())
    assert saved["metadata"]["untrained_latent_tokens"]
    assert not saved["metadata"]["initialized_latent_tokens"]
    assert saved["datasets"][str(dataset)]["successful"] == 3


def test_svamp_population_is_explicit_and_test_only_is_supported(monkeypatch):
    from datasets import Dataset
    splits = {"train": Dataset.from_list([{"question_concat": "train", "Answer": 1}]),
              "test": Dataset.from_list([{"question_concat": "test", "Answer": 2}])}
    monkeypatch.setattr(evaluator, "load_dataset", lambda name: splits)
    assert evaluator.load_ood_dataset("svamp") == [("train", "1"), ("test", "2")]
    assert evaluator.load_ood_dataset("svamp", svamp_split="test") == [("test", "2")]
    assert evaluator.dataset_source("svamp", "all")["split"] == "train+test"
