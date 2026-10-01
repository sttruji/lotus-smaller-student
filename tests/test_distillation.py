from types import SimpleNamespace
import pytest
import torch
from datasets import Dataset
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from configuration import accumulation_divisor, apply_overrides, load_hierarchical_yaml
from dataset import get_cot_latent_dataset
from distillation import assert_shared_tokenizer, layer_pairs, load_teacher, reconstruct_explicit_batch
from export_student import export_student, student_backbone_state
from run_metadata import write_run_manifest
from smoke_distillation import tiny_problem
from inference_model import build_inference_model


def test_cross_size_backward_updates_student_and_projections_only():
    model, batch = tiny_problem()
    teacher_before = {k: v.clone() for k, v in model.teacher_model.state_dict().items()}
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(**batch)
    assert output.n_codi_valid == 2
    assert output.n_student_cot_valid == 10
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.distill_alignment.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.base_causallm.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.teacher_model.parameters())
    assert not model.teacher_model.training
    optimizer.step()
    assert all(torch.equal(v, teacher_before[k]) for k, v in model.teacher_model.state_dict().items())
    state = model.state_dict()
    assert any(k.startswith("distill_alignment.") for k in state)
    assert not any(k.startswith("teacher") for k in state)
    restored, _ = tiny_problem()
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(restored(**batch).loss, model(**batch).loss)
    inference_state = student_backbone_state(state)
    bare = LlamaForCausalLM(model.base_causallm.config)
    bare.load_state_dict(inference_state, strict=True)


def test_alignment_has_no_answer_token_leakage():
    model, batch = tiny_problem()
    before = model(**batch).codi_loss
    changed = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    answer = changed["answer_labels"] != -100
    changed["input_ids"][answer] = 20
    changed["answer_labels"][answer] = 20
    changed["labels"][answer] = 20
    torch.testing.assert_close(before, model(**changed).codi_loss)


def test_checkpointing_preserves_cached_student_objective():
    cached, batch = tiny_problem()
    checkpointed, _ = tiny_problem()
    checkpointed.base_causallm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    original = cached(**batch)
    recomputed = checkpointed(**batch)
    torch.testing.assert_close(original.main_loss, recomputed.main_loss)
    torch.testing.assert_close(original.codi_loss, recomputed.codi_loss)
    original.loss.backward()
    recomputed.loss.backward()
    for (_, a), (_, b) in zip(cached.named_parameters(), checkpointed.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4)


def test_reconstruction_accounts_for_padding_and_visible_cot():
    _, batch = tiny_problem()
    result = reconstruct_explicit_batch(batch["input_ids"], batch["attention_mask"], batch["answer_labels"],
                                        batch["replaced_cot_steps"], 61, 62, 0, hidden_offset=5)
    assert result.teacher_input_ids.tolist() == [[2, 3, 4, 5, 8, 9], [2, 6, 7, 0, 0, 0]]
    assert result.teacher_positions.tolist() == [5, 2]
    assert result.student_positions.tolist() == [2, 1]
    assert result.labels.tolist() == [[-100, -100, 4, 5, 8, 9, 10, 1], [-100, 6, 7, 10, 1, -100, -100, -100]]


@pytest.mark.parametrize("checkpointing", [False, True])
def test_fully_latent_student_restores_more_cot_steps_than_loops(checkpointing):
    model, _ = tiny_problem()
    if checkpointing:
        model.base_causallm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dataset = Dataset.from_list([{"question_tokenized": [2, 3], "steps_tokenized": [[4, 5], [6], [7], [8, 9]],
                                 "answer_tokenized": [10, 1], "idx": 0}])
    config = SimpleNamespace(uniform_prob=0, max_latent_stage=1, no_cot=False, c_thought=2,
                             fixed_latent_tokens=0, replace_all_cot_at_max_stage=True)
    latent = get_cot_latent_dataset(1, dataset, config, 61, 63, 62)[0]
    assert latent["input_ids"] == [2, 3, 61, 63, 63, 62, 10, 1]
    assert len(latent["replaced_cot_steps"]) == 4
    batch = {key: torch.tensor([latent[key]]) for key in ("input_ids", "attention_mask", "labels", "answer_labels", "position_ids")}
    batch["replaced_cot_steps"] = torch.tensor([[[4, 5], [6, 0], [7, 0], [8, 9]]])
    output = model(**batch, n_looped_iters=1)
    output.loss.backward()
    assert output.n_codi_valid == 1
    assert output.n_main_valid == 2
    assert torch.isfinite(output.loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.base_causallm.parameters())


def test_stage_zero_can_train_before_latents_exist():
    model, _ = tiny_problem()
    ids = torch.tensor([[2, 3, 61, 62, 4, 5, 10, 1]])
    answer = torch.tensor([[-100, -100, -100, -100, -100, -100, 10, 1]])
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=torch.arange(8).unsqueeze(0),
                   labels=answer, answer_labels=answer, replaced_cot_steps=torch.empty(1, 0, 0, dtype=torch.long), n_looped_iters=0)
    assert output.n_codi_valid == 1 and output.n_student_cot_valid == 4
    output.loss.backward()


def test_training_wrapper_can_generate_without_answer_labels(monkeypatch):
    model, batch = tiny_problem()
    model.eval()
    # Timing synchronization is irrelevant to a CPU correctness test.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    prefix = batch["input_ids"][:1, :6]
    with torch.no_grad():
        generated = model.generate(prefix, torch.ones_like(prefix), n_looped_iters=1, max_new_tokens=2)
    assert generated.shape[1] > prefix.shape[1]


def test_unwrapped_validation_copy_matches_student_predictions():
    model, batch = tiny_problem()
    model.eval()
    copy = build_inference_model(model.base_causallm.config, model.state_dict(), looped=True,
                                 latent_id=63, start_id=61, end_id=62, eos_id=1, pad_id=0, c_thought=2)
    assert copy.teacher_model is None and copy.distill_alignment is None
    with torch.no_grad():
        torch.testing.assert_close(model(**batch).logits, copy(**batch).logits)


def tokenizer():
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    vocab.update({f"token{i}": i for i in range(4, 61)})
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>", unk_token="<unk>")


def test_independent_teacher_loading_and_tokenizer_contract(tmp_path):
    model, _ = tiny_problem()
    source = tmp_path / "teacher"
    model.teacher_model.save_pretrained(source)
    tok = tokenizer()
    tok.save_pretrained(source)
    student_tok = tokenizer()
    student_tok.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    config = SimpleNamespace(teacher_model_path=str(source), teacher_model_id=str(source), model_id="unused-student",
                             teacher_tokenizer_id=str(source), bf16=False)
    loaded = load_teacher(config, student_tok, "cpu")
    assert (loaded.config.hidden_size, loaded.config.num_hidden_layers) == (24, 3)
    assert all(not p.requires_grad for p in loaded.parameters())
    config.teacher_model_path = str(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        load_teacher(config, student_tok, "cpu")
    incompatible = tokenizer()
    incompatible.add_tokens(["different-token"])
    with pytest.raises(ValueError, match="share token IDs"):
        assert_shared_tokenizer(student_tok, incompatible)


def test_export_has_only_student_and_can_reload(tmp_path):
    model, _ = tiny_problem()
    initial = tmp_path / "initial"
    tokenizer().save_pretrained(initial)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    config = {"model_id": str(initial), "cot": False, "looped": True, "c_thought": 2,
              "max_latent_stage": 1, "batch_size_training": 2, "resume": 0}
    write_run_manifest(tmp_path, config, model, 1)
    architecture = export_student(checkpoint, tmp_path / "run_manifest.json", tmp_path / "exported")
    reloaded = LlamaForCausalLM.from_pretrained(tmp_path / "exported")
    assert not architecture["teacher_required"] and not architecture["training_projections_required"]
    assert sum(p.numel() for p in reloaded.parameters()) == sum(p.numel() for p in model.base_causallm.parameters())
    assert not any("distill" in key or "teacher" in key for key in reloaded.state_dict())


def test_layer_mapping_and_partial_accumulation_window():
    assert layer_pairs(16, 28) == [(1, 2), (2, 4), (3, 5), (4, 7), (5, 9), (6, 11), (7, 12), (8, 14),
                                   (9, 16), (10, 18), (11, 19), (12, 21), (13, 23), (14, 25), (15, 26), (16, 28)]
    assert layer_pairs(16, 28, "final") == [(16, 28)]
    value = torch.tensor(0.0, requires_grad=True)
    for i in range(5):
        (value * (i + 1) / accumulation_divisor(i, 5, 3)).backward()
    assert value.grad == 6.5  # mean(1,2,3) + mean(4,5)


def test_baseline_configs_share_data_and_keep_cot_sft_explicit():
    student = load_hierarchical_yaml("args/research/cot_sft_student_1b.yaml")
    teacher = load_hierarchical_yaml("args/research/cot_sft_teacher_3b.yaml")
    kd = load_hierarchical_yaml("args/research/distill_3b_to_looped_1b.yaml")
    for config in (student, teacher):
        assert config["cot"] and not config["looped"]
        assert config["codi_loss_weight"] == 0 and config["student_cot_loss_weight"] == 0
        assert all(config[key] == kd[key] for key in ("train_path", "val_path", "test_path", "seed", "num_epochs"))
    assert kd["replace_all_cot_at_max_stage"] and kd["main_answer_only"]
    assert apply_overrides(kd, ["seed=2"])["seed"] == 2
