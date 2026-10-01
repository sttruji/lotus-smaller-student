"""Download-free CPU training/checkpoint smoke test, using tiny random Llamas."""
import argparse
import json
from pathlib import Path
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from lotus import Lotus


def tiny_problem():
    torch.manual_seed(7)
    student_config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=2, num_attention_heads=2,
                                 num_key_value_heads=2, max_position_embeddings=128,
                                 bos_token_id=2, eos_token_id=1, pad_token_id=0)
    teacher_config = LlamaConfig(vocab_size=64, hidden_size=24, intermediate_size=48,
                                 num_hidden_layers=3, num_attention_heads=3,
                                 num_key_value_heads=3, max_position_embeddings=128,
                                 bos_token_id=2, eos_token_id=1, pad_token_id=0)
    model = Lotus(LlamaForCausalLM(student_config), 63, 61, 62, 1, pad_token_id=0, c_thought=2,
                  teacher_model=LlamaForCausalLM(teacher_config), codi_loss_weight=1.0,
                  codi_loss_type="normalized_l1", codi_alignment="relative_depth",
                  student_cot_loss_weight=0.1, main_answer_only=True, require_smaller_student=True)
    ids = torch.tensor([[2, 3, 61, 63, 63, 62, 8, 9, 10, 1],
                        [0, 2, 61, 63, 63, 62, 7, 10, 1, 0]])
    mask = (ids != 0).long()
    answer_labels = torch.full_like(ids, -100)
    answer_labels[0, 8:] = ids[0, 8:]
    answer_labels[1, 7:9] = ids[1, 7:9]
    batch = {"input_ids": ids, "attention_mask": mask, "labels": answer_labels.clone(),
             "answer_labels": answer_labels, "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
             "replaced_cot_steps": torch.tensor([[[4, 5, 0, 0]], [[6, 0, 0, 0]]]),
             "n_looped_iters": 1}
    return model, batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output-dir", default="./outputs/cpu-smoke")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    torch.set_num_threads(1)
    model, batch = tiny_problem()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    teacher_before = {k: v.clone() for k, v in model.teacher_model.state_dict().items()}
    for _ in range(args.steps):
        optimizer.zero_grad()
        output = model(**batch)
        output.loss.backward()
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.distill_alignment.parameters()):
            raise RuntimeError("No projection gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(output.loss.detach()))
    if any(p.grad is not None for p in model.teacher_model.parameters()):
        raise RuntimeError("Teacher received gradients")
    if not all(torch.equal(v, teacher_before[k]) for k, v in model.teacher_model.state_dict().items()):
        raise RuntimeError("Teacher weights changed")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    restored, _ = tiny_problem()
    restored.load_state_dict(torch.load(out / "model.pt", weights_only=True), strict=True)
    restored.train()
    torch.testing.assert_close(restored(**batch).loss, model(**batch).loss)
    report = {"steps": args.steps, "losses": losses, "teacher_frozen": True,
              "checkpoint_roundtrip": True, "student_width_depth": [16, 2],
              "teacher_width_depth": [24, 3], "layer_pairs": model.distill_alignment.pairs}
    (out / "smoke_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
