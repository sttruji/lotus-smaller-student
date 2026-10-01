"""Export student backbone weights, excluding teacher and training-only heads."""
import argparse
import json
from pathlib import Path
import torch


def student_backbone_state(state, cot=False):
    if cot:
        if any(k.startswith("base_causallm.") for k in state):
            raise ValueError("--cot cannot export a looped checkpoint")
        return state
    result = {k.removeprefix("base_causallm."): v for k, v in state.items() if k.startswith("base_causallm.")}
    if not result:
        raise ValueError("Expected a LOTUS checkpoint with base_causallm.* weights")
    return result


def export_student(checkpoint, manifest_path, output_dir):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {out}")
    manifest = json.loads(Path(manifest_path).read_text())
    config = manifest["config"]
    checkpoint = Path(checkpoint)
    checkpoint = checkpoint / "model.pt" if checkpoint.is_dir() else checkpoint
    state = student_backbone_state(torch.load(checkpoint, map_location="cpu", weights_only=True), cot=config["cot"])
    model_config = dict(manifest["student_config"])
    model_type = model_config.pop("model_type")
    hf_config = AutoConfig.for_model(model_type, **model_config)
    key = next(k for k in state if k.endswith(("wte.weight", "embed_tokens.weight")))
    hf_config.vocab_size = state[key].shape[0]
    dtype = state[key].dtype
    model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=dtype)
    model.load_state_dict(state, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config.get("model_revision"))
    tokenizer.pad_token = tokenizer.eos_token
    if not config["cot"]:
        for token in ("<|start-latent|>", "<|end-latent|>", "<|latent|>"):
            tokenizer.add_tokens(token)
    if len(tokenizer) != hf_config.vocab_size:
        raise ValueError("Exported tokenizer size does not match the trained embedding table")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    architecture = {
        "cot": config["cot"], "looped": config["looped"],
        "c_thought": config["c_thought"],
        "n_looped_iters": config["max_latent_stage"] if config["looped"] else 0,
        "latent_injection_mode": config.get("latent_injection_mode", "add"),
        "unique_parameters": sum(p.numel() for p in model.parameters()),
        "teacher_required": False, "training_projections_required": False,
        "source_checkpoint": str(checkpoint), "source_git_commit": manifest.get("git_commit"),
        "usage": "Use the LOTUS wrapper and matching loop/token settings for looped inference; plain HF generate does not perform recurrence.",
    }
    (out / "lotus_config.json").write_text(json.dumps(architecture, indent=2) + "\n")
    return architecture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(export_student(args.checkpoint, args.manifest, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
