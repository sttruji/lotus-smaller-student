"""Cheap configuration/data checks, with optional HF config/tokenizer checks."""
import argparse
import json
from pathlib import Path
from configuration import apply_overrides, load_hierarchical_yaml


def validate_configuration(config):
    for key in ("train_max_examples", "val_max_examples"):
        value = config.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{key} must be null or a positive integer")
    if not config.get("looped") and (config.get("student_cot_loss_weight", 0) > 0 or config.get("main_answer_only")):
        raise ValueError("Student auxiliary CoT and main_answer_only are LOTUS wrapper objectives")
    for key in ("batch_size_training", "gradient_accumulation_steps", "num_epochs", "epochs_per_stage"):
        value = config.get(key, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get("codi_loss_weight", 0) > 0:
        if not config.get("looped") or not config.get("teacher_model_path"):
            raise ValueError("Distillation requires looped=true and teacher_model_path")
        if config.get("codi_alignment") not in ("relative_depth", "final", "legacy"):
            raise ValueError("Unsupported codi_alignment")
        if config.get("codi_alignment") != "legacy" and config.get("codi_loss_type") != "normalized_l1":
            raise ValueError("Cross-size alignment requires normalized_l1")
    if config.get("replace_all_cot_at_max_stage"):
        if config.get("cumulative_stages") or config.get("intermediate_loss_weight", 0) > 0 or config.get("inter_answer_loss_weight", 0) > 0:
            raise ValueError("Full-CoT replacement currently supports single-stage batches without per-step intermediate supervision")
        if config.get("max_latent_stage", 0) < 1 or config.get("num_epochs", 0) <= config["max_latent_stage"] * config["epochs_per_stage"]:
            raise ValueError("Training must reach and train the fully latent stage")


def validate_data(config):
    summary = {}
    for key in ("train_path", "val_path", "test_path"):
        path = Path(config[key])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key}: {path}; run bash preprocessing/gsm_icot.bash first")
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Expected a nonempty JSON list: {path}")
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError(f"Invalid question at {path}:{i}")
            if not isinstance(row.get("answer"), str) or not row["answer"].strip():
                raise ValueError(f"Invalid answer at {path}:{i}")
            if not isinstance(row.get("steps"), list) or not all(isinstance(s, str) for s in row["steps"]):
                raise ValueError(f"Invalid reasoning steps at {path}:{i}")
        summary[key] = {"path": str(path), "examples": len(rows), "max_steps": max(len(row["steps"]) for row in rows)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--config-only", action="store_true", help="Do not access datasets or the Hub")
    parser.add_argument("--check-models", action="store_true", help="Read HF configs/tokenizers, without loading weights")
    args = parser.parse_args()
    config = apply_overrides(load_hierarchical_yaml(args.config_file), args.set)
    validate_configuration(config)
    report = {"name": config["name"], "model_id": config["model_id"], "config_valid": True}
    if not args.config_only:
        report["data"] = validate_data(config)
    if args.check_models:
        from transformers import AutoConfig, AutoTokenizer
        from distillation import assert_shared_tokenizer, layer_pairs
        student = AutoConfig.from_pretrained(config["model_id"], revision=config.get("model_revision"))
        student_tok = AutoTokenizer.from_pretrained(config["model_id"], revision=config.get("model_revision"))
        source = config.get("load_model_path")
        if source and not source.startswith(("/", ".", "~")):
            initial = AutoConfig.from_pretrained(source, revision=config.get("student_revision"))
            if (initial.hidden_size, initial.num_hidden_layers) != (student.hidden_size, student.num_hidden_layers):
                raise ValueError("Student initialization width/depth does not match its backbone")
        if config.get("codi_loss_weight", 0) > 0:
            teacher_id = config.get("teacher_model_id") or config["model_id"]
            teacher_source = config["teacher_model_path"]
            path = Path(teacher_source).expanduser()
            if teacher_source.startswith(("/", ".", "~")) and not path.exists():
                raise FileNotFoundError(path)
            config_source = str(path) if path.is_dir() and (path / "config.json").exists() else teacher_id if path.exists() else teacher_source
            teacher = AutoConfig.from_pretrained(config_source, revision=config.get("teacher_revision"))
            teacher_tok = AutoTokenizer.from_pretrained(config.get("teacher_tokenizer_id") or teacher_id,
                                                       revision=config.get("teacher_tokenizer_revision"))
            assert_shared_tokenizer(student_tok, teacher_tok)
            mode = config.get("codi_alignment", "legacy")
            if mode == "legacy":
                if (student.hidden_size, student.num_hidden_layers) != (teacher.hidden_size, teacher.num_hidden_layers):
                    raise ValueError("Legacy alignment requires identical widths and depths")
                report["layer_mapping"] = [(i, i) for i in range(student.num_hidden_layers + 1)]
            else:
                report["layer_mapping"] = layer_pairs(student.num_hidden_layers, teacher.num_hidden_layers, mode)
            report["width_mapping"] = [student.hidden_size, teacher.hidden_size]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
