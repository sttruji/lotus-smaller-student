"""Record model/data provenance without importing GPU libraries."""
import hashlib
import json
import subprocess
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_manifest(save_dir, config, model, world_size):
    import torch
    import transformers

    base = getattr(model, "base_causallm", model)
    teacher = getattr(model, "teacher_model", None)
    alignment = getattr(model, "distill_alignment", None)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = {
        "config": config, "git_commit": commit, "git_dirty": dirty,
        "world_size": world_size,
        "effective_batch_examples": world_size * config["batch_size_training"] * config.get("gradient_accumulation_steps", 1),
        "student_backbone_parameters": sum(p.numel() for p in base.parameters()),
        "training_projection_parameters": 0 if alignment is None else sum(p.numel() for p in alignment.parameters()),
        "teacher_parameters": 0 if teacher is None else sum(p.numel() for p in teacher.parameters()),
        "layer_mapping": [] if alignment is None else alignment.pairs,
        "student_config": base.config.to_dict(),
        "teacher_config": None if teacher is None else teacher.config.to_dict(),
        "student_backbone_commit": getattr(base.config, "_commit_hash", None),
        "teacher_checkpoint_commit": None if teacher is None else getattr(teacher.config, "_commit_hash", None),
        "dataset_sha256": {key: file_sha256(value) for key, value in config.items()
                           if key in ("train_path", "val_path", "test_path") and Path(value).is_file()},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda},
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "boundary": "before first supervised answer token (including the ### delimiter)",
        "loop_count": "n_looped_iters additional passes after an initial latent pass (R+1 total)",
    }
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Preserve provenance on resume instead of overwriting the initial run.
    destination = out / (f"manifest_resume_{config['resume']}.json" if config.get("resume", 0) else "run_manifest.json")
    destination.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest
