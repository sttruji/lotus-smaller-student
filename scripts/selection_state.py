"""Preserve validation-based checkpoint selection across epoch resumes."""
import json
import math
from pathlib import Path


def restore_best_accuracy(save_dir, checkpoint_best=-1.0, *, require_fully_latent=False):
    """Reconcile old periodic state with the final artifact in this run directory."""
    best = float(checkpoint_best)
    if not math.isfinite(best) or not -1.0 <= best <= 1.0:
        raise ValueError("Invalid best_acc in the resumed training state")
    directory = Path(save_dir)
    metadata_path = directory / "checkpoint_final_metadata.json"
    if not (directory / "checkpoint_final").is_file() or not metadata_path.is_file():
        return best
    metadata = json.loads(metadata_path.read_text())
    if require_fully_latent and metadata.get("all_cot_removed") is not True:
        return best
    accuracy = float(metadata["val_accuracy"])
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"Invalid val_accuracy in {metadata_path}")
    return max(best, accuracy)


def update_checkpoint_best_accuracy(checkpoint_dir, best_acc):
    """Update the selection score without changing the saved optimizer/RNG state."""
    import torch

    path = Path(checkpoint_dir) / "training_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["best_acc"] = best_acc
    temporary = path.with_name(".training_state.pt.tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
