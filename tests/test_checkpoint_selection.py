import json
from unittest.mock import patch

import pytest
import torch

from selection_state import restore_best_accuracy, update_checkpoint_best_accuracy


def save_final(directory, accuracy, *, fully_latent=True):
    torch.save({"weight": torch.tensor([accuracy])}, directory / "checkpoint_final")
    (directory / "checkpoint_final_metadata.json").write_text(json.dumps({
        "val_accuracy": accuracy, "all_cot_removed": fully_latent,
    }))


def test_resume_preserves_better_final_from_old_periodic_state(tmp_path):
    save_final(tmp_path, 0.80)
    best = restore_best_accuracy(tmp_path, 0.60, require_fully_latent=True)
    assert best == 0.80
    assert not 0.70 > best  # A worse resumed epoch must not replace the final artifact.
    assert 0.85 > best
    assert restore_best_accuracy(tmp_path, 0.90, require_fully_latent=True) == 0.90


def test_post_validation_score_survives_checkpoint_roundtrip(tmp_path):
    state = {"epoch": 7, "best_acc": 0.60, "total_train_steps": 42,
             "lr_scheduler": {"last_epoch": 42},
             "rng_state": {"torch": torch.tensor([1, 2, 3], dtype=torch.uint8)}}
    torch.save(state, tmp_path / "training_state.pt")
    update_checkpoint_best_accuracy(tmp_path, 0.80)
    restored = torch.load(tmp_path / "training_state.pt", weights_only=False)
    assert restored["best_acc"] == 0.80
    assert restored["epoch"] == 7 and restored["total_train_steps"] == 42
    assert restored["lr_scheduler"] == state["lr_scheduler"]
    torch.testing.assert_close(restored["rng_state"]["torch"], state["rng_state"]["torch"])
    assert restore_best_accuracy(tmp_path, restored["best_acc"]) == 0.80


def test_cot_assisted_final_does_not_block_fully_latent_selection(tmp_path):
    save_final(tmp_path, 0.95, fully_latent=False)
    assert restore_best_accuracy(tmp_path, -1.0, require_fully_latent=True) == -1.0
    assert restore_best_accuracy(tmp_path, -1.0) == 0.95


def test_missing_final_artifact_does_not_restore_orphaned_metadata(tmp_path):
    save_final(tmp_path, 0.80)
    (tmp_path / "checkpoint_final").unlink()
    assert restore_best_accuracy(tmp_path, 0.60) == 0.60


def test_failed_selection_state_write_preserves_previous_checkpoint(tmp_path):
    state_path = tmp_path / "training_state.pt"
    torch.save({"best_acc": 0.60}, state_path)
    with patch.object(torch, "save", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            update_checkpoint_best_accuracy(tmp_path, 0.80)
    assert torch.load(state_path, weights_only=False)["best_acc"] == 0.60
    assert not (tmp_path / ".training_state.pt.tmp").exists()


@pytest.mark.parametrize("accuracy", [float("nan"), float("inf"), 1.1, -0.1])
def test_invalid_final_accuracy_fails_before_selection(tmp_path, accuracy):
    save_final(tmp_path, accuracy)
    with pytest.raises(ValueError, match="Invalid val_accuracy"):
        restore_best_accuracy(tmp_path, 0.60)
