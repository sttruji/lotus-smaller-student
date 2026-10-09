"""Completed, checksummed checkpoint bundles for interruptible single-GPU runs.

A directory name alone never establishes completeness. The small COMPLETED.json
record is written last, after the large files have been closed and read back.
This protocol detects partial Drive writes; it cannot guarantee cloud-side sync.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid


MARKER = "COMPLETED.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_bundle(path, *, verify_hashes=True, expected_kind=None):
    path = Path(path)
    record = json.loads((path / MARKER).read_text())
    if record.get("schema_version") != 1 or record.get("kind") not in ("epoch", "best"):
        raise ValueError(f"Unknown checkpoint protocol: {path}")
    if expected_kind and record["kind"] != expected_kind:
        raise ValueError(f"Wrong checkpoint kind: {path}")
    files = record.get("files", {})
    required = {"model.pt", "optimizer.pt", "training_state.pt"} if record["kind"] == "epoch" else {"model.pt", "metadata.json"}
    if set(files) != required:
        raise ValueError(f"Incomplete checkpoint file list: {path}")
    if type(record.get("epoch")) is not int or record["epoch"] < 1:
        raise ValueError(f"Invalid checkpoint epoch: {path}")
    for name, info in files.items():
        artifact = path / name
        if not artifact.is_file() or artifact.stat().st_size != info["bytes"]:
            raise ValueError(f"Missing or truncated artifact: {artifact}")
        if verify_hashes and sha256(artifact) != info["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {artifact}")
    return record


def completed_epochs(save_dir, *, verify_hashes=True):
    valid = []
    for path in Path(save_dir).glob("checkpoint_*"):
        match = re.fullmatch(r"checkpoint_(\d+)", path.name)
        if not match or not path.is_dir():
            continue
        try:
            record = validate_bundle(path, verify_hashes=verify_hashes, expected_kind="epoch")
            if record["epoch"] != int(match[1]):
                raise ValueError("Directory epoch disagrees with completion record")
            valid.append((record["epoch"], path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Ignoring incomplete/corrupt checkpoint {path}: {error}", flush=True)
    return sorted(valid)


def best_checkpoint(save_dir, *, verify_hashes=True):
    valid = []
    for path in (Path(save_dir) / "best_checkpoints").glob("epoch_*"):
        try:
            record = validate_bundle(path, verify_hashes=verify_hashes, expected_kind="best")
            metadata = json.loads((path / "metadata.json").read_text())
            score = metadata["val_accuracy"]
            if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                raise ValueError("Invalid validation accuracy")
            if metadata["epoch"] != record["epoch"]:
                raise ValueError("Best checkpoint epoch mismatch")
            valid.append((score, -record["epoch"], path, metadata))
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Ignoring incomplete/corrupt best checkpoint {path}: {error}", flush=True)
    if not valid:
        return None
    _, _, path, metadata = max(valid, key=lambda row: (row[0], row[1]))
    return path, metadata


def write_bundle(destination, payloads, *, kind, epoch):
    """Write a new generation; an existing completed generation is immutable."""
    import torch

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (destination / MARKER).exists():
        try:
            validate_bundle(destination)
        except (OSError, ValueError, KeyError, TypeError):
            quarantine = destination.with_name(f".quarantined-{destination.name}-{uuid.uuid4().hex}")
            destination.rename(quarantine)
            print(f"Preserved corrupt checkpoint in {quarantine}", flush=True)
        else:
            raise FileExistsError(f"Refusing to overwrite completed checkpoint: {destination}")
    # A previous interrupted attempt at this exact epoch is safe to discard.
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir()
    record = {"schema_version": 1, "kind": kind, "epoch": epoch, "files": {}}
    for name, value in payloads.items():
        path = destination / name
        if name.endswith(".json"):
            atomic_json(path, value)
        else:
            with path.open("wb") as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
        record["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    # Verify a second read before publishing the completion marker.
    for name, info in record["files"].items():
        if sha256(destination / name) != info["sha256"]:
            raise OSError(f"Checkpoint read-back failed: {destination / name}")
    atomic_json(destination / MARKER, record)
    validate_bundle(destination, verify_hashes=False, expected_kind=kind)
    return destination


def prune_epochs(save_dir, keep):
    if keep < 2:
        raise ValueError("Transactional checkpoints must retain at least two epochs")
    # Prune only after a new completed generation is durable to the mounted FS.
    valid = completed_epochs(save_dir, verify_hashes=True)
    for _, path in valid[:-keep]:
        shutil.rmtree(path)
        print(f"Removed old completed checkpoint {path}", flush=True)


def prune_best(save_dir, keep=2):
    valid = []
    for path in (Path(save_dir) / "best_checkpoints").glob("epoch_*"):
        try:
            validate_bundle(path, verify_hashes=True, expected_kind="best")
            metadata = json.loads((path / "metadata.json").read_text())
            valid.append((metadata["val_accuracy"], -metadata["epoch"], path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    valid.sort(key=lambda row: (row[0], row[1]), reverse=True)
    for _, _, path in valid[keep:]:
        shutil.rmtree(path)
