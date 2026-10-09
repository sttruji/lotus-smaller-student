"""Drive persistence and orchestration used by notebooks/colab_training.ipynb.

Run with the notebook's isolated Python, not the Colab kernel interpreter.
Credentials are inherited through HF_TOKEN and never serialized to Drive.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

from persistent_checkpoints import atomic_json, best_checkpoint, completed_epochs, sha256, validate_bundle


REPO = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "e06a32ee5e4cd117171daeb4755d2a97ece62761"
DATA_DIR_NAME = "gsm8k-icot-e06a32ee"
SOURCE_SPLITS = {
    "train": (385620, "0a3909a9e7d8d2f7ad6b8c7b5608aa744988d835f9bf874d4cf06ca77df6bf8c"),
    "valid": (500, "04f9bddfe8ac4846571b1e2b612b289c95770b5012fdb34cc36a597b041afb87"),
    "test": (1319, "e464efa2713fbcf423bdbfb482c543a6a086188c3d97c294b1d19a70c2a515e8"),
}


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def context(session_file):
    session = json.loads(Path(session_file).read_text())
    root = Path(session["drive_root"])
    if not os.path.ismount("/content/drive") or not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Drive is not mounted. Run notebook cell 3; refusing to write to ephemeral storage.")
    if not root.is_absolute() or not root.resolve().is_relative_to(Path("/content/drive/MyDrive").resolve()):
        raise ValueError("drive_root must be a dedicated folder inside /content/drive/MyDrive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", session["run_id"]):
        raise ValueError("run_id must be a simple folder name (letters, numbers, _, -, .)")
    root.mkdir(parents=True, exist_ok=True)
    run = root / "runs" / session["run_id"]
    return session, root, run, root / "datasets" / DATA_DIR_NAME


@contextmanager
def local_lock(run):
    """Prevent overlapping launches in this VM; Drive is not a distributed lock."""
    directory = Path("/content/lotus-process-locks")
    directory.mkdir(exist_ok=True)
    with (directory / (digest_object(str(run)) + ".lock")).open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This run is already active in this runtime") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def execute(command, log_path):
    """Stream bounded notebook output and stop the entire torchrun process group."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(str(item) for item in command), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", WANDB_MODE="disabled",
               TOKENIZERS_PARALLELISM="false", TQDM_MININTERVAL="10")
    with log_path.open("a", buffering=1024 * 1024) as log:
        process = subprocess.Popen([str(item) for item in command], cwd=REPO, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors="replace", start_new_session=True)
        last_output = last_flush = 0.0
        try:
            for line in process.stdout:
                log.write(line)
                now = time.monotonic()
                if now - last_output >= 10 or any(word in line.lower() for word in
                        ("checkpoint", "error", "traceback", "accuracy on validation", "restored", "epoch: ")):
                    print(line.rstrip(), flush=True)
                    last_output = now
                if now - last_flush >= 10:
                    log.flush()
                    last_flush = now
            code = process.wait()
            if code:
                raise RuntimeError(f"Command exited {code}. Full log: {log_path}. Rerun the cell to resume completed work.")
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            process.stdout.close()
            log.flush()


def verify_data(directory):
    manifest_path = directory / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Persistent dataset is not ready. Run notebook cell 4 once.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("source_revision") != SOURCE_REVISION or set(manifest.get("splits", {})) != set(SOURCE_SPLITS):
        raise ValueError("Unexpected persistent dataset manifest; use a new dedicated dataset directory")
    for split, (count, source_hash) in SOURCE_SPLITS.items():
        info = manifest["splits"][split]
        path = directory / f"gsm_{split}.json"
        if info["examples"] != count or info["source_sha256"] != source_hash or sha256(path) != info["json_sha256"]:
            raise ValueError(f"Persistent dataset integrity check failed: {path}. Repair with cell 4.")
    return manifest


def prepare_data(session_file):
    _, _, _, directory = context(session_file)
    directory.mkdir(parents=True, exist_ok=True)
    with local_lock(directory):
        try:
            verify_data(directory)
            print(f"All three splits already verified; no download: {directory}")
            return
        except (OSError, ValueError, KeyError):
            pass
        manifest = {"source_revision": SOURCE_REVISION, "splits": {}}
        for split, (expected_count, expected_hash) in SOURCE_SPLITS.items():
            host = "media.githubusercontent.com/media" if split == "train" else "raw.githubusercontent.com"
            url = f"https://{host}/da03/Internalize_CoT_Step_by_Step/{SOURCE_REVISION}/data/gsm8k/{split}.txt"
            raw = directory / f".gsm_{split}.download"
            if not raw.is_file() or sha256(raw) != expected_hash:
                print(f"Downloading pinned {split} split to {directory} ...", flush=True)
                for attempt in range(4):
                    try:
                        request = urllib.request.Request(url, headers={"User-Agent": "lotus-colab/1"})
                        with urllib.request.urlopen(request, timeout=60) as response, raw.open("wb") as output:
                            shutil.copyfileobj(response, output, length=1024 * 1024)
                        if sha256(raw) != expected_hash:
                            raise ValueError(f"Pinned source hash mismatch: {split}")
                        break
                    except (OSError, ValueError):
                        if attempt == 3:
                            raise
                        time.sleep(2 ** attempt)
            rows = []
            with raw.open() as stream:
                for index, line in enumerate(stream):
                    fields = line.split("||")
                    if len(fields) != 2 or "##" not in fields[1]:
                        raise ValueError(f"Malformed source: {split}:{index}")
                    # Same transformation as preprocessing/gsm_icot.py.
                    row = {"question": fields[0], "steps": fields[1].split("##")[0].strip().split(" "),
                           "answer": line.split("##")[-1].strip()}
                    if not row["question"].strip() or not row["answer"]:
                        raise ValueError(f"Empty source fields: {split}:{index}")
                    rows.append(row)
            if len(rows) != expected_count:
                raise ValueError(f"Wrong {split} size: {len(rows)} != {expected_count}")
            destination = directory / f"gsm_{split}.json"
            atomic_json(destination, rows)
            manifest["splits"][split] = {"examples": len(rows), "source_sha256": expected_hash,
                                        "json_sha256": sha256(destination), "source_url": url}
            raw.unlink()
            print(f"Prepared and verified {split}: {len(rows):,} examples", flush=True)
        atomic_json(directory / "dataset_manifest.json", manifest)
        verify_data(directory)
        print(f"Persistent data ready: {directory}")


def hardware(*, training):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select Runtime > Change runtime type > GPU in Colab")
    properties = torch.cuda.get_device_properties(0)
    memory = properties.total_memory / 1024**3
    if properties.major < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe requires native BF16 (Ampere or newer). A T4 is not supported.")
    available = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable"):
            available[key] = int(value.split()[0]) / 1024**2
    info = {"gpu": properties.name, "gpu_gib": memory, "host_gib": available}
    print(json.dumps(info, indent=2), flush=True)
    if training and (memory < 35 or available.get("MemTotal", 0) < 30 or available.get("MemAvailable", 0) < 20):
        raise RuntimeError("Use a GPU with at least 35 GiB VRAM and a high-RAM runtime (30 GiB total, 20 GiB available). "
                           "The full 3B teacher + full-finetuned 1B recipe is not configured for T4/L4 or low host RAM.")
    return info


def environment():
    import importlib.metadata
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "datasets", "accelerate", "huggingface_hub", "numpy", "safetensors")}


def current_commit():
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True).strip():
        raise RuntimeError("Tracked repository files have changed; use a clean checkout before training")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def recipe(session, run, data_dir):
    from configuration import load_hierarchical_yaml
    from preflight_distillation import validate_configuration
    config = load_hierarchical_yaml(REPO / "args/research/distill_3b_to_looped_1b.yaml")
    if (session["microbatch"], session["accumulation"]) not in ((2, 64), (8, 16)):
        raise ValueError("Select the recorded 2/64 or 8/16 batch profile")
    config.update({
        "name": run.name, "save_path": str(run.parent), "num_epochs": session["epochs"],
        "batch_size_training": session["microbatch"], "gradient_accumulation_steps": session["accumulation"],
        "train_max_examples": session.get("train_max_examples"), "val_max_examples": session.get("val_max_examples"),
        "train_path": str(data_dir / "gsm_train.json"), "val_path": str(data_dir / "gsm_valid.json"),
        "test_path": str(data_dir / "gsm_test.json"), "transactional_checkpoints": True,
        "checkpoint_keep_last": 2, "save_every_epochs": 1, "dataset_num_proc": 2,
        "allow_resume_from_checkpoint": False, "resume": 0, "debug": False,
    })
    validate_configuration(config)
    return config


def pin_models(config):
    from huggingface_hub import HfApi
    api = HfApi()
    revisions = {}
    for key in ("model_id", "load_model_path", "teacher_model_path", "teacher_tokenizer_id"):
        repo_id = config[key]
        if repo_id not in revisions:
            revisions[repo_id] = api.model_info(repo_id).sha
    config.update(model_revision=revisions[config["model_id"]],
                  student_revision=revisions[config["load_model_path"]],
                  teacher_revision=revisions[config["teacher_model_path"]],
                  teacher_tokenizer_revision=revisions[config["teacher_tokenizer_id"]])
    return revisions


def train(session_file):
    import yaml
    session, root, run, data_dir = context(session_file)
    with local_lock(run):
        data_manifest = verify_data(data_dir)
        config = recipe(session, run, data_dir)
        commit = current_commit()
        identity = {"recipe": dict(config), "data": data_manifest, "git_commit": commit,
                    "python": list(sys.version_info[:2]), "packages": environment()}
        run.mkdir(parents=True, exist_ok=True)
        plan_path = run / "run_plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if identity != plan["identity"]:
                raise ValueError("Run configuration, data, code or environment changed. Restore the original settings or choose a NEW RUN_ID in cell 1.")
        else:
            if list(run.glob("checkpoint_*")) or (run / "run_manifest.json").exists():
                raise ValueError("Existing training artifacts have no run plan; use a new RUN_ID")
            hardware(training=True)
            revisions = pin_models(config)
            freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
            (run / "requirements.freeze.txt").write_text(freeze)
            plan = {"schema_version": 1, "identity": identity, "training_config": config,
                    "model_revisions": revisions, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            atomic_json(plan_path, plan)
        checkpoints = completed_epochs(run)
        existing = list(run.glob("checkpoint_*"))
        had_completed_work = (any((path / "COMPLETED.json").exists() for path in existing)
                              or any((run / "history").glob("epoch_*.json"))
                              or any(re.fullmatch(r"checkpoint_([2-9]|\d{2,})", path.name) for path in existing))
        if not checkpoints and had_completed_work:
            raise RuntimeError("Checkpoint artifacts exist but none passed integrity checks. Refusing to silently restart. "
                               "Inspect the folder; use a new RUN_ID for a deliberate fresh run.")
        completed = checkpoints[-1][0] if checkpoints else 0
        if completed >= config["num_epochs"]:
            print(f"Training already complete: epoch {completed}/{config['num_epochs']}. Run cell 6.")
            return
        hardware(training=True)
        # The complete record is followed by a CPU deserialization of small trusted state.
        import torch
        config = dict(plan["training_config"])
        if checkpoints:
            path = checkpoints[-1][1]
            state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
            if state["epoch"] != completed or not all(key in state for key in ("rng_state", "lr_scheduler", "best_acc")):
                raise ValueError("Invalid training state in completed checkpoint")
            config.update(resume=completed, load_model_path=str(path))
            print(f"Automatically resuming AFTER completed epoch {completed}: {path}", flush=True)
        else:
            print("Starting from the pinned LOTUS CoT initialization", flush=True)
        session_config = Path("/content/lotus-session-training.yaml")
        session_config.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"Batch {config['batch_size_training']}/{config['gradient_accumulation_steps']} = 128; target {config['num_epochs']} epochs")
        print(f"Persistent run: {run}\nKeep at least 60 GiB free in Drive for caches, snapshots, best weights and exports.", flush=True)
        execute([sys.executable, "scripts/preflight_distillation.py", str(session_config), "--check-models"], run / "logs/preflight.log")
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        execute([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=1",
                 "scripts/run.py", str(session_config)], run / f"logs/train-{stamp}.log")
        latest = completed_epochs(run)
        if not latest or latest[-1][0] < config["num_epochs"]:
            raise RuntimeError("Trainer exited without completing the requested epochs; inspect the persistent log")
        print("Training completed and the final epoch checkpoint passed integrity checks. Run cell 6.")


def snapshot(repo_id, revision):
    from huggingface_hub import HfApi, snapshot_download
    files = HfApi().list_repo_files(repo_id, revision=revision)
    weights = [name for name in files if name.endswith(".safetensors")]
    if not weights:
        weights = [name for name in files if name.startswith("pytorch_model") and name.endswith(".bin")]
    if not weights:
        raise ValueError(f"No HF model weights: {repo_id}@{revision}")
    extras = [name for name in files if name.endswith((".json", ".model", ".txt", ".tiktoken"))]
    return snapshot_download(repo_id, revision=revision, allow_patterns=weights + extras)


def ready_export(path, fingerprint):
    try:
        ready = json.loads((path / "EXPORT_COMPLETED.json").read_text())
        files = ready["files"]
        return (ready["fingerprint"] == fingerprint
                and {"config.json", "tokenizer_config.json", "lotus_config.json"}.issubset(files)
                and any(name.endswith(".safetensors") for name in files)
                and all((path / name).is_file() and sha256(path / name) == checksum for name, checksum in files.items()))
    except (OSError, ValueError, KeyError):
        return False


def evaluate(session_file):
    session, _, run, data_dir = context(session_file)
    with local_lock(run):
        verify_data(data_dir)
        if not (run / "run_plan.json").is_file():
            raise RuntimeError("No training run plan exists. Complete cell 5 before evaluation.")
        plan = json.loads((run / "run_plan.json").read_text())
        if current_commit() != plan["identity"]["git_commit"] or environment() != plan["identity"]["packages"]:
            raise ValueError("Evaluation must use the pinned code and environment; rerun cells 1, 2, 3")
        epochs = completed_epochs(run)
        if not epochs or epochs[-1][0] != plan["training_config"]["num_epochs"]:
            raise RuntimeError("Training is not complete. Run cell 5; evaluation does not silently choose a partially trained model.")
        best = best_checkpoint(run)
        if best is None or not best[1].get("all_cot_removed"):
            raise RuntimeError("No verified fully latent best checkpoint was selected")
        gpu_info = hardware(training=False)
        gpu = {key: gpu_info[key] for key in ("gpu", "gpu_gib")}
        best_path, best_info = best
        export = run / "exports" / best_path.name
        fingerprint = digest_object({"source": validate_bundle(best_path), "commit": current_commit()})
        if not ready_export(export, fingerprint):
            if export.exists():
                shutil.rmtree(export)
            export.parent.mkdir(parents=True, exist_ok=True)
            execute([sys.executable, "scripts/export_student.py", "--checkpoint", best_path,
                     "--manifest", run / "run_manifest.json", "--output-dir", export], run / "logs/export.log")
            atomic_json(export / "EXPORT_COMPLETED.json", {"fingerprint": fingerprint,
                        "files": {str(path.relative_to(export)): sha256(path) for path in export.rglob("*") if path.is_file()}})
        population = json.loads((data_dir / "gsm_test.json").read_text())
        limit = session.get("eval_max_examples")
        if limit is not None:
            if type(limit) is not int or limit < 1 or limit > len(population):
                raise ValueError("eval_max_examples must be null or between 1 and the full test count")
            population = population[:limit]
        eval_dir = run / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        population_path = eval_dir / f"population-{digest_object(population)[:16]}.json"
        if not population_path.exists():
            atomic_json(population_path, population)
        jobs = [("distilled_looped_1b", str(export), False, fingerprint)]
        if session.get("evaluate_baselines", True):
            original = plan["training_config"]
            for label, key in (("published_cot_1b", "load_model_path"), ("published_cot_3b", "teacher_model_path")):
                repo_id = original[key]
                revision = plan["model_revisions"][repo_id]
                jobs.append((label, snapshot(repo_id, revision), True, f"{repo_id}@{revision}"))
        cap = session["eval_max_new_tokens"]
        if type(cap) is not int or cap < 1:
            raise ValueError("eval_max_new_tokens must be positive")
        reports = []
        for label, model, cot, model_fingerprint in jobs:
            spec = {"model": model_fingerprint, "population": sha256(population_path), "max_new_tokens": cap,
                    "gpu": gpu, "packages": environment(), "commit": current_commit(), "cot": cot}
            signature = digest_object(spec)
            destination = eval_dir / f"{label}-{signature[:16]}"
            destination.mkdir(exist_ok=True)
            metrics, preds = destination / "metrics.json", destination / "predictions.json"
            completed_path = destination / "EVALUATION_COMPLETED.json"
            reusable = False
            if completed_path.exists() and metrics.exists() and preds.exists():
                done = json.loads(completed_path.read_text())
                reusable = (done.get("signature") == signature and done["metrics_sha256"] == sha256(metrics)
                            and done["predictions_sha256"] == sha256(preds))
            if not reusable:
                completed_path.unlink(missing_ok=True)
                command = [sys.executable, "scripts/eval.py", "--model_id", model,
                           "--datasets", population_path, "--max_new_tokens", str(cap), "--device", "cuda:0",
                           "--save_metrics", metrics, "--save_preds", preds]
                command += (["--cot", "--c_thought", "1"] if cot else [])
                execute(command, destination / "evaluation.log")
                report = json.loads(metrics.read_text())
                if report["metadata"]["run_status"] != "complete" or not all(
                        value["valid_for_accuracy"] for value in report["datasets"].values()):
                    raise RuntimeError(f"Evaluation report is incomplete or invalid: {metrics}")
                atomic_json(completed_path, {"signature": signature, "label": label, "spec": spec,
                            "metrics_sha256": sha256(metrics), "predictions_sha256": sha256(preds)})
            else:
                print(f"Reusing verified completed evaluation: {label}", flush=True)
            reports.append({"label": label, "path": str(metrics), "sha256": sha256(metrics)})
        atomic_json(eval_dir / "latest_comparison.json", {"reports": reports, "best_checkpoint": str(best_path),
                    "population_examples": len(population), "test_is_subset": limit is not None,
                    "max_new_tokens": cap})
        print(f"Evaluation ready: {eval_dir / 'latest_comparison.json'}. Run cell 7.")


def plot(session_file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    session, _, run, _ = context(session_file)
    if not (run / "evaluation/latest_comparison.json").is_file():
        raise RuntimeError("No completed comparison exists. Run cell 6 before plotting.")
    comparison = json.loads((run / "evaluation/latest_comparison.json").read_text())
    reports = list(comparison["reports"])
    for path in session.get("baseline_metrics", []):
        reports.append({"label": Path(path).parent.name, "path": path, "sha256": sha256(path)})
    rows = []
    expected = None
    for item in reports:
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Report changed after evaluation: {path}")
        report = json.loads(path.read_text())
        meta = report["metadata"]
        if meta.get("schema_version") != 2 or meta.get("run_status") != "complete" or meta.get("untrained_latent_tokens"):
            raise ValueError(f"Only complete schema-2 trained-model reports are comparable: {path}")
        if len(report["datasets"]) != 1:
            raise ValueError("Import one-dataset baseline reports")
        result = next(iter(report["datasets"].values()))
        key = (result["dataset_sha256"], result["total"], meta["max_new_tokens"], meta["decoding"], meta["dtype"],
               meta["batch_size"], meta["answer_extraction"], meta["accuracy_policy"], meta["timing_scope"], meta.get("warmup_examples", 0))
        if not result["valid_for_accuracy"] or result["failed"] or result["unattempted"]:
            raise ValueError(f"Failed/incomplete population cannot be plotted as a baseline: {path}")
        if expected is not None and key != expected:
            raise ValueError(f"Mismatched population/output cap/decoding/dtype/batch: {path}")
        expected = key
        rows.append({"model": item["label"], "accuracy_percent": result["accuracy"] * 100,
                     "examples": result["total"], "failed": result["failed"],
                     "truncation_percent": result["truncation_rate"] * 100,
                     "seconds_per_example": result["avg_inference_time"], "tokens_per_example": result["avg_gen_tokens"],
                     "peak_gpu_allocated_gib": result["peak_mem_alloc_gb"], "gpu": meta["gpu"],
                     "torch": meta["torch_version"], "transformers": meta["transformers_version"],
                     "parameters": meta["unique_parameters"], "metrics_file": str(path)})
    table = pd.DataFrame(rows)
    if table["model"].duplicated().any():
        raise ValueError("Baseline labels are duplicated. Put imported reports in folders with unique baseline names.")
    output = run / "results"
    output.mkdir(exist_ok=True)
    table.to_csv(output / "comparison.csv", index=False)
    print(table.drop(columns=["metrics_file"]).to_string(index=False))
    comparable_hardware = len(table[["gpu", "torch", "transformers"]].drop_duplicates()) == 1
    axes_count = 3 if comparable_hardware else 2
    fig, axes = plt.subplots(1, axes_count, figsize=(6 * axes_count, 5))
    for axis, column, title in ((axes[0], "accuracy_percent", "Test accuracy (%)"),
                                (axes[1], "truncation_percent", "Output truncation (%)")):
        axis.bar(table["model"], table[column])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylim(0, 100)
    if comparable_hardware:
        axes[2].bar(table["model"], table["seconds_per_example"])
        axes[2].set_title("Full generation seconds/example")
        axes[2].tick_params(axis="x", rotation=25)
    else:
        print("Latency chart omitted: reports use different GPUs or library versions.")
    fig.suptitle(f"{'Test SUBSET' if comparison['test_is_subset'] else 'GSM8K test'} — {comparison['population_examples']} examples, cap {comparison['max_new_tokens']}")
    fig.tight_layout()
    fig.savefig(output / "baseline_comparison.png", dpi=150)
    plt.close(fig)
    history = [json.loads(path.read_text()) for path in (run / "history").glob("epoch_*.json")]
    history.sort(key=lambda row: row["epoch"])
    if history:
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        fig, axis = plt.subplots(figsize=(9, 4))
        axis.plot([row["epoch"] for row in history], [None if row["val_accuracy"] is None else 100 * row["val_accuracy"] for row in history], marker="o")
        axis.set(xlabel="Completed epoch", ylabel="Validation accuracy (%)", ylim=(0, 100),
                 title="Curriculum validation (training parser; stages differ)")
        fig.tight_layout()
        fig.savefig(output / "training_history.png", dpi=150)
        plt.close(fig)
    print(f"Saved plots and CSVs to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-data", "train", "evaluate", "plot"))
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    {"prepare-data": prepare_data, "train": train, "evaluate": evaluate, "plot": plot}[args.action](args.session)


if __name__ == "__main__":
    main()
