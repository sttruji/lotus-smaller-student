# Persistent Google Colab run notebook

[Open in Colab](https://colab.research.google.com/github/sttruji/lotus-smaller-student/blob/main/notebooks/colab_training.ipynb)
or download [colab_training.ipynb](../notebooks/colab_training.ipynb) and upload a
copy to your school Colab account. Save the copy in Drive so your cell-1 settings
survive a new runtime.

## The seven cells

| Cell | Action |
| --- | --- |
| 1 | Set the dedicated Drive path, run ID and batch profile; clone the public repository |
| 2 | Install CUDA PyTorch and dependencies into an isolated, ephemeral virtualenv |
| 3 | Mount Drive, read HF_TOKEN securely, restore an existing run's code/packages, configure paths |
| 4 | Download and verify the original augmented GSM8K splits into Drive once |
| 5 | Train, or automatically resume the latest verified completed epoch |
| 6 | Export the completed student and evaluate it alongside optional published CoT baselines |
| 7 | Display baseline/validation plots and save PNGs and CSV tables to Drive |

First session: **1 → 2 → 3 → 4 → 5**, followed by **6 → 7** when training finishes.
Every new training session: **1 → 2 → 3 → 5**. Data is verified locally on Drive;
cell 5 does not download it again. Re-run cell 4 only to prepare or repair data.

## Before starting

- Select a GPU runtime with high host RAM. Preflight requires native BF16,
  at least 35 GiB VRAM, 30 GiB host RAM total and 20 GiB host RAM available.
  A100 40/80 GB runtimes meet the GPU gate. T4 and 24 GiB L4 runtimes do not.
  These are planning gates; the full training recipe has not been profiled on Colab.
- Add a read-only **HF_TOKEN** in Colab Secrets and grant the notebook access.
  The account must have accepted the access conditions for Llama 3.2 1B and 3B.
  Cell 3 offers a hidden prompt if a secret is unavailable. Tokens are inherited
  by subprocesses and are never serialized in the run plan or saved to Drive.
- Choose your own dedicated **DRIVE_ROOT** inside mounted MyDrive. Allow
  approximately 60 GiB free for Hub caches, two full epoch snapshots, two best
  generations, an inference export and temporary writes. Interrupted/quarantined
  artifacts can require more. Check the school's actual Drive quota; mounted
  filesystem free-space reports do not establish cloud-side quota.
- Keep only one active runtime per **RUN_ID**. The helper prevents duplicate
  launches inside one VM; it does not provide a lock across independent Colab VMs.

The default is **2/64**, effective batch 128. **8/16** is selectable in cell 1
after profiling the full stage-6 workload. There is no automatic precision,
batch, teacher-size or dataset reduction. Optional train/validation limits in
cell 1 are for a separate small pilot with a new run ID.

A bounded first pilot can use `RUN_ID="gsm-distill-pilot"`,
`TRAIN_MAX_EXAMPLES=128`, `VAL_MAX_EXAMPLES=8` and `EPOCHS=7`. Seven epochs reach
stage 6 without changing the curriculum. This first-N sample does not establish
worst-case sequence memory or model accuracy. Return to full-split settings and
a new run ID for the actual experiment.

The isolated environment uses official PyTorch 2.7.1 CUDA 12.6 wheels plus
[requirements-colab.txt](../requirements-colab.txt). It leaves the notebook
kernel's installed packages intact and needs no kernel restart. The first run
records a package freeze and Python major/minor version. Cell 3 restores that
freeze and the recorded Git commit before resuming. A different Python version
requires selecting a compatible runtime. CUDA driver and GPU differences can
still affect numerical reproducibility.

## Persistent layout

~~~text
MyDrive/lotus-research/                 # DRIVE_ROOT; editable in cell 1
  cache/hub/                           # Downloaded HF snapshots
  datasets/gsm8k-icot-e06a32ee/
    gsm_train.json                     # 385,620 examples
    gsm_valid.json                     # 500 examples
    gsm_test.json                      # 1,319 examples
    dataset_manifest.json              # Source and prepared-data SHA-256
  runs/gsm-distill-b128-mb2-seed0/       # RUN_ID; editable before first launch
    run_plan.json                      # Recipe, code, versions, pinned Hub revisions
    requirements.freeze.txt
    run_manifest.json
    manifest_resume_N.json
    checkpoint_N/
      model.pt                         # Student and training projection weights
      optimizer.pt
      training_state.pt                # Epoch, scheduler, RNG, best validation score
      COMPLETED.json                   # Written last; file sizes and SHA-256
    best_checkpoints/epoch_N_GENERATION/
      model.pt
      metadata.json
      COMPLETED.json
    history/epoch_N.json
    logs/
    exports/
    evaluation/
      latest_comparison.json
      MODEL_SIGNATURE/
        metrics.json
        predictions.json
        EVALUATION_COMPLETED.json
    results/
      comparison.csv
      baseline_comparison.png
      training_history.csv
      training_history.png
~~~

The checkout, virtualenv, session config and Arrow preprocessing cache live under
`/content` and can disappear between sessions. All essential training artifacts
live under the declared Drive root.

## Checkpoint and resume contract

The notebook opts into `transactional_checkpoints=true`; existing command-line
runs retain their original checkpoint layout. The Colab protocol requires one
GPU, optimizer continuity, saving every epoch and retaining at least two epochs.

After an epoch's **training and validation** finish, the trainer writes model,
optimizer and training state, closes and flushes each file, checks its SHA-256 on
read-back, then publishes `COMPLETED.json`. The previous completed epochs remain
available throughout the write. Old completed snapshots are pruned only after
the new generation is complete. Two best validation generations are kept
separately, using the same protocol; selection remains restricted to the fully
latent stage.

Cell 5 verifies completed checkpoint file sizes and hashes and chooses the
highest epoch number that passes. It restores model/projection weights, AdamW,
scheduler and learning rate, RNG and curriculum progress. Corrupt/incomplete
newer directories are ignored. A corrupt generation at a repeated epoch is
quarantined rather than overwritten. If completed work exists but no checkpoint
passes, the notebook stops instead of silently starting a fresh experiment.
An interrupted first epoch can restart from the same pinned initialization.

The recipe, split hashes, Git commit, Python version and core library versions
must match the saved run plan. Changing the batch profile or training population
requires a new run ID. Test evaluation limits and the baseline toggle can change
without changing training; they produce separate evaluation cache signatures.
Completed training is detected before starting another training process.

**Recovery is at epoch boundaries, not within an epoch.** The unfinished epoch
is repeated. A full fully latent epoch may exceed a Colab session's lifetime;
this notebook cannot make an epoch longer than the allocated session finish.
Use a separate bounded pilot to estimate the epoch duration, then choose compute
with sufficient session length. No keep-alive or resource-limit workaround is used.

Drive mounting does not guarantee remote synchronization: completion markers
and checksums establish the mounted filesystem's contents, not a cloud durability
acknowledgment. Google also documents runtime variability and Drive I/O limits.
See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Recovering from a cell-2 ensurepip failure

A user reported that the original `venv.EnvBuilder(with_pip=True)` failed in
Colab's Python 3.13 `ensurepip` step, before dependency installation. That attempt
can leave `bin/python` behind without a working pip. Checking only whether the
Python executable exists is insufficient for retrying setup.

Cell 2 now creates/reuses the environment with `with_pip=False, clear=False`,
then uses the notebook interpreter's pip with `--python` to install pip into the
target environment. This avoids `ensurepip` and completes the partial environment.
See [pip's documented support for pip-free environments](https://pip.pypa.io/en/stable/topics/python-option/).

A saved notebook copy does not update its cells when cell 1 fetches new repo code.
For an existing copy, replace the original `if not PYTHON.is_file()` block and the
following pip-upgrade command in cell 2 with:

~~~python
venv.EnvBuilder(with_pip=False, clear=False).create(VENV_DIR)
subprocess.run([sys.executable, "-m", "pip", "--python", str(PYTHON),
                "install", "--upgrade", "pip"], check=True)
~~~

Re-run cell 2, then continue with cell 3. No manual deletion of the partial
environment is needed. This bootstrap relies on the notebook interpreter having
pip 22.3 or newer, which introduced `--python`.

## Evaluation and comparison

Cell 6 requires all planned training epochs and a verified fully latent best
checkpoint. It exports student-only weights and evaluates models sequentially
on the same ordered GSM8K test population, with greedy decoding, BF16, batch one
and an explicit default output cap of 512. Optional `EVAL_MAX_EXAMPLES` evaluates
the first N test rows and labels the result as a subset; it does not alter training.

Default baselines are the pinned published **CoT 1B initialization** and
**CoT 3B teacher** used in this run. These references have not received matching
continued training. They establish a useful starting comparison, not a controlled
KD ablation. For controlled baselines, train the matched configurations described
in [RESEARCH_SCAFFOLD.md](../RESEARCH_SCAFFOLD.md) separately and add their complete
schema-2 report paths to `BASELINE_METRICS` in cell 1, then re-run cells 3 and 7.

Evaluation writes predictions, schema-2 metrics and completion/checksum records.
Re-running cell 6 skips completed models only when the model, population, output
cap, GPU, code and environment match. An interrupted individual model evaluation
restarts that model; earlier finished model evaluations remain reusable.

Plots reject failed/incomplete reports, untrained latent-token baselines, and
mismatched ordered populations, output caps, decoding, dtype, batch size or scoring
policies. Latency plots are omitted across different GPUs or library versions.
Accuracy, truncation and generation time appear together; full tables also include
tokens, peak allocation and parameter counts. Validation history uses the training
parser and changing curriculum stages; it is separate from standalone test accuracy.
See [EVALUATION.md](EVALUATION.md) for accounting details.

## Validation status

The notebook is delivered without saved outputs or credentials. Its JSON structure,
cell order and Python syntax are reviewed locally. No school-account, Drive-mount,
CUDA training, Colab dependency installation or end-to-end resume run has been
performed. The earlier 54 CPU tests predate this notebook and do not validate the
new persistence protocol. GPU and Drive behavior still need an actual pilot.

Relevant implementation: [colab_runner.py](../scripts/colab_runner.py),
[persistent_checkpoints.py](../scripts/persistent_checkpoints.py), [run.py](../scripts/run.py),
and [dataset.py](../scripts/dataset.py).
