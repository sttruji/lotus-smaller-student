#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
CONFIG="${CONFIG:-args/research/distill_3b_to_looped_1b.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PYTHON="${PYTHON:-python}"
# Logging is optional; the default needs no W&B credentials.
export WANDB_MODE="${WANDB_MODE:-disabled}"
if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo 'NPROC_PER_NODE must be a positive integer' >&2
  exit 2
fi
EXTRA_ARGS=("$@")
if [[ -n "${RUN_NAME:-}" ]]; then
  EXTRA_ARGS+=(--name "$RUN_NAME")
fi
# --name only affects the training entry point, so preflight forwards --set only.
CHECK_ARGS=()
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
  if [[ "${EXTRA_ARGS[i]}" == --set ]]; then
    CHECK_ARGS+=(--set "${EXTRA_ARGS[i+1]}")
    ((i+=1))
  fi
done
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  "$PYTHON" scripts/preflight_distillation.py "$CONFIG" --config-only "${CHECK_ARGS[@]}"
else
  "$PYTHON" scripts/preflight_distillation.py "$CONFIG" "${CHECK_ARGS[@]}"
fi
CMD=("$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE"
     scripts/run.py "$CONFIG" "${EXTRA_ARGS[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'Training command: '
  printf '%q ' "${CMD[@]}"
  printf '\n'
else
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "GPU training requires a CUDA PyTorch environment"'
  exec "${CMD[@]}"
fi
