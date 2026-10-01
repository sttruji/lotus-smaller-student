#!/usr/bin/env bash
#SBATCH --job-name=lotus-distill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=lotus-distill-%j.log
set -euo pipefail
# Submit from the repository root, in an environment with the CUDA stack and
# requirements.txt already installed. Set account/partition via sbatch flags.
cd "${LOTUS_REPO_DIR:-$SLURM_SUBMIT_DIR}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
exec bash ./launch_distillation.sh "$@"
