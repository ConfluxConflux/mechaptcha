#!/usr/bin/env bash
#SBATCH --job-name=mechaptcha-dino-probe
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --output=dino/logs/%x_%j.out
#SBATCH --error=dino/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p dino/logs

export PYTHONUNBUFFERED=1
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export HF_HOME="${HF_HOME:-/tmp/hf-home}"

# Extract DINOv2 activations from the paired experiments and train probes.
# Pass dino.run flags through, e.g.:
#   sbatch dino/run_slurm.sh --checkpoint dino_runs/dinov2-small/best.pt \
#     --experiments data/experiments/siddharthmb/2026.mechaptcha.linear-probe-experiments-giant-20260525 \
#     --output dino_results/dinov2-small
uv run python -m dino.run "$@"
