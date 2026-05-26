#!/usr/bin/env bash
#SBATCH --job-name=mechaptcha-exp-hf
#SBATCH --account=nlp
#SBATCH --partition=john
#SBATCH --output=dataset-generation/logs/%x_%j.out
#SBATCH --error=dataset-generation/logs/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=21-00:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p dataset-generation/logs

export PYTHONUNBUFFERED=1
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"

if [[ -z "${HF_TOKEN:-}" && -r "$HOME/.shell/secrets/hf_token_write" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$HOME/.shell/secrets/hf_token_write")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

N="${N:-100000}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
HF_REPO_ID="${HF_REPO_ID:-siddharthmb/2026.mechaptcha.linear-probe-experiments-giant-20260525}"

uv run python dataset-generation/build_experiments.py \
  --n "$N" \
  --push-to-hf \
  --hf-repo-id "$HF_REPO_ID" \
  --workers "$WORKERS"
