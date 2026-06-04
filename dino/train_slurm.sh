#!/usr/bin/env bash
#SBATCH --job-name=mechaptcha-dino-train
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

if [[ -z "${HF_TOKEN:-}" && -r "$HOME/.shell/secrets/hf_token_write" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$HOME/.shell/secrets/hf_token_write")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

# LoRA fine-tune the DINOv2 backbone on the CAPTCHA transcription task.
# Pass any dino.train flags through, e.g. --model-name facebook/dinov2-base.
uv run python -m dino.train "$@"
