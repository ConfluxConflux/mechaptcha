#!/usr/bin/env bash
#SBATCH --job-name=mechaptcha-train
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --output=train/logs/%x_%j.out
#SBATCH --error=train/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p train/logs

export PYTHONUNBUFFERED=1
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb-mechaptcha}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb-config}"

if [[ -z "${HF_TOKEN:-}" && -r "$HOME/.shell/secrets/hf_token_write" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(cat "$HOME/.shell/secrets/hf_token_write")"
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NPROC_PER_NODE="$SLURM_GPUS_ON_NODE"
elif [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="${SLURM_GPUS_PER_NODE%%(*}"
else
  NPROC_PER_NODE=1
fi

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  uv run torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train/scripts/train.py "$@"
else
  uv run python train/scripts/train.py "$@"
fi
