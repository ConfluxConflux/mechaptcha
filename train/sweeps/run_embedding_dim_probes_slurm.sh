#!/usr/bin/env bash
#SBATCH --job-name=mechaptcha-probes
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

mkdir -p train/logs probe_results/sweeps

export PYTHONUNBUFFERED=1
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"

EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-data/experiments/siddharthmb/2026.mechaptcha.linear-probe-experiments-giant-20260525}"

declare -A RUNS_BY_DIM=(
  [64]="runs/captcha-cnn/20260525-154713_slurm-15555201"
  [128]="runs/captcha-cnn/20260525-155019_slurm-15555201"
  [256]="runs/captcha-cnn/20260525-155304_slurm-15555201"
  [384]="runs/captcha-cnn/20260525-155601_slurm-15555201"
  [512]="runs/captcha-cnn/20260525-155917_slurm-15555201"
  [1024]="runs/captcha-cnn/20260525-160221_slurm-15555201"
  [2048]="runs/captcha-cnn/20260525-160556_slurm-15555201"
)

for dim in 64 128 256 384 512 1024 2048; do
  run_dir="${RUNS_BY_DIM[$dim]}"
  checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    checkpoint="$run_dir/checkpoints/last.pt"
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint for embedding_dim=$dim under $run_dir" >&2
    exit 1
  fi

  output_dir="probe_results/sweeps/$dim"
  activations_dir="$output_dir/activations"
  mkdir -p "$output_dir" "$activations_dir"

  echo "=== embedding_dim=$dim ==="
  echo "checkpoint: $checkpoint"
  echo "output: $output_dir"
  uv run python -m probe.run \
    --checkpoint "$checkpoint" \
    --experiments "$EXPERIMENTS_ROOT" \
    --activations "$activations_dir" \
    --output "$output_dir"
done
