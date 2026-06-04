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

# Extract activations, run linear probes, then automatically run MLP probes on
# the same activations. Pass dino.run flags through for the linear probe stage.
uv run python -m dino.run "$@"

# Parse --output from the forwarded args to locate the activations for MLP.
OUTPUT=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --output)   OUTPUT="${args[i+1]}" ;;
    --output=*) OUTPUT="${args[i]#--output=}" ;;
  esac
done

if [[ -n "$OUTPUT" ]]; then
  ACTIVATIONS="${OUTPUT}/activations"
  MLP_OUTPUT="${OUTPUT}/mlp"
  echo "Running MLP probes -> ${MLP_OUTPUT}"
  uv run python -m dino.run --probe-only --classifier mlp \
    --activations "$ACTIVATIONS" \
    --output "$MLP_OUTPUT" \
    --no-plot
  echo "MLP probes done -> ${MLP_OUTPUT}/results.json"
fi

# Regenerate cross-model comparison charts now that this run's results are in.
echo "Regenerating charts..."
uv run python charts/make_charts.py
echo "Charts done."
