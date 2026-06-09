#!/usr/bin/env bash
# Run CPU-only DINO probe classifiers from cached activations.
#
# Usage:
#   sbatch dino/probe_cpu_slurm.sh --activations /path/to/activations --output dino_results/model/sparse_logistic --classifier sparse_logistic
#
# Required arguments:
#   --activations DIR   Cached activation directory containing one subdirectory per experiment.
#   --output DIR        Directory where results.json will be written.
#
# Optional arguments:
#   --classifier NAME   Probe classifier to run. Use sparse_logistic for L1-sparse logistic
#                       probes, mlp for CPU MLP probes, or logistic_regression for a CPU
#                       linear baseline. Defaults to sparse_logistic.
#   --max-iter N        Maximum optimizer iterations. Increase when sklearn reports
#                       convergence warnings; decrease for faster approximate reruns.
#                       Defaults to 1000.
#   --no-charts         Skip cross-model chart regeneration after the probe finishes.
#
# Any additional arguments are forwarded to python -m dino.run.
#SBATCH --job-name=mechaptcha-dino-cpu-probe
#SBATCH --account=nlp
#SBATCH --partition=john
#SBATCH --output=dino/logs/%x_%j.out
#SBATCH --error=dino/logs/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p dino/logs

export PYTHONUNBUFFERED=1
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export HF_HOME="${HF_HOME:-/tmp/hf-home}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

ACTIVATIONS=""
OUTPUT=""
CLASSIFIER="sparse_logistic"
MAX_ITER="1000"
RUN_CHARTS=1
FORWARDED_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --activations)
      ACTIVATIONS="$2"
      shift 2
      ;;
    --activations=*)
      ACTIVATIONS="${1#--activations=}"
      shift
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --output=*)
      OUTPUT="${1#--output=}"
      shift
      ;;
    --classifier)
      CLASSIFIER="$2"
      shift 2
      ;;
    --classifier=*)
      CLASSIFIER="${1#--classifier=}"
      shift
      ;;
    --max-iter)
      MAX_ITER="$2"
      shift 2
      ;;
    --max-iter=*)
      MAX_ITER="${1#--max-iter=}"
      shift
      ;;
    --no-charts)
      RUN_CHARTS=0
      shift
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$ACTIVATIONS" ]]; then
  echo "Error: --activations DIR is required." >&2
  exit 2
fi
if [[ -z "$OUTPUT" ]]; then
  echo "Error: --output DIR is required." >&2
  exit 2
fi

echo "Activations -> ${ACTIVATIONS}"
echo "Output      -> ${OUTPUT}"
echo "Classifier  -> ${CLASSIFIER}"
echo "Max iter    -> ${MAX_ITER}"

uv run python -m dino.run --probe-only \
  --classifier "$CLASSIFIER" \
  --max-iter "$MAX_ITER" \
  --activations "$ACTIVATIONS" \
  --output "$OUTPUT" \
  --no-plot \
  "${FORWARDED_ARGS[@]}"

if [[ "$RUN_CHARTS" -eq 1 ]]; then
  echo "Regenerating charts..."
  uv run python charts/make_charts.py
  echo "Charts done."
fi
