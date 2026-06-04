#!/usr/bin/env bash
#SBATCH --job-name=mechaptcha-cnn-probe
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --output=probe/logs/%x_%j.out
#SBATCH --error=probe/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Extract CNN activations and train linear + MLP probes.
# Activations go to NFS scratch (MECHAPTCHA_CACHE) so --probe-only works across jobs.
# Only results.json files and charts are written back to the persistent output dir.
#
# Usage:
#   sbatch probe/run_slurm.sh --checkpoint runs/captcha-cnn/best.pt \
#       --experiments data/experiments/siddharthmb/... --output probe_results
#
#   # Probe only (activations already cached from a previous run):
#   sbatch probe/run_slurm.sh --probe-only --output probe_results

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p probe/logs

export PYTHONUNBUFFERED=1
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export HF_HOME="${HF_HOME:-/tmp/hf-home}"

ACTIVATION_CACHE="${MECHAPTCHA_CACHE:-/nlp/scr/siddharth/mechaptcha/activations}"
CNN_ACTIVATIONS="${ACTIVATION_CACHE}/cnn"
mkdir -p "$CNN_ACTIVATIONS"
echo "Activations -> ${CNN_ACTIVATIONS}"

# Parse --output from forwarded args.
PERSISTENT_OUTPUT="probe_results"
PROBE_ONLY=0
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --output)    PERSISTENT_OUTPUT="${args[i+1]}" ;;
    --output=*)  PERSISTENT_OUTPUT="${args[i]#--output=}" ;;
    --probe-only)PROBE_ONLY=1 ;;
  esac
done

# Probe outputs go to node-local scratch; result JSONs are copied to persistent dir.
SCRATCH="${LOCAL_SCRATCH:-/tmp}/mechaptcha-artifacts/cnn"
mkdir -p "$SCRATCH"
echo "Scratch output  -> ${SCRATCH}  (discarded after job)"

# Strip --output and --activations from forwarded args.
FILTERED_ARGS=()
skip_next=0
for arg in "${args[@]}"; do
  if [[ $skip_next -eq 1 ]]; then skip_next=0; continue; fi
  case "$arg" in
    --output|--activations) skip_next=1 ;;
    --output=*|--activations=*) ;;
    *) FILTERED_ARGS+=("$arg") ;;
  esac
done

# Run linear probes (extraction or probe-only, both with cached activations).
uv run python -m probe.run "${FILTERED_ARGS[@]}" \
  --activations "$CNN_ACTIVATIONS" \
  --output "$SCRATCH"

# MLP and sparse probes on the same cached activations.
for CLF in mlp sparse_logistic; do
  CLF_SCRATCH="${SCRATCH}/${CLF}"
  echo "Running ${CLF} probes -> ${CLF_SCRATCH}"
  uv run python -m probe.run --probe-only --classifier "$CLF" \
    --activations "$CNN_ACTIVATIONS" \
    --output "$CLF_SCRATCH" \
    --no-plot
  echo "${CLF} probes done."
done

# Copy result files to persistent output dir.
for f in results.json; do
  [[ -f "${SCRATCH}/${f}" ]] && cp "${SCRATCH}/${f}" "${PERSISTENT_OUTPUT}/${f}"
done
for CLF in mlp sparse_logistic; do
  CLF_SCRATCH="${SCRATCH}/${CLF}"
  if [[ -f "${CLF_SCRATCH}/results.json" ]]; then
    mkdir -p "${PERSISTENT_OUTPUT}/${CLF}"
    cp "${CLF_SCRATCH}/results.json" "${PERSISTENT_OUTPUT}/${CLF}/results.json"
  fi
done
echo "Persisted results -> ${PERSISTENT_OUTPUT}/"

# Regenerate cross-model comparison charts.
echo "Regenerating charts..."
uv run python charts/make_charts.py
echo "Charts done."
