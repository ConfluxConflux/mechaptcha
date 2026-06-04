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

# Activation files are large but need to survive between jobs so --probe-only works.
# They live in NFS scratch (MECHAPTCHA_CACHE), not the repo and not node-local scratch.
# Override by setting MECHAPTCHA_CACHE in the environment before submitting.
ACTIVATION_CACHE="${MECHAPTCHA_CACHE:-/nlp/scr/siddharth/mechaptcha/activations}"

# Parse --output and --activations from forwarded args.
PERSISTENT_OUTPUT=""
CALLER_ACTIVATIONS=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --output)       PERSISTENT_OUTPUT="${args[i+1]}" ;;
    --output=*)     PERSISTENT_OUTPUT="${args[i]#--output=}" ;;
    --activations)  CALLER_ACTIVATIONS="${args[i+1]}" ;;
    --activations=*)CALLER_ACTIVATIONS="${args[i]#--activations=}" ;;
  esac
done

# Activations go to the NFS cache (persistent, reusable with --probe-only).
# Caller can override with --activations if they want a different location.
if [[ -n "$CALLER_ACTIVATIONS" ]]; then
  ACTIVATIONS="$CALLER_ACTIVATIONS"
elif [[ -n "$PERSISTENT_OUTPUT" ]]; then
  SLUG="$(basename "$PERSISTENT_OUTPUT")"
  ACTIVATIONS="${ACTIVATION_CACHE}/${SLUG}"
else
  ACTIVATIONS="${ACTIVATION_CACHE}/run-$$"
fi
mkdir -p "$ACTIVATIONS"
echo "Activations -> ${ACTIVATIONS}"

# Probe outputs (results JSON, charts) go to node-local scratch; only the small
# result files are copied to the persistent output dir at the end.
SCRATCH="${LOCAL_SCRATCH:-/tmp}/mechaptcha-artifacts"
if [[ -n "$PERSISTENT_OUTPUT" ]]; then
  SLUG="$(basename "$PERSISTENT_OUTPUT")"
  SCRATCH_OUTPUT="${SCRATCH}/${SLUG}"
else
  SCRATCH_OUTPUT="${SCRATCH}/run-$$"
fi
mkdir -p "$SCRATCH_OUTPUT"
echo "Scratch output  -> ${SCRATCH_OUTPUT}  (discarded after job)"

# Strip --output from forwarded args (we replace it with scratch path).
FILTERED_ARGS=()
skip_next=0
for arg in "${args[@]}"; do
  if [[ $skip_next -eq 1 ]]; then skip_next=0; continue; fi
  case "$arg" in
    --output)   skip_next=1 ;;
    --output=*) ;;
    *)          FILTERED_ARGS+=("$arg") ;;
  esac
done

# Run extraction + linear probes.
uv run python -m dino.run "${FILTERED_ARGS[@]}" \
  --output "$SCRATCH_OUTPUT" \
  --activations "$ACTIVATIONS"

# Run MLP and sparse probes on the same (cached) activations.
for CLF in mlp sparse_logistic; do
  CLF_SCRATCH="${SCRATCH_OUTPUT}/${CLF}"
  echo "Running ${CLF} probes -> ${CLF_SCRATCH}"
  uv run python -m dino.run --probe-only --classifier "$CLF" \
    --activations "$ACTIVATIONS" \
    --output "$CLF_SCRATCH" \
    --no-plot
  echo "${CLF} probes done."
done

# Copy small result files to the persistent output dir.
if [[ -n "$PERSISTENT_OUTPUT" ]]; then
  for f in results.json transcription_accuracy.json; do
    [[ -f "${SCRATCH_OUTPUT}/${f}" ]] && cp "${SCRATCH_OUTPUT}/${f}" "${PERSISTENT_OUTPUT}/${f}"
  done
  for CLF in mlp sparse_logistic; do
    CLF_SCRATCH="${SCRATCH_OUTPUT}/${CLF}"
    if [[ -f "${CLF_SCRATCH}/results.json" ]]; then
      mkdir -p "${PERSISTENT_OUTPUT}/${CLF}"
      cp "${CLF_SCRATCH}/results.json" "${PERSISTENT_OUTPUT}/${CLF}/results.json"
    fi
  done
  echo "Persisted results -> ${PERSISTENT_OUTPUT}/"
fi

# Regenerate cross-model comparison charts.
echo "Regenerating charts..."
uv run python charts/make_charts.py
echo "Charts done."
