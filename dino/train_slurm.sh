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

# Parse --output and --experiments from the forwarded args so we can derive
# the probe job's checkpoint/output paths and auto-submit it as a dependent.
CHECKPOINT=""
EXPERIMENTS="${DINO_EXPERIMENTS:-data/experiments/siddharthmb/2026.mechaptcha.linear-probe-experiments-giant-20260525}"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --output)       CHECKPOINT="${args[i+1]}" ;;
    --output=*)     CHECKPOINT="${args[i]#--output=}" ;;
    --experiments)  EXPERIMENTS="${args[i+1]}" ;;
    --experiments=*)EXPERIMENTS="${args[i]#--experiments=}" ;;
  esac
done

# Derive the probe output dir from the checkpoint path:
#   dino_runs/<slug>/best.pt  ->  dino_results/<slug>
if [[ -n "$CHECKPOINT" ]]; then
  SLUG="$(basename "$(dirname "$CHECKPOINT")")"
  PROBE_OUTPUT="dino_results/${SLUG}"
  PROBE_PARTITION="${SLURM_JOB_PARTITION:-jag-standard},sc-loprio"

  PROBE_JOB=$(sbatch \
    --partition="$PROBE_PARTITION" \
    --parsable \
    --dependency="afterok:${SLURM_JOB_ID}" \
    dino/run_slurm.sh \
      --checkpoint "$CHECKPOINT" \
      --experiments "$EXPERIMENTS" \
      --output "$PROBE_OUTPUT")
  echo "Queued probe job ${PROBE_JOB} (after train job ${SLURM_JOB_ID}) -> ${PROBE_OUTPUT}"
else
  echo "Warning: --output not specified; skipping auto-probe submission."
fi

# LoRA fine-tune the backbone on the CAPTCHA transcription task.
uv run python -m dino.train "$@"
