# dino — generalizing the probe result to pretrained ViT backbones

Tests whether the original finding — *behavioral invariance ≠ representational
erasure* — holds for models we did **not** train from scratch. Instead of the
custom `CaptchaCNN`, we take a pretrained ViT, adapt it to the CAPTCHA
transcription task with **LoRA**, and probe its intermediate blocks.

Two backbone families are supported (set `--backbone`):
- **`dinov2`** — self-supervised (`facebook/dinov2-{small,base,large}`), LoRA on query/value
- **`clip`** — language-supervised vision tower (`openai/clip-vit-{base-patch16,large-patch14}`),
  LoRA on q_proj/v_proj

They are structurally identical for our purposes (ViTs exposing per-block hidden states with a
CLS token at index 0); only loading, input normalisation, and LoRA target names differ. Adding
CLIP required no change to the probe pipeline.

## Why LoRA (and not a strictly frozen backbone)

A strictly frozen backbone never feels any pressure to ignore the distortion, so
finding the distortion in its features is trivial and says nothing about the
thesis. LoRA lets the transcription loss reshape the backbone's behavior — and its
intermediate representations — cheaply, *without training from scratch*. The
resulting model is **behaviorally invariant** (reads batch_a ≈ batch_b); the
question is then whether the distortion is still linearly decodable across depth.

## Pipeline

The downstream probe machinery is reused verbatim — `dino/extract.py` writes the
same `{split}_{batch}_{layer}.npy` layout the CNN produces, so `probe.fit`,
`probe.results`, and `probe.plot` work unchanged. Only the activation source and
backbone-loading differ.

```
dino/config.py   DinoConfig (backbone id, LoRA params, token reduction)
dino/model.py    DinoCaptchaModel = DINOv2 + LoRA + 5 character heads
dino/train.py    LoRA fine-tune on the transcription set (jacobcohen/mechaptcha)
dino/extract.py  per-block activations + per-batch transcription accuracy
dino/run.py      extract -> probe -> plot  (reuses probe.*)
```

Layer names: `input` (grayscale raw-pixel baseline, matches the CNN), `block_0` …
`block_{L-1}` (token-reduced hidden states; `mean`-pooled patch tokens ≈ the CNN's
global-avg-pool), `embedding` (pooled CLS the heads read), `logits`.

## Usage

```bash
# 1. LoRA fine-tune the backbone (GPU; via Slurm)
#    train_slurm.sh auto-submits the probe job after training completes.
sbatch dino/train_slurm.sh --model-name facebook/dinov2-small \
    --train-size 50000 --epochs 6 --output dino_runs/dinov2-small-lora/best.pt

# 2. Extract activations + probe + plot (if running manually)
EXP=data/experiments/siddharthmb/2026.mechaptcha.linear-probe-experiments-giant-20260525
sbatch dino/run_slurm.sh --checkpoint dino_runs/dinov2-small-lora/best.pt \
    --experiments "$EXP" --output dino_results/dinov2-small-lora

# Re-probe existing activations with a different classifier
uv run python -m dino.run --probe-only --classifier mlp \
    --activations dino_results/dinov2-small-lora/activations --output dino_results/dinov2-small-lora/mlp
```

## Outputs (under `--output`)

- `results.json` — `experiment -> layer -> {train_acc, test_acc}` (same schema as the CNN)
- `transcription_accuracy.json` — per-batch seq/char accuracy = the behavioral-invariance check
- `heatmap.png`, `chart_lines.png`

Compare `dino_results/.../results.json` against the CNN's `probe_results/full/results.json`.
