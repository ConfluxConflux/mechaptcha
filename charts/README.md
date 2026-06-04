# charts

Cross-model comparison figures for the distortion-probe experiment. Regenerate with:

```bash
uv run python charts/make_charts.py
```

The script auto-discovers whichever `results.json` runs exist (CNN + each pretrained-ViT
run under `dino_results/`), so re-running it picks up new backbones as their jobs finish.

## Figures

- **`decodability_vs_depth.png`** — the headline. Mean linear-probe accuracy (distortion
  batch_a vs batch_b, non-control experiments) vs *normalised depth* (0 = input → 1 = logits),
  one line per model. Every model rises to a high plateau through its body, then attenuates
  toward the behaviorally-invariant output but stays well above chance: behavioral invariance
  ≠ representational erasure, across architectures and training signals.
- **`peak_vs_output.png`** — per model, peak decodability across depth vs decodability at the
  final logits. Quantifies the "retained but partially suppressed" story.
- **`heatmap_<model>.png`** — per-model experiment × layer heatmap.

Models compared: the from-scratch **CNN**, **DINOv2** (self-supervised, LoRA-adapted), and
**CLIP** (language-supervised vision tower, LoRA-adapted). See [`dino/`](../dino/README.md).

Original single-run CNN figures: [run figures](../runs/captcha-cnn/20260525-125407_slurm-15554167/figures).
