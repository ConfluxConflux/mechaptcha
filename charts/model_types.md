# Chart Model Types

This document defines the model labels used in `charts/` and how each model was trained to solve the CAPTCHA transcription task.

## Task Setup

All task-trained models predict a 5-character CAPTCHA string. Architecturally, the ViT/CLIP/DINO variants use a pretrained vision backbone plus one linear classification head per character position. Each head predicts one character class, and the five heads together produce the transcription.

The chart probes are separate from the transcription task. They train classifiers on saved intermediate activations to distinguish paired image batches, so a model can have highly decodable visual features even when its CAPTCHA transcription accuracy is poor.

## Model Summary

| Chart label | Source results | Backbone | Task training | What was trainable for CAPTCHA transcription? | Task interpretation |
| --- | --- | --- | --- | --- | --- |
| `cnn` | `probe_results/full/` | Custom `CaptchaCNN` | Trained from scratch on CAPTCHA transcription | All CNN convolution blocks, embedding layer, and character heads | Native task model for this project. |
| `clip-b` | `dino_results/clip-vit-base/` | `openai/clip-vit-base-patch16` vision tower | No CAPTCHA task training | Nothing meaningful for the task; CAPTCHA heads are newly initialized and untrained | Use only as a pretrained visual representation baseline. Its transcription logits are expected to be garbage. |
| `clip-b-frozen` | `dino_results/clip-vit-base-frozen/` | `openai/clip-vit-base-patch16` vision tower | Heads-only CAPTCHA training | Character heads only; CLIP vision tower stays frozen | Tests whether frozen pretrained CLIP-B vision features are enough for CAPTCHA reading with a lightweight readout. |
| `clip-b-lora` | `dino_results/clip-vit-base-lora/` | `openai/clip-vit-base-patch16` vision tower | LoRA fine-tuned on CAPTCHA transcription | LoRA adapters plus character heads | Task-adapted CLIP baseline. Validation sequence accuracy recorded as `0.921`. |
| `dinov2-b-frozen` | `dino_results/dinov2-base-frozen/` | `facebook/dinov2-base` | Heads-only CAPTCHA training | Character heads only; DINOv2 backbone stays frozen | Base-sized analogue of `dinov2-s-frozen`. Tests whether frozen pretrained DINOv2-B features are enough for CAPTCHA reading. |
| `dinov2-b-lora` | `dino_results/dinov2-base-lora/` | `facebook/dinov2-base` | LoRA fine-tuned on CAPTCHA transcription | LoRA adapters plus character heads | Task-adapted DINOv2-Base baseline. Validation sequence accuracy recorded as `0.939`. |
| `dinov2-s-frozen` | `dino_results/dinov2-small-frozen/` | `facebook/dinov2-small` | Heads-only CAPTCHA training | Character heads only; DINOv2 backbone stayed frozen | Tests whether frozen pretrained DINOv2-S features are enough for CAPTCHA reading. They were not: validation sequence accuracy recorded as `0.0`. |
| `dinov2-s-lora` | `dino_results/dinov2-small-lora/` | `facebook/dinov2-small` | LoRA fine-tuned on CAPTCHA transcription | LoRA adapters plus character heads | Task-adapted DINOv2-Small baseline. Validation sequence accuracy recorded as `0.893`. |
| `vit-b-lora` | `dino_results/vit-base-supervised-lora/` | `vit_base_patch16_224` from `timm` | LoRA fine-tuned on CAPTCHA transcription | LoRA adapters plus character heads | Task-adapted supervised ImageNet ViT-B baseline. Validation sequence accuracy recorded as `0.826`. |

## Important Distinctions

### `clip-b` vs `clip-b-lora`

`clip-b` is pretrained CLIP used without CAPTCHA fine-tuning. CLIP is a vision-language model, but the CLIP vision tower does not natively output CAPTCHA text. Our wrapper attaches CAPTCHA character heads so the code has a `logits` layer, but for `clip-b` those heads are random and untrained.

`clip-b-frozen` uses the same CLIP vision backbone and trains only the character heads. The CLIP transformer remains fixed.

`clip-b-lora` uses the same CLIP vision backbone, but trains LoRA adapters and character heads on the CAPTCHA transcription task.

### `dinov2-s-frozen` vs `dinov2-s-lora`

`dinov2-s-frozen` freezes the pretrained DINOv2-Small transformer and trains only the small character heads. This asks whether the pretrained representation is already sufficient for the task with a lightweight readout.

`dinov2-s-lora` trains LoRA adapters inside the DINOv2-Small transformer, in addition to the character heads. This lets the representation adapt to CAPTCHA transcription.

`dinov2-b-frozen` and `dinov2-b-lora` follow the same distinction at the DINOv2-Base scale.

### Pretrained-only Vision Backbones

Pretrained-only backbones such as `clip-b` and direct DINOv2 baselines are meaningful for representation probes, not for native CAPTCHA transcription. DINOv2 and the CLIP vision tower output visual features, not text. Any reported transcription accuracy for untrained character heads should be treated as a sanity check, not as the model's native OCR performance.

## Metrics Referenced by Charts

The validation sequence accuracies above come from the corresponding `dino_runs/*/metrics.json` files when present. The `cnn` chart is based on the trained `CaptchaCNN` checkpoint and probe artifacts under `probe_results/full/`.
