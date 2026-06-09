# Chart Model Types

This document defines the model labels used in `charts/` and how each model was trained to solve the CAPTCHA transcription task.

## TLDR
### DINOv2
- `dinov2-s`: [DINOv2-Small](https://huggingface.co/facebook/dinov2-small). It does not natively output CAPTCHA text, so the pipeline attaches 5 character heads at the end, one for each output character. Those heads are random/untrained in the pretrained-only run, so transcription accuracy is expected to be garbage. This is a representation-only baseline.
- `dinov2-s-frozen`: DINOv2-Small with no transformer weights changed and no LoRA adapters. The backbone is frozen and only the transcription heads are trained. It learns how to recognize CAPTCHA text from existing pretrained features; it does not learn new visual features.
- `dinov2-s-lora`: DINOv2-Small with LoRA adapters and character heads trained jointly on the CAPTCHA transcription task. This can adjust visual features via LoRA and performs much better on the task than `dinov2-s-frozen`.
- `dinov2-b`: [DINOv2-Base](https://huggingface.co/facebook/dinov2-base). Same protocol as `dinov2-s`, but at Base scale.
- `dinov2-b-frozen`: Same as `dinov2-s-frozen`, but larger.
- `dinov2-b-lora`: Same as `dinov2-s-lora`, but larger.

### CLIP
- `clip-b`: [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16). CLIP is a vision-language model, but its vision tower is an encoder, not an image-to-text decoder. We attach 5 random/untrained CAPTCHA heads in this pretrained-only baseline, so task logits are not meaningful.
- `clip-b-frozen`: CLIP-B vision tower frozen, CAPTCHA character heads trained.
- `clip-b-lora`: CLIP-B vision tower with LoRA adapters and character heads trained jointly.

### ViT
- `vit-b-frozen`: supervised ImageNet ViT-B from `timm`, frozen backbone, CAPTCHA character heads trained.
- `vit-b-lora`: supervised ImageNet ViT-B from `timm`, LoRA adapters and CAPTCHA character heads trained jointly. This is a control for pretraining objective versus DINOv2 and CLIP.

### CNN
- `cnn`: Custom CNN trained from scratch on the CAPTCHA task, including all convolution blocks, embedding layer, and character heads.

NOTE: `charts/transcription_accuracy.csv` is CAPTCHA transcription accuracy measured on the **probe/stress-test experiment images**, not on the original HF validation set. For pretrained-only rows with random/untrained heads, this is a sanity check and should not be interpreted as native OCR ability.

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
| `dinov2-b` | `dino_results/dinov2-base/` | `facebook/dinov2-base` | No CAPTCHA task training | Nothing meaningful for the task; CAPTCHA heads are newly initialized and untrained | Use only as a pretrained visual representation baseline. Its transcription logits are expected to be garbage. |
| `dinov2-b-frozen` | `dino_results/dinov2-base-frozen/` | `facebook/dinov2-base` | Heads-only CAPTCHA training | Character heads only; DINOv2 backbone stays frozen | Base-sized analogue of `dinov2-s-frozen`. Tests whether frozen pretrained DINOv2-B features are enough for CAPTCHA reading. |
| `dinov2-b-lora` | `dino_results/dinov2-base-lora/` | `facebook/dinov2-base` | LoRA fine-tuned on CAPTCHA transcription | LoRA adapters plus character heads | Task-adapted DINOv2-Base baseline. Validation sequence accuracy recorded as `0.939`. |
| `dinov2-s` | `dino_results/dinov2-small/` | `facebook/dinov2-small` | No CAPTCHA task training | Nothing meaningful for the task; CAPTCHA heads are newly initialized and untrained | Use only as a pretrained visual representation baseline. Its transcription logits are expected to be garbage. |
| `dinov2-s-frozen` | `dino_results/dinov2-small-frozen/` | `facebook/dinov2-small` | Heads-only CAPTCHA training | Character heads only; DINOv2 backbone stayed frozen | Tests whether frozen pretrained DINOv2-S features are enough for CAPTCHA reading. They were not: validation sequence accuracy recorded as `0.0`. |
| `dinov2-s-lora` | `dino_results/dinov2-small-lora/` | `facebook/dinov2-small` | LoRA fine-tuned on CAPTCHA transcription | LoRA adapters plus character heads | Task-adapted DINOv2-Small baseline. Validation sequence accuracy recorded as `0.893`. |
| `vit-b-frozen` | `dino_results/vit-base-supervised-frozen/` | `vit_base_patch16_224` from `timm` | Heads-only CAPTCHA training | Character heads only; supervised ImageNet ViT-B backbone stays frozen | Tests whether frozen supervised ImageNet ViT-B features are enough for CAPTCHA reading with a lightweight readout. |
| `vit-b-lora` | `dino_results/vit-base-supervised-lora/` | `vit_base_patch16_224` from `timm` | LoRA fine-tuned on CAPTCHA transcription | LoRA adapters plus character heads | Task-adapted supervised ImageNet ViT-B baseline. Validation sequence accuracy recorded as `0.826`. |

## Important Distinctions

### `clip-b` vs `clip-b-lora`

`clip-b` is pretrained CLIP used without CAPTCHA fine-tuning. CLIP is a vision-language model, but the CLIP vision tower does not natively output CAPTCHA text. Our wrapper attaches CAPTCHA character heads so the code has a `logits` layer, but for `clip-b` those heads are random and untrained.

`clip-b-frozen` uses the same CLIP vision backbone and trains only the character heads. The CLIP transformer remains fixed.

`clip-b-lora` uses the same CLIP vision backbone, but trains LoRA adapters and character heads on the CAPTCHA transcription task.

The CLIP retrieval protocol discussed separately would score image embeddings against candidate text embeddings. That is a different native-CLIP experiment and is not what these chart folders currently report.

### `dinov2-s-frozen` vs `dinov2-s-lora`

`dinov2-s-frozen` freezes the pretrained DINOv2-Small transformer and trains only the small character heads. This asks whether the pretrained representation is already sufficient for the task with a lightweight readout.

`dinov2-s-lora` trains LoRA adapters inside the DINOv2-Small transformer, in addition to the character heads. This lets the representation adapt to CAPTCHA transcription.

The LoRA adapters and character heads are trained at the same time from the same transcription loss. We do not first train heads and then train LoRA.

`dinov2-b-frozen` and `dinov2-b-lora` follow the same distinction at the DINOv2-Base scale.

### `vit-b-frozen` vs `vit-b-lora`

`vit-b-frozen` freezes the supervised ImageNet ViT-B backbone and trains only the CAPTCHA character heads.

`vit-b-lora` trains LoRA adapters inside the supervised ImageNet ViT-B transformer, in addition to the character heads.

The ViT-B base model is not trained from scratch. It starts from the `timm` pretrained `vit_base_patch16_224` weights.

### Pretrained-only Vision Backbones

Pretrained-only backbones such as `clip-b` and direct DINOv2 baselines are meaningful for representation probes, not for native CAPTCHA transcription. DINOv2 and the CLIP vision tower output visual features, not text. Any reported transcription accuracy for untrained character heads should be treated as a sanity check, not as the model's native OCR performance.

## Metrics Referenced by Charts

The validation sequence accuracies above come from the corresponding `dino_runs/*/metrics.json` files when present. The `cnn` chart is based on the trained `CaptchaCNN` checkpoint and probe artifacts under `probe_results/full/`.
