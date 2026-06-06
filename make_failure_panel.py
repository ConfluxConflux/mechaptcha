"""Three CAPTCHA failures stacked vertically, with perturbation labels (no ground truth)."""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset

from probe.extract import load_model
from train.model.charset import DEFAULT_ALPHABET

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = REPO_ROOT / "runs/siddharth/best.pt"
DATASET     = "jacobcohen/mechaptcha"
N_SAMPLE    = 2000        # how many val examples to scan for failures
IMAGE_SIZE  = (64, 160)
OUT         = REPO_ROOT / "failure_panel.png"

PERTURBATION_DISPLAY = {
    "blur":             "Blur",
    "dots":             "Dot noise",
    "salt_pepper":      "Salt-and-pepper noise",
    "rotation":         "Rotation",
    "wave":             "Letter wave",
    "wavy_line":        "Wavy line",
    "easy_line":        "Horizontal line",
    "hard_line":        "Angled line",
    "two_lines":        "Two lines",
    "bold":             "Bold font",
    "italic":           "Italic font",
    "dumb_control":     "Same-data control",
    "variation_control": "Same-distribution control",
}

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model…")
model = load_model(CHECKPOINT)
model.eval()
alphabet = DEFAULT_ALPHABET

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset…")
raw = load_dataset(DATASET, split=f"val[:{N_SAMPLE}]")

# Bool metadata columns = perturbation flags
SKIP = {"image", "text", "id"}
bool_cols = [c for c, f in raw.features.items()
             if c not in SKIP and str(f.dtype) == "bool"]
print("Perturbation columns:", bool_cols)

# ── Run inference ─────────────────────────────────────────────────────────────
from torchvision import transforms
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

failures = []   # list of (pil_image, prediction_str, perturbations: list[str])

print("Running inference…")
with torch.no_grad():
    for example in raw:
        img = example["image"]
        text = example["text"]
        tensor = transform(img).unsqueeze(0)   # [1, 1, H, W]
        logits = model(tensor)                  # [1, 5, 26]
        pred_indices = logits.argmax(dim=-1)[0].tolist()
        pred_text = "".join(alphabet[i] for i in pred_indices)

        if pred_text == text:
            continue  # model got it right — skip

        perturbs = [PERTURBATION_DISPLAY.get(c, c)
                    for c in bool_cols if example.get(c)]
        failures.append((img, pred_text, perturbs))

        if len(failures) >= 50:
            break   # enough candidates; stop early

print(f"Found {len(failures)} failures in {N_SAMPLE} examples.")
if len(failures) < 3:
    raise SystemExit("Not enough failures found — try increasing N_SAMPLE.")

# ── Pick 3 with varied perturbations ─────────────────────────────────────────
# Prefer examples from different perturbation categories where possible.
chosen = []
seen_perturbs: set[str] = set()
for img, pred, perturbs in failures:
    if not perturbs:
        continue
    key = perturbs[0]
    if key not in seen_perturbs:
        chosen.append((img, pred, perturbs))
        seen_perturbs.add(key)
    if len(chosen) == 3:
        break

# Fall back to first three if diversity filter didn't fill slots
for item in failures:
    if len(chosen) == 3:
        break
    if item not in chosen:
        chosen.append(item)

# ── Build figure ──────────────────────────────────────────────────────────────
FIG_W = 3.5     # single column width
IMG_H = 0.90    # inches per CAPTCHA image (160×64 aspect → ~2.5× wide)
PAD   = 0.35    # inches between images (for label)
TOP   = 0.12
BOT   = 0.12

fig_h = TOP + 3 * IMG_H + 2 * PAD + BOT
fig, axes = plt.subplots(3, 1, figsize=(FIG_W, fig_h))
fig.patch.set_facecolor("white")

plt.subplots_adjust(
    left=0.0, right=1.0,
    top=1 - TOP / fig_h,
    bottom=BOT / fig_h,
    hspace=PAD / IMG_H,
)

for ax, (img, pred, perturbs) in zip(axes, chosen):
    # Show image (grayscale PIL → resize for display)
    from torchvision.transforms.functional import to_tensor, resize
    import PIL
    disp = img.convert("L").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), PIL.Image.LANCZOS)
    ax.imshow(disp, cmap="gray", vmin=0, vmax=255, aspect="auto",
              interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_edgecolor("#888")

    # Perturbation label above the image
    label = ", ".join(perturbs) if perturbs else "no perturbation"
    ax.set_title(label, fontsize=7, color="#333333", pad=3,
                 fontstyle="italic")

fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
