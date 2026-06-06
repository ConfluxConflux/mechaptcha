"""2×2 panel of four CAPTCHAs the model failed on — no answers shown."""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from datasets import load_dataset

TARGET_TEXTS = {"pzxpo", "xxweq", "cjhwq", "hqsir"}
DATASET = "jacobcohen/mechaptcha"
OUT = REPO_ROOT / "challenge_panel.png"

PERTURBATION_DISPLAY = {
    "blur":              "blur",
    "dots":              "dot noise",
    "salt_pepper":       "salt-and-pepper noise",
    "rotation":          "rotation",
    "wave":              "letter wave",
    "wavy_line":         "wavy line",
    "easy_line":         "horizontal line",
    "hard_line":         "angled line",
    "two_lines":         "two lines",
    "bold":              "bold font",
    "italic":            "italic font",
    "dumb_control":      None,   # skip
    "variation_control": None,   # skip
    "char_jitter":       None,   # skip (minor cosmetic, not distinctive)
    "spacing_litter":    None,   # skip
    "spacing_jitter":    None,   # skip
}

print("Loading dataset…")
raw = load_dataset(DATASET, split="val")

SKIP = {"image", "text", "id"}
bool_cols = [c for c, f in raw.features.items()
             if c not in SKIP and str(f.dtype) == "bool"]

found: dict[str, dict] = {}
for example in raw:
    t = example["text"]
    if t in TARGET_TEXTS and t not in found:
        found[t] = example
    if len(found) == len(TARGET_TEXTS):
        break

print(f"Found: {list(found.keys())}")

# Fixed display order matching the user's request
ORDER = ["pzxpo", "xxweq", "cjhwq", "hqsir"]
items = [found[t] for t in ORDER if t in found]

# ── Figure: 2×2 grid (full text width — spans both columns) ───────────────────
FIG_W = 7.0
fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 3.2),
                         gridspec_kw={"hspace": 0.28, "wspace": 0.06})
fig.patch.set_facecolor("white")

# Perturbations to skip when labelling (control conditions, not real distortions)
SKIP_PERTURBS = {"same-data control", "same-distribution control", "char jitter"}

for ax, example in zip(axes.flat, items):
    img = example["image"].convert("L")
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto",
              interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#aaa")

    perturbs = [PERTURBATION_DISPLAY.get(c, c) for c in bool_cols
                if example.get(c) and PERTURBATION_DISPLAY.get(c, c) not in SKIP_PERTURBS]
    label = ", ".join(perturbs[:4]) if perturbs else "no perturbation"
    ax.set_xlabel(label, fontsize=7, color="#444", labelpad=4, fontstyle="italic")

fig.savefig(OUT, dpi=250, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
