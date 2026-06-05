"""
Architecture diagram for the paper: CNN with linear probe tap-points.

Portrait format (~3.5 × 7 in) designed for a single column in a two-column paper.
Probe dots mark every representation the probing study taps:
  input → conv_block_0..3 → pool → embedding → logits
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent / "arch_diagram.png"

# ── Palette ───────────────────────────────────────────────────────────────────
C = dict(
    bg      = "#FFFFFF",
    border  = "#2B2D42",
    arrow   = "#4A4A6E",
    text    = "#1A1A2E",
    sub     = "#4E4E72",
    probe   = "#7B2D8B",
    # layer type colours
    input_c = "#E4EAF4",   # light slate  — input image
    conv_c  = "#B8D4EE",   # cornflower   — conv blocks
    pool_c  = "#F5D8B4",   # peach        — pooling
    embed_c = "#B8E8C4",   # mint         — fully-connected embedding
    out_c   = "#DDD0EC",   # lavender     — output heads
)

# ── Figure geometry (inches = data units) ─────────────────────────────────────
FIG_W   = 3.5
FIG_H   = 7.0
CX      = FIG_W / 2

BOX_W   = 2.75
BOX_H   = 0.505
GAP     = 0.295    # vertical space between box bottom and next box top
TOP_PAD = 0.22
PROBE_MS = 10      # marker size in points (renders as true circle)

# ── Layer stack (display_name, sublabel, color) ────────────────────────────────
LAYERS: list[tuple[str, str, str]] = [
    ("Input Image",
     "1 × 64 × 160",
     C["input_c"]),
    ("Conv Block 1",
     "Conv2d(1→64, 3×3)  +  ReLU  +  MaxPool2d(2)",
     C["conv_c"]),
    ("Conv Block 2",
     "Conv2d(64→128, 3×3)  +  ReLU  +  MaxPool2d(2)",
     C["conv_c"]),
    ("Conv Block 3",
     "Conv2d(128→256, 3×3)  +  ReLU  +  MaxPool2d(2)",
     C["conv_c"]),
    ("Conv Block 4",
     "Conv2d(256→384, 3×3)  +  ReLU  +  MaxPool2d(2)",
     C["conv_c"]),
    ("Adaptive Avg Pool",
     "→  384 × 4 × 10",
     C["pool_c"]),
    ("Embedding",
     "Linear(15 360 → 512)  +  ReLU",
     C["embed_c"]),
    ("Output Heads",
     "5 × Linear(512 → 26)",
     C["out_c"]),
]
N = len(LAYERS)

# ── Probe labels (shown right of each probe dot) ───────────────────────────────
PROBE_LABELS = [
    "input",
    "conv_block_0",
    "conv_block_1",
    "conv_block_2",
    "conv_block_3",
    "pool",
    "embedding",
    "logits",
]


def box_top(i: int) -> float:
    """Y-coordinate of top edge of box i (y increases upward)."""
    return FIG_H - TOP_PAD - i * (BOX_H + GAP)


# ── Build figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── Draw each layer box ────────────────────────────────────────────────────────
for i, (name, sub, color) in enumerate(LAYERS):
    t  = box_top(i)
    cy = t - BOX_H / 2
    bx = CX - BOX_W / 2

    ax.add_patch(FancyBboxPatch(
        (bx, t - BOX_H), BOX_W, BOX_H,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=C["border"], linewidth=0.8, zorder=3,
    ))
    ax.text(CX, cy + 0.092, name,
            ha="center", va="center", fontsize=7.8, fontweight="bold",
            color=C["text"], zorder=4)
    ax.text(CX, cy - 0.108, sub,
            ha="center", va="center", fontsize=5.8, color=C["sub"], zorder=4)

# ── Draw arrows + probe dots ───────────────────────────────────────────────────
# Probe i sits on the arrow BELOW box i, representing the output of box i.
for i in range(N):
    y_bot = box_top(i) - BOX_H

    if i < N - 1:
        y_next_top = box_top(i + 1)
        y_mid = (y_bot + y_next_top) / 2

        ax.annotate(
            "",
            xy=(CX, y_next_top + 0.01),
            xytext=(CX, y_bot - 0.01),
            arrowprops=dict(arrowstyle="->", color=C["arrow"],
                            lw=0.9, mutation_scale=11),
            zorder=2,
        )
        probe_y = y_mid
    else:
        # Logits: short downward stub below last box
        probe_y = y_bot - GAP / 2
        ax.annotate(
            "",
            xy=(CX, probe_y - 0.04),
            xytext=(CX, y_bot - 0.01),
            arrowprops=dict(arrowstyle="->", color=C["arrow"],
                            lw=0.9, mutation_scale=11),
            zorder=2,
        )

    ax.plot(CX, probe_y, "o",
            ms=PROBE_MS, color=C["probe"],
            mec="white", mew=1.0, zorder=5)

    # Probe label to the right of the dot
    label_x = CX + BOX_W / 2 * 0.58
    ax.text(label_x, probe_y, PROBE_LABELS[i],
            ha="left", va="center", fontsize=5.5, color=C["probe"],
            fontstyle="italic", zorder=5)


fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor=C["bg"])
plt.close(fig)
print(f"Saved: {OUT}")
