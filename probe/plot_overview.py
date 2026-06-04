"""Method overview diagram: shows Batch A vs B concept + CaptchaCNN architecture + probe tap-points."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_method_overview(
    data_dir: Path,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.image import imread

    fig = plt.figure(figsize=(18, 10), facecolor="white")
    fig.patch.set_facecolor("white")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.97, "Method Overview", ha="center", va="top",
             fontsize=22, fontweight="bold", color="#111")

    # ═══════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN: concept text box + perturbation examples
    # ═══════════════════════════════════════════════════════════════════════════
    ax_text = fig.add_axes([0.01, 0.60, 0.28, 0.33])
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    ax_text.axis("off")
    ax_text.add_patch(FancyBboxPatch((0, 0), 1, 1,
        boxstyle="round,pad=0.03", facecolor="#f7f7f7", edgecolor="#999", linewidth=1.5))

    concept = (
        "Synthetic text CAPTCHAs — sequences of\n"
        "five letters rendered in random fonts.\n\n"
        r"$\bf{Batch\ A}$" + "  has a visual perturbation\n"
        "(noise, lines, font style, …).\n\n"
        r"$\bf{Batch\ B}$" + "  is the clean version\n"
        "with the same letters & fonts."
    )
    ax_text.text(0.5, 0.55, concept, ha="center", va="center",
                 fontsize=10.5, color="#222", linespacing=1.6,
                 transform=ax_text.transAxes)

    # Colour the "Batch A" / "Batch B" labels
    ax_text.text(0.50, 0.445, "Batch A", ha="center", va="center",
                 fontsize=11, fontweight="bold", color="#c0392b",
                 transform=ax_text.transAxes)
    ax_text.text(0.50, 0.285, "Batch B", ha="center", va="center",
                 fontsize=11, fontweight="bold", color="#2471a3",
                 transform=ax_text.transAxes)

    # ── Perturbation thumbnail grid ────────────────────────────────────────────
    PERTURB_EXPS = [
        ("easy_line",   "horiz. line"),
        ("hard_line",   "angled line"),
        ("salt_pepper", "salt & pepper"),
        ("blur",        "blur"),
        ("dots",        "dots"),
        ("italic",      "italic font"),
        ("rotation",    "rotation"),
        ("wave",        "letter wave"),
    ]

    n_cols = 4
    n_rows = 2
    thumb_w = 0.90 / n_cols
    thumb_h = 0.22
    grid_left = 0.01
    grid_bot  = 0.30

    for idx, (exp, label) in enumerate(PERTURB_EXPS):
        col = idx % n_cols
        row = idx // n_cols
        left = grid_left + col * (thumb_w + 0.005)
        bot  = grid_bot + (n_rows - 1 - row) * (thumb_h + 0.03)

        img_a_path = data_dir / exp / "batch_a" / "images" / "000000.png"
        img_b_path = data_dir / exp / "batch_b" / "images" / "000000.png"
        if not img_a_path.exists():
            continue

        ax_a = fig.add_axes([left, bot + thumb_h * 0.52, thumb_w, thumb_h * 0.46])
        ax_b = fig.add_axes([left, bot,                  thumb_w, thumb_h * 0.46])
        for ax_img, path, ec in [(ax_a, img_a_path, "#c0392b"), (ax_b, img_b_path, "#2471a3")]:
            img = imread(str(path))
            ax_img.imshow(img, cmap="gray", aspect="auto")
            ax_img.axis("off")
            for spine in ax_img.spines.values():
                spine.set_visible(True)
                spine.set_color(ec)
                spine.set_linewidth(1.8)

        # label centred between the two thumbnails
        fig.text(left + thumb_w / 2, bot + thumb_h * 1.04, label,
                 ha="center", va="bottom", fontsize=7.5, color="#444", style="italic")

    # ═══════════════════════════════════════════════════════════════════════════
    # CENTRE: Sample A/B image pair (large, with boxes)
    # ═══════════════════════════════════════════════════════════════════════════
    sample_exp = "easy_line"
    img_a = imread(str(data_dir / sample_exp / "batch_a" / "images" / "000000.png"))
    img_b = imread(str(data_dir / sample_exp / "batch_b" / "images" / "000000.png"))

    ax_sa = fig.add_axes([0.31, 0.73, 0.18, 0.17])
    ax_sb = fig.add_axes([0.31, 0.56, 0.18, 0.17])
    for ax_s, img, ec, tag in [
        (ax_sa, img_a, "#c0392b", "Batch A  (perturbed)"),
        (ax_sb, img_b, "#2471a3", "Batch B  (clean)"),
    ]:
        ax_s.imshow(img, cmap="gray", aspect="auto")
        ax_s.axis("off")
        for spine in ax_s.spines.values():
            spine.set_visible(True)
            spine.set_color(ec)
            spine.set_linewidth(2.5)
        ax_s.set_title(tag, fontsize=9, color=ec, fontweight="bold", pad=3)

    # ═══════════════════════════════════════════════════════════════════════════
    # ARCHITECTURE PIPELINE (bottom half)
    # ═══════════════════════════════════════════════════════════════════════════
    ax_arch = fig.add_axes([0.01, 0.04, 0.97, 0.46])
    ax_arch.set_xlim(0, 10)
    ax_arch.set_ylim(0, 2.4)
    ax_arch.axis("off")

    cy = 1.45   # vertical centre of the pipeline

    PROBE_COLOR   = "#7b2fa3"
    BLOCK_COLORS  = {
        "input":       "#555555",
        "conv_block_0": "#2471a3",
        "conv_block_1": "#2471a3",
        "conv_block_2": "#2471a3",
        "conv_block_3": "#2471a3",
        "pool":         "#1a8a47",
        "embedding":    "#7b2fa3",
        "heads":        "#c0392b",
    }
    BLOCK_LABELS = {
        "input":        "Input\n1×64×160",
        "conv_block_0": "ConvBlock 0\nConv2d 1→64\n+ ReLU + MaxPool\n→ 64×32×80",
        "conv_block_1": "ConvBlock 1\nConv2d 64→128\n+ ReLU + MaxPool\n→ 128×16×40",
        "conv_block_2": "ConvBlock 2\nConv2d 128→256\n+ ReLU + MaxPool\n→ 256×8×20",
        "conv_block_3": "ConvBlock 3\nConv2d 256→384\n+ ReLU + MaxPool\n→ 384×4×10",
        "pool":         "AdaptiveAvgPool\n→ 384×4×10",
        "embedding":    "Embedding\nFlatten+Linear\n→ 512-d",
        "heads":        "5 Output Heads\n512→36 each\n(5 chars × 36 cls)",
    }
    PROBE_LAYERS = ["input", "conv_block_0", "conv_block_1", "conv_block_2",
                    "conv_block_3", "pool", "embedding"]

    ORDER = ["input", "conv_block_0", "conv_block_1", "conv_block_2",
             "conv_block_3", "pool", "embedding", "heads"]
    n = len(ORDER)
    box_w = 0.90
    gap   = 0.28
    total = n * box_w + (n - 1) * gap
    x0    = (10 - total) / 2

    positions = {}
    for i, key in enumerate(ORDER):
        positions[key] = x0 + i * (box_w + gap)

    BOX_HEIGHTS = {
        "input":        0.52,
        "conv_block_0": 0.80,
        "conv_block_1": 0.80,
        "conv_block_2": 0.80,
        "conv_block_3": 0.80,
        "pool":         0.65,
        "embedding":    0.70,
        "heads":        0.72,
    }

    for key in ORDER:
        x = positions[key]
        h = BOX_HEIGHTS[key]
        col = BLOCK_COLORS[key]
        ax_arch.add_patch(FancyBboxPatch(
            (x, cy - h / 2), box_w, h,
            boxstyle="round,pad=0.05",
            facecolor=col, edgecolor="#333", linewidth=1.3, alpha=0.90, zorder=3,
        ))
        ax_arch.text(x + box_w / 2, cy, BLOCK_LABELS[key],
                     ha="center", va="center", fontsize=6.2, color="white",
                     zorder=4, linespacing=1.35, fontweight="bold")

    # Arrows between blocks
    for i in range(len(ORDER) - 1):
        x_left  = positions[ORDER[i]] + box_w
        x_right = positions[ORDER[i + 1]]
        ax_arch.annotate(
            "", xy=(x_right, cy), xytext=(x_left, cy),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.4), zorder=2,
        )

    # Probe dots + drop-lines
    probe_y_top = cy - max(BOX_HEIGHTS.values()) / 2 - 0.06
    probe_y_dot = probe_y_top - 0.42
    probe_y_label = probe_y_dot - 0.20

    for layer in PROBE_LAYERS:
        tx = positions[layer] + box_w / 2
        ax_arch.plot([tx, tx], [probe_y_top, probe_y_dot + 0.09],
                     color=PROBE_COLOR, lw=1.3, linestyle="--", zorder=1)
        circle = plt.Circle((tx, probe_y_dot), 0.088,
                             color=PROBE_COLOR, zorder=5)
        ax_arch.add_patch(circle)
        ax_arch.text(tx, probe_y_dot, "P",
                     ha="center", va="center", fontsize=7, color="white",
                     fontweight="bold", zorder=6)
        short = layer.replace("conv_block_", "cb")
        ax_arch.text(tx, probe_y_label, f"probe\n({short})",
                     ha="center", va="top", fontsize=5.8, color=PROBE_COLOR)

    # Legend dot
    ax_arch.add_patch(plt.Circle((x0 - 0.05, probe_y_dot), 0.072,
                                 color=PROBE_COLOR, zorder=5))
    ax_arch.text(x0 - 0.05, probe_y_dot, "P",
                 ha="center", va="center", fontsize=6, color="white",
                 fontweight="bold", zorder=6)
    ax_arch.text(x0 + 0.10, probe_y_dot, "= probe tap-point",
                 ha="left", va="center", fontsize=7.5, color=PROBE_COLOR)

    # ── Bottom callout ellipse ─────────────────────────────────────────────────
    from matplotlib.patches import Ellipse
    ell_y = 0.16
    ell = Ellipse((5, ell_y), width=9.6, height=0.56,
                  facecolor="#7b2fa3", edgecolor="#5a1f80",
                  linewidth=1.5, alpha=0.93, zorder=7)
    ax_arch.add_patch(ell)
    callout = (
        "At each probe tap-point we train a logistic regression to classify "
        "Batch A vs Batch B — a feature the model is trained to ignore.\n"
        "Where in the network does the model truly forget the perturbation? "
        "How does probe accuracy change as representations deepen?"
    )
    ax_arch.text(5, ell_y, callout, ha="center", va="center",
                 fontsize=8.5, color="white", linespacing=1.45,
                 zorder=8, wrap=True,
                 bbox=dict(boxstyle="round", facecolor="none", edgecolor="none"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    plot_method_overview(
        data_dir=Path("data/experiments_test"),
        output_path=Path("probe_results/chart_method_overview.png"),
    )
