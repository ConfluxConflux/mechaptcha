"""Method overview diagram: Batch A vs B concept + CaptchaCNN architecture + probe tap-points."""
from __future__ import annotations

from pathlib import Path


def plot_method_overview(
    data_dir: Path,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
    from matplotlib.image import imread

    fig = plt.figure(figsize=(18, 10.5), facecolor="white")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.985, "Method Overview", ha="center", va="top",
             fontsize=22, fontweight="bold", color="#111")

    # ══════════════════════════════════════════════════════════════════════════
    # TOP-LEFT: concept text box
    # ══════════════════════════════════════════════════════════════════════════
    ax_txt = fig.add_axes([0.01, 0.56, 0.22, 0.40])
    ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1); ax_txt.axis("off")
    ax_txt.add_patch(FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
        boxstyle="round,pad=0.03", facecolor="#f5f5f5", edgecolor="#aaa", lw=1.5))

    ax_txt.text(0.5, 0.90,
                "Synthetic text CAPTCHAs —\n"
                "sequences of five letters\n"
                "rendered in random fonts,\n"
                "made by Claude.",
                ha="center", va="top", fontsize=10, color="#222", linespacing=1.6)

    # Batch A line (coloured label inline)
    ax_txt.text(0.12, 0.52, "Batch A", ha="left", va="top",
                fontsize=10.5, fontweight="bold", color="#c0392b")
    ax_txt.text(0.50, 0.52,
                "has a visual\nperturbation (noise, lines,\nfont style, geometry…).",
                ha="left", va="top", fontsize=9.8, color="#222", linespacing=1.5)

    # Batch B line
    ax_txt.text(0.12, 0.22, "Batch B", ha="left", va="top",
                fontsize=10.5, fontweight="bold", color="#2471a3")
    ax_txt.text(0.50, 0.22,
                "is the same text\n& fonts without any\nperturbation.",
                ha="left", va="top", fontsize=9.8, color="#222", linespacing=1.5)

    # ══════════════════════════════════════════════════════════════════════════
    # EXAMPLE PAIRS — two columns of 4 perturbation types
    # ══════════════════════════════════════════════════════════════════════════
    COLUMNS = [
        [
            ("easy_line",   "Horizontal line"),
            ("salt_pepper", "Salt & pepper"),
            ("italic",      "Italic font"),
            ("rotation",    "Rotation"),
        ],
        [
            ("blur",      "Blur"),
            ("dots",      "Dots"),
            ("hard_line", "Angled line"),
            ("wave",      "Letter wave"),
        ],
    ]

    pair_w   = 0.155    # width of one CAPTCHA thumbnail
    pair_h   = 0.088    # height of one thumbnail
    pair_gap = 0.010    # gap between A and B
    row_step = 0.135    # vertical step between rows

    col_lefts = [0.245, 0.245 + pair_w + 0.055]  # left edges of the two columns
    top_start = 0.920  # figure y of top of first A image

    for col_idx, col_exps in enumerate(COLUMNS):
        left = col_lefts[col_idx]
        for row_idx, (exp, label) in enumerate(col_exps):
            img_a_path = data_dir / exp / "batch_a" / "images" / "000000.png"
            img_b_path = data_dir / exp / "batch_b" / "images" / "000000.png"
            if not img_a_path.exists():
                continue

            top_a = top_start - row_idx * row_step

            # Perturbation label
            fig.text(left + pair_w / 2, top_a + 0.006, label,
                     ha="center", va="bottom", fontsize=7.8, style="italic", color="#444")

            # Batch A
            ax_a = fig.add_axes([left, top_a - pair_h, pair_w, pair_h])
            ax_a.imshow(imread(str(img_a_path)), cmap="gray", aspect="auto")
            ax_a.axis("off")
            for sp in ax_a.spines.values():
                sp.set_visible(True); sp.set_color("#c0392b"); sp.set_linewidth(2.2)
            ax_a.set_ylabel("A", color="#c0392b", fontsize=8, fontweight="bold",
                            rotation=0, labelpad=3, va="center")

            # Batch B
            top_b = top_a - pair_h - pair_gap - pair_h
            ax_b = fig.add_axes([left, top_b, pair_w, pair_h])
            ax_b.imshow(imread(str(img_b_path)), cmap="gray", aspect="auto")
            ax_b.axis("off")
            for sp in ax_b.spines.values():
                sp.set_visible(True); sp.set_color("#2471a3"); sp.set_linewidth(2.2)
            ax_b.set_ylabel("B", color="#2471a3", fontsize=8, fontweight="bold",
                            rotation=0, labelpad=3, va="center")

    # ══════════════════════════════════════════════════════════════════════════
    # ARCHITECTURE PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    ax = fig.add_axes([0.01, 0.03, 0.97, 0.42])
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.2); ax.axis("off")

    PROBE_COL = "#7b2fa3"

    PIPELINE = [
        # (key, colour, label, box_height, has_probe)
        ("input",        "#555555", "Input\n1×64×160",                      0.50, True),
        ("conv_block_0", "#1a5c8a", "ConvBlock 0\nConv 1→64\n→ 64×32×80",  0.72, True),
        ("conv_block_1", "#1a5c8a", "ConvBlock 1\nConv 64→128\n→128×16×40", 0.72, True),
        ("conv_block_2", "#1a5c8a", "ConvBlock 2\nConv 128→256\n→256×8×20", 0.72, True),
        ("conv_block_3", "#1a5c8a", "ConvBlock 3\nConv 256→384\n→384×4×10", 0.72, True),
        ("pool",         "#1a7a44", "AdaptiveAvgPool\n→ 384×4×10",          0.60, True),
        ("embedding",    "#6a2390", "Embedding\nLinear → 512-d",            0.65, True),
        ("heads",        "#a83232", "5 Output Heads\n512 → 36 each",        0.60, False),
    ]

    n = len(PIPELINE)
    box_w = 0.87
    gap   = 0.265
    total = n * box_w + (n - 1) * gap
    x0    = (10 - total) / 2
    cy    = 1.42

    xs = [x0 + i * (box_w + gap) for i in range(n)]
    max_h = max(h for *_, h, _ in PIPELINE)

    for i, (key, col, label, h, _probe) in enumerate(PIPELINE):
        ax.add_patch(FancyBboxPatch((xs[i], cy - h/2), box_w, h,
            boxstyle="round,pad=0.05", facecolor=col, edgecolor="#333",
            lw=1.3, alpha=0.92, zorder=3))
        ax.text(xs[i] + box_w/2, cy, label, ha="center", va="center",
                fontsize=6.0, color="white", fontweight="bold", zorder=4, linespacing=1.35)

    # Arrows between blocks
    for i in range(n - 1):
        ax.annotate("", xy=(xs[i+1], cy), xytext=(xs[i] + box_w, cy),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.4), zorder=2)

    # Probe dots + drop-lines
    probe_top = cy - max_h / 2 - 0.06
    probe_dot = probe_top - 0.35

    for i, (key, col, label, h, has_probe) in enumerate(PIPELINE):
        if not has_probe:
            continue
        tx = xs[i] + box_w / 2
        ax.plot([tx, tx], [probe_top, probe_dot + 0.075],
                color=PROBE_COL, lw=1.3, ls="--", zorder=1)
        ax.add_patch(plt.Circle((tx, probe_dot), 0.080, color=PROBE_COL, zorder=5))
        ax.text(tx, probe_dot, "P", ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold", zorder=6)
        short = key.replace("conv_block_", "cb")
        ax.text(tx, probe_dot - 0.155, f"probe\n({short})",
                ha="center", va="top", fontsize=5.4, color=PROBE_COL)

    # Callout ellipse
    ell_y = 0.285
    ax.add_patch(Ellipse((5, ell_y), width=9.5, height=0.70,
                         facecolor="#7b2fa3", edgecolor="#5a1f80", lw=1.5, alpha=0.93, zorder=7))
    ax.text(5, ell_y,
            "At each probe tap-point (P) we train a logistic regression to classify Batch A vs Batch B\n"
            "— a signal the model is trained to suppress. Where does the model truly forget the perturbation?\n"
            "How does probe accuracy evolve as representations deepen through the network?",
            ha="center", va="center", fontsize=8.8, color="white", linespacing=1.5, zorder=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    plot_method_overview(
        data_dir=Path("data/experiments_test"),
        output_path=Path("probe_results/chart_method_overview.png"),
    )
