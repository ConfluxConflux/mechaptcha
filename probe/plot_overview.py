"""Method overview diagram: Batch A vs B concept + CaptchaCNN architecture + probe tap-points."""
from __future__ import annotations

from pathlib import Path


def plot_method_overview(
    data_dir: Path,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Ellipse
    from matplotlib.image import imread

    fig = plt.figure(figsize=(18, 10), facecolor="white")

    # ══════════════════════════════════════════════════════════════════════════
    # TOP BAND: title + concept box + sample pairs (one per perturbation group)
    # ══════════════════════════════════════════════════════════════════════════
    fig.text(0.5, 0.97, "Method Overview", ha="center", va="top",
             fontsize=21, fontweight="bold", color="#111")

    # ── Concept text box (top-left) ───────────────────────────────────────────
    ax_txt = fig.add_axes([0.01, 0.58, 0.24, 0.36])
    ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1); ax_txt.axis("off")
    ax_txt.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.03", facecolor="#f5f5f5", edgecolor="#aaa", lw=1.5))
    ax_txt.text(0.5, 0.82, "Synthetic text CAPTCHAs —\n"
                            "sequences of five letters\n"
                            "rendered in random fonts.",
                ha="center", va="top", fontsize=10.5, color="#222", linespacing=1.55)
    ax_txt.text(0.5, 0.50,
                "Batch A  has a visual perturbation\n"
                "(noise, lines, font style, …).",
                ha="center", va="top", fontsize=10.5, color="#222", linespacing=1.55)
    ax_txt.text(0.18, 0.495, "Batch A", ha="left", va="top",
                fontsize=10.5, fontweight="bold", color="#c0392b")
    ax_txt.text(0.5, 0.23,
                "Batch B  is the same text & fonts\n"
                "without any perturbation.",
                ha="center", va="top", fontsize=10.5, color="#222", linespacing=1.55)
    ax_txt.text(0.18, 0.225, "Batch B", ha="left", va="top",
                fontsize=10.5, fontweight="bold", color="#2471a3")

    # ── Three example perturbation pairs ──────────────────────────────────────
    EXAMPLES = [
        ("easy_line",   "Horizontal line"),
        ("salt_pepper", "Salt & pepper noise"),
        ("italic",      "Italic font"),
        ("rotation",    "Rotation"),
    ]

    pair_w, pair_h = 0.155, 0.095
    pair_gap = 0.015
    pair_left = 0.26
    pair_tops = [0.895, 0.895 - pair_h - 0.10, 0.895 - 2*(pair_h + 0.10),
                 0.895 - 3*(pair_h + 0.10)]

    for (exp, label), top in zip(EXAMPLES, pair_tops):
        img_a_path = data_dir / exp / "batch_a" / "images" / "000000.png"
        img_b_path = data_dir / exp / "batch_b" / "images" / "000000.png"
        if not img_a_path.exists():
            continue
        # Label above
        fig.text(pair_left + pair_w / 2, top + pair_h + 0.005, label,
                 ha="center", va="bottom", fontsize=8.5, style="italic", color="#444")
        # Batch A (top)
        ax_a = fig.add_axes([pair_left, top, pair_w, pair_h])
        ax_a.imshow(imread(str(img_a_path)), cmap="gray", aspect="auto")
        ax_a.axis("off")
        for sp in ax_a.spines.values():
            sp.set_visible(True); sp.set_color("#c0392b"); sp.set_linewidth(2.2)
        ax_a.set_ylabel("A", color="#c0392b", fontsize=8, fontweight="bold", rotation=0,
                         labelpad=4, va="center")
        # Batch B (below)
        ax_b = fig.add_axes([pair_left, top - pair_h - pair_gap, pair_w, pair_h])
        ax_b.imshow(imread(str(img_b_path)), cmap="gray", aspect="auto")
        ax_b.axis("off")
        for sp in ax_b.spines.values():
            sp.set_visible(True); sp.set_color("#2471a3"); sp.set_linewidth(2.2)
        ax_b.set_ylabel("B", color="#2471a3", fontsize=8, fontweight="bold", rotation=0,
                         labelpad=4, va="center")

    # ── Second column of perturbations ────────────────────────────────────────
    EXAMPLES2 = [
        ("blur",      "Blur"),
        ("dots",      "Dots"),
        ("hard_line", "Angled line"),
        ("wave",      "Letter wave"),
    ]
    pair_left2 = pair_left + pair_w + 0.055

    for (exp, label), top in zip(EXAMPLES2, pair_tops):
        img_a_path = data_dir / exp / "batch_a" / "images" / "000000.png"
        img_b_path = data_dir / exp / "batch_b" / "images" / "000000.png"
        if not img_a_path.exists():
            continue
        fig.text(pair_left2 + pair_w / 2, top + pair_h + 0.005, label,
                 ha="center", va="bottom", fontsize=8.5, style="italic", color="#444")
        ax_a = fig.add_axes([pair_left2, top, pair_w, pair_h])
        ax_a.imshow(imread(str(img_a_path)), cmap="gray", aspect="auto")
        ax_a.axis("off")
        for sp in ax_a.spines.values():
            sp.set_visible(True); sp.set_color("#c0392b"); sp.set_linewidth(2.2)
        ax_b = fig.add_axes([pair_left2, top - pair_h - pair_gap, pair_w, pair_h])
        ax_b.imshow(imread(str(img_b_path)), cmap="gray", aspect="auto")
        ax_b.axis("off")
        for sp in ax_b.spines.values():
            sp.set_visible(True); sp.set_color("#2471a3"); sp.set_linewidth(2.2)

    # ══════════════════════════════════════════════════════════════════════════
    # ARCHITECTURE PIPELINE (lower half)
    # ══════════════════════════════════════════════════════════════════════════
    ax = fig.add_axes([0.01, 0.03, 0.97, 0.40])
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.0); ax.axis("off")

    PROBE_COL = "#7b2fa3"
    PIPELINE = [
        ("input",        "#555",     "Input\n1×64×160",                   0.50, True),
        ("conv_block_0", "#1a5c8a",  "ConvBlock 0\nConv 1→64\n→64×32×80", 0.72, True),
        ("conv_block_1", "#1a5c8a",  "ConvBlock 1\nConv 64→128\n→128×16×40", 0.72, True),
        ("conv_block_2", "#1a5c8a",  "ConvBlock 2\nConv 128→256\n→256×8×20", 0.72, True),
        ("conv_block_3", "#1a5c8a",  "ConvBlock 3\nConv 256→384\n→384×4×10", 0.72, True),
        ("pool",         "#1a7a44",  "AvgPool\n→384×4×10",                0.60, True),
        ("embedding",    "#6a2390",  "Embedding\nLinear→512",             0.65, True),
        ("heads",        "#a83232",  "5 Output Heads\n512→36 each",       0.60, False),
    ]

    n = len(PIPELINE)
    box_w = 0.88
    gap   = 0.265
    total = n * box_w + (n - 1) * gap
    x0    = (10 - total) / 2
    cy    = 1.30

    xs = [x0 + i * (box_w + gap) for i in range(n)]

    for i, (key, col, label, h, has_probe) in enumerate(PIPELINE):
        x = xs[i]
        ax.add_patch(FancyBboxPatch((x, cy - h/2), box_w, h,
            boxstyle="round,pad=0.05", facecolor=col, edgecolor="#333",
            linewidth=1.3, alpha=0.92, zorder=3))
        ax.text(x + box_w/2, cy, label, ha="center", va="center",
                fontsize=6.0, color="white", fontweight="bold", zorder=4, linespacing=1.35)

    # Arrows
    for i in range(n - 1):
        ax.annotate("", xy=(xs[i+1], cy), xytext=(xs[i] + box_w, cy),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.4), zorder=2)

    # Probe dots
    probe_top = cy - max(h for *_, h, _ in PIPELINE) / 2 - 0.06
    probe_dot = probe_top - 0.38

    for i, (key, col, label, h, has_probe) in enumerate(PIPELINE):
        if not has_probe:
            continue
        tx = xs[i] + box_w / 2
        ax.plot([tx, tx], [probe_top, probe_dot + 0.08],
                color=PROBE_COL, lw=1.3, ls="--", zorder=1)
        circle = plt.Circle((tx, probe_dot), 0.082, color=PROBE_COL, zorder=5)
        ax.add_patch(circle)
        ax.text(tx, probe_dot, "P", ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold", zorder=6)
        short = key.replace("conv_block_", "cb")
        ax.text(tx, probe_dot - 0.16, f"probe\n({short})",
                ha="center", va="top", fontsize=5.5, color=PROBE_COL)

    # Legend
    lx = xs[-1] + box_w + 0.12
    ax.add_patch(plt.Circle((lx + 0.09, probe_dot), 0.068, color=PROBE_COL, zorder=5))
    ax.text(lx + 0.09, probe_dot, "P", ha="center", va="center",
            fontsize=5.5, color="white", fontweight="bold", zorder=6)
    ax.text(lx + 0.22, probe_dot, "= probe\ntap-point",
            ha="left", va="center", fontsize=6.5, color=PROBE_COL, linespacing=1.3)

    # ── Callout ellipse ──────────────────────────────────────────────────────
    ell_y = 0.30
    ax.add_patch(Ellipse((5, ell_y), width=9.5, height=0.72,
                         facecolor="#7b2fa3", edgecolor="#5a1f80",
                         lw=1.5, alpha=0.93, zorder=7))
    ax.text(5, ell_y,
            "At each probe tap-point we train a logistic regression to classify Batch A vs Batch B\n"
            "— a feature the model is trained to ignore. Where in the network does it truly forget?\n"
            "How does probe accuracy evolve as representations deepen?",
            ha="center", va="center", fontsize=8.8, color="white",
            linespacing=1.5, zorder=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    plot_method_overview(
        data_dir=Path("data/experiments_test"),
        output_path=Path("probe_results/chart_method_overview.png"),
    )
