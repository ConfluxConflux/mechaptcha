"""Generate a compact LaTeX table comparing linear vs MLP probe accuracy."""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from probe.results import load_results

lin = load_results(REPO_ROOT / "probe_results/results.json")
mlp = load_results(REPO_ROOT / "probe_results/full_mlp/results.json")

# Drop 'pool' — identical to cb3 for this input size
LAYERS = ["conv_block_0", "conv_block_1", "conv_block_2", "conv_block_3", "embedding"]
COL_HEADS = [r"\textit{cb}0", r"\textit{cb}1", r"\textit{cb}2",
             r"\textit{cb}3", r"emb"]

CATEGORIES: dict[str, list[str]] = {
    "Pixel noise":   ["blur", "dots", "salt_pepper"],
    "Line artifact": ["wavy_line", "easy_line", "hard_line", "two_lines"],
    "Geometric":     ["rotation", "wave"],
    "Font style":    ["bold", "italic"],
    "Controls":      ["dumb_control", "variation_control"],
}

DISPLAY = {
    "blur":              "Blur",
    "dots":              "Dot noise",
    "salt_pepper":       r"Salt \& pepper",
    "wavy_line":         "Wavy line",
    "easy_line":         r"Horiz.\ line",
    "hard_line":         "Angled line",
    "two_lines":         "Two lines",
    "rotation":          "Rotation",
    "wave":              "Letter wave",
    "bold":              "Bold",
    "italic":            "Italic",
    "dumb_control":      "Same-data ctrl.",
    "variation_control": "Same-dist.\ ctrl.",
}


def cell(exp: str, layer: str) -> str:
    l = lin.get(exp, {}).get(layer)
    m = mlp.get(exp, {}).get(layer)
    if l is None or m is None:
        return "--"
    lv = round(l.test_acc * 100)
    d  = round((m.test_acc - l.test_acc) * 100)
    if d == 0:
        return f"${lv}$"
    sign = "+" if d > 0 else "$-$"
    if d < 0:
        return rf"${lv}\ (\text{{\scriptsize {sign}{abs(d)}}})\!$"
    return rf"${lv}\ (\text{{\scriptsize +{d}}})\!$"


ncols = len(LAYERS)
col_spec = "l" + "r" * ncols

lines: list[str] = []
lines += [
    r"\begin{table}[t]",
    r"\centering",
    r"\setlength{\tabcolsep}{4pt}",
    r"\caption{Linear probe accuracy (\%) at each CNN layer, with MLP-probe delta in parentheses.}",
    r"\label{tab:linear_vs_mlp}",
    rf"\begin{{tabular}}{{{col_spec}}}",
    r"\toprule",
    "Distortion & " + " & ".join(COL_HEADS) + r" \\",
    r"\midrule",
]

first_group = True
for cat, exps in CATEGORIES.items():
    exps_present = [e for e in exps if e in lin]
    if not exps_present:
        continue
    if not first_group:
        lines.append(r"\addlinespace[2pt]")
    first_group = False
    lines.append(rf"\multicolumn{{{ncols + 1}}}{{l}}{{\textit{{{cat}}}}} \\")
    for exp in exps_present:
        cells = " & ".join(cell(exp, l) for l in LAYERS)
        lines.append(rf"\quad {DISPLAY[exp]} & {cells} \\")

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]

OUT = REPO_ROOT / "probe_table.tex"
OUT.write_text("\n".join(lines) + "\n")
print(OUT.read_text())
print(f"\nSaved: {OUT}")
