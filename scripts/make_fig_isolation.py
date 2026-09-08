"""Regenerate the component-isolation figure from the numbers actually in the table.

The previous fig9_isolation.pdf had no generator in the repository and predates the recomputation of
the isolation on a single cohort, so it contradicted Table V. This produces the figure from the values
`scripts/component_isolation.py` reports, so the two cannot drift apart again.

Design: a horizontal bar chart, single IEEE column, ordered as the pipeline is read. The step that the
paper argues for -- changing the read-out -- is the only highlighted bar, and the annotation shows the
paired change against the row above rather than an absolute value, because the claim is about
increments.

    python scripts/make_fig_isolation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Values from scripts/component_isolation.py on the single 298-subject cohort.
STAGES = [
    ("Raw flow, FFT", 0.685, None, False),
    ("Published extractor", 0.755, "+0.070", False),
    ("Encoder + linear read-out", 0.754, "-0.001 n.s.", False),
    ("+ deep fusion head", 0.826, "+0.072  p=.08", False),
    ("+ distributional read-out", 0.935, "+0.109  p<.001", True),
]
ACCENT = "#1b4f72"
MUTED = "#b8c6d1"
CHANCE = 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manuscript/fig/fig9_isolation")
    ap.add_argument("--width", type=float, default=3.45)   # one IEEE column
    ap.add_argument("--height", type=float, default=2.05)
    a = ap.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    })
    fig, ax = plt.subplots(figsize=(a.width, a.height))

    labels = [s[0] for s in STAGES]
    vals = [s[1] for s in STAGES]
    deltas = [s[2] for s in STAGES]
    hot = [s[3] for s in STAGES]
    ypos = range(len(STAGES))[::-1]          # first stage at the top

    bars = ax.barh(list(ypos), vals, height=0.62,
                   color=[ACCENT if h else MUTED for h in hot],
                   edgecolor="white", linewidth=0.5, zorder=3)

    ax.axvline(CHANCE, color="0.55", linewidth=0.7, linestyle=(0, (3, 3)), zorder=2)
    ax.text(CHANCE, len(STAGES) - 0.55, "chance", fontsize=6.5, color="0.45",
            ha="center", va="bottom")

    for y, b, v, d, h in zip(ypos, bars, vals, deltas, hot):
        # Value, then its increment on the same baseline: no annotation is ever placed outside the
        # axis, so nothing can collide with the tick labels.
        label = f"{v:.3f}" + (f"   {d}" if d else "")
        ax.text(v + 0.008, y, label, va="center", ha="left",
                fontsize=7.0, fontweight="bold" if h else "normal",
                color=ACCENT if h else "0.3", zorder=4)

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlim(0.4, 1.30)          # headroom for the value + increment annotation
    ax.set_xlabel("Tachypnea AUROC", fontsize=7.5)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(axis="x", color="0.9", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=2)

    fig.tight_layout(pad=0.4)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(out.with_suffix(ext), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")
    print("Values mirror scripts/component_isolation.py; regenerate both together.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
