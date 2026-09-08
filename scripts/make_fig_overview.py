"""Figure 1: how the clinical decision is formed, and why it is not the same as estimating the rate.

Three panels, all from real artifacts rather than illustration:

  (a) the two read-outs, sharing one encoder. The only difference is what is done with the encoder's
      output: collapse it to a scalar and compare to the WHO cut point, or keep it as a distribution
      over rates and integrate the mass above that cut point.
  (b) one real neonate, decoded from the released checkpoint. Its regressed rate saturates below the
      cut point -- the camera literally cannot read a rate that high -- while a fifth of its salience
      mass still lies above the cut point, which is enough to flag it.
  (c) the same picture for every tachypneic neonate in the cohort, so (b) is not a lucky case.

The mechanism the figure is meant to make obvious: a regress-then-threshold pipeline scores cases by a
monotone function of its own rate estimate, so its *ranking is its rate*. When the rate saturates, the
ranking saturates with it, and no threshold can recover the case. Integrating probability mass does not
have that property.

Run:  python scripts/make_fig_overview.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_resp_salience_ssl import Encoder, SalHead, prep, BINS  # noqa: E402

RUN = os.path.join(ROOT, "outputs", "paper_runs", "pedrr_kfold_salience_published")
CKPT_RUNS = [os.path.join(ROOT, "outputs", "paper_runs",
                          f"pedrr_kfold_salience_nossl332det_seed{i}") for i in range(6)]
CASE = "ID-1266"                      # true RR 78.5, regressed 42.4, mass above cutoff 0.22
OUT = os.path.join(ROOT, "manuscript", "fig_overview")

INK, MUTED = "#1b2a38", "#8a949e"
REG, DIST = "#c0603a", "#1b4f72"
FLAG, MISS = "#4a8a5a", "#c0392b"


def load_preds():
    ps = sorted(glob.glob(os.path.join(RUN, "fold*", "test_eval", "prism_breath_predictions.csv")))
    return pd.concat([pd.read_csv(p) for p in ps], ignore_index=True)


def salience_for(subject, fold, signal_path):
    """Decode one subject's salience curve, averaging the seeds exactly as the model does."""
    x = np.load(signal_path, allow_pickle=True)
    sig = torch.tensor(prep(x["flow"])).unsqueeze(0).unsqueeze(0)
    curves = []
    for r in CKPT_RUNS:
        ck = os.path.join(r, f"fold{fold}", "train", "resp_salience_best.pt")
        if not os.path.exists(ck):
            continue
        c = torch.load(ck, map_location="cpu", weights_only=False)
        enc, head = Encoder(), SalHead()
        enc.load_state_dict(c["encoder_state_dict"]); head.load_state_dict(c["head_state_dict"])
        enc.eval(); head.eval()
        with torch.no_grad():
            s = head(enc(sig)).squeeze(0).numpy()
        curves.append(s / (s.sum() + 1e-8))
    if not curves:
        raise SystemExit("no checkpoints found; run the training queue first")
    return np.mean(curves, axis=0)


# --------------------------------------------------------------------------------------------
def panel_a(ax):
    # Tight limits: the schematic spans y 1.3-8.6, and leaving the axis at 0-10 stranded it in the
    # upper half with a band of dead space underneath.
    ax.axis("off"); ax.set_xlim(0, 11.4); ax.set_ylim(1.15, 8.75)

    def box(x, y, w, h, text, fc="white", ec=INK, fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    fc=fc, ec=ec, lw=0.9, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                zorder=3, color=INK, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                     lw=0.9, color=color, zorder=1))

    # shared front end
    box(0.1, 4.1, 1.5, 1.6, "chest\nvideo")
    box(2.0, 4.1, 1.7, 1.6, "shared\nencoder", fc="#eef2f5", bold=True)
    arrow(1.65, 4.9, 1.95, 4.9)
    ax.text(2.85, 3.85, "identical up to here", ha="center", va="top", fontsize=6.3, color=MUTED)

    # upper branch: regress then threshold
    arrow(3.75, 5.3, 4.35, 7.1, REG)
    box(4.4, 6.4, 2.0, 1.4, "rate  $\\hat{r}$\n(one number)", ec=REG)
    arrow(6.45, 7.1, 6.95, 7.1, REG)
    box(7.0, 6.4, 2.8, 1.4, "$\\hat{r} \\geq \\tau_{\\mathrm{age}}$ ?", ec=REG)
    arrow(9.85, 7.1, 10.35, 7.1, REG)
    ax.text(10.45, 7.1, "decision", ha="left", va="center", fontsize=7.0, color=REG)
    ax.text(4.4, 8.15, "Regress, then threshold", fontsize=7.6, color=REG, fontweight="bold")
    ax.text(4.4, 6.22, "ranks cases by a monotone function of $\\hat{r}$:\n"
                       "the ranking $=$ the rate estimate",
            fontsize=6.4, color=MUTED, va="top")

    # lower branch: read the distribution
    arrow(3.75, 4.5, 4.35, 2.8, DIST)
    box(4.4, 2.1, 2.0, 1.4, "salience $s(r)$\nover rates", ec=DIST)
    arrow(6.45, 2.8, 6.95, 2.8, DIST)
    box(7.0, 2.1, 2.8, 1.4, "$p=\\sum_{r\\geq\\tau_{\\mathrm{age}}} s(r)$", ec=DIST)
    arrow(9.85, 2.8, 10.35, 2.8, DIST)
    ax.text(10.45, 2.8, "decision", ha="left", va="center", fontsize=7.0, color=DIST)
    # Title above its own boxes, mirroring the upper branch; description below. Previously the title
    # sat under the description and the two collided.
    ax.text(4.4, 3.78, "Read the distribution (this paper)", fontsize=7.6, color=DIST,
            fontweight="bold")
    ax.text(4.4, 1.92, "uses the whole distribution: mass above the cut point\n"
                       "survives a saturated point estimate",
            fontsize=6.4, color=MUTED, va="top")
    ax.set_title("(a) One encoder, two ways to reach a clinical decision", fontsize=8.5, loc="left")


def panel_b(ax, sal, row):
    tau = float(row.threshold_bpm); rhat = float(row.pred_rr_bpm); true = float(row.gt_rr_bpm)
    mass = float(sal[BINS >= tau].sum())
    ax.fill_between(BINS, 0, sal, where=(BINS >= tau), color=DIST, alpha=0.30, zorder=2,
                    label=f"mass above cut point $=$ {mass:.2f}")
    ax.plot(BINS, sal, color=DIST, lw=1.4, zorder=3)
    ax.axvline(tau, color="0.35", lw=1.0, ls="--", zorder=4)
    ax.axvline(true, color=INK, lw=1.0, zorder=4)
    ax.plot([rhat], [0], marker="v", ms=7, color=REG, clip_on=False, zorder=6)

    top = sal.max()
    ax.set_ylim(0, top * 1.42)
    ax.annotate(f"regressed rate {rhat:.0f}\n(saturated, below cut)", xy=(rhat, top * 0.03),
                xytext=(rhat - 2, top * 0.52), fontsize=6.4, color=REG, ha="right",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=REG))
    ax.text(tau - 1.5, top * 1.30, "WHO cut point\n60", fontsize=6.4, color="0.35",
            ha="right", va="top")
    ax.text(true + 1.5, top * 1.30, f"true rate\n{true:.0f}", fontsize=6.4, color=INK,
            ha="left", va="top")
    ax.set_xlabel("Respiratory rate (breaths/min)")
    ax.set_ylabel("salience  $s(r)$")
    ax.set_xlim(6, 90)
    ax.legend(frameon=False, loc="upper left", fontsize=6.4, handlelength=1.1, borderpad=0.2)
    ax.set_title(f"(b) A real neonate the rate cannot reach", fontsize=8.5, loc="left")


def panel_c(ax, d, thr_band):
    n = d[(d.age_months < 2) & (d.gt_tachy == 1)].copy()
    n = n.sort_values("gt_rr_bpm").reset_index(drop=True)
    y = np.arange(len(n))
    flagged = n.tachy_prob.to_numpy(float) > thr_band

    ax.hlines(y, n.pred_rr_bpm, n.gt_rr_bpm, color="0.85", lw=1.4, zorder=1)
    ax.scatter(n.pred_rr_bpm, y, s=26, color=REG, zorder=3, label="regressed rate")
    ax.scatter(n.gt_rr_bpm, y, s=26, color=INK, zorder=3, marker="D", label="true rate")
    ax.axvline(60, color="0.35", lw=1.0, ls="--", zorder=2)
    ax.text(60 - 1, len(n) - 0.35, "WHO cut point", fontsize=6.3, color="0.35", ha="right", va="top")

    # One marker column rather than ten repetitions of the same word.
    ax.scatter(np.full(flagged.sum(), 92.0), y[flagged], s=22, marker="s", color=FLAG,
               zorder=3, label="flagged by mass read-out")
    if (~flagged).any():
        ax.scatter(np.full((~flagged).sum(), 92.0), y[~flagged], s=26, marker="x", color=MISS,
                   zorder=3, lw=1.4, label="missed")
    ax.set_xlim(20, 97); ax.set_ylim(-1.7, len(n) - 0.2)
    ax.set_yticks([]); ax.set_xticks([20, 40, 60, 80])
    ax.set_xlabel("Respiratory rate (breaths/min)")
    ax.set_ylabel(f"tachypneic neonates (n={len(n)})")
    # Below the data rather than on top of it: the lowest row sits at y=0, so the strip under it is free.
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=3,
              fontsize=6.3, handlelength=1.0, columnspacing=1.0, handletextpad=0.4)
    ax.set_title("(c) Every regressed rate falls short; the mass read-out still flags "
                 f"{int(flagged.sum())}/{len(n)}", fontsize=8.5, loc="left")
    return int(flagged.sum()), len(n)


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.titlesize": 8.5, "axes.labelsize": 7.6,
                         "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                         "axes.linewidth": 0.7, "axes.spines.top": False,
                         "axes.spines.right": False})
    d = load_preds()
    row = d[d.subject_id == CASE].iloc[0]
    sal = salience_for(CASE, int(row.fold), row.signal_path)

    # The deployed <2mo operating point, from the per-band read-out (Sec. age).
    thr_band = 0.077

    fig = plt.figure(figsize=(7.16, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.05], hspace=0.42, wspace=0.28,
                          left=0.075, right=0.975, top=0.93, bottom=0.10)
    panel_a(fig.add_subplot(gs[0, :]))
    panel_b(fig.add_subplot(gs[1, 0]), sal, row)
    nf, nt = panel_c(fig.add_subplot(gs[1, 1]), d, thr_band)

    fig.savefig(OUT + ".pdf", dpi=600, bbox_inches="tight")
    fig.savefig(OUT + ".png", dpi=600, bbox_inches="tight")
    print(f"case {CASE}: true {row.gt_rr_bpm:.1f}, regressed {row.pred_rr_bpm:.1f}, "
          f"mass above cut {sal[BINS >= row.threshold_bpm].sum():.3f}")
    print(f"panel (c): {nf}/{nt} tachypneic neonates flagged by the mass read-out")
    print(f"wrote {OUT}.pdf and .png")


if __name__ == "__main__":
    raise SystemExit(main())
