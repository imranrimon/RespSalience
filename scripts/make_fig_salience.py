"""Regenerate the RespSalience figure from the stored predictions.

The previous fig8_salience.pdf had no generator in the repository and its per-band panel carried values
from an earlier 297-subject analysis, so it disagreed with the headline table. Both panels are now
computed from the same prediction files the tables use.

(a) Predicted against true rate. This is the mechanism the paper argues for, made visible: the
    regression read-out compresses toward the training mean and stops tracking above roughly
    35 breaths/min, while the distributional read-out continues up the diagonal. Everything to the
    right of the WHO neonatal threshold is where that difference decides a clinical call.
(b) Tachypnea AUROC by WHO age band with subject-bootstrap intervals, proposed against baseline.

    python scripts/make_fig_salience.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_main_run import load, scored  # noqa: E402

PRISM = "pedrr_kfold_main332"
SALIENCE = [f"pedrr_kfold_salience_ssl332_seed{i}" for i in range(3)]
ACCENT = "#1b4f72"
MUTED = "#9aa7b1"
BANDS = [("<2mo", lambda a: a < 2), ("2-12mo", lambda a: (a >= 2) & (a < 12)),
         ("12-60mo", lambda a: a >= 12)]


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(roc_auc_score(y[i], s[i]))
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if out else (np.nan, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manuscript/fig/fig8_salience")
    a = ap.parse_args()

    ref = load("pedrr_kfold_flowonly_agefix")
    age_map = dict(zip(ref["subject_id"].astype(str), ref["age_months"].astype(float)))
    P = scored(PRISM, age_map)
    S = [scored(r, age_map) for r in SALIENCE]
    sid = sorted(set(P.sid) & set(S[0].sid))
    p = P[P.sid.isin(sid)].sort_values("sid").reset_index(drop=True)
    sg = [s[s.sid.isin(sid)].sort_values("sid").reset_index(drop=True) for s in S]
    sal_rr = np.mean([x.pred_rr_bpm.to_numpy(float) for x in sg], axis=0)
    sal_p = np.mean([x.tachy_prob.to_numpy(float) for x in sg], axis=0)
    gt = p.gt_rr_bpm.to_numpy(float)
    y = p.y.to_numpy(int)
    age = p.age.to_numpy(float)

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1]})

    # ---- (a) predicted vs true -------------------------------------------------------------
    ax1.plot([10, 85], [10, 85], color="0.6", lw=0.7, ls=(0, (4, 3)), zorder=1, label="ideal")
    ax1.scatter(gt, p.pred_rr_bpm, s=7, c=MUTED, alpha=0.65, linewidths=0,
                label="regression baseline", zorder=2)
    ax1.scatter(gt, sal_rr, s=7, c=ACCENT, alpha=0.75, linewidths=0,
                label="distributional (ours)", zorder=3)
    hi = gt >= 60
    ceil = float(np.median(p.pred_rr_bpm.to_numpy(float)[hi])) if hi.any() else np.nan
    ax1.axhline(ceil, color=MUTED, lw=0.8, ls=":", zorder=2)
    ax1.annotate(f"baseline saturates\nnear {ceil:.0f} br/min", xy=(78, ceil), xytext=(60, ceil - 22),
                 fontsize=6.4, color="0.35", ha="center",
                 arrowprops=dict(arrowstyle="->", lw=0.6, color="0.5"))
    ax1.axvspan(60, 85, color="#f2c9c9", alpha=0.35, zorder=0)
    ax1.text(72.5, 88, "neonatal tachypnea", fontsize=6.3, color="#8c3a3a", ha="center")
    ax1.set_xlim(10, 85); ax1.set_ylim(5, 92)
    ax1.set_xlabel("True rate (breaths/min)"); ax1.set_ylabel("Predicted rate (breaths/min)")
    ax1.set_title("(a) The regression read-out stops tracking", fontsize=8, pad=8)
    ax1.legend(fontsize=6.3, frameon=False, loc="upper left", handletextpad=0.4, borderpad=0.2)

    # ---- (b) per-band AUROC ----------------------------------------------------------------
    names, pv, sv, pe, se = [], [], [], [], []
    for label, sel in BANDS:
        m = sel(age)
        if m.sum() == 0 or y[m].sum() in (0, m.sum()):
            continue
        ap_ = roc_auc_score(y[m], p.tachy_prob.to_numpy(float)[m])
        as_ = roc_auc_score(y[m], sal_p[m])
        lo1, hi1 = boot_ci(y[m], p.tachy_prob.to_numpy(float)[m])
        lo2, hi2 = boot_ci(y[m], sal_p[m])
        names.append(f"{label}\n(n={int(m.sum())})")
        pv.append(ap_); sv.append(as_)
        pe.append([ap_ - lo1, hi1 - ap_]); se.append([as_ - lo2, hi2 - as_])

    x = np.arange(len(names)); w = 0.36
    ax2.bar(x - w / 2, pv, w, yerr=np.array(pe).T, color=MUTED, label="regression baseline",
            error_kw=dict(lw=0.7, capsize=2, ecolor="0.35"), zorder=3)
    ax2.bar(x + w / 2, sv, w, yerr=np.array(se).T, color=ACCENT, label="distributional (ours)",
            error_kw=dict(lw=0.7, capsize=2, ecolor="0.2"), zorder=3)
    for xi, (a_, b_) in enumerate(zip(pv, sv)):
        ax2.text(xi, min(1.005, max(a_, b_) + 0.055), f"+{b_ - a_:.2f}", ha="center",
                 fontsize=6.6, color=ACCENT, fontweight="bold")
    ax2.axhline(0.5, color="0.6", lw=0.7, ls=(0, (3, 3)), zorder=1)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=7)
    ax2.set_ylim(0.4, 1.10); ax2.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax2.set_ylabel("Tachypnea AUROC")
    ax2.set_title("(b) Largest gain where the screen is weakest", fontsize=8, pad=8)
    ax2.legend(fontsize=6.3, frameon=False, loc="lower right", handletextpad=0.4, borderpad=0.2)
    ax2.grid(axis="y", color="0.92", lw=0.5, zorder=0); ax2.set_axisbelow(True)

    fig.tight_layout(pad=0.5, w_pad=1.6)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(out.with_suffix(ext), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")
    print("per-band AUROC:", {n.split(chr(10))[0]: (round(a_, 3), round(b_, 3))
                              for n, a_, b_ in zip(names, pv, sv)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
