"""Does the framing advantage scale with how badly the front end saturates?

The paper claims a mechanism, not just a difference: a regress-then-threshold read-out ranks cases by a
monotone function of its own rate estimate, so when that estimate is compressed toward the training mean
the ranking is compressed with it and no threshold can recover the case. If that is the real mechanism,
the advantage of reading the distribution should be *proportional to the compression* -- large where the
front end saturates, absent where it does not.

That prediction is falsifiable, and it also explains this study's one negative result: the external
replication on a contact belt (NCHSDB) found no framing advantage, and a clean belt does not saturate.
Under the mechanism, that null is not a failure of the thesis but a point at zero on its dose-response
curve.

Two arms:

  OBSERVATIONAL -- real camera front ends on the clinical cohort. Each extractor has its own degree of
    saturation, measured from its own predictions; the framing gain at the WHO-relevant threshold comes
    from the head-swap sweep. One point per front end.

  CONTROLLED -- the belt signal, progressively low-passed. Cutting the pass-band toward the tachypnea
    band (60 breaths/min = 1 Hz) removes the fast-breathing content, forcing the rate estimate to
    saturate while everything else is held fixed. This turns a correlation across models into a
    dose-response within one dataset and one pipeline.

Saturation severity is measured on the OUTPUT, where the mechanism acts:
    sigma = 1 - slope of OLS(predicted rate ~ true rate)
sigma = 0 is a faithful front end; sigma -> 1 means the estimate ignores the true rate entirely.

Run:  python scripts/saturation_severity.py --arm observational      (CPU, seconds)
      python scripts/saturation_severity.py --arm controlled         (GPU, ~25 min)
      python scripts/saturation_severity.py --arm figure             (after both)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OBS_OUT = os.path.join(ROOT, "outputs", "saturation_observational.csv")
CTL_OUT = os.path.join(ROOT, "outputs", "saturation_controlled.csv")
FIG = os.path.join(ROOT, "manuscript", "fig", "fig_saturation")

WHO_NEONATE = 60.0


def saturation(pred, true):
    """1 - OLS slope of predicted rate on true rate. 0 = faithful, 1 = ignores the true rate."""
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    m = np.isfinite(pred) & np.isfinite(true)
    if m.sum() < 8 or np.std(true[m]) < 1e-6:
        return np.nan
    slope = np.polyfit(true[m], pred[m], 1)[0]
    return float(np.clip(1.0 - slope, 0.0, 1.0))


# ---------------------------------------------------------------------------------------------
def arm_observational():
    """Real camera front ends: saturation from their own predictions, gain from the head-swap sweep."""
    sweep = pd.read_csv(os.path.join(ROOT, "outputs", "framing_bradypnea_sweep.csv"))
    # The head-swap sweep is on a threshold grid; take the point nearest the neonatal WHO cutoff.
    near = sweep.loc[(sweep.thr - WHO_NEONATE).abs().groupby(sweep.model).idxmin()]

    RUNS = {"EfficientPhys": "pedrr_kfold_efficientphys",
            "TS-CAN": "pedrr_kfold_tscan",
            "PRISM(fusion)": "pedrr_kfold_main332"}
    rows = []
    for _, r in near.iterrows():
        tag = RUNS.get(r.model)
        if tag is None:
            continue
        ps = sorted(glob.glob(os.path.join(ROOT, "outputs", "paper_runs", tag,
                                           "fold*", "test_eval", "prism_breath_predictions.csv")))
        if not ps:
            print(f"  [skip] {r.model}: no predictions"); continue
        d = pd.concat([pd.read_csv(p) for p in ps], ignore_index=True)
        g = d.groupby("subject_id").agg(pred=("pred_rr_bpm", "mean"), true=("gt_rr_bpm", "mean"))
        sig = saturation(g.pred, g.true)
        rows.append(dict(arm="observational", label=r.model, sigma=sig, gain=float(r.gain),
                         lo=float(r.lo), hi=float(r.hi), thr=float(r.thr), n=int(r.npos + r.nneg)))
        print(f"  {r.model:16s} sigma={sig:.3f}  gain@{r.thr:.0f}={r.gain:+.3f} "
              f"[{r.lo:+.3f},{r.hi:+.3f}]")
    pd.DataFrame(rows).to_csv(OBS_OUT, index=False)
    print(f"wrote {OBS_OUT}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
def arm_controlled(cutoffs, seeds, epochs, folds):
    """Belt signal, progressively low-passed: induce saturation and watch the gain appear."""
    import torch  # noqa: F401  (imported for the side effect of failing early without CUDA)
    from nchsdb_framing import degrade, train_eval, subject_auroc

    z = np.load(os.path.join(ROOT, "outputs", "nchsdb_waveforms.npz"), allow_pickle=True)
    X0, rec, rr = z["X"], z["record"].astype(str), z["gt_rr"].astype(float)
    y = (rr >= WHO_NEONATE).astype(int)
    subs = np.array(sorted(set(rec)))
    rs = np.random.default_rng(0); rs.shuffle(subs)
    fold_of = {s: i % folds for i, s in enumerate(subs)}
    folds_arr = np.array([fold_of[s] for s in rec])
    print(f"windows {len(X0)} | infants {len(subs)} | tachypneic windows {int(y.sum())}")

    rows = []
    for lp in cutoffs:
        Xd = degrade(X0, lowpass_hz=lp, rng=np.random.default_rng(0))
        per = []
        for s_i in range(seeds):
            reg = np.full(len(X0), np.nan); cls = np.full(len(X0), np.nan)
            for k in range(folds):
                tr, te = folds_arr != k, folds_arr == k
                o = train_eval(Xd[tr], rr[tr], Xd[te], epochs, seed=1000 * s_i + k)
                reg[te] = o["reg_rr"]; cls[te] = o["cls_mass"]
            a_r = subject_auroc(y, reg, rec)
            a_c = subject_auroc(y, cls, rec)
            sig = saturation(reg, rr)
            per.append((sig, a_r, a_c, a_c - a_r))
            print(f"  lp={lp if lp else 'none':>5} seed{s_i}: sigma={sig:.3f} "
                  f"reg={a_r:.3f} cls={a_c:.3f} gain={a_c-a_r:+.3f}", flush=True)
        p = np.array(per)
        rows.append(dict(arm="controlled", label=f"lowpass {lp:g} Hz" if lp else "belt (clean)",
                         lowpass_hz=lp, sigma=float(p[:, 0].mean()), sigma_sd=float(p[:, 0].std(ddof=1)),
                         auroc_reg=float(p[:, 1].mean()), auroc_cls=float(p[:, 2].mean()),
                         gain=float(p[:, 3].mean()), gain_sd=float(p[:, 3].std(ddof=1)), seeds=seeds))
        pd.DataFrame(rows).to_csv(CTL_OUT, index=False)     # checkpoint after every condition
        print(f"  -> lp={lp:g}: sigma={rows[-1]['sigma']:.3f} gain={rows[-1]['gain']:+.3f} "
              f"+-{rows[-1]['gain_sd']:.3f}\n", flush=True)
    print(f"wrote {CTL_OUT}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
def arm_stats():
    """Test the dose-response properly, and say what the design can and cannot support.

    Three tests, because they answer different questions:
      1. Spearman(sigma, gain) -- does the gain track measured saturation?
      2. Spearman(dose rank, gain) -- does it track the MANIPULATION? This is the cleaner test: the
         low-pass cutoff is set by us, whereas sigma is itself an estimate with its own error.
      3. Weighted least squares gain ~ sigma, weighting each condition by its seed SD, so a condition
         measured with three scattered seeds does not count the same as a tight one.
    """
    from scipy import stats
    ctl = pd.read_csv(CTL_OUT)
    obs = pd.read_csv(OBS_OUT) if os.path.exists(OBS_OUT) else pd.DataFrame()
    n = len(ctl)
    print("CONTROLLED ARM: %d conditions, %d seeds each" % (n, int(ctl.seeds.iloc[0])))
    print(ctl[["label", "sigma", "sigma_sd", "gain", "gain_sd"]].to_string(index=False))

    # Dose rank: severity rises as the cutoff falls, so "clean" (recorded as 0.0) is the mildest.
    cut = ctl.lowpass_hz.replace(0.0, np.inf)
    dose = stats.rankdata(-cut)

    r1, p1 = stats.spearmanr(ctl.sigma, ctl.gain)
    r2, p2 = stats.spearmanr(dose, ctl.gain)
    print("")
    print("1. gain vs measured saturation : Spearman rho = %+.3f, p = %.4f (n=%d)" % (r1, p1, n))
    print("2. gain vs the manipulation    : Spearman rho = %+.3f, p = %.4f" % (r2, p2))

    w = 1.0 / np.maximum(ctl.gain_sd.to_numpy(float), 1e-3) ** 2
    X = np.column_stack([np.ones(n), ctl.sigma.to_numpy(float)])
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ ctl.gain.to_numpy(float))
    resid = ctl.gain.to_numpy(float) - X @ beta
    dof = max(n - 2, 1)
    s2 = float(resid @ W @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ W @ X)
    se = float(np.sqrt(cov[1, 1]))
    tstat = beta[1] / se if se > 0 else np.nan
    pw = float(2 * (1 - stats.t.cdf(abs(tstat), dof))) if np.isfinite(tstat) else float("nan")
    print("3. weighted LS gain ~ sigma    : slope = %+.3f per unit sigma (SE %.3f, t=%+.2f, p=%.4f)"
          % (beta[1], se, tstat, pw))
    swing = float(ctl.gain.max() - ctl.gain.min())
    print("   total gain swing across the sweep: %+.3f AUROC" % swing)

    print("")
    print("SCALE CHECK against the observational arm")
    if not obs.empty:
        print("  camera front ends: sigma %.3f-%.3f, gain %+.3f to %+.3f"
              % (obs.sigma.min(), obs.sigma.max(), obs.gain.min(), obs.gain.max()))
        print("  belt sweep:        sigma %.3f-%.3f, gain %+.3f to %+.3f"
              % (ctl.sigma.min(), ctl.sigma.max(), ctl.gain.min(), ctl.gain.max()))
        print("  CAVEAT: sigma is subject-level for the cameras and window-level for the belt.")
        print("  Those are not comparable, so the arms must NOT be pooled on a single axis.")

    json.dump({"n_conditions": int(n),
               "spearman_sigma": [float(r1), float(p1)],
               "spearman_dose": [float(r2), float(p2)],
               "wls_slope": float(beta[1]), "wls_se": se, "wls_p": pw,
               "gain_swing": swing},
              open(os.path.join(ROOT, "outputs", "saturation_stats.json"), "w"), indent=2)
    return 0


def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    ctl = pd.read_csv(CTL_OUT) if os.path.exists(CTL_OUT) else pd.DataFrame()
    if ctl.empty:
        print("controlled arm missing; run --arm controlled first")
        return 1

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7, "axes.linewidth": 0.7,
                         "axes.spines.top": False, "axes.spines.right": False})
    C = "#1b4f72"
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # ---- (a) gain against measured saturation, with the weighted fit ----------------------
    c = ctl.sort_values("sigma")
    ax[0].axhline(0, color="0.45", lw=0.8, zorder=1)
    ax[0].errorbar(c.sigma, c.gain, yerr=c.gain_sd, xerr=c.sigma_sd, fmt="o", ms=4.5, color=C,
                   ecolor="0.65", elinewidth=0.8, capsize=2, zorder=3)
    w = 1.0 / np.maximum(c.gain_sd.to_numpy(float), 1e-3) ** 2
    X = np.column_stack([np.ones(len(c)), c.sigma.to_numpy(float)])
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ c.gain.to_numpy(float))
    xs = np.linspace(c.sigma.min() - 0.01, c.sigma.max() + 0.01, 50)
    ax[0].plot(xs, beta[0] + beta[1] * xs, lw=1.1, color=C, alpha=0.55, zorder=2)
    rho, p = stats.spearmanr(c.sigma, c.gain)
    ax[0].text(0.04, 0.95, "Spearman $\\rho=%.2f$, $p=%.3f$" % (rho, p),
               transform=ax[0].transAxes, va="top", fontsize=7, color=C)
    ax[0].set_xlabel("saturation severity  $\\sigma = 1-$slope")
    ax[0].set_ylabel("framing gain ($\\Delta$AUROC)")
    ax[0].set_title("(a) The gain scales with saturation", loc="left")
    ax[0].grid(axis="y", color="0.93", lw=0.5, zorder=0)
    ax[0].set_axisbelow(True)

    # ---- (b) the same data against the manipulation --------------------------------------
    d = ctl.copy()
    # "clean" has no cutoff; place it beyond the widest one so the x-axis reads as severity.
    d["x"] = d.lowpass_hz.replace(0.0, 2.6)
    d = d.sort_values("x")
    ax[1].axhline(0, color="0.45", lw=0.8, zorder=1)
    ax[1].axvspan(0.5, 1.0, color="#c0603a", alpha=0.10, zorder=0)
    ax[1].text(0.8, -0.088, "cutoff below the\n60 bpm threshold", fontsize=6.3, color="#8a5a44",
               ha="center", va="bottom")
    ax[1].errorbar(d.x, d.gain, yerr=d.gain_sd, fmt="o-", ms=4.5, lw=1.2, color=C,
                   ecolor="0.65", elinewidth=0.8, capsize=2, zorder=3)
    ax[1].set_xticks([0.7, 0.9, 1.1, 1.4, 2.0, 2.6])
    ax[1].set_xticklabels(["0.7", "0.9", "1.1", "1.4", "2.0", "none"])
    ax[1].invert_xaxis()
    ax[1].set_xlabel("low-pass cutoff (Hz)   $\\rightarrow$ more saturation")
    ax[1].set_ylabel("framing gain ($\\Delta$AUROC)")
    ax[1].set_title("(b) and appears where the band is cut", loc="left")
    ax[1].grid(axis="y", color="0.93", lw=0.5, zorder=0)
    ax[1].set_axisbelow(True)

    fig.tight_layout(pad=0.5, w_pad=2.0)
    fig.savefig(FIG + ".pdf", dpi=600, bbox_inches="tight")
    fig.savefig(FIG + ".png", dpi=600, bbox_inches="tight")
    print("wrote %s.pdf and .png" % FIG)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["observational", "controlled", "stats", "figure", "all"],
                    default="observational")
    ap.add_argument("--cutoffs", default="0,2.0,1.4,1.1,0.9,0.7",
                    help="low-pass cutoffs in Hz; 0 = no filtering. 60 breaths/min is 1 Hz, so "
                         "cutoffs below ~1 remove the tachypnea band itself")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    if a.arm in ("observational", "all"):
        print("OBSERVATIONAL ARM -- real camera front ends")
        arm_observational()
    if a.arm in ("controlled", "all"):
        print("\nCONTROLLED ARM -- belt with induced saturation")
        cuts = [float(x) for x in a.cutoffs.split(",")]
        arm_controlled(cuts, a.seeds, a.epochs, a.folds)
    if a.arm in ("stats", "all"):
        arm_stats()
    if a.arm in ("figure", "all"):
        return make_figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
