"""(#15) Calibration of PRISM tachy_prob vs corrected gt_tachy on the full pedrr-332.

Reports: logistic recalibration intercept a and slope b from logit(gt) ~ a + b*logit(prob)
(a=0, b=1 is perfectly calibrated), Brier score, log-loss, ECE (10-bin), and ICI (integrated
calibration index = mean |prob - lowess(observed|prob)|). CPU-only, read-only cached predictions.

Run:  python scripts/calibration_full.py
"""
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
import _corrected_common as C

RUN = "outputs/paper_runs/pedrr_kfold_learned"
EPS = 1e-6


def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return tot


if __name__ == "__main__":
    d = C.load_corrected(RUN)
    y = d["gt_tachy"].values.astype(int)
    p = np.clip(d["tachy_prob"].values.astype(float), EPS, 1 - EPS)
    print(f"PRISM full n={len(d)}  prevalence={y.mean():.4f} ({int(y.sum())})  mean(prob)={p.mean():.4f}\n")

    # logistic recalibration: logit(gt) ~ a + b*logit(prob)
    lp = np.log(p / (1 - p))
    X = sm.add_constant(lp)
    res = sm.Logit(y, X).fit(disp=0)
    a, b = res.params[0], res.params[1]
    a_ci, b_ci = res.conf_int()[0], res.conf_int()[1]

    brier = brier_score_loss(y, p)
    ll = log_loss(y, p, labels=[0, 1])
    e = ece(y, p, 10)

    # ICI (Austin & Steyerberg): loess of observed on predicted, mean |pred - smoothed obs|.
    # it=0 => non-robust loess (robustifying iterations collapse the fit for a binary outcome).
    sm_fit = lowess(y, p, frac=0.75, it=0, return_sorted=True)
    obs_hat = np.clip(np.interp(p, sm_fit[:, 0], sm_fit[:, 1]), 0, 1)
    ici = np.mean(np.abs(p - obs_hat))

    print("=== logistic recalibration  logit(gt) ~ a + b*logit(prob)  (ideal a=0, b=1) ===")
    print(f"  intercept a = {a:+.4f}  [95% CI {a_ci[0]:+.4f}, {a_ci[1]:+.4f}]")
    print(f"  slope     b = {b:+.4f}  [95% CI {b_ci[0]:+.4f}, {b_ci[1]:+.4f}]")
    print(f"\n=== proper scores / calibration error ===")
    print(f"  Brier    = {brier:.4f}")
    print(f"  Log-loss = {ll:.4f}")
    print(f"  ECE(10)  = {e:.4f}")
    print(f"  ICI      = {ici:.4f}")
