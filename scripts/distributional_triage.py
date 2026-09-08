"""Drive the escalation triage from the DISTRIBUTIONAL read-out instead of the regressed rate.

Why this experiment exists
--------------------------
The paper's thesis is that a screening decision should be read off the salience distribution
(mass above the WHO age threshold) rather than off a regressed rate that is then thresholded.
The deployed escalation pathway, however, does the opposite: it forms a conformal interval around
`pred_rr_bpm` and refers whenever that interval straddles the cutoff. So the triage layer is built
on exactly the read-out the paper argues against, and it inherits that read-out's weakness -- in the
neonatal band the regressed rate saturates near 30 for true rates of 60-80, so the interval has to be
made enormous (~19 breaths/min) before it can straddle the cutoff, which forces a high referral rate
everywhere else as collateral.

This script builds the coherent alternative and measures it against the incumbent:

  A. regression + Mondrian conformal interval   (the incumbent; what the paper currently reports)
  B. distributional two-threshold triage        (proposed: abstain on the uncertain band of tachy_prob)

Rule B. Per age band, pick (t_lo, t_hi) so that
    NORMAL  if p < t_lo        subject to  false-clearance rate  P(tachypneic | NORMAL) <= delta
    ALERT   if p > t_hi        subject to  false-alarm rate      P(healthy | ALERT)     <= gamma
    REFER   otherwise
choosing the WIDEST clearing/alerting regions that respect those constraints, i.e. the SMALLEST
referral rate consistent with the safety budget. Both thresholds are fitted leave-one-fold-out, so a
subject's own fold never informs the thresholds applied to it -- the same leakage discipline as the
conformal arm.

Both arms are evaluated on the identical subjects, so referral rate and retained sensitivity are
directly comparable.

CPU-only; reads cached per-fold prediction CSVs. No training, no GPU.

Run:  python scripts/distributional_triage.py
      python scripts/distributional_triage.py --delta 0.05 --gamma 0.30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = ["<2mo", "2-12mo", "12-60mo"]
# The published model: SSL-free, deterministic, six seeds. See train_resp_salience_ssl.py.
SALIENCE = [f"pedrr_kfold_salience_nossl332det_seed{i}" for i in range(6)]
PRISM = "pedrr_kfold_main332"
ALPHA = 0.20


def band(a):
    return "<2mo" if a < 2 else ("2-12mo" if a < 12 else "12-60mo")


def conf_q(res, alpha=ALPHA):
    res = np.asarray(res, float)
    n = len(res)
    if n == 0:
        return np.nan
    k = min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n)
    return float(np.sort(res)[k - 1])


def load_run(run):
    """Subject-level frame for one k-fold run."""
    ps = sorted(glob.glob(os.path.join(ROOT, "outputs", "paper_runs", run,
                                       "fold*", "test_eval", "prism_breath_predictions.csv")))
    if not ps:
        return None
    parts = []
    for p in ps:
        x = pd.read_csv(p)
        x["fold"] = int(re.search(r"fold(\d+)", p).group(1))
        parts.append(x)
    df = pd.concat(parts, ignore_index=True)
    g = df.groupby("subject_id", sort=False)
    sub = g.agg(fold=("fold", "first"), age_months=("age_months", "first"),
                gt_rr_bpm=("gt_rr_bpm", "mean"), gt_tachy=("gt_tachy", "max"),
                pred_rr_bpm=("pred_rr_bpm", "mean"), tachy_prob=("tachy_prob", "mean"),
                T=("threshold_bpm", "first")).reset_index()
    sub["band"] = sub["age_months"].apply(band)
    sub["res"] = (sub["pred_rr_bpm"] - sub["gt_rr_bpm"]).abs()
    return sub


def ensemble(runs):
    """Seed-ensemble the salience model exactly as the headline table does (mean of probabilities)."""
    ds = [load_run(r) for r in runs]
    ds = [d for d in ds if d is not None]
    if not ds:
        return None
    sid = sorted(ds[0].subject_id)
    base = ds[0].set_index("subject_id").loc[sid].copy()
    base["tachy_prob"] = np.mean([d.set_index("subject_id").loc[sid].tachy_prob.to_numpy(float)
                                  for d in ds], axis=0)
    base["pred_rr_bpm"] = np.mean([d.set_index("subject_id").loc[sid].pred_rr_bpm.to_numpy(float)
                                   for d in ds], axis=0)
    base["res"] = (base["pred_rr_bpm"] - base["gt_rr_bpm"]).abs()
    return base.reset_index()


# ----------------------------------------------------------------------------------------------
# Arm A: regression + Mondrian conformal interval (the incumbent)
# ----------------------------------------------------------------------------------------------
def triage_conformal(sub):
    q = np.empty(len(sub))
    for k in sorted(sub.fold.unique()):
        for b in sub.band.unique():
            q[((sub.fold == k) & (sub.band == b)).values] = conf_q(
                sub.loc[(sub.fold != k) & (sub.band == b), "res"].to_numpy(float))
    lo = sub.pred_rr_bpm.to_numpy(float) - q
    hi = sub.pred_rr_bpm.to_numpy(float) + q
    T = sub["T"].to_numpy(float)
    return np.where(hi < T, 0, np.where(lo >= T, 1, 2)).astype(int), q


# ----------------------------------------------------------------------------------------------
# Arm B: distributional two-threshold triage
# ----------------------------------------------------------------------------------------------
def _fit_thresholds(p, y, delta, gamma):
    """Widest clearing / alerting regions whose empirical error respects the safety budget.

    t_lo: largest cut with P(y=1 | p < t_lo) <= delta   (few tachypneic children are cleared)
    t_hi: smallest cut with P(y=0 | p > t_hi) <= gamma  (alerts are not overwhelmingly false)
    Returns (t_lo, t_hi) with t_lo <= t_hi enforced, so the REFER band is never negative-width.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    if len(p) == 0:
        return 0.0, 1.0
    cuts = np.unique(np.concatenate([[0.0], p, [1.0]]))

    t_lo = 0.0
    for c in cuts:                      # ascending: keep widening the cleared region while it is safe
        m = p < c
        if m.sum() == 0:
            t_lo = c
            continue
        if y[m].mean() <= delta:
            t_lo = c
        else:
            break

    t_hi = 1.0
    for c in cuts[::-1]:                # descending: keep widening the alert region while it is precise
        m = p > c
        if m.sum() == 0:
            t_hi = c
            continue
        if (1.0 - y[m].mean()) <= gamma:
            t_hi = c
        else:
            break

    if t_lo > t_hi:                     # budgets in conflict: fall back to a single split point
        t_lo = t_hi = 0.5 * (t_lo + t_hi)
    return float(t_lo), float(t_hi)


def triage_distributional(sub, delta, gamma, per_band=True):
    """Leave-one-fold-out two-threshold abstention on the salience mass."""
    code = np.empty(len(sub), int)
    lo_used = np.empty(len(sub))
    hi_used = np.empty(len(sub))
    groups = sub.band.unique() if per_band else ["__all__"]
    for k in sorted(sub.fold.unique()):
        for b in groups:
            fit = (sub.fold != k) if b == "__all__" else ((sub.fold != k) & (sub.band == b))
            app = (sub.fold == k) if b == "__all__" else ((sub.fold == k) & (sub.band == b))
            if app.sum() == 0:
                continue
            t_lo, t_hi = _fit_thresholds(sub.loc[fit, "tachy_prob"], sub.loc[fit, "gt_tachy"],
                                         delta, gamma)
            p = sub.loc[app, "tachy_prob"].to_numpy(float)
            code[app.values] = np.where(p < t_lo, 0, np.where(p > t_hi, 1, 2))
            lo_used[app.values] = t_lo
            hi_used[app.values] = t_hi
    return code, lo_used, hi_used


# ----------------------------------------------------------------------------------------------
def screen_metrics(sub, col):
    ret = sub[sub[col] != 2]
    y = ret["gt_tachy"].to_numpy(int)
    dec = (ret[col].to_numpy() == 1).astype(int)
    tp = int(((dec == 1) & (y == 1)).sum()); fn = int(((dec == 0) & (y == 1)).sum())
    tn = int(((dec == 0) & (y == 0)).sum()); fp = int(((dec == 1) & (y == 0)).sum())
    au = np.nan
    if len(ret) and y.min() != y.max():
        au = roc_auc_score(y, ret["tachy_prob"].to_numpy(float))
    return dict(n_ret=len(ret), n_pos=int(y.sum()),
                sens=tp / (tp + fn) if (tp + fn) else np.nan,
                spec=tn / (tn + fp) if (tn + fp) else np.nan,
                auroc=au, missed=fn)


def report(sub, col, title):
    print(f"\n{'=' * 84}\n{title}\n{'=' * 84}")
    print(f"  {'band':9s} {'n':>4s} {'REFER%':>8s} {'nRet':>5s} {'pos':>4s} "
          f"{'Sens':>7s} {'Spec':>7s} {'AUROC':>7s} {'missed':>7s}")
    for b in BANDS + ["overall"]:
        s = sub if b == "overall" else sub[sub.band == b]
        m = screen_metrics(s, col)
        f = lambda v: f"{v:.3f}" if v == v else "    n/a"
        print(f"  {b:9s} {len(s):>4d} {100 * (s[col] == 2).mean():>8.1f} {m['n_ret']:>5d} "
              f"{m['n_pos']:>4d} {f(m['sens']):>7s} {f(m['spec']):>7s} {f(m['auroc']):>7s} "
              f"{m['missed']:>7d}")

    n2 = sub[(sub.band == "<2mo") & (sub.gt_tachy == 1)]
    cam = (n2.pred_rr_bpm.to_numpy(float) >= n2["T"].to_numpy(float)).astype(int)
    miss_cam = int((cam == 0).sum())
    miss_tri = int((n2[col] == 0).sum())
    print(f"\n  tachypneic neonates n={len(n2)}: camera-only clears {miss_cam}, "
          f"triage clears {miss_tri} -> rescues {miss_cam - miss_tri}; "
          f"referred {int((n2[col] == 2).sum())}, alerted {int((n2[col] == 1).sum())}")
    ov = screen_metrics(sub, col)
    return dict(refer=float((sub[col] == 2).mean()), sens=ov["sens"], spec=ov["spec"],
                auroc=ov["auroc"], missed_total=ov["missed"],
                rescued=miss_cam - miss_tri, miss_tri=miss_tri, n_tachy_neo=len(n2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.05,
                    help="max tolerated tachypnea prevalence among auto-cleared cases")
    ap.add_argument("--gamma", type=float, default=0.30,
                    help="max tolerated false-alarm rate among auto-alerted cases")
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "distributional_triage.json"))
    A = ap.parse_args()

    sal = ensemble(SALIENCE)
    pri = load_run(PRISM)
    if sal is None or pri is None:
        print("missing runs"); return 1
    # Evaluate every arm on the identical subjects.
    common = sorted(set(sal.subject_id) & set(pri.subject_id))
    sal = sal[sal.subject_id.isin(common)].reset_index(drop=True)
    pri = pri[pri.subject_id.isin(common)].reset_index(drop=True)
    print(f"n={len(sal)} subjects (RespSalience 3-seed ensemble vs PRISM main run, identical cohort)")
    print(f"safety budget: delta={A.delta} (max tachypnea rate among cleared), "
          f"gamma={A.gamma} (max false-alarm rate among alerts)")

    out = {}
    pri["conf"], qp = triage_conformal(pri)
    out["A_prism_conformal"] = report(
        pri, "conf", "A. INCUMBENT -- PRISM regression + Mondrian conformal interval")

    sal["conf"], qs = triage_conformal(sal)
    out["A2_salience_conformal"] = report(
        sal, "conf", "A2. CONTROL -- RespSalience regression head + Mondrian conformal interval")

    sal["dist"], tlo, thi = triage_distributional(sal, A.delta, A.gamma)
    sal["t_lo"], sal["t_hi"] = tlo, thi
    out["B_salience_distributional"] = report(
        sal, "dist", "B. PROPOSED -- RespSalience distributional two-threshold triage")

    print("\n  fitted thresholds by band (mean over folds):")
    for b in BANDS:
        s = sal[sal.band == b]
        print(f"    {b:9s} clear if p < {s.t_lo.mean():.3f}   alert if p > {s.t_hi.mean():.3f}")

    # Headline comparison ------------------------------------------------------------------
    a, b_ = out["A_prism_conformal"], out["B_salience_distributional"]
    print(f"\n{'=' * 84}\nHEAD-TO-HEAD (identical cohort, identical leakage discipline)\n{'=' * 84}")
    print(f"  {'':34s} {'A: conformal':>14s} {'B: distributional':>18s}")
    for k, lab, fmt in (("refer", "referral rate", "{:.1%}"), ("sens", "retained sensitivity", "{:.3f}"),
                        ("spec", "retained specificity", "{:.3f}"),
                        ("missed_total", "tachypneic missed (all ages)", "{:.0f}"),
                        ("miss_tri", "tachypneic neonates cleared", "{:.0f}"),
                        ("rescued", "neonates rescued vs camera", "{:.0f}")):
        print(f"  {lab:34s} {fmt.format(a[k]):>14s} {fmt.format(b_[k]):>18s}")

    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump({"delta": A.delta, "gamma": A.gamma, "n": len(sal), **out},
              open(A.out, "w"), indent=2, default=float)
    print(f"\nwrote {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
