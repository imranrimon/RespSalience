"""#24 (MatQCP align->conformal): quantile-mapping alignment BEFORE Mondrian conformal.

The earlier raw Mondrian conformal failed to fix neonatal coverage (<2mo stayed ~0.55) because the
neonatal predictions are systematically BIASED (large under-prediction) and a symmetric interval
around a biased prediction cannot cover. MatQCP's fix (Rahman et al., MatQCP): first learn a monotone
quantile-mapping g: pred -> target-scale to remove the systematic output-space bias, THEN calibrate
Mondrian split-conformal on the aligned residuals.

Per fold (k-fold CV): fit g on val (map empirical quantiles of val preds to val gts), align val+test
preds, compute residuals on the ALIGNED preds, estimate pooled + per-WHO-band conformal radii, apply
to test. Reports per-band empirical coverage + width for: (a) raw pooled (current), (b) align+pooled,
(c) align+Mondrian; and the RR-MAE before/after alignment.

Caveat: uses val for both alignment and calibration (MatQCP uses disjoint splits); a demonstration of
the align->conformal principle on the CV predictions. No retrain.

Usage: python scripts/mondrian_conformal.py --run-root outputs/paper_runs/pedrr_kfold_learned --alpha 0.20
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BANDS = [("<2mo", 0, 2), ("2-12mo", 2, 12), ("12-60mo", 12, 60)]


def band_of(a):
    for b, lo, hi in BANDS:
        if lo <= a < hi:
            return b
    return "12-60mo"


def qhat(scores, alpha):
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        return np.nan
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(s[k - 1])


def fit_qmap(pred_cal, gt_cal):
    """Monotone non-parametric quantile map: pred -> target scale (histogram matching)."""
    sp, sg = np.sort(np.asarray(pred_cal, float)), np.sort(np.asarray(gt_cal, float))
    return lambda x: np.interp(x, sp, sg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/paper_runs/pedrr_kfold_learned")
    ap.add_argument("--alpha", type=float, default=0.20)
    args = ap.parse_args()
    root = Path(args.run_root)
    rows = []
    for fold in sorted(root.glob("fold*")):
        vp = fold / "val_eval" / "prism_breath_predictions.csv"
        tp = fold / "test_eval" / "prism_breath_predictions.csv"
        if not (vp.exists() and tp.exists()):
            continue
        val, test = pd.read_csv(vp), pd.read_csv(tp)
        for d in (val, test):
            d["band"] = d.age_months.apply(band_of)
        g = fit_qmap(val.pred_rr_bpm.values, val.gt_rr_bpm.values)   # align on val
        val["al"] = g(val.pred_rr_bpm.values)
        test["al"] = g(test.pred_rr_bpm.values)
        val["r_raw"] = (val.gt_rr_bpm - val.pred_rr_bpm).abs()
        val["r_al"] = (val.gt_rr_bpm - val.al).abs()
        qh_raw = qhat(val.r_raw.values, args.alpha)
        qh_alp = qhat(val.r_al.values, args.alpha)
        qh_alb = {b: qhat(val[val.band == b].r_al.values, args.alpha) for b, _, _ in BANDS}
        for _, r in test.iterrows():
            qb = qh_alb.get(r["band"])
            qb = qh_alp if (qb is None or not np.isfinite(qb)) else qb
            e_raw = abs(r.gt_rr_bpm - r.pred_rr_bpm)
            e_al = abs(r.gt_rr_bpm - r.al)
            rows.append({"band": r["band"], "mae_raw": e_raw, "mae_al": e_al,
                         "cov_raw": e_raw <= qh_raw, "w_raw": 2 * qh_raw,
                         "cov_alp": e_al <= qh_alp, "w_alp": 2 * qh_alp,
                         "cov_alm": e_al <= qb, "w_alm": 2 * qb})
    d = pd.DataFrame(rows)
    print(f"[align->conformal] {len(d)} test cases  alpha={args.alpha} (target coverage {1-args.alpha:.2f})\n")
    print(f"{'band':<9}{'n':>4}  | {'MAE raw':>8}{'MAE algn':>9}  | "
          f"{'cov RAW':>8}{'cov ALIGN+pool':>16}{'cov ALIGN+Mondrian':>20}  {'width A+M':>10}")
    for b, _, _ in BANDS + [("OVERALL", 0, 999)]:
        g = d if b == "OVERALL" else d[d.band == b]
        if len(g) == 0:
            continue
        print(f"{b:<9}{len(g):>4}  | {g.mae_raw.mean():>8.2f}{g.mae_al.mean():>9.2f}  | "
              f"{g.cov_raw.mean():>8.3f}{g.cov_alp.mean():>16.3f}{g.cov_alm.mean():>20.3f}  {g.w_alm.mean():>10.1f}")
    print("\nRead: alignment (quantile-map) de-biases predictions; align+Mondrian should bring each band")
    print("toward the 0.80 target — esp. <2mo, which raw pooled/Mondrian under-covered (~0.55-0.67).")


if __name__ == "__main__":
    raise SystemExit(main())
