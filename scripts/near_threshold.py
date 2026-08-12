"""(#5) Near-threshold stratification for PRISM on the full pedrr-332 (corrected labels).

Stratify each subject by |gt_rr - WHO_threshold| into <5, 5-10, >=10 bpm bands; per stratum report
n, prevalence, Sens, Spec, AUROC, MAE, and #false-negatives. Tests whether PRISM's clinical value
concentrates near the WHO tachypnea boundary. CPU-only, read-only on cached predictions.

Run:  python scripts/near_threshold.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import _corrected_common as C
import full_metric_comparison as F

RUN = "outputs/paper_runs/pedrr_kfold_learned"
STRATA = [("<5", 0, 5), ("5-10", 5, 10), (">=10", 10, np.inf)]


def stratum_report(d):
    y = d["gt_tachy"].values.astype(int)
    dec = (d["pred_rr"].values >= d["threshold"].values).astype(int)
    tp = int(((dec == 1) & (y == 1)).sum()); fn = int(((dec == 0) & (y == 1)).sum())
    tn = int(((dec == 0) & (y == 0)).sum()); fp = int(((dec == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    auroc = roc_auc_score(y, d["tachy_prob"].values) if len(np.unique(y)) == 2 else np.nan
    mae = np.abs(d["pred_rr"].values - d["gt_rr"].values).mean()
    return dict(n=len(d), prev=y.mean(), sens=sens, spec=spec, auroc=auroc, mae=mae, fn=fn, npos=int(y.sum()))


if __name__ == "__main__":
    d = C.load_corrected(RUN)
    d["dist"] = np.abs(d["gt_rr"].values - d["threshold"].values)
    print(f"PRISM full n={len(d)} corrected prevalence={d.gt_tachy.mean():.4f} ({int(d.gt_tachy.sum())})\n")
    print(f"{'stratum':>8}{'n':>5}{'pos':>5}{'prev':>7}{'Sens':>7}{'Spec':>7}{'AUROC':>8}{'MAE':>7}{'#FN':>5}")
    rows = {}
    for name, lo, hi in STRATA:
        s = d[(d.dist >= lo) & (d.dist < hi)]
        r = stratum_report(s); rows[name] = r
        au = f"{r['auroc']:.3f}" if r['auroc'] == r['auroc'] else "  nan"
        print(f"{name:>8}{r['n']:>5}{r['npos']:>5}{r['prev']:>7.3f}{r['sens']:>7.3f}{r['spec']:>7.3f}{au:>8}{r['mae']:>7.2f}{r['fn']:>5}")
    # overall
    ov = stratum_report(d)
    print(f"{'ALL':>8}{ov['n']:>5}{ov['npos']:>5}{ov['prev']:>7.3f}{ov['sens']:>7.3f}{ov['spec']:>7.3f}{ov['auroc']:>8.3f}{ov['mae']:>7.2f}{ov['fn']:>5}")
