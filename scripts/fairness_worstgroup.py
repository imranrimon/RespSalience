"""(#19) Worst-group fairness for PRISM on the full pedrr-332 (corrected labels).

Subgroups: age band (<2mo / 2-12mo / 12-60mo), sex (1=Male, 2=Female), and site from the .dta
facility_code (1=ICMH, 2=Upazila Health Complex, 3=Community Clinic). Per subgroup reports
Sens / Spec / AUROC and the positive count; then the WORST-group value per metric. Flags subgroups
whose n or positive count is too small for a reliable estimate. CPU-only, read-only cached predictions.

Run:  python scripts/fairness_worstgroup.py
"""
import numpy as np
from sklearn.metrics import roc_auc_score
import _corrected_common as C

RUN = "outputs/paper_runs/pedrr_kfold_learned"
SMALL_N = 30   # flag subgroups smaller than this
SMALL_POS = 10  # flag subgroups with fewer positives than this (AUROC/Sens unstable)


def grp_metrics(d):
    y = d["gt_tachy"].values.astype(int)
    dec = (d["pred_rr"].values >= d["threshold"].values).astype(int)
    tp = int(((dec == 1) & (y == 1)).sum()); fn = int(((dec == 0) & (y == 1)).sum())
    tn = int(((dec == 0) & (y == 0)).sum()); fp = int(((dec == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    auroc = roc_auc_score(y, d["tachy_prob"].values) if len(np.unique(y)) == 2 else np.nan
    return dict(n=len(d), pos=int(y.sum()), sens=sens, spec=spec, auroc=auroc)


def report_axis(d, axis_name, groups):
    print(f"\n=== {axis_name} ===")
    print(f"{'group':>24}{'n':>5}{'pos':>5}{'Sens':>8}{'Spec':>8}{'AUROC':>8}  flag")
    rows = {}
    for label, sub in groups:
        r = grp_metrics(sub); rows[label] = r
        flag = []
        if r['n'] < SMALL_N: flag.append(f"small n<{SMALL_N}")
        if r['pos'] < SMALL_POS: flag.append(f"pos<{SMALL_POS}")
        au = f"{r['auroc']:.3f}" if r['auroc'] == r['auroc'] else "nan"
        print(f"{label:>24}{r['n']:>5}{r['pos']:>5}{r['sens']:>8.3f}{r['spec']:>8.3f}{au:>8}  {'; '.join(flag)}")
    for met in ["sens", "spec", "auroc"]:
        vals = {k: v[met] for k, v in rows.items() if v[met] == v[met]}
        if vals:
            wk = min(vals, key=vals.get)
            print(f"   WORST {met.upper():5} = {vals[wk]:.3f}  ({wk})")
    return rows


if __name__ == "__main__":
    d = C.load_corrected(RUN)
    print(f"PRISM full n={len(d)}  prevalence={d.gt_tachy.mean():.4f} ({int(d.gt_tachy.sum())})")

    report_axis(d, "AGE BAND", [(b, d[d.band == b]) for b in ["<2mo", "2-12mo", "12-60mo"]])
    report_axis(d, "SEX", [(C.SEX_LABEL[s], d[d.sex == s]) for s in [1, 2]])
    report_axis(d, "SITE (facility_code)", [(C.FAC_LABEL[f], d[d.fac == f]) for f in [1, 2, 3]])
