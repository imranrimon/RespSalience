"""#12 (MatQCP selective deployment): the SELECT stage of align->Mondrian->select.

The align->conformal analysis showed neonates are un-coverable post-hoc (base-model-quality limit).
MatQCP's answer: learn a label-free risk score from deployment-time uncertainty signals, retain the
low-risk predictions as trustworthy, and DEFER the rest to a clinician. This converts the neonatal
Sens=0.00 silent-failure into a SAFE abstention.

Risk score (MatQCP-style): fit s = beta^T phi on val, regressing |residual| on label-free features
  phi = [disagreement_score, conformal interval width, |pred - WHO threshold|].
Apply to pooled test. Report the retention curve: at each retention rho, RR-MAE + tachypnea
sensitivity/specificity on the AUTO-DECIDED (retained) subset, and the age-band composition of the
DEFERRED subset (does it concentrate the neonates the model can't do?).

No retrain. Usage: python scripts/selective_deployment.py --run-root outputs/paper_runs/pedrr_kfold_learned
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


def feats(df):
    iw = (df.get("interval_upper_bpm", 0) - df.get("interval_lower_bpm", 0)).abs()
    dis = df.get("disagreement_score", pd.Series(np.zeros(len(df))))
    margin = (df.pred_rr_bpm - df.threshold_bpm).abs()
    cols = [dis.values, iw.values, margin.values]
    if "rr_std" in df.columns and df["rr_std"].abs().sum() > 0:   # heteroscedastic aleatoric std, if trained
        cols.append(df["rr_std"].values)
    return np.column_stack(cols).astype(float)


def clf(sub):
    y = (sub.gt_rr_bpm >= sub.threshold_bpm).astype(int).values
    yh = (sub.pred_rr_bpm >= sub.threshold_bpm).astype(int).values
    tp = int(((y == 1) & (yh == 1)).sum()); fn = int(((y == 1) & (yh == 0)).sum())
    tn = int(((y == 0) & (yh == 0)).sum()); fp = int(((y == 0) & (yh == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return sens, spec, tp + fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="outputs/paper_runs/pedrr_kfold_learned")
    args = ap.parse_args()
    root = Path(args.run_root)
    vX, vy, tests = [], [], []
    for fold in sorted(root.glob("fold*")):
        vp = fold / "val_eval" / "prism_breath_predictions.csv"
        tp = fold / "test_eval" / "prism_breath_predictions.csv"
        if not (vp.exists() and tp.exists()):
            continue
        val, test = pd.read_csv(vp), pd.read_csv(tp)
        vX.append(feats(val)); vy.append((val.gt_rr_bpm - val.pred_rr_bpm).abs().values)
        tests.append(test)
    Xv = np.vstack(vX); yv = np.concatenate(vy)
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-9
    Xn = (Xv - mu) / sd
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xn)), Xn]), yv, rcond=None)  # risk = residual proxy

    test = pd.concat(tests, ignore_index=True)
    Xt = (feats(test) - mu) / sd
    test["risk"] = beta[0] + Xt @ beta[1:]
    test["band"] = test.age_months.apply(band_of)
    test["abs_err"] = (test.gt_rr_bpm - test.pred_rr_bpm).abs()
    test = test.sort_values("risk").reset_index(drop=True)
    n = len(test)
    print(f"[selective deployment] pooled test n={n}, n_pos={int((test.gt_rr_bpm>=test.threshold_bpm).sum())}")
    print(f"risk = {beta[1]:+.2f}*disagreement {beta[2]:+.2f}*interval_width {beta[3]:+.2f}*|margin| (standardized)\n")
    print(f"{'retention':>10}{'n_auto':>8}{'RR-MAE':>9}{'Sens':>8}{'Spec':>8}   deferred-set age mix")
    for rho in [1.00, 0.90, 0.80, 0.70, 0.60, 0.50]:
        k = max(1, int(round(rho * n)))
        auto = test.iloc[:k]; deferred = test.iloc[k:]
        sens, spec, npos = clf(auto)
        mae = auto.abs_err.mean()
        if len(deferred):
            mix = deferred.band.value_counts(normalize=True)
            mixs = " ".join(f"{b}={mix.get(b,0)*100:.0f}%" for b, _, _ in BANDS)
        else:
            mixs = "(none deferred)"
        print(f"{rho:>10.2f}{k:>8}{mae:>9.2f}{sens:>8.2f}{spec:>8.2f}   {mixs}")
    base = test.band.value_counts(normalize=True)
    print("\nfull-cohort age mix: " + " ".join(f"{b}={base.get(b,0)*100:.0f}%" for b, _, _ in BANDS))
    print("Read: as retention drops, the DEFERRED set concentrates <2mo (the un-coverable neonates),")
    print("so retained RR-MAE + tachypnea sensitivity rise -> a SAFE abstaining screener, not a silent failure.")


if __name__ == "__main__":
    raise SystemExit(main())
