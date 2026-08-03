"""Clinical-endpoint evaluation harness for non-contact tachypnea/ARI screening.

This is the evaluation the redesigned PRISM-Breath paper is judged on (see
docs/PRISM_BREATH_CLINICAL_REDESIGN.md, Pillar A). It reframes the metric from
RR-MAE to the clinical decision — fast breathing (tachypnea) at WHO age thresholds
(>=60 BPM <2mo, >=50 2-12mo, >=40 12-60mo) — reported BY AGE BAND and aggregated
across seeds, with:
  * RR agreement: MAE, RMSE, Pearson r, Bland-Altman bias + 95% limits of agreement
  * Tachypnea classification: sensitivity, specificity, PPV, NPV, F1, balanced acc
  * Probabilistic: AUROC, AUPRC (from tachy_prob)
  * Operating points: spec@sens>=0.90, sens@spec>=0.80, Youden-J
  * Selective prediction (risk-coverage): defer least-confident cases; coverage needed
    to reach sens>=0.90 on auto-decided cases
  * Net-benefit / decision curve across the clinically relevant threshold-probability grid

Reads the per-seed prediction CSVs written by evaluate_prism_breath.py.

Usage:
  python scripts/clinical_endpoint_eval.py \
      --run-root outputs/paper_runs/pedrr_indist --setting calibrated \
      --output-dir outputs/paper_runs/pedrr_indist/clinical_endpoint
  # or explicit CSVs:
  python scripts/clinical_endpoint_eval.py --predictions a.csv b.csv c.csv --output-dir ...
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False

# WHO ARI age bands (months) and the tachypnea threshold (breaths/min) for each.
AGE_BANDS = [("<2mo", 0, 2, 60), ("2-12mo", 2, 12, 50), ("12-60mo", 12, 60, 40)]
SETTING_SUBDIRS = {"calibrated": "test_eval", "uncalibrated": "test_eval_uncalibrated"}


def _seed_from_path(path: Path) -> int:
    m = re.search(r"seed[_-]?(\d+)", str(path))
    return int(m.group(1)) if m else -1


def resolve_prediction_paths(predictions: List[str], run_root: str, setting: str) -> List[Path]:
    if predictions:
        return [Path(p) for p in predictions]
    root = Path(run_root)
    subdir = SETTING_SUBDIRS[setting]
    hits = sorted(root.glob(f"seed*/{subdir}/prism_breath_predictions.csv"))
    if not hits:
        raise FileNotFoundError(f"No prediction CSVs under {root}/seed*/{subdir}/")
    return hits


def load_predictions(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["seed"] = _seed_from_path(p)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # drop rows with NaN RR (extractor failures) so metrics don't choke
    nan_cols = [c for c in ["gt_bpm", "pred_bpm"] if c in df.columns]
    if nan_cols:
        before = len(df)
        df = df.dropna(subset=nan_cols).reset_index(drop=True)
        if len(df) < before:
            print(f"[load] dropped {before - len(df)} rows with NaN RR")
    # WHO threshold: prefer the per-row threshold_bpm the evaluator wrote; else derive.
    if "threshold_bpm" not in df.columns or df["threshold_bpm"].isna().any():
        df["threshold_bpm"] = df["age_months"].apply(_who_threshold)
    # gt tachypnea label: prefer explicit gt_tachy; else derive from gt vs threshold.
    if "gt_tachy" not in df.columns:
        df["gt_tachy"] = (df["gt_bpm"] >= df["threshold_bpm"]).astype(int)
    df["pred_tachy"] = (df["pred_bpm"] >= df["threshold_bpm"]).astype(int)
    df["age_band"] = df["age_months"].apply(_age_band)
    return df


def _who_threshold(age_months: float) -> float:
    for _, lo, hi, thr in AGE_BANDS:
        if lo <= age_months < hi:
            return float(thr)
    return 40.0  # >=60mo falls back to the oldest band's threshold


def _age_band(age_months: float) -> str:
    for name, lo, hi, _ in AGE_BANDS:
        if lo <= age_months < hi:
            return name
    return "12-60mo"


def _safe(n, d):
    return float(n) / float(d) if d else float("nan")


def rr_metrics(sub: pd.DataFrame) -> Dict[str, float]:
    gt, pr = sub["gt_bpm"].to_numpy(float), sub["pred_bpm"].to_numpy(float)
    diff = pr - gt
    r = float(np.corrcoef(gt, pr)[0, 1]) if gt.std() > 1e-9 and pr.std() > 1e-9 else float("nan")
    return {
        "n": int(len(sub)),
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "pearson_r": r,
        "ba_bias": float(diff.mean()),
        "ba_loa_lo": float(diff.mean() - 1.96 * diff.std()),
        "ba_loa_hi": float(diff.mean() + 1.96 * diff.std()),
    }


def clf_metrics(sub: pd.DataFrame) -> Dict[str, float]:
    y = sub["gt_tachy"].to_numpy(int)
    yhat = sub["pred_tachy"].to_numpy(int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    sens = _safe(tp, tp + fn)
    spec = _safe(tn, tn + fp)
    out = {
        "n_pos": int(y.sum()),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": _safe(tp, tp + fp),
        "npv": _safe(tn, tn + fn),
        "f1": _safe(2 * tp, 2 * tp + fp + fn),
        "balanced_acc": float(np.nanmean([sens, spec])),
    }
    if _HAVE_SK and 0 < y.sum() < len(y) and "tachy_prob" in sub.columns:
        s = sub["tachy_prob"].to_numpy(float)
        try:
            out["auroc"] = float(roc_auc_score(y, s))
            out["auprc"] = float(average_precision_score(y, s))
        except Exception:
            out["auroc"] = out["auprc"] = float("nan")
    else:
        out["auroc"] = out["auprc"] = float("nan")
    return out


def operating_points(sub: pd.DataFrame) -> Dict[str, float]:
    """Threshold sweep on tachy_prob: spec@sens>=.90, sens@spec>=.80, Youden-J."""
    y = sub["gt_tachy"].to_numpy(int)
    if not (_HAVE_SK and 0 < y.sum() < len(y) and "tachy_prob" in sub.columns):
        return {"spec_at_sens90": float("nan"), "sens_at_spec80": float("nan"),
                "youden_j": float("nan"), "youden_thr": float("nan")}
    s = sub["tachy_prob"].to_numpy(float)
    fpr, tpr, thr = roc_curve(y, s)
    spec = 1 - fpr
    spec_at_sens90 = float(np.nanmax(np.where(tpr >= 0.90, spec, np.nan))) if (tpr >= 0.90).any() else float("nan")
    sens_at_spec80 = float(np.nanmax(np.where(spec >= 0.80, tpr, np.nan))) if (spec >= 0.80).any() else float("nan")
    j = tpr - fpr
    k = int(np.argmax(j))
    return {"spec_at_sens90": spec_at_sens90, "sens_at_spec80": sens_at_spec80,
            "youden_j": float(j[k]), "youden_thr": float(thr[k])}


def risk_coverage(sub: pd.DataFrame, coverages=(1.0, 0.9, 0.8, 0.7, 0.6)) -> pd.DataFrame:
    """Selective prediction: keep the most-confident `cov` fraction (confidence =
    distance of predicted RR from the WHO threshold), defer the rest to a human.
    Report sensitivity/specificity on the auto-decided cases at each coverage."""
    margin = np.abs(sub["pred_bpm"].to_numpy(float) - sub["threshold_bpm"].to_numpy(float))
    order = np.argsort(-margin)  # most confident (largest margin) first
    y = sub["gt_tachy"].to_numpy(int)[order]
    yhat = sub["pred_tachy"].to_numpy(int)[order]
    n = len(sub)
    rows = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        yy, yh = y[:k], yhat[:k]
        tp = int(((yy == 1) & (yh == 1)).sum()); fn = int(((yy == 1) & (yh == 0)).sum())
        tn = int(((yy == 0) & (yh == 0)).sum()); fp = int(((yy == 0) & (yh == 1)).sum())
        rows.append({"coverage": cov, "n_auto": k,
                     "sensitivity": _safe(tp, tp + fn), "specificity": _safe(tn, tn + fp),
                     "referral_rate": 1 - cov})
    return pd.DataFrame(rows)


def net_benefit_seed(sub: pd.DataFrame, pt_grid: np.ndarray) -> pd.DataFrame:
    """Decision-curve net benefit using tachy_prob as the risk score.
    NB(pt) = TP/N - FP/N * pt/(1-pt); compared to treat-all and treat-none."""
    y = sub["gt_tachy"].to_numpy(int)
    n = len(y)
    prev = y.mean()
    score = sub["tachy_prob"].to_numpy(float) if "tachy_prob" in sub.columns else sub["pred_tachy"].to_numpy(float)
    rows = []
    for pt in pt_grid:
        flag = (score >= pt).astype(int)
        tp = int(((y == 1) & (flag == 1)).sum()); fp = int(((y == 0) & (flag == 1)).sum())
        nb = tp / n - fp / n * (pt / (1 - pt)) if pt < 1 else float("nan")
        nb_all = prev - (1 - prev) * (pt / (1 - pt)) if pt < 1 else float("nan")
        rows.append({"pt": float(pt), "nb_model": nb, "nb_treat_all": nb_all, "nb_treat_none": 0.0})
    return pd.DataFrame(rows)


def _agg(per_seed: pd.DataFrame, keys: List[str], metrics: List[str]) -> pd.DataFrame:
    g = per_seed.groupby(keys)
    out = g[metrics].agg(["mean", "std"]).reset_index()
    out.columns = keys + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    return out


# --------------------------------------------------------------------------
# Subject-level bootstrap 95% CIs. This is the PRIMARY uncertainty measure:
# seed-std only captures training stochasticity on a fixed test set; it does
# NOT reflect sampling uncertainty over infants. With small n this dominates.
# --------------------------------------------------------------------------
def _seed_average(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed rows to one row per clip (seed-averaged predictions)."""
    key = "signal_path" if "signal_path" in df.columns else "name"
    grp = df.groupby(key)
    first_cols = [c for c in ["subject_id", "gt_bpm", "gt_tachy", "threshold_bpm",
                              "age_months", "age_band"] if c in df.columns]
    d = grp[first_cols].first().reset_index()
    for c in ["pred_bpm", "tachy_prob"]:
        if c in df.columns:
            d[c] = grp[c].mean().values
    d["pred_tachy"] = (d["pred_bpm"] >= d["threshold_bpm"]).astype(int)
    if "subject_id" not in d.columns:
        d["subject_id"] = d[key]
    return d


def _clip_metrics(sub: pd.DataFrame) -> Dict[str, float]:
    """MAE + tachypnea sens/spec/auroc on one-row-per-clip data."""
    gt = sub["gt_bpm"].to_numpy(float); pr = sub["pred_bpm"].to_numpy(float)
    y = sub["gt_tachy"].to_numpy(int); yhat = sub["pred_tachy"].to_numpy(int)
    tp = int(((y == 1) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
    auroc = float("nan")
    if _HAVE_SK and 0 < y.sum() < len(y) and "tachy_prob" in sub.columns:
        try:
            auroc = float(roc_auc_score(y, sub["tachy_prob"].to_numpy(float)))
        except Exception:
            auroc = float("nan")
    return {"mae": float(np.abs(pr - gt).mean()), "sensitivity": _safe(tp, tp + fn),
            "specificity": _safe(tn, tn + fp), "auroc": auroc}


def bootstrap_ci(dclip: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Subject-level bootstrap 95% CIs for OVERALL + each WHO band. Resamples
    SUBJECTS with replacement (respecting multi-clip subjects)."""
    rng = np.random.default_rng(seed)
    metric_names = ["mae", "sensitivity", "specificity", "auroc"]
    rows = []
    for band in [b[0] for b in AGE_BANDS] + ["OVERALL"]:
        sub = dclip if band == "OVERALL" else dclip[dclip["age_band"] == band]
        if len(sub) == 0:
            continue
        subs = sub["subject_id"].to_numpy()
        uniq = np.unique(subs)
        sub_to_idx = {u: np.where(subs == u)[0] for u in uniq}
        point = _clip_metrics(sub)
        draws = {m: [] for m in metric_names}
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([sub_to_idx[u] for u in pick])
            mm = _clip_metrics(sub.iloc[idx])
            for m in metric_names:
                draws[m].append(mm[m])
        row = {"band": band, "n": int(len(sub)), "n_pos": int(sub["gt_tachy"].sum())}
        for m in metric_names:
            arr = np.asarray(draws[m], float); arr = arr[np.isfinite(arr)]
            row[m] = point[m]
            row[f"{m}_lo"] = float(np.nanpercentile(arr, 2.5)) if arr.size else float("nan")
            row[f"{m}_hi"] = float(np.nanpercentile(arr, 97.5)) if arr.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Clinical-endpoint (tachypnea) evaluation by WHO age band.")
    ap.add_argument("--predictions", nargs="*", default=[])
    ap.add_argument("--run-root", default="")
    ap.add_argument("--setting", default="calibrated", choices=sorted(SETTING_SUBDIRS))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pt-min", type=float, default=0.01)
    ap.add_argument("--pt-max", type=float, default=0.50)
    ap.add_argument("--pt-step", type=float, default=0.01)
    ap.add_argument("--n-boot", type=int, default=2000, help="Subject-level bootstrap resamples for 95pct CIs (0 to skip)")
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    paths = resolve_prediction_paths(args.predictions, args.run_root, args.setting)
    df = load_predictions(paths)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted(df["seed"].unique())
    print(f"[load] {len(df)} rows, seeds={seeds}, N/seed={len(df)//max(1,len(seeds))}")

    # Per-seed metrics for RR + classification, per age band and OVERALL.
    rr_rows, clf_rows, op_rows = [], [], []
    for seed, dseed in df.groupby("seed"):
        for band in [b[0] for b in AGE_BANDS] + ["OVERALL"]:
            sub = dseed if band == "OVERALL" else dseed[dseed["age_band"] == band]
            if len(sub) == 0:
                continue
            rr_rows.append({"seed": seed, "band": band, **rr_metrics(sub)})
            clf_rows.append({"seed": seed, "band": band, **clf_metrics(sub)})
            op_rows.append({"seed": seed, "band": band, **operating_points(sub)})
    rr_ps, clf_ps, op_ps = pd.DataFrame(rr_rows), pd.DataFrame(clf_rows), pd.DataFrame(op_rows)

    rr_agg = _agg(rr_ps, ["band"], ["n", "mae", "rmse", "pearson_r", "ba_bias", "ba_loa_lo", "ba_loa_hi"])
    clf_agg = _agg(clf_ps, ["band"], ["n_pos", "sensitivity", "specificity", "ppv", "npv", "f1", "balanced_acc", "auroc", "auprc"])
    op_agg = _agg(op_ps, ["band"], ["spec_at_sens90", "sens_at_spec80", "youden_j"])

    # Selective prediction + net benefit are computed pooled per seed (OVERALL), then averaged.
    rc = pd.concat([risk_coverage(d).assign(seed=s) for s, d in df.groupby("seed")], ignore_index=True)
    rc_agg = rc.groupby("coverage")[["sensitivity", "specificity", "referral_rate"]].agg(["mean", "std"]).reset_index()
    rc_agg.columns = ["coverage"] + [f"{m}_{st}" for m in ["sensitivity", "specificity", "referral_rate"] for st in ("mean", "std")]
    pt_grid = np.arange(args.pt_min, args.pt_max + 1e-9, args.pt_step)
    nb = pd.concat([net_benefit_seed(d, pt_grid).assign(seed=s) for s, d in df.groupby("seed")], ignore_index=True)
    nb_agg = nb.groupby("pt")[["nb_model", "nb_treat_all", "nb_treat_none"]].mean().reset_index()

    rr_agg.to_csv(out_dir / "rr_by_band.csv", index=False)
    clf_agg.to_csv(out_dir / "tachypnea_by_band.csv", index=False)
    op_agg.to_csv(out_dir / "operating_points.csv", index=False)
    rc_agg.to_csv(out_dir / "risk_coverage.csv", index=False)
    nb_agg.to_csv(out_dir / "decision_curve.csv", index=False)

    # ---- subject-level bootstrap 95% CIs (PRIMARY uncertainty) ----
    boot_agg = None
    if args.n_boot > 0:
        dclip = _seed_average(df)
        boot_agg = bootstrap_ci(dclip, n_boot=args.n_boot, seed=args.boot_seed)
        boot_agg.to_csv(out_dir / "bootstrap_ci.csv", index=False)

    # coverage needed for mean sensitivity >= 0.90
    cov90 = rc_agg[rc_agg["sensitivity_mean"] >= 0.90]["coverage"].max()
    summary = {
        "n_total": int(len(df)), "seeds": [int(s) for s in seeds],
        "overall_rr": rr_agg[rr_agg["band"] == "OVERALL"].to_dict("records")[0],
        "overall_tachypnea": clf_agg[clf_agg["band"] == "OVERALL"].to_dict("records")[0],
        "coverage_for_sens90": (float(cov90) if pd.notna(cov90) else None),
        "overall_bootstrap_ci": (boot_agg[boot_agg["band"] == "OVERALL"].to_dict("records")[0]
                                 if boot_agg is not None and (boot_agg["band"] == "OVERALL").any() else None),
    }
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)

    # ---- printed Table 1 ----
    pd.set_option("display.width", 200, "display.max_columns", 30)
    def fmt(agg, band, cols):
        row = agg[agg["band"] == band].iloc[0]
        return "  ".join(f"{c}={row[f'{c}_mean']:.2f}±{row[f'{c}_std']:.2f}" for c in cols)
    print("\n===== CLINICAL ENDPOINT — Table 1 (mean±std over seeds) =====")
    print("\n[RR agreement by WHO age band]")
    for b in [x[0] for x in AGE_BANDS] + ["OVERALL"]:
        if (rr_agg["band"] == b).any():
            print(f"  {b:8}: " + fmt(rr_agg, b, ["n", "mae", "rmse", "pearson_r", "ba_bias"]))
    print("\n[Tachypnea classification by WHO age band]")
    for b in [x[0] for x in AGE_BANDS] + ["OVERALL"]:
        if (clf_agg["band"] == b).any():
            print(f"  {b:8}: " + fmt(clf_agg, b, ["n_pos", "sensitivity", "specificity", "ppv", "auroc"]))
    print("\n[Operating points (OVERALL)]")
    print("  " + fmt(op_agg, "OVERALL", ["spec_at_sens90", "sens_at_spec80", "youden_j"]))
    print("\n[Selective prediction: sensitivity vs coverage (defer least-confident)]")
    for _, r in rc_agg.iterrows():
        print(f"  coverage={r['coverage']:.2f} (refer {r['referral_rate_mean']*100:.0f}%): "
              f"sens={r['sensitivity_mean']:.2f}±{r['sensitivity_std']:.2f}  spec={r['specificity_mean']:.2f}")
    print(f"\n  -> coverage needed for mean sens>=0.90: {summary['coverage_for_sens90']}")
    if boot_agg is not None:
        print(f"\n===== SUBJECT-LEVEL BOOTSTRAP 95pct CI (PRIMARY uncertainty; n_boot={args.n_boot}; seed-std above = training stability only) =====")
        for _, r in boot_agg.iterrows():
            print(f"  {r['band']:8} (n={int(r['n'])}, n_pos={int(r['n_pos'])}): "
                  f"MAE={r['mae']:.2f}[{r['mae_lo']:.2f},{r['mae_hi']:.2f}]  "
                  f"AUROC={r['auroc']:.3f}[{r['auroc_lo']:.2f},{r['auroc_hi']:.2f}]  "
                  f"Sens={r['sensitivity']:.2f}[{r['sensitivity_lo']:.2f},{r['sensitivity_hi']:.2f}]  "
                  f"Spec={r['specificity']:.2f}[{r['specificity_lo']:.2f},{r['specificity_hi']:.2f}]")
    print(f"\n[written] {out_dir}/  (rr_by_band, tachypnea_by_band, operating_points, risk_coverage, decision_curve, bootstrap_ci, summary.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
