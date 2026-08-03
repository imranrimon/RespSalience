"""
Clinical-rigor analysis for PRISM-Breath: decision-curve (net-benefit) analysis
and multi-seed subgroup breakdowns.

This operates on the prediction CSVs exported by ``scripts/evaluate_prism_breath.py``
(``prism_breath_predictions.csv``), which contain per-video ``gt_rr_bpm``,
``pred_rr_bpm``, ``threshold_bpm``, ``gt_tachy``, ``triage_code``, ``tachy_prob``,
``age_months``, ``dataset`` and ``disagreement_score``.

It produces two things the flagship paper needs beyond raw MAE:

1) Decision Curve Analysis (Vickers & Elkin net benefit) for the age-conditioned
   tachypnea decision, comparing the model's risk score against the "treat all"
   and "treat none" reference strategies, plus an idealized PRISM triage policy
   that defers UNCERTAIN_REVIEW cases to a clinician. This answers "is acting on
   the model clinically worthwhile, and over what threshold range?" — not just
   "is the AUC high?".

2) Multi-seed subgroup tables (mean +/- std across seeds) of MAE/RMSE, triage
   sensitivity/specificity, review rate and interval coverage, broken down by
   dataset, WHO age bin, and modality-disagreement bin. This is where dataset
   shift and age dependence become visible.

Usage:
    # Aggregate all seed runs under a study root (calibrated test predictions)
    python scripts/clinical_analysis.py --run-root outputs/paper_runs/<RUN> \
        --setting calibrated --output-dir outputs/paper_runs/<RUN>/clinical

    # Or pass explicit prediction CSVs (one per seed)
    python scripts/clinical_analysis.py \
        --predictions s42/test_eval/prism_breath_predictions.csv \
                      s52/test_eval/prism_breath_predictions.csv \
        --output-dir outputs/clinical --plot
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# WHO-aligned age bins (months) and disagreement bins, matching the thresholds
# encoded in prism_breath_model.tensor_tachypnea_threshold.
AGE_BINS = [-1, 2, 12, 60, 300]
AGE_LABELS = ["<2m", "2-12m", "12-60m", ">60m"]
DISAGREEMENT_BINS = [-0.01, 0.2, 0.4, 0.7, 1.01]
DISAGREEMENT_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.7", "0.7-1.0"]

SETTING_SUBDIRS = {
    "calibrated": "test_eval",
    "uncalibrated": "test_eval_uncalibrated",
    "external": "external_eval",
    "val": "val_eval",
}


def _seed_from_path(path: Path) -> int:
    match = re.search(r"seed[_-]?(\d+)", str(path))
    return int(match.group(1)) if match else -1


def resolve_prediction_paths(predictions: List[str], run_root: str, setting: str) -> List[Path]:
    paths: List[Path] = []
    for raw in predictions or []:
        p = Path(raw)
        if p.exists():
            paths.append(p)
        else:
            print(f"[WARN] missing prediction CSV: {p}")
    if run_root:
        root = Path(run_root)
        subdir = SETTING_SUBDIRS.get(setting, "test_eval")
        # Match both <root>/*seed*/<subdir>/... and <root>/<subdir>/...
        discovered = sorted(root.glob(f"**/{subdir}/prism_breath_predictions.csv"))
        for p in discovered:
            if p not in paths:
                paths.append(p)
    return paths


def load_predictions(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if df.empty:
            continue
        seed = _seed_from_path(p)
        df["seed"] = seed
        # Robust fallbacks for older CSVs.
        if "gt_tachy" not in df.columns and {"gt_rr_bpm", "threshold_bpm"} <= set(df.columns):
            df["gt_tachy"] = (df["gt_rr_bpm"].astype(float) >= df["threshold_bpm"].astype(float)).astype(int)
        frames.append(df)
        print(f"[load] {p} (seed={seed}, n={len(df)})")
    if not frames:
        raise FileNotFoundError("No non-empty prediction CSVs resolved.")
    return pd.concat(frames, ignore_index=True)


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "age_months" in out.columns:
        out["age_bin"] = pd.cut(out["age_months"].astype(float), bins=AGE_BINS, labels=AGE_LABELS)
    if "disagreement_score" in out.columns:
        out["disagreement_bin"] = pd.cut(
            out["disagreement_score"].astype(float), bins=DISAGREEMENT_BINS, labels=DISAGREEMENT_LABELS
        )
    return out


def _metrics(sub: pd.DataFrame) -> Dict[str, float]:
    gt = sub["gt_rr_bpm"].astype(float).to_numpy()
    pred = sub["pred_rr_bpm"].astype(float).to_numpy()
    triage = sub["triage_code"].astype(int).to_numpy()
    gt_tachy = sub["gt_tachy"].astype(int).to_numpy()
    lower = sub["interval_lower_bpm"].astype(float).to_numpy() if "interval_lower_bpm" in sub else None
    upper = sub["interval_upper_bpm"].astype(float).to_numpy() if "interval_upper_bpm" in sub else None
    positives = gt_tachy == 1
    negatives = gt_tachy == 0
    out = {
        "n": int(len(sub)),
        "mae": float(np.mean(np.abs(pred - gt))),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "sensitivity": float(np.mean(triage[positives] == 1)) if positives.any() else float("nan"),
        "specificity": float(np.mean(triage[negatives] == 0)) if negatives.any() else float("nan"),
        "review_rate": float(np.mean(triage == 2)),
    }
    if lower is not None and upper is not None:
        out["coverage"] = float(np.mean((gt >= lower) & (gt <= upper)))
    return out


def subgroup_across_seeds(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-seed subgroup metrics aggregated to mean/std across seeds."""
    rows = []
    for (seed, key), sub in df.groupby(["seed", group_col], dropna=False, observed=True):
        if sub.empty:
            continue
        rows.append({"seed": seed, group_col: key, **_metrics(sub)})
    if not rows:
        return pd.DataFrame()
    per_seed = pd.DataFrame(rows)
    metric_cols = [c for c in ["n", "mae", "rmse", "sensitivity", "specificity", "review_rate", "coverage"] if c in per_seed]
    agg = per_seed.groupby(group_col, observed=True)[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    agg = agg.reset_index()
    agg["n_seeds"] = per_seed.groupby(group_col, observed=True).size().reindex(agg[group_col]).to_numpy()
    return agg


def decision_curve_seed(sub: pd.DataFrame, pt_grid: np.ndarray) -> pd.DataFrame:
    """Net benefit (Vickers & Elkin) for one seed over a threshold-probability grid."""
    n = len(sub)
    gt = sub["gt_tachy"].astype(int).to_numpy()
    prev = float(np.mean(gt))
    has_prob = "tachy_prob" in sub.columns
    prob = sub["tachy_prob"].astype(float).to_numpy() if has_prob else None
    triage = sub["triage_code"].astype(int).to_numpy() if "triage_code" in sub.columns else None

    rows = []
    for pt in pt_grid:
        w = pt / (1.0 - pt)
        row = {"pt": float(pt), "nb_treat_all": prev - (1.0 - prev) * w, "nb_treat_none": 0.0}
        if has_prob:
            treat = prob >= pt
            tp = int(np.sum(treat & (gt == 1)))
            fp = int(np.sum(treat & (gt == 0)))
            row["nb_model"] = tp / n - (fp / n) * w
        if triage is not None:
            # Idealized PRISM triage: act on TACHYPNEA_ALERT, defer UNCERTAIN_REVIEW
            # to a clinician assumed to resolve it correctly (deferred positives
            # caught, deferred negatives not falsely treated).
            tp_t = int(np.sum(((triage == 1) | (triage == 2)) & (gt == 1)))
            fp_t = int(np.sum((triage == 1) & (gt == 0)))
            row["nb_triage_defer"] = tp_t / n - (fp_t / n) * w
        rows.append(row)
    return pd.DataFrame(rows)


def decision_curve_across_seeds(df: pd.DataFrame, pt_grid: np.ndarray) -> pd.DataFrame:
    per_seed = []
    for seed, sub in df.groupby("seed"):
        curve = decision_curve_seed(sub, pt_grid)
        curve["seed"] = seed
        per_seed.append(curve)
    allc = pd.concat(per_seed, ignore_index=True)
    value_cols = [c for c in allc.columns if c not in ("pt", "seed")]
    agg = allc.groupby("pt")[value_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    return agg.reset_index()


def _plot_decision_curve(curve: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot] matplotlib unavailable, skipping figure: {exc}")
        return

    # Colorblind-safe (Okabe-Ito) lines; clear axes for a paper figure.
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    series = [
        ("nb_model_mean", "nb_model_std", "PRISM risk score", "#0072B2"),
        ("nb_triage_defer_mean", "nb_triage_defer_std", "PRISM triage (idealized review)", "#009E73"),
        ("nb_treat_all_mean", None, "Treat all", "#D55E00"),
    ]
    for mean_col, std_col, label, color in series:
        if mean_col not in curve.columns:
            continue
        ax.plot(curve["pt"], curve[mean_col], label=label, color=color, linewidth=2)
        if std_col and std_col in curve.columns and curve[std_col].notna().any():
            ax.fill_between(
                curve["pt"], curve[mean_col] - curve[std_col], curve[mean_col] + curve[std_col],
                color=color, alpha=0.15, linewidth=0,
            )
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1, label="Treat none")
    ymin = float(np.nanmin([curve[c].min() for c in curve.columns if c.endswith("_mean")]))
    ax.set_ylim(max(-0.05, ymin - 0.02), None)
    ax.set_xlabel("Threshold probability $p_t$")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve: age-conditioned tachypnea triage")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Decision-curve + subgroup clinical analysis for PRISM-Breath.")
    parser.add_argument("--predictions", nargs="*", default=[], help="Explicit prediction CSVs (one per seed).")
    parser.add_argument("--run-root", default="", help="Study root to auto-discover seed prediction CSVs.")
    parser.add_argument("--setting", default="calibrated", choices=sorted(SETTING_SUBDIRS),
                        help="Which eval subdir to discover under --run-root.")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis CSVs (and optional figure).")
    parser.add_argument("--pt-min", type=float, default=0.01)
    parser.add_argument("--pt-max", type=float, default=0.60)
    parser.add_argument("--pt-step", type=float, default=0.01)
    parser.add_argument("--plot", action="store_true", help="Also render a decision-curve PNG.")
    args = parser.parse_args()

    paths = resolve_prediction_paths(args.predictions, args.run_root, args.setting)
    if not paths:
        raise SystemExit("No prediction CSVs found. Provide --predictions or --run-root.")

    df = add_bins(load_predictions(paths))
    n_seeds = df["seed"].nunique()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[analysis] {len(paths)} file(s), {n_seeds} seed(s), {len(df)} rows -> {out_dir}")

    # Overall (treat 'overall' as a single group)
    df_overall = df.assign(overall="overall")
    subgroup_across_seeds(df_overall, "overall").to_csv(out_dir / "clinical_overall.csv", index=False)

    # Subgroups
    for group_col, fname in (("dataset", "clinical_by_dataset.csv"),
                             ("age_bin", "clinical_by_age.csv"),
                             ("disagreement_bin", "clinical_by_disagreement.csv")):
        if group_col in df.columns:
            table = subgroup_across_seeds(df, group_col)
            if not table.empty:
                table.to_csv(out_dir / fname, index=False)
                print(f"[subgroup] {group_col}: {len(table)} groups -> {fname}")

    # Decision curve
    pt_grid = np.arange(args.pt_min, args.pt_max + 1e-9, args.pt_step)
    curve = decision_curve_across_seeds(df, pt_grid)
    curve.to_csv(out_dir / "decision_curve.csv", index=False)
    print(f"[decision-curve] {len(curve)} thresholds -> decision_curve.csv")

    # Net-benefit headline: max advantage of the model over 'treat all' across pt.
    if "nb_model_mean" in curve.columns:
        adv = curve["nb_model_mean"] - curve["nb_treat_all_mean"]
        best_i = int(np.nanargmax(adv.to_numpy()))
        print(f"[decision-curve] peak model-vs-treat-all net-benefit advantage "
              f"{adv.iloc[best_i]:+.4f} at p_t={curve['pt'].iloc[best_i]:.2f}")

    if args.plot:
        _plot_decision_curve(curve, out_dir / "decision_curve.png")

    print("[OK] clinical analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
