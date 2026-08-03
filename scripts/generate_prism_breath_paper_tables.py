"""Aggregate PRISM-Breath paper outputs into publication-ready CSV tables.

This script targets the journal-facing PRISM-Breath run layout:

- main_seed*/val_eval/prism_breath_summary.json
- main_seed*/test_eval_uncalibrated/prism_breath_summary.json
- main_seed*/test_eval/prism_breath_summary.json
- study_seed*/ablation_summary.csv
- expanded_seed*/benchmark/expanded_paper_leaderboard.csv
- official_air_seed*/benchmark/official_air_leaderboard.csv
- legacy_seed*/benchmark/legacy_canonical_leaderboard.csv

It writes consistent CSV tables for:
- flagship operating points
- ablation mean/std summaries
- expanded paper baseline mean/std summaries
- official AIR baseline mean/std summaries
- legacy baseline mean/std summaries
- best-seed subgroup analyses by dataset and age bin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _mean_std_aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.columns
        if col not in group_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_cols:
        return pd.DataFrame()
    grouped = df.groupby(group_cols, as_index=False)[numeric_cols].agg(["mean", "std"])
    grouped.columns = [
        "_".join([str(x) for x in col if x]).strip("_") if isinstance(col, tuple) else str(col)
        for col in grouped.columns
    ]
    return grouped


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Wrote {path}")


def load_main_summaries(run_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    settings = [
        ("val", "val_eval"),
        ("test_uncalibrated", "test_eval_uncalibrated"),
        ("test_calibrated", "test_eval"),
    ]
    main_seed_dirs = list(sorted(run_root.glob("main_seed*")))
    if not main_seed_dirs:
        flat_rows: List[Dict] = []
        for setting_name, subdir in settings:
            summary_path = run_root / subdir / "prism_breath_summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            flat_rows.append({"seed": -1, "setting": setting_name, **payload})
        if flat_rows:
            return pd.DataFrame(flat_rows)

    for seed_dir in main_seed_dirs:
        try:
            seed = int(seed_dir.name.replace("main_seed", ""))
        except ValueError:
            continue
        for setting_name, subdir in settings:
            summary_path = seed_dir / subdir / "prism_breath_summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            rows.append({"seed": seed, "setting": setting_name, **payload})
    return pd.DataFrame(rows)


def load_ablation_summaries(run_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(run_root.glob("study_seed*/ablation_summary.csv")):
        try:
            seed = int(csv_path.parent.name.replace("study_seed", ""))
        except ValueError:
            seed = -1
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_legacy_leaderboards(run_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(run_root.glob("legacy_seed*/benchmark/legacy_canonical_leaderboard.csv")):
        try:
            seed = int(csv_path.parts[-3].replace("legacy_seed", ""))
        except ValueError:
            seed = -1
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_expanded_paper_leaderboards(run_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(run_root.glob("expanded_seed*/benchmark/expanded_paper_leaderboard.csv")):
        try:
            seed = int(csv_path.parts[-3].replace("expanded_seed", ""))
        except ValueError:
            seed = -1
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_official_air_leaderboards(run_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in sorted(run_root.glob("official_air_seed*/benchmark/official_air_leaderboard.csv")):
        try:
            seed = int(csv_path.parts[-3].replace("official_air_seed", ""))
        except ValueError:
            seed = -1
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_prediction_frame(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    groups = []
    for key, sub in df.groupby(group_col, dropna=False, observed=True):
        if sub.empty:
            continue
        gt = sub["gt_rr_bpm"].astype(float).to_numpy()
        pred = sub["pred_rr_bpm"].astype(float).to_numpy()
        triage = sub["triage_code"].astype(int).to_numpy()
        gt_tachy = sub["gt_tachy"].astype(int).to_numpy()
        lower = sub["interval_lower_bpm"].astype(float).to_numpy()
        upper = sub["interval_upper_bpm"].astype(float).to_numpy()

        positives = gt_tachy == 1
        negatives = gt_tachy == 0
        groups.append(
            {
                group_col: key,
                "n": int(len(sub)),
                "mae": float(np.mean(np.abs(pred - gt))),
                "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
                "review_rate": float(np.mean(triage == 2)),
                "coverage": float(np.mean((gt >= lower) & (gt <= upper))),
                "sensitivity": float(np.mean(triage[positives] == 1)) if positives.any() else float("nan"),
                "specificity": float(np.mean(triage[negatives] == 0)) if negatives.any() else float("nan"),
                "pred_mean": float(np.mean(pred)),
                "gt_mean": float(np.mean(gt)),
            }
        )
    return pd.DataFrame(groups)


def build_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [-1, 2, 12, 60, 300]
    labels = ["<2m", "2-12m", "12-60m", ">60m"]
    out["age_bin"] = pd.cut(out["age_months"], bins=bins, labels=labels)
    return out


def load_prediction_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def export_best_seed_subgroups(run_root: Path, output_dir: Path, best_seed: int) -> None:
    seed_dir = run_root / f"main_seed{best_seed}"
    if not seed_dir.exists():
        seed_dir = run_root
    settings = {
        "uncalibrated": seed_dir / "test_eval_uncalibrated" / "prism_breath_predictions.csv",
        "calibrated": seed_dir / "test_eval" / "prism_breath_predictions.csv",
    }
    for setting_name, csv_path in settings.items():
        df = load_prediction_csv(csv_path)
        if df is None:
            continue
        dataset_table = summarize_prediction_frame(df, group_col="dataset")
        _safe_write_csv(dataset_table, output_dir / f"best_seed_{best_seed}_{setting_name}_by_dataset.csv")

        by_age = summarize_prediction_frame(build_age_bins(df), group_col="age_bin")
        _safe_write_csv(by_age, output_dir / f"best_seed_{best_seed}_{setting_name}_by_age.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate PRISM-Breath paper outputs into publication-ready CSV tables.")
    parser.add_argument("--run-root", required=True, help="Root paper run directory, e.g. outputs/paper_runs/20260318_172659")
    parser.add_argument("--output-dir", default="", help="Output directory (default: <run-root>/paper_tables)")
    parser.add_argument("--best-seed", type=int, default=42, help="Seed used for subgroup analysis exports")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.exists():
        raise SystemExit(f"Run root not found: {run_root}")
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "paper_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    main_df = load_main_summaries(run_root)
    if not main_df.empty:
        _safe_write_csv(main_df, output_dir / "prism_main_seedwise.csv")
        main_agg = _mean_std_aggregate(main_df, group_cols=["setting"])
        _safe_write_csv(main_agg, output_dir / "prism_main_mean_std.csv")

    ablation_df = load_ablation_summaries(run_root)
    if not ablation_df.empty:
        _safe_write_csv(ablation_df, output_dir / "prism_ablation_seedwise.csv")
        ablation_agg = _mean_std_aggregate(ablation_df, group_cols=["variant"])
        _safe_write_csv(ablation_agg, output_dir / "prism_ablation_mean_std.csv")

    legacy_df = load_legacy_leaderboards(run_root)
    if not legacy_df.empty:
        _safe_write_csv(legacy_df, output_dir / "legacy_baselines_seedwise.csv")
        legacy_agg = _mean_std_aggregate(legacy_df, group_cols=["name", "paradigm"])
        _safe_write_csv(legacy_agg, output_dir / "legacy_baselines_mean_std.csv")

    expanded_df = load_expanded_paper_leaderboards(run_root)
    if not expanded_df.empty:
        _safe_write_csv(expanded_df, output_dir / "expanded_paper_baselines_seedwise.csv")
        expanded_agg = _mean_std_aggregate(expanded_df, group_cols=["name", "paradigm"])
        _safe_write_csv(expanded_agg, output_dir / "expanded_paper_baselines_mean_std.csv")

    official_air_df = load_official_air_leaderboards(run_root)
    if not official_air_df.empty:
        _safe_write_csv(official_air_df, output_dir / "official_air_baselines_seedwise.csv")
        official_air_agg = _mean_std_aggregate(official_air_df, group_cols=["name", "paradigm"])
        _safe_write_csv(official_air_agg, output_dir / "official_air_baselines_mean_std.csv")

    if not expanded_df.empty:
        combined_frames = [expanded_df]
    else:
        combined_frames = [df for df in [legacy_df, official_air_df] if not df.empty]
    if combined_frames:
        combined_df = pd.concat(combined_frames, ignore_index=True)
        _safe_write_csv(combined_df, output_dir / "baseline_comparison_seedwise.csv")
        combined_agg = _mean_std_aggregate(combined_df, group_cols=["name", "paradigm"])
        _safe_write_csv(combined_agg, output_dir / "baseline_comparison_mean_std.csv")

    export_best_seed_subgroups(run_root, output_dir, best_seed=int(args.best_seed))
    print(f"[DONE] Paper tables written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
