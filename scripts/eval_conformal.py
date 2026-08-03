"""
Evaluate split conformal intervals on held-out test units.

Inputs:
- per-sample prediction CSV (from run_fair_benchmark --sample-output)
- calibration CSV (from calibrate_conformal.py)
"""

import argparse
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_disagreement_bins(text: str) -> Tuple[float, float]:
    vals = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        vals.append(float(raw))
    if len(vals) < 2:
        raise ValueError("Need at least two comma-separated disagreement bin thresholds, e.g. 0.2,0.4")
    vals = sorted(vals)
    return float(vals[0]), float(vals[1])


def _disagreement_bin(score: float, bins: Tuple[float, float]) -> str:
    lo, hi = bins
    s = float(score)
    if s <= lo:
        return "low"
    if s <= hi:
        return "mid"
    return "high"


def _stable_uniform(key: str, seed: int) -> float:
    payload = f"{seed}|{key}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    value = int(digest, 16)
    return value / float(16 ** 16 - 1)


def _ensure_split_unit(df: pd.DataFrame, split_unit: str) -> pd.Series:
    if split_unit in df.columns:
        return df[split_unit].astype(str).fillna("NA")
    return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)


def load_samples(samples_text: str, sample_glob: str) -> pd.DataFrame:
    paths: List[Path] = []
    for raw in _parse_csv_list(samples_text):
        p = Path(raw)
        if p.exists():
            paths.append(p)
        else:
            print(f"[WARN] Missing sample CSV: {p}")
    if sample_glob.strip():
        paths.extend(sorted([p for p in Path(".").glob(sample_glob) if p.is_file()]))

    if not paths:
        raise SystemExit("No sample CSV files found.")

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if df.empty:
                continue
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Failed to read {p}: {exc}")

    if not frames:
        raise SystemExit("Sample CSVs were found but no readable rows were loaded.")

    out = pd.concat(frames, ignore_index=True)
    required = {"name", "pred_bpm", "gt_bpm"}
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise SystemExit(f"Missing required sample columns: {missing}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate split conformal coverage and interval quality.")
    parser.add_argument("--samples", type=str, default="", help="Comma-separated sample CSV files")
    parser.add_argument("--sample-glob", type=str, default="", help="Optional glob for sample CSV files")
    parser.add_argument("--calibration", type=str, required=True, help="Calibration CSV from calibrate_conformal.py")
    parser.add_argument("--output", type=str, required=True, help="Output conformal metrics CSV")
    parser.add_argument("--detailed-output", type=str, default="", help="Optional per-sample interval CSV")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level")
    parser.add_argument("--calibration-ratio", type=float, default=0.3, help="Calibration split ratio")
    parser.add_argument("--split-unit", type=str, default="subject", help="Split unit column")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument(
        "--group-by-source",
        action="store_true",
        help="Evaluate separately per source dataset in addition to model/paradigm",
    )
    parser.add_argument(
        "--condition-on-disagreement",
        action="store_true",
        help="Evaluate conformal metrics per disagreement bin (low/mid/high).",
    )
    parser.add_argument(
        "--disagreement-col",
        type=str,
        default="disagreement_score",
        help="Sample CSV column used for disagreement conditioning.",
    )
    parser.add_argument(
        "--disagreement-bins",
        type=str,
        default="0.2,0.4",
        help="Two comma-separated thresholds for low/mid/high bins.",
    )
    args = parser.parse_args()

    samples = load_samples(args.samples, args.sample_glob).copy()
    samples["pred_bpm"] = pd.to_numeric(samples["pred_bpm"], errors="coerce")
    samples["gt_bpm"] = pd.to_numeric(samples["gt_bpm"], errors="coerce")
    samples = samples.dropna(subset=["name", "pred_bpm", "gt_bpm"]).reset_index(drop=True)
    samples["abs_error"] = np.abs(samples["gt_bpm"] - samples["pred_bpm"])
    samples["sq_error"] = (samples["gt_bpm"] - samples["pred_bpm"]) ** 2

    if "paradigm" not in samples.columns:
        samples["paradigm"] = "Unknown"
    if "source" not in samples.columns:
        samples["source"] = "Unknown"

    disagreement_bins = _parse_disagreement_bins(args.disagreement_bins)
    if args.condition_on_disagreement:
        if args.disagreement_col not in samples.columns:
            print(
                f"[WARN] --condition-on-disagreement requested, but '{args.disagreement_col}' not found. "
                "Falling back to zeros."
            )
            samples[args.disagreement_col] = 0.0
        samples[args.disagreement_col] = pd.to_numeric(samples[args.disagreement_col], errors="coerce").fillna(0.0)
        samples["disagreement_bin"] = samples[args.disagreement_col].map(
            lambda x: _disagreement_bin(float(x), disagreement_bins)
        )

    split_unit_vals = _ensure_split_unit(samples, args.split_unit)
    samples["__split_unit"] = split_unit_vals
    samples["is_calibration"] = samples["__split_unit"].map(
        lambda x: _stable_uniform(str(x), args.seed) < args.calibration_ratio
    )
    test_samples = samples[~samples["is_calibration"]].copy()

    calib_path = Path(args.calibration)
    if not calib_path.exists():
        raise SystemExit(f"Calibration CSV not found: {calib_path}")
    calib_df = pd.read_csv(calib_path)
    if calib_df.empty:
        raise SystemExit("Calibration CSV is empty.")

    group_cols = ["name", "paradigm"]
    if args.group_by_source:
        group_cols.append("source")
        if "source" not in calib_df.columns:
            raise SystemExit("Calibration CSV missing 'source' column but --group-by-source was requested.")
    if args.condition_on_disagreement:
        if "disagreement_bin" not in calib_df.columns:
            raise SystemExit(
                "Calibration CSV missing 'disagreement_bin' but --condition-on-disagreement was requested."
            )
        group_cols.append("disagreement_bin")

    qhat_map = {}
    for _, row in calib_df.iterrows():
        key = tuple([row[c] for c in group_cols])
        qhat_map[key] = {
            "qhat": float(row.get("qhat", np.nan)),
            "n_cal": int(row.get("n_cal", 0)),
            "alpha": float(row.get("alpha", args.alpha)),
            "target_coverage": float(row.get("target_coverage", 1.0 - args.alpha)),
        }

    metric_rows = []
    detailed_frames = []
    for keys, group in test_samples.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key = tuple(keys)
        if key not in qhat_map:
            continue
        qhat = qhat_map[key]["qhat"]
        if np.isnan(qhat):
            continue

        lower = group["pred_bpm"] - qhat
        upper = group["pred_bpm"] + qhat
        covered = ((group["gt_bpm"] >= lower) & (group["gt_bpm"] <= upper)).astype(int)
        coverage = float(covered.mean()) if len(group) > 0 else float("nan")
        target_cov = qhat_map[key]["target_coverage"]

        row = dict(zip(group_cols, keys))
        row.update(
            {
                "alpha": float(qhat_map[key]["alpha"]),
                "target_coverage": float(target_cov),
                "empirical_coverage": coverage,
                "coverage_gap": float(coverage - target_cov) if not np.isnan(coverage) else float("nan"),
                "undercoverage": float(max(0.0, target_cov - coverage)) if not np.isnan(coverage) else float("nan"),
                "avg_interval_width": float(2.0 * qhat),
                "qhat": float(qhat),
                "MAE_test": float(group["abs_error"].mean()),
                "RMSE_test": float(np.sqrt(group["sq_error"].mean())),
                "n_cal": int(qhat_map[key]["n_cal"]),
                "n_test": int(len(group)),
            }
        )
        metric_rows.append(row)

        if args.detailed_output.strip():
            d = group.copy()
            d["lower"] = lower
            d["upper"] = upper
            d["covered"] = covered
            d["qhat"] = float(qhat)
            detailed_frames.append(d)

    if not metric_rows:
        raise SystemExit("No conformal evaluation rows produced. Check group keys/samples/calibration.")

    out_df = pd.DataFrame(metric_rows).sort_values(by=["name", "paradigm"]).reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[OK] Saved conformal evaluation: {output_path}")

    if args.detailed_output.strip() and detailed_frames:
        detailed_path = Path(args.detailed_output)
        detailed_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(detailed_frames, ignore_index=True).to_csv(detailed_path, index=False)
        print(f"[OK] Saved detailed per-sample intervals: {detailed_path}")


if __name__ == "__main__":
    main()
