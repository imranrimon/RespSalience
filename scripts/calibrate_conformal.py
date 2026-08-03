"""
Calibrate split conformal quantiles for respiratory rate regression.

Input CSV must contain at least:
- name
- pred_bpm
- gt_bpm

Recommended extra columns:
- paradigm
- source
- subject
- seed
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
    # Fallback: deterministic row identifier if split unit column is missing.
    return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)


def _conformal_quantile(errors: np.ndarray, alpha: float) -> float:
    """Finite-sample split conformal quantile with ceil((n+1)(1-alpha))."""
    n = len(errors)
    if n == 0:
        return float("nan")
    ordered = np.sort(errors.astype(float))
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = max(1, min(rank, n))
    return float(ordered[rank - 1])


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
            df["sample_file"] = str(p)
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Failed to read {p}: {exc}")

    if not frames:
        raise SystemExit("Sample CSVs were found but no readable rows were loaded.")

    out = pd.concat(frames, ignore_index=True)
    alias_pairs = [
        ("pred_rr_bpm", "pred_bpm"),
        ("gt_rr_bpm", "gt_bpm"),
        ("dataset", "source"),
        ("subject_id", "subject"),
    ]
    for source_col, target_col in alias_pairs:
        if target_col not in out.columns and source_col in out.columns:
            out[target_col] = out[source_col]
    if "name" not in out.columns:
        out["name"] = "PRISM-Breath"
    required = {"name", "pred_bpm", "gt_bpm"}
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise SystemExit(f"Missing required sample columns: {missing}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Calibrate split conformal quantiles from per-sample RR predictions.")
    parser.add_argument("--samples", type=str, default="", help="Comma-separated sample CSV files")
    parser.add_argument("--sample-glob", type=str, default="", help="Optional glob for sample CSV files")
    parser.add_argument("--output", type=str, required=True, help="Output calibration CSV")
    parser.add_argument("--split-output", type=str, default="", help="Optional output CSV with split assignments")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level (e.g., 0.1 for 90% coverage)")
    parser.add_argument(
        "--calibration-ratio",
        type=float,
        default=0.3,
        help="Fraction of split units allocated to calibration (0,1)",
    )
    parser.add_argument("--split-unit", type=str, default="subject", help="Column used for deterministic split")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument(
        "--use-all-samples",
        action="store_true",
        help="Use every provided sample for calibration instead of splitting within the CSV.",
    )
    parser.add_argument(
        "--group-by-source",
        action="store_true",
        help="Calibrate separately per source dataset in addition to model/paradigm",
    )
    parser.add_argument(
        "--condition-on-disagreement",
        action="store_true",
        help="Calibrate separate qhat for disagreement bins (low/mid/high).",
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

    if not (0.0 < args.alpha < 1.0):
        raise SystemExit("--alpha must be in (0,1)")
    if not args.use_all_samples and not (0.0 < args.calibration_ratio < 1.0):
        raise SystemExit("--calibration-ratio must be in (0,1)")

    samples = load_samples(args.samples, args.sample_glob).copy()
    samples["pred_bpm"] = pd.to_numeric(samples["pred_bpm"], errors="coerce")
    samples["gt_bpm"] = pd.to_numeric(samples["gt_bpm"], errors="coerce")
    samples = samples.dropna(subset=["name", "pred_bpm", "gt_bpm"]).reset_index(drop=True)
    samples["abs_error"] = np.abs(samples["gt_bpm"] - samples["pred_bpm"])

    if "paradigm" not in samples.columns:
        samples["paradigm"] = "Unknown"
    if "source" not in samples.columns:
        samples["source"] = "Unknown"

    unit_vals = _ensure_split_unit(samples, args.split_unit)
    samples["__split_unit"] = unit_vals
    if args.use_all_samples:
        samples["is_calibration"] = True
    else:
        samples["is_calibration"] = samples["__split_unit"].map(
            lambda x: _stable_uniform(str(x), args.seed) < args.calibration_ratio
        )

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

    group_cols = ["name", "paradigm"]
    if args.group_by_source:
        group_cols.append("source")
    if args.condition_on_disagreement:
        group_cols.append("disagreement_bin")

    rows = []
    for keys, group in samples.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))

        cal = group[group["is_calibration"]]
        test = group[~group["is_calibration"]]

        # Guard against degenerate tiny groups.
        if cal.empty and not group.empty:
            cal = group.iloc[[0]]
            test = group.iloc[1:]
        if not args.use_all_samples and test.empty and len(group) > 1:
            test = group.iloc[[-1]]
            cal = group.iloc[:-1]

        qhat = _conformal_quantile(cal["abs_error"].to_numpy(), args.alpha)
        row.update(
            {
                "alpha": float(args.alpha),
                "target_coverage": float(1.0 - args.alpha),
                "calibration_ratio": float(1.0 if args.use_all_samples else args.calibration_ratio),
                "split_unit": args.split_unit,
                "n_total": int(len(group)),
                "n_cal": int(len(cal)),
                "n_test": int(len(test)),
                "qhat": qhat,
            }
        )
        rows.append(row)

    calib_df = pd.DataFrame(rows).sort_values(by=group_cols).reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calib_df.to_csv(output_path, index=False)
    print(f"[OK] Saved conformal calibration: {output_path}")

    if args.split_output.strip():
        split_df = samples[
            [
                c
                for c in [
                    "name",
                    "paradigm",
                    "source",
                    args.split_unit if args.split_unit in samples.columns else "__split_unit",
                    "__split_unit",
                    "is_calibration",
                    args.disagreement_col if args.disagreement_col in samples.columns else None,
                    "disagreement_bin" if "disagreement_bin" in samples.columns else None,
                    "pred_bpm",
                    "gt_bpm",
                    "abs_error",
                    "seed" if "seed" in samples.columns else None,
                    "dataset_filter" if "dataset_filter" in samples.columns else None,
                ]
                if c is not None and c in samples.columns
            ]
        ].copy()
        split_output_path = Path(args.split_output)
        split_output_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(split_output_path, index=False)
        print(f"[OK] Saved split assignments: {split_output_path}")


if __name__ == "__main__":
    main()
