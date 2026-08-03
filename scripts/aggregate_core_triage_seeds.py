"""
Aggregate seed-wise calibrated triage leaderboard CSVs into one mean/std table.

Default input pattern:
  <run_dir>/leaderboards/core_triage_calibrated_seed*.csv

Default output:
  <run_dir>/leaderboards/core_triage_calibrated_seed_aggregate.csv
"""

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _seed_from_filename(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return -1


def _resolve_inputs(run_dir: Path, inputs_text: str, input_glob: str) -> List[Path]:
    paths: List[Path] = []

    for raw in _parse_csv_list(inputs_text):
        p = Path(raw)
        if p.exists() and p.is_file():
            paths.append(p)
        else:
            print(f"[WARN] Missing input CSV: {p}")

    if input_glob.strip():
        base = run_dir if run_dir.exists() else Path(".")
        globbed = sorted([p for p in base.glob(input_glob) if p.is_file()])
        for p in globbed:
            if p not in paths:
                paths.append(p)

    return paths


def aggregate_seed_csvs(csv_paths: List[Path], group_cols: List[str]) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"[WARN] Empty CSV skipped: {path}")
                continue
            df["seed"] = _seed_from_filename(path)
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}")

    if not frames:
        raise SystemExit("No valid seed CSV rows loaded.")

    all_df = pd.concat(frames, ignore_index=True)

    valid_group_cols = [c for c in group_cols if c in all_df.columns]
    if not valid_group_cols:
        raise SystemExit(f"None of requested group columns exist: {group_cols}")

    numeric_cols = []
    for c in all_df.columns:
        if c in valid_group_cols or c == "seed":
            continue
        if pd.api.types.is_numeric_dtype(all_df[c]):
            numeric_cols.append(c)

    if not numeric_cols:
        raise SystemExit("No numeric metric columns found to aggregate.")

    grouped = all_df.groupby(valid_group_cols, as_index=False)[numeric_cols].agg(["mean", "std"])
    grouped.columns = [
        "_".join([x for x in col if x]).strip("_") if isinstance(col, tuple) else str(col)
        for col in grouped.columns
    ]

    seed_count = (
        all_df.groupby(valid_group_cols, as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "n_seeds"})
    )
    out = grouped.merge(seed_count, on=valid_group_cols, how="left")

    sort_candidates = [c for c in ["MAE_overall_mean", "RMSE_overall_mean", "name"] if c in out.columns]
    if sort_candidates:
        out = out.sort_values(by=sort_candidates, na_position="last")
    return out.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate core_triage_calibrated_seed*.csv into a single mean/std leaderboard table"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Camera-ready run directory (contains leaderboards/)",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default="",
        help="Optional comma-separated explicit CSV paths",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="leaderboards/core_triage_calibrated_seed*.csv",
        help="Glob pattern relative to --run-dir",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="leaderboards/core_triage_calibrated_seed_aggregate.csv",
        help="Output CSV path relative to --run-dir, unless absolute path is provided",
    )
    parser.add_argument(
        "--group-cols",
        type=str,
        default="name,paradigm",
        help="Comma-separated grouping columns",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"--run-dir does not exist: {run_dir}")

    csv_paths = _resolve_inputs(run_dir, args.inputs, args.input_glob)
    if not csv_paths:
        raise SystemExit("No input CSV files found.")

    group_cols = _parse_csv_list(args.group_cols)
    out_df = aggregate_seed_csvs(csv_paths, group_cols=group_cols)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = run_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"[OK] Aggregated {len(csv_paths)} seed file(s).")
    print(f"[OK] Output: {output_path}")


if __name__ == "__main__":
    main()

