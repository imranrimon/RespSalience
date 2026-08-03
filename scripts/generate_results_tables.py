"""
Generate camera-ready result tables from benchmark outputs.

Inputs:
- Preferred: leaderboard CSVs from run_camera_ready_pipeline
  (../leaderboards/core_leaderboard_*.csv, modality_leaderboard_*.csv)
- Fallback: ablation_report_*.csv files inside results_dir

Outputs:
- core_results_table.csv/.md/.tex
- modality_results_table.csv/.md/.tex
- combined_results_table.csv/.md/.tex
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


STANDARD_COLS = [
    "name",
    "paradigm",
    "N_total",
    "MAE",
    "RMSE",
    "Pearson_r",
    "IoA2",
    "IoA3",
]

CONFORMAL_COLS = [
    "name",
    "paradigm",
    "disagreement_bin",
    "n_cal",
    "n_test",
    "alpha",
    "target_coverage",
    "empirical_coverage",
    "coverage_gap",
    "undercoverage",
    "avg_interval_width",
    "qhat",
    "MAE_test",
    "RMSE_test",
]


def _latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _parse_seed(path: Path) -> int:
    m = re.search(r"seed(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def _standardize_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["name"] = df.get("name", "")
    out["paradigm"] = df.get("paradigm", "")

    if "N_total" in df.columns:
        out["N_total"] = pd.to_numeric(df["N_total"], errors="coerce").fillna(0).astype(int)
    elif "N" in df.columns:
        out["N_total"] = pd.to_numeric(df["N"], errors="coerce").fillna(0).astype(int)
    else:
        out["N_total"] = 0

    # Accept either raw metrics or pre-aggregated mean columns.
    out["MAE"] = pd.to_numeric(
        df["MAE_overall"] if "MAE_overall" in df.columns else df.get("MAE_overall_mean"),
        errors="coerce",
    )
    out["RMSE"] = pd.to_numeric(
        df["RMSE_overall"] if "RMSE_overall" in df.columns else df.get("RMSE_overall_mean"),
        errors="coerce",
    )
    out["Pearson_r"] = pd.to_numeric(
        df["Pearson_r_overall"] if "Pearson_r_overall" in df.columns else df.get("Pearson_r_overall_mean"),
        errors="coerce",
    )
    out["IoA2"] = pd.to_numeric(
        df["IoA2_overall"] if "IoA2_overall" in df.columns else df.get("IoA2_overall_mean"),
        errors="coerce",
    )
    out["IoA3"] = pd.to_numeric(
        df["IoA3_overall"] if "IoA3_overall" in df.columns else df.get("IoA3_overall_mean"),
        errors="coerce",
    )

    out = out.dropna(subset=["name"])
    out = out[STANDARD_COLS]
    out = out.sort_values(by=["MAE", "RMSE", "name"], na_position="last").reset_index(drop=True)
    return out


def _pick_numeric(df: pd.DataFrame, *candidates: str) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series([float("nan")] * len(df))


def _standardize_conformal(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["name"] = df.get("name", "")
    out["paradigm"] = df.get("paradigm", "")
    out["disagreement_bin"] = df["disagreement_bin"] if "disagreement_bin" in df.columns else "all"
    out["n_cal"] = _pick_numeric(df, "n_cal", "n_cal_mean")
    out["n_test"] = _pick_numeric(df, "n_test", "n_test_mean")
    out["alpha"] = _pick_numeric(df, "alpha", "alpha_mean")
    out["target_coverage"] = _pick_numeric(df, "target_coverage", "target_coverage_mean")
    out["empirical_coverage"] = _pick_numeric(df, "empirical_coverage", "empirical_coverage_mean")
    out["coverage_gap"] = _pick_numeric(df, "coverage_gap", "coverage_gap_mean")
    out["undercoverage"] = _pick_numeric(df, "undercoverage", "undercoverage_mean")
    out["avg_interval_width"] = _pick_numeric(df, "avg_interval_width", "avg_interval_width_mean")
    out["qhat"] = _pick_numeric(df, "qhat", "qhat_mean")
    out["MAE_test"] = _pick_numeric(df, "MAE_test", "MAE_test_mean")
    out["RMSE_test"] = _pick_numeric(df, "RMSE_test", "RMSE_test_mean")
    out = out.dropna(subset=["name"])
    out = out[CONFORMAL_COLS]
    out = out.sort_values(
        by=["undercoverage", "avg_interval_width", "name", "disagreement_bin"],
        na_position="last",
    ).reset_index(drop=True)
    return out


def _aggregate_seed_files(seed_files: List[Path]) -> Optional[pd.DataFrame]:
    if not seed_files:
        return None

    frames: List[pd.DataFrame] = []
    for f in seed_files:
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            df["seed"] = _parse_seed(f)
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Failed to read {f}: {exc}")

    if not frames:
        return None

    all_df = pd.concat(frames, ignore_index=True)
    metrics = []
    for c in all_df.columns:
        if c in ("name", "paradigm", "seed"):
            continue
        if pd.api.types.is_numeric_dtype(all_df[c]):
            metrics.append(c)
    if not metrics:
        return None

    grouped = all_df.groupby(["name", "paradigm"], as_index=False)[metrics].agg(["mean", "std"])
    grouped.columns = [
        "_".join([x for x in col if x]).strip("_") if isinstance(col, tuple) else str(col)
        for col in grouped.columns
    ]
    # Keep mean names available as raw names for downstream standardizers.
    for c in list(grouped.columns):
        if c.endswith("_mean"):
            base = c[: -len("_mean")]
            if base not in grouped.columns:
                grouped = grouped.rename(columns={c: base})
    return grouped


def _load_leaderboard(leaderboards_dir: Path, prefix: str) -> Optional[pd.DataFrame]:
    agg_path = leaderboards_dir / f"{prefix}_seed_aggregate.csv"
    if agg_path.exists():
        try:
            return pd.read_csv(agg_path)
        except Exception as exc:
            print(f"[WARN] Failed to read aggregate file {agg_path}: {exc}")

    seed_files = sorted(leaderboards_dir.glob(f"{prefix}_seed*.csv"))
    seed_files = [p for p in seed_files if not p.name.endswith("_aggregate.csv") and not p.name.endswith("_raw.csv")]
    return _aggregate_seed_files(seed_files)


def _load_fallback_from_ablation(results_dir: Path) -> Optional[pd.DataFrame]:
    files = sorted(results_dir.glob("ablation_report_*.csv"))
    if not files:
        return None

    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}")

    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)
    if "name" not in raw.columns:
        return None

    out = pd.DataFrame()
    out["name"] = raw["name"]
    out["paradigm"] = raw["paradigm"] if "paradigm" in raw.columns else ""
    out["N_total"] = pd.to_numeric(raw["N"], errors="coerce").fillna(0).astype(int) if "N" in raw.columns else 0
    out["MAE"] = pd.to_numeric(raw.get("MAE"), errors="coerce")
    out["RMSE"] = pd.to_numeric(raw.get("RMSE"), errors="coerce")
    out["Pearson_r"] = pd.to_numeric(raw.get("Pearson_r"), errors="coerce")
    out["IoA2"] = pd.to_numeric(raw.get("IoA@2BPM"), errors="coerce")
    out["IoA3"] = pd.to_numeric(raw.get("IoA@3BPM"), errors="coerce")

    metric_cols = ["MAE", "RMSE", "Pearson_r", "IoA2", "IoA3"]
    agg = out.groupby(["name", "paradigm"], as_index=False).agg(
        {
            "N_total": "sum",
            **{c: "mean" for c in metric_cols},
        }
    )
    agg = agg[STANDARD_COLS]
    return agg.sort_values(by=["MAE", "RMSE", "name"], na_position="last").reset_index(drop=True)


def _fmt(x, decimals: int = 3) -> str:
    if pd.isna(x):
        return "-"
    if isinstance(x, (int,)) or (isinstance(x, float) and float(x).is_integer()):
        return str(int(x))
    return f"{float(x):.{decimals}f}"


def _write_csv(df: pd.DataFrame, out_path: Path, decimals: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    for col in ["MAE", "RMSE", "Pearson_r", "IoA2", "IoA3"]:
        if col in out_df.columns:
            out_df[col] = out_df[col].map(lambda x: _fmt(x, decimals))
    out_df.to_csv(out_path, index=False)
    print(f"[OK] CSV: {out_path}")


def _write_markdown(df: pd.DataFrame, out_path: Path, decimals: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Camera-Ready Results Table",
        "",
        "| Model | Paradigm | N | MAE | RMSE | Pearson_r | IoA@2 | IoA@3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| {name} | {paradigm} | {n} | {mae} | {rmse} | {pr} | {ioa2} | {ioa3} |".format(
                name=r.get("name", ""),
                paradigm=r.get("paradigm", ""),
                n=_fmt(r.get("N_total"), 0),
                mae=_fmt(r.get("MAE"), decimals),
                rmse=_fmt(r.get("RMSE"), decimals),
                pr=_fmt(r.get("Pearson_r"), decimals),
                ioa2=_fmt(r.get("IoA2"), decimals),
                ioa3=_fmt(r.get("IoA3"), decimals),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] MD: {out_path}")


def _write_latex(df: pd.DataFrame, out_path: Path, decimals: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Camera-ready results table",
        "\\begin{tabular}{l l c c c c c c}",
        "\\hline",
        "Model & Paradigm & N & MAE & RMSE & Pearson\\_r & IoA@2 & IoA@3 \\\\",
        "\\hline",
    ]
    for _, r in df.iterrows():
        lines.append(
            "{name} & {paradigm} & {n} & {mae} & {rmse} & {pr} & {ioa2} & {ioa3} \\\\".format(
                name=_latex_escape(r.get("name", "")),
                paradigm=_latex_escape(r.get("paradigm", "")),
                n=_fmt(r.get("N_total"), 0),
                mae=_fmt(r.get("MAE"), decimals),
                rmse=_fmt(r.get("RMSE"), decimals),
                pr=_fmt(r.get("Pearson_r"), decimals),
                ioa2=_fmt(r.get("IoA2"), decimals),
                ioa3=_fmt(r.get("IoA3"), decimals),
            )
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] TEX: {out_path}")


def _write_outputs(df: pd.DataFrame, prefix: str, output_dir: Path, output_format: str, decimals: int):
    csv_path = output_dir / f"{prefix}.csv"
    md_path = output_dir / f"{prefix}.md"
    tex_path = output_dir / f"{prefix}.tex"

    if output_format in ("csv", "both"):
        _write_csv(df, csv_path, decimals)
    if output_format in ("markdown", "both"):
        _write_markdown(df, md_path, decimals)
    if output_format in ("latex", "both"):
        _write_latex(df, tex_path, decimals)


def _write_conformal_csv(df: pd.DataFrame, out_path: Path, decimals: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df.copy()
    for col in [
        "alpha",
        "target_coverage",
        "empirical_coverage",
        "coverage_gap",
        "undercoverage",
        "avg_interval_width",
        "qhat",
        "MAE_test",
        "RMSE_test",
    ]:
        if col in out_df.columns:
            out_df[col] = out_df[col].map(lambda x: _fmt(x, decimals))
    for col in ["n_cal", "n_test"]:
        if col in out_df.columns:
            out_df[col] = out_df[col].map(lambda x: _fmt(x, 0))
    out_df.to_csv(out_path, index=False)
    print(f"[OK] CSV: {out_path}")


def _write_conformal_markdown(df: pd.DataFrame, out_path: Path, decimals: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Conformal Results Table",
        "",
        "| Model | Paradigm | N_cal | N_test | alpha | target_cov | empirical_cov | undercoverage | width | qhat | MAE_test | RMSE_test |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| {name} | {paradigm} | {n_cal} | {n_test} | {alpha} | {target} | {emp} | {under} | {width} | {qhat} | {mae} | {rmse} |".format(
                name=r.get("name", ""),
                paradigm=r.get("paradigm", ""),
                n_cal=_fmt(r.get("n_cal"), 0),
                n_test=_fmt(r.get("n_test"), 0),
                alpha=_fmt(r.get("alpha"), decimals),
                target=_fmt(r.get("target_coverage"), decimals),
                emp=_fmt(r.get("empirical_coverage"), decimals),
                under=_fmt(r.get("undercoverage"), decimals),
                width=_fmt(r.get("avg_interval_width"), decimals),
                qhat=_fmt(r.get("qhat"), decimals),
                mae=_fmt(r.get("MAE_test"), decimals),
                rmse=_fmt(r.get("RMSE_test"), decimals),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] MD: {out_path}")


def _write_conformal_latex(df: pd.DataFrame, out_path: Path, decimals: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Conformal results table",
        "\\begin{tabular}{l l c c c c c c c c c c}",
        "\\hline",
        "Model & Paradigm & Ncal & Ntest & alpha & TargetCov & EmpCov & UnderCov & Width & qhat & MAEtest & RMSEtest \\\\",
        "\\hline",
    ]
    for _, r in df.iterrows():
        lines.append(
            "{name} & {paradigm} & {n_cal} & {n_test} & {alpha} & {target} & {emp} & {under} & {width} & {qhat} & {mae} & {rmse} \\\\".format(
                name=_latex_escape(r.get("name", "")),
                paradigm=_latex_escape(r.get("paradigm", "")),
                n_cal=_fmt(r.get("n_cal"), 0),
                n_test=_fmt(r.get("n_test"), 0),
                alpha=_fmt(r.get("alpha"), decimals),
                target=_fmt(r.get("target_coverage"), decimals),
                emp=_fmt(r.get("empirical_coverage"), decimals),
                under=_fmt(r.get("undercoverage"), decimals),
                width=_fmt(r.get("avg_interval_width"), decimals),
                qhat=_fmt(r.get("qhat"), decimals),
                mae=_fmt(r.get("MAE_test"), decimals),
                rmse=_fmt(r.get("RMSE_test"), decimals),
            )
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] TEX: {out_path}")


def _write_conformal_outputs(df: pd.DataFrame, prefix: str, output_dir: Path, output_format: str, decimals: int):
    csv_path = output_dir / f"{prefix}.csv"
    md_path = output_dir / f"{prefix}.md"
    tex_path = output_dir / f"{prefix}.tex"

    if output_format in ("csv", "both"):
        _write_conformal_csv(df, csv_path, decimals)
    if output_format in ("markdown", "both"):
        _write_conformal_markdown(df, md_path, decimals)
    if output_format in ("latex", "both"):
        _write_conformal_latex(df, tex_path, decimals)


def main():
    parser = argparse.ArgumentParser(description="Generate camera-ready result tables from benchmark outputs.")
    parser.add_argument("--results_dir", type=str, default="data/ablation_results")
    parser.add_argument("--format", choices=["csv", "latex", "markdown", "both"], default="both")
    parser.add_argument("--output", type=str, default="outputs/tables")
    parser.add_argument("--decimals", type=int, default=3)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = results_dir.parent if results_dir.name == "ablation_cache" else results_dir
    leaderboards_dir = run_dir / "leaderboards"

    core_df = None
    modality_df = None
    conformal_core_df = None
    conformal_modality_df = None
    if leaderboards_dir.exists():
        core_raw = _load_leaderboard(leaderboards_dir, "core_leaderboard")
        modality_raw = _load_leaderboard(leaderboards_dir, "modality_leaderboard")
        conformal_core_raw = _load_leaderboard(leaderboards_dir, "conformal_core")
        conformal_modality_raw = _load_leaderboard(leaderboards_dir, "conformal_modality")
        if core_raw is not None and not core_raw.empty:
            core_df = _standardize_leaderboard(core_raw)
        if modality_raw is not None and not modality_raw.empty:
            modality_df = _standardize_leaderboard(modality_raw)
        if conformal_core_raw is not None and not conformal_core_raw.empty:
            conformal_core_df = _standardize_conformal(conformal_core_raw)
        if conformal_modality_raw is not None and not conformal_modality_raw.empty:
            conformal_modality_df = _standardize_conformal(conformal_modality_raw)

    if core_df is None and modality_df is None and conformal_core_df is None and conformal_modality_df is None:
        fallback_df = _load_fallback_from_ablation(results_dir)
        if fallback_df is None or fallback_df.empty:
            raise SystemExit(
                f"No leaderboard or ablation report data found in: {results_dir} (or {leaderboards_dir})"
            )
        _write_outputs(fallback_df, "combined_results_table", output_dir, args.format, args.decimals)
        print("[INFO] Generated combined table from ablation_report fallback only.")
        return

    if core_df is not None and not core_df.empty:
        _write_outputs(core_df, "core_results_table", output_dir, args.format, args.decimals)
    if modality_df is not None and not modality_df.empty:
        _write_outputs(modality_df, "modality_results_table", output_dir, args.format, args.decimals)

    parts = [df for df in [core_df, modality_df] if df is not None and not df.empty]
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.sort_values(by=["MAE", "RMSE", "name"], na_position="last")
        combined = combined.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)
        _write_outputs(combined, "combined_results_table", output_dir, args.format, args.decimals)

    if conformal_core_df is not None and not conformal_core_df.empty:
        _write_conformal_outputs(conformal_core_df, "conformal_core_results_table", output_dir, args.format, args.decimals)
    if conformal_modality_df is not None and not conformal_modality_df.empty:
        _write_conformal_outputs(
            conformal_modality_df, "conformal_modality_results_table", output_dir, args.format, args.decimals
        )
    conformal_parts = [df for df in [conformal_core_df, conformal_modality_df] if df is not None and not df.empty]
    if conformal_parts:
        conformal_combined = pd.concat(conformal_parts, ignore_index=True)
        conformal_combined = conformal_combined.sort_values(
            by=["undercoverage", "avg_interval_width", "name", "disagreement_bin"], na_position="last"
        )
        dedup_cols = ["name", "paradigm"]
        if "disagreement_bin" in conformal_combined.columns:
            dedup_cols.append("disagreement_bin")
        conformal_combined = conformal_combined.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
        _write_conformal_outputs(
            conformal_combined, "conformal_combined_results_table", output_dir, args.format, args.decimals
        )

    print(f"[DONE] Wrote tables to: {output_dir}")


if __name__ == "__main__":
    main()
