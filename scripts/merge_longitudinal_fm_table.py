"""
Merge longitudinal FM result CSVs into one camera-ready table.

Expected per-input CSV columns (any one alias is accepted):
- model or name
- mae_next_bpm or MAE
- rmse_next_bpm or RMSE
- acc_deterioration or ACC
- f1_deterioration or F1
- optional: error
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DEFAULT_INPUTS = (
    "outputs/exp3_longitudinal/tabpfn_results.csv,"
    "outputs/exp3_longitudinal/timesfm_results.csv,"
    "outputs/exp3_longitudinal/chronos_results.csv"
)


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _pick_first(row: Dict[str, str], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in row and str(row[key]).strip() != "":
            return str(row[key]).strip()
    return None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _read_rows(path: Path, keep_failed: bool = False) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=2):
            model = _pick_first(raw, ("model", "name"))
            mae = _to_float(_pick_first(raw, ("MAE", "mae_next_bpm", "mae")))
            rmse = _to_float(_pick_first(raw, ("RMSE", "rmse_next_bpm", "rmse")))
            acc = _to_float(_pick_first(raw, ("ACC", "acc_deterioration", "accuracy", "acc")))
            f1 = _to_float(_pick_first(raw, ("F1", "f1_deterioration", "f1")))
            error = _pick_first(raw, ("error",))

            if model is None:
                print(f"[WARN] {path}:{idx} missing model/name, skipping.")
                continue

            if error and not keep_failed:
                print(f"[INFO] {path}:{idx} model='{model}' has error, skipping.")
                continue

            if not keep_failed and any(x is None for x in (mae, rmse, acc, f1)):
                print(f"[WARN] {path}:{idx} model='{model}' missing metric(s), skipping.")
                continue

            rows.append(
                {
                    "model": model,
                    "MAE": mae,
                    "RMSE": rmse,
                    "ACC": acc,
                    "F1": f1,
                    "source_file": str(path),
                    "error": error or "",
                }
            )
    return rows


def _dedupe_best_by_mae(rows: List[Dict]) -> List[Dict]:
    best: Dict[str, Dict] = {}
    for row in rows:
        model = row["model"]
        mae = row["MAE"]
        if model not in best:
            best[model] = row
            continue
        current_best = best[model]
        if mae is not None and (current_best["MAE"] is None or mae < current_best["MAE"]):
            best[model] = row
    return list(best.values())


def _format_float(x: Optional[float], decimals: int) -> str:
    if x is None:
        return "-"
    return f"{x:.{decimals}f}"


def _write_csv(path: Path, rows: List[Dict], decimals: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "MAE", "RMSE", "ACC", "F1"])
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "model": r["model"],
                    "MAE": _format_float(r["MAE"], decimals),
                    "RMSE": _format_float(r["RMSE"], decimals),
                    "ACC": _format_float(r["ACC"], decimals),
                    "F1": _format_float(r["F1"], decimals),
                }
            )


def _write_markdown(path: Path, rows: List[Dict], decimals: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Camera-Ready Longitudinal FM Table",
        "",
        "| Model | MAE | RMSE | ACC | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {model} | {mae} | {rmse} | {acc} | {f1} |".format(
                model=r["model"],
                mae=_format_float(r["MAE"], decimals),
                rmse=_format_float(r["RMSE"], decimals),
                acc=_format_float(r["ACC"], decimals),
                f1=_format_float(r["F1"], decimals),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(path: Path, rows: List[Dict], decimals: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Camera-ready longitudinal FM table",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Model & MAE & RMSE & ACC & F1 \\\\",
        "\\hline",
    ]
    for r in rows:
        lines.append(
            "{model} & {mae} & {rmse} & {acc} & {f1} \\\\".format(
                model=_latex_escape(r["model"]),
                mae=_format_float(r["MAE"], decimals),
                rmse=_format_float(r["RMSE"], decimals),
                acc=_format_float(r["ACC"], decimals),
                f1=_format_float(r["F1"], decimals),
            )
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Merge TabPFN/TimesFM/Chronos CSVs into one camera-ready table")
    parser.add_argument(
        "--inputs",
        type=str,
        default=DEFAULT_INPUTS,
        help="Comma-separated CSV paths",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="",
        help="Optional glob pattern to include extra CSV files",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/exp3_longitudinal/camera_ready_fm_table",
        help="Output prefix (writes .csv .md .tex)",
    )
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument(
        "--keep-failed",
        action="store_true",
        help="Keep rows that have errors / missing metrics",
    )
    args = parser.parse_args()

    input_paths: List[Path] = []
    for p in _parse_csv_list(args.inputs):
        path = Path(p)
        if path.exists():
            input_paths.append(path)
        else:
            print(f"[WARN] Missing input: {path}")

    if args.input_glob.strip():
        globbed = [p for p in Path(".").glob(args.input_glob) if p.is_file()]
        for p in sorted(globbed):
            if p not in input_paths:
                input_paths.append(p)

    if not input_paths:
        raise SystemExit("No input CSV files found.")

    merged_rows: List[Dict] = []
    for path in input_paths:
        rows = _read_rows(path, keep_failed=args.keep_failed)
        merged_rows.extend(rows)

    if not merged_rows:
        raise SystemExit("No valid rows found in provided CSV files.")

    rows = _dedupe_best_by_mae(merged_rows)
    rows = sorted(rows, key=lambda r: (float("inf") if r["MAE"] is None else r["MAE"], r["model"]))

    output_prefix = Path(args.output_prefix)
    out_csv = output_prefix.with_suffix(".csv")
    out_md = output_prefix.with_suffix(".md")
    out_tex = output_prefix.with_suffix(".tex")

    _write_csv(out_csv, rows, args.decimals)
    _write_markdown(out_md, rows, args.decimals)
    _write_latex(out_tex, rows, args.decimals)

    print(f"Merged {len(rows)} model rows from {len(input_paths)} file(s).")
    print(f"CSV : {out_csv}")
    print(f"MD  : {out_md}")
    print(f"TEX : {out_tex}")


if __name__ == "__main__":
    main()

