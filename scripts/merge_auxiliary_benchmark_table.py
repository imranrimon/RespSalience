"""
Merge auxiliary MINI-RGBD and ICBHI results into one camera-ready table.

Inputs:
- MINI summary CSV from eval_mini_rgbd_aux.py
- ICBHI JSON from eval_icbhi.py

Outputs:
- CSV / Markdown / LaTeX table
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _fmt(x, decimals: int = 3) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)
    except Exception:
        return str(x)
    if pd.isna(v):
        return "-"
    return f"{v:.{decimals}f}"


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


def load_mini_row(mini_summary_csv: Path) -> Dict:
    if not mini_summary_csv.exists():
        return {}
    df = pd.read_csv(mini_summary_csv)
    if df.empty:
        return {}
    r = df.iloc[0]
    return {
        "Dataset": "MINI-RGBD",
        "Task": "Image-sequence transfer (auxiliary)",
        "Modality": "RGB/Depth image sequence",
        "N": int(r.get("n_sequences_success", r.get("n_sequences_total", 0))),
        "MAE": None,
        "RMSE": None,
        "IoA@2": None,
        "IoA@3": None,
        "Conf": float(r.get("confidence_mean")) if "confidence_mean" in r else None,
        "Proxy": f"plausible_rate={_fmt(r.get('plausible_rate'), 3)}",
        "Notes": "No direct RR labels; proxy quality only.",
    }


def load_icbhi_row(icbhi_json: Path) -> Dict:
    if not icbhi_json.exists():
        return {}
    data = json.loads(icbhi_json.read_text(encoding="utf-8"))
    return {
        "Dataset": "ICBHI",
        "Task": "Audio-only RR (auxiliary)",
        "Modality": "Audio",
        "N": int(data.get("n", 0)),
        "MAE": data.get("mae"),
        "RMSE": None,
        "IoA@2": data.get("ioa_2bpm"),
        "IoA@3": data.get("ioa_3bpm"),
        "Conf": None,
        "Proxy": "",
        "Notes": "From paired ICBHI annotations.",
    }


def write_markdown(path: Path, rows: List[Dict], decimals: int):
    lines = [
        "# Auxiliary Benchmark Table",
        "",
        "| Dataset | Task | Modality | N | MAE | RMSE | IoA@2 | IoA@3 | Conf | Proxy | Notes |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {Dataset} | {Task} | {Modality} | {N} | {MAE} | {RMSE} | {IoA2} | {IoA3} | {Conf} | {Proxy} | {Notes} |".format(
                Dataset=r["Dataset"],
                Task=r["Task"],
                Modality=r["Modality"],
                N=r["N"],
                MAE=_fmt(r["MAE"], decimals),
                RMSE=_fmt(r["RMSE"], decimals),
                IoA2=_fmt(r["IoA@2"], decimals),
                IoA3=_fmt(r["IoA@3"], decimals),
                Conf=_fmt(r["Conf"], decimals),
                Proxy=r["Proxy"],
                Notes=r["Notes"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: List[Dict], decimals: int):
    lines = [
        "% Auxiliary benchmark table",
        "\\begin{tabular}{lllrrrrrll}",
        "\\hline",
        "Dataset & Task & Modality & N & MAE & RMSE & IoA@2 & IoA@3 & Conf & Notes \\\\",
        "\\hline",
    ]
    for r in rows:
        lines.append(
            "{d} & {t} & {m} & {n} & {mae} & {rmse} & {ioa2} & {ioa3} & {conf} & {notes} \\\\".format(
                d=_latex_escape(r["Dataset"]),
                t=_latex_escape(r["Task"]),
                m=_latex_escape(r["Modality"]),
                n=r["N"],
                mae=_fmt(r["MAE"], decimals),
                rmse=_fmt(r["RMSE"], decimals),
                ioa2=_fmt(r["IoA@2"], decimals),
                ioa3=_fmt(r["IoA@3"], decimals),
                conf=_fmt(r["Conf"], decimals),
                notes=_latex_escape(r["Notes"]),
            )
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Merge MINI-RGBD + ICBHI auxiliary results into one table")
    parser.add_argument("--mini-summary-csv", type=str, default="outputs/auxiliary/mini_rgbd_summary.csv")
    parser.add_argument("--icbhi-json", type=str, default="data/ablation_results/icbhi_audio_results.json")
    parser.add_argument("--output-prefix", type=str, default="outputs/auxiliary/auxiliary_benchmark_table")
    parser.add_argument("--decimals", type=int, default=3)
    args = parser.parse_args()

    rows = []
    mini_row = load_mini_row(Path(args.mini_summary_csv))
    if mini_row:
        rows.append(mini_row)
    else:
        print(f"[WARN] MINI summary missing/empty: {args.mini_summary_csv}")

    icbhi_row = load_icbhi_row(Path(args.icbhi_json))
    if icbhi_row:
        rows.append(icbhi_row)
    else:
        print(f"[WARN] ICBHI JSON missing/empty: {args.icbhi_json}")

    if not rows:
        raise SystemExit("No auxiliary rows to write.")

    out_prefix = Path(args.output_prefix)
    out_csv = out_prefix.with_suffix(".csv")
    out_md = out_prefix.with_suffix(".md")
    out_tex = out_prefix.with_suffix(".tex")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    write_markdown(out_md, rows, args.decimals)
    write_latex(out_tex, rows, args.decimals)

    print(f"[OK] CSV : {out_csv}")
    print(f"[OK] MD  : {out_md}")
    print(f"[OK] TEX : {out_tex}")


if __name__ == "__main__":
    main()

