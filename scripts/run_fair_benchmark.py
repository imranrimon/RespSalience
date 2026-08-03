"""
Run a fair, repeatable benchmark across modality matrix + external baselines.

This script wraps AblationOrchestrator with a controlled baseline set and
produces one consolidated leaderboard CSV across selected datasets.
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import random


_DLL_DIR_HANDLES = []


def _configure_windows_runtime() -> None:
    if os.name != "nt":
        return

    python_path = Path(sys.executable).resolve()
    env_root = python_path.parent
    path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    existing = {part.lower() for part in path_parts if part}

    candidate_paths = [
        env_root / "Lib" / "site-packages" / "torch" / "lib",
        env_root,
        env_root / "Library" / "mingw-w64" / "bin",
        env_root / "Library" / "usr" / "bin",
        env_root / "Library" / "bin",
        env_root / "Scripts",
        env_root / "bin",
    ]

    prepended = []
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if not candidate.exists():
            continue
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(candidate_str))
            except OSError:
                pass
        if candidate_str.lower() not in existing:
            prepended.append(candidate_str)
            existing.add(candidate_str.lower())

    if prepended:
        os.environ["PATH"] = os.pathsep.join(prepended + path_parts)

    os.environ.setdefault("CONDA_PREFIX", str(env_root))
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


_configure_windows_runtime()

import numpy as np
import torch

import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.run_ablation_study import AblationOrchestrator


DEFAULT_BASELINES = [
    # Classical / modality baselines
    "Flow-Only",
    "Depth-Only",
    "Fused-SNR",
    # Internal learned models
    "Transformer-Fusion",
    "Tri-Modal VLM-Gated",
    "Quad-Modal Full (Ours)",
    "Quad-Modal Full + Robust Fusion",
    "Quad-Modal Full + Robust Fusion + Conformal Triage",
    # Robustness (missing modality)
    "Quad-Modal-Full (Drop Audio)",
    "Quad-Modal-Full (Drop Flow)",
    "Quad-Modal-Full (Drop Depth)",
    # External baselines (included only if available in environment)
    "DeepPhys",
    # NOTE: "TSCAN" is intentionally excluded from the default suite — see
    # UNSTABLE_BASELINES below. Run it explicitly with --allow-unstable-baselines.
]

DEFAULT_DATASETS = ["AIR-125", "AIR-400", "Pediatric-RR"]

# Baselines with known stability problems that are excluded from default runs.
# They remain runnable via --allow-unstable-baselines for transparency, but are
# dropped (with a warning) otherwise so a default suite never silently OOMs.
UNSTABLE_BASELINES = {
    "TSCAN": "known GPU OOM / memory instability on long infant videos (FRAME_DEPTH=20)",
}


def filter_unstable_baselines(baselines: List[str], allow_unstable: bool) -> List[str]:
    if allow_unstable:
        return list(baselines)
    kept: List[str] = []
    for name in baselines:
        if name in UNSTABLE_BASELINES:
            print(f"[SKIP] Excluding unstable baseline '{name}': {UNSTABLE_BASELINES[name]}. "
                  f"Pass --allow-unstable-baselines to include it.")
        else:
            kept.append(name)
    return kept


def _set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _load_json_mapping(text: str) -> Dict[str, str]:
    if not text:
        return {}
    payload = text.strip()
    if payload.startswith("{"):
        return json.loads(payload)
    path = Path(payload)
    if not path.exists():
        raise FileNotFoundError(f"JSON mapping not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _weighted_mean(sub_df: pd.DataFrame, col: str) -> float:
    if col not in sub_df.columns:
        return float("nan")
    if "N" in sub_df.columns and sub_df["N"].sum() > 0:
        weights = sub_df["N"].astype(float)
        return float((sub_df[col].astype(float) * weights).sum() / weights.sum())
    return float(sub_df[col].astype(float).mean())


def build_leaderboard(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()

    rows = []
    grouped = combined.groupby(["name", "paradigm"], dropna=False)

    for (name, paradigm), sub in grouped:
        row = {
            "name": name,
            "paradigm": paradigm,
            "N_total": int(sub["N"].sum()) if "N" in sub.columns else 0,
            "MAE_overall": _weighted_mean(sub, "MAE"),
            "RMSE_overall": _weighted_mean(sub, "RMSE"),
            "Pearson_r_overall": _weighted_mean(sub, "Pearson_r"),
            "IoA2_overall": _weighted_mean(sub, "IoA@2BPM"),
            "IoA3_overall": _weighted_mean(sub, "IoA@3BPM"),
        }

        for source, ssub in sub.groupby("source"):
            key = str(source).replace("-", "_")
            row[f"MAE_{key}"] = _weighted_mean(ssub, "MAE")
            row[f"RMSE_{key}"] = _weighted_mean(ssub, "RMSE")
            row[f"N_{key}"] = int(ssub["N"].sum()) if "N" in ssub.columns else 0

        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty and "MAE_overall" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(by="MAE_overall", ascending=True)

    return leaderboard


def run_fair_benchmark(
    datasets: List[str],
    baselines: List[str],
    limit: int,
    results_dir: str,
    output_csv: str,
    seed: int = 42,
    disable_vlm: bool = False,
    sample_output_csv: str = "",
    conformal_calibration_csv: str = "",
    conformal_default_qhat: float = 6.0,
    conformal_condition_on_disagreement: bool = False,
    manifest_path: str = "",
    manifest_label: str = "",
    pretrained_overrides: Dict[str, str] | None = None,
    allow_unstable_baselines: bool = False,
):
    _set_global_seed(seed)
    all_reports = []
    all_samples = []
    missing_by_run = {}
    pretrained_overrides = dict(pretrained_overrides or {})
    baselines = filter_unstable_baselines(list(baselines), allow_unstable_baselines)

    if manifest_path:
        label = manifest_label or Path(manifest_path).stem
        run_specs = [(label, manifest_path)]
    else:
        run_specs = [(dataset, "") for dataset in datasets]

    for dataset, current_manifest in run_specs:
        print(f"\n{'=' * 72}")
        if current_manifest:
            print(f"Running fair benchmark on canonical manifest: {current_manifest}")
        else:
            print(f"Running fair benchmark on dataset filter: {dataset}")
        print(f"{'=' * 72}")

        orchestrator = AblationOrchestrator(output_dir=results_dir, seed=seed)
        orchestrator.disable_vlm = disable_vlm
        available = {b["name"]: b for b in orchestrator.baselines}

        selected = [dict(available[name]) for name in baselines if name in available]
        missing = [name for name in baselines if name not in available]
        missing_by_run[dataset] = missing

        if not selected:
            print(f"[WARN] No selected baselines are available for dataset={dataset}. Skipping.")
            continue

        for baseline in selected:
            override_ckpt = pretrained_overrides.get(baseline["name"])
            if override_ckpt:
                if baseline.get("runner") == "external":
                    baseline["checkpoint"] = override_ckpt
                else:
                    baseline["pretrained"] = override_ckpt

        if conformal_calibration_csv:
            for b in selected:
                if b.get("use_conformal_triage", False):
                    b["conformal_calibration_csv"] = conformal_calibration_csv
                    b["conformal_default_qhat"] = float(conformal_default_qhat)
                    b["conformal_condition_on_disagreement"] = bool(conformal_condition_on_disagreement)

        print(f"Selected baselines ({len(selected)}): {[b['name'] for b in selected]}")
        if missing:
            print(f"[INFO] Missing baselines in this environment: {missing}")

        orchestrator.baselines = selected
        orchestrator._baseline_explicitly_set = True

        report = orchestrator.run_evaluation(
            limit=limit if limit > 0 else None,
            dataset_filter=None if current_manifest else dataset,
            manifest_path=current_manifest or None,
            generate_plots_flag=False,
        )
        if report is None or report.empty:
            print(f"[WARN] Empty report for dataset={dataset}")
            continue

        report = report.copy()
        # Ablation manager often returns a multi-index report
        # (paradigm, source, name) as index rather than columns.
        if "name" not in report.columns or "paradigm" not in report.columns:
            report = report.reset_index()
        report["dataset_filter"] = dataset
        all_reports.append(report)

        # Export compact per-sample rows for downstream conformal calibration/eval.
        sample_df = pd.DataFrame(orchestrator.ablation_mgr.results)
        if not sample_df.empty:
            def _parse_telemetry(val):
                if isinstance(val, dict):
                    return val
                if isinstance(val, str):
                    txt = val.strip()
                    if txt.startswith("{") and txt.endswith("}"):
                        try:
                            parsed = ast.literal_eval(txt)
                            if isinstance(parsed, dict):
                                return parsed
                        except Exception:
                            return {}
                return {}

            if "cross_modal_telemetry" in sample_df.columns:
                telemetry = sample_df["cross_modal_telemetry"].apply(_parse_telemetry)
                sample_df["model_disagreement_score"] = telemetry.apply(
                    lambda t: float(t.get("model_disagreement_score", 0.0) or 0.0)
                )
                sample_df["bpm_disagreement_ratio"] = telemetry.apply(
                    lambda t: float(t.get("bpm_disagreement_ratio", 0.0) or 0.0)
                )
                sample_df["disagreement_detected"] = telemetry.apply(
                    lambda t: bool(t.get("disagreement_detected", False))
                )
                sample_df["disagreement_score"] = sample_df[
                    ["model_disagreement_score", "bpm_disagreement_ratio"]
                ].max(axis=1)

            keep_cols = [
                "name",
                "paradigm",
                "source",
                "subject",
                "pred_bpm",
                "gt_bpm",
                "abs_error",
                "squared_error",
                "fusion_strategy",
                "model_disagreement_score",
                "bpm_disagreement_ratio",
                "disagreement_score",
                "disagreement_detected",
            ]
            compact = sample_df[[c for c in keep_cols if c in sample_df.columns]].copy()
            compact["dataset_filter"] = dataset
            compact["seed"] = seed
            all_samples.append(compact)

    if not all_reports:
        raise RuntimeError("No benchmark reports were generated.")

    combined = pd.concat(all_reports, ignore_index=True)
    leaderboard = build_leaderboard(combined)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    leaderboard.to_csv(output_path, index=False)
    raw_path = output_path.with_name(output_path.stem + "_raw.csv")
    combined.to_csv(raw_path, index=False)

    print(f"\nSaved leaderboard: {output_path}")
    print(f"Saved raw combined report: {raw_path}")

    if sample_output_csv:
        if all_samples:
            sample_output_path = Path(sample_output_csv)
            sample_output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(all_samples, ignore_index=True).to_csv(sample_output_path, index=False)
            print(f"Saved compact per-sample rows: {sample_output_path}")
        else:
            print("[WARN] No per-sample rows available to write sample output.")

    for ds, miss in missing_by_run.items():
        if miss:
            print(f"[INFO] {ds} missing baselines: {miss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fair benchmark and export consolidated leaderboard")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset filters (e.g., AIR-125,AIR-400,Pediatric-RR)",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default=",".join(DEFAULT_BASELINES),
        help="Comma-separated baseline names to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of videos per dataset (0 = all)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/ablation_results",
        help="Directory for ablation outputs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ablation_results/fair_benchmark_leaderboard.csv",
        help="Output CSV path for consolidated leaderboard",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--disable-vlm",
        action="store_true",
        help="Disable live VLM inference during benchmark evaluation",
    )
    parser.add_argument(
        "--sample-output",
        type=str,
        default="",
        help="Optional output CSV path for compact per-sample predictions (for conformal analysis)",
    )
    parser.add_argument(
        "--conformal-calibration-csv",
        type=str,
        default="",
        help="Optional calibration CSV to enable architecture-level Conformal Triage qhat lookup.",
    )
    parser.add_argument(
        "--conformal-default-qhat",
        type=float,
        default=6.0,
        help="Fallback qhat for conformal triage baselines.",
    )
    parser.add_argument(
        "--conformal-condition-on-disagreement",
        action="store_true",
        help="Condition conformal triage qhat lookup on disagreement bins (requires matching calibration CSV).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Optional canonical manifest JSON. When provided, runs all selected baselines on this exact split.",
    )
    parser.add_argument(
        "--manifest-label",
        type=str,
        default="",
        help="Optional label used in exported CSVs when --manifest is provided.",
    )
    parser.add_argument(
        "--pretrained-overrides-json",
        type=str,
        default="",
        help="Optional JSON object or JSON file mapping baseline name to checkpoint path.",
    )
    parser.add_argument(
        "--allow-unstable-baselines",
        action="store_true",
        help=f"Include baselines excluded by default for stability reasons ({', '.join(sorted(UNSTABLE_BASELINES))}).",
    )

    args = parser.parse_args()

    run_fair_benchmark(
        datasets=_parse_csv_list(args.datasets),
        baselines=_parse_csv_list(args.baselines),
        limit=args.limit,
        results_dir=args.results_dir,
        output_csv=args.output,
        seed=args.seed,
        disable_vlm=args.disable_vlm,
        sample_output_csv=args.sample_output,
        conformal_calibration_csv=args.conformal_calibration_csv,
        conformal_default_qhat=args.conformal_default_qhat,
        conformal_condition_on_disagreement=args.conformal_condition_on_disagreement,
        manifest_path=args.manifest,
        manifest_label=args.manifest_label,
        pretrained_overrides=_load_json_mapping(args.pretrained_overrides_json),
        allow_unstable_baselines=args.allow_unstable_baselines,
    )
