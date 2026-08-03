"""Run the expanded paper baseline suite on canonical benchmark splits.

This script is the paper-facing wrapper for:
- the full legacy repo baseline family ("old repo all")
- the official AIR-family baselines (AIRFlowNet/VIRENet, EfficientPhys, MTTS-CAN, DeepPhys)
- focused modern paper subsets

It does not attempt to download or install external modern repos such as
PhysFormer or FactorizePhys. Those remain optional future integrations unless
their code is vendored locally.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.run_ablation_study import AblationOrchestrator
from scripts.run_fair_benchmark import run_fair_benchmark


OFFICIAL_AIR_BASELINES = [
    "AIRFlowNet-Official",
    "DeepPhys-Official",
    "MTTS-CAN-Official",
    "EfficientPhys-Official",
]

MODERN_PAPER_BASELINES = [
    "AIRFlowNet-Official",
    "EfficientPhys-Official",
    "MTTS-CAN-Official",
    "DeepPhys-Official",
    "TSCAN",
    "DeepPhys",
    "Contrastive-Finetune",
    "Transformer-Fusion",
    "Flow-Only",
    "Depth-Only",
    "Fused-VMD",
]

UNAVAILABLE_OPTIONAL_SOTA = {
    "PhysFormer": "PhysFormer is not vendored locally in this workspace.",
    "FactorizePhys": "FactorizePhys is not vendored locally in this workspace.",
}


def _parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _load_json_mapping(text: str) -> Dict[str, str]:
    payload = str(text or "").strip()
    if not payload:
        return {}
    if payload.startswith("{"):
        return json.loads(payload)
    path = Path(payload)
    if not path.exists():
        raise FileNotFoundError(f"JSON mapping not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_command(cmd: Sequence[str]) -> None:
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(list(cmd), check=True, cwd=PROJECT_ROOT)


def _available_baselines(seed: int, output_dir: str) -> List[str]:
    orchestrator = AblationOrchestrator(output_dir=output_dir, seed=seed)
    return [item["name"] for item in orchestrator.baselines]


def _resolve_preset(preset: str, available: Sequence[str]) -> List[str]:
    available_set = set(available)
    if preset == "old_repo_all":
        return [name for name in available if name not in OFFICIAL_AIR_BASELINES]
    if preset == "paper_modern":
        return [name for name in MODERN_PAPER_BASELINES if name in available_set]
    if preset == "official_air_only":
        return [name for name in OFFICIAL_AIR_BASELINES if name in available_set]
    if preset == "paper_full":
        return [name for name in available]
    raise ValueError(f"Unknown preset: {preset}")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_checkpoint_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the expanded paper baseline suite on canonical benchmark splits.")
    parser.add_argument("--train-manifest", default="data/organized/respiratory_benchmark_train.json")
    parser.add_argument("--val-manifest", default="data/organized/respiratory_benchmark_val.json")
    parser.add_argument("--test-manifest", default="data/organized/respiratory_benchmark_test.json")
    parser.add_argument("--output-root", required=True, help="Output root for configs, checkpoints, and reports")
    parser.add_argument(
        "--preset",
        default="paper_full",
        choices=["old_repo_all", "paper_modern", "official_air_only", "paper_full"],
        help="Baseline preset to run",
    )
    parser.add_argument("--baselines", default="", help="Optional explicit comma-separated baseline list")
    parser.add_argument("--exclude-baselines", default="", help="Optional comma-separated baselines to exclude")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Optional per-manifest limit passed to run_fair_benchmark")
    parser.add_argument("--enable-vlm", action="store_true", help="Allow live VLM inference for VLM-tagged legacy baselines")
    parser.add_argument("--skip-existing", action="store_true", help="Skip official AIR training or benchmarking when outputs already exist")
    parser.add_argument("--dry-run", action="store_true", help="Only write selection metadata without running baselines")
    parser.add_argument(
        "--allow-unstable-baselines",
        action="store_true",
        help="Include baselines excluded by default for stability reasons (e.g. TSCAN OOM).",
    )
    parser.add_argument("--pretrained-overrides-json", default="", help="Optional JSON/file mapping baseline name -> checkpoint path")
    parser.add_argument("--skip-official-air-train", action="store_true", help="Do not train official AIR-family baselines before benchmark")
    parser.add_argument("--official-air-train-epochs", type=int, default=15)
    parser.add_argument("--official-air-batch-size", type=int, default=4)
    parser.add_argument("--official-air-lr", type=float, default=1e-3)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    available = _available_baselines(seed=args.seed, output_dir=str(output_root / "inventory"))
    if args.baselines:
        selected = _parse_csv_list(args.baselines)
    else:
        selected = _resolve_preset(args.preset, available)

    excluded = set(_parse_csv_list(args.exclude_baselines))
    selected = [name for name in selected if name not in excluded]

    available_set = set(available)
    missing_requested = [name for name in selected if name not in available_set]
    supported_selected = [name for name in selected if name in available_set]

    unavailable_optional = {name: msg for name, msg in UNAVAILABLE_OPTIONAL_SOTA.items() if name in selected}
    official_selected = [name for name in supported_selected if name in OFFICIAL_AIR_BASELINES]

    selection_payload = {
        "preset": args.preset,
        "seed": int(args.seed),
        "available_count": len(available),
        "selected_count": len(supported_selected),
        "selected_baselines": supported_selected,
        "missing_requested": missing_requested,
        "official_air_selected": official_selected,
        "optional_unavailable": unavailable_optional,
    }
    selection_path = output_root / "expanded_baseline_selection.json"
    _write_json(selection_path, selection_payload)
    print(f"[OK] Wrote selection metadata to {selection_path}")

    if missing_requested:
        print(f"[WARN] Requested baselines not available in this environment: {missing_requested}")
    for name, message in unavailable_optional.items():
        print(f"[INFO] {name}: {message}")

    merged_checkpoint_map = _load_json_mapping(args.pretrained_overrides_json)
    official_root = output_root / "official_air"
    official_checkpoint_map_path = official_root / "official_air_checkpoint_map.json"

    if official_selected and not args.skip_official_air_train:
        cmd = [
            sys.executable,
            "scripts/run_official_air_baselines.py",
            "--train-manifest",
            args.train_manifest,
            "--val-manifest",
            args.val_manifest,
            "--test-manifest",
            args.test_manifest,
            "--output-root",
            str(official_root),
            "--baselines",
            ",".join(official_selected),
            "--train-epochs",
            str(args.official_air_train_epochs),
            "--batch-size",
            str(args.official_air_batch_size),
            "--lr",
            str(args.official_air_lr),
            "--seed",
            str(args.seed),
            "--stages",
            "prepare,train",
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.dry_run:
            print("[DRY RUN]", " ".join(cmd))
        else:
            _run_command(cmd)

    if official_checkpoint_map_path.exists():
        merged_checkpoint_map.update(_load_checkpoint_map(official_checkpoint_map_path))

    merged_checkpoint_map_path = output_root / "benchmark" / "expanded_checkpoint_map.json"
    _write_json(merged_checkpoint_map_path, merged_checkpoint_map)

    benchmark_dir = output_root / "benchmark"
    leaderboard_path = benchmark_dir / "expanded_paper_leaderboard.csv"
    predictions_path = benchmark_dir / "expanded_paper_predictions.csv"

    if args.dry_run:
        print(f"[DRY RUN] Selected {len(supported_selected)} baselines:")
        for name in supported_selected:
            print(f"  - {name}")
        return 0

    if args.skip_existing and leaderboard_path.exists() and predictions_path.exists():
        print(f"[SKIP] expanded paper baseline benchmark -> existing outputs found in {benchmark_dir}")
        print(f"[OK] Leaderboard: {leaderboard_path}")
        print(f"[OK] Predictions: {predictions_path}")
        return 0

    if not supported_selected:
        raise RuntimeError("No supported baselines selected for the expanded paper suite.")

    run_fair_benchmark(
        datasets=[],
        baselines=supported_selected,
        limit=int(args.limit),
        results_dir=str(benchmark_dir),
        output_csv=str(leaderboard_path),
        seed=int(args.seed),
        disable_vlm=not bool(args.enable_vlm),
        sample_output_csv=str(predictions_path),
        manifest_path=args.test_manifest,
        manifest_label="respiratory_benchmark_test",
        pretrained_overrides=merged_checkpoint_map,
        allow_unstable_baselines=bool(args.allow_unstable_baselines),
    )

    print(f"[OK] Expanded paper leaderboard: {leaderboard_path}")
    print(f"[OK] Expanded paper predictions: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
