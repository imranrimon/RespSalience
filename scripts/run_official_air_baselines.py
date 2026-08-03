"""Train official AIR-family baselines on canonical AIR subject splits and benchmark them."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.run_fair_benchmark import run_fair_benchmark


DATASET_NAME_MAP = {
    "AIR-125": "AIR_125",
    "AIR-400": "AIR_400",
}

DEFAULT_BASELINES = [
    "AIRFlowNet-Official",
    "DeepPhys-Official",
    "MTTS-CAN-Official",
    "EfficientPhys-Official",
]

BASELINE_SPECS = {
    "AIRFlowNet-Official": {
        "model_name": "VIRENet",
        "template": PROJECT_ROOT / "external" / "air-400" / "configs" / "body_roi" / "virenet_6_fold_cv_air400_with_roi_coarse2fine_of_body.yaml",
    },
    "DeepPhys-Official": {
        "model_name": "DeepPhys",
        "template": PROJECT_ROOT / "external" / "air-400" / "configs" / "body_roi" / "deepphys_6_fold_cv_air400_with_roi_coarse2fine_of_body.yaml",
    },
    "MTTS-CAN-Official": {
        "model_name": "TSCAN",
        "template": PROJECT_ROOT / "external" / "air-400" / "configs" / "body_roi" / "tscan_6_fold_cv_air400_with_roi_coarse2fine_of_body.yaml",
    },
    "EfficientPhys-Official": {
        "model_name": "EfficientPhys",
        "template": PROJECT_ROOT / "external" / "air-400" / "configs" / "body_roi" / "efficientphys_6_fold_cv_air400_with_roi_coarse2fine_of_body.yaml",
    },
}


def _parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _load_manifest(path: str | Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _air_subjects_from_manifest(entries: Iterable[Dict]) -> Dict[str, List[str]]:
    subjects: Dict[str, set[str]] = {dataset_name: set() for dataset_name in DATASET_NAME_MAP.values()}
    for entry in entries:
        dataset_name = DATASET_NAME_MAP.get(str(entry.get("dataset", "")))
        if not dataset_name:
            continue
        subject_id = str(entry.get("subject_id", "")).strip()
        if subject_id:
            subjects[dataset_name].add(subject_id)
    return {dataset_name: sorted(values) for dataset_name, values in subjects.items() if values}


def _split_subjects(train_manifest: str, val_manifest: str, test_manifest: str) -> Dict[str, Dict[str, List[str]]]:
    split_map = {
        "TRAIN": _air_subjects_from_manifest(_load_manifest(train_manifest)),
        "VALID": _air_subjects_from_manifest(_load_manifest(val_manifest)),
        "TEST": _air_subjects_from_manifest(_load_manifest(test_manifest)),
    }
    if not any(split_map["TRAIN"].values()):
        raise ValueError("Canonical train manifest does not contain AIR-125 or AIR-400 subjects.")
    if not any(split_map["VALID"].values()):
        raise ValueError("Canonical val manifest does not contain AIR-125 or AIR-400 subjects.")
    if not any(split_map["TEST"].values()):
        raise ValueError("Canonical test manifest does not contain AIR-125 or AIR-400 subjects.")
    return split_map


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _load_template(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _split_entries(
    split_subjects: Dict[str, List[str]],
    data_format: str,
    preprocess: Dict,
) -> List[Dict]:
    entries = []
    for dataset_name in ["AIR_400", "AIR_125"]:
        subjects = split_subjects.get(dataset_name, [])
        if not subjects:
            continue
        entries.append(
            {
                "DATASET": dataset_name,
                "SPLIT_TYPE": "custom",
                "SPLIT_SUBJECTS": list(subjects),
                "SPLIT_RATIO": 0.0,
                "DATA_FORMAT": data_format,
                "PREPROCESS": deepcopy(preprocess),
            }
        )
    return entries


def _expected_checkpoint(config: Dict) -> Path:
    model_name = str(config["MODEL"]["NAME"])
    output_dir = Path(str(config["DATA_PATH"]["OUTPUT_DIR"]))
    return output_dir / "model" / model_name / str(config["NAME"]) / f"{model_name}_best.pth"


def _build_config(
    baseline_name: str,
    spec: Dict,
    split_map: Dict[str, Dict[str, List[str]]],
    output_root: Path,
    train_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Dict:
    config = _load_template(Path(spec["template"]))
    config_name = f"{_slugify(baseline_name)}_canonical_air_subject_split"
    config["NAME"] = config_name
    config["SEED"] = int(seed)

    if "CV" in config:
        config["CV"]["ENABLE_CV"] = False

    config.setdefault("LOGGING", {})
    config["LOGGING"]["USE_WANDB"] = False
    config.setdefault("SYSTEM", {})
    config["SYSTEM"]["NUM_WORKERS"] = 0 if os.name == "nt" else int(config["SYSTEM"].get("NUM_WORKERS", 10))
    config["SYSTEM"]["PIN_MEMORY"] = bool(config["SYSTEM"].get("PIN_MEMORY", True))

    config["TRAIN"]["BATCH_SIZE"] = int(batch_size)
    config["TRAIN"]["EPOCHES"] = int(train_epochs)
    config["TRAIN"]["LR"] = float(lr)
    config.setdefault("INFERENCE", {})
    config["INFERENCE"]["BATCH_SIZE"] = int(max(1, min(batch_size, 8)))

    train_template = config["TRAIN"]["DATA"][0] if isinstance(config["TRAIN"]["DATA"], list) else config["TRAIN"]["DATA"]
    common_preprocess = deepcopy(train_template["PREPROCESS"])
    common_data_format = str(train_template["DATA_FORMAT"])

    config["TRAIN"]["DATA"] = _split_entries(split_map["TRAIN"], common_data_format, common_preprocess)
    config["VALID"]["DATA"] = _split_entries(split_map["VALID"], common_data_format, common_preprocess)
    config["TEST"]["DATA"] = _split_entries(split_map["TEST"], common_data_format, common_preprocess)

    data_root = PROJECT_ROOT / "data"
    model_output_dir = output_root / "official_outputs" / _slugify(baseline_name)
    cache_dir = output_root / "official_cache"
    detector_path = data_root / "yolov8n.pt"

    config["DATA_PATH"]["AIR_125"] = str((data_root / "AIR_125").resolve())
    config["DATA_PATH"]["AIR_400"] = str((data_root / "AIR_400").resolve())
    config["DATA_PATH"]["COHFACE"] = str((data_root / "COHFACE").resolve())
    config["DATA_PATH"]["CACHE_DIR"] = str(cache_dir.resolve())
    config["DATA_PATH"]["OUTPUT_DIR"] = str(model_output_dir.resolve())
    config["DATA_PATH"]["BODY_DETECTOR_PATH"] = str(detector_path.resolve())
    config["DATA_PATH"]["FACE_DETECTOR_PATH"] = str(detector_path.resolve())

    return config


def _write_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _run_training(config_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_SILENT", "true")
    temp_root = (PROJECT_ROOT / "outputs" / "runtime_tmp").resolve()
    wandb_root = (PROJECT_ROOT / "outputs" / "wandb_tmp").resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    wandb_root.mkdir(parents=True, exist_ok=True)
    env.setdefault("TMP", str(temp_root))
    env.setdefault("TEMP", str(temp_root))
    env.setdefault("TMPDIR", str(temp_root))
    env.setdefault("WANDB_DIR", str(wandb_root))
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "external" / "air-400" / "main.py").resolve()),
        "--config",
        str(config_path.resolve()),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)


def _write_training_summary(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train official AIR-family baselines on canonical AIR subject splits and benchmark them.")
    parser.add_argument("--train-manifest", default="data/organized/respiratory_benchmark_train.json")
    parser.add_argument("--val-manifest", default="data/organized/respiratory_benchmark_val.json")
    parser.add_argument("--test-manifest", default="data/organized/respiratory_benchmark_test.json")
    parser.add_argument("--output-root", default="outputs/official_air_baselines")
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES), help="Comma-separated official baseline names")
    parser.add_argument("--train-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", default="prepare,train,benchmark", help="Comma-separated stages: prepare,train,benchmark")
    parser.add_argument("--skip-existing", action="store_true", help="Skip training/benchmarking when expected outputs already exist")
    args = parser.parse_args()

    stages = {item.strip().lower() for item in args.stages.split(",") if item.strip()}
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    split_map = _split_subjects(args.train_manifest, args.val_manifest, args.test_manifest)
    baselines = _parse_csv_list(args.baselines)
    unknown = [name for name in baselines if name not in BASELINE_SPECS]
    if unknown:
        raise ValueError(f"Unsupported official baseline(s): {unknown}")

    config_dir = output_root / "configs"
    summary_rows = []
    checkpoint_map: Dict[str, str] = {}

    for baseline_name in baselines:
        spec = BASELINE_SPECS[baseline_name]
        config = _build_config(
            baseline_name=baseline_name,
            spec=spec,
            split_map=split_map,
            output_root=output_root,
            train_epochs=args.train_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
        config_path = config_dir / f"{_slugify(baseline_name)}.yaml"
        _write_yaml(config_path, config)

        checkpoint_path = _expected_checkpoint(config)
        summary_rows.append(
            {
                "baseline": baseline_name,
                "model_name": spec["model_name"],
                "seed": int(args.seed),
                "config_path": str(config_path),
                "checkpoint_path": str(checkpoint_path),
                "train_subjects": split_map["TRAIN"],
                "val_subjects": split_map["VALID"],
                "test_subjects": split_map["TEST"],
            }
        )

        if "train" in stages:
            if args.skip_existing and checkpoint_path.exists():
                print(f"[SKIP] {baseline_name} training -> existing checkpoint at {checkpoint_path}")
            else:
                _run_training(config_path)

        if checkpoint_path.exists():
            checkpoint_map[baseline_name] = str(checkpoint_path)

    summary_path = output_root / "official_air_training_summary.json"
    checkpoint_map_path = output_root / "official_air_checkpoint_map.json"
    _write_training_summary(summary_path, summary_rows)
    _write_training_summary(checkpoint_map_path, checkpoint_map)
    print(f"[OK] Wrote summary to {summary_path}")
    print(f"[OK] Wrote checkpoint map to {checkpoint_map_path}")

    if "benchmark" in stages:
        if not checkpoint_map:
            raise RuntimeError("No official AIR checkpoints found; cannot run benchmark stage.")

        benchmark_dir = output_root / "benchmark"
        leaderboard_path = benchmark_dir / "official_air_leaderboard.csv"
        predictions_path = benchmark_dir / "official_air_predictions.csv"
        if args.skip_existing and leaderboard_path.exists() and predictions_path.exists():
            print(f"[SKIP] official AIR benchmark -> existing outputs found in {benchmark_dir}")
        else:
            run_fair_benchmark(
                datasets=[],
                baselines=baselines,
                limit=0,
                results_dir=str(benchmark_dir),
                output_csv=str(leaderboard_path),
                seed=args.seed,
                disable_vlm=True,
                sample_output_csv=str(predictions_path),
                manifest_path=args.test_manifest,
                manifest_label="respiratory_benchmark_test",
                pretrained_overrides=checkpoint_map,
            )
        print(f"[OK] Official AIR leaderboard: {leaderboard_path}")
        print(f"[OK] Official AIR predictions: {predictions_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
