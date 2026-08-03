"""Train and evaluate legacy strong baselines on canonical benchmark splits.

This script aligns the older baseline families with the paper-ready PRISM-Breath
protocol. It optionally retrains learned legacy models on the canonical
train/val manifests, then evaluates all requested baselines on the canonical
test manifest through the same benchmark runner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.run_fair_benchmark import run_fair_benchmark
from scripts.train_finetune import train_finetune
from scripts.train_ssl_benchmark import train_ssl_benchmark


DEFAULT_BASELINES = [
    "Flow-Only",
    "Depth-Only",
    "Fused-VMD",
    "Contrastive-Finetune",
    "Transformer-Fusion",
    "DeepPhys",
]


def _parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _require_path(path: str | Path, label: str) -> str:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return str(resolved)


def _should_train(baselines: List[str], name: str) -> bool:
    return any(baseline == name for baseline in baselines)


def run_legacy_canonical_benchmark(
    signal_dir: str,
    train_manifest: str,
    val_manifest: str,
    test_manifest: str,
    output_root: str,
    baselines: List[str],
    ssl_epochs: int,
    finetune_epochs: int,
    batch_size: int,
    ssl_lr: float,
    finetune_lr: float,
    weight_decay: float,
    seq_len: int,
    d_model: int,
    seed: int,
    disable_vlm: bool,
    force_retrain: bool,
    skip_existing: bool,
):
    signal_dir = _require_path(signal_dir, "Signal directory")
    train_manifest = _require_path(train_manifest, "Train manifest")
    val_manifest = _require_path(val_manifest, "Val manifest")
    test_manifest = _require_path(test_manifest, "Test manifest")

    out_root = Path(output_root)
    pretrain_dir = out_root / "pretraining"
    model_dir = out_root / "models"
    benchmark_dir = out_root / "benchmark"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_map: Dict[str, str] = {}

    contrast_ssl_ckpt = pretrain_dir / "legacy_contrast_ssl.pt"
    contrast_ft_ckpt = model_dir / "legacy_contrast_finetune_best.pth"
    transformer_ckpt = model_dir / "legacy_transformer_fusion_best.pth"

    if _should_train(baselines, "Contrastive-Finetune"):
        if force_retrain or not contrast_ssl_ckpt.exists():
            train_ssl_benchmark(
                "contrast",
                epochs=ssl_epochs,
                batch_size=batch_size,
                lr=ssl_lr,
                data_dir=signal_dir,
                seed=seed,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                output_path=str(contrast_ssl_ckpt),
                seq_len=seq_len,
                d_model=d_model,
            )
        if force_retrain or not contrast_ft_ckpt.exists():
            train_finetune(
                data_dir=signal_dir,
                pretrained_path=str(contrast_ssl_ckpt),
                output_path=str(contrast_ft_ckpt),
                epochs=finetune_epochs,
                batch_size=batch_size,
                lr=finetune_lr,
                seed=seed,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                seq_len=seq_len,
                d_model=d_model,
                weight_decay=weight_decay,
            )
        checkpoint_map["Contrastive-Finetune"] = str(contrast_ft_ckpt)

    if _should_train(baselines, "Transformer-Fusion"):
        if force_retrain or not transformer_ckpt.exists():
            train_finetune(
                data_dir=signal_dir,
                pretrained_path="",
                output_path=str(transformer_ckpt),
                epochs=finetune_epochs,
                batch_size=batch_size,
                lr=finetune_lr,
                seed=seed,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                seq_len=seq_len,
                d_model=d_model,
                weight_decay=weight_decay,
            )
        checkpoint_map["Transformer-Fusion"] = str(transformer_ckpt)

    checkpoint_map_path = benchmark_dir / "legacy_checkpoint_map.json"
    with open(checkpoint_map_path, "w", encoding="utf-8") as handle:
        json.dump(checkpoint_map, handle, indent=2)

    leaderboard_path = benchmark_dir / "legacy_canonical_leaderboard.csv"
    sample_output_path = benchmark_dir / "legacy_canonical_predictions.csv"
    if skip_existing and leaderboard_path.exists() and sample_output_path.exists():
        print(f"[SKIP] legacy benchmark -> existing outputs found in {benchmark_dir}")
    else:
        run_fair_benchmark(
            datasets=[],
            baselines=baselines,
            limit=0,
            results_dir=str(benchmark_dir),
            output_csv=str(leaderboard_path),
            seed=seed,
            disable_vlm=disable_vlm,
            sample_output_csv=str(sample_output_path),
            manifest_path=test_manifest,
            manifest_label="respiratory_benchmark_test",
            pretrained_overrides=checkpoint_map,
        )

    print("\nLegacy canonical benchmark complete")
    print(f"  Leaderboard: {leaderboard_path}")
    print(f"  Predictions: {sample_output_path}")
    print(f"  Checkpoints: {checkpoint_map_path}")


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate legacy strong baselines on canonical benchmark splits.")
    parser.add_argument("--signal-dir", required=True, help="Directory containing canonical extracted benchmark signals")
    parser.add_argument("--train-manifest", required=True, help="Canonical train manifest JSON")
    parser.add_argument("--val-manifest", required=True, help="Canonical val manifest JSON")
    parser.add_argument("--test-manifest", required=True, help="Canonical test manifest JSON")
    parser.add_argument("--output-root", required=True, help="Root output directory for checkpoints and reports")
    parser.add_argument(
        "--baselines",
        type=str,
        default=",".join(DEFAULT_BASELINES),
        help="Comma-separated baseline names to include",
    )
    parser.add_argument("--ssl-epochs", type=int, default=50, help="Epochs for contrastive pretraining")
    parser.add_argument("--finetune-epochs", type=int, default=50, help="Epochs for supervised fine-tuning")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for learned legacy baselines")
    parser.add_argument("--ssl-lr", type=float, default=1e-3, help="Learning rate for SSL pretraining")
    parser.add_argument("--finetune-lr", type=float, default=3e-4, help="Learning rate for supervised fine-tuning")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for supervised fine-tuning")
    parser.add_argument("--seq-len", type=int, default=300, help="Signal sequence length")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer hidden size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--disable-vlm", action="store_true", help="Disable live VLM inference during baseline evaluation")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain learned baselines even if checkpoints already exist")
    parser.add_argument("--skip-existing", action="store_true", help="Skip benchmark evaluation if leaderboard outputs already exist")
    args = parser.parse_args()

    run_legacy_canonical_benchmark(
        signal_dir=args.signal_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        test_manifest=args.test_manifest,
        output_root=args.output_root,
        baselines=_parse_csv_list(args.baselines),
        ssl_epochs=args.ssl_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        ssl_lr=args.ssl_lr,
        finetune_lr=args.finetune_lr,
        weight_decay=args.weight_decay,
        seq_len=args.seq_len,
        d_model=args.d_model,
        seed=args.seed,
        disable_vlm=args.disable_vlm,
        force_retrain=args.force_retrain,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
