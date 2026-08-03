"""End-to-end runner for PRISM-Breath pretraining, ablations, and conformal evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_VARIANTS = [
    "prism_breath",
    "prism_shared_trunk",
    "prism_no_scan",
    "prism_no_spectral",
    "prism_no_codebook",
    "prism_no_prototypes",
    "airflow_compact",
]


def run_command(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def should_skip(skip_existing: bool, *paths: Path) -> bool:
    if not skip_existing:
        return False
    return all(path.exists() for path in paths)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full PRISM-Breath study pipeline.")
    parser.add_argument("--signal-dir", default="data/processed_signals")
    parser.add_argument("--train-manifest", default="data/organized/respiratory_benchmark_train.json")
    parser.add_argument("--val-manifest", default="data/organized/respiratory_benchmark_val.json")
    parser.add_argument("--test-manifest", default="data/organized/respiratory_benchmark_test.json")
    parser.add_argument("--output-root", default="outputs/prism_breath_study")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--stages", default="pretrain,train,validate,calibrate,test")
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--disagreement-bins", default="0.8,0.9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true", help="Skip stages whose expected output files already exist")
    args = parser.parse_args()

    stages = {item.strip().lower() for item in args.stages.split(",") if item.strip()}
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    pretrain_dir = output_root / "pretraining"
    pretrain_ckpt = pretrain_dir / "prism_breath_pretrained.pt"

    pretrain_history = pretrain_dir / "pretraining_history.json"
    if "pretrain" in stages:
        if should_skip(args.skip_existing, pretrain_ckpt, pretrain_history):
            print(f"[SKIP] pretrain -> existing outputs found at {pretrain_dir}")
        else:
            run_command(
                [
                    python,
                    "scripts/pretrain_prism_breath.py",
                    "--signal-dir",
                    args.signal_dir,
                    "--train-manifest",
                    args.train_manifest,
                    "--val-manifest",
                    args.val_manifest,
                    "--output-dir",
                    str(pretrain_dir),
                    "--model-variant",
                    "prism_breath",
                    "--seq-len",
                    str(args.seq_len),
                    "--d-model",
                    str(args.d_model),
                    "--batch-size",
                    str(args.batch_size),
                    "--epochs",
                    str(args.pretrain_epochs),
                    "--seed",
                    str(args.seed),
                ]
            )

    summaries = []
    for variant in args.variants:
        variant_dir = output_root / variant
        train_dir = variant_dir / "train"
        val_dir = variant_dir / "val_eval"
        test_dir = variant_dir / "test_eval"
        conformal_dir = variant_dir / "conformal"
        checkpoint_path = train_dir / "prism_breath_best.pt"
        training_history = train_dir / "training_history.json"
        val_summary = val_dir / "prism_breath_summary.json"
        val_predictions = val_dir / "prism_breath_predictions.csv"
        calibration_csv = conformal_dir / "prism_qhat.csv"
        test_summary = test_dir / "prism_breath_summary.json"
        test_predictions = test_dir / "prism_breath_predictions.csv"

        if "train" in stages:
            if should_skip(args.skip_existing, checkpoint_path, training_history):
                print(f"[SKIP] train {variant} -> existing outputs found at {train_dir}")
            else:
                cmd = [
                    python,
                    "scripts/train_prism_breath.py",
                    "--signal-dir",
                    args.signal_dir,
                    "--train-manifest",
                    args.train_manifest,
                    "--val-manifest",
                    args.val_manifest,
                    "--output-dir",
                    str(train_dir),
                    "--model-variant",
                    variant,
                    "--seq-len",
                    str(args.seq_len),
                    "--d-model",
                    str(args.d_model),
                    "--batch-size",
                    str(args.batch_size),
                    "--epochs",
                    str(args.train_epochs),
                    "--seed",
                    str(args.seed),
                ]
                if pretrain_ckpt.exists() and variant != "airflow_compact":
                    cmd.extend(["--encoder-pretrained", str(pretrain_ckpt)])
                    cmd.extend(["--freeze-encoder-epochs", "2"])
                run_command(cmd)

        if "validate" in stages:
            if should_skip(args.skip_existing, val_summary, val_predictions):
                print(f"[SKIP] validate {variant} -> existing outputs found at {val_dir}")
            else:
                run_command(
                    [
                        python,
                        "scripts/evaluate_prism_breath.py",
                        "--signal-dir",
                        args.signal_dir,
                        "--manifest",
                        args.val_manifest,
                        "--checkpoint",
                        str(checkpoint_path),
                        "--output-dir",
                        str(val_dir),
                        "--batch-size",
                        str(args.batch_size),
                        "--disagreement-bins",
                        str(args.disagreement_bins),
                    ]
                )

        if "calibrate" in stages:
            if should_skip(args.skip_existing, calibration_csv):
                print(f"[SKIP] calibrate {variant} -> existing output found at {calibration_csv}")
            else:
                run_command(
                    [
                        python,
                        "scripts/calibrate_conformal.py",
                        "--samples",
                        str(val_dir / "prism_breath_predictions.csv"),
                        "--output",
                        str(calibration_csv),
                        "--use-all-samples",
                        "--group-by-source",
                        "--condition-on-disagreement",
                        "--split-unit",
                        "subject",
                        "--disagreement-bins",
                        str(args.disagreement_bins),
                    ]
                )

        if "test" in stages:
            if should_skip(args.skip_existing, test_summary, test_predictions):
                print(f"[SKIP] test {variant} -> existing outputs found at {test_dir}")
            else:
                cmd = [
                    python,
                    "scripts/evaluate_prism_breath.py",
                    "--signal-dir",
                    args.signal_dir,
                    "--manifest",
                    args.test_manifest,
                    "--checkpoint",
                    str(checkpoint_path),
                    "--output-dir",
                    str(test_dir),
                    "--batch-size",
                    str(args.batch_size),
                ]
                if calibration_csv.exists():
                    cmd.extend(
                        [
                            "--calibration-csv",
                            str(calibration_csv),
                            "--condition-on-disagreement",
                            "--disagreement-bins",
                            str(args.disagreement_bins),
                        ]
                    )
                run_command(cmd)

            summary_path = test_dir / "prism_breath_summary.json"
            if summary_path.exists():
                row = {"variant": variant, **load_json(summary_path)}
                summaries.append(row)

    if summaries:
        summary_csv = output_root / "ablation_summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"[OK] Saved ablation summary to {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
