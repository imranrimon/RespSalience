"""
Camera-ready experiment pipeline.

Stages:
1) extract_signals
2) ssl_pretrain
3) ssl_finetune
4) semi_train
5) benchmark_core
6) benchmark_modality
7) conformal
8) tables
9) package

This orchestrates training + baselines + modality ablations across datasets.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _prepend_env_runtime_paths(env: Dict[str, str], python_executable: str) -> Dict[str, str]:
    if os.name != "nt":
        return env

    python_path = Path(python_executable).resolve()
    env_root = python_path.parent
    path_parts = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
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
        if candidate.exists() and candidate_str.lower() not in existing:
            prepended.append(candidate_str)
            existing.add(candidate_str.lower())

    if prepended:
        env["PATH"] = os.pathsep.join(prepended + path_parts)

    env.setdefault("CONDA_PREFIX", str(env_root))
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return env

DEFAULT_DATASETS = ["AIR-125", "AIR-400", "Pediatric-RR"]

DEFAULT_CORE_BASELINES = [
    "Flow-Only",
    "Depth-Only",
    "Fused-SNR",
    "Fused-VMD",
    "MSM-Finetune",
    "SAE-Finetune",
    "Contrastive-Finetune",
    "Transformer-Fusion",
    "Tri-Modal VLM-Gated",
    "Quad-Modal Full (Ours)",
    "Quad-Modal Full + Robust Fusion",
    "Quad-Modal Full + Robust Fusion + Conformal Triage",
    "DeepPhys",
    "TSCAN",
]

DEFAULT_MODALITY_BASELINES = [
    "Tri-Modal VLM-Gated",
    "Quad-Modal Full (Ours)",
    "Quad-Modal Full + Robust Fusion",
    "Quad-Modal Full + Robust Fusion + Conformal Triage",
    "Quad-Modal-Full (Drop Audio)",
    "Quad-Modal-Full (Drop Flow)",
    "Quad-Modal-Full (Drop Depth)",
]

ALL_STAGES = [
    "extract_signals",
    "ssl_pretrain",
    "ssl_finetune",
    "semi_train",
    "benchmark_core",
    "benchmark_modality",
    "conformal",
    "tables",
    "package",
]


def _csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _count_npz(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.npz")))


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_from_filename(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return -1


class CameraReadyPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.python = sys.executable
        self.datasets = _csv_list(args.datasets)
        self.core_baselines = _csv_list(args.core_baselines)
        self.modality_baselines = _csv_list(args.modality_baselines)
        self.benchmark_seeds = [int(x) for x in _csv_list(args.benchmark_seeds)]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name if args.run_name else f"camera_ready_{timestamp}"
        preferred_root = Path(args.output_root)
        self.run_dir = preferred_root / run_name
        if not self._mkdir_safe(self.run_dir):
            fallback_root = PROJECT_ROOT / "outputs" / "camera_ready_runs"
            print(
                f"[WARN] Could not create output root '{preferred_root}'. "
                f"Falling back to '{fallback_root}'."
            )
            self.run_dir = fallback_root / run_name
            if not self._mkdir_safe(self.run_dir):
                raise PermissionError(
                    f"Unable to create run directory at '{preferred_root}' or fallback '{fallback_root}'."
                )

        self.logs_dir = self.run_dir / "logs"
        self.leaderboards_dir = self.run_dir / "leaderboards"
        self.tables_dir = self.run_dir / "tables"
        self.ablation_cache_dir = self.run_dir / "ablation_cache"

        for d in [self.run_dir, self.logs_dir, self.leaderboards_dir, self.tables_dir, self.ablation_cache_dir]:
            if not self._mkdir_safe(d):
                raise PermissionError(f"Unable to create directory: {d}")

        self.manifest_path = self.run_dir / "run_manifest.json"
        self.summary_path = self.run_dir / "run_summary.md"

        self.manifest: Dict = {
            "run_name": run_name,
            "created_utc": _now_utc(),
            "datasets": self.datasets,
            "core_baselines": self.core_baselines,
            "modality_baselines": self.modality_baselines,
            "benchmark_seeds": self.benchmark_seeds,
            "stages_requested": self._resolve_stages(),
            "args": vars(args),
            "steps": [],
        }

    @staticmethod
    def _mkdir_safe(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    def _resolve_stages(self) -> List[str]:
        stages = _csv_list(self.args.stages)
        if len(stages) == 1 and stages[0].lower() == "all":
            return ALL_STAGES
        invalid = [s for s in stages if s not in ALL_STAGES]
        if invalid:
            raise ValueError(f"Invalid stage(s): {invalid}. Valid: {ALL_STAGES}")
        return stages

    def _write_manifest(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def _run_cmd(self, step_name: str, cmd: List[str], allow_fail: bool = False) -> bool:
        log_path = self.logs_dir / f"{step_name}.log"
        record = {
            "step": step_name,
            "command": cmd,
            "started_utc": _now_utc(),
            "log": str(log_path),
            "status": "running",
        }
        self.manifest["steps"].append(record)
        self._write_manifest()

        print(f"\n{'=' * 88}")
        print(f"[STEP] {step_name}")
        print(f"[CMD ] {' '.join(cmd)}")
        print(f"{'=' * 88}")

        if self.args.dry_run:
            record["status"] = "dry_run"
            record["finished_utc"] = _now_utc()
            self._write_manifest()
            return True

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(self.args.seed)
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env = _prepend_env_runtime_paths(env, self.python)
        yolo_cfg_dir = self.run_dir / "ultralytics_cfg"
        yolo_cfg_dir.mkdir(parents=True, exist_ok=True)
        env["YOLO_CONFIG_DIR"] = str(yolo_cfg_dir)

        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"$ {' '.join(cmd)}\n\n")
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                lf.write(line)
            return_code = proc.wait()

        record["finished_utc"] = _now_utc()
        record["return_code"] = return_code
        record["status"] = "ok" if return_code == 0 else "failed"
        self._write_manifest()

        if return_code != 0 and not allow_fail:
            raise RuntimeError(f"Step failed: {step_name} (rc={return_code})")
        return return_code == 0

    def _maybe_skip_file(self, output_path: Path) -> bool:
        return self.args.resume and output_path.exists()

    def stage_extract_signals(self):
        processed = PROJECT_ROOT / "data" / "processed_signals"
        if self.args.resume and _count_npz(processed) >= self.args.min_processed_signals:
            print(f"[SKIP] extract_signals: found {_count_npz(processed)} processed signals.")
        else:
            self._run_cmd(
                "extract_processed_signals",
                [self.python, "scripts/preprocess_dataset.py"],
            )

        signal_dir = PROJECT_ROOT / "data" / "ablation_results" / "signals"
        if self.args.resume and _count_npz(signal_dir) >= self.args.min_ablation_signals:
            print(f"[SKIP] extract_vlm_signals: found {_count_npz(signal_dir)} ablation signals.")
            return

        for dataset in self.datasets:
            cmd = [
                self.python,
                "scripts/run_ablation_study.py",
                "--dataset",
                dataset,
                "--save-signals",
                "--baseline",
                self.args.signal_baseline,
                "--seed",
                str(self.args.seed),
            ]
            if self.args.limit > 0:
                cmd += ["--limit", str(self.args.limit)]
            self._run_cmd(f"extract_vlm_signals_{dataset}", cmd)

    def stage_ssl_pretrain(self):
        methods = ["msm", "sae", "contrast"]
        for method in methods:
            out = PROJECT_ROOT / "data" / "models" / f"ssl_benchmark_{method}.pth"
            if self._maybe_skip_file(out):
                print(f"[SKIP] ssl_pretrain_{method}: {out} exists.")
                continue
            self._run_cmd(
                f"ssl_pretrain_{method}",
                [
                    self.python,
                    "scripts/train_ssl_benchmark.py",
                    "--method",
                    method,
                    "--epochs",
                    str(self.args.ssl_epochs),
                    "--data_dir",
                    str(self.args.ssl_data_dir),
                    "--seed",
                    str(self.args.seed),
                ],
            )

    def stage_ssl_finetune(self):
        configs = [
            ("msm", "data/models/ssl_benchmark_msm.pth", "data/models/finetuned_msm.pth"),
            ("sae", "data/models/ssl_benchmark_sae.pth", "data/models/finetuned_sae.pth"),
            ("contrast", "data/models/ssl_benchmark_contrast.pth", "data/models/finetuned_contrast.pth"),
        ]
        for name, pretrained, output in configs:
            pretrained_path = PROJECT_ROOT / pretrained
            output_path = PROJECT_ROOT / output
            if self._maybe_skip_file(output_path):
                print(f"[SKIP] ssl_finetune_{name}: {output_path} exists.")
                continue
            if not pretrained_path.exists():
                print(f"[WARN] ssl_finetune_{name}: missing pretrained {pretrained_path}, skipping.")
                continue
            self._run_cmd(
                f"ssl_finetune_{name}",
                [
                    self.python,
                    "scripts/train_finetune.py",
                    "--pretrained",
                    pretrained,
                    "--output",
                    output,
                    "--epochs",
                    str(self.args.finetune_epochs),
                    "--data_dir",
                    str(self.args.finetune_data_dir),
                    "--seed",
                    str(self.args.seed),
                ],
            )

    def stage_semi_train(self):
        pretrain_path = PROJECT_ROOT / "data" / "models" / "spmat_bt.pth"
        finetune_path = PROJECT_ROOT / "data" / "models" / "vlm_gated_fused.pth"

        if not self._maybe_skip_file(pretrain_path):
            self._run_cmd(
                "semi_pretrain_spmat_bt",
                [
                    self.python,
                    "scripts/train_vlm_fusion.py",
                    "--stage",
                    "pretrain",
                    "--data_dir",
                    str(self.args.vlm_data_dir),
                    "--epochs",
                    str(self.args.vlm_pretrain_epochs),
                    "--seed",
                    str(self.args.seed),
                ],
            )
        else:
            print(f"[SKIP] semi_pretrain_spmat_bt: {pretrain_path} exists.")

        if not self._maybe_skip_file(finetune_path):
            self._run_cmd(
                "semi_finetune_vlm_gated",
                [
                    self.python,
                    "scripts/train_vlm_fusion.py",
                    "--stage",
                    "finetune",
                    "--data_dir",
                    str(self.args.vlm_data_dir),
                    "--ft_epochs",
                    str(self.args.vlm_finetune_epochs),
                    "--seed",
                    str(self.args.seed),
                ],
            )
        else:
            print(f"[SKIP] semi_finetune_vlm_gated: {finetune_path} exists.")

    def _aggregate_seed_csvs(self, csv_paths: List[Path], output_path: Path):
        frames = []
        for p in csv_paths:
            if not p.exists():
                continue
            df = pd.read_csv(p)
            df["seed"] = _seed_from_filename(p)
            frames.append(df)
        if not frames:
            return

        all_df = pd.concat(frames, ignore_index=True)
        metric_cols = [c for c in [
            "MAE_overall", "RMSE_overall", "Pearson_r_overall", "IoA2_overall", "IoA3_overall"
        ] if c in all_df.columns]
        if not metric_cols:
            return

        agg = all_df.groupby(["name", "paradigm"], as_index=False)[metric_cols].agg(["mean", "std"])
        agg.columns = [
            "_".join([x for x in col if x]).strip("_") if isinstance(col, tuple) else str(col)
            for col in agg.columns
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(output_path, index=False)

    def _aggregate_seed_csvs_generic(self, csv_paths: List[Path], output_path: Path, group_cols: List[str]):
        frames = []
        for p in csv_paths:
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if df.empty:
                continue
            df["seed"] = _seed_from_filename(p)
            frames.append(df)
        if not frames:
            return

        all_df = pd.concat(frames, ignore_index=True)
        group_cols = [c for c in group_cols if c in all_df.columns]
        if not group_cols:
            print(f"[WARN] No valid group columns found for aggregate: {output_path}")
            return

        numeric_cols = []
        for c in all_df.columns:
            if c in group_cols or c == "seed":
                continue
            if pd.api.types.is_numeric_dtype(all_df[c]):
                numeric_cols.append(c)

        if not numeric_cols:
            return

        agg = all_df.groupby(group_cols, as_index=False)[numeric_cols].agg(["mean", "std"])
        agg.columns = [
            "_".join([x for x in col if x]).strip("_") if isinstance(col, tuple) else str(col)
            for col in agg.columns
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(output_path, index=False)

    def _run_benchmark(self, stage_prefix: str, baselines: List[str], output_prefix: str):
        outputs = []
        for seed in self.benchmark_seeds:
            out_csv = self.leaderboards_dir / f"{output_prefix}_seed{seed}.csv"
            sample_csv = self.leaderboards_dir / f"{output_prefix}_samples_seed{seed}.csv"
            if self.args.resume and out_csv.exists() and sample_csv.exists():
                print(f"[SKIP] {stage_prefix}_seed{seed}: {out_csv} exists.")
                outputs.append(out_csv)
                continue

            cmd = [
                self.python,
                "scripts/run_fair_benchmark.py",
                "--datasets",
                ",".join(self.datasets),
                "--baselines",
                ",".join(baselines),
                "--results-dir",
                str(self.ablation_cache_dir),
                "--output",
                str(out_csv),
                "--seed",
                str(seed),
                "--sample-output",
                str(sample_csv),
            ]
            if not self.args.enable_vlm_benchmark:
                cmd.append("--disable-vlm")
            if self.args.limit > 0:
                cmd += ["--limit", str(self.args.limit)]
            if self.args.benchmark_conformal_calibration_csv.strip():
                cmd += [
                    "--conformal-calibration-csv",
                    str(self.args.benchmark_conformal_calibration_csv),
                    "--conformal-default-qhat",
                    str(self.args.benchmark_conformal_default_qhat),
                ]
                if self.args.benchmark_conformal_condition_on_disagreement:
                    cmd.append("--conformal-condition-on-disagreement")

            self._run_cmd(f"{stage_prefix}_seed{seed}", cmd)
            outputs.append(out_csv)

        if len(outputs) > 1:
            agg_out = self.leaderboards_dir / f"{output_prefix}_seed_aggregate.csv"
            self._aggregate_seed_csvs(outputs, agg_out)
            print(f"[INFO] Seed aggregate saved: {agg_out}")

    def stage_benchmark_core(self):
        self._run_benchmark(
            stage_prefix="benchmark_core",
            baselines=self.core_baselines,
            output_prefix="core_leaderboard",
        )

    def stage_benchmark_modality(self):
        self._run_benchmark(
            stage_prefix="benchmark_modality",
            baselines=self.modality_baselines,
            output_prefix="modality_leaderboard",
        )

    def _run_conformal_for_prefix(self, sample_prefix: str, output_prefix: str):
        outputs = []
        for seed in self.benchmark_seeds:
            sample_csv = self.leaderboards_dir / f"{sample_prefix}_samples_seed{seed}.csv"
            calib_csv = self.leaderboards_dir / f"{output_prefix}_calibration_seed{seed}.csv"
            out_csv = self.leaderboards_dir / f"{output_prefix}_seed{seed}.csv"
            split_csv = self.leaderboards_dir / f"{output_prefix}_split_seed{seed}.csv"
            detailed_csv = self.leaderboards_dir / f"{output_prefix}_detailed_seed{seed}.csv"

            if self._maybe_skip_file(out_csv):
                print(f"[SKIP] {output_prefix}_seed{seed}: {out_csv} exists.")
                outputs.append(out_csv)
                continue

            if not sample_csv.exists():
                print(f"[WARN] Missing sample CSV for conformal stage: {sample_csv}. Skipping seed={seed}.")
                continue

            calibrate_cmd = [
                self.python,
                "scripts/calibrate_conformal.py",
                "--samples",
                str(sample_csv),
                "--output",
                str(calib_csv),
                "--split-output",
                str(split_csv),
                "--alpha",
                str(self.args.conformal_alpha),
                "--calibration-ratio",
                str(self.args.conformal_calibration_ratio),
                "--split-unit",
                str(self.args.conformal_split_unit),
                "--seed",
                str(seed),
                "--disagreement-col",
                str(self.args.conformal_disagreement_col),
                "--disagreement-bins",
                str(self.args.conformal_disagreement_bins),
            ]
            if self.args.conformal_condition_on_disagreement:
                calibrate_cmd.append("--condition-on-disagreement")

            self._run_cmd(
                f"{output_prefix}_calibrate_seed{seed}",
                calibrate_cmd,
            )

            eval_cmd = [
                self.python,
                "scripts/eval_conformal.py",
                "--samples",
                str(sample_csv),
                "--calibration",
                str(calib_csv),
                "--output",
                str(out_csv),
                "--detailed-output",
                str(detailed_csv),
                "--alpha",
                str(self.args.conformal_alpha),
                "--calibration-ratio",
                str(self.args.conformal_calibration_ratio),
                "--split-unit",
                str(self.args.conformal_split_unit),
                "--seed",
                str(seed),
                "--disagreement-col",
                str(self.args.conformal_disagreement_col),
                "--disagreement-bins",
                str(self.args.conformal_disagreement_bins),
            ]
            if self.args.conformal_condition_on_disagreement:
                eval_cmd.append("--condition-on-disagreement")

            self._run_cmd(
                f"{output_prefix}_eval_seed{seed}",
                eval_cmd,
            )
            outputs.append(out_csv)

        if len(outputs) > 1:
            agg_out = self.leaderboards_dir / f"{output_prefix}_seed_aggregate.csv"
            self._aggregate_seed_csvs_generic(
                outputs,
                agg_out,
                group_cols=["name", "paradigm", "source", "disagreement_bin"],
            )
            print(f"[INFO] Conformal seed aggregate saved: {agg_out}")

    def stage_conformal(self):
        self._run_conformal_for_prefix("core_leaderboard", "conformal_core")
        self._run_conformal_for_prefix("modality_leaderboard", "conformal_modality")

    def stage_tables(self):
        self._run_cmd(
            "generate_tables",
            [
                self.python,
                "scripts/generate_results_tables.py",
                "--results_dir",
                str(self.ablation_cache_dir),
                "--format",
                "both",
                "--output",
                str(self.tables_dir),
            ],
            allow_fail=False,
        )

    def stage_package(self):
        artifacts = sorted(
            [str(p.relative_to(self.run_dir)) for p in self.run_dir.rglob("*") if p.is_file()]
        )
        lines = [
            "# Camera-Ready Run Summary",
            "",
            f"- Run directory: `{self.run_dir}`",
            f"- Created (UTC): `{self.manifest['created_utc']}`",
            f"- Datasets: `{', '.join(self.datasets)}`",
            f"- Benchmark seeds: `{', '.join(map(str, self.benchmark_seeds))}`",
            "",
            "## Key Artifacts",
            "",
        ]
        for path in artifacts:
            if path.startswith("leaderboards/") or path.startswith("tables/") or path.endswith("run_manifest.json"):
                lines.append(f"- `{path}`")
        lines += ["", "## All Files", ""]
        for path in artifacts:
            lines.append(f"- `{path}`")

        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[INFO] Summary written: {self.summary_path}")

    def run(self):
        stage_map = {
            "extract_signals": self.stage_extract_signals,
            "ssl_pretrain": self.stage_ssl_pretrain,
            "ssl_finetune": self.stage_ssl_finetune,
            "semi_train": self.stage_semi_train,
            "benchmark_core": self.stage_benchmark_core,
            "benchmark_modality": self.stage_benchmark_modality,
            "conformal": self.stage_conformal,
            "tables": self.stage_tables,
            "package": self.stage_package,
        }

        self._write_manifest()
        for stage in self.manifest["stages_requested"]:
            stage_map[stage]()
        self.manifest["finished_utc"] = _now_utc()
        self._write_manifest()
        print(f"\n[DONE] Camera-ready pipeline finished. Artifacts: {self.run_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run camera-ready training + ablation pipeline")
    parser.add_argument("--stages", type=str, default="all", help=f"Comma-separated stages or 'all'. Valid: {ALL_STAGES}")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--core-baselines", type=str, default=",".join(DEFAULT_CORE_BASELINES))
    parser.add_argument("--modality-baselines", type=str, default=",".join(DEFAULT_MODALITY_BASELINES))
    parser.add_argument("--benchmark-seeds", type=str, default="42")
    parser.add_argument("--enable-vlm-benchmark", action="store_true", help="Enable live VLM inference during benchmark stages (slow)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Optional per-dataset video limit for quick runs")

    parser.add_argument("--ssl-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--vlm-pretrain-epochs", type=int, default=50)
    parser.add_argument("--vlm-finetune-epochs", type=int, default=30)

    parser.add_argument("--ssl-data-dir", type=str, default="data/processed_signals")
    parser.add_argument("--finetune-data-dir", type=str, default="data/ablation_results/signals")
    parser.add_argument("--vlm-data-dir", type=str, default="data/ablation_results/signals")

    parser.add_argument("--signal-baseline", type=str, default="Fused-VMD")
    parser.add_argument("--min-processed-signals", type=int, default=10)
    parser.add_argument("--min-ablation-signals", type=int, default=10)
    parser.add_argument("--conformal-alpha", type=float, default=0.1)
    parser.add_argument("--conformal-calibration-ratio", type=float, default=0.3)
    parser.add_argument("--conformal-split-unit", type=str, default="subject")
    parser.add_argument("--conformal-condition-on-disagreement", action="store_true")
    parser.add_argument("--conformal-disagreement-col", type=str, default="disagreement_score")
    parser.add_argument("--conformal-disagreement-bins", type=str, default="0.2,0.4")
    parser.add_argument("--benchmark-conformal-calibration-csv", type=str, default="")
    parser.add_argument("--benchmark-conformal-default-qhat", type=float, default=6.0)
    parser.add_argument("--benchmark-conformal-condition-on-disagreement", action="store_true")

    parser.add_argument("--output-root", type=str, default="outputs/camera_ready_runs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--resume", action="store_true", help="Skip steps whose key outputs already exist")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    pipeline = CameraReadyPipeline(args)
    pipeline.run()
