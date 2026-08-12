# RUN: C:\Users\rimon\anaconda3\envs\resp_env\python.exe scripts\data_efficiency.py --signal-dir data/pedrr_signals_learned
#      (full sweep, sequential/1-wide; prefer running through scripts/run_queue.py for the 2-wide GPU pool.)
#      Single config (what run_queue launches):  ... scripts\data_efficiency.py --only --mode ssl --fold 0 --frac 0.25
#      Emit the queue job list (JSONL):          ... scripts\data_efficiency.py --emit-jobs
#      Aggregate results after the pool drains:  ... scripts\data_efficiency.py --summarize
"""
#18 -- DATA-EFFICIENCY curve for PRISM-Breath (scratch vs SSL) on the pedrr k-folds.

WHAT
----
For every (train-subject fraction) x (fold) x (mode in {scratch, ssl}):
  1. Subsample that fold's train.json to the fraction at the SUBJECT level (deterministic,
     nested across fractions, so 0.10 subjects subset 0.25 subset ... subset 1.0). Subject-level
     subsampling guarantees no subject leaks into that fold's val/test (the folds are already
     subject-disjoint; we only ever shrink the TRAIN subject pool).
  2. Train PRISM-Breath (model_variant=prism_breath) on the subsampled train manifest,
     validating on the fold's val.json.
       * mode=scratch : supervised training only (no pretraining).
       * mode=ssl     : per-config self-supervised masked-signal pretraining (pretrain_prism_breath)
                        on the SAME subsampled signals, then supervised fine-tune warm-started from it
                        (freeze-encoder-epochs 2). This isolates "does SSL help at each label budget".
  3. Evaluate on the fold's test.json and record tachypnea AUROC + RR-MAE.

Each config writes:  outputs/data_efficiency/<mode>/fold<k>/frac<pct>/{train,pretrain,test_eval}/...,
a per-config result.json, and a .done marker. `--summarize` rolls every result.json up into
outputs/data_efficiency/summary.{csv,json} (per-fold mean+-std AND a pooled-across-folds AUROC).

AGE FIX: signals are read from data/pedrr_signals_learned (age-patched) and, crucially, the
respiratory benchmark dataset prefers the manifest's age_months (data/organized/kfold_learned carries
corrected age), so every config trains/evaluates with the corrected WHO tachypnea thresholds.

TIME (single RTX 8000, 2-wide via run_queue): the full 5 fractions x 5 folds x 2 modes = 50 configs
takes ~12-24 h. A scratch config is one ~100-epoch fit on <=226 subjects (~15-40 min); an ssl config
adds a masked-recon pretrain (~15-30 min). Single-process launchers, num_workers=0.

RESUME: every config is gated by its own .done marker; re-running skips finished configs. Safe to
kill/restart. THIS SANDBOX CANNOT TRAIN (CUDA spawn / WinError 5) -- launch on the real GPU box.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.respiratory_benchmark_dataset import signal_filename_from_entry  # noqa: E402

DEFAULT_FRACTIONS = "0.10,0.25,0.50,0.75,1.0"
DEFAULT_FOLDS = "0,1,2,3,4"
DEFAULT_MODES = "scratch,ssl"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def pct(frac: float) -> str:
    return f"{int(round(frac * 100)):03d}"


def load_manifest(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def available_entries(entries: List[Dict], signal_dir: Path) -> List[Dict]:
    """Keep only entries whose extracted .npz exists, so the fraction is over usable data."""
    out = []
    for e in entries:
        if (signal_dir / signal_filename_from_entry(e)).exists():
            out.append(e)
    return out


def nested_subsample(entries: List[Dict], frac: float, fold: int, seed: int) -> List[Dict]:
    """Deterministic, subject-level, NESTED subsample of a fold's train entries.

    A single per-fold subject ordering is drawn from RandomState(seed*1000+fold); each fraction takes
    the first ceil(frac*N) subjects, so smaller fractions are strict subsets of larger ones.
    """
    import numpy as np

    subjects = sorted({str(e["subject_id"]) for e in entries})
    rng = np.random.RandomState(seed * 1000 + fold)
    order = list(subjects)
    rng.shuffle(order)
    n = max(1, math.ceil(frac * len(order)))
    chosen = set(order[:n])
    return [e for e in entries if str(e["subject_id"]) in chosen]


def write_manifest(path: Path, entries: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)


def run(cmd, dry: bool) -> None:
    printable = " ".join(str(c) for c in cmd)
    print("+ " + printable, flush=True)
    if dry:
        return
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))


def config_dir(out_root: Path, mode: str, fold: int, frac: float) -> Path:
    return out_root / mode / f"fold{fold}" / f"frac{pct(frac)}"


def metrics_from_csv(csv_path: Path) -> Dict[str, float]:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = pd.read_csv(csv_path)
    rr_pred = df["pred_rr_bpm"] if "pred_rr_bpm" in df else df["pred_bpm"]
    rr_gt = df["gt_rr_bpm"] if "gt_rr_bpm" in df else df["gt_bpm"]
    mae = float(np.mean(np.abs(rr_pred.to_numpy(float) - rr_gt.to_numpy(float))))
    y = df["gt_tachy"].to_numpy(int)
    score = df["tachy_prob"].to_numpy(float)
    auroc = float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else float("nan")
    return {"auroc": auroc, "mae": mae, "n_test": int(len(df)), "n_pos": int(y.sum())}


# --------------------------------------------------------------------------- #
# run one config
# --------------------------------------------------------------------------- #
def run_one(args, mode: str, fold: int, frac: float) -> None:
    signal_dir = Path(args.signal_dir if os.path.isabs(args.signal_dir) else ROOT / args.signal_dir)
    kf = Path(args.kfold_dir if os.path.isabs(args.kfold_dir) else ROOT / args.kfold_dir)
    out_root = Path(args.output_dir if os.path.isabs(args.output_dir) else ROOT / args.output_dir)

    cdir = config_dir(out_root, mode, fold, frac)
    done = cdir / ".done"
    if done.exists() and not args.overwrite:
        print(f"[skip] {mode} fold{fold} frac{pct(frac)} (.done exists)", flush=True)
        return

    train_manifest = kf / f"fold{fold}" / "train.json"
    val_manifest = kf / f"fold{fold}" / "val.json"
    test_manifest = kf / f"fold{fold}" / "test.json"

    # ---- subject-level nested subsample of the fold train set --------------- #
    entries = available_entries(load_manifest(train_manifest), signal_dir)
    subset = nested_subsample(entries, frac, fold, args.seed)
    sub_manifest = cdir / "train_sub.json"
    write_manifest(sub_manifest, subset)
    n_sub_subjects = len({str(e["subject_id"]) for e in subset})
    print(f"[config] {mode} fold{fold} frac{pct(frac)}: "
          f"{n_sub_subjects}/{len({str(e['subject_id']) for e in entries})} train subjects, "
          f"{len(subset)} clips", flush=True)

    train_dir = cdir / "train"
    test_dir = cdir / "test_eval"
    ckpt = train_dir / "prism_breath_best.pt"

    # ---- optional SSL pretrain (mode=ssl) ---------------------------------- #
    encoder_pretrained = ""
    freeze = 0
    if mode == "ssl":
        pre_dir = cdir / "pretrain"
        pre_ckpt = pre_dir / "prism_breath_pretrained.pt"
        pre_manifest = sub_manifest if args.ssl_pretrain_on == "subsample" else train_manifest
        if pre_ckpt.exists() and not args.overwrite:
            print(f"[skip pretrain] {pre_ckpt} exists", flush=True)
        else:
            run([args.python, "scripts/pretrain_prism_breath.py",
                 "--signal-dir", args.signal_dir,
                 "--train-manifest", pre_manifest,
                 "--val-manifest", val_manifest,
                 "--output-dir", pre_dir,
                 "--model-variant", "prism_breath",
                 "--seq-len", args.seq_len, "--d-model", args.d_model,
                 "--batch-size", args.batch_size,
                 "--epochs", args.pretrain_epochs,
                 "--num-workers", 0,
                 "--seed", args.seed], args.dry_run)
        encoder_pretrained = str(pre_ckpt)
        freeze = args.ssl_freeze_epochs

    # ---- supervised training ----------------------------------------------- #
    if ckpt.exists() and not args.overwrite:
        print(f"[skip train] {ckpt} exists", flush=True)
    else:
        cmd = [args.python, "scripts/train_prism_breath.py",
               "--signal-dir", args.signal_dir,
               "--train-manifest", sub_manifest,
               "--val-manifest", val_manifest,
               "--output-dir", train_dir,
               "--model-variant", "prism_breath",
               "--seq-len", args.seq_len, "--d-model", args.d_model,
               "--epochs", args.epochs, "--lr", args.lr,
               "--batch-size", args.batch_size,
               "--freeze-encoder-epochs", freeze,
               "--modality-dropout-prob", 0.0,
               "--num-workers", 0,
               "--seed", args.seed]
        if args.early_stop_patience:
            cmd += ["--early-stop-patience", args.early_stop_patience]
        if encoder_pretrained:
            cmd += ["--encoder-pretrained", encoder_pretrained]
        run(cmd, args.dry_run)

    # ---- evaluate on the fold test set ------------------------------------- #
    pred_csv = test_dir / "prism_breath_predictions.csv"
    if pred_csv.exists() and not args.overwrite:
        print(f"[skip eval] {pred_csv} exists", flush=True)
    else:
        run([args.python, "scripts/evaluate_prism_breath.py",
             "--signal-dir", args.signal_dir,
             "--manifest", test_manifest,
             "--checkpoint", ckpt,
             "--output-dir", test_dir,
             "--num-workers", 0], args.dry_run)

    if args.dry_run:
        return

    # ---- record per-config metrics + done marker --------------------------- #
    res = {"mode": mode, "fold": fold, "fraction": frac,
           "n_train_subjects": n_sub_subjects, "n_train_clips": len(subset),
           "ssl_pretrain_on": args.ssl_pretrain_on if mode == "ssl" else None}
    try:
        res.update(metrics_from_csv(pred_csv))
    except Exception as e:  # pragma: no cover - defensive
        res["metrics_error"] = str(e)
    with open(cdir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    done.write_text("ok\n", encoding="utf-8")
    print(f"[done] {mode} fold{fold} frac{pct(frac)}: "
          f"AUROC={res.get('auroc')!r} MAE={res.get('mae')!r}", flush=True)


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def summarize(args) -> None:
    import numpy as np
    import pandas as pd

    out_root = Path(args.output_dir if os.path.isabs(args.output_dir) else ROOT / args.output_dir)
    rows = []
    for res_path in out_root.glob("*/fold*/frac*/result.json"):
        try:
            rows.append(json.load(open(res_path)))
        except Exception:
            continue
    if not rows:
        print(f"[summarize] no result.json under {out_root}", flush=True)
        return
    df = pd.DataFrame(rows)

    # per (mode, fraction): mean+-std over folds of the per-fold AUROC / MAE
    agg = (df.groupby(["mode", "fraction"])
             .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
                  mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                  n_folds=("fold", "nunique"))
             .reset_index()
             .sort_values(["mode", "fraction"]))

    # pooled-across-folds AUROC per (mode, fraction) from the raw test predictions
    from sklearn.metrics import roc_auc_score
    pooled = []
    for (mode, frac), _ in df.groupby(["mode", "fraction"]):
        ys, ss, maes = [], [], []
        for fold in sorted(df[(df["mode"] == mode) & (df["fraction"] == frac)]["fold"].unique()):
            csv = config_dir(out_root, mode, int(fold), float(frac)) / "test_eval" / "prism_breath_predictions.csv"
            if not csv.exists():
                continue
            d = pd.read_csv(csv)
            ys.append(d["gt_tachy"].to_numpy(int))
            ss.append(d["tachy_prob"].to_numpy(float))
            rr_pred = d["pred_rr_bpm"] if "pred_rr_bpm" in d else d["pred_bpm"]
            rr_gt = d["gt_rr_bpm"] if "gt_rr_bpm" in d else d["gt_bpm"]
            maes.append(np.abs(rr_pred.to_numpy(float) - rr_gt.to_numpy(float)))
        if ys:
            y = np.concatenate(ys); s = np.concatenate(ss)
            pooled.append({"mode": mode, "fraction": frac,
                           "pooled_auroc": float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan"),
                           "pooled_mae": float(np.concatenate(maes).mean()),
                           "pooled_n": int(len(y))})
    pooled_df = pd.DataFrame(pooled)
    if not pooled_df.empty:
        agg = agg.merge(pooled_df, on=["mode", "fraction"], how="left")

    out_root.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_root / "summary.csv", index=False)
    json.dump({"per_config": rows, "summary": agg.to_dict("records")},
              open(out_root / "summary.json", "w"), indent=2)
    print(agg.to_string(index=False), flush=True)
    print(f"\n[summarize] wrote {out_root/'summary.csv'} and summary.json", flush=True)


# --------------------------------------------------------------------------- #
# emit jobs (JSONL, one atomic config per line) for scripts/run_queue.py
# --------------------------------------------------------------------------- #
def emit_jobs(args, fractions, folds, modes) -> None:
    for mode in modes:
        for fold in folds:
            for frac in fractions:
                name = f"de_{mode}_f{fold}_frac{pct(frac)}"
                argv = [args.python, "scripts/data_efficiency.py", "--only",
                        "--mode", mode, "--fold", str(fold), "--frac", str(frac),
                        "--signal-dir", args.signal_dir, "--kfold-dir", args.kfold_dir,
                        "--output-dir", args.output_dir,
                        "--epochs", str(args.epochs), "--pretrain-epochs", str(args.pretrain_epochs),
                        "--lr", str(args.lr), "--batch-size", str(args.batch_size),
                        "--seq-len", str(args.seq_len), "--d-model", str(args.d_model),
                        "--seed", str(args.seed), "--ssl-pretrain-on", args.ssl_pretrain_on,
                        "--ssl-freeze-epochs", str(args.ssl_freeze_epochs)]
                if args.early_stop_patience:
                    argv += ["--early-stop-patience", str(args.early_stop_patience)]
                print(json.dumps({"name": name, "argv": argv, "cwd": str(ROOT)}))


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal-dir", default="data/pedrr_signals_learned")
    ap.add_argument("--kfold-dir", default="data/organized/kfold_learned")
    ap.add_argument("--output-dir", default="outputs/data_efficiency")
    ap.add_argument("--fractions", default=DEFAULT_FRACTIONS)
    ap.add_argument("--folds", default=DEFAULT_FOLDS)
    ap.add_argument("--modes", default=DEFAULT_MODES, help="comma list subset of: scratch,ssl")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--pretrain-epochs", type=int, default=100)
    ap.add_argument("--early-stop-patience", type=int, default=20)
    ap.add_argument("--ssl-freeze-epochs", type=int, default=2, help="freeze encoder for N epochs after SSL warm-start")
    ap.add_argument("--ssl-pretrain-on", choices=["subsample", "fullfold"], default="subsample",
                    help="SSL masked-recon pretrain corpus: the subsampled train (default, honest label-budget "
                         "curve) or the fold's full train set")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=300)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--python", default=sys.executable, help="python used for emitted/sub jobs")
    # execution modes
    ap.add_argument("--only", action="store_true", help="run exactly ONE config (--mode/--fold/--frac)")
    ap.add_argument("--mode", default="scratch")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--emit-jobs", action="store_true", help="print JSONL job list for run_queue.py")
    ap.add_argument("--summarize", action="store_true", help="aggregate result.json -> summary.{csv,json}")
    ap.add_argument("--overwrite", action="store_true", help="recompute even if a .done/checkpoint exists")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    if args.summarize:
        summarize(args)
        return 0
    if args.emit_jobs:
        emit_jobs(args, fractions, folds, modes)
        return 0
    if args.only:
        run_one(args, args.mode, args.fold, args.frac)
        return 0

    # full sweep (sequential / 1-wide; use run_queue.py for the 2-wide GPU pool)
    for mode in modes:
        for fold in folds:
            for frac in fractions:
                run_one(args, mode, fold, frac)
    if not args.dry_run:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
