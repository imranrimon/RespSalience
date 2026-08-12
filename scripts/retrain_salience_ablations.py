# RUN: C:\Users\rimon\anaconda3\envs\resp_env\python.exe scripts\retrain_salience_ablations.py --variants all
#      (sequential/1-wide; prefer scripts/run_queue.py for the 2-wide GPU pool.)
#      Single variant (what run_queue launches):  ... scripts\retrain_salience_ablations.py --only lean
#      Emit the queue job list (JSONL):           ... scripts\retrain_salience_ablations.py --variants all --emit-jobs
#      List variants + resolved commands:         ... scripts\retrain_salience_ablations.py --list
"""
Corrected-age (#agefix) RETRAIN of the PRISM-Breath ablation / model variants and the RespSalience
seeds, on the age-patched signals (data/pedrr_signals_learned + data/organized/kfold_learned, whose
manifests carry the corrected age_months that the benchmark dataset prefers).

Every model/flag variant's exact invocation was recovered from the ORIGINAL run's saved checkpoint
config (vars(args) stored in outputs/paper_runs/pedrr_kfold_<variant>/fold0/train/prism_breath_best.pt):
common budget epochs=100, lr=3e-4, batch=16, seed=42, freeze-encoder-epochs=0, modality-dropout=0.0,
num_workers=0, and -- for the standard-encoder variants -- warm-started from the SHARED learned SSL
init outputs/paper_runs/pedrr_kfold_learned/pretrain/prism_breath_pretrained.pt (age-independent
masked-recon, so reusing it under corrected age is faithful). Outputs go to a *_agefix run tag so the
originals are never clobbered:  outputs/paper_runs/pedrr_kfold_<variant>_agefix/.

VARIANTS (all reproduce the recovered command exactly; --variants subsets by name or group):
  group 'ablations' :
    abl_no_scan (prism_no_scan) | abl_no_spectral (prism_no_spectral) |
    abl_no_codebook (prism_no_codebook) | abl_no_prototypes (prism_no_prototypes) |
    abl_shared_trunk (prism_shared_trunk) | abl_het_only (--het-weight 0.1) |
    abl_synth_only (--synth-highrr-prob 0.25) | abl_moddrop01 (--modality-dropout-prob 0.1)
  group 'models' :
    lean (prism_lean) | freqfwd (--use-freq-forward-fusion) | snrfusion (--use-snr-fusion) |
    hetsynth (--het-weight 0.1 --synth-highrr-prob 0.25) | nopretrain (no SSL init) |
    flowonly (signal-dir data/pedrr_signals_learned_flowonly, shared SSL init) |
    mamba (--model-variant mamba, own SSL pretrain) | cliffphys (--model-variant cliffphys, own SSL) |
    airflow_compact (--model-variant airflow_compact, own compact SSL)
  group 'salience' :
    salience_seed{0,1,2} (train_resp_salience.py) | salience_ssl_seed{0,1,2} (train_resp_salience_ssl.py) |
    salience_sinc_seed0 (train_resp_salience.py --sincnet)
    -> RespSalience reads its labels/thresholds from pedrr_kfold_learned/fold*/test_eval CSVs, so this
       script first RE-EVALUATES the existing learned checkpoints on the corrected test manifests
       (in-place, resume-safe, mkdir-locked) to propagate corrected age before training. Salience
       outputs land at pedrr_kfold_salience<...>_agefix_seed<s>/ (originals preserved).

DEPRECATED (excluded unless --include-deprecated):
  v3 -- reproducible (--model-variant prism_breath, own SSL pretrain) BUT its signal-dir
  data/pedrr_signals_v3 is the SUPERSEDED/contaminated flow extraction removed in Task #24; retraining
  it would re-introduce the val contamination. Left out on purpose; see the report.

TIME (single RTX 8000, 2-wide via run_queue): the ablation/model variants are 5-fold ~2.5-3 h each
(~25-40 h for the full set at 2-wide); the salience seeds are GPU-light (~5-10 min each). RESUME: per
variant, per fold .done markers; re-running skips finished work. Single-process, num_workers=0.
THIS SANDBOX CANNOT TRAIN (CUDA spawn / WinError 5) -- launch on the real GPU box.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
KFOLD = ROOT / "data" / "organized" / "kfold_learned"
LEARNED_RUN = ROOT / "outputs" / "paper_runs" / "pedrr_kfold_learned"
SHARED_SSL_INIT = LEARNED_RUN / "pretrain" / "prism_breath_pretrained.pt"
PAPER_RUNS = ROOT / "outputs" / "paper_runs"


# --------------------------------------------------------------------------- #
# variant registry  (spec recovered from each original run's checkpoint config)
# --------------------------------------------------------------------------- #
def prism_variant(model_variant: str, flags=None, init="shared", signal_dir="data/pedrr_signals_learned",
                  modality_dropout=0.0) -> Dict:
    return {"kind": "prism", "model_variant": model_variant, "flags": list(flags or []),
            "init": init, "signal_dir": signal_dir, "modality_dropout": modality_dropout}


def salience_variant(script: str, seed: int, tag: str, sincnet=False) -> Dict:
    return {"kind": "salience", "script": script, "seed": seed, "tag": tag, "sincnet": sincnet}


REGISTRY: Dict[str, Dict] = {
    # ---- ablations (model_variant toggles + training-flag toggles) --------- #
    "abl_no_scan":       prism_variant("prism_no_scan"),
    "abl_no_spectral":   prism_variant("prism_no_spectral"),
    "abl_no_codebook":   prism_variant("prism_no_codebook"),
    "abl_no_prototypes": prism_variant("prism_no_prototypes"),
    "abl_shared_trunk":  prism_variant("prism_shared_trunk"),
    "abl_het_only":      prism_variant("prism_breath", flags=["--het-weight", "0.1"]),
    "abl_synth_only":    prism_variant("prism_breath", flags=["--synth-highrr-prob", "0.25"]),
    "abl_moddrop01":     prism_variant("prism_breath", modality_dropout=0.1),
    # ---- model / fusion variants ------------------------------------------ #
    "lean":              prism_variant("prism_lean"),
    "freqfwd":           prism_variant("prism_breath", flags=["--use-freq-forward-fusion"]),
    "snrfusion":         prism_variant("prism_breath", flags=["--use-snr-fusion"]),
    "hetsynth":          prism_variant("prism_breath", flags=["--het-weight", "0.1", "--synth-highrr-prob", "0.25"]),
    "nopretrain":        prism_variant("prism_breath", init="none"),
    "flowonly":          prism_variant("prism_breath", signal_dir="data/pedrr_signals_learned_flowonly"),
    "mamba":             prism_variant("mamba", init="per_variant"),
    "cliffphys":         prism_variant("cliffphys", init="per_variant"),
    "airflow_compact":   prism_variant("airflow_compact", init="per_variant"),
    # ---- RespSalience seeds ----------------------------------------------- #
    "salience_seed0":      salience_variant("train_resp_salience.py", 0, "_agefix"),
    "salience_seed1":      salience_variant("train_resp_salience.py", 1, "_agefix"),
    "salience_seed2":      salience_variant("train_resp_salience.py", 2, "_agefix"),
    "salience_ssl_seed0":  salience_variant("train_resp_salience_ssl.py", 0, "_ssl_agefix"),
    "salience_ssl_seed1":  salience_variant("train_resp_salience_ssl.py", 1, "_ssl_agefix"),
    "salience_ssl_seed2":  salience_variant("train_resp_salience_ssl.py", 2, "_ssl_agefix"),
    "salience_sinc_seed0": salience_variant("train_resp_salience.py", 0, "_sinc_agefix", sincnet=True),
}

# reproducible but excluded by default (superseded/contaminated v3 flow signal-dir)
DEPRECATED = {
    "v3": prism_variant("prism_breath", init="per_variant", signal_dir="data/pedrr_signals_v3"),
}

GROUPS = {
    "ablations": ["abl_no_scan", "abl_no_spectral", "abl_no_codebook", "abl_no_prototypes",
                  "abl_shared_trunk", "abl_het_only", "abl_synth_only", "abl_moddrop01"],
    "models": ["lean", "freqfwd", "snrfusion", "hetsynth", "nopretrain", "flowonly",
               "mamba", "cliffphys", "airflow_compact"],
    "salience": ["salience_seed0", "salience_seed1", "salience_seed2",
                 "salience_ssl_seed0", "salience_ssl_seed1", "salience_ssl_seed2",
                 "salience_sinc_seed0"],
}


def resolve_selection(spec: str, include_deprecated: bool) -> List[str]:
    reg = dict(REGISTRY)
    if include_deprecated:
        reg.update(DEPRECATED)
    names: List[str] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok == "all":
            names += GROUPS["ablations"] + GROUPS["models"] + GROUPS["salience"]
            if include_deprecated:
                names += list(DEPRECATED)
        elif tok in GROUPS:
            names += GROUPS[tok]
        elif tok in reg:
            names.append(tok)
        else:
            raise SystemExit(f"[retrain] unknown variant/group {tok!r}. "
                             f"Known: all, {', '.join(GROUPS)}, or {', '.join(reg)}")
    # de-dup, keep order
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def get_spec(name: str) -> Dict:
    if name in REGISTRY:
        return REGISTRY[name]
    if name in DEPRECATED:
        return DEPRECATED[name]
    raise SystemExit(f"[retrain] unknown variant {name!r}")


# --------------------------------------------------------------------------- #
def run(cmd, dry: bool) -> None:
    printable = " ".join(str(c) for c in cmd)
    print("+ " + printable, flush=True)
    if dry:
        return
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))


def _lock(path: Path, timeout=1800):
    """Atomic mkdir lock so concurrent salience jobs don't race the in-place source re-eval."""
    lock = path.with_suffix(".lock")
    start = time.time()
    while True:
        try:
            os.mkdir(lock)
            return lock
        except FileExistsError:
            if time.time() - start > timeout:
                return None  # give up waiting; proceed (the .done markers still guard idempotency)
            time.sleep(3.0)


def ensure_salience_source(args) -> None:
    """Re-evaluate the existing learned checkpoints on the corrected test manifests, overwriting
    pedrr_kfold_learned/fold*/test_eval CSVs IN PLACE so RespSalience reads corrected age/thresholds.
    Resume-safe (agefix_refresh.done per fold) and mkdir-locked against concurrent salience jobs."""
    for k in range(5):
        out_dir = LEARNED_RUN / f"fold{k}" / "test_eval"
        done = out_dir / "agefix_refresh.done"
        if done.exists():
            continue
        ckpt = LEARNED_RUN / f"fold{k}" / "train" / "prism_breath_best.pt"
        if not ckpt.exists():
            print(f"[warn] salience-source fold{k}: missing {ckpt}; cannot refresh", flush=True)
            continue
        lock = _lock(done)
        try:
            if done.exists():
                continue
            print(f"[salience-source] re-eval learned fold{k} on corrected age (in-place)", flush=True)
            run([args.python, "scripts/evaluate_prism_breath.py",
                 "--signal-dir", args.signal_dir,
                 "--manifest", KFOLD / f"fold{k}" / "test.json",
                 "--checkpoint", ckpt,
                 "--output-dir", out_dir,
                 "--num-workers", 0], args.dry_run)
            if not args.dry_run:
                done.write_text("ok\n", encoding="utf-8")
        finally:
            if lock is not None:
                try:
                    os.rmdir(lock)
                except OSError:
                    pass


# --------------------------------------------------------------------------- #
def init_path_for(name: str, spec: Dict, args) -> str:
    """Resolve the encoder-pretrained init for a prism variant; build a per-variant SSL pretrain
    when the variant's encoder differs from the shared PRISM encoder (mamba/cliffphys/airflow_compact)."""
    if spec["init"] == "none":
        return ""
    if spec["init"] == "shared":
        if not SHARED_SSL_INIT.exists():
            print(f"[warn] shared SSL init missing at {SHARED_SSL_INIT}; training {name} from scratch", flush=True)
            return ""
        return str(SHARED_SSL_INIT)
    # per_variant: pretrain once for this variant's encoder, reuse across folds
    out = PAPER_RUNS / f"pedrr_kfold_{name}_agefix" / "pretrain"
    ckpt = out / "prism_breath_pretrained.pt"
    if not ckpt.exists():
        run([args.python, "scripts/pretrain_prism_breath.py",
             "--signal-dir", spec["signal_dir"],
             "--train-manifest", KFOLD / "fold0" / "train.json",
             "--val-manifest", KFOLD / "fold0" / "val.json",
             "--output-dir", out,
             "--model-variant", spec["model_variant"],
             "--seq-len", args.seq_len, "--d-model", args.d_model,
             "--batch-size", args.batch_size,
             "--epochs", args.pretrain_epochs,
             "--num-workers", 0,
             "--seed", args.seed], args.dry_run)
    return str(ckpt)


def run_prism_variant(name: str, spec: Dict, folds: List[int], args) -> None:
    out_root = PAPER_RUNS / f"pedrr_kfold_{name}_agefix"
    init = init_path_for(name, spec, args)
    for k in folds:
        fdir = out_root / f"fold{k}"
        done = fdir / ".done"
        if done.exists() and not args.overwrite:
            print(f"[skip] {name} fold{k} (.done exists)", flush=True)
            continue
        train_dir = fdir / "train"
        test_dir = fdir / "test_eval"
        ckpt = train_dir / "prism_breath_best.pt"
        if not (ckpt.exists() and not args.overwrite):
            cmd = [args.python, "scripts/train_prism_breath.py",
                   "--signal-dir", spec["signal_dir"],
                   "--train-manifest", KFOLD / f"fold{k}" / "train.json",
                   "--val-manifest", KFOLD / f"fold{k}" / "val.json",
                   "--output-dir", train_dir,
                   "--model-variant", spec["model_variant"],
                   "--seq-len", args.seq_len, "--d-model", args.d_model,
                   "--epochs", args.epochs, "--lr", args.lr,
                   "--batch-size", args.batch_size,
                   "--freeze-encoder-epochs", args.freeze_encoder_epochs,
                   "--modality-dropout-prob", spec["modality_dropout"],
                   "--num-workers", 0,
                   "--seed", args.seed] + spec["flags"]
            if args.early_stop_patience:
                cmd += ["--early-stop-patience", args.early_stop_patience]
            if init:
                cmd += ["--encoder-pretrained", init]
            run(cmd, args.dry_run)
        if not (test_dir / "prism_breath_predictions.csv").exists() or args.overwrite:
            run([args.python, "scripts/evaluate_prism_breath.py",
                 "--signal-dir", spec["signal_dir"],
                 "--manifest", KFOLD / f"fold{k}" / "test.json",
                 "--checkpoint", ckpt,
                 "--output-dir", test_dir,
                 "--num-workers", 0], args.dry_run)
        if not args.dry_run:
            done.write_text("ok\n", encoding="utf-8")
    print(f"[done] prism variant {name} -> {out_root}", flush=True)


def run_salience_variant(name: str, spec: Dict, args) -> None:
    ensure_salience_source(args)
    out_dir = PAPER_RUNS / f"pedrr_kfold_salience{spec['tag']}_seed{spec['seed']}"
    done = out_dir / "agefix_salience.done"
    if done.exists() and not args.overwrite:
        print(f"[skip] salience {name} (.done exists)", flush=True)
        return
    cmd = [args.python, f"scripts/{spec['script']}",
           "--seed", spec["seed"], "--epochs", args.salience_epochs, "--tag", spec["tag"]]
    if spec["sincnet"]:
        cmd += ["--sincnet"]
    run(cmd, args.dry_run)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        done.write_text("ok\n", encoding="utf-8")
    print(f"[done] salience variant {name} -> {out_dir}", flush=True)


def run_variant(name: str, args) -> None:
    spec = get_spec(name)
    if spec["kind"] == "salience":
        run_salience_variant(name, spec, args)
    else:
        folds = [int(x) for x in args.folds.split(",") if x.strip()]
        run_prism_variant(name, spec, folds, args)


# --------------------------------------------------------------------------- #
def emit_jobs(names: List[str], args) -> None:
    for name in names:
        argv = [args.python, "scripts/retrain_salience_ablations.py", "--only", name,
                "--signal-dir", args.signal_dir,
                "--epochs", str(args.epochs), "--pretrain-epochs", str(args.pretrain_epochs),
                "--lr", str(args.lr), "--batch-size", str(args.batch_size),
                "--seq-len", str(args.seq_len), "--d-model", str(args.d_model),
                "--seed", str(args.seed), "--freeze-encoder-epochs", str(args.freeze_encoder_epochs),
                "--salience-epochs", str(args.salience_epochs), "--folds", args.folds]
        if args.early_stop_patience:
            argv += ["--early-stop-patience", str(args.early_stop_patience)]
        print(json.dumps({"name": f"var_{name}", "argv": argv, "cwd": str(ROOT)}))


def list_variants(names: List[str]) -> None:
    for name in names:
        spec = get_spec(name)
        if spec["kind"] == "salience":
            extra = " --sincnet" if spec["sincnet"] else ""
            print(f"{name:22s} salience: {spec['script']} --seed {spec['seed']} --tag {spec['tag']}{extra}")
        else:
            print(f"{name:22s} prism:  --model-variant {spec['model_variant']} "
                  f"init={spec['init']} moddrop={spec['modality_dropout']} "
                  f"signal-dir={spec['signal_dir']} flags={' '.join(spec['flags']) or '-'}")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="all", help="'all', a group (ablations|models|salience), or names")
    ap.add_argument("--only", default="", help="run exactly ONE variant by name (what run_queue launches)")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--signal-dir", default="data/pedrr_signals_learned",
                    help="default signal dir (per-variant registry entries override, e.g. flowonly)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--pretrain-epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=300)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-encoder-epochs", type=int, default=0,
                    help="matches the recovered variant configs (0 even with the shared SSL init)")
    ap.add_argument("--early-stop-patience", type=int, default=20)
    ap.add_argument("--salience-epochs", type=int, default=70)
    ap.add_argument("--python", default=sys.executable, help="python used for emitted/sub jobs")
    ap.add_argument("--include-deprecated", action="store_true", help="also allow the deprecated v3 variant")
    ap.add_argument("--emit-jobs", action="store_true", help="print JSONL job list for run_queue.py")
    ap.add_argument("--list", action="store_true", help="print resolved per-variant commands and exit")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.only:
        run_variant(args.only, args)
        return 0
    names = resolve_selection(args.variants, args.include_deprecated)
    if args.list:
        list_variants(names)
        return 0
    if args.emit_jobs:
        emit_jobs(names, args)
        return 0
    for name in names:
        run_variant(name, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
