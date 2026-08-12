# RUN: envs\resp_env\python.exe scripts\retrain_agefix.py --skip-existing
"""
Reviewer item #1 -- CLEAN AGE-FIX RETRAIN of the two headline contributions.

WHY
----
An age-0-newborn bug stored age_months=6 for 9 true newborns. Their WHO tachypnea
threshold was therefore 50 BPM (2-12mo band) instead of 60 BPM (<2mo band), which
biased every age-derived column (threshold_bpm / gt_tachy) and the salience tachy_prob
(salience mass above threshold). The .npz caches AND the manifests have now been PATCHED
to true age, and src/core/respiratory_benchmark_dataset.py prefers the manifest age
(verified: data/organized/kfold_learned carries newborns at age_months=0.0). So any
retrain from here uses the CORRECT age with zero further edits.

WHAT THIS DOES  (three resume-safe stages; nothing is re-run if its .done marker exists)
----------------------------------------------------------------------------------------
  A. PRISM 5-fold agefix retrain  -- thin wrapper over scripts/run_prism_kfold.py:
     retrains PRISM (model_variant=prism_breath) on the age-patched signals that
     pedrr_kfold_learned used (data/pedrr_signals_learned), tag `agefix`, epochs 100,
     --early-stop-patience 20, warm-started from the SHARED SSL init (age-independent
     masked-signal pretrain -- reusing it is faithful and saves ~1.5-2 h), then runs the
     standard eval so the corrected checkpoints emit a fresh predictions set at
     outputs/paper_runs/pedrr_kfold_agefix/fold*/test_eval/prism_breath_predictions.csv.

  B. Salience source-CSV refresh (default ON; --no-refresh-source-eval to skip):
     the RespSalience trainers read threshold_bpm / gt_tachy from
     pedrr_kfold_learned/fold*/test_eval CSVs, which still hold the stale age. Re-EVAL
     (no training) the EXISTING pedrr_kfold_learned/fold*/train checkpoints on the
     corrected test manifests -> overwrites those CSVs IN PLACE with corrected age /
     threshold / gt_tachy (pred_rr is deterministic and unchanged). This is exactly the
     age-fix propagation; it is required for stage C to see corrected thresholds.

  C. Salience retrain (its neonatal number was the most age-sensitive):
     re-run scripts/train_resp_salience.py (plain) and train_resp_salience_ssl.py (SSL)
     for seeds 0,1,2 -> refreshes pedrr_kfold_salience_seed{0,1,2} and
     pedrr_kfold_salience_ssl_seed{0,1,2} with corrected-age predictions.

TIME ESTIMATE (RTX 8000):  ~4-6 h total.
  * Stage A PRISM 5-fold: ~2.5-3 h (~16-60 min/fold; faster with --early-stop-patience 20).
  * Stage B re-eval (5 forward passes, no training): ~5-15 min.
  * Stage C salience (6 CPU/GPU-light runs, 70 epochs each on ~300 signals): ~30-60 min.

ASSUMPTIONS TO VERIFY BEFORE FIRING
-----------------------------------
  * envs\\resp_env\\python.exe resolves to C:\\Users\\rimon\\anaconda3\\envs\\resp_env\\python.exe;
    run from repo root G:\\Respiratory Project. (Subprocesses inherit sys.executable.)
  * --signal-dir data/pedrr_signals_learned currently holds 297 .npz (35 were deleted by the
    #24 val-contamination repair). To retrain on the full 332, first restore them per
    docs/REPRODUCIBILITY.md B1 (python scripts/build_learned_flow_signals.py) and re-run;
    otherwise agefix trains on the same 297 the current learned run effectively used.
  * LR CAVEAT: run_prism_kfold.py does not expose --lr, so PRISM trains at train_prism_breath's
    default lr=1e-3. The headline pedrr_kfold_learned used lr=3e-4 (recovered from its
    checkpoint config). This is the single hyperparameter that differs from the headline; the
    agefix run is a clean re-train and 1e-3 is stable, but if you need bit-for-bit headline
    hyperparameters, drive train_prism_breath directly with --lr 3e-4 instead.
  * Shared SSL init: outputs/paper_runs/pedrr_kfold_learned/pretrain/prism_breath_pretrained.pt
    (present). It is age-independent (masked-signal reconstruction), so reusing it under
    corrected age is faithful; the encoder is fully trainable (freeze-encoder-epochs 0).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
KFOLD = ROOT / "data" / "organized" / "kfold_learned"
LEARNED_RUN = ROOT / "outputs" / "paper_runs" / "pedrr_kfold_learned"
SHARED_SSL_INIT = LEARNED_RUN / "pretrain" / "prism_breath_pretrained.pt"


def run(cmd, dry: bool) -> None:
    printable = " ".join(str(c) for c in cmd)
    print("+ " + printable, flush=True)
    if dry:
        return
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))


def marker_done(path: Path) -> bool:
    return path.exists()


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def stage_prism(args) -> None:
    out = ROOT / "outputs" / "paper_runs" / f"pedrr_kfold_{args.tag}"
    done = out / "agefix_prism.done"
    if args.skip_existing and marker_done(done):
        print(f"[skip] stage A PRISM agefix -> {done} exists", flush=True)
        return
    cmd = [
        PY, "scripts/run_prism_kfold.py",
        "--signal-dir", args.signal_dir,
        "--tag", args.tag,
        "--epochs", args.epochs,
        "--early-stop-patience", args.early_stop_patience,
        "--freeze-encoder-epochs", 0,
    ]
    if SHARED_SSL_INIT.exists():
        cmd += ["--encoder-pretrained", str(SHARED_SSL_INIT)]
    else:
        print(f"[warn] shared SSL init missing at {SHARED_SSL_INIT}; training from scratch", flush=True)
    if args.skip_existing:
        cmd += ["--skip-existing"]
    run(cmd, args.dry_run)
    # verify all 5 folds emitted predictions before marking done
    preds = list((out).glob("fold*/test_eval/prism_breath_predictions.csv"))
    if not args.dry_run and len(preds) < 5:
        raise SystemExit(f"[abort] expected 5 fold predictions under {out}, found {len(preds)}")
    if not args.dry_run:
        touch(done)


def stage_refresh_source(args) -> None:
    if not args.refresh_source_eval:
        print("[skip] stage B source-CSV refresh (disabled)", flush=True)
        return
    for k in range(5):
        ckpt = LEARNED_RUN / f"fold{k}" / "train" / "prism_breath_best.pt"
        out_dir = LEARNED_RUN / f"fold{k}" / "test_eval"
        done = out_dir / "agefix_refresh.done"
        if args.skip_existing and marker_done(done):
            print(f"[skip] stage B fold{k} refresh -> exists", flush=True)
            continue
        if not ckpt.exists():
            print(f"[warn] fold{k}: missing {ckpt}; cannot refresh salience source CSV", flush=True)
            continue
        print(f"[stage B] re-eval learned fold{k} on corrected age (IN PLACE overwrite of {out_dir})", flush=True)
        run([
            PY, "scripts/evaluate_prism_breath.py",
            "--signal-dir", args.signal_dir,
            "--manifest", KFOLD / f"fold{k}" / "test.json",
            "--checkpoint", ckpt,
            "--output-dir", out_dir,
        ], args.dry_run)
        if not args.dry_run:
            touch(done)


def stage_salience(args) -> None:
    seeds = [s.strip() for s in str(args.salience_seeds).split(",") if s.strip() != ""]
    jobs = []
    if args.salience_plain:
        jobs += [("train_resp_salience.py", s, f"pedrr_kfold_salience_seed{s}") for s in seeds]
    if args.salience_ssl:
        jobs += [("train_resp_salience_ssl.py", s, f"pedrr_kfold_salience_ssl_seed{s}") for s in seeds]
    for script, seed, run_name in jobs:
        done = ROOT / "outputs" / "paper_runs" / run_name / "agefix_salience.done"
        if args.skip_existing and marker_done(done):
            print(f"[skip] salience {script} seed{seed} -> exists", flush=True)
            continue
        run([PY, f"scripts/{script}", "--seed", seed, "--epochs", args.salience_epochs], args.dry_run)
        if not args.dry_run:
            touch(done)


def main() -> int:
    ap = argparse.ArgumentParser(description="#1 clean age-fix retrain (PRISM 5-fold + RespSalience).")
    ap.add_argument("--signal-dir", default="data/pedrr_signals_learned")
    ap.add_argument("--tag", default="agefix", help="PRISM run tag -> outputs/paper_runs/pedrr_kfold_<tag>")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--early-stop-patience", type=int, default=20)
    # salience
    ap.add_argument("--salience-seeds", default="0,1,2")
    ap.add_argument("--salience-epochs", type=int, default=70)
    ap.add_argument("--salience-plain", dest="salience_plain", action="store_true", default=True)
    ap.add_argument("--no-salience-plain", dest="salience_plain", action="store_false")
    ap.add_argument("--salience-ssl", dest="salience_ssl", action="store_true", default=True)
    ap.add_argument("--no-salience-ssl", dest="salience_ssl", action="store_false")
    # stage toggles
    ap.add_argument("--no-prism", dest="do_prism", action="store_false", default=True)
    ap.add_argument("--no-salience", dest="do_salience", action="store_false", default=True)
    ap.add_argument("--refresh-source-eval", dest="refresh_source_eval", action="store_true", default=True,
                    help="re-eval existing learned checkpoints so salience reads corrected thresholds (default ON)")
    ap.add_argument("--no-refresh-source-eval", dest="refresh_source_eval", action="store_false")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without executing")
    args = ap.parse_args()

    print(f"[retrain_agefix] python={PY}", flush=True)
    if args.do_prism:
        stage_prism(args)
    if args.do_salience:
        stage_refresh_source(args)
        stage_salience(args)
    print("[retrain_agefix] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
