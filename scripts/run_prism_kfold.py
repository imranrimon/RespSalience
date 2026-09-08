"""Drive PRISM 5-fold CV for a given signal dir: per fold, train_prism_breath + evaluate_prism_breath
on the kfold_learned manifests -> outputs/paper_runs/pedrr_kfold_<tag>/fold*/test_eval/prism_breath_predictions.csv.
Reproduces pedrr_kfold_learned (--signal-dir data/pedrr_signals_learned --tag learned) and produces the
B2 leakage-free control (--signal-dir data/pedrr_signals_learned_clean --tag learned_clean).

Set --epochs to MATCH your original pedrr_kfold_learned training so leaky-vs-clean is a fair control.
Run in an unrestricted GPU terminal:
    set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
    python scripts/run_prism_kfold.py --signal-dir data/pedrr_signals_learned_clean --tag learned_clean --epochs 40
"""
import argparse, subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KF = ROOT / "data" / "organized" / "kfold_learned"
PY = sys.executable

def run(cmd):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-dir", required=True)
    ap.add_argument("--tag", required=True, help="output run tag, e.g. learned_clean -> pedrr_kfold_learned_clean")
    ap.add_argument("--epochs", type=int, default=100,
                    help="MATCH your original pedrr_kfold_learned budget (journal suite TrainEpochs=100), not a low placeholder")
    ap.add_argument("--freeze-encoder-epochs", type=int, default=2, help="match headline (journal suite uses 2)")
    ap.add_argument("--encoder-pretrained", default="",
                    help="fixed SSL encoder checkpoint (rarely needed; prefer --pretrain-epochs for per-fold SSL)")
    ap.add_argument("--pretrain-epochs", type=int, default=0,
                    help="if >0, run per-fold SSL pretrain (pretrain_prism_breath) first -> FAITHFUL control (headline used 100)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--selection-metric", choices=["mae","auroc","hybrid"], default="",
                    help="Passed through to train_prism_breath.py; empty keeps its default (mae).")
    ap.add_argument("--folds", default="",
                    help="Comma-separated fold indices to run, e.g. '0,1,2'. Default: all five. "
                         "Use to split the CV across at most TWO concurrent processes on this box -- "
                         "a 4-way split has triggered a GPU TDR reset (exit 0x40010004).")
    ap.add_argument("--decision-loss-weight", type=float, default=0.0,
                    help="Tier-2c: pass through to train_prism_breath -> train PRISM with the net-benefit "
                         "decision loss (e.g. 0.3). 0 = standard PRISM (default).")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="pass through to train_prism_breath: stop each fold when val plateaus -> guarantees "
                         "convergence regardless of --epochs (recommend 15-20)")
    ap.add_argument("--resume-from-best", action="store_true",
                    help="warm-start each fold from its existing run/prism_breath_best.pt and CONTINUE training "
                         "(resume the current run to full convergence; use WITHOUT --skip-existing, pair with "
                         "--early-stop-patience so already-converged folds exit fast)")
    a = ap.parse_args()
    sd = a.signal_dir if os.path.isabs(a.signal_dir) else str(ROOT / a.signal_dir)
    if not os.path.isdir(sd) or not any(f.endswith(".npz") for f in os.listdir(sd)):
        sys.exit(f"[run_prism_kfold] no .npz signals in '{a.signal_dir}'. Build them first, e.g. for the\n"
                 f"  leakage-free control:  python scripts/build_efficientphys_flow_cleanfold.py  "
                 f"(creates data/pedrr_signals_learned_clean)")
    OUT = ROOT / "outputs" / "paper_runs" / f"pedrr_kfold_{a.tag}"
    folds = [int(x) for x in a.folds.split(",") if x.strip() != ""] if a.folds else list(range(5))
    if any(k < 0 or k > 4 for k in folds):
        sys.exit(f"[run_prism_kfold] --folds must be within 0..4, got {folds}")
    print(f"[run_prism_kfold] running folds {folds}", flush=True)
    for k in folds:
        fd = OUT / f"fold{k}"
        pred = fd / "test_eval" / "prism_breath_predictions.csv"
        if a.skip_existing and pred.exists():
            print(f"[skip] fold{k} (exists)", flush=True); continue
        print(f"\n===== PRISM fold{k}  signal-dir={a.signal_dir}  epochs={a.epochs} =====", flush=True)
        enc = a.encoder_pretrained
        if a.pretrain_epochs > 0:                       # per-fold SSL pretrain -> faithful control
            pdir = fd / "pretrain"
            run([PY, "scripts/pretrain_prism_breath.py",
                 "--signal-dir", a.signal_dir,
                 "--train-manifest", KF / f"fold{k}" / "train.json",
                 "--val-manifest",   KF / f"fold{k}" / "val.json",
                 "--output-dir",     pdir,
                 "--epochs", a.pretrain_epochs])
            enc = pdir / "prism_breath_pretrained.pt"
        init_from = ""
        freeze = a.freeze_encoder_epochs
        if a.resume_from_best:                      # existing headline uses fold*/train/; driver writes fold*/run/
            for cand in (fd / "run" / "prism_breath_best.pt", fd / "train" / "prism_breath_best.pt"):
                if cand.exists():
                    init_from, freeze = cand, 0     # continuing -> keep encoder trainable
                    print(f"[resume] fold{k}: warm-starting from {cand}", flush=True)
                    break
        train_cmd = [PY, "scripts/train_prism_breath.py",
                     "--signal-dir", a.signal_dir,
                     "--train-manifest", KF / f"fold{k}" / "train.json",
                     "--val-manifest",   KF / f"fold{k}" / "val.json",
                     "--output-dir",     fd / "run",
                     "--epochs", a.epochs,
                     "--freeze-encoder-epochs", freeze]
        if enc:
            train_cmd += ["--encoder-pretrained", enc]
        if init_from:
            train_cmd += ["--init-from", init_from]
        if a.selection_metric:
            train_cmd += ["--selection-metric", a.selection_metric]
        if a.early_stop_patience:
            train_cmd += ["--early-stop-patience", a.early_stop_patience]
        if a.decision_loss_weight:
            train_cmd += ["--decision-loss-weight", a.decision_loss_weight]
        run(train_cmd)
        run([PY, "scripts/evaluate_prism_breath.py",
             "--signal-dir", a.signal_dir,
             "--manifest",   KF / f"fold{k}" / "test.json",
             "--checkpoint", fd / "run" / "prism_breath_best.pt",
             "--output-dir", fd / "test_eval"])
    print(f"\nDONE: PRISM 5-fold -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
