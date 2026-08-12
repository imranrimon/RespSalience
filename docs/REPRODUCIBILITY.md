# Reproducibility & runbook — PRISM-Breath experiments section

Environment: conda `resp_env` (see `environment.yml`, UTF-16). Run all commands from repo root.
Data splits are checked in under `data/organized/` (subject-disjoint, deterministic). Prediction
CSVs live under `outputs/paper_runs/<run>/fold*/test_eval/prism_breath_predictions.csv`.

## A. Analysis / evaluation (CPU-only — reproduce every table & figure)
These read cached prediction CSVs; no GPU or multiprocessing; run as-is.

| Result | Command |
|---|---|
| Main matched-compute comparison (PRISM 6/10, DeLong/McNemar) | `python scripts/full_metric_comparison.py --deepphys100` |
| Subgroup / fairness (WHO age band + recording site) | `python scripts/subgroup_fairness.py --deepphys100` |
| Decision-curve / net benefit | `python scripts/decision_curve.py --deepphys100` |
| Calibration & abstention diagnostics (ECE, coverage, risk-coverage, triage) | `python scripts/calibration_diagnostics.py` |
| E7 fix: age-conditional conformal + straddle triage (OLD vs NEW; `--write` emits `*_recal.csv`) | `python scripts/recalibrate_triage.py [--write]` |
| Framing head-swap (architecture-agnostic; reg vs ROCKET per backbone) | `python scripts/framing_headswap.py --matched` |
| Number reconciliation (canonical PRISM numbers on 332 vs 298 bases) | `python scripts/reconcile_numbers.py` |
| External validation (NCHSDB): framing / degradation sweep / robustness / figure | `python scratchpad/nchsdb_framing.py`, `nchsdb_final.py`, `nchsdb_robustness.py`, `make_fig_extval.py` |

Deployable E7 fix is wired into `scripts/evaluate_prism_breath.py` (opt-in, gated on
`src/config/age_conditional_conformal.json`) via `recalibrate_triage.apply_deploy_triage`.

**NOTE — NCHSDB scripts** currently live in the session scratchpad; for release move them into
`scripts/nchsdb/` (`nchsdb_edf.py`, `nchsdb_framing.py`, `nchsdb_degrade_sweep.py`, `nchsdb_final.py`,
`nchsdb_robustness.py`, `make_fig_extval.py`). NCHSDB access = PhysioNet credentialed + session-cookie.

## B. Training / extraction (OPTIONAL polish — GPU + multiprocessing, run externally)
**None of these change the headline results, which are locked.** Task #24 is resolved by DROP
(the 35 contaminated val signals were deleted; conformal uses the k-fold LOFO method, §A). B1–B4 are
refinements for a stronger paper, not gates. They require DataLoader/coarse2fine multiprocessing,
which the sandbox blocks (`WinError 5`); launch externally like DeepPhys-100ep. Each is multi-hour.
NB: B1 (`build_learned_flow_signals.py`) drives pyflow COARSE2FINE, which can be slow/fragile — run
it isolated; if it hangs at frame 0 with 0% CPU it has crashed, retry or skip (it's optional).

1. **Finish the val-contamination repair (Task #24).** The 35 v3-contaminated val flow signals were
   deleted; re-extract the learned versions (restores `data/pedrr_signals_learned` to 332):
   ```
   python scripts/build_learned_flow_signals.py
   ```
   Then refresh conformal/coverage: `python scripts/recalibrate_triage.py --write`.

2. **Leakage-free EfficientPhys control (B2) — script written & syntax-checked** (`build_efficientphys_flow_cleanfold.py`,
   mirrors the TSCAN clean-fold, uses the EfficientPhys-aware extract_split). Per fold: train
   EfficientPhys on fold-train, extract fold-test -> zero flow leakage.
   ```
   set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
   python scripts/build_efficientphys_flow_cleanfold.py        # -> data/pedrr_signals_learned_clean
   # then PRISM k-fold on the clean flow (--signal-dir data/pedrr_signals_learned_clean,
   #   output outputs/paper_runs/pedrr_kfold_learned_clean), then the CPU comparison glue:
   python scripts/leakage_control.py     # leaky-vs-clean dAUROC (leakage delta); expected ~-0.014
   ```
   `leakage_control.py` is CPU-only (runnable here) and self-guards: it prints the produce-it recipe
   until the clean k-fold exists, then reports PRISM AUROC leaky vs clean + subject-bootstrap CI + DeLong.

3. **Modern baseline: PhysFormer (B3).** PhysFormer is an RGB video transformer (its own trainer,
   loss, and `(B,3,T,H,W)` clip input) — it does NOT fit the frame-wise air-400 harness that the 4
   optical-flow extractors use, so run it in its own repo and fold it in with the CPU glue
   (`scripts/physformer_to_csv.py`, written + compiled):
   ```
   # a) git clone https://github.com/ubicomp-lab/rPPG-Toolbox   (tested PhysFormer + PhysFormerTrainer)
   # b) preprocess the pedrr videos to rPPG-Toolbox RGB chunks; train PhysFormer per fold on the
   #    kfold_learned train subjects; infer RR (FFT of the rPPG output) for the held-out test subjects
   # c) save {subject_id_or_subject_video: pred_rr_bpm} -> outputs/physformer_preds.json
   python scripts/physformer_to_csv.py    # CPU (runnable in-session) -> pedrr_kfold_physformer/fold*/...
   # d) add "PhysFormer": "outputs/paper_runs/pedrr_kfold_physformer" to full_metric_comparison.RUNS; re-run
   ```
   Real work = (b): RGB preprocessing + 5-fold training on your GPU. Steps (c)/(d) are CPU glue.
   Expected: PhysFormer's regression-then-threshold loses to PRISM's classification like the others,
   *reinforcing* the architecture-agnostic framing (E2). MTTS-CAN/PhysNet add little (older/near-dup of TS-CAN).

4. **Refresh ablations (B4)** — variants confirmed in `train_prism_breath.py --model-variant`
   (prism_shared_trunk / prism_no_scan / prism_no_spectral / prism_no_codebook / prism_no_prototypes /
   airflow_compact); `run_prism_breath_study.py` iterates them. Re-run on the current signals, then
   regenerate `tab:abl`:
   ```
   set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
   python scripts/run_prism_breath_study.py --signal-dir data/prism_breath_signals --output-root outputs/prism_breath_study --skip-existing
   ```

## C. Session-environment constraints (why B must run externally)
- No multiprocessing worker spawn (`WinError 5 DuplicateHandle`) — blocks coarse2fine + DataLoader workers.
- Long background jobs are killed; only externally-launched processes (e.g. DeepPhys-100ep) survive.
- Process kill is restricted; `/RU SYSTEM` scheduled-task creation is Access-denied.
- Even a detached `schtasks` job fails: it intermittently clears `WinError 5` but then the coarse2fine
  optical-flow Pool **stalls at 0** (workers can't do handle IPC) or dies on a console event.
- **=> Run section B in an INTERACTIVE terminal you control (RDP/local), exactly like DeepPhys-100ep.**
  If you hit `forrtl: error (200) ... window-CLOSE`, `set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` first.
  A ready wrapper with that fix already set: `C:\Users\rimon\prism_reextract.bat` (job B1).
