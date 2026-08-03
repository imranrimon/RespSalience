# PRISM-Breath Pipeline

## 1. Build Canonical Manifests
```bash
python scripts/build_respiratory_benchmark_manifest.py
```

Outputs:
- `data/organized/respiratory_benchmark_manifest.json`
- `data/organized/respiratory_benchmark_train.json`
- `data/organized/respiratory_benchmark_val.json`
- `data/organized/respiratory_benchmark_test.json`
- leave-one-dataset-out split files

## 2. Extract Signals
```bash
python scripts/extract_benchmark_signals.py --manifest data/organized/respiratory_benchmark_train.json --output data/prism_breath_signals
python scripts/extract_benchmark_signals.py --manifest data/organized/respiratory_benchmark_val.json --output data/prism_breath_signals
python scripts/extract_benchmark_signals.py --manifest data/organized/respiratory_benchmark_test.json --output data/prism_breath_signals
```

## 3. Pretrain Compatible Encoders
This is the new in-family SSL path. It replaces the older mismatch where SSL was trained on a different backbone and could not be transferred cleanly into PRISM-Breath.

```bash
python scripts/pretrain_prism_breath.py \
  --signal-dir data/prism_breath_signals \
  --train-manifest data/organized/respiratory_benchmark_train.json \
  --val-manifest data/organized/respiratory_benchmark_val.json \
  --output-dir outputs/prism_breath_pretrain
```

## 4. Fine-Tune a Paper Variant
Supported variants:
- `prism_breath`
- `prism_shared_trunk`
- `prism_no_scan`
- `prism_no_spectral`
- `prism_no_codebook`
- `prism_no_prototypes`
- `airflow_compact`

Example:
```bash
python scripts/train_prism_breath.py \
  --signal-dir data/prism_breath_signals \
  --train-manifest data/organized/respiratory_benchmark_train.json \
  --val-manifest data/organized/respiratory_benchmark_val.json \
  --output-dir outputs/prism_breath \
  --model-variant prism_breath \
  --encoder-pretrained outputs/prism_breath_pretrain/prism_breath_pretrained.pt
```

Default AIR-125 mitigation is now built into the training path:
- source-balanced sampling with an AIR-125 boost
- higher AIR-125 loss weight
- source/age-conditioned domain adapters in PRISM-Breath
- stronger AIR-125-targeted signal augmentation

Useful override flags:
- `--disable-source-balanced-sampling`
- `--air125-sampling-boost 1.25`
- `--air125-loss-weight 2.0`
- `--disable-air125-augmentation`
- `--air125-augment-prob 0.55`
- `--air125-augment-strength 1.75`
- `--disable-domain-adapter`

## 5. Evaluate on Validation
```bash
python scripts/evaluate_prism_breath.py \
  --signal-dir data/prism_breath_signals \
  --manifest data/organized/respiratory_benchmark_val.json \
  --checkpoint outputs/prism_breath/prism_breath_best.pt \
  --output-dir outputs/prism_breath_val_eval
```

The exported CSV is conformal-ready and includes:
- `name`, `paradigm`, `source`, `subject`
- `pred_bpm`, `gt_bpm`
- disagreement, interval width, prototype evidence
- CodePhys-style clean-codebook evidence

## 6. Calibrate Conformal Safety Intervals
```bash
python scripts/calibrate_conformal.py \
  --samples outputs/prism_breath_val_eval/prism_breath_predictions.csv \
  --output outputs/prism_breath_conformal/prism_qhat.csv \
  --use-all-samples \
  --group-by-source \
  --condition-on-disagreement \
  --split-unit subject
```

## 7. Evaluate on Test With Conformal Calibration
```bash
python scripts/evaluate_prism_breath.py \
  --signal-dir data/prism_breath_signals \
  --manifest data/organized/respiratory_benchmark_test.json \
  --checkpoint outputs/prism_breath/prism_breath_best.pt \
  --output-dir outputs/prism_breath_eval \
  --calibration-csv outputs/prism_breath_conformal/prism_qhat.csv \
  --condition-on-disagreement
```

## 8. Run the Full Study
```bash
python scripts/run_prism_breath_study.py \
  --signal-dir data/prism_breath_signals \
  --output-root outputs/prism_breath_study
```

## 9. One-Command Journal Suite
To launch or resume the full journal pipeline from PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_prism_breath_journal_suite.ps1
```

Useful stage controls:
- `-RunId journal_suite_resume`
- `-ForceMainEvalOnly`
- `-SkipManifestBuild`
- `-SkipSignalExtraction`
- `-SkipMainRuns`
- `-SkipAblations`
- `-SkipOfficialAirBaselines`
- `-SkipLegacyBaselines`
- `-SkipExpandedPaperBaselines`
- `-SkipPaperTables`

To recompute only PRISM-Breath validation, conformal calibration, and test
outputs from existing checkpoints in a prior run:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_prism_breath_journal_suite.ps1 `
  -RunId journal_suite_resume `
  -ForceMainEvalOnly
```

This mode:
- skips manifest rebuilding
- skips signal extraction
- skips PRISM pretraining and fine-tuning
- skips ablations and legacy baselines
- reruns PRISM validation, conformal calibration, uncalibrated test, and calibrated test

## 10. Official AIR-Family Baselines
To train and benchmark official AIR-family models on canonical AIR subject splits:
```bash
python scripts/run_official_air_baselines.py \
  --train-manifest data/organized/respiratory_benchmark_train.json \
  --val-manifest data/organized/respiratory_benchmark_val.json \
  --test-manifest data/organized/respiratory_benchmark_test.json \
  --output-root outputs/official_air_baselines \
  --baselines AIRFlowNet-Official,DeepPhys-Official,MTTS-CAN-Official,EfficientPhys-Official \
  --skip-existing
```

This wrapper:
- builds canonical AIR-only subject splits from the main manifests
- generates official `external/air-400` configs
- trains the official models with fixed seeds
- benchmarks the resulting checkpoints through the shared paper leaderboard path

The journal suite can run the same stage automatically unless `-SkipOfficialAirBaselines` is provided.

## 11. Expanded Paper Baseline Suite
To run the full old-repo paper baseline family plus official AIR-family baselines:
```bash
python scripts/run_expanded_paper_baselines.py \
  --train-manifest data/organized/respiratory_benchmark_train.json \
  --val-manifest data/organized/respiratory_benchmark_val.json \
  --test-manifest data/organized/respiratory_benchmark_test.json \
  --output-root outputs/expanded_paper_baselines \
  --preset paper_full \
  --skip-existing
```

Useful presets:
- `old_repo_all`: all legacy repo baselines except the official AIR-family additions
- `paper_modern`: focused modern paper subset including official AIRFlowNet/EfficientPhys/MTTS-CAN
- `official_air_only`: only official AIR-family baselines
- `paper_full`: everything currently available in the workspace

The journal suite can run the same stage automatically unless `-SkipExpandedPaperBaselines` is provided.

Note:
- `PhysFormer` and `FactorizePhys` are not yet vendored locally, so they are not part of the runnable preset list.
- The expanded paper benchmark defaults to offline-safe evaluation (`disable_vlm=True` inside the wrapper) unless you explicitly enable live VLM usage.

To resume a partially completed study without retraining finished variants or
rerunning completed evaluation stages, add:
```bash
python scripts/run_prism_breath_study.py \
  --signal-dir data/prism_breath_signals \
  --output-root outputs/prism_breath_study \
  --skip-existing
```

This will:
- pretrain encoder branches
- train each ablation variant
- evaluate on validation
- fit disagreement-conditioned conformal intervals
- evaluate on test
- save `outputs/prism_breath_study/ablation_summary.csv`

## 12. External-Method Baselines (CliffPhys / Mamba)

Two modern literature backbones are available as first-class `--model-variant`
values, so they train, evaluate, and calibrate through the exact same protocol as
PRISM-Breath (reliability fusion, age-conditioned triage, conformal intervals):

- `mamba` — a genuine selective state-space (Mamba-style) temporal backbone
  (`src/core/baseline_wrappers.py::SelectiveSSM`), the linear-time model behind
  PhysMamba / RhythmMamba, here applied to infant respiration.
- `cliffphys` — Clifford geometric-algebra layers over the Cl(2,0) geometric
  product (`CliffordLinear`), the core inductive bias of CliffPhys. This repo's
  benchmark path uses pre-extracted 1D signals, so it is a signal-space
  adaptation ("CliffPhys-Signal") that keeps the Clifford mixing while cross-modal
  depth/flow coupling is handled by the shared reliability fusion.

Both subclass `PrismBreathModel` and swap only the per-modality encoder; the
factory `build_prism_breath_model_from_config` dispatches them, so evaluation
needs no changes. PRISM-specific modules (expert routing, clean codebook,
prototypes, domain adapter) are disabled for these variants so the comparison
isolates the backbone.

```bash
python scripts/train_prism_breath.py --model-variant mamba \
  --signal-dir data/prism_breath_signals \
  --train-manifest data/organized/respiratory_benchmark_train.json \
  --val-manifest data/organized/respiratory_benchmark_val.json \
  --output-dir outputs/baselines/mamba
# evaluate_prism_breath.py reconstructs the variant from the checkpoint config automatically
python scripts/evaluate_prism_breath.py \
  --signal-dir data/prism_breath_signals \
  --manifest data/organized/respiratory_benchmark_test.json \
  --checkpoint outputs/baselines/mamba/prism_breath_best.pt \
  --output-dir outputs/baselines/mamba/test_eval
```

Or add them to a study sweep: `run_prism_breath_study.py --variants cliffphys mamba prism_breath ...`
(the encoders differ from the PRISM pretraining backbone, so any `--encoder-pretrained`
weights simply do not transfer — these baselines train from scratch, which is expected).
Smoke test: `python src/core/test_baseline_wrappers.py`.

## 13. Clinical-Rigor Analysis (Decision Curve + Subgroups)

`scripts/clinical_analysis.py` consumes the `prism_breath_predictions.csv` files
(across seeds) and produces the safety/decision artifacts the flagship paper needs
beyond raw MAE:

- **Decision Curve Analysis** (Vickers & Elkin net benefit) for the age-conditioned
  tachypnea decision — the model's risk score and an idealized PRISM triage policy
  (act on `TACHYPNEA_ALERT`, defer `UNCERTAIN_REVIEW` to a clinician) vs the
  "treat all" / "treat none" references, over a threshold-probability grid, with
  mean±std across seeds → `decision_curve.csv` (+ `--plot` → `decision_curve.png`).
- **Multi-seed subgroup tables** (mean±std) of MAE/RMSE, triage
  sensitivity/specificity, review rate and interval coverage, by **dataset**,
  **WHO age bin** (`<2m`, `2-12m`, `12-60m`, `>60m`) and **modality-disagreement
  bin** → `clinical_by_dataset.csv`, `clinical_by_age.csv`, `clinical_by_disagreement.csv`.

```bash
# Aggregate all seed runs under a study root (calibrated test predictions)
python scripts/clinical_analysis.py --run-root outputs/paper_runs/<RUN_ID> \
  --setting calibrated --output-dir outputs/paper_runs/<RUN_ID>/clinical --plot

# Same on the external OOD cohort (evaluate on the external manifest first)
python scripts/clinical_analysis.py --run-root outputs/paper_runs/<RUN_ID> \
  --setting external --output-dir outputs/paper_runs/<RUN_ID>/clinical_external --plot
```

## Reviewer-Facing Claims Supported by the Pipeline
- subject-disjoint evaluation
- cross-dataset protocol support
- AIRFlow-style compact small-data baseline
- expert-routing, scan, spectral, and codebook ablations
- compatible SSL pretraining and fine-tuning
- official AIRFlowNet/DeepPhys/MTTS-CAN/EfficientPhys baseline training wrappers
- expanded old-repo baseline suite runner for paper tables
- disagreement-conditioned conformal calibration
- prototype and counterfactual evidence outputs
