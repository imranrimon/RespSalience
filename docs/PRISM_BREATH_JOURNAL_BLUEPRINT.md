# PRISM-Breath Journal Blueprint

## 1. Flagship Paper Scope

The top-tier journal version should tell **one** story:

**contactless pediatric respiratory-rate estimation with safe age-conditioned tachypnea triage under dataset shift**

That scope is tight enough to be credible, clinically meaningful, and supported
by the current benchmark/data path.

Everything below should stay outside the main paper claims unless new labels are
collected:

- etiologic ARI diagnosis
- longitudinal deterioration forecasting
- physiologic state recognition
- audio-first disease reasoning
- generic foundation-model reasoning claims

## 2. Recommended Project Structure

The repo should be presented to reviewers and collaborators in four layers.

### A. Benchmark layer
- `scripts/build_respiratory_benchmark_manifest.py`
- `src/utils/respiratory_benchmark_registry.py`
- `scripts/extract_benchmark_signals.py`
- `src/core/respiratory_benchmark_dataset.py`

Purpose:
- define the paper cohort
- enforce canonical splits
- create leakage-safe signal files

### B. Method layer
- `src/core/prism_breath_model.py`
- `src/core/prism_breath_pretraining.py`
- `scripts/pretrain_prism_breath.py`
- `scripts/train_prism_breath.py`
- `scripts/evaluate_prism_breath.py`

Purpose:
- masked respiratory pretraining
- PRISM-Breath training
- explainable evaluation

### C. Study layer
- `scripts/run_prism_breath_study.py`
- `scripts/run_official_air_baselines.py`
- `scripts/run_expanded_paper_baselines.py`
- `scripts/calibrate_conformal.py`
- `scripts/run_legacy_canonical_benchmark.py`
- `scripts/run_fair_benchmark.py`
- `scripts/generate_prism_breath_paper_tables.py`

Purpose:
- ablations
- official AIR-family baseline training
- expanded baseline suite for paper comparison
- legacy baseline comparison
- calibration
- seed aggregation
- paper tables

### D. Paper layer
- `docs/PRISM_BREATH_REDESIGN.md`
- `docs/PRISM_BREATH_PIPELINE.md`
- `docs/LEGACY_BASELINE_INTEGRATION.md`
- `docs/PRISM_BREATH_JOURNAL_BLUEPRINT.md`

Purpose:
- explain the paper scope
- document the runbook
- align tasks, claims, and outputs

## 3. Goal / Objective / Task Mapping

### Goal
Build a clinically meaningful non-contact respiratory AI system for children.

### Objectives
1. Estimate pediatric RR robustly from contactless signals.
2. Convert RR into age-conditioned tachypnea triage.
3. quantify uncertainty and abstain when evidence is weak.
4. generalize across heterogeneous datasets and sensing conditions.
5. provide explanations that are useful to reviewers and clinicians.

### Proposed tasks
1. Canonical benchmark curation across AIR-125, AIR-400, Pediatric-RR.
2. In-family masked respiratory pretraining.
3. PRISM-Breath training and evaluation.
4. Internal ablation study.
5. Legacy baseline comparison.
6. Conformal calibration and selective triage analysis.
7. Case-based explainability analysis.

### Mapping verdict
- **Yes**, the proposed tasks map clearly to the goals and objectives **if** the
  paper is scoped to RR estimation + selective tachypnea triage.
- **No**, the mapping becomes unclear if the paper still tries to claim ARI
  diagnosis, longitudinal forecasting, and foundation-model clinical reasoning
  at the same time.

## 4. Technical Maturity Assessment

### Mature enough for a serious paper
- canonical benchmark manifests
- split-aware signal extraction
- PRISM-Breath training/evaluation path
- ablation infrastructure
- legacy baseline integration

### Implemented but not yet fully journal-mature
- conformal selective triage
- prototype/codebook explanation claims
- leave-one-dataset-out model selection
- automated paper-table generation

### Future work, not mature enough for main claims
- etiology/comorbidity reasoning
- longitudinal risk forecasting
- audio-diagnostic claims
- synthetic text-aligned foundation branches

## 5. How To Design This As a Top-Tier Journal Paper

### Core positioning
Do not sell this as a generic multimodal platform.

Sell it as:

**a reliability-aware respiratory benchmark and model for safe contactless pediatric tachypnea screening**

### What makes it top-tier
1. Clear clinical framing.
2. Canonical multi-dataset evaluation.
3. Strong baseline suite.
4. Robustness and abstention, not MAE alone.
5. Honest failure analysis.
6. Explainability tied to evidence, not just saliency.

### Required main contributions
1. A canonical pediatric respiratory benchmark with subject-disjoint evaluation.
2. A physiology-grounded selective model that estimates RR and can defer.
3. A rigorous comparison against classical, learned, and compact small-data baselines.
4. A rigorous comparison against official AIRFlowNet/DeepPhys/MTTS-CAN/EfficientPhys baselines where applicable.
5. A safety analysis showing what happens under calibration and dataset shift.

### Claims to avoid unless improved
- “deployment-ready triage”
- “robust uncertainty-aware clinical decision support”
- “all proposed modules consistently contribute”

## 6. Methodology Section Design

Recommended structure:

1. **Problem Setup**
   - define RR estimation and age-conditioned tachypnea triage
   - define `NORMAL_SAFE`, `TACHYPNEA_ALERT`, `UNCERTAIN_REVIEW`

2. **Datasets and Protocol**
   - AIR-125, AIR-400, Pediatric-RR
   - subject-disjoint train/val/test
   - mention leave-one-dataset-out as an external-style extension

3. **Signal Extraction**
   - explain depth / flow / audio surrogate signal generation
   - fixed-length standardized respiratory signals

4. **Respiratory Representation Pretraining**
   - RespMAE-style masked reconstruction
   - contrastive augmentation
   - same encoder family as the downstream model

5. **PRISM-Breath**
   - per-modality encoding
   - reliability-aware fusion
   - spectral + scan components
   - clean-breathing codebook
   - prototype evidence
   - selective triage outputs

6. **Training Objectives**
   - RR regression
   - triage losses
   - uncertainty / review loss
   - fusion consistency
   - codebook regularization

7. **Baselines and Ablations**
   - classical baselines
   - learned legacy baselines
   - AIRFlow-inspired compact baseline
   - PRISM-Breath internal ablations

8. **Evaluation Metrics**
   - MAE, RMSE
   - sensitivity, specificity
   - review rate
   - coverage
   - subgroup analyses

## 7. Results Analysis Section Design

Use this exact flow.

### 7.1 Main benchmark performance
Start with PRISM-Breath vs:
- Flow-Only
- Depth-Only
- Fused-VMD
- Contrastive-Finetune
- Transformer-Fusion
- DeepPhys
- AIRFlow-inspired compact baseline

Focus on:
- MAE
- RMSE
- overall ranking

### 7.2 Safety-aware operating points
Compare:
- uncalibrated PRISM-Breath
- calibrated PRISM-Breath

Report:
- sensitivity
- specificity
- review rate
- coverage
- mean qhat

Interpret whether calibration meaningfully improves safety or becomes too conservative.

### 7.3 Ablation study
Compare:
- `prism_breath`
- `prism_shared_trunk`
- `prism_no_scan`
- `prism_no_spectral`
- `prism_no_codebook`
- `prism_no_prototypes`
- `airflow_compact`

Interpret each component only if it helps consistently across seeds.

### 7.4 Subgroup analysis
Break down by:
- dataset
- age bin

This is where domain shift and age dependence should be analyzed.

### 7.5 Explainability and failure analysis
Show:
- modality weights
- disagreement
- prototype evidence
- codebook evidence
- modality-drop counterfactuals

Pick examples from:
- correct confident cases
- correct review cases
- major AIR-125 failure cases

### 7.6 Limitations
State explicitly:
- conservative calibration if applicable
- AIR-125 difficulty
- missing LODO val split
- incomplete external baseline support on long videos

## 8. Production Run Commands

### 8.1 Environment setup
```powershell
$ErrorActionPreference = "Stop"
Set-Location "G:\Respiratory Project"   # repository root

$PY = "C:\Users\rimon\anaconda3\envs\resp_env\python.exe"
$RUN = Get-Date -Format "yyyyMMdd_HHmmss"
$ROOT = "outputs\paper_runs\$RUN"
$SIG = "data\paper_signals"
$env:YOLO_CONFIG_DIR = "$PWD\.ultralytics"
$env:WANDB_MODE = "disabled"
$env:WANDB_SILENT = "true"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
New-Item -ItemType Directory -Force $env:YOLO_CONFIG_DIR, $ROOT, $SIG | Out-Null
Start-Transcript -Path "$ROOT\run.log"
```

### 8.2 Build manifests
```powershell
& $PY scripts/build_respiratory_benchmark_manifest.py `
  --base-dir data `
  --output-dir data\organized
```

### 8.3 Extract signals
```powershell
& $PY scripts/extract_benchmark_signals.py --manifest data\organized\respiratory_benchmark_train.json --output $SIG --config src\config\test_infer.yaml --checkpoint data\models\VIRENet_Coarse2fine_Body_best.pth
& $PY scripts/extract_benchmark_signals.py --manifest data\organized\respiratory_benchmark_val.json   --output $SIG --config src\config\test_infer.yaml --checkpoint data\models\VIRENet_Coarse2fine_Body_best.pth
& $PY scripts/extract_benchmark_signals.py --manifest data\organized\respiratory_benchmark_test.json  --output $SIG --config src\config\test_infer.yaml --checkpoint data\models\VIRENet_Coarse2fine_Body_best.pth
```

### 8.4 Main PRISM-Breath suite
```powershell
foreach ($seed in 42,52,62) {
  $S = "$ROOT\main_seed$seed"

  & $PY scripts/pretrain_prism_breath.py `
    --signal-dir $SIG `
    --train-manifest data\organized\respiratory_benchmark_train.json `
    --val-manifest data\organized\respiratory_benchmark_val.json `
    --output-dir "$S\pretrain" `
    --model-variant prism_breath `
    --seq-len 300 `
    --d-model 64 `
    --batch-size 16 `
    --epochs 100 `
    --lr 1e-3 `
    --contrastive-weight 0.15 `
    --seed $seed

  & $PY scripts/train_prism_breath.py `
    --signal-dir $SIG `
    --train-manifest data\organized\respiratory_benchmark_train.json `
    --val-manifest data\organized\respiratory_benchmark_val.json `
    --output-dir "$S\train" `
    --model-variant prism_breath `
    --seq-len 300 `
    --d-model 64 `
    --num-experts 3 `
    --num-prototypes 6 `
    --codebook-size 12 `
    --epochs 100 `
    --batch-size 16 `
    --lr 3e-4 `
    --weight-decay 1e-4 `
    --encoder-pretrained "$S\pretrain\prism_breath_pretrained.pt" `
    --freeze-encoder-epochs 2 `
    --modality-dropout-prob 0.1 `
    --seed $seed

  & $PY scripts/evaluate_prism_breath.py `
    --signal-dir $SIG `
    --manifest data\organized\respiratory_benchmark_val.json `
    --checkpoint "$S\train\prism_breath_best.pt" `
    --output-dir "$S\val_eval" `
    --batch-size 16 `
    --disagreement-bins 0.8,0.9 `
    --counterfactual-limit 50

  & $PY scripts/calibrate_conformal.py `
    --samples "$S\val_eval\prism_breath_predictions.csv" `
    --output "$S\conformal\prism_qhat.csv" `
    --alpha 0.1 `
    --use-all-samples `
    --group-by-source `
    --condition-on-disagreement `
    --split-unit subject `
    --disagreement-bins 0.8,0.9

  & $PY scripts/evaluate_prism_breath.py `
    --signal-dir $SIG `
    --manifest data\organized\respiratory_benchmark_test.json `
    --checkpoint "$S\train\prism_breath_best.pt" `
    --output-dir "$S\test_eval_uncalibrated" `
    --batch-size 16 `
    --disagreement-bins 0.8,0.9 `
    --counterfactual-limit 100

  & $PY scripts/evaluate_prism_breath.py `
    --signal-dir $SIG `
    --manifest data\organized\respiratory_benchmark_test.json `
    --checkpoint "$S\train\prism_breath_best.pt" `
    --output-dir "$S\test_eval" `
    --batch-size 16 `
    --calibration-csv "$S\conformal\prism_qhat.csv" `
    --condition-on-disagreement `
    --disagreement-bins 0.8,0.9 `
    --counterfactual-limit 100
}
```

### 8.5 PRISM-Breath ablations
```powershell
foreach ($seed in 42,52,62) {
  & $PY scripts/run_prism_breath_study.py `
    --signal-dir $SIG `
    --train-manifest data\organized\respiratory_benchmark_train.json `
    --val-manifest data\organized\respiratory_benchmark_val.json `
    --test-manifest data\organized\respiratory_benchmark_test.json `
    --output-root "$ROOT\study_seed$seed" `
    --variants prism_breath prism_shared_trunk prism_no_scan prism_no_spectral prism_no_codebook prism_no_prototypes airflow_compact `
    --seq-len 300 `
    --d-model 64 `
    --batch-size 16 `
    --pretrain-epochs 100 `
    --train-epochs 100 `
    --disagreement-bins 0.8,0.9 `
    --seed $seed
}
```

### 8.6 Stable legacy baseline suite
`TSCAN` is excluded from the default production suite until its memory path is fixed.

```powershell
foreach ($seed in 42,52,62) {
  & $PY scripts/run_legacy_canonical_benchmark.py `
    --signal-dir $SIG `
    --train-manifest data\organized\respiratory_benchmark_train.json `
    --val-manifest data\organized\respiratory_benchmark_val.json `
    --test-manifest data\organized\respiratory_benchmark_test.json `
    --output-root "$ROOT\legacy_seed$seed" `
    --baselines "Flow-Only,Depth-Only,Fused-VMD,Contrastive-Finetune,Transformer-Fusion,DeepPhys" `
    --ssl-epochs 100 `
    --finetune-epochs 100 `
    --batch-size 16 `
    --ssl-lr 1e-3 `
    --finetune-lr 3e-4 `
    --weight-decay 1e-4 `
    --seq-len 300 `
    --d-model 64 `
    --seed $seed `
    --disable-vlm
}
```

### 8.7 Aggregate paper tables
```powershell
& $PY scripts/generate_prism_breath_paper_tables.py `
  --run-root $ROOT `
  --output-dir "$ROOT\paper_tables" `
  --best-seed 42
```

### 8.8 Close the session
```powershell
Stop-Transcript
```

## 9. What Would Make This Journal-Ready

Minimum:
- PRISM-Breath consistently beats stable baselines.
- at least 3-seed reporting.
- internal ablations support the main architectural claims.
- AIR-125 failure analysis is explicit.

Stronger:
- add a clean LODO validation protocol.
- fix memory-safe external baselines or exclude them transparently.
- add a first-class qualitative case-study figure set.
- if calibration remains too conservative, present it as a limitation rather than a headline contribution.
