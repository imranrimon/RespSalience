# Camera-Ready Experiment Runbook

## One-command full run

```bash
python scripts/run_camera_ready_pipeline.py --stages all --benchmark-seeds 42,52,62 --resume
```

`run_camera_ready_pipeline.py` disables live VLM inference during benchmark stages by default (to avoid slow sequential GPU pipeline calls).  
If you need live VLM text reasoning during benchmark, add:

```bash
--enable-vlm-benchmark
```

This executes:
1. Signal extraction
2. SSL pretraining (`msm`, `sae`, `contrast`)
3. SSL fine-tuning
4. Semi-supervised training (`SPMat-BT`, `VLM-Gated`)
5. Core benchmark (methods across datasets)
6. Modality ablation benchmark
7. Table generation
8. Artifact packaging

## Stage-wise runs (recommended for long jobs)

### Stage A: Data + training

```bash
python scripts/run_camera_ready_pipeline.py \
  --stages extract_signals,ssl_pretrain,ssl_finetune,semi_train \
  --benchmark-seeds 42 \
  --resume
```

### Stage B: Camera-ready benchmarks + tables

```bash
python scripts/run_camera_ready_pipeline.py \
  --stages benchmark_core,benchmark_modality,tables,package \
  --benchmark-seeds 42,52,62 \
  --resume
```

## Dataset/modality control

- Restrict datasets:

```bash
python scripts/run_camera_ready_pipeline.py --datasets AIR-125,AIR-400 --stages benchmark_core
```

- Quick smoke run:

```bash
python scripts/run_camera_ready_pipeline.py --stages benchmark_core --limit 5 --dry-run
```

## Outputs

Each run creates:

- `outputs/camera_ready_runs/<run_name>/run_manifest.json`
- `outputs/camera_ready_runs/<run_name>/leaderboards/*.csv`
- `outputs/camera_ready_runs/<run_name>/tables/*`
- `outputs/camera_ready_runs/<run_name>/run_summary.md`
