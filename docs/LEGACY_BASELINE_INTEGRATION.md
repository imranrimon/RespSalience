# Legacy Baseline Integration

This benchmark keeps the paper story centered on one task:

- contactless pediatric respiratory-rate estimation
- age-conditioned tachypnea triage
- uncertainty-aware review

Legacy baselines are integrated as comparison models, not as separate project
tracks. They now plug into the same canonical manifests used by PRISM-Breath.

## Strong legacy baseline set

- `Flow-Only`
- `Depth-Only`
- `Fused-VMD`
- `Contrastive-Finetune`
- `Transformer-Fusion`
- `DeepPhys`
- `TSCAN`

## Fairness rule

All learned legacy baselines must be trained on:

- `data/organized/respiratory_benchmark_train.json`
- `data/organized/respiratory_benchmark_val.json`

and evaluated on:

- `data/organized/respiratory_benchmark_test.json`

This avoids comparing PRISM-Breath against older checkpoints trained on
different ad hoc splits.

## New integration points

- `scripts/run_ablation_study.py --manifest ...`
  Runs classical and external legacy baselines on an exact canonical split.
- `scripts/run_fair_benchmark.py --manifest ...`
  Exports a comparable leaderboard from that canonical split.
- `scripts/train_ssl_benchmark.py --train-manifest ... --val-manifest ...`
  Retrains legacy SSL encoders on canonical train/val signals.
- `scripts/train_finetune.py --train-manifest ... --val-manifest ...`
  Fine-tunes legacy transformer baselines on canonical train/val signals.
- `scripts/run_legacy_canonical_benchmark.py`
  One-command runner for the paper-grade legacy benchmark.

## Recommended paper tables

1. Main benchmark table
   Compare `PRISM-Breath`, `airflow_compact`, and the strong legacy baseline set.

2. PRISM-Breath ablation table
   Compare `prism_shared_trunk`, `prism_no_scan`, `prism_no_spectral`,
   `prism_no_codebook`, and `prism_no_prototypes`.

3. Safety table
   Compare uncalibrated vs conformal-calibrated PRISM-Breath operating points.
