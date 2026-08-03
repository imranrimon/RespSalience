# PRISM-Breath - Consolidated Results Tables
_Auto-generated from prediction CSVs; see PRISM_BREATH_PAPER_POSITIONING.md for narrative._

## Table 1. Clinical operating points (3-seed averaged, subject bootstrap 95% CI)

| Cohort | n | pos | MAE | Sens [95% CI] | Spec | AUROC [95% CI] |
|---|--:|--:|--:|--:|--:|--:|
| Full cohort | 332 | 86 | 6.05 | 0.64 [0.54,0.74] | 0.91 | 0.892 [0.848,0.932] |
| **>=2mo (screen)** | 303 | 78 | 5.50 | 0.71 [0.60,0.80] | 0.91 | 0.919 [0.881,0.950] |
|   2-12mo | 143 | 49 | 7.13 | 0.63 [0.49,0.76] | 0.93 | 0.888 [0.823,0.941] |
|   12-60mo | 160 | 29 | 4.05 | 0.83 [0.68,0.96] | 0.89 | 0.950 [0.914,0.978] |
| <2mo (DEFERRED) | 29 | 8 | - | referred to clinician | - | - |

## Table 2. PRISM vs published extractors (matched 5-fold CV, 298 subjects)

| Method | RR-MAE | Tachypnea AUROC | vs PRISM (DeLong) |
|---|--:|--:|---|
| **PRISM (ours)** | 6.19 | **0.879** | - |
| EfficientPhys | 6.10 | 0.755 | p=0.001 |
| TSCAN | 6.15 | 0.726 | p<0.001 |
| VIRENet | 5.91 | 0.760 | p=0.001 |
| DeepPhys | 5.61 | 0.762 | p=0.002 |

## Table 3. Ablations & lean model (pooled 5-fold)

| Variant | MAE | Sens | AUROC |
|---|--:|--:|--:|
| Full PRISM | 6.19 | 0.66 | 0.879 |
| **Lean** (drop prototypes+codebook+het) | 6.17 | 0.64 | 0.867 |
| - prototypes | 6.08 | 0.66 | 0.875 |
| - codebook | 6.22 | 0.64 | 0.882 |
| - scan block | 6.00 | 0.60 | 0.874 |
| - spectral features | 6.39 | 0.58 | 0.868 |
| - separate trunks | 6.28 | 0.60 | 0.884 |
| - SSL pretrain | 6.43 | 0.55 | 0.862 |

## Table 4. Extractor-robustness & fusion isolation (pooled 5-fold)

| Configuration | MAE | Sens | AUROC | Note |
|---|--:|--:|--:|---|
| PRISM (EfficientPhys flow) | 6.19 | 0.66 | 0.879 | reference |
| Flow-only (depth/audio off) | 6.07 | 0.60 | 0.860 | beats EfficientPhys AUROC p=0.002 |
| TSCAN-flow (leaky) | 5.23 | 0.74 | 0.911 | INFLATED by extractor leakage |
| TSCAN-flow (leakage-free) | 5.86 | 0.59 | 0.897 | ~=PRISM: extractor-robust |
| SNR-gated fusion | 6.24 | 0.63 | 0.881 | de-collapses weights; null clinically |

## Table 5. Radar validation & cross-dataset generalization

| Evaluation | Metric | Note |
|---|---|---|
| Radar, thorax phantom (20-80 BPM) | **RR-MAE 0.00, r=1.00** | recovers neonatal band RGB can't (~26-35) |
| Radar, **50 real children** vs clinical ref | **RR-MAE 1.76, r=0.96** (86% <=3 BPM) | validated on real cohort |
| PRISM in-domain AIR-400 | RR-MAE 2.93 | ~= AIRFlowNet 2.9 (infant SOTA) |
| PRISM in-domain AIR-125 | RR-MAE 4.02 | (5 subjects) |
| PRISM zero-shot AIR-125/400 | 4.83 / 4.92 | learned flow transfers |
| Hand-crafted flow (v3) zero-shot | 11.3 / 11.8 | collapses -> learned-flow justified |

## Table 6. Clinical utility, calibration, equity (pooled)

| Property | Result |
|---|---|
| Calibration | ECE 0.064, Brier 0.110 |
| Decision curve | net benefit > treat-all/none at every threshold |
| Equity by sex | M AUROC 0.890 vs F 0.871; MAE gap n.s. |
| Tachypnea vs difficulty-breathing | true-RR AUROC 0.59; PRISM 0.54 (weak proxy) |
| Skin-tone equity | not auditable (single-population) |