# Reproducing the results

This document maps every table and claim in the manuscript to the artifact and command that produce it.
It is written so that a reader with the datasets can regenerate the paper, and so that a reader without
them can still see exactly what was run.

## 0. Fast check: does the paper match its own artifacts?

```bash
python scripts/verify_manuscript_claims.py
```

Re-derives every checkable number from the stored per-fold predictions (not from cached summary CSVs)
and prints it beside the value in the manuscript. Exits non-zero on any mismatch.

Current status: **53 claims checked, 53 match, 0 mismatches.**

Run this after any change to the manuscript or to an analysis script. It has caught, among other
things, two tables reported on different cohort definitions and a malformed artifact whose columns were
shifted one place left of their headers.

## 1. Datasets

None are redistributed. Obtain them from their providers first.

| Dataset | Used for | Location expected by the code |
|---|---|---|
| Pediatric-RR-332 | primary cohort, all headline results | `data/Video Dataset for Respiratory Rate Counting in Children/` |
| NCH Sleep DataBank 3.1.0 | external replication | `data/nchsdb/Sleep_Data/*.edf` |
| AIR-125 / AIR-400 | cross-dataset generalization | `data/AIR_125/`, `data/AIR_400/` |
| Children radar set | radar future-work numbers | `data/child_radar/Children Dataset/` |
| Neonatal thorax phantom | radar future-work numbers | `data/radar_neonatal_sim/` |

**Reference standard note.** The Pediatric-RR rate distributed in the Stata file is a multi-reader video
expert panel consensus, not a single bedside count. `RR_method = Manual` is constant across all 332
records and denotes the recording condition. The evidence is in the values themselves: 40.4% are exact
half-integers and none are quarter-values, which is what a mean of two integer counts produces and a
single count cannot.

## 2. Signal extraction

Split first, then extract — the ordering is deliberate and leakage-safe.

```bash
python scripts/build_respiratory_benchmark_manifest.py         # subject-disjoint deterministic splits
python scripts/build_learned_flow_signals.py                   # learned optical flow -> per-video .npz
```

Extraction gotcha, recorded because it cost a day: `pyflow` segfaults on crops of 24x24 or smaller, with
no traceback. A mid-clip YOLO detection failure can produce such a crop. Signals are upscaled above that
floor before flow is computed.

## 3. Training

### The proposed model (RespSalience)

```bash
for S in 0 1 2; do
  python scripts/train_resp_salience_ssl.py --seed $S --kfold pedrr_kfold_main332 --tag _ssl332
done
```

Ablation (self-supervised pretraining removed):

```bash
for S in 0 1 2; do
  python scripts/train_resp_salience_ssl.py --seed $S --no-ssl --kfold pedrr_kfold_main332 --tag _nossl332
done
```

### The regression baseline (PRISM-Breath)

```bash
python scripts/run_prism_kfold.py --signal-dir data/pedrr_signals_learned --tag main332 \
    --epochs 100 --pretrain-epochs 100 --freeze-encoder-epochs 2
```

Add `--selection-metric auroc` for the decision-aware selection arm used in `tab:selmatrix`.

### Published extractors

```bash
python scripts/run_efficientphys_cv.py
python scripts/run_extractor_cv.py --model-name DeepPhys      # also TSCAN, VIRENet
```

## 4. Result-by-result map

| Paper element | Command | Artifact |
|---|---|---|
| `tab:main` | `python scripts/verify_manuscript_claims.py` | per-fold prediction CSVs |
| `tab:headline` | training commands above | `outputs/paper_runs/pedrr_kfold_salience_ssl332_seed*` |
| `tab:selmatrix` | `python scripts/selection_matrix.py` | `*_aurocsel` run roots |
| `tab:sota` | `python scripts/full_metric_comparison.py --deepphys100 --prism-run outputs/paper_runs/pedrr_kfold_main332` | — |
| `tab:abl` | `python scripts/regenerate_supplementary_tables.py --prism-run pedrr_kfold_main332` | `manuscript/supplementary_tables_regenerated.tex` |
| `tab:ceiling` | `python scripts/label_noise_ceiling.py` | simulation, seeded |
| `tab:conformal` | `python scripts/age_conditional_conformal.py --run outputs/paper_runs/pedrr_kfold_main332` | `age_conditional_conformal.json` |
| `sec:age` read-out | `python scripts/operating_point_readout.py --run-root outputs/paper_runs/pedrr_kfold_salience_ssl332_seed0` | `outputs/operating_point_*` |
| `sec:age` deployed rule | `python scripts/hybrid_readout.py` | — |
| `sec:extval` | `python scripts/nchsdb_select_records.py` → download → `python scripts/nchsdb_extract.py --cohort outputs/nchsdb_cohort.csv` → `python scripts/nchsdb_headswap.py` | `outputs/nchsdb_*` |
| `sec:selmatrix` audit | `python scripts/identify_baseline_checkpoint.py --model-name EfficientPhys --fold 0` | — |
| Subgroup / fairness | `python scripts/subgroup_fairness.py --deepphys100 --prism-run outputs/paper_runs/pedrr_kfold_main332` | — |
| Calibration, conformal | `python scripts/calibration_diagnostics.py --deepphys100` | — |
| Radar (future work) | `python scripts/radar_tof_rr.py --rec-dir <recording>`, `python scripts/radar_children_rr.py` | `outputs/radar_children.json` |

## 5. External replication (NCH Sleep DataBank)

Requires PhysioNet credentialed access.

```bash
python scripts/nchsdb_select_records.py --n 70          # writes outputs/nchsdb_urls.txt
# download those 70 records (~40 GB); do NOT mirror the whole dataset (~2 TB)
wget -c -np --user <USER> --ask-password -P data/nchsdb/Sleep_Data/ -i outputs/nchsdb_urls.txt

python scripts/nchsdb_extract.py --cohort outputs/nchsdb_cohort.csv
python scripts/nchsdb_headswap.py                       # deterministic, matches the original method
python scripts/nchsdb_framing.py --seeds 5              # learned heads; MUST be seed-averaged
```

Two things that will otherwise mislead you:

- **Seed-average the learned comparison.** Single runs at the same nominal seed produced deltas of
  −0.062 and +0.108 for the same configuration. One run of this size cannot support a conclusion.
- **Two channel montages ship in release 3.1.0.** Some records use `Resp Thoracic` / `Resp Airflow` /
  `Resp PTAF`, others `Resp Chest` / `Resp Flow` / `XFlow`. Matching only the first silently discards
  22% of the cohort. End-tidal CO₂ carries a corrupt digital calibration range in a majority of records
  and is unusable, so the reference is airflow-only.

## 6. Known artifact defects

Recorded so that nobody re-derives a wrong number from them:

- `outputs/objective_comparison/objective_comparison_summary.csv` has 12 header names for 13 columns.
  Every field sits one place left of its label; the true AUROC is under the column labelled `RMSE`.
  Read it positionally.
- Older run directories carry absolute `f:/` paths in logs from an earlier machine. Ignore them.
- `pedrr_kfold_abl_no_codebook` is the codebook ablation, correctly named. It is **not** the main run,
  despite an earlier note in the project history claiming so.

## 7. Environment

```bash
conda env create -f environment.yml     # authoritative pinned spec; note: UTF-16 encoded
conda activate resp_env
```

Hardware used: single NVIDIA GPU, 15 GB host RAM. Host memory, not the GPU, is the binding constraint
for the extraction and re-selection sweeps — run them one at a time, one fold per process.
