# PRISM-Breath — Non-Contact Pediatric Respiratory Screening

Research code for **non-contact respiratory-rate (RR) estimation and tachypnea screening in infants**
from RGB video (optical flow), monocular depth, and audio. Respiration lives in the 0.15–1.5 Hz band
(~9–90 BPM). The clinical endpoint is **tachypnea by WHO age band** (≥60 BPM <2mo, ≥50 2–12mo,
≥40 12–60mo).

> **Scope (honest):** validated as a tachypnea screen for infants **≥2 months** (Sens 0.71, AUROC 0.919,
> 3-seed, calibrated), with **mandatory clinician referral for <2 months** where RGB extraction is
> provably unreliable. Tachypnea is a screening sign, not a disease diagnosis — see limitations in
> `docs/PRISM_BREATH_REVIEWER_CRITIQUE.md`.

## Headline results
- **PRISM beats 4 published extractors on tachypnea AUROC** (0.879 vs 0.73–0.76, all p≤0.002) at
  comparable RR-MAE — *raw MAE ≠ clinical utility*.
- **Neonatal fix:** RGB flow physically saturates ~35 BPM for true 60–80; **radar recovers it** —
  MAE 0.00 on a thorax phantom (20–80 BPM) and **MAE 1.76 (r=0.96) on 50 real children**.
- **Lean model** (drop prototype bank + codebook + het head) = full PRISM (all n.s.) — an
  ablation-justified, Occam-clean design. SSL pretraining *is* load-bearing (Sens +0.116).
- Calibrated (ECE 0.064), positive net benefit on decision-curve analysis, no sex disparity.

See `docs/PRISM_BREATH_RESULTS_TABLES.md` (6 tables) and `docs/PRISM_BREATH_PAPER_POSITIONING.md`.

## Repository structure
```
prism-breath/
├── src/                     # importable package (prism_breath)
│   ├── core/                # models: prism_breath_model, fusion, signal_processor, diagnosis...
│   ├── utils/               # dataset registry, ground-truth, dataset paths
│   ├── config/              # YAML configs (root-relative paths by design)
│   └── ui/  api/            # Streamlit dashboard, FastAPI
├── scripts/                 # run-from-root CLIs (paths are project-root-relative — run from repo root)
├── tests/                   # standalone self-checking test_*.py (no pytest suite)
├── docs/                    # design docs, runbooks, reviewer critique, RESULTS_TABLES, POSITIONING
├── manuscript/             # IEEE MTT paper (Overleaf-ready): prism_breath.tex + .bib + 600-DPI figs
├── tools/                   # one-off utilities
├── data/ outputs/ external/ Paper/  # gitignored (datasets, checkpoints, vendored 3rd-party, copyrighted ref PDFs)
├── pipeline.py              # single-video CLI / dashboard / vitals entry point
├── pyproject.toml  requirements.txt  environment.yml   # environment.yml is authoritative (conda resp_env)
```

### Scripts by category (organizing scheme — paths kept flat by design; see below)
Scripts stay flat in `scripts/` **on purpose**: they resolve the project root via
`Path(__file__).resolve().parents[1]` and import each other via `sys.path`, so moving them into
subfolders would break the pipeline. They are organized *logically* as:
- **Manifests/data:** `build_respiratory_benchmark_manifest`, `make_kfold_manifests`
- **Extraction:** `extract_benchmark_signals`, `build_learned_flow_signals`, `build_tscan_flow_signals`,
  `train_learned_extractor_pedrr`, `radar_tof_rr`
- **Training:** `pretrain_prism_breath`, `train_prism_breath`
- **Eval/CV:** `evaluate_prism_breath`, `run_efficientphys_cv`, `run_extractor_cv`, `reseed_deepphys_fold3`,
  `calibrate_conformal`, `age_conditional_conformal`
- **Analysis:** `clinical_endpoint_eval`, `compare_clinical_models`, `generate_prism_breath_paper_tables`

## Getting started
```bash
conda env create -f environment.yml && conda activate resp_env   # authoritative, pinned (torch 2.8 cu129)
# Single-video CLI
python pipeline.py --mode cli --video <path> --age <months>
# PRISM-Breath pipeline (run from repo root):
python scripts/build_respiratory_benchmark_manifest.py
python scripts/extract_benchmark_signals.py --manifest <split.json> --output data/prism_breath_signals
python scripts/pretrain_prism_breath.py --signal-dir data/prism_breath_signals ...
python scripts/train_prism_breath.py --model-variant prism_lean ...      # prism_lean = the shippable model
python scripts/evaluate_prism_breath.py --checkpoint <best.pt> --manifest <split.json> ...
```
Full runnable flags: `docs/PRISM_BREATH_PIPELINE.md`. Dashboard: `streamlit run src/ui/dashboard.py`.

## Notes
- **Run scripts from the repo root** — configs use root-relative paths.
- `external/` is vendored third-party (air-400, Infant-Respiration-Estimation, pyflow) — **do not edit**.
- `data/`, `outputs/`, `external/` are gitignored; weights (`*.pth/*.pt`) are not versioned.
