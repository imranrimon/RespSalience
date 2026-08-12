# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is
Non-contact **pediatric respiratory monitoring** research code: estimate respiratory rate (BPM)
and flag Acute Respiratory Infections (ARI, via tachypnea) in infants from RGB video, monocular
depth, optical flow, and audio. Respiration lives in the 0.15–1.5 Hz band (~9–90 BPM). **Two
systems coexist and meet at signal extraction:**
1. **Live inference / demo** — `pipeline.py` → `RespiratoryController`, Streamlit dashboard, FastAPI.
2. **PRISM-Breath benchmark/paper pipeline** — manifest-driven, multi-dataset training/eval/ablation for the journal paper (the current focus).

## Environment
- Conda env `resp_env`, Python 3.13. `environment.yml` is the authoritative pinned spec; `requirements.txt` is a minimal subset. GPU/CUDA build. Note the file is UTF-16 encoded.
- Conda env is `resp_env` (matches `environment.yml`; `train_all.bat` now activates the same name).

## Commands

### Live inference / demo
- Full legacy experiment pipeline: `python scripts/run_full_system.py` (or `run_full_system.bat`). Stages: signal extraction → SSL pretrain (msm/sae/contrast) → semi-SL (VLM labels) → supervised finetune → ablation eval → results tables → launch dashboard. **Resume-safe**: each stage skips if its output model/`.npz` already exists.
- Dashboard only: `streamlit run src/ui/dashboard.py`
- Single-video CLI: `python pipeline.py --mode cli --video <path> --age <months>` (also `--mode dashboard|vitals`)

### PRISM-Breath pipeline (full flags in docs/PRISM_BREATH_PIPELINE.md)
Ordered stages — note the **split-then-extract** ordering is deliberate (leakage-safe):
1. `python scripts/build_respiratory_benchmark_manifest.py` → `data/organized/respiratory_benchmark_{train,val,test}.json` (subject-disjoint, deterministic SHA-256 splits)
2. `python scripts/extract_benchmark_signals.py --manifest <split.json> --output data/prism_breath_signals` → per-video `.npz`
3. `python scripts/pretrain_prism_breath.py --signal-dir data/prism_breath_signals ...`
4. `python scripts/train_prism_breath.py --model-variant prism_breath ...` (variants: `prism_breath`, `prism_shared_trunk`, `prism_no_scan`, `prism_no_spectral`, `prism_no_codebook`, `prism_no_prototypes`, `airflow_compact`)
5. `python scripts/evaluate_prism_breath.py --checkpoint <best.pt> --manifest <split.json> ...`
6. `python scripts/calibrate_conformal.py --samples <val_predictions.csv> ...`
- One-command full study: `python scripts/run_prism_breath_study.py --signal-dir data/prism_breath_signals --output-root outputs/prism_breath_study --skip-existing`
- Full resume-safe journal suite (Windows): `powershell -ExecutionPolicy Bypass -File scripts/run_prism_breath_journal_suite.ps1` (stage-skip flags: `-ForceMainEvalOnly`, `-SkipAblations`, `-RunId <id>`, …)
- Paper baselines: `scripts/run_official_air_baselines.py`, `scripts/run_expanded_paper_baselines.py --preset paper_full`
- Paper tables: `python scripts/generate_prism_breath_paper_tables.py --run-root outputs/paper_runs/<RUN_ID>`

### Tests
**No pytest suite.** Tests are standalone scripts, each with a `__main__` that self-checks one
component; run from the project root, e.g. `python src/core/test_physiological_classifier.py`.
They live beside their modules in `src/core/test_*.py` (there is no `-k` selector — running the
file runs its whole suite). The top-level `tests/` directory is empty.

## Architecture

### Live inference path — `src/core/main_controller.py::RespiratoryController` (`process_video`)
A ~25-flag orchestrator (`enable_*`) that runs: ingest frames (OpenCV, adaptive-resolution OOM
fallback) → ROI (`anatomical_stabilizer.py` YOLO-pose chest box, falling back to `external/air-400`
`PreProcessor` body/face, then center-crop) → per-modality signals: depth (`depth_engine.py`,
HuggingFace Depth-Anything-V2), optical flow, audio spectrogram → per-modality BPM+confidence
(`signal_processor.py`: Butterworth 0.15–1.5 Hz bandpass, optional VMD, FFT peak, SNR) → **layered
fusion fallback ladder** (supervised regressor → transformer/Perceiver/ImageBind fusion → robust
disagreement-aware fusion → SNR-weighted heuristic, so a BPM always returns) → diagnosis
(`diagnosis.py::ARIDiagnostic`, WHO age thresholds) + `vlm_engine.py` clinical context + optional
conformal triage interval. Fusion backbones: `multimodal_fusion_models.py`
(`TemporalFusionTransformer`, `MambaFusionModel`), `perceiver_fusion.py`, `imagebind_fusion.py`,
`supervised_fusion.py`, `modality_reliability.py::RobustFusionTransformer` (see
CROSS_MODAL_IMPLEMENTATION_SUMMARY.md). `MODALITY_NAMES = ("depth","flow","audio")`.

### PRISM-Breath path — the flagship model
`prism_breath_model.py` (`PrismBreathModel` + dataclass `PrismBreathConfig.from_variant` + `PrismBreathLoss`):
per-modality `RespiratorySignalEncoder` branches → reliability/agreement-weighted fusion, plus
`PrototypeEvidenceBank`, `CleanBreathingCodebook`, `SourceAgeAdapter` (domain adaptation), and
counterfactual/triage heads — every capability a `use_*` flag for ablations. Datasets flow through
`src/utils/respiratory_benchmark_registry.py` / `dataset_paths.py` / `respiratory_ground_truth.py`
and `src/core/respiratory_benchmark_dataset.py`; `.npz` files follow a `{dataset}_{subject}_{video}.npz`
provenance contract (parsed back by `parse_signal_name`). `selective_triage.py` provides
disagreement-conditioned conformal safety intervals.

### Critical gotchas
- **Vendored repos on sys.path**: `main_controller.py` appends `external/air-400` to `sys.path`, so `from models.vire_net import VIRENet` / `from processors.pre_processor import PreProcessor` resolve against `external/air-400`, **not** `src`. Treat everything under `external/` as third-party (`air-400`, `Infant-Respiration-Estimation` = PhysNet/TS-CAN/DeepPhys/EfficientPhys/VIRENet, `pyflow`) — don't edit it as project source.
- **`Respiratory-rate-monitoring/`** is a superseded legacy RealSense/Haar-cascade prototype, not part of the current pipeline.
- **`src/api/server.py`** currently returns a **mocked** BPM (55.0) — it is not the real `process_video` path.
- **WHO tachypnea thresholds are duplicated** in `diagnosis.py::ARIDiagnostic` and `prism_breath_model.py::tensor_tachypnea_threshold` (≥60 BPM <2mo, ≥50 2–12mo, ≥40 12–60mo) — keep them in sync when editing.
- **Config paths are project-root-relative**: `src/config/test_infer.yaml` uses relative paths (`data/yolov8m.pt`, `output`), so run scripts from the repo root (as the runbooks do) for them to resolve. Its uppercase-key schema mirrors `external/air-400` configs; Ultralytics `settings.json` `datasets_dir` is relative too. (Historical `outputs/`/`*.log` artifacts may still contain old absolute `f:/` paths — ignore those.)
- **`data/` is huge and gitignored** (~125k files: MINI-RGBD frames, AIR_125/AIR_400, ICBHI audio, mimic4wdb, a 6 GB video zip). Never glob/enumerate `data/` wholesale — target specific subpaths. Weights (`*.pth/*.pt/*.pkl`) and `outputs/` are also gitignored.
- **`smil/`** is the SMIL infant body model (SMPL-variant `.pkl` priors) for synthetic MINI-RGBD data — not part of the runtime.
- Dataset short names used throughout: **AIR-125 / AIR-400** (infant respiration video), **Pediatric-RR** (-332), **MINI-RGBD** (synthetic infant depth), **ICBHI** (respiratory audio), **MIMIC-IV-WDB** (waveform).

## Docs worth reading first
`docs/PRISM_BREATH_PIPELINE.md` (runnable pipeline), `docs/PRISM_BREATH_REDESIGN.md` +
`docs/PRISM_BREATH_JOURNAL_BLUEPRINT.md` (design narrative & reviewer-facing plan),
`docs/CAMERA_READY_RUNBOOK.md`, `README.md`, `CROSS_MODAL_IMPLEMENTATION_SUMMARY.md`.
Best code entry points: `main_controller.py`, `prism_breath_model.py`, `signal_processor.py`,
`diagnosis.py`, `respiratory_benchmark_registry.py`, `scripts/run_prism_breath_study.py`.
