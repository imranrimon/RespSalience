# Dataset-Driven Scope (March 5, 2026)

This project can proceed now with features that are supported by currently available labels.

## Implement Now (Supported by current labels)

- RR/BPM estimation and ARI thresholding from AIR-125, AIR-400, Pediatric-RR.
- Cross-modal safety checks and robust fallback fusion (`enable_robust_fusion=True`) to handle modality conflicts.
- Explainability overlays and model diagnostics that do not require new clinical class labels.

## Future Work (Requires new labels/cohort design)

### Challenge #1: Physiological State Classification
- Missing labels: sleep state (REM/NREM), feeding, crying/distress aligned to video time.
- Current labels are mainly RR/BPM, not behavioral state.

### Challenge #3: Longitudinal Progression
- Missing labels: repeated linked measurements over hours/days with outcome/intervention labels.
- Current datasets are mostly per-video snapshots.

### Challenge #7: Etiology/Comorbidity
- Missing labels: confirmed diagnosis classes tied to infant multimodal recordings (e.g., pneumonia vs asthma vs CHF vs sepsis).
- Current infant video datasets do not provide diagnosis-linked etiology classes at this granularity.

## Recommended Data Collection Additions

- Add synchronized event labels to existing videos:
  - `state`: sleep/awake/feeding/crying (+ REM/NREM if available)
  - `episode`: distress onset/offset
- Build a longitudinal table keyed by `patient_id`, `timestamp`, `visit_id`, and outcomes.
- Add clinician-confirmed etiology labels and severity scores per encounter.
