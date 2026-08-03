# PRISM-Breath Redesign

## Scope
The journal paper should focus on one supported task:

**contactless pediatric respiratory-rate estimation with safe tachypnea triage under modality failure and dataset shift**

Everything else in the repo should move to future work unless new labels are collected:
- physiologic state recognition
- longitudinal deterioration forecasting
- etiology / comorbidity reasoning
- audio-first diagnostic claims

## What Was Implemented
This redesign adds a dedicated journal path instead of extending the exploratory platform:

- `scripts/build_respiratory_benchmark_manifest.py`
  - builds canonical manifests for `AIR-125`, `AIR-400`, and `Pediatric-RR`
  - saves subject-disjoint train/val/test files
  - saves leave-one-dataset-out splits for external-style validation
- `src/utils/respiratory_benchmark_registry.py`
  - unifies raw dataset scanning and fixed split generation
- `src/core/respiratory_benchmark_dataset.py`
 How can we   - loads signal `.npz` files from the journal manifests
- `src/core/prism_breath_model.py`
  - implements `PRISM-Breath`: a physiology-grounded, reliability-aware, selective model
  - exposes paper ablation variants:
    - `prism_shared_trunk`
    - `prism_no_scan`
    - `prism_no_spectral`
    - `prism_no_codebook`
    - `airflow_compact`
- `src/core/prism_breath_pretraining.py`
  - compatible encoder pretraining for the same branch family used downstream
- `scripts/train_prism_breath.py`
  - trains the main model or any ablation variant on extracted signals
- `scripts/pretrain_prism_breath.py`
  - runs RespMAE-style masked pretraining plus contrastive augmentation
- `scripts/evaluate_prism_breath.py`
  - exports prediction tables, clinical triage outputs, prototype evidence, and counterfactual drop-modality explanations
- `scripts/run_prism_breath_study.py`
  - orchestrates pretraining, ablations, conformal calibration, and test evaluation

The Pediatric-RR path mismatch was also fixed centrally through:
- `src/utils/dataset_paths.py`
- `src/utils/respiratory_ground_truth.py`
- `scripts/build_dataset_manifest.py`

## Core Research Question
Can a contactless multimodal respiratory model estimate pediatric RR accurately across heterogeneous datasets and make safer tachypnea decisions by:
- downweighting unreliable modalities
- abstaining when evidence overlaps the clinical threshold
- explaining which evidence drove the decision

## Model Design
`PRISM-Breath` is not a generic fusion transformer. It is designed around the journal claim.

1. Per-modality physiology encoding
- Each signal branch extracts temporal structure, multi-scale motion cues, spectral summaries, and morphology statistics.

2. Reliability-aware fusion
- Each modality predicts RR, a scale term, and a quality score.
- Pairwise agreement between modality-specific RR estimates becomes an explicit disagreement signal.
- Fusion weights depend on quality, scale, and agreement instead of blind concatenation.

3. Selective clinical decision layer
- The model outputs RR, an interval width (`qhat`), tachypnea probability, and review probability.
- Final triage is `NORMAL_SAFE`, `TACHYPNEA_ALERT`, or `UNCERTAIN_REVIEW`.

4. Clean-breathing denoising memory
- A CodePhys-inspired codebook retrieves clean respiratory patterns and denoises the latent state before the final decision.

5. Explainability hooks
- prototype evidence bank
- per-modality RR / quality / fusion weight
- disagreement score
- counterfactual modality-drop deltas

## Why This Is More Innovative
The innovation is not “more modalities.” It is **safe respiratory AI under uncertainty**.

That is stronger because real pediatric deployment fails on:
- motion
- occlusion
- crying
- missing audio
- low-quality depth or flow
- dataset/site shift

The redesign makes those failure modes part of the method and evaluation instead of treating them as noise.

## Latest SOTA Inspirations
These ideas were chosen from recent primary-source work and translated into a respiratory setting.

1. Small-data respiration modeling:
- Song et al., WACV 2026, *Overcoming Small Data Limitations in Video-Based Infant Respiration Estimation*  
  https://openaccess.thecvf.com/content/WACV2026/html/Song_Overcoming_Small_Data_Limitations_in_Video-Based_Infant_Respiration_Estimation_WACV_2026_paper.html
- Takeaway for this repo: respiration benchmarks are still small and heterogeneous, so journal design must prioritize data efficiency and cross-dataset generalization.

2. Physiology-guided experts:
- Zhang et al., TPAMI 2024, *PhysMLE: Physiology-Guided Mixture of Low-Rank Experts for Predicting Multiple Physiological Measurements From Facial Videos*  
  https://pubmed.ncbi.nlm.nih.gov/40031788/
- Takeaway for this repo: route information using physiology-aware expert behavior instead of one fixed fusion rule.

3. Frequency-aware state-space modeling from another biosignal domain:
- Wang et al., 2025, *CardiacMamba: A Frequency-Augmented State Space Model for ECG-Based Arrhythmia Detection*  
  https://arxiv.org/abs/2502.13624
- Takeaway for this repo: combine sequence-state updates with explicit spectral structure instead of relying only on attention.

4. Generalizable remote physiological modeling:
- Liu et al., 2025, *CodePhys: A Generalizable and Explainable Approach to Remote Physiological Measurement*  
  https://pubmed.ncbi.nlm.nih.gov/40031604/
- Takeaway for this repo: separate a latent physiological representation from the raw sensing stream and make the representation inspectable.

5. Prototype-based medical interpretability:
- Du et al., Med Image Anal 2025, *ProtoMM: Prototype-Based Multimodal Learning for Interpretable Computational Pathology*  
  https://pubmed.ncbi.nlm.nih.gov/41168627/
- Takeaway for this repo: explanations should connect a sample to learned evidence patterns, not only heatmaps.

6. Counterfactual explanations for time series:
- Delaney et al., *Glacier: Guided Learning of Counterfactual Explanations for Time Series Classification under Probabilistic Temporal Constraints*  
  https://arxiv.org/abs/2405.15431
- Takeaway for this repo: counterfactual reasoning is a better explanation tool for clinicians than saliency alone.

7. Uncertainty-aware clinical deployment:
- van Bree et al., Nat Commun 2025, *Minimising and Quantifying Uncertainty in AI-Informed Decision Support Systems in Medicine*  
  https://pubmed.ncbi.nlm.nih.gov/40833408/
- Takeaway for this repo: the model should know when to review rather than forcing a diagnosis.

## Recommended Journal Contribution List
1. A unified benchmark for contactless pediatric RR estimation and safe tachypnea triage across AIR-125, AIR-400, and Pediatric-RR.
2. A physiology-grounded, reliability-aware selective model that fuses complementary motion signals under disagreement.
3. An explainability package with prototype evidence, modality reliability, and counterfactual drop-modality analysis.
4. A cross-dataset evaluation protocol centered on robustness and abstention, not just MAE.

## Suggested Paper Titles
- `PRISM-Breath: Physiology-Grounded Reliability-Aware Selective Modeling for Contactless Pediatric Tachypnea Triage`
- `SafeBreath: Uncertainty-Aware Contactless Respiratory Monitoring for Pediatric Tachypnea Screening`

## Expected Journal Experiment Matrix
- In-dataset train/val/test on canonical subject-disjoint splits
- Leave-one-dataset-out evaluation
- Missing-modality ablations
- Disagreement-conditioned calibration
- Review-rate / sensitivity / specificity tradeoff
- Explanation case studies with modality-drop counterfactuals

## Future Work Boundary
Keep these as future work until labels are available:
- etiologic ARI diagnosis
- longitudinal risk forecasting
- physiologic state classification
- richer audio-visual diagnostic reasoning

## Foundation Model Boundary
- Use masked respiratory pretraining as an initialization tool for PRISM-Breath.
- Keep synthetic text-signal alignment (`CLISP`) out of the main paper until real matched waveform-text corpora are available.
- Treat the foundation-model branch as auxiliary representation learning, not as the headline journal claim.
