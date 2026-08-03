# PRISM-Breath — Unified Narrative: Reliable Non-Contact RR Deployment under Domain/Label-Fidelity Shift

This is the conceptual backbone for the journal paper. It reframes the project's scattered findings
as one problem — **reliable deployment of a video respiratory-rate estimator under domain and
label-fidelity shift** — and adopts the MatQCP post-hoc reliability paradigm (align → conformal →
select; Rahman et al., *MatQCP*) as the backbone that ties calibration (#24), abstention (#12),
cross-dataset transfer (#23), and the neonatal limitation into a single story.

## 1. The reframing
Non-contact infant RR is not a pure accuracy problem; it is a **deployment** problem. A model trained
on one cohort (Pediatric-RR, manual-count labels, 25 fps) must be trusted on other devices, frame
rates, and populations — where systematic bias, not variance, dominates the error. Every result we
have fits this frame:
- **fps shift** — deploying on 10/15 fps AIR inflates RR ~2.5x (MAE 47) unless normalized (MAE 3.1).
- **domain shift** — pedrr->AIR-125 shows a systematic +8 BPM over-prediction.
- **label-fidelity shift** — GT is manual breath-counting (pedrr) vs belt-waveform (AIR-125); the
  training label itself is a low-fidelity proxy.
- **subgroup collapse** — the model is *confidently wrong* on neonates (<2mo, Sens 0.00).

## 2. The MatQCP backbone (all post-hoc, no retraining)
**Stage 1 — Target-space alignment (quantile mapping).** A monotone map g: pred -> target scale
removes systematic output-space bias. Result: **AIR-125 MAE 9.3 -> 4.7, bias +8 -> +2** (`air_quantile_align.py`).
Applied adaptively (only where the calibration split shows bias — it correctly leaves unbiased AIR-400 alone).

**Stage 2 — Mondrian conformal (align -> conformal).** Per-regime residual radii on the *aligned*
predictions. Result: 2-12mo and OVERALL coverage move to the 0.80 target (0.77 -> **0.82**);
`mondrian_conformal.py`. Crucially, `<2mo` **remains un-coverable** (0.48-0.62) — proving the neonatal
gap is a **base-model-quality** limit, not a calibration one.

**Stage 3 — Selective deployment (align -> conformal -> select).** A label-free risk score defers the
least-reliable predictions. Finding (`selective_deployment.py`): the model's *own* uncertainty
(disagreement, interval width) **cannot** flag neonates, because it is confidently wrong there —
uncertainty-based abstention lowers screening sensitivity. **The honest deployment policy is therefore
population-scoping:** the system is validated for **2-60mo** and **abstains on `<2mo` by age**, referring
neonates to a clinician. This converts the Sens=0.00 silent failure into a *safe* abstention.

## 3. What survives as contributions (all powered, all honest)
1. **A validated, calibrated, safely-abstaining non-contact RR/tachypnea screener for 2-60mo** —
   subject-disjoint 5-fold CV, n=332, no leakage: overall AUROC 0.879 [0.83,0.92], **12-60mo AUROC
   0.948 [0.90,0.98], Sens 0.86**; align+conformal coverage ~0.80; abstains on the un-modeled neonatal regime.
2. **fps-normalization is essential for cross-dataset deployment** — a clean methodological result with
   a 15x effect (MAE 47 -> 3.1); the pedrr-trained model then **transfers to AIR-400 (MAE 3.12)**,
   competitive with the SOTA infant extractor despite never seeing AIR.
3. **A rigorously characterized negative on the neonatal frontier** — monocular RGB intensity/depth
   cannot resolve high-RR breathing by *any* ROI/decomposition method (three independent negatives:
   motion-weighting, real Depth-Anything-V2, PCA/ICA spatial decomposition); only optical flow contains
   the signal, and it is high-variance. This scopes the claim and motivates new modalities.
4. **The MatQCP reliability layer** applied to non-contact vitals: alignment fixes cross-dataset bias,
   conformal calibrates the coverable population, and selective deployment makes the neonatal failure safe.

## 4. Negative results (a feature, reported honestly)
- In-dist learned-flow ≈ hand-crafted (DeLong p=0.73-0.92) — the fusion beats dumb baselines but the
  front-end choice does not move the in-dist endpoint.
- The #22 freq-forward fusion fix backfired under powered CV (Sens -0.105). The 4-clip "recovery" did
  not survive proper training + evaluation — a case study in why powered evaluation matters.
- Depth extraction upgrades and uncertainty-based neonatal abstention both failed. Post-hoc layers
  cannot rescue a base model that is confidently wrong on a subgroup.

## 5. Future directions (retraining/data-gated; validated where cheap)
- **Synthesize high-RR training data** by time-compression (concept validated to <0.5 BPM) to attack
  neonatal data scarcity — the base-model fix that a post-hoc layer cannot substitute for.
- **Heteroscedastic uncertainty head** so the model is *un*confident where it is wrong -> enables the
  uncertainty-based abstention that currently fails.
- **Robust optical flow** (temporal super-resolution, flow ensembles, motion-magnification on flow) or
  **new modalities** (radar/thermal) for the neonatal frontier.
- **Higher-fidelity GT / ICC** using the AIR-125 belt-waveform reference; **cross-site** analysis
  (pedrr 3 sites). Skin-tone/device fairness is not possible with current metadata (stated as a limitation).

## Analysis scripts (all no-retrain)
`mondrian_conformal.py` (align->conformal), `selective_deployment.py` (select), `air_quantile_align.py`
(cross-dataset alignment), `compare_clinical_models.py` (DeLong/McNemar), `clinical_endpoint_eval.py`
(bootstrap CIs), `ab_spatial_decomp.py` / `ab_extraction_upgrade.py` (extraction negatives).
