# PRISM-Breath — Experimental Plan, Ablations, Baselines & Evaluation Protocol

Target: a top-tier venue (WACV/CVPR/MICCAI/IEEE JBHI/TBME). This is the canonical, honest
plan reflecting everything the validity audit established. It supersedes ad-hoc run notes.

## 0. Where we are (state as of the audit)

**Done (single-split, 3-seed, Pediatric-RR-332 in-dist):**
- `pedrr_v3` (hand-crafted `gated_consensus` flow) — baseline.
- `pedrr_retune_fe0` — learned flow + audio content-aware mask + fps-normalized load; recipe
  `modality-dropout 0, freeze-encoder 0` (the re-tune winner).
- `pedrr_retune_fe0_audiofix` — audio-mask A/B.
- `ablation_fe0` — full architecture ablation, 5 variants.

**Tooling (in-repo, verified):** subject-level bootstrap CIs + DeLong + McNemar
(`clinical_endpoint_eval.py`, `compare_clinical_models.py`); dumb+oracle baselines
(`validate_fe0.py`); conformal coverage check; deterministic training.

**Established findings (all with proper statistics):**
1. In-dist learned-flow ≈ hand-crafted v3 — **not significant** (DeLong p=0.73–0.92, McNemar p=1.0).
2. The fusion **does** beat dumb baselines significantly (ΔAUROC +0.16, ΔSens +0.35 vs median-of-3).
3. High-RR tachypnea **false-negatives are extractor-limited**: the benchmark "depth" is a mean-ROI
   **intensity proxy** whose signal peaks at 13–19 BPM for gt 58–72 (fast breathing physically absent).
4. Dead audio channel on pedrr (all-zeros placeholder) — now masked content-aware.
5. fps bug affects **cross-dataset only** (pedrr is uniform 25 fps); fix coded + unit-verified.
6. Model is **over-parameterized** for n=232 (no single component individually significant in-dist).
7. Conformal is **miscalibrated by band** (empirical coverage <2mo 0.67 vs nominal 0.80; intervals ±11 BPM).
8. GT is **manual breath-counting** with no documented error → sub-2–3 BPM differences are within the GT floor.

**Staged, not run:** upgraded extractor (`src/core/upgraded_respiration_extractor.py`) + A/B
(`scripts/ab_extraction_upgrade.py`); k-fold harness (`scripts/make_kfold_manifests.py`,
`run_kfold_cv.ps1`); shared CV pretrain READY; folds 0/5.

## 1. What's next to run (ordered by value ÷ cost)

| # | Experiment | Cost | Why first |
|---|---|---|---|
| A | **Extraction A/B** (Tier-1 intensity, CPU, minutes) | trivial | Decides the flagship lever cheaply — is high-RR info recoverable? |
| B | **k-fold CV baseline** (needs ~8 h awake) | high | The powered whole-cohort numbers; kills selection-on-test |
| C | **#22 freq-forward CV** (gated behind B) | high | Clinical verification of the fusion fix at real power |
| D | **#23 cross-dataset** (fps-backfill AIR + eval fe0; no retrain) | low | The generalization headline |
| E | **#24 per-band (Mondrian) conformal + normalized score** | low | Fix the neonatal miscalibration + narrow intervals |
| F | **Depth upgrade → re-extract → retrain → CV** (only if A wins) | highest | The SOTA lever for the neonatal/high-RR frontier |

## 2. Step-by-step experimental plan (phases)

**Phase 0 — Front-end validation (decide the extractor).**
- **Tier-1 (intensity) — RUN, FAILED (2026-07-22).** Motion-magnified + abdomen-ROI + harmonic-aware
  did NOT recover the FN clips (gt 58–65 → ~10 BPM) and *regressed* the normal-RR controls: the
  in-band-power weighting locks onto a ~10 BPM low-frequency confound (lighting/gross-motion), not the
  subtle fast breathing. MAE proxy 24.3 → upgrade 30.0. Conclusion: **the high-RR signal is not in the
  intensity pixels**; intensity-based depth salvage is dead.
- **Tier-2 (real Depth-Anything-V2) — RUN, FAILED (2026-07-23).** Real monocular depth ALSO did not
  recover the FN clips (gt 58–65 → 8.6–12.9 BPM); MAE proxy 24.3 → 28.8. Same failure mode as Tier-1
  (ROI aggregation locks onto a ~10 BPM confound). **Conclusion: monocular depth — proxy OR real —
  is confound-limited for neonatal high RR via spatial aggregation.**
- **F (depth re-extraction) is CANCELLED.** Both depth sources fail; do not spend GPU swapping one
  broken depth channel for another. `reextract_depth_upgraded.py` stays unused.
- **Definitive redirect:** optical FLOW is the only modality that resolves high RR (flow-head 68.5/55/87
  on these exact clips). The high-RR win comes from the **#22 freq-forward fusion** (trust flow under
  disagreement), verified via the CV — NOT from the depth front-end. This is a publishable finding:
  *monocular depth is confound-limited for neonatal tachypnea; only flow resolves it → fusion must be
  flow-forward under disagreement.*

**Phase 1 — In-distribution benchmark (main table).** Run B: subject-disjoint 5-fold CV on
Pediatric-RR-332; pool all folds → one 332-subject clinical endpoint with bootstrap CIs. Report the
full model + all baselines under the *identical* CV protocol.

**Phase 2 — Ablations (under CV).** Re-run the architecture + front-end + fusion + modality ablations
inside the CV so every number is powered and selection-clean (§3).

**Phase 3 — Cross-dataset generalization.** Run D: train on pedrr → evaluate on AIR-125/AIR-400
(fps-normalized), and if data allows cross-device / cross-skin-tone stratification. This is the
contribution the in-dist parity result forces us toward.

**Phase 4 — Uncertainty & triage.** Run E + decision-curve/risk-coverage analysis. Report the
disagreement-conditioned conformal selective-triage as a *validated* abstaining screener.

**Phase 5 — Clinical grounding (aspirational).** Source an ARI-outcome cohort (external) to convert
"RR-threshold detection" into "screening". Until then, frame honestly as a screening *aid*.

## 3. Ablation design (each isolates ONE factor; 3-seed × CV; vs full model with DeLong/McNemar)

**Front-end (Phase 0/F):** proxy-intensity → {+abdomen ROI} → {+motion-magnification} →
{+harmonic-aware RR} → {real depth}. Reported at signal level (dominant-BPM vs gt) AND end-to-end.

**Architecture (done, re-run under CV):** `prism_no_scan`, `prism_no_spectral`, `prism_no_codebook`,
`prism_no_prototypes`, `prism_shared_trunk` (expert-routing off) vs full `prism_breath`.

**Fusion (#22):** `use_freq_forward_fusion` on/off; `modality-dropout` 0 vs 0.1; audio content-aware
mask on/off.

**Modality:** depth-only, flow-only, depth+flow, +audio (only on sources with real audio).

**SSL pretrain:** with vs without encoder pretraining.

Every ablation is a paired comparison on the same folds → paired bootstrap CI + DeLong/McNemar,
with Holm–Bonferroni across the ablation family.

## 4. Baselines (all under the SAME train/eval/conformal protocol)

- **Dumb fusion:** mean-of-3, median-of-3, best-quality-channel; **oracle-best-channel** (upper bound).
- **Hand-crafted:** v3 `gated_consensus`.
- **Published extractors** (repo has them; same protocol via the model factory): PhysNet, DeepPhys,
  TS-CAN/MTTS-CAN, EfficientPhys, AIRFlowNet/VIRENet; external-method wrappers `cliffphys`, `mamba`.
- **SOTA reference:** EfficientPhys + Coarse2Fine + body-ROI (MAE 3.70 on AIR-400).
- **Single-modality vs multimodal** to justify fusion.

## 5. Evaluation protocol

- **Primary:** subject-disjoint **k-fold CV**, pooled to whole-cohort. Secondary: a single held-out
  test touched once. Splits are deterministic and subject-disjoint (no identity leakage).
- **Reported BY WHO age band + OVERALL** (<2mo ≥60, 2–12mo ≥50, 12–60mo ≥40 BPM).
- **RR agreement:** MAE, RMSE, Pearson r, **Bland–Altman** bias + 95% LoA, **ICC** (given manual-count GT).
- **Tachypnea decision:** sensitivity, specificity, PPV, NPV, F1, balanced accuracy, AUROC, AUPRC.
- **Operating points:** spec@sens≥0.90, sens@spec≥0.80, Youden-J.
- **Selective prediction:** risk–coverage, coverage needed for sens≥0.90.
- **Decision-curve net benefit** across the clinically relevant threshold-probability grid.
- **Conformal:** empirical coverage vs nominal 1−α **per band**, mean interval width.

## 6. Result analysis for a top-tier venue

1. **Uncertainty is subject-level bootstrap 95% CI** — seed-std is demoted to a "training stability"
   footnote. (Seed-std understated real uncertainty ~6–10×.)
2. **Significance for every head-to-head:** DeLong (AUROC), McNemar (decision at threshold), paired
   bootstrap CI on the difference. Report **effect size + CI**, not bare p-values.
3. **GT-noise floor honesty:** report agreement (ICC/Bland–Altman); explicitly declare method
   differences below the GT's manual-count error (~2–3 BPM) as within-noise.
4. **Failure modes reported, never aggregated away:** the <2mo band (n_pos tiny; AUROC CI [0,1]) and
   RR>55 subset get their own honest section. High-RR limitation stated plainly.
5. **Multiple-comparisons control:** Holm–Bonferroni / BH across the ablation and baseline families.
6. **Pre-registration:** lock the final config + primary endpoint before the external/cross-dataset
   evaluation, so the headline is selection-bias-free (past single-split "winners" were selection-contaminated).
7. **Reproducibility:** deterministic kernels + seeded dataloaders; release code, manifests, and the
   exact fold assignment.
8. **Construct-validity framing:** state clearly this validates RR-threshold detection on a
   counting cohort; ARI-screening is the aspiration, contingent on Phase 5.

**Contribution framing (what makes it top-tier, given in-dist parity):** not "another multimodal RR
estimator," but (i) a **reliability-calibrated, disagreement-conditioned abstaining screener** validated
with coverage + decision curves; (ii) **cross-dataset generalization** where single-modality methods
collapse; (iii) a **high-RR/neonatal frontier** advance via the extraction upgrade. The MAE race is
crowded and we are at parity there — the novelty is the calibrated, generalizable *decision* layer plus
the neonatal extraction fix.
