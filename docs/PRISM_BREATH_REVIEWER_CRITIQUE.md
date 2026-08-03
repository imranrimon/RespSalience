# PRISM-Breath — Adversarial Reviewer Critique & Paradigm-Shift Directions
_Grounded in the actual 5-fold CV numbers (2026-08). Severity: 🔴 fatal-to-claim, 🟠 serious, 🟡 fixable._

## Verdict in one line
What the evidence actually supports: **"a non-contact RR estimator whose learned flow + trained
head rank tachypnea better (AUROC 0.869) than four thresholded rPPG extractors, on ONE lab dataset."**
What the paper currently claims: **"non-contact pediatric ARI screening."** The gap between those two
sentences is the review.

## Major shortcomings

### 🔴 1. Total blindness to neonatal tachypnea — the highest-risk group
`<2mo` sensitivity = **0.00** across every variant, ablation, and mitigation. Concretely, the 8
neonatal tachypnea cases have true RR 60–80 BPM but PRISM predicts **30–51** — it *never* exceeds
~52 BPM. The estimator's dynamic range **saturates ~50**, structurally below the ≥60 neonatal
threshold, so it is guaranteed to miss every neonatal case. This is not underpowering; it is a
regression-to-the-mean + band/extractor resolution limit (synth-aug didn't fix it). For a paper
whose north star is pediatric ARI, the system **works where it's least needed (older infants, lower
thresholds) and fails where it matters most (neonates, highest pneumonia mortality).**

### 🔴 2. "ARI screening" is unsupported — ARI is never measured
There are no infection/disease labels anywhere. Pipeline = measure RR → threshold at WHO cutoffs →
"tachypnea" → *inferred* ARI. Tachypnea is a weak proxy (present in fever, crying, pain, cardiac
disease, metabolic acidosis; WHO-IMCI tachypnea has limited PPV for pneumonia). The work measures a
proxy (tachypnea) of a proxy (spot RR) and claims the target (ARI). Honest scope = "RR estimation +
tachypnea flagging." Drop or heavily caveat the ARI framing until real disease labels exist.

### 🟠 3. Single-dataset clinical validation; the screening endpoint is never externally validated
Every clinical number rests on Pediatric-RR-332 (n_pos=86). Cross-dataset (AIR) has **zero**
tachypnea (degenerate low-RR), so it validates RR-MAE only — the tachypnea/screening endpoint has
**no external validation**. One dataset, one lab, one GT protocol. No skin-tone / lighting /
demographic breakdown → equity unaudited (rPPG has documented skin-tone bias; flow-based respiration
bias is untested).

### 🟠 4. Ground truth is manual breath-counting
The reference standard is the noisy manual count WHO itself flags for poor inter-rater reliability.
A screening tool is being validated against a weak reference. No RIP-belt / capnography / impedance
gold standard anywhere.

### 🟠 5. Single spot estimate, no behavioral-state awareness
Clinical tachypnea = sustained count over ≥60 s when the child is calm, ideally repeated. A single
video-derived RR is state-blind: crying/feeding/motion inflate RR (false positives) or corrupt the
signal (false negatives). No behavioral-state gating or decision-level artifact rejection.

### 🟠 6. Leakage in the flow channel and SSL pretrain
The flow extractor trains on the full/default split then extracts test-fold subjects; SSL pretrains
on all subjects' inputs. Both inject optimism into absolute numbers. (The clean-leakage per-fold
TSCAN run tests the flow part; the SSL representation leakage remains.)

### 🟡 7. Over-engineering vs. load-bearing contribution
Ablations: MAE flat (6.0–6.4) and sensitivity moves only a few infants when removing
prototypes/codebook/domain-adapter; flow-only ≈ full PRISM (depth marginal, audio dead). The
elaborate apparatus earns little on the anchor endpoint. Worse: the trained tachy head **underperforms
raw-RR ranking** (AUROC 0.869 < 0.877) — the head isn't load-bearing. A reviewer will ask which
components justify their complexity; the honest answer looks like "a well-trained RR regressor on a
good flow signal + reliability masking."
_NB: the reassuring half — PRISM's raw-RR AUROC 0.877 > EfficientPhys 0.786 and its common-margin
AUROC 0.820 > 0.755, so the headline is NOT a "trained-head vs hand-set-sigmoid" artifact. The
better signal is real._

### 🟡 8. Underpowered + single-seed
n_pos=86 pooled, <2mo n_pos=8; many McNemar/Sens comparisons n.s. Main CV is single-seed (42) —
and DeepPhys just demonstrated how seed-fragile from-scratch training is (fold3 collapse). Seed-average
(≥3) the headline PRISM operating points before publishing them.

### 🟡 9. Conformal "safety" is hollow exactly where safety matters
Pooled α=0.20, but <2mo is uncoverable (coverage 0.48–0.62). The safety guarantee does not hold for
neonates — the group that most needs a safe defer. Age-conditional conformal is mandatory, and even
then the neonatal problem is upstream (the RR is wrong, not merely uncertain).

## Paradigm-shift directions (not incremental)

1. **Spot-RR-threshold → state-aware continuous monitoring + deterioration trend.** Monitor the RR
   *trajectory* with behavioral-state gating; flag *sustained* tachypnea and *rising* trends (PEWS-like),
   which are more actionable and robust than one reading.

2. **Solve neonates by changing the modality, not the model.** RGB flow at 25 fps can't resolve tiny,
   fast (1–1.5 Hz), often-swaddled neonates in cribs/incubators under variable light. Move to
   crib-realistic sensing — **thermal** (periodic nasal/oral heat, works in the dark, no skin-tone bias),
   **mmWave radar / RF** (through-swaddle chest displacement), or **event/high-fps depth** for high-RR
   resolution. Make the neonate the PRIMARY design target. Biggest white space: neonatal-specific
   contactless RR datasets barely exist.

3. **Proxy → outcome: a prospective clinical study.** ED/clinic cohort with RT-PCR/clinical ARI
   diagnosis + gold-standard RR (RIP/capnography) + diverse skin tones/lighting/real cribs + outcomes
   (admission, deterioration). Validate the SCREENING claim against disease and outcomes, with an equity
   audit. This single move changes what the paper *is* (signal-processing → clinical-impact).

4. **Physiology-informed foundation model for infant vitals.** One cross-modal SSL model pretrained on
   large unlabeled infant video (+depth/thermal/audio), learning HR, RR, SpO2-proxy and their couplings
   (RSA HR–RR coupling as a consistency/anomaly prior); few-shot adapt to neonates. Multi-vital
   redundancy + built-in plausibility checks attack small-data and single-modality fragility.

5. **Uncertainty as an outcome-validated triage policy.** confident-normal / confident-tachypneic /
   defer-to-clinician, with guarantees that hold **per age band**, tuned to downstream cost (missed
   neonatal pneumonia ≫ false alarm) at a fixed nurse-review budget.

6. **Unifying frame: equitable, offline, phone-based neonatal screener for LMIC settings** (where
   pneumonia kills most and pulse-ox/clinicians are scarce). Designing FOR that deployment forces the
   right constraints — equity, robustness, neonatal focus, edge latency, no gold-standard hardware.

## Post-review evidence (staged fixes + verdict, 2026-08)

**#1 neonatal — LOCALIZED & (partly) FIXED-BY-REFRAME.** Proven to be a *universal RGB-extractor*
saturation: for the 8 `<2mo` positives (true 60–80), the flow waveform's own FFT peak averages 34.9
BPM and ALL SIX extractors output ~25–40 (ID-1272 gt=80 → every method ~26). Not head/training-fixable;
requires modality change. **Honest fix shipped:** age-gated defer → `>=2mo` screen **Sens 0.73
[0.63,0.83], AUROC 0.907 [0.865,0.945]** (n=303, pos=78; defers 29 neonates / 8 tachypneic, none
falsely cleared). Defer must be age-based: `disagreement_score` is *lower* on missed positives (0.849
vs 0.937) — the model is confidently wrong, uncertainty can't flag it.

**New finding — the reliability fusion is INERT.** flow_weight=0.49 / depth_weight=0.51 *identically*
across all three age bands; it does not downweight the broken neonatal flow. The "reliability-weighted"
claim is not load-bearing (compounds #7).

**#9 conformal — quantified.** Pooled α=0.20 gives coverage `<2mo`=0.59, 2–12mo=0.71, 12–60mo=0.88
(robs young to pay old). Fix = Mondrian per-age-band α (post-hoc, cheap).

**#8 seed — STAGED.** Seeds 43/44 of the main CV queued (reuse shared SSL pretrain); `combine_seeds.py`
reports 3-seed averaged operating points with bootstrap CIs.

**#6 flow-leakage — RUNNING.** Per-fold leakage-free TSCAN-flow build + CV in flight.

**CORRECTION (2026-08) — #7 was overstated; the tachy head is NOT dead weight.** Tested the proposed
fix (replace head with val-fit isotonic on raw RR): it is WORSE — AUROC 0.879 (head) vs 0.822 (isotonic
raw-RR), ECE 0.064 vs 0.073. Raw-RR's tiny intrinsic ranking edge (0.888 test-ranking vs 0.879 head)
does not survive proper val→test calibration (isotonic plateaus destroy ranking). Keep the head. My
earlier "head < raw RR → dead weight" claim is retracted.

**#9 Mondrian age-conditional conformal — tested.** Per-band qhat lifts 2–12mo but `<2mo` coverage
stays 0.59 (neonatal errors ~30 BPM dwarf any qhat ≤ ~8); covering <2mo at 0.80 needs qhat≈30 = defer.
Confirms: Mondrian for ≥2mo + age-gated defer for <2mo.

**Fusion diagnosis (#C).** The mechanism exists — `fusion_weights = softmax(quality·agreement/scale)` —
but `quality` is a free learned head that collapsed to ~constant, so on pedrr's 2 live modalities it
degenerates to 0.49/0.51. Fix = ground `quality` in COMPUTED spectral SNR of each input (staged retrain).

## What I'd need to see to accept (as a journal reviewer)
- Neonatal sensitivity > 0 with a credible mechanism (modality change or dynamic-range fix), OR the
  claim narrowed to the age bands where it works, stated up front.
- A second clinical dataset **with real tachypnea prevalence**, ideally external site.
- Gold-standard RR reference for at least a validation subset.
- Seed-averaged operating points with age-band CIs; leakage-free extractor (in progress).
- Equity/subgroup analysis, and the ARI claim either substantiated with disease labels or removed.

## Literature-grounded comparison (where PRISM sits in the 2023–2026 field)

Positioned against published work, PRISM does not advance extraction, its fusion/SSL are behind the
frontier, and its fatal case is solved elsewhere. Every claim below is tied to a cited number.

**A. Does not advance the infant-extraction SOTA.** AIRFlowNet reports **MAE ≈ 2.9 brpm in-domain on
AIR-125** vs 4.7–6.2 for adult rPPG backbones (Wan/Ostadabbas, MICCAIW 2023, arXiv:2307.13110); WACV
2026 (Song et al.) extends to AIR-400 under small data. PRISM's AIR-125 is **~4.7 zero-shot** — in the
adult-baseline band, ~1.8 brpm behind the infant SOTA, and it was NEVER trained/eval'd in-domain on AIR
against AIRFlowNet. PRISM's flow channel *is* EfficientPhys, so it inherits (not improves) extraction;
ablations show the win is a downstream head that underperforms raw-RR ranking (0.869 < 0.877).

**B. The neonatal fatal flaw is solved in other modalities.** Thermal: neonatal RR MAE **≈1.5 brpm**,
with normal/tachypnea/apnea differentiation (J Clin Monit Comput 2022 PMC10175339; SIViP 2025). Radar
(mmWave/FMCW/UWB): through-cot neonatal breathing in the NICU, dark/swaddle-robust (Sci Rep 2022
s41598-022-08836-3; Frontiers Peds 2026; JMIR 2025). PRISM's `<2mo` Sens=0.00 is a solved problem — just
not in RGB.

**C. Fusion is behind the field and empirically inert.** CardiacMamba (RGB-RF, state-space fusion, 2025,
arXiv:2502.13624) is genuinely complementary cross-modal fusion. PRISM's RGB+depth+audio has two dead
channels and constant 0.49/0.51 reliability weights — non-functional fusion on correlated modalities
that fail together.

**D. SSL/architecture not competitive with the frontier.** rPPG-MAE, RhythmFormer (periodic sparse
attention, Pattern Recognition 2025), PhySU-Net (SSL long-context, arXiv:2402.11913), VLM-bootstrapped
self-supervised rPPG (arXiv:2407.08507), physiological foundation models (arXiv:2606.07365) — none used
as baselines. SSL on ~265 subjects is not a foundation-model contribution in 2026.

**E. Clinical validation below the existing device bar.** ChARM (BMC Digital Health 2025) validates
automated RR against a **Video Expert Panel** on suspected-pneumonia children 0–59mo, honestly reporting
~1-in-8 failures on clinically challenging cases. PRISM uses single manual-count GT, one lab dataset, no
pneumonia outcome, and a worse failure profile (100% of neonates). Real-NICU literature (Oxford 30
preterm/426h, npj Digital Med 2019; RGB-D NICU arXiv:2412.06012) is neither compared nor approached.

**F. Benchmarking gaps.** No in-domain AIRFlowNet, thermal, radar, RhythmFormer/rPPG-MAE, or
foundation-model comparison; ARI claimed via a tachypnea proxy while the clinical literature ties
automated RR to pneumonia + VEP references.

**Legitimate narrow contribution.** Reframing extraction-SOTA as a *clinical screening endpoint*
(tachypnea AUROC, not MAE) is real: raw-RR AUROC 0.877 > EfficientPhys 0.786 on the same flow, and the
age-gated defer yields a safe ≥2mo screen (Sens 0.73 / AUROC 0.907). Workshop-grade, not yet flagship.

**The single paradigm move:** stop competing as another RGB rPPG extractor and become *the first
equitable, uncertainty-guaranteed, neonatal-capable non-contact tachypnea screen* by adding a thermal or
radar channel and validating to the ChARM/VEP clinical bar. See `PRISM_BREATH_NEONATAL_PARADIGM_PLAN.md`.
