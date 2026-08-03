# PRISM-Breath — Lapse Mitigation & Paradigm Roadmap

How to mitigate each reviewer lapse (see `PRISM_BREATH_REVIEWER_CRITIQUE.md`), tiered by cost and by
what is executable in THIS repo (compute-only) vs. what needs new data/hardware. Rule of thumb:
**Tier 1 makes the current paper honest and defensible; Tier 2–3 make it field-changing.**

---

## Tier 0 — Honest reframing (no compute, do immediately)
- Retitle scope: **"non-contact RR estimation + tachypnea flagging for infants ≥2 mo"**, not "ARI
  screening." State the `<2mo` referral policy up front as a design choice, not a hidden failure.
- Position PRISM as a **clinical-decision layer on top of any extractor**, not a new extractor — this
  is its genuine, defensible contribution and sidesteps Lapse A.
- Report the ≥2mo operating range (Sens 0.73 / AUROC 0.907) as the headline; full-cohort as secondary.

## Tier 1 — Camera-ready compute mitigations (this repo, days–weeks)
Executable now on existing data/GPU. Each maps to a lapse.

| Lapse | Mitigation (runnable here) | Status |
|---|---|---|
| A (behind AIRFlowNet; extractor-agnostic claim untested) | **In-domain AIR train + eval PRISM vs AIRFlowNet**, and add **AIRFlowNet as a PRISM flow backbone** (extend the TSCAN-flow robustness result) → reframes "doesn't improve extraction" into "improves the clinical endpoint regardless of extractor" | TODO |
| C (inert fusion 0.49/0.51) | **SNR-gate the reliability fusion** (make weights a trained function of per-modality signal quality) and re-check weight variance; OR drop the dead depth/audio channels and present flow+head honestly | TODO |
| D (SSL/arch dated) | Add **RhythmFormer** and an **rPPG-MAE-style SSL** flow backbone as baselines + PRISM backbones | TODO |
| F (benchmarking gaps) | Run AIRFlowNet (in-domain), RhythmFormer under the SAME 5-fold clinical protocol | TODO |
| 7 (tachy head < raw RR) | Drop the trained tachy head; rank/threshold on **calibrated raw RR** (isotonic on val) | TODO |
| 8 (single seed) | **Seed-average (42/43/44)** with bootstrap CIs | RUNNING (PID 51672) |
| 6-flow (extractor leakage) | **Per-fold leakage-free TSCAN-flow** build + CV | RUNNING (PID 33480) |
| 9 (conformal hollow for neonates) | **Mondrian per-age-band conformal** (post-hoc, cheap) + age-gated defer | SPECIFIED |
| 1-safety (neonatal false-normal) | **Age-gated defer** (<2mo → refer) | SHIPPED (Sens 0.73/AUROC 0.907 on ≥2mo) |

## Tier 2 — New-data mitigations (weeks–months, need collaborators/hardware)
| Lapse | Mitigation | Resource |
|---|---|---|
| B (neonatal unsolvable in RGB) | **Thermal or radar neonatal channel.** Thermal: FLIR/Seek (~$200–500), nasal-region periodic heat → ~1.5 brpm neonatal (PMC10175339). Radar: TI IWR6843 mmWave EVM (~$300) → through-cot chest displacement, dark/swaddle-robust (Sci Rep 2022). Start with a **10–20 infant pilot** including `<2mo`. | thermal/radar sensor + IRB |
| E (weak GT) | **Video Expert Panel (VEP)** re-annotation of a Pediatric-RR subset (matches ChARM's reference standard), or a RIP-belt validation subset | clinical annotators |
| 3 (single dataset) | Acquire/curate a **second dataset with real tachypnea prevalence** (AIR has none); ideally external site | data-sharing |

## Tier 3 — Paradigm redesign (the field-changer, 6–12 months)
1. **Neonatal-first multimodal foundation model**: RGB + thermal/radar, SSL-pretrained multi-vital
   (HR+RR) with **RSA HR–RR coupling as a built-in plausibility prior**; few-shot adapt to neonates.
   Anchors: CardiacMamba (RGB-RF SSM fusion), rPPG-MAE, PhySU-Net, PPG foundation models.
2. **Prospective clinical validation to the ChARM/VEP bar**: suspected-pneumonia cohort 0–59mo,
   VEP + gold-standard RR reference, pneumonia/deterioration outcomes, **equity audit across skin
   tone/lighting**. Converts signal-processing → clinical-impact.
3. **Continuous deterioration/trend monitoring** (not spot-threshold), matching real NICU use
   (Oxford 426h; RGB-D NICU 2024) with behavioral-state gating.
4. **Equitable offline edge screener for LMIC** — the deployment that forces neonatal focus, equity,
   robustness, and latency as first-class constraints.

## Recommended execution order
1. **Now (compute):** SNR-gated fusion (C) + AIRFlowNet/RhythmFormer backbones & baselines (A/D/F) +
   Mondrian conformal (9) + raw-RR calibration (7). These + the two running jobs (6,8) close every
   *compute-fixable* lapse and make the paper honest.
2. **Next (data):** thermal/radar neonatal pilot (B) + VEP GT subset (E) — the two that convert the
   fatal neonatal lapse and the weak-GT lapse.
3. **Then (redesign):** Tier-3 foundation model + prospective validation.
