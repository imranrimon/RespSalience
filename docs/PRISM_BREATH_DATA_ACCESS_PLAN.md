# PRISM-Breath — Data-Access Plan for the Unblocked Directions

Two high-value datasets that address the harshest review lapses (disease outcome, weak GT, single
dataset) WITHOUT hardware/IRB. Both need a step only the PI can take (a browser download / an
institutional request) — a coding session can't clear those from here. Everything else is drafted.

---

## 1. VEP-referenced pediatric RR video dataset — Mendeley `72dd3rkttf`
**URL:** https://data.mendeley.com/datasets/72dd3rkttf/2  (DOI 10.17632/72dd3rkttf.2, open, CC-BY)
**Contents:** 332 MP4 videos (~60 s) of chest-wall movement, children <5 in Bangladesh presenting with
cough / difficulty breathing; **RR assessed by a Video Expert Panel (VEP)** + a Stata `.dta` with
demographics, symptoms, VEP-RR.

**Why it matters (corrects two earlier claims):**
- The **VEP reference standard I said "wasn't public" IS public here** — this is the ChARM-family
  reference, far stronger than our single manual counts (lapse #4 / Tier 2C).
- **Symptomatic (ARI-relevant) cohort** — closer to the screening use-case than Pediatric-RR.
- Candidate **external validation** set (lapse #3).

**⚠️ Critical caveat — check for overlap first.** It is **332 videos**, children <5, Bangladesh, VEP-RR
— and our anchor is **Pediatric-RR-332**. These may be the SAME cohort (or its source). Before using it
as "external," diff subject IDs / video hashes against Pediatric-RR; if they overlap it is NOT
independent and must not be double-counted. This is the first thing to verify on download.

**How to get it (browser — API blocks automation from this environment):**
`!start https://data.mendeley.com/datasets/72dd3rkttf/2` → "Download All", or download the `.dta`
metadata alone first (small) to run the overlap check before pulling the videos.

## 2. PREPARE child-pneumonia database — retrospective outcome validation
> **SCOPING DECISION (2026-08): OUTSCOPED for this paper → future work.** PREPARE answers an
> epidemiology question (does tachypnea predict pneumonia outcomes?) adjacent to — not part of — our
> non-contact *measurement* contribution. The tachypnea→pneumonia rationale is established WHO
> literature we cite; the VEP `diff_breath_observed` probe already gives a free disease-link check
> (true RR AUROC 0.59, PRISM 0.54). Managed access + WHO joint-authorship = heavyweight for marginal
> gain. Revisit ONLY if the central claim becomes "improves pneumonia outcomes" (a separate clinical
> paper). Access details retained below for that future step.
**What:** 285,839 pneumonia episodes (41 datasets), **RR at admission + 24 h intervals to day 14**,
IMCI danger signs, SpO2, **9,185 deaths** (outcome). Managed access.
**Access procedure:** email **Yasir Bin Nisar, WHO MNCAH (nisary@who.int)** with a proposed-analysis
outline → WHO + PREPARE study-group approval → data-use agreement (joint ownership / joint publication).
Held at WHO HQ. No IRB, no hardware — a data-use request only.

**Drafted analysis outline (ready to submit):**
> *Title:* Age-banded diagnostic value of WHO tachypnea thresholds for in-hospital outcomes in child
> pneumonia — a retrospective analysis of the PREPARE database.
> *Aim:* Quantify, by age band (<2mo / 2–12mo / 12–59mo), the sensitivity/specificity/PPV of the WHO
> fast-breathing thresholds (≥60/≥50/≥40) at admission for **in-hospital mortality and respiratory
> support**, and whether a continuous RR / RR-trend improves discrimination over the binary threshold.
> *Method:* subject-level logistic / survival models, age-band bootstrap CIs, calibration; compare
> operating points to PRISM's non-contact estimates (bridges "tachypnea→outcome", lapse #2).
> *Output:* the first outcome-anchored evidence that the screening endpoint PRISM targets is
> prognostically meaningful — converts the proxy critique into a validated link.

## 3. Thermal method dev (adult public data; neonatal thermal stays future)
Public adult thermal + RR-GT sets for prototyping the thermal channel: **SIM1** (68 subj, LWIR),
**RR-rPPG** (119 facial + RR GT), **FLIR benchmark** (71). Real-neonate thermal remains future work.

---

## Honest boundary
- **Scriptable-from-here:** the radar/ToF neonatal test (downloading), the thermal adult method dev.
- **Needs the PI:** the Mendeley browser download (then overlap check), the PREPARE WHO request. Both
  are desk-doable (no IRB/hardware) — just not automatable from this shell.
