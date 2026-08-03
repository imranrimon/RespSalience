# PRISM-Breath — Tier 2/3 Protocols (execution-ready, public-data-first)

The dataset audit (2026-08) changes the picture: the **radar path is implementable on PUBLIC data with
zero hardware/IRB**; thermal (neonatal) and pneumonia-outcome remain genuine future work. Each item
below is either RUNNABLE-NOW (public data + this repo) or a DEFER with the honest blocker stated.

---

## Tier 2A — Radar/depth neonatal channel  ✅ RUNNABLE NOW (public data)
The single fix for the fatal `<2mo` lapse, without a sensor.

**Public datasets:**
- **Neonatal thorax-simulator ToF + radar** (PMC10805796): radar + Time-of-Flight depth, **RR 5–80 BPM**
  (spans the exact 60–80 neonatal band PRISM saturates on), synchronized reference. Phantom, open-source.
- **Radar-recorded child vital signs** (MDPI Sensors 2021, 21/7/2412): **50 real children <13 yr**, FMCW
  radar + clinical reference (BSM6501K), public repo + MATLAB extraction example.
- **120 GHz mmWave + reference** (Nature Sci Data 2026); **multi-target FMCW** (PMC11532813) — method dev.

**Experiment (runnable):**
1. Download the thorax-simulator (ToF+radar) + child-radar datasets.
2. Build a standard FMCW/ToF RR extractor: range-FFT → phase of the chest bin → band-pass 0.15–1.5 Hz →
   FFT peak. Reuse `signal_processor.py` band-pass/FFT-peak/SNR.
3. **Key test:** RR-MAE **stratified by rate**, especially **60–90 BPM** — show radar/ToF recovers the
   neonatal band where every RGB extractor saturates at ~35 (see `neonatal-extractor-limit-reframe`).
4. **Fusion:** add radar as a 4th modality to PRISM via the new **SNR-gated fusion** (`--use-snr-fusion`);
   the SNR gate will trust radar when the RGB flow SNR collapses (neonatal) — the intended mechanism.

**Deliverable:** the first neonatal-capable PRISM configuration, validated on public data. Flips the
paper's worst result. Caveat to state: simulator/child-vehicular data ≠ real neonates in cribs — frame
as feasibility, and list a real-neonate radar cohort as the confirmatory future step.

## Tier 2B — Thermal neonatal channel  ⚠️ DEFER (no public dataset)
Thermal achieves ~1.5 brpm on neonates (PMC10175339) but **no neonatal thermal dataset is public** (only
pilots). Blocker: data, not method. Future work; or approach the pilot authors for a data-sharing MOU.

## Tier 2C — Gold-standard GT / disease labels  ⚠️ MOSTLY DEFER
- ChARM VEP pneumonia video set: **not released**. Defer the pneumonia-outcome validation.
- Partial public resource: **neonatal lung-ultrasound dataset, 86 patients incl. Transient Tachypnea of
  the Newborn (TTN)** — real tachypnea *disease* labels, but ultrasound modality. Usable to study the
  tachypnea→disease link, not to validate the video pipeline directly.
- Actionable now: **VEP-style re-annotation of a Pediatric-RR subset** by 2–3 trained counters to
  quantify our manual-count GT noise (needs annotators, not hardware).

## Tier 3A — Neonatal-first multimodal foundation model (spec)
- Inputs: RGB-flow + radar/ToF (+thermal when available), one shared SSL encoder (rPPG-MAE / PhySU-Net
  style masked pretraining), multi-vital heads (HR + RR).
- Physiology prior: **RSA HR–RR coupling** as a consistency/anomaly term (flag when coupling is
  implausible → built-in plausibility check the current model lacks).
- Fusion: the SNR-gated reliability layer (now implemented) generalized to N modalities.
- Adaptation: few-shot / meta-learn the neonatal regime (data-scarce).

## Tier 3B — Prospective clinical validation (statistical plan, to the ChARM/VEP bar)
- Population: children 0–59 mo with suspected ARI; reference = VEP + gold-standard RR (RIP/capnography);
  outcome = clinician ARI/pneumonia dx + admission/deterioration.
- Endpoints: age-band Sens/Spec/AUROC with subject bootstrap CIs; **equity audit** across skin tone /
  lighting; abstention/defer rate and its effect on missed cases at a fixed review budget.
- Powering: size for a ±0.07 Sens CI in the `<2mo` band (the current fatal gap), ≈ the limiting stratum.
- Blocker: prospective cohort + IRB + clinical partners — genuine future work.

---

## REVISION (2026-08) — "genuinely blocked" was too pessimistic; public-data paths found
- **Disease outcome / "tachypnea ≠ ARI" (was Tier 2C/3B):** the **PREPARE database — 285,839 children
  with pneumonia, 41 datasets, RR + IMCI danger signs + SpO2 + OUTCOMES** (PMC9798037, FAIR, managed
  access via data-use request) enables a **retrospective** validation that tachypnea predicts pneumonia
  outcomes. No IRB/hardware — just a data-use agreement. This is the biggest unblock.
- **Second RR video dataset (external validation, was "single dataset"):** a public **pediatric RR video
  dataset, children <5** (BMC Res Notes 2026, 10.1186/s13104-026-07677-x) — candidate external set.
- **Thermal method dev (was Tier 2B):** public **ADULT** thermal datasets — SIM1 (68, LWIR), RR-rPPG
  (119 facial + RR GT), FLIR benchmark (71) — prototype the thermal RR method now; real-neonate thermal
  stays future.
- **Still inherently future:** a truly PROSPECTIVE trial, and real-neonate-in-crib thermal/radar (the
  simulator + child-radar datasets are proxies) — confirmatory, not blocking.

## Execution order (what a session CAN do vs CANNOT)
1. **Now, in this repo:** SNR-gated fusion retrain (STAGED, PID 23928) → in-domain AIR vs VIRENet →
   **Tier 2A radar on public data** (download + extractor + high-RR-band test + radar-PRISM fusion).
2. **Needs external resources (future work / limitations):** Tier 2B thermal data, Tier 2C VEP/pneumonia
   labels, Tier 3B prospective cohort. State these explicitly as limitations — do not overclaim.
