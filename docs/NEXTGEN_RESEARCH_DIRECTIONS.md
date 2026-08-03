# NextGen Research Directions for PRISM-Breath

*SOTA landscape, gap analysis, and a prioritized plan to turn this project into a strong journal contribution.*
*Compiled 2026-07-04 from a fact-checked literature sweep (2022–2026). Every numeric claim below carries a citation; MAE/RMSE are in breaths-per-minute (BPM) unless noted. Read §5 first if you only want the plan.*

---

## 0. TL;DR

The entire *methodological frontier* of camera-based physiology — efficient state-space (Mamba) backbones, codebook denoising, depth+optical-flow geometric models, multitask HR+RR — has been built and validated **on adult facial video for heart rate**. Infant *respiration* is stuck in a small-data regime where the best method is a plain optical-flow net at ~2.9 BPM MAE and depth methods sit near ~5 BPM. **The genuine white space is the transfer + clinical-generalization gap**, not inventing yet another fusion block. PRISM-Breath already has most of the *machinery* (reliability fusion, codebook, domain adapters, conformal triage). The winning move is to **re-scope to one clinically-framed story — robust, calibrated infant RR + tachypnea triage that generalizes across sites/ages/skin-tone — and be the first to bring the adult-HR frontier (Mamba efficiency, codebook denoising, depth+flow) to infant respiration on a real clinical external set (Pediatric-RR-332).**

---

## 1. SOTA map (2022–2026)

### 1a. Infant / neonatal specific (the population that matters)

| Method / source | Modality | Dataset (n) | Reported error | Key limitation (self-reported) |
|---|---|---|---|---|
| **AIRFlowNet** (arXiv 2307.13110, MICCAIW 2023) | Optical flow, infant-tuned | AIR-125 (125 vids / 8 subj) | **~2.9 BPM MAE** (adult models 4.7–6.2 on same data) | Tiny data; only one public infant RR set existed |
| **AIR-400 / WACV 2026** (arXiv 2512.06888) | RGB, reproducible infant pipeline | AIR-400 (400 vids / 18 subj) | Extends AIR-125; core thesis = *"overcoming small-data limitations"* | Still ~18 subjects; no reproducible adult-derived algo worked on infants |
| **RGB-D depth NICU** (arXiv 2412.06012 / Sci Rep 2025, PMC12081704) | Azure Kinect depth + Fourier | 14 preterm, RR-GT on 3 ventilated / 6 non-vent | **MAE 4.83 (vent) / 4.99 (non-vent)** | Single well-resourced hospital; tiny n; hardware depth |
| **EVM NICU** (PMC11573411 / PubMed 39559825, 2024) | Eulerian magnification + adaptive peak | 13 neonates, 288 vids | **ICC 0.91 / 0.88**, Pearson 84.7% / 77.4% | *Only clinically-stable infants (no true apnea)*; limb-motion sensitive; **~9.41 s per second of video** compute |
| **Pediatric-RR-332** (BMC Res Notes 2026, doi:10.1186/s13104-026-07677-x) | RGB chest-wall video | 332 vids, Bangladeshi children 0–59 mo | *Data descriptor* — real clinical corpus; human video-panel agreed within 2 BPM in **88.7%** of cases (mean diff −0.3) | Ground truth is expert-panel, not gold sensor |

**Takeaway:** infant RR is small-n, single-center, and the strongest numbers are not apples-to-apples (different populations, motion regimes, references). AIRFlowNet's ~2.9 and depth's ~5 are the honest infant baselines to beat.

### 1b. The adult-facial frontier (transferable methods, HR-centric)

| Method / source | What it contributes | Reported error | Transfer status |
|---|---|---|---|
| **rPPG-Toolbox** (NeurIPS 2023, arXiv 2210.00716) | Canonical suite: DeepPhys, PhysNet, TS-CAN, EfficientPhys, PhysFormer, BigSmall | — | You already vendor most of these |
| **MTTS-CAN** (NeurIPS 2020, arXiv 2006.03790) | Multitask HR+RR, **>150 fps on ARM CPU** | — | Real-time, but adult |
| **CliffPhys** (ECCV 2024, air.unimi.it 11393) | **Monocular depth scalar field + 2D optical-flow vector field → Clifford neural layers** | RR **MAE 0.83 / RMSE 1.97** on COHFACE; beats MTTS-CAN 2.83, PhysFormer 3.72, ContrastPhys 4.73, BigSmall 5.58 | **Validates PRISM's exact depth+flow choice — but on adults.** Untransferred to infants |
| **PhysMamba** (CCBR 2024, arXiv 2409.12031) | Temporal-Difference Mamba + SlowFast state-space backbone | HR-only | **HR-only, adults, no RR, no infants** |
| **RhythmMamba** (arXiv 2404.06483) | Mamba frame-stem; **+319% throughput, −23% peak memory** vs Transformer | HR-only (0.75–2.5 Hz) | **HR-only, adults, no RR, no infants** |
| **CodePhys** (IEEE TCSVT/TBME 2025, arXiv 2502.07526) | rPPG as **code-query in a noise-free codebook** built from clean PPG | HR-only | **HR-only** — your "clean-breathing codebook" is the RR adaptation nobody published |
| **Complex-valued multitask rPPG+RR** (Sci Rep 2025, s41598-025-23103-x) | Joint HR+RR CNN; quantifies skin-tone shift | RR MAE **0.69 (Caucasian) → 0.85 (Asian) → 0.88 (African-American)**; rPPG 0.69 → 0.93 | Documents the fairness gap |

### 1c. Other non-contact modalities (context; lower-verification — see caveats)

These surfaced in the sweep but were not in the top adversarially-verified set, so treat the numbers as indicative, not settled:
- **UWB radar** (Vitalthings Guardian M10): RR MAE ~0.31 BPM in an adult ED — the accuracy ceiling contactless can reach with active sensing.
- **Thermal** (YOLO-nostril + Kalman on Jetson Orin Nano): MAE ~0.57 BPM, 10 adults, real-time edge.
- **Optical-flow survey** (ACM Comp. Surveys 2025): optical-flow motion methods beat rPPG/DL for *RGB respiration*; LCV-RAFT best; COHFACE RAFT MAE ~0.64 BPM — corroborates that flow is the right primary RGB signal.
- **Demographic-bias meta-analysis** (npj Digital Medicine 2025): chrominance rPPG degrades from MAE 5.2 → 14.1 across Fitzpatrick skin types (HR).
- **Multimodal edge-robot RR** (2026): RGB+thermal+NIR+low-light with brightness-adaptive sensor selection on Jetson.

---

## 2. Open gaps the literature itself names

1. **Small, single-center, low-diversity data.** AIR: ~18 infants; NICU RGB-D: 14 preterm, RR-GT on 3; EVM: 13 neonates. Every infant result has wide uncertainty and near-zero external validation.
2. **No transfer of the modern frontier to infants/RR.** Mamba (PhysMamba, RhythmMamba) and codebook denoising (CodePhys) are HR-only, adult-only. CliffPhys proves depth+flow works *for adult RR* but was never run on infants.
3. **Domain shift is real and quantified but unsolved for infants.** Skin-tone monotonic degradation (0.69→0.88 RR MAE; 5.2→14.1 HR MAE). Nobody has an age/site/skin-tone generalization benchmark for infant RR.
4. **Motion & apnea are failure modes, not features.** NICU EVM works *only on clinically stable infants*, is limb-motion sensitive, and cannot detect true apnea — yet apnea is the highest-value neonatal event.
5. **Uncertainty / calibrated abstention is essentially absent** from published camera-RR infant work.
6. **Clinical + edge validation missing.** Compute costs like ~9.41 s/s (EVM) are deployment-hostile; the actual ARI-screening use case is low-resource (LMIC) settings — exactly Pediatric-RR-332's Bangladesh cohort — where no camera model has been validated.

---

## 3. Datasets & their limits

| Dataset | Content | Limitation |
|---|---|---|
| **AIR-125 / AIR-400** | Infant RR video, varied pose/light/camera | ~8→18 subjects; primary but small |
| **Pediatric-RR-332** | 332 real clinical chest-wall videos, Bangladesh 0–59 mo | **Underused gold: real ARI-screening population.** Expert-panel GT |
| **MINI-RGBD** | Synthetic infant RGB-D (SMIL body model) | Synthetic; good for pretraining/motion-sim, not clinical claims |
| **ICBHI / Resp. Sound DB** | Respiratory audio | Adult-heavy, cross-modal only |
| **MIMIC-IV-WDB** | Waveforms | Not video; pretraining/aux only |

**Under-exploited:** Pediatric-RR-332 is a *real clinical LMIC cohort* — it should be your **external test set**, turning generalization from a claim into a demonstrated result.

---

## 4. Emerging directions (2024–2026) mapped to what you already have

| Direction | SOTA status | PRISM-Breath status | Verdict |
|---|---|---|---|
| State-space / **Mamba** backbones | Hot, HR-only | You have `MambaFusionModel`, `AdaptiveScanBlock` | **Under-exploited — first-mover on infant RR** |
| **Codebook / diffusion denoising** | CodePhys (HR-only); you have `ConditionalDiffusion1D` | Clean-breathing codebook implemented | **Adapt to RR = novel** |
| **Depth + optical-flow** geometric | CliffPhys (adult RR, ECCV'24) | Depth + flow already core | Add CliffPhys as a baseline; you're aligned |
| Multitask HR+RR | MTTS-CAN, mature | Not a current claim | Optional |
| **Conformal / calibrated uncertainty** | Rare in camera-RR | You have disagreement-conditioned conformal | **Differentiator — push to clinical decision-curve** |
| **VLM clinical context** | Immature, over-claimed field-wide | Implemented | Keep secondary (weak labels / narrator), don't headline |
| Federated / privacy | Sparse for pediatrics | Not implemented | Future work |
| **Edge / low-latency** | Thermal/Mamba edge demos exist | Not benchmarked | **High-value for LMIC story** |

---

## 5. The plan — NextGen problem statements, ranked

> **Framing recommendation:** the repo feels haphazard because it spans etiology, longitudinal forecasting, foundation-model reasoning, and audio diagnosis. Your own `PRISM_BREATH_JOURNAL_BLUEPRINT.md` already says to cut those. **Do that.** Pick **one headline + two method novelties** from below. Everything else becomes "future work."

### Tier 1 — pick ONE as the headline

**H1. Cross-population generalization benchmark for infant RR + safe tachypnea triage.**
*Problem:* every infant method is single-center; nobody has shown generalization across site, age, skin-tone, lighting for infant RR.
*Solution (mostly infra you have):* canonical subject-disjoint + **leave-one-dataset-out** protocol; train on AIR-400 (+MINI-RGBD pretrain), **external-test on Pediatric-RR-332**; source/age domain adapters (`SourceAgeAdapter`); report **subgroup MAE by age bin, dataset, and a lighting/skin-tone proxy**; couple with **conformal selective triage** and a **clinical decision-curve / net-benefit analysis** at WHO age thresholds.
*Why it wins:* directly attacks gaps #1, #3, #5, #6; honest, hard to referee-reject, and clinically legible. This is the strongest single story.

### Tier 2 — pick TWO as method contributions

**M1. Motion- & apnea-robust respiration (event detection, not just a number).**
*Problem:* SOTA explicitly fails on limb motion and cannot detect apnea — the most actionable neonatal event.
*Solution:* add an **apnea/respiratory-pause head** + **motion-artifact gating** driven by your `modality_reliability` / disagreement signal; pretrain pause/motion representations self-supervised on unlabeled + MINI-RGBD synthetic video with **simulated apnea and limb-motion augmentations** (extend `src/utils/augmentations.py`). Report event sensitivity/specificity, not only RR MAE.

**M2. Efficient state-space backbone + edge deployment for LMIC screening.**
*Problem:* Mamba's efficiency (RhythmMamba: +319% throughput, −23% memory) has never touched infant RR; the real use case is cheap edge hardware.
*Solution:* benchmark a **Mamba/`AdaptiveScanBlock` temporal backbone** against your Transformer fusion for RR; report **accuracy vs latency/energy on an edge device (Jetson)** via `scripts/export_edge.py`. First Mamba-for-infant-RR + first infant-RR edge result.

**M3. Clean-breathing codebook + diffusion denoising for corrupted infant signals.**
*Problem:* CodePhys's codebook denoising is HR-only.
*Solution:* formalize a **clean-breathing prior** learned from high-SNR signals (your codebook) and/or a **1D conditional-diffusion denoiser** (`ConditionalDiffusion1D` already exists) to restore motion-corrupted infant waveforms; ablate codebook-size and denoiser on/off. First codebook/diffusion denoising for respiration.

### Tier 3 — keep as secondary / future work
- VLM as **weak-label generator + explainability narrator** (not a diagnostic claim).
- Etiologic ARI diagnosis, longitudinal deterioration, comorbidity reasoning — **explicitly future work** until new labels exist (per blueprint).

### Recommended combination
**H1 (headline) + M1 + M2.** It yields a coherent paper: *"A robust, efficient, calibrated system for contactless infant respiratory-rate estimation and tachypnea triage that generalizes to a real clinical LMIC cohort and detects motion/apnea failure states — deployable on the edge."* M3 becomes a strong ablation section rather than a third headline.

---

## 6. How to complete the work (concrete, mapped to your repo)

Status legend: ✅ implemented in-repo · ☐ still to do.

1. ☐ **Freeze scope** to H1+M1+M2 (edit `PRISM_BREATH_JOURNAL_BLUEPRINT.md`); move etiology/longitudinal/audio-diagnosis to a "Future Work" appendix.
2. ✅ **Wire Pediatric-RR-332 as the external test set.** `build_respiratory_benchmark_manifest.py` now holds it out entirely (default `--external-test-datasets Pediatric-RR`), writing `data/organized/respiratory_benchmark_external_test.json`; train/val/test are AIR-only and subject-disjoint. `--pool-external` reverts to legacy pooling. (`respiratory_benchmark_registry.py`, `dataset_paths.py`). LODO splits also retained.
3. ✅ **Two modern baselines added:** `--model-variant cliffphys` (Clifford Cl(2,0) geometric-product layers) and `--model-variant mamba` (genuine selective SSM) in `src/core/baseline_wrappers.py`, dispatched through the shared train/eval/conformal protocol. Smoke test: `python src/core/test_baseline_wrappers.py`.
4. ✅ **Apnea/motion heads + augmentation:** `PrismBreathModel` now emits `apnea_prob`/`motion_prob`; `src/utils/augmentations.py` gained `simulate_apnea`/`simulate_motion_artifact`/`apply_event_augmentations`; train with `--enable-event-augmentation` (gated weak supervision, backward compatible). Event probs surface in the eval CSV.
5. ✅ **Edge latency benchmark:** `python scripts/export_edge.py --benchmark --variants prism_breath mamba cliffphys airflow_compact --batch-sizes 1 8 --quantize --report <json>` reports params, fp32/INT8 size, latency percentiles and throughput per variant. (Honest finding: the pure-Python selective scan makes `mamba` CPU-slow — a fused/associative-scan kernel is needed to realize the efficiency claim on the edge; CliffPhys's vectorized layers are fastest.)
6. ◑ **Clinical rigor:** ✅ decision-curve / net-benefit + multi-seed subgroup analysis implemented in `scripts/clinical_analysis.py` (net benefit vs "treat all"/"treat none" and an idealized-triage policy; subgroup mean±std by dataset / WHO age bin / disagreement bin; `--plot` renders the decision-curve figure). ☐ Still to do: actually run the 3 seeds (42/52/62) end-to-end and point the analysis at those outputs (calibrated + external settings); optionally add a lighting proxy once a label exists.
7. ◑ **Reproducibility hygiene (fixes the "haphazard" feel):** ✅ stale `f:/` paths fixed — `src/config/test_infer.yaml` and both Ultralytics `settings.json` are now project-root-relative, `train_all.bat` env-name aligned to `resp_env`. ✅ memory-unsafe TSCAN handled — dropped from default suites and gated behind `--allow-unstable-baselines` (`run_fair_benchmark.py` / `run_expanded_paper_baselines.py`), plus a real mitigation (batch size forced to 1 for TSCAN in `external_runner.py`). ☐ Still to do: release manifests + code.
8. ☐ **Honesty section:** state the small-n uncertainty, apples-to-oranges cross-method caveats, and any over-conservative calibration as limitations rather than burying them — referees reward this in clinical-AI venues.

### New runbook commands (implemented above)
```bash
# External-test manifest (Pediatric-RR-332 held out as OOD)
python scripts/build_respiratory_benchmark_manifest.py --base-dir data --output-dir data/organized

# Baseline + event-robust training
python scripts/train_prism_breath.py --model-variant mamba    ...   # or cliffphys
python scripts/train_prism_breath.py --model-variant prism_breath --enable-event-augmentation ...

# Evaluate on the external OOD cohort (any checkpoint)
python scripts/evaluate_prism_breath.py --manifest data/organized/respiratory_benchmark_external_test.json \
  --checkpoint <ckpt> --signal-dir data/prism_breath_signals --output-dir outputs/external_eval

# Edge latency comparison
python scripts/export_edge.py --benchmark --variants prism_breath mamba cliffphys airflow_compact \
  --batch-sizes 1 8 --quantize --report outputs/edge_latency.json
```

**Suggested venue fit:** IEEE JBHI / TBME, MICCAI, or a WACV/CVPR applications track — all match the "robust + calibrated + clinically-validated contactless pediatric vitals" framing.

---

## 7. Caveats on this research

- Fast-moving 2024–2026 area; several sources are very recent (WACV 2026 AIR-400, BMC 2026 Pediatric-RR-332) or preprints — table values were verified but venue-final numbers may shift.
- Strongest infant evidence is tiny-n (n=3–18); all infant MAE/ICC figures have wide uncertainty.
- Cross-method MAE is **not** apples-to-apples: adult 0.83 BPM (CliffPhys/COHFACE) ≠ infant 2.9 (AIRFlowNet/AIR-125) ≠ neonatal ~5 (RGB-D NICU). Lower adult numbers do **not** imply infant transfer — the opposite is documented (adult models score 4.7–6.2 on infant data).
- Several baseline comparisons are authors' own re-implementations.
- Thermal, radar/RF, diffusion, federated, and VLM sub-topics had no top-tier *verified* claims in this pass (only indicative numbers in §1c) — treat them as under-explored, not settled.

## Key sources
- Song et al., *Overcoming Small Data Limitations in Video-Based Infant Respiration Estimation*, WACV 2026 — arXiv:2512.06888
- *AIRFlowNet / AIR-125*, MICCAIW 2023 — arXiv:2307.13110
- RGB-D NICU RR, Sci Rep 2025 — arXiv:2412.06012 / PMC12081704
- EVM NICU RR, 2024 — PMC11573411 / PubMed 39559825
- *Pediatric-RR-332*, BMC Research Notes 2026 — doi:10.1186/s13104-026-07677-x
- *rPPG-Toolbox*, NeurIPS 2023 — arXiv:2210.00716
- *MTTS-CAN*, NeurIPS 2020 — arXiv:2006.03790
- *CliffPhys*, ECCV 2024 — air.unimi.it/2434/1125560
- *PhysMamba*, CCBR 2024 — arXiv:2409.12031
- *RhythmMamba* — arXiv:2404.06483
- *CodePhys*, IEEE 2025 — arXiv:2502.07526
- Complex-valued multitask rPPG+RR (skin-tone shift), Sci Rep 2025 — nature.com/articles/s41598-025-23103-x
