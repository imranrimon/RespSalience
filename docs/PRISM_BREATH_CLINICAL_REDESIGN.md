# PRISM-Breath — Clinical-Screening Redesign Blueprint

_Decided 2026-07-11. North star chosen by the PI: **validated non-contact tachypnea/ARI
screening for low-resource settings.** This doc is the concrete "how"._

## 0. Why redesign (evidence, not vibes)
1. **The old objective measured the wrong thing.** After fixing the AIR-125 GT bug (2–3× fps
   inflation), the in-distribution AIR benchmark is degenerate — RR ~24 BPM, std ~1.4, **zero
   tachypnea**. Old headline (MAE 3.44, r 0.987) was a labeling artifact; a constant predictor
   ties SOTA there.
2. **The architecture doesn't earn its complexity.** Ablations already show `prism_no_scan`
   (3.29) and `prism_shared_trunk` (3.81) beat the full model; scan block, expert routing, and
   Mamba/Clifford backbones don't pay off in the small-data regime.
3. **The bottleneck is data + clinical grounding, not modeling.** The only data with real RR
   spread + tachypnea is Pediatric-RR (was mis-labeled "external"); public infant *tachypnea
   video* essentially doesn't exist.

**Redesign in one line:** stop competing on RR-MAE on tiny degenerate benchmarks; move the
complexity budget from *architecture* to *data + clinical validation*, and reframe around the
decision that has value — trustworthy non-contact tachypnea/ARI screening under scarcity & shift.

## 1. Reframed problem & success metrics
- **Task:** per-infant **fast-breathing (tachypnea) classification at WHO age thresholds**
  (≥60 BPM <2mo, ≥50 2–12mo, ≥40 12–60mo) — a clinical decision, reported **by age band**.
  RR regression becomes a secondary/means-to-an-end output.
- **Primary metrics:** sensitivity, specificity, PPV/NPV, AUROC **per age band**, plus
  **decision-curve / net-benefit** at clinically relevant thresholds (use existing
  `scripts/clinical_analysis.py`). Bland–Altman for RR agreement. Operating point framed as
  "≥90% sensitivity at a controlled referral/abstain rate" (don't miss pneumonia).
- **Anchor dataset:** Pediatric-RR-332 (RR 16–96, 72 tachypnea, 51% infants, WHO age bands) —
  now the primary in-distribution benchmark (`pedrr_indist_{train,val,test}.json`, 232/50/50).

## 2. Keep / Cut / Build
**Keep (recast with clinical motivation):**
- Reliability-weighted multimodal fusion → **graceful degradation** when a modality is
  missing/unreliable (phone RGB-only fallback). This is now a *deployment* story, not accuracy.
- SSL pretraining, source/age domain adapters → **cross-site generalization**.
- Conformal / selective triage → **calibrated abstention** (safety).
- Edge export tooling → **low-cost deployment** evidence.
- The GT-audit rigor → a **benchmark-fidelity contribution** (see Pillar D).

**Cut / demote:** scan block, expert routing (ablation-negative); RR-MAE-on-AIR as a headline;
codebook/prototypes unless they help *classification* (re-ablate under the new objective); keep
VLM context light (nice-to-have, not load-bearing). Demote AIR-125/400 to a low-variance
robustness probe.

**Build (the four pillars below).**

## 3. The four pillars
### A. Clinical-endpoint harness (near-term, cheap — do first)
Tachypnea classification by age band (sens/spec/PPV/NPV/AUROC) + net-benefit/decision-curve +
Bland–Altman. This becomes the paper's Table 1 and makes the pivot real. Feeds directly off the
Pediatric-RR run already in flight.

### B. Generative data for the positive class (the methods contribution)
The classifier needs tachypnea it can't get from real data. Synthesize breathing across the
**full RR range incl. 60–90 BPM**:
- Start simple & physical: RR-conditioned chest-motion + respiration-waveform simulator (drive
  depth/flow/audio surrogates at target RR). Leverage MINI-RGBD (synthetic infant depth) and the
  figshare ToF+radar **phantom (5–80 BPM)** as controlled-RR anchors.
- Optional escalation: diffusion/generative video of respiration conditioned on RR.
- **Claim to prove:** synthetic tachypnea augmentation improves **sensitivity on held-out real
  tachypnea** (Pediatric-RR positives), with a sim-to-real ablation. This attacks the field's
  core blocker head-on.

### C. Trustworthy abstention under shift (the safety mechanism)
Conformal coverage broke under external shift (observed: external coverage 0.64 < target). Move
to **shift-aware selective prediction** (Mondrian/group-conditional or shift-adaptive conformal;
or learned abstention with guarantees). Report coverage honestly under external shift. Clinical
operating point = target sensitivity with controlled referral rate.

### D. Benchmark-fidelity + deployment realism
- **Label-fidelity method:** generalize the GT audit (detect sampling-rate/scale/label errors in
  RR video datasets); present the AIR-125 bug as a cautionary case study — publishable rigor.
- **Deployment:** RGB-only degradation, skin-tone/lighting/camera/site robustness, edge latency.
- **External validation:** Cambridge Meerkat (CC-BY, download now, signal-level probe), phantom
  (tachypnea-range calibration), pursue CHEO/Carleton DUA for a real external cohort.

## 4. Evidence hierarchy (paper narrative)
1. In-distribution (Pediatric-RR): tachypnea classification by age band + calibrated abstention.
2. Generative-augmented: synthetic tachypnea → higher real-tachypnea sensitivity (sim-to-real).
3. External (Meerkat / phantom / CHEO): generalization + honest coverage-under-shift.
4. Deployment: RGB-only fallback, robustness, edge latency.
5. Benchmark-fidelity: the label-audit method + AIR case study.

## 5. Phased execution
- **P0 (now, in flight):** Pediatric-RR 3-seed train/eval → first clinical numbers.
- **P1 (cheap):** clinical-endpoint harness (Pillar A) — the concrete pivot.
- **P2:** generative tachypnea data + sim-to-real (Pillar B).
- **P3:** shift-aware abstention (Pillar C).
- **P4:** external validation + deployment/robustness (Pillar D).
- **P5:** re-ablate the simplified model under the classification objective; write the audit.

## 6. Positioning
Target **npj Digital Medicine / Lancet Digital Health / PLOS Digital Health / IEEE JBHI** —
clinical-impact venues where "non-contact ARI screening for under-5s in low-resource settings,
validated + calibrated + deployable" is a strong, uncrowded story. Context: ARI/pneumonia is a
leading cause of under-5 mortality; WHO IMCI decisions hinge on fast-breathing counts that are
error-prone in the field. A camera/phone screener is genuinely novel and useful.

## 7. Open questions (pending in-flight deep-research `wey0rxln5`)
- Best generative approach for infant respiration (physics vs diffusion) and its SOTA precedent.
- Which shift-aware uncertainty method has the strongest track record under clinical shift.
- Closest prior art to a non-contact ARI screener (novelty delta) and the right clinical baselines
  (e.g., UNICEF/ARIDA automated RR counters).
