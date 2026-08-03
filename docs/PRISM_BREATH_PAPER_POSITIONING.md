# PRISM-Breath — Paper Positioning (framing improvements)

The experiments are done; how they're *framed* is high-leverage and free. Three moves:

## 1. Lead with the paradigm shift, not RGB screening
The strongest, most novel result is: **RGB optical flow physically cannot see the neonatal band (60–90
BPM) — every extractor saturates at ~26–35 — but radar recovers it exactly (MAE 0.00 across 20–80 BPM,
laser-validated; being extended to 50 real children).** Open with this. It reframes the contribution
from "another rPPG method" (a saturated field where AIRFlowNet leads at ~3 BPM) to "the first
characterization of the RGB neonatal limit + a sensing-modality fix." That is a field-level insight.

## 2. Headline the thesis: *raw MAE ≠ clinical utility*
The cleanest, most defensible message: four SOTA extractors match-or-beat PRISM on **RR-MAE**, yet all
are significantly worse on the **clinical endpoint (tachypnea AUROC)** — because PRISM optimizes
*screening discrimination*, not point error. Back it with: PRISM AUROC 0.869–0.919 vs 0.727–0.752
(p≤0.002); raw-RR AUROC 0.877 > 0.786 (not a head artifact); **decision-curve net benefit positive and
> treat-all/none at every threshold**; calibrated `tachy_prob` (ECE 0.064). This is a metric-choice
argument the whole rPPG field should hear.

## 3. Present the lean model as a methodological contribution
The load-bearing audit + `prism_lean` validation (lean = full, all n.s.) shows performance comes from
**SSL-pretrained encoders (spectral + scan + separate trunks) → simple fusion → RR + tachy head** — the
prototype bank, codebook, het head, and reliability-weighting are *empirically decorative* (with data
behind every cut; SSL pretrain IS load-bearing, Sens +0.116 sig). Ship the lean model and present the
audit — it turns "over-engineered" into "principled, ablation-justified design."

## Scope / title
"**Non-contact respiratory-rate estimation and tachypnea screening in infants ≥2 months, with a
sensing-modality path to the neonate**" — NOT "ARI screening." State the `<2mo` referral policy and the
tachypnea≠disease limitation (VEP AUROC 0.59) up front; honesty is a strength here.

## Honest limitations to state plainly (each disarms a reviewer)
neonatal-RGB limit (defer) · tachypnea is a weak disease proxy (VEP 0.59) · single clinical dataset +
manual-count GT · no skin-tone audit (single-population; sex audit = no disparity) · radar validated on
phantom + 50 real children but not real neonates-in-cribs · TSCAN-flow ≈ PRISM (extractor-robust; the
apparent gain was leakage — caught by the clean-leakage run).

## The narrative arc (results order)
(1) RGB neonatal limit characterized → (2) radar recovers it (paradigm fix) → (3) for ≥2mo, PRISM is a
validated screen (Sens 0.71/AUROC 0.919, seed-averaged, calibrated, net-benefit-positive) → (4) MAE≠
utility (SOTA table) → (5) lean model + load-bearing audit → (6) cross-dataset + in-domain AIR (2.93 ≈
AIRFlowNet) → (7) honest limitations + future work (prospective, PREPARE, real-neonate radar).
