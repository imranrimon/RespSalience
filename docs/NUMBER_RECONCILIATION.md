# Number reconciliation across paper tables

Authoritative mapping so every table uses ONE subject basis and ONE metric definition, clearly
labeled. Source of truth: `scripts/reconcile_numbers.py` (run from repo root). Numbers below are the
current run; the 298-matched baseline row finalizes after DeepPhys-100ep + the #24 re-extraction.

## The two axes of confusion
1. **Subject basis.** PRISM-family covers all **332** pedrr subjects; the published extractors were
   cached for only **298** ("hang-free"). So any PRISM-vs-baseline table MUST use the 298 matched
   intersection (paired/fair); PRISM-only tables (ablation, component isolation, calibration) use 332.
2. **"Regression" definition.** Three different scores have all been called "PRISM":
   - **rate-margin** (`pred_rr - WHO_threshold`) = the *pure* regression-then-threshold rule.
   - **`tachy_prob` head** = a *learned* classifier on fused features (NOT pure regression).
   - **ROCKET / salience** = classification on the raw signal.
   The framing thesis's regression baseline should be the **rate-margin**, not the head.

## Canonical PRISM numbers (label N on every table)
| basis | AUROC head | AUROC **margin** (=regression) | AUROC rawrr | AUPRC | Sp@Se90 | MAE |
|---|---|---|---|---|---|---|
| **332 full** | 0.879 | **0.830** | 0.888 | 0.794 | 0.671 | 6.19 |
| **298 matched** | 0.869 | **0.820** | 0.877 | 0.792 | 0.644 | 6.40 |

Baselines (298 only): EfficientPhys 0.755 / TS-CAN 0.726 / VIRENet 0.760 / DeepPhys 0.762 (head==margin, their prob is monotonic in margin).

## CONVENTION (adopt everywhere)
- **PRISM-vs-baseline tables → 298 matched.** PRISM AUROC there is **0.869** (not 0.879).
- **PRISM-only tables → 332 full**, labeled "$N{=}332$".
- **Framing regression baseline = rate-margin** (332: 0.830 / 298: 0.820), not the 0.879 head.
- Every table caption states N and the score definition.

## Table-by-table audit
| Table | Basis | Issue → fix |
|---|---|---|
| tab:data | — | none |
| tab:simple (component isolation) | 332 | OK; label $N{=}332$. "Full PRISM 0.879" = head (fine here, PRISM-only). |
| **tab:framing** | mixed! | (1) extractors "0.73–0.76" are **298** inside a 332 table → split or label per row. (2) "Full PRISM **0.879** / Regression→threshold" is the **head**; add/relabel the **pure-regression margin 0.830** as the baseline → framing gain becomes 0.830→0.925 (**+0.095**, larger & cleaner) rather than 0.879→0.925 (+0.046). |
| **tab:sota** | 298 | PRISM must read **0.869** (matched), not 0.879. (Prior auto-table showed 0.879 from an older run — supersede with the final harness run.) |
| Table 1 (tab:clin) | 332, 3-seed | 0.892 (full) vs tab:framing 0.879 (single-run) — reconcile: state both are PRISM but one is 3-seed-averaged, or use one run throughout. |
| tab:abl | 332 | OK; label $N{=}332$. |
| **draft tab:opchar** | 298 | ✓ already 0.869 etc. Add "$N{=}298$ matched" to caption. |
| draft tab:extval | NCHSDB n=65 | separate cohort; no pedrr reconciliation. |
| draft tab:calib | 332 | PRISM-only conformal; label $N{=}332$. |
| draft §subgroup | 298 | ✓; label. |
| **draft §framing (E2 para)** | mixed | EfficientPhys/TS-CAN are 298; PRISM row was 332 (reg 0.830). Recompute PRISM on 298 (reg **0.820**) so all three rows are 298, OR label the PRISM row $N{=}332$. |

## Open (resolve at finalization)
- 0.869 vs 0.879 vs 0.892 all denote "PRISM tachypnea AUROC" on different (basis, seeding) — pick the
  final run + basis per table and regenerate. The auto/manual `full_metric_comparison.py --deepphys100`
  run is the canonical 298 source; a 3-seed 332 run is the canonical PRISM-only source.
