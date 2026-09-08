# Main body vs. appendix: what earns a place, and why

The results section has grown by accretion — most subsections were added to answer a specific reviewer
objection, and they are now ordered by the history of the rebuttal rather than by the argument. This is
a proposed allocation. The test applied to each item is:

> **Does the paper's central claim collapse without it?**
> If yes, main body. If it only *defends* the claim against a specific objection, appendix.
> If it neither supports nor defends the claim, cut it.

The central claim, stated once so everything can be measured against it:

> *How the clinical decision is formed matters more than the architecture that produces the rate.
> A regress-then-threshold pipeline cannot dissociate rate accuracy from clinical utility, because its
> ranking is its rate; reading the decision off a distribution can.*

---

## Main body — the argument (keep)

| Section | Why it must stay |
|---|---|
| **Fig. 1 overview** | States the mechanism visually and grounds it in one real neonate. This is the paper. |
| **RespSalience + PRISM-Breath (method)** | The two read-outs being compared. Irreducible. |
| **Primary clinical performance** (`tab:headline`, `tab:main`) | The headline result, and the one-protocol comparison against five published extractors. |
| **The clinical task is classification, not regression** (`sec:framing`) | The thesis stated and tested across backbones. The strongest section in the paper. |
| **Isolating the objective from the representation** (`sec:objective`) | The frozen-encoder head-swap. This is what turns a correlation into a causal claim, and no other experiment does that job. |
| **Component isolation** (`sec:simple`) | Shows the gain is the read-out, not the machinery. Now reinforced by the pretraining null. |
| **Age bands and the neonatal read-out** (`sec:age`, `tab:readout`) | The clinical payload: a band that looks unscreenable under regression is screenable under the distribution. |
| **Where RR-MAE stops tracking utility** (`sec:sota`) | Establishes that the field's default metric does not measure the clinical endpoint — the premise the whole paper rests on. |
| **External replication that fails** (`sec:extval`) | A negative result on our own thesis. Removing it would be the single most damaging thing we could do to the paper's credibility. |
| **Discussion / limitations** | Contains the neonatal blind spot, single-cohort caveat, and reference-standard dependence. |

## Move to the appendix — defences, not arguments

| Section | Rationale for moving |
|---|---|
| **Checkpoint selection is part of the decision rule** (`sec:selmatrix`) | A genuinely good finding, but it defends the comparison's fairness rather than advancing the thesis. Keep two sentences in the main body pointing to the appendix: extractors were never checkpoint-selected, and we gave every model the same treatment. |
| **The triage layer should use the same read-out** (`sec:disttriage`) | Conceptually important — the incumbent triage contradicts the thesis — but its effects are modest and only partly replicate across seed halves. Compress to one paragraph in `sec:age` and move the table and the split analysis to the appendix. |
| **Pooled vs. Mondrian conformal comparison** (inside `sec:age`) | Justifies a calibration choice. Belongs with the conformal appendix. |
| **Architecture comparisons** (`sec:abl`) and **Why we abandoned fusion** | Both say "the machinery does not matter." That point is made once, better, by the component isolation. Merge into a single short main-body sentence, full tables to the appendix. |
| **Extractor robustness and a retracted result** | Honest, but it is housekeeping about a prior claim. Appendix, referenced from limitations. |
| **Cross-dataset generalization** (AIR-125/400) | These sets contain no fast-breathing cases, so they cannot test the thesis. That limitation is worth one sentence in the main body; the numbers belong in the appendix. |
| **Calibration, decision curve, equity** (`sec:calib`) | ECE and net benefit support deployability, not the central claim. Keep the ECE figure reference in the main body; move DCA and subgroup tables. |

## Consider cutting

- **Radar** and **`tab:clin`** were already cut. Do not reintroduce them: the radar set is not an
  independent cohort for our purposes and invites a scope objection.
- **Edge-deployment cost (parameters and CPU)** — no reviewer asked, and it invites an efficiency
  comparison the paper is not designed to win.
- **Decision-focused training does not improve on calibration alone** — a null on a variant nobody
  proposed. One sentence in future work is enough.

---

## Results that are missing and would strengthen the paper

Ordered by value per unit of work.

1. **A second cohort with real tachypnea.** This is the paper's binding limitation: every positive
   result rests on one 332-subject cohort. NCHSDB was the attempt and it is negative for reasons we
   now understand (a clean contact belt does not saturate, so there is no read-out gap to recover).
   The thesis predicts the gain appears *only* where the front end saturates — which is a testable
   prediction, and the strongest possible framing of the negative result.
2. **A prospective operating-point check.** Every operating point is chosen out-of-fold but on the
   same cohort. Even a small held-out site would convert "chosen on calibration data" into
   "validated prospectively."
3. **Inter-reader variability on our own predictions.** The reference is a two-reader consensus; the
   label-noise ceiling analysis is a simulation. Real repeat readings on a subset would replace a
   sensitivity analysis with a measurement.
4. **A saturation-severity axis.** The thesis says the gain scales with how much the front end
   saturates. We have the ingredients (extractors of differing quality, the NCHSDB belt as a
   non-saturating control) to plot gain against saturation directly. This would turn the central
   claim from a comparison into a dose-response relationship, and would explain the negative
   replication rather than merely reporting it.

Item 4 is the one to do next if there is appetite for another experiment: it uses only existing data,
and it converts the paper's biggest apparent weakness — a failed replication — into evidence for the
mechanism.
