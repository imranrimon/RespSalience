# RespSalience

**Decision-aware contactless pediatric tachypnea screening from RGB video.**

[![checks](https://github.com/imranrimon/RespSalience/actions/workflows/checks.yml/badge.svg)](https://github.com/imranrimon/RespSalience/actions/workflows/checks.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Research Use](https://img.shields.io/badge/license-Research%20Use-lightgrey.svg)](LICENSE)
[![Claims verified](https://img.shields.io/badge/manuscript%20claims-95%2F95%20verified-brightgreen.svg)](scripts/verify_manuscript_claims.py)

Tachypnea — an age-elevated breathing rate — is the WHO's frontline sign for pediatric acute
respiratory infection, and counting it by eye is the step that fails most often where the burden is
highest. This repository estimates it from an ordinary RGB camera, with no contact and no added
hardware.

Its central claim is not a better rate estimator. It is that **how the clinical decision is formed
matters more than the architecture that produces the rate**. A model that emits a *distribution* over
respiratory rates and decides by integrating the probability mass above the WHO age threshold beats a
multimodal regression system that is otherwise strictly larger — and beats every published extractor
we could run under one protocol.

| Method | Type | RR-MAE ↓ | Tachypnea AUROC ↑ | ΔAUROC | DeLong *p* |
|---|---|---|---|---|---|
| **RespSalience** *(proposed)* | distributional | **5.21** | **0.935** | — | — |
| PRISM-Breath *(our baseline)* | regression | 6.42 | 0.826 | +0.109 | 3.2×10⁻⁵ |
| DeepPhys | regression | 5.62 | 0.777 | +0.158 | 1.9×10⁻⁵ |
| VIRENet | regression | 5.91 | 0.760 | +0.174 | 4.2×10⁻⁶ |
| EfficientPhys | regression | 6.10 | 0.755 | +0.180 | 2.2×10⁻⁶ |
| PhysFormer | regression | 6.65 | 0.734 | +0.200 | 4.5×10⁻¹⁰ |
| TS-CAN | regression | 6.15 | 0.726 | +0.208 | 1.8×10⁻⁷ |

<sub>298-subject cohort common to all methods, subject-disjoint five-fold CV, corrected age labels,
identical decision rule. RespSalience is the three-seed ensemble. Paired exact McNemar on the WHO
decision at a single operating point is **not** significant for any pair (*p* = 0.117–1.000); the
ranking evidence is the AUROC gap, and we report both.</sub>

RespSalience is the **simpler** of the two models: one channel, no multimodal fusion, no auxiliary
components. That is the paper's point, not an accident of tuning.

---

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Reproducing the paper](#reproducing-the-paper)
- [What the paper establishes](#what-the-paper-establishes)
- [Repository layout](#repository-layout)
- [Data access](#data-access)
- [Intended use and limitations](#intended-use-and-limitations)
- [Citing this work](#citing-this-work)

## Installation

`environment.yml` is the authoritative pinned specification (note: it is UTF-16 encoded).
`requirements.txt` is a minimal pip subset for analysis-only use.

```bash
git clone https://github.com/imranrimon/RespSalience.git
cd RespSalience
conda env create -f environment.yml
conda activate resp_env
```

A CUDA build of PyTorch is required for training. The analysis and verification entry points are
CPU-only and run against cached per-fold predictions.

> Scripts resolve paths relative to the repository root by design. Run them from the root, as the
> commands below do.

## Quickstart

Estimate a rate and a WHO band decision for one video:

```bash
python pipeline.py --mode cli --video path/to/clip.mp4 --age 7
```

Confirm the published numbers on your checkout — no GPU, no dataset download:

```bash
python scripts/verify_manuscript_claims.py
```

## Reproducing the paper

Every quantitative claim in the manuscript is re-derived from stored artifacts by a single command.
It prints each claim beside the value recomputed from the underlying per-fold predictions, and exits
non-zero on any mismatch:

```bash
python scripts/verify_manuscript_claims.py          # 95 claims checked, 95 match
python scripts/verify_manuscript_claims.py --tol 0.002
```

This is deliberately a *re-derivation*, not a re-read: nothing in it trusts a cached summary table
where the underlying predictions are available. It exists because numbers were carried between drafts
by hand and drifted, and it is the guard against that recurring.

| Paper element | Command |
|---|---|
| `tab:main` — proposed model vs prior work | `python scripts/verify_manuscript_claims.py` |
| `tab:headline` — proposed vs baseline, per age band | `python scripts/train_resp_salience_ssl.py --seed {0,1,2} --kfold pedrr_kfold_main332 --tag _ssl332` |
| `tab:sota` — five published extractors | `python scripts/full_metric_comparison.py --deepphys100 --prism-run outputs/paper_runs/pedrr_kfold_main332` |
| `tab:selmatrix` — selection rule × model family | `python scripts/selection_matrix.py` |
| `tab:abl` — ablation of the regression baseline | `python scripts/regenerate_supplementary_tables.py --prism-run pedrr_kfold_main332` |
| `tab:ceiling` — reference-standard sensitivity analysis | `python scripts/label_noise_ceiling.py` |
| `tab:disttriage` — triage read-out comparison | `python scripts/distributional_triage.py` |
| `fig:escalation` — neonatal escalation pathway | `python scripts/escalation_pathway.py --run-root pedrr_kfold_main332` |
| `fig:framing` — decision-threshold sweep | `python scripts/framing_bradypnea.py` |
| `sec:age` — age-conditional read-out | `python scripts/operating_point_readout.py --run-root outputs/paper_runs/pedrr_kfold_salience_ssl332_seed0` |
| `sec:extval` — external replication (negative) | `python scripts/nchsdb_headswap.py` and `python scripts/nchsdb_framing.py --seeds 5` |
| Subgroup / fairness | `python scripts/subgroup_fairness.py --deepphys100 --prism-run outputs/paper_runs/pedrr_kfold_main332` |
| Calibration and conformal coverage | `python scripts/calibration_diagnostics.py --deepphys100` |

[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) covers the full pipeline, including signal extraction
and training from raw video. [`docs/PRISM_BREATH_PIPELINE.md`](docs/PRISM_BREATH_PIPELINE.md)
documents the manifest-driven benchmark stages.

## What the paper establishes

**A regress-then-threshold pipeline cannot dissociate regression accuracy from clinical utility,**
because its ranking *is* its rate estimate. This is why RR-MAE ranks the published extractors almost
perfectly (Spearman ρ = −0.90) and yet stops tracking utility the moment the objective changes.
`scripts/selection_matrix.py` demonstrates it by intervention rather than by correlation.

**Checkpoint selection is part of the decision rule.** Selecting on validation regression error rather
than on the clinical endpoint measurably changes that endpoint — and the published extractors we
compare against were never checkpoint-selected at all; their drivers evaluate whatever weights remain
resident when training returns. `scripts/identify_baseline_checkpoint.py` verifies this bit-exactly,
and `scripts/reselect_baseline_checkpoints.py` gives every model the same treatment before comparison.

**The uncertainty layer inherits the read-out's weakness.** The deployed escalation triage forms a
conformal interval around the *regressed* rate, which is the read-out the paper argues against.
Replacing it with abstention on the salience mass cuts total downstream burden from 52.4% to 41.9%
(paired reduction 0.106 [0.051, 0.163], McNemar *p* = 3.1×10⁻⁴) — see `scripts/distributional_triage.py`.

**A negative external replication, reported as such.** The framing advantage does *not* replicate on an
independent neonatal cohort with a clean contact belt. We report the null, and localise the claim to
degraded and saturating front ends rather than quietly dropping it.

## Repository layout

```
src/                  importable package (installed as `respsalience`)
  core/               models, signal processing, fusion, diagnosis
  utils/              dataset registries, ground-truth handling, paths
  config/             YAML/JSON configs (conformal widths, inference)
  api/  ui/           FastAPI service and Streamlit dashboard
scripts/              run-from-root entry points: extraction, training, evaluation, analysis
manuscript/           LaTeX sources, 600-DPI figures, bibliography
docs/                 pipeline documentation, runbooks, design notes
tools/                developer utilities (manuscript consistency checks)
tests/                standalone self-checking modules (no pytest suite)
external/             vendored third-party baselines — treat as read-only
```

`data/`, `outputs/`, and `external/` are gitignored: datasets, checkpoints, and vendored code are not
redistributed here.

## Data access

No dataset is redistributed by this repository. Each remains governed by its provider's terms.

| Dataset | Role | Access |
|---|---|---|
| Pediatric-RR-332 | primary clinical cohort | Mendeley `72dd3rkttf` (public) |
| NCH Sleep DataBank | external replication | PhysioNet, credentialed access |
| AIR-125 / AIR-400 | cross-dataset generalization | from the original authors |

The Pediatric-RR reference rate is a **multi-reader video expert panel consensus**, not a single
bedside count: 40.4% of released values are exact half-integers with no quarter-values, the signature
of a two-reader mean. This matters for how the reference standard is described and for what a
label-noise ceiling analysis can conclude.

## Intended use and limitations

This is a **research artifact, not a medical device.** It has not been cleared or approved by any
regulatory authority and must not be used to inform care for an individual patient. Tachypnea is a
screening sign, not a diagnosis.

Known limitations, stated because they bound what the results mean:

- **Neonates (<2 months) are escalated, not screened.** Every RGB extractor we tested saturates near
  30 breaths/min for true rates of 60–80, so camera sensitivity in this band collapses. The system
  routes these cases to a contact or radar measurement by policy rather than clearing them.
- **One clinical cohort.** The primary results rest on a single 332-subject pediatric cohort across
  three sites. The one external replication we ran is negative and reported as such.
- **Reference-standard noise bounds the achievable ceiling** — see the sensitivity analysis in
  `scripts/label_noise_ceiling.py`, which is labelled as such rather than used to explain away results.
- **The screening advantage is concentrated in hard regimes** (younger bands, the highest-prevalence
  site) and is null in the easiest ones.

## Citing this work

The accompanying manuscript is under review; please cite the software metadata in
[`CITATION.cff`](CITATION.cff) in the meantime. GitHub renders a formatted citation from that file
under *Cite this repository*.

## License

Research use. Non-commercial academic research and teaching, with attribution. See [`LICENSE`](LICENSE)
for the full terms, including the medical-device disclaimer and the dataset carve-out.

## Contact

Muhammad Imran Hossain — West Virginia University — <muhammadimran.hossain@mail.wvu.edu>
