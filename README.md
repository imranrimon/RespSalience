# RespSalience

Code and analysis for **decision-aware contactless pediatric tachypnea screening**.

The clinical target is tachypnea by WHO age band (≥60 breaths/min under 2 months, ≥50 at 2–12 months,
≥40 at 12–60 months) estimated from an ordinary RGB camera. The central finding is that **how the
clinical decision is formed matters more than the architecture that produces the rate**: a model that
emits a *distribution over respiratory rates* and decides by integrating the mass above the age
threshold outperforms a multimodal regression system that is otherwise strictly larger.

| model | role | RR-MAE ↓ | tachypnea AUROC ↑ |
|---|---|---|---|
| **RespSalience** | proposed | **5.21** | **0.935** |
| PRISM-Breath | regression baseline | 6.42 | 0.826 |
| Best published extractor (DeepPhys) | prior work | 5.62 | 0.777 |

*n = 298 matched subjects, subject-disjoint five-fold CV, corrected age labels. RespSalience is the
three-seed ensemble. All six paired comparisons significant at p < 10⁻⁴.*

RespSalience is the **simpler** of our two models: one channel, no multimodal fusion, no auxiliary
components. That is the point of the paper, not an accident.

## What this repository is

Research code accompanying the manuscript in [`manuscript/`](manuscript/). It is a reproduction
artifact, not a deployable medical device. **Tachypnea is a screening sign, not a diagnosis**, and
nothing here is cleared for clinical use.

## Reproducing the paper

Every quantitative claim in the manuscript is re-derived from stored artifacts by a single command:

```bash
python scripts/verify_manuscript_claims.py
```

It prints each claim beside the value recomputed from the underlying per-fold predictions and exits
non-zero on any mismatch. At the time of writing: **53 claims checked, 53 match.**

The table below maps each result in the paper to the command that produces it.

| Paper element | Command |
|---|---|
| `tab:main` — proposed model vs prior work | `python scripts/verify_manuscript_claims.py` |
| `tab:headline` — proposed vs baseline, per age band | `python scripts/train_resp_salience_ssl.py --seed {0,1,2} --kfold pedrr_kfold_main332 --tag _ssl332` |
| `tab:selmatrix` — selection rule × model family | `python scripts/selection_matrix.py` |
| `tab:sota` — five published extractors | `python scripts/full_metric_comparison.py --deepphys100 --prism-run outputs/paper_runs/pedrr_kfold_main332` |
| `tab:abl` — ablation of the regression baseline | `python scripts/regenerate_supplementary_tables.py --prism-run pedrr_kfold_main332` |
| `tab:ceiling` — reference-standard sensitivity analysis | `python scripts/label_noise_ceiling.py` |
| `sec:age` — age-conditional read-out | `python scripts/operating_point_readout.py --run-root outputs/paper_runs/pedrr_kfold_salience_ssl332_seed0` |
| `sec:age` — selective read-out (deployed rule) | `python scripts/hybrid_readout.py` |
| `sec:extval` — external replication (negative) | `python scripts/nchsdb_headswap.py` and `python scripts/nchsdb_framing.py --seeds 5` |
| `tab:crossdata` — cross-dataset generalization | `python scripts/run_expanded_paper_baselines.py --preset paper_full` |
| Subgroup / fairness | `python scripts/subgroup_fairness.py --deepphys100 --prism-run outputs/paper_runs/pedrr_kfold_main332` |
| Calibration and conformal coverage | `python scripts/calibration_diagnostics.py --deepphys100` |

See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) for the full pipeline, including signal extraction
and training from raw video.

## Two findings worth reading the code for

**Checkpoint selection is part of the decision rule.** Selecting a checkpoint by validation regression
error rather than by the clinical endpoint measurably changes the endpoint, and the published
extractors we compare against were never checkpoint-selected at all — their drivers evaluate whatever
weights remain resident after training returns. `scripts/identify_baseline_checkpoint.py` verifies this
bit-exactly; `scripts/reselect_baseline_checkpoints.py` gives every model the same treatment.

**A regress-then-threshold pipeline cannot dissociate regression accuracy from clinical utility**,
because its ranking *is* its rate estimate. This is why RR-MAE ranks the five published extractors
almost perfectly (Spearman ρ = −0.90) yet fails across a change of objective. `scripts/selection_matrix.py`
demonstrates it by intervention.

## Environment

Conda environment `resp_env`, Python 3.13, CUDA build. `environment.yml` is the authoritative pinned
spec (note: UTF-16 encoded); `requirements.txt` is a minimal pip subset.

```bash
conda env create -f environment.yml
conda activate resp_env
```

## Data

None of the datasets are redistributed here.

| Dataset | Role | Access |
|---|---|---|
| Pediatric-RR-332 | primary clinical cohort | Mendeley `72dd3rkttf` (public) |
| NCH Sleep DataBank | external replication | PhysioNet credentialed access |
| AIR-125 / AIR-400 | cross-dataset generalization | from the original authors |

The Pediatric-RR reference rate is a **multi-reader video expert panel consensus**, not a single
bedside count — 40.4% of released values are exact half-integers with no quarter-values, the signature
of a two-reader mean. This matters for how the reference standard is described.

## Repository layout

```
src/respsalience/     importable library (models, signal processing, evaluation)
scripts/              entry points: extraction, training, evaluation, analysis
manuscript/           LaTeX sources, figures, bibliography
docs/                 pipeline documentation and design notes
tools/                developer utilities
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

See [`LICENSE`](LICENSE).
