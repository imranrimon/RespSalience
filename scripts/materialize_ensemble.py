"""Write the seed-ensembled predictions out as an ordinary run directory.

The published model is a seed ensemble, but most analysis scripts take a single --run-root and
re-derive from per-fold prediction CSVs. Rather than teach each of them to ensemble (and risk them
ensembling slightly differently), materialise the ensemble once, in the standard layout, so every
downstream script sees the published model by pointing at one path.

Averaging is over the per-subject predictions the seeds agree on, fold assignment and ground truth
are carried through unchanged (they are identical across seeds by construction, and this asserts it).

Run:  python scripts/materialize_ensemble.py \
        --runs pedrr_kfold_salience_nossl332det_seed0..5 --out pedrr_kfold_salience_published
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "outputs", "paper_runs")

# Columns that are model OUTPUT and must be averaged; everything else is carried through and checked.
AVERAGED = ["pred_rr_bpm", "tachy_prob", "pred_bpm"]


def expand(spec):
    """Accept an explicit list, or a 'prefix_seed0..5' range."""
    out = []
    for s in spec:
        m = re.match(r"^(.*seed)(\d+)\.\.(\d+)$", s)
        if m:
            out += [f"{m.group(1)}{i}" for i in range(int(m.group(2)), int(m.group(3)) + 1)]
        else:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = expand(a.runs)
    print(f"ensembling {len(runs)} runs -> {a.out}")

    folds = sorted({int(re.search(r"fold(\d+)", p).group(1))
                    for p in glob.glob(os.path.join(RUNS, runs[0], "fold*"))})
    for k in folds:
        frames = []
        for r in runs:
            p = os.path.join(RUNS, r, f"fold{k}", "test_eval", "prism_breath_predictions.csv")
            if not os.path.exists(p):
                raise SystemExit(f"missing: {p}")
            frames.append(pd.read_csv(p).sort_values("subject_id").reset_index(drop=True))

        base = frames[0].copy()
        # The seeds must describe the same subjects and the same labels, or averaging is meaningless.
        for f in frames[1:]:
            if not base["subject_id"].equals(f["subject_id"]):
                raise SystemExit(f"fold{k}: subject sets differ between seeds")
            if "gt_rr_bpm" in base and not np.allclose(base["gt_rr_bpm"], f["gt_rr_bpm"]):
                raise SystemExit(f"fold{k}: ground truth differs between seeds")

        for c in AVERAGED:
            if c in base.columns:
                base[c] = np.mean([f[c].to_numpy(float) for f in frames], axis=0)

        out = os.path.join(RUNS, a.out, f"fold{k}", "test_eval")
        os.makedirs(out, exist_ok=True)
        base.to_csv(os.path.join(out, "prism_breath_predictions.csv"), index=False)
        print(f"  fold{k}: {len(base)} rows")
    print(f"wrote outputs/paper_runs/{a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
