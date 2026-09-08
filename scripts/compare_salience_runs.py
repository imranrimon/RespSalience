"""Does the checkpoint-saving re-run reproduce the canonical RespSalience predictions?

The manuscript's headline numbers are re-derived from the prediction files in
pedrr_kfold_salience_ssl332_seed{0,1,2}. Re-training to obtain releasable weights produces a fresh
set under ..._ssl332ckpt_seed{0,1,2}. Those two need not agree: cuDNN kernel selection and
non-deterministic reductions mean a rerun at the same seed can differ, and this project has already
been bitten by seed-level sign flips.

So the rerun is NOT promoted automatically. This script quantifies the gap and says plainly whether
the released checkpoints correspond to the published numbers.

Run:  python scripts/compare_salience_runs.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "outputs", "paper_runs")
CANON = "pedrr_kfold_salience_ssl332_seed{}"
RERUN = "pedrr_kfold_salience_ssl332ckpt_seed{}"


def load(run):
    ps = sorted(glob.glob(os.path.join(RUNS, run, "fold*", "test_eval",
                                       "prism_breath_predictions.csv")))
    if not ps:
        return None
    d = pd.concat([pd.read_csv(p) for p in ps], ignore_index=True)
    return d.groupby("subject_id", sort=False).agg(
        gt_tachy=("gt_tachy", "max"), gt_rr_bpm=("gt_rr_bpm", "mean"),
        tachy_prob=("tachy_prob", "mean"), pred_rr_bpm=("pred_rr_bpm", "mean")).reset_index()


def ens(runs):
    ds = [load(r) for r in runs]
    if any(d is None for d in ds):
        return None
    sid = sorted(ds[0].subject_id)
    b = ds[0].set_index("subject_id").loc[sid].copy()
    b["tachy_prob"] = np.mean([d.set_index("subject_id").loc[sid].tachy_prob.to_numpy(float) for d in ds], axis=0)
    b["pred_rr_bpm"] = np.mean([d.set_index("subject_id").loc[sid].pred_rr_bpm.to_numpy(float) for d in ds], axis=0)
    return b.reset_index()


def au(d):
    return roc_auc_score(d.gt_tachy.to_numpy(int), d.tachy_prob.to_numpy(float))


def mae(d):
    return float((d.pred_rr_bpm - d.gt_rr_bpm).abs().mean())


def main():
    print(f"{'run':>10s} {'canonical AUROC':>16s} {'rerun AUROC':>13s} "
          f"{'canonical MAE':>14s} {'rerun MAE':>11s} {'max |dProb|':>12s}")
    missing = False
    for s in range(3):
        c, r = load(CANON.format(s)), load(RERUN.format(s))
        if c is None:
            print(f"  seed{s}: canonical run missing"); missing = True; continue
        if r is None:
            print(f"  seed{s}: rerun missing -- has train_salience_checkpoints.ps1 finished?")
            missing = True
            continue
        m = c.merge(r, on="subject_id", suffixes=("_c", "_r"))
        dp = float((m.tachy_prob_c - m.tachy_prob_r).abs().max())
        print(f"{'seed'+str(s):>10s} {au(c):>16.4f} {au(r):>13.4f} {mae(c):>14.3f} "
              f"{mae(r):>11.3f} {dp:>12.4f}")

    if missing:
        print("\nIncomplete: cannot judge reproduction yet.")
        return 1

    ce, re_ = ens([CANON.format(i) for i in range(3)]), ens([RERUN.format(i) for i in range(3)])
    m = ce.merge(re_, on="subject_id", suffixes=("_c", "_r"))
    dp = float((m.tachy_prob_c - m.tachy_prob_r).abs().max())
    print(f"\n{'ENSEMBLE':>10s} {au(ce):>16.4f} {au(re_):>13.4f} {mae(ce):>14.3f} "
          f"{mae(re_):>11.3f} {dp:>12.4f}")

    d_au, d_mae = au(re_) - au(ce), mae(re_) - mae(ce)
    print(f"\nensemble delta: AUROC {d_au:+.4f}, MAE {d_mae:+.3f}")
    print(f"published headline: AUROC 0.935, MAE 5.21")

    # A checkpoint release is honest only if the weights correspond to the published numbers.
    if abs(d_au) <= 0.005 and abs(d_mae) <= 0.05:
        print("\nVERDICT: the rerun reproduces the published ensemble within tolerance.\n"
              "The checkpoints may be released as corresponding to the reported results.")
        return 0
    print("\nVERDICT: the rerun does NOT reproduce the published ensemble.\n"
          "Do not describe these checkpoints as the ones behind the reported numbers. Either release\n"
          "them with the rerun's own metrics stated, or report the run-to-run spread as a robustness\n"
          "result -- but do not silently swap the canonical predictions for these.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
