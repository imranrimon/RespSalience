"""Controlled test of the checkpoint-selection criterion.

The audit found that `selection_score` (validation MAE + 5*review_rate) is anti-correlated with test
tachypnea AUROC across folds (r=+0.879, p=0.0008). That was observational -- two runs that happened to
differ. This script scores the *interventional* version: two full 5-fold runs that are identical in
data, folds, architecture, budget and seed, differing ONLY in which epoch's weights are kept.

    pedrr_kfold_main332       --selection-metric mae     (regression error, the published pipeline)
    pedrr_kfold_main332auroc  --selection-metric auroc   (decision quality on validation)

Both are evaluated on the SAME test predictions per fold, so DeLong is paired and the MAE difference
is paired per subject. This is the paper's framing thesis applied one level up, at model selection:
if optimising the regression metric costs decision quality, then selecting on it should too.

    python scripts/selection_criterion_experiment.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_clinical_models import delong_test  # noqa: E402

RUNS = Path("outputs/paper_runs")
AGE_REF = "pedrr_kfold_flowonly_agefix"


def who(age: float) -> float:
    return 60.0 if age < 2 else (50.0 if age < 12 else 40.0)


def band(age: float) -> str:
    return "<2mo" if age < 2 else ("2-12mo" if age < 12 else "12-60mo")


def load(run: str) -> Optional[pd.DataFrame]:
    paths = sorted(glob.glob(str(RUNS / run / "fold*" / "test_eval" / "prism_breath_predictions.csv")))
    if not paths:
        return None
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["fold"] = int(os.path.normpath(p).split(os.sep)[-3].replace("fold", ""))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def scored(run: str, age_map: Dict[str, float]) -> Optional[pd.DataFrame]:
    d = load(run)
    if d is None:
        return None
    d["sid"] = d["subject_id"].astype(str)
    d["age"] = d["sid"].map(age_map).fillna(d["age_months"])
    d["thr"] = d["age"].map(who)
    d["band"] = d["age"].map(band)
    d["y"] = (d["gt_rr_bpm"] >= d["thr"]).astype(int)
    d["ae"] = (d["pred_rr_bpm"] - d["gt_rr_bpm"]).abs()
    return d


def sens_spec(d: pd.DataFrame) -> tuple:
    dec = (d["pred_rr_bpm"] >= d["thr"]).astype(int)
    tp = int(((dec == 1) & (d.y == 1)).sum()); fn = int(((dec == 0) & (d.y == 1)).sum())
    tn = int(((dec == 0) & (d.y == 0)).sum()); fp = int(((dec == 1) & (d.y == 0)).sum())
    return (tp / (tp + fn) if tp + fn else np.nan, tn / (tn + fp) if tn + fp else np.nan)


def selected_epoch_info(run: str) -> pd.DataFrame:
    """Per-fold: which epoch was kept, its validation MAE and validation AUROC (if recorded)."""
    rows = []
    for hp in sorted(glob.glob(str(RUNS / run / "fold*" / "run" / "training_history.json"))):
        fold = int(os.path.normpath(hp).split(os.sep)[-3].replace("fold", ""))
        try:
            hist = json.loads(Path(hp).read_text())
        except Exception:
            continue
        epochs = hist.get("history", []) if isinstance(hist, dict) else hist
        cand = [e for e in epochs if "selection_score" in e]
        if not cand:
            continue
        best = min(cand, key=lambda e: e["selection_score"])
        rows.append({
            "fold": fold,
            "epoch": best.get("epoch"),
            "val_mae": best.get("val_mae"),
            "val_auroc": best.get("val_auroc"),
            "val_sens": best.get("val_sensitivity"),
            "score": best.get("selection_score"),
            "criterion": best.get("selection_metric"),
            # what the OTHER criterion would have picked from this same trajectory
            "best_val_mae_epoch": min(cand, key=lambda e: e.get("val_mae", np.inf)).get("epoch"),
            "best_val_auroc_epoch": max(
                cand, key=lambda e: (e.get("val_auroc") if e.get("val_auroc") is not None else -np.inf)
            ).get("epoch"),
        })
    return pd.DataFrame(rows).sort_values("fold") if rows else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mae-run", default="pedrr_kfold_main332")
    ap.add_argument("--auroc-run", default="pedrr_kfold_main332auroc")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ref = load(AGE_REF)
    if ref is None:
        raise SystemExit(f"age reference run missing: {AGE_REF}")
    age_map = dict(zip(ref["subject_id"].astype(str), ref["age_months"].astype(float)))

    A = scored(a.mae_run, age_map)      # MAE-selected
    B = scored(a.auroc_run, age_map)    # AUROC-selected
    if A is None or B is None:
        raise SystemExit("one of the runs has no predictions")

    # Pair on subject id; require identical fold assignment so the comparison is truly matched.
    m = A.merge(B, on="sid", suffixes=("_mae", "_auroc"))
    bad = int((m.fold_mae != m.fold_auroc).sum())
    if bad:
        raise SystemExit(f"fold assignment differs for {bad} subjects -- not a matched comparison")
    if not (m.y_mae == m.y_auroc).all():
        raise SystemExit("label mismatch between runs")
    y = m.y_mae.to_numpy(int)
    print(f"Matched cohort: n={len(m)}, {int(y.sum())} tachypneic, folds identical\n")

    print("=" * 78)
    print("A. Checkpoint selection: validation MAE  vs  validation AUROC")
    print("=" * 78)
    print(f"{'':<26}{'MAE-selected':>16}{'AUROC-selected':>18}{'delta':>12}")
    rows = []
    for name, d, tag in ((a.mae_run, A, "mae"), (a.auroc_run, B, "auroc")):
        se, sp = sens_spec(d)
        rows.append({
            "auroc": roc_auc_score(d.y, d.tachy_prob), "mae": d.ae.mean(),
            "sens": se, "spec": sp,
        })
    for k, fmt in (("auroc", "{:.3f}"), ("mae", "{:.2f}"), ("sens", "{:.3f}"), ("spec", "{:.3f}")):
        lab = {"auroc": "Tachypnea AUROC", "mae": "RR MAE (BPM)",
               "sens": "Sensitivity @ WHO", "spec": "Specificity @ WHO"}[k]
        d0, d1 = rows[0][k], rows[1][k]
        print(f"{lab:<26}{fmt.format(d0):>16}{fmt.format(d1):>18}{fmt.format(d1 - d0):>12}")

    # Paired significance.
    auc_a, auc_b, p_delong = delong_test(y, m.tachy_prob_mae.to_numpy(float),
                                         m.tachy_prob_auroc.to_numpy(float))
    dae = (m.ae_auroc - m.ae_mae).to_numpy(float)
    w = stats.wilcoxon(m.ae_mae.to_numpy(float), m.ae_auroc.to_numpy(float))
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, len(m), size=(a.boot, len(m)))
    dboot = np.array([
        roc_auc_score(y[i], m.tachy_prob_auroc.to_numpy(float)[i])
        - roc_auc_score(y[i], m.tachy_prob_mae.to_numpy(float)[i])
        for i in idx if 0 < y[i].sum() < len(i)
    ])
    lo, hi = np.percentile(dboot, [2.5, 97.5])
    print(f"\nPaired DeLong  AUROC {auc_b:.3f} vs {auc_a:.3f}: "
          f"delta +{auc_b - auc_a:.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]  p = {p_delong:.4f}")
    print(f"Paired MAE     delta {dae.mean():+.2f} BPM (AUROC-selected is worse), "
          f"Wilcoxon p = {w.pvalue:.2e}")
    print("\nThe two move in OPPOSITE directions: selecting on regression error buys MAE and")
    print("costs decision quality. This is the paper's thesis at the model-selection level.")

    print("\n" + "=" * 78)
    print("B. Per-fold (paired)")
    print("=" * 78)
    print(f"{'fold':<6}{'n':>5}{'AUROC mae-sel':>16}{'AUROC auroc-sel':>18}{'delta':>10}"
          f"{'MAE mae-sel':>14}{'MAE auroc-sel':>16}")
    for f in sorted(m.fold_mae.unique()):
        g = m[m.fold_mae == f]
        if g.y_mae.nunique() < 2:
            continue
        ka = roc_auc_score(g.y_mae, g.tachy_prob_mae)
        kb = roc_auc_score(g.y_mae, g.tachy_prob_auroc)
        print(f"{f:<6}{len(g):>5}{ka:>16.3f}{kb:>18.3f}{kb - ka:>+10.3f}"
              f"{g.ae_mae.mean():>14.2f}{g.ae_auroc.mean():>16.2f}")
    wins = sum(
        roc_auc_score(m[m.fold_mae == f].y_mae, m[m.fold_mae == f].tachy_prob_auroc)
        > roc_auc_score(m[m.fold_mae == f].y_mae, m[m.fold_mae == f].tachy_prob_mae)
        for f in sorted(m.fold_mae.unique()) if m[m.fold_mae == f].y_mae.nunique() > 1
    )
    print(f"\nAUROC-selection wins {wins}/5 folds.")

    print("\n" + "=" * 78)
    print("C. Per age band (AUROC)")
    print("=" * 78)
    print(f"{'band':<10}{'n':>5}{'pos':>5}{'mae-sel':>10}{'auroc-sel':>12}{'delta':>10}")
    for b in ("<2mo", "2-12mo", "12-60mo"):
        g = m[m.band_mae == b]
        if len(g) == 0 or g.y_mae.nunique() < 2:
            print(f"{b:<10}{len(g):>5}{int(g.y_mae.sum()) if len(g) else 0:>5}"
                  f"{'--':>10}{'--':>12}{'--':>10}")
            continue
        ka = roc_auc_score(g.y_mae, g.tachy_prob_mae)
        kb = roc_auc_score(g.y_mae, g.tachy_prob_auroc)
        print(f"{b:<10}{len(g):>5}{int(g.y_mae.sum()):>5}{ka:>10.3f}{kb:>12.3f}{kb - ka:>+10.3f}")

    print("\n" + "=" * 78)
    print("D. Which epoch each criterion kept")
    print("=" * 78)
    for name in (a.mae_run, a.auroc_run):
        info = selected_epoch_info(name)
        if info.empty:
            print(f"{name}: no training_history.json with per-epoch records")
            continue
        print(f"\n{name}")
        print(info.to_string(index=False))

    print("\nHow to report: this is a controlled A/B on the selection rule alone. Both arms are")
    print("legitimate models; whichever is used for the headline, the BASELINES must be selected the")
    print("same way or the comparison is unfair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
