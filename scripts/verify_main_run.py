"""Re-audit the paper's PRISM numbers against a freshly trained main run.

The reproducibility audit found that every reported number reproduces from the archive EXCEPT those
depending on the main PRISM run, whose predictions were missing. Once that run is regenerated this
script scores it under the canonical label set and prints, for each affected cell, the committed
value beside the newly measured one so the delta is explicit rather than silently adopted.

Canonical labels: WHO thresholds on CORRECTED ages (nine neonates were stored with a stale
age_months=6). The correction is read from a reference run, not hard-coded.

    python scripts/verify_main_run.py --prism-run pedrr_kfold_main332
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUNS = Path("outputs/paper_runs")
AGE_REF = "pedrr_kfold_flowonly_agefix"

# Committed values that depend on the main run, with where they appear.
COMMITTED = {
    "tab:abl Full PRISM":        {"mae": 6.19, "sens": 0.65, "auroc": 0.881, "n": 332},
    "tab:s1 Pooled":             {"mae": 6.19, "sens": 0.65, "auroc": 0.881, "n": 332},
    "tab:s4 PRISM-Breath (full)": {"mae": 6.19, "sens": None, "auroc": 0.881, "n": 332},
    "tab:simple + fusion head":  {"mae": None, "sens": None, "auroc": 0.881, "n": 332},
    "tab:framing Full PRISM":    {"mae": None, "sens": None, "auroc": 0.881, "n": 332},
}


def who(age: float) -> float:
    return 60.0 if age < 2 else (50.0 if age < 12 else 40.0)


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
    d["y"] = (d["gt_rr_bpm"] >= d["thr"]).astype(int)
    return d


def metrics(d: pd.DataFrame) -> Dict[str, float]:
    dec = (d["pred_rr_bpm"] >= d["thr"]).astype(int)
    tp = int(((dec == 1) & (d.y == 1)).sum()); fn = int(((dec == 0) & (d.y == 1)).sum())
    return {
        "n": len(d), "pos": int(d.y.sum()),
        "mae": float((d["pred_rr_bpm"] - d["gt_rr_bpm"]).abs().mean()),
        "sens": tp / (tp + fn) if (tp + fn) else float("nan"),
        "auroc": float(roc_auc_score(d.y, d["tachy_prob"])) if d.y.nunique() > 1 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prism-run", default="pedrr_kfold_main332")
    ap.add_argument("--tolerance-auroc", type=float, default=0.005)
    ap.add_argument("--tolerance-mae", type=float, default=0.05)
    a = ap.parse_args()

    ref = load(AGE_REF)
    if ref is None:
        raise SystemExit(f"age reference run missing: {AGE_REF}")
    age_map = dict(zip(ref["subject_id"].astype(str), ref["age_months"].astype(float)))

    d = scored(a.prism_run, age_map)
    if d is None:
        raise SystemExit(f"run not found or has no predictions: {a.prism_run}")
    m = metrics(d)
    print(f"Run            : {a.prism_run}")
    print(f"Subjects       : {m['n']}  ({m['pos']} tachypneic, corrected labels)")
    print(f"Per-fold folds : {sorted(d.fold.unique())}\n")
    print(f"Measured: MAE {m['mae']:.2f}   Sens {m['sens']:.2f}   AUROC {m['auroc']:.3f}\n")

    print("Committed value vs newly measured (cells that depended on the missing run)")
    print(f"  {'cell':<28} {'committed':<30} {'measured':<30} verdict")
    print(f"  {'-'*28} {'-'*30} {'-'*30} -------")
    for cell, c in COMMITTED.items():
        parts_c, parts_m, ok = [], [], True
        for k, tol in (("mae", a.tolerance_mae), ("sens", 0.02), ("auroc", a.tolerance_auroc)):
            if c.get(k) is None:
                continue
            cv, mv = c[k], m[k]
            fmt = "{:.3f}" if k == "auroc" else "{:.2f}"
            parts_c.append(f"{k} {fmt.format(cv)}")
            parts_m.append(f"{k} {fmt.format(mv)}")
            if not np.isfinite(mv) or abs(mv - cv) > tol:
                ok = False
        print(f"  {cell:<28} {', '.join(parts_c):<30} {', '.join(parts_m):<30} "
              f"{'matches' if ok else 'CHANGED'}")

    # Matched-subset comparison against the published extractors.
    eff = scored("pedrr_kfold_efficientphys", age_map)
    if eff is not None:
        matched = set(eff.sid)
        sub = d[d.sid.isin(matched)]
        if len(sub) and sub.y.nunique() > 1:
            print(f"\nMatched-subset check (n={len(sub)}, the head-to-head cohort)")
            print(f"  PRISM AUROC {roc_auc_score(sub.y, sub.tachy_prob):.3f}   "
                  f"MAE {float((sub.pred_rr_bpm - sub.gt_rr_bpm).abs().mean()):.2f}")
            print("  committed tab:sota PRISM row: AUROC 0.874, MAE 6.40")

    print("\nIf any row says CHANGED, the paper's headline moves and the affected tables and prose "
          "must be updated together -- do not adopt a new number in one place only.")
    print(f"Regenerate the supplement with:\n  python scripts/regenerate_supplementary_tables.py "
          f"--prism-run {a.prism_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
