"""Recompute the failure-mode and equity numbers in manuscript/appendix_evals.tex from one artifact.

The audit found two defects in that appendix, independent of each other:
  1. its header credited `pedrr_kfold_learned` (297 subjects) for a 332-subject analysis, and
  2. it used the stale 86-positive label set, before nine neonates' ages were corrected (now 84).

Both are fixed by recomputing every cell from a single run under a single, corrected label set.
Ages come from the reference run that carries the correction, exactly as `verify_main_run.py` does.

    python scripts/regenerate_appendix_evals.py --prism-run pedrr_kfold_main332
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_main_run import load, scored  # noqa: E402

SITES = {1: "ICMH", 2: "Upazila", 3: "Community"}
DTA = Path("data/Video Dataset for Respiratory Rate Counting in Children/Stata File/"
           "Data Respiratory Rate Counting from Video Recording.dta")


def band(a: float) -> str:
    return "<2mo" if a < 2 else ("2-12mo" if a < 12 else "12-60mo")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prism-run", default="pedrr_kfold_main332")
    a = ap.parse_args()

    ref = load("pedrr_kfold_flowonly_agefix")
    if ref is None:
        raise SystemExit("age reference run missing")
    age_map = dict(zip(ref["subject_id"].astype(str), ref["age_months"].astype(float)))
    d = scored(a.prism_run, age_map)
    if d is None:
        raise SystemExit(f"run not found: {a.prism_run}")
    d["band"] = d["age"].map(band)
    d["dec"] = (d["pred_rr_bpm"] >= d["thr"]).astype(int)
    # Derive the proxies before slicing, so every case-class view carries them.
    d["absmargin"] = (d["pred_rr_bpm"] - d["thr"]).abs()
    d["minqual"] = d[["depth_quality", "flow_quality", "audio_quality"]].min(axis=1)

    tp = d[(d.dec == 1) & (d.y == 1)]; fn = d[(d.dec == 0) & (d.y == 1)]
    fp = d[(d.dec == 1) & (d.y == 0)]; tn = d[(d.dec == 0) & (d.y == 0)]
    sens = len(tp) / max(1, len(tp) + len(fn)); spec = len(tn) / max(1, len(tn) + len(fp))
    print(f"Run: {a.prism_run}   n={len(d)}   positives={int(d.y.sum())} (corrected labels)")
    print(f"WHO decision rule: TP={len(tp)} FN={len(fn)} FP={len(fp)} TN={len(tn)}  "
          f"Sens {sens:.2f}  Spec {spec:.2f}  AUROC {roc_auc_score(d.y, d.tachy_prob):.3f}\n")

    print("=== tab:fail_means (mean reliability proxy by case class) ===")
    rows = [("Disagreement", "disagreement_score", 3), ("Min. modality qual.", "minqual", 3),
            ("Apnea prob.", "apnea_prob", 3), ("Motion prob.", "motion_prob", 3),
            ("|r-tau| (br/min)", "absmargin", 2), ("Age (months)", "age", 1)]
    print(f"{'Proxy':<22}{'TP':>9}{'FN':>9}{'FP':>9}{'Overall':>10}")
    for lab, col, dp in rows:
        if col not in d.columns:
            print(f"{lab:<22}{'(absent)':>9}"); continue
        f = f"{{:>9.{dp}f}}"
        print(f"{lab:<22}" + "".join(f.format(g[col].mean()) for g in (tp, fn, fp))
              + f"{d[col].mean():>10.{dp}f}")
    print(f"{'n (subjects)':<22}{len(tp):>9}{len(fn):>9}{len(fp):>9}{len(d):>10}")

    print("\n=== tab:fail_tertile (miss rate among positives by proxy tertile) ===")
    pos = d[d.y == 1].copy()
    for lab, col in (("Motion prob.", "motion_prob"), ("Apnea prob.", "apnea_prob"),
                     ("Disagreement", "disagreement_score"), ("|r-tau|", "absmargin")):
        if col not in pos.columns:
            continue
        try:
            pos["t"] = pd.qcut(pos[col], 3, labels=["low", "mid", "high"], duplicates="drop")
        except ValueError:
            continue
        miss = pos.groupby("t", observed=True).apply(
            lambda g: pd.Series({"n": len(g), "miss_rate": float((g.dec == 0).mean())}))
        print(f"  {lab}: " + "  ".join(f"{i} n={int(r.n)} miss={r.miss_rate:.2f}"
                                       for i, r in miss.iterrows()))

    print("\n=== how well each proxy ranks FN above TP among positives (AUROC; 0.5 = no signal) ===")
    lab_fn = np.r_[np.ones(len(fn), int), np.zeros(len(tp), int)]
    for lab, col in (("Motion prob.", "motion_prob"), ("Apnea prob.", "apnea_prob"),
                     ("Disagreement", "disagreement_score"), ("|r-tau|", "absmargin"),
                     ("Age (months)", "age")):
        if col not in d.columns:
            continue
        s = np.r_[fn[col].to_numpy(float), tp[col].to_numpy(float)]
        if not np.all(np.isfinite(s)):
            continue
        print(f"  {lab:<16} AUROC(FN vs TP) = {roc_auc_score(lab_fn, s):.3f}")

    print("\n=== tab:equity (AUROC by age band and site) ===")
    aur = {}
    for b in ("<2mo", "2-12mo", "12-60mo"):
        g = d[d.band == b]
        aur[b] = (len(g), int(g.y.sum()),
                  roc_auc_score(g.y, g.tachy_prob) if g.y.nunique() > 1 else float("nan"))
        print(f"  {b:<9} n={aur[b][0]:>4} pos={aur[b][1]:>3} AUROC={aur[b][2]:.3f}")
    vals = [v[2] for v in aur.values() if np.isfinite(v[2])]
    print(f"  age equity gap = {max(vals) - min(vals):.3f}")

    # Site comes from the cohort's Stata metadata (`facility_code`), NOT from the subject-id prefix:
    # the ids do not encode facility, and a prefix heuristic produced 137/92/37 against the true
    # 137/92/103.
    site_map = {}
    if DTA.exists():
        meta = pd.read_stata(DTA, convert_categoricals=False)
        key = "child_id" if "child_id" in meta.columns else meta.columns[0]
        for _, r in meta.iterrows():
            raw = str(r[key]).strip()
            for form in {raw, f"ID-{raw}", raw.replace("ID-", "")}:
                site_map[form] = SITES.get(int(r["facility_code"]))
    else:
        print(f"\n[warn] cohort metadata not found at {DTA}; skipping the site table")
    if site_map:
        d["site"] = d["sid"].astype(str).map(site_map)
        unmapped = int(d["site"].isna().sum())
        if unmapped:
            print(f"\n[warn] {unmapped} subject(s) unmatched to a facility; excluded from site rows")
        svals = []
        for s in ("ICMH", "Upazila", "Community"):
            g = d[d.site == s]
            if len(g) == 0 or g.y.nunique() < 2:
                continue
            au = roc_auc_score(g.y, g.tachy_prob); svals.append(au)
            print(f"  {s:<11} n={len(g):>4} pos={int(g.y.sum()):>3} AUROC={au:.3f}")
        if len(svals) > 1:
            print(f"  site equity gap = {max(svals) - min(svals):.3f}")
    print("\nPaste these into manuscript/appendix_evals.tex and set its header provenance to "
          f"outputs/paper_runs/{a.prism_run}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
