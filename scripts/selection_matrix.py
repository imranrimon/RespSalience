"""The selection-rule x model-family matrix: the fair form of the head-to-head comparison.

The paper's checkpoint-selection experiment showed that selecting PRISM's weights on validation
regression error costs tachypnea AUROC. Comparing a decision-selected PRISM against the published
extractors would be unfair unless they are given the same treatment, so `reselect_baseline_checkpoints.py`
re-selects each extractor's epoch on validation AUROC from its own saved per-epoch checkpoints.

An audit along the way found that the published extractor runs were not checkpoint-selected at all:
their drivers extract predictions from whatever weights are resident when `train()` returns, i.e. the
LAST epoch. So the matrix has three columns, not two, and the middle one is what the paper's original
table actually used.

    python scripts/selection_matrix.py
    python scripts/selection_matrix.py --latex     # emits the supplementary table body
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_main_run import load, scored  # noqa: E402
from compare_clinical_models import delong_test  # noqa: E402

RUNS = Path("outputs/paper_runs")

# (display name, published/last-epoch run, val-AUROC-re-selected run)
FAMILIES = [
    ("EfficientPhys", "pedrr_kfold_efficientphys", "pedrr_kfold_efficientphys_aurocsel"),
    ("TS-CAN", "pedrr_kfold_tscan", "pedrr_kfold_tscan_aurocsel"),
    ("VIRENet", "pedrr_kfold_virenet", "pedrr_kfold_virenet_aurocsel"),
    ("DeepPhys", "pedrr_kfold_deepphys_100ep", "pedrr_kfold_deepphys_100ep_aurocsel"),
]
PRISM_MAE = "pedrr_kfold_main332"
PRISM_AUROC = "pedrr_kfold_main332auroc"


def prep(tag, age_map):
    d = scored(tag, age_map)
    if d is None:
        return None
    d["ae"] = (d["pred_rr_bpm"] - d["gt_rr_bpm"]).abs()
    return d


def folds_done(tag) -> int:
    return len(list((RUNS / tag).glob("fold*/selection_report.json"))) if (RUNS / tag).is_dir() else 0


def selected_epochs(tag):
    out = {}
    for f in sorted((RUNS / tag).glob("fold*/selection_report.json")):
        r = json.loads(f.read_text())
        out[r["fold"]] = (r["selected_epoch"], r.get("mae_selected_epoch"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    ref = load("pedrr_kfold_flowonly_agefix")
    if ref is None:
        raise SystemExit("age reference run missing")
    age_map = dict(zip(ref["subject_id"].astype(str), ref["age_months"].astype(float)))

    rows = []
    pm, pa = prep(PRISM_MAE, age_map), prep(PRISM_AUROC, age_map)

    for name, pub_tag, sel_tag in FAMILIES:
        pub = prep(pub_tag, age_map)
        n_done = folds_done(sel_tag)
        sel = prep(sel_tag, age_map) if n_done == 5 else None
        rec = {"name": name, "n_folds_reselected": n_done}
        if pub is not None:
            rec["published_auroc"] = roc_auc_score(pub.y, pub.tachy_prob)
            rec["published_mae"] = float(pub.ae.mean())
            rec["n"] = len(pub)
        if sel is not None:
            common = sorted(set(pub.sid) & set(sel.sid))
            p = pub[pub.sid.isin(common)].sort_values("sid").reset_index(drop=True)
            s = sel[sel.sid.isin(common)].sort_values("sid").reset_index(drop=True)
            y = p.y.to_numpy(int)
            a_s, a_p, pv = delong_test(y, s.tachy_prob.to_numpy(float), p.tachy_prob.to_numpy(float))
            rec.update({"reselected_auroc": a_s, "reselected_mae": float(sel.ae.mean()),
                        "delta": a_s - a_p, "p": pv,
                        "epochs": {k: v[0] for k, v in selected_epochs(sel_tag).items()}})
        rows.append(rec)

    print("SELECTION RULE x MODEL FAMILY")
    print("(published extractor runs are LAST-EPOCH: their drivers never load a selected checkpoint)\n")
    print(f"{'model':<16}{'n':>5}{'last-epoch':>12}{'val-AUROC sel':>15}{'delta':>9}{'DeLong p':>10}"
          f"{'MAE last':>10}{'MAE sel':>9}   status")
    for r in rows:
        base = f"{r['name']:<16}{r.get('n',0):>5}{r.get('published_auroc',float('nan')):>12.3f}"
        if "reselected_auroc" in r:
            print(base + f"{r['reselected_auroc']:>15.3f}{r['delta']:>+9.3f}{r['p']:>10.4f}"
                         f"{r['published_mae']:>10.2f}{r['reselected_mae']:>9.2f}   complete")
        else:
            print(base + f"{'--':>15}{'--':>9}{'--':>10}"
                         f"{r.get('published_mae',float('nan')):>10.2f}{'--':>9}"
                         f"   re-selection {r['n_folds_reselected']}/5 folds")

    if pm is not None and pa is not None:
        am_ = roc_auc_score(pm.y, pm.tachy_prob); aa_ = roc_auc_score(pa.y, pa.tachy_prob)
        print(f"\n{'PRISM-Breath':<16}{len(pm):>5}{am_:>12.3f}{aa_:>15.3f}"
              f"{aa_-am_:>+9.3f}{'':>10}{pm.ae.mean():>10.2f}{pa.ae.mean():>9.2f}"
              "   (val-MAE selected, not last-epoch)")

    incomplete = [r["name"] for r in rows if "reselected_auroc" not in r]
    if incomplete:
        print(f"\nNOT YET COMPLETE: {', '.join(incomplete)} -- do not report the matrix until all five "
              "families have all five folds. Partial columns are not a result.")
    elif a.latex:
        print("\n% --- supplementary table body ---")
        for r in rows:
            print(f"{r['name']} & {r['published_auroc']:.3f} & {r['reselected_auroc']:.3f} & "
                  f"${r['delta']:+.3f}$ & ${r['p']:.3f}$ \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
