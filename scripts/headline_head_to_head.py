"""Head-to-head of the regenerated PRISM run against the published extractors, on matched cohorts.

The paper's tab:sota compares PRISM to five published extractors on the intersection cohort. That row
depended on the main PRISM run, which was missing; it has now been regenerated under two checkpoint
selection rules. This script re-runs the comparison for BOTH arms so the choice of headline is made on
evidence rather than on which number is larger.

Every comparison is restricted to the subject intersection of the two runs being compared, uses the
corrected age labels, and reports paired DeLong plus a subject-level bootstrap CI on the difference.

    python scripts/headline_head_to_head.py
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
from compare_clinical_models import delong_test  # noqa: E402

BASELINES = [
    ("EfficientPhys", "pedrr_kfold_efficientphys"),
    ("TS-CAN", "pedrr_kfold_tscan"),
    ("VIRENet", "pedrr_kfold_virenet"),
    ("DeepPhys", "pedrr_kfold_deepphys_100ep"),
    ("PhysFormer", "pedrr_kfold_physformer"),
]
ARMS = [
    ("PRISM (val-MAE selection)", "pedrr_kfold_main332"),
    ("PRISM (val-AUROC selection)", "pedrr_kfold_main332auroc"),
]


def boot_delta(y, sa, sb, n=4000, seed=0):
    """Subject-level bootstrap CI for AUROC(a) - AUROC(b)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, int); sa = np.asarray(sa, float); sb = np.asarray(sb, float)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args()

    ref = load("pedrr_kfold_flowonly_agefix")
    if ref is None:
        raise SystemExit("age reference run missing")
    age_map = dict(zip(ref["subject_id"].astype(str), ref["age_months"].astype(float)))

    def prep(tag):
        d = scored(tag, age_map)
        if d is not None:
            d["ae"] = (d["pred_rr_bpm"] - d["gt_rr_bpm"]).abs()
        return d

    arms = {name: prep(tag) for name, tag in ARMS}
    bases = {}
    for name, tag in BASELINES:
        d = prep(tag)
        if d is None:
            print(f"[warn] baseline missing: {name} ({tag})")
            continue
        bases[name] = d

    for arm_name, P in arms.items():
        if P is None:
            continue
        print("=" * 92)
        print(arm_name)
        print("=" * 92)
        print(f"{'baseline':<16}{'n':>5}{'base AUROC':>12}{'PRISM AUROC':>13}{'delta':>9}"
              f"{'95% CI':>20}{'DeLong p':>11}{'base MAE':>10}{'PRISM MAE':>11}")
        for bname, B in bases.items():
            common = set(P.sid) & set(B.sid)
            # PhysFormer stores a non-finite tachy_prob for 2 subjects; a DeLong comparison cannot
            # rank those, so drop them pairwise and report the cohort size actually used.
            for d in (P, B):
                common &= set(d.loc[np.isfinite(d.tachy_prob.to_numpy(float)), "sid"])
            dropped = len(set(P.sid) & set(B.sid)) - len(common)
            common = sorted(common)
            p = P[P.sid.isin(common)].sort_values("sid").reset_index(drop=True)
            b = B[B.sid.isin(common)].sort_values("sid").reset_index(drop=True)
            if not (p.y.to_numpy() == b.y.to_numpy()).all():
                print(f"{bname:<16} label mismatch -- skipped")
                continue
            if dropped:
                print(f"{'':<16}  [{bname}: dropped {dropped} subject(s) with non-finite score]")
            y = p.y.to_numpy(int)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            ap_, ab_, pv = delong_test(y, p.tachy_prob.to_numpy(float), b.tachy_prob.to_numpy(float))
            lo, hi = boot_delta(y, p.tachy_prob.to_numpy(float), b.tachy_prob.to_numpy(float), a.boot)
            print(f"{bname:<16}{len(y):>5}{ab_:>12.3f}{ap_:>13.3f}{ap_ - ab_:>+9.3f}"
                  f"{f'[{lo:+.3f},{hi:+.3f}]':>20}{pv:>11.4f}"
                  f"{b.ae.mean():>10.2f}{p.ae.mean():>11.2f}")
        print()

    # Direct arm-vs-arm on the intersection cohort used by the baseline table.
    A, Bm = arms.get("PRISM (val-MAE selection)"), arms.get("PRISM (val-AUROC selection)")
    if A is not None and Bm is not None and bases:
        matched = set.intersection(*[set(d.sid) for d in bases.values()])
        for label, sel in (("full 332 cohort", set(A.sid)), ("matched head-to-head cohort", matched)):
            p1 = A[A.sid.isin(sel)].sort_values("sid").reset_index(drop=True)
            p2 = Bm[Bm.sid.isin(sel)].sort_values("sid").reset_index(drop=True)
            y = p1.y.to_numpy(int)
            a1, a2, pv = delong_test(y, p2.tachy_prob.to_numpy(float), p1.tachy_prob.to_numpy(float))
            print(f"{label:<30} n={len(y):<5} MAE-sel {a2:.3f} -> AUROC-sel {a1:.3f} "
                  f"(delta {a1 - a2:+.3f}, DeLong p={pv:.3f}); "
                  f"MAE {p1.ae.mean():.2f} -> {p2.ae.mean():.2f}")

    print("\nFairness note: the baselines' own checkpoints were selected by validation regression error.")
    print("Using the val-AUROC arm as the headline compares a decision-selected PRISM against")
    print("MAE-selected baselines. Report the MAE-selected arm as the matched headline, and the")
    print("val-AUROC arm as the selection-criterion experiment, unless baselines are re-selected too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
