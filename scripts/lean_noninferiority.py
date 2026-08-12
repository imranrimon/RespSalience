"""(#14) Lean non-inferiority: PRISM-full vs the lean variant (pedrr_kfold_lean), corrected labels.

Paired tachypnea-AUROC difference (full - lean) on the common clip set, with a subject-grouped
paired bootstrap 95% CI, evaluated against a non-inferiority margin of 0.02 AUROC. Lean is judged
non-inferior to full if the UPPER 95% bound of (full - lean) < margin (i.e. full does not beat lean
by more than the margin). CPU-only, read-only cached predictions. Reports clearly if no lean run.

Run:  python scripts/lean_noninferiority.py
"""
import os, glob, re
import numpy as np
from sklearn.metrics import roc_auc_score
import _corrected_common as C

FULL = "outputs/paper_runs/pedrr_kfold_learned"
MARGIN = 0.02
B = 5000
SEED = 0


def find_lean():
    # 'lean' as a standalone token in the basename (excludes 'clean' -> preceded by a letter)
    pat = re.compile(r"(?<![a-z])lean(?![a-z])")
    cands = [p for p in glob.glob("outputs/paper_runs/*lean*")
             if os.path.isdir(p) and pat.search(os.path.basename(p))
             and glob.glob(os.path.join(p, "fold*", "test_eval", "prism_breath_predictions.csv"))]
    return sorted(cands)


if __name__ == "__main__":
    leans = find_lean()
    if not leans:
        print("NO lean-variant run found under outputs/paper_runs/*lean* with fold*/test_eval predictions.")
        print("Non-inferiority CI cannot be computed. (Not fabricating a result.)")
        raise SystemExit(0)
    lean_root = leans[0]
    print(f"lean run = {lean_root}" + (f"  (also found: {leans[1:]})" if len(leans) > 1 else ""))

    full = C.load_corrected(FULL)
    lean = C.load_corrected(lean_root)
    common = sorted(set(full.clip_key) & set(lean.clip_key))
    f = full[full.clip_key.isin(common)].sort_values("clip_key").reset_index(drop=True)
    l = lean[lean.clip_key.isin(common)].sort_values("clip_key").reset_index(drop=True)
    assert (f.gt_tachy.values == l.gt_tachy.values).all(), "corrected labels differ across runs"

    y = f.gt_tachy.values.astype(int)
    pf, pl = f.tachy_prob.values, l.tachy_prob.values
    au_f, au_l = roc_auc_score(y, pf), roc_auc_score(y, pl)
    delta = au_f - au_l
    print(f"\ncommon n={len(f)}  prevalence={y.mean():.4f} ({int(y.sum())})")
    print(f"AUROC full = {au_f:.4f}   AUROC lean = {au_l:.4f}   delta(full-lean) = {delta:+.4f}")

    # subject-grouped paired bootstrap (subject==clip here; resample subjects, apply to both models)
    subs = f.subject_id.values
    uniq = np.unique(subs)
    idxby = {s: np.where(subs == s)[0] for s in uniq}
    rng = np.random.RandomState(SEED)
    deltas = []
    for _ in range(B):
        samp = rng.choice(uniq, len(uniq), replace=True)
        ix = np.concatenate([idxby[s] for s in samp])
        yy = y[ix]
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(roc_auc_score(yy, pf[ix]) - roc_auc_score(yy, pl[ix]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"paired subject-bootstrap 95% CI of delta(full-lean): [{lo:+.4f}, {hi:+.4f}]  (B={len(deltas)})")
    print(f"\nnon-inferiority margin = {MARGIN}")
    ni = hi < MARGIN
    print(f"upper 95% bound {hi:+.4f} {'<' if ni else '>='} margin {MARGIN}  =>  "
          f"lean is {'NON-INFERIOR' if ni else 'NOT shown non-inferior'} to full at margin {MARGIN}")
