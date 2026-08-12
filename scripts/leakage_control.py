"""B2 comparison glue: flow-extractor leakage control for EfficientPhys. Compares PRISM on the
LEAKY flow (full-split EfficientPhys, the 0.869 headline) vs the LEAKAGE-FREE flow (per-fold
EfficientPhys from build_efficientphys_flow_cleanfold.py). Only the flow channel differs (depth/audio
identical), so the AUROC gap = the flow-extractor leakage inflation. Mirrors manuscript Table 4's
TSCAN leaky-vs-clean row (delta ~0.014). Subject-grouped bootstrap + DeLong.

Once the clean k-fold exists, this drops the leakage delta straight into the paper. Run:
    python scripts/leakage_control.py
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_metric_comparison import load_run
from compare_clinical_models import delong_test

LEAKY = "outputs/paper_runs/pedrr_kfold_learned"          # PRISM w/ full-split (leaky) EfficientPhys flow
CLEAN = "outputs/paper_runs/pedrr_kfold_learned_clean"    # PRISM w/ per-fold leakage-free flow (after B2)

def boot_delta(y, pL, pC, grp, B=3000):
    rng = np.random.RandomState(0); subs = np.array(sorted(set(grp)))
    idxby = {s: np.where(grp == s)[0] for s in subs}; D = []
    for _ in range(B):
        idx = np.concatenate([idxby[s] for s in rng.choice(subs, len(subs), replace=True)]); yy = y[idx]
        if yy.sum() < 3 or (yy == 0).sum() < 3: continue
        D.append(roc_auc_score(yy, pL[idx]) - roc_auc_score(yy, pC[idx]))
    D = np.array(D)
    return float(np.mean(D)), float(np.percentile(D, 2.5)), float(np.percentile(D, 97.5)), float(np.mean(D > 0))

if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--leaky", default=LEAKY, help="leaky PRISM k-fold run root (default: original pedrr_kfold_learned; "
                     "use a matched --tag learned_repro run for a guaranteed-matched control)")
    _ap.add_argument("--clean", default=CLEAN, help="leakage-free PRISM k-fold run root (run_prism_kfold --tag learned_clean)")
    _a = _ap.parse_args()
    LEAKY, CLEAN = _a.leaky, _a.clean
    if not glob.glob(os.path.join(CLEAN, "fold*", "test_eval", "prism_breath_predictions.csv")):
        print(f"[leakage-control] clean run not found: {CLEAN}\nProduce it first (external GPU terminal):")
        print("  1) set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1")
        print("  2) python scripts/build_efficientphys_flow_cleanfold.py     # -> data/pedrr_signals_learned_clean")
        print("  3) run the SAME PRISM k-fold you used for pedrr_kfold_learned, but with")
        print("       --signal-dir data/pedrr_signals_learned_clean  and output dir pedrr_kfold_learned_clean")
        print("       (i.e. train_prism_breath.py per fold on the clean signals, or the journal suite with that signal-dir)")
        print("  Then re-run this script to get the leakage delta.")
        sys.exit()
    L = load_run(LEAKY); C = load_run(CLEAN)
    for d in (L, C):
        if "subject_id" not in d: d["subject_id"] = d["clip_key"].str.rsplit("_", n=1).str[0]
    common = sorted(set(L["clip_key"]) & set(C["clip_key"]))
    L = L[L["clip_key"].isin(common)].sort_values("clip_key").reset_index(drop=True)
    C = C[C["clip_key"].isin(common)].sort_values("clip_key").reset_index(drop=True)
    y = L["gt_tachy"].values.astype(int); pL = L["tachy_prob"].values; pC = C["tachy_prob"].values
    grp = L["subject_id"].values
    aL, aC = roc_auc_score(y, pL), roc_auc_score(y, pC)
    _, _, pdl = delong_test(y, pC, pL)
    md, lo, hi, pw = boot_delta(y, pL, pC, grp)
    print("=== FLOW-EXTRACTOR LEAKAGE CONTROL (EfficientPhys): PRISM leaky vs leakage-free ===")
    print(f"  common clips: {len(common)}")
    print(f"  PRISM AUROC  LEAKY (full-split flow):    {aL:.3f}")
    print(f"  PRISM AUROC  CLEAN (per-fold flow):      {aC:.3f}")
    print(f"  leakage inflation dAUROC = {aL - aC:+.3f}  [subject-boot 95% CI {lo:+.3f},{hi:+.3f}]  "
          f"P(leaky>clean)={pw:.3f}  DeLong p={pdl:.4g}")
    print(f"  => report CLEAN {aC:.3f} as the leakage-free headline, OR disclose LEAKY {aL:.3f} with this "
          f"~{aL - aC:+.3f} control (cf. TSCAN-clean ~-0.014, tab:framing/Table 4).")
