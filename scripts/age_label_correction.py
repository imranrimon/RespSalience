"""Age-ID reconciliation: the signal .npz cache mislabeled 9 true age-0 newborns as age_months=6
(old falsy-0 -> infant-prior default), which mis-banded them (<2mo -> 2-12mo) and applied the wrong
WHO threshold (>=50 instead of >=60). The manifest is correct; the code fix
(respiratory_benchmark_dataset.py) now prefers the manifest age. This script recomputes the CORRECTED
clinical numbers on the CACHED predictions by joining the authoritative age from the Stata metadata,
so the paper's numbers can be corrected without a retrain. CPU-only.

Caveat: the model's pred_rr/tachy_prob for those 9 subjects were computed with age=6 in the age
adapter + internal threshold head; a clean re-extract+retrain (staged, GPU) is the fully-clean version.
The 9/332 effect on ranking scores is expected to be small; the decision-level correction here is exact.
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

DTA = "data/Video Dataset for Respiratory Rate Counting in Children/Stata File/Data Respiratory Rate Counting from Video Recording.dta"
RUN = sys.argv[1] if len(sys.argv) > 1 else "outputs/paper_runs/pedrr_kfold_learned"

def who(a): return 60 if a < 2 else (50 if a < 12 else 40)
def band(a): return "<2mo" if a < 2 else ("2-12mo" if a < 12 else "12-60mo")

dta = pd.read_stata(DTA, convert_categoricals=False)
dta["filename"] = dta["filename"].astype(str)
age_map = dict(zip(dta["filename"], dta["age"].astype(int)))

rows = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(RUN, "fold*/test_eval/prism_breath_predictions.csv")))]
d = pd.concat(rows, ignore_index=True)
d["video_id"] = d["video_id"].astype(str)
d["age_true"] = d["video_id"].map(age_map)
assert d["age_true"].notna().all(), "unmatched videos"
d["thr_true"] = d["age_true"].apply(who).astype(float)
d["tachy_true"] = (d["gt_rr_bpm"] >= d["thr_true"]).astype(int)
d["band_old"] = d["age_months"].apply(band)
d["band_true"] = d["age_true"].apply(band)

y_old, y_new = d["gt_tachy"].astype(int).values, d["tachy_true"].values
p = d["tachy_prob"].values
flips = int((y_old != y_new).sum())
print(f"run: {RUN}")
print(f"n={len(d)}  |  label flips (old->corrected): {flips}  |  mis-banded subjects: {int((d.band_old!=d.band_true).sum())}")
print(f"prevalence: old {y_old.mean():.4f} ({y_old.sum()})  ->  corrected {y_new.mean():.4f} ({y_new.sum()})")

def sens_spec(y, pred_pos):
    tp=((pred_pos==1)&(y==1)).sum(); fn=((pred_pos==0)&(y==1)).sum()
    tn=((pred_pos==0)&(y==0)).sum(); fp=((pred_pos==1)&(y==0)).sum()
    return tp/(tp+fn) if tp+fn else np.nan, tn/(tn+fp) if tn+fp else np.nan

dec_old = (d["pred_rr_bpm"].values >= d["threshold_bpm"].values).astype(int)
dec_new = (d["pred_rr_bpm"].values >= d["thr_true"].values).astype(int)
print("\n--- overall (WHO decision on pred_rr; AUROC on tachy_prob) ---")
print(f"{'':10}{'AUROC':>9}{'Sens':>8}{'Spec':>8}")
se,sp=sens_spec(y_old,dec_old); print(f"{'old':10}{roc_auc_score(y_old,p):>9.4f}{se:>8.3f}{sp:>8.3f}")
se,sp=sens_spec(y_new,dec_new); print(f"{'corrected':10}{roc_auc_score(y_new,p):>9.4f}{se:>8.3f}{sp:>8.3f}")

print("\n--- <2mo band (the reviewer's small-n concern) ---")
for tag, bcol, ycol, acol in [("old","band_old",y_old,"age_months"),("corrected","band_true",y_new,"age_true")]:
    mask = (d[bcol]=="<2mo").values
    yy = (y_old if tag=="old" else y_new)[mask]; pp=p[mask]
    au = roc_auc_score(yy,pp) if yy.min()!=yy.max() else float('nan')
    print(f"  {tag:10} n={mask.sum():>3} pos={int(yy.sum()):>3}  AUROC={au:.3f}")
