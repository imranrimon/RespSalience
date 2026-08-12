"""CANONICAL recompute (review #11): ONE evaluator -> every clinical number in the paper, with the
age-newborn fix applied UNIFORMLY to all models (join authoritative .dta age on video_id, recompute WHO
threshold + gt_tachy). Prints corrected values + before/after deltas so every table/figure can be updated
from a single source. Reuses full_metric_comparison (load, point_metrics, boot) and compare_clinical_models
(delong). CPU-only, cached predictions.
"""
import os, sys, glob, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import full_metric_comparison as F
from compare_clinical_models import delong_test
from sklearn.metrics import roc_auc_score

RUNS = dict(F.RUNS); RUNS["DeepPhys"] = "outputs/paper_runs/pedrr_kfold_deepphys_100ep"  # matched-budget, per tab:sota
DTA = "data/Video Dataset for Respiratory Rate Counting in Children/Stata File/Data Respiratory Rate Counting from Video Recording.dta"
dta = pd.read_stata(DTA, convert_categoricals=False); dta["filename"] = dta["filename"].astype(str)
AGE = dict(zip(dta["filename"], dta["age"].astype(int)))
def who(a): return 60.0 if a < 2 else (50.0 if a < 12 else 40.0)
def band(a): return "<2mo" if a < 2 else ("2-12mo" if a < 12 else "12-60mo")

def load_corrected(run_root):
    paths = sorted(glob.glob(os.path.join(run_root, "fold*", "test_eval", "prism_breath_predictions.csv")))
    df = pd.concat([pd.read_csv(p).assign(fold=i) for i, p in enumerate(paths)], ignore_index=True)
    df["video_id"] = df["video_id"].astype(str)
    df["clip_key"] = df["subject_id"].astype(str) + "_" + df["video_id"]
    g = df.groupby("clip_key").agg(subject_id=("subject_id", "first"), video_id=("video_id", "first"),
            fold=("fold", "first"), gt_rr=("gt_rr_bpm", "mean"), pred_rr=("pred_rr_bpm", "mean"),
            threshold_old=("threshold_bpm", "first"), gt_tachy_old=("gt_tachy", "first"),
            tachy_prob=("tachy_prob", "mean")).reset_index()
    g["age"] = g["video_id"].map(AGE)
    g["threshold"] = g["age"].apply(who)                    # CORRECTED WHO threshold
    g["gt_tachy"] = (g["gt_rr"] >= g["threshold"]).astype(int)  # CORRECTED label
    g["band"] = g["age"].apply(band)
    return g

data = {m: load_corrected(r) for m, r in RUNS.items()}
common = set.intersection(*[set(d["clip_key"]) for d in data.values()])
data = {m: d[d["clip_key"].isin(common)].sort_values("clip_key").reset_index(drop=True) for m, d in data.items()}
P = data[F.PRISM]
print(f"common matched set n={len(P)}  |  corrected prevalence={P['gt_tachy'].mean():.4f} ({int(P['gt_tachy'].sum())})"
      f"  [was {P['gt_tachy_old'].mean():.4f} ({int(P['gt_tachy_old'].sum())})]")
print(f"label flips on common set: {int((P['gt_tachy']!=P['gt_tachy_old']).sum())}\n")

# ---- (1) headline metric suite, corrected ----
print("=== (1) HEADLINE (matched common set, corrected labels) ===")
print(f"{'model':14}{'MAE':>7}{'AUROC':>8}{'AUROCraw':>9}{'AUPRC':>7}{'Se@Sp90':>8}{'Sp@Se90':>8}{'Sens':>7}{'Spec':>7}")
mets = {}
for m, d in data.items():
    o = F.point_metrics(d); mets[m] = o
    print(f"{m:14}{o['MAE']:>7.2f}{o['AUROC']:>8.3f}{o['AUROC_rawrr']:>9.3f}{o['AUPRC']:>7.3f}"
          f"{o['Se@Sp90']:>8.3f}{o['Sp@Se90']:>8.3f}{o['Sens']:>7.3f}{o['Spec']:>7.3f}")

# ---- (2) DeLong + exact McNemar PRISM vs each (corrected) ----
from scipy.stats import binomtest
print("\n=== (2) DeLong + McNemar vs PRISM (corrected labels/decisions) ===")
y = P["gt_tachy"].values.astype(int); pp = P["tachy_prob"].values
corrP = ((P["pred_rr"].values >= P["threshold"].values).astype(int) == y)
for m in RUNS:
    if m == F.PRISM: continue
    _, aB, pv = delong_test(y, pp, data[m]["tachy_prob"].values)
    corrB = ((data[m]["pred_rr"].values >= data[m]["threshold"].values).astype(int) == y)
    b = int((corrP & ~corrB).sum()); c = int((~corrP & corrB).sum())
    pmc = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    print(f"  {m:14} AUROC(DeLong)={aB:.3f}  DeLong p={pv:.2e}  McNemar p={pmc:.3f}")

# ---- (3) per age-band AUROC, corrected (all models) ----
print("\n=== (3) per-band tachypnea AUROC (corrected) ===")
print(f"{'model':14}{'overall':>9}{'2-12mo':>9}{'12-60mo':>9}{'<2mo':>8}")
for m, d in data.items():
    r = {"overall": roc_auc_score(d.gt_tachy, d.tachy_prob)}
    for b in ["2-12mo", "12-60mo", "<2mo"]:
        s = d[d.band == b]
        r[b] = roc_auc_score(s.gt_tachy, s.tachy_prob) if s.gt_tachy.nunique() == 2 else float("nan")
    print(f"{m:14}{r['overall']:>9.3f}{r['2-12mo']:>9.3f}{r['12-60mo']:>9.3f}{r['<2mo']:>8.3f}")
nb = {b: int((P.band == b).sum()) for b in ["<2mo", "2-12mo", "12-60mo"]}
pos = {b: int(P[P.band == b].gt_tachy.sum()) for b in ["<2mo", "2-12mo", "12-60mo"]}
print(f"  band n (corrected): {nb}  positives: {pos}")

# ---- (4) PRISM full-332 per-fold + per-band (corrected) ----
Pfull = load_corrected(RUNS[F.PRISM])
print(f"\n=== (4) PRISM per-fold (full {len(Pfull)}, corrected) ===")
print(f"{'fold':6}{'n':>4}{'pos':>4}{'MAE':>7}{'Sens':>7}{'Spec':>7}{'AUROC':>8}")
aurs = []
for f in sorted(Pfull.fold.unique()):
    s = Pfull[Pfull.fold == f]; o = F.point_metrics(s); aurs.append(o["AUROC"])
    print(f"{f:<6}{len(s):>4}{int(s.gt_tachy.sum()):>4}{o['MAE']:>7.2f}{o['Sens']:>7.2f}{o['Spec']:>7.2f}{o['AUROC']:>8.3f}")
of = F.point_metrics(Pfull)
print(f"{'POOLED':6}{len(Pfull):>4}{int(Pfull.gt_tachy.sum()):>4}{of['MAE']:>7.2f}{of['Sens']:>7.2f}{of['Spec']:>7.2f}{of['AUROC']:>8.3f}")
print(f"  per-fold AUROC mean±sd: {np.mean(aurs):.3f}±{np.std(aurs):.3f}  |  full-332 prevalence {Pfull.gt_tachy.mean():.4f} ({int(Pfull.gt_tachy.sum())})")
for b in ["<2mo","2-12mo","12-60mo"]:
    s=Pfull[Pfull.band==b]; au=roc_auc_score(s.gt_tachy,s.tachy_prob) if s.gt_tachy.nunique()==2 else float('nan')
    o=F.point_metrics(s); print(f"  full-332 {b:8} n={len(s):>3} pos={int(s.gt_tachy.sum()):>3} AUROC={au:.3f} Sens={o['Sens']:.3f} MAE={o['MAE']:.2f}")
