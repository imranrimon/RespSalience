"""Finish the #41 propagation: recompute EVERY remaining paper number on corrected labels via the
age-corrected full_metric_comparison.load_run. Covers ablation variants (tab:abl/simple/framing/s4),
salience recovery (neonatal), per-site (tab:sitesubgroup), per-fold all-models (tab:s2/s3), and the
3-seed operating points (tab:opchar). CPU-only, cached predictions."""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import full_metric_comparison as F   # load_run is now age-corrected
from sklearn.metrics import roc_auc_score

R = "outputs/paper_runs"
def au(y, s): return roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan")
def band(a): return "<2mo" if a < 2 else ("2-12mo" if a < 12 else "12-60mo")
AGE = F._AGE

def pm(run):
    """MAE, Sens, Spec, AUROC on full-332 (corrected)."""
    try: d = F.load_run(f"{R}/{run}")
    except Exception as e: return None
    y = d["gt_tachy"].values.astype(int); dec = (d["pred_rr"] >= d["threshold"]).values.astype(int)
    tp=((dec==1)&(y==1)).sum(); fn=((dec==0)&(y==1)).sum(); tn=((dec==0)&(y==0)).sum(); fp=((dec==1)&(y==0)).sum()
    e=(d["pred_rr"]-d["gt_rr"]).values
    return dict(n=len(d), MAE=np.abs(e).mean(), Sens=tp/(tp+fn) if tp+fn else np.nan,
                Spec=tn/(tn+fp) if tn+fp else np.nan, AUROC=au(y, d["tachy_prob"].values))

print("=== ABLATION / VARIANT runs (full-332, corrected) ===")
VAR = {"Full PRISM":"pedrr_kfold_learned","Lean":"pedrr_kfold_lean","Flow-only":"pedrr_kfold_flowonly",
 "no scan":"pedrr_kfold_abl_no_scan","no spectral":"pedrr_kfold_abl_no_spectral",
 "no codebook":"pedrr_kfold_abl_no_codebook","no prototypes":"pedrr_kfold_abl_no_prototypes",
 "het only":"pedrr_kfold_abl_het_only","shared trunk":"pedrr_kfold_abl_shared_trunk",
 "synth only":"pedrr_kfold_abl_synth_only","moddrop":"pedrr_kfold_abl_moddrop01",
 "freq-forward":"pedrr_kfold_freqfwd","het+synth":"pedrr_kfold_hetsynth",
 "airflow-compact":"pedrr_kfold_airflow_compact","mamba":"pedrr_kfold_mamba",
 "cliffphys":"pedrr_kfold_cliffphys","no pretrain":"pedrr_kfold_nopretrain","v3 flow":"pedrr_kfold_v3",
 "snr-fusion":"pedrr_kfold_snrfusion"}
print(f"{'variant':18}{'n':>4}{'MAE':>7}{'Sens':>7}{'Spec':>7}{'AUROC':>8}")
for name, run in VAR.items():
    m = pm(run)
    if m: print(f"{name:18}{m['n']:>4}{m['MAE']:>7.2f}{m['Sens']:>7.2f}{m['Spec']:>7.2f}{m['AUROC']:>8.3f}")
    else: print(f"{name:18}  (missing)")

print("\n=== SALIENCE neonatal recovery (<2mo AUROC, corrected) ===")
def band_auroc(run, b="<2mo"):
    try: d = F.load_run(f"{R}/{run}")
    except Exception: return None
    d["age"] = d["video_id"].astype(str).map(AGE); s = d[d["age"].apply(lambda a: band(a)==b if a==a else False)]
    return au(s["gt_tachy"].values.astype(int), s["tachy_prob"].values) if len(s) else None
for grp, runs in [("salience", ["pedrr_kfold_salience_seed0","pedrr_kfold_salience_seed1","pedrr_kfold_salience_seed2"]),
                  ("salience+SSL", ["pedrr_kfold_salience_ssl_seed0","pedrr_kfold_salience_ssl_seed1","pedrr_kfold_salience_ssl_seed2"])]:
    vals=[band_auroc(r) for r in runs]; vals=[v for v in vals if v is not None]
    ov=[au(F.load_run(f"{R}/{r}")["gt_tachy"].astype(int), F.load_run(f"{R}/{r}")["tachy_prob"]) for r in runs]
    print(f"  {grp:14} <2mo AUROC seeds={[f'{v:.3f}' for v in vals]}  mean={np.mean(vals):.3f}  | overall mean={np.mean(ov):.3f}")

print("\n=== PER-SITE AUROC (6 models, corrected; facility 1=ICMH 2=Upazila 3=Community) ===")
FAC = dict(zip(F._dta["filename"].astype(str), F._dta["facility_code"].astype(int)))
RUNS = dict(F.RUNS); RUNS["DeepPhys"]="outputs/paper_runs/pedrr_kfold_deepphys_100ep"
data = {m: F.load_run(r if r.startswith("outputs") else f"{R}/{r}") for m,r in
        {k:(v if v.startswith('outputs') else v) for k,v in RUNS.items()}.items()}
data = {m: F.load_run(r) for m,r in RUNS.items()}
common = set.intersection(*[set(d["clip_key"]) for d in data.values()])
data = {m: d[d["clip_key"].isin(common)].sort_values("clip_key").reset_index(drop=True) for m,d in data.items()}
print(f"{'model':14}{'ICMH':>8}{'Upazila':>9}{'Communty':>9}")
for m,d in data.items():
    d=d.copy(); d["fac"]=d["video_id"].astype(str).map(FAC)
    r=[au(d[d.fac==f]["gt_tachy"].astype(int), d[d.fac==f]["tachy_prob"]) for f in (1,2,3)]
    print(f"{m:14}{r[0]:>8.3f}{r[1]:>9.3f}{r[2]:>9.3f}")
nsite={f:int((data['PRISM-Breath']['video_id'].astype(str).map(FAC)==f).sum()) for f in (1,2,3)}
print(f"  site n (298 common): {nsite}")

print("\n=== 3-SEED operating points (learned s42/s43/s44 avg, corrected) ===")
seeds=["pedrr_kfold_learned","pedrr_kfold_learned_s43","pedrr_kfold_learned_s44"]
def cohort_metrics(runs, sel):
    accs=[]
    for run in runs:
        d=F.load_run(f"{R}/{run}"); d["age"]=d["video_id"].astype(str).map(AGE)
        s=d[d["age"].apply(sel)]; y=s["gt_tachy"].astype(int)
        dec=(s["pred_rr"]>=s["threshold"]).astype(int)
        accs.append((np.abs(s["pred_rr"]-s["gt_rr"]).mean(),
                     ((dec==1)&(y==1)).sum()/max(((y==1).sum()),1), au(y,s["tachy_prob"])))
    a=np.array(accs); return a.mean(0)
for nm,sel in [("Full cohort", lambda a:True),(">=2mo", lambda a:a>=2),("2-12mo",lambda a:2<=a<12),("12-60mo",lambda a:a>=12)]:
    try:
        mae,se,ac=cohort_metrics(seeds,sel); print(f"  {nm:12} MAE={mae:.2f} Sens={se:.2f} AUROC={ac:.3f}")
    except Exception as e: print(f"  {nm}: {e}")
