"""Unified full-metric comparison harness: PRISM-Breath vs baselines on the matched pedrr
5-fold CV, on the COMMON intersected clip set, with subject-grouped bootstrap 95% CIs and
paired DeLong/McNemar vs PRISM.

Metrics: tachypnea AUROC (on each model's tachy_prob), AUROC_margin (rank by raw pred_rr-thr,
removes the trained-head vs hand-set-logistic confound), AUPRC, Se@Sp=0.90, WHO-decision
Sens/Spec, RR MAE/RMSE. Manufactures the "wins most metrics" table.

Reuses scripts/compare_clinical_models.py (delong_test, mcnemar_exact). Read-only on data.
Run from repo root:  python scripts/full_metric_comparison.py
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_clinical_models import delong_test, mcnemar_exact

RUNS = {  # model -> run_root (complete matched 5-fold runs). DeepPhys=regular; 100ep swaps in when done.
    "PRISM-Breath":  "outputs/paper_runs/pedrr_kfold_learned",
    "EfficientPhys": "outputs/paper_runs/pedrr_kfold_efficientphys",
    "TS-CAN":        "outputs/paper_runs/pedrr_kfold_tscan",
    "VIRENet":       "outputs/paper_runs/pedrr_kfold_virenet",
    "DeepPhys":      "outputs/paper_runs/pedrr_kfold_deepphys",
    "PhysFormer":  "outputs/paper_runs/pedrr_kfold_physformer",  # B3
}
PRISM = "PRISM-Breath"
HIGHER_BETTER = {"AUROC","AUROC_margin","AUROC_rawrr","AUPRC","Se@Sp90","Sp@Se90","Sens","Spec"}
LOWER_BETTER  = {"MAE","RMSE"}

# Authoritative age from the Stata metadata -> corrects the stale-.npz age-0 newborn bug uniformly for
# EVERY caller of load_run (join on video_id==filename; recompute WHO threshold + gt_tachy).
_DTA = "data/Video Dataset for Respiratory Rate Counting in Children/Stata File/Data Respiratory Rate Counting from Video Recording.dta"
try:
    _dta = pd.read_stata(_DTA, convert_categoricals=False)
    _AGE = dict(zip(_dta["filename"].astype(str), _dta["age"].astype(int)))
except Exception:
    _AGE = {}
def _who(a): return 60.0 if a < 2 else (50.0 if a < 12 else 40.0)

def load_run(run_root):
    paths = sorted(glob.glob(os.path.join(run_root,"fold*","test_eval","prism_breath_predictions.csv"))) \
         or sorted(glob.glob(os.path.join(run_root,"seed*","test_eval","prism_breath_predictions.csv")))
    if not paths: raise FileNotFoundError(f"no predictions under {run_root}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "gt_rr_bpm"   not in df and "gt_bpm"   in df: df["gt_rr_bpm"]   = df["gt_bpm"]
    if "pred_rr_bpm" not in df and "pred_bpm" in df: df["pred_rr_bpm"] = df["pred_bpm"]
    df["clip_key"] = df["subject_id"].astype(str)+"_"+df["video_id"].astype(str)
    g = df.groupby("clip_key").agg(subject_id=("subject_id","first"), video_id=("video_id","first"),
            gt_rr=("gt_rr_bpm","mean"), pred_rr=("pred_rr_bpm","mean"),
            threshold=("threshold_bpm","first"), gt_tachy=("gt_tachy","first"),
            tachy_prob=("tachy_prob","mean")).reset_index()
    if _AGE:
        age = g["video_id"].astype(str).map(_AGE)
        fixed = age.notna()
        g.loc[fixed, "threshold"] = age[fixed].apply(_who).values
        g.loc[fixed, "gt_tachy"]  = (g.loc[fixed, "gt_rr"] >= g.loc[fixed, "threshold"]).astype(int).values
    return g

def point_metrics(d):
    y=d["gt_tachy"].values.astype(int); prob=d["tachy_prob"].values
    margin=(d["pred_rr"]-d["threshold"]).values; e=(d["pred_rr"]-d["gt_rr"]).values
    o={"MAE":np.abs(e).mean(),"RMSE":np.sqrt((e**2).mean())}
    if len(np.unique(y))==2:
        o["AUROC"]=roc_auc_score(y,prob); o["AUROC_margin"]=roc_auc_score(y,margin)
        o["AUPRC"]=average_precision_score(y,prob)
        fpr,tpr,_=roc_curve(y,prob); m=fpr<=0.10
        o["Se@Sp90"]=float(np.max(tpr[m])) if m.any() else 0.0
        hs=tpr>=0.90  # screening: specificity achievable at >=90% sensitivity
        o["Sp@Se90"]=float(np.max(1.0-fpr[hs])) if hs.any() else 0.0
        o["AUROC_rawrr"]=roc_auc_score(y,d["pred_rr"].values)  # rank by raw RR (reviewer 0.877 control)
    dec=(d["pred_rr"]>=d["threshold"]).values.astype(int)
    tp=((dec==1)&(y==1)).sum(); fn=((dec==0)&(y==1)).sum()
    tn=((dec==0)&(y==0)).sum(); fp=((dec==1)&(y==0)).sum()
    o["Sens"]=tp/(tp+fn) if (tp+fn) else np.nan
    o["Spec"]=tn/(tn+fp) if (tn+fp) else np.nan
    return o

def subj_boot(d, B=2000, seed=0):
    rng=np.random.RandomState(seed); subs=d["subject_id"].unique()
    idxby={s:d.index[d["subject_id"]==s].values for s in subs}
    acc={}
    for _ in range(B):
        samp=rng.choice(subs,len(subs),replace=True)
        bd=d.loc[np.concatenate([idxby[s] for s in samp])]
        if bd["gt_tachy"].nunique()<2: continue
        for k,v in point_metrics(bd).items(): acc.setdefault(k,[]).append(v)
    return {k:(np.nanpercentile(v,2.5),np.nanpercentile(v,97.5)) for k,v in acc.items()}

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--deepphys100",action="store_true",help="use matched-compute pedrr_kfold_deepphys_100ep")
    ap.add_argument("--deepphys",default=None,help="override DeepPhys run root")
    ap.add_argument("--prism-run",default=None,
                    help="override the PRISM run root (e.g. outputs/paper_runs/pedrr_kfold_main332). "
                         "The default points at a 297-subject run that predates the regenerated main run.")
    a=ap.parse_args()
    if a.prism_run: RUNS[PRISM]=a.prism_run
    if a.deepphys100: RUNS["DeepPhys"]="outputs/paper_runs/pedrr_kfold_deepphys_100ep"
    elif a.deepphys: RUNS["DeepPhys"]=a.deepphys
    print(f"DeepPhys run = {RUNS['DeepPhys']}",flush=True)
    data={m:load_run(r) for m,r in RUNS.items()}
    common=set.intersection(*[set(d["clip_key"]) for d in data.values()])
    print(f"models={list(data)} | common intersected clips = {len(common)}",flush=True)
    data={m:d[d["clip_key"].isin(common)].sort_values("clip_key").reset_index(drop=True) for m,d in data.items()}
    pts={m:point_metrics(d) for m,d in data.items()}
    cis={m:subj_boot(d) for m,d in data.items()}
    cols=["AUROC","AUROC_margin","AUROC_rawrr","AUPRC","Se@Sp90","Sp@Se90","Sens","Spec","MAE","RMSE"]
    print("\n=== FULL METRIC SUITE (common set, subject-bootstrap 95% CI) ===",flush=True)
    hdr="{:<14}".format("model")+"".join("{:>22}".format(c) for c in cols); print(hdr,flush=True)
    for m in RUNS:
        row="{:<14}".format(m)
        for c in cols:
            v=pts[m].get(c,np.nan); lo,hi=cis[m].get(c,(np.nan,np.nan))
            row+="{:>22}".format(f"{v:.3f}[{lo:.3f},{hi:.3f}]")
        print(row,flush=True)
    # wins (PRISM best point-estimate among all models)
    print("\n=== PRISM 'wins' (best point estimate among all models) ===",flush=True)
    wins=0
    for c in cols:
        vals={m:pts[m].get(c,np.nan) for m in RUNS}
        best=min(vals,key=lambda k:vals[k]) if c in LOWER_BETTER else max(vals,key=lambda k:vals[k])
        w = best==PRISM; wins+= w
        print(f"  {c:14s} best={best:13s} PRISM={vals[PRISM]:.3f}  {'<-- PRISM' if w else ''}",flush=True)
    print(f"  => PRISM best on {wins}/{len(cols)} metrics",flush=True)
    # paired vs PRISM
    print("\n=== PAIRED vs PRISM (DeLong AUROC, McNemar decision) ===",flush=True)
    yp=data[PRISM]["gt_tachy"].values.astype(int); pp=data[PRISM]["tachy_prob"].values
    decp=(data[PRISM]["pred_rr"]>=data[PRISM]["threshold"]).values.astype(int)
    for m in RUNS:
        if m==PRISM: continue
        d=data[m]; pb=d["tachy_prob"].values
        aP,aB,pdl=delong_test(yp,pp,pb)
        decb=(d["pred_rr"]>=d["threshold"]).values.astype(int)
        # mcnemar_exact takes the DECISIONS and computes correctness itself; passing (dec==y)
        # correctness indicators here double-applied ==y (a bug that inflated the PhysFormer result).
        b,c,pmc=mcnemar_exact(yp,decp,decb)
        print(f"  PRISM {aP:.3f} vs {m:13s} {aB:.3f}  DeLong p={pdl:.4g} | McNemar p={pmc:.4g}",flush=True)
