"""Tier-2c evaluation: does training PRISM on the differentiable net-benefit (decision-curve) loss +
calibrated abstention (pedrr_kfold_decision) improve the CLINICAL DECISION vs standard PRISM
(pedrr_kfold_learned)? Compares tachypnea AUROC, WHO-decision Sens/Spec, and net benefit across the
screening operating range, on the common clip set. CPU-only."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_metric_comparison import load_run
from sklearn.metrics import roc_auc_score

std = load_run("outputs/paper_runs/pedrr_kfold_learned")
dec = load_run("outputs/paper_runs/pedrr_kfold_decision")
common = sorted(set(std["clip_key"]) & set(dec["clip_key"]))
std = std[std["clip_key"].isin(common)].sort_values("clip_key").reset_index(drop=True)
dec = dec[dec["clip_key"].isin(common)].sort_values("clip_key").reset_index(drop=True)
y = std["gt_tachy"].values.astype(int)
prev = y.mean()
print(f"common n={len(y)}  prevalence={prev:.3f}\n")

def wh_sens_spec(d):
    dec_ = (d["pred_rr"].values >= d["threshold"].values).astype(int)
    tp=((dec_==1)&(y==1)).sum(); fn=((dec_==0)&(y==1)).sum()
    tn=((dec_==0)&(y==0)).sum(); fp=((dec_==1)&(y==0)).sum()
    return tp/(tp+fn), tn/(tn+fp)

def nb(prob, pts):
    N=len(y); out={}
    for pt in pts:
        dd=(prob>=pt).astype(int)
        tp=((dd==1)&(y==1)).sum(); fp=((dd==1)&(y==0)).sum()
        out[pt]=(tp-(pt/(1-pt))*fp)/N
    return out

pts=[0.05,0.10,0.15,0.20,0.25,0.30]
print(f"{'model':16}{'AUROC':>8}{'Sens':>7}{'Spec':>7}   net benefit @ p_t")
for name,d in [("standard PRISM",std),("decision PRISM",dec)]:
    p=d["tachy_prob"].values
    au=roc_auc_score(y,p); se,sp=wh_sens_spec(d); nbd=nb(p,pts)
    print(f"{name:16}{au:>8.3f}{se:>7.3f}{sp:>7.3f}   "+" ".join(f"{pt:.2f}:{nbd[pt]:+.3f}" for pt in pts))
ta={pt:(prev-(pt/(1-pt))*(1-prev)) for pt in pts}
print(f"{'treat-all':16}{'':>8}{'':>7}{'':>7}   "+" ".join(f"{pt:.2f}:{ta[pt]:+.3f}" for pt in pts))
print(f"{'treat-none':16}{'':>8}{'':>7}{'':>7}   "+" ".join(f"{pt:.2f}:{0.0:+.3f}" for pt in pts))

# paired NB delta (decision - standard), subject bootstrap at the central screening pt=0.15
def boot_nb_delta(pt=0.15, B=2000, seed=0):
    rng=np.random.RandomState(seed); n=len(y); idx=np.arange(n); D=[]
    ps=std["tachy_prob"].values; pd=dec["tachy_prob"].values
    for _ in range(B):
        s=rng.choice(idx,n,replace=True); yy=y[s]
        if yy.sum()<3: continue
        def nb1(p):
            dd=(p[s]>=pt).astype(int); tp=((dd==1)&(yy==1)).sum(); fp=((dd==1)&(yy==0)).sum()
            return (tp-(pt/(1-pt))*fp)/len(s)
        D.append(nb1(pd)-nb1(ps))
    D=np.array(D); return np.mean(D),np.percentile(D,2.5),np.percentile(D,97.5),np.mean(D>0)
m,lo,hi,pw=boot_nb_delta()
print(f"\nNB delta (decision - standard) @ p_t=0.15: {m:+.4f} [{lo:+.4f},{hi:+.4f}]  P(decision>standard)={pw:.3f}")
