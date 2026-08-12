"""Decision-curve analysis (net benefit) — clinical-utility row. NB(pt) = TP/N - FP/N * pt/(1-pt),
deciding tachypnea when tachy_prob >= pt. Screening cares about LOW pt (catch sick kids, tolerate
false alarms). Compares each model vs treat-all / treat-none, subject-bootstrap 95% CI at key pt.
PRISM's calibrated probs (ECE 0.070) should dominate the baselines' ad-hoc logistic.

Run:  python scripts/decision_curve.py [--deepphys100]
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_metric_comparison import RUNS, load_run

PTS = [0.10, 0.20, 0.30, 0.40]

def nb(y, prob, pt):
    dec = prob >= pt; N=len(y)
    tp=((dec)&(y==1)).sum(); fp=((dec)&(y==0)).sum()
    return tp/N - fp/N*(pt/(1-pt))

def nb_boot(d, pt, B=1500):
    rng=np.random.RandomState(0); subs=d["subject_id"].unique()
    idxby={s:d.index[d["subject_id"]==s].values for s in subs}; V=[]
    y=d["gt_tachy"].values.astype(int); p=d["tachy_prob"].values
    for _ in range(B):
        idx=np.concatenate([idxby[s] for s in rng.choice(subs,len(subs),replace=True)])
        V.append(nb(y[idx],p[idx],pt))
    return np.percentile(V,2.5),np.percentile(V,97.5)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--deepphys100",action="store_true"); a=ap.parse_args()
    if a.deepphys100: RUNS["DeepPhys"]="outputs/paper_runs/pedrr_kfold_deepphys_100ep"
    data={m:load_run(r) for m,r in RUNS.items()}
    for m in data:  # load_run drops subject_id? re-derive from clip_key
        if "subject_id" not in data[m]: data[m]["subject_id"]=data[m]["clip_key"].str.rsplit("_",n=1).str[0]
    common=set.intersection(*[set(d["clip_key"]) for d in data.values()])
    data={m:d[d["clip_key"].isin(common)].reset_index(drop=True) for m,d in data.items()}
    prev=data["PRISM-Breath"]["gt_tachy"].mean(); N=len(common)
    print(f"common clips={N} | tachypnea prevalence={prev:.3f}\n")
    print("=== DECISION-CURVE net benefit (subject-bootstrap 95% CI); higher=more clinical utility ===")
    print("{:<15}".format("pt ->")+"".join(f"{pt:>20.2f}" for pt in PTS));
    # reference strategies
    row="{:<15}".format("treat-all")
    for pt in PTS: row+=f"{prev-(1-prev)*(pt/(1-pt)):>20.3f}"
    print(row); print("{:<15}".format("treat-none")+"".join(f"{0.0:>20.3f}" for _ in PTS))
    for m in RUNS:
        d=data[m]; y=d["gt_tachy"].values.astype(int); p=d["tachy_prob"].values
        row="{:<15}".format(m)
        for pt in PTS:
            v=nb(y,p,pt); lo,hi=nb_boot(d,pt)
            row+=f"{v:.3f}[{lo:.2f},{hi:.2f}]".rjust(20)
        print(row)
    print("\n screening operates at LOW pt; a model whose NB exceeds treat-all AND every baseline there = most clinical utility.")
