"""E7: calibration & abstention diagnostics on the matched pedrr 5-fold predictions.
- ECE + MCE + Brier + reliability bins for tachy_prob (all models; PRISM head vs baselines'
  hand-set logistic of RR margin).
- Conformal empirical coverage vs nominal 1-alpha=0.80, overall + per WHO age band (PRISM;
  checks the age-miscalibration concern).
- Risk-coverage / selective prediction: abstain most-uncertain first, does tachypnea-decision
  error on the retained drop? AURC (lower=better). PRISM disagreement_score vs prob-confidence
  vs baselines' prob-confidence -> shows disagreement-conditioned triage adds value.
- Triage-code summary: accuracy/sensitivity on auto-decided vs referred.

Run:  python scripts/calibration_diagnostics.py [--deepphys100]
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_metric_comparison import RUNS

NOMINAL = 0.80   # 1 - alpha (alpha=0.20, pooled conformal)
BANDS = [("overall",-1,999),("2-12mo",2,12),("12-60mo",12,61),("<2mo",-1,2)]
OPT = ["qhat","interval_lower_bpm","interval_upper_bpm","triage_code","disagreement_score"]

def load(run_root):
    paths = sorted(glob.glob(os.path.join(run_root,"fold*","test_eval","prism_breath_predictions.csv"))) \
         or sorted(glob.glob(os.path.join(run_root,"seed*","test_eval","prism_breath_predictions.csv")))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "gt_rr_bpm"   not in df and "gt_bpm"   in df: df["gt_rr_bpm"]   = df["gt_bpm"]
    if "pred_rr_bpm" not in df and "pred_bpm" in df: df["pred_rr_bpm"] = df["pred_bpm"]
    df["clip_key"] = df["subject_id"].astype(str)+"_"+df["video_id"].astype(str)
    agg = {"subject_id":("subject_id","first"),"age_months":("age_months","first"),
           "gt_rr":("gt_rr_bpm","mean"),"pred_rr":("pred_rr_bpm","mean"),
           "threshold":("threshold_bpm","first"),"gt_tachy":("gt_tachy","first"),
           "tachy_prob":("tachy_prob","mean")}
    for c in OPT:
        if c in df.columns: agg[c]=(c,"first" if c=="triage_code" else "mean")
    return df.groupby("clip_key").agg(**agg).reset_index()

def ece_brier(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); ece=mce=0.0; rows=[]
    for i in range(bins):
        m=(p>=edges[i])&(p<edges[i+1]) if i<bins-1 else (p>=edges[i])&(p<=edges[i+1])
        if m.sum()==0: continue
        conf=p[m].mean(); acc=y[m].mean(); w=m.sum()/len(p)
        ece+=w*abs(acc-conf); mce=max(mce,abs(acc-conf)); rows.append((edges[i],edges[i+1],conf,acc,int(m.sum())))
    return ece,mce,float(np.mean((p-y)**2)),rows

def aurc(unc,correct):
    """area under risk-coverage: retain most-certain (lowest unc) first."""
    order=np.argsort(unc); c=correct[order]; n=len(c)
    covs=np.arange(1,n+1)/n; risk=1-np.cumsum(c)/np.arange(1,n+1)
    return float(np.trapz(risk,covs)), covs, risk

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--deepphys100",action="store_true"); a=ap.parse_args()
    if a.deepphys100: RUNS["DeepPhys"]="outputs/paper_runs/pedrr_kfold_deepphys_100ep"
    data={m:load(r) for m,r in RUNS.items()}
    common=set.intersection(*[set(d["clip_key"]) for d in data.values()])
    data={m:d[d["clip_key"].isin(common)].reset_index(drop=True) for m,d in data.items()}
    print(f"common clips={len(common)}\n")

    print("=== (1) PROBABILITY CALIBRATION (tachy_prob) ===")
    print("{:<14}{:>8}{:>8}{:>8}{:>8}".format("model","ECE","MCE","Brier","AURC"))
    for m in RUNS:
        d=data[m]; y=d["gt_tachy"].values.astype(int); p=np.clip(d["tachy_prob"].values,0,1)
        ece,mce,brier,_=ece_brier(y,p)
        correct=((d["pred_rr"]>=d["threshold"]).values.astype(int)==y).astype(float)
        conf=np.abs(p-0.5); au,_,_=aurc(-conf,correct)   # high conf -> retain first
        print("{:<14}{:>8.3f}{:>8.3f}{:>8.3f}{:>8.3f}".format(m,ece,mce,brier,au))

    P=data["PRISM-Breath"]
    if "interval_lower_bpm" in P.columns:
        print(f"\n=== (2) CONFORMAL COVERAGE (RR in interval; nominal={NOMINAL:.2f}) ===")
        for bn,lo,hi in BANDS:
            # Half-open [lo, hi) to match the WHO bands. The previous strict-on-both-sides filter
            # silently dropped every subject whose age fell exactly on a boundary (2 or 12 months),
            # so the per-band counts summed to 266 against an overall n of 298.
            s=P[(P["age_months"]>=lo)&(P["age_months"]<hi)] if bn!="overall" else P
            if len(s)<10: print(f"  {bn:8s}: n={len(s)} (small)"); continue
            cov=((s["gt_rr"]>=s["interval_lower_bpm"])&(s["gt_rr"]<=s["interval_upper_bpm"])).mean()
            w=(s["interval_upper_bpm"]-s["interval_lower_bpm"]).mean()
            flag="OK" if abs(cov-NOMINAL)<=0.07 else ("UNDER" if cov<NOMINAL else "OVER")
            print(f"  {bn:8s}: n={len(s):3d} coverage={cov:.3f} width={w:.1f}  [{flag}]")

    if "disagreement_score" in P.columns:
        print("\n=== (3) SELECTIVE PREDICTION (risk-coverage; retain most-certain) ===")
        y=P["gt_tachy"].values.astype(int); correct=((P["pred_rr"]>=P["threshold"]).values.astype(int)==y).astype(float)
        rankers={"disagreement":P["disagreement_score"].values, "prob-conf":-np.abs(P["tachy_prob"].values-0.5)}
        print("  coverage:   " + "".join(f"{c:>8.1f}" for c in [1.0,0.9,0.8,0.7,0.6]))
        for name,unc in rankers.items():
            au,covs,risk=aurc(unc,correct)
            pts=[risk[np.argmin(np.abs(covs-c))] for c in [1.0,0.9,0.8,0.7,0.6]]
            print(f"  {name:11s}:"+"".join(f"{r:>8.3f}" for r in pts)+f"   AURC={au:.3f}")
        print("  (lower risk at lower coverage = good abstention; disagreement AURC < prob-conf AURC = triage adds value)")

    if "triage_code" in P.columns:
        print("\n=== (4) SELECTIVE TRIAGE (auto-decided vs referred) ===")
        auto=P[P["triage_code"]!=2]; ref=P[P["triage_code"]==2]  # code2=UNCERTAIN_REVIEW=refer
        for nm,s in [("ALL",P),("AUTO-DECIDED (code!=2)",auto),("REFERRED (code2)",ref)]:
            if len(s)==0: continue
            y=s["gt_tachy"].values.astype(int); dec=(s["pred_rr"]>=s["threshold"]).values.astype(int)
            acc=(dec==y).mean(); sens=(dec[y==1]==1).mean() if (y==1).any() else float("nan")
            print(f"  {nm:24s} n={len(s):3d} ({100*len(s)/len(P):4.0f}%)  acc={acc:.3f}  sens={sens:.3f}  tachy={int(y.sum())}")
