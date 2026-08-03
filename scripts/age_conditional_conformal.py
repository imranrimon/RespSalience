"""Age-conditional (Mondrian) split-conformal for the RR estimate. Pooled conformal robs the young to
pay the old (coverage <2mo=0.59, 2-12mo=0.71, 12-60mo=0.88); this computes a SEPARATE qhat per WHO age
band from validation residuals and applies it per band -> each band hits its target coverage, and <2mo
lands at an interval so wide it amounts to 'defer' (consistent with the RGB-neonatal reframe)."""
import argparse, glob, json
import numpy as np, pandas as pd
BANDS=[("<2mo",0,2),("2-12mo",2,12),("12-60mo",12,60)]
def band(a): return next((n for n,lo,hi in BANDS if lo<=a<hi),"12-60mo")
def pool(run,kind):
    cs=sorted(glob.glob(f"{run}/fold*/{kind}/prism_breath_predictions.csv"))
    d=pd.concat([pd.read_csv(c) for c in cs],ignore_index=True).dropna(subset=["gt_bpm","pred_bpm"])
    d["k"]=d.subject_id.astype(str)+"_"+d.video_id.astype(str)
    g=d.groupby("k").agg(gt=("gt_bpm","first"),pred=("pred_bpm","mean"),age=("age_months","first")).reset_index()
    g["band"]=g["age"].apply(band); return g
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",default="outputs/paper_runs/pedrr_kfold_learned")
    ap.add_argument("--alpha",type=float,default=0.20); ap.add_argument("--out",default="outputs/paper_runs/pedrr_kfold_learned/age_conditional_conformal.json")
    a=ap.parse_args()
    val=pool(a.run,"val_eval"); test=pool(a.run,"test_eval")
    print(f"Mondrian conformal (alpha={a.alpha}, target cov={1-a.alpha:.2f})")
    print(f"  {'band':<9}{'qhat(BPM)':>10}{'test_cov':>10}{'n_test':>8}{'interval':>16}")
    res={}
    for n,lo,hi in BANDS:
        v=val[val["band"]==n]; t=test[test["band"]==n]
        if len(v)<3 or len(t)==0: continue
        r=np.abs(v["pred"]-v["gt"]).values; q=float(np.quantile(r,min(1.0,np.ceil((len(r)+1)*(1-a.alpha))/len(r))))
        cov=float((np.abs(t["pred"]-t["gt"]).values<=q).mean())
        res[n]={"qhat":q,"coverage":cov,"n":len(t)}
        tag=" (=> DEFER: interval too wide to be useful)" if q>=20 else ""
        print(f"  {n:<9}{q:>10.1f}{cov:>10.2f}{len(t):>8d}{'+/-'+f'{q:.0f}':>16}{tag}")
    json.dump(res,open(a.out,"w"),indent=2); print(f"  -> {a.out}")
if __name__=="__main__": main()
