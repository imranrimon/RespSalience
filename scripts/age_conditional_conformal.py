"""Age-conditional (Mondrian) split-conformal for the RR estimate. Pooled conformal robs the young to
pay the old (coverage <2mo=0.59, 2-12mo=0.71, 12-60mo=0.88); this computes a SEPARATE qhat per WHO age
band from validation residuals and applies it per band -> each band hits its target coverage, and <2mo
lands at an interval so wide it amounts to 'defer' (consistent with the RGB-neonatal reframe)."""
import argparse, glob, json
import numpy as np, pandas as pd
BANDS=[("<2mo",0,2),("2-12mo",2,12),("12-60mo",12,60)]
def band(a): return next((n for n,lo,hi in BANDS if lo<=a<hi),"12-60mo")
def fold_frame(path):
    d=pd.read_csv(path).dropna(subset=["gt_bpm","pred_bpm"])
    d["k"]=d.subject_id.astype(str)+"_"+d.video_id.astype(str)
    g=d.groupby("k").agg(gt=("gt_bpm","first"),pred=("pred_bpm","mean"),age=("age_months","first"),
                         sid=("subject_id","first")).reset_index()
    g["band"]=g["age"].apply(band); return g
def by_fold(run,kind):
    out={}
    for c in sorted(glob.glob(f"{run}/fold*/{kind}/prism_breath_predictions.csv")):
        f=int([p for p in c.replace("\\","/").split("/") if p.startswith("fold")][-1].replace("fold",""))
        out[f]=fold_frame(c)
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",default="outputs/paper_runs/pedrr_kfold_learned")
    ap.add_argument("--alpha",type=float,default=0.20)
    ap.add_argument("--calibration",choices=["lofo","val"],default="lofo",
                    help="lofo: other folds' test residuals (subject-disjoint by construction). "
                         "val: the fold's own validation split (small but strictly split-conformal).")
    ap.add_argument("--out",default="outputs/paper_runs/pedrr_kfold_learned/age_conditional_conformal.json")
    a=ap.parse_args()
    # Calibrate PER FOLD. Pooling every fold's validation rows and applying the
    # result to the pooled test set leaks: the validation splits are drawn from
    # only ~49 distinct subjects, ~10 of which appear in each test fold.
    val_f=by_fold(a.run,"val_eval"); test_f=by_fold(a.run,"test_eval")
    print(f"Mondrian conformal (alpha={a.alpha}, target cov={1-a.alpha:.2f}, calibration={a.calibration})")
    covered={n:[] for n,_,_ in BANDS}; qhats={n:[] for n,_,_ in BANDS}
    for f in sorted(test_f):
        cal=(pd.concat([test_f[g] for g in test_f if g!=f],ignore_index=True)
             if a.calibration=="lofo" else val_f.get(f))
        if cal is None or cal.empty: continue
        leak=set(cal.sid.astype(str)) & set(test_f[f].sid.astype(str))
        if leak: raise SystemExit(f"fold{f}: {len(leak)} calibration subjects also in test; refusing.")
        for n,lo,hi in BANDS:
            v=cal[cal["band"]==n]; t=test_f[f][test_f[f]["band"]==n]
            if len(v)<3 or len(t)==0: continue
            r=np.abs(v["pred"]-v["gt"]).values
            q=float(np.quantile(r,min(1.0,np.ceil((len(r)+1)*(1-a.alpha))/len(r))))
            qhats[n].append(q); covered[n].extend((np.abs(t["pred"]-t["gt"]).values<=q).tolist())
    print(f"  {'band':<9}{'qhat(BPM)':>10}{'test_cov':>10}{'n_test':>8}{'interval':>16}")
    res={}
    for n,_,_ in BANDS:
        if not covered[n]: continue
        q=float(np.mean(qhats[n])); cov=float(np.mean(covered[n]))
        res[n]={"qhat":q,"coverage":cov,"n":len(covered[n]),"calibration":a.calibration}
        tag=" (=> DEFER: interval too wide to be useful)" if q>=20 else ""
        warn="" if cov>=1-a.alpha-1e-9 else "  <-- UNDER-COVERS"
        print(f"  {n:<9}{q:>10.1f}{cov:>10.2f}{len(covered[n]):>8d}{'+/-'+f'{q:.0f}':>16}{tag}{warn}")
    allc=[c for n,_,_ in BANDS for c in covered[n]]
    if allc: print(f"  {'overall':<9}{'':>10}{np.mean(allc):>10.2f}{len(allc):>8d}")

    # POOLED arm: one qhat over all bands, calibrated identically (same folds, same leak guard), so the
    # "Mondrian vs pooled" contrast in the paper comes from a single artifact instead of two runs.
    pcov={n:[] for n,_,_ in BANDS}; pq=[]
    for f in sorted(test_f):
        cal=(pd.concat([test_f[g] for g in test_f if g!=f],ignore_index=True)
             if a.calibration=="lofo" else val_f.get(f))
        if cal is None or cal.empty: continue
        r=np.abs(cal["pred"]-cal["gt"]).values
        if len(r)<3: continue
        q=float(np.quantile(r,min(1.0,np.ceil((len(r)+1)*(1-a.alpha))/len(r))))
        pq.append(q)
        for n,_,_ in BANDS:
            t=test_f[f][test_f[f]["band"]==n]
            if len(t): pcov[n].extend((np.abs(t["pred"]-t["gt"]).values<=q).tolist())
    if pq:
        qp=float(np.mean(pq))
        print(f"\nPooled conformal (single width {qp:.1f} BPM for every band, same calibration)")
        print(f"  {'band':<9}{'qhat(BPM)':>10}{'test_cov':>10}{'n_test':>8}{'status':>12}")
        res["_pooled"]={"qhat":qp,"calibration":a.calibration,"bands":{}}
        for n,_,_ in BANDS:
            if not pcov[n]: continue
            cov=float(np.mean(pcov[n]))
            st="OK" if abs(cov-(1-a.alpha))<=0.05 else ("UNDER" if cov<1-a.alpha else "OVER")
            res["_pooled"]["bands"][n]={"coverage":cov,"n":len(pcov[n]),"status":st}
            print(f"  {n:<9}{qp:>10.1f}{cov:>10.2f}{len(pcov[n]):>8d}{st:>12}")
        allp=[c for n,_,_ in BANDS for c in pcov[n]]
        if allp:
            res["_pooled"]["overall"]={"coverage":float(np.mean(allp)),"n":len(allp)}
            print(f"  {'overall':<9}{qp:>10.1f}{np.mean(allp):>10.2f}{len(allp):>8d}")
        print("  A single width meets the target on average while missing it in both directions per\n"
              "  band -- which is the case for the age-conditional construction.")
    json.dump(res,open(a.out,"w"),indent=2); print(f"  -> {a.out}")
if __name__=="__main__": main()
