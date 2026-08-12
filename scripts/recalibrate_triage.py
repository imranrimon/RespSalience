"""E7 FIX (post-hoc, leakage-free): age-conditional conformal + sensitivity-first interval-straddle
triage, replacing the disagreement_score-based triage. Validates OLD vs NEW on the pedrr 5-fold
predictions. If it improves per-band coverage + auto-decided sensitivity, wire into the eval pipeline.

Age-conditional qhat = per-WHO-band conformal quantile of |pred-gt| from the OTHER folds
(leave-one-fold-out => no leakage). Triage: hi<T -> auto NORMAL; lo>=T -> auto ALERT; else REFER.

Run:  python scripts/recalibrate_triage.py
"""
import glob, re, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

DEPLOY_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "src", "config", "age_conditional_conformal.json")
_STATUS = {0: "NORMAL_SAFE", 1: "TACHYPNEA_ALERT", 2: "UNCERTAIN_REVIEW"}

def apply_deploy_triage(df, cfg_path=DEPLOY_CFG):
    """DEPLOYABLE E7 fix (no fold structure): override interval_lower/upper_bpm, qhat, triage_code,
    triage_status using the persisted per-WHO-band conformal qhat + sensitivity-first interval-straddle
    triage (hi<thr->NORMAL, lo>=thr->ALERT, else REFER). Call from any inference/eval path."""
    cfg = json.load(open(cfg_path)); qb = cfg["qhat_by_band"]
    def _band(a): return "<2mo" if a < 2 else ("2-12mo" if a < 12 else "12-60mo")
    pr = df["pred_rr_bpm"].values.astype(float)
    age = df["age_months"].values.astype(float)
    thr = df["threshold_bpm"].values.astype(float)
    q = np.array([qb[_band(a)] for a in age])
    lo, hi = pr - q, pr + q
    tri = np.where(hi < thr, 0, np.where(lo >= thr, 1, 2)).astype(int)
    df = df.copy()
    df["qhat"] = q; df["qhat_source"] = "age_conditional_deploy"
    df["interval_lower_bpm"] = lo; df["interval_upper_bpm"] = hi
    df["triage_code"] = tri; df["triage_status"] = [_STATUS[c] for c in tri]
    return df

ALPHA = 0.20
RUN = "outputs/paper_runs/pedrr_kfold_learned"

def band(a): return "<2mo" if a<2 else ("2-12mo" if a<12 else "12-60mo")
def conf_q(res, alpha=ALPHA):
    res=np.asarray(res,float); n=len(res)
    if n==0: return np.nan
    k=min(max(int(np.ceil((n+1)*(1-alpha))),1),n)
    return float(np.sort(res)[k-1])
def cover(lo,hi,gt): return float(((gt>=lo)&(gt<=hi)).mean())
def triage(lo,hi,T):  # 0=auto-NORMAL, 1=auto-ALERT, 2=REFER (sensitivity-first straddle)
    return np.where(hi<T,0,np.where(lo>=T,1,2))

def load():
    rows=[]
    for p in sorted(glob.glob(f"{RUN}/fold*/test_eval/prism_breath_predictions.csv")):
        d=pd.read_csv(p); d["fold"]=int(re.search(r"fold(\d+)",p).group(1)); rows.append(d)
    df=pd.concat(rows,ignore_index=True)
    df["band"]=df["age_months"].apply(band); df["res"]=(df["pred_rr_bpm"]-df["gt_rr_bpm"]).abs()
    df["T"]=df["threshold_bpm"]
    return df

def triage_stats(df, code_col, normal_code=0):
    y=df["gt_tachy"].values.astype(int); dec=(df["pred_rr_bpm"]>=df["T"]).values.astype(int)
    auto=df[code_col]!=2; N=len(df)
    ya,da=y[auto],dec[auto]
    acc_auto=(da==ya).mean() if auto.any() else np.nan
    sens_auto=(da[ya==1]==1).mean() if (ya==1).any() else np.nan
    # DANGEROUS error: true tachypnea auto-cleared as NORMAL
    fn_autoclear=int(((df[code_col]==normal_code)&(df["gt_tachy"]==1)).sum())
    return dict(auto_pct=100*auto.mean(), refer_pct=100*(~auto).mean(),
                acc_auto=acc_auto, sens_auto=sens_auto, fn_autocleared=fn_autoclear)

def recalibrate(df, alpha=ALPHA):
    """Age-conditional (leave-one-fold-out) conformal + straddle triage. Returns df with
    new columns nlo/nhi/ntriage. Reusable by the eval pipeline (replaces disagreement triage)."""
    folds=sorted(df["fold"].unique()); nlo=np.empty(len(df)); nhi=np.empty(len(df))
    for k in folds:
        for b in df["band"].unique():
            q=conf_q(df[(df.fold!=k)&(df.band==b)]["res"], alpha)
            m=(df.fold==k)&(df.band==b)
            nlo[m]=df.loc[m,"pred_rr_bpm"]-q; nhi[m]=df.loc[m,"pred_rr_bpm"]+q
    df=df.copy(); df["nlo"],df["nhi"]=nlo,nhi
    df["ntriage"]=triage(df["nlo"].values,df["nhi"].values,df["T"].values)
    return df

STATUS={0:"NORMAL_SAFE",1:"TACHYPNEA_ALERT",2:"UNCERTAIN_REVIEW"}
def write_corrected(df):
    """Write per-fold *_recal.csv with age-conditional interval + straddle triage as canonical cols.
    load() preserves per-fold row order (raw concat, no groupby), so rows align 1:1 with the source."""
    for k in sorted(df["fold"].unique()):
        src=f"{RUN}/fold{k}/test_eval/prism_breath_predictions.csv"
        out=src.replace(".csv","_recal.csv")
        o=pd.read_csv(src).reset_index(drop=True); s=df[df.fold==k].reset_index(drop=True)
        assert len(o)==len(s), f"row mismatch fold{k}: {len(o)} vs {len(s)}"
        o["interval_lower_bpm"]=s["nlo"].values; o["interval_upper_bpm"]=s["nhi"].values
        o["qhat"]=(s["nhi"]-s["pred_rr_bpm"]).values
        o["triage_code"]=s["ntriage"].values; o["triage_status"]=[STATUS[c] for c in s["ntriage"]]
        o.to_csv(out,index=False)
    print(f"wrote {len(df['fold'].unique())} *_recal.csv files under {RUN}/fold*/test_eval/",flush=True)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--write",action="store_true"); A=ap.parse_args()
    df=load(); folds=sorted(df["fold"].unique())
    print(f"n={len(df)} folds={folds} bands={df['band'].value_counts().to_dict()}\n")

    df=recalibrate(df)
    qtab={b:round((df[df.band==b]["nhi"]-df[df.band==b]["pred_rr_bpm"]).mean(),1) for b in df["band"].unique()}
    print("age-conditional qhat (mean):",qtab,"\n")

    print("=== (2) PER-BAND CONFORMAL COVERAGE (nominal 0.80) ===")
    print("{:<9}{:>8}{:>10}{:>10}{:>10}".format("band","n","OLD cov","NEW cov","NEW width"))
    for b in ["overall","2-12mo","12-60mo","<2mo"]:
        s=df if b=="overall" else df[df.band==b]
        oc=cover(s["interval_lower_bpm"],s["interval_upper_bpm"],s["gt_rr_bpm"])
        nc=cover(s["nlo"],s["nhi"],s["gt_rr_bpm"]); w=(s["nhi"]-s["nlo"]).mean()
        print("{:<9}{:>8}{:>10.3f}{:>10.3f}{:>10.1f}".format(b,len(s),oc,nc,w))

    print("\n=== (3+4) TRIAGE: OLD (disagreement) vs NEW (straddle, sensitivity-first) ===")
    print("{:<8}{:>9}{:>9}{:>10}{:>10}{:>14}".format("policy","auto%","refer%","acc_auto","sens_auto","FN_autocleared"))
    for nm,col in [("OLD","triage_code"),("NEW","ntriage")]:
        s=triage_stats(df,col)
        print("{:<8}{:>9.0f}{:>9.0f}{:>10.3f}{:>10.3f}{:>14d}".format(
            nm,s["auto_pct"],s["refer_pct"],s["acc_auto"],s["sens_auto"],s["fn_autocleared"]))
    print("  (goal: NEW auto-cleared FN << OLD; sens_auto up; still meaningful auto% -> workload cut)")
    if A.write: write_corrected(df)
