"""E2 (signature experiment): make the framing thesis ARCHITECTURE-AGNOSTIC. For each backbone,
compare regression-then-threshold (the extractor's OWN pred_rr -> WHO margin) vs
classification-on-signal (ROCKET on the extractor's respiratory WAVEFORM), on the SAME subjects,
subject-grouped 5-fold CV, DeLong + subject-bootstrap dAUROC. If cls >= reg for every backbone,
the framing gain is not a PRISM artifact. (Regression uses the extractor's real RR -> fair, not a
naive FFT strawman.)

Run:  python scripts/framing_headswap.py
"""
import os, sys, glob, warnings
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import signal as ss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_clinical_models import delong_test

LEN=300; dev="cuda" if torch.cuda.is_available() else "cpu"
# backbone -> (waveform signal dir, its k-fold run for the regression pred_rr)
SETS={
    "EfficientPhys": ("data/pedrr_signals_learned", "outputs/paper_runs/pedrr_kfold_efficientphys"),
    "TS-CAN":        ("data/pedrr_signals_tscan",    "outputs/paper_runs/pedrr_kfold_tscan"),
    "PRISM(fusion)": ("data/pedrr_signals_learned",  "outputs/paper_runs/pedrr_kfold_learned"),
}

def load_waveforms(d):
    out={}
    for f in glob.glob(os.path.join(d,"Pediatric-RR_*.npz")):
        z=np.load(f,allow_pickle=True)
        if "flow" not in z.files: continue
        x=np.asarray(z["flow"],float)
        if x.std()<1e-6: continue
        out[f"{z['subject_id']}_{z['video_id']}"]=x
    return out

def load_reg(run):
    paths=sorted(glob.glob(os.path.join(run,"fold*","test_eval","prism_breath_predictions.csv")))
    df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    if "pred_rr_bpm" not in df and "pred_bpm" in df: df["pred_rr_bpm"]=df["pred_bpm"]
    df["clip_key"]=df["subject_id"].astype(str)+"_"+df["video_id"].astype(str)
    g=df.groupby("clip_key").agg(subject_id=("subject_id","first"),pred_rr=("pred_rr_bpm","mean"),
        threshold=("threshold_bpm","first"),gt_tachy=("gt_tachy","first")).reset_index()
    return g

def rocket(X):
    torch.manual_seed(0); xt=torch.tensor(X,device=dev).unsqueeze(1); feats=[]
    for ks,dil in [(9,1),(9,2),(9,4),(11,1),(11,4),(7,2),(5,1),(13,2)]:
        w=torch.randn(96,1,ks,device=dev); b=torch.randn(96,device=dev)
        c=torch.nn.functional.conv1d(xt,w,padding=(ks//2)*dil,dilation=dil)
        feats.append((c>b[None,:,None]).float().mean(-1).cpu().numpy()); feats.append(c.max(-1).values.cpu().numpy())
    return np.concatenate(feats,1)

def cv_prob(F,y,grp):
    uniq=sorted(set(grp)); rng=np.random.RandomState(0)
    up=[u for u in uniq if y[grp==u].sum()>0]; un=[u for u in uniq if y[grp==u].sum()==0]
    rng.shuffle(up); rng.shuffle(un); fold={}
    for i,u in enumerate(up): fold[u]=i%5
    for i,u in enumerate(un): fold[u]=i%5
    fk=np.array([fold[g] for g in grp]); prob=np.full(len(y),np.nan)
    for k in range(5):
        tr=fk!=k; te=fk==k
        if te.sum()==0: continue
        if y[tr].sum()<2 or (~y[tr].astype(bool)).sum()<2: prob[te]=y[tr].mean(); continue
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=0.2)); clf.fit(F[tr],y[tr])
        prob[te]=clf.predict_proba(F[te])[:,1]
    return prob

def boot_delta(y,reg,cls,grp,B=2000):
    rng=np.random.RandomState(0); subs=np.array(sorted(set(grp))); idxby={s:np.where(grp==s)[0] for s in subs}
    D=[]
    for _ in range(B):
        idx=np.concatenate([idxby[s] for s in rng.choice(subs,len(subs),replace=True)]); yy=y[idx]
        if yy.sum()<3 or (yy==0).sum()<3: continue
        D.append(roc_auc_score(yy,cls[idx])-roc_auc_score(yy,reg[idx]))
    D=np.array(D); return np.percentile(D,2.5),np.percentile(D,97.5),float(np.mean(D>0))

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--matched",action="store_true",
        help="restrict all backbones to the common 298 matched set (reconciled basis)"); A=ap.parse_args()
    COMMON=None
    if A.matched:
        from full_metric_comparison import RUNS as ALLRUNS, load_run as _lr
        COMMON=set.intersection(*[set(_lr(r)["clip_key"]) for r in ALLRUNS.values()])
    print(f"=== E2 FRAMING HEAD-SWAP: regression-then-threshold vs ROCKET classification-on-signal"
          f"{' (matched N=%d)'%len(COMMON) if COMMON else ''} ===")
    print("{:<15}{:>5}{:>6}{:>10}{:>10}{:>16}{:>10}{:>9}".format(
        "backbone","n","pos%","AUROC_reg","AUROC_cls","dAUROC[95%CI]","P(c>r)","DeLong"))
    for name,(sd,run) in SETS.items():
        wav=load_waveforms(sd); reg=load_reg(run)
        reg=reg[reg["clip_key"].isin(wav)]
        if COMMON is not None: reg=reg[reg["clip_key"].isin(COMMON)]
        reg=reg.reset_index(drop=True)
        X=np.array([ (lambda z:(z-z.mean())/(z.std()+1e-6))(ss.resample(wav[k],LEN)).astype(np.float32)
                     for k in reg["clip_key"] ])
        y=reg["gt_tachy"].values.astype(int); grp=reg["subject_id"].values
        rscore=(reg["pred_rr"]-reg["threshold"]).values
        if y.sum()<5 or (y==0).sum()<5: print(f"{name}: degenerate"); continue
        prob=cv_prob(rocket(X),y,grp)
        aR=roc_auc_score(y,rscore); aC=roc_auc_score(y,prob)
        _,_,pdl=delong_test(y,rscore,prob); lo,hi,pw=boot_delta(y,rscore,prob,grp)
        print("{:<15}{:>5}{:>5.0f}%{:>10.3f}{:>10.3f}{:>16}{:>10.3f}{:>9.3g}".format(
            name,len(y),100*y.mean(),aR,aC,f"{aC-aR:+.3f}[{lo:+.2f},{hi:+.2f}]",pw,pdl))
    print("\n reg = extractor's own pred_rr -> WHO margin (fair); cls = ROCKET on its waveform. Same subjects, subject-grouped 5-fold CV.")
