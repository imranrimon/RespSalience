"""Subgroup / fairness table: PRISM vs baselines stratified by (a) WHO age band and
(b) recording SITE, on the matched pedrr 5-fold common set, with subject-grouped bootstrap
95% CIs. Age = clinical-difficulty axis (neonatal frontier); site = deployment-generalization
/ fairness across 3 real Bangladeshi care settings. Sex/skin-tone need external metadata
(empty vep_mendeley; doc reports sex audit = no disparity, skin-tone single-population).

Run from repo root:  python scripts/subgroup_fairness.py [--deepphys100]
"""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_metric_comparison import RUNS, point_metrics, subj_boot

SHOW = ["AUROC", "Sp@Se90", "Se@Sp90", "Sens", "MAE"]

def load_run_age(run_root):
    paths = sorted(glob.glob(os.path.join(run_root,"fold*","test_eval","prism_breath_predictions.csv"))) \
         or sorted(glob.glob(os.path.join(run_root,"seed*","test_eval","prism_breath_predictions.csv")))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "gt_rr_bpm"   not in df and "gt_bpm"   in df: df["gt_rr_bpm"]   = df["gt_bpm"]
    if "pred_rr_bpm" not in df and "pred_bpm" in df: df["pred_rr_bpm"] = df["pred_bpm"]
    df["clip_key"] = df["subject_id"].astype(str)+"_"+df["video_id"].astype(str)
    return df.groupby("clip_key").agg(subject_id=("subject_id","first"),
            age_months=("age_months","first"), gt_rr=("gt_rr_bpm","mean"),
            pred_rr=("pred_rr_bpm","mean"), threshold=("threshold_bpm","first"),
            gt_tachy=("gt_tachy","first"), tachy_prob=("tachy_prob","mean")).reset_index()

def site_map():
    m={}
    for f in glob.glob("data/organized/pedrr_indist_*.json"):
        d=json.load(open(f)); ents=d if isinstance(d,list) else d.get("videos",d.get("entries",d))
        for e in ents:
            if isinstance(e,dict) and "site_id" in e:
                m[f"{e['subject_id']}_{e['video_id']}"]="site "+str(e["site_id"]).strip()[:1]
    return m

def band_of(a): return "<2mo" if a<2 else ("2-12mo" if a<12 else "12-60mo")

def report(title, data, assign, order):
    print(f"\n######## {title} (subject-bootstrap 95% CI) ########",flush=True)
    ref=data["PRISM-Breath"]
    for g in ["overall"]+order:
        keys=set(ref["clip_key"]) if g=="overall" else {k for k,v in assign.items() if v==g}
        keys&=set(ref["clip_key"])
        sub=ref[ref["clip_key"].isin(keys)]; n=len(sub); npos=int(sub["gt_tachy"].sum())
        print(f"\n=== {g}: n={n}, tachy={npos} ({100*npos/max(n,1):.0f}%) ===",flush=True)
        if n<15 or npos<3 or npos==n:
            print("   (too small / degenerate)",flush=True); continue
        print("{:<14}".format("model")+"".join("{:>20}".format(c) for c in SHOW),flush=True)
        for m in RUNS:
            d=data[m][data[m]["clip_key"].isin(keys)].reset_index(drop=True)
            pts=point_metrics(d); ci=subj_boot(d, B=1500)
            row="{:<14}".format(m)
            for c in SHOW:
                v=pts.get(c,np.nan); lo,hi=ci.get(c,(np.nan,np.nan))
                row+="{:>20}".format(f"{v:.3f}[{lo:.2f},{hi:.2f}]")
            print(row,flush=True)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--deepphys100",action="store_true"); a=ap.parse_args()
    if a.deepphys100: RUNS["DeepPhys"]="outputs/paper_runs/pedrr_kfold_deepphys_100ep"
    data={m:load_run_age(r) for m,r in RUNS.items()}
    common=set.intersection(*[set(d["clip_key"]) for d in data.values()])
    data={m:d[d["clip_key"].isin(common)].reset_index(drop=True) for m,d in data.items()}
    print(f"common clips={len(common)}",flush=True)
    ref=data["PRISM-Breath"]
    age_assign={r.clip_key:band_of(r.age_months) for r in ref.itertuples()}
    report("AGE BAND (clinical difficulty)", data, age_assign, ["2-12mo","12-60mo","<2mo"])
    smap={k:v for k,v in site_map().items() if k in common}
    sites=sorted(set(smap.values()))
    print(f"\nsite coverage on common set: { {s:sum(1 for v in smap.values() if v==s) for s in sites} }",flush=True)
    report("RECORDING SITE (deployment generalization)", data, smap, sites)
