"""Tier-1: WHEN does classification-on-signal beat regression-then-threshold?

Characterizes the framing gain (AUROC_cls - AUROC_reg, and decision-accuracy advantage) as a
function of GT-free difficulty axes on the pedrr per-fold predictions:
  * reg framing  = rank/threshold by (pred_rr - WHO_threshold)
  * cls framing  = PRISM tachy_prob (trained classification head)
  * noise proxy  = disagreement_score (cross-modality), min modality quality
  * threshold proximity = |pred_rr - threshold|  (estimate near the WHO cutoff)
  * RR level     = age band (WHO thresholds -> higher RR at younger age)

Thesis: the gain is a decision-theoretic consequence of thresholding a NOISY estimate near the
cutoff -> it concentrates in the near-threshold x high-noise cell and vanishes when the signal is
clean or the case is far from the cutoff. CPU-only; reads cached CSVs.
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

RUN = "outputs/paper_runs/pedrr_kfold_learned"
paths = sorted(glob.glob(os.path.join(RUN, "fold*", "test_eval", "prism_breath_predictions.csv")))
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
df["clip"] = df["subject_id"].astype(str) + "_" + df["video_id"].astype(str)
num = ["gt_rr_bpm","pred_rr_bpm","threshold_bpm","age_months","gt_tachy","tachy_prob",
       "disagreement_score","depth_quality","flow_quality","audio_quality"]
g = df.groupby("clip").agg({**{c:"mean" for c in num}, "subject_id":"first"}).reset_index()

y   = g["gt_tachy"].round().astype(int).values
reg = (g["pred_rr_bpm"] - g["threshold_bpm"]).values      # regression framing score (margin)
cls = g["tachy_prob"].values                              # classification framing score
grp = g["subject_id"].values
g["margin_abs"] = np.abs(g["pred_rr_bpm"] - g["threshold_bpm"])   # distance of ESTIMATE from cutoff
g["minqual"]    = g[["depth_quality","flow_quality","audio_quality"]].min(axis=1)

def auroc(yy, s):
    return roc_auc_score(yy, s) if len(np.unique(yy)) == 2 else np.nan

def boot_gain(mask, B=2000, seed=0):
    rng = np.random.RandomState(seed)
    yy, cc, rr, gg = y[mask], cls[mask], reg[mask], grp[mask]
    subs = np.unique(gg); idxby = {s: np.where(gg == s)[0] for s in subs}; D = []
    for _ in range(B):
        idx = np.concatenate([idxby[s] for s in rng.choice(subs, len(subs), replace=True)])
        if len(np.unique(yy[idx])) < 2: continue
        D.append(auroc(yy[idx], cc[idx]) - auroc(yy[idx], rr[idx]))
    D = np.array(D)
    return (np.mean(D), np.percentile(D, 2.5), np.percentile(D, 97.5), float(np.mean(D > 0))) if len(D) else (np.nan,)*4

print(f"n={len(g)} subjects, tachypnea prevalence={y.mean():.2f}\n")
aC, aR = auroc(y, cls), auroc(y, reg)
m, lo, hi, pw = boot_gain(np.ones(len(g), bool))
print(f"OVERALL  cls AUROC={aC:.3f}  reg AUROC={aR:.3f}  GAIN={aC-aR:+.3f}  [boot {lo:+.3f},{hi:+.3f}]  P(gain>0)={pw:.3f}")

def strat(col, name, labels=("low","mid","high")):
    print(f"\n-- framing gain by {name} tertile --")
    try:
        t = pd.qcut(g[col], 3, labels=labels, duplicates="drop")
    except Exception:
        print("   (cannot bin)"); return
    for lab in labels:
        mask = (t == lab).values
        if mask.sum() == 0: continue
        yy = y[mask]
        if len(np.unique(yy)) < 2:
            print(f"   {lab:4s} n={mask.sum():3d} tachy={int(yy.sum()):2d}  (one class)"); continue
        gg = auroc(yy, cls[mask]) - auroc(yy, reg[mask])
        rng = f"[{g[col][mask].min():.2f},{g[col][mask].max():.2f}]"
        print(f"   {lab:4s} n={mask.sum():3d} tachy={int(yy.sum()):2d}  range{rng:16s} cls-reg gain={gg:+.3f}")

strat("disagreement_score", "cross-modality disagreement (noise, GT-free)")
strat("minqual", "min modality quality (lower=noisier)", labels=("low","mid","high"))
strat("margin_abs", "|pred_rr - threshold| (estimate distance from cutoff)")
strat("age_months", "age (months): younger -> higher RR")

# ---- decision-level (operating point) mechanism ----
reg_dec = (g["pred_rr_bpm"] >= g["threshold_bpm"]).astype(int).values
cls_dec = (g["tachy_prob"]  >= 0.5).astype(int).values
net = (cls_dec == y).astype(int) - (reg_dec == y).astype(int)      # +1 cls wins the decision, -1 reg wins
flip = net != 0
print(f"\n=== DECISION-FLIP MECHANISM (op point: reg=pred>=thr, cls=prob>=0.5) ===")
print(f"  cls and reg DISAGREE on {flip.sum()}/{len(g)} subjects; among disagreements cls is correct "
      f"{(net[flip] > 0).mean():.1%} of the time")
print(f"  mean |pred_rr-thr| : disagreements={g['margin_abs'][flip].mean():.1f} vs agreements={g['margin_abs'][~flip].mean():.1f} BPM")
print(f"  mean disagreement_score : disagreements={g['disagreement_score'][flip].mean():.3f} vs agreements={g['disagreement_score'][~flip].mean():.3f}")

# ---- 2D phase diagram: net cls decision advantage over (noise x proximity) ----
print(f"\n=== PHASE DIAGRAM: mean cls decision advantage (net) by noise x threshold-proximity ===")
noise_t = pd.qcut(g["disagreement_score"], 3, labels=["clean","mid","noisy"], duplicates="drop")
prox_t  = pd.qcut(g["margin_abs"], 3, labels=["near","mid","far"], duplicates="drop")   # small margin = near
grid_net = pd.crosstab(noise_t, prox_t, values=net, aggfunc="mean").round(3)
grid_n   = pd.crosstab(noise_t, prox_t)
print("mean net (cls_correct - reg_correct):"); print(grid_net.to_string())
print("\ncell counts:"); print(grid_n.to_string())

# ---- figure ----
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    # (a) 1D gain vs each axis
    axes = [("disagreement_score","disagreement"),("margin_abs","|est-cutoff|"),("age_months","age(mo)")]
    for col, nm in axes:
        t = pd.qcut(g[col], 3, labels=[0,1,2], duplicates="drop").astype(int)
        gains = []
        for b in [0,1,2]:
            mk = (t==b).values; yy=y[mk]
            gains.append(auroc(yy,cls[mk])-auroc(yy,reg[mk]) if len(np.unique(yy))==2 else np.nan)
        ax[0].plot([0,1,2], gains, marker="o", label=nm)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xticks([0,1,2]); ax[0].set_xticklabels(["low","mid","high"])
    ax[0].set_ylabel("framing gain ($\\Delta$AUROC)"); ax[0].set_xlabel("difficulty tertile")
    ax[0].legend(fontsize=8, frameon=False); ax[0].set_title("(a) gain grows with difficulty", fontsize=11)
    # (b) 2D phase diagram heatmap
    M = grid_net.reindex(index=["clean","mid","noisy"], columns=["far","mid","near"]).values.astype(float)
    im = ax[1].imshow(M, cmap="RdBu_r", vmin=-np.nanmax(np.abs(M)), vmax=np.nanmax(np.abs(M)))
    ax[1].set_xticks([0,1,2]); ax[1].set_xticklabels(["far","mid","near"]); ax[1].set_xlabel("proximity to cutoff")
    ax[1].set_yticks([0,1,2]); ax[1].set_yticklabels(["clean","mid","noisy"]); ax[1].set_ylabel("estimation noise")
    for i in range(3):
        for j in range(3):
            if np.isfinite(M[i,j]): ax[1].text(j,i,f"{M[i,j]:+.2f}",ha="center",va="center",fontsize=9)
    ax[1].set_title("(b) cls decision advantage", fontsize=11); fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout(); fig.savefig("manuscript/fig_framing_phase.png", dpi=600); fig.savefig("manuscript/fig_framing_phase.pdf", dpi=600)
    print("\n[fig] wrote manuscript/fig_framing_phase.{png,pdf}")
except Exception as e:
    print(f"\n[fig] skipped ({e})")
