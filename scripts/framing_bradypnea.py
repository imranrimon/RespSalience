"""Tier-2a: make the framing thesis THRESHOLD-AGNOSTIC (not tachypnea-specific).

The paper's signature claim is that classification-on-signal (ROCKET on the raw respiratory
waveform) beats regression-then-threshold (the extractor's own pred_rr -> WHO margin) for infant
TACHYPNEA (high-RR). Here we show the SAME framing gain holds at the opposite end -- a BRADYPNEA
(low-RR) cutoff -- and across a full sweep of RR thresholds, so the advantage is a property of the
DECISION FRAMING, not of the specific WHO tachypnea thresholds.

We reuse framing_headswap.py verbatim (load_waveforms, rocket, cv_prob, boot_delta, SETS) so the
bradypnea numbers use the IDENTICAL subject-grouped 5-fold CV / subject-bootstrap conventions as
the tachypnea head-swap and are directly comparable.

reg (bradypnea) = CUTOFF - pred_rr        (higher = more bradypneic; the extractor's OWN RR, fair)
cls (bradypnea) = ROCKET on the waveform, subject-grouped CV probability
Sweep: label_thr = 1[gt_rr > thr]; reg = pred_rr - thr (== rank by pred_rr); cls = ROCKET(label_thr).
       gain(thr) = AUROC_cls - AUROC_reg. (By the AUROC label/score-flip symmetry, the gain at a low
       thr is exactly the low-tail / bradypnea-detection gain, so one signed sweep covers both tails.)

Run:  python scripts/framing_bradypnea.py [--cutoff 30]
"""
import os, sys, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, glob
from scipy import signal as ss
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# reuse the EXACT tachypnea head-swap machinery
from framing_headswap import load_waveforms, rocket, cv_prob, boot_delta, SETS, LEN
from compare_clinical_models import delong_test


def load_reg_ext(run):
    """Like framing_headswap.load_reg but also carries gt_rr (continuous) and age (months)."""
    paths = sorted(glob.glob(os.path.join(run, "fold*", "test_eval", "prism_breath_predictions.csv")))
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "pred_rr_bpm" not in df and "pred_bpm" in df:
        df["pred_rr_bpm"] = df["pred_bpm"]
    df["clip_key"] = df["subject_id"].astype(str) + "_" + df["video_id"].astype(str)
    g = df.groupby("clip_key").agg(
        subject_id=("subject_id", "first"), pred_rr=("pred_rr_bpm", "mean"),
        gt_rr=("gt_rr_bpm", "mean"), age=("age_months", "first"),
        threshold=("threshold_bpm", "first"), gt_tachy=("gt_tachy", "first")).reset_index()
    return g


def build_matched(sd, run):
    """Return (reg dataframe, ROCKET feature matrix F) for the subjects with a usable waveform."""
    wav = load_waveforms(sd); reg = load_reg_ext(run)
    reg = reg[reg["clip_key"].isin(wav)].reset_index(drop=True)
    X = np.array([(lambda z: (z - z.mean()) / (z.std() + 1e-6))(ss.resample(wav[k], LEN)).astype(np.float32)
                  for k in reg["clip_key"]])
    F = rocket(X)                      # deterministic (seed 0) -> compute once, reuse for every threshold
    return reg, F


def brady_row(name, reg, F, cutoff):
    """Bradypnea head-swap in the exact tachypnea-report format."""
    y = (reg["gt_rr"].values <= cutoff).astype(int)
    grp = reg["subject_id"].values
    rscore = (cutoff - reg["pred_rr"].values)          # higher = more bradypneic (extractor's own RR)
    if y.sum() < 5 or (y == 0).sum() < 5:
        return None
    prob = cv_prob(F, y, grp)
    aR = roc_auc_score(y, rscore); aC = roc_auc_score(y, prob)
    _, _, pdl = delong_test(y, rscore, prob)
    lo, hi, pw = boot_delta(y, rscore, prob, grp)
    return dict(name=name, n=len(y), pos=100 * y.mean(), aR=aR, aC=aC,
                gain=aC - aR, lo=lo, hi=hi, pw=pw, pdl=pdl)


def sweep_backbone(reg, F, thresholds, min_pos=8, boot=True):
    """gain(thr) for label 1[gt_rr>thr]; reg = pred_rr - thr (rank by pred_rr); cls = ROCKET(label)."""
    grp = reg["subject_id"].values; gt = reg["gt_rr"].values; pred = reg["pred_rr"].values
    rows = []
    for thr in thresholds:
        y = (gt > thr).astype(int)
        npos, nneg = int(y.sum()), int((y == 0).sum())
        if npos < min_pos or nneg < min_pos:          # degenerate imbalance -> skip, flag
            rows.append(dict(thr=thr, prev=y.mean(), npos=npos, nneg=nneg, aR=np.nan, aC=np.nan,
                             gain=np.nan, lo=np.nan, hi=np.nan, pw=np.nan, degenerate=True))
            continue
        rscore = pred - thr
        prob = cv_prob(F, y, grp)
        aR = roc_auc_score(y, rscore); aC = roc_auc_score(y, prob)
        lo = hi = pw = np.nan
        if boot:
            lo, hi, pw = boot_delta(y, rscore, prob, grp, B=1000)
        rows.append(dict(thr=thr, prev=y.mean(), npos=npos, nneg=nneg, aR=aR, aC=aC,
                         gain=aC - aR, lo=lo, hi=hi, pw=pw, degenerate=False))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=30.0, help="bradypnea cutoff (gt_rr <= cutoff)")
    A = ap.parse_args()
    CUT = A.cutoff

    # ---- 0. RR distribution + cutoff justification (from PRISM run; identical GT for all backbones) ----
    ref = load_reg_ext("outputs/paper_runs/pedrr_kfold_learned")
    rr = ref["gt_rr"].values
    print("=== pedrr ground-truth RR distribution (n=%d clips) ===" % len(rr))
    qs = [0, .05, .10, .15, .20, .25, .50, .75, .90, 1.0]
    print("  quantiles:", {f"{int(q*100)}%": round(float(np.quantile(rr, q)), 1) for q in qs})
    print("  mean %.1f  median %.1f  tachypnea(WHO) prevalence %.3f" %
          (rr.mean(), np.median(rr), ref["gt_tachy"].mean()))
    print("  low-cutoff prevalences 1[gt_rr<=c]:",
          {c: round(float((rr <= c).mean()), 3) for c in [24, 26, 28, 30, 32]})
    print("  --> CUTOFF = %g BPM  (prevalence %.3f; ~mirrors the WHO tachypnea base rate %.3f,\n"
          "      so bradypnea vs tachypnea are matched in difficulty/imbalance and directly comparable)"
          % (CUT, (rr <= CUT).mean(), ref["gt_tachy"].mean()))
    print("  clinical note: infant normal RR lower bound is ~30 (<12mo) / ~22 (12-24mo) / ~20 (24-60mo);")
    print("  gt_rr<=%g is frankly low for the <12mo majority but low-normal for toddlers -- a low-RR" % CUT)
    print("  OPERATING POINT to probe the framing, not a validated clinical bradypnea criterion.\n")

    # ---- 1-4. Bradypnea head-swap per backbone (mirror the tachypnea output exactly) ----
    print("=== BRADYPNEA HEAD-SWAP (gt_rr <= %g): regression-then-threshold vs ROCKET-on-signal ===" % CUT)
    print("{:<15}{:>5}{:>6}{:>10}{:>10}{:>18}{:>9}{:>9}".format(
        "backbone", "n", "pos%", "AUROC_reg", "AUROC_cls", "dAUROC[95%CI]", "P(c>r)", "DeLong"))
    matched = {}
    brady = {}
    for name, (sd, run) in SETS.items():
        reg, F = build_matched(sd, run); matched[name] = (reg, F)
        r = brady_row(name, reg, F, CUT)
        if r is None:
            print(f"{name}: degenerate"); continue
        brady[name] = r
        print("{:<15}{:>5}{:>5.0f}%{:>10.3f}{:>10.3f}{:>18}{:>9.3f}{:>9.3g}".format(
            name, r["n"], r["pos"], r["aR"], r["aC"],
            f"{r['gain']:+.3f}[{r['lo']:+.2f},{r['hi']:+.2f}]", r["pw"], r["pdl"]))
    print("  reg = extractor's own pred_rr -> (CUTOFF - pred_rr) margin (fair);  cls = ROCKET on its waveform.\n")

    # ---- 5. Threshold sweep: gain(thr) for label 1[gt_rr>thr] across the observed range ----
    THRS = np.arange(22, 67, 3)     # BPM; 1[gt_rr>thr] gets degenerate below/above -> auto-skipped+flagged
    print("=== THRESHOLD SWEEP  gain = AUROC_cls - AUROC_reg  (label 1[gt_rr>thr]) ===")
    sweeps = {}
    for name, (reg, F) in matched.items():
        sw = sweep_backbone(reg, F, THRS); sweeps[name] = sw
    # tidy table: thr | prev | per-backbone gain
    hdr = "{:>6}{:>7}".format("thr", "prev") + "".join("{:>16}".format(n) for n in matched)
    print(hdr)
    for i, thr in enumerate(THRS):
        prev = sweeps[list(matched)[0]].iloc[i]["prev"]
        cells = []
        for n in matched:
            row = sweeps[n].iloc[i]
            if row["degenerate"]:
                cells.append("{:>16}".format("(imbal.)"))
            else:
                cells.append("{:>16}".format(f"{row['gain']:+.3f}"))
        print("{:>6.0f}{:>7.2f}".format(thr, prev) + "".join(cells))
    # summary: mean gain over non-degenerate thresholds + where largest
    print("\n  per-backbone sweep summary (non-degenerate thresholds only):")
    for n in matched:
        sw = sweeps[n]; ok = sw[~sw["degenerate"]]
        best = ok.loc[ok["gain"].idxmax()]
        frac_pos = float((ok["gain"] > 0).mean())
        print("    %-15s mean_gain=%+.3f  frac(gain>0)=%.2f  max_gain=%+.3f @ thr=%g (prev %.2f)"
              % (n, ok["gain"].mean(), frac_pos, best["gain"], best["thr"], best["prev"]))

    # consistency check: sweep gain near CUTOFF vs the bradypnea head-swap gain (flip symmetry)
    print("\n  [consistency] bradypnea gain(<=%g) vs sweep gain(>%g) should ~match (AUROC flip symmetry):" % (CUT, CUT))
    for n in matched:
        if n not in brady:
            continue
        sw = sweeps[n]; near = sw.iloc[(sw["thr"] - CUT).abs().idxmin()]
        print("    %-15s brady(<=%g)=%+.3f   sweep(>%g@thr=%g)=%+.3f"
              % (n, CUT, brady[n]["gain"], CUT, near["thr"],
                 near["gain"] if not near["degenerate"] else np.nan))

    # ---- 6. Figure: (a) bradypnea reg-vs-cls bars per backbone  (b) sweep gain vs threshold ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                             "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9})
        names = [n for n in matched if n in brady]
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
        # (a) grouped reg vs cls bars
        xb = np.arange(len(names)); w = 0.36
        aR = [brady[n]["aR"] for n in names]; aC = [brady[n]["aC"] for n in names]
        ax[0].bar(xb - w/2, aR, w, label="reg (pred_rr margin)", color="#9ea7b3")
        ax[0].bar(xb + w/2, aC, w, label="cls (ROCKET on signal)", color="#2f6f9f")
        for i, n in enumerate(names):
            g = brady[n]
            ax[0].errorbar(xb[i] + w/2, g["aC"], yerr=[[max(0, -g["lo"])], [max(0, g["hi"])]],
                           fmt="none", ecolor="k", elinewidth=0.8, capsize=2)
            ax[0].text(xb[i], max(g["aR"], g["aC"]) + 0.03, f"{g['gain']:+.3f}", ha="center", fontsize=8)
        ax[0].axhline(0.5, color="k", lw=0.6, ls=":")
        ax[0].set_xticks(xb); ax[0].set_xticklabels([n.replace("(fusion)", "") for n in names], fontsize=8)
        ax[0].set_ylim(0.4, 1.0); ax[0].set_ylabel("AUROC")
        ax[0].set_title("(a) bradypnea (RR$\\leq$%g)" % CUT, fontsize=11)
        ax[0].legend(fontsize=8, frameon=False, loc="lower right")
        # (b) sweep
        colors = {"EfficientPhys": "#c0603a", "TS-CAN": "#4a8a5a", "PRISM(fusion)": "#2f6f9f"}
        for n in matched:
            sw = sweeps[n][~sweeps[n]["degenerate"]]
            ax[1].plot(sw["thr"], sw["gain"], marker="o", ms=3, lw=1.4,
                       color=colors.get(n, None), label=n.replace("(fusion)", ""))
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].axvline(CUT, color="#666", lw=0.8, ls="--")
        ax[1].text(CUT + 0.4, ax[1].get_ylim()[1], "brady\ncutoff", fontsize=8, va="top", color="#666")
        for who in (40, 50, 60):
            ax[1].axvline(who, color="#bbb", lw=0.6, ls=":")
        ax[1].text(50, ax[1].get_ylim()[0], "WHO tachy 40/50/60", fontsize=8, ha="center",
                   va="bottom", color="#999")
        ax[1].set_xlabel("RR threshold (BPM)")
        ax[1].set_ylabel("framing gain ($\\Delta$AUROC)")
        ax[1].set_title("(b) gain is threshold-agnostic", fontsize=11)
        ax[1].legend(fontsize=8, frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig("manuscript/fig_framing_thresholds.png", dpi=600)
        fig.savefig("manuscript/fig_framing_thresholds.pdf", dpi=600)
        print("\n[fig] wrote manuscript/fig_framing_thresholds.{png,pdf}")
    except Exception as e:
        print(f"\n[fig] skipped ({e})")
