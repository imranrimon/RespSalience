"""Tier-2b: reframe the <2mo neonatal screening gap as a DESIGNED safe-escalation pathway.

Established in this project: (i) <2mo tachypnea screening from RGB video collapses -- a universal
RGB-extractor saturation (all extractors read ~30 BPM for a true 60-80), so camera sensitivity <2mo
~= 0; (ii) radar (public thorax-phantom / VEP set) recovers the true neonatal rate ~exactly (MAE ~0)
with zero new hardware; (iii) PRISM carries an age-conditional conformal + disagreement triage that
outputs NORMAL / ALERT / REFER.

This script quantifies the reframing: the camera reliably screens the ages it can (>=2mo), and a
conformal-straddle triage ROUTES the cases it cannot safely screen (esp. <2mo) to REFER = escalation
(radar / contact). Escalated <2mo cases are then recovered downstream by radar (referenced, not scored
here -- we have no radar per-subject data in these CSVs).

Triage semantics (from scripts/recalibrate_triage.py + src/config/age_conditional_conformal.json):
  triage_code 0 = NORMAL_SAFE      -> NORMAL  (retained, screened as not-tachypneic)
  triage_code 1 = TACHYPNEA_ALERT  -> ALERT   (retained, screened as tachypneic)
  triage_code 2 = UNCERTAIN_REVIEW -> REFER   (ESCALATE -- routed out to radar/contact)
Straddle rule on the age-conditional conformal interval [pred-qhat, pred+qhat] vs WHO threshold T:
  hi < T -> NORMAL(0);  lo >= T -> ALERT(1);  else (interval straddles T) -> REFER(2).

The escalation behaviour depends entirely on the conformal width being AGE-CONDITIONAL: the persisted
CSV triage_code uses a single model-level qhat (~7-9 BPM for every band) and does NOT preferentially
escalate neonates. The E7 age-conditional qhat (<2mo ~= 24-26 BPM, from the large neonatal residuals)
makes the <2mo interval wide enough to straddle the WHO cutoff -> REFER. We report BOTH, honestly.

Primary triage = leave-one-fold-out (LOFO) age-conditional conformal (leakage-free; identical
construction to recalibrate_triage.recalibrate). Cross-checked against the deployed config qhat.

CPU-only; reads cached per-fold CSVs. No training, no GPU, no multiprocessing.
Run:  python scripts/escalation_pathway.py
"""
import os, sys, glob, re, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# NOTE: this default is NOT evidence of which run produced the published numbers. The paper's
# escalation figures (79%/48%/34%, 8 tachypneic neonates, n=332) come from the 332-subject main run,
# not from this 297-subject default -- confirmed by appendix tab:escalation (n=332) and tab:retained
# (retained AUROC 0.881). Always pass --run-root explicitly when regenerating for the manuscript.
_DEFAULT_RUN = os.environ.get("ESCALATION_RUN", "pedrr_kfold_learned")
_argv = sys.argv[1:]
if "--run-root" in _argv:
    _DEFAULT_RUN = _argv[_argv.index("--run-root") + 1]
if "--fig-out" in _argv:
    _FIG_NAME = _argv[_argv.index("--fig-out") + 1]
else:
    _FIG_NAME = "fig_escalation"

RUN = _DEFAULT_RUN if os.path.isabs(_DEFAULT_RUN) else os.path.join(ROOT, "outputs", "paper_runs", _DEFAULT_RUN)
CFG = os.path.join(ROOT, "src", "config", "age_conditional_conformal.json")
FIG = os.path.join(ROOT, "manuscript", "fig", _FIG_NAME)
print(f"[escalation_pathway] run-root = {RUN}", flush=True)
ALPHA = 0.20
BANDS = ["<2mo", "2-12mo", "12-60mo"]
STATUS = {0: "NORMAL_SAFE", 1: "TACHYPNEA_ALERT", 2: "UNCERTAIN_REVIEW"}
LABEL = {0: "NORMAL", 1: "ALERT", 2: "REFER(escalate)"}


def band(a):
    return "<2mo" if a < 2 else ("2-12mo" if a < 12 else "12-60mo")


def conf_q(res, alpha=ALPHA):
    """Split-conformal quantile of absolute residuals (same as recalibrate_triage.conf_q)."""
    res = np.asarray(res, float); n = len(res)
    if n == 0:
        return np.nan
    k = min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n)
    return float(np.sort(res)[k - 1])


def straddle(lo, hi, T):
    """0=NORMAL (hi<T), 1=ALERT (lo>=T), 2=REFER (interval straddles the WHO cutoff)."""
    return np.where(hi < T, 0, np.where(lo >= T, 1, 2)).astype(int)


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(RUN, "fold*", "test_eval", "prism_breath_predictions.csv"))):
        d = pd.read_csv(p)
        d["fold"] = int(re.search(r"fold(\d+)", p).group(1))
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["band"] = df["age_months"].apply(band)
    df["res"] = (df["pred_rr_bpm"] - df["gt_rr_bpm"]).abs()
    df["T"] = df["threshold_bpm"].astype(float)
    return df


def aggregate_subject(df):
    """Subject-level aggregation (mean prediction / first-per-subject label). In pedrr each subject
    has exactly one test video, so this is 1:1 with rows -- we run it explicitly for honesty and to
    stay correct if a subject ever contributes multiple videos."""
    g = df.groupby("subject_id", sort=False)
    spec = dict(
        fold=("fold", "first"),
        age_months=("age_months", "first"),
        band=("band", "first"),
        gt_rr_bpm=("gt_rr_bpm", "mean"),
        gt_tachy=("gt_tachy", "max"),          # any tachypneic clip -> subject tachypneic
        pred_rr_bpm=("pred_rr_bpm", "mean"),
        tachy_prob=("tachy_prob", "mean"),
        T=("T", "first"),
    )
    # The persisted triage columns exist only for runs evaluated after selective_triage was wired in.
    # They drive the honesty cross-check, not the primary analysis, so carry them only when present.
    for col, src in (("csv_triage_code", "triage_code"), ("model_triage_code", "model_triage_code")):
        if src in df.columns:
            spec[col] = (src, "first")
    agg = g.agg(**spec).reset_index()
    agg["res"] = (agg["pred_rr_bpm"] - agg["gt_rr_bpm"]).abs()
    return agg


def lofo_ageconditional_triage(df):
    """Leave-one-fold-out age-conditional conformal + straddle triage (leakage-free)."""
    folds = sorted(df["fold"].unique())
    q = np.empty(len(df))
    for k in folds:
        for b in df["band"].unique():
            qb = conf_q(df.loc[(df.fold != k) & (df.band == b), "res"].values)
            q[((df.fold == k) & (df.band == b)).values] = qb
    lo = df["pred_rr_bpm"].values - q
    hi = df["pred_rr_bpm"].values + q
    return straddle(lo, hi, df["T"].values), q


def deploy_ageconditional_triage(df):
    """Deployed persisted config qhat (in-sample, cross-check only)."""
    cfg = json.load(open(CFG)); qb = cfg["qhat_by_band"]
    q = np.array([qb[b] for b in df["band"]])
    lo = df["pred_rr_bpm"].values - q
    hi = df["pred_rr_bpm"].values + q
    return straddle(lo, hi, df["T"].values), q


def auroc(y, s):
    y = np.asarray(y, int)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, np.asarray(s, float)))


def screen_metrics(sub, triage_col):
    """Screening quality on the RETAINED (non-escalated) subset. The screening call on a retained
    case IS the triage code: 1=ALERT=predicted-positive, 0=NORMAL=predicted-negative. AUROC uses
    tachy_prob over retained cases (threshold-independent)."""
    ret = sub[sub[triage_col] != 2]
    y = ret["gt_tachy"].values.astype(int)
    dec = (ret[triage_col].values == 1).astype(int)          # ALERT = positive screen
    tp = int(((dec == 1) & (y == 1)).sum()); fn = int(((dec == 0) & (y == 1)).sum())
    tn = int(((dec == 0) & (y == 0)).sum()); fp = int(((dec == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(n_ret=len(ret), n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()),
                sens=sens, spec=spec, auroc=auroc(y, ret["tachy_prob"].values),
                tp=tp, fn=fn, tn=tn, fp=fp)


def refer_rate(sub, triage_col):
    return float((sub[triage_col] == 2).mean())


def hline():
    print("-" * 78)


def report(sub, triage_col, title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    # (1) escalation (REFER) rate by band -------------------------------------------------
    print("\n(1) ESCALATION (REFER) RATE BY AGE BAND  [system routes rather than screens]")
    print(f"    {'band':8s} {'n':>4s} {'REFER%':>8s} {'NORMAL%':>8s} {'ALERT%':>8s}  {'#tachy':>6s}")
    for b in BANDS + ["overall"]:
        s = sub if b == "overall" else sub[sub.band == b]
        n = len(s)
        r = 100 * (s[triage_col] == 2).mean()
        nm = 100 * (s[triage_col] == 0).mean()
        al = 100 * (s[triage_col] == 1).mean()
        print(f"    {b:8s} {n:>4d} {r:>8.1f} {nm:>8.1f} {al:>8.1f}  {int(s.gt_tachy.sum()):>6d}")

    # (2) retained-case screening quality by band -----------------------------------------
    print("\n(2) SELECTIVE SCREENING ON RETAINED (non-escalated) CASES")
    print(f"    {'band':8s} {'nRet':>5s} {'pos':>4s} {'Sens':>7s} {'Spec':>7s} {'AUROC':>7s}")
    band_screen = {}
    for b in BANDS + ["overall"]:
        s = sub if b == "overall" else sub[sub.band == b]
        m = screen_metrics(s, triage_col); band_screen[b] = m
        def f(x): return f"{x:.3f}" if x == x else "  n/a"
        print(f"    {b:8s} {m['n_ret']:>5d} {m['n_pos']:>4d} {f(m['sens']):>7s} {f(m['spec']):>7s} {f(m['auroc']):>7s}")

    # (3) the avoided failure -- <2mo tachypnea ------------------------------------------
    n2 = sub[sub.band == "<2mo"]
    tpos = n2[n2.gt_tachy == 1]
    # WITHOUT escalation: every <2mo screened by camera; decision = pred_rr >= WHO threshold
    cam_pos = (tpos["pred_rr_bpm"].values >= tpos["T"].values).astype(int)
    sens_noesc = cam_pos.mean() if len(tpos) else np.nan
    fn_noesc = int((cam_pos == 0).sum())                       # falsely cleared neonates
    # WITH escalation: a tachy neonate is dangerously cleared only if triage==0 (NORMAL)
    fn_esc = int((tpos[triage_col] == 0).sum())
    caught_alert = int((tpos[triage_col] == 1).sum())          # directly flagged by camera
    escalated = int((tpos[triage_col] == 2).sum())             # routed to radar
    rescued = fn_noesc - fn_esc
    print("\n(3) AVOIDED FAILURE -- tachypneic <2mo neonates")
    print(f"    tachypneic <2mo neonates: {len(tpos)}  (true RR range "
          f"{tpos.gt_rr_bpm.min():.0f}-{tpos.gt_rr_bpm.max():.0f}, pred saturates "
          f"{tpos.pred_rr_bpm.min():.0f}-{tpos.pred_rr_bpm.max():.0f})")
    print(f"    WITHOUT escalation (camera screens all): Sens={sens_noesc:.3f}  "
          f"-> {fn_noesc}/{len(tpos)} falsely cleared as NORMAL")
    print(f"    WITH escalation: {caught_alert} caught as ALERT, {escalated} routed to REFER "
          f"(radar), {fn_esc} still auto-cleared as NORMAL")
    print(f"    ==> escalation RESCUES {rescued}/{len(tpos)} tachypneic neonates from a false 'normal'")

    # (4) coverage / benefit trade-off ----------------------------------------------------
    ov = band_screen["overall"]
    cov = 1 - refer_rate(sub, triage_col)
    # full-coverage baseline: screen everyone by camera decision (pred>=T), no escalation
    y = sub["gt_tachy"].values.astype(int); cam = (sub["pred_rr_bpm"].values >= sub["T"].values).astype(int)
    sens_full = cam[y == 1].mean(); spec_full = (cam[y == 0] == 0).mean()
    print("\n(4) COVERAGE / BENEFIT TRADE-OFF")
    print(f"    full coverage (screen 100%, no escalation): Sens={sens_full:.3f}  Spec={spec_full:.3f}")
    print(f"    with escalation: coverage={100*cov:.0f}% screened ({100*(1-cov):.0f}% escalated), "
          f"retained Sens={ov['sens']:.3f}  Spec={ov['spec']:.3f}  AUROC={ov['auroc']:.3f}")
    print(f"    -> a {100*(1-cov):.0f}% coverage cost buys retained-Sens {sens_full:.3f}->{ov['sens']:.3f}")
    return band_screen, dict(sens_noesc=sens_noesc, fn_noesc=fn_noesc, fn_esc=fn_esc,
                             rescued=rescued, n_tpos=len(tpos), coverage=cov,
                             sens_full=sens_full, spec_full=spec_full)


def camera_all_sens(sub, b):
    """Sensitivity if the CAMERA screened every case in band b (no escalation): decision pred>=T."""
    s = sub[sub.band == b]; pos = s[s.gt_tachy == 1]
    if len(pos) == 0:
        return np.nan
    return float((pos["pred_rr_bpm"].values >= pos["T"].values).mean())


def make_figure(sub, triage_col, band_screen, avoided):
    """Clean two-panel figure, authored at full text width so nothing is shrunk:
    (a) escalation (REFER) rate by age band -- younger bands are routed out more;
    (b) fate of the tachypneic neonates -- camera-only vs with-escalation false-clears (8 -> 2),
    the rescued cases stacked on top. Large fonts, minimal annotation."""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 11.5, "axes.labelsize": 11,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10})
    C_REFER, C_OLD, C_MISS, C_RESCUE = "#C44E52", "#DD8452", "#C44E52", "#55A868"
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    x = np.arange(len(BANDS))

    # (a) escalation (REFER) rate by band ------------------------------------------------
    refer = [100 * (sub[sub.band == b][triage_col] == 2).mean() for b in BANDS]
    ax[0].bar(x, refer, width=0.62, color=[C_REFER if b == "<2mo" else C_OLD for b in BANDS],
              edgecolor="k", linewidth=0.6)
    for xi, v in zip(x, refer):
        ax[0].text(xi, v + 2.5, f"{v:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax[0].set_xticks(x); ax[0].set_xticklabels(BANDS)
    ax[0].set_ylabel("Escalated (REFER) [%]")
    ax[0].set_ylim(0, 100)
    ax[0].set_title("(a) Escalation rate by age band")

    # (b) fate of the tachypneic neonates: camera-only vs with escalation ----------------
    miss0, miss1 = avoided["fn_noesc"], avoided["fn_esc"]   # false-clears without / with escalation
    rescued, ntp = avoided["rescued"], avoided["n_tpos"]
    xb = np.array([0.0, 1.0])
    ax[1].bar(xb[0], miss0, width=0.55, color=C_MISS, edgecolor="k", linewidth=0.6)
    ax[1].bar(xb[1], miss1, width=0.55, color=C_MISS, edgecolor="k", linewidth=0.6,
              label="falsely cleared")
    ax[1].bar(xb[1], rescued, width=0.55, bottom=miss1, color=C_RESCUE, edgecolor="k", linewidth=0.6,
              label="rescued to radar")
    ax[1].text(xb[0], miss0 + 0.12, f"{miss0}", ha="center", va="bottom", fontweight="bold")
    ax[1].text(xb[1], miss1 / 2, f"{miss1}", ha="center", va="center", color="white", fontweight="bold")
    ax[1].text(xb[1], miss1 + rescued / 2, f"{rescued}", ha="center", va="center", color="white",
               fontweight="bold")
    ax[1].set_xticks(xb); ax[1].set_xticklabels(["camera\nonly", "with\nescalation"])
    # Short label: these are counts, so the axis must tick in whole neonates, and n belongs in the
    # caption rather than in a y-label long enough to run into panel (a)'s title.
    ax[1].set_ylabel("Tachypneic neonates")
    ax[1].set_ylim(0, ntp + 1.2)
    ax[1].set_yticks(np.arange(0, ntp + 1, 2))
    ax[1].set_title(f"(b) Missed neonates cut {miss0}→{miss1} (n={ntp})")
    # Both bars reach nearly the full height, so an in-axes legend sits on the data; put it under
    # the panel, below the two-line tick labels, where it cannot collide with either.
    ax[1].legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                 ncol=2, handlelength=1.3, columnspacing=1.4, borderpad=0.2)

    fig.tight_layout(w_pad=3.0)
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(FIG + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG + ".pdf", dpi=600, bbox_inches="tight")
    print(f"\nwrote {FIG}.png and {FIG}.pdf")


if __name__ == "__main__":
    df = load()
    print(f"loaded n={len(df)} rows, folds={sorted(df.fold.unique())}")

    # --- triage semantics (as requested: print distinct triage_code/status + mapping) ---
    HAS_CSV_TRIAGE = "triage_code" in df.columns
    if HAS_CSV_TRIAGE:
        print("\nTRIAGE SEMANTICS (persisted CSV columns):")
        print("  distinct triage_code   :", sorted(df["triage_code"].unique()))
        print("  distinct triage_status :", sorted(df["triage_status"].unique()))
        print("  code<->status map:")
        for c in sorted(df["triage_code"].unique()):
            st = df.loc[df.triage_code == c, "triage_status"].iloc[0]
            print(f"    code {c} = {st:16s} -> {LABEL[c]}")
        print("  (REFER = escalate = routed out to radar/contact)")
        print("  persisted qhat_source :", df["qhat_source"].unique().tolist(),
              "-> single model-level width, NOT age-conditional")
    else:
        print("\nTRIAGE SEMANTICS: this run predates the persisted triage columns; the LOFO primary "
              "and the deployed-config cross-check are both computed here, and the persisted-triage "
              "honesty check is skipped.")

    sub = aggregate_subject(df)
    print(f"\nsubject-level: n_subjects={len(sub)} (== n_rows {len(df)}: one test video per subject)")

    # attach the three triage policies
    sub["lofo"], qlofo = lofo_ageconditional_triage(sub)
    sub["deploy"], qdep = deploy_ageconditional_triage(sub)
    print("\nage-conditional qhat (mean BPM per band):")
    print(f"    {'band':8s} {'LOFO':>7s} {'deploy_cfg':>11s} {'model-level(CSV)':>17s}")
    for b in BANDS:
        ql = qlofo[(sub.band == b).values].mean()
        qd = qdep[(sub.band == b).values].mean()
        qm = df.loc[df.band == b, "qhat"].mean()
        print(f"    {b:8s} {ql:>7.1f} {qd:>11.1f} {qm:>17.1f}")

    # PRIMARY analysis: leakage-free LOFO age-conditional straddle triage
    band_screen, avoided = report(sub, "lofo",
                                  "PRIMARY: LOFO age-conditional conformal straddle triage (leakage-free)")

    # cross-check: deployed config qhat
    report(sub, "deploy", "CROSS-CHECK: deployed config qhat (in-sample)")

    # honesty check: the PERSISTED model-level triage does it escalate <2mo?
    if HAS_CSV_TRIAGE:
        report(sub, "csv_triage_code",
               "HONESTY CHECK: persisted model-level triage_code (single qhat ~8 BPM, NOT age-conditional)")

    make_figure(sub, "lofo", band_screen, avoided)
