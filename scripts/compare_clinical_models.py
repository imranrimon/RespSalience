"""Paired two-model clinical comparison with proper significance tests.

Answers "is model A significantly better than model B on the SAME test cases?"
using tests that respect the pairing (same infants scored by both models):
  * DeLong test for the paired AUROC difference (correlated ROC curves)
  * McNemar exact test for the tachypnea decision (discordant pairs)
  * Subject-level bootstrap 95% CI on the paired difference of MAE/AUROC/Sens/Spec

Seed-averages each model to one row per clip, then aligns the two models on the
intersection of signal_path so every compared case is the same infant/clip.

Usage:
  python scripts/compare_clinical_models.py \
      --model-a outputs/paper_runs/pedrr_retune_fe0 --label-a fe0_learned \
      --model-b outputs/paper_runs/pedrr_v3        --label-b v3_gated
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from sklearn.metrics import roc_auc_score

AGE_BANDS = [("<2mo", 0, 2, 60), ("2-12mo", 2, 12, 50), ("12-60mo", 12, 60, 40)]


def _age_band(a):
    for name, lo, hi, _ in AGE_BANDS:
        if lo <= a < hi:
            return name
    return "12-60mo"


def load_seed_avg(run_root: str) -> pd.DataFrame:
    root = Path(run_root)
    # matches both single-split (seed*/) and k-fold CV (fold*/) layouts
    paths = sorted(root.glob("*/test_eval/prism_breath_predictions.csv"))
    if not paths:
        raise FileNotFoundError(f"no */test_eval CSVs under {root}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["clip_key"] = df["subject_id"].astype(str) + "_" + df["video_id"].astype(str)
    g = df.groupby("clip_key")
    first = ["subject_id", "video_id", "gt_rr_bpm", "threshold_bpm", "gt_tachy", "age_months"]
    d = g[first].first().reset_index()
    for c in ["pred_rr_bpm", "tachy_prob"]:
        d[c] = g[c].mean().values
    d["age_band"] = d["age_months"].apply(_age_band)
    d["pred_tachy"] = (d["pred_rr_bpm"] >= d["threshold_bpm"]).astype(int)
    return d


# ---------- fast DeLong (Sun & Xu 2014) for two paired ROC curves ----------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_test(y, sa, sb):
    """Two-sided p-value for AUC(a) - AUC(b), paired (same y). Returns (auc_a, auc_b, p)."""
    y = np.asarray(y, int)
    order = np.argsort(-y, kind="mergesort")   # positives (y=1) first
    preds = np.vstack([np.asarray(sa, float)[order], np.asarray(sb, float)[order]])
    m = int(y.sum()); n = len(y) - m
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")
    pos, neg = preds[:, :m], preds[:, m:]
    k = 2
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        p = float("nan")
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = float(2 * norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), p


def mcnemar_exact(y, pa, pb):
    """Exact McNemar on the tachypnea DECISION correctness of A vs B (paired)."""
    ca = (pa == y); cb = (pb == y)
    b = int((ca & ~cb).sum())   # A right, B wrong
    c = int((~ca & cb).sum())   # A wrong, B right
    if b + c == 0:
        return b, c, 1.0
    p = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    return b, c, p


def metrics(d):
    gt = d["gt_rr_bpm"].to_numpy(float); pr = d["pred_rr_bpm"].to_numpy(float)
    y = d["gt_tachy"].to_numpy(int); yh = d["pred_tachy"].to_numpy(int)
    tp = int(((y == 1) & (yh == 1)).sum()); fn = int(((y == 1) & (yh == 0)).sum())
    tn = int(((y == 0) & (yh == 0)).sum()); fp = int(((y == 0) & (yh == 1)).sum())
    au = roc_auc_score(y, d["tachy_prob"]) if 0 < y.sum() < len(y) else float("nan")
    return {"mae": float(np.abs(pr - gt).mean()),
            "auroc": float(au),
            "sens": tp / (tp + fn) if tp + fn else float("nan"),
            "spec": tn / (tn + fp) if tn + fp else float("nan")}


def paired_boot_diff(da, db, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    subs = da["subject_id"].to_numpy(); uniq = np.unique(subs)
    s2i = {u: np.where(subs == u)[0] for u in uniq}
    keys = ["mae", "auroc", "sens", "spec"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([s2i[u] for u in pick])
        ma, mb = metrics(da.iloc[idx]), metrics(db.iloc[idx])
        for k in keys:
            draws[k].append(ma[k] - mb[k])
    out = {}
    for k in keys:
        arr = np.asarray(draws[k], float); arr = arr[np.isfinite(arr)]
        out[k] = (float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True); ap.add_argument("--label-a", default="A")
    ap.add_argument("--model-b", required=True); ap.add_argument("--label-b", default="B")
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()

    da = load_seed_avg(args.model_a); db = load_seed_avg(args.model_b)
    common = sorted(set(da.clip_key) & set(db.clip_key))
    da = da[da.clip_key.isin(common)].sort_values("clip_key").reset_index(drop=True)
    db = db[db.clip_key.isin(common)].sort_values("clip_key").reset_index(drop=True)
    assert (da.clip_key.values == db.clip_key.values).all()
    # drop clips where either model's RR prediction is NaN (extractor failures)
    keep = da.pred_rr_bpm.notna().values & db.pred_rr_bpm.notna().values
    if (~keep).sum():
        print(f"[warn] dropping {int((~keep).sum())} clips with NaN prediction")
        da, db = da[keep].reset_index(drop=True), db[keep].reset_index(drop=True)
    # use model-A's labels as the shared reference (minor age-source discrepancies otherwise)
    n_mismatch = int((da.gt_tachy.values != db.gt_tachy.values).sum())
    if n_mismatch:
        print(f"[warn] {n_mismatch} label mismatches (age-source diff); using model-A gt_tachy as reference")
    db = db.copy(); db["gt_tachy"] = da["gt_tachy"].values; db["threshold_bpm"] = da["threshold_bpm"].values
    y = da.gt_tachy.to_numpy(int)
    print(f"[align] {len(da)} paired clips, {da.subject_id.nunique()} subjects, n_pos={int(y.sum())}")

    ma, mb = metrics(da), metrics(db)
    ci = paired_boot_diff(da, db, args.n_boot)
    print(f"\n{'metric':<8}{args.label_a:>14}{args.label_b:>14}   diff(A-B) [95% CI]")
    for k, lbl in [("mae", "MAE"), ("auroc", "AUROC"), ("sens", "Sens"), ("spec", "Spec")]:
        diff = ma[k] - mb[k]
        sig = "" if ci[k][0] <= 0 <= ci[k][1] else "  *sig"
        print(f"{lbl:<8}{ma[k]:>14.3f}{mb[k]:>14.3f}   {diff:+.3f} [{ci[k][0]:+.3f},{ci[k][1]:+.3f}]{sig}")

    aa, ab, pdel = delong_test(y, da.tachy_prob, db.tachy_prob)
    b, c, pmc = mcnemar_exact(y, da.pred_tachy.to_numpy(int), db.pred_tachy.to_numpy(int))
    print(f"\n[DeLong]  AUROC {args.label_a}={aa:.3f} vs {args.label_b}={ab:.3f}  ->  p = {pdel:.3f}")
    print(f"[McNemar] tachypnea decision: {args.label_a}-right/{args.label_b}-wrong={b}, "
          f"{args.label_a}-wrong/{args.label_b}-right={c}  ->  exact p = {pmc:.3f}")
    print("\nNOTE: with n_pos~17 these tests are underpowered; non-significant p != equivalence.")


if __name__ == "__main__":
    raise SystemExit(main())
