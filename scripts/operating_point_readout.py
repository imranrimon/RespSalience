"""Recover the neonatal band by fixing the read-out, not the sensor.

`scripts/acquisition_limit_probe.py` established that neonatal breathing survives
extraction -- the true rate is a detectable spectral peak in 100% of fast signals
-- and that the model ranks it acceptably (tachypnea AUROC 0.746 for <2mo). What
destroys neonatal detection is the *decision rule*: thresholding a regressed rate
that has been shrunk toward the training mean (bias -15.3 breaths/min at
RR >= 60) means the prediction almost never crosses the WHO cutoff.

The fix needs no new sensor, no retraining, and no new modality: read the
distributional mass the model already produces, and set the operating point per
age band. This script does that with the threshold chosen **only on calibration
data**, since picking it on the test set is what makes such a result worthless.

Comparators, all on identical scores from the same trained model:

  regression      pred_rr >= tau_age            (the rule in the paper)
  global-0.5      tachy_prob >= 0.5             (mass, but one global cut point)
  per-band        tachy_prob >= t_band          (mass, cut point per age band)

Run from the repository root:

    python scripts/operating_point_readout.py --run-root outputs/paper_runs/pedrr_kfold_v3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cost_aware_deferral import (  # noqa: E402
    ALL_BANDS,
    BANDS,
    calibration_for_fold,
    load_by_fold,
)


def select_threshold(cal: pd.DataFrame, target_sens: float,
                     min_positives: int = 5) -> Tuple[float, bool]:
    """Largest tachy_prob cut point whose calibration sensitivity clears target.

    Sensitivity is non-increasing in the threshold, so the largest admissible cut
    point is the one that meets the sensitivity floor at the best specificity.
    Returns (threshold, selected) -- selected is False when the band has too few
    positives to choose responsibly, in which case the caller should fall back.
    """
    y = cal["gt_tachy"].to_numpy(dtype=int)
    s = cal["tachy_prob"].to_numpy(dtype=float)
    n_pos = int((y == 1).sum())
    if n_pos < min_positives:
        return float("nan"), False
    best = 0.0
    for t in np.unique(np.concatenate([np.linspace(0, 1, 201), s])):
        sens = float(((s >= t) & (y == 1)).sum() / n_pos)
        if sens >= target_sens:
            best = float(t)
        else:
            break
    return best, True


def shrink_thresholds(per_band: Dict[str, float], pooled: float,
                      n_pos: Dict[str, int], k: float) -> Dict[str, float]:
    """Partial pooling of the per-band cut points toward the pooled one.

    A band with few positives cannot estimate its own operating point reliably,
    but it is not exchangeable with the others either. Shrinking by
    w = n_pos / (n_pos + k) lets a band earn its own threshold in proportion to
    the evidence it carries: k is the number of positives at which a band is
    weighted equally between its own estimate and the pooled one.
    """
    out = {}
    for b, t in per_band.items():
        if not np.isfinite(t):
            out[b] = pooled
            continue
        w = n_pos.get(b, 0) / (n_pos.get(b, 0) + k) if k > 0 else 1.0
        out[b] = float(w * t + (1 - w) * pooled)
    return out


def decisions(df: pd.DataFrame, rule: str, thresholds: Optional[Dict[str, float]] = None,
              global_t: float = 0.5) -> np.ndarray:
    if rule == "regression":
        return (df["pred_rr_bpm"].to_numpy(dtype=float)
                >= df["threshold_bpm"].to_numpy(dtype=float)).astype(int)
    if rule == "global":
        return (df["tachy_prob"].to_numpy(dtype=float) >= global_t).astype(int)
    t = np.array([(thresholds or {}).get(b, global_t) for b in df["band"]], dtype=float)
    return (df["tachy_prob"].to_numpy(dtype=float) >= t).astype(int)


def sens_spec(y: np.ndarray, d: np.ndarray) -> Tuple[float, float, int, int]:
    tp = int(((d == 1) & (y == 1)).sum()); fn = int(((d == 0) & (y == 1)).sum())
    tn = int(((d == 0) & (y == 0)).sum()); fp = int(((d == 1) & (y == 0)).sum())
    return (tp / (tp + fn) if tp + fn else float("nan"),
            tn / (tn + fp) if tn + fp else float("nan"), tp + fn, tn + fp)


def boot_ci(y: np.ndarray, d: np.ndarray, rng: np.random.Generator,
            n_boot: int = 4000) -> Dict[str, Tuple[float, float]]:
    idx = np.arange(y.size)
    se, sp = [], []
    for _ in range(n_boot):
        k = rng.choice(idx, size=idx.size, replace=True)
        a, b, npos, nneg = sens_spec(y[k], d[k])
        if npos:
            se.append(a)
        if nneg:
            sp.append(b)
    def ci(v):
        v = np.asarray(v, dtype=float); v = v[np.isfinite(v)]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v.size else (np.nan, np.nan)
    return {"sens": ci(se), "spec": ci(sp)}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-band operating point on the distributional mass.")
    p.add_argument("--run-root", default="outputs/paper_runs/pedrr_kfold_v3")
    p.add_argument("--out-dir", default="outputs/operating_point")
    p.add_argument("--calibration", choices=["oof", "val"], default="oof")
    p.add_argument("--target-sens", type=float, default=0.70)
    p.add_argument("--sweep", default="0.60,0.70,0.80,0.90")
    p.add_argument("--shrinkage-k", type=float, default=10.0,
                   help="Positives at which a band is weighted equally between its own threshold "
                        "and the pooled one; 0 disables shrinkage")
    p.add_argument("--shrinkage-sweep", default="0,5,10,20,40")
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-age-fix", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    age_fix = not args.no_age_fix

    test_by_fold = load_by_fold(run_root, "test_eval", age_fix=age_fix)
    # Out-of-fold calibration reads the other folds' test predictions, so a run
    # without persisted validation splits is still usable.
    try:
        val_by_fold = load_by_fold(run_root, "val_eval", age_fix=age_fix)
    except FileNotFoundError:
        if args.calibration == "val":
            raise SystemExit(f"{run_root} has no val_eval splits; use --calibration oof.")
        val_by_fold = {}

    print(f"Run root       : {run_root}")
    print(f"Calibration    : {args.calibration} (thresholds NEVER touch the evaluated fold)")
    print(f"Target sens    : {args.target_sens:.2f}\n")

    def run_at(target: float) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray], List[Dict]]:
        frames, per_rule, picks = [], {"regression": [], "global": [], "per-band": [], "shrunk": []}, []
        for fold in sorted(test_by_fold):
            cal = calibration_for_fold(fold, val_by_fold, test_by_fold, args.calibration)
            thr, ok = {}, {}
            pooled_t, _ = select_threshold(cal, target)
            for b in BANDS:
                sub = cal[cal["band"] == b]
                t, sel = select_threshold(sub, target)
                thr[b] = t if sel else pooled_t          # fall back rather than guess
                ok[b] = sel
                picks.append({"fold": fold, "band": b, "threshold": thr[b],
                              "band_specific": sel, "cal_n": len(sub),
                              "cal_pos": int(sub["gt_tachy"].sum())})
            npos = {b: int(cal[cal["band"] == b]["gt_tachy"].sum()) for b in BANDS}
            thr_shrunk = shrink_thresholds(thr, pooled_t, npos, float(args.shrinkage_k))
            test = test_by_fold[fold]
            frames.append(test)
            for rule in per_rule:
                per_rule[rule].append(
                    decisions(test, "per-band" if rule == "shrunk" else rule,
                              thr_shrunk if rule == "shrunk" else thr))
        pooled = pd.concat(frames, ignore_index=True)
        return pooled, pooled["gt_tachy"].to_numpy(dtype=int), \
               {k: np.concatenate(v) for k, v in per_rule.items()}, picks

    pooled, y, dec, picks = run_at(float(args.target_sens))
    bands = pooled["band"].to_numpy()

    rows = []
    for rule, d in dec.items():
        for band in ALL_BANDS:
            mask = np.ones(y.size, dtype=bool) if band == "overall" else (bands == band)
            if not mask.any():
                continue
            se, sp, npos, nneg = sens_spec(y[mask], d[mask])
            ci = boot_ci(y[mask], d[mask], rng, int(args.n_boot))
            rows.append({"rule": rule, "band": band, "n": int(mask.sum()), "pos": npos,
                         "sens": se, "sens_lo": ci["sens"][0], "sens_hi": ci["sens"][1],
                         "spec": sp, "spec_lo": ci["spec"][0], "spec_hi": ci["spec"][1]})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / f"operating_point_{run_root.name}.csv", index=False)
    pd.DataFrame(picks).to_csv(out_dir / f"selected_thresholds_{run_root.name}.csv", index=False)

    print("Sensitivity / specificity with subject bootstrap 95% CIs "
          "(identical model and scores; only the read-out differs)")
    for band in ALL_BANDS:
        sub = res[res.band == band]
        if sub.empty:
            continue
        r0 = sub.iloc[0]
        print(f"\n  {band}  (n={int(r0['n'])}, positives={int(r0['pos'])})")
        for _, r in sub.iterrows():
            print(f"    {r['rule']:<12} sens {r['sens']:.3f} [{r['sens_lo']:.3f},{r['sens_hi']:.3f}]"
                  f"   spec {r['spec']:.3f} [{r['spec_lo']:.3f},{r['spec_hi']:.3f}]")

    print("\nPaired McNemar vs the regression rule (discordant true positives)")
    for band in ALL_BANDS:
        mask = np.ones(y.size, dtype=bool) if band == "overall" else (bands == band)
        m, p = dec["regression"][mask], dec["per-band"][mask]
        yy = y[mask]
        b = int(((yy == 1) & (m == 0) & (p == 1)).sum())
        c = int(((yy == 1) & (m == 1) & (p == 0)).sum())
        pv = binomtest(b, b + c, 0.5).pvalue if b + c else float("nan")
        print(f"  {band:<9} recovered {b}, lost {c}"
              + (f", p={pv:.4f}" if np.isfinite(pv) else ", n/a"))

    print("\nSelected thresholds (calibration-only, per fold)")
    pk = pd.DataFrame(picks)
    g = pk.groupby("band").agg(mean_t=("threshold", "mean"), sd_t=("threshold", "std"),
                               band_specific=("band_specific", "mean"),
                               cal_pos=("cal_pos", "mean")).reindex(BANDS)
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nShrinkage sweep: does partial pooling stabilise the neonatal operating point?")
    print(f"  {'k':>5}{'<2mo sens':>11}{'<2mo spec':>11}{'overall sens':>14}{'overall spec':>14}"
          f"{'<2mo thr sd':>13}")
    base_k = args.shrinkage_k
    for kk in [float(x) for x in str(args.shrinkage_sweep).split(",") if x.strip()]:
        args.shrinkage_k = kk
        pk, yk, deck, pick_k = run_at(float(args.target_sens))
        bk = pk["band"].to_numpy()
        so, po, _, _ = sens_spec(yk, deck["shrunk"])
        mk = bk == "<2mo"
        sn, pn, _, _ = sens_spec(yk[mk], deck["shrunk"][mk])
        pkdf = pd.DataFrame(pick_k)
        npos = {b: pkdf[pkdf.band == b]["cal_pos"].mean() for b in BANDS}
        pooled_mean = pkdf.groupby("fold")["threshold"].mean()
        raw = pkdf[pkdf.band == "<2mo"]["threshold"].to_numpy(dtype=float)
        w = npos["<2mo"] / (npos["<2mo"] + kk) if kk > 0 else 1.0
        sd = float(np.std(w * raw + (1 - w) * pooled_mean.to_numpy(), ddof=1))
        print(f"  {kk:>5.0f}{sn:>11.3f}{pn:>11.3f}{so:>14.3f}{po:>14.3f}{sd:>13.3f}")
    args.shrinkage_k = base_k

    print("\nSensitivity-target sweep (overall / <2mo), all leakage-safe")
    print(f"  {'target':>7}{'overall sens':>14}{'overall spec':>14}{'<2mo sens':>11}{'<2mo spec':>11}")
    sweep_rows = []
    for t in [float(x) for x in str(args.sweep).split(",") if x.strip()]:
        pooled_t, yt, dect, _ = run_at(t)
        bt = pooled_t["band"].to_numpy()
        so, po, _, _ = sens_spec(yt, dect["per-band"])
        mask = bt == "<2mo"
        sn, pn, _, _ = sens_spec(yt[mask], dect["per-band"][mask])
        print(f"  {t:>7.2f}{so:>14.3f}{po:>14.3f}{sn:>11.3f}{pn:>11.3f}")
        sweep_rows.append({"target_sens": t, "overall_sens": so, "overall_spec": po,
                           "neonatal_sens": sn, "neonatal_spec": pn})
    pd.DataFrame(sweep_rows).to_csv(out_dir / f"sens_sweep_{run_root.name}.csv", index=False)

    print(f"\nWrote {out_dir}/operating_point_{run_root.name}.csv (+ thresholds, sweep)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
