"""What does the stronger guarantee cost, and can we afford it?

Two caveats hang over the certified triage layer:

  (i)  conformal risk control buys E[risk] <= alpha, an expectation over
       deployments, not a per-cohort high-probability statement;
  (ii) the Clopper-Pearson bound that *would* give the high-probability
       statement certifies almost nothing, because at ~20 positives per age band
       it bounds even a zero-error rate at ~0.17.

Caveat (ii) is an artefact of *where* the bound is applied, not of the bound.
The guarantee a screening programme wants is population-level -- "at most alpha
of screened children are auto-cleared while tachypneic" -- so the risk should be
bounded once over the whole calibration set, using all ~68 positives, rather than
three times over ~20 each. Age-conditioning is preserved by indexing the three
band thresholds with a single shared quantile level: the decision rule stays
per-band, but the hypothesis being tested is one-dimensional, so there is no
multiplicity penalty and the bound sees every positive.

That moves the zero-error Clopper-Pearson bound from 0.17 to 0.05 and makes the
high-probability guarantee affordable. This script prices both guarantee types
against each other so the paper can state which one it claims and what it cost.

Run from the repository root:

    python scripts/guarantee_strength_tradeoff.py --run-root outputs/paper_runs/pedrr_kfold_v3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cost_aware_deferral import (  # noqa: E402
    ALL_BANDS,
    BANDS,
    CostModel,
    PolicyThreshold,
    apply_policy,
    calibration_for_fold,
    clopper_pearson_upper,
    load_by_fold,
    metrics_for_triage,
)

NORMAL, ALERT, REFER = 0, 1, 2


def band_thresholds(cal: pd.DataFrame, q_normal: float, q_alert: float
                    ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-band thresholds indexed by shared quantile levels.

    The rule stays age-conditional -- each band gets its own cut point -- but
    both cut points move together along one scalar, so the family of hypotheses
    is one-dimensional.
    """
    lo, hi = {}, {}
    for b in BANDS:
        s = cal.loc[cal["band"] == b, "tachy_prob"].to_numpy(dtype=float)
        if s.size == 0:
            lo[b], hi[b] = -1.0, 2.0
            continue
        lo[b] = float(np.quantile(s, q_normal)) if q_normal > 0 else -1.0
        hi[b] = float(np.quantile(s, q_alert)) if q_alert < 1 else 2.0
    return lo, hi


def pooled_counts(cal: pd.DataFrame, lo: Dict[str, float], hi: Dict[str, float]
                  ) -> Tuple[int, int, int, int]:
    """(cleared positives, alerted negatives, n_pos, n_neg) pooled over bands."""
    s = cal["tachy_prob"].to_numpy(dtype=float)
    y = cal["gt_tachy"].to_numpy(dtype=int)
    b = cal["band"].to_numpy()
    a = np.array([lo.get(x, -1.0) for x in b], dtype=float)
    c = np.array([hi.get(x, 2.0) for x in b], dtype=float)
    fn = int(((s <= a) & (y == 1)).sum())
    fp = int(((s >= c) & (y == 0)).sum())
    return fn, fp, int((y == 1).sum()), int((y == 0).sum())


def rate_bound(k: int, n: int, delta: float, bound: str) -> float:
    """Upper bound on a class-conditional rate.

    cp  : P(rate <= bound) >= 1 - delta         (high-probability, per cohort)
    crc : conformal-risk-control pseudo-count   (expectation over deployments)
    """
    if n <= 0:
        return 1.0
    if bound == "cp":
        return float(clopper_pearson_upper(k, n, delta))
    return float((k + 1) / (n + 1))


def fit_pooled(
    cal: pd.DataFrame, alpha_missed: float, alpha_false_alert: float,
    pi_lo: float, pi_hi: float, delta: float, bound: str,
    pooling: str, grid_size: int = 101,
) -> Dict[str, PolicyThreshold]:
    """Largest certified decision groups under the chosen bound and pooling."""
    qs = np.linspace(0.0, 1.0, grid_size)
    # The missed-case side is the scarce one; give it most of the budget. The
    # false-alert side is bounded on ~200 negatives and needs very little.
    d_m, d_fa = delta * 0.8, delta * 0.2

    scopes = [("__all__", cal)] if pooling == "shared" else [(b, cal[cal["band"] == b]) for b in BANDS]

    lo_sel: Dict[str, float] = {b: -1.0 for b in BANDS}
    hi_sel: Dict[str, float] = {b: 2.0 for b in BANDS}
    feasible: Dict[str, bool] = {b: False for b in BANDS}

    for name, sub in scopes:
        if sub.empty:
            continue
        bands_here = BANDS if name == "__all__" else (name,)
        # NORMAL side: walk q upward while the certified risk stays in budget.
        best_q = None
        for q in qs:
            lo, _ = band_thresholds(sub, q, 1.0)
            fn, _, n_pos, _ = pooled_counts(sub, lo, {b: 2.0 for b in BANDS})
            if pi_hi * rate_bound(fn, n_pos, d_m, bound) <= alpha_missed + 1e-12:
                best_q = q
            else:
                break
        # ALERT side: walk the upper quantile downward.
        best_qa = None
        for qa in qs[::-1]:
            _, hi = band_thresholds(sub, 0.0, qa)
            _, fp, _, n_neg = pooled_counts(sub, {b: -1.0 for b in BANDS}, hi)
            if (1 - pi_lo) * rate_bound(fp, n_neg, d_fa, bound) <= alpha_false_alert + 1e-12:
                best_qa = qa
            else:
                break
        if best_q is None or best_qa is None:
            continue
        lo_q, hi_q = band_thresholds(sub, best_q, best_qa)
        for b in bands_here:
            if lo_q.get(b, -1.0) < hi_q.get(b, 2.0):
                lo_sel[b], hi_sel[b] = lo_q.get(b, -1.0), hi_q.get(b, 2.0)
                feasible[b] = True

    return {
        b: PolicyThreshold("certified", b, 0,
                           normal_threshold=lo_sel[b] if feasible[b] else -1.0,
                           alert_threshold=hi_sel[b] if feasible[b] else 2.0,
                           feasible=feasible[b])
        for b in BANDS
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Price the strong guarantee against the weak one.")
    p.add_argument("--run-root", default="outputs/paper_runs/pedrr_kfold_v3")
    p.add_argument("--out-dir", default="")
    p.add_argument("--calibration", choices=["oof", "val"], default="oof")
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--pi-lo", type=float, default=0.10)
    p.add_argument("--pi-hi", type=float, default=0.30)
    p.add_argument("--alpha-false-alert", type=float, default=0.15)
    p.add_argument("--alphas", default="0.01,0.02,0.03,0.05")
    p.add_argument("--no-age-fix", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / "deferral_policy" / run_root.name
    out_dir.mkdir(parents=True, exist_ok=True)
    costs = CostModel()
    alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]

    val_by_fold = load_by_fold(run_root, "val_eval", age_fix=not args.no_age_fix)
    test_by_fold = load_by_fold(run_root, "test_eval", age_fix=not args.no_age_fix)
    cal_by_fold = {f: calibration_for_fold(f, val_by_fold, test_by_fold, args.calibration)
                   for f in sorted(test_by_fold)}

    n_pos_band = {b: int(cal_by_fold[0][cal_by_fold[0]["band"] == b]["gt_tachy"].sum()) for b in BANDS}
    print(f"Run root   : {run_root}")
    print(f"Deployment : certified for prevalence in [{args.pi_lo:.2f}, {args.pi_hi:.2f}]")
    print(f"Positives  : pooled {sum(n_pos_band.values())}, per band {n_pos_band}")
    print(f"Budget     : delta={args.delta} split 80/20 across the missed and false-alert sides\n")

    rows = []
    for bound in ("crc", "cp"):
        for pooling in ("band", "shared"):
            for alpha in alphas:
                frames, triages = [], []
                for fold in sorted(test_by_fold):
                    by_band = fit_pooled(cal_by_fold[fold], alpha, float(args.alpha_false_alert),
                                         float(args.pi_lo), float(args.pi_hi), float(args.delta),
                                         bound, pooling)
                    frames.append(test_by_fold[fold])
                    triages.append(apply_policy(test_by_fold[fold], by_band, "certified"))
                pooled = pd.concat(frames, ignore_index=True)
                triage = np.concatenate(triages)
                m = metrics_for_triage(pooled, triage, "certified", "overall", costs)
                y = pooled["gt_tachy"].to_numpy(dtype=int)
                # Realised risk re-expressed at the certified worst-case prevalence.
                n_pos = max(int((y == 1).sum()), 1)
                fnr = m["fn"] / n_pos
                rows.append({
                    "bound": bound, "pooling": pooling, "alpha_missed": alpha,
                    "refer_rate": m["refer_rate"], "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"], "false_normals": m["fn"],
                    "risk_at_pi_hi": float(args.pi_hi) * fnr,
                    "within_budget": bool(float(args.pi_hi) * fnr <= alpha + 1e-12),
                    "decides_any": bool(m["refer_rate"] < 0.999),
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "guarantee_strength_tradeoff.csv", index=False)

    label = {("crc", "band"): "CRC, per band", ("crc", "shared"): "CRC, pooled",
             ("cp", "band"): "High-prob (CP), per band", ("cp", "shared"): "High-prob (CP), pooled"}
    print("Referral rate required by each guarantee (lower is better; '--' = certifies nothing)")
    print(f"  {'guarantee':<26}" + "".join(f"{'a=' + format(a, '.3g'):>12}" for a in alphas))
    for bound in ("crc", "cp"):
        for pooling in ("band", "shared"):
            sub = df[(df.bound == bound) & (df.pooling == pooling)].set_index("alpha_missed")
            cells = []
            for a in alphas:
                r = sub.loc[a]
                cells.append("--" if not r["decides_any"] else f"{r['refer_rate']:.1%}")
            print(f"  {label[(bound, pooling)]:<26}" + "".join(f"{c:>12}" for c in cells))

    print("\nSensitivity at the same settings")
    print(f"  {'guarantee':<26}" + "".join(f"{'a=' + format(a, '.3g'):>12}" for a in alphas))
    for bound in ("crc", "cp"):
        for pooling in ("band", "shared"):
            sub = df[(df.bound == bound) & (df.pooling == pooling)].set_index("alpha_missed")
            cells = []
            for a in alphas:
                r = sub.loc[a]
                cells.append("--" if not r["decides_any"] or pd.isna(r["sensitivity"])
                             else f"{r['sensitivity']:.3f}")
            print(f"  {label[(bound, pooling)]:<26}" + "".join(f"{c:>12}" for c in cells))

    ok = df[df.decides_any]
    print(f"\nBudget respected on held-out data: {int(ok['within_budget'].sum())}/{len(ok)} "
          f"settings that decide anything")
    strong = df[(df.bound == 'cp') & (df.pooling == 'shared') & df.decides_any]
    weak = df[(df.bound == 'crc') & (df.pooling == 'shared') & df.decides_any]
    if not strong.empty and not weak.empty:
        j = strong.merge(weak, on="alpha_missed", suffixes=("_cp", "_crc"))
        if not j.empty:
            print("\nPrice of the high-probability guarantee (pooled, vs the expectation guarantee)")
            for _, r in j.iterrows():
                print(f"  alpha={r['alpha_missed']:.3g}: refer {r['refer_rate_crc']:.1%} -> "
                      f"{r['refer_rate_cp']:.1%} ({(r['refer_rate_cp'] - r['refer_rate_crc']) * 100:+.1f} pts)")

    print(f"\nWrote {out_dir / 'guarantee_strength_tradeoff.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
