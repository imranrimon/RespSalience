"""Release-side risk audit for PRISM-Breath triage under prevalence shift.

A deferral policy can hold its calibration-time guarantee and still become
unsafe once deployment prevalence moves away from the prevalence it was
calibrated at: sensitivity and specificity are prevalence-invariant, but PPV and
NPV are not. This script audits the decisions the policy makes *on its own* --
the auto-released ALERT and NORMAL cases -- rather than the referral rate alone.

Two views, which should agree:

  analytic   Under label (prior) shift the class-conditional decision rates
             P(decide d | y) are fixed, so PPV, NPV and referral rate at any
             deployment prevalence follow in closed form. This also gives an
             exact break-even prevalence: the point where the guarantee is
             void.

  bootstrap  Subject-level resampling of the held-out test set to the target
             prevalence, within age band so that case mix is held fixed and
             only the label prior moves. Gives percentile intervals.

Policies are re-fitted through scripts/cost_aware_deferral.py, so calibration
stays leakage-safe and there is a single source of truth for the decision rules.

Run from the repository root, for example:

    python scripts/prevalence_shift_audit.py --run-root outputs/paper_runs/pedrr_kfold_v3
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
    POLICIES,
    CostModel,
    OperatingPoints,
    PolicyThreshold,
    apply_policy,
    build_operating_points,
    calibration_for_fold,
    fit_policies_for_fold,
    load_by_fold,
)

NORMAL, ALERT, REFER = 0, 1, 2


# --------------------------------------------------------------------------
# Class-conditional decision rates (the quantities that survive label shift)
# --------------------------------------------------------------------------

def conditional_rates(y: np.ndarray, triage: np.ndarray) -> Dict[str, float]:
    """P(decide d | y) for d in {NORMAL, ALERT, REFER} and y in {0, 1}."""
    out: Dict[str, float] = {}
    for label, name in ((1, "pos"), (0, "neg")):
        mask = y == label
        n = int(mask.sum())
        for code, dname in ((NORMAL, "normal"), (ALERT, "alert"), (REFER, "refer")):
            out[f"p_{dname}_given_{name}"] = float((triage[mask] == code).sum() / n) if n else float("nan")
        out[f"n_{name}"] = n
    return out


def pooled_conditional_rates(
    rates_by_band: Dict[str, Dict[str, float]], sizes: Dict[str, int]
) -> Dict[str, float]:
    """Aggregate per-band rates by band SIZE, matching within-band prevalence shift.

    Reading P(decide|y) off the pooled test set instead weights each band by its
    natural positive count, which is not invariant once every band is reweighted
    to a common prevalence. Weighting by band size is, and it makes the analytic
    curve agree with the bootstrap.
    """
    total = sum(sizes.get(b, 0) for b in rates_by_band)
    if total == 0:
        return {}
    out: Dict[str, float] = {}
    keys = ["p_normal_given_pos", "p_alert_given_pos", "p_refer_given_pos",
            "p_normal_given_neg", "p_alert_given_neg", "p_refer_given_neg"]
    for key in keys:
        acc, wsum = 0.0, 0.0
        for band, rates in rates_by_band.items():
            w = sizes.get(band, 0)
            val = rates.get(key, float("nan"))
            if w > 0 and np.isfinite(val):
                acc += w * val
                wsum += w
        out[key] = acc / wsum if wsum > 0 else float("nan")
    out["n_pos"] = sum(int(r.get("n_pos", 0)) for r in rates_by_band.values())
    out["n_neg"] = sum(int(r.get("n_neg", 0)) for r in rates_by_band.values())
    return out


def analytic_operating_metrics(rates: Dict[str, float], prevalence: float) -> Dict[str, float]:
    """PPV/NPV/referral at a deployment prevalence, under label shift."""
    pi = float(prevalence)
    p1n, p0n = rates["p_normal_given_pos"], rates["p_normal_given_neg"]
    p1a, p0a = rates["p_alert_given_pos"], rates["p_alert_given_neg"]
    p1r, p0r = rates["p_refer_given_pos"], rates["p_refer_given_neg"]

    normal_mass = pi * p1n + (1 - pi) * p0n
    alert_mass = pi * p1a + (1 - pi) * p0a
    npv = ((1 - pi) * p0n / normal_mass) if normal_mass > 0 else float("nan")
    ppv = (pi * p1a / alert_mass) if alert_mass > 0 else float("nan")
    retained = normal_mass + alert_mass
    return {
        "prevalence": pi,
        "refer_rate": pi * p1r + (1 - pi) * p0r,
        "ppv": ppv,
        "npv": npv,
        "false_omission_rate": 1.0 - npv if np.isfinite(npv) else float("nan"),
        # The release-side harm: truly tachypneic infants the system clears itself.
        "missed_per_1000_screened": 1000.0 * pi * p1n,
        "auto_normal_share": normal_mass,
        "auto_alert_share": alert_mass,
        "retained_share": retained,
        # Prevalence-invariant, shown to make the contrast explicit.
        "sensitivity_retained": (p1a / (p1a + p1n)) if (p1a + p1n) > 0 else float("nan"),
        "specificity_retained": (p0n / (p0n + p0a)) if (p0n + p0a) > 0 else float("nan"),
    }


def npv_breakeven_prevalence(rates: Dict[str, float], target: float) -> float:
    """Largest deployment prevalence at which NPV >= target still holds.

    NPV falls as prevalence rises, so this is an upper bound on safe deployment
    prevalence. Returns 1.0 if the guarantee never breaks, nan if undefined.
    """
    p1n, p0n = rates["p_normal_given_pos"], rates["p_normal_given_neg"]
    if not np.isfinite(p1n) or not np.isfinite(p0n) or (p1n + p0n) == 0:
        return float("nan")
    if p1n <= 0:
        return 1.0  # never clears a true positive
    denom = p0n * (1 - target) + target * p1n
    return float(np.clip(p0n * (1 - target) / denom, 0.0, 1.0)) if denom > 0 else float("nan")


def ppv_breakeven_prevalence(rates: Dict[str, float], target: float) -> float:
    """Smallest deployment prevalence at which PPV >= target still holds.

    PPV rises with prevalence, so this is a lower bound: below it the alert
    group is mostly false alarms.
    """
    p1a, p0a = rates["p_alert_given_pos"], rates["p_alert_given_neg"]
    if not np.isfinite(p1a) or not np.isfinite(p0a):
        return float("nan")
    if p0a <= 0:
        return 0.0  # never alerts on a true negative
    if p1a <= 0:
        return float("nan")
    denom = target * p0a + (1 - target) * p1a
    return float(np.clip(target * p0a / denom, 0.0, 1.0)) if denom > 0 else float("nan")


# --------------------------------------------------------------------------
# Bootstrap under within-band prevalence shift
# --------------------------------------------------------------------------

def _empirical_metrics(y: np.ndarray, triage: np.ndarray) -> Dict[str, float]:
    n = max(y.size, 1)
    tp = int(((triage == ALERT) & (y == 1)).sum())
    fp = int(((triage == ALERT) & (y == 0)).sum())
    tn = int(((triage == NORMAL) & (y == 0)).sum())
    fn = int(((triage == NORMAL) & (y == 1)).sum())
    refer = int((triage == REFER).sum())
    return {
        "refer_rate": refer / n,
        "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
        "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
        "false_omission_rate": fn / (tn + fn) if (tn + fn) else float("nan"),
        "missed_per_1000_screened": 1000.0 * fn / n,
    }


def bootstrap_shift(
    df: pd.DataFrame,
    triage: np.ndarray,
    prevalence: float,
    n_boot: int,
    rng: np.random.Generator,
    band: str = "overall",
) -> Dict[str, Tuple[float, float, float]]:
    """Resample to `prevalence` within each age band, holding case mix fixed."""
    y = df["gt_tachy"].to_numpy(dtype=int)
    bands = df["band"].to_numpy()
    scope = np.ones(len(df), dtype=bool) if band == "overall" else (bands == band)

    pools: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for b in BANDS:
        idx = np.flatnonzero(scope & (bands == b))
        if idx.size == 0:
            continue
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        if pos.size == 0 or neg.size == 0:
            continue  # cannot re-weight a band with only one class present
        pools.append((pos, neg, idx.size))
    if not pools:
        return {}

    draws: Dict[str, List[float]] = {}
    for _ in range(n_boot):
        picked: List[np.ndarray] = []
        for pos, neg, size in pools:
            n_pos = int(round(prevalence * size))
            n_pos = min(max(n_pos, 0), size)
            picked.append(rng.choice(pos, size=n_pos, replace=True))
            picked.append(rng.choice(neg, size=size - n_pos, replace=True))
        sel = np.concatenate(picked)
        for key, val in _empirical_metrics(y[sel], triage[sel]).items():
            draws.setdefault(key, []).append(val)

    out: Dict[str, Tuple[float, float, float]] = {}
    for key, vals in draws.items():
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            out[key] = (float("nan"), float("nan"), float("nan"))
        else:
            out[key] = (float(finite.mean()),
                        float(np.percentile(finite, 2.5)),
                        float(np.percentile(finite, 97.5)))
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def fit_all_policies(
    run_root: Path,
    calibration: str,
    alpha: float,
    grid_size: int,
    delta: float,
    target_ppv: float,
    target_npv: float,
    costs: CostModel,
    method: str,
    max_refer_rate: float,
    prevalence_range: Tuple[float, float] = (0.10, 0.50),
    age_fix: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], float]:
    val_by_fold = load_by_fold(run_root, "val_eval", age_fix=age_fix)
    test_by_fold = load_by_fold(run_root, "test_eval", age_fix=age_fix)

    frames: List[pd.DataFrame] = []
    per_policy: Dict[str, List[np.ndarray]] = {p: [] for p in POLICIES}
    cal_prevalences: List[float] = []

    for fold in sorted(test_by_fold):
        cal = calibration_for_fold(fold, val_by_fold, test_by_fold, calibration)
        cal_prevalences.append(float(cal["gt_tachy"].mean()))
        ops = build_operating_points(cal, fold, grid_size, delta)
        fitted = fit_policies_for_fold(cal, fold, alpha, grid_size, target_ppv, target_npv,
                                       costs, delta, max_refer_rate, ops_by_band=ops, method=method,
                                       prevalence_range=prevalence_range)
        test = test_by_fold[fold]
        frames.append(test)
        for policy in POLICIES:
            by_band = {t.band: t for t in fitted if t.policy == policy}
            per_policy[policy].append(apply_policy(test, by_band, policy))

    df = pd.concat(frames, ignore_index=True)
    triages = {p: np.concatenate(v) for p, v in per_policy.items()}
    return df, triages, float(np.mean(cal_prevalences))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit auto-released PRISM-Breath triage decisions under prevalence shift."
    )
    p.add_argument("--run-root", default="outputs/paper_runs/pedrr_kfold_v3")
    p.add_argument("--out-dir", default="", help="Default: outputs/deferral_policy/<run-name>")
    p.add_argument("--calibration", choices=["oof", "val"], default="oof")
    p.add_argument("--certification", choices=["fixed-sequence", "bonferroni"], default="fixed-sequence")
    p.add_argument("--alpha", type=float, default=0.20)
    p.add_argument("--grid-size", type=int, default=41)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--target-ppv", type=float, default=0.60)
    p.add_argument("--target-npv", type=float, default=0.90)
    p.add_argument("--max-refer-rate", type=float, default=1.0)
    p.add_argument("--deploy-prevalence-min", type=float, default=0.10)
    p.add_argument("--deploy-prevalence-max", type=float, default=0.50)
    p.add_argument("--miss-cost", type=float, default=10.0)
    p.add_argument("--false-alert-cost", type=float, default=1.0)
    p.add_argument("--referral-cost", type=float, default=0.3)
    p.add_argument("--prevalences", default="0.05,0.10,0.15,0.20,0.26,0.30,0.40,0.50,0.60")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-age-fix", action="store_true")
    p.add_argument("--skip-bootstrap", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / "deferral_policy" / run_root.name
    out_dir.mkdir(parents=True, exist_ok=True)
    costs = CostModel(miss=args.miss_cost, false_alert=args.false_alert_cost, referral=args.referral_cost)
    prevalences = [float(x) for x in str(args.prevalences).split(",") if x.strip()]
    rng = np.random.default_rng(int(args.seed))

    df, triages, cal_prevalence = fit_all_policies(
        run_root, args.calibration, args.alpha, args.grid_size, args.delta,
        args.target_ppv, args.target_npv, costs, args.certification, args.max_refer_rate,
        prevalence_range=(float(args.deploy_prevalence_min), float(args.deploy_prevalence_max)),
        age_fix=not args.no_age_fix,
    )
    observed_prevalence = float(df["gt_tachy"].mean())
    y = df["gt_tachy"].to_numpy(dtype=int)

    print(f"Run root            : {run_root}")
    print(f"Held-out subjects   : {len(df)}  ({int(y.sum())} tachypneic)")
    print(f"Calibration prev.   : {cal_prevalence:.3f}   Test prevalence: {observed_prevalence:.3f}")
    print(f"Guarantee targets   : PPV>={args.target_ppv:.2f}, NPV>={args.target_npv:.2f} "
          f"({args.certification})")
    print(f"Shift model         : within-band label shift, case mix held fixed")

    # ---- class-conditional rates and break-even prevalences -------------
    band_sizes = {b: int((df["band"] == b).sum()) for b in BANDS}
    rate_rows, curve_rows = [], []
    for policy, triage in triages.items():
        per_band = {
            b: conditional_rates(y[(df["band"] == b).to_numpy()], triage[(df["band"] == b).to_numpy()])
            for b in BANDS if band_sizes[b] > 0
        }
        for band in ALL_BANDS:
            mask = np.ones(len(df), dtype=bool) if band == "overall" else (df["band"] == band).to_numpy()
            if not mask.any():
                continue
            rates = pooled_conditional_rates(per_band, band_sizes) if band == "overall" else per_band[band]
            if not rates:
                continue
            npv_star = npv_breakeven_prevalence(rates, args.target_npv)
            ppv_star = ppv_breakeven_prevalence(rates, args.target_ppv)
            rate_rows.append({
                "policy": policy, "band": band, **rates,
                "cal_prevalence": cal_prevalence, "test_prevalence": float(y[mask].mean()),
                "target_npv": args.target_npv, "npv_safe_up_to_prevalence": npv_star,
                "target_ppv": args.target_ppv, "ppv_valid_from_prevalence": ppv_star,
                "npv_holds_at_test_prevalence": bool(
                    np.isfinite(npv_star) and float(y[mask].mean()) <= npv_star),
            })
            for pi in prevalences:
                curve_rows.append({"policy": policy, "band": band, "view": "analytic",
                                   **analytic_operating_metrics(rates, pi)})

    rates_df = pd.DataFrame(rate_rows)
    curves_df = pd.DataFrame(curve_rows)

    # ---- bootstrap check -------------------------------------------------
    if not args.skip_bootstrap:
        boot_rows = []
        for policy, triage in triages.items():
            for band in ALL_BANDS:
                for pi in prevalences:
                    res = bootstrap_shift(df, triage, pi, int(args.n_boot), rng, band)
                    if not res:
                        continue
                    row = {"policy": policy, "band": band, "view": "bootstrap", "prevalence": pi}
                    for key, (mean, lo, hi) in res.items():
                        row[key] = mean
                        row[f"{key}_lo"] = lo
                        row[f"{key}_hi"] = hi
                    boot_rows.append(row)
        if boot_rows:
            curves_df = pd.concat([curves_df, pd.DataFrame(boot_rows)], ignore_index=True)

    rates_df.to_csv(out_dir / "prevalence_breakeven.csv", index=False)
    curves_df.to_csv(out_dir / "prevalence_shift_curves.csv", index=False)

    # ---- report ----------------------------------------------------------
    print("\nRelease-side safety envelope (overall)")
    show = rates_df[rates_df.band == "overall"][[
        "policy", "p_normal_given_pos", "p_alert_given_neg", "test_prevalence",
        "npv_safe_up_to_prevalence", "ppv_valid_from_prevalence"]].copy()
    show.columns = ["policy", "P(clear|tachy)", "P(alert|normal)", "test prev",
                    f"NPV>={args.target_npv:.2f} up to prev", f"PPV>={args.target_ppv:.2f} from prev"]
    for col in show.columns[1:]:
        show[col] = show[col].map(lambda x: "--" if pd.isna(x) else f"{float(x):.3f}")
    print(show.to_string(index=False))

    print("\nAuto-cleared true tachypnea per 1000 screened, by deployment prevalence (analytic)")
    piv = (curves_df[(curves_df.view == "analytic") & (curves_df.band == "overall")]
           .pivot_table(index="policy", columns="prevalence", values="missed_per_1000_screened"))
    print(piv.round(1).to_string())

    print("\nNPV of the auto-cleared group, by deployment prevalence (analytic)")
    piv = (curves_df[(curves_df.view == "analytic") & (curves_df.band == "overall")]
           .pivot_table(index="policy", columns="prevalence", values="npv"))
    print(piv.round(3).to_string())

    if not args.skip_bootstrap:
        print("\nAnalytic vs. bootstrap agreement (overall NPV, max absolute difference)")
        a = curves_df[(curves_df.view == "analytic") & (curves_df.band == "overall")]
        b = curves_df[(curves_df.view == "bootstrap") & (curves_df.band == "overall")]
        merged = a.merge(b, on=["policy", "prevalence"], suffixes=("_a", "_b"))
        for policy, grp in merged.groupby("policy"):
            diff = (grp["npv_a"] - grp["npv_b"]).abs()
            diff = diff[np.isfinite(diff)]
            print(f"  {policy:<20} max |analytic - bootstrap| = "
                  f"{diff.max():.4f}" if diff.size else f"  {policy:<20} n/a")

    print("\nVerdict")
    for _, r in rates_df[rates_df.band == "overall"].iterrows():
        star = r["npv_safe_up_to_prevalence"]
        if not np.isfinite(star):
            print(f"  {r['policy']:<20} makes no auto-NORMAL decisions; no release-side risk to audit.")
            continue
        verdict = "safe across the swept range" if star >= max(prevalences) else (
            f"VOID above prevalence {star:.1%}")
        print(f"  {r['policy']:<20} NPV>={args.target_npv:.2f} {verdict} "
              f"(test prevalence {r['test_prevalence']:.1%})")

    print("\nWrote outputs")
    for name in ("prevalence_breakeven.csv", "prevalence_shift_curves.csv"):
        print(f"  {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
