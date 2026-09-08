"""Conformal risk control for PRISM-Breath triage.

Certifying PPV/NPV is sample-hungry: both are proportions *within* a decision
group, so the effective sample size is the size of that group, not of the
calibration set. On this cohort that is what forces the certified policy to
defer 84% of cases -- the guarantee is limited by roughly 68 positive
calibration subjects, not by the model.

Conformal risk control (CRC) targets a *population* risk instead:

    R_normal(a) = P(score <= a  and  y = 1)     auto-cleared while tachypneic
    R_alert(b)  = P(score >= b  and  y = 0)     auto-alerted while healthy

Both are means over every calibration subject, so the effective sample size is
the whole calibration set. Each is monotone in its threshold, which is all CRC
requires, and the guarantee

    E[R(lambda_hat)] <= alpha,   lambda_hat = sup{ lambda : (n*R_hat + B)/(n+1) <= alpha }

is exact in finite samples with **no multiplicity correction** -- the two
thresholds are separate risks with separate targets, so there is nothing to
split alpha across either.

The risk is also the quantity a screening programme actually budgets: alpha is
"missed tachypnea per screened child", directly convertible to per-1000 harm.

Run from the repository root:

    python scripts/conformal_risk_triage.py --run-root outputs/paper_runs/pedrr_kfold_v3
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
    OperatingPoints,
    PolicyThreshold,
    apply_policy,
    build_operating_points,
    calibration_for_fold,
    fit_coverage_conformal,
    load_by_fold,
    metrics_for_triage,
    prespecified_grid,
    select_ppv_npv,
)

NORMAL, ALERT, REFER = 0, 1, 2


# --------------------------------------------------------------------------
# Conformal risk control
# --------------------------------------------------------------------------

def crc_normal_threshold(scores: np.ndarray, y: np.ndarray, alpha: float,
                         grid: np.ndarray) -> Tuple[float, float]:
    """Largest NORMAL threshold whose missed-case risk is certified at alpha.

    R(a) = P(score <= a and y = 1) is non-decreasing in a, so the CRC condition
    ``(n*R_hat(a) + B)/(n+1) <= alpha`` selects a prefix of the grid and the
    answer is the last threshold that satisfies it. Returns (-1, nan) when even
    the emptiest decision group cannot meet alpha, which is the instruction to
    refer rather than to decide.
    """
    n = int(y.size)
    if n == 0:
        return -1.0, float("nan")
    best_a, best_r = -1.0, float("nan")
    for a in grid:
        r_hat = float(((scores <= a) & (y == 1)).sum() / n)
        if (n * r_hat + 1.0) / (n + 1) <= alpha:
            best_a, best_r = float(a), r_hat
        else:
            break  # monotone: once it fails it stays failed
    return best_a, best_r


def crc_alert_threshold(scores: np.ndarray, y: np.ndarray, alpha: float,
                        grid: np.ndarray) -> Tuple[float, float]:
    """Smallest ALERT threshold whose false-alert risk is certified at alpha.

    R(b) = P(score >= b and y = 0) is non-increasing in b, so we walk the grid
    downward and keep the last threshold that satisfies the CRC condition.
    """
    n = int(y.size)
    if n == 0:
        return 2.0, float("nan")
    best_b, best_r = 2.0, float("nan")
    for b in grid[::-1]:
        r_hat = float(((scores >= b) & (y == 0)).sum() / n)
        if (n * r_hat + 1.0) / (n + 1) <= alpha:
            best_b, best_r = float(b), r_hat
        else:
            break
    return best_b, best_r


def fit_crc(ops_cal: pd.DataFrame, band: str, fold: int, alpha_missed: float,
            alpha_false_alert: float, grid_size: int) -> PolicyThreshold:
    grid = prespecified_grid(grid_size)
    scores = ops_cal["tachy_prob"].to_numpy(dtype=float)
    y = ops_cal["gt_tachy"].to_numpy(dtype=int)
    a, r_normal = crc_normal_threshold(scores, y, alpha_missed, grid)
    b, r_alert = crc_alert_threshold(scores, y, alpha_false_alert, grid)

    feasible = a >= 0.0 and b <= 1.0 and a < b
    if a >= b:
        # The two certified groups would overlap; shrink the NORMAL side, whose
        # risk only decreases as the threshold falls.
        a = float(max(0.0, b - 1e-9))
        feasible = b <= 1.0
    return PolicyThreshold(
        policy="crc",
        band=band,
        fold=fold,
        normal_threshold=a if feasible else -1.0,
        alert_threshold=b if feasible else 2.0,
        feasible=bool(feasible),
        calibration_n=int(y.size),
        calibration_positives=int(y.sum()),
        ppv_lcb=r_alert,
        npv_lcb=r_normal,
        infeasible_reason="" if feasible else "no threshold pair meets both risk budgets",
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def realised_risks(df: pd.DataFrame, triage: np.ndarray) -> Dict[str, float]:
    y = df["gt_tachy"].to_numpy(dtype=int)
    n = max(y.size, 1)
    return {
        "risk_missed": float(((triage == NORMAL) & (y == 1)).sum() / n),
        "risk_false_alert": float(((triage == ALERT) & (y == 0)).sum() / n),
        "missed_per_1000": 1000.0 * float(((triage == NORMAL) & (y == 1)).sum() / n),
    }


def run_policy(test_by_fold, by_fold_band, policy) -> Tuple[pd.DataFrame, np.ndarray]:
    frames, triages = [], []
    for fold in sorted(test_by_fold):
        test = test_by_fold[fold]
        frames.append(test)
        triages.append(apply_policy(test, by_fold_band[fold], policy))
    return pd.concat(frames, ignore_index=True), np.concatenate(triages)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Conformal risk control for PRISM-Breath triage.")
    p.add_argument("--run-root", default="outputs/paper_runs/pedrr_kfold_v3")
    p.add_argument("--out-dir", default="")
    p.add_argument("--calibration", choices=["oof", "val"], default="oof")
    p.add_argument("--grid-size", type=int, default=101)
    p.add_argument("--ltt-grid-size", type=int, default=41,
                   help="Grid for the LTT comparator. Fixed-sequence testing stops at the first "
                        "hypothesis it cannot reject, so a finer grid is strictly more conservative; "
                        "this keeps the comparator at the setting used elsewhere in the paper.")
    p.add_argument("--alpha-missed", type=float, default=0.01,
                   help="Budget for auto-cleared-while-tachypneic, per screened child")
    p.add_argument("--alpha-false-alert", type=float, default=0.10,
                   help="Budget for auto-alerted-while-healthy, per screened child")
    p.add_argument("--no-age-fix", action="store_true")
    p.add_argument("--compare-ltt", action="store_true", default=True)
    p.add_argument("--target-ppv", type=float, default=0.60)
    p.add_argument("--target-npv", type=float, default=0.90)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / "deferral_policy" / run_root.name
    out_dir.mkdir(parents=True, exist_ok=True)
    age_fix = not args.no_age_fix
    costs = CostModel()

    val_by_fold = load_by_fold(run_root, "val_eval", age_fix=age_fix)
    test_by_fold = load_by_fold(run_root, "test_eval", age_fix=age_fix)

    print(f"Run root      : {run_root}")
    print(f"Risk budgets  : missed <= {args.alpha_missed:.3f}/screened, "
          f"false alert <= {args.alpha_false_alert:.3f}/screened")
    print(f"Grid          : {args.grid_size} thresholds; CRC needs no multiplicity correction")

    cal_by_fold, crc_by_fold, ltt_by_fold, cov_by_fold = {}, {}, {}, {}
    for fold in sorted(test_by_fold):
        cal = calibration_for_fold(fold, val_by_fold, test_by_fold, args.calibration)
        cal_by_fold[fold] = cal
        ops = build_operating_points(cal, fold, int(args.ltt_grid_size), 0.10)
        crc_band, ltt_band, cov_band = {}, {}, {}
        for band in BANDS:
            sub = cal[cal["band"] == band]
            if sub.empty:
                crc_band[band] = PolicyThreshold("crc", band, fold, -1.0, 2.0, feasible=False)
                ltt_band[band] = PolicyThreshold("ppv_npv_ltt", band, fold, -1.0, 2.0, feasible=False)
                cov_band[band] = PolicyThreshold("coverage_conformal", band, fold, feasible=False)
                continue
            crc_band[band] = fit_crc(sub, band, fold, float(args.alpha_missed),
                                     float(args.alpha_false_alert), int(args.grid_size))
            cov_band[band] = fit_coverage_conformal(sub, band, fold, 0.20)
            o = ops[band]
            ltt_band[band] = (PolicyThreshold("ppv_npv_ltt", band, fold, -1.0, 2.0, feasible=False)
                              if o is None else
                              select_ppv_npv(o, args.target_ppv, args.target_npv, True, costs))
        crc_by_fold[fold] = crc_band
        ltt_by_fold[fold] = ltt_band
        cov_by_fold[fold] = cov_band

    pooled, crc_triage = run_policy(test_by_fold, crc_by_fold, "crc")
    _, ltt_triage = run_policy(test_by_fold, ltt_by_fold, "ppv_npv_ltt")
    _, cov_triage = run_policy(test_by_fold, cov_by_fold, "coverage_conformal")

    rows = []
    for name, triage in (("Coverage (paper)", cov_triage),
                         ("LTT (PPV/NPV)", ltt_triage),
                         ("CRC (risk control)", crc_triage)):
        for band in ALL_BANDS:
            if band == "overall":
                sub, t = pooled, triage
            else:
                mask = (pooled["band"] == band).to_numpy()
                sub, t = pooled[mask], triage[mask]
            if sub.empty:
                continue
            m = metrics_for_triage(sub, t, name, band, costs)
            m.update(realised_risks(sub, t))
            rows.append(m)
    summary = pd.DataFrame(rows)

    print("\nHeld-out comparison (per-fold calibration, decisions pooled)")
    show = summary[["policy", "band", "n", "refer_rate", "sensitivity", "specificity",
                    "npv", "false_normals", "risk_missed", "missed_per_1000"]].copy()
    for c in ["refer_rate", "sensitivity", "specificity", "npv", "risk_missed"]:
        show[c] = show[c].map(lambda x: "--" if pd.isna(x) else f"{float(x):.3f}")
    show["missed_per_1000"] = show["missed_per_1000"].map(lambda x: f"{float(x):.1f}")
    print(show.to_string(index=False))

    over = summary[(summary.band == "overall")].set_index("policy")
    print("\nSample-efficiency of the guarantee (overall)")
    for name in ("Coverage (paper)", "LTT (PPV/NPV)", "CRC (risk control)"):
        if name not in over.index:
            continue
        r = over.loc[name]
        sens = "--" if pd.isna(r["sensitivity"]) else f"{r['sensitivity']:.3f}"
        print(f"  {name:<20} refer {r['refer_rate']:.1%}  sens {sens}  "
              f"{int(r['false_normals'])} missed ({r['missed_per_1000']:.1f}/1000)")

    # Validity: does the realised risk respect the budget it was given?
    print("\nValidity check: realised risk vs budget (held-out)")
    sweep = []
    for a in (0.005, 0.0075, 0.01, 0.02, 0.03, 0.05):
        by_fold = {}
        for fold in sorted(test_by_fold):
            band_map = {}
            for band in BANDS:
                sub = cal_by_fold[fold][cal_by_fold[fold]["band"] == band]
                band_map[band] = (PolicyThreshold("crc", band, fold, -1.0, 2.0, feasible=False)
                                  if sub.empty else
                                  fit_crc(sub, band, fold, a, float(args.alpha_false_alert),
                                          int(args.grid_size)))
            by_fold[fold] = band_map
        pooled_a, triage_a = run_policy(test_by_fold, by_fold, "crc")
        r = realised_risks(pooled_a, triage_a)
        m = metrics_for_triage(pooled_a, triage_a, "crc", "overall", costs)
        sweep.append({"alpha_missed": a, "realised_risk_missed": r["risk_missed"],
                      "within_budget": r["risk_missed"] <= a,
                      "refer_rate": m["refer_rate"], "sensitivity": m["sensitivity"],
                      "specificity": m["specificity"], "false_normals": m["false_normals"],
                      "missed_per_1000": r["missed_per_1000"]})
    sw = pd.DataFrame(sweep)
    disp = sw.copy()
    for c in ["realised_risk_missed", "refer_rate", "sensitivity", "specificity"]:
        disp[c] = disp[c].map(lambda x: "--" if pd.isna(x) else f"{float(x):.3f}")
    print(disp.to_string(index=False))
    n_ok = int(sw["within_budget"].sum())
    print(f"  budget respected in {n_ok}/{len(sw)} settings")

    summary.to_csv(out_dir / "crc_comparison.csv", index=False)
    sw.to_csv(out_dir / "crc_alpha_sweep.csv", index=False)
    print(f"\nWrote\n  {out_dir / 'crc_comparison.csv'}\n  {out_dir / 'crc_alpha_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
