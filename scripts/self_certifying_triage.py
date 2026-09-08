"""Self-certifying triage: estimate deployment prevalence, then certify at it.

The prevalence-robust certificate in scripts/cost_aware_deferral.py needs a
*declared* deployment range and must assume the worst point in it. Declaring a
wide range to be safe makes the certificate so conservative it licenses almost
no decisions; declaring a narrow one is a promise nobody can keep.

A deployment site does not know its prevalence, but it does have unlabelled
scores. Under label shift the class-conditional score distributions are fixed,
so prevalence is identifiable from those scores alone:

    mu = P(score >= tau) = pi * sens + (1 - pi) * fpr      =>   pi = (mu - fpr) / (sens - fpr)

which is the Rogan-Gladen / BBSE estimator. Propagating exact binomial bounds
through it -- pi is increasing in mu and decreasing in both sens and fpr -- gives
a finite-sample interval [pi_lo, pi_hi] rather than a point estimate, and the
certificate is then issued at the worst point of *that* interval. The result is
a triage layer that certifies itself against the prevalence it is actually
deployed into, with a total error budget split between estimating the prior and
bounding the class-conditional rates.

Validation resamples the held-out cohort to known prevalences, hides the labels,
and checks (i) that the interval covers the truth, (ii) that the resulting
certificate holds, and (iii) how much less it has to defer than a fixed
conservative range.

Run from the repository root:

    python scripts/self_certifying_triage.py --run-root outputs/paper_runs/pedrr_kfold_v3
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
    BANDS,
    CostModel,
    OperatingPoints,
    PolicyThreshold,
    apply_policy,
    build_operating_points,
    calibration_for_fold,
    clopper_pearson_lower,
    clopper_pearson_upper,
    load_by_fold,
    metrics_for_triage,
    select_ppv_npv_robust,
)

NORMAL, ALERT, REFER = 0, 1, 2


# --------------------------------------------------------------------------
# Prior estimation under label shift
# --------------------------------------------------------------------------

def bbse_point(mu: float, sens: float, fpr: float) -> float:
    """Rogan-Gladen / BBSE point estimate of deployment prevalence."""
    if not np.isfinite(sens) or not np.isfinite(fpr) or (sens - fpr) <= 1e-9:
        return float("nan")
    return float(np.clip((mu - fpr) / (sens - fpr), 0.0, 1.0))


def bbse_interval(
    n_target_pos_calls: int, n_target: int,
    tp: int, n_pos: int, fp: int, n_neg: int,
    delta: float,
) -> Tuple[float, float, float]:
    """Finite-sample interval for deployment prevalence.

    pi = (mu - fpr) / (sens - fpr) is increasing in mu and decreasing in both
    sens and fpr, so the extremes of the box give the extremes of pi. The budget
    delta is split three ways across the binomials for mu, sens and fpr.
    """
    d = delta / 3.0
    mu_lo = clopper_pearson_lower(n_target_pos_calls, n_target, d)
    mu_hi = clopper_pearson_upper(n_target_pos_calls, n_target, d)
    sens_lo = clopper_pearson_lower(tp, n_pos, d)
    sens_hi = clopper_pearson_upper(tp, n_pos, d)
    fpr_lo = clopper_pearson_lower(fp, n_neg, d)
    fpr_hi = clopper_pearson_upper(fp, n_neg, d)

    mu_hat = n_target_pos_calls / max(n_target, 1)
    sens_hat = tp / max(n_pos, 1)
    fpr_hat = fp / max(n_neg, 1)
    point = bbse_point(mu_hat, sens_hat, fpr_hat)

    # Worst case up: largest mu, smallest sens, smallest fpr.
    hi = bbse_point(mu_hi, sens_lo, fpr_lo)
    # Worst case down: smallest mu, largest sens, largest fpr.
    lo = bbse_point(mu_lo, sens_hi, fpr_hi)
    if not np.isfinite(hi):
        hi = 1.0
    if not np.isfinite(lo):
        lo = 0.0
    return float(np.clip(lo, 0.0, 1.0)), float(point), float(np.clip(hi, 0.0, 1.0))


def em_prior(probs: np.ndarray, pi_source: float, iters: int = 200, tol: float = 1e-8) -> float:
    """Saerens-Latinne-Decaestecker EM (MLLS) prior estimate.

    Uses the calibrated probabilities directly, so it is usually tighter than
    BBSE, but it has no exact finite-sample interval; reported for comparison.
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    pi = float(pi_source)
    for _ in range(iters):
        w1 = (pi / pi_source) * p
        w0 = ((1 - pi) / (1 - pi_source)) * (1 - p)
        post = w1 / (w1 + w0)
        new = float(post.mean())
        if abs(new - pi) < tol:
            break
        pi = new
    return float(np.clip(pi, 0.0, 1.0))


# --------------------------------------------------------------------------
# Deployment simulation
# --------------------------------------------------------------------------

def resample_to_prevalence(df: pd.DataFrame, prevalence: float,
                           rng: np.random.Generator, size_mult: float = 1.0) -> np.ndarray:
    """Row indices of a cohort re-weighted to `prevalence` within each band.

    `size_mult` scales the deployment cohort, modelling a site that has
    accumulated more screens before asking for a certificate.
    """
    y = df["gt_tachy"].to_numpy(dtype=int)
    bands = df["band"].to_numpy()
    picked: List[np.ndarray] = []
    for b in BANDS:
        idx = np.flatnonzero(bands == b)
        if idx.size == 0:
            continue
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        if pos.size == 0 or neg.size == 0:
            picked.append(rng.choice(idx, size=int(round(size_mult * idx.size)), replace=True))
            continue
        size = int(round(size_mult * idx.size))
        n_pos = int(round(prevalence * size))
        n_pos = min(max(n_pos, 0), size)
        picked.append(rng.choice(pos, size=n_pos, replace=True))
        picked.append(rng.choice(neg, size=size - n_pos, replace=True))
    return np.concatenate(picked) if picked else np.array([], dtype=int)


def source_rates(cal: pd.DataFrame, tau: float) -> Tuple[int, int, int, int]:
    """(tp, n_pos, fp, n_neg) for the hard rule score >= tau on calibration."""
    y = cal["gt_tachy"].to_numpy(dtype=int)
    s = cal["tachy_prob"].to_numpy(dtype=float)
    call = s >= tau
    return (int((call & (y == 1)).sum()), int((y == 1).sum()),
            int((call & (y == 0)).sum()), int((y == 0).sum()))


def best_tau(cal: pd.DataFrame, grid_size: int = 101) -> float:
    """Threshold maximising sens - fpr (Youden's J) on the calibration set.

    The estimator divides by (sens - fpr), so that gap is exactly the factor by
    which deployment sampling noise is amplified into the prevalence interval.
    Choosing tau to maximise it is free -- it uses calibration labels only --
    and materially narrows the interval versus a default of 0.5.
    """
    y = cal["gt_tachy"].to_numpy(dtype=int)
    s = cal["tachy_prob"].to_numpy(dtype=float)
    n_pos, n_neg = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    best, best_j = 0.5, -np.inf
    for t in np.linspace(0.0, 1.0, grid_size):
        call = s >= t
        j = (call & (y == 1)).sum() / n_pos - (call & (y == 0)).sum() / n_neg
        if j > best_j:
            best_j, best = j, float(t)
    return best


def select_crc_robust(
    ops: OperatingPoints,
    alpha_missed: float,
    alpha_false_alert: float,
    pi_lo: float,
    pi_hi: float,
    delta: float,
    bound: str = "crc",
) -> PolicyThreshold:
    """Prevalence-adaptive conformal risk control -- the composition of B and C.

    The population risk factorises into a prevalence and a class-conditional
    rate, and only the latter has to be estimated from labels:

        P(clear and tachypneic) = pi * P(clear | y=1) <= pi_hi * fnr_upper
        P(alert and healthy)    = (1-pi) * P(alert | y=0) <= (1-pi_lo) * fpr_upper

    Bounding the class-conditional rate uses every positive in the calibration
    set rather than only the members of a decision group, which is why this
    certifies where the PPV/NPV version cannot. Both factors are monotone in
    their threshold, so a single scan finds the largest certified decision
    groups. delta is split between the two bounds.
    """
    g = int(ops.grid.size)
    if g == 0:
        return PolicyThreshold("crc_robust", ops.band, ops.fold, -1.0, 2.0, feasible=False)
    n_pos = int(ops.positives)
    n_neg = int(ops.n - ops.positives)
    d = delta / 2.0

    if bound == "cp":
        # High-probability: P(risk <= alpha) >= 1 - delta. Stronger claim, and on
        # a band with ~20 positives a Clopper-Pearson bound on a rate of zero is
        # still ~0.15, which is what makes it certify so little.
        fnr_hi = np.nan_to_num(
            np.array([clopper_pearson_upper(int(k), n_pos, d) for k in ops.fn_of_a]), nan=1.0)
        fpr_hi = np.nan_to_num(
            np.array([clopper_pearson_upper(int(k), n_neg, d) for k in ops.fp_of_b]), nan=1.0)
    else:
        # Conformal risk control: E[risk] <= alpha. The finite-sample correction
        # is a single pseudo-count, (k+1)/(n+1), rather than a tail bound.
        fnr_hi = (ops.fn_of_a.astype(float) + 1.0) / (n_pos + 1)
        fpr_hi = (ops.fp_of_b.astype(float) + 1.0) / (n_neg + 1)

    ok_normal = (pi_hi * fnr_hi) <= alpha_missed + 1e-12
    ok_alert = ((1.0 - pi_lo) * fpr_hi) <= alpha_false_alert + 1e-12

    if not ok_normal[0]:
        return PolicyThreshold("crc_robust", ops.band, ops.fold, -1.0, 2.0, feasible=False,
                               calibration_n=ops.n, calibration_positives=n_pos,
                               infeasible_reason="missed-case budget unreachable at pi_hi")
    i = 0
    while i + 1 < g and ok_normal[i + 1]:
        i += 1

    if not ok_alert[g - 1]:
        return PolicyThreshold("crc_robust", ops.band, ops.fold, -1.0, 2.0, feasible=False,
                               calibration_n=ops.n, calibration_positives=n_pos,
                               infeasible_reason="false-alert budget unreachable at pi_lo")
    j = g - 1
    while j - 1 >= 0 and ok_alert[j - 1]:
        j -= 1

    if j < 1:
        return PolicyThreshold("crc_robust", ops.band, ops.fold, -1.0, 2.0, feasible=False,
                               calibration_n=ops.n, calibration_positives=n_pos,
                               infeasible_reason="no room between certified thresholds")
    i = min(i, j - 1)
    return PolicyThreshold(
        "crc_robust", ops.band, ops.fold,
        normal_threshold=float(ops.grid[i]), alert_threshold=float(ops.grid[j]),
        feasible=True, calibration_n=ops.n, calibration_positives=n_pos,
        npv_lcb=float(pi_hi * fnr_hi[i]), ppv_lcb=float((1.0 - pi_lo) * fpr_hi[j]),
    )


def certify_crc_and_apply(
    ops_by_band: Dict[str, Optional[OperatingPoints]],
    sub: pd.DataFrame,
    pi_lo: float, pi_hi: float,
    alpha_missed: float, alpha_false_alert: float, delta: float, bound: str = "crc",
) -> np.ndarray:
    by_band: Dict[str, PolicyThreshold] = {}
    for band in BANDS:
        ops = ops_by_band.get(band)
        by_band[band] = (
            PolicyThreshold("crc_robust", band, 0, -1.0, 2.0, feasible=False)
            if ops is None else
            select_crc_robust(ops, alpha_missed, alpha_false_alert, pi_lo, pi_hi, delta, bound)
        )
    return apply_policy(sub, by_band, "crc_robust")


def certify_and_apply(
    ops_by_band: Dict[str, Optional[OperatingPoints]],
    sub: pd.DataFrame,
    pi_lo: float,
    pi_hi: float,
    target_ppv: float,
    target_npv: float,
    costs: CostModel,
) -> np.ndarray:
    by_band: Dict[str, PolicyThreshold] = {}
    for band in BANDS:
        ops = ops_by_band.get(band)
        by_band[band] = (
            PolicyThreshold("ppv_npv_robust", band, 0, -1.0, 2.0, feasible=False)
            if ops is None else
            select_ppv_npv_robust(ops, target_ppv, target_npv, costs, pi_lo, pi_hi)
        )
    return apply_policy(sub, by_band, "ppv_npv_robust")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Self-certifying triage under prior shift.")
    p.add_argument("--run-root", default="outputs/paper_runs/pedrr_kfold_v3")
    p.add_argument("--out-dir", default="")
    p.add_argument("--calibration", choices=["oof", "val"], default="oof")
    p.add_argument("--grid-size", type=int, default=41)
    p.add_argument("--delta-prior", type=float, default=0.05,
                   help="Error budget for the prevalence interval")
    p.add_argument("--delta-cert", type=float, default=0.05,
                   help="Error budget for the class-conditional certificate")
    p.add_argument("--tau", type=float, default=-1.0,
                   help="Hard-call threshold for BBSE; <0 selects it on calibration by Youden's J")
    p.add_argument("--deploy-mult", default="1,3,9",
                   help="Deployment cohort size multipliers (1 = the 332-subject cohort)")
    p.add_argument("--target-ppv", type=float, default=0.60)
    p.add_argument("--target-npv", type=float, default=0.90)
    p.add_argument("--alpha-missed", type=float, default=0.02,
                   help="Budget for auto-cleared-while-tachypneic, per screened child")
    p.add_argument("--alpha-false-alert", type=float, default=0.15,
                   help="Budget for auto-alerted-while-healthy, per screened child")
    p.add_argument("--bound", choices=["crc", "cp"], default="crc",
                   help="crc: E[risk] <= alpha. cp: P(risk <= alpha) >= 1-delta (stronger, "
                        "far more conservative at this number of positives).")
    p.add_argument("--certificate", choices=["crc", "ppv_npv"], default="crc",
                   help="crc composes the risk-control certificate with the estimated prior")
    p.add_argument("--declared-max", type=float, default=0.50,
                   help="Upper end of the fixed declared range, the baseline to beat")
    p.add_argument("--prevalences", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    p.add_argument("--n-rep", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-age-fix", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / "deferral_policy" / run_root.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    costs = CostModel()
    age_fix = not args.no_age_fix
    prevalences = [float(x) for x in str(args.prevalences).split(",") if x.strip()]

    val_by_fold = load_by_fold(run_root, "val_eval", age_fix=age_fix)
    test_by_fold = load_by_fold(run_root, "test_eval", age_fix=age_fix)

    ops_by_fold, cal_by_fold = {}, {}
    for fold in sorted(test_by_fold):
        cal = calibration_for_fold(fold, val_by_fold, test_by_fold, args.calibration)
        cal_by_fold[fold] = cal
        ops_by_fold[fold] = build_operating_points(cal, fold, int(args.grid_size), float(args.delta_cert))

    print(f"Run root        : {run_root}")
    print(f"Error budget    : delta_prior={args.delta_prior} + delta_cert={args.delta_cert} "
          f"= {args.delta_prior + args.delta_cert:.2f} total")
    print(f"Targets         : PPV>={args.target_ppv:.2f}, NPV>={args.target_npv:.2f}")
    print(f"Baseline        : fixed declared range [0.10, {args.declared_max:.2f}]")
    print(f"Deployment sim  : within-band resampling, {args.n_rep} replicates per prevalence")

    tau = float(args.tau)
    if tau < 0:
        taus = [best_tau(cal_by_fold[f]) for f in sorted(cal_by_fold)]
        tau = float(np.mean(taus))
        print(f"BBSE threshold  : tau={tau:.3f} (Youden-optimal on calibration, per-fold {taus})")
    mults = [float(x) for x in str(args.deploy_mult).split(",") if x.strip()]
    base_n = sum(len(test_by_fold[f]) for f in test_by_fold)
    print(f"Cohort sizes    : {[int(round(m * base_n)) for m in mults]}\n")

    rows = []
    for mult in mults:
      for pi_true in prevalences:
        cover, self_ref, fixed_ref, oracle_ref = [], [], [], []
        self_npv, fixed_npv, oracle_npv = [], [], []
        self_hold, fixed_hold = [], []
        pi_hats, pi_his, pi_los, em_hats = [], [], [], []
        for _ in range(int(args.n_rep)):
            # Build one deployment cohort per fold, score it with that fold's certificate.
            tri_self, tri_fixed, tri_oracle, frames = [], [], [], []
            mu_calls = tp = n_pos = fp = n_neg = 0
            probs_all = []
            for fold in sorted(test_by_fold):
                test = test_by_fold[fold]
                idx = resample_to_prevalence(test, pi_true, rng, size_mult=mult)
                sub = test.iloc[idx].reset_index(drop=True)
                frames.append(sub)
                probs_all.append(sub["tachy_prob"].to_numpy(dtype=float))
                mu_calls += int((sub["tachy_prob"].to_numpy(dtype=float) >= tau).sum())
                a, b, c, d = source_rates(cal_by_fold[fold], tau)
                tp += a; n_pos += b; fp += c; n_neg += d
            deploy = pd.concat(frames, ignore_index=True)
            n_target = len(deploy)

            # --- estimate the prior from unlabelled scores only ---
            pi_lo, pi_hat, pi_hi = bbse_interval(mu_calls, n_target, tp, n_pos, fp, n_neg,
                                                 float(args.delta_prior))
            pi_src = n_pos / max(n_pos + n_neg, 1)
            pi_em = em_prior(np.concatenate(probs_all), pi_src)
            cover.append(pi_lo <= pi_true <= pi_hi)
            pi_hats.append(pi_hat); pi_his.append(pi_hi); pi_los.append(pi_lo); em_hats.append(pi_em)

            # --- certify at the estimated / fixed / oracle range ---
            for fold in sorted(test_by_fold):
                sub = frames[fold]
                if args.certificate == "crc":
                    def cert(lo, hi, _s=sub, _f=fold):
                        return certify_crc_and_apply(ops_by_fold[_f], _s, lo, hi,
                                                     float(args.alpha_missed),
                                                     float(args.alpha_false_alert),
                                                     float(args.delta_cert), args.bound)
                else:
                    def cert(lo, hi, _s=sub, _f=fold):
                        return certify_and_apply(ops_by_fold[_f], _s, lo, hi,
                                                 args.target_ppv, args.target_npv, costs)
                tri_self.append(cert(pi_lo, pi_hi))
                tri_fixed.append(cert(0.10, float(args.declared_max)))
                tri_oracle.append(cert(pi_true, pi_true))
            for name, tri, ref, npvs, holds in (
                ("self", tri_self, self_ref, self_npv, self_hold),
                ("fixed", tri_fixed, fixed_ref, fixed_npv, fixed_hold),
                ("oracle", tri_oracle, oracle_ref, oracle_npv, None),
            ):
                t = np.concatenate(tri)
                m = metrics_for_triage(deploy, t, name, "overall", costs)
                ref.append(m["refer_rate"])
                npvs.append(m["npv"])
                if holds is not None:
                    # Validate the quantity the certificate actually promised.
                    # The CRC certificate bounds the population missed-case risk,
                    # not NPV; checking NPV would score a promise never made.
                    if args.certificate == "crc":
                        realised = m["fn"] / max(m["n"], 1)
                        holds.append(bool(realised <= float(args.alpha_missed) + 1e-12))
                    else:
                        holds.append(bool(np.isnan(m["npv"]) or m["npv"] >= args.target_npv))

        def mean(x):
            a = np.asarray(x, dtype=float); a = a[np.isfinite(a)]
            return float(a.mean()) if a.size else float("nan")

        rows.append({
            "deploy_n": int(round(mult * base_n)), "size_mult": mult,
            "prevalence_true": pi_true,
            "pi_hat_bbse": mean(pi_hats), "pi_hi_bbse": mean(pi_his), "pi_lo_bbse": mean(pi_los),
            "interval_width": mean(pi_his) - mean(pi_los), "pi_hat_em": mean(em_hats),
            "interval_coverage": float(np.mean(cover)),
            "self_refer": mean(self_ref), "fixed_refer": mean(fixed_ref), "oracle_refer": mean(oracle_ref),
            "self_npv": mean(self_npv), "fixed_npv": mean(fixed_npv), "oracle_npv": mean(oracle_npv),
            "self_npv_holds": float(np.mean(self_hold)), "fixed_npv_holds": float(np.mean(fixed_hold)),
        })
        r = rows[-1]
        print(f"  N={r['deploy_n']:>5} pi={pi_true:.2f}  BBSE {r['pi_hat_bbse']:.3f} (hi {r['pi_hi_bbse']:.3f}, "
              f"cov {r['interval_coverage']:.2f})  EM {r['pi_hat_em']:.3f}  |  "
              f"refer self {r['self_refer']:.1%} vs fixed {r['fixed_refer']:.1%}  |  "
              f"guarantee holds self {r['self_npv_holds']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "self_certifying_triage.csv", index=False)

    base = df[df.size_mult == mults[0]]
    print("\nPrior estimation from unlabelled scores (cohort n=%d)" % int(base.deploy_n.iloc[0]))
    est = base[["prevalence_true", "pi_hat_bbse", "pi_hat_em", "pi_lo_bbse", "pi_hi_bbse",
                "interval_coverage"]].copy()
    est["bbse_error"] = (base["pi_hat_bbse"] - base["prevalence_true"]).abs()
    est["em_error"] = (base["pi_hat_em"] - base["prevalence_true"]).abs()
    print(est.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"  mean |error|: BBSE {est['bbse_error'].mean():.3f}   EM {est['em_error'].mean():.3f}")
    print(f"  interval coverage (nominal {1 - args.delta_prior:.2f}): "
          f"{df['interval_coverage'].mean():.3f} mean, {df['interval_coverage'].min():.3f} worst")

    print("\nHow the certificate sharpens as a site accumulates screens")
    piv = df.pivot_table(index="prevalence_true", columns="deploy_n",
                         values=["interval_width", "self_refer"])
    print(piv.round(3).to_string())

    print("\nWhat self-certification buys (referral rate at equal validity)")
    for n_dep in sorted(df.deploy_n.unique()):
        sub = df[df.deploy_n == n_dep]
        gain = (sub["fixed_refer"] - sub["self_refer"])
        print(f"  cohort n={n_dep:>5}: refer fixed {sub['fixed_refer'].mean():.1%} -> "
              f"self {sub['self_refer'].mean():.1%} (oracle {sub['oracle_refer'].mean():.1%}); "
              f"mean gain {gain.mean() * 100:+.1f} pts, best {gain.max() * 100:+.1f}; "
              f"guarantee met in {sub['self_npv_holds'].mean():.3f} of replicates "
              f"(fixed: {sub['fixed_npv_holds'].mean():.3f})")

    print(f"\nWrote {out_dir / 'self_certifying_triage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
