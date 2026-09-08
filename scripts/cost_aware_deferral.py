"""Cost-aware and PPV/NPV-guaranteed deferral for PRISM-Breath.

This is a CPU-only post-hoc analysis over existing PRISM-Breath prediction
CSVs. It does not retrain a model or run inference. It compares three deferral
policies for the tachypnea screening decision:

  coverage_conformal   Mondrian (age-conditional) interval coverage, the policy
                       used in the main paper. Guarantees that the predicted RR
                       interval covers the true RR at rate 1-alpha; says nothing
                       about the reliability of the resulting ALERT/NORMAL call.

  cost_aware           Chooses the two decision thresholds to minimise an explicit
                       expected clinical cost (missed tachypnea >> false alert >
                       referral) rather than fixing a coverage target.

  ppv_npv_plugin       Chooses thresholds so that the *empirical* PPV of the ALERT
                       group and NPV of the NORMAL group meet targets on the
                       calibration set. This is the naive version and is included
                       to show that a plug-in guarantee does not transfer.

  ppv_npv_ltt          The same targets certified with finite-sample validity via
                       Learn-then-Test: a threshold pair is admissible only if the
                       Bonferroni-corrected one-sided Clopper-Pearson lower
                       confidence bound on both PPV and NPV clears the target at
                       level delta. If no pair is certifiable the band is REFERRED
                       rather than silently auto-decided.

Calibration is leakage-safe with respect to the evaluated test fold:

  --calibration oof    (default) For test fold i, calibrate on the pooled
                       out-of-fold predictions of folds j != i. Test folds are
                       subject-disjoint, so the calibration subjects are disjoint
                       from fold i's test subjects by construction. This is the
                       cross-conformal / CV+ construction; the calibration
                       predictions come from sibling fold models rather than the
                       deployed one.
  --calibration val    Strict split-conformal: fold i's own validation split
                       (n=39 here). Subject-disjoint from fold i's test, but far
                       too small to certify anything -- reported as the
                       sample-size sensitivity analysis.

Run from the repository root, for example:

    python scripts/cost_aware_deferral.py --run-root outputs/paper_runs/pedrr_kfold_v3
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist


BANDS: Tuple[str, ...] = ("<2mo", "2-12mo", "12-60mo")
ALL_BANDS: Tuple[str, ...] = ("overall",) + BANDS
SHORT_STATUS = {0: "NORMAL", 1: "ALERT", 2: "REFER"}
POLICIES: Tuple[str, ...] = (
    "coverage_conformal",
    "cost_aware",
    "ppv_npv_plugin",
    "ppv_npv_ltt",
    "ppv_npv_robust",
)


# --------------------------------------------------------------------------
# Labels and bands
# --------------------------------------------------------------------------

def who_threshold(age_months: float) -> float:
    """WHO age-banded tachypnea threshold (breaths/min).

    Defined for under-fives only; the IMCI fast-breathing thresholds do not
    extend past 60 months. Callers must screen out older subjects rather than
    silently receiving an out-of-scope cutoff.
    """
    if age_months < 2:
        return 60.0
    if age_months < 12:
        return 50.0
    if age_months <= 60:
        return 40.0
    raise ValueError(f"WHO tachypnea thresholds are undefined above 60 months (got {age_months}).")


def age_band(age_months: float) -> str:
    """Age band matching who_threshold's domain exactly."""
    if age_months < 2:
        return "<2mo"
    if age_months < 12:
        return "2-12mo"
    if age_months <= 60:
        return "12-60mo"
    raise ValueError(f"Age band is undefined above 60 months (got {age_months}).")


# --------------------------------------------------------------------------
# Finite-sample bounds
# --------------------------------------------------------------------------

def clopper_pearson_lower(successes: int, trials: int, delta: float) -> float:
    """One-sided exact (Clopper-Pearson) lower confidence bound at level delta.

    P(p >= bound) >= 1 - delta. Returns 0.0 when there are no successes and
    nan when there are no trials (the quantity is undefined, not achieved).
    """
    if trials <= 0:
        return float("nan")
    if successes <= 0:
        return 0.0
    if successes >= trials:
        return float(delta ** (1.0 / trials))
    return float(beta_dist.ppf(delta, successes, trials - successes + 1))


def clopper_pearson_upper(successes: int, trials: int, delta: float) -> float:
    """One-sided exact (Clopper-Pearson) upper confidence bound at level delta."""
    if trials <= 0:
        return float("nan")
    if successes >= trials:
        return 1.0
    if successes <= 0:
        return float(1.0 - delta ** (1.0 / trials))
    return float(beta_dist.ppf(1.0 - delta, successes + 1, trials - successes))


def _cp_upper_vec(successes: np.ndarray, trials: np.ndarray, delta: float) -> np.ndarray:
    out = np.full(successes.shape, np.nan, dtype=float)
    empty = trials <= 0
    full = (~empty) & (successes >= trials)
    zero = (~empty) & (successes <= 0)
    mid = (~empty) & (~full) & (~zero)
    out[full] = 1.0
    out[zero] = 1.0 - delta ** (1.0 / np.maximum(trials[zero], 1))
    if mid.any():
        out[mid] = beta_dist.ppf(1.0 - delta, successes[mid] + 1, trials[mid] - successes[mid])
    return out


def clopper_pearson_interval(successes: int, trials: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Two-sided exact binomial confidence interval, for reporting."""
    if trials <= 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if successes == 0 else float(beta_dist.ppf(alpha / 2, successes, trials - successes + 1))
    hi = 1.0 if successes == trials else float(beta_dist.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return (lo, hi)


def min_n_to_certify(p_hat: float, target: float, delta: float, n_max: int = 20000) -> float:
    """Smallest calibration n whose CP lower bound clears `target`, at rate p_hat.

    Answers "how much calibration data would this guarantee have needed?".
    """
    if not np.isfinite(p_hat) or p_hat <= target:
        return float("inf")
    n = 1
    while n <= n_max:
        k = int(math.floor(p_hat * n))
        if clopper_pearson_lower(k, n, delta) >= target:
            return float(n)
        n += 1
    return float("inf")


def conformal_qhat(errors: Iterable[float], alpha: float) -> float:
    vals = np.asarray(list(errors), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    rank = int(math.ceil((vals.size + 1) * (1.0 - alpha)))
    rank = max(1, min(rank, vals.size))
    return float(np.sort(vals)[rank - 1])


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def infer_fold(path: Path) -> int:
    match = re.search(r"fold(\d+)", str(path))
    if not match:
        raise ValueError(f"Could not infer fold number from path: {path}")
    return int(match.group(1))


def _prepare(df: pd.DataFrame, recompute_labels: bool) -> pd.DataFrame:
    required = ["subject_id", "video_id", "age_months", "gt_rr_bpm", "pred_rr_bpm", "tachy_prob"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"predictions are missing required columns: {missing}")

    for col in ["age_months", "gt_rr_bpm", "pred_rr_bpm", "tachy_prob"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["age_months", "gt_rr_bpm", "pred_rr_bpm", "tachy_prob"]).copy()

    over = df["age_months"] > 60
    if over.any():
        print(f"  warning: dropping {int(over.sum())} subject(s) older than 60 months; "
              f"WHO fast-breathing thresholds do not cover them.")
        df = df[~over].copy()

    if recompute_labels or "threshold_bpm" not in df.columns:
        df["threshold_bpm"] = df["age_months"].map(who_threshold)
    else:
        df["threshold_bpm"] = pd.to_numeric(df["threshold_bpm"], errors="coerce")
        df["threshold_bpm"] = df["threshold_bpm"].fillna(df["age_months"].map(who_threshold))

    df["gt_tachy"] = (df["gt_rr_bpm"] >= df["threshold_bpm"]).astype(int)
    df["band"] = df["age_months"].map(age_band)
    df["clip_key"] = df["subject_id"].astype(str) + "_" + df["video_id"].astype(str)

    grouped = df.groupby(["fold", "clip_key"], sort=False)
    out = grouped.agg(
        subject_id=("subject_id", "first"),
        video_id=("video_id", "first"),
        age_months=("age_months", "first"),
        band=("band", "first"),
        threshold_bpm=("threshold_bpm", "first"),
        gt_rr_bpm=("gt_rr_bpm", "mean"),
        pred_rr_bpm=("pred_rr_bpm", "mean"),
        tachy_prob=("tachy_prob", "mean"),
        gt_tachy=("gt_tachy", "max"),
        source_csv=("source_csv", "first"),
    ).reset_index()
    out["residual_abs"] = (out["pred_rr_bpm"] - out["gt_rr_bpm"]).abs()
    return out


# Run carrying the corrected ages: nine neonates were stored with a stale
# age_months=6 and so were scored against the 2-12mo WHO threshold instead of
# the neonatal one.
AGE_REFERENCE_RUN = Path("outputs/paper_runs/pedrr_kfold_flowonly_agefix")


def corrected_age_map(reference: Path = AGE_REFERENCE_RUN) -> Dict[str, float]:
    paths = sorted(glob.glob(str(reference / "fold*" / "test_eval" / "prism_breath_predictions.csv")))
    if not paths:
        raise FileNotFoundError(f"Age reference run not found: {reference}")
    ref = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return dict(zip(ref["subject_id"].astype(str), ref["age_months"].astype(float)))


def load_by_fold(run_root: Path, split: str, recompute_labels: bool = True,
                 age_fix: bool = True) -> Dict[int, pd.DataFrame]:
    pattern = run_root / "fold*" / split / "prism_breath_predictions.csv"
    paths = sorted(Path(p) for p in glob.glob(str(pattern)))
    if not paths:
        raise FileNotFoundError(f"No prediction CSVs found at {pattern}")
    fix = corrected_age_map() if age_fix else {}
    out: Dict[int, pd.DataFrame] = {}
    for path in paths:
        df = pd.read_csv(path)
        df["fold"] = infer_fold(path)
        df["source_csv"] = str(path)
        if fix:
            mapped = df["subject_id"].astype(str).map(fix)
            df["age_months"] = mapped.fillna(df["age_months"])
        out[infer_fold(path)] = _prepare(df, recompute_labels)
    return out


def assert_no_leakage(cal: pd.DataFrame, test: pd.DataFrame, fold: int) -> None:
    overlap = set(cal["subject_id"].astype(str)) & set(test["subject_id"].astype(str))
    if overlap:
        raise SystemExit(
            f"Calibration/test subject leakage on fold {fold}: {len(overlap)} shared subjects "
            f"(e.g. {sorted(overlap)[:5]}). Refusing to report a guarantee fitted on test subjects."
        )


def calibration_for_fold(
    fold: int,
    val_by_fold: Dict[int, pd.DataFrame],
    test_by_fold: Dict[int, pd.DataFrame],
    scheme: str,
) -> pd.DataFrame:
    if scheme == "val":
        cal = val_by_fold[fold].copy()
    elif scheme == "oof":
        others = [test_by_fold[g] for g in sorted(test_by_fold) if g != fold]
        cal = pd.concat(others, ignore_index=True)
    else:
        raise SystemExit(f"Unknown --calibration scheme: {scheme}")
    assert_no_leakage(cal, test_by_fold[fold], fold)
    return cal


# --------------------------------------------------------------------------
# Decision rules and metrics
# --------------------------------------------------------------------------

def triage_from_interval(pred_rr: np.ndarray, threshold: np.ndarray, qhat: float) -> np.ndarray:
    lower = pred_rr - qhat
    upper = pred_rr + qhat
    return np.where(upper < threshold, 0, np.where(lower >= threshold, 1, 2)).astype(int)


def triage_from_thresholds(scores: np.ndarray, normal_threshold: float, alert_threshold: float) -> np.ndarray:
    triage = np.full(scores.shape, 2, dtype=int)
    triage[scores <= normal_threshold] = 0
    triage[scores >= alert_threshold] = 1
    return triage


def confusion_counts(y_true: np.ndarray, triage: np.ndarray) -> Dict[str, int]:
    y = np.asarray(y_true, dtype=int)
    t = np.asarray(triage, dtype=int)
    pred_pos = t == 1
    pred_neg = t == 0
    return {
        "n": int(y.size),
        "positives": int((y == 1).sum()),
        "retained_n": int((t != 2).sum()),
        "refer_n": int((t == 2).sum()),
        "tp": int((pred_pos & (y == 1)).sum()),
        "fp": int((pred_pos & (y == 0)).sum()),
        "tn": int((pred_neg & (y == 0)).sum()),
        "fn": int((pred_neg & (y == 1)).sum()),
    }


def ratio(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def metrics_for_triage(
    df: pd.DataFrame,
    triage: np.ndarray,
    policy: str,
    band: str,
    costs: "CostModel",
) -> Dict[str, float | int | str]:
    counts = confusion_counts(df["gt_tachy"].to_numpy(dtype=int), triage)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    refer_n = counts["refer_n"]
    total_cost = costs.miss * fn + costs.false_alert * fp + costs.referral * refer_n
    n = max(counts["n"], 1)
    return {
        "policy": policy,
        "band": band,
        "n": counts["n"],
        "positives": counts["positives"],
        "refer_rate": refer_n / n,
        "retained_n": counts["retained_n"],
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "false_omission_rate": ratio(fn, tn + fn),
        "alert_n": tp + fp,
        "normal_n": tn + fn,
        "false_normals": fn,
        "false_alerts": fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "refer_n": refer_n,
        "expected_cost": total_cost / n,
        "total_cost": total_cost,
    }


@dataclass(frozen=True)
class CostModel:
    """Relative clinical costs. A missed tachypnea case dominates.

    `referral` sits below `false_alert`: escalating a case for a second look
    (repeat count, radar, clinician) is cheaper than an unnecessary pneumonia
    workup. When referral == false_alert the referral option is weakly dominated
    and the cost-aware policy degenerates to never deferring.
    """

    miss: float = 10.0
    false_alert: float = 1.0
    referral: float = 0.3


@dataclass
class PolicyThreshold:
    policy: str
    band: str
    fold: int
    normal_threshold: float = float("nan")
    alert_threshold: float = float("nan")
    qhat: float = float("nan")
    feasible: bool = True
    calibration_n: int = 0
    calibration_positives: int = 0
    calibration_refer_rate: float = float("nan")
    calibration_ppv: float = float("nan")
    calibration_npv: float = float("nan")
    calibration_cost: float = float("nan")
    ppv_lcb: float = float("nan")
    npv_lcb: float = float("nan")
    n_candidates: int = 0
    delta_effective: float = float("nan")
    infeasible_reason: str = ""


# --------------------------------------------------------------------------
# Policy fitting
# --------------------------------------------------------------------------

def prespecified_grid(grid_size: int) -> np.ndarray:
    """Grid fixed in advance, independent of the calibration data.

    Learn-then-Test validity requires the candidate set to be pre-specified, so
    this deliberately does NOT mix in observed score values.
    """
    return np.linspace(0.0, 1.0, int(grid_size))


def iter_threshold_pairs(vals: Sequence[float]):
    for normal_threshold in vals:
        for alert_threshold in vals:
            if normal_threshold < alert_threshold:
                yield float(normal_threshold), float(alert_threshold)


def n_threshold_pairs(grid_size: int) -> int:
    g = int(grid_size)
    return g * (g - 1) // 2


@dataclass
class OperatingPoints:
    """Every pre-specified (t_normal, t_alert) pair scored on one calibration set.

    The NORMAL group depends only on t_normal and the ALERT group only on
    t_alert, so all G(G-1)/2 pairs are obtained from two cumulative counts
    rather than a re-scan per pair. Confidence bounds are evaluated once here
    and reused for every guarantee target.
    """

    band: str
    fold: int
    n: int
    positives: int
    t_normal: np.ndarray
    t_alert: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    tn: np.ndarray
    fn: np.ndarray
    refer_n: np.ndarray
    ppv: np.ndarray
    npv: np.ndarray
    ppv_lcb: np.ndarray
    npv_lcb: np.ndarray
    delta_effective: float
    # Per-threshold marginals, used by the fixed-sequence procedure. The ALERT
    # group depends only on t_alert and the NORMAL group only on t_normal, so
    # the two guarantees can be certified along two separate ordered families
    # instead of one 2-D grid.
    grid: np.ndarray = field(default_factory=lambda: np.empty(0))
    tp_of_b: np.ndarray = field(default_factory=lambda: np.empty(0))
    fp_of_b: np.ndarray = field(default_factory=lambda: np.empty(0))
    tn_of_a: np.ndarray = field(default_factory=lambda: np.empty(0))
    fn_of_a: np.ndarray = field(default_factory=lambda: np.empty(0))
    ppv_lcb_seq: np.ndarray = field(default_factory=lambda: np.empty(0))
    npv_lcb_seq: np.ndarray = field(default_factory=lambda: np.empty(0))
    pair_index: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    # Class-conditional bounds. These are prevalence-free, so a guarantee built
    # on them can be certified for a whole range of deployment prevalences
    # instead of only the prevalence the calibration set happened to have.
    sens_alert_lcb: np.ndarray = field(default_factory=lambda: np.empty(0))
    fpr_alert_ucb: np.ndarray = field(default_factory=lambda: np.empty(0))
    spec_normal_lcb: np.ndarray = field(default_factory=lambda: np.empty(0))
    fnr_normal_ucb: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def n_alert(self) -> np.ndarray:
        return self.tp + self.fp

    @property
    def n_normal(self) -> np.ndarray:
        return self.tn + self.fn

    @property
    def refer_rate(self) -> np.ndarray:
        return self.refer_n / max(self.n, 1)


def _cp_lower_vec(successes: np.ndarray, trials: np.ndarray, delta: float) -> np.ndarray:
    """Vectorised one-sided Clopper-Pearson lower bound."""
    out = np.full(successes.shape, np.nan, dtype=float)
    empty = trials <= 0
    zero = (~empty) & (successes <= 0)
    full = (~empty) & (successes >= trials)
    mid = (~empty) & (~zero) & (~full)
    out[zero] = 0.0
    out[full] = delta ** (1.0 / np.maximum(trials[full], 1))
    if mid.any():
        out[mid] = beta_dist.ppf(delta, successes[mid], trials[mid] - successes[mid] + 1)
    return out


def enumerate_operating_points(
    cal: pd.DataFrame, band: str, fold: int, grid_size: int, delta_effective: float,
    delta_sequential: float = float("nan"),
) -> OperatingPoints:
    grid = prespecified_grid(grid_size)
    y = cal["gt_tachy"].to_numpy(dtype=int)
    s = cal["tachy_prob"].to_numpy(dtype=float)
    n = int(y.size)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    # Counts as a function of a single threshold.
    le = s[None, :] <= grid[:, None]          # NORMAL side, indexed by t_normal
    ge = s[None, :] >= grid[:, None]          # ALERT side, indexed by t_alert
    tn_of_a = (le & (y == 0)[None, :]).sum(axis=1)
    fn_of_a = (le & (y == 1)[None, :]).sum(axis=1)
    tp_of_b = (ge & (y == 1)[None, :]).sum(axis=1)
    fp_of_b = (ge & (y == 0)[None, :]).sum(axis=1)

    ia, ib = np.triu_indices(len(grid), k=1)  # strictly t_normal < t_alert
    tn, fn = tn_of_a[ia], fn_of_a[ia]
    tp, fp = tp_of_b[ib], fp_of_b[ib]
    refer_n = n - (tn + fn + tp + fp)

    n_alert, n_normal = tp + fp, tn + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        ppv = np.where(n_alert > 0, tp / np.maximum(n_alert, 1), np.nan)
        npv = np.where(n_normal > 0, tn / np.maximum(n_normal, 1), np.nan)

    pair_index = np.full((len(grid), len(grid)), -1, dtype=int)
    pair_index[ia, ib] = np.arange(ia.size)

    return OperatingPoints(
        band=band, fold=fold, n=n, positives=int(y.sum()),
        t_normal=grid[ia], t_alert=grid[ib],
        tp=tp, fp=fp, tn=tn, fn=fn, refer_n=refer_n,
        ppv=ppv, npv=npv,
        ppv_lcb=_cp_lower_vec(tp, n_alert, delta_effective),
        npv_lcb=_cp_lower_vec(tn, n_normal, delta_effective),
        delta_effective=delta_effective,
        grid=grid,
        tp_of_b=tp_of_b, fp_of_b=fp_of_b, tn_of_a=tn_of_a, fn_of_a=fn_of_a,
        ppv_lcb_seq=_cp_lower_vec(tp_of_b, tp_of_b + fp_of_b, delta_sequential),
        npv_lcb_seq=_cp_lower_vec(tn_of_a, tn_of_a + fn_of_a, delta_sequential),
        pair_index=pair_index,
        sens_alert_lcb=_cp_lower_vec(tp_of_b, np.full_like(tp_of_b, n_pos), delta_sequential / 2),
        fpr_alert_ucb=_cp_upper_vec(fp_of_b, np.full_like(fp_of_b, n_neg), delta_sequential / 2),
        spec_normal_lcb=_cp_lower_vec(tn_of_a, np.full_like(tn_of_a, n_neg), delta_sequential / 2),
        fnr_normal_ucb=_cp_upper_vec(fn_of_a, np.full_like(fn_of_a, n_pos), delta_sequential / 2),
    )


def fit_coverage_conformal(cal: pd.DataFrame, band: str, fold: int, alpha: float) -> PolicyThreshold:
    qhat = conformal_qhat(cal["residual_abs"], alpha)
    return PolicyThreshold(
        policy="coverage_conformal",
        band=band,
        fold=fold,
        qhat=qhat,
        feasible=bool(np.isfinite(qhat)),
        calibration_n=int(len(cal)),
        calibration_positives=int(cal["gt_tachy"].sum()),
    )


def _threshold_from_ops(ops: OperatingPoints, idx: int, policy: str, costs: CostModel) -> PolicyThreshold:
    total_cost = costs.miss * ops.fn[idx] + costs.false_alert * ops.fp[idx] + costs.referral * ops.refer_n[idx]
    return PolicyThreshold(
        policy=policy,
        band=ops.band,
        fold=ops.fold,
        normal_threshold=float(ops.t_normal[idx]),
        alert_threshold=float(ops.t_alert[idx]),
        feasible=True,
        calibration_n=ops.n,
        calibration_positives=ops.positives,
        calibration_refer_rate=float(ops.refer_n[idx] / max(ops.n, 1)),
        calibration_ppv=float(ops.ppv[idx]),
        calibration_npv=float(ops.npv[idx]),
        calibration_cost=float(total_cost / max(ops.n, 1)),
        ppv_lcb=float(ops.ppv_lcb[idx]),
        npv_lcb=float(ops.npv_lcb[idx]),
        n_candidates=int(ops.t_normal.size),
        delta_effective=ops.delta_effective,
    )


def _infeasible(ops: OperatingPoints, policy: str, reason: str) -> PolicyThreshold:
    """Refer the whole band rather than auto-deciding without a basis."""
    return PolicyThreshold(
        policy=policy, band=ops.band, fold=ops.fold,
        normal_threshold=-1.0, alert_threshold=2.0, feasible=False,
        calibration_n=ops.n, calibration_positives=ops.positives,
        calibration_refer_rate=1.0, n_candidates=int(ops.t_normal.size),
        delta_effective=ops.delta_effective, infeasible_reason=reason,
    )


def _lexsort_pick(keys: Sequence[np.ndarray], mask: np.ndarray) -> Optional[int]:
    """Index of the mask-admissible point that is smallest under `keys` in order."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    # np.lexsort takes the last key as primary, so reverse.
    order = np.lexsort(tuple(k[idx] for k in reversed(keys)))
    return int(idx[order[0]])


def select_cost_aware(ops: OperatingPoints, costs: CostModel, max_refer_rate: float) -> PolicyThreshold:
    total_cost = costs.miss * ops.fn + costs.false_alert * ops.fp + costs.referral * ops.refer_n
    mask = ops.refer_rate <= max_refer_rate + 1e-12
    idx = _lexsort_pick([total_cost, ops.fn, ops.refer_n, ops.fp], mask)
    if idx is None:
        return _infeasible(ops, "cost_aware", f"no point within referral budget {max_refer_rate:.0%}")
    return _threshold_from_ops(ops, idx, "cost_aware", costs)


def select_ppv_npv_sequential(
    ops: OperatingPoints,
    target_ppv: float,
    target_npv: float,
    costs: CostModel,
    max_refer_rate: float = 1.0,
) -> PolicyThreshold:
    """Fixed-sequence Learn-then-Test certification.

    Bonferroni over all G(G-1)/2 threshold pairs is needlessly conservative: at
    this prevalence it certifies nothing. Fixed-sequence testing controls the
    family-wise error without that penalty by walking a pre-specified ordering
    and stopping at the first hypothesis it cannot reject.

    The ALERT group depends only on t_alert and the NORMAL group only on
    t_normal, so the two guarantees form two independent ordered families; delta
    is split evenly between them. Each family starts at its most conservative
    threshold and relaxes while the bound still clears the target, so the chosen
    point is the largest certified decision group. An empty decision group
    satisfies its guarantee vacuously (nothing is decided, so nothing can be
    wrong) and does not stop the walk.
    """
    policy = "ppv_npv_ltt"
    g = int(ops.grid.size)
    if g == 0:
        return _infeasible(ops, policy, "no operating points")

    n_alert_of_b = ops.tp_of_b + ops.fp_of_b
    n_normal_of_a = ops.tn_of_a + ops.fn_of_a
    passes_alert = (n_alert_of_b == 0) | (np.nan_to_num(ops.ppv_lcb_seq, nan=-1.0) + 1e-12 >= target_ppv)
    passes_normal = (n_normal_of_a == 0) | (np.nan_to_num(ops.npv_lcb_seq, nan=-1.0) + 1e-12 >= target_npv)

    if not passes_alert[g - 1]:
        return _infeasible(ops, policy, "PPV not certifiable at any alert threshold")
    j = g - 1
    while j - 1 >= 0 and passes_alert[j - 1]:
        j -= 1

    if not passes_normal[0]:
        return _infeasible(ops, policy, "NPV not certifiable at any normal threshold")
    i = 0
    while i + 1 < g and passes_normal[i + 1]:
        i += 1

    # The two groups must not overlap. Shrinking the NORMAL group is safe: every
    # index below i was already certified on the same walk.
    if j < 1:
        return _infeasible(ops, policy, "certified alert threshold leaves no room for a normal group")
    i = min(i, j - 1)

    idx = int(ops.pair_index[i, j])
    if idx < 0:
        return _infeasible(ops, policy, "no valid threshold pair after overlap resolution")
    if ops.n_alert[idx] == 0 and ops.n_normal[idx] == 0:
        # Certified vacuously by deciding nothing; report it as what it is.
        return _infeasible(ops, policy, "certified point makes no ALERT or NORMAL decisions")
    if ops.refer_rate[idx] > max_refer_rate + 1e-12:
        return _infeasible(ops, policy, f"certified point exceeds referral budget {max_refer_rate:.0%}")

    picked = _threshold_from_ops(ops, idx, policy, costs)
    picked.ppv_lcb = float(ops.ppv_lcb_seq[j])
    picked.npv_lcb = float(ops.npv_lcb_seq[i])
    picked.n_candidates = 2 * g
    return picked


def select_ppv_npv_robust(
    ops: OperatingPoints,
    target_ppv: float,
    target_npv: float,
    costs: CostModel,
    prevalence_min: float,
    prevalence_max: float,
    max_refer_rate: float = 1.0,
) -> PolicyThreshold:
    """Certify PPV/NPV across a range of deployment prevalences, not just one.

    A certificate fitted at the calibration prevalence says nothing once
    prevalence moves, and PPV/NPV move in opposite directions: PPV degrades as
    prevalence falls, NPV as it rises. Certifying on the class-conditional rates
    instead -- which are invariant under label shift -- and evaluating them at
    the worst prevalence in the deployment range gives a guarantee that survives
    the shift.

        PPV(pi) = pi*sens / (pi*sens + (1-pi)*fpr)        worst at pi_min
        NPV(pi) = (1-pi)*spec / ((1-pi)*spec + pi*fnr)    worst at pi_max

    Substituting a lower bound on sens/spec and an upper bound on fpr/fnr makes
    each expression a valid lower bound on the true value. delta is split four
    ways: two ordered families, two bounds each.
    """
    policy = "ppv_npv_robust"
    g = int(ops.grid.size)
    if g == 0:
        return _infeasible(ops, policy, "no operating points")

    pi_lo, pi_hi = float(prevalence_min), float(prevalence_max)
    n_alert_of_b = ops.tp_of_b + ops.fp_of_b
    n_normal_of_a = ops.tn_of_a + ops.fn_of_a

    sens_lo = np.nan_to_num(ops.sens_alert_lcb, nan=0.0)
    fpr_hi = np.nan_to_num(ops.fpr_alert_ucb, nan=1.0)
    spec_lo = np.nan_to_num(ops.spec_normal_lcb, nan=0.0)
    fnr_hi = np.nan_to_num(ops.fnr_normal_ucb, nan=1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ppv_den = pi_lo * sens_lo + (1 - pi_lo) * fpr_hi
        ppv_bound = np.where(ppv_den > 0, pi_lo * sens_lo / np.maximum(ppv_den, 1e-12), np.nan)
        npv_den = (1 - pi_hi) * spec_lo + pi_hi * fnr_hi
        npv_bound = np.where(npv_den > 0, (1 - pi_hi) * spec_lo / np.maximum(npv_den, 1e-12), np.nan)

    passes_alert = (n_alert_of_b == 0) | (np.nan_to_num(ppv_bound, nan=-1.0) + 1e-12 >= target_ppv)
    passes_normal = (n_normal_of_a == 0) | (np.nan_to_num(npv_bound, nan=-1.0) + 1e-12 >= target_npv)

    if not passes_alert[g - 1]:
        return _infeasible(ops, policy, f"PPV not certifiable at prevalence {pi_lo:.2f}")
    j = g - 1
    while j - 1 >= 0 and passes_alert[j - 1]:
        j -= 1

    if not passes_normal[0]:
        return _infeasible(ops, policy, f"NPV not certifiable at prevalence {pi_hi:.2f}")
    i = 0
    while i + 1 < g and passes_normal[i + 1]:
        i += 1

    if j < 1:
        return _infeasible(ops, policy, "certified alert threshold leaves no room for a normal group")
    i = min(i, j - 1)

    idx = int(ops.pair_index[i, j])
    if idx < 0:
        return _infeasible(ops, policy, "no valid threshold pair after overlap resolution")
    if ops.n_alert[idx] == 0 and ops.n_normal[idx] == 0:
        return _infeasible(ops, policy, "certified point makes no ALERT or NORMAL decisions")
    if ops.refer_rate[idx] > max_refer_rate + 1e-12:
        return _infeasible(ops, policy, f"certified point exceeds referral budget {max_refer_rate:.0%}")

    picked = _threshold_from_ops(ops, idx, policy, costs)
    picked.ppv_lcb = float(ppv_bound[j])
    picked.npv_lcb = float(npv_bound[i])
    picked.n_candidates = 2 * g
    return picked


def select_ppv_npv(
    ops: OperatingPoints,
    target_ppv: float,
    target_npv: float,
    certify: bool,
    costs: CostModel,
    max_refer_rate: float = 1.0,
    method: str = "fixed-sequence",
) -> PolicyThreshold:
    """Pick the least-deferring admissible point.

    certify=False accepts a pair on its empirical PPV/NPV (plug-in).
    certify=True requires a finite-sample certificate, via either fixed-sequence
    testing (default) or a Bonferroni correction over the full threshold grid.
    """
    if certify and method == "fixed-sequence":
        return select_ppv_npv_sequential(ops, target_ppv, target_npv, costs, max_refer_rate)

    policy = "ppv_npv_ltt" if certify else "ppv_npv_plugin"
    ppv_stat = ops.ppv_lcb if certify else ops.ppv
    npv_stat = ops.npv_lcb if certify else ops.npv

    # A guarantee needs at least one ALERT and one NORMAL decision; otherwise
    # PPV/NPV are undefined rather than achieved.
    decidable = (ops.n_alert > 0) & (ops.n_normal > 0)
    if not decidable.any():
        return _infeasible(ops, policy, "no ALERT/NORMAL split possible")

    meets_ppv = (np.nan_to_num(ppv_stat, nan=-1.0) + 1e-12) >= target_ppv
    meets_npv = (np.nan_to_num(npv_stat, nan=-1.0) + 1e-12) >= target_npv
    within_budget = ops.refer_rate <= (max_refer_rate + 1e-12)
    admissible = decidable & meets_ppv & meets_npv & within_budget

    idx = _lexsort_pick([ops.refer_n, ops.fn, ops.fp], admissible)
    if idx is None:
        reason = ("no threshold pair certifiable at target" if certify
                  else "no threshold pair met empirical target")
        return _infeasible(ops, policy, reason)
    return _threshold_from_ops(ops, idx, policy, costs)


def build_operating_points(
    cal_df: pd.DataFrame, fold: int, grid_size: int, delta: float
) -> Dict[str, Optional[OperatingPoints]]:
    delta_eff = delta / max(n_threshold_pairs(grid_size), 1)   # Bonferroni over pairs
    delta_seq = delta / 2.0                                    # split across the two ordered families
    out: Dict[str, Optional[OperatingPoints]] = {}
    for band in BANDS:
        cal = cal_df[cal_df["band"] == band]
        out[band] = None if cal.empty else enumerate_operating_points(
            cal, band, fold, grid_size, delta_eff, delta_seq
        )
    return out


def fit_policies_for_fold(
    cal_df: pd.DataFrame,
    fold: int,
    alpha: float,
    grid_size: int,
    target_ppv: float,
    target_npv: float,
    costs: CostModel,
    delta: float,
    max_refer_rate: float = 1.0,
    ops_by_band: Optional[Dict[str, Optional[OperatingPoints]]] = None,
    method: str = "fixed-sequence",
    prevalence_range: Tuple[float, float] = (0.10, 0.50),
) -> List[PolicyThreshold]:
    if ops_by_band is None:
        ops_by_band = build_operating_points(cal_df, fold, grid_size, delta)
    thresholds: List[PolicyThreshold] = []
    for band in BANDS:
        ops = ops_by_band[band]
        if ops is None:
            for policy in POLICIES:
                thresholds.append(
                    PolicyThreshold(policy, band, fold, -1.0, 2.0, feasible=False,
                                    infeasible_reason="no calibration data in band")
                )
            continue
        cal = cal_df[cal_df["band"] == band]
        thresholds.append(fit_coverage_conformal(cal, band, fold, alpha))
        thresholds.append(select_cost_aware(ops, costs, max_refer_rate))
        thresholds.append(select_ppv_npv(ops, target_ppv, target_npv, False, costs, max_refer_rate, method))
        thresholds.append(select_ppv_npv(ops, target_ppv, target_npv, True, costs, max_refer_rate, method))
        thresholds.append(select_ppv_npv_robust(ops, target_ppv, target_npv, costs,
                                                prevalence_range[0], prevalence_range[1], max_refer_rate))
    return thresholds


def apply_policy(df: pd.DataFrame, by_band: Dict[str, PolicyThreshold], policy: str) -> np.ndarray:
    triage = np.full(len(df), 2, dtype=int)
    for band, threshold in by_band.items():
        mask = (df["band"] == band).to_numpy()
        if not mask.any():
            continue
        if not threshold.feasible:
            triage[mask] = 2
        elif policy == "coverage_conformal":
            triage[mask] = triage_from_interval(
                df.loc[mask, "pred_rr_bpm"].to_numpy(dtype=float),
                df.loc[mask, "threshold_bpm"].to_numpy(dtype=float),
                threshold.qhat,
            )
        else:
            triage[mask] = triage_from_thresholds(
                df.loc[mask, "tachy_prob"].to_numpy(dtype=float),
                threshold.normal_threshold,
                threshold.alert_threshold,
            )
    return triage


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def evaluate_policy(test_df: pd.DataFrame, triage: np.ndarray, policy: str, costs: CostModel) -> List[Dict]:
    rows = []
    for band in ALL_BANDS:
        if band == "overall":
            sub, sub_triage = test_df, triage
        else:
            mask = (test_df["band"] == band).to_numpy()
            sub, sub_triage = test_df[mask], triage[mask]
        if sub.empty:
            continue
        rows.append(metrics_for_triage(sub, sub_triage, policy, band, costs))
    return rows


def guarantee_transfer_table(
    summary_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    target_ppv: float,
    target_npv: float,
) -> pd.DataFrame:
    """Target vs. calibration-empirical vs. held-out test, with exact CIs.

    This is the table that shows whether a claimed guarantee actually transfers.
    """
    rows = []
    for policy in ("ppv_npv_plugin", "ppv_npv_ltt"):
        for band in ALL_BANDS:
            sub = summary_df[(summary_df.policy == policy) & (summary_df.band == band)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            cal = thresholds_df[(thresholds_df.policy == policy) & (thresholds_df.band == band)]
            ppv_lo, ppv_hi = clopper_pearson_interval(int(r["tp"]), int(r["alert_n"]))
            npv_lo, npv_hi = clopper_pearson_interval(int(r["tn"]), int(r["normal_n"]))
            rows.append({
                "policy": policy,
                "band": band,
                "target_ppv": target_ppv,
                "cal_ppv": float(cal["calibration_ppv"].mean()) if not cal.empty else float("nan"),
                "test_ppv": float(r["ppv"]),
                "test_ppv_lo": ppv_lo,
                "test_ppv_hi": ppv_hi,
                "ppv_holds": bool(np.isfinite(r["ppv"]) and r["ppv"] >= target_ppv),
                "target_npv": target_npv,
                "cal_npv": float(cal["calibration_npv"].mean()) if not cal.empty else float("nan"),
                "test_npv": float(r["npv"]),
                "test_npv_lo": npv_lo,
                "test_npv_hi": npv_hi,
                "npv_holds": bool(np.isfinite(r["npv"]) and r["npv"] >= target_npv),
                "alert_n": int(r["alert_n"]),
                "normal_n": int(r["normal_n"]),
                "refer_rate": float(r["refer_rate"]),
            })
    return pd.DataFrame(rows)


def feasibility_table(
    cal_by_fold: Dict[int, pd.DataFrame],
    grid_size: int,
    target_ppv: float,
    target_npv: float,
    delta: float,
) -> pd.DataFrame:
    """How much calibration data would each band's guarantee have needed?

    Uses the best empirical PPV/NPV attainable on the pooled calibration data at
    a non-degenerate operating point, then inverts the Clopper-Pearson bound.
    """
    delta_eff = delta / max(n_threshold_pairs(grid_size), 1)
    pooled = pd.concat(cal_by_fold.values(), ignore_index=True).drop_duplicates(subset=["subject_id"])
    grid = prespecified_grid(grid_size)
    rows = []
    for band in BANDS:
        cal = pooled[pooled["band"] == band]
        if cal.empty:
            continue
        y = cal["gt_tachy"].to_numpy(dtype=int)
        scores = cal["tachy_prob"].to_numpy(dtype=float)
        best_ppv, best_npv = float("nan"), float("nan")
        for normal_threshold, alert_threshold in iter_threshold_pairs(grid):
            counts = confusion_counts(y, triage_from_thresholds(scores, normal_threshold, alert_threshold))
            n_alert, n_normal = counts["tp"] + counts["fp"], counts["tn"] + counts["fn"]
            # Require a non-trivial decision group so the rate is not 1/1.
            if n_alert >= 5:
                p = counts["tp"] / n_alert
                best_ppv = p if not np.isfinite(best_ppv) else max(best_ppv, p)
            if n_normal >= 5:
                p = counts["tn"] / n_normal
                best_npv = p if not np.isfinite(best_npv) else max(best_npv, p)
        rows.append({
            "band": band,
            "cal_n_available": int(len(cal)),
            "cal_positives": int(y.sum()),
            "best_attainable_ppv": best_ppv,
            "target_ppv": target_ppv,
            "min_alert_n_to_certify_ppv": min_n_to_certify(best_ppv, target_ppv, delta_eff),
            "best_attainable_npv": best_npv,
            "target_npv": target_npv,
            "min_normal_n_to_certify_npv": min_n_to_certify(best_npv, target_npv, delta_eff),
            "delta_effective": delta_eff,
        })
    return pd.DataFrame(rows)


def _pool_policy(
    test_by_fold: Dict[int, pd.DataFrame],
    by_fold_band: Dict[int, Dict[str, PolicyThreshold]],
    policy: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    frames, triages = [], []
    for fold in sorted(test_by_fold):
        test = test_by_fold[fold]
        frames.append(test)
        triages.append(apply_policy(test, by_fold_band[fold], policy))
    return pd.concat(frames, ignore_index=True), np.concatenate(triages)


def guarantee_frontier(
    test_by_fold: Dict[int, pd.DataFrame],
    ops_by_fold: Dict[int, Dict[str, Optional[OperatingPoints]]],
    costs: CostModel,
    targets_ppv: Sequence[float],
    targets_npv: Sequence[float],
    certify: bool,
    max_refer_rate: float,
    method: str = "fixed-sequence",
) -> pd.DataFrame:
    """Which (PPV, NPV) guarantees are actually certifiable, and at what referral cost?

    Fixing a single target and reporting "infeasible" understates what the system
    can promise. This sweeps the target grid so the paper can state the strongest
    guarantee the available calibration data supports.
    """
    policy = "ppv_npv_ltt" if certify else "ppv_npv_plugin"
    rows = []
    for t_ppv in targets_ppv:
        for t_npv in targets_npv:
            by_fold_band: Dict[int, Dict[str, PolicyThreshold]] = {}
            certified: Dict[str, int] = {b: 0 for b in BANDS}
            for fold in sorted(test_by_fold):
                by_band: Dict[str, PolicyThreshold] = {}
                for band in BANDS:
                    ops = ops_by_fold[fold][band]
                    if ops is None:
                        by_band[band] = PolicyThreshold(policy, band, fold, -1.0, 2.0, feasible=False,
                                                        infeasible_reason="no calibration data in band")
                        continue
                    picked = select_ppv_npv(ops, t_ppv, t_npv, certify, costs, max_refer_rate, method)
                    by_band[band] = picked
                    certified[band] += int(picked.feasible)
                by_fold_band[fold] = by_band
            pooled, triage = _pool_policy(test_by_fold, by_fold_band, policy)
            for band in ALL_BANDS:
                if band == "overall":
                    sub, sub_triage = pooled, triage
                    n_certified = sum(certified.values())
                    n_possible = len(BANDS) * len(test_by_fold)
                else:
                    mask = (pooled["band"] == band).to_numpy()
                    sub, sub_triage = pooled[mask], triage[mask]
                    n_certified, n_possible = certified[band], len(test_by_fold)
                if sub.empty:
                    continue
                m = metrics_for_triage(sub, sub_triage, policy, band, costs)
                rows.append({
                    "policy": policy, "target_ppv": t_ppv, "target_npv": t_npv, "band": band,
                    "folds_certified": n_certified, "folds_possible": n_possible,
                    "refer_rate": m["refer_rate"], "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"], "test_ppv": m["ppv"], "test_npv": m["npv"],
                    "ppv_holds": bool(np.isfinite(m["ppv"]) and m["ppv"] >= t_ppv),
                    "npv_holds": bool(np.isfinite(m["npv"]) and m["npv"] >= t_npv),
                    "false_normals": m["false_normals"], "false_alerts": m["false_alerts"],
                    "expected_cost": m["expected_cost"],
                })
    return pd.DataFrame(rows)


def robust_envelope(
    test_by_fold: Dict[int, pd.DataFrame],
    ops_by_fold: Dict[int, Dict[str, Optional[OperatingPoints]]],
    costs: CostModel,
    target_ppv: float,
    target_npv: float,
    prevalence_min: float,
    prevalence_maxima: Sequence[float],
    max_refer_rate: float,
) -> pd.DataFrame:
    """How far can the guarantee be certified before it stops being certifiable?

    Sweeping the upper end of the deployment prevalence range turns "the
    certificate is void above X%" from a post-hoc audit finding into a
    pre-specified operating envelope the deployment can be held to.
    """
    rows = []
    for pi_max in prevalence_maxima:
        by_fold_band: Dict[int, Dict[str, PolicyThreshold]] = {}
        certified: Dict[str, int] = {b: 0 for b in BANDS}
        for fold in sorted(test_by_fold):
            by_band: Dict[str, PolicyThreshold] = {}
            for band in BANDS:
                ops = ops_by_fold[fold][band]
                if ops is None:
                    by_band[band] = PolicyThreshold("ppv_npv_robust", band, fold, -1.0, 2.0,
                                                    feasible=False,
                                                    infeasible_reason="no calibration data in band")
                    continue
                picked = select_ppv_npv_robust(ops, target_ppv, target_npv, costs,
                                               prevalence_min, pi_max, max_refer_rate)
                by_band[band] = picked
                certified[band] += int(picked.feasible)
            by_fold_band[fold] = by_band
        pooled, triage = _pool_policy(test_by_fold, by_fold_band, "ppv_npv_robust")
        for band in ALL_BANDS:
            if band == "overall":
                sub, sub_triage = pooled, triage
                n_cert, n_poss = sum(certified.values()), len(BANDS) * len(test_by_fold)
            else:
                mask = (pooled["band"] == band).to_numpy()
                sub, sub_triage = pooled[mask], triage[mask]
                n_cert, n_poss = certified[band], len(test_by_fold)
            if sub.empty:
                continue
            m = metrics_for_triage(sub, sub_triage, "ppv_npv_robust", band, costs)
            # The certificate is a claim about prevalences in [pi_min, pi_max].
            # Scoring it at the test set's own prevalence answers a different
            # question, so re-express the held-out decisions at the prevalence
            # the certificate actually claims (exact under label shift).
            y_sub = sub["gt_tachy"].to_numpy(dtype=int)
            n_pos, n_neg = int((y_sub == 1).sum()), int((y_sub == 0).sum())
            p1n = m["fn"] / n_pos if n_pos else float("nan")
            p0n = m["tn"] / n_neg if n_neg else float("nan")
            p1a = m["tp"] / n_pos if n_pos else float("nan")
            p0a = m["fp"] / n_neg if n_neg else float("nan")
            npv_at_max = (((1 - pi_max) * p0n) / ((1 - pi_max) * p0n + pi_max * p1n)
                          if np.isfinite(p1n) and np.isfinite(p0n)
                          and ((1 - pi_max) * p0n + pi_max * p1n) > 0 else float("nan"))
            ppv_at_min = ((prevalence_min * p1a)
                          / (prevalence_min * p1a + (1 - prevalence_min) * p0a)
                          if np.isfinite(p1a) and np.isfinite(p0a)
                          and (prevalence_min * p1a + (1 - prevalence_min) * p0a) > 0 else float("nan"))
            rows.append({
                "prevalence_min": prevalence_min, "prevalence_max": pi_max, "band": band,
                "target_ppv": target_ppv, "target_npv": target_npv,
                "folds_certified": n_cert, "folds_possible": n_poss,
                "refer_rate": m["refer_rate"], "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "test_prevalence": float(y_sub.mean()) if y_sub.size else float("nan"),
                "test_ppv_at_test_prev": m["ppv"], "test_npv_at_test_prev": m["npv"],
                "npv_at_prevalence_max": npv_at_max, "ppv_at_prevalence_min": ppv_at_min,
                "npv_holds_at_prevalence_max": bool(np.isfinite(npv_at_max) and npv_at_max >= target_npv),
                "ppv_holds_at_prevalence_min": bool(np.isfinite(ppv_at_min) and ppv_at_min >= target_ppv),
                "false_normals": m["false_normals"], "false_alerts": m["false_alerts"],
            })
    return pd.DataFrame(rows)


def cost_sensitivity(
    test_by_fold: Dict[int, pd.DataFrame],
    ops_by_fold: Dict[int, Dict[str, Optional[OperatingPoints]]],
    miss_costs: Sequence[float],
    referral_costs: Sequence[float],
    max_refer_rate: float,
) -> pd.DataFrame:
    """Does the cost-aware policy's behaviour depend on the assumed cost ratio?"""
    rows = []
    for miss in miss_costs:
        for refer in referral_costs:
            costs = CostModel(miss=float(miss), false_alert=1.0, referral=float(refer))
            by_fold_band: Dict[int, Dict[str, PolicyThreshold]] = {}
            for fold in sorted(test_by_fold):
                by_band: Dict[str, PolicyThreshold] = {}
                for band in BANDS:
                    ops = ops_by_fold[fold][band]
                    by_band[band] = (
                        PolicyThreshold("cost_aware", band, fold, -1.0, 2.0, feasible=False)
                        if ops is None else select_cost_aware(ops, costs, max_refer_rate)
                    )
                by_fold_band[fold] = by_band
            pooled, triage = _pool_policy(test_by_fold, by_fold_band, "cost_aware")
            m = metrics_for_triage(pooled, triage, "cost_aware", "overall", costs)
            rows.append({
                "miss_cost": miss, "false_alert_cost": 1.0, "referral_cost": refer,
                "refer_rate": m["refer_rate"], "sensitivity": m["sensitivity"],
                "specificity": m["specificity"], "false_normals": m["false_normals"],
                "false_alerts": m["false_alerts"], "expected_cost": m["expected_cost"],
            })
    return pd.DataFrame(rows)


def print_table(rows: pd.DataFrame) -> None:
    cols = ["policy", "band", "n", "positives", "refer_rate", "sensitivity", "specificity",
            "ppv", "npv", "false_normals", "false_alerts", "expected_cost"]
    display = rows[cols].copy()
    for col in ["refer_rate", "sensitivity", "specificity", "ppv", "npv", "expected_cost"]:
        display[col] = display[col].map(lambda x: "--" if pd.isna(x) else f"{float(x):.3f}")
    print(display.to_string(index=False))


def clinical_interpretation(summary: pd.DataFrame, transfer: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    overall = summary[summary["band"] == "overall"].set_index("policy")
    if "coverage_conformal" not in overall.index:
        return
    base = overall.loc["coverage_conformal"]
    print("\nClinical interpretation (vs. the coverage-based policy used in the paper)")
    for policy in ("cost_aware", "ppv_npv_plugin", "ppv_npv_ltt"):
        if policy not in overall.index:
            continue
        row = overall.loc[policy]
        fn_delta = int(row["false_normals"]) - int(base["false_normals"])
        direction = "fewer" if fn_delta < 0 else "more" if fn_delta > 0 else "the same number of"
        print(f"  {policy}: {direction} false-normal tachypnea cases "
              f"({int(base['false_normals'])} -> {int(row['false_normals'])}); "
              f"referral {float(base['refer_rate']):.1%} -> {float(row['refer_rate']):.1%}.")

    if not transfer.empty:
        print("\nDoes the claimed guarantee transfer to held-out data?")
        for _, r in transfer[transfer.band == "overall"].iterrows():
            ppv_txt = "--" if not np.isfinite(r["test_ppv"]) else f"{r['test_ppv']:.3f}"
            npv_txt = "--" if not np.isfinite(r["test_npv"]) else f"{r['test_npv']:.3f}"
            verdict = "HOLDS" if (r["ppv_holds"] and r["npv_holds"]) else "VIOLATED"
            print(f"  {r['policy']:<16} target PPV>={r['target_ppv']:.2f}/NPV>={r['target_npv']:.2f} -> "
                  f"test PPV {ppv_txt}, NPV {npv_txt}  [{verdict}]")

    infeasible = thresholds[~thresholds["feasible"].astype(bool)]
    if not infeasible.empty:
        grp = infeasible.groupby(["policy", "band"]).size().reset_index(name="folds")
        print("\nBands where no policy was certifiable (REFER-all fallback):")
        for _, r in grp.iterrows():
            reason = infeasible[(infeasible.policy == r["policy"]) & (infeasible.band == r["band"])]
            reason = reason["infeasible_reason"].dropna().astype(str)
            why = reason.iloc[0] if len(reason) else ""
            print(f"  {r['policy']:<16} {r['band']:<8} {int(r['folds'])}/5 folds  ({why})")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare coverage-, cost-, and PPV/NPV-guaranteed PRISM-Breath deferral policies."
    )
    parser.add_argument("--run-root", default="outputs/paper_runs/pedrr_kfold_v3")
    parser.add_argument("--out-dir", default="", help="Default: outputs/deferral_policy/<run-name>")
    parser.add_argument("--calibration", choices=["oof", "val"], default="oof",
                        help="oof: other folds' test predictions (CV+ style, n~265). "
                             "val: fold's own validation split (strict split-conformal, n=39).")
    parser.add_argument("--alpha", type=float, default=0.20, help="Miscoverage for coverage_conformal")
    parser.add_argument("--grid-size", type=int, default=41,
                        help="Pre-specified threshold grid over [0,1]; sets the Bonferroni correction")
    parser.add_argument("--delta", type=float, default=0.10,
                        help="Family-wise error level for the Learn-then-Test guarantee")
    parser.add_argument("--certification", choices=["fixed-sequence", "bonferroni"],
                        default="fixed-sequence",
                        help="Multiplicity control for the LTT guarantee. Bonferroni over all "
                             "threshold pairs is valid but far too conservative at this prevalence.")
    parser.add_argument("--miss-cost", type=float, default=10.0)
    parser.add_argument("--false-alert-cost", type=float, default=1.0)
    parser.add_argument("--referral-cost", type=float, default=0.3)
    parser.add_argument("--target-ppv", type=float, default=0.90)
    parser.add_argument("--target-npv", type=float, default=0.95)
    parser.add_argument("--max-refer-rate", type=float, default=1.0,
                        help="Operational referral budget: reject policies deferring more than this")
    parser.add_argument("--deploy-prevalence-min", type=float, default=0.10,
                        help="Lowest deployment prevalence the robust certificate must cover")
    parser.add_argument("--deploy-prevalence-max", type=float, default=0.50,
                        help="Highest deployment prevalence the robust certificate must cover")
    parser.add_argument("--no-age-fix", action="store_true",
                        help="Score against the stored ages instead of the corrected ones")
    parser.add_argument("--skip-cost-sweep", action="store_true")
    parser.add_argument("--skip-frontier", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if not (0.0 < args.alpha < 1.0):
        raise SystemExit("--alpha must be in (0,1)")
    if not (0.0 < args.delta < 1.0):
        raise SystemExit("--delta must be in (0,1)")
    if args.grid_size < 3:
        raise SystemExit("--grid-size must be at least 3")
    if not (0.0 <= args.target_ppv <= 1.0 and 0.0 <= args.target_npv <= 1.0):
        raise SystemExit("--target-ppv and --target-npv must be in [0,1]")

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / "deferral_policy" / run_root.name
    costs = CostModel(miss=float(args.miss_cost), false_alert=float(args.false_alert_cost),
                      referral=float(args.referral_cost))

    age_fix = not args.no_age_fix
    val_by_fold = load_by_fold(run_root, "val_eval", age_fix=age_fix)
    test_by_fold = load_by_fold(run_root, "test_eval", age_fix=age_fix)
    n_cand = n_threshold_pairs(args.grid_size)
    print(f"Run root         : {run_root}")
    print(f"Folds            : {sorted(test_by_fold)}")
    print(f"Labels           : {'age-corrected' if age_fix else 'stored ages'}")
    print(f"Calibration      : {args.calibration} "
          f"({'other folds test predictions, CV+ style' if args.calibration == 'oof' else 'own validation split'})")
    print(f"Costs            : miss={costs.miss}, false_alert={costs.false_alert}, referral={costs.referral}")
    if args.certification == "fixed-sequence":
        print(f"LTT              : delta={args.delta}, fixed-sequence over two ordered families "
              f"({args.grid_size} thresholds each), delta'={args.delta / 2:.3g} per family")
    else:
        print(f"LTT              : delta={args.delta}, Bonferroni over {n_cand} candidate pairs, "
              f"delta'={args.delta / n_cand:.3g}")

    all_thresholds: List[PolicyThreshold] = []
    cal_by_fold: Dict[int, pd.DataFrame] = {}
    per_policy_triage: Dict[str, List[np.ndarray]] = {p: [] for p in POLICIES}
    test_frames: List[pd.DataFrame] = []

    ops_by_fold: Dict[int, Dict[str, Optional[OperatingPoints]]] = {}
    for fold in sorted(test_by_fold):
        cal = calibration_for_fold(fold, val_by_fold, test_by_fold, args.calibration)
        cal_by_fold[fold] = cal
        ops_by_fold[fold] = build_operating_points(cal, fold, int(args.grid_size), float(args.delta))
        fitted = fit_policies_for_fold(cal, fold, float(args.alpha), int(args.grid_size),
                                       float(args.target_ppv), float(args.target_npv),
                                       costs, float(args.delta), float(args.max_refer_rate),
                                       ops_by_band=ops_by_fold[fold], method=args.certification,
                                       prevalence_range=(float(args.deploy_prevalence_min),
                                                         float(args.deploy_prevalence_max)))
        all_thresholds.extend(fitted)
        test = test_by_fold[fold]
        test_frames.append(test)
        for policy in POLICIES:
            by_band = {t.band: t for t in fitted if t.policy == policy}
            per_policy_triage[policy].append(apply_policy(test, by_band, policy))
        print(f"  fold{fold}: calibrated on n={len(cal)} subjects "
              f"({int(cal['gt_tachy'].sum())} tachypneic), applied to n={len(test)} held-out subjects")

    test_df = pd.concat(test_frames, ignore_index=True)
    policy_triages = {p: np.concatenate(v) for p, v in per_policy_triage.items()}

    thresholds_df = pd.DataFrame([t.__dict__ for t in all_thresholds])
    summary_rows: List[Dict] = []
    for policy in POLICIES:
        summary_rows.extend(evaluate_policy(test_df, policy_triages[policy], policy, costs))
    summary_df = pd.DataFrame(summary_rows)
    summary_df["policy_order"] = summary_df["policy"].map({p: i for i, p in enumerate(POLICIES)})
    summary_df["band_order"] = summary_df["band"].map({b: i for i, b in enumerate(ALL_BANDS)})
    summary_df = summary_df.sort_values(["policy_order", "band_order"]).drop(columns=["policy_order", "band_order"])

    transfer_df = guarantee_transfer_table(summary_df, thresholds_df,
                                           float(args.target_ppv), float(args.target_npv))
    feasibility_df = feasibility_table(cal_by_fold, int(args.grid_size),
                                       float(args.target_ppv), float(args.target_npv), float(args.delta))

    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds_df.to_csv(out_dir / "policy_thresholds.csv", index=False)
    summary_df.to_csv(out_dir / "policy_comparison_by_band.csv", index=False)
    transfer_df.to_csv(out_dir / "guarantee_transfer.csv", index=False)
    feasibility_df.to_csv(out_dir / "guarantee_feasibility.csv", index=False)

    subjects = test_df.copy()
    for policy, triage in policy_triages.items():
        subjects[f"{policy}_triage_code"] = triage
        subjects[f"{policy}_triage_label"] = [SHORT_STATUS[int(x)] for x in triage]
    subjects.to_csv(out_dir / "policy_comparison_by_subject.csv", index=False)

    print("\nHeld-out test comparison (per-fold calibration, decisions pooled)")
    print_table(summary_df)

    print("\nGuarantee transfer: calibration vs. held-out test")
    if not transfer_df.empty:
        show = transfer_df[["policy", "band", "target_ppv", "cal_ppv", "test_ppv", "ppv_holds",
                            "target_npv", "cal_npv", "test_npv", "npv_holds", "refer_rate"]].copy()
        for col in ["cal_ppv", "test_ppv", "cal_npv", "test_npv", "refer_rate"]:
            show[col] = show[col].map(lambda x: "--" if pd.isna(x) else f"{float(x):.3f}")
        print(show.to_string(index=False))

    print("\nCalibration sample size needed to certify each guarantee")
    if not feasibility_df.empty:
        show = feasibility_df.copy()
        for col in ["best_attainable_ppv", "best_attainable_npv"]:
            show[col] = show[col].map(lambda x: "--" if pd.isna(x) else f"{float(x):.3f}")
        print(show[["band", "cal_n_available", "cal_positives", "best_attainable_ppv",
                    "min_alert_n_to_certify_ppv", "best_attainable_npv",
                    "min_normal_n_to_certify_npv"]].to_string(index=False))

    clinical_interpretation(summary_df, transfer_df, thresholds_df)

    written = ["policy_thresholds.csv", "policy_comparison_by_band.csv",
               "guarantee_transfer.csv", "guarantee_feasibility.csv",
               "policy_comparison_by_subject.csv"]

    if not args.skip_frontier:
        frontier = guarantee_frontier(
            test_by_fold, ops_by_fold, costs,
            targets_ppv=(0.60, 0.70, 0.75, 0.80, 0.85, 0.90),
            targets_npv=(0.80, 0.85, 0.90, 0.925, 0.95, 0.98),
            certify=True, max_refer_rate=float(args.max_refer_rate), method=args.certification,
        )
        frontier.to_csv(out_dir / "guarantee_frontier.csv", index=False)
        written.append("guarantee_frontier.csv")
        print("\nCertifiable guarantee frontier (Learn-then-Test, all bands certified, overall)")
        ok = frontier[(frontier.band == "overall") & (frontier.folds_certified == frontier.folds_possible)]
        if ok.empty:
            print("  No (PPV, NPV) target in the swept grid was certifiable in every band and fold.")
            per_band = frontier[(frontier.band != "overall") &
                                (frontier.folds_certified == frontier.folds_possible)]
            if not per_band.empty:
                best = (per_band.sort_values(["band", "target_npv", "target_ppv"],
                                             ascending=[True, False, False])
                        .groupby("band").head(1))
                print("  Strongest guarantee certifiable per band (all 5 folds):")
                for _, r in best.iterrows():
                    print(f"    {r['band']:<8} PPV>={r['target_ppv']:.2f}, NPV>={r['target_npv']:.3f} "
                          f"-> refer {r['refer_rate']:.1%}, test PPV {r['test_ppv']:.3f}, "
                          f"NPV {r['test_npv']:.3f}, missed {int(r['false_normals'])}")
            else:
                print("  No band certifies any swept target at this calibration size.")
        else:
            show = ok.sort_values(["target_npv", "target_ppv"], ascending=False)
            show = show[["target_ppv", "target_npv", "refer_rate", "sensitivity",
                         "specificity", "test_ppv", "test_npv", "false_normals"]]
            for col in ["refer_rate", "sensitivity", "specificity", "test_ppv", "test_npv"]:
                show[col] = show[col].map(lambda x: "--" if pd.isna(x) else f"{float(x):.3f}")
            print(show.to_string(index=False))

    envelope = robust_envelope(
        test_by_fold, ops_by_fold, costs, float(args.target_ppv), float(args.target_npv),
        float(args.deploy_prevalence_min),
        prevalence_maxima=(0.10, 0.15, 0.20, 0.26, 0.30, 0.35, 0.40, 0.50),
        max_refer_rate=float(args.max_refer_rate),
    )
    envelope.to_csv(out_dir / "robust_envelope.csv", index=False)
    written.append("robust_envelope.csv")
    print(f"\nPrevalence-robust certificate: how far it certifies "
          f"(PPV>={args.target_ppv:.2f} from prev {args.deploy_prevalence_min:.2f}, "
          f"NPV>={args.target_npv:.2f} up to prev shown)")
    show = envelope[envelope.band == "overall"][
        ["prevalence_max", "folds_certified", "folds_possible", "refer_rate",
         "sensitivity", "specificity", "npv_at_prevalence_max",
         "npv_holds_at_prevalence_max", "false_normals"]].copy()
    for col in ["refer_rate", "sensitivity", "specificity", "npv_at_prevalence_max"]:
        show[col] = show[col].map(lambda x: "--" if pd.isna(x) else f"{float(x):.3f}")
    print(show.to_string(index=False))

    full = envelope[(envelope.band != "overall") &
                    (envelope.folds_certified == envelope.folds_possible)]
    print("  Highest deployment prevalence certifiable in every fold, by band:")
    for band in BANDS:
        sub = full[full.band == band]
        if sub.empty:
            print(f"    {band:<8} not certifiable at any swept prevalence")
            continue
        r = sub.sort_values("prevalence_max").iloc[-1]
        print(f"    {band:<8} up to prevalence {r['prevalence_max']:.0%}  "
              f"(refer {r['refer_rate']:.1%}, held-out NPV at that prevalence "
              f"{r['npv_at_prevalence_max']:.3f})")

    if not args.skip_cost_sweep:
        sweep = cost_sensitivity(test_by_fold, ops_by_fold,
                                 miss_costs=(2.0, 5.0, 10.0, 20.0),
                                 referral_costs=(0.1, 0.3, 0.5, 1.0),
                                 max_refer_rate=float(args.max_refer_rate))
        sweep.to_csv(out_dir / "cost_sensitivity.csv", index=False)
        written.append("cost_sensitivity.csv")
        print("\nCost sensitivity (cost_aware, overall)")
        show = sweep.copy()
        for col in ["refer_rate", "sensitivity", "specificity", "expected_cost"]:
            show[col] = show[col].map(lambda x: "--" if pd.isna(x) else f"{float(x):.3f}")
        print(show.to_string(index=False))

    print("\nWrote outputs")
    for name in written:
        print(f"  {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
