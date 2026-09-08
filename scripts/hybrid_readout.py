"""A selective age-conditional read-out: use the per-band cut point only where it earns its place.

`scripts/operating_point_readout.py` shows, consistently across seeds, that replacing the
regression rule with a per-band cut point on the distributional mass recovers neonates (4-5 recovered,
0 lost in every seed) but *costs* true positives in the 12-60mo band (6-8 lost, 0 recovered,
McNemar p=0.008-0.031). Applied blindly to every band it is roughly a wash.

That asymmetry is not noise and it has an explanation: in the older band the salience model already
discriminates near-perfectly (AUROC 0.96) and the regression rule is already well placed there, so
moving the cut point can only lose. In the neonatal band the ranking is equally good but the regressed
rate almost never crosses the WHO cutoff, so the cut point is the only thing standing between a good
ranking and a usable screen.

So the deployable rule is selective: read the distributional mass at a calibrated per-band cut point
for `<2mo`, and keep the regression rule where it already works. Thresholds are still chosen on
calibration folds only.

    python scripts/hybrid_readout.py
    python scripts/hybrid_readout.py --apply-bands "<2mo,2-12mo" --target-sens 0.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_main_run import load, scored  # noqa: E402

SEEDS = [f"pedrr_kfold_salience_ssl332_seed{i}" for i in range(3)]
BANDS = ("<2mo", "2-12mo", "12-60mo")


def band_of(a):
    return "<2mo" if a < 2 else ("2-12mo" if a < 12 else "12-60mo")


def pick_threshold(cal, target):
    """Lowest cut point on the distributional mass meeting the target sensitivity on calibration."""
    pos = cal[cal.y == 1]
    if len(pos) == 0:
        return 0.5
    qs = np.unique(np.round(pos.tachy_prob.to_numpy(float), 4))
    # The LARGEST cut point still meeting the sensitivity target is the most specific one that does.
    ok = [t for t in np.sort(qs) if (pos.tachy_prob >= t).mean() >= target]
    return float(ok[-1]) if ok else 0.0


def evaluate(d, apply_bands, target):
    """Leave-one-fold-out: thresholds from the other folds only."""
    dec = np.zeros(len(d), int)
    d = d.reset_index(drop=True)
    for f in sorted(d.fold.unique()):
        te = d.fold == f
        cal = d[~te]
        for b in BANDS:
            m = te & (d.band == b)
            if not m.any():
                continue
            if b in apply_bands:
                t = pick_threshold(cal[cal.band == b], target)
                dec[m.to_numpy()] = (d.loc[m, "tachy_prob"] >= t).astype(int)
            else:
                dec[m.to_numpy()] = (d.loc[m, "pred_rr_bpm"] >= d.loc[m, "thr"]).astype(int)
    return dec


def rates(y, dec):
    tp = int(((dec == 1) & (y == 1)).sum()); fn = int(((dec == 0) & (y == 1)).sum())
    tn = int(((dec == 0) & (y == 0)).sum()); fp = int(((dec == 1) & (y == 0)).sum())
    return (tp / max(1, tp + fn), tn / max(1, tn + fp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply-bands", default="<2mo",
                    help="comma-separated bands that use the distributional cut point")
    ap.add_argument("--target-sens", type=float, default=0.70)
    a = ap.parse_args()
    apply_bands = tuple(s.strip() for s in a.apply_bands.split(",") if s.strip())

    ref = load("pedrr_kfold_flowonly_agefix")
    age_map = dict(zip(ref["subject_id"].astype(str), ref["age_months"].astype(float)))

    rows, mcn = [], {b: [0, 0] for b in ("overall",) + BANDS}
    for tag in SEEDS:
        d = scored(tag, age_map)
        if d is None:
            raise SystemExit(f"missing run {tag}")
        d["band"] = d["age"].map(band_of)
        base = (d.pred_rr_bpm >= d.thr).astype(int).to_numpy()
        hyb = evaluate(d, apply_bands, a.target_sens)
        y = d.y.to_numpy(int)
        for lab, m in [("overall", np.ones(len(y), bool))] + [(b, (d.band == b).to_numpy()) for b in BANDS]:
            s_b, p_b = rates(y[m], base[m]); s_h, p_h = rates(y[m], hyb[m])
            rows.append({"seed": tag[-1], "band": lab, "sens_reg": s_b, "spec_reg": p_b,
                         "sens_hyb": s_h, "spec_hyb": p_h})
            rec = int(((y == 1) & (base == 0) & (hyb == 1))[m].sum())
            lost = int(((y == 1) & (base == 1) & (hyb == 0))[m].sum())
            mcn[lab][0] += rec; mcn[lab][1] += lost
    r = pd.DataFrame(rows).groupby("band").mean(numeric_only=True).reindex(["overall", *BANDS])

    print(f"Selective read-out: distributional cut point applied to {apply_bands}, "
          f"regression rule elsewhere (target sens {a.target_sens:.2f}, 3 seeds)\n")
    print(f"{'band':<10}{'Sens reg':>10}{'Sens hyb':>10}{'  ':<2}{'Spec reg':>10}{'Spec hyb':>10}")
    for b, v in r.iterrows():
        print(f"{b:<10}{v.sens_reg:>10.3f}{v.sens_hyb:>10.3f}{'':<2}{v.spec_reg:>10.3f}{v.spec_hyb:>10.3f}")

    print("\nPaired discordant true positives, pooled over the 3 seeds (exact McNemar)")
    for b, (rec, lost) in mcn.items():
        p = binomtest(min(rec, lost), rec + lost, 0.5).pvalue if rec + lost else 1.0
        flag = "" if p >= 0.05 else ("  <-- net GAIN" if rec > lost else "  <-- net LOSS")
        print(f"  {b:<10} recovered {rec:>3}, lost {lost:>3}, p={p:.4f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
