"""Radar vs. clinical-monitor breathing rate on the public 'Children Dataset'.

The paper reports a radar row validated against a reference monitor on real children, but no script in
this repository produced it. This is that script.

Data layout (as distributed):
  FMCW Radar/Heart Rate & Breathing Rate/Breath_<k>.csv       one breathing rate per line (per second)
  Nihon Kohden/Heart Rate & Breathing Rate/Ref_Breath_<k>.csv  the reference monitor's rates, one row,
                                                               comma separated, same cadence

Both are already rates in breaths/min, so this is an agreement analysis, not an extraction: per
recording we take the median over the jointly-valid samples (radar warm-up writes leading zeros, which
are missing data and are excluded, not treated as 0 BPM), then report MAE, correlation and the fraction
of recordings within 3 breaths/min -- the tolerance the paper quotes.

    python scripts/radar_children_rr.py
    python scripts/radar_children_rr.py --agg mean --json outputs/radar_children.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "child_radar" / "Children Dataset"
RADAR_DIR = BASE / "FMCW Radar" / "Heart Rate & Breathing Rate"
REF_DIR = BASE / "Nihon Kohden" / "Heart Rate & Breathing Rate"


def read_series(path: Path) -> np.ndarray:
    """Read a rate series stored either one-per-line or as a single comma-separated row."""
    raw = Path(path).read_text().strip()
    if not raw:
        return np.array([])
    parts = re.split(r"[,\s]+", raw)
    v = pd.to_numeric(pd.Series(parts), errors="coerce").to_numpy(float)
    return v[np.isfinite(v)]


def valid_mask(x: np.ndarray) -> np.ndarray:
    # Exactly-zero samples are the sensor's warm-up / dropout marker, not a measurement of 0 BPM.
    return np.isfinite(x) & (x > 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", choices=["median", "mean"], default="median")
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--min-valid", type=int, default=30, help="minimum jointly-valid samples")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    if not RADAR_DIR.is_dir() or not REF_DIR.is_dir():
        raise SystemExit(f"dataset not found under {BASE}")

    agg = np.median if a.agg == "median" else np.mean
    rows, skipped = [], []
    for rp in sorted(glob.glob(str(RADAR_DIR / "Breath_*.csv")),
                     key=lambda p: int(re.search(r"Breath_(\d+)", os.path.basename(p)).group(1))):
        k = int(re.search(r"Breath_(\d+)", os.path.basename(rp)).group(1))
        fp = REF_DIR / f"Ref_Breath_{k}.csv"
        if not fp.exists():
            skipped.append((k, "no reference")); continue
        r, f = read_series(Path(rp)), read_series(fp)
        n = min(len(r), len(f))
        if n == 0:
            skipped.append((k, "empty")); continue
        r, f = r[:n], f[:n]
        m = valid_mask(r) & valid_mask(f)
        if m.sum() < a.min_valid:
            skipped.append((k, f"only {int(m.sum())} valid samples")); continue
        rows.append({"rec": k, "n_valid": int(m.sum()),
                     "radar": float(agg(r[m])), "ref": float(agg(f[m]))})

    if not rows:
        raise SystemExit("no usable recordings")
    d = pd.DataFrame(rows)
    err = (d.radar - d.ref).abs()
    mae = float(err.mean()); rmse = float(np.sqrt(((d.radar - d.ref) ** 2).mean()))
    r = float(np.corrcoef(d.radar, d.ref)[0, 1])
    within = float((err <= a.tolerance).mean())
    bias = float((d.radar - d.ref).mean())
    lo, hi = bias - 1.96 * float((d.radar - d.ref).std(ddof=1)), bias + 1.96 * float((d.radar - d.ref).std(ddof=1))

    print(f"Children radar vs. Nihon Kohden reference   (per-recording {a.agg})")
    print(f"  recordings used : {len(d)}   skipped: {len(skipped)}")
    print(f"  reference RR    : {d.ref.min():.1f}-{d.ref.max():.1f} (mean {d.ref.mean():.1f})")
    print(f"  radar RR        : {d.radar.min():.1f}-{d.radar.max():.1f} (mean {d.radar.mean():.1f})")
    print(f"  MAE             : {mae:.2f} breaths/min")
    print(f"  RMSE            : {rmse:.2f}")
    print(f"  Pearson r       : {r:.3f}")
    print(f"  within {a.tolerance:.0f} br/min : {within*100:.0f}%  ({int((err<=a.tolerance).sum())}/{len(d)})")
    print(f"  Bland-Altman    : bias {bias:+.2f}, 95% LoA [{lo:+.2f}, {hi:+.2f}]")
    if skipped:
        print(f"  skipped recordings: {skipped[:8]}{' ...' if len(skipped) > 8 else ''}")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"n": len(d), "agg": a.agg, "mae": mae, "rmse": rmse, "pearson_r": r,
             "within_tolerance": within, "tolerance": a.tolerance, "bias": bias,
             "loa": [lo, hi], "per_recording": rows, "skipped": skipped}, indent=2))
        print(f"  -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
