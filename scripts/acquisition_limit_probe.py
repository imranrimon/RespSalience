"""Is neonatal breathing erased at acquisition, or merely out-ranked?

The paper states that high-rate respiratory information is "attenuated in the raw
signal, but not provably erased". That hedge decides where the neonatal blind
spot has to be fixed: if the true fast rate is present in the extracted signal
but out-ranked by a lower-frequency confound, a better read-out can recover it
and the fix is a model. If there is no spectral evidence at the true rate at all,
no read-out can recover it and the fix is a sensor.

This settles it directly on the cached signals, with no training. For every
subject we compute the respiratory-band spectrum and ask three questions of
increasing generosity:

  1. does the spectral maximum land on the true rate?          (what we do now)
  2. is the true rate among the top-k spectral peaks?          (a better read-out)
  3. is there ANY peak at the true rate above the noise floor? (an oracle read-out)

Question 3 is the one that matters: it is the ceiling of every possible read-out
of this signal. If neonates fail question 3, the information is gone.

The probe also checks the band-pass itself. Respiration is filtered to
0.15-1.5 Hz, and 1.5 Hz is 90 breaths/min -- so the neonatal range sits against
the upper edge of the band, where any filter attenuates. Whether the pipeline is
discarding the signal it is looking for is worth knowing separately from what the
sensor captured.

Run from the repository root:

    python scripts/acquisition_limit_probe.py --signal-dir data/pedrr_signals_v2
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch

FS = 25.0  # canonical common time base (Hz)


def rr_band(rr_bpm: float) -> str:
    if rr_bpm < 30:
        return "<30"
    if rr_bpm < 45:
        return "30-45"
    if rr_bpm < 60:
        return "45-60"
    return ">=60"


def age_band(age_months: float) -> str:
    if age_months < 2:
        return "<2mo"
    if age_months < 12:
        return "2-12mo"
    return "12-60mo"


def spectrum(x: np.ndarray, fs: float = FS, nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 64:
        return np.array([]), np.array([])
    x = x - x.mean()
    nperseg = int(min(nperseg, x.size))
    f, p = welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann", detrend="linear")
    return f, p


def analyse_signal(x: np.ndarray, gt_bpm: float, f_lo: float, f_hi: float,
                   tol_frac: float, tol_min_bpm: float, top_k: int) -> Dict[str, float]:
    f, p = spectrum(x)
    if f.size == 0:
        return {}
    band = (f >= f_lo) & (f <= f_hi)
    if band.sum() < 8:
        return {}
    fb, pb = f[band], p[band]
    bpm = fb * 60.0

    noise_floor = float(np.median(pb))
    argmax_bpm = float(bpm[int(np.argmax(pb))])

    tol = max(tol_frac * gt_bpm, tol_min_bpm)
    # Peaks, strongest first.
    idx, _ = find_peaks(pb)
    if idx.size == 0:
        idx = np.array([int(np.argmax(pb))])
    order = idx[np.argsort(pb[idx])[::-1]]
    peak_bpm = bpm[order]

    hit = np.abs(peak_bpm - gt_bpm) <= tol
    rank_true = int(np.argmax(hit) + 1) if hit.any() else -1

    # Power exactly at the true rate, relative to the band noise floor.
    j = int(np.argmin(np.abs(bpm - gt_bpm)))
    snr_true = float(pb[j] / noise_floor) if noise_floor > 0 else float("nan")
    snr_argmax = float(pb.max() / noise_floor) if noise_floor > 0 else float("nan")

    return {
        "argmax_bpm": argmax_bpm,
        "argmax_hit": bool(abs(argmax_bpm - gt_bpm) <= tol),
        "rank_true": rank_true,
        "in_top_k": bool(0 < rank_true <= top_k),
        "any_peak_at_true": bool(rank_true > 0),
        "snr_true": snr_true,
        "snr_argmax": snr_argmax,
        "snr_ratio": float(snr_true / snr_argmax) if snr_argmax else float("nan"),
        "n_peaks": int(order.size),
    }


def band_edge_check(x: np.ndarray, gt_bpm: float) -> Dict[str, float]:
    """How much power sits above the 1.5 Hz band edge the pipeline discards?"""
    f, p = spectrum(x)
    if f.size == 0:
        return {}
    total = (f >= 0.15) & (f <= 3.0)
    inband = (f >= 0.15) & (f <= 1.5)
    above = (f > 1.5) & (f <= 3.0)
    tp = float(p[total].sum())
    return {
        "frac_power_above_edge": float(p[above].sum() / tp) if tp > 0 else float("nan"),
        "true_rate_hz": gt_bpm / 60.0,
        "true_above_edge": bool(gt_bpm / 60.0 > 1.5),
    }


def load_signals(signal_dir: Path) -> List[Dict]:
    rows = []
    for path in sorted(glob.glob(str(signal_dir / "*.npz"))):
        try:
            z = np.load(path, allow_pickle=True)
        except Exception:
            continue
        try:
            gt = float(z["gt"])
        except Exception:
            continue
        if not np.isfinite(gt) or gt <= 0:
            continue
        rows.append({
            "file": os.path.basename(path),
            "subject_id": str(z["subject_id"]) if "subject_id" in z else "",
            "gt_bpm": gt,
            "age_months": float(z["age_months"]) if "age_months" in z else float("nan"),
            "depth": np.asarray(z["depth"], dtype=float) if "depth" in z else None,
            "flow": np.asarray(z["flow"], dtype=float) if "flow" in z else None,
        })
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test whether high-rate respiration survives acquisition.")
    p.add_argument("--signal-dir", default="data/pedrr_signals_v2")
    p.add_argument("--out-dir", default="outputs/acquisition_limit")
    p.add_argument("--f-lo", type=float, default=0.15)
    p.add_argument("--f-hi", type=float, default=1.8,
                   help="Analysis band edge in Hz. Deliberately above the 1.5 Hz pipeline edge so "
                        "the probe can see content the pipeline discards.")
    p.add_argument("--tol-frac", type=float, default=0.10)
    p.add_argument("--tol-min-bpm", type=float, default=3.0)
    p.add_argument("--top-k", type=int, default=3)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    signal_dir = Path(args.signal_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_signals(signal_dir)
    if not rows:
        raise SystemExit(f"No usable .npz signals under {signal_dir}")
    print(f"Signals      : {len(rows)} from {signal_dir}")
    print(f"Analysis band: {args.f_lo}-{args.f_hi} Hz "
          f"({args.f_lo * 60:.0f}-{args.f_hi * 60:.0f} breaths/min); pipeline band ends at 1.5 Hz")
    print(f"Tolerance    : max({args.tol_frac:.0%} of true rate, {args.tol_min_bpm:.0f} breaths/min)\n")

    recs = []
    for r in rows:
        for modality in ("flow", "depth"):
            x = r.get(modality)
            if x is None or x.size == 0 or not np.isfinite(x).any() or np.allclose(x, 0):
                continue
            m = analyse_signal(x, r["gt_bpm"], args.f_lo, args.f_hi,
                               args.tol_frac, args.tol_min_bpm, int(args.top_k))
            if not m:
                continue
            m.update(band_edge_check(x, r["gt_bpm"]))
            recs.append({
                "subject_id": r["subject_id"], "modality": modality,
                "gt_bpm": r["gt_bpm"], "age_months": r["age_months"],
                "rr_band": rr_band(r["gt_bpm"]), "age_band": age_band(r["age_months"]),
                **m,
            })
    df = pd.DataFrame(recs)
    df.to_csv(out_dir / "acquisition_limit_per_subject.csv", index=False)

    for modality in ("flow", "depth"):
        sub = df[df.modality == modality]
        if sub.empty:
            continue
        print(f"=== {modality} ===")
        g = sub.groupby("rr_band").agg(
            n=("gt_bpm", "size"),
            argmax_hit=("argmax_hit", "mean"),
            top_k=("in_top_k", "mean"),
            any_peak=("any_peak_at_true", "mean"),
            snr_true=("snr_true", "median"),
            snr_argmax=("snr_argmax", "median"),
        ).reindex(["<30", "30-45", "45-60", ">=60"]).dropna(how="all")
        g.columns = ["n", "argmax hit", f"top-{args.top_k}", "any peak", "SNR@true", "SNR@argmax"]
        print(g.to_string(float_format=lambda x: f"{x:.3f}"))
        print()

    print("Verdict on the neonatal band (true rate >= 60 breaths/min)")
    for modality in ("flow", "depth"):
        fast = df[(df.modality == modality) & (df.rr_band == ">=60")]
        if fast.empty:
            continue
        n = len(fast)
        print(f"  {modality}: n={n}  argmax {fast.argmax_hit.mean():.1%}  "
              f"top-{args.top_k} {fast.in_top_k.mean():.1%}  any peak {fast.any_peak_at_true.mean():.1%}  "
              f"median SNR at true rate {fast.snr_true.median():.2f}x noise floor")
        recoverable = fast.any_peak_at_true.mean()
        if recoverable >= 0.5:
            print(f"     -> the true rate is present in over half of neonatal signals but out-ranked: "
                  f"a read-out problem, recoverable in silico.")
        else:
            print(f"     -> no spectral evidence at the true rate in {1 - recoverable:.0%} of cases: "
                  f"an acquisition limit no read-out can undo.")

    print("\nIs the 1.5 Hz pipeline band edge itself discarding the signal?")
    edge = df[df.modality == "flow"]
    for b in ["<30", "30-45", "45-60", ">=60"]:
        s = edge[edge.rr_band == b]
        if s.empty:
            continue
        print(f"  true RR {b:<6} n={len(s):3d}  power above 1.5 Hz {s.frac_power_above_edge.median():.3f}  "
              f"true rate above edge: {int(s.true_above_edge.sum())}/{len(s)}")

    print(f"\nWrote {out_dir / 'acquisition_limit_per_subject.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
