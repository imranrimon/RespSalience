"""FMCW-radar / ToF respiratory-rate extractor for the neonatal thorax-simulator dataset
(figshare 6809370). THE test: does radar/ToF recover 60-80 BPM where every RGB extractor
saturated at ~35 (see memory neonatal-extractor-limit-reframe)?

Data format (from the dataset's scripts pack):
  - Radar : ROS `rostopic echo -p /iq` CSV -> %time + IQ field columns.
  - ToF   : z-encoded depth PNG frames, timestamp in filename.
  - Laser : CSV [timestamp, distance] -> the GROUND-TRUTH breathing reference.

RR pipeline (shared): detrend -> Butterworth band-pass 0.1-1.6 Hz (6-96 BPM, covers neonatal
tachypnea) -> Hann FFT -> dominant in-band peak -> BPM. Radar uses the phase of the max-band-energy
range bin; ToF uses mean depth in a central chest ROI.

Usage: python scripts/radar_tof_rr.py --rec-dir data/radar_neonatal_sim/<unzipped recording dir>
       python scripts/radar_tof_rr.py --radar-csv <iq.csv> --laser-csv <laser.csv>   # explicit files
"""
from __future__ import annotations
import argparse, glob, os, re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend


def band_fft_rr(x, fs, lo=0.1, hi=1.6):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 32 or fs <= 0 or not np.isfinite(fs):
        return float("nan")
    x = detrend(x)
    ny = fs / 2.0
    lo_n, hi_n = max(lo, 0.03) / ny, min(hi, ny * 0.98) / ny
    if not (0 < lo_n < hi_n < 1):
        return float("nan")
    try:
        b, a = butter(3, [lo_n, hi_n], btype="band")
        x = filtfilt(b, a, x)
    except Exception:
        pass
    n = len(x); f = np.fft.rfftfreq(n, d=1.0 / fs)
    P = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return float("nan")
    return float(f[band][np.argmax(P[band])] * 60.0)


def _time_fs(t):
    t = np.asarray(t, float)
    if t.max() > 1e17:   t = t / 1e9      # ns
    elif t.max() > 1e14: t = t / 1e6      # us
    elif t.max() > 1e11: t = t / 1e3      # ms
    t = t - t[0]
    dt = np.median(np.diff(t))
    return t, (1.0 / dt if dt > 0 else float("nan"))


def laser_rr(csv):
    """Ground-truth RR from the laser micrometer distance signal."""
    d = pd.read_csv(csv, header=None, comment="%").values.astype(float)
    t, fs = _time_fs(d[:, 0])
    z = d[:, -1]                                   # distance = last column
    return band_fft_rr(z, fs), fs, len(z)


def radar_rr(csv):
    """RR from the ROS IQ dump: reconstruct complex per range bin, pick the max in-band-energy bin,
    take its unwrapped phase (chest displacement), extract RR."""
    df = pd.read_csv(csv)
    tcol = next((c for c in df.columns if "time" in c.lower()), df.columns[0])
    t, fs = _time_fs(df[tcol].values)
    fcols = [c for c in df.columns if c != tcol and np.issubdtype(df[c].dtype, np.number)]
    M = df[fcols].values.astype(float)             # (T, K)
    K = M.shape[1]
    # interleaved I/Q -> complex per bin (K even); else fall back to magnitude of all columns
    if K >= 2 and K % 2 == 0:
        comp = M[:, 0::2] + 1j * M[:, 1::2]         # (T, K/2) complex range bins
    else:
        comp = M.astype(complex)
    # pick the range bin whose (magnitude or phase) carries the most respiratory-band energy
    best_rr, best_energy = float("nan"), -1.0
    for j in range(comp.shape[1]):
        phase = np.unwrap(np.angle(comp[:, j]))
        rr = band_fft_rr(phase, fs)
        # energy proxy: variance of the band-passed phase
        e = np.nanvar(phase) if np.isfinite(rr) else -1.0
        if e > best_energy:
            best_energy, best_rr = e, rr
    return best_rr, fs, comp.shape


def tof_rr(png_dir):
    """RR from ToF depth PNG frames: central-ROI mean z over time."""
    import imageio.v2 as imageio
    pngs = sorted(glob.glob(os.path.join(png_dir, "*.png")))
    if len(pngs) < 32:
        return float("nan"), float("nan"), 0
    def ts(p):
        m = re.search(r"(\d{10,})", os.path.basename(p))
        return float(m.group(1)) if m else 0.0
    pngs.sort(key=ts)
    zt = []
    for p in pngs:
        im = np.asarray(imageio.imread(p)).astype(float)
        if im.ndim == 3: im = im[..., 0]
        h, w = im.shape
        roi = im[h//3:2*h//3, w//3:2*w//3]          # central chest ROI
        roi = roi[roi > 0]
        zt.append(roi.mean() if roi.size else np.nan)
    tvec = np.array([ts(p) for p in pngs]); _, fs = _time_fs(tvec)
    return band_fft_rr(zt, fs), fs, len(zt)


def _nominal_from_name(path):
    m = re.search(r"(\d+)\s*BPM", str(path), re.IGNORECASE)
    return float(m.group(1)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-dir"); ap.add_argument("--radar-csv"); ap.add_argument("--laser-csv"); ap.add_argument("--tof-dir")
    a = ap.parse_args()
    radar_csv, laser_csv, tof_dir = a.radar_csv, a.laser_csv, a.tof_dir
    if a.rec_dir:
        rc = glob.glob(os.path.join(a.rec_dir, "**", "*iq*.csv"), recursive=True)
        lc = [c for c in glob.glob(os.path.join(a.rec_dir, "**", "*.csv"), recursive=True) if "iq" not in c.lower()]
        radar_csv = radar_csv or (rc[0] if rc else None)
        laser_csv = laser_csv or (lc[0] if lc else None)
        td = glob.glob(os.path.join(a.rec_dir, "**"), recursive=True)
        tof_dir = tof_dir or next((d for d in td if os.path.isdir(d) and glob.glob(os.path.join(d, "*.png"))), None)
    nominal = _nominal_from_name(a.rec_dir or radar_csv or laser_csv or "")
    print(f"nominal set-RR (from name): {nominal:.0f} BPM")
    if laser_csv:
        rr, fs, n = laser_rr(laser_csv); print(f"  LASER (ground truth) RR = {rr:6.1f} BPM   (fs={fs:.1f}Hz, n={n})")
    if radar_csv:
        rr, fs, shp = radar_rr(radar_csv); print(f"  RADAR             RR = {rr:6.1f} BPM   (fs={fs:.1f}Hz, bins={shp})")
    if tof_dir:
        rr, fs, n = tof_rr(tof_dir); print(f"  ToF (depth)       RR = {rr:6.1f} BPM   (fs={fs:.1f}Hz, frames={n})")
    print(f"\n  >>> RGB extractors gave ~26-35 BPM for a true-80 neonate. Compare above. <<<")


if __name__ == "__main__":
    main()
