"""Upgraded respiration signal extractor (experiment: high-RR / neonatal frontier).

Motivation (see #22 root-cause): the current benchmark "depth" channel is a whole-ROI
mean-intensity band-pass proxy. Spatial averaging over the ROI CANCELS the sub-pixel,
spatially-varying motion of fast shallow infant breathing, so its signal collapses to
low-frequency drift (dominant 13-19 BPM for gt 58-72). This module fixes that with three
targeted upgrades, each independently ablatable:

  (1) MOTION MAGNIFICATION via per-pixel in-band-power weighting. Instead of a flat mean,
      each pixel's temporal signal is band-passed and the ROI is aggregated with weights
      proportional to each pixel's respiratory-band power. The few pixels that actually
      oscillate at the breathing rate dominate, instead of being washed out. (This is the
      signal-extraction analogue of Eulerian magnification: emphasize the moving pixels.)
  (2) ABDOMEN-FOCUSED ROI. Infants are diaphragmatic/abdominal breathers; abdomen motion
      >> chest motion. Default ROI targets the lower-central torso.
  (3) HARMONIC-AWARE RR. Replace argmax-FFT (which picks sub/super-harmonics -> halving or
      doubling) with a Harmonic Product Spectrum peak, reinforcing the true fundamental.

Plus a widened band (0.13-1.70 Hz = ~8-102 BPM) to match the flow channel and cover neonates.

Two signal sources:
  source="intensity" (Tier 1, no model): tests whether the high-RR info is PRESENT in the
      pixels but destroyed by averaging. Cheap, CPU-only, decisive.
  source="depth" (Tier 2): real monocular depth via src.core.depth_engine (Depth-Anything-V2),
      the physical chest-rise-in-z signal. Heavier; used if Tier 1 is insufficient.

Nothing here edits external/. Standalone so the current extractor stays as the A/B baseline.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
from scipy.signal import detrend, get_window

DEFAULT_LOW_HZ = 0.13   # ~8 BPM
DEFAULT_HIGH_HZ = 1.70  # ~102 BPM (matches the learned-flow band; covers neonatal tachypnea)

# (y0, y1, x0, x1) as fractions of frame height/width.
ROI_BOXES = {
    "proxy":   (0.20, 0.80, 0.25, 0.75),   # replicates the current mean-intensity ROI (baseline)
    "chest":   (0.18, 0.55, 0.25, 0.75),
    "abdomen": (0.45, 0.90, 0.28, 0.72),   # infant diaphragmatic breathing
    "torso":   (0.15, 0.92, 0.20, 0.80),
}


def _decode_gray(video_path: str, ds: Tuple[int, int] = (160, 90), max_frames: int = 1400) -> Tuple[np.ndarray, float]:
    if cv2 is None:
        raise RuntimeError("opencv (cv2) is required to decode video")
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        fr = cv2.resize(fr, ds, interpolation=cv2.INTER_AREA)
        frames.append(fr.astype(np.float32))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return np.stack(frames, 0), fps  # (T, H, W)


def _roi_slice(H: int, W: int, kind: str):
    y0, y1, x0, x1 = ROI_BOXES.get(kind, ROI_BOXES["torso"])
    return slice(int(y0 * H), max(int(y0 * H) + 1, int(y1 * H))), slice(int(x0 * W), max(int(x0 * W) + 1, int(x1 * W)))


def motion_weighted_waveform(stack: np.ndarray, fps: float, low: float, high: float,
                             weight_gamma: float = 2.0, magnify: bool = True) -> Tuple[np.ndarray, float]:
    """Aggregate a (T, H, W) ROI stack into one waveform, weighting pixels by respiratory
    in-band power so breathing-moving pixels dominate. Returns (waveform, inband_concentration)."""
    T = stack.shape[0]
    P = stack.reshape(T, -1).astype(np.float64)
    P = detrend(P, axis=0, type="linear")
    win = get_window("hann", T)[:, None]
    F = np.fft.rfft(P * win, axis=0)                    # (nf, N)
    f = np.fft.rfftfreq(T, d=1.0 / fps)
    band = (f >= low) & (f <= high)
    inband_power = (np.abs(F[band]) ** 2).sum(0)        # (N,)
    total_power = (np.abs(F[1:]) ** 2).sum(0) + 1e-12
    # per-pixel band-passed signal
    Fb = np.zeros_like(F); Fb[band] = F[band]
    Pb = np.fft.irfft(Fb, n=T, axis=0)                  # (T, N)
    if magnify:
        w = inband_power ** weight_gamma
    else:
        w = np.ones_like(inband_power)                  # flat mean (ablation: no magnification)
    w = w / (w.sum() + 1e-12)
    wave = (Pb * w[None, :]).sum(1)
    # ROI-level respiratory in-band concentration (a spectral SNR proxy for reliability)
    Sw = np.abs(np.fft.rfft((wave - wave.mean()) * get_window("hann", T)))
    conc = float((Sw[band] ** 2).sum() / ((Sw[1:] ** 2).sum() + 1e-12))
    return wave.astype(np.float32), conc


def harmonic_rr(wave: np.ndarray, fps: float, low: float, high: float, n_harm: int = 4) -> float:
    """Harmonic Product Spectrum RR: reinforces the true fundamental, suppressing the
    sub-/super-harmonic peaks that cause halving/doubling."""
    n = len(wave)
    win = get_window("hann", n)
    S = np.abs(np.fft.rfft((wave - wave.mean()) * win)).astype(np.float64)
    f = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (f >= low) & (f <= high)
    hps = S.copy()
    for h in range(2, n_harm + 1):
        ds = S[::h]
        hps[:len(ds)] *= ds
    hps[~band] = 0.0
    if not band.any() or hps.max() <= 0:
        return float("nan")
    return float(f[int(np.argmax(hps))] * 60.0)


def argmax_rr(wave: np.ndarray, fps: float, low: float, high: float) -> float:
    n = len(wave)
    S = np.abs(np.fft.rfft(wave - wave.mean()))
    f = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (f >= low) & (f <= high)
    S = np.where(band, S, 0.0)
    return float(f[int(np.argmax(S))] * 60.0)


def _depth_stack(video_path: str, ds: Tuple[int, int], max_frames: int) -> Tuple[np.ndarray, float]:
    """Tier-2: per-frame monocular depth via the existing Depth-Anything-V2 engine."""
    from src.core.depth_engine import DepthEngine  # lazy: heavy import, GPU
    if cv2 is None:
        raise RuntimeError("opencv required")
    engine = DepthEngine()
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    maps = []
    while len(maps) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        d = engine.estimate_depth(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))  # HxW monocular depth map
        maps.append(cv2.resize(np.asarray(d, np.float32), ds, interpolation=cv2.INTER_AREA))
    cap.release()
    if not maps:
        raise RuntimeError(f"no depth frames from {video_path}")
    return np.stack(maps, 0), fps


def extract_upgraded(video_path: str, roi: str = "abdomen", source: str = "intensity",
                     low: float = DEFAULT_LOW_HZ, high: float = DEFAULT_HIGH_HZ,
                     ds: Tuple[int, int] = (160, 90), max_frames: int = 1400,
                     magnify: bool = True, harmonic: bool = True,
                     weight_gamma: float = 2.0) -> Dict:
    """Return {waveform, rr_bpm, inband_concentration, fps, roi, source}."""
    if source == "depth":
        stack, fps = _depth_stack(video_path, ds, max_frames)
    else:
        stack, fps = _decode_gray(video_path, ds, max_frames)
    ys, xs = _roi_slice(stack.shape[1], stack.shape[2], roi)
    roi_stack = stack[:, ys, xs]
    wave, conc = motion_weighted_waveform(roi_stack, fps, low, high, weight_gamma=weight_gamma, magnify=magnify)
    rr = harmonic_rr(wave, fps, low, high) if harmonic else argmax_rr(wave, fps, low, high)
    return {"waveform": wave, "rr_bpm": rr, "inband_concentration": conc,
            "fps": fps, "roi": roi, "source": source}


def extract_proxy_baseline(video_path: str, ds: Tuple[int, int] = (160, 90), max_frames: int = 1400) -> Dict:
    """Replicates the CURRENT depth proxy: whole-central-ROI mean intensity, band 0.20-1.30 Hz,
    argmax-FFT RR. Used as the A/B baseline."""
    stack, fps = _decode_gray(video_path, ds, max_frames)
    ys, xs = _roi_slice(stack.shape[1], stack.shape[2], "proxy")
    mi = stack[:, ys, xs].reshape(stack.shape[0], -1).mean(1)
    mi = detrend(mi.astype(np.float64), type="linear")
    rr = argmax_rr(mi, fps, 0.20, 1.30)
    return {"waveform": mi.astype(np.float32), "rr_bpm": rr, "fps": fps, "roi": "proxy", "source": "intensity"}
