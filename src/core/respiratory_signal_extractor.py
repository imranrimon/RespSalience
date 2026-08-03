"""Production respiratory signal extractor: chest-ROI vertical optical flow + band-pass intensity.

This is the extraction that fixed the 12-60mo "sensing ceiling". On Pediatric-RR it lifted
tachypnea AUROC in the 12-60mo band from 0.51 (chance) to 0.95 and halved overall RR MAE
(11.2 -> 5.6) after retraining, because the previous global depth/flow pipeline
(`RespiratoryController.process_video`) was discarding the breathing signal for larger / more
mobile children. Validated raw single-modality recovery (recovered-RR vs GT over all 332 clips):
12-60mo r=0.83, 2-12mo r=0.54, <2mo r=0.33.

Per RGB video it produces three standardized 1-D signals (RR lives in the frequency, which
standardization preserves):
  * flow  <- mean VERTICAL Farneback optical flow inside a central chest ROI  (the winner)
  * depth <- band-pass (0.20-1.30 Hz) filtered ROI mean-intensity  (complementary intensity view)
  * audio <- zeros  (the video datasets here carry no usable respiratory audio track; the key is
             still written so downstream modality-availability masks match the validated run)

`scripts/extract_improved_signals.py` (QA / one-off) and `scripts/extract_benchmark_signals.py`
(the manifest-driven paper pipeline) both import `extract_video_signals` from here, so the exact
signal a model trains on is defined in ONE place and re-extraction stays reproducible. Do not fork
the numerical body; change it here if it must change, then re-extract + re-validate.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import cv2

# --- validated configuration (do not change without re-extracting + re-validating) -------------
BAND: Tuple[float, float] = (0.20, 1.30)      # Hz; covers ~12-90 BPM
MAX_FRAMES: int = 1400
DS: Tuple[int, int] = (160, 90)               # cv2.resize target (width, height) for speed
ROI_FRAC: Tuple[float, float, float, float] = (0.20, 0.80, 0.25, 0.75)  # (y0, y1, x0, x1)
# Farneback params, kept positional-equivalent to the validated call:
#   pyr_scale=0.5, levels=2, winsize=15, iterations=2, poly_n=5, poly_sigma=1.1, flags=0
_FB = (0.5, 2, 15, 2, 5, 1.1, 0)


def band_rr(x: np.ndarray, fps: float, band: Tuple[float, float] = BAND) -> float:
    """FFT peak (in BPM) of a cubic-detrended, Hann-windowed signal, restricted to `band`."""
    x = np.asarray(x, float)
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 3), np.arange(len(x)))
    if x.std() < 1e-9:
        return float("nan")
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), d=1.0 / fps)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return float("nan")
    Pb = P.copy()
    Pb[~m] = 0
    return float(f[int(np.argmax(Pb))] * 60.0)


def standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std (mean-center only if variance is ~0)."""
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x - x.mean()


# --- Tier-1 "gated_consensus" variant: chest+abdomen search region + outlier-rejecting consensus.
# Validated in the A/B probe (extraction-sota-and-tier1-probe memory): recovers cross-framing cases
# (AIR-125 recovered-RR r 0.10->0.87) and improves neonatal MAE, while holding in-distribution
# (pedrr 12-60mo r 0.93). It does NOT fix AIR-400 (that needs the learned recipe). Default variant
# stays "fixed" so the validated pedrr_v2 signals reproduce byte-for-byte.
SEARCH_FRAC: Tuple[float, float, float, float] = (0.10, 0.95, 0.15, 0.85)  # chest+abdomen + slack
_CHEST_BOX: Tuple[float, float, float, float] = (0.05, 0.55, 0.15, 0.85)   # within the search region
_ABDOMEN_BOX: Tuple[float, float, float, float] = (0.45, 0.95, 0.15, 0.85)


def inband_concentration(x: np.ndarray, fps: float, band: Tuple[float, float] = BAND) -> float:
    """Peak in-band power / total power of a cubic-detrended, Hann-windowed signal (periodicity)."""
    x = np.asarray(x, float)
    if len(x) < 8 or x.std() < 1e-9:
        return 0.0
    w = (x - np.polyval(np.polyfit(np.arange(len(x)), x, 3), np.arange(len(x)))) * np.hanning(len(x))
    P = np.abs(np.fft.rfft(w)) ** 2
    f = np.fft.rfftfreq(len(w), d=1.0 / fps)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or P.sum() < 1e-12:
        return 0.0
    return float(P[m].max() / (P.sum() + 1e-12))


def _dense_vflow(reg: np.ndarray) -> np.ndarray:
    """(T, h, w) grayscale ROI -> (T-1, h, w) per-pixel VERTICAL Farneback flow."""
    T = reg.shape[0]
    vf = np.empty((T - 1,) + reg.shape[1:], np.float32)
    prev = reg[0]
    for t in range(1, T):
        fl = cv2.calcOpticalFlowFarneback(prev.astype(np.uint8), reg[t].astype(np.uint8), None, *_FB)
        vf[t - 1] = fl[..., 1]
        prev = reg[t]
    return vf


def _box_mean(vf: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    t, h, w = vf.shape
    y0, y1, x0, x1 = box
    return vf[:, int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].reshape(t, -1).mean(1)


def _gated_consensus_signal(vf: np.ndarray, fps: float, band: Tuple[float, float] = BAND) -> np.ndarray:
    """Median-RR consensus over {chest, abdomen, best-in-band PCA} candidate flow signals: return the
    candidate time series whose recovered RR is closest to the median (rejects the outlier)."""
    chest = _box_mean(vf, _CHEST_BOX)
    abd = _box_mean(vf, _ABDOMEN_BOX)
    cands = [chest, abd]
    try:
        t = vf.shape[0]
        X = vf[:, ::3, ::3].reshape(t, -1).astype(np.float64)
        X = X - X.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        best_c, pca_sig = -1.0, None
        for j in range(min(6, U.shape[1])):
            c = inband_concentration(U[:, j], fps, band)
            if c > best_c:
                best_c, pca_sig = c, U[:, j]
        if pca_sig is not None:
            cands.append(pca_sig)
    except np.linalg.LinAlgError:
        pass
    scored = [(sig, band_rr(sig, fps, band)) for sig in cands]
    scored = [z for z in scored if z[1] == z[1]]  # drop NaN recovered-RR
    if not scored:
        return chest
    med = float(np.median([z[1] for z in scored]))
    return min(scored, key=lambda z: abs(z[1] - med))[0]


def extract_video_signals(
    video_path: str,
    variant: str = "fixed",
    max_frames: int = MAX_FRAMES,
    ds: Tuple[int, int] = DS,
    band: Tuple[float, float] = BAND,
    roi_frac: Tuple[float, float, float, float] = ROI_FRAC,
    min_frames: int = 30,
) -> Optional[Dict[str, object]]:
    """Extract standardized depth/flow/audio respiration signals from one RGB video.

    variant="fixed" (default): the validated pedrr_v2 extractor -- mean vertical Farneback flow in a
    fixed central chest ROI (reproduces pedrr_v2 byte-for-byte). variant="gated_consensus": the
    Tier-1 robustness upgrade -- dense vertical flow over a chest+abdomen search region with an
    outlier-rejecting median-RR consensus over {chest, abdomen, PCA} candidates.

    Returns a dict with float32 arrays ``flow``, ``depth``, ``audio`` (length T), the video ``fps``,
    and QA recovered-RR ``rr_flow`` / ``rr_depth`` (BPM). None if unreadable / shorter than min_frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(fr, ds), cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(frames) < min_frames:
        return None

    vol = np.stack(frames)
    T, H, W = vol.shape

    if variant == "gated_consensus":
        sy0, sy1, sx0, sx1 = SEARCH_FRAC
        reg = vol[:, int(sy0 * H):int(sy1 * H), int(sx0 * W):int(sx1 * W)]
        core = _gated_consensus_signal(_dense_vflow(reg), fps, band)      # length T-1
        flow = np.concatenate(([0.0], np.asarray(core, float)))           # leading 0 -> length T
        mi = reg.reshape(T, -1).mean(1).astype(np.float64)
    elif variant == "fixed":
        ys, ye, xs, xe = (int(roi_frac[0] * H), int(roi_frac[1] * H),
                          int(roi_frac[2] * W), int(roi_frac[3] * W))
        roi = vol[:, ys:ye, xs:xe]
        # mean VERTICAL Farneback optical flow in the chest ROI (leading 0 to keep length T)
        flow = [0.0]
        prev = roi[0]
        for t in range(1, T):
            fl = cv2.calcOpticalFlowFarneback(prev.astype(np.uint8), roi[t].astype(np.uint8),
                                              None, *_FB)
            flow.append(float(fl[..., 1].mean()))
            prev = roi[t]
        flow = np.array(flow)
        mi = roi.reshape(T, -1).mean(1).astype(np.float64)
    else:
        raise ValueError(f"unknown variant: {variant!r} (expected 'fixed' or 'gated_consensus')")

    # depth proxy: band-pass-filtered mean-intensity (intensity-based respiration view)
    F = np.fft.rfft(mi - mi.mean())
    f = np.fft.rfftfreq(len(mi), d=1.0 / fps)
    F[(f < band[0]) | (f > band[1])] = 0
    depth_sig = np.fft.irfft(F, n=len(mi))

    flow = np.asarray(flow, float)
    return {
        "flow": standardize(flow).astype(np.float32),
        "depth": standardize(depth_sig).astype(np.float32),
        "audio": np.zeros(len(flow), np.float32),
        "fps": float(fps),
        "rr_flow": band_rr(flow, fps, band),
        "rr_depth": band_rr(depth_sig, fps, band),
    }
