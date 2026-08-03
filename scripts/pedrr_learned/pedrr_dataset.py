"""PedRRDataset: a BaseDataset (air-400) reader for the Pediatric-RR benchmark.

This lives in PROJECT space and only *imports* / *subclasses* the vendored
air-400 code (`external/air-400`). It never modifies anything under `external/`.

Design
------
The Pediatric-RR in-distribution manifests (`data/organized/pedrr_indist_*.json`)
give a *scalar* respiratory rate per video (`gt_rr_bpm`). The air-400 trainer's
`psd_mse` loss (`loss/psdmse.py::PSDMSE`) supervises the *power spectrum*, so it
needs a per-frame *waveform* label, not a scalar. We therefore synthesize a
constant-frequency cosine at `gt_rr_bpm`, reusing the exact math of
`air400_dataset.py::AIR400Dataset._build_via_waveform` for a single segment:

    period  = 60 / rr        (seconds per breath)
    label[t] = cos( (t/fs) * 2*pi / period )  =  cos( 2*pi * (rr/60) * (t/fs) )

whose dominant FFT frequency is exactly `rr/60` Hz (== `rr` BPM). The base class
then resamples this to the video length (a no-op here) and applies the same
`DiffNormalized` transform used for real waveforms; `d/dt cos(2*pi*f*t)` keeps the
peak at `f`, so PSD-MSE is supervised toward the correct RR.

`get_raw_data(data_path)` reads a *combined* manifest (union of the train/val/test
splits, written by the driver) so that the base class's subject-disjoint
`SPLIT_TYPE=custom` / `SPLIT_SUBJECTS` mechanism selects each split. Each subject
has exactly one video, so subject-level aggregation in the trainer == video-level.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

# Vendored air-400 lives on sys.path so `from dataloaders...` resolves against it
# (never modified — imported only). Bootstrap here so this module is importable /
# runnable standalone as well as from the driver.
_AIR400_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "external", "air-400",
)
if _AIR400_DIR not in sys.path:
    sys.path.insert(0, _AIR400_DIR)

from dataloaders.base_dataset import BaseDataset  # noqa: E402  (needs sys.path above)


class PedRRDataset(BaseDataset):
    """Pediatric-RR reader that emits cosine waveform labels from scalar RR."""

    DATASET_NAME = "PEDRR"

    # ------------------------------------------------------------------ #
    # Raw data enumeration
    # ------------------------------------------------------------------ #
    @classmethod
    def get_raw_data(cls, data_path: str) -> Sequence[Dict[str, Any]]:
        """Enumerate clip metadata from a combined Pediatric-RR manifest JSON.

        `data_path` is the path to a JSON *list* of entries, each with at least
        ``subject_id``, ``video_id``, ``full_path`` and ``gt_rr_bpm`` (and
        optionally a precomputed ``fs`` and ``age_months``). The base class
        filters these by subject for each split via ``SPLIT_SUBJECTS``.
        """
        logger = logging.getLogger(cls.__name__)

        if not os.path.exists(data_path):
            logger.error("Pediatric-RR combined manifest not found: %s", data_path)
            raise FileNotFoundError(f"Pediatric-RR combined manifest not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as handle:
            entries = json.load(handle)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Combined manifest {data_path} must be a non-empty JSON list")

        dirs: List[Dict[str, Any]] = []
        for entry in entries:
            subject = str(entry["subject_id"])
            video_id = str(entry["video_id"])
            video_path = entry["full_path"]

            fs = entry.get("fs", None)
            if fs is None or int(fs) <= 0:
                try:
                    cap = cv2.VideoCapture(video_path)
                    fs = int(round(cap.get(cv2.CAP_PROP_FPS)))
                    cap.release()
                except Exception:
                    logger.exception("Failed to read FPS for %s", video_path)
                    continue
            fs = int(fs)
            if fs <= 0:
                logger.warning("Skipping %s: non-positive fps", video_path)
                continue

            dirs.append({
                # `index` must be unique per clip and must NOT contain '_input_'.
                "index": f"{subject}_{video_id}",
                "subject": subject,          # matches SPLIT_SUBJECTS entries (e.g. 'ID-1115')
                "video_path": video_path,
                "label_path": video_path,    # unused for PedRR (label is synthesized)
                "label_format": "pedrr_cosine",
                "fs": fs,
                "gt_rr_bpm": float(entry["gt_rr_bpm"]),
                "age_months": entry.get("age_months", None),
            })

        if not dirs:
            raise ValueError(f"No valid Pediatric-RR clips resolved from {data_path}")

        logger.info("Pediatric-RR: resolved %d clips from %s", len(dirs), data_path)
        return dirs

    # ------------------------------------------------------------------ #
    # Label synthesis
    # ------------------------------------------------------------------ #
    @staticmethod
    def synthesize_cosine_label(num_frames: int, fs: int, rr_bpm: float) -> np.ndarray:
        """Constant-frequency cosine at `rr_bpm` (mirrors AIR400 single-segment case).

        Dominant FFT frequency == rr_bpm/60 Hz == rr_bpm BPM.
        """
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if fs <= 0:
            fs = 30
        period = max(60.0 / max(float(rr_bpm), 1e-6), 1e-6)  # seconds per breath
        t = np.arange(num_frames, dtype=np.float32) / float(fs)
        waveform = np.cos(t * (2.0 * np.pi / period))
        return waveform.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Frame reading (with an optional smoke-test frame cap)
    # ------------------------------------------------------------------ #
    def _smoke_max_frames(self) -> int:
        """Optional cap on decoded frames (smoke tests only); 0 == no cap."""
        try:
            return int(self.preprocess_config.get("SMOKE_MAX_FRAMES", 0) or 0)
        except Exception:
            return 0

    def _read_video_capped(self, video_file: str, max_frames: int) -> np.ndarray:
        """Like BaseDataset._read_video but stops after `max_frames` frames.

        Honours the same decode-time resize (`self.decode_size`) as the base
        reader so the pipeline is identical apart from truncation.
        """
        vid = cv2.VideoCapture(video_file)
        vid.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = vid.read()
        frames: List[np.ndarray] = []
        while success:
            if self.decode_size is not None:
                frame = cv2.resize(frame, self.decode_size, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.nan_to_num(frame, copy=False)
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
            success, frame = vid.read()
        vid.release()
        if len(frames) == 0:
            raise RuntimeError(f"No frames decoded from: {video_file}")
        return np.stack(frames, axis=0)

    def _read_video_wave_files(self, data_dir: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Read frames and synthesize the per-frame cosine label at `gt_rr_bpm`."""
        max_frames = self._smoke_max_frames()
        if max_frames > 0:
            frames = self._read_video_capped(data_dir["video_path"], max_frames)
        else:
            frames = self._read_video(data_dir["video_path"])

        fs = int(data_dir.get("fs", 30))
        rr_bpm = float(data_dir["gt_rr_bpm"])
        labels = self.synthesize_cosine_label(int(frames.shape[0]), fs, rr_bpm)
        return frames, labels


# ---------------------------------------------------------------------- #
# Self-check (run: python scripts/pedrr_learned/pedrr_dataset.py)
# ---------------------------------------------------------------------- #
def _fft_peak_bpm(sig: np.ndarray, fs: int, low_hz: float = 0.1, high_hz: float = 2.0) -> float:
    """Dominant FFT frequency of `sig` within [low_hz, high_hz], in BPM."""
    from scipy import signal as sps

    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - sig.mean()
    n = 2 ** (len(sig) - 1).bit_length()
    f, p = sps.periodogram(sig, fs=fs, nfft=n, detrend=False)
    band = (f >= low_hz) & (f <= high_hz)
    return float(f[band][np.argmax(p[band])] * 60.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("== PedRR cosine label unit check ==")
    fs = 25
    dur_s = 60  # full-length clip -> fine FFT resolution
    n = fs * dur_s
    ok = True
    for rr in [16.0, 27.0, 33.5, 50.0, 74.0, 96.5]:
        lab = PedRRDataset.synthesize_cosine_label(n, fs, rr)
        # raw cosine
        peak_raw = _fft_peak_bpm(lab, fs)
        # DiffNormalized (as the trainer sees it) then cumsum-recovered
        diff = np.diff(lab)
        diff = diff / (np.std(diff) + 1e-8)
        peak_diff = _fft_peak_bpm(np.cumsum(diff), fs)
        bin_bpm = (fs / (2 ** (n - 1).bit_length())) * 60.0
        good = abs(peak_raw - rr) <= bin_bpm and abs(peak_diff - rr) <= 2 * bin_bpm
        ok = ok and good
        print(f"rr={rr:6.1f}  raw_peak={peak_raw:6.2f}  diffnorm_peak={peak_diff:6.2f}  "
              f"(fft bin={bin_bpm:.2f} bpm)  {'OK' if good else 'FAIL'}")
    print("ALL OK" if ok else "SOME FAILED")
