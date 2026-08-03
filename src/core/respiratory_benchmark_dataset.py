"""
Signal dataset utilities for the respiratory benchmark pipeline.

The benchmark path expects pre-extracted signal files generated from the canonical
manifests. It supports the repo's standard `.npz` format and a few legacy
fallbacks to reduce friction while the data pipeline is being consolidated.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def slugify(text: str) -> str:
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def signal_filename_from_entry(entry: Dict) -> str:
    dataset = slugify(entry.get("dataset", "unknown"))
    subject = slugify(entry.get("subject_id", "unknown_subject"))
    video = slugify(entry.get("video_id", "unknown_video"))
    return f"{dataset}_{subject}_{video}.npz"


def parse_signal_name(path: Path) -> Dict[str, str]:
    stem = path.stem
    parts = stem.split("_")
    dataset = parts[0] if len(parts) >= 1 else "unknown"
    subject = parts[1] if len(parts) >= 2 else "unknown_subject"
    video = parts[2] if len(parts) >= 3 else stem
    return {"dataset": dataset, "subject_id": subject, "video_id": video}


CANONICAL_DATASET_NAME_MAP = {
    "AIR-125": "AIR-125",
    "AIR_125": "AIR-125",
    "air-125": "AIR-125",
    "air_125": "AIR-125",
    "AIR-400": "AIR-400",
    "AIR_400": "AIR-400",
    "air-400": "AIR-400",
    "air_400": "AIR-400",
    "Pediatric-RR": "Pediatric-RR",
    "Pediatric-RR-332": "Pediatric-RR",
    "Pediatric_RR_332": "Pediatric-RR",
    "pediatric-rr": "Pediatric-RR",
}

DATASET_ID_MAP = {
    "unknown": 0,
    "AIR-125": 1,
    "AIR-400": 2,
    "Pediatric-RR": 3,
}


def canonical_dataset_name(name: str) -> str:
    text = str(name or "").strip()
    return CANONICAL_DATASET_NAME_MAP.get(text, text or "unknown")


def dataset_name_to_id(name: str) -> int:
    return int(DATASET_ID_MAP.get(canonical_dataset_name(name), 0))


class RespiratoryBenchmarkDataset(Dataset):
    def __init__(
        self,
        signal_dir: str | Path,
        manifest_path: Optional[str | Path] = None,
        seq_len: int = 300,
        include_audio: bool = True,
        file_glob: str = "*.npz",
        training: bool = False,
        enable_augmentation: bool = False,
        general_augment_prob: float = 0.12,
        air125_augment_prob: float = 0.55,
        air125_augment_strength: float = 1.75,
        enable_event_augmentation: bool = False,
        apnea_prob: float = 0.15,
        motion_prob: float = 0.2,
        synth_highrr_prob: float = 0.0,
    ):
        self.synth_highrr_prob = float(synth_highrr_prob)
        self.signal_dir = Path(signal_dir)
        self.seq_len = int(seq_len)
        self.include_audio = bool(include_audio)
        self.training = bool(training)
        self.enable_augmentation = bool(enable_augmentation)
        self.general_augment_prob = float(general_augment_prob)
        self.air125_augment_prob = float(air125_augment_prob)
        self.air125_augment_strength = float(air125_augment_strength)
        self.enable_event_augmentation = bool(enable_event_augmentation)
        self.apnea_prob = float(apnea_prob)
        self.motion_prob = float(motion_prob)
        self.records: List[Dict] = []

        if manifest_path:
            manifest_file = Path(manifest_path)
            with open(manifest_file, "r", encoding="utf-8") as handle:
                manifest_entries = json.load(handle)
            for entry in manifest_entries:
                sig_path = self.signal_dir / signal_filename_from_entry(entry)
                if sig_path.exists():
                    record = dict(entry)
                    record["dataset"] = canonical_dataset_name(record.get("dataset", "unknown"))
                    record["signal_path"] = str(sig_path.resolve())
                    self.records.append(record)
        else:
            for sig_path in sorted(self.signal_dir.glob(file_glob)):
                parsed = parse_signal_name(sig_path)
                parsed["dataset"] = canonical_dataset_name(parsed.get("dataset", "unknown"))
                parsed["signal_path"] = str(sig_path.resolve())
                self.records.append(parsed)

        if not self.records:
            raise FileNotFoundError(
                f"No benchmark signal files resolved from signal_dir={self.signal_dir} "
                f"manifest_path={manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        path = Path(str(record["signal_path"]))
        data = np.load(path, allow_pickle=False)

        has_depth = "depth" in data or "signal" in data
        has_flow = "flow" in data or "signal" in data
        has_audio = self.include_audio and "audio" in data

        depth = self._pick_signal(data, "depth")
        flow = self._pick_signal(data, "flow", fallback_like=depth)
        audio = self._pick_signal(data, "audio", fallback_like=depth) if self.include_audio else np.zeros_like(depth)

        # Content-aware availability: a constant/zero channel carries no signal
        # (e.g. the all-zeros audio placeholder written for video-only datasets like
        # Pediatric-RR). Mark it UNAVAILABLE so reliability-fusion never trusts a
        # degenerate constant channel — z-norm of a constant collapses to zeros and
        # the per-modality head otherwise emits a fixed (wrong) BPM for every clip.
        def _informative(x: np.ndarray) -> bool:
            return bool(np.std(np.asarray(x, dtype=np.float64)) > 1e-6)

        has_depth = has_depth and _informative(depth)
        has_flow = has_flow and _informative(flow)
        has_audio = has_audio and _informative(audio)

        gt_rr = self._pick_scalar(data, ["gt", "bpm"], default=record.get("gt_rr_bpm", 0.0))
        age_months = self._pick_scalar(data, ["age_months"], default=record.get("age_months", 6))

        dataset = self._pick_string(data, ["source_dataset", "dataset"], default=record.get("dataset", "unknown"))
        dataset = canonical_dataset_name(dataset)
        subject = self._pick_string(data, ["subject_id", "subject"], default=record.get("subject_id", "unknown_subject"))
        video = self._pick_string(data, ["video_id"], default=record.get("video_id", path.stem))

        # Per-clip source fps (written by the extractor for cross-dataset clips);
        # 0.0 when absent -> _process_signal treats it as already-canonical (no resample).
        src_fps = self._pick_scalar(data, ["fps"], default=0.0)
        depth = self._process_signal(depth, fps=src_fps)
        flow = self._process_signal(flow, fps=src_fps)
        audio = self._process_signal(audio, fps=src_fps)

        # Synth high-RR augmentation: time-compress the z-normed signals to synthesize a higher
        # respiratory rate with a correctly-scaled label -> attacks neonatal (high-RR) data scarcity.
        if (self.training and self.enable_augmentation and self.synth_highrr_prob > 0
                and gt_rr > 12.0 and np.random.rand() < self.synth_highrr_prob):
            target = float(np.random.uniform(55.0, 90.0))
            c = target / gt_rr
            if 1.2 <= c <= 4.0:
                depth = self._time_compress(depth, c)
                flow = self._time_compress(flow, c)
                audio = self._time_compress(audio, c)
                gt_rr = target

        if self.training and self.enable_augmentation:
            depth = self._augment_signal(depth, dataset=dataset, modality="depth")
            flow = self._augment_signal(flow, dataset=dataset, modality="flow")
            audio = self._augment_signal(audio, dataset=dataset, modality="audio")

        apnea_target = 0.0
        motion_target = 0.0
        if self.training and self.enable_event_augmentation:
            from src.utils.augmentations import apply_event_augmentations

            augmented, events = apply_event_augmentations(
                {"depth": depth, "flow": flow, "audio": audio},
                apnea_prob=self.apnea_prob,
                motion_prob=self.motion_prob,
            )
            # Re-normalize so injected windows keep their event signature while
            # the input distribution stays consistent with the model's inputs.
            depth = self._renormalize(augmented["depth"])
            flow = self._renormalize(augmented["flow"])
            audio = self._renormalize(augmented["audio"])
            apnea_target = float(events["apnea"])
            motion_target = float(events["motion"])

        availability = np.array(
            [
                float(has_depth),
                float(has_flow),
                float(has_audio),
            ],
            dtype=np.float32,
        )

        return {
            "depth": torch.from_numpy(depth).float(),
            "flow": torch.from_numpy(flow).float(),
            "audio": torch.from_numpy(audio).float(),
            "gt_rr": torch.tensor(float(gt_rr)).float(),
            "age_months": torch.tensor(float(age_months)).float(),
            "apnea_target": torch.tensor(float(apnea_target)).float(),
            "motion_target": torch.tensor(float(motion_target)).float(),
            "modality_mask": torch.from_numpy(availability).float(),
            "dataset": dataset,
            "dataset_id": torch.tensor(dataset_name_to_id(dataset), dtype=torch.long),
            "subject_id": subject,
            "video_id": video,
            "signal_path": str(path),
        }

    def dataset_counts(self) -> Dict[str, int]:
        counts = Counter(canonical_dataset_name(record.get("dataset", "unknown")) for record in self.records)
        return dict(sorted(counts.items()))

    def source_balanced_sample_weights(
        self,
        power: float = 1.0,
        source_boosts: Optional[Dict[str, float]] = None,
    ) -> List[float]:
        counts = self.dataset_counts()
        boosts = {canonical_dataset_name(key): float(value) for key, value in (source_boosts or {}).items()}
        weights: List[float] = []
        for record in self.records:
            dataset = canonical_dataset_name(record.get("dataset", "unknown"))
            count = max(1, int(counts.get(dataset, 1)))
            base = 1.0 / float(count) if power > 0 else 1.0
            if power != 1.0 and power > 0:
                base = base ** float(power)
            weights.append(float(base * boosts.get(dataset, 1.0)))
        return weights

    def _pick_signal(self, data: np.lib.npyio.NpzFile, key: str, fallback_like: Optional[np.ndarray] = None) -> np.ndarray:
        if key in data:
            return np.asarray(data[key], dtype=np.float32).flatten()
        if "signal" in data and key in ("depth", "flow"):
            return np.asarray(data["signal"], dtype=np.float32).flatten()
        if fallback_like is not None:
            return np.zeros_like(np.asarray(fallback_like, dtype=np.float32).flatten())
        return np.zeros(self.seq_len, dtype=np.float32)

    @staticmethod
    def _pick_scalar(data: np.lib.npyio.NpzFile, keys: List[str], default: float = 0.0) -> float:
        for key in keys:
            if key in data:
                value = np.asarray(data[key]).reshape(-1)
                if value.size:
                    return float(value[0])
        return float(default)

    @staticmethod
    def _pick_string(data: np.lib.npyio.NpzFile, keys: List[str], default: str = "") -> str:
        for key in keys:
            if key in data:
                value = np.asarray(data[key]).reshape(-1)
                if value.size:
                    return str(value[0])
        return str(default)

    def _process_signal(self, signal: np.ndarray, fps: float = 0.0, canonical_fps: float = 25.0) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float32).flatten()
        if sig.size == 0:
            sig = np.zeros(self.seq_len, dtype=np.float32)

        # FPS normalization: the model regresses BPM from cycles-per-window, so a
        # clip recorded at a different frame rate would be mis-scaled. Resample to a
        # canonical fps BEFORE the fixed-length crop. Inert when fps is unknown (0)
        # or already canonical -> in-dist Pediatric-RR (uniform 25 fps) is unchanged.
        if fps and fps > 0 and abs(fps - canonical_fps) > 0.1 and sig.size > 1:
            new_len = max(2, int(round(sig.size * canonical_fps / float(fps))))
            xp = np.linspace(0.0, 1.0, num=sig.size, dtype=np.float64)
            xq = np.linspace(0.0, 1.0, num=new_len, dtype=np.float64)
            sig = np.interp(xq, xp, sig).astype(np.float32)

        if sig.size > self.seq_len:
            start = max(0, (sig.size - self.seq_len) // 2)
            sig = sig[start : start + self.seq_len]
        elif sig.size < self.seq_len:
            pad = self.seq_len - sig.size
            if sig.size == 1:
                sig = np.pad(sig, (0, pad), mode="edge")
            else:
                sig = np.pad(sig, (0, pad), mode="constant")

        mean = float(sig.mean())
        std = float(sig.std())
        if std > 1e-6:
            sig = (sig - mean) / std
        else:
            sig = sig - mean
        return sig.astype(np.float32, copy=False)

    @staticmethod
    def _time_compress(sig: np.ndarray, c: float) -> np.ndarray:
        """Compress a fixed-length signal in time by factor c (breathing c-times faster), tiling back
        to the original length, then z-renormalize. Verified to scale the dominant RR by ~c (<0.5 BPM)."""
        s = np.asarray(sig, dtype=np.float64).flatten()
        n = len(s)
        m = max(2, int(round(n / c)))
        comp = np.interp(np.linspace(0, n - 1, m), np.arange(n), s)
        out = np.resize(comp, n)
        mean, std = out.mean(), out.std()
        return ((out - mean) / std if std > 1e-6 else out - mean).astype(np.float32)

    @staticmethod
    def _renormalize(signal: np.ndarray) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float32).flatten()
        mean = float(sig.mean())
        std = float(sig.std())
        if std > 1e-6:
            sig = (sig - mean) / std
        else:
            sig = sig - mean
        return sig.astype(np.float32, copy=False)

    def _augment_signal(self, signal: np.ndarray, dataset: str, modality: str) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float32).copy()
        if sig.size == 0:
            return sig

        is_air125 = canonical_dataset_name(dataset) == "AIR-125"
        if is_air125:
            prob = self.air125_augment_prob
            strength = self.air125_augment_strength
        else:
            prob = self.general_augment_prob
            strength = 1.0

        if np.random.rand() >= prob:
            return sig

        modality_scale = {"depth": 1.0, "flow": 1.1, "audio": 0.7}.get(modality, 1.0)
        strength = float(strength * modality_scale)

        if np.random.rand() < 0.85:
            amp = 1.0 + np.random.uniform(-0.12, 0.12) * strength
            sig *= float(amp)

        if np.random.rand() < 0.9:
            noise_std = 0.02 * strength
            sig += np.random.normal(0.0, noise_std, size=sig.shape).astype(np.float32)

        if np.random.rand() < 0.55:
            phase = np.random.uniform(0.0, 2.0 * np.pi)
            freq = np.random.uniform(0.35, 1.4)
            drift = (0.08 * strength) * np.sin(np.linspace(0.0, freq * 2.0 * np.pi, sig.size, dtype=np.float32) + phase)
            sig += drift.astype(np.float32)

        if np.random.rand() < 0.45:
            max_shift = max(1, int(0.05 * self.seq_len * strength))
            shift = int(np.random.randint(-max_shift, max_shift + 1))
            sig = np.roll(sig, shift)

        if np.random.rand() < 0.5:
            seg_len = max(4, int(np.random.uniform(0.04, 0.12) * self.seq_len))
            start = int(np.random.randint(0, max(1, self.seq_len - seg_len + 1)))
            attenuation = float(np.random.uniform(0.0, 0.25))
            sig[start : start + seg_len] *= attenuation

        if np.random.rand() < 0.35:
            kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
            sig = np.convolve(sig, kernel, mode="same").astype(np.float32)

        mean = float(sig.mean())
        std = float(sig.std())
        if std > 1e-6:
            sig = (sig - mean) / std
        else:
            sig = sig - mean
        return sig.astype(np.float32, copy=False)
