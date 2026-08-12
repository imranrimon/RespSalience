"""Train PhysFormer per-fold on Pediatric-RR and emit RR predictions for full_metric_comparison.

PhysFormer (Yu et al., CVPR-2022 -- ``ViT_ST_ST_Compact3_TDC_gra_sharp`` in
``external/rPPG-Toolbox/neural_methods/model/PhysFormer.py``) is a modern RGB-video
transformer baseline. It does **not** fit the frame-wise air-400 harness (its own
NegPearson + frequency loss, its own trainer), so we produce its RR predictions with
this standalone driver, then hand the JSON to ``scripts/physformer_to_csv.py`` (B3 glue)
which converts them to the standard per-fold ``prism_breath_predictions.csv`` schema.

WHAT THIS DOES
--------------
For each of the 5 subject-disjoint folds in ``data/organized/kfold_learned/fold{k}``:
  * builds train/val/test datasets by **reusing the exact air-400 PedRR preprocessing
    cache** (``data/pedrr_learned_cache/full``) -- identical YOLO body-ROI crop,
    coarse2fine optical flow, standardization, and 180-frame chunking used to train the
    other learned extractors (EfficientPhys / TS-CAN / VIRENet / DeepPhys). Only the
    *subject partition* differs per fold; the cached ``.npy`` chunks are split-agnostic
    and shared across folds (keyed by preprocess hash + ``{subject}_{video}`` index).
  * trains PhysFormer full-budget, selects the epoch with the lowest **validation RR MAE**
    (computed with the project's respiratory FFT band, not PhysFormer's HR band),
  * infers RR (BPM) for the held-out test subjects via FFT of the predicted waveform in
    the Pediatric-RR band, aggregated (mean) per video,
  * writes ``outputs/physformer_preds.json`` = ``{"<subject_id>_<video_id>": rr_bpm}``.

============================ DATA-ADAPTER / LABEL DESIGN =============================
(All claims verified by reading the code -- file:line evidence in the report below.)

Cache tensor (``pedrr_learned_cache/full/PEDRR/<hash>/{sub}_{vid}_input_NNN.npy``):
    shape (180, 96, 96, 6) NDHWC float32  == [flow(dx,dy,mag) | standardized-RGB].
    (pre_processor.py:243  ``np.concatenate([flow, frames], axis=-1)``.)
    ``BaseDataset.__getitem__`` (base_dataset.py:88) transposes NDHWC->NDCHW because the
    PedRR config uses ``DATA_FORMAT='NDCHW'``, so an item is (D=180, C=6, 96, 96).
    Channels [0:3]=flow, [3:6]=standardized RGB frames (report_rr in
    train_learned_extractor_pedrr.py:319 confirms: flow is the FIRST ``in_channels``).

Label (``..._label_NNN.npy``): shape (180,) -- a constant-frequency cosine synthesized at
    ``gt_rr_bpm`` (pedrr_dataset.py:123) then DiffNormalized (LABEL_NORMALIZE_TYPE), whose
    dominant FFT frequency is exactly rr/60 Hz. This IS a per-clip respiratory *waveform*,
    so PhysFormer's NegPearson loss has a valid target. (There is no *real* respiratory
    waveform GT for Pediatric-RR -- only a scalar RR per video -- so this synthetic
    waveform is the same supervision signal the other extractors already use.)

INPUT ADAPTER (per item):  (180,6,96,96)
    -> select 3 channels (``--input frames`` = std-RGB[3:6], default; ``--input flow`` = [0:3])
    -> crop time to ``--chunk-len`` (<=180, multiple of 4)
    -> spatial bilinear resize 96 -> 128  (MANDATORY: PhysFormer hard-codes a 4x4 patch
       grid = resize/(8*patch_size); 96 -> 3x3 grid crashes. resize=32*patch_size=128.)
    -> permute to (3, T, 128, 128).  Batched -> (B, 3, T, 128, 128) as forward() expects.
    Default input = standardized RGB frames because PhysFormer is natively an RGB model
    (Standardized is one of its supported DATA_TYPEs; its Stem has BatchNorm3d so the exact
    scale is irrelevant). ``--input flow`` makes PhysFormer vs EfficientPhys a pure-backbone
    comparison (identical input modality) -- see report for the tradeoff.

LOSS:  loss = pearson_w * NegPearson(norm_rPPG, label) + freq_w * PSDMSE(rPPG, label, fs)
    * NegPearson (PhysNetNegPearsonLoss.py) is PhysFormer's signature temporal loss.
    * PSDMSE (air-400 loss/psdmse.py) over the Pediatric-RR band **0.20-1.70 Hz** is the
      IDENTICAL frequency supervision used to train EfficientPhys/TS-CAN/VIRENet here.
    We deliberately DROP PhysFormer's native ``TorchLossComputer`` CE/DLDL frequency loss
    and its ``get_hr`` model-selection: both hard-code a HEART-RATE band
    (PhysFormerLossComputer.py:63/76/103/110 use bpm 40-180; PhysFormerTrainer.py:272
    ``get_hr`` uses 30-180 bpm = 0.5-3.0 Hz). Pediatric-RR spans 16-96.5 BPM
    (0.27-1.61 Hz), so those HR bands would EXCLUDE / mis-supervise every clip below
    ~40 BPM. PSDMSE(0.20-1.70 Hz) keeps supervision correct and the comparison fair.

RR INFERENCE:  FFT (periodogram) of the predicted rPPG, peak within 0.20-1.70 Hz -> BPM,
    fs = 25 (all 332 PedRR videos are 25 fps). Mean over a video's chunks -> one RR/video.
    The JSON key ``{subject}_{video}`` == BaseDataset ``filename`` == what
    physformer_to_csv.py expects (each subject has exactly one video, so video==subject).

============================== RUN (on a GPU box) ===================================
    python scripts/train_physformer_pedrr.py --epochs 15          # all 5 folds
    python scripts/train_physformer_pedrr.py --fold 0 --epochs 15 # one fold
    python scripts/train_physformer_pedrr.py --skip-existing      # resume
Then:  python scripts/physformer_to_csv.py   (reads outputs/physformer_preds.json)

NOTE: This box CANNOT train (WinError 5 / no GPU multiprocessing); run externally.
num_workers=0 everywhere; small batch. Resumable (per-fold, incremental JSON write).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Deterministic cuBLAS workspace must be set before torch is imported (only matters if
# --strict-deterministic is used, which calls air-400's use_deterministic_algorithms).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]                       # G:\Respiratory Project
AIR400_DIR = ROOT / "external" / "air-400"
PEDRR_PKG_DIR = ROOT / "scripts" / "pedrr_learned"
RPPG_DIR = ROOT / "external" / "rPPG-Toolbox"

# air-400 goes to the FRONT so ``import main`` / ``loss`` / ``models`` / ``dataloaders``
# resolve to it (rPPG-Toolbox ships a colliding main.py + a top-level config.py). rPPG-Toolbox
# is APPENDED (lowest priority) so it can never shadow an air-400 module; ``neural_methods``
# is unique to it and still resolves. pedrr_learned holds PedRRDataset.
for _p in (str(AIR400_DIR), str(PEDRR_PKG_DIR)):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(PEDRR_PKG_DIR))
sys.path.insert(0, str(AIR400_DIR))
if str(RPPG_DIR) not in sys.path:
    sys.path.append(str(RPPG_DIR))

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402
from scipy import signal as sps                                  # noqa: E402
from torch.utils.data import DataLoader, Dataset                 # noqa: E402

import main as air_main                                          # noqa: E402  (air-400/main.py)
from pedrr_dataset import PedRRDataset                           # noqa: E402  (scripts/pedrr_learned)
from loss.psdmse import PSDMSE                                   # noqa: E402  (air-400)
from neural_methods.model.PhysFormer import (                    # noqa: E402  (rPPG-Toolbox)
    ViT_ST_ST_Compact3_TDC_gra_sharp,
)
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson  # noqa: E402

logger = logging.getLogger("physformer_pedrr")

# --------------------------------------------------------------------------- #
# Constants (must stay in lockstep with the existing 89 GB cache & siblings)
# --------------------------------------------------------------------------- #
DATASET_NAME = "PEDRR"

# Pediatric-RR respiratory band (train_learned_extractor_pedrr.py:82-83). Manifest RR
# spans 16.0-96.5 BPM (0.267-1.608 Hz); bracketed with headroom.
PEDRR_LOW_HZ = 0.20
PEDRR_HIGH_HZ = 1.70

# EXACT preprocess block of the SOTA body_roi coarse2fine recipe used to build the cache.
# Copied verbatim from train_learned_extractor_pedrr.py:BASE_PREPROCESS with CHUNK_LENGTH
# pinned to 180. This dict's hash == the cache subdir name
# (4ab759328c...); DO NOT edit any value or the cache stops matching and preprocessing
# would re-run (needs GPU YOLO + optical flow).
BASE_PREPROCESS = {
    "DO_DOWNSAMPLE_BEFORE_PREPROCESS": True,
    "DOWNSAMPLE_SIZE_BEFORE_PREPROCESS": [320, 180],
    "DATA_NORMALIZE_TYPE": "Standardized",
    "LABEL_NORMALIZE_TYPE": "DiffNormalized",
    "FLOW_NORMALIZE_TYPE": "Standardized",
    "DO_CHUNK": True,
    "CHUNK_LENGTH": 180,                    # <- FIXED (cache hash); model T set via --chunk-len
    "DO_CROP_INFANT_REGION": True,
    "DYNAMIC_DETECTION": True,
    "DYNAMIC_DETECTION_FREQUENCY": 10,
    "INFANT_REGION": "Body",
    "BODY_CONF_THRESHOLD": 0.25,
    "FACE_CONF_THRESHOLD": 0.5,
    "CHEST_SHIFT_ALPHA": 0.2,
    "DO_OPTICAL_FLOW": True,
    "OF_METHOD": "coarse2fine",
    "DO_DOWNSAMPLE_BEFORE_TRAINING": True,
    "DOWNSAMPLE_SIZE_BEFORE_TRAINING": [96, 96],
}
CACHE_CHUNK_LEN = int(BASE_PREPROCESS["CHUNK_LENGTH"])           # 180
IN_CHANNELS = 3                                                  # flow[0:3] / frames[3:6]

# PhysFormer hyperparameters (rPPG-Toolbox config.py:342-348 defaults).
PF_PATCH_SIZE = 4
PF_DIM = 96
PF_FF_DIM = 144
PF_NUM_HEADS = 4
PF_NUM_LAYERS = 12
PF_THETA = 0.7
PF_GRA_SHARP = 2.0


# --------------------------------------------------------------------------- #
# Tee: mirror stdout/stderr to a timestamped logfile (self-redirect)
# --------------------------------------------------------------------------- #
class _Tee:
    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


# --------------------------------------------------------------------------- #
# Respiratory FFT -> BPM (numpy; runs on CPU after .cpu().numpy())
# --------------------------------------------------------------------------- #
def fft_rr_bpm(sig: np.ndarray, fs: float,
               low_hz: float = PEDRR_LOW_HZ, high_hz: float = PEDRR_HIGH_HZ) -> float:
    """Dominant FFT frequency of ``sig`` within [low_hz, high_hz], in BPM.

    Mirrors pedrr_dataset._fft_peak_bpm (periodogram, zero-padded to next pow2) but pinned
    to the Pediatric-RR band -- the same recovery logic the other extractors' PostProcessor
    uses. ``sig`` is the predicted rPPG waveform (already a pulse-like signal; no cumsum).
    """
    sig = np.asarray(sig, dtype=np.float64).ravel()
    if sig.size < 4 or not np.isfinite(sig).all():
        return float("nan")
    sig = sig - sig.mean()
    nfft = int(2 ** int(np.ceil(np.log2(max(sig.size, 4)))) * 4)   # 4x zero-pad for resolution
    f, p = sps.periodogram(sig, fs=fs, nfft=nfft, detrend="constant")
    band = (f >= low_hz) & (f <= high_hz)
    if not band.any():
        return float("nan")
    return float(f[band][int(np.argmax(p[band]))] * 60.0)


# --------------------------------------------------------------------------- #
# Combined manifest (union of every fold's videos) for the PedRR reader.
# The reader enumerates all videos; per-fold SPLIT_SUBJECTS then selects the split.
# --------------------------------------------------------------------------- #
def build_combined_manifest(kfold_dir: Path, out_path: Path, rebuild: bool) -> str:
    """Union all kfold_learned fold entries into one PedRR manifest (dedup by video).

    ``fs`` is taken from the existing pedrr_combined_manifest.json when present (all 332
    PedRR videos are 25 fps) so we avoid re-opening every video; falls back to cv2.
    """
    if out_path.exists() and not rebuild:
        logger.info("Using cached combined manifest: %s", out_path)
        return str(out_path)

    # fs lookup from the sibling combined manifest, if it exists.
    fs_lookup: dict[str, int] = {}
    sib = ROOT / "data" / "pedrr_learned_cache" / "pedrr_combined_manifest.json"
    if sib.exists():
        for e in json.load(open(sib, encoding="utf-8")):
            fs_lookup[f"{e['subject_id']}_{e['video_id']}"] = int(e.get("fs", 0) or 0)

    seen: set[str] = set()
    combined: list[dict] = []
    for k in range(5):
        for split in ("train", "val", "test"):
            fp = kfold_dir / f"fold{k}" / f"{split}.json"
            for e in json.load(open(fp, encoding="utf-8")):
                key = f"{e['subject_id']}_{e['video_id']}"
                if key in seen:
                    continue
                seen.add(key)
                fs = fs_lookup.get(key, 0)
                if fs <= 0:
                    import cv2
                    cap = cv2.VideoCapture(e["full_path"])
                    fs = int(round(cap.get(cv2.CAP_PROP_FPS)))
                    cap.release()
                combined.append({
                    "subject_id": str(e["subject_id"]),
                    "video_id": str(e["video_id"]),
                    "full_path": e["full_path"],
                    "gt_rr_bpm": float(e["gt_rr_bpm"]),
                    "age_months": e.get("age_months"),
                    "fs": int(fs),
                })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(combined, open(out_path, "w"), indent=2)
    logger.info("Wrote combined manifest (%d unique videos): %s", len(combined), out_path)
    return str(out_path)


def cached_keys(cache_dir: str) -> frozenset:
    """'<subject>_<video>' keys already in the coarse2fine preprocess cache. Restricting to
    these means NO coarse2fine multiprocessing runs -> single-process, deadlock-free, schtasks-safe."""
    import glob
    return frozenset(os.path.basename(f).split("_input_")[0]
                     for f in glob.glob(os.path.join(cache_dir, "PEDRR", "*", "*_input_*.npy")))


def fold_subjects(kfold_dir: Path, k: int, split: str, cached=None) -> list[str]:
    entries = json.load(open(kfold_dir / f"fold{k}" / f"{split}.json", encoding="utf-8"))
    return [str(e["subject_id"]) for e in entries
            if cached is None or f"{e['subject_id']}_{e['video_id']}" in cached]


def fold_test_keys(kfold_dir: Path, k: int, cached=None) -> list[str]:
    entries = json.load(open(kfold_dir / f"fold{k}" / "test.json", encoding="utf-8"))
    return [f"{e['subject_id']}_{e['video_id']}" for e in entries
            if cached is None or f"{e['subject_id']}_{e['video_id']}" in cached]


# --------------------------------------------------------------------------- #
# air-400 config for one fold (mirrors train_learned_extractor_pedrr.build_config)
# --------------------------------------------------------------------------- #
def build_air_config(manifest: str, subjects_by_split: dict[str, list[str]],
                     cache_dir: str, output_dir: str, seed: int) -> dict:
    def data_entry(subjects: list[str]) -> dict:
        return {
            "DATASET": DATASET_NAME,
            "SPLIT_TYPE": "custom",
            "SPLIT_SUBJECTS": list(subjects),
            "SPLIT_RATIO": 0.0,
            "DATA_FORMAT": "NDCHW",
            "PREPROCESS": copy.deepcopy(BASE_PREPROCESS),
        }

    return {
        "NAME": "physformer_pedrr",
        "SEED": int(seed),
        "DATA_PATH": {
            DATASET_NAME: manifest,
            "CACHE_DIR": cache_dir,
            "OUTPUT_DIR": output_dir,
            "BODY_DETECTOR_PATH": str(ROOT / "data" / "yolov8m.pt"),
            "FACE_DETECTOR_PATH": str(ROOT / "data" / "yolov8n-face.pt"),
        },
        # MODEL is unused by the dataset reader but kept for parity/clarity.
        "MODEL": {"NAME": "PhysFormer"},
        "TRAIN": {"DATA": [data_entry(subjects_by_split["train"])]},
        "VALID": {"DATA": [data_entry(subjects_by_split["val"])]},
        "TEST": {"DATA": [data_entry(subjects_by_split["test"])]},
        "SYSTEM": {"NUM_WORKERS": 0, "PIN_MEMORY": bool(torch.cuda.is_available())},
    }


# --------------------------------------------------------------------------- #
# Adapter: air-400 (D=180, C=6, 96, 96) item  ->  PhysFormer (3, T, 128, 128)
# --------------------------------------------------------------------------- #
class PhysFormerAdapter(Dataset):
    """Wrap an air-400 ConcatDataset; convert each item to a PhysFormer clip + waveform."""

    def __init__(self, base: Dataset, input_kind: str, chunk_len: int, resize: int):
        self.base = base
        assert input_kind in ("frames", "flow")
        # frames = standardized RGB[3:6]; flow = [0:3]
        self.ch = slice(3, 6) if input_kind == "frames" else slice(0, 3)
        self.chunk_len = int(chunk_len)
        self.resize = int(resize)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        data, label, filename, chunk_id, _dataset, fs = self.base[i]
        # data: (D=180, C=6, 96, 96) float32 NDCHW
        x = torch.from_numpy(np.ascontiguousarray(data))[: self.chunk_len, self.ch]  # (T,3,96,96)
        # spatial bilinear resize 96 -> resize (per-frame; T is the batch dim here). CPU op,
        # so it stays deterministic even under torch.use_deterministic_algorithms.
        x = F.interpolate(x, size=(self.resize, self.resize),
                          mode="bilinear", align_corners=False)                        # (T,3,R,R)
        x = x.permute(1, 0, 2, 3).contiguous()                                          # (3,T,R,R)
        y = torch.from_numpy(np.ascontiguousarray(label)).float()[: self.chunk_len]     # (T,)
        return x, y, str(filename), str(chunk_id), int(fs)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, pin: bool):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=pin,
        drop_last=False, worker_init_fn=air_main.seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(chunk_len: int, resize: int, dropout: float, device: torch.device):
    grid = resize // (8 * PF_PATCH_SIZE)   # Stem downsamples spatial by 8, patch_embed by patch
    if grid != 4:
        raise ValueError(
            f"PhysFormer hard-codes a 4x4 patch grid; resize/(8*patch_size) = {grid} != 4. "
            f"Use resize == {32 * PF_PATCH_SIZE} (got resize={resize}, patch_size={PF_PATCH_SIZE})."
        )
    if chunk_len % PF_PATCH_SIZE != 0:
        raise ValueError(f"--chunk-len ({chunk_len}) must be a multiple of patch_size ({PF_PATCH_SIZE}).")
    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(chunk_len, resize, resize),
        patches=(PF_PATCH_SIZE,) * 3,
        dim=PF_DIM, ff_dim=PF_FF_DIM, num_heads=PF_NUM_HEADS, num_layers=PF_NUM_LAYERS,
        dropout_rate=dropout, theta=PF_THETA,
    )
    return model.to(device)


def normalize_waveform(x: torch.Tensor) -> torch.Tensor:
    """Per-sample zero-mean/unit-std along time, as PhysFormerTrainer does before NegPearson."""
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True) + 1e-6
    return (x - mu) / sd


# --------------------------------------------------------------------------- #
# Eval: per-video RR (mean over chunks) + optional MAE against GT
# --------------------------------------------------------------------------- #
@torch.no_grad()
def infer_rr(model, loader, device) -> dict[str, float]:
    """Return {filename '<subject>_<video>': mean predicted RR (BPM) over its chunks}."""
    model.eval()
    per_video: dict[str, list[float]] = defaultdict(list)
    for data, _label, filenames, _chunk_ids, fs in loader:
        data = data.float().to(device)
        rppg, _, _, _ = model(data, PF_GRA_SHARP)                # (B, T)
        rppg = rppg.detach().cpu().numpy()
        fs = fs.numpy() if torch.is_tensor(fs) else np.asarray(fs)
        for b in range(rppg.shape[0]):
            rr = fft_rr_bpm(rppg[b], float(fs[b]))
            if np.isfinite(rr):
                per_video[filenames[b]].append(rr)
    return {k: float(np.mean(v)) for k, v in per_video.items() if len(v) > 0}


def rr_mae(preds: dict[str, float], gt_by_key: dict[str, float]) -> tuple[float, int]:
    errs = [abs(preds[k] - gt_by_key[k]) for k in preds if k in gt_by_key]
    return (float(np.mean(errs)) if errs else float("nan")), len(errs)


# --------------------------------------------------------------------------- #
# Train one fold
# --------------------------------------------------------------------------- #
def train_one_fold(k: int, args, combined_manifest: str, kfold_dir: Path,
                   cache_dir: str, output_root: Path, device: torch.device, cached=None) -> dict[str, float]:
    subjects_by_split = {s: fold_subjects(kfold_dir, k, s, cached) for s in ("train", "val", "test")}
    logger.info("fold%d subjects -> train=%d val=%d test=%d", k,
                len(subjects_by_split["train"]), len(subjects_by_split["val"]),
                len(subjects_by_split["test"]))

    fold_out = output_root / f"fold{k}"
    fold_out.mkdir(parents=True, exist_ok=True)

    config = build_air_config(combined_manifest, subjects_by_split, cache_dir,
                              str(fold_out), args.seed)
    air_main.listify_datasets(config)
    # Builds/loads the shared cache (any of the 34 uncached videos are preprocessed here on
    # the GPU box via the standard air-400 YOLO+coarse2fine path -- identical to siblings).
    datasets = air_main.get_split_datasets(config, {DATASET_NAME: PedRRDataset})

    train_ds = PhysFormerAdapter(datasets["TRAIN"], args.input, args.chunk_len, args.resize)
    val_ds = (PhysFormerAdapter(datasets["VALID"], args.input, args.chunk_len, args.resize)
              if datasets["VALID"] is not None else None)
    test_ds = (PhysFormerAdapter(datasets["TEST"], args.input, args.chunk_len, args.resize)
               if datasets["TEST"] is not None else None)

    pin = False  # host RAM is tight (shared w/ bdsl_graph+Chrome); pinned memory is non-pageable and worsens thrashing
    train_loader = make_loader(train_ds, args.batch, True, args.seed, pin)
    val_loader = make_loader(val_ds, args.infer_batch, False, args.seed, pin) if val_ds else None
    test_loader = make_loader(test_ds, args.infer_batch, False, args.seed, pin) if test_ds else None
    logger.info("fold%d loaders: train_chunks=%d val_chunks=%s test_chunks=%s", k,
                len(train_ds), len(val_ds) if val_ds else None, len(test_ds) if test_ds else None)

    # GT lookup (RR by '<subject>_<video>') for validation model-selection / logging.
    gt_by_key: dict[str, float] = {}
    for split in ("val", "test"):
        for e in json.load(open(kfold_dir / f"fold{k}" / f"{split}.json", encoding="utf-8")):
            gt_by_key[f"{e['subject_id']}_{e['video_id']}"] = float(e["gt_rr_bpm"])

    model = build_model(args.chunk_len, args.resize, args.dropout, device)
    if args.init_checkpoint:
        sd = torch.load(args.init_checkpoint, map_location="cpu")
        sd = {(kk[7:] if kk.startswith("module.") else kk): vv for kk, vv in sd.items()}
        miss, unexp = model.load_state_dict(sd, strict=False)
        logger.info("fold%d warm-start %s (missing=%d unexpected=%d)", k,
                    args.init_checkpoint, len(miss), len(unexp))

    criterion_pearson = Neg_Pearson()
    criterion_freq = PSDMSE(high_pass=PEDRR_HIGH_HZ, low_pass=PEDRR_LOW_HZ)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 2), gamma=0.5)

    best_val_mae, best_state, best_epoch = float("inf"), None, -1

    for epoch in range(args.epochs):
        model.train()
        ep_pear, ep_freq = [], []
        for data, label, _fn, _ci, fs in train_loader:
            data = data.float().to(device)
            label = label.float().to(device)                     # (B, T) DiffNormalized cosine
            optimizer.zero_grad()
            rppg, _, _, _ = model(data, PF_GRA_SHARP)            # (B, T)
            l_pear = criterion_pearson(normalize_waveform(rppg), label)
            # PSDMSE per-sample so each clip uses its own fs (all 25 here, but robust).
            l_freq = rppg.new_zeros(())
            B = rppg.shape[0]
            for b in range(B):
                l_freq = l_freq + criterion_freq(rppg[b:b + 1], label[b:b + 1], int(fs[b]))
            l_freq = l_freq / max(B, 1)
            loss = args.pearson_weight * l_pear + args.freq_weight * l_freq
            loss.backward()
            optimizer.step()
            ep_pear.append(float(l_pear.detach()))
            ep_freq.append(float(l_freq.detach()))
        scheduler.step()

        # Model selection: validation RR MAE in the RESPIRATORY band (not PhysFormer's HR band).
        val_mae, val_n = float("nan"), 0
        if val_loader is not None:
            val_preds = infer_rr(model, val_loader, device)
            val_mae, val_n = rr_mae(val_preds, gt_by_key)
        logger.info("fold%d epoch %02d/%d  NegPearson=%.4f  PSDMSE=%.5f  val_RR_MAE=%.3f (n=%d)",
                    k, epoch, args.epochs, float(np.mean(ep_pear)), float(np.mean(ep_freq)),
                    val_mae, val_n)

        take = (val_loader is None) or (np.isfinite(val_mae) and val_mae < best_val_mae)
        if take:
            best_val_mae = val_mae if np.isfinite(val_mae) else best_val_mae
            best_epoch = epoch
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("fold%d best epoch=%d val_RR_MAE=%.3f", k, best_epoch, best_val_mae)
    torch.save(model.state_dict(), fold_out / f"physformer_fold{k}.pt")

    # Test inference.
    if test_loader is None:
        logger.warning("fold%d has no test loader; no predictions produced.", k)
        return {}
    test_preds = infer_rr(model, test_loader, device)
    t_mae, t_n = rr_mae(test_preds, gt_by_key)
    logger.info("fold%d TEST RR MAE=%.3f over n=%d videos (%d predicted)",
                k, t_mae, t_n, len(test_preds))
    json.dump({"fold": k, "best_epoch": best_epoch, "val_rr_mae": best_val_mae,
               "test_rr_mae": t_mae, "n_test_pred": len(test_preds)},
              open(fold_out / "physformer_metrics.json", "w"), indent=2)
    return test_preds


# --------------------------------------------------------------------------- #
# Incremental JSON write (atomic)
# --------------------------------------------------------------------------- #
def write_preds(preds: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    json.dump({k: float(v) for k, v in preds.items()}, open(tmp, "w"), indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=15,
                    help="Full-budget epochs per fold (siblings use 15; PhysFormer canonical is "
                         "often higher, e.g. 20-30).")
    ap.add_argument("--chunk-len", type=int, default=CACHE_CHUNK_LEN,
                    help=f"Model temporal length T (<= cached {CACHE_CHUNK_LEN}, multiple of "
                         f"{PF_PATCH_SIZE}). Default uses the whole cached clip.")
    ap.add_argument("--resize", type=int, default=32 * PF_PATCH_SIZE,
                    help="Spatial size fed to PhysFormer (must be 32*patch_size=128; the model "
                         "hard-codes a 4x4 patch grid).")
    ap.add_argument("--input", choices=["frames", "flow"], default="frames",
                    help="Which 3 cache channels: 'frames'=standardized RGB[3:6] (PhysFormer-"
                         "native, default); 'flow'=optical-flow[0:3] (same input as EfficientPhys "
                         "-> pure-backbone comparison).")
    ap.add_argument("--batch", type=int, default=2, help="Train batch size (memory-tight box).")
    ap.add_argument("--infer-batch", type=int, default=2, help="Val/test batch size.")
    ap.add_argument("--lr", type=float, default=1e-4, help="Adam LR (PhysFormer canonical 1e-4).")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pearson-weight", type=float, default=1.0)
    ap.add_argument("--freq-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=100, help="Match sibling extractor convention.")
    ap.add_argument("--fold", type=int, default=None, choices=[0, 1, 2, 3, 4],
                    help="Run a single fold (default: all 5).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip a fold whose test keys are already all present in the preds JSON.")
    ap.add_argument("--cached-only", action="store_true",
                    help="Only use subjects already in the coarse2fine cache (drops the few uncached "
                         "clips) -> NO coarse2fine multiprocessing -> deadlock-free + Task-Scheduler-safe.")
    ap.add_argument("--init-checkpoint", type=str, default="",
                    help="Optional .pt to warm-start the model before training a fold (default: none).")
    ap.add_argument("--preds-out", type=str, default=str(ROOT / "outputs" / "physformer_preds.json"))
    ap.add_argument("--cache-dir", type=str, default=str(ROOT / "data" / "pedrr_learned_cache" / "full"),
                    help="Reuse the existing 89 GB body_roi coarse2fine cache (hash-matched).")
    ap.add_argument("--output-root", type=str,
                    default=str(ROOT / "outputs" / "paper_runs" / "pedrr_kfold_physformer"))
    ap.add_argument("--rebuild-manifest", action="store_true")
    ap.add_argument("--strict-deterministic", action="store_true",
                    help="Use air-400 set_random_seeds (torch.use_deterministic_algorithms=True). "
                         "May error on a 3D op lacking a deterministic CUDA kernel; off by default.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_len > CACHE_CHUNK_LEN:
        logger.warning("Clamping --chunk-len %d -> cached %d (larger would need re-preprocessing).",
                       args.chunk_len, CACHE_CHUNK_LEN)
        args.chunk_len = CACHE_CHUNK_LEN

    kfold_dir = ROOT / "data" / "organized" / "kfold_learned"
    output_root = Path(args.output_root)
    preds_path = Path(args.preds_out)

    logs_dir = ROOT / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_fh = open(logs_dir / f"train_physformer_pedrr_{ts}.log", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="[%(levelname)s] %(name)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=== PhysFormer / Pediatric-RR k-fold ===")
    logger.info("device=%s  input=%s  chunk_len=%d  resize=%d  epochs=%d  batch=%d  lr=%g",
                device, args.input, args.chunk_len, args.resize, args.epochs, args.batch, args.lr)
    logger.info("cache_dir=%s  preds_out=%s", args.cache_dir, preds_path)

    # Reproducibility.
    if args.strict_deterministic:
        air_main.set_random_seeds(args.seed)
    else:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    combined_manifest = build_combined_manifest(
        kfold_dir, ROOT / "data" / "pedrr_learned_cache" / "physformer_kfold_manifest.json",
        args.rebuild_manifest)

    # Resume: load any predictions already on disk.
    preds: dict[str, float] = {}
    if preds_path.exists():
        try:
            preds = {str(kk): float(vv) for kk, vv in json.load(open(preds_path)).items()}
            logger.info("Loaded %d existing predictions from %s", len(preds), preds_path)
        except Exception:
            logger.exception("Could not parse existing %s; starting fresh.", preds_path)

    cached = cached_keys(args.cache_dir) if args.cached_only else None
    if cached is not None:
        logger.info("cached-only: %d clips in cache; uncached clips are skipped (no coarse2fine).", len(cached))
    folds = [args.fold] if args.fold is not None else [0, 1, 2, 3, 4]
    for k in folds:
        tkeys = fold_test_keys(kfold_dir, k, cached)
        if args.skip_existing and all(kk in preds and np.isfinite(preds[kk]) for kk in tkeys):
            logger.info("fold%d: all %d test predictions present -> skipping.", k, len(tkeys))
            continue
        logger.info("---- fold %d ----", k)
        t0 = datetime.now()
        try:
            fold_preds = train_one_fold(k, args, combined_manifest, kfold_dir,
                                        args.cache_dir, output_root, device, cached)
        except Exception:
            logger.exception("fold%d FAILED; continuing to next fold.", k)
            continue
        preds.update(fold_preds)
        write_preds(preds, preds_path)      # incremental: a later crash keeps completed folds
        logger.info("fold%d done in %s; wrote %d total preds -> %s",
                    k, datetime.now() - t0, len(preds), preds_path)

    write_preds(preds, preds_path)
    logger.info("ALL DONE: %d predictions -> %s", len(preds), preds_path)
    logger.info("Next: python scripts/physformer_to_csv.py --preds %s", preds_path)
    log_fh.flush()


if __name__ == "__main__":
    main()
