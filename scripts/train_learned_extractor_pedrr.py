"""Train the learned infant-respiration extractor on Pediatric-RR.

Model : EfficientPhys (air-400) over 3-channel Coarse2Fine optical flow,
        with a YOLO *body* ROI crop -- the SOTA `body_roi` recipe.
Data  : Pediatric-RR in-distribution manifests (`data/organized/pedrr_indist_*.json`),
        with a synthesized constant-frequency cosine waveform label per video
        (see scripts/pedrr_learned/pedrr_dataset.py).

This driver lives in PROJECT space. It *reuses* the vendored air-400 machinery by
import (BaseDataset, EfficientPhys, BaseTrainer, PSDMSE, PreProcessor, PostProcessor,
and the split/loader helpers in `main`) and *monkeypatches at runtime* -- it never
edits anything under `external/`. Two runtime overrides adapt the AIR-tuned code to
Pediatric-RR's wider RR range (16-96.5 BPM):

  1. PSD-MSE loss band  -> trainer.criterion['PEDRR']         (air-400 default is 0.3-0.8 Hz)
  2. RR post-process band -> trainer.post_processor           (air-400 default is 0.3-1.0 Hz infant / 0.08-0.5 Hz adult)

Both are widened to PEDRR_LOW_HZ..PEDRR_HIGH_HZ so high-RR clips are supervised /
evaluated correctly.

Default invocation runs a SHORT smoke test. Pass --full for the real study run.
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

# Deterministic cuBLAS workspace must be set before torch is imported so that
# torch.use_deterministic_algorithms(True) (in air-400's set_random_seeds) works.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]                 # G:\Respiratory Project
AIR400_DIR = ROOT / "external" / "air-400"
PEDRR_PKG_DIR = ROOT / "scripts" / "pedrr_learned"
for _p in (str(AIR400_DIR), str(PEDRR_PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
from torch.utils.data import DataLoader                     # noqa: E402

import main as air_main                                     # noqa: E402  (air-400/main.py)
from models.efficient_phys import EfficientPhys             # noqa: E402
from trainers.base_trainer import BaseTrainer               # noqa: E402
from loss.psdmse import PSDMSE                              # noqa: E402
from processors.post_processor import PostProcessor         # noqa: E402
from pedrr_dataset import PedRRDataset                      # noqa: E402
import processors.pre_processor as _pre_processor           # noqa: E402

logger = logging.getLogger("pedrr_train")

# --------------------------------------------------------------------------- #
# Robustness: cap the coarse2fine optical-flow Pool worker count.
# air-400's PreProcessor spawns min(n_pairs, cpu_count()) workers PER VIDEO
# (16 on this box), each holding a whole clip's frames in RAM. On large (1080p)
# clips that transient spike can silently OOM-kill a worker, leaving pool.imap()
# to hang forever with no traceback -- observed: a 1080p val clip stalled the
# entire run for ~24h at flow "0/1509". Capping workers bounds peak RAM. The
# name `cpu_count` is bound *inside* pre_processor (it did
# `from multiprocessing import cpu_count`), so we rebind it there -- never
# editing anything under external/. Tune via PEDRR_FLOW_WORKERS (default 8).
# --------------------------------------------------------------------------- #
_FLOW_WORKERS = max(1, int(os.environ.get("PEDRR_FLOW_WORKERS", "8")))
_pre_processor.cpu_count = lambda: _FLOW_WORKERS
logger.info("Capped coarse2fine flow Pool workers to %d (PEDRR_FLOW_WORKERS)", _FLOW_WORKERS)

# --------------------------------------------------------------------------- #
# Pediatric-RR RR band.  Manifest RR spans 16.0 - 96.5 BPM == 0.267 - 1.608 Hz.
# We bracket it with headroom: 0.20 Hz (12 BPM) .. 1.70 Hz (102 BPM).
# NOTE: this is DELIBERATELY wider than air-400's infant cap (0.3-1.0 Hz / 18-60
# BPM); using the AIR band would mis-supervise and mis-read every clip above 60 BPM.
# --------------------------------------------------------------------------- #
PEDRR_LOW_HZ = 0.20
PEDRR_HIGH_HZ = 1.70

DATASET_NAME = "PEDRR"

# Base preprocess block = the SOTA body_roi coarse2fine config, verbatim except
# CHUNK_LENGTH / SMOKE_MAX_FRAMES which the driver injects per run.
BASE_PREPROCESS = {
    "DO_DOWNSAMPLE_BEFORE_PREPROCESS": True,
    "DOWNSAMPLE_SIZE_BEFORE_PREPROCESS": [320, 180],
    "DATA_NORMALIZE_TYPE": "Standardized",
    "LABEL_NORMALIZE_TYPE": "DiffNormalized",
    "FLOW_NORMALIZE_TYPE": "Standardized",
    "DO_CHUNK": True,
    "CHUNK_LENGTH": 180,
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

# Per-mode defaults; explicit CLI flags override these.
SMOKE_DEFAULTS = dict(epochs=1, batch=2, infer_batch=2, lr=1e-3, seed=100,
                      chunk_length=60, max_frames=120,
                      train_limit=2, val_limit=1, test_limit=1)
FULL_DEFAULTS = dict(epochs=15, batch=4, infer_batch=8, lr=1e-3, seed=100,
                     chunk_length=180, max_frames=0,
                     train_limit=0, val_limit=0, test_limit=0)


class PedRRPostProcessor(PostProcessor):
    """PostProcessor that forces the Pediatric-RR band (ignores infant_flag).

    Mirrors air-400 PostProcessor.post_process exactly, but pins the band to
    [low_hz, high_hz] instead of the AIR-tuned infant/adult presets so RR up to
    ~102 BPM is recovered. Reuses the parent's static filtering/FFT helpers.
    """

    def __init__(self, low_hz: float, high_hz: float) -> None:
        super().__init__()
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)

    def post_process(self, sig, fs=30, diff_flag=True, infant_flag=False,
                     use_bandpass=True, eval_method="FFT"):
        processed = sig.detach().cpu().numpy()
        if diff_flag:
            processed = self._detrend_signal(np.cumsum(processed), 100)
        else:
            processed = self._detrend_signal(processed, 100)

        low, high = self.low_hz, self.high_hz
        if use_bandpass:
            processed = self._bandpass_filter(processed, fs, low, high)

        if eval_method == "FFT":
            sig_rr = self._calculate_fft_rr(processed, fs, low, high)
        elif eval_method == "Peak":
            sig_rr = self._calculate_peak_rr(processed, fs)
        else:
            raise ValueError("Please use FFT or Peak method to calculate RR.")
        return processed, sig_rr


# --------------------------------------------------------------------------- #
# Manifest handling
# --------------------------------------------------------------------------- #
def build_combined_manifest(rebuild: bool) -> tuple[str, dict[str, list[str]], dict[str, float]]:
    """Union the 3 pedrr split manifests into one (with per-video cv2 fps).

    Returns (combined_path, split->subjects, subject->gt_rr_bpm).
    """
    import cv2

    out_dir = ROOT / "data" / "pedrr_learned_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "pedrr_combined_manifest.json"

    split_subjects: dict[str, list[str]] = {}
    subject_gt: dict[str, float] = {}
    raw_by_split: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        entries = json.load(open(ROOT / "data" / "organized" / f"pedrr_indist_{split}.json"))
        raw_by_split[split] = entries
        split_subjects[split] = [str(e["subject_id"]) for e in entries]
        for e in entries:
            subject_gt[str(e["subject_id"])] = float(e["gt_rr_bpm"])

    if combined_path.exists() and not rebuild:
        logger.info("Using cached combined manifest: %s", combined_path)
        return str(combined_path), split_subjects, subject_gt

    logger.info("Building combined manifest (reading cv2 fps for all clips)...")
    combined: list[dict] = []
    for split in ("train", "val", "test"):
        for e in raw_by_split[split]:
            cap = cv2.VideoCapture(e["full_path"])
            fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
            cap.release()
            combined.append({
                "subject_id": str(e["subject_id"]),
                "video_id": str(e["video_id"]),
                "full_path": e["full_path"],
                "gt_rr_bpm": float(e["gt_rr_bpm"]),
                "age_months": e.get("age_months"),
                "fs": fps,
                "split": split,
            })
    json.dump(combined, open(combined_path, "w"), indent=2)
    logger.info("Wrote combined manifest (%d clips): %s", len(combined), combined_path)
    return str(combined_path), split_subjects, subject_gt


def _limit(subjects: list[str], n: int) -> list[str]:
    """First `n` subjects (n<=0 -> all)."""
    return list(subjects) if n <= 0 else list(subjects)[:max(1, n)]


# --------------------------------------------------------------------------- #
# Config assembly
# --------------------------------------------------------------------------- #
def build_config(p: dict, combined_manifest: str, split_subjects: dict,
                 cache_dir: str, output_dir: str, name: str) -> dict:
    pre = copy.deepcopy(BASE_PREPROCESS)
    pre["CHUNK_LENGTH"] = int(p["chunk_length"])
    if int(p["max_frames"]) > 0:
        pre["SMOKE_MAX_FRAMES"] = int(p["max_frames"])  # honoured by PedRRDataset only

    def data_entry(subjects: list[str]) -> dict:
        return {
            "DATASET": DATASET_NAME,
            "SPLIT_TYPE": "custom",
            "SPLIT_SUBJECTS": list(subjects),
            "SPLIT_RATIO": 0.0,
            "DATA_FORMAT": "NDCHW",
            "PREPROCESS": copy.deepcopy(pre),
        }

    train_subj = _limit(split_subjects["train"], p["train_limit"])
    val_subj = _limit(split_subjects["val"], p["val_limit"])
    test_subj = _limit(split_subjects["test"], p["test_limit"])
    logger.info("Split sizes -> train=%d val=%d test=%d subjects",
                len(train_subj), len(val_subj), len(test_subj))

    return {
        "NAME": name,
        "SEED": int(p["seed"]),
        "DATA_PATH": {
            DATASET_NAME: combined_manifest,
            "CACHE_DIR": cache_dir,
            "OUTPUT_DIR": output_dir,
            "BODY_DETECTOR_PATH": str(ROOT / "data" / "yolov8m.pt"),
            "FACE_DETECTOR_PATH": str(ROOT / "data" / "yolov8n-face.pt"),
        },
        "MODEL": {"NAME": "EfficientPhys", "LOSS": "psd_mse",
                  "FRAME_DEPTH": 10, "IN_CHANNELS": 3, "DROP_RATE": 0.2},
        "TRAIN": {"BATCH_SIZE": int(p["batch"]), "EPOCHES": int(p["epochs"]),
                  "LR": float(p["lr"]), "DATA": [data_entry(train_subj)]},
        "VALID": {"DATA": [data_entry(val_subj)]},
        "TEST": {"DATA": [data_entry(test_subj)]},
        "INFERENCE": {"BATCH_SIZE": int(p["infer_batch"]),
                      "EVALUATION_METHOD": "FFT", "USE_POST_PROCESS": True},
        "LOGGING": {"USE_WANDB": False},
        "SYSTEM": {"NUM_WORKERS": 0, "PIN_MEMORY": bool(torch.cuda.is_available())},
    }


# --------------------------------------------------------------------------- #
# Model init / warm start
# --------------------------------------------------------------------------- #
def build_model(config: dict) -> EfficientPhys:
    frame_h = config["TRAIN"]["DATA"][0]["PREPROCESS"]["DOWNSAMPLE_SIZE_BEFORE_TRAINING"][1]
    return EfficientPhys(
        in_channels=config["MODEL"]["IN_CHANNELS"],
        frame_depth=config["MODEL"]["FRAME_DEPTH"],
        img_size=frame_h,
    )


def load_init_checkpoint(model: torch.nn.Module, path: str) -> None:
    sd = torch.load(path, map_location="cpu")
    if any(k.startswith("module.") for k in sd):
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info("Warm-started from %s (missing=%d, unexpected=%d)",
                path, len(missing), len(unexpected))


# --------------------------------------------------------------------------- #
# Loaders (mirrors air-400/main.py loader construction)
# --------------------------------------------------------------------------- #
def build_loaders(config: dict, datasets: dict, seed: int):
    num_workers = int(config["SYSTEM"]["NUM_WORKERS"])
    pin = bool(config["SYSTEM"]["PIN_MEMORY"])
    train_loader = DataLoader(
        datasets["TRAIN"], batch_size=config["TRAIN"]["BATCH_SIZE"], shuffle=True,
        num_workers=num_workers, pin_memory=pin,
        worker_init_fn=air_main.seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = None
    if datasets["VALID"] is not None:
        val_loader = DataLoader(
            datasets["VALID"], batch_size=config["INFERENCE"]["BATCH_SIZE"], shuffle=False,
            num_workers=num_workers, pin_memory=pin, worker_init_fn=air_main.seed_worker)
    test_loader = None
    if datasets["TEST"] is not None:
        test_loader = DataLoader(
            datasets["TEST"], batch_size=config["INFERENCE"]["BATCH_SIZE"], shuffle=False,
            num_workers=num_workers, pin_memory=pin, worker_init_fn=air_main.seed_worker)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# Per-subject RR report (surfaces predicted-RR vs GT that trainer.test() hides)
# --------------------------------------------------------------------------- #
def report_rr(trainer: BaseTrainer, loader: DataLoader, subject_gt: dict[str, float]) -> None:
    """Print predicted RR (model), label-derived RR, and manifest GT per subject."""
    if loader is None:
        return
    dev = trainer.device
    preds, labs = defaultdict(list), defaultdict(list)
    ds_map, fs_map = {}, {}

    trainer.model.eval()
    with torch.no_grad():
        for batch in loader:
            data, labels, subj_idx, _chunk, ds_names, fs_list = batch
            data, labels = data.to(dev), labels.to(dev)
            if trainer.model_name.lower() not in ["deepphys", "tscan"]:
                if trainer.do_optical_flow:
                    data = data[:, :, :trainer.in_channels, ...]
                else:
                    data = data[:, :, trainer.in_channels:, ...]
            N, D, C, H, W = data.shape
            data = data.view(N * D, C, H, W)
            labels = labels.view(-1, 1)
            valid = (data.shape[0] // trainer.base_len) * trainer.base_len
            data, labels = data[:valid], labels[:valid]
            if trainer.model_name.lower().startswith("efficientphys"):
                last = torch.unsqueeze(data[-1], 0).repeat(trainer.num_of_gpu, 1, 1, 1)
                data = torch.cat((data, last), 0)
            pred = trainer.model(data)
            sp, sl, dm, fm = trainer._aggregate_batch_by_subject(
                pred, labels, subj_idx, ds_names, fs_list)
            for s in sp:
                preds[s].extend(sp[s]); labs[s].extend(sl[s]); ds_map[s] = dm[s]; fs_map[s] = fm[s]

    print("\n=== Per-subject RR (predicted vs GT) ===")
    print(f"{'subject':<26}{'fs':>4}{'pred_RR':>10}{'label_RR':>10}{'gt_RR':>8}{'abs_err':>9}")
    for s in preds:
        pred_subj = torch.cat(preds[s], 0)
        lab_subj = torch.cat(labs[s], 0)
        fs = fs_map[s]
        _, pred_rr = trainer.post_processor.post_process(
            pred_subj, fs=fs, diff_flag=trainer.diff_flag_test, use_bandpass=True,
            eval_method=trainer.eval_method)
        _, lab_rr = trainer.post_processor.post_process(
            lab_subj, fs=fs, diff_flag=trainer.diff_flag_test, use_bandpass=True,
            eval_method=trainer.eval_method)
        # subject filename == "{subject_id}_{video_id}"; recover subject_id prefix
        subj_id = "-".join(s.split("_")[0:1]) if "ID-" in s else s.rsplit("_", 1)[0]
        gt = subject_gt.get(s.rsplit("_", 1)[0], subject_gt.get(subj_id, float("nan")))
        print(f"{s:<26}{fs:>4}{pred_rr:>10.2f}{lab_rr:>10.2f}{gt:>8.1f}{abs(pred_rr - gt):>9.2f}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="Full study run (all subjects, 15 epochs). Default is a short smoke test.")
    # training params (None -> take from mode defaults)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--infer-batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--chunk-length", type=int, default=None, help="Must be a multiple of FRAME_DEPTH (10).")
    ap.add_argument("--max-frames", type=int, default=None, help="Cap decoded frames/clip (smoke). 0=full video.")
    ap.add_argument("--train-limit", type=int, default=None, help="# train subjects (0=all).")
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)
    # misc
    ap.add_argument("--init-checkpoint", type=str, default=None,
                    help="Optional EfficientPhys .pth to warm-start from "
                         "(e.g. data/models/EfficientPhys_Coarse2fine_Body_best.pth).")
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--rebuild-manifest", action="store_true")
    ap.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    return ap.parse_args()


def resolve_params(args: argparse.Namespace) -> dict:
    base = FULL_DEFAULTS if args.full else SMOKE_DEFAULTS
    p = dict(base)
    for k in ("epochs", "batch", "infer_batch", "lr", "seed", "chunk_length",
              "max_frames", "train_limit", "val_limit", "test_limit"):
        v = getattr(args, k)
        if v is not None:
            p[k] = v
    if p["chunk_length"] % 10 != 0:
        raise ValueError(f"CHUNK_LENGTH ({p['chunk_length']}) must be a multiple of FRAME_DEPTH (10)")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    p = resolve_params(args)
    mode = "FULL" if args.full else "SMOKE"

    cache_dir = args.cache_dir or str(
        ROOT / "data" / "pedrr_learned_cache" / ("full" if args.full else "smoke"))
    output_dir = args.output_dir or str(
        ROOT / "outputs" / "paper_runs" / "pedrr_learned" / ("full" if args.full else "smoke"))
    name = args.name or f"efficientphys_c2f_body_pedrr_{mode.lower()}"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=== PedRR learned extractor: %s mode ===", mode)
    logger.info("params: %s", p)
    logger.info("cache_dir=%s", cache_dir)
    logger.info("output_dir=%s", output_dir)

    # Reproducibility (air-400's seeding; use_deterministic_algorithms needs the
    # CUBLAS env set at top). --no-deterministic relaxes it if an op lacks a
    # deterministic kernel.
    if args.deterministic:
        air_main.set_random_seeds(int(p["seed"]))
    else:
        np.random.seed(int(p["seed"])); torch.manual_seed(int(p["seed"]))

    combined_manifest, split_subjects, subject_gt = build_combined_manifest(args.rebuild_manifest)
    config = build_config(p, combined_manifest, split_subjects, cache_dir, output_dir, name)
    air_main.listify_datasets(config)

    # Build datasets (this triggers YOLO ROI + coarse2fine preprocessing & caching)
    # via air-400's own split/concat logic, injecting PedRRDataset as the reader.
    datasets = air_main.get_split_datasets(config, {DATASET_NAME: PedRRDataset})
    train_loader, val_loader, test_loader = build_loaders(config, datasets, int(p["seed"]))
    logger.info("Loaders: train_batches=%d val=%s test=%s",
                len(train_loader),
                len(val_loader) if val_loader else None,
                len(test_loader) if test_loader else None)

    model = build_model(config)
    if args.init_checkpoint:
        load_init_checkpoint(model, args.init_checkpoint)

    trainer = BaseTrainer(model=model, config=config,
                          train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

    # --- runtime overrides (do NOT touch external/) --------------------------
    # 1. Widen the PSD-MSE supervision band to cover pedrr RR (default is 0.08-0.5 Hz here).
    trainer.criterion[DATASET_NAME] = PSDMSE(high_pass=PEDRR_HIGH_HZ, low_pass=PEDRR_LOW_HZ)
    # 2. Widen the RR post-process band (default infant_flag=False -> 0.08-0.5 Hz).
    trainer.post_processor = PedRRPostProcessor(PEDRR_LOW_HZ, PEDRR_HIGH_HZ)
    logger.info("RR band set to %.2f-%.2f Hz (%.0f-%.0f BPM) for loss + post-process",
                PEDRR_LOW_HZ, PEDRR_HIGH_HZ, PEDRR_LOW_HZ * 60, PEDRR_HIGH_HZ * 60)

    t0 = datetime.now()
    trainer.train()
    logger.info("Training wall-clock: %s", datetime.now() - t0)

    metrics = trainer.test()
    print("\n=== trainer.test() metrics (test_mae is RR MAE in BPM) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    report_rr(trainer, test_loader, subject_gt)
    logger.info("Done (%s mode).", mode)


if __name__ == "__main__":
    main()
