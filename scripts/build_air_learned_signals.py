"""#23 cross-dataset: build AIR learned-flow PRISM-contract signals.

Runs the pedrr-trained EfficientPhys (learned extractor) on each AIR-125/AIR-400 clip via the
air-400 InferenceDataset (coarse2fine + YOLO body-ROI), captures the POST-PROCESSED waveform
(pred_seq), resamples it to the v3 signal length, and writes:

    data/air_signals_learned/AIR-{125|400}_{subj}_{vid}.npz
      = LEARNED flow  +  v3 depth/audio/gt/age (copied)  +  'fps' key (native 10/15 fps)

The 'fps' key is the crux of #23: PRISM's load-time resampler normalizes 10/15 fps -> canonical
25 fps, undoing the "AIR catastrophe" (RR inflated by 25/fps). Post-process band = pedrr TRAINING
band 0.20-1.70 Hz so the AIR flow is distributionally comparable to the pedrr-learned flow the
model was trained on.

Reuses the proven per-clip machinery from scratchpad/run_air_eval_learned.py. Resume-safe.
Caps coarse2fine flow workers (ID-1322 deadlock guard). Run once; not concurrently with another flow job.
"""
import os, sys, json, argparse
from pathlib import Path

REPO = r"G:\Respiratory Project"
AIR = os.path.join(REPO, "external", "air-400")
sys.path.insert(0, AIR)
os.chdir(AIR)

os.environ.setdefault("PEDRR_FLOW_WORKERS", "8")
import numpy as np
import torch
from torch.utils.data import DataLoader
import processors.pre_processor as _pp
_pp.cpu_count = lambda: int(os.environ.get("PEDRR_FLOW_WORKERS", "8"))  # cap BEFORE any preprocess

from dataloaders.inference_dataset import InferenceDataset
import infer as air_infer
from infer import build_model, run_inference
from main import set_random_seeds
from processors.post_processor import PostProcessor

CKPT = os.path.join(REPO, "outputs", "paper_runs", "pedrr_learned", "full", "model",
                    "EfficientPhys", "efficientphys_c2f_body_pedrr_full", "EfficientPhys_best.pth")
V3_DIR = os.path.join(REPO, "data", "air_signals_v3")
OUT_DIR = os.path.join(REPO, "data", "air_signals_learned")
MANIFESTS = [os.path.join(REPO, "data", "organized", "respiratory_benchmark_lodo_air-125_test.json"),
             os.path.join(REPO, "data", "organized", "respiratory_benchmark_lodo_air-400_test.json")]
LOW_HZ, HIGH_HZ = 0.20, 1.70   # pedrr TRAINING band (matches build_learned_flow_signals)


class FixedBandPP(PostProcessor):
    def post_process(self, sig, fs=30, diff_flag=True, infant_flag=None, use_bandpass=True, eval_method="FFT"):
        p = sig.detach().cpu().numpy()
        p = self._detrend_signal(np.cumsum(p) if diff_flag else p, 100)
        if use_bandpass:
            p = self._bandpass_filter(p, fs, LOW_HZ, HIGH_HZ)
        rr = self._calculate_fft_rr(p, fs, LOW_HZ, HIGH_HZ) if eval_method == "FFT" else self._calculate_peak_rr(p, fs)
        return p, rr


air_infer.PostProcessor = FixedBandPP

PRE = {
    'DO_DOWNSAMPLE_BEFORE_PREPROCESS': True, 'DOWNSAMPLE_SIZE_BEFORE_PREPROCESS': [320, 180],
    'DATA_NORMALIZE_TYPE': 'Standardized', 'LABEL_NORMALIZE_TYPE': 'DiffNormalized',
    'FLOW_NORMALIZE_TYPE': 'Standardized', 'DO_CHUNK': True, 'CHUNK_LENGTH': 180,
    'DO_CROP_INFANT_REGION': True, 'DYNAMIC_DETECTION': True, 'DYNAMIC_DETECTION_FREQUENCY': 10,
    'INFANT_REGION': 'Body', 'BODY_CONF_THRESHOLD': 0.25, 'FACE_CONF_THRESHOLD': 0.5,
    'CHEST_SHIFT_ALPHA': 0.2, 'DO_OPTICAL_FLOW': True, 'OF_METHOD': 'coarse2fine',
    'DO_DOWNSAMPLE_BEFORE_TRAINING': True, 'DOWNSAMPLE_SIZE_BEFORE_TRAINING': [96, 96],
}


def make_config(video_file):
    return {
        'NAME': 'air_learned', 'MODEL': {'NAME': 'EfficientPhys', 'IN_CHANNELS': 3,
        'FRAME_DEPTH': 10, 'DROP_RATE': 0.2, 'LOSS': 'psd_mse'},
        'DATA_PATH': {'BODY_DETECTOR_PATH': os.path.join(REPO, 'data', 'yolov8m.pt'),
        'FACE_DETECTOR_PATH': os.path.join(REPO, 'data', 'yolov8n-face.pt'),
        'OUTPUT_DIR': os.path.join(REPO, 'outputs', 'paper_runs', 'air_learned'), 'VIDEO_FILE': video_file},
        'TEST': {'DATA': {'INFANT_FLAG': True, 'DATASET': 'AIR', 'DATA_FORMAT': 'NDCHW', 'PREPROCESS': PRE}},
        'INFERENCE': {'BATCH_SIZE': 8, 'EVALUATION_METHOD': 'FFT', 'USE_POST_PROCESS': True},
        'LOGGING': {'USE_WANDB': False},
    }


def load_ckpt(model, path):
    sd = torch.load(path, map_location='cpu')
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model


def main():
    import logging
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)     # 0 = all
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("air_build"); logger.setLevel(logging.ERROR)
    os.makedirs(OUT_DIR, exist_ok=True)

    clips = []
    for mf in MANIFESTS:
        for e in json.load(open(mf)):
            p = e.get("full_path")
            if p and os.path.exists(p):
                clips.append({"dataset": e["dataset"], "subj": e["subject_id"],
                              "vid": e["video_id"], "gt": float(e.get("gt_rr_bpm") or 0.0), "path": p})
    if args.limit > 0:
        clips = clips[:args.limit]
    print(f"[air_build] {len(clips)} clips  band={LOW_HZ}-{HIGH_HZ}Hz  workers={_pp.cpu_count()}", flush=True)

    model = load_ckpt(build_model(make_config(clips[0]["path"]), logger), CKPT)
    print("[air_build] pedrr EfficientPhys loaded", flush=True)

    saved = miss = skip = 0
    errs = []
    for i, c in enumerate(clips):
        name = f"{c['dataset']}_{c['subj']}_{c['vid']}.npz"
        out_npz = os.path.join(OUT_DIR, name)
        v3_npz = os.path.join(V3_DIR, name)
        if os.path.exists(out_npz) and not args.overwrite:
            skip += 1; continue
        if not os.path.exists(v3_npz):
            miss += 1; print(f"  [miss v3] {name}", flush=True); continue
        try:
            set_random_seeds(100)
            ds = InferenceDataset(make_config(c["path"]), c["path"])   # coarse2fine ONCE
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
            out = run_inference(make_config(c["path"]), model, loader, logger)
            wave = np.asarray(out["pred_seq"], dtype=np.float64).ravel()
            fps = float(ds.fs)
            rr = float(out["pred_rr_bpm"])
        except Exception as ex:
            errs.append(name); print(f"  ERR {name}: {type(ex).__name__}: {ex}", flush=True); continue

        z = dict(np.load(v3_npz, allow_pickle=True))
        n_v3 = int(np.asarray(z["depth"]).shape[0])
        if len(wave) != n_v3 and len(wave) > 1:   # reconcile length to keep depth/audio aligned
            wave = np.interp(np.linspace(0, 1, n_v3), np.linspace(0, 1, len(wave)), wave)
        z["flow"] = wave.astype(np.float32)
        z["fps"] = np.float32(fps)                 # <-- the crux: enables load-time fps normalization
        np.savez_compressed(out_npz, **z)
        saved += 1
        print(f"  {i+1}/{len(clips)} {name} fps={fps:.0f} extractor_rr={rr:.1f} gt={c['gt']:.1f}", flush=True)

    print(f"\nDONE: saved={saved} skipped={skip} miss_v3={miss} err={len(errs)} -> {OUT_DIR}", flush=True)
    if errs:
        print(f"ERRORS ({len(errs)}): {errs[:10]}", flush=True)


if __name__ == "__main__":
    main()
