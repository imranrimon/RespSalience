# RUN: conda run -n resp_env python scripts/physics_vs_algorithm.py --signal-dir data/pedrr_signals_learned --checkpoint outputs/paper_runs/pedrr_kfold_learned/fold{fold}/train/prism_breath_best.pt
#      CPU-only fast pass (raw + flow stages, no model):
#      conda run -n resp_env python scripts/physics_vs_algorithm.py --signal-dir data/pedrr_signals_learned --stages raw,flow
"""
Physics-vs-algorithm saturation probe  (reviewer #3).

HYPOTHESIS
----------
The neonatal / high-RR failure of camera respiratory-rate estimation is an
ALGORITHMIC ceiling, not a PHYSICAL one. If the respiratory oscillation at the
true frequency f_gt = RR/60 Hz is still present in the *raw* ROI mean-intensity
signal even for high-RR clips, but its spectral energy is progressively lost as
the signal passes through the pipeline (optical-flow waveform -> learned encoder
-> scalar RR head, whose output plateaus ~26-30 bpm), then the high-rate
information is DESTROYED by the extractor, not physically absent. If instead the
raw-ROI spectral energy at f_gt itself collapses at high RR, the limit is
physical (camera temporal resolution / motion aliasing).

WHAT IT MEASURES (per clip, stratified by GT RR band)
-----------------------------------------------------
At each pipeline STAGE, spectral energy concentrated at f_gt, quantified two ways:
  * snr_at_fgt : band power in [f_gt +/- tol] / total power in [0.1, 2.0] Hz
  * hit        : does the dominant in-band spectral peak fall within tol of f_gt
  * peak_bpm   : dominant in-band frequency (bpm)

Stages:
  raw     raw ROI mean-intensity signal (center crop, per-frame gray mean)  [CPU]
  flow    optical-flow waveform  (the `flow` channel of the extracted .npz)  [CPU]
  encoder frozen PRISM flow-branch temporal features (max/mean over channels) [GPU-pref]
  rr      scalar RR head output: pred_rr, recovery = pred_rr / gt_rr          [GPU-pref]
          (a scalar has no spectrum; "energy at f_gt" degenerates to whether
           the predicted rate lands at the true rate -> the recovery ratio.)

The raw + flow stages run CPU-only. The encoder + rr stages need the frozen model
(GPU strongly preferred; they are auto-skipped if --checkpoint is unavailable or
absent from --stages).

RESUME / EXECUTION
------------------
Single-process. Raw signals are cached under outputs/physics_saturation/raw_signals/
and per-clip rows are appended incrementally, so a re-run resumes where it stopped.
Raw-video reading is the expensive part; use --max-per-band / --limit for a fast
stratified subset. THIS BOX CANNOT run the model stages (WinError 5) -- launch on
the RTX 8000, or use --stages raw,flow for the CPU-only physical-presence probe.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from scipy import signal as sps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KFOLD = ROOT / "data" / "organized" / "kfold_learned"
CANONICAL_FS = 25.0


# --------------------------------------------------------------------------- #
# Spectral energy at f_gt
# --------------------------------------------------------------------------- #
def spectral_metrics(sig: np.ndarray, fs: float, f_gt: float,
                     tol_hz: float, band: Tuple[float, float] = (0.1, 2.0)) -> Dict[str, float]:
    x = np.asarray(sig, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 8 or fs <= 0:
        return {"snr_at_fgt": float("nan"), "hit": float("nan"), "peak_bpm": float("nan")}
    x = sps.detrend(x, type="linear")
    w = np.hanning(x.size)
    nfft = int(2 ** int(np.ceil(np.log2(max(x.size, 8)))) * 4)  # 4x zero-pad
    f, p = sps.periodogram(x * w, fs=fs, nfft=nfft, detrend=False)
    in_band = (f >= band[0]) & (f <= band[1])
    if not in_band.any() or p[in_band].sum() <= 0:
        return {"snr_at_fgt": float("nan"), "hit": float("nan"), "peak_bpm": float("nan")}
    near = in_band & (np.abs(f - f_gt) <= tol_hz)
    snr = float(p[near].sum() / p[in_band].sum()) if near.any() else 0.0
    fb = f[in_band]
    pb = p[in_band]
    peak_f = float(fb[int(np.argmax(pb))])
    hit = float(abs(peak_f - f_gt) <= tol_hz)
    return {"snr_at_fgt": snr, "hit": hit, "peak_bpm": peak_f * 60.0}


# --------------------------------------------------------------------------- #
# Raw ROI mean-intensity signal from the source video (CPU)
# --------------------------------------------------------------------------- #
def extract_raw_roi_signal(video_path: str, roi_frac: float, max_frames: int) -> Tuple[np.ndarray, float]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros(0, dtype=np.float32), 0.0
    fs = float(cap.get(cv2.CAP_PROP_FPS)) or CANONICAL_FS
    vals: List[float] = []
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        ch, cw = int(h * roi_frac), int(w * roi_frac)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        roi = frame[y0:y0 + ch, x0:x0 + cw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        vals.append(float(gray.mean()))
        n += 1
    cap.release()
    return np.asarray(vals, dtype=np.float32), fs


# --------------------------------------------------------------------------- #
# Frozen model (encoder + scalar RR stages)
# --------------------------------------------------------------------------- #
class ModelBank:
    """Lazily loads one frozen PRISM per fold + its processed-signal dataset."""

    def __init__(self, checkpoint_pattern: str, signal_dir: str):
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pattern = checkpoint_pattern
        self.per_fold = "{fold}" in checkpoint_pattern
        self.signal_dir = signal_dir
        self._models: Dict[int, Tuple] = {}
        self._ds: Dict[int, Tuple] = {}

    def _load(self, fold: int):
        import torch
        from src.core.prism_breath_model import build_prism_breath_model_from_config
        from src.core.respiratory_benchmark_dataset import RespiratoryBenchmarkDataset
        if fold in self._models:
            return self._models[fold], self._ds[fold]
        ckpt_path = Path(self.pattern.format(fold=fold)) if self.per_fold else Path(self.pattern)
        ckpt_path = ckpt_path if ckpt_path.is_absolute() else ROOT / ckpt_path
        if not ckpt_path.exists():
            self._models[fold] = None
            self._ds[fold] = None
            return None, None
        ckpt = torch.load(ckpt_path, map_location=self.device)
        config = ckpt.get("config", {})
        model_config = config.get("model_config", config)
        model = build_prism_breath_model_from_config(model_config).to(self.device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        seq_len = int(model_config.get("seq_len", config.get("seq_len", 300)))
        ds = RespiratoryBenchmarkDataset(signal_dir=self.signal_dir,
                                         manifest_path=str(KFOLD / f"fold{fold}" / "test.json"),
                                         seq_len=seq_len)
        idx = {f"{r['subject_id']}_{r['video_id']}": i for i, r in enumerate(ds.records)}
        self._models[fold] = (model, seq_len)
        self._ds[fold] = (ds, idx)
        return self._models[fold], self._ds[fold]

    def score(self, fold: int, clip_key: str, f_gt: float, tol_hz: float) -> Optional[Dict[str, float]]:
        import torch
        loaded, dsi = self._load(fold)
        if loaded is None or dsi is None:
            return None
        model, _ = loaded
        ds, idx = dsi
        if clip_key not in idx:
            return None
        item = ds[idx[clip_key]]
        depth = item["depth"].unsqueeze(0).to(self.device)
        flow = item["flow"].unsqueeze(0).to(self.device)
        audio = item["audio"].unsqueeze(0).to(self.device)
        age = item["age_months"].unsqueeze(0).to(self.device)
        mask = item["modality_mask"].unsqueeze(0).to(self.device)
        did = item["dataset_id"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = model(depth=depth, flow=flow, audio=audio, age_months=age,
                        modality_mask=mask, dataset_ids=did)
            pred_rr = float(out["rr_pred"][0].cpu())
            # encoder temporal features of the flow branch (the respiratory channel)
            seq = model.branches["flow"]._encode_sequence(flow)  # (1, T, D)
            seq_np = seq[0].cpu().numpy().T  # (D, T)
        fs = CANONICAL_FS
        snrs, hits, peaks = [], [], []
        for ch in seq_np:
            m = spectral_metrics(ch, fs, f_gt, tol_hz)
            if np.isfinite(m["snr_at_fgt"]):
                snrs.append(m["snr_at_fgt"]); hits.append(m["hit"]); peaks.append(m["peak_bpm"])
        if not snrs:
            enc = {"enc_snr_max": float("nan"), "enc_snr_mean": float("nan"), "enc_hit_any": float("nan")}
        else:
            best = int(np.argmax(snrs))
            enc = {"enc_snr_max": float(np.max(snrs)), "enc_snr_mean": float(np.mean(snrs)),
                   "enc_hit_any": float(max(hits)), "enc_peak_bpm_best": float(peaks[best])}
        enc["pred_rr"] = pred_rr
        return enc


# --------------------------------------------------------------------------- #
def band_of(rr: float, edges: List[float]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= rr < edges[i + 1]:
            return f"[{edges[i]:.0f},{edges[i+1]:.0f})"
    return f">= {edges[-1]:.0f}" if rr >= edges[-1] else f"< {edges[0]:.0f}"


def build_clip_index() -> Dict[str, Dict]:
    """Union of all fold TEST clips (each pedrr clip appears once) + its fold."""
    clips: Dict[str, Dict] = {}
    for k in range(5):
        for e in json.load(open(KFOLD / f"fold{k}" / "test.json", encoding="utf-8")):
            key = f"{e['subject_id']}_{e['video_id']}"
            if key not in clips:
                clips[key] = {
                    "clip_key": key,
                    "subject_id": str(e["subject_id"]),
                    "video_id": str(e["video_id"]),
                    "full_path": e.get("full_path", ""),
                    "gt_rr": float(e["gt_rr_bpm"]),
                    "age_months": e.get("age_months"),
                    "fold": k,
                }
    return clips


def signal_path_for(signal_dir: Path, clip: Dict) -> Path:
    return signal_dir / f"Pediatric-RR_{clip['subject_id']}_{clip['video_id']}.npz"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal-dir", default="data/pedrr_signals_learned")
    ap.add_argument("--checkpoint",
                    default="outputs/paper_runs/pedrr_kfold_learned/fold{fold}/train/prism_breath_best.pt",
                    help="Frozen model for encoder/rr stages. `{fold}` => each clip scored by its own fold's "
                         "model (leakage-safe). Omit encoder,rr from --stages to skip the model entirely.")
    ap.add_argument("--output-dir", default="outputs/physics_saturation")
    ap.add_argument("--stages", default="raw,flow,encoder,rr",
                    help="comma subset of raw,flow,encoder,rr (raw+flow are CPU-only).")
    ap.add_argument("--bands", default="16,25,35,45,60,97", help="GT-RR band edges (bpm)")
    ap.add_argument("--tol-bpm", type=float, default=2.0, help="+/- tolerance around f_gt (bpm)")
    ap.add_argument("--roi-frac", type=float, default=0.5, help="center-crop fraction for raw ROI")
    ap.add_argument("--max-frames", type=int, default=1500, help="cap frames read per video (~60s @25fps)")
    ap.add_argument("--max-per-band", type=int, default=0, help="cap clips per GT-RR band (0=all)")
    ap.add_argument("--limit", type=int, default=0, help="global cap on clips processed (0=all)")
    ap.add_argument("--overwrite", action="store_true", help="ignore existing per-clip rows / raw caches")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    edges = [float(x) for x in args.bands.split(",")]
    tol_hz = args.tol_bpm / 60.0
    signal_dir = ROOT / args.signal_dir if not os.path.isabs(args.signal_dir) else Path(args.signal_dir)
    out = ROOT / args.output_dir
    (out / "raw_signals").mkdir(parents=True, exist_ok=True)
    rows_csv = out / "per_clip_spectral.csv"

    want_model = ("encoder" in stages) or ("rr" in stages)
    bank = ModelBank(args.checkpoint, str(signal_dir)) if want_model else None
    if want_model:
        print(f"[model stages] device={bank.device} (GPU strongly preferred). "
              f"checkpoint pattern={args.checkpoint}")

    clips = build_clip_index()
    print(f"pedrr clips: {len(clips)}  stages={stages}  bands={edges}")

    # resume: existing rows
    done: set = set()
    existing_rows: List[Dict] = []
    if rows_csv.exists() and not args.overwrite:
        with open(rows_csv, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                existing_rows.append(r)
                done.add(r["clip_key"])
        print(f"resume: {len(done)} clips already scored")

    # stratified selection order + per-band caps
    order = sorted(clips.values(), key=lambda c: c["gt_rr"])
    per_band_count: Dict[str, int] = defaultdict(int)
    fieldnames = ["clip_key", "subject_id", "video_id", "fold", "gt_rr", "band", "f_gt_hz",
                  "raw_fs", "raw_snr", "raw_hit", "raw_peak_bpm",
                  "flow_snr", "flow_hit", "flow_peak_bpm",
                  "enc_snr_max", "enc_snr_mean", "enc_hit_any", "enc_peak_bpm_best",
                  "pred_rr", "pred_err", "recovery"]

    new_file = not rows_csv.exists() or args.overwrite
    fh = open(rows_csv, "w" if new_file else "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if new_file:
        writer.writeheader()

    processed = 0
    for clip in order:
        key = clip["clip_key"]
        b = band_of(clip["gt_rr"], edges)
        if args.max_per_band and per_band_count[b] >= args.max_per_band:
            continue
        if key in done:
            per_band_count[b] += 1
            continue
        if args.limit and processed >= args.limit:
            break

        f_gt = clip["gt_rr"] / 60.0
        row = {k: "" for k in fieldnames}
        row.update({"clip_key": key, "subject_id": clip["subject_id"], "video_id": clip["video_id"],
                    "fold": clip["fold"], "gt_rr": f"{clip['gt_rr']:.2f}", "band": b, "f_gt_hz": f"{f_gt:.4f}"})

        # ---- raw ROI stage (CPU) --------------------------------------- #
        if "raw" in stages:
            cache = out / "raw_signals" / f"{key}.npy"
            raw_fs = CANONICAL_FS
            if cache.exists() and not args.overwrite:
                raw = np.load(cache)
                meta = out / "raw_signals" / f"{key}.fs"
                raw_fs = float(meta.read_text()) if meta.exists() else CANONICAL_FS
            else:
                if clip["full_path"] and os.path.exists(clip["full_path"]):
                    raw, raw_fs = extract_raw_roi_signal(clip["full_path"], args.roi_frac, args.max_frames)
                    np.save(cache, raw.astype(np.float32))
                    (out / "raw_signals" / f"{key}.fs").write_text(f"{raw_fs}")
                else:
                    raw = np.zeros(0, dtype=np.float32)
            m = spectral_metrics(raw, raw_fs, f_gt, tol_hz)
            row.update({"raw_fs": f"{raw_fs:.2f}", "raw_snr": f"{m['snr_at_fgt']:.4f}",
                        "raw_hit": m["hit"], "raw_peak_bpm": f"{m['peak_bpm']:.2f}"})

        # ---- optical-flow waveform stage (CPU, from .npz) -------------- #
        if "flow" in stages:
            sp = signal_path_for(signal_dir, clip)
            if sp.exists():
                z = np.load(sp, allow_pickle=True)
                flow = np.asarray(z["flow"], dtype=np.float64) if "flow" in z.files else np.zeros(0)
                m = spectral_metrics(flow, CANONICAL_FS, f_gt, tol_hz)
                row.update({"flow_snr": f"{m['snr_at_fgt']:.4f}", "flow_hit": m["hit"],
                            "flow_peak_bpm": f"{m['peak_bpm']:.2f}"})

        # ---- encoder + scalar RR stages (GPU-preferred) --------------- #
        if bank is not None:
            enc = bank.score(clip["fold"], key, f_gt, tol_hz)
            if enc is not None:
                if "encoder" in stages:
                    row.update({"enc_snr_max": f"{enc.get('enc_snr_max', float('nan')):.4f}",
                                "enc_snr_mean": f"{enc.get('enc_snr_mean', float('nan')):.4f}",
                                "enc_hit_any": enc.get("enc_hit_any", ""),
                                "enc_peak_bpm_best": f"{enc.get('enc_peak_bpm_best', float('nan')):.2f}"})
                if "rr" in stages:
                    pr = enc.get("pred_rr", float("nan"))
                    row.update({"pred_rr": f"{pr:.2f}", "pred_err": f"{pr - clip['gt_rr']:.2f}",
                                "recovery": f"{pr / clip['gt_rr']:.3f}" if clip["gt_rr"] > 0 else ""})

        writer.writerow(row)
        fh.flush()
        existing_rows.append(row)
        per_band_count[b] += 1
        processed += 1
        if processed % 10 == 0:
            print(f"  ...{processed} new clips scored", flush=True)
    fh.close()
    print(f"per-clip rows -> {rows_csv}  (+{processed} new)")

    # ---- per-band aggregation ----------------------------------------- #
    def fnum(v):
        try:
            x = float(v)
            return x if np.isfinite(x) else np.nan
        except (TypeError, ValueError):
            return np.nan

    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in existing_rows:
        b = r.get("band", "")
        for col in ("raw_snr", "raw_hit", "flow_snr", "flow_hit", "enc_snr_max", "enc_hit_any",
                    "pred_rr", "recovery"):
            v = fnum(r.get(col, ""))
            if np.isfinite(v):
                agg[b][col].append(v)

    band_order = [f"[{edges[i]:.0f},{edges[i+1]:.0f})" for i in range(len(edges) - 1)]
    summary = {}
    print("\n=== SPECTRAL ENERGY AT f_gt BY GT-RR BAND (mean; hit=peak-in-band rate) ===")
    print("{:<12}{:>6}{:>10}{:>8}{:>10}{:>8}{:>10}{:>8}{:>10}{:>10}".format(
        "band", "n", "raw_snr", "raw_hit", "flow_snr", "flw_hit", "enc_snr", "enc_hit", "pred_rr", "recovery"))
    for b in band_order + [k for k in agg if k not in band_order]:
        if b not in agg:
            continue
        d = agg[b]
        n = max((len(d[c]) for c in d), default=0)
        rec = {c: (float(np.mean(d[c])) if d[c] else float("nan")) for c in
               ("raw_snr", "raw_hit", "flow_snr", "flow_hit", "enc_snr_max", "enc_hit_any", "pred_rr", "recovery")}
        rec["n"] = n
        summary[b] = rec
        print("{:<12}{:>6}{:>10.3f}{:>8.2f}{:>10.3f}{:>8.2f}{:>10.3f}{:>8.2f}{:>10.1f}{:>10.3f}".format(
            b, n, rec["raw_snr"], rec["raw_hit"], rec["flow_snr"], rec["flow_hit"],
            rec["enc_snr_max"], rec["enc_hit_any"], rec["pred_rr"], rec["recovery"]))

    json.dump(summary, open(out / "band_summary.json", "w"), indent=2)
    print(f"\nband summary -> {out / 'band_summary.json'}")
    print("\nInterpretation: raw_snr / raw_hit staying HIGH in the top RR bands while enc_snr and "
          "recovery COLLAPSE => algorithmic ceiling (info present, extractor destroys it). raw_snr "
          "itself collapsing at high RR => physical limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
