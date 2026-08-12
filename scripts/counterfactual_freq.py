# RUN: conda run -n resp_env python scripts/counterfactual_freq.py --signal-dir data/pedrr_signals_learned --checkpoint outputs/paper_runs/pedrr_kfold_learned/fold{fold}/train/prism_breath_best.pt
"""
Frequency counterfactuals & causal ablations  (reviewer #11).

HYPOTHESIS
----------
A trustworthy tachypnea model must be CAUSALLY driven by the respiratory
frequency. Two predictions follow:
  (1) Time-scaling the input respiratory waveform by a factor alpha (breathing
      alpha-times faster/slower) must move the predicted RR by ~alpha and move
      the predicted tachypnea probability MONOTONICALLY in the same direction
      relative to the age threshold -- clips just below threshold should cross
      it as alpha>1, clips just above should fall back as alpha<1.
  (2) Destroying the respiratory information should degrade the prediction:
      * phase-randomization keeps the POWER spectrum (hence RR) but destroys
        waveform morphology -> a frequency-driven model should stay ~stable
        (pred_rr and tachy_prob roughly preserved);
      * removing the dominant respiratory spectral peak (band notch) destroys
        the rate itself -> pred_rr and tachy_prob should be strongly disrupted.

If both hold, the model reads the respiratory frequency causally rather than
exploiting a spurious correlate.

TRANSFORMS (applied to the processed depth/flow/audio the model actually sees)
------------------------------------------------------------------------------
  time-scale alpha in {0.8, 0.9, 1.1, 1.2}  (reuses the dataset's verified
      `_time_compress`, which shifts the dominant RR by ~alpha to <0.5 bpm);
  phase-randomize (magnitude spectrum preserved, phases scrambled);
  notch (zero the dominant respiratory bin +/- tol, inverse FFT).

RESUME / EXECUTION
------------------
Single-process, num_workers=0, GPU-preferred (each fold's clips are forwarded in
batches per variant). Resume-safe: per-fold CSVs under outputs/counterfactual/ and
`.done` markers; re-running skips completed folds. `{fold}` in --checkpoint scores
each clip with its own fold's model (leakage-safe); a plain path uses one model for
all folds. THIS BOX CANNOT run the model (WinError 5) -- launch on the RTX 8000.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.respiratory_benchmark_dataset import RespiratoryBenchmarkDataset  # noqa: E402
from src.core.prism_breath_model import build_prism_breath_model_from_config  # noqa: E402

KFOLD = ROOT / "data" / "organized" / "kfold_learned"
CANONICAL_FS = 25.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHAS = [0.8, 0.9, 1.1, 1.2]  # baseline (1.0) handled separately


# --------------------------------------------------------------------------- #
# Signal transforms  (operate on a z-normed 1-D signal; renormalize afterwards)
# --------------------------------------------------------------------------- #
def _renorm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    m, s = float(x.mean()), float(x.std())
    return ((x - m) / s if s > 1e-6 else x - m).astype(np.float32)


def time_scale(sig: np.ndarray, alpha: float) -> np.ndarray:
    """Shift the dominant frequency by ~alpha, keeping length fixed.

    Delegates to the dataset's verified `_time_compress` (compress by c=alpha):
    alpha>1 => faster breathing (higher RR), alpha<1 => slower (lower RR)."""
    if not np.isfinite(alpha) or abs(alpha - 1.0) < 1e-6:
        return _renorm(sig)
    return RespiratoryBenchmarkDataset._time_compress(np.asarray(sig, dtype=np.float64), float(alpha))


def phase_randomize(sig: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Randomize FFT phases while preserving the magnitude spectrum (RR preserved)."""
    x = np.asarray(sig, dtype=np.float64).ravel()
    n = x.size
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    phases = rng.uniform(0, 2 * np.pi, size=mag.shape)
    phases[0] = 0.0  # keep DC real
    if n % 2 == 0:
        phases[-1] = 0.0  # Nyquist real for even length
    y = np.fft.irfft(mag * np.exp(1j * phases), n=n)
    return _renorm(y)


def notch_peak(sig: np.ndarray, fs: float, tol_hz: float,
               band=(0.1, 1.7)) -> np.ndarray:
    """Zero the dominant respiratory-band bin +/- tol_hz, inverse FFT (rate destroyed)."""
    x = np.asarray(sig, dtype=np.float64).ravel()
    n = x.size
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if in_band.any():
        mag = np.abs(spec)
        mag_band = np.where(in_band, mag, 0.0)
        f0 = freqs[int(np.argmax(mag_band))]
        kill = np.abs(freqs - f0) <= tol_hz
        spec = spec.copy()
        spec[kill] = 0.0
    y = np.fft.irfft(spec, n=n)
    return _renorm(y)


# --------------------------------------------------------------------------- #
# Frozen model per fold
# --------------------------------------------------------------------------- #
def load_model(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    config = ckpt.get("config", {})
    model_config = config.get("model_config", config)
    model = build_prism_breath_model_from_config(model_config).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    seq_len = int(model_config.get("seq_len", config.get("seq_len", 300)))
    return model, seq_len


@torch.no_grad()
def forward_variant(model, depth, flow, audio, age, mask, did, batch_size):
    """Batched forward -> (tachy_prob, pred_rr, threshold, triage) numpy arrays."""
    n = depth.shape[0]
    probs, rrs, thrs, tri = [], [], [], []
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        out = model(
            depth=torch.tensor(depth[sl], dtype=torch.float32, device=DEVICE),
            flow=torch.tensor(flow[sl], dtype=torch.float32, device=DEVICE),
            audio=torch.tensor(audio[sl], dtype=torch.float32, device=DEVICE),
            age_months=torch.tensor(age[sl], dtype=torch.float32, device=DEVICE),
            modality_mask=torch.tensor(mask[sl], dtype=torch.float32, device=DEVICE),
            dataset_ids=torch.tensor(did[sl], dtype=torch.long, device=DEVICE),
        )
        probs.append(out["tachy_prob"].cpu().numpy())
        rrs.append(out["rr_pred"].cpu().numpy())
        thrs.append(out["threshold_bpm"].cpu().numpy())
        tri.append(out["triage_code"].cpu().numpy())
    return (np.concatenate(probs), np.concatenate(rrs), np.concatenate(thrs), np.concatenate(tri))


def transform_stack(depth, flow, audio, mask, fn) -> tuple:
    """Apply `fn` to every informative channel (mask>0) of every clip."""
    d2, f2, a2 = depth.copy(), flow.copy(), audio.copy()
    for i in range(depth.shape[0]):
        if mask[i, 0] > 0:
            d2[i] = fn(depth[i])
        if mask[i, 1] > 0:
            f2[i] = fn(flow[i])
        if mask[i, 2] > 0:
            a2[i] = fn(audio[i])
    return d2, f2, a2


# --------------------------------------------------------------------------- #
def gather_fold(signal_dir, fold, seq_len):
    ds = RespiratoryBenchmarkDataset(signal_dir=signal_dir,
                                     manifest_path=str(KFOLD / f"fold{fold}" / "test.json"),
                                     seq_len=seq_len)
    depth, flow, audio, age, mask, did, gt, subj, clip = [], [], [], [], [], [], [], [], []
    for i in range(len(ds)):
        it = ds[i]
        depth.append(it["depth"].numpy()); flow.append(it["flow"].numpy()); audio.append(it["audio"].numpy())
        age.append(float(it["age_months"])); mask.append(it["modality_mask"].numpy())
        did.append(int(it["dataset_id"])); gt.append(float(it["gt_rr"]))
        subj.append(str(it["subject_id"])); clip.append(f"{it['subject_id']}_{it['video_id']}")
    return {
        "depth": np.stack(depth), "flow": np.stack(flow), "audio": np.stack(audio),
        "age": np.asarray(age, dtype=np.float32), "mask": np.stack(mask),
        "did": np.asarray(did), "gt": np.asarray(gt, dtype=np.float64),
        "subject": np.asarray(subj), "clip": np.asarray(clip),
    }


def process_fold(model, data, args, fold: int) -> List[Dict]:
    depth, flow, audio = data["depth"], data["flow"], data["audio"]
    age, mask, did = data["age"], data["mask"], data["did"]
    rng = np.random.RandomState(args.seed)

    # baseline
    b_prob, b_rr, thr, b_tri = forward_variant(model, depth, flow, audio, age, mask, did, args.batch_size)

    variants: Dict[str, tuple] = {}
    for a in ALPHAS:
        d2, f2, a2 = transform_stack(depth, flow, audio, mask, lambda s, a=a: time_scale(s, a))
        variants[f"a{a}"] = forward_variant(model, d2, f2, a2, age, mask, did, args.batch_size)
    dpr, fpr, apr = transform_stack(depth, flow, audio, mask, lambda s: phase_randomize(s, rng))
    variants["phaserand"] = forward_variant(model, dpr, fpr, apr, age, mask, did, args.batch_size)
    tol_hz = args.tol_bpm / 60.0
    dn, fn_, an = transform_stack(depth, flow, audio, mask, lambda s: notch_peak(s, CANONICAL_FS, tol_hz))
    variants["notch"] = forward_variant(model, dn, fn_, an, age, mask, did, args.batch_size)

    rows: List[Dict] = []
    alpha_axis = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    for i in range(len(data["clip"])):
        probs_curve = np.array([variants["a0.8"][0][i], variants["a0.9"][0][i], b_prob[i],
                                variants["a1.1"][0][i], variants["a1.2"][0][i]])
        rr_curve = np.array([variants["a0.8"][1][i], variants["a0.9"][1][i], b_rr[i],
                             variants["a1.1"][1][i], variants["a1.2"][1][i]])
        # direction: correlation of tachy_prob with alpha (>0 == expected direction)
        if np.std(probs_curve) < 1e-9:
            corr = 0.0
        else:
            corr = float(np.corrcoef(alpha_axis, probs_curve)[0, 1])
        # RR should scale ~alpha: slope of rr_curve/baseline vs alpha near 1
        rr_ratio_12 = float(rr_curve[-1] / b_rr[i]) if b_rr[i] != 0 else float("nan")
        row = {
            "clip_key": data["clip"][i], "subject_id": data["subject"][i], "fold": int(fold),
            "gt_rr": round(float(data["gt"][i]), 2), "threshold": round(float(thr[i]), 1),
            "age_months": round(float(age[i]), 1),
            "gt_tachy": int(data["gt"][i] >= thr[i]),
            "base_prob": round(float(b_prob[i]), 5), "base_rr": round(float(b_rr[i]), 3),
            "base_triage": int(b_tri[i]),
            "prob_a0.8": round(float(variants["a0.8"][0][i]), 5),
            "prob_a0.9": round(float(variants["a0.9"][0][i]), 5),
            "prob_a1.1": round(float(variants["a1.1"][0][i]), 5),
            "prob_a1.2": round(float(variants["a1.2"][0][i]), 5),
            "rr_a0.8": round(float(variants["a0.8"][1][i]), 3),
            "rr_a0.9": round(float(variants["a0.9"][1][i]), 3),
            "rr_a1.1": round(float(variants["a1.1"][1][i]), 3),
            "rr_a1.2": round(float(variants["a1.2"][1][i]), 3),
            "prob_alpha_corr": round(corr, 4),
            "dir_correct": int(corr > 0),
            "rr_ratio_a1.2": round(rr_ratio_12, 3),
            "phaserand_prob": round(float(variants["phaserand"][0][i]), 5),
            "phaserand_rr": round(float(variants["phaserand"][1][i]), 3),
            "notch_prob": round(float(variants["notch"][0][i]), 5),
            "notch_rr": round(float(variants["notch"][1][i]), 3),
        }
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal-dir", default="data/pedrr_signals_learned")
    ap.add_argument("--checkpoint",
                    default="outputs/paper_runs/pedrr_kfold_learned/fold{fold}/train/prism_breath_best.pt",
                    help="`{fold}` => per-fold leakage-safe model (default); plain path => one model for all folds.")
    ap.add_argument("--output-dir", default="outputs/counterfactual")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--tol-bpm", type=float, default=2.0, help="notch half-width around the dominant peak (bpm)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="recompute folds even if their CSV exists")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    signal_dir = str(ROOT / args.signal_dir) if not os.path.isabs(args.signal_dir) else args.signal_dir
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    per_fold = "{fold}" in args.checkpoint
    if not per_fold:
        print("[WARN] one --checkpoint for all folds (leakage on 4/5 folds); use `{fold}` for the clean run.")
    print(f"device={DEVICE}  folds={folds}  signal_dir={args.signal_dir}")

    all_rows: List[Dict] = []
    for k in folds:
        fold_csv = out / f"fold{k}_counterfactual.csv"
        done = out / f"fold{k}.done"
        if fold_csv.exists() and done.exists() and not args.overwrite:
            print(f"[fold{k}] present -> skipping")
            with open(fold_csv, encoding="utf-8") as fh:
                all_rows.extend(list(csv.DictReader(fh)))
            continue
        ckpt = Path(args.checkpoint.format(fold=k)) if per_fold else Path(args.checkpoint)
        ckpt = ckpt if ckpt.is_absolute() else ROOT / ckpt
        if not ckpt.exists():
            print(f"[fold{k}] missing checkpoint {ckpt} -> skipping")
            continue
        model, seq_len = load_model(ckpt)
        data = gather_fold(signal_dir, k, seq_len)
        print(f"[fold{k}] {len(data['clip'])} test clips  (ckpt={ckpt.name})")
        rows = process_fold(model, data, args, k)
        with open(fold_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        done.write_text("ok", encoding="utf-8")
        all_rows.extend(rows)
        del model

    if not all_rows:
        print("[abort] no folds produced rows.")
        return 1

    # combined CSV
    combined = out / "counterfactual_all.csv"
    with open(combined, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    # ---- summary ------------------------------------------------------- #
    def col(name):
        return np.array([float(r[name]) for r in all_rows], dtype=np.float64)

    base_prob = col("base_prob"); base_rr = col("base_rr"); thr = col("threshold")
    probs = {a: col(f"prob_a{a}") for a in ALPHAS}
    rrs = {a: col(f"rr_a{a}") for a in ALPHAS}
    dir_correct = col("dir_correct")
    gt_tachy = col("gt_tachy")

    # threshold-crossing: predicted-normal at baseline that flip to alert when sped up
    base_pred_normal = base_rr < thr
    base_pred_tachy = base_rr >= thr
    flip_up = float(np.mean((rrs[1.2][base_pred_normal] >= thr[base_pred_normal]))) if base_pred_normal.any() else float("nan")
    flip_down = float(np.mean((rrs[0.8][base_pred_tachy] < thr[base_pred_tachy]))) if base_pred_tachy.any() else float("nan")

    summary = {
        "n_clips": len(all_rows),
        "mean_tachy_prob": {
            "alpha_0.8": float(np.mean(probs[0.8])), "alpha_0.9": float(np.mean(probs[0.9])),
            "alpha_1.0": float(np.mean(base_prob)),
            "alpha_1.1": float(np.mean(probs[1.1])), "alpha_1.2": float(np.mean(probs[1.2])),
        },
        "mean_pred_rr": {
            "alpha_0.8": float(np.mean(rrs[0.8])), "alpha_0.9": float(np.mean(rrs[0.9])),
            "alpha_1.0": float(np.mean(base_rr)),
            "alpha_1.1": float(np.mean(rrs[1.1])), "alpha_1.2": float(np.mean(rrs[1.2])),
        },
        "mean_rr_ratio_vs_expected": {  # observed pred_rr(alpha)/pred_rr(1.0) ; ideal == alpha
            f"alpha_{a}": float(np.mean(rrs[a] / np.where(base_rr == 0, np.nan, base_rr))) for a in ALPHAS
        },
        "direction_correct_frac": float(np.mean(dir_correct)),
        "threshold_crossing": {
            "pred_normal_flip_to_alert_at_1.2": flip_up,
            "pred_tachy_flip_to_normal_at_0.8": flip_down,
            "n_pred_normal": int(base_pred_normal.sum()),
            "n_pred_tachy": int(base_pred_tachy.sum()),
        },
        "ablations": {
            "phaserand_mean_abs_drr": float(np.mean(np.abs(col("phaserand_rr") - base_rr))),
            "phaserand_mean_dprob": float(np.mean(col("phaserand_prob") - base_prob)),
            "notch_mean_abs_drr": float(np.mean(np.abs(col("notch_rr") - base_rr))),
            "notch_mean_dprob": float(np.mean(col("notch_prob") - base_prob)),
        },
    }
    json.dump(summary, open(out / "counterfactual_summary.json", "w"), indent=2)

    print("\n=== FREQUENCY COUNTERFACTUALS (means over {} clips) ===".format(len(all_rows)))
    print("alpha        0.8     0.9     1.0     1.1     1.2   (expect prob & RR to rise with alpha)")
    mp = summary["mean_tachy_prob"]; mr = summary["mean_pred_rr"]
    print("tachy_prob {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f}".format(
        mp["alpha_0.8"], mp["alpha_0.9"], mp["alpha_1.0"], mp["alpha_1.1"], mp["alpha_1.2"]))
    print("pred_rr    {:>7.2f} {:>7.2f} {:>7.2f} {:>7.2f} {:>7.2f}".format(
        mr["alpha_0.8"], mr["alpha_0.9"], mr["alpha_1.0"], mr["alpha_1.1"], mr["alpha_1.2"]))
    print(f"direction-correct fraction: {summary['direction_correct_frac']:.3f}")
    print(f"pred-normal -> alert at alpha=1.2: {flip_up:.3f}  |  pred-tachy -> normal at alpha=0.8: {flip_down:.3f}")
    ab = summary["ablations"]
    print(f"phase-rand  |dRR|={ab['phaserand_mean_abs_drr']:.2f}  dprob={ab['phaserand_mean_dprob']:+.3f}  "
          f"(expect SMALL: power spectrum/RR preserved)")
    print(f"notch-peak  |dRR|={ab['notch_mean_abs_drr']:.2f}  dprob={ab['notch_mean_dprob']:+.3f}  "
          f"(expect LARGE: respiratory rate destroyed)")
    print(f"\nwrote: {combined}\n       {out / 'counterfactual_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
