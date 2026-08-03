"""Pillar B: generate synthetic tachypnea training signals by RR-conditioned time-warping.

The Pediatric-RR benchmark has real RR spread (16-96) but few tachypnea positives, and
PRISM systematically under-predicts high RR (Bland-Altman bias -3.45, worst -11 for
neonates) -> tachypnea sensitivity 0.20. Public infant *tachypnea* video does not exist.

Approach (defensible + exact): take REAL Pediatric-RR train clips and rescale their
breathing frequency by a warp factor (target_rr / source_rr), producing signals with
realistic morphology (noise, baseline wander, depth/flow coupling — because it IS real
data) at controlled target RR, emphasising the missing 50-90 BPM tachypnea zone. Audio is
kept zero (matches the dataset). Signals are written as SYNTH-TACHY_*.npz alongside the
real signals, and an augmented train manifest is emitted.

Usage:
  python scripts/generate_synthetic_respiration.py \
      --train-manifest data/organized/pedrr_indist_train.json \
      --signal-dir data/paper_signals --n-synth 400 \
      --out-manifest data/organized/pedrr_indist_train_synthaug.json --seed 42
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np

WHO_BANDS = [(0, 2, 60), (2, 12, 50), (12, 60, 40)]


def who_threshold(age_months: float) -> float:
    for lo, hi, thr in WHO_BANDS:
        if lo <= age_months < hi:
            return float(thr)
    return 40.0


def warp_signal(x: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    """Rescale the temporal frequency of x by `factor` (freq *= factor), keeping length N.
    factor>1 speeds breathing up (higher RR). Tiles then interpolates to avoid seams; adds
    a random phase so repeated draws from the same clip differ."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 8 or x.std() < 1e-9:
        return x.copy()
    reps = int(np.ceil(factor)) + 2
    xt = np.tile(x, reps)
    phase = float(rng.uniform(0, n))
    idx = np.arange(n) * factor + phase
    y = np.interp(idx, np.arange(len(xt)), xt)
    return y.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", default="data/organized/pedrr_indist_train.json")
    ap.add_argument("--signal-dir", default="data/paper_signals")
    ap.add_argument("--out-manifest", default="data/organized/pedrr_indist_train_synthaug.json")
    ap.add_argument("--n-synth", type=int, default=400)
    ap.add_argument("--rr-min", type=float, default=35.0, help="min target RR")
    ap.add_argument("--rr-max", type=float, default=90.0, help="max target RR")
    ap.add_argument("--tachy-frac", type=float, default=0.8, help="fraction of synth samples that are tachypnic")
    ap.add_argument("--amp-jitter", type=float, default=0.15)
    ap.add_argument("--noise-frac", type=float, default=0.05, help="added Gaussian noise as fraction of signal std")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sigdir = Path(args.signal_dir)
    real = json.load(open(args.train_manifest))
    real = [e for e in real if str(e.get("dataset", e.get("source", ""))) == "Pediatric-RR"]

    # Load source signals (only clips with a usable depth signal).
    sources = []
    for e in real:
        p = sigdir / f"Pediatric-RR_{e['subject_id']}_{e['video_id']}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        gt = float(d["gt"])
        if gt <= 0 or np.asarray(d["depth"], float).std() < 1e-6:
            continue
        sources.append({"depth": np.asarray(d["depth"], float), "flow": np.asarray(d["flow"], float),
                        "audio_len": len(np.asarray(d["audio"], float)), "gt": gt})
    if not sources:
        raise RuntimeError("No usable source signals found.")
    print(f"[src] {len(sources)} real train clips usable (RR {min(s['gt'] for s in sources):.0f}"
          f"-{max(s['gt'] for s in sources):.0f})")

    synth_entries = []
    written = 0
    for i in range(args.n_synth):
        src = sources[int(rng.integers(len(sources)))]
        # target RR: emphasise tachypnea; the rest spread across mid-high normal.
        if rng.random() < args.tachy_frac:
            target = float(rng.uniform(max(48.0, args.rr_min), args.rr_max))
        else:
            target = float(rng.uniform(args.rr_min, 48.0))
        factor = target / src["gt"]
        factor = float(np.clip(factor, 0.4, 4.0))
        target = src["gt"] * factor  # keep target consistent with the clipped factor

        depth = warp_signal(src["depth"], factor, rng)
        flow = warp_signal(src["flow"], factor, rng)
        # amplitude jitter + light noise for diversity
        depth *= (1.0 + rng.uniform(-args.amp_jitter, args.amp_jitter))
        depth += rng.normal(0, args.noise_frac * (depth.std() + 1e-6), size=len(depth)).astype(np.float32)
        flow = np.clip(flow * (1.0 + rng.uniform(-args.amp_jitter, args.amp_jitter)), 0.0, 1.0)
        audio = np.zeros(src["audio_len"] or len(depth), np.float32)  # match dataset (audio absent)

        # age so the sample is a genuine tachypnea positive when target is high.
        eligible = [(lo, hi) for lo, hi, thr in WHO_BANDS if target >= thr]
        if not eligible:
            eligible = [(12, 60)]
        lo, hi = eligible[int(rng.integers(len(eligible)))]
        age = int(rng.integers(max(1, lo), hi))

        subj = f"synth{i:04d}"
        vid = "000"
        name = f"SYNTH-TACHY_{subj}_{vid}.npz"
        np.savez(sigdir / name, depth=depth, flow=flow, audio=audio,
                 gt=np.float64(round(target, 1)), vlm_label=False, vlm_text="N/A (synthetic)",
                 source_dataset="SYNTH-TACHY", subject_id=subj, video_id=vid, age_months=age)
        synth_entries.append({
            "dataset": "SYNTH-TACHY", "subject_id": subj, "video_id": vid,
            "age_months": age, "gt_rr_bpm": round(target, 1), "gt_source": "synthetic_timewarp",
            "protocol_split": "train_synth", "paper_scope": "respiratory_benchmark_rr_triage",
        })
        written += 1

    # augmented manifest = real train + synth
    out = real + synth_entries
    json.dump(out, open(args.out_manifest, "w"), indent=2)
    rrs = np.array([e["gt_rr_bpm"] for e in synth_entries])
    print(f"[synth] wrote {written} clips | target RR min/med/max={rrs.min():.0f}/{np.median(rrs):.0f}/{rrs.max():.0f} "
          f"| tachy(>=50)={int((rrs>=50).sum())}")
    print(f"[manifest] {args.out_manifest}: {len(real)} real + {len(synth_entries)} synth = {len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
