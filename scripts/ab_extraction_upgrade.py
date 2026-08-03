"""A/B experiment: current depth proxy vs the upgraded extractor, at the SIGNAL level.

Decisive & cheap (CPU, no retrain): does motion-magnification + abdomen-ROI + harmonic-aware
RR recover the true high RR on the clips where the current proxy fails, WITHOUT breaking the
easy (normal-RR) controls? Runs on the 4 known tachypnea false-negative clips plus low-RR
controls, and ablates each upgrade component so we learn which one carries the win.

If Tier-1 (intensity source) already recovers high RR, the info was in the pixels and averaging
destroyed it -> a cheap fix. If not, add --source depth (Tier-2, Depth-Anything-V2) -> the root fix.

Then, and only then, is a full re-extraction + retrain justified. This script commits neither.

Usage:
  python scripts/ab_extraction_upgrade.py                 # Tier-1 intensity, FN clips + controls
  python scripts/ab_extraction_upgrade.py --source depth  # Tier-2 real depth
  python scripts/ab_extraction_upgrade.py --n-controls 6
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.upgraded_respiration_extractor import (
    extract_upgraded, extract_proxy_baseline, motion_weighted_waveform,
    harmonic_rr, argmax_rr, _decode_gray, _roi_slice, DEFAULT_LOW_HZ, DEFAULT_HIGH_HZ,
)

# The four confirmed high-RR tachypnea false-negatives (gt >> proxy).
FN_CLIPS = [("ID-1177", "100-4680", 63.5), ("ID-1330", "100-4927", 65.5),
            ("ID-1405", "100-5063", 61.0), ("ID-2205", "100-5318", 58.5)]
MANIFESTS = ["data/organized/pedrr_indist_train.json",
             "data/organized/pedrr_indist_val.json",
             "data/organized/pedrr_indist_test.json"]


def load_index():
    idx = {}
    for m in MANIFESTS:
        for r in json.load(open(m)):
            idx[(r["subject_id"], r["video_id"])] = r
    return idx


def variants(path, low, high, source):
    """Return {label: rr} for the proxy + full upgrade + per-component ablations."""
    out = {}
    out["proxy(base)"] = extract_proxy_baseline(path)["rr_bpm"]
    # decode once; run ROI/aggregation/RR ablations on the same stack
    stack, fps = _decode_gray(path) if source == "intensity" else (None, None)
    def run(roi, magnify, harmonic):
        if source == "intensity":
            ys, xs = _roi_slice(stack.shape[1], stack.shape[2], roi)
            wave, _ = motion_weighted_waveform(stack[:, ys, xs], fps, low, high, magnify=magnify)
            return harmonic_rr(wave, fps, low, high) if harmonic else argmax_rr(wave, fps, low, high)
        return extract_upgraded(path, roi=roi, source=source, low=low, high=high,
                                magnify=magnify, harmonic=harmonic)["rr_bpm"]
    out["abd+mag+harm(full)"] = run("abdomen", True, True)
    out["abd+FLATmean+harm"]  = run("abdomen", False, True)   # ablate magnification
    out["abd+mag+ARGMAX"]     = run("abdomen", True, False)   # ablate harmonic-aware
    out["chest+mag+harm"]     = run("chest", True, True)      # ROI check
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="intensity", choices=["intensity", "depth"])
    ap.add_argument("--n-controls", type=int, default=4)
    ap.add_argument("--low", type=float, default=DEFAULT_LOW_HZ)
    ap.add_argument("--high", type=float, default=DEFAULT_HIGH_HZ)
    args = ap.parse_args()

    idx = load_index()
    # controls = lowest-gt clips (easy, normal-RR) not in the FN set
    fn_keys = {(s, v) for s, v, _ in FN_CLIPS}
    controls = sorted(((r["gt_rr_bpm"], k, r) for k, r in idx.items() if k not in fn_keys))[:args.n_controls]
    rows = [(s, v, gt, "FN") for s, v, gt in FN_CLIPS] + [(k[0], k[1], gt, "ctrl") for gt, k, r in controls]

    labels = ["proxy(base)", "abd+mag+harm(full)", "abd+FLATmean+harm", "abd+mag+ARGMAX", "chest+mag+harm"]
    print(f"[source={args.source}] band={args.low:.2f}-{args.high:.2f} Hz\n")
    header = f"{'clip':<20}{'grp':<5}{'gt':>6} | " + " ".join(f"{l:>19}" for l in labels)
    print(header); print("-" * len(header))
    err = {l: [] for l in labels}
    for s, v, gt, grp in rows:
        rec = idx.get((s, v))
        if rec is None:
            print(f"{s}_{v:<12} MISSING from manifest"); continue
        path = rec.get("full_path")
        if not path or not Path(path).exists():
            print(f"{s}_{v:<12} video not found: {path}"); continue
        try:
            res = variants(path, args.low, args.high, args.source)
        except Exception as e:
            print(f"{s}_{v:<12} FAILED: {e}"); continue
        cells = []
        for l in labels:
            rr = res.get(l, float("nan"))
            err[l].append(abs(rr - gt))
            cells.append(f"{rr:>6.1f}({rr-gt:+5.1f})     ")
        print(f"{s}_{v:<12}{grp:<5}{gt:>6.1f} | " + " ".join(c.strip().rjust(19) for c in cells))
    print("\n[MAE by method]  (FN clips are the target; controls must NOT regress)")
    for l in labels:
        vals = err[l]
        print(f"  {l:<22} MAE={sum(vals)/max(1,len(vals)):.2f}  (n={len(vals)})")
    print("\nRead: does 'abd+mag+harm(full)' pull the FN clips to ~gt while controls stay accurate?")
    print("The ablation columns show whether magnification, harmonic-RR, or the abdomen ROI carries it.")


if __name__ == "__main__":
    raise SystemExit(main())
