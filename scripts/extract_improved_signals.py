"""Fork B2: re-extract Pediatric-RR respiration signals with a corrected extractor.

B1 proved the 12-60mo breathing signal is recoverable (chest-ROI vertical optical flow
r=0.94 vs the old pipeline's r=0.03). This rebuilds the per-clip 1-D modality signals:
  * flow  <- mean VERTICAL Farneback optical flow in a central chest ROI (the winner)
  * depth <- top respiratory-band PCA component of the ROI per-pixel time series (complementary)
  * audio <- zeros (Pediatric-RR has no usable audio track)
Signals are standardized (RR lives in the frequency, which standardization preserves), gt is
preserved from the existing npz/manifest, and new npz are written to a fresh dir so the
originals are untouched. Emits per-age-band recovered-RR-vs-GT QA so we can confirm no band
regresses before retraining.

Usage:
  python scripts/extract_improved_signals.py --out-dir data/pedrr_signals_v2 [--qa 8]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path("G:/Respiratory Project")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Single source of truth for the extraction numerics: the same function the paper pipeline
# (scripts/extract_benchmark_signals.py) uses. This file is the QA / one-off driver around it.
from src.core.respiratory_signal_extractor import extract_video_signals as extract  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/pedrr_signals_v2")
    ap.add_argument("--old-signal-dir", default="data/paper_signals")
    ap.add_argument("--qa", type=int, default=0, help="if >0, only process a stratified QA subset")
    args = ap.parse_args()

    # union of Pediatric-RR clips across the pedrr splits
    seen, clips = set(), []
    for sp in ["train", "val", "test"]:
        for e in json.load(open(REPO / f"data/organized/pedrr_indist_{sp}.json")):
            key = (e["subject_id"], e["video_id"])
            if str(e.get("dataset", "")) == "Pediatric-RR" and key not in seen:
                seen.add(key)
                p = e.get("full_path")
                if p and os.path.exists(p):
                    clips.append((e["subject_id"], e["video_id"], int(e.get("age_months", 6)),
                                  float(e["gt_rr_bpm"]), p))
    clips.sort(key=lambda c: c[3])
    if args.qa:
        idx = np.linspace(0, len(clips) - 1, args.qa).astype(int)
        clips = [clips[i] for i in idx]
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {len(clips)} clips -> {out}")

    qa = []
    for i, (s, v, age, gt, p) in enumerate(clips):
        r = extract(p)
        if r is None:
            print(f"  [skip] {s}/{v} (unreadable)")
            continue
        if not args.qa:
            np.savez(out / f"Pediatric-RR_{s}_{v}.npz",
                     depth=r["depth"], flow=r["flow"], audio=r["audio"],
                     gt=np.float64(gt), vlm_label=False, vlm_text="N/A",
                     source_dataset="Pediatric-RR", subject_id=s, video_id=v, age_months=age)
        qa.append({"age": age, "gt": gt, "rr_flow": r["rr_flow"], "rr_depth": r["rr_depth"]})
        if i % 25 == 0 or args.qa:
            print(f"  {i+1}/{len(clips)} {s}/{v} age={age} gt={gt:5.1f} rr_flow={r['rr_flow']:5.1f} rr_depth={r['rr_depth']:5.1f}")

    # QA: per-band recovered-RR vs GT correlation
    df = qa
    def band(a): return "<2mo" if a < 2 else ("2-12mo" if a < 12 else "12-60mo")
    print("\n[QA] recovered-RR (flow) vs GT by age band:")
    import collections
    groups = collections.defaultdict(list)
    for r in df:
        groups[band(r["age"])].append(r)
    for b, rows in sorted(groups.items()):
        g = np.array([x["gt"] for x in rows]); vf = np.array([x["rr_flow"] for x in rows], float)
        mask = np.isfinite(vf)
        rr = np.corrcoef(g[mask], vf[mask])[0, 1] if mask.sum() > 2 and np.std(vf[mask]) > 1e-6 else float("nan")
        mae = np.abs(g[mask] - vf[mask]).mean() if mask.sum() else float("nan")
        print(f"  {b:8}: N={len(rows):3d}  flow r={rr:+.2f}  MAE={mae:.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
