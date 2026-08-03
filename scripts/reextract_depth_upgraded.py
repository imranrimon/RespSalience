"""#3 (conditional on the Tier-2 A/B winning): re-extract the DEPTH channel for all
Pediatric-RR clips with the upgraded extractor, merged with the existing learned flow.

Writes data/pedrr_signals_upgraded/Pediatric-RR_{subj}_{vid}.npz = upgraded depth + the
learned flow/audio/gt from data/pedrr_signals_learned. The learned flow (which already
captures high RR) is preserved; only 'depth' is replaced. Then retrain via the CV harness:

  python scripts/make_kfold_manifests.py --signal-dir data/pedrr_signals_upgraded --out-dir data/organized/kfold_upgraded
  # then run_kfold_cv.ps1 with Sig=data/pedrr_signals_upgraded, KF=data/organized/kfold_upgraded, new Base

Resume-safe (skips clips already written). GPU (Depth-Anything-V2) when --source depth.
Parameterize --source/--roi/--harmonic to the config the Tier-2 A/B selected.

Usage (only after Tier-2 wins):
  python scripts/reextract_depth_upgraded.py --source depth --roi abdomen
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.upgraded_respiration_extractor import extract_upgraded, DEFAULT_LOW_HZ, DEFAULT_HIGH_HZ

MANIFESTS = ["data/organized/pedrr_indist_train.json",
             "data/organized/pedrr_indist_val.json",
             "data/organized/pedrr_indist_test.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="depth", choices=["depth", "intensity"])
    ap.add_argument("--roi", default="abdomen")
    ap.add_argument("--harmonic", action="store_true", default=True)
    ap.add_argument("--no-harmonic", dest="harmonic", action="store_false")
    ap.add_argument("--low", type=float, default=DEFAULT_LOW_HZ)
    ap.add_argument("--high", type=float, default=DEFAULT_HIGH_HZ)
    ap.add_argument("--learned-dir", default="data/pedrr_signals_learned")
    ap.add_argument("--out-dir", default="data/pedrr_signals_upgraded")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    learned = Path(args.learned_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    recs = {}
    for m in MANIFESTS:
        for r in json.load(open(m)):
            recs[(r["subject_id"], r["video_id"])] = r

    saved = miss = skip = 0
    for (subj, vid), r in recs.items():
        name = f"Pediatric-RR_{subj}_{vid}.npz"
        dst = out / name
        if dst.exists() and not args.overwrite:
            skip += 1; continue
        base = learned / name
        if not base.exists():
            miss += 1; continue
        path = r.get("full_path")
        if not path or not Path(path).exists():
            miss += 1; continue
        try:
            up = extract_upgraded(path, roi=args.roi, source=args.source,
                                  low=args.low, high=args.high, harmonic=args.harmonic)
        except Exception as e:
            print(f"[fail] {name}: {e}"); miss += 1; continue
        z = dict(np.load(base, allow_pickle=True))
        z["depth"] = np.asarray(up["waveform"], dtype=np.float32)  # replace ONLY depth
        np.savez_compressed(dst, **z)
        saved += 1
        if saved % 25 == 0:
            print(f"  ...{saved} written")
    print(f"DONE: saved={saved} skipped={skip} missing={miss} -> {out}  "
          f"(source={args.source} roi={args.roi} harmonic={args.harmonic})")


if __name__ == "__main__":
    raise SystemExit(main())
