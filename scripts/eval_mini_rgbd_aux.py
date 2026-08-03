"""
Auxiliary MINI-RGBD evaluation (image-sequence domain transfer, no RR ground truth).

This script reports proxy quality metrics:
- predicted RR distribution
- confidence distribution
- success rate
- physiologic plausibility rate

Outputs:
- per-sequence CSV
- summary CSV
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.universal_adapter import UniversalAdapter


def discover_mini_rgbd_sequences(root: Path) -> List[Path]:
    """
    Supports both layouts:
    1) data/MINI-RGBD/MINI-RGBD_web/01/rgb
    2) data/MINI-RGBD/01/rgb
    """
    candidates = [root / "MINI-RGBD_web", root]
    found: List[Path] = []
    seen = set()

    for base in candidates:
        if not base.exists():
            continue
        for seq_dir in sorted(base.glob("*")):
            if not seq_dir.is_dir():
                continue
            if not seq_dir.name.isdigit():
                continue
            rgb_dir = seq_dir / "rgb"
            if not rgb_dir.exists():
                continue
            if not any(rgb_dir.glob("*.png")):
                continue
            key = str(rgb_dir.resolve())
            if key not in seen:
                seen.add(key)
                found.append(rgb_dir)
    return found


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MINI-RGBD as auxiliary image-sequence benchmark.")
    parser.add_argument("--data-root", type=str, default="data/MINI-RGBD")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of sequences to process (0 = all)")
    parser.add_argument("--plausible-min-bpm", type=float, default=6.0)
    parser.add_argument("--plausible-max-bpm", type=float, default=90.0)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/auxiliary/mini_rgbd_per_sequence.csv",
        help="Per-sequence output CSV",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="outputs/auxiliary/mini_rgbd_summary.csv",
        help="Summary output CSV",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    sequences = discover_mini_rgbd_sequences(data_root)
    if not sequences:
        raise SystemExit(f"No MINI-RGBD sequence folders found under: {data_root}")
    if args.limit > 0:
        sequences = sequences[: args.limit]

    adapter = UniversalAdapter()
    rows = []

    print(f"[INFO] Found {len(sequences)} MINI-RGBD sequences.")
    for seq in sequences:
        seq_id = seq.parent.name
        row = {
            "dataset": "MINI-RGBD",
            "sequence_id": seq_id,
            "sequence_path": str(seq),
            "fps": float(args.fps),
            "n_frames": 0,
            "rr_bpm_pred": float("nan"),
            "confidence": float("nan"),
            "method": "",
            "modalities_used": "",
            "status": "ok",
            "error": "",
        }
        try:
            bundle = adapter.load(str(seq), fps=args.fps)
            result = adapter.estimate_rr(bundle)
            n_frames = int(bundle.frames.shape[0]) if bundle.frames is not None else 0

            row["n_frames"] = n_frames
            row["rr_bpm_pred"] = _safe_float(result.get("rr_bpm"))
            row["confidence"] = _safe_float(result.get("confidence"))
            row["method"] = str(result.get("method", ""))
            row["modalities_used"] = ",".join(result.get("modalities_used", []))
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        rows.append(row)

    per_df = pd.DataFrame(rows)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per_df.to_csv(out_csv, index=False)

    ok = per_df[per_df["status"] == "ok"].copy()
    rr = pd.to_numeric(ok["rr_bpm_pred"], errors="coerce")
    conf = pd.to_numeric(ok["confidence"], errors="coerce")
    plausible = ((rr >= args.plausible_min_bpm) & (rr <= args.plausible_max_bpm)).astype(float)

    summary = pd.DataFrame(
        [
            {
                "dataset": "MINI-RGBD",
                "n_sequences_total": int(len(per_df)),
                "n_sequences_success": int(len(ok)),
                "success_rate": float(len(ok) / len(per_df)) if len(per_df) > 0 else 0.0,
                "rr_mean": float(rr.mean()) if len(rr) > 0 else float("nan"),
                "rr_std": float(rr.std()) if len(rr) > 0 else float("nan"),
                "rr_median": float(rr.median()) if len(rr) > 0 else float("nan"),
                "rr_q1": float(rr.quantile(0.25)) if len(rr) > 0 else float("nan"),
                "rr_q3": float(rr.quantile(0.75)) if len(rr) > 0 else float("nan"),
                "confidence_mean": float(conf.mean()) if len(conf) > 0 else float("nan"),
                "confidence_std": float(conf.std()) if len(conf) > 0 else float("nan"),
                "plausible_rate": float(plausible.mean()) if len(plausible) > 0 else float("nan"),
                "plausible_range_bpm": f"[{args.plausible_min_bpm}, {args.plausible_max_bpm}]",
                "note": "No RR ground truth in MINI-RGBD; report proxy transfer quality only.",
            }
        ]
    )

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)

    print(f"[OK] Saved MINI per-sequence CSV: {out_csv}")
    print(f"[OK] Saved MINI summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
