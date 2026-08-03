import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _pick_signal(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    depth = np.asarray(npz.get("depth", np.array([], dtype=np.float32)), dtype=np.float32).flatten()
    flow = np.asarray(npz.get("flow", np.array([], dtype=np.float32)), dtype=np.float32).flatten()

    # Prefer flow if it has usable variance; fallback to depth.
    if flow.size > 16 and float(np.std(flow)) > 1e-6:
        sig = flow
    elif depth.size > 16 and float(np.std(depth)) > 1e-6:
        sig = depth
    elif flow.size > 0:
        sig = flow
    else:
        sig = depth

    if sig.size == 0:
        return np.zeros(300, dtype=np.float32)

    # Min-max normalize to [-1, 1] to match existing foundation preprocessing.
    lo = float(np.min(sig))
    hi = float(np.max(sig))
    rng = hi - lo
    if rng > 1e-6:
        sig = (2.0 * ((sig - lo) / rng) - 1.0).astype(np.float32)
    else:
        sig = np.zeros_like(sig, dtype=np.float32)

    return sig


def _make_text(rr: float, baseline_name: str, child_id: str) -> str:
    if rr >= 30:
        status = "tachypnea"
    elif rr <= 15:
        status = "bradypnea"
    else:
        status = "eupnea"
    return (
        f"Pediatric respiratory recording from child {child_id}. "
        f"Estimated/annotated respiratory rate is approximately {int(round(rr))} bpm. "
        f"Breathing status suggests {status}. "
        f"Signal extracted from baseline pipeline: {baseline_name}."
    )


def build_dataset(input_dir: Path, output_dir: Path, baseline_priority: List[str]) -> Dict[str, int]:
    output_signals = output_dir / "signals"
    output_signals.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("Pediatric-RR_*.npz"))
    if not files:
        raise FileNotFoundError(f"No Pediatric-RR npz files found in {input_dir}")

    # Group by child ID and keep preferred baseline per child.
    by_child: Dict[str, Dict[str, Path]] = {}
    for path in files:
        parts = path.stem.split("_", 2)
        if len(parts) < 3:
            continue
        _, child_id, baseline_name = parts
        by_child.setdefault(child_id, {})[baseline_name] = path

    rows = []
    selected = 0
    for child_id, baseline_map in sorted(by_child.items()):
        chosen_baseline = None
        chosen_path = None
        for b in baseline_priority:
            if b in baseline_map:
                chosen_baseline = b
                chosen_path = baseline_map[b]
                break
        if chosen_path is None:
            # Deterministic fallback if preferred list does not match.
            chosen_baseline = sorted(baseline_map.keys())[0]
            chosen_path = baseline_map[chosen_baseline]

        data = np.load(chosen_path, allow_pickle=True)
        signal = _pick_signal(data)

        gt_raw = data.get("gt", np.nan)
        try:
            gt = float(np.asarray(gt_raw).reshape(-1)[0])
        except Exception:
            gt = float("nan")

        out_path = output_signals / f"PED_{child_id}.npz"
        np.savez_compressed(out_path, signal=signal)

        rr_for_text = gt if np.isfinite(gt) else 20.0
        rows.append(
            {
                "patient_id": f"PED_{child_id}",
                "condition": "Pediatric-RR",
                "true_bpm": gt if np.isfinite(gt) else "",
                "clinical_text": _make_text(rr_for_text, chosen_baseline, child_id),
                "has_sensor_loss": False,
                "signal_path": str(out_path.resolve()),
                "source": "Pediatric-RR",
                "baseline_name": chosen_baseline,
                "source_npz": str(chosen_path.resolve()),
            }
        )
        selected += 1

    meta_path = output_dir / "clinical_notes_metadata.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)

    return {
        "input_files": len(files),
        "subjects": len(by_child),
        "selected": selected,
        "metadata": str(meta_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Pediatric-RR foundation dataset from ablation signal npz files.")
    parser.add_argument("--input-dir", type=str, default="data/ablation_results/signals")
    parser.add_argument("--output-dir", type=str, default="data/foundation_data/pediatric")
    parser.add_argument(
        "--baseline-priority",
        type=str,
        default="Fused-VMD,Fused-Keypoint,Depth-Only,Flow-Only",
        help="Comma-separated baseline preference used to choose one signal per child",
    )
    args = parser.parse_args()

    stats = build_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        baseline_priority=_parse_csv_list(args.baseline_priority),
    )

    print("[DONE] Pediatric foundation dataset ready")
    print(f"  Input files   : {stats['input_files']}")
    print(f"  Subjects      : {stats['subjects']}")
    print(f"  Selected rows : {stats['selected']}")
    print(f"  Metadata CSV  : {stats['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())