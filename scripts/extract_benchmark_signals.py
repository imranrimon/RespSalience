"""
Split-aware signal extraction for leakage-safe training.

Reads a manifest JSON (for example a train split only), extracts per-video depth/flow/audio
respiration signals plus ground-truth RR (and optional VLM surrogate labels), and writes one
``.npz`` per video following the ``{dataset}_{subject}_{video}.npz`` provenance contract that
``src/core/respiratory_benchmark_dataset.py`` reads back.

Two extraction methods (``--method``):
  * ``improved`` (default) -- chest-ROI VERTICAL optical flow + band-pass intensity depth,
    from ``src.core.respiratory_signal_extractor.extract_video_signals``. This is the extraction
    that lifted 12-60mo tachypnea AUROC from 0.51 (chance) to 0.95 and halved overall RR MAE
    (11.2 -> 5.6) on Pediatric-RR after retraining, because the legacy global depth/flow path was
    discarding the breathing signal for larger / mobile children. It needs only numpy + cv2 (no
    model load) and applies to the RGB-video datasets (AIR-125, AIR-400, Pediatric-RR). For any
    other dataset in the manifest it transparently falls back to the controller path below.
  * ``controller`` -- the legacy ``RespiratoryController.process_video`` path (Depth-Anything-V2
    depth, global optical flow, audio spectrogram, optional VLM). Kept for reproducing older
    signal dirs and for non-RGB-video datasets. Heavy deps are imported lazily, only if used.

The ``.npz`` schema (keys, dtypes, and the always-present zero ``audio`` key) is identical across
both methods so downstream modality-availability masks and readers are unaffected.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.respiratory_signal_extractor import extract_video_signals
from src.utils.respiratory_ground_truth import (
    load_air125_ground_truth,
    load_air400_ground_truth,
    load_pediatric_rr_ground_truth,
)

# Datasets whose entries are RGB videos the improved extractor understands. Anything else in a
# manifest falls back to the controller path.
VIDEO_DATASETS = {"AIR-125", "AIR-400", "Pediatric-RR", "Pediatric-RR-332"}


def _slugify(text: str) -> str:
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _resolve_gt(entry: dict):
    """Resolve GT RR from dataset-specific loaders.

    Returns:
        (gt_bpm, age_months)
    """
    dataset = str(entry.get("dataset", "")).strip()
    video_path = str(entry.get("full_path", ""))

    gt_bpm = None
    age_months = 6

    if dataset == "AIR-400":
        gt_bpm = load_air400_ground_truth(video_path)
    elif dataset == "AIR-125":
        gt_bpm = load_air125_ground_truth(video_path)
    elif dataset in ("Pediatric-RR", "Pediatric-RR-332"):
        ped_gt = load_pediatric_rr_ground_truth(video_path)
        if ped_gt:
            gt_bpm = ped_gt.get("respiratory_rate")
            age_months = int(ped_gt.get("age_months", 6))

    return gt_bpm, age_months


def _gt_and_age(entry: dict):
    """GT RR + age for a manifest entry.

    Prefer the manifest's precomputed ``gt_rr_bpm`` -- this is the value the AIR-125 GT fix patched
    and the value the validated ``pedrr_signals_v2`` run trained on -- and fall back to the
    dataset-specific loaders when it is absent.
    """
    g = entry.get("gt_rr_bpm")
    if g is not None:
        try:
            return float(g), int(entry.get("age_months", 6) or 6)
        except (TypeError, ValueError):
            pass
    return _resolve_gt(entry)


# Lazily-instantiated live-inference controller (heavy: torch / mediapipe / Depth-Anything).
_CONTROLLER = None


def _get_controller(config_path: str, checkpoint_path: str):
    global _CONTROLLER
    if _CONTROLLER is None:
        from src.core.main_controller import RespiratoryController  # lazy import: heavy deps

        _CONTROLLER = RespiratoryController(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            enable_depth=True,
            enable_flow=True,
            enable_audio=True,
            enable_vmd=False,
            enable_keypoint=True,
            enable_transformer=False,
            enable_supervised=False,
        )
    return _CONTROLLER


def _signals_improved(video_path: str, variant: str = "fixed"):
    """depth/flow/audio via the productionized extractor. variant: 'fixed' (validated pedrr_v2) or
    'gated_consensus' (Tier-1 chest+abdomen consensus). None if unreadable."""
    r = extract_video_signals(video_path, variant=variant)
    if r is None:
        return None
    return {
        "depth": r["depth"],
        "flow": r["flow"],
        "audio": r["audio"],
        "vlm_label": False,
        "vlm_text": "N/A",
    }


def _signals_controller(video_path, age_months, enable_vlm, vlm_model_id, config_path, checkpoint_path):
    """depth/flow/audio (+ optional VLM) via the legacy RespiratoryController path."""
    controller = _get_controller(config_path, checkpoint_path)
    result = controller.process_video(
        video_path,
        age_months=age_months,
        vlm_model_id=vlm_model_id if enable_vlm else None,
    )
    return {
        "depth": np.asarray(result.get("depth_signal", []), dtype=np.float32),
        "flow": np.asarray(result.get("flow_signal", []), dtype=np.float32),
        "audio": np.asarray(result.get("audio_signal", []), dtype=np.float32),
        "vlm_label": bool(result.get("vlm_distress", False)),
        "vlm_text": str(result.get("vlm_assessment", "N/A")),
    }


def extract_benchmark_signals(
    manifest_path: str,
    output_dir: str,
    config_path: str,
    checkpoint_path: str,
    enable_vlm: bool,
    vlm_model_id: str,
    overwrite: bool,
    limit: int,
    method: str,
    extract_variant: str = "fixed",
):
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if limit and limit > 0:
        manifest = manifest[:limit]

    print(f"Loaded {len(manifest)} entries from {manifest_file}")
    print(f"Extraction method: {method}" + (f" (variant={extract_variant})" if method == "improved" else ""))
    if method == "improved" and enable_vlm:
        print("[NOTE] --enable-vlm is ignored for method=improved (VLM needs the controller path); "
              "vlm_label/vlm_text are written as False/'N/A'.")

    saved = 0
    skipped_existing = 0
    skipped_missing_gt = 0
    failed = 0
    fell_back = 0

    for entry in tqdm(manifest, desc="Extracting split signals"):
        dataset_raw = str(entry.get("dataset", "unknown"))
        dataset = _slugify(dataset_raw)
        subject = _slugify(entry.get("subject_id", "unknown_subject"))
        video_id = _slugify(entry.get("video_id", Path(str(entry.get("full_path", "vid"))).stem))
        video_path = str(entry.get("full_path", ""))

        if not video_path:
            failed += 1
            continue

        out_name = f"{dataset}_{subject}_{video_id}.npz"
        out_path = out_dir / out_name

        if out_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        gt_bpm, age_months = _gt_and_age(entry)
        if gt_bpm is None:
            skipped_missing_gt += 1
            continue

        try:
            use_improved = method == "improved" and dataset_raw in VIDEO_DATASETS
            sig = _signals_improved(video_path, extract_variant) if use_improved else None
            if sig is None:
                # Non-video dataset under method=improved, or an unreadable video: fall back to the
                # controller path so a mixed manifest still fully extracts.
                if use_improved:
                    fell_back += 1
                sig = _signals_controller(
                    video_path, age_months, enable_vlm, vlm_model_id, config_path, checkpoint_path
                )

            np.savez_compressed(
                out_path,
                depth=np.asarray(sig["depth"], dtype=np.float32),
                flow=np.asarray(sig["flow"], dtype=np.float32),
                audio=np.asarray(sig["audio"], dtype=np.float32),
                gt=float(gt_bpm),
                vlm_label=bool(sig["vlm_label"]),
                vlm_text=str(sig["vlm_text"]),
                source_dataset=dataset_raw,
                subject_id=subject,
                video_id=video_id,
                age_months=int(age_months),
                fps=float(sig.get("fps", 0.0) or 0.0),  # source frame rate; enables fps-normalization at load (cross-dataset)
            )
            saved += 1
        except Exception as exc:
            failed += 1
            print(f"[WARN] Failed: {video_path} -> {exc}")

    print("\nExtraction complete")
    print(f"  Method: {method}")
    print(f"  Saved: {saved}")
    print(f"  Skipped existing: {skipped_existing}")
    print(f"  Skipped missing GT: {skipped_missing_gt}")
    if fell_back:
        print(f"  Fell back to controller (non-video / unreadable): {fell_back}")
    print(f"  Failed: {failed}")
    print(f"  Output dir: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe benchmark signal extraction")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to split manifest JSON (for example data/organized/pedrr_indist_train.json)",
    )
    parser.add_argument(
        "--output",
        default="data/semi_sl/train_signals",
        help="Output directory for extracted .npz files",
    )
    parser.add_argument(
        "--method",
        choices=["improved", "controller"],
        default="improved",
        help="improved = chest-ROI vertical-flow + band-pass intensity (default, validated); "
             "controller = legacy RespiratoryController.process_video path",
    )
    parser.add_argument(
        "--extract-variant",
        choices=["fixed", "gated_consensus"],
        default="fixed",
        help="improved-method extractor variant: fixed = validated pedrr_v2 (default); "
             "gated_consensus = Tier-1 chest+abdomen outlier-rejecting consensus (framing-robust)",
    )
    parser.add_argument(
        "--config",
        default="src/config/test_infer.yaml",
        help="Controller config YAML (only used by --method controller)",
    )
    parser.add_argument(
        "--checkpoint",
        default="data/models/VIRENet_Coarse2fine_Body_best.pth",
        help="Body detector checkpoint used by the controller (only used by --method controller)",
    )
    parser.add_argument(
        "--enable-vlm",
        action="store_true",
        help="Enable VLM assessment (only effective with --method controller)",
    )
    parser.add_argument(
        "--vlm-model-id",
        default="Salesforce/blip-image-captioning-base",
        help="HuggingFace VLM model id when --enable-vlm is used",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N manifest entries (0 = all)",
    )

    args = parser.parse_args()

    extract_benchmark_signals(
        manifest_path=args.manifest,
        output_dir=args.output,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        enable_vlm=args.enable_vlm,
        vlm_model_id=args.vlm_model_id,
        overwrite=args.overwrite,
        limit=args.limit,
        method=args.method,
        extract_variant=args.extract_variant,
    )


if __name__ == "__main__":
    main()
