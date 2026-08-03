"""Evaluate PRISM-Breath and export explainable respiratory-triage predictions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.selective_triage import DisagreementConditionedConformal
from src.core.respiratory_benchmark_dataset import RespiratoryBenchmarkDataset
from src.core.prism_breath_model import PrismBreathModel, build_prism_breath_model_from_config


def _to_float(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def compute_summary(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    gt = np.asarray([row["gt_rr_bpm"] for row in rows], dtype=np.float32)
    pred = np.asarray([row["pred_rr_bpm"] for row in rows], dtype=np.float32)
    triage = np.asarray([row["triage_code"] for row in rows], dtype=np.int64)
    tachy = np.asarray([row["gt_tachy"] for row in rows], dtype=np.int64)
    lower = np.asarray([row["interval_lower_bpm"] for row in rows], dtype=np.float32)
    upper = np.asarray([row["interval_upper_bpm"] for row in rows], dtype=np.float32)
    qhat = np.asarray([row["qhat"] for row in rows], dtype=np.float32)

    positives = tachy == 1
    negatives = tachy == 0
    return {
        "n": int(len(rows)),
        "mae": float(np.mean(np.abs(pred - gt))),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "sensitivity": float(np.mean(triage[positives] == 1)) if positives.any() else float("nan"),
        "specificity": float(np.mean(triage[negatives] == 0)) if negatives.any() else float("nan"),
        "review_rate": float(np.mean(triage == 2)),
        "coverage": float(np.mean((gt >= lower) & (gt <= upper))),
        "mean_qhat": float(np.mean(qhat)),
    }


def save_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PRISM-Breath and export explainable predictions.")
    parser.add_argument("--signal-dir", default="data/processed_signals")
    parser.add_argument("--manifest", default="data/organized/respiratory_benchmark_test.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/prism_breath_eval")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--calibration-csv", default="", help="Optional calibration CSV from scripts/calibrate_conformal.py")
    parser.add_argument("--condition-on-disagreement", action="store_true")
    parser.add_argument(
        "--disagreement-bins",
        default="0.2,0.4",
        help="Two comma-separated thresholds used when selecting disagreement-conditioned qhat bins.",
    )
    parser.add_argument("--counterfactual-limit", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    model_config = config.get("model_config", config)
    model = build_prism_breath_model_from_config(model_config).to(device)
    # Tolerant load: checkpoints trained before the apnea/motion event heads were
    # added lack those parameters; strict=False lets them evaluate unchanged (the
    # event heads stay unused for RR/triage).
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)} keys (expected for pre-event-head checkpoints)")
    model.eval()
    model_name = config.get("model_name", model.model_name())

    dataset = RespiratoryBenchmarkDataset(
        signal_dir=args.signal_dir,
        manifest_path=args.manifest,
        seq_len=int(model_config.get("seq_len", config.get("seq_len", 300))),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    calibrator = None
    if args.calibration_csv:
        bins = tuple(sorted(float(x.strip()) for x in args.disagreement_bins.split(",") if x.strip()))
        if len(bins) < 2:
            raise SystemExit("--disagreement-bins requires two comma-separated values, e.g. 0.8,0.9")
        calibrator = DisagreementConditionedConformal(
            calibration_csv=args.calibration_csv,
            default_qhat=6.0,
            disagreement_bins=(bins[0], bins[1]),
        )

    all_rows: List[Dict] = []
    cf_budget = max(0, int(args.counterfactual_limit))

    with torch.no_grad():
        for batch in loader:
            depth = batch["depth"].to(device)
            flow = batch["flow"].to(device)
            audio = batch["audio"].to(device)
            age = batch["age_months"].to(device)
            dataset_ids = batch["dataset_id"].to(device)
            modality_mask = batch["modality_mask"].to(device)

            outputs = model(
                depth=depth,
                flow=flow,
                audio=audio,
                age_months=age,
                modality_mask=modality_mask,
                dataset_ids=dataset_ids,
            )
            proto_sim = outputs["prototype_similarity"].detach().cpu()
            proto_vals, proto_idx = torch.topk(proto_sim, k=min(2, proto_sim.shape[-1]), dim=-1)

            qhat = outputs["qhat"].detach().cpu().numpy()
            pred_rr = outputs["rr_pred"].detach().cpu().numpy()
            rr_std = (np.exp(0.5 * outputs["rr_logvar"].detach().cpu().numpy())
                      if "rr_logvar" in outputs else np.zeros_like(pred_rr))  # heteroscedastic aleatoric std
            disagreement = outputs["disagreement_score"].detach().cpu().numpy()
            threshold = outputs["threshold_bpm"].detach().cpu().numpy()
            triage_code = outputs["triage_code"].detach().cpu().numpy()
            modality_pred = outputs["modality_pred_tensor"].detach().cpu().numpy()
            modality_weight = outputs["fusion_weights_tensor"].detach().cpu().numpy()
            modality_quality = outputs["quality_tensor"].detach().cpu().numpy()

            for idx in range(depth.shape[0]):
                final_qhat = float(qhat[idx])
                qhat_source = "model"
                if calibrator is not None:
                    calibrated_qhat, meta = calibrator.select_qhat(
                        model_name=model_name,
                        paradigm="respiratory_benchmark",
                        source=batch["dataset"][idx],
                        disagreement_score=float(disagreement[idx]),
                        condition_on_disagreement=args.condition_on_disagreement,
                    )
                    final_qhat = max(final_qhat, float(calibrated_qhat))
                    qhat_source = meta.get("qhat_source", "calibration")

                lower = float(pred_rr[idx] - final_qhat)
                upper = float(pred_rr[idx] + final_qhat)
                triage = 2
                if upper < threshold[idx]:
                    triage = 0
                elif lower >= threshold[idx]:
                    triage = 1

                row = {
                    "name": model_name,
                    "paradigm": "respiratory_benchmark",
                    "source": batch["dataset"][idx],
                    "subject": batch["subject_id"][idx],
                    "dataset": batch["dataset"][idx],
                    "subject_id": batch["subject_id"][idx],
                    "video_id": batch["video_id"][idx],
                    "signal_path": batch["signal_path"][idx],
                    "pred_bpm": float(pred_rr[idx]),
                    "gt_bpm": _to_float(batch["gt_rr"][idx].cpu().numpy()),
                    "gt_rr_bpm": _to_float(batch["gt_rr"][idx].cpu().numpy()),
                    "pred_rr_bpm": float(pred_rr[idx]),
                    "rr_std": float(rr_std[idx]),
                    "threshold_bpm": float(threshold[idx]),
                    "age_months": _to_float(batch["age_months"][idx].cpu().numpy()),
                    "qhat": float(final_qhat),
                    "qhat_source": qhat_source,
                    "interval_lower_bpm": lower,
                    "interval_upper_bpm": upper,
                    "triage_code": int(triage),
                    "triage_status": PrismBreathModel.triage_name(int(triage)),
                    "model_triage_code": int(triage_code[idx]),
                    "gt_tachy": int(_to_float(batch["gt_rr"][idx].cpu().numpy()) >= float(threshold[idx])),
                    "tachy_prob": float(outputs["tachy_prob"][idx].detach().cpu()),
                    "review_prob": float(outputs["review_prob"][idx].detach().cpu()),
                    "apnea_prob": float(outputs["apnea_prob"][idx].detach().cpu()),
                    "motion_prob": float(outputs["motion_prob"][idx].detach().cpu()),
                    "disagreement_score": float(disagreement[idx]),
                    "depth_rr_bpm": float(modality_pred[idx, 0]),
                    "flow_rr_bpm": float(modality_pred[idx, 1]),
                    "audio_rr_bpm": float(modality_pred[idx, 2]),
                    "depth_weight": float(modality_weight[idx, 0]),
                    "flow_weight": float(modality_weight[idx, 1]),
                    "audio_weight": float(modality_weight[idx, 2]),
                    "depth_quality": float(modality_quality[idx, 0]),
                    "flow_quality": float(modality_quality[idx, 1]),
                    "audio_quality": float(modality_quality[idx, 2]),
                    "prototype_1_idx": int(proto_idx[idx, 0]),
                    "prototype_1_similarity": float(proto_vals[idx, 0]),
                    "prototype_2_idx": int(proto_idx[idx, 1]) if proto_idx.shape[1] > 1 else -1,
                    "prototype_2_similarity": float(proto_vals[idx, 1]) if proto_vals.shape[1] > 1 else float("nan"),
                    "clean_code_idx": int(outputs["clean_code_idx"][idx].detach().cpu()),
                    "clean_code_weight": float(outputs["clean_code_weight"][idx].detach().cpu()),
                    "clean_code_gate": float(outputs["clean_code_gate"][idx].detach().cpu()),
                }

                if cf_budget > 0:
                    cf = model.counterfactual_report(
                        depth=depth[idx : idx + 1],
                        flow=flow[idx : idx + 1],
                        audio=audio[idx : idx + 1],
                        age_months=age[idx : idx + 1],
                        modality_mask=modality_mask[idx : idx + 1],
                        dataset_ids=dataset_ids[idx : idx + 1],
                    )
                    for name in ("depth", "flow", "audio"):
                        row[f"cf_drop_{name}_delta_rr"] = float(cf[name]["delta_rr"][0].detach().cpu())
                        row[f"cf_drop_{name}_triage"] = PrismBreathModel.triage_name(
                            int(cf[name]["triage_code"][0].detach().cpu())
                        )
                    cf_budget -= 1
                else:
                    for name in ("depth", "flow", "audio"):
                        row[f"cf_drop_{name}_delta_rr"] = float("nan")
                        row[f"cf_drop_{name}_triage"] = ""

                all_rows.append(row)

    summary = compute_summary(all_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "prism_breath_predictions.csv", all_rows)
    with open(output_dir / "prism_breath_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[PRISM-Breath Evaluation]")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  predictions: {output_dir / 'prism_breath_predictions.csv'}")
    print(f"  summary: {output_dir / 'prism_breath_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
