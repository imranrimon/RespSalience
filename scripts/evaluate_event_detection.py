"""
Evaluate PRISM-Breath's apnea / motion detection heads.

No dataset carries real apnea or motion labels, so detection is measured against
*synthetically injected* events (the same augmentations used at training time):
for every clip we build a clean version (negative) and versions with an injected
apnea pause and an injected motion burst (positives), run the model, and score
the apnea/motion head probabilities with ROC-AUC and average precision.

This is an honest, clearly-labeled synthetic-event evaluation — it validates that
the heads respond to the failure modes they were trained to flag, which is the
evidence the M1 (motion/apnea robustness) claim needs.

Usage:
    # single checkpoint
    python scripts/evaluate_event_detection.py --checkpoint <ckpt.pt> \
        --signal-dir data/paper_signals \
        --manifest data/organized/respiratory_benchmark_test.json \
        --output-dir outputs/event_eval

    # aggregate across the 3-seed run (main_seed*/train/prism_breath_best.pt)
    python scripts/evaluate_event_detection.py --run-root outputs/paper_runs/threeseed_real \
        --signal-dir data/paper_signals \
        --manifest data/organized/respiratory_benchmark_test.json \
        --output-dir outputs/paper_runs/threeseed_real/event_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.prism_breath_model import build_prism_breath_model_from_config
from src.core.respiratory_benchmark_dataset import RespiratoryBenchmarkDataset
from src.utils.augmentations import simulate_apnea, simulate_motion_artifact

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    HAS_SKLEARN = False


def _seed_from_path(path: Path) -> int:
    match = re.search(r"seed[_-]?(\d+)", str(path))
    return int(match.group(1)) if match else -1


def resolve_checkpoints(checkpoint: str, run_root: str, checkpoints: Optional[List[str]] = None) -> List[Path]:
    paths: List[Path] = []
    for raw in ([checkpoint] if checkpoint else []) + list(checkpoints or []):
        p = Path(raw)
        if p.exists():
            paths.append(p)
        else:
            print(f"[WARN] missing checkpoint: {p}")
    if run_root:
        paths.extend(sorted(Path(run_root).glob("**/train/prism_breath_best.pt")))
    # de-dup preserving order
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _renorm(sig: np.ndarray) -> np.ndarray:
    return RespiratoryBenchmarkDataset._renormalize(sig)


def build_event_variants(sample: Dict, rng: np.random.Generator) -> Dict[str, Dict[str, np.ndarray]]:
    """Return clean/apnea/motion signal triplets for one clip."""
    depth = sample["depth"].numpy()
    flow = sample["flow"].numpy()
    audio = sample["audio"].numpy()

    ap_d, _, _ = simulate_apnea(depth, rng=rng)
    ap_f, _, _ = simulate_apnea(flow, rng=rng)
    ap_a, _, _ = simulate_apnea(audio, rng=rng)
    mo_d, _, _ = simulate_motion_artifact(depth, rng=rng)
    mo_f, _, _ = simulate_motion_artifact(flow, rng=rng)
    mo_a, _, _ = simulate_motion_artifact(audio, rng=rng)

    return {
        "clean": {"depth": depth, "flow": flow, "audio": audio},
        "apnea": {"depth": _renorm(ap_d), "flow": _renorm(ap_f), "audio": _renorm(ap_a)},
        "motion": {"depth": _renorm(mo_d), "flow": _renorm(mo_f), "audio": _renorm(mo_a)},
    }


@torch.no_grad()
def score_condition(model, records: List[Dict], condition: str, device, batch_size: int) -> Dict[str, np.ndarray]:
    apnea_probs, motion_probs = [], []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        depth = torch.tensor(np.stack([r[condition]["depth"] for r in chunk]), dtype=torch.float32, device=device)
        flow = torch.tensor(np.stack([r[condition]["flow"] for r in chunk]), dtype=torch.float32, device=device)
        audio = torch.tensor(np.stack([r[condition]["audio"] for r in chunk]), dtype=torch.float32, device=device)
        age = torch.tensor(np.stack([r["age_months"] for r in chunk]), dtype=torch.float32, device=device)
        mask = torch.tensor(np.stack([r["modality_mask"] for r in chunk]), dtype=torch.float32, device=device)
        ds_id = torch.tensor(np.stack([r["dataset_id"] for r in chunk]), dtype=torch.long, device=device)
        out = model(depth=depth, flow=flow, audio=audio, age_months=age, modality_mask=mask, dataset_ids=ds_id)
        apnea_probs.append(out["apnea_prob"].detach().cpu().numpy())
        motion_probs.append(out["motion_prob"].detach().cpu().numpy())
    return {"apnea_prob": np.concatenate(apnea_probs), "motion_prob": np.concatenate(motion_probs)}


def _auc_ap(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Dict[str, float]:
    y = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    s = np.concatenate([pos_scores, neg_scores])
    if not HAS_SKLEARN or len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "ap": float("nan")}
    return {"auroc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}


def evaluate_checkpoint(ckpt_path: Path, signal_dir: str, manifest: str, device, batch_size: int, seed: int) -> Dict:
    state = torch.load(ckpt_path, map_location=device)
    cfg = state.get("config", {})
    model_config = cfg.get("model_config", cfg) if isinstance(cfg, dict) else {}
    seq_len = int((model_config or {}).get("seq_len", 300))
    model = build_prism_breath_model_from_config(model_config).to(device)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    dataset = RespiratoryBenchmarkDataset(signal_dir=signal_dir, manifest_path=manifest, seq_len=seq_len)
    records = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        rng = np.random.default_rng(seed * 100003 + idx)
        variants = build_event_variants(sample, rng)
        records.append({
            **variants,
            "age_months": np.float32(sample["age_months"].item()),
            "modality_mask": sample["modality_mask"].numpy(),
            "dataset_id": np.int64(sample["dataset_id"].item()),
        })

    clean = score_condition(model, records, "clean", device, batch_size)
    apnea = score_condition(model, records, "apnea", device, batch_size)
    motion = score_condition(model, records, "motion", device, batch_size)

    apnea_metrics = _auc_ap(apnea["apnea_prob"], clean["apnea_prob"])
    motion_metrics = _auc_ap(motion["motion_prob"], clean["motion_prob"])
    return {
        "n_clips": len(records),
        "apnea_auroc": apnea_metrics["auroc"],
        "apnea_ap": apnea_metrics["ap"],
        "motion_auroc": motion_metrics["auroc"],
        "motion_ap": motion_metrics["ap"],
        "apnea_prob_clean_mean": float(np.mean(clean["apnea_prob"])),
        "apnea_prob_event_mean": float(np.mean(apnea["apnea_prob"])),
        "motion_prob_clean_mean": float(np.mean(clean["motion_prob"])),
        "motion_prob_event_mean": float(np.mean(motion["motion_prob"])),
    }


def _agg(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PRISM-Breath apnea/motion detection on synthetic events.")
    parser.add_argument("--checkpoint", default="", help="Single checkpoint .pt")
    parser.add_argument("--checkpoints", nargs="*", default=[], help="Explicit list of checkpoint .pt files")
    parser.add_argument("--run-root", default="", help="Discover **/train/prism_breath_best.pt under this root")
    parser.add_argument("--signal-dir", default="data/paper_signals")
    parser.add_argument("--manifest", default="data/organized/respiratory_benchmark_test.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducible event injection")
    args = parser.parse_args()

    if not HAS_SKLEARN:
        print("[WARN] scikit-learn unavailable; ROC-AUC/AP will be NaN.")

    device = torch.device(args.device)
    ckpts = resolve_checkpoints(args.checkpoint, args.run_root, args.checkpoints)
    if not ckpts:
        raise SystemExit("No checkpoints found. Provide --checkpoint or --run-root.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_ckpt = []
    for ckpt in ckpts:
        seed = _seed_from_path(ckpt)
        res = evaluate_checkpoint(ckpt, args.signal_dir, args.manifest, device, args.batch_size, args.seed)
        res["checkpoint"] = str(ckpt)
        res["seed"] = seed
        per_ckpt.append(res)
        print(f"[{ckpt.parent.parent.name}] apnea AUROC={res['apnea_auroc']:.3f} AP={res['apnea_ap']:.3f} | "
              f"motion AUROC={res['motion_auroc']:.3f} AP={res['motion_ap']:.3f} "
              f"(apnea prob clean->event {res['apnea_prob_clean_mean']:.2f}->{res['apnea_prob_event_mean']:.2f}, "
              f"motion {res['motion_prob_clean_mean']:.2f}->{res['motion_prob_event_mean']:.2f})")

    # per-checkpoint CSV
    with open(out_dir / "event_detection_per_seed.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_ckpt[0].keys()))
        writer.writeheader()
        writer.writerows(per_ckpt)

    summary = {
        "n_checkpoints": len(per_ckpt),
        "manifest": args.manifest,
        "apnea_auroc": _agg([r["apnea_auroc"] for r in per_ckpt]),
        "apnea_ap": _agg([r["apnea_ap"] for r in per_ckpt]),
        "motion_auroc": _agg([r["motion_auroc"] for r in per_ckpt]),
        "motion_ap": _agg([r["motion_ap"] for r in per_ckpt]),
    }
    with open(out_dir / "event_detection_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[event-detection] apnea AUROC {summary['apnea_auroc']['mean']:.3f}+/-{summary['apnea_auroc']['std']:.3f} | "
          f"motion AUROC {summary['motion_auroc']['mean']:.3f}+/-{summary['motion_auroc']['std']:.3f} "
          f"over {len(per_ckpt)} checkpoint(s)")
    print(f"[OK] wrote {out_dir/'event_detection_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
