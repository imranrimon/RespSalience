"""
Few-shot target-domain adaptation for the external (Pediatric-RR) gap.

Zero-shot, PRISM-Breath trained on AIR reaches ~15.6 BPM MAE on the held-out
Pediatric-RR-332 clinical cohort. This script measures how quickly that gap
closes when a few labeled target subjects are available — the realistic clinical
deployment scenario (a new site labels a handful of patients).

Protocol (leakage-safe): Pediatric-RR is 1 video per subject, so a random draw of
K subjects for adaptation is automatically subject-disjoint from the evaluation
set. For each K we fine-tune a copy of the base model on the K adaptation clips
and evaluate on the remaining subjects, repeated over several random draws
(mean +/- std). By default the signal encoders are frozen and only the
source/age domain adapter + prediction heads adapt — the principled low-variance
few-shot setup (the domain adapter's Pediatric-RR embedding is untrained at
zero-shot, so this is exactly the module built to specialize to a new source).

Usage:
    python scripts/adapt_external.py \
        --checkpoint outputs/paper_runs/threeseed_real/main_seed42/train/prism_breath_best.pt \
        --signal-dir data/paper_signals \
        --external-manifest data/organized/respiratory_benchmark_external_test.json \
        --k-shots 0 5 10 25 50 --repeats 3 --device cpu \
        --output-dir outputs/paper_runs/threeseed_real/adapt_external
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.prism_breath_model import (
    PrismBreathLoss,
    build_prism_breath_model_from_config,
    tensor_tachypnea_threshold,
)
from src.core.respiratory_benchmark_dataset import RespiratoryBenchmarkDataset


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_base(checkpoint: str, device):
    state = torch.load(checkpoint, map_location=device)
    cfg = state.get("config", {})
    model_config = cfg.get("model_config", cfg) if isinstance(cfg, dict) else {}
    seq_len = int((model_config or {}).get("seq_len", 300))
    model = build_prism_breath_model_from_config(model_config).to(device)
    model.load_state_dict(state["state_dict"], strict=False)
    return model, seq_len


def write_manifest(entries: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle)


@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    gts, preds, thrs, tris = [], [], [], []
    for batch in loader:
        out = model(
            depth=batch["depth"].to(device), flow=batch["flow"].to(device), audio=batch["audio"].to(device),
            age_months=batch["age_months"].to(device), modality_mask=batch["modality_mask"].to(device),
            dataset_ids=batch["dataset_id"].to(device),
        )
        gts.append(batch["gt_rr"].numpy())
        preds.append(out["rr_pred"].cpu().numpy())
        thrs.append(out["threshold_bpm"].cpu().numpy())
        tris.append(out["triage_code"].cpu().numpy())
    return (np.concatenate(gts), np.concatenate(preds), np.concatenate(thrs), np.concatenate(tris))


def _triage_from_qhat(pred, thr, qhat):
    """0=NORMAL_SAFE, 1=TACHYPNEA_ALERT, 2=UNCERTAIN_REVIEW using pred +/- qhat vs threshold."""
    lower, upper = pred - qhat, pred + qhat
    tri = np.full_like(pred, 2, dtype=int)
    tri[upper < thr] = 0
    tri[lower >= thr] = 1
    return tri


def _triage_metrics(gt, pred, thr, tri):
    tac = (gt >= thr).astype(int)
    pos, neg = tac == 1, tac == 0
    return {
        "sensitivity": float(np.mean(tri[pos] == 1)) if pos.any() else float("nan"),
        "specificity": float(np.mean(tri[neg] == 0)) if neg.any() else float("nan"),
        "review_rate": float(np.mean(tri == 2)),
    }


def evaluate(model, loader, device, recal_qhat=None) -> Dict[str, float]:
    gt, pred, thr, tri = _collect(model, loader, device)
    out = {
        "n": int(len(gt)),
        "mae": float(np.mean(np.abs(pred - gt))),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
    }
    # Default triage = the model's own (source-calibrated) triage code.
    default = _triage_metrics(gt, pred, thr, tri)
    out.update({"sensitivity": default["sensitivity"], "specificity": default["specificity"],
                "review_rate": default["review_rate"]})
    # Recalibrated triage using a target-domain qhat (few-shot conformal on the adaptation subjects).
    if recal_qhat is not None:
        recal = _triage_metrics(gt, pred, thr, _triage_from_qhat(pred, thr, recal_qhat))
        out.update({"recal_qhat": float(recal_qhat), "recal_sensitivity": recal["sensitivity"],
                    "recal_specificity": recal["specificity"], "recal_review_rate": recal["review_rate"]})
    return out


@torch.no_grad()
def target_qhat(model, loader, device, alpha: float) -> float:
    """Split-conformal qhat from target adaptation residuals (finite-sample corrected)."""
    gt, pred, _, _ = _collect(model, loader, device)
    residuals = np.abs(pred - gt)
    n = len(residuals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(residuals, level, method="higher"))


def finetune(model, loader, device, epochs: int, lr: float, weight_decay: float, freeze_encoder: bool):
    for name, param in model.named_parameters():
        param.requires_grad = not (freeze_encoder and name.startswith("branches."))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    criterion = PrismBreathLoss()
    model.train()
    for _ in range(epochs):
        for batch in loader:
            out = model(
                depth=batch["depth"].to(device), flow=batch["flow"].to(device), audio=batch["audio"].to(device),
                age_months=batch["age_months"].to(device), modality_mask=batch["modality_mask"].to(device),
                dataset_ids=batch["dataset_id"].to(device),
            )
            loss = criterion(out, gt_rr=batch["gt_rr"].to(device), age_months=batch["age_months"].to(device))["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            optimizer.step()


def main() -> int:
    parser = argparse.ArgumentParser(description="Few-shot target adaptation on the external Pediatric-RR cohort.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--signal-dir", default="data/paper_signals")
    parser.add_argument("--external-manifest", default="data/organized/respiratory_benchmark_external_test.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k-shots", type=int, nargs="+", default=[0, 5, 10, 25, 50])
    parser.add_argument("--repeats", type=int, default=3, help="Random adaptation-set draws per K (K>0)")
    parser.add_argument("--adapt-epochs", type=int, default=25)
    parser.add_argument("--recal-alpha", type=float, default=0.2, help="Target-domain conformal miscoverage for triage recalibration")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--full-finetune", action="store_true", help="Also fine-tune the signal encoders (default: frozen)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_manifests"

    entries = json.load(open(args.external_manifest))
    print(f"[adapt] external cohort: {len(entries)} entries | freeze_encoder={not args.full_finetune}")

    _, seq_len = load_base(args.checkpoint, device)
    rows = []
    for k in args.k_shots:
        reps = 1 if k == 0 else args.repeats
        rep_metrics = []
        for r in range(reps):
            seed_all(args.seed + r)
            shuffled = list(entries)
            random.Random(args.seed + r).shuffle(shuffled)
            adapt_entries = shuffled[:k]
            eval_entries = shuffled[k:]

            eval_manifest = tmp_dir / f"k{k}_r{r}_eval.json"
            write_manifest(eval_entries, eval_manifest)
            eval_ds = RespiratoryBenchmarkDataset(args.signal_dir, str(eval_manifest), seq_len=seq_len)
            eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

            model, _ = load_base(args.checkpoint, device)
            recal_qhat = None
            if k > 0:
                adapt_manifest = tmp_dir / f"k{k}_r{r}_adapt.json"
                write_manifest(adapt_entries, adapt_manifest)
                adapt_ds = RespiratoryBenchmarkDataset(args.signal_dir, str(adapt_manifest), seq_len=seq_len)
                adapt_loader = DataLoader(adapt_ds, batch_size=min(args.batch_size, max(1, k)), shuffle=True)
                finetune(model, adapt_loader, device, args.adapt_epochs, args.lr, args.weight_decay,
                         freeze_encoder=not args.full_finetune)
                # Target-domain triage recalibration: conformal qhat on the same K adaptation subjects.
                recal_qhat = target_qhat(model, DataLoader(adapt_ds, batch_size=args.batch_size), device, args.recal_alpha)
            m = evaluate(model, eval_loader, device, recal_qhat=recal_qhat)
            rep_metrics.append(m)

        def _agg(key):
            vals = [x.get(key, float("nan")) for x in rep_metrics]
            return float(np.nanmean(vals))
        mae = np.array([x["mae"] for x in rep_metrics])
        row = {
            "k_shots": k, "repeats": reps, "eval_n": rep_metrics[0]["n"],
            "mae_mean": float(mae.mean()), "mae_std": float(mae.std()),
            "rmse_mean": _agg("rmse"),
            "sensitivity_mean": _agg("sensitivity"), "review_rate_mean": _agg("review_rate"),
            "recal_sensitivity_mean": _agg("recal_sensitivity"),
            "recal_specificity_mean": _agg("recal_specificity"),
            "recal_review_rate_mean": _agg("recal_review_rate"),
        }
        rows.append(row)
        print(f"  K={k:3d} (x{reps}): MAE={row['mae_mean']:6.2f}+/-{row['mae_std']:.2f}  "
              f"default sens={row['sensitivity_mean']:.2f} | recal sens={row['recal_sensitivity_mean']:.2f} "
              f"spec={row['recal_specificity_mean']:.2f} review={row['recal_review_rate_mean']:.2f}")

    import csv
    with open(out_dir / "few_shot_adaptation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    zero = next((r for r in rows if r["k_shots"] == 0), None)
    best = min(rows, key=lambda r: r["mae_mean"])
    if zero:
        print(f"\n[adapt] zero-shot MAE {zero['mae_mean']:.2f} -> best {best['mae_mean']:.2f} "
              f"at K={best['k_shots']} ({100*(zero['mae_mean']-best['mae_mean'])/zero['mae_mean']:.0f}% reduction)")
    print(f"[OK] wrote {out_dir/'few_shot_adaptation.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
