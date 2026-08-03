"""
Hyperparameter Sweep for Respiratory Rate Estimation Models.

This script systematically evaluates combinations of:
  - d_model: Transformer embedding dimension
  - num_layers: Transformer depth
  - learning_rate: AdamW step size
  - denoiser_type: Which DL denoiser to pre-apply to signals
  - recon_weight: Weighting of the SSL reconstruction loss

Results are saved to data/ablation_results/hp_sweep_results.csv.
The best configuration is printed at the end and can be passed directly
to train_vlm_fusion.py for the final training run.

Usage:
    python scripts/hp_sweep.py                     # Full sweep
    python scripts/hp_sweep.py --quick             # Reduced 4-combo quick search
    python scripts/hp_sweep.py --epochs 10         # Fewer epochs per trial
"""
import os
import sys
import csv
import json
import argparse
import itertools
import time
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.train_vlm_fusion import VLMSignalDataset, apply_modality_dropout
from src.core.multimodal_fusion_models import TemporalFusionTransformer


def _ensure_signal_2d(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor shape to (B, L) for 1D fusion models.

    Some extracted .npz files may store signals with extra channel dimensions,
    which can become 3D in a batch (B, C, L). The fusion model expects (B, L)
    before it adds its own channel dimension.
    """
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        # Prefer squeezing singleton channel; otherwise average channels.
        if x.shape[1] == 1:
            return x.squeeze(1)
        return x.mean(dim=1)
    if x.dim() == 1:
        return x.unsqueeze(0)
    # Fallback for unexpected ranks: flatten all non-batch dims.
    return x.view(x.shape[0], -1)

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH SPACE
# ─────────────────────────────────────────────────────────────────────────────
FULL_GRID = {
    "d_model":       [64, 128],
    "num_layers":    [2, 4, 6],
    "lr":            [5e-5, 1e-4, 3e-4],
    "recon_weight":  [0.05, 0.1, 0.5],
}

# Quick mode: pre-selected high-ROI subset (8 combos)
QUICK_GRID = {
    "d_model":       [64, 128],
    "num_layers":    [4],
    "lr":            [1e-4, 3e-4],
    "recon_weight":  [0.05, 0.1],
}


def run_trial(config: dict, data_dir: str, epochs: int, batch_size: int, device: str) -> dict:
    """
    Run one training trial with the given hyperparameter config.
    Returns a dict with config + {train_mae, val_mae, best_epoch, duration_s}.
    """
    d_model      = config["d_model"]
    num_layers   = config["num_layers"]
    lr           = config["lr"]
    recon_weight = config["recon_weight"]
    vlm_dim      = 2048

    full_dataset = VLMSignalDataset(data_dir, mode="finetune")
    if len(full_dataset) == 0:
        return {**config, "val_mae": float("inf"), "train_mae": float("inf"),
                "best_epoch": 0, "duration_s": 0, "error": "no_data"}

    # 90/10 split (seeded for reproducibility across trials)
    val_size   = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Model — we only support nhead divisors of d_model cleanly
    nhead = 8 if d_model >= 128 else 4
    model = TemporalFusionTransformer(d_model=d_model, nhead=nhead,
                                      num_layers=num_layers, vlm_dim=vlm_dim).to(device)
    vlm_proj = nn.Sequential(nn.Linear(1, 256), nn.ReLU(), nn.Linear(256, vlm_dim)).to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(vlm_proj.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val_mae   = float("inf")
    best_train_mae = float("inf")
    best_epoch     = 0
    patience_cnt   = 0
    PATIENCE       = 7

    t0 = time.time()

    for epoch in range(epochs):
        # ── Train ──
        model.train(); vlm_proj.train()
        train_mae_sum = 0.0
        for batch in train_loader:
            depth     = _ensure_signal_2d(batch["depth"].to(device))
            flow      = _ensure_signal_2d(batch["flow"].to(device))
            audio     = _ensure_signal_2d(batch["audio"].to(device))
            gt        = batch["gt"].to(device).unsqueeze(1)
            vlm_label = batch["vlm_label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            d_drop, f_drop, a_drop = apply_modality_dropout(depth, flow, audio, drop_prob=0.2)
            vlm_emb = vlm_proj(vlm_label)

            pred       = model(d_drop, f_drop, audio_signal=a_drop, vlm_emb=vlm_emb)
            recon_mse  = model.compute_anomaly_score(depth, flow, audio_signal=audio, vlm_emb=vlm_emb).mean()
            loss       = criterion(pred, gt) + recon_weight * recon_mse

            loss.backward()
            optimizer.step()
            train_mae_sum += torch.mean(torch.abs(pred.detach() - gt)).item()

        avg_train_mae = train_mae_sum / len(train_loader)

        # ── Validate ──
        model.eval(); vlm_proj.eval()
        val_mae_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                depth     = _ensure_signal_2d(batch["depth"].to(device))
                flow      = _ensure_signal_2d(batch["flow"].to(device))
                audio     = _ensure_signal_2d(batch["audio"].to(device))
                gt        = batch["gt"].to(device).unsqueeze(1)
                vlm_label = batch["vlm_label"].to(device).unsqueeze(1)
                vlm_emb   = vlm_proj(vlm_label)
                pred      = model(depth, flow, audio_signal=audio, vlm_emb=vlm_emb)
                val_mae_sum += torch.mean(torch.abs(pred - gt)).item()
        avg_val_mae = val_mae_sum / len(val_loader)
        scheduler.step(avg_val_mae)

        if avg_val_mae < best_val_mae:
            best_val_mae   = avg_val_mae
            best_train_mae = avg_train_mae
            best_epoch     = epoch + 1
            patience_cnt   = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break   # Early stopping

    return {
        **config,
        "nhead":       nhead,
        "train_mae":   round(best_train_mae, 3),
        "val_mae":     round(best_val_mae, 3),
        "best_epoch":  best_epoch,
        "duration_s":  round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for fusion transformer")
    parser.add_argument("--data_dir", type=str,  default="data/ablation_results/signals")
    parser.add_argument("--epochs",   type=int,  default=20, help="Max epochs per trial")
    parser.add_argument("--batch",    type=int,  default=32)
    parser.add_argument("--quick",    action="store_true", help="Use reduced quick grid")
    parser.add_argument("--out",      type=str,  default="data/ablation_results/hp_sweep_results.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid   = QUICK_GRID if args.quick else FULL_GRID

    # Build all combinations
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    configs = [dict(zip(keys, c)) for c in combos]

    print(f"{'='*60}")
    print(f" HYPERPARAMETER SWEEP — {len(configs)} configs × {args.epochs} max epochs")
    print(f" Device: {device} | Data: {args.data_dir}")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = []

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Config: {cfg}")
        try:
            result = run_trial(cfg, args.data_dir, args.epochs, args.batch, device)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {**cfg, "val_mae": float("inf"), "train_mae": float("inf"),
                      "best_epoch": 0, "duration_s": 0, "error": str(e)}
        results.append(result)
        print(f"  -> Val MAE={result['val_mae']:.3f} BPM | Train MAE={result['train_mae']:.3f} | "
              f"Best epoch={result['best_epoch']} | Time={result['duration_s']}s")

        # Save incrementally (crash-safe)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(" SWEEP COMPLETE — TOP 5 CONFIGS BY VAL MAE")
    print(f"{'='*60}")
    sorted_results = sorted(results, key=lambda x: x.get("val_mae", float("inf")))
    for rank, r in enumerate(sorted_results[:5], 1):
        print(f"  #{rank}: Val MAE={r['val_mae']:.3f} BPM | {r}")

    best = sorted_results[0]
    print(f"\n{'='*60}")
    print(" RECOMMENDED TRAINING COMMAND")
    print(f"{'='*60}")
    print(f"\npython scripts/train_vlm_fusion.py --stage both --epochs 50 --ft_epochs 30 \\\n"
          f"    # Best config: d_model={best.get('d_model')}, num_layers={best.get('num_layers')}, "
          f"lr={best.get('lr')}, recon_weight={best.get('recon_weight')}")
    print(f"\nFull results saved to: {args.out}")

    # Also save JSON for programmatic use
    json_out = args.out.replace(".csv", ".json")
    with open(json_out, "w") as f:
        json.dump({"best": best, "all": sorted_results}, f, indent=2)
    print(f"JSON results saved to: {json_out}")


if __name__ == "__main__":
    main()
