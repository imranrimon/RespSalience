import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import math
import sys
import os
import random
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.multimodal_fusion_models import TemporalFusionTransformer
from src.core.respiratory_ssl_datasets import RespiratorySignalDataset


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _evaluate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for batch in loader:
            depth = batch['depth'].to(device)
            flow = batch['flow'].to(device)
            gt = batch['gt'].to(device).unsqueeze(1)
            pred = model(depth, flow)
            loss = criterion(pred, gt)
            mae = torch.mean(torch.abs(pred - gt))
            total_loss += float(loss.item())
            total_mae += float(mae.item())

    return {
        "loss": total_loss / max(1, len(loader)),
        "mae": total_mae / max(1, len(loader)),
    }


def train_finetune(
    data_dir,
    pretrained_path,
    output_path,
    epochs=30,
    batch_size=32,
    lr=1e-4,
    seed=42,
    train_manifest=None,
    val_manifest=None,
    seq_len=300,
    d_model=64,
    weight_decay=1e-4,
):
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset
    print(f"Loading dataset from {data_dir}...")
    if train_manifest:
        train_dataset = RespiratorySignalDataset(
            data_dir,
            split='train',
            mode='finetune',
            seq_len=seq_len,
            manifest_path=train_manifest,
        )
    else:
        train_dataset = RespiratorySignalDataset(data_dir, split='train', mode='finetune', seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = None
    if val_manifest:
        val_dataset = RespiratorySignalDataset(
            data_dir,
            split='val',
            mode='finetune',
            seq_len=seq_len,
            manifest_path=val_manifest,
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    if len(train_dataset) == 0:
        print("Error: No data found.")
        return

    # 2. Model
    model = TemporalFusionTransformer(seq_len=seq_len, d_model=d_model, nhead=4, num_layers=4)
    
    # Load Pretrained Backbone
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading Foundation Model from {pretrained_path}")
        state = torch.load(pretrained_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
    else:
        print("Warning: Foundation model not found. Training from scratch.")
        
    model.to(device)
    
    # 3. Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) # Lower LR for fine-tuning
    best_score = float("inf")
    history = []
    
    print(f"Starting Fine-tuning for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_mae = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            depth = batch['depth'].to(device)
            flow = batch['flow'].to(device)
            gt = batch['gt'].to(device).unsqueeze(1) # (B, 1)
            
            optimizer.zero_grad()
            
            # Forward (Predict BPM)
            pred = model(depth, flow)
            
            loss = criterion(pred, gt)
            mae = torch.mean(torch.abs(pred - gt))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            pbar.set_postfix({'mae': f"{mae.item():.2f}"})
            
        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_train_mae = total_mae / max(1, len(train_loader))
        val_metrics = _evaluate(model, val_loader, device) if val_loader is not None else None
        selection_score = val_metrics["mae"] if val_metrics is not None else avg_train_mae
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "train_mae": float(avg_train_mae),
                "val_loss": None if val_metrics is None else float(val_metrics["loss"]),
                "val_mae": None if val_metrics is None else float(val_metrics["mae"]),
                "selection_score": float(selection_score),
            }
        )
        summary = f"Epoch {epoch+1} Done. Train MAE: {avg_train_mae:.2f} BPM"
        if val_metrics is not None:
            summary += f" | Val MAE: {val_metrics['mae']:.2f} BPM"
        print(summary)

        if selection_score < best_score:
            best_score = float(selection_score)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"[OK] Saved best checkpoint to {output_path}")
        
    # Save
    output_path = Path(output_path)
    history_path = output_path.with_name(output_path.stem + "_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": seed,
                "pretrained_path": pretrained_path,
                "output_path": str(output_path),
                "best_score": float(best_score),
                "history": history,
            },
            handle,
            indent=2,
        )
    print(f"History saved to {history_path}")
    return str(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ablation_results/signals")
    parser.add_argument("--pretrained", type=str, default="", help="Optional pretrained checkpoint for initialization")
    parser.add_argument("--output", type=str, default="data/models/finetuned_model.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--train-manifest", type=str, default="", help="Optional canonical train manifest")
    parser.add_argument("--val-manifest", type=str, default="", help="Optional canonical val manifest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    train_finetune(
        args.data_dir,
        args.pretrained,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        train_manifest=args.train_manifest or None,
        val_manifest=args.val_manifest or None,
        seq_len=args.seq_len,
        d_model=args.d_model,
        weight_decay=args.weight_decay,
    )
