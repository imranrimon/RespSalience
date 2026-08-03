import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import sys
import random
import json
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.multimodal_fusion_models import TemporalFusionTransformer
from src.core.respiratory_ssl_datasets import RespiratorySignalDataset, MaskedRespiratoryDataset
from src.core.losses import SupervisedBarlowTwinsLoss


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _make_ssl_dataset(method, data_dir, split, manifest_path, seq_len):
    common = {
        "data_dir": data_dir,
        "split": split,
        "seq_len": seq_len,
        "manifest_path": manifest_path,
    }
    if method in ['msm', 'sae']:
        return MaskedRespiratoryDataset(mask_ratio=0.5, **common)
    return RespiratorySignalDataset(mode='pretrain', **common)


def _run_epoch(model, loader, method, optimizer, device, barlow_loss, mse_loss):
    total_loss = 0.0
    is_train = optimizer is not None
    model.train(mode=is_train)
    phase = "train" if is_train else "val"
    pbar = tqdm(loader, desc=f"{phase.title()} batches", leave=False)

    for batch in pbar:
        if optimizer is not None:
            optimizer.zero_grad()

        if method == 'msm':
            masked_depth = batch['masked_depth'].to(device)
            masked_flow = batch['masked_flow'].to(device)
            tgt_depth = batch['target_depth'].to(device)
            tgt_flow = batch['target_flow'].to(device)
            recon = model(masked_depth, masked_flow, mode='recon')
            recon_depth = recon[:, 0, :]
            recon_flow = recon[:, 1, :]
            loss = mse_loss(recon_depth, tgt_depth) + mse_loss(recon_flow, tgt_flow)
        elif method == 'sae':
            masked_depth = batch['masked_depth'].to(device)
            masked_flow = batch['masked_flow'].to(device)
            tgt_depth = batch['target_depth'].to(device)
            tgt_flow = batch['target_flow'].to(device)
            z = model(masked_depth, masked_flow, mode='emb')
            recon = model(masked_depth, masked_flow, mode='recon')
            recon_depth = recon[:, 0, :]
            recon_flow = recon[:, 1, :]
            rec_loss = mse_loss(recon_depth, tgt_depth) + mse_loss(recon_flow, tgt_flow)
            sparsity_loss = torch.mean(torch.abs(z))
            loss = rec_loss + 0.01 * sparsity_loss
        else:
            v1_depth = batch['v1_depth'].to(device)
            v1_flow = batch['v1_flow'].to(device)
            v2_depth = batch['v2_depth'].to(device)
            v2_flow = batch['v2_flow'].to(device)
            z1 = model(v1_depth, v1_flow, return_embedding=True)
            z2 = model(v2_depth, v2_flow, return_embedding=True)
            loss = barlow_loss(z1, z2, labels=None)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / max(1, len(loader))


def train_ssl_benchmark(
    method,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    data_dir="data/ablation_results/signals",
    seed=42,
    train_manifest=None,
    val_manifest=None,
    output_path=None,
    seq_len=300,
    d_model=64,
):
    print(f"============================================================")
    print(f" SSL BENCHMARK: {method.upper()}")
    print(f"============================================================")
    set_global_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("data/models", exist_ok=True)
    
    # 1. Dataset Selection
    train_dataset = _make_ssl_dataset(method, data_dir, 'train', train_manifest, seq_len)
    if len(train_dataset) == 0:
        print("Error: No data found.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = None
    if val_manifest:
        val_dataset = _make_ssl_dataset(method, data_dir, 'val', val_manifest, seq_len)
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if method in ['msm', 'sae']:
        print("Loaded Masked Dataset (50% block masking).")
    else:
        print("Loaded Contrastive Dataset (Time warping).")
    
    # 2. Model
    model = TemporalFusionTransformer(num_layers=4, d_model=d_model, seq_len=seq_len) # Match Foundation Config
    model.to(device)
    
    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    barlow_loss = SupervisedBarlowTwinsLoss(lambda_coeff=0.005)
    mse_loss = nn.MSELoss()
    
    if not output_path:
        output_path = f"data/models/ssl_benchmark_{method}.pth"
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')
    history = []

    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, method, optimizer, device, barlow_loss, mse_loss)
        val_loss = float("nan")
        if val_loader is not None:
            with torch.no_grad():
                val_loss = _run_epoch(model, val_loader, method, None, device, barlow_loss, mse_loss)

        score = val_loss if val_loader is not None else train_loss
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": None if np.isnan(val_loss) else float(val_loss),
                "selection_score": float(score),
            }
        )
        print(
            f"Epoch {epoch+1} Done. "
            f"Train Loss: {train_loss:.4f}"
            + (f" | Val Loss: {val_loss:.4f}" if val_loader is not None else "")
        )

        if score < best_loss:
            best_loss = float(score)
            torch.save(model.state_dict(), output_file)

    history_path = output_file.with_name(output_file.stem + "_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": method,
                "seed": seed,
                "output_path": str(output_file),
                "best_score": float(best_loss),
                "history": history,
            },
            handle,
            indent=2,
        )

    print(f"Training Complete. Best Model Saved: {output_file}")
    print(f"History saved to: {history_path}")
    return str(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=['msm', 'sae', 'contrast'], required=True, help="SSL Method")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data_dir", type=str, default=None, help="Path to signal directory")
    parser.add_argument("--train-manifest", type=str, default="", help="Optional canonical train manifest")
    parser.add_argument("--val-manifest", type=str, default="", help="Optional canonical val manifest")
    parser.add_argument("--output", type=str, default="", help="Output checkpoint path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Smart default for data_dir
    target_dir = args.data_dir
    if target_dir is None:
        if os.path.exists("data/processed_signals") and len(os.listdir("data/processed_signals")) > 10:
            target_dir = "data/processed_signals"
            print(f"Using processed signals from: {target_dir}")
        else:
            target_dir = "data/ablation_results/signals"
            print(f"Using default ablation signals from: {target_dir}")
    
    train_ssl_benchmark(
        args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=target_dir,
        seed=args.seed,
        train_manifest=args.train_manifest or None,
        val_manifest=args.val_manifest or None,
        output_path=args.output or None,
        seq_len=args.seq_len,
        d_model=args.d_model,
    )
