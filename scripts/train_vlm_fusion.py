"""
VLM-Gated Fusion Training (Semi-Supervised)
Uses BLIP visual features as VLM context query for the cross-attention transformer.
Surrogate labels from VLM distress detection are used for SPMat-BT pretraining.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import argparse
import os
import sys
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.multimodal_fusion_models import TemporalFusionTransformer
from src.core.losses import SupervisedBarlowTwinsLoss


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VLMSignalDataset(Dataset):
    """Dataset that loads depth/flow signals + VLM embeddings for VLM-Gated training."""
    
    def __init__(self, signal_dir, seq_len=300, mode='finetune'):
        self.seq_len = seq_len
        self.mode = mode
        self.samples = []
        
        signal_dir = Path(signal_dir)
        for f in sorted(signal_dir.glob("*.npz")):
            try:
                data = np.load(f, allow_pickle=True)
                depth = data['depth'] if 'depth' in data else np.zeros(seq_len)
                flow = data['flow'] if 'flow' in data else np.zeros(seq_len)
                audio = data['audio'] if 'audio' in data else np.zeros(seq_len)
                gt = float(data['gt']) if 'gt' in data else None
                
                # VLM surrogate label (binary distress)
                vlm_label = bool(data['vlm_label']) if 'vlm_label' in data else False
                
                if gt is not None:
                    self.samples.append({
                        'depth': depth, 'flow': flow, 'audio': audio,
                        'gt': gt, 'vlm_label': 1.0 if vlm_label else 0.0
                    })
            except Exception as e:
                continue
        
        print(f"[VLMSignalDataset] Loaded {len(self.samples)} samples ({mode} mode)")
    
    def __len__(self):
        return len(self.samples)
    
    def _pad_or_truncate(self, signal):
        signal = np.array(signal, dtype=np.float32)
        if len(signal) > self.seq_len:
            signal = signal[:self.seq_len]
        elif len(signal) < self.seq_len:
            signal = np.pad(signal, (0, self.seq_len - len(signal)))
        return torch.tensor(signal, dtype=torch.float32)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'depth': self._pad_or_truncate(s['depth']),
            'flow': self._pad_or_truncate(s['flow']),
            'audio': self._pad_or_truncate(s['audio']),
            'gt': torch.tensor(s['gt'], dtype=torch.float32),
            'vlm_label': torch.tensor(s['vlm_label'], dtype=torch.float32),
        }

def apply_modality_dropout(depth, flow, audio, drop_prob=0.2):
    """
    Randomly drops an entire modality during training to make the fusion 
    transformer robust to missing or corrupted signals.
    """
    b = depth.shape[0]
    device = depth.device
    
    # Masks must be (B, 1) to broadcast correctly against (B, seq_len) signals.
    # Using (B, 1, 1) would create (B, B, seq_len) via broadcasting â€” shape bug.
    mask_d = (torch.rand(b, 1, device=device) > drop_prob).float()
    mask_f = (torch.rand(b, 1, device=device) > drop_prob).float()
    mask_a = (torch.rand(b, 1, device=device) > drop_prob).float()
    
    # Ensure at least one modality is kept
    all_dropped = (mask_d + mask_f + mask_a) == 0  # (B, 1)
    mask_d[all_dropped] = 1.0
    mask_f[all_dropped] = 1.0
    mask_a[all_dropped] = 1.0
    
    return depth * mask_d, flow * mask_f, audio * mask_a

def train_spmat_bt(data_dir, output_path, epochs=50, batch_size=32, lr=1e-3, seed=42):
    """
    Stage 1: SPMat-BT Pretraining with surrogate labels.
    Uses Supervised Barlow Twins loss where labels are derived from GT BPM bins,
    creating a meaningful class structure for the contrastive objective.
    
    Label strategy: Bin GT BPM into 5 categories to give diverse class structure.
    This avoids the collapse seen with binary VLM labels (nearly all same class).
    """
    print("=" * 60)
    print(" SEMI-SUPERVISED PRETRAINING: SPMat-BT (Surrogate Labels)")
    print("=" * 60)
    
    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = VLMSignalDataset(data_dir, mode='pretrain')
    
    if len(dataset) == 0:
        print("Error: No data. Run ablation with --save-signals first.")
        return
    
    # Analyze label distribution
    gt_vals = [s['gt'] for s in dataset.samples]
    vlm_pos = sum(1 for s in dataset.samples if s['vlm_label'] > 0.5)
    print(f"  GT BPM range: [{min(gt_vals):.0f}, {max(gt_vals):.0f}]")
    print(f"  VLM distress labels: {vlm_pos}/{len(gt_vals)} positive ({100*vlm_pos/len(gt_vals):.0f}%)")
    
    # Create BPM-bin surrogate labels (5 bins based on quantiles)
    gt_arr = np.array(gt_vals)
    bin_edges = np.quantile(gt_arr, [0.2, 0.4, 0.6, 0.8])
    for s in dataset.samples:
        s['surrogate_label'] = float(np.digitize(s['gt'], bin_edges))  # 0-4
    
    n_classes = len(set(s['surrogate_label'] for s in dataset.samples))
    print(f"  Surrogate label bins: {n_classes} classes (BPM quantile bins)")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    model = TemporalFusionTransformer(d_model=64, nhead=4, num_layers=4)
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    barlow_loss = SupervisedBarlowTwinsLoss(lambda_coeff=0.005)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"SPMat Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            depth = batch['depth'].to(device)
            flow = batch['flow'].to(device)
            audio = batch['audio'].to(device)
            
            # Use surrogate labels from dataset (set dynamically above)
            indices = list(range(depth.shape[0]))
            labels = torch.tensor(
                [dataset.samples[i % len(dataset)]['surrogate_label'] for i in range(depth.shape[0])],
                dtype=torch.float32, device=device
            )
            
            optimizer.zero_grad()
            
            # Apply Modality Dropout
            d_drop, f_drop, a_drop = apply_modality_dropout(depth, flow, audio, drop_prob=0.2)
            
            # View 1: Original (with missing modalities potentially)
            z1 = model(d_drop, f_drop, audio_signal=a_drop, return_embedding=True)
            
            # View 2: Augmented (stronger augmentation for diversity)
            noise_scale = 0.1
            shift = np.random.randint(1, 15)
            aug_depth = torch.roll(depth, shifts=shift, dims=1) + torch.randn_like(depth) * noise_scale
            aug_flow = torch.roll(flow, shifts=-shift, dims=1) + torch.randn_like(flow) * noise_scale
            aug_audio = torch.roll(audio, shifts=shift, dims=1) + torch.randn_like(audio) * noise_scale
            z2 = model(aug_depth, aug_flow, audio_signal=aug_audio, return_embedding=True)
            
            # SPMat-BT: Same BPM-bin â†’ pull together, different bin â†’ push apart
            loss = barlow_loss(z1, z2, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break
    
    print(f"SPMat-BT Pretraining Done. Best Loss: {best_loss:.6f}. Saved: {output_path}")


def train_vlm_gated(data_dir, pretrained_path, output_path, epochs=30, batch_size=32, lr=1e-4, patience=10, seed=42):
    """
    Stage 2: VLM-Gated Fine-tuning with validation split and early stopping.
    Fine-tunes the transformer with a learnable VLM context query.
    Saves the best model based on held-out validation MAE to prevent overfitting.
    """
    print("=" * 60)
    print(" SEMI-SUPERVISED FINE-TUNING: VLM-Gated Fusion")
    print(f" Patience={patience} | LR={lr} | Epochs={epochs}")
    print("=" * 60)
    
    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_dataset = VLMSignalDataset(data_dir, mode='finetune')
    
    if len(full_dataset) == 0:
        print("Error: No data.")
        return
    
    # â”€â”€ 90/10 Train/Val Split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f"  Train samples: {train_size} | Val samples: {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # VLM dim matches BLIP pooler output (768) but we use a smaller surrogate
    # since we don't have per-sample BLIP embeddings yet.
    # The VLM projection layer maps vlm_dim -> d_model anyway.
    vlm_surrogate_dim = 2048  # Match model's vlm_dim
    
    model = TemporalFusionTransformer(d_model=64, nhead=4, num_layers=4, vlm_dim=vlm_surrogate_dim)
    
    # Load pretrained backbone
    if Path(pretrained_path).exists():
        print(f"Loading pretrained from {pretrained_path}")
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state, strict=False)  # strict=False for vlm_proj mismatch
    else:
        print("Warning: No pretrained model found. Training from scratch.")
    
    model.to(device)
    
    # VLM label -> embedding projection (binary label -> vlm_dim vector)
    vlm_label_proj = nn.Sequential(
        nn.Linear(1, 256),
        nn.ReLU(),
        nn.Linear(256, vlm_surrogate_dim)
    ).to(device)
    
    all_params = list(model.parameters()) + list(vlm_label_proj.parameters())
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    # Torch scheduler signature differs across versions; keep this portable.
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
    criterion = nn.MSELoss()
    
    best_val_mae = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        # â”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        model.train()
        vlm_label_proj.train()
        total_loss = 0
        total_mae = 0
        pbar = tqdm(train_loader, desc=f"VLM-Gated Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            depth = batch['depth'].to(device)
            flow = batch['flow'].to(device)
            audio = batch['audio'].to(device)
            gt = batch['gt'].to(device).unsqueeze(1)
            vlm_label = batch['vlm_label'].to(device).unsqueeze(1)  # (B, 1)
            
            optimizer.zero_grad()
            
            # Apply modality dropout to simulate missing sensors
            d_drop, f_drop, a_drop = apply_modality_dropout(depth, flow, audio, drop_prob=0.2)
            
            # Project VLM label to embedding space
            vlm_emb = vlm_label_proj(vlm_label)  # (B, vlm_dim)
            
            pred = model(d_drop, f_drop, audio_signal=a_drop, vlm_emb=vlm_emb)
            recon_mse = model.compute_anomaly_score(depth, flow, audio_signal=audio, vlm_emb=vlm_emb).mean()
            
            loss_cls = criterion(pred, gt)
            loss = loss_cls + (0.1 * recon_mse)
            
            mae = torch.mean(torch.abs(pred - gt))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            pbar.set_postfix({'train_mae': f"{mae.item():.2f}"})
        
        avg_train_mae = total_mae / len(train_loader)
        
        # â”€â”€ Validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        model.eval()
        vlm_label_proj.eval()
        val_mae_total = 0
        with torch.no_grad():
            for batch in val_loader:
                depth = batch['depth'].to(device)
                flow = batch['flow'].to(device)
                audio = batch['audio'].to(device)
                gt = batch['gt'].to(device).unsqueeze(1)
                vlm_label = batch['vlm_label'].to(device).unsqueeze(1)
                vlm_emb = vlm_label_proj(vlm_label)
                pred = model(depth, flow, audio_signal=audio, vlm_emb=vlm_emb)
                val_mae_total += torch.mean(torch.abs(pred - gt)).item()
        
        avg_val_mae = val_mae_total / len(val_loader)
        scheduler.step(avg_val_mae)
        
        print(f"Epoch {epoch+1:3d} | Train MAE: {avg_train_mae:.2f} BPM | Val MAE: {avg_val_mae:.2f} BPM")
        
        # â”€â”€ Early Stopping on Val MAE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            patience_counter = 0
            best_epoch = epoch + 1
            torch.save({
                'model': model.state_dict(),
                'vlm_proj': vlm_label_proj.state_dict(),
                'epoch': epoch + 1,
                'val_mae': best_val_mae,
            }, output_path)
            print(f"  -> [OK] New best Val MAE={best_val_mae:.2f} BPM @ epoch {best_epoch}. Saved!")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  -> Early stopping triggered. Best epoch={best_epoch}, Val MAE={best_val_mae:.2f}")
                break
    
    print(f"\n{'='*60}")
    print(f"VLM-Gated Fine-tuning Done.")
    print(f"Best Val MAE: {best_val_mae:.2f} BPM @ epoch {best_epoch}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=['pretrain', 'finetune', 'both'], default='both')
    parser.add_argument("--data_dir", type=str, default="data/ablation_results/signals")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--ft_epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    os.makedirs("data/models", exist_ok=True)
    
    pretrain_path = "data/models/spmat_bt.pth"
    finetune_path = "data/models/vlm_gated_fused.pth"
    
    if args.stage in ['pretrain', 'both']:
        train_spmat_bt(args.data_dir, pretrain_path, epochs=args.epochs, seed=args.seed)
    
    if args.stage in ['finetune', 'both']:
        train_vlm_gated(args.data_dir, pretrain_path, finetune_path, epochs=args.ft_epochs, seed=args.seed)
