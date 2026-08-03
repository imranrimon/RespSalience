import os
import sys
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Suppress vmdpy "invalid value in scalar divide" on near-zero segments (harmless)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="vmdpy")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.core.dl_denoiser import ResUNet1D, WaveNetAutoencoder, TransformerAutoencoder1D, ConditionalDiffusion1D, CNN_BiLSTM_Denoiser, MambaDenoiser
from src.core.tfmae_denoiser import TFMAE_Denoiser
from src.core.signal_processor import SignalProcessor

class NoisyRespirationDataset(Dataset):
    """
    Loads pre-extracted respiratory signals (already VMD-cleaned in Phase 1)
    and synthesises noisy versions on-the-fly for denoiser supervised training.

    The Phase-1 signals (data/ablation_results/signals/*.npz) are already
    Fused-VMD output — re-running VMD here would waste hours and add no value.
    A fast Butterworth bandpass (0.1-0.8 Hz) is applied as the only cleanup step.
    """
    def __init__(self, data_dir="data/ablation_results/signals", seq_len=300):
        self.seq_len = seq_len
        self.clean_signals = []

        from scipy.signal import butter, filtfilt

        def fast_bandpass(sig, fs=30.0, lo=0.1, hi=0.8):
            """Fast Butterworth bandpass — microseconds per signal."""
            nyq = fs / 2.0
            b, a = butter(4, [lo / nyq, hi / nyq], btype='band')
            try:
                return filtfilt(b, a, sig).astype(np.float32)
            except Exception:
                return sig.astype(np.float32)

        all_npz = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".npz"):
                    all_npz.append(os.path.join(root, f))

        print(f"Loading {len(all_npz)} signals from {data_dir}  (fast mode, no VMD)...")
        for path in all_npz:
            try:
                data  = np.load(path)
                # Prefer 'depth'; fall back to 'flow' or 'audio'
                sig = None
                for key in ('depth', 'flow', 'audio'):
                    val = data.get(key)
                    if val is not None and len(val) > 50:
                        sig = val.astype(np.float32)
                        break
                if sig is None:
                    continue

                clean = fast_bandpass(sig)

                # Skip flat / near-zero signals
                if np.max(clean) - np.min(clean) > 1e-6:
                    self.clean_signals.append(clean)
            except Exception:
                pass

        print(f"Loaded {len(self.clean_signals)} valid signals.")


    def __len__(self):
        # Artificially expand dataset size through heavy synthetic augmentation
        return len(self.clean_signals) * 10 

    def _pad_or_crop(self, sig):
        if len(sig) > self.seq_len:
            start = np.random.randint(0, len(sig) - self.seq_len)
            return sig[start:start+self.seq_len]
        else:
            return np.pad(sig, (0, self.seq_len - len(sig)))

    def _synthesize_noise(self, clean_sig):
        # 1. Baseline Drift (Low Freq sinusoid)
        t = np.linspace(0, 10, self.seq_len)
        drift = np.sin(2 * np.pi * np.random.uniform(0.01, 0.05) * t) * np.random.uniform(0.5, 2.0)
        
        # 2. Additive White Gaussian Noise
        awgn = np.random.randn(self.seq_len) * np.random.uniform(0.1, 0.5)
        
        # 3. Sudden spikes (motion artifacts)
        spikes = np.zeros(self.seq_len)
        if np.random.rand() > 0.5:
            idx = np.random.randint(0, self.seq_len)
            spikes[idx] = np.random.uniform(3, 5) * np.random.choice([-1, 1])
            
        noisy = clean_sig + drift + awgn + spikes
        
        # Min/Max normalize both to [-1, 1] range for Tanh activations
        def normalize(s):
            _range = np.max(s) - np.min(s)
            if _range < 1e-6: return s
            return 2 * ((s - np.min(s)) / _range) - 1
            
        return normalize(noisy), normalize(clean_sig)

    def __getitem__(self, idx):
        base_idx = idx % len(self.clean_signals)
        clean = self._pad_or_crop(self.clean_signals[base_idx])
        noisy, clean = self._synthesize_noise(clean)
        
        return {
            'noisy': torch.tensor(noisy, dtype=torch.float32).unsqueeze(0), # (1, L)
            'clean': torch.tensor(clean, dtype=torch.float32).unsqueeze(0)  # (1, L)
        }

def train_denoiser(model_type="unet", epochs=50, batch_size=32, lr=1e-3, patience=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_type.upper()} on {device} (patience={patience})...")
    
    dataset = NoisyRespirationDataset()
    if len(dataset) == 0:
        print("Error: No signals found. Run classical ablation first to save .npz files.")
        return
        
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    if model_type == "unet":
        model = ResUNet1D().to(device)
    elif model_type == "transformer":
        model = TransformerAutoencoder1D().to(device)
    elif model_type == "diffusion":
        model = ConditionalDiffusion1D(timesteps=50).to(device)
    elif model_type == "cnn-bilstm":
        model = CNN_BiLSTM_Denoiser().to(device)
    elif model_type == "tfmae":
        model = TFMAE_Denoiser().to(device)
    elif model_type == "mamba":
        model = MambaDenoiser().to(device)
    else:
        model = WaveNetAutoencoder().to(device)
        
    criterion = nn.L1Loss() # L1 Loss is better for preserving signal peaks than MSE
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    os.makedirs("data/models/denoisers", exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            
            optimizer.zero_grad()
            
            if model_type == "diffusion":
                loss = model.compute_loss(clean, noisy)
            elif model_type == "tfmae":
                # TFMAE learns to reconstruct clean sequence from heavily masked clean sequence
                loss = model.compute_anomaly_score(clean).mean()
            else:
                pred = model(noisy)
                loss = criterion(pred, clean)
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch['noisy'].to(device)
                clean = batch['clean'].to(device)
                
                if model_type == "diffusion":
                    loss = model.compute_loss(clean, noisy).item()
                elif model_type == "tfmae":
                    loss = model.compute_anomaly_score(clean).mean().item()
                else:
                    pred = model(noisy)
                    loss = criterion(pred, clean).item()
                    
                val_loss += loss
                
        val_loss /= len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1} | Train Loss (L1): {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            save_path = f"data/models/denoisers/{model_type}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  -> ✓ New best! Saved to {save_path}")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  -> Early stopping at epoch {epoch+1}. Best was epoch {best_epoch} (val={best_val_loss:.4f})")
                break
    
    print(f"\n{'='*50}")
    print(f"[{model_type.upper()}] Training Done. Best Val Loss={best_val_loss:.4f} @ epoch {best_epoch}")
    print(f"Saved: data/models/denoisers/{model_type}_best.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["unet", "wavenet", "transformer", "diffusion", "cnn-bilstm", "tfmae", "mamba"], default="unet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    train_denoiser(model_type=args.model, epochs=args.epochs, patience=args.patience,
                   lr=args.lr, batch_size=args.batch_size)
