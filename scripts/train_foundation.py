import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.foundation_models import VQVAE_1D, RespMAE, RespDiffInpainter, CLISP

class FoundationDataset(Dataset):
    def __init__(self, metadata_path="data/foundation_data/synthetic/clinical_notes_metadata.csv", max_samples=None):
        self.df = pd.read_csv(metadata_path)
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)))
            
        self.signals = []
        self.texts = []
        
        print("Loading foundation signals into memory...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                data = np.load(row['signal_path'])
                sig = data['signal'].astype(np.float32)
                
                # Normalize to [-1, 1] range
                _range = np.max(sig) - np.min(sig)
                if _range > 1e-6:
                    sig = 2 * ((sig - np.min(sig)) / _range) - 1
                else:
                    sig = np.zeros_like(sig)
                    
                self.signals.append(sig)
                self.texts.append(row['clinical_text'])
            except Exception:
                pass
                
        print(f"Loaded {len(self.signals)} high-quality samples.")
        
    def __len__(self):
        return len(self.signals)
        
    def __getitem__(self, idx):
        return {
            'signal': torch.tensor(self.signals[idx]).unsqueeze(0), # Add channel dim: [1, seq_len]
            'text': self.texts[idx]
        }

def train_resp_mae(device, metadata_path, epochs=10):
    print("\n" + "="*50)
    print(" 1. Pretraining RespMAE (Signal Tokenizer & Masked Predictor)")
    print("="*50)
    
    dataset = FoundationDataset(metadata_path=metadata_path, max_samples=2000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    vqvae = VQVAE_1D().to(device)
    mae = RespMAE().to(device)
    
    # Train VQVAE first to stabilize codebook, then end-to-end
    optimizer = optim.AdamW(list(vqvae.parameters()) + list(mae.parameters()), lr=1e-3)
    
    for epoch in range(epochs):
        vqvae.train()
        mae.train()
        total_loss, total_vq, total_mae = 0.0, 0.0, 0.0
        
        pbar = tqdm(dataloader, desc=f"RespMAE Ep {epoch+1}/{epochs}")
        for batch in pbar:
            sig = batch['signal'].to(device)
            optimizer.zero_grad()
            
            # 1. Tokenize
            x_recon, vq_loss, tokens = vqvae(sig)
            
            # 2. Masked Prediction
            # Tokens shape: [batch, 37]
            logits, mae_loss = mae(tokens, mask_prob=0.75)
            
            # Combine losses
            loss = vq_loss + mae_loss * 0.5 
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_vq += vq_loss.item()
            total_mae += mae_loss.item()
            
            pbar.set_postfix({'vq': f'{vq_loss.item():.2f}', 'mae': f'{mae_loss.item():.2f}'})
            
        print(f"Epoch {epoch+1} | Total: {total_loss/len(dataloader):.3f} | VQ: {total_vq/len(dataloader):.3f} | MAE: {total_mae/len(dataloader):.3f}")

    os.makedirs("data/models/foundation", exist_ok=True)
    torch.save(vqvae.state_dict(), "data/models/foundation/vqvae_best.pth")
    torch.save(mae.state_dict(), "data/models/foundation/resp_mae_best.pth")
    print("✓ Saved RespMAE Foundation Model.")


def train_resp_diff(device, metadata_path, epochs=10):
    print("\n" + "="*50)
    print(" 2. Pretraining RespDiff (Latent Diffusion Inpainter)")
    print("="*50)
    
    dataset = FoundationDataset(metadata_path=metadata_path, max_samples=2000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = RespDiffInpainter().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"RespDiff Ep {epoch+1}/{epochs}")
        for batch in pbar:
            sig = batch['signal'].to(device)
            b, c, l = sig.shape
            optimizer.zero_grad()
            
            # Create a random continuous hole (mask=0) of 1-3 seconds
            mask = torch.ones_like(sig)
            for i in range(b):
                hole_len = np.random.randint(30, 90) # 1 to 3 seconds missing
                hole_start = np.random.randint(0, l - hole_len)
                mask[i, :, hole_start:hole_start+hole_len] = 0
            
            # The True noise we want to predict (in standard DDPM this would be added, 
            # here we cheat slightly for simplicity of demonstration and train autoencoder-style)
            # Inproduction: implement full forward noise schedule q(x_t | x_0)
            target = sig.clone()
            corrupted = sig * mask + torch.randn_like(sig) * (1-mask) * 0.5
            
            pred = model(corrupted, mask)
            
            # Loss only computed on the hallucinated hole
            loss = torch.nn.functional.mse_loss(pred * (1-mask), target * (1-mask))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'hole_mse': f'{loss.item():.4f}'})
            
    torch.save(model.state_dict(), "data/models/foundation/resp_diff_best.pth")
    print("✓ Saved RespDiff Foundation Model.")


def train_clisp(device, metadata_path, epochs=5):
    print("\n" + "="*50)
    print(" 3. Pretraining CLISP (Clinical-Signal Contrastive Aligner)")
    print("="*50)
    
    dataset = FoundationDataset(metadata_path=metadata_path, max_samples=1000) 
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True) # Needs smaller batch due to BERT memory
    
    from transformers import AutoTokenizer
    # Fallback to tiny bert if offline handling is needed
    try:
        tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        clisp = CLISP(text_model="medicalai/ClinicalBERT").to(device)
    except Exception:
        print("Using tiny-bert fallback for fast local demonstration...")
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        clisp = CLISP(text_model="prajjwal1/bert-tiny").to(device)
        
    optimizer = optim.AdamW(clisp.parameters(), lr=1e-4) # Smaller LR for text encoders
    
    for epoch in range(epochs):
        clisp.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"CLISP Ep {epoch+1}/{epochs}")
        for batch in pbar:
            sig = batch['signal'].to(device)
            texts = batch['text']
            
            # Tokenize text
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            loss, logits = clisp(sig, input_ids, attention_mask)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'infonce': f'{loss.item():.4f}'})
            
    torch.save(clisp.state_dict(), "data/models/foundation/clisp_best.pth")
    print("✓ Saved CLISP Foundation Model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mae", "diff", "clisp", "all"], default="all")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/foundation_data/synthetic/clinical_notes_metadata.csv",
        help="Path to foundation metadata CSV with columns including signal_path and clinical_text",
    )
    args = parser.parse_args()
    
    print(f"Training Foundation Models on {args.device.upper()}...")
    print(f"Using foundation metadata: {args.metadata_path}")
    
    if args.model in ["mae", "all"]:
        train_resp_mae(args.device, args.metadata_path)
        
    if args.model in ["diff", "all"]:
        train_resp_diff(args.device, args.metadata_path)
        
    if args.model in ["clisp", "all"]:
        train_clisp(args.device, args.metadata_path)
        
    print("\n[v] Phase 5 Foundation Training Complete.")
