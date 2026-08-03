import torch
import torch.nn as nn
import cv2
import numpy as np

try:
    from transformers import VideoMAEModel
    HAS_VIDEOMAE = True
except ImportError:
    HAS_VIDEOMAE = False

class VideoMAE_3D_Encoder(nn.Module):
    """
    Video Masked Autoencoder (VideoMAE) feature extractor wrapper.
    Instead of relying on 2D explicit Flow/Depth extraction, this model
    takes a raw 3D video clip (e.g., 16 frames) and extracts dense spatio-temporal features
    using a pretrained ViT backbone trained on video reconstruction (SSL).
    """
    def __init__(self, d_model=64, pretrained_model="MCG-NJU/videomae-base", freeze_backbone=True):
        super().__init__()
        
        self.use_hf = HAS_VIDEOMAE
        
        if self.use_hf:
            # We attempt to load the HF VideoMAE Model
            try:
                self.videomae = VideoMAEModel.from_pretrained(pretrained_model)
                self.hidden_size = self.videomae.config.hidden_size
                if freeze_backbone:
                    for param in self.videomae.parameters():
                        param.requires_grad = False
                
                # Projection from VideoMAE embedding (768) to QuadModal d_model (64)
                self.proj = nn.Linear(self.hidden_size, d_model)
                print(f"[VideoMAE] Successfully loaded pretrained {pretrained_model}")
            except Exception as e:
                print(f"[VideoMAE] HF Model load failed: {e}. Falling back to 3D CNN Proxy.")
                self.use_hf = False
        
        if not self.use_hf:
            # Fallback 3D CNN proxy representing spatio-temporal modeling
            self.hidden_size = 512
            self.videomae = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2)), # keep temporal dim mostly
                
                nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 2)),
                
                nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(256),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 2)),
                
                nn.Conv3d(256, self.hidden_size, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(self.hidden_size),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((None, 1, 1)) # Collapse spatial, keep temporal
            )
            self.proj = nn.Conv1d(self.hidden_size, d_model, kernel_size=1)
            
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (Batch, NumFrames, Channels, Height, Width)
                         expected format for HF VideoMAE.
            If using proxy: (Batch, Channels, NumFrames, Height, Width)
        """
        B, num_frames = pixel_values.shape[0], pixel_values.shape[1]
        
        if self.use_hf:
            # HF expects (B, num_frames, C, H, W) mostly, but check docs.
            # Outputs last_hidden_state: (B, seq_length, hidden_size) where seq_len is usually (num_frames/2 * H/16 * W/16)
            out = self.videomae(pixel_values=pixel_values)
            features = out.last_hidden_state
            
            # Map back to sequence length shape
            x = self.proj(features) # (B, patches, d_model)
            x = x.transpose(1, 2)
            # Interpolate the arbitrary patch count back to a standard length if needed
            x = nn.functional.interpolate(x, size=num_frames, mode='linear')
            return x # (B, d_model, num_frames)
            
        else:
            # Convert (B, T, C, H, W) -> (B, C, T, H, W) for Conv3d
            x = pixel_values.transpose(1, 2)
            features = self.videomae(x) # (B, hidden_size, T', 1, 1)
            features = features.squeeze(-1).squeeze(-1) # (B, hidden_size, T')
            
            x = self.proj(features) # (B, d_model, T')
            if x.shape[-1] != num_frames:
                x = nn.functional.interpolate(x, size=num_frames, mode='linear')
            return x # (B, d_model, num_frames)

if __name__ == "__main__":
    print("Testing VideoMAE Proxy...")
    model = VideoMAE_3D_Encoder() # Force proxy if HF not ready
    dummy_video = torch.randn(2, 16, 3, 224, 224) # Batch=2, 16 frames, RGB, 224x224
    out = model(dummy_video)
    print(f"VideoMAE Output Shape: {out.shape}") # Expect (2, 64, 16)
