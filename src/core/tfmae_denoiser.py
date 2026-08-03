import torch
import torch.nn as nn

class PatchEmbed1D(nn.Module):
    """
    1D Patch Embedding layer. Splits a 1D sequence into non-overlapping patches.
    """
    def __init__(self, in_channels=1, patch_size=15, d_model=64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, L)
        x = self.proj(x)
        # Output: (B, d_model, NumPatches)
        x = x.transpose(1, 2)
        # Output: (B, NumPatches, d_model)
        return x

class TFMAE_Denoiser(nn.Module):
    """
    Temporal Masked Autoencoder (TimeMAE / TFMAE styled).
    Used for both robust 1D signal denoising and advanced SSL Anomaly Detection.
    By training the model to reconstruct heavily masked physiological signals,
    it learns the true periodic manifold of breathing. If a patient experiences Apnea,
    the model will fail to reconstruct the masked periods accurately, yielding a high anomaly score.
    """
    def __init__(self, seq_len=300, patch_size=15, in_channels=1, d_model=64, 
                 enc_heads=4, enc_layers=4, dec_d_model=32, dec_heads=2, dec_layers=2, mask_ratio=0.5):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbed1D(in_channels, patch_size, d_model)
        
        # 2. ENCODER
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model), requires_grad=False)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=enc_heads, dim_feedforward=d_model*4, batch_first=True, activation='gelu'),
            num_layers=enc_layers
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        
        # 3. DECODER
        self.decoder_embed = nn.Linear(d_model, dec_d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_d_model))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dec_d_model), requires_grad=False)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dec_d_model, nhead=dec_heads, dim_feedforward=dec_d_model*4, batch_first=True, activation='gelu'),
            num_layers=dec_layers
        )
        self.decoder_norm = nn.LayerNorm(dec_d_model)
        
        # Output projection back to raw patch values
        self.decoder_pred = nn.Linear(dec_d_model, patch_size * in_channels)
        
        # Sequence Regressor Output (if used for downstream BPM prediction)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio=None):
        """
        Args:
            x: (B, C, L)
        Returns:
            latent: (B, len_keep+1, d_model)
            mask: (B, num_patches)
            ids_restore: (B, num_patches)
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        x = self.patch_embed(x) # (B, num_patches, d_model)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # masking
        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)
            ids_restore = torch.arange(x.shape[1], device=x.device).unsqueeze(0).repeat(x.shape[0], 1)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer
        x = self.encoder(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.expand(x.shape[0], self.num_patches + 1 - x.shape[1], -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1) # append cls token
        
        # add pos embed
        x = x + self.decoder_pos_embed
        
        # apply Transformer
        x = self.decoder(x)
        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)
        
        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward(self, seq, mask_ratio=None):
        """
        Args:
            seq: (B, L) or (B, C, L)
        """
        if seq.ndim == 2:
            seq = seq.unsqueeze(1) # (B, 1, L)
            
        latent, mask, ids_restore = self.forward_encoder(seq, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore) # (B, num_patches, patch_size * C)
        
        # Reshape to original format
        B = seq.shape[0]
        C = seq.shape[1]
        pred_seq = pred.reshape(B, self.num_patches * self.patch_size, C).transpose(1, 2)
        
        return pred_seq, mask

    def compute_anomaly_score(self, seq, mask_ratio=None):
        """
        Returns the MSE error calculated *only* on the masked patches.
        This represents the MAE Anomaly Score.
        """
        if seq.ndim == 2:
            seq = seq.unsqueeze(1) # (B, 1, L)
            
        pred_seq, mask = self.forward(seq, mask_ratio)
        
        # Create patch-level ground truth to easily multiply with mask
        # seq: (B, C, L)
        B, C, L = seq.shape
        # reshape to (B, num_patches, patch_size * C)
        target = seq.transpose(1, 2).contiguous().view(B, self.num_patches, self.patch_size * C)
        
        pred = pred_seq.transpose(1, 2).contiguous().view(B, self.num_patches, self.patch_size * C)
        
        # MSE per patch
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # (B, num_patches), mean over patch pixels
        
        # Only compute loss on masked patches
        # mask: 1 is masked out, 0 is kept
        # (B)
        loss = (loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        return loss

if __name__ == "__main__":
    print("Testing TFMAE_Denoiser...")
    model = TFMAE_Denoiser()
    # Mock depth signal: (Batch, SeqLen)
    dummy_signal = torch.randn(4, 300)
    
    recon_sig, mask = model(dummy_signal)
    print(f"Input: {dummy_signal.shape} -> Recon: {recon_sig.shape}, Mask: {mask.shape}")
    
    anomaly_scores = model.compute_anomaly_score(dummy_signal)
    print(f"Anomaly Scores: {anomaly_scores}")
