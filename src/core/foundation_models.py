import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer

# ==============================================================================
# 1. RespMAE: Signal Tokenization & Masked Autoencoder
# ==============================================================================

class VQVAE_1D(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE) for 1D physiological signals.
    Learns a discrete codebook of 'breathing morphology tokens' allowing us to 
    treat continuous waveforms like sequences of discrete text words.
    """
    def __init__(self, in_channels=1, hidden_dim=64, num_embeddings=512, embedding_dim=64):
        super().__init__()
        
        # Encoder: compress time dimension
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim*2, embedding_dim, kernel_size=4, stride=2, padding=1)
        )
        
        # Quantization Codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # Decoder: reconstruct signal
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim*2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, in_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        # x shape: [batch, 1, seq_len]
        z_e = self.encoder(x)  # [batch, emb_dim, seq_len_compressed]
        z_e_transposed = z_e.permute(0, 2, 1).contiguous()
        
        # Find nearest embeddings
        d = torch.cdist(z_e_transposed, self.codebook.weight) # Euclidean distance
        encoding_indices = torch.argmin(d, dim=-1) # [batch, seq_len_compressed]
        z_q = self.codebook(encoding_indices).permute(0, 2, 1) # [batch, emb_dim, seq_len_compressed]
        
        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        x_recon = self.decoder(z_q_st)
        
        # Ensure exact sequence length match
        if x_recon.shape[-1] != x.shape[-1]:
            x_recon = F.interpolate(x_recon, size=x.shape[-1], mode='linear', align_corners=False)
        
        # VQ Loss (Commitment loss)
        vq_loss = F.mse_loss(z_q, z_e.detach()) + 0.25 * F.mse_loss(z_e, z_q.detach())
        recon_loss = F.mse_loss(x_recon, x)
        loss = recon_loss + vq_loss
        
        return x_recon, loss, encoding_indices

class RespMAE(nn.Module):
    """
    Masked Autoencoder trained on discrete VQ-VAE tokens. 
    Drop 75% of signal tokens and force the Transformer to predict them, 
    learning deep semantic understanding of respiratory waveforms zero-shot.
    """
    def __init__(self, vocab_size=512, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size # Special mask token
        
        self.embedding = nn.Embedding(vocab_size + 1, d_model) # +1 for [MASK]
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, tokens, mask_prob=0.75):
        # tokens: [batch, seq_len] of integers from VQ-VAE
        batch_size, seq_len = tokens.shape
        
        # Create random mask
        mask = torch.rand(batch_size, seq_len, device=tokens.device) < mask_prob
        
        # Replace masked tokens with [MASK] token id
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token_id
        
        # Pass through transformer
        x = self.embedding(masked_tokens)
        out = self.transformer(x)
        
        # Predict original vocabulary
        logits = self.head(out) # [batch, seq_len, vocab_size]
        
        # Only compute loss on the masked tokens
        loss = F.cross_entropy(logits[mask], tokens[mask])
        
        return logits, loss

# ==============================================================================
# 2. RespDiff: Latent Diffusion for Signal Inpainting
# ==============================================================================

class RespDiffInpainter(nn.Module):
    """
    1D Diffusion model designed for missing-data imputation (Inpainting).
    When a sensor fails (or patient moves heavily), it hallucinates (generates)
    the missing physiological waveform segment conditioned on the clean context.
    """
    def __init__(self, channels=1, hidden_dim=64):
        super().__init__()
        # Simplified 1D UNet Backbone for DDPM
        self.net = nn.Sequential(
            nn.Conv1d(channels + 1, hidden_dim, 3, padding=1), # +1 for mask conditioning
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, channels, 3, padding=1)
        )
        self.timesteps = 100
        
    def forward(self, x, mask, t=None):
        # x: corrupted/noisy signal [batch, 1, seq_len]
        # mask: 1 where signal is valid, 0 where signal is missing [batch, 1, seq_len]
        # Condition on the mask
        x_cond = torch.cat([x, mask], dim=1) 
        return self.net(x_cond)

    def inpaint(self, corrupted_signal, mask):
        """
        Hallucinates the missing segment mathematically using DDPM.
        mask == 0 indicates missing data to be hallucinated.
        """
        batch, c, l = corrupted_signal.shape
        device = corrupted_signal.device
        
        # Start with pure noise
        x_gen = torch.randn_like(corrupted_signal)
        
        # Simplified iterative denoising loop (DDIM-style fast sampling)
        for t in reversed(range(self.timesteps)):
            # Force known regions to be the true signal
            x_gen = x_gen * (1 - mask) + corrupted_signal * mask
            
            # Predict noise to remove
            predicted_noise = self(x_gen, mask)
            
            # Take a denoising step (simplified equation)
            alpha = 1 - (t / self.timesteps)
            x_gen = x_gen - (1-alpha) * predicted_noise
            
        return x_gen * (1 - mask) + corrupted_signal * mask


# ==============================================================================
# 3. CLISP: Contrastive Text-Signal Pretraining
# ==============================================================================

class CLISP(nn.Module):
    """
    Clinical-Signal Pretraining (CLIP for physiological waveforms).
    Aligns 1D signal representations with clinical text embeddings into a joint
    latent space, enabling zero-shot diagnosis from raw waves.
    """
    def __init__(self, embed_dim=256, text_model="medicalai/ClinicalBERT"):
        super().__init__()
        
        # 1. Signal Encoder (1D ResNet style)
        self.signal_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim)
        )
        
        # 2. Text Encoder (HuggingFace Transformers)
        try:
            self.text_encoder = AutoModel.from_pretrained(text_model)
        except Exception:
            # Fallback to a tiny model if ClinicalBERT is not cached/downloaded
            print(f"Warning: {text_model} not found locally, falling back to distilbert.")
            self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
            
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        
        # Learned temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_signal(self, signal):
        return F.normalize(self.signal_encoder(signal), dim=-1)

    def encode_text(self, text_input_ids, text_attention_mask):
        text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        # Use CLS token representation
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        text_embeds = self.text_projection(text_embeds)
        return F.normalize(text_embeds, dim=-1)

    def forward(self, signal, text_input_ids, text_attention_mask):
        # Embed both modalities
        signal_embeds = self.encode_signal(signal)
        text_embeds = self.encode_text(text_input_ids, text_attention_mask)
        
        # Cosine similarities
        logit_scale = self.logit_scale.exp()
        logits_per_signal = logit_scale * signal_embeds @ text_embeds.t()
        logits_per_text = logits_per_signal.t()
        
        # Contrastive (InfoNCE) Loss
        labels = torch.arange(logits_per_signal.shape[0], device=signal.device)
        loss_s = F.cross_entropy(logits_per_signal, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_s + loss_t) / 2
        
        return loss, logits_per_signal

if __name__ == "__main__":
    print("Testing Foundation Models Initialization...")
    
    # 1. Test RespMAE Tokenizer
    print("\n[RespMAE Tokenizer]")
    dummy_signal = torch.randn(2, 1, 300)
    vqvae = VQVAE_1D(in_channels=1, hidden_dim=32, num_embeddings=128, embedding_dim=16)
    x_recon, vq_loss, tokens = vqvae(dummy_signal)
    print(f"Signal: {dummy_signal.shape} -> Tokens: {tokens.shape}")
    
    # 2. Test RespMAE Transformer
    print("\n[RespMAE Masked Predictor]")
    mae = RespMAE(vocab_size=128, d_model=64, num_layers=2)
    logits, mae_loss = mae(tokens)
    print(f"Masked prediction loss: {mae_loss.item():.4f}")
    
    # 3. Test RespDiff Inpainter
    print("\n[RespDiff Inpainter]")
    diff = RespDiffInpainter(channels=1)
    mask = torch.ones_like(dummy_signal)
    mask[:, :, 100:150] = 0 # Simulate missing chunk in the middle
    inpainted = diff.inpaint(dummy_signal, mask)
    print(f"Inpainted Signal (Zero-context filled): {inpainted.shape}")
    
    # 4. Test CLISP (Text-Signal Alignment)
    print("\n[CLISP Contrastive Aligner]")
    clisp = CLISP(embed_dim=128, text_model="prajjwal1/bert-tiny") # Fast dummy text encoder
    
    # Dummy HuggingFace inputs (batch of 2)
    dummy_input_ids = torch.randint(0, 1000, (2, 16))
    dummy_attn_mask = torch.ones(2, 16)
    
    loss, logits = clisp(dummy_signal, dummy_input_ids, dummy_attn_mask)
    print(f"Contrastive Loss (InfoNCE): {loss.item():.4f}")
    print(f"Logits shape (Signal vs Text batch): {logits.shape}")

    print("\n[v] All Foundation Models initialized correctly.")
