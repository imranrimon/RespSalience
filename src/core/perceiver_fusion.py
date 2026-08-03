import torch
import torch.nn as nn
from src.core.audio_encoders import ASTEncoder

class PerceiverFusionModel(nn.Module):
    """
    Perceiver IO inspired Fusion Architecture for Quad-Modal signals.
    Instead of self-attending over 3*SeqLen tokens (which is O(N^2) and expensive),
    it uses a small fixed-size latent array to cross-attend to the dense input signals.
    This solves scaling issues with high frequency inputs.
    """
    def __init__(self, seq_len=300, d_model=64, num_latents=64, latent_dim=128, nhead=4, num_layers=4, vlm_dim=2048, use_ast_audio=False):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.use_ast_audio = use_ast_audio
        
        # Signal Encoders 
        self.depth_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.flow_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        if self.use_ast_audio:
            self.audio_encoder = ASTEncoder(d_model=d_model)
        else:
            self.audio_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        
        # Modality Type Embeddings to distinguish concatenated inputs
        self.modality_embed = nn.Embedding(3, d_model)
        
        # Positional Encoding for the sequences
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Perceiver Latent Array
        # The latent array gathers information from the input array via Cross-Attention
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        
        # Optional VLM guidance: We inject VLM context into the latents initially
        self.vlm_proj = nn.Linear(vlm_dim, latent_dim) if vlm_dim > 0 else None
        
        # 1. Cross-Attention: Latent (Q) attend to Inputs (K, V)
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, kdim=d_model, vdim=d_model, num_heads=nhead, batch_first=True)
        
        # 2. Latent Transformer (Self-Attention within latent space)
        self.latent_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=512, batch_first=True),
            num_layers=num_layers
        )
        
        # 3. Output Regressor decodes the averaged latent state into a single BPM
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1) # Predicts BPM
        )
        
        # Anomaly Decoder (for Apnea SSL)
        # Reconstructs the original signals from the latent array
        self.decoder_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, seq_len * 3)
        )

    def forward(self, depth_signal, flow_signal, audio_signal=None, vlm_emb=None, return_embedding=False, mode='cls'):
        B = depth_signal.shape[0]
        
        # Expand channels
        x_depth = depth_signal.unsqueeze(1)
        x_flow = flow_signal.unsqueeze(1)
        if audio_signal is None:
            audio_signal = torch.zeros_like(depth_signal)
        x_audio = audio_signal.unsqueeze(1)
        
        # Encode (Outputs: B, d_model, SeqLen)
        e_depth = self.depth_encoder(x_depth)
        e_flow = self.flow_encoder(x_flow)
        if self.use_ast_audio:
            e_audio = self.audio_encoder(audio_signal)
        else:
            e_audio = self.audio_encoder(x_audio)
            
        # Transpose to B, SeqLen, d_model
        e_depth = e_depth.transpose(1, 2)
        e_flow = e_flow.transpose(1, 2)
        e_audio = e_audio.transpose(1, 2)
        
        # Add Positional & Modality Embeddings
        e_depth = e_depth + self.pos_embedding + self.modality_embed(torch.full((B, 1), 0, dtype=torch.long, device=depth_signal.device))
        e_flow = e_flow + self.pos_embedding + self.modality_embed(torch.full((B, 1), 1, dtype=torch.long, device=depth_signal.device))
        e_audio = e_audio + self.pos_embedding + self.modality_embed(torch.full((B, 1), 2, dtype=torch.long, device=depth_signal.device))
        
        # Concatenate inputs into one large array pool: (B, 3*SeqLen, d_model)
        features = torch.cat([e_depth, e_flow, e_audio], dim=1) 
        
        # Prepare latents
        # Inject VLM contextual cues into the latent initial state if provided
        z = self.latents.expand(B, -1, -1) # (B, NumLatents, latent_dim)
        if vlm_emb is not None and self.vlm_proj is not None:
            v_context = self.vlm_proj(vlm_emb).unsqueeze(1) # (B, 1, latent_dim)
            # Add VLM context directly to the latents (broadcasted or replaced)
            # We will use it to substitute the first latent token 
            # so the model inherently conditions on language context!
            z = torch.cat([v_context, z[:, 1:, :]], dim=1)
            
        # 1. Cross-Attention Bottleneck
        # The latents acts as Queries, the massive signal array acts as Keys & Values
        # (B, NumLatents, latent_dim) <- CrossAttn <- (B, 3*SeqLen, d_model)
        z, _ = self.cross_attn(query=z, key=features, value=features)
        
        # 2. Deep Latent Processing (Self-Attention)
        z = self.latent_transformer(z) # (B, NumLatents, latent_dim)
        
        # Collapse along latent sequence dimension (mean pooling)
        z_out = z.mean(dim=1) # (B, latent_dim)
        
        if return_embedding:
            return z_out
            
        # Classification/Regression
        if mode == 'cls':
            return self.regressor(z_out)
        elif mode == 'recon':
            return self.decoder_head(z_out)
        elif mode == 'both':
            return self.regressor(z_out), self.decoder_head(z_out)

    def compute_anomaly_score(self, depth_signal, flow_signal, audio_signal, vlm_emb=None):
        """Calculates SSL Reconstruction MSE for Apnea Detection."""
        B = depth_signal.shape[0]
        device = depth_signal.device
        
        recon = self.forward(depth_signal, flow_signal, audio_signal, vlm_emb, mode='recon')
        
        # Target is concatenated original sequence
        orig = torch.cat([depth_signal, flow_signal, audio_signal], dim=1) # (B, SeqLen*3)
        
        mse = torch.mean((recon - orig) ** 2, dim=1) # (B)
        return mse

if __name__ == "__main__":
    print("Testing PerceiverFusionModel instantiation...")
    model = PerceiverFusionModel()
    depth = torch.randn(2, 300)
    flow = torch.randn(2, 300)
    audio = torch.randn(2, 300)
    vlm = torch.randn(2, 2048)
    
    bpm = model(depth, flow, audio, vlm_emb=vlm, mode='cls')
    recon = model(depth, flow, audio, vlm_emb=vlm, mode='recon')
    score = model.compute_anomaly_score(depth, flow, audio, vlm_emb=vlm)
    
    print(f"BPM Out: {bpm.shape}")
    print(f"Recon Out: {recon.shape}")
    print(f"Anomaly Score: {score.shape}")
