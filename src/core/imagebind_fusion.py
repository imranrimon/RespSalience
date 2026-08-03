import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.audio_encoders import ASTEncoder
from src.core.multimodal_fusion_models import TemporalFusionTransformer

class ImageBindLiteFusion(nn.Module):
    """
    Alignment-First Fusion Model inspired by ImageBind.
    Projects all modalities (Depth, Flow, Audio, VLM) into a joint embedding space
    and applies a contrastive InfoNCE loss to align them *before* cross-attention fusion.
    This creates an extremely robust representation, especially when modalities drop out.
    """
    def __init__(self, seq_len=300, d_model=64, nhead=4, num_layers=4, vlm_dim=2048, use_ast_audio=False, temperature=0.07):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_ast_audio = use_ast_audio
        self.temperature = temperature
        
        # 1. Modality-Specific Encoders
        self.depth_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.flow_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        if self.use_ast_audio:
            self.audio_encoder = ASTEncoder(d_model=d_model)
        else:
            self.audio_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
            
        # 2. Alignment Projectors: Maps from sequence encoding to a single unified "Joint Embedding" (1D vector per modality)
        self.depth_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.flow_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.audio_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.vlm_proj = nn.Linear(vlm_dim, d_model) if vlm_dim > 0 else None
        
        # 3. Downstream Task Fusion (Using our standard Transformer Architecture but operating on the aligned features)
        # We decouple the feature extraction from the sequence fusion.
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, seq_len))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True),
            num_layers=num_layers
        )
        
        self.fusion_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward_encoders(self, x_depth, x_flow, audio_signal, x_audio):
        emb_depth = self.depth_encoder(x_depth)
        emb_flow = self.flow_encoder(x_flow)
        if self.use_ast_audio:
            emb_audio = self.audio_encoder(audio_signal)
        else:
            emb_audio = self.audio_encoder(x_audio)
        return emb_depth, emb_flow, emb_audio

    def compute_alignment_loss(self, depth_signal, flow_signal, audio_signal=None, vlm_emb=None):
        """
        Computes the InfoNCE Contrastive Loss to pull corresponding modalities together 
        and push unmatched batch pairs apart in the joint embedding space.
        """
        B = depth_signal.shape[0]
        x_depth = depth_signal.unsqueeze(1)
        x_flow = flow_signal.unsqueeze(1)
        if audio_signal is None:
            audio_signal = torch.zeros_like(depth_signal)
        x_audio = audio_signal.unsqueeze(1)
        
        emb_depth, emb_flow, emb_audio = self.forward_encoders(x_depth, x_flow, audio_signal, x_audio)
        
        # Project to joint space (B, d_model)
        z_depth = F.normalize(self.depth_proj(emb_depth), dim=-1)
        z_flow = F.normalize(self.flow_proj(emb_flow), dim=-1)
        z_audio = F.normalize(self.audio_proj(emb_audio), dim=-1)
        
        # Collect available modalities for contrastive pairs (Depth is our anchor)
        loss = 0.0
        pairs = 0
        
        # InfoNCE (Depth <-> Flow)
        logits_df = (z_depth @ z_flow.T) / self.temperature
        labels = torch.arange(B, device=depth_signal.device)
        loss += F.cross_entropy(logits_df, labels) + F.cross_entropy(logits_df.T, labels)
        pairs += 1
        
        # InfoNCE (Depth <-> Audio)
        logits_da = (z_depth @ z_audio.T) / self.temperature
        loss += F.cross_entropy(logits_da, labels) + F.cross_entropy(logits_da.T, labels)
        pairs += 1
        
        # InfoNCE (Depth <-> VLM)
        if vlm_emb is not None and self.vlm_proj is not None:
            z_vlm = F.normalize(self.vlm_proj(vlm_emb), dim=-1)
            logits_dv = (z_depth @ z_vlm.T) / self.temperature
            loss += F.cross_entropy(logits_dv, labels) + F.cross_entropy(logits_dv.T, labels)
            pairs += 1
            
        return loss / (pairs * 2)

    def forward(self, depth_signal, flow_signal, audio_signal=None, vlm_emb=None, return_embedding=False):
        B = depth_signal.shape[0]
        x_depth = depth_signal.unsqueeze(1)
        x_flow = flow_signal.unsqueeze(1)
        if audio_signal is None:
            audio_signal = torch.zeros_like(depth_signal)
        x_audio = audio_signal.unsqueeze(1)
        
        emb_depth, emb_flow, emb_audio = self.forward_encoders(x_depth, x_flow, audio_signal, x_audio)
        
        # Add Positional Embedding
        emb_depth = emb_depth + self.pos_embedding
        emb_flow = emb_flow + self.pos_embedding
        emb_audio = emb_audio + self.pos_embedding
        
        # Transpose for Transformer (B, SeqLen, D)
        seq_depth = emb_depth.transpose(1, 2)
        seq_flow = emb_flow.transpose(1, 2)
        seq_audio = emb_audio.transpose(1, 2)
        
        # Sequence Concat pool
        pool = torch.cat([seq_depth, seq_flow, seq_audio], dim=1) # (B, 3*Seq, D)
        
        # VLM Context acts as Query
        if vlm_emb is not None and self.vlm_proj is not None:
            query = self.vlm_proj(vlm_emb).unsqueeze(1) # (B, 1, D)
        else:
             # If no VLM, just mean pool the sequences to get a query
             query = pool.mean(dim=1, keepdim=True) # (B, 1, D)
             
        # Cross Attention: VLM Context attends to aligned sensor pool
        attn_out, _ = self.fusion_attention(query=query, key=pool, value=pool)
        
        # Pass through causal sequence transformer to gather sequence structure
        out = self.transformer(attn_out)
        
        # Regress
        pooled_out = out.squeeze(1) # (B, D)
        
        if return_embedding:
            return pooled_out
            
        bpm_pred = self.regressor(pooled_out)
        return bpm_pred

if __name__ == "__main__":
    print("Testing ImageBindLiteFusion...")
    model = ImageBindLiteFusion()
    depth = torch.randn(2, 300)
    flow = torch.randn(2, 300)
    audio = torch.randn(2, 300)
    vlm = torch.randn(2, 2048)
    
    bpm = model(depth, flow, audio, vlm_emb=vlm)
    loss = model.compute_alignment_loss(depth, flow, audio, vlm_emb=vlm)
    
    print(f"BPM Out: {bpm.shape}")
    print(f"Alignment InfoNCE Loss: {loss.item()}")
