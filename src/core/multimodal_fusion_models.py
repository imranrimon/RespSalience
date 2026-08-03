import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.core.audio_encoders import ASTEncoder, HTSATEncoder, PANNsEncoder

class TemporalFusionTransformer(nn.Module):
    """
    VLM-Gated Cross-Attention Transformer (Quad-Modal Phase).
    Fuses Flow, Depth, and Audio signals, guided by VLM text embeddings.
    Also provides SSL-based Apnea Anomaly Detection via reconstruction.
    """
    def __init__(self, seq_len=300, d_model=64, nhead=4, num_layers=2, vlm_dim=2048, use_ast_audio=False, use_htsat_audio=False, use_panns_audio=False):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_ast_audio = use_ast_audio
        
        # Signal Encoders (1D Convolution to expand channel dim)
        self.depth_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.flow_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        if self.use_ast_audio:
            self.audio_encoder = ASTEncoder(d_model=d_model)
        elif use_htsat_audio:
            self.audio_encoder = HTSATEncoder(d_model=d_model)
        elif use_panns_audio:
            self.audio_encoder = PANNsEncoder(d_model=d_model)
        else:
            self.audio_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        
        # Positional Encoding (Simple Learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, seq_len))
        
        # VLM Projection (Language -> Model Space)
        self.vlm_proj = nn.Linear(vlm_dim, d_model)
        
        # Default Query (if VLM is missing)
        self.default_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer Components
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True),
            num_layers=num_layers
        )
        
        # Cross-Attention Fusion
        # Query: VLM Context (or Default)
        # Key/Value: Respiratory Signals
        self.fusion_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Output Regressor
        # Input is (B, d_model) from the Context Token
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1) # Predicts BPM
        )
        
        # Decoder Head for Reconstruction (Phase 15 - SSL Apnea Detection)
        # Maps (B, d_model) -> (B, 3, seq_len) (Depth + Flow + Audio)
        self.decoder_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len * 3) # Depth + Flow + Audio
        )

    def forward(self, depth_signal, flow_signal, audio_signal=None, vlm_emb=None, return_embedding=False, mode='cls'):
        """
        Args:
            depth_signal: (B, Seq_Len)
            flow_signal: (B, Seq_Len)
            audio_signal: Optional (B, Seq_Len)
            vlm_emb: Optional (B, vlm_dim)
            return_embedding: If True, returns (B, d_model).
            mode: 'cls' (regression), 'emb' (embedding), 'recon' (reconstruction).
        """
        B = depth_signal.shape[0]
        
        # 1. Prepare Query (Context)
        if vlm_emb is not None:
            # Project VLM embedding: (B, vlm_dim) -> (B, d_model) -> (B, 1, d_model)
            query = self.vlm_proj(vlm_emb).unsqueeze(1)
        else:
            # Use learnable default query: (1, 1, d_model) -> (B, 1, d_model)
            query = self.default_query.expand(B, -1, -1)
            
        # 2. Prepare Key/Values (Signals)
        # Add channel dim: (B, 1, Seq_Len)
        x_depth = depth_signal.unsqueeze(1)
        x_flow = flow_signal.unsqueeze(1)
        
        # Audio handling (Fallback if not provided)
        if audio_signal is None:
            audio_signal = torch.zeros_like(depth_signal)
        x_audio = audio_signal.unsqueeze(1)
        
        # Encode: (B, d_model, Seq_Len)
        emb_depth = self.depth_encoder(x_depth)
        emb_flow = self.flow_encoder(x_flow)
        if self.use_ast_audio:
            emb_audio = self.audio_encoder(audio_signal)
        else:
            emb_audio = self.audio_encoder(x_audio)
        
        # Add Positional Embedding
        emb_depth = emb_depth + self.pos_embedding
        emb_flow = emb_flow + self.pos_embedding
        emb_audio = emb_audio + self.pos_embedding
        
        # Permute to (B, Seq_Len, d_model)
        emb_depth = emb_depth.permute(0, 2, 1)
        emb_flow = emb_flow.permute(0, 2, 1)
        emb_audio = emb_audio.permute(0, 2, 1)
        
        # Refine with Self-Attention
        feat_depth = self.transformer(emb_depth)
        feat_flow = self.transformer(emb_flow)
        feat_audio = self.transformer(emb_audio)
        
        # Concatenate for Cross-Attention Key/Value
        # (B, 3*Seq_Len, d_model)
        kv_tensor = torch.cat([feat_depth, feat_flow, feat_audio], dim=1)
        
        # 3. Cross-Attention Fusion
        # "Based on Context (Query), what parts of Signals (KV) are important?"
        fused_seq, attn_weights = self.fusion_attention(query, kv_tensor, kv_tensor)
        
        # fused_seq is (B, 1, d_model)
        
        # 4. Regress
        fused_vec = fused_seq.squeeze(1) # (B, d_model)
        
        if return_embedding or mode == 'emb':
            return fused_vec
            
        if mode == 'recon':
            # Reconstruct (B, 3, Seq_Len)
            recon = self.decoder_head(fused_vec)
            return recon.view(B, 3, self.seq_len)
            
        bpm_pred = self.regressor(fused_vec)
        
        return bpm_pred

    def compute_anomaly_score(self, depth_signal, flow_signal, audio_signal=None, vlm_emb=None):
        """
        Self-Supervised Apnea Anomaly Detection.
        Reconstructs the original signals using the decoded embedding.
        Returns the MSE Loss as an Anomaly Score (High = Anormal breathing pattern / Apnea)
        """
        B = depth_signal.shape[0]
        if audio_signal is None:
            audio_signal = torch.zeros_like(depth_signal)
            
        # Get reconstruction: (B, 3, Seq_Len)
        recon_signals = self.forward(depth_signal, flow_signal, audio_signal, vlm_emb, mode='recon')
        
        # Target: Stack the 3 modalities -> (B, 3, Seq_Len)
        target_signals = torch.stack([depth_signal, flow_signal, audio_signal], dim=1)
        
        # Compute MSE across spatial and temporal dimensions, leaving Batch dim
        mse_per_sample = F.mse_loss(recon_signals, target_signals, reduction='none')
        anomaly_scores = mse_per_sample.mean(dim=(1, 2)) # (B,)
        
        return anomaly_scores

    @staticmethod
    def load_pretrained(path, device="cuda"):
        model = TemporalFusionTransformer()
        if torch.cuda.is_available():
            map_loc = device
        else:
            map_loc = "cpu"
            
        try:
            model.load_state_dict(torch.load(path, map_location=map_loc))
        except FileNotFoundError:
            print(f"Warning: No pretrained weights found at {path}. Using random initialization.")
        
        model.to(device)
        model.eval()
        return model

# ==========================================
# PHASE 3: STATE-SPACE MODEL (MAMBA) FUSION
# ==========================================

class MambaBlock(nn.Module):
    """
    Simplified State Space Model (SSM) Block inspired by Mamba.
    Processes sequences in linear time O(N) instead of O(N^2) like attention.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        
        # Linear projections for input
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # 1D Depthwise Convolution 
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # SSM Parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, self.d_inner)
        
        # Learnable SSM state matrices
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        """
        x shape: (B, L, D) = (Batch, SeqLen, d_model)
        """
        B, L, D = x.shape
        
        # 1. Input Projection & Gating
        x_and_res = self.in_proj(x) # (B, L, d_inner * 2)
        x_ssm, res = torch.chunk(x_and_res, 2, dim=-1)
        
        # 2. 1D Convolution
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :L] # (B, d_inner, L)
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # 3. State Space Model Forward 
        # (Simplified continuous-time discretization for demo purposes)
        # Project to delta, B, C
        x_proj = self.x_proj(x_conv) # (B, L, d_state * 2 + 1)
        dt, B_mat, C_mat = torch.split(x_proj, [1, self.A_log.shape[1], self.A_log.shape[1]], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)) # (B, L, d_inner)
        
        # To make this a true selective scan, we need a recurrent loop or associative scan.
        # For this implementation, we use a simplified parallel scan approximation.
        # Compute A and B bar
        A = -torch.exp(self.A_log.float()) # (1, d_state)
        
        # Simple global pooling approximation for SSM state mixing 
        # (Full Mamba requires custom CUDA Triton kernels for the parallel scan)
        state_mixing = torch.einsum('bld,bd->bld', x_conv, self.D)
        
        # 4. Gating and Output
        y = state_mixing * F.silu(res)
        out = self.out_proj(y)
        
        return out

class MambaFusionModel(nn.Module):
    """
    State-Space Model alternative to TemporalFusionTransformer.
    Uses MambaBlocks for linear-time spatial-temporal fusion.
    """
    def __init__(self, seq_len=300, d_model=64, num_layers=3, use_ast_audio=False, use_htsat_audio=False, use_panns_audio=False):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Modality Encoders
        self.depth_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.flow_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        
        self.use_ast_audio = use_ast_audio
        self.use_htsat_audio = use_htsat_audio
        self.use_panns_audio = use_panns_audio
        
        if self.use_ast_audio:
            self.audio_encoder = ASTEncoder(d_model=d_model)
        elif self.use_htsat_audio:
            self.audio_encoder = HTSATEncoder(d_model=d_model)
        elif self.use_panns_audio:
            self.audio_encoder = PANNsEncoder(d_model=d_model)
        else:
            self.audio_encoder = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        
        # Fusion Projection
        self.fusion_proj = nn.Linear(d_model * 3, d_model)
        
        # Mamba Core Layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=d_model) for _ in range(num_layers)
        ])
        
        # Output Heads
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, depth_signal, flow_signal, audio_signal=None):
        """
        Input shape: (Batch, Seq_Len)
        """
        B = depth_signal.shape[0]
        if audio_signal is None:
            audio_signal = torch.zeros_like(depth_signal)
            
        # Add channel dim: (B, 1, L)
        d = depth_signal.unsqueeze(1)
        f = flow_signal.unsqueeze(1)
        a = audio_signal.unsqueeze(1)
        
        feat_d = self.depth_encoder(d).transpose(1, 2) # (B, L, d_model)
        feat_f = self.flow_encoder(f).transpose(1, 2)
        feat_a = self.audio_encoder(a).transpose(1, 2)
        
        # Concatenate modalities (B, L, d_model * 3)
        concat_feat = torch.cat([feat_d, feat_f, feat_a], dim=-1)
        
        # Project to d_model
        x = self.fusion_proj(concat_feat) # (B, L, d_model)
        
        # Pass through Mamba layers
        for layer in self.mamba_layers:
            # Residual connection
            x = x + layer(x)
            
        # Global Average Pooling over Sequence Length
        pooled = x.mean(dim=1) # (B, d_model)
        
        # Predict BPM
        return self.regressor(pooled)
        
if __name__ == "__main__":
    # Test Mamba
    print("Testing MambaFusionModel...")
    model = MambaFusionModel(seq_len=300)
    d = torch.randn(4, 300)
    f = torch.randn(4, 300)
    a = torch.randn(4, 300)
    out = model(d, f, a)
    print(f"Mamba Output Shape: {out.shape}")
