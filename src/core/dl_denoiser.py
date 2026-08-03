import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.functional as F
import math

class ResResidualBlock1D(nn.Module):
    """1D Residual Block for deep feature extraction without vanishing gradients."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResUNet1D(nn.Module):
    """
    1D U-Net with Residual blocks. 
    Excellent for mapping noisy drift sequences to clean respiratory waveforms.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super().__init__()
        
        # Encoder (Downsampling)
        self.enc1 = ResResidualBlock1D(in_channels, base_filters)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = ResResidualBlock1D(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool1d(2)
        
        self.enc3 = ResResidualBlock1D(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = ResResidualBlock1D(base_filters * 4, base_filters * 8)
        
        # Decoder (Upsampling & Skip Connections)
        self.upconv3 = nn.ConvTranspose1d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ResResidualBlock1D(base_filters * 8, base_filters * 4) # 4 from skip + 4 from upconv
        
        self.upconv2 = nn.ConvTranspose1d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResResidualBlock1D(base_filters * 4, base_filters * 2)
        
        self.upconv1 = nn.ConvTranspose1d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ResResidualBlock1D(base_filters * 2, base_filters)
        
        # Final Output Layer
        self.final_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)
        # Using Tanh because normalized physiological signals typically map bounded ranges
        self.out_act = nn.Tanh()

    def forward(self, x):
        # x shape: (Batch, Channels, Sequence_Length)
        
        # Encoding path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoding path with skip connections
        # F.interpolate handles non-power-of-2 sequence lengths if upconv size mismatches
        d3 = self.upconv3(b)
        if d3.shape[2] != e3.shape[2]: d3 = F.interpolate(d3, size=e3.shape[2], mode='linear')
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        if d2.shape[2] != e2.shape[2]: d2 = F.interpolate(d2, size=e2.shape[2], mode='linear')
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        if d1.shape[2] != e1.shape[2]: d1 = F.interpolate(d1, size=e1.shape[2], mode='linear')
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        return self.out_act(out)

class WaveNetAutoencoder(nn.Module):
    """
    Dilated Causal Convolution Autoencoder.
    Models long-range temporal dependencies without massive parameters.
    SOTA for generating clean audio/time-series.
    """
    def __init__(self, in_channels=1, out_channels=1, residual_channels=32, dilation_depth=6):
        super().__init__()
        
        self.initial_conv = nn.Conv1d(in_channels, residual_channels, kernel_size=1)
        
        self.dilated_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        
        for i in range(dilation_depth):
            dilation = 2 ** i
            # Casual/Dilated Conv
            self.dilated_convs.append(
                nn.Conv1d(residual_channels, residual_channels * 2, kernel_size=3, 
                          padding=dilation, dilation=dilation)
            )
            self.skip_convs.append(nn.Conv1d(residual_channels, residual_channels, kernel_size=1))
            if i < dilation_depth - 1:
                self.residual_convs.append(nn.Conv1d(residual_channels, residual_channels, kernel_size=1))
                
        # Bottleneck/Decoder collapse
        self.final_conv1 = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.final_conv2 = nn.Conv1d(residual_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.initial_conv(x)
        skip_connections = 0
        
        for i in range(len(self.dilated_convs)):
            residual = x
            
            # Gated Activation (Tanh * Sigmoid)
            filter_gate = self.dilated_convs[i](x)
            filter_gate = filter_gate[..., :x.shape[-1]]  # Crop padding
            
            f, g = torch.chunk(filter_gate, 2, dim=1)
            f = torch.tanh(f)
            g = torch.sigmoid(g)
            x_gated = f * g
            
            s = self.skip_convs[i](x_gated)
            skip_connections += s
            
            if i < len(self.residual_convs):
                x = self.residual_convs[i](x_gated) + residual
                
        # Final aggregation
        x = self.relu(skip_connections)
        x = self.final_conv1(x)
        x = self.relu(x)
        x = self.final_conv2(x)
        return self.tanh(x)

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, SeqLen, d_model)
        return x + self.pe[:, :x.size(1), :]

class TransformerAutoencoder1D(nn.Module):
    """
    1D Transformer Autoencoder.
    Uses global Self-Attention to capture periodic breathing cycles across the entire sequence.
    SOTA for handling complex periodic patterns with varying amplitudes.
    """
    def __init__(self, in_channels=1, out_channels=1, d_model=64, num_layers=4, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding1D(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, out_channels)
        self.out_act = nn.Tanh()

    def forward(self, x):
        # x shape: (Batch, Channels, SeqLen)
        x = x.transpose(1, 2) # (Batch, SeqLen, Channels)
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x)
        
        x = self.output_proj(x)
        
        x = x.transpose(1, 2) # Back to (Batch, Channels, SeqLen)
        return self.out_act(x)

# ==========================================
# 4. CONDITIONAL 1D DIFFUSION (PHASE 3 SOTA)
# ==========================================

class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, t):
        # t shape: (Batch,)
        half_dim = self.d_model // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.mlp(embeddings)

class DiffWaveBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, res_channels)
        self.conv = nn.Conv1d(res_channels, res_channels * 2, kernel_size=3, padding=dilation, dilation=dilation)
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)

    def forward(self, x, t_emb):
        t_proj = self.time_mlp(t_emb).unsqueeze(-1)
        x_in = x + t_proj
        
        out = self.conv(x_in)
        out = out[..., :x.shape[-1]]  # Cut padding
        filter_act, gate_act = out.chunk(2, dim=1)
        
        act = torch.tanh(filter_act) * torch.sigmoid(gate_act)
        
        res = self.res_conv(act)
        skip = self.skip_conv(act)
        
        return (x + res) * 0.7071, skip # Scale for stability

class DiffWaveNet(nn.Module):
    """
    DiffWave backbone for 1D physiological waveform generation/denoising.
    Extracts time-step embedding and passes it through dilated casual conv blocks.
    """
    def __init__(self, in_channels=2, out_channels=1, res_channels=64, skip_channels=64, num_layers=6):
        super().__init__()
        self.time_dim = 128
        self.init_conv = nn.Conv1d(in_channels, res_channels, kernel_size=1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        
        self.blocks = nn.ModuleList()
        # 2 cycles of dilation [1, 2, 4]
        cycle_length = 3
        for i in range(num_layers):
            dilation = 2 ** (i % cycle_length)
            self.blocks.append(DiffWaveBlock(res_channels, skip_channels, dilation, self.time_dim))
            
        self.skip_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)
        
    def forward(self, x, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1).float()
        t_emb = self.time_embed(t)
        x = self.init_conv(x)
        
        skips = 0
        for block in self.blocks:
            x, skip = block(x, t_emb)
            skips += skip
            
        out = skips / (len(self.blocks) ** 0.5)
        out = F.relu(out)
        out = self.skip_conv1(out)
        out = F.relu(out)
        out = self.skip_conv2(out)
        return out

class ConditionalDiffusion1D(nn.Module):
    """
    1D Denoising Diffusion Probabilistic Model (DDPM).
    Conditioned on the original noisy signal to iteratively generate the clean breathing waveform.
    """
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        self.net = DiffWaveNet(in_channels=2, out_channels=1) # 2: x_t + condition
        
        # Precompute diffusion schedule parameters
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
    def forward(self, condition):
        """Inference: Reverse Diffusion Process (Sampling)"""
        b, c, l = condition.shape
        device = condition.device
        
        # Start from pure noise
        x = torch.randn((b, 1, l), device=device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            
            # Predict noise added at step t
            model_input = torch.cat([x, condition], dim=1)
            predicted_noise = self.net(model_input, t.float())
            
            alpha = (1.0 - self.betas[i])
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise
            
        return torch.tanh(x) # Constrain output to [-1, 1]

    def compute_loss(self, clean_target, condition):
        """Training: Forward Diffusion Process mapping"""
        b = clean_target.shape[0]
        device = clean_target.device
        
        # Random time steps
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        
        # Generate random noise
        noise = torch.randn_like(clean_target)
        
        # Corrupt the clean target precisely to timestep t
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(b, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(b, 1, 1)
        
        x_noisy = sqrt_alpha_cumprod_t * clean_target + sqrt_one_minus_alpha_cumprod_t * noise
        
        # Model tries to predict the noise injected
        model_input = torch.cat([x_noisy, condition], dim=1)
        predicted_noise = self.net(model_input, t.float())
        
        return F.mse_loss(predicted_noise, noise)

if __name__ == "__main__":
    # Test tensor: Batch 4, 1 Channel, Seq 300
    dummy_input = torch.randn(4, 1, 300)
    
    print("Testing ResUNet-1D...")
    unet = ResUNet1D()
    out1 = unet(dummy_input)
    print(f"Output shape: {out1.shape}") # Expected: [4, 1, 300]
    
    print("Testing WaveNetAutoencoder...")
    wave = WaveNetAutoencoder()
    out2 = wave(dummy_input)
    print(f"Output shape: {out2.shape}") # Expected: [4, 1, 300]
    
    print("Testing TransformerAutoencoder1D...")
    transformer = TransformerAutoencoder1D()
    out3 = transformer(dummy_input)
    print(f"Output shape: {out3.shape}") # Expected: [4, 1, 300]
    
    print("Testing ConditionalDiffusion1D...")
    diffusion = ConditionalDiffusion1D(timesteps=10) # 10 steps for fast test
    out4 = diffusion(dummy_input) # Sampling forward pass
    loss = diffusion.compute_loss(torch.randn_like(dummy_input), dummy_input) # Train step
    print(f"Sampling shape: {out4.shape}")
    print(f"Train loss shape: {loss.shape} -> Value: {loss.item():.4f}")

# ==========================================
# 5. SPATIAL-TEMPORAL HYBRID (CNN-BiLSTM)
# ==========================================

class CNN_BiLSTM_Denoiser(nn.Module):
    """
    Classic SOTA Baseline for physiological time-series (pre-Transformer era).
    Uses 1D CNNs for local spatial/temporal feature extraction, followed by a 
    Bidirectional LSTM to model the forward and backward breathing phase shifts.
    """
    def __init__(self, in_channels=1, out_channels=1, cnn_filters=32, lstm_hidden=64, num_layers=2):
        super().__init__()
        
        # Spatial Feature Extractor (Local patterns)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2) # Sequence length reduces by 4x
        )
        
        # Temporal Sequence Mapper (Phase modeling)
        self.bilstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Reconstructor / Upsampler
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(lstm_hidden * 2, cnn_filters, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(cnn_filters, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Constrain output bounds
        )
        
    def forward(self, x):
        # x: (Batch, Channels, SeqLen)
        
        # Extract features
        x_cnn = self.cnn(x) # (Batch, Filters, SeqLen // 4)
        
        # Prepare for LSTM: (Batch, SeqLen', Features)
        x_seq = x_cnn.transpose(1, 2)
        
        # Run BiLSTM
        lstm_out, _ = self.bilstm(x_seq) # (Batch, SeqLen', Hidden*2)
        
        # Prepare for Upsampling: (Batch, Hidden*2, SeqLen')
        x_up = lstm_out.transpose(1, 2)
        
        # Reconstruct to original length
        out = self.upsample(x_up)
        
        # Handle potential padding mismatches from transpose convolutions
        if out.shape[2] != x.shape[2]:
            out = F.interpolate(out, size=x.shape[2], mode='linear')
            
        return out

class MambaDenoiser(nn.Module):
    """
    S4/Mamba-Backbone Denoiser. 
    State-Space Models are exceptionally fast for long context. 
    This denoiser takes the 1D signal and uses selective state spaces to reconstruct it.
    """
    def __init__(self, in_channels=1, out_channels=1, d_model=64, num_layers=4):
        super().__init__()
        self.d_model = d_model
        
        self.in_proj = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
        
        self.layers = nn.ModuleList([
            # Mamba/SSMs are causal sequence models; unidirectional LSTM is a better fallback 
            # and keeps input/output dimensions strictly at d_model for residual stacking.
            nn.LSTM(d_model, d_model, batch_first=True, bidirectional=False)
            for _ in range(num_layers)
        ])
        
        self.out_proj = nn.Conv1d(d_model, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        h = self.in_proj(x).transpose(1, 2)
        
        for layer in self.layers:
            h_out, _ = layer(h)
            h = h + h_out  # Residual connection
            
        return self.out_proj(h.transpose(1, 2))
        
if __name__ == "__main__":
    # Test CNN-BiLSTM
    dummy_input = torch.randn(4, 1, 300)
    print("Testing CNN_BiLSTM_Denoiser...")
    bilstm = CNN_BiLSTM_Denoiser()
    out5 = bilstm(dummy_input)
    print(f"Output shape: {out5.shape}")
