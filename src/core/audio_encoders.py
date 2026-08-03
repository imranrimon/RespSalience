import torch
import torch.nn as nn

try:
    from transformers import ASTModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class ASTEncoder(nn.Module):
    """
    Audio Spectrogram Transformer (AST) feature extractor wrapper.
    Replaces the simple Conv1d mel-spectrogram encoder with a pretrained AST backbone.
    Takes 1D raw waveform or computed spectrogram features, processes them through AST,
    and projects the sequence of embeddings to the desired d_model dimension
    to match the Quad-Modal fusion dimension.
    """
    def __init__(self, d_model=64, pretrained_model="MIT/ast-finetuned-audioset-10-10-0.4593", freeze_backbone=True):
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library is required for ASTEncoder. Run: pip install transformers")
            
        # Load the pretrained AST model (without the classification head)
        self.ast = ASTModel.from_pretrained(pretrained_model)
        
        # AudioSet AST outputs 768-dim embeddings per time patch
        self.ast_hidden_size = self.ast.config.hidden_size 
        
        # We need to project the AST embeddings (768) down to our fusion d_model (e.g., 64)
        self.proj = nn.Linear(self.ast_hidden_size, d_model)
        
        # AST is huge. Freeze it by default to prevent overfitting on our small dataset
        if freeze_backbone:
            for param in self.ast.parameters():
                param.requires_grad = False
                
    def forward(self, input_values):
        """
        Args:
            input_values: (Batch, SeqLen) or (Batch, Freq, Time)
                 HuggingFace AST typically expects precomputed spectrograms 
                 processed by ASTFeatureExtractor of shape (B, 1024, 128).
                 If we feed raw 1D audio (B, L), we need to handle it.
                 For simplicity in this pipeline, we assume `input_values` are the raw 
                 1D waveforms extracted from the video, and we let the caller or 
                 internal logic reshape it.
                 
                 However, since our current pipeline provides `audio_signal` as a 1D sequence 
                 of length 300 (which was originally just rolling mean of spectrograms), 
                 we will upsample/reshape it here to mock a spectrogram structure for the AST 
                 to consume if it's 1D, or accept the true spectrogram.
        """
        B = input_values.shape[0]
        
        # If input is our 1D sequential mock audio of shape (B, SeqLen)
        if input_values.ndim == 2:
            # We inflate the 1D signal into a pseudo-spectrogram geometry (1024 time bins x 128 freq bins) 
            # expected by standard AST. In a real scenario, you'd compute the MelSpectrogram here.
            # (Batch, 1024, 128)
            seq_len = input_values.shape[1]
            
            # Simple projection to get 128 freq bins per time step
            pseudo_spec = input_values.unsqueeze(-1).expand(-1, -1, 128) # (B, 300, 128)
            
            # AST expects specific length (usually 1024). We interpolate temporal dim.
            pseudo_spec = pseudo_spec.transpose(1, 2) # (B, 128, 300)
            pseudo_spec = torch.nn.functional.interpolate(pseudo_spec, size=1024, mode='linear') # (B, 128, 1024)
            pseudo_spec = pseudo_spec.transpose(1, 2) # (B, 1024, 128)
            
            input_values = pseudo_spec

        # AST outputs: last_hidden_state, pooler_output
        outputs = self.ast(input_values)
        
        # last_hidden_state: (B, NumPatches, 768)
        # We project this to (B, NumPatches, d_model)
        x = self.proj(outputs.last_hidden_state)
        
        # We need to map NumPatches (usually 1214) back to our system's SeqLen (300)
        # Transpose for interpolate: (B, d_model, NumPatches)
        x = x.transpose(1, 2)
        x = torch.nn.functional.interpolate(x, size=300, mode='linear')
        
        # Back to (B, SeqLen, d_model) -> our fusion transformer expects (B, d_model, SeqLen) 
        # from the modality encoders, wait no.
        # Actually our current conv1D encoders output (B, d_model, SeqLen).
        
        return x # (B, d_model, 300)

class PANNsEncoder(nn.Module):
    """
    Proxy for Pretrained Audio Neural Networks (PANNs).
    Typically a ResNet38 or CNN14 trained on AudioSet.
    We implement a CNN14 style 1D mapping to represent the PANNs feature extraction logic.
    """
    def __init__(self, d_model=64):
        super().__init__()
        # CNN14 simplified proxy for 1D signals
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, d_model, kernel_size=1) 
        )
    def forward(self, x):
        # x: (B, SeqLen)
        if x.ndim == 2:
            x = x.unsqueeze(1) # (B, 1, SeqLen)
        return self.conv_blocks(x)

class HTSATEncoder(nn.Module):
    """
    Proxy for Hierarchical Token-Semantic Audio Transformer (HTS-AT).
    A Swin-Transformer based model for audio classification.
    We simulate the hierarchical structure using progressively downsampling Transformer layers
    followed by an up-sampling to match the sequence length.
    """
    def __init__(self, d_model=64):
        super().__init__()
        self.patch_embed = nn.Conv1d(1, d_model, kernel_size=4, stride=4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*2, batch_first=True)
        self.layer1 = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.downsample = nn.Conv1d(d_model, d_model*2, kernel_size=2, stride=2)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=d_model*2, nhead=8, dim_feedforward=d_model*4, batch_first=True)
        self.layer2 = nn.TransformerEncoder(encoder_layer2, num_layers=2)
        
        # Upsample back to B, d_model, SeqLen
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(d_model*2, d_model, kernel_size=2, stride=2),
            nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=4)
        )
        
    def forward(self, x):
        # x: (B, SeqLen) e.g., (B, 300)
        orig_len = x.shape[-1]
        if x.ndim == 2:
            x = x.unsqueeze(1)
            
        x = self.patch_embed(x) # (B, d_model, SeqLen/4)
        x = x.transpose(1, 2)
        x = self.layer1(x)
        x = x.transpose(1, 2)
        
        x = self.downsample(x) # (B, d_model*2, SeqLen/8)
        x = x.transpose(1, 2)
        x = self.layer2(x)
        x = x.transpose(1, 2)
        
        x = self.upsample(x)
        # Ensure length matches
        if x.shape[-1] != orig_len:
            x = nn.functional.interpolate(x, size=orig_len, mode='linear')
            
        return x # (B, d_model, SeqLen)

if __name__ == "__main__":
    print("Testing ASTEncoder with 1D pseudo-signal...")
    try:
        model = ASTEncoder(d_model=64)
        dummy_1d_audio = torch.randn(2, 300)
        out = model(dummy_1d_audio)
        print(f"AST Output Shape: {out.shape}")  # Expected (2, 64, 300)
    except ImportError as e:
        print(e)
        
    print("Testing PANNsEncoder...")
    panns = PANNsEncoder()
    out2 = panns(dummy_1d_audio)
    print(f"PANNs Output Shape: {out2.shape}") # Expected (2, 64, 300)
    
    print("Testing HTSATEncoder...")
    htsat = HTSATEncoder()
    out3 = htsat(dummy_1d_audio)
    print(f"HTS-AT Output Shape: {out3.shape}") # Expected (2, 64, 300)
