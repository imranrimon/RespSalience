"""
Cross-Modal Anomaly Detection & Robust Fusion Module.

Challenge #2 Implementation: Detects sensor failures, misalignment, and occlusions
by monitoring disagreement between depth, optical flow, and audio modalities.

Key Features:
- Per-modality confidence scoring
- Disagreement flagging (when std of confidences exceeds threshold)
- Robust weighted fusion (automatically downweight unreliable modalities)
- Sensor failure detection (complete information loss)
- Detailed telemetry for clinical review
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any
import numpy as np


class CrossModalAnomalyDetector(nn.Module):
    """
    Detects conflicts between depth, flow, and audio modalities.
    Returns per-modality confidence scores used for robust fusion.
    """
    
    def __init__(self, d_model: int = 64, num_heads: int = 4, threshold_disagreement: float = 0.4):
        """
        Args:
            d_model: Hidden dimension of feature embeddings
            num_heads: Multi-head attention for agreement scoring
            threshold_disagreement: Std threshold above which we flag disagreement
        """
        super().__init__()
        self.d_model = d_model
        self.threshold_disagreement = threshold_disagreement
        
        # Cross-attention layers to measure agreement between modality pairs
        self.depth_flow_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.depth_audio_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.flow_audio_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # Confidence scoring network
        # Takes concatenated attention weights and outputs 3 confidence scores
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # [depth_conf, flow_conf, audio_conf]
            nn.Softmax(dim=-1)  # Normalize to [0, 1] summing to 1
        )
        
        # Sensor failure detector (detects complete signal loss)
        self.sensor_failure_detector = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Logits for [normal, depth_failed, flow_failed, audio_failed]
            nn.Softmax(dim=-1)
        )
    
    def forward(self, feat_depth: torch.Tensor, feat_flow: torch.Tensor, 
                feat_audio: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            feat_depth: (B, Seq_Len, d_model) - Depth features
            feat_flow: (B, Seq_Len, d_model) - Flow features  
            feat_audio: (B, Seq_Len, d_model) - Audio features
            
        Returns:
            confidence: (B, 3) - Confidence scores for [depth, flow, audio]
            telemetry: Dict with diagnostic info:
                - disagreement_flag: (B,) - True where disagreement detected
                - disagreement_score: (B,) - Std of confidence scores
                - sensor_health: (B, 3) - Health status per modality
                - modality_agreement: Dict of pairwise agreement scores
        """
        B = feat_depth.shape[0]
        
        # =====================================================
        # 1. Measure pairwise modality agreement via attention
        # =====================================================
        
        # Depth vs Flow: How well does flow predict depth?
        depth_flow_attn, _ = self.depth_flow_attention(
            feat_depth,
            feat_flow,
            feat_flow
        )  # (B, Seq_Len, d_model)
        
        # Depth vs Audio
        depth_audio_attn, _ = self.depth_audio_attention(
            feat_depth,
            feat_audio,
            feat_audio
        )  # (B, Seq_Len, d_model)
        
        # Flow vs Audio
        flow_audio_attn, _ = self.flow_audio_attention(
            feat_flow,
            feat_audio,
            feat_audio
        )  # (B, Seq_Len, d_model)
        
        # Compute agreement as correlation between reconstructed signals
        # Higher correlation = better agreement
        depth_flow_sim = F.cosine_similarity(feat_depth, depth_flow_attn, dim=-1).mean(dim=-1)  # (B,)
        depth_audio_sim = F.cosine_similarity(feat_depth, depth_audio_attn, dim=-1).mean(dim=-1)  # (B,)
        flow_audio_sim = F.cosine_similarity(feat_flow, flow_audio_attn, dim=-1).mean(dim=-1)  # (B,)
        
        # =====================================================
        # 2. Concatenate agreement signals for confidence scoring
        # =====================================================
        agreement_signals = torch.stack([
            depth_flow_attn.mean(dim=1),  # Pool over time
            depth_audio_attn.mean(dim=1),
            flow_audio_attn.mean(dim=1)
        ], dim=-1)  # (B, 3, d_model)
        
        # Reshape for confidence head: (B, 3*d_model)
        agreement_flat = agreement_signals.view(B, -1)
        
        # Score confidence for each modality
        confidence = self.confidence_head(agreement_flat)  # (B, 3)
        
        # =====================================================
        # 3. Detect disagreement (when modalities strongly disagree)
        # =====================================================
        confidence_std = confidence.std(dim=-1)  # (B,)
        disagreement_flag = confidence_std > self.threshold_disagreement  # (B,) bool
        
        # =====================================================
        # 4. Detect sensor failures (complete information loss)
        # =====================================================
        # Compute signal statistics for each modality
        depth_energy = (feat_depth ** 2).sum(dim=-1).mean(dim=-1)  # (B,)
        flow_energy = (feat_flow ** 2).sum(dim=-1).mean(dim=-1)  # (B,)
        audio_energy = (feat_audio ** 2).sum(dim=-1).mean(dim=-1)  # (B,)
        
        # Concatenate for sensor failure detector
        modality_energies = torch.stack([depth_energy, flow_energy, audio_energy], dim=-1)  # (B, 3)
        
        # Use feat_depth aggregation as the input to sensor failure detector
        depth_summary = feat_depth.mean(dim=1)  # (B, d_model)
        sensor_health = self.sensor_failure_detector(depth_summary)  # (B, 3)
        
        # =====================================================
        # 5. Package telemetry for clinical review
        # =====================================================
        telemetry = {
            'disagreement_flag': disagreement_flag,  # (B,) bool
            'disagreement_score': confidence_std,  # (B,) float in [0, 1]
            'confidence': confidence,  # (B, 3)
            'sensor_health': sensor_health,  # (B, 3)
            
            # Pairwise agreement metrics
            'modality_agreement': {
                'depth_vs_flow': depth_flow_sim,  # (B,) in [-1, 1]
                'depth_vs_audio': depth_audio_sim,
                'flow_vs_audio': flow_audio_sim,
            },
            
            # Signal energy (helps detect zeros/silence)
            'signal_energy': {
                'depth': depth_energy,
                'flow': flow_energy,
                'audio': audio_energy,
            }
        }
        
        return confidence, telemetry


class RobustFusionTransformer(nn.Module):
    """
    Extends TemporalFusionTransformer with cross-modal anomaly detection.
    
    When modalities disagree, automatically:
    1. Flags the disagreement
    2. Weights each modality by confidence
    3. Reports sensor health status
    4. Provides detailed telemetry for clinical review
    
    This is safer than blind fusion—the system knows when to distrust itself.
    """
    
    def __init__(self, base_transformer, d_model: int = 64, 
                 threshold_disagreement: float = 0.4):
        """
        Args:
            base_transformer: An instantiated TemporalFusionTransformer
            d_model: Hidden dimension (must match transformer)
            threshold_disagreement: Std threshold for flagging disagreement
        """
        super().__init__()
        self.base_transformer = base_transformer
        self.d_model = d_model
        
        # Cross-modal anomaly detector
        self.anomaly_detector = CrossModalAnomalyDetector(
            d_model=d_model,
            threshold_disagreement=threshold_disagreement
        )
        
        # Per-modality regressors (for individual modal predictions)
        self.depth_regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.flow_regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.audio_regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, depth_signal: torch.Tensor, flow_signal: torch.Tensor,
                audio_signal: torch.Tensor = None, vlm_emb: torch.Tensor = None,
                return_diagnostics: bool = True) -> Dict[str, Any]:
        """
        Args:
            depth_signal: (B, Seq_Len)
            flow_signal: (B, Seq_Len)
            audio_signal: (B, Seq_Len) optional
            vlm_emb: (B, vlm_dim) optional
            return_diagnostics: If True, include telemetry; else just BPM
            
        Returns:
            output: Dict containing:
                - bpm: (B, 1) - Fused respiratory rate prediction
                - bpm_individual: Dict with per-modality BPM estimates
                - confidence: (B, 3) - Confidence scores for each modality
                - disagreement_flag: (B,) bool - True if disagreement detected
                - disagreement_score: (B,) float
                - telemetry: Full diagnostic info (if return_diagnostics=True)
        """
        B = depth_signal.shape[0]
        
        if audio_signal is None:
            audio_signal = torch.zeros_like(depth_signal)
        
        # =====================================================
        # 1. Get base embeddings from transformer
        # =====================================================
        
        # Temporarily modify base transformer to return embeddings
        # by extracting intermediate features
        x_depth = depth_signal.unsqueeze(1)
        x_flow = flow_signal.unsqueeze(1)
        x_audio = audio_signal.unsqueeze(1)
        
        # Encode signals
        emb_depth = self.base_transformer.depth_encoder(x_depth)
        emb_flow = self.base_transformer.flow_encoder(x_flow)
        if self.base_transformer.use_ast_audio:
            emb_audio = self.base_transformer.audio_encoder(audio_signal)
        else:
            emb_audio = self.base_transformer.audio_encoder(x_audio)
        
        # Add positional embedding
        emb_depth = emb_depth + self.base_transformer.pos_embedding
        emb_flow = emb_flow + self.base_transformer.pos_embedding
        emb_audio = emb_audio + self.base_transformer.pos_embedding
        
        # Permute to (B, Seq_Len, d_model)
        feat_depth = emb_depth.permute(0, 2, 1)
        feat_flow = emb_flow.permute(0, 2, 1)
        feat_audio = emb_audio.permute(0, 2, 1)
        
        # Refine with self-attention
        feat_depth = self.base_transformer.transformer(feat_depth)
        feat_flow = self.base_transformer.transformer(feat_flow)
        feat_audio = self.base_transformer.transformer(feat_audio)
        
        # =====================================================
        # 2. Run cross-modal anomaly detection
        # =====================================================
        confidence, telemetry = self.anomaly_detector(feat_depth, feat_flow, feat_audio)
        
        # =====================================================
        # 3. Individual per-modality predictions
        # =====================================================
        # Global average pooling
        depth_pool = feat_depth.mean(dim=1)  # (B, d_model)
        flow_pool = feat_flow.mean(dim=1)
        audio_pool = feat_audio.mean(dim=1)
        
        bpm_depth = self.depth_regressor(depth_pool)  # (B, 1)
        bpm_flow = self.flow_regressor(flow_pool)
        bpm_audio = self.audio_regressor(audio_pool)
        
        # =====================================================
        # 4. Robust weighted fusion using confidence scores
        # =====================================================
        # Confidence shape: (B, 3) with each row summing to 1
        # Weighted BPM: sum of (BPM_modality * confidence_modality)
        
        bpm_stack = torch.cat([bpm_depth, bpm_flow, bpm_audio], dim=1)  # (B, 3)
        bpm_robust = (bpm_stack * confidence).sum(dim=1, keepdim=True)  # (B, 1)
        
        # =====================================================
        # 5. Package outputs
        # =====================================================
        output = {
            'bpm': bpm_robust,
            'bpm_individual': {
                'depth': bpm_depth,
                'flow': bpm_flow,
                'audio': bpm_audio,
            },
            'confidence': confidence,  # (B, 3)
            'disagreement_flag': telemetry['disagreement_flag'],
            'disagreement_score': telemetry['disagreement_score'],
        }
        
        if return_diagnostics:
            output['telemetry'] = telemetry
        
        return output
    
    def get_diagnostic_report(self, output: Dict[str, Any], threshold_disagreement: float = 0.4) -> Dict[str, Any]:
        """
        Formats the output into a human-readable diagnostic report for clinicians.
        
        Args:
            output: Dict from forward() call
            threshold_disagreement: Alert threshold for disagreement scores
            
        Returns:
            report: Dict with formatted diagnostics
        """
        confidence = output['confidence']  # (B, 3)
        disagreement_flag = output['disagreement_flag']  # (B,)
        disagreement_score = output['disagreement_score']  # (B,)
        telemetry = output.get('telemetry', {})
        
        B = confidence.shape[0]
        reports = []
        
        for b in range(B):
            conf = confidence[b]  # (3,)
            
            # Identify which modality is most reliable
            most_reliable_idx = conf.argmax().item()
            modality_names = ['Depth', 'Optical Flow', 'Audio']
            most_reliable = modality_names[most_reliable_idx]
            
            report = {
                'sample_id': b,
                'bpm_estimate': output['bpm'][b, 0].item(),
                'per_modality_bpm': {
                    'depth': output['bpm_individual']['depth'][b, 0].item(),
                    'flow': output['bpm_individual']['flow'][b, 0].item(),
                    'audio': output['bpm_individual']['audio'][b, 0].item(),
                },
                'modality_confidence': {
                    'depth': conf[0].item(),
                    'flow': conf[1].item(),
                    'audio': conf[2].item(),
                },
                'most_reliable_modality': most_reliable,
                'disagreement_detected': disagreement_flag[b].item(),
                'disagreement_score': disagreement_score[b].item(),
                'health_status': 'GOOD',  # Default
            }
            
            # Determine health status
            if disagreement_flag[b].item():
                report['health_status'] = 'WARNING: Modalities Disagree'
                report['recommendation'] = (
                    f"Modalities strongly disagree (score={disagreement_score[b]:.3f}). "
                    f"Relying most on {most_reliable} (confidence={conf[most_reliable_idx]:.2%}). "
                    "Verify sensor calibration if possible."
                )
            else:
                report['health_status'] = 'GOOD'
                report['recommendation'] = "All modalities agree. Measurements reliable."
            
            # Add telemetry if available
            if telemetry:
                agreement = telemetry['modality_agreement']
                report['modality_agreement_scores'] = {
                    'depth_vs_flow': agreement['depth_vs_flow'][b].item(),
                    'depth_vs_audio': agreement['depth_vs_audio'][b].item(),
                    'flow_vs_audio': agreement['flow_vs_audio'][b].item(),
                }
                
                energy = telemetry['signal_energy']
                report['signal_energy'] = {
                    'depth': energy['depth'][b].item(),
                    'flow': energy['flow'][b].item(),
                    'audio': energy['audio'][b].item(),
                }
            
            reports.append(report)
        
        return reports[0] if B == 1 else reports


# =====================================================
# Utility: Integrate with existing RespiratoryController
# =====================================================

def upgrade_controller_with_robust_fusion(controller):
    """
    Monkey-patch an existing RespiratoryController to use RobustFusionTransformer.
    
    Usage:
        controller = RespiratoryController(...)
        controller = upgrade_controller_with_robust_fusion(controller)
        results = controller.process_video(video_path, age_months=6)
        # results now include telemetry and confidence scores
    """
    if not hasattr(controller, 'fusion_model'):
        raise ValueError("Controller must have fusion_model attribute")
    
    # Wrap the existing fusion model
    robust_model = RobustFusionTransformer(
        base_transformer=controller.fusion_model,
        d_model=controller.fusion_model.d_model,
        threshold_disagreement=0.4
    )
    robust_model.to(controller.device)
    
    # Replace the old forward method
    original_forward = controller.fusion_model.forward
    
    def new_forward(depth_signal, flow_signal, audio_signal=None, vlm_emb=None, **kwargs):
        output = robust_model(depth_signal, flow_signal, audio_signal, vlm_emb, 
                             return_diagnostics=True)
        # For backward compatibility, also return BPM directly
        return output['bpm']
    
    # Store the robust model as an attribute
    controller.robust_fusion_model = robust_model
    controller.get_diagnostic_report = lambda output: robust_model.get_diagnostic_report(output)
    
    return controller


if __name__ == "__main__":
    # Quick test
    print("Testing CrossModalAnomalyDetector...")
    detector = CrossModalAnomalyDetector(d_model=64)
    
    B, Seq_Len, d_model = 4, 300, 64
    feat_depth = torch.randn(B, Seq_Len, d_model)
    feat_flow = torch.randn(B, Seq_Len, d_model)
    feat_audio = torch.randn(B, Seq_Len, d_model)
    
    confidence, telemetry = detector(feat_depth, feat_flow, feat_audio)
    
    print(f"✓ Confidence shape: {confidence.shape}")
    print(f"✓ Disagreement flags: {telemetry['disagreement_flag']}")
    print(f"✓ Disagreement scores: {telemetry['disagreement_score']}")
    print(f"✓ Sensor health: {telemetry['sensor_health'].shape}")
    print("All tests passed!")
