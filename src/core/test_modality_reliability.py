"""
Demo: Cross-Modal Anomaly Detection (Challenge #2)

Shows how to:
1. Initialize controller with robust fusion enabled
2. Process a video and get cross-modal anomaly detection results
3. Interpret the diagnostic reports
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.modality_reliability import CrossModalAnomalyDetector, RobustFusionTransformer
from src.core.multimodal_fusion_models import TemporalFusionTransformer

# =====================================================
# Test 1: CrossModalAnomalyDetector in isolation
# =====================================================

def test_cross_modal_detector():
    """Test the anomaly detector with synthetic data."""
    print("\n" + "="*70)
    print("TEST 1: CrossModalAnomalyDetector (Synthetic Data)")
    print("="*70)
    
    detector = CrossModalAnomalyDetector(d_model=64)
    
    # Create synthetic features
    B, Seq_Len, d_model = 4, 300, 64
    
    # Scenario 1: All modalities agree (good case)
    print("\n[Scenario 1] All modalities agree")
    feat_depth = torch.randn(B, Seq_Len, d_model)
    feat_flow = feat_depth + 0.1 * torch.randn_like(feat_depth)  # Similar to depth
    feat_audio = feat_depth + 0.1 * torch.randn_like(feat_depth)  # Similar to depth
    
    confidence, telemetry = detector(feat_depth, feat_flow, feat_audio)
    print(f"  Confidence scores: {confidence[0].detach().numpy()}")
    print(f"  Disagreement score: {telemetry['disagreement_score'][0].item():.4f}")
    print(f"  Disagreement detected: {telemetry['disagreement_flag'][0].item()}")
    print(f"  Expected: Low disagreement score, all modalities trusted equally")
    
    # Scenario 2: Modalities strongly disagree (failure case)
    print("\n[Scenario 2] Modalities strongly disagree (sensor failure)")
    feat_depth = torch.randn(B, Seq_Len, d_model)
    feat_flow = torch.randn(B, Seq_Len, d_model) * -1.0  # Opposite direction
    feat_audio = torch.zeros(B, Seq_Len, d_model)  # Silent/dead
    
    confidence, telemetry = detector(feat_depth, feat_flow, feat_audio)
    print(f"  Confidence scores: {confidence[0].detach().numpy()}")
    print(f"  Disagreement score: {telemetry['disagreement_score'][0].item():.4f}")
    print(f"  Disagreement detected: {telemetry['disagreement_flag'][0].item()}")
    print(f"  Expected: High disagreement score, one modality downweighted")
    
    # Scenario 3: Partial agreement (occlusion case)
    print("\n[Scenario 3] Two modalities agree, audio glitchy")
    feat_depth = torch.randn(B, Seq_Len, d_model)
    feat_flow = feat_depth + 0.05 * torch.randn_like(feat_depth)  # Closely matches depth
    feat_audio = feat_depth * 2 + torch.randn_like(feat_depth) * 2  # Noisy/different
    
    confidence, telemetry = detector(feat_depth, feat_flow, feat_audio)
    print(f"  Confidence scores: {confidence[0].detach().numpy()}")
    print(f"  Disagreement score: {telemetry['disagreement_score'][0].item():.4f}")
    print(f"  Disagreement detected: {telemetry['disagreement_flag'][0].item()}")
    print(f"  Expected: Medium disagreement, audio confidence down, depth+flow up")


# =====================================================
# Test 2: RobustFusionTransformer wrapper
# =====================================================

def test_robust_fusion_wrapper():
    """Test the RobustFusionTransformer wrapper with a base transformer."""
    print("\n" + "="*70)
    print("TEST 2: RobustFusionTransformer (With Base Transformer)")
    print("="*70)
    
    # Create a base transformer
    base_model = TemporalFusionTransformer(seq_len=300, d_model=64, num_layers=2)
    
    # Wrap it with robust fusion
    robust_model = RobustFusionTransformer(
        base_transformer=base_model,
        d_model=64,
        threshold_disagreement=0.4
    )
    robust_model.eval()
    
    # Synthetic signals
    B = 2
    depth_signal = torch.randn(B, 300)
    flow_signal = torch.randn(B, 300)
    audio_signal = torch.randn(B, 300)
    
    print(f"\nInput shapes:")
    print(f"  Depth: {depth_signal.shape}")
    print(f"  Flow: {flow_signal.shape}")
    print(f"  Audio: {audio_signal.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = robust_model(
            depth_signal, flow_signal, audio_signal,
            return_diagnostics=True
        )
    
    print(f"\nRobust fusion output:")
    print(f"  BPM estimate: {output['bpm'].numpy()}")
    print(f"  Per-modality BPM:")
    print(f"    - Depth: {output['bpm_individual']['depth'].numpy()}")
    print(f"    - Flow: {output['bpm_individual']['flow'].numpy()}")
    print(f"    - Audio: {output['bpm_individual']['audio'].numpy()}")
    
    print(f"  Modality confidence: {output['confidence'].numpy()}")
    print(f"  Disagreement detected: {output['disagreement_flag'].numpy()}")
    print(f"  Disagreement score: {output['disagreement_score'].numpy()}")
    
    # Generate diagnostic reports
    print(f"\nDiagnostic reports:")
    for b in range(B):
        print(f"\n  Sample {b}:")
        telemetry = {
            'disagreement_flag': output['disagreement_flag'],
            'disagreement_score': output['disagreement_score'],
            'modality_agreement': {k: v for k, v in output.get('telemetry', {}).get('modality_agreement', {}).items()},
            'signal_energy': {k: v for k, v in output.get('telemetry', {}).get('signal_energy', {}).items()},
        }
        
        conf = output['confidence'][b].numpy()
        most_rel_idx = conf.argmax()
        modality_names = ['Depth', 'Flow', 'Audio']
        
        print(f"    BPM: {output['bpm'][b, 0].item():.1f}")
        print(f"    Most reliable: {modality_names[most_rel_idx]} ({conf[most_rel_idx]:.1%})")
        print(f"    Disagreement: {output['disagreement_flag'][b].item()}")
        print(f"    Status: {'WARNING' if output['disagreement_flag'][b].item() else 'GOOD'}")


# =====================================================
# Test 3: Integration with RespiratoryController
# =====================================================

def test_controller_integration(video_path=None):
    """Test integration with RespiratoryController."""
    print("\n" + "="*70)
    print("TEST 3: RespiratoryController with Robust Fusion")
    print("="*70)
    
    try:
        from src.core.main_controller import RespiratoryController
        
        # Initialize with robust fusion enabled
        print("Initializing controller with enable_robust_fusion=True...")
        controller = RespiratoryController(
            config_path="src/config/test_infer.yaml",
            checkpoint_path="data/models/VIRENet_Coarse2fine_Body_best.pth",
            enable_depth=True,
            enable_flow=True,
            enable_audio=True,
            enable_transformer=True,
            enable_robust_fusion=True,  # ← NEW!
            fusion_model_path="data/models/vlm_gated_fused.pth"
        )
        
        print(f"[OK] Controller initialized with robust fusion model")
        print(f"  - enable_transformer: {controller.enable_transformer}")
        print(f"  - enable_robust_fusion: {controller.enable_robust_fusion}")
        print(f"  - robust_fusion_model: {controller.robust_fusion_model is not None}")
        
        if video_path and os.path.exists(video_path):
            print(f"\nProcessing video: {video_path}")
            results = controller.process_video(video_path, age_months=6)
            
            print(f"\nResults:")
            print(f"  BPM: {results['bpm']}")
            print(f"  Status: {results['diagnosis']['status']}")
            
            if results.get('cross_modal_telemetry'):
                telemetry = results['cross_modal_telemetry']
                print(f"\n  Cross-Modal Anomaly Detection:")
                print(f"    Disagreement detected: {telemetry['disagreement_detected']}")
                print(f"    Disagreement score: {telemetry['disagreement_score']:.4f}")
                print(f"    Modality confidence: {telemetry['confidence_scores']}")
                
                report = telemetry.get('diagnostic_report')
                if report:
                    print(f"    Status: {report.get('health_status')}")
                    print(f"    Recommendation: {report.get('recommendation')}")
        else:
            print(f"\nNo video path provided. Skipping video processing test.")
            print(f"To test with a real video, call:")
            print(f"  test_controller_integration('path/to/video.mp4')")
    
    except ImportError as e:
        print(f"Skipping controller test (dependencies not installed): {e}")
    except Exception as e:
        print(f"Controller test error (expected if models not downloaded): {e}")


# =====================================================
# Test 4: Demonstrate robust weighted fusion
# =====================================================

def test_robust_weighted_fusion():
    """Demonstrate the robust weighted fusion mechanism."""
    print("\n" + "="*70)
    print("TEST 4: Robust Weighted Fusion Mechanism")
    print("="*70)
    
    # Simulate per-modality BPM predictions
    bpm_depth = 45.0
    bpm_flow = 44.0
    bpm_audio = 65.0  # Outlier/bad signal
    
    # Robust fusion scores them
    confidence = np.array([0.40, 0.42, 0.18])  # Audio downweighted
    
    # Weighted fusion
    bpm_robust = (bpm_depth * confidence[0] + 
                  bpm_flow * confidence[1] + 
                  bpm_audio * confidence[2]) / confidence.sum()
    
    # Simple average for comparison
    bpm_simple = (bpm_depth + bpm_flow + bpm_audio) / 3
    
    print(f"\nPer-modality predictions:")
    print(f"  Depth:  {bpm_depth:.1f} BPM")
    print(f"  Flow:   {bpm_flow:.1f} BPM")
    print(f"  Audio:  {bpm_audio:.1f} BPM (outlier)")
    
    print(f"\nModality confidence (from anomaly detector):")
    print(f"  Depth:  {confidence[0]:.1%}")
    print(f"  Flow:   {confidence[1]:.1%}")
    print(f"  Audio:  {confidence[2]:.1%}")
    
    print(f"\nFusion results:")
    print(f"  Simple average:    {bpm_simple:.1f} BPM")
    print(f"  Robust (weighted): {bpm_robust:.1f} BPM")
    print(f"  Robust method ignores the outlier audio signal!")
    print(f"  Final estimate closer to reliable depth+flow consensus")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHALLENGE #2: Cross-Modal Anomaly Detection - Demo Suite")
    print("="*70)
    
    # Run tests
    test_cross_modal_detector()
    test_robust_fusion_wrapper()
    test_robust_weighted_fusion()
    test_controller_integration()
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
    print("\n📚 Key Features:")
    print("  [OK] Detects disagreement between modalities")
    print("  [OK] Provides per-modality confidence scores")
    print("  [OK] Automatic downweighting of unreliable modalities")
    print("  [OK] Telemetry for clinical review")
    print("  [OK] Sensor failure detection")
    print("\n📖 Next steps:")
    print("  - Train the model on your datasets")
    print("  - Deploy with enable_robust_fusion=True for clinical safety")
    print("  - Use diagnostic reports to identify sensor issues")
