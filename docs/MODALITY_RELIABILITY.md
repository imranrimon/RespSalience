# Cross-Modal Anomaly Detection (Challenge #2)

## Overview

**Cross-Modal Anomaly Detection** is a safety-critical feature that monitors disagreement between the three respiratory monitoring modalities (depth, optical flow, audio) to flag sensor failures, misalignments, or occlusions in real-time.

**Why it matters for clinical deployment:**
- Sensors fail silently → system trusts wrong data → misdiagnosis
- RobustFusionTransformer detects when modalities disagree
- Automatically downweights unreliable modalities
- Provides detailed telemetry for clinical review

---

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│ RespiratoryController (Main entry point)                     │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    Fusion Model          Robust Fusion Model
    (Base: Transformer)   (NEW - Wraps base model)
         │                       │
         ├─► Depth Encoder       │
         ├─► Flow Encoder    CrossModalAnomalyDetector
         ├─► Audio Encoder       │
         └─► Transformer    ├─► Pairwise Attention
                            ├─► Confidence Scoring
                            ├─► Sensor Health Detection
                            └─► Per-Modality Regressors
                                 ├─► Depth Regressor
                                 ├─► Flow Regressor
                                 └─► Audio Regressor

Output:
- bpm: Robust BPM estimate (confidently weighted fusion)
- confidence: Per-modality confidence scores
- disagreement_flag: Boolean - was disagreement detected?
- disagreement_score: Magnitude of disagreement (0-1)
- telemetry: Detailed diagnostic info for clinician review
```

### CrossModalAnomalyDetector

Measures pairwise agreement between modalities:

1. **Cross-Attention Matching** (measures how well one modality predicts another):
   ```
   depth_flow_sim = cosine_similarity(depth_features, flow_features)
   depth_audio_sim = cosine_similarity(depth_features, audio_features)
   flow_audio_sim = cosine_similarity(flow_features, audio_features)
   ```

2. **Confidence Scoring** (learns which modalities are trustworthy):
   ```python
   confidence = softmax(
       network([depth_flow_attn, depth_audio_attn, flow_audio_attn])
   )
   # Output: (batch, 3) where each row sums to 1.0
   # If all same: [0.33, 0.33, 0.33] = "all equally trusted"
   # If one fails: [0.40, 0.42, 0.18] = "audio downweighted"
   ```

3. **Disagreement Detection** (when std of confidences exceeds threshold):
   ```python
   disagreement_score = std(confidence[0, :])  # (batch,)
   disagreement_flag = disagreement_score > threshold  # (batch,) bool
   ```

4. **Sensor Failure Detection** (detects complete information loss):
   ```
   signal_energy = sum(features^2, dim=-1).mean(dim=-1)
   # Low energy = silent sensor (no data)
   ```

### RobustFusionTransformer

Wrapper that:
1. Extracts embeddings from base transformer
2. Runs cross-modal anomaly detector
3. Gets per-modality predictions
4. **Robust weighted fusion**: `BPM = sum(BPM_i * confidence_i)`
5. Returns detailed diagnostics

---

## Usage

### Basic: Enable Robust Fusion in Controller

```python
from src.core.main_controller import RespiratoryController

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

# Process video
results = controller.process_video("video.mp4", age_months=6)

# Check for anomalies
if results.get('cross_modal_telemetry'):
    telemetry = results['cross_modal_telemetry']
    if telemetry['disagreement_detected']:
        print(f"⚠️  WARNING: {telemetry['diagnostic_report']['health_status']}")
        print(f"   {telemetry['diagnostic_report']['recommendation']}")
```

### Advanced: Direct Use

```python
from src.core.modality_reliability import RobustFusionTransformer
from src.core.multimodal_fusion_models import TemporalFusionTransformer
import torch

# Wrap existing fusion model
base_model = TemporalFusionTransformer(...)
robust_model = RobustFusionTransformer(
    base_transformer=base_model,
    d_model=64,
    threshold_disagreement=0.4
)

# Forward pass with diagnostics
depth_signal = torch.randn(batch_size, 300)  # (B, Seq_Len)
flow_signal = torch.randn(batch_size, 300)
audio_signal = torch.randn(batch_size, 300)

output = robust_model(
    depth_signal, flow_signal, audio_signal,
    return_diagnostics=True
)

# Access results
bpm = output['bpm']  # (B, 1) - robust estimate
confidence = output['confidence']  # (B, 3) - per-modality scores
disagreement_flag = output['disagreement_flag']  # (B,) bool
telemetry = output['telemetry']  # Full diagnostic info

# Generate human-readable diagnostic report
reports = robust_model.get_diagnostic_report(output)
for report in reports:
    print(f"BPM: {report['bpm_estimate']:.1f}")
    print(f"Status: {report['health_status']}")
    print(f"Most reliable: {report['most_reliable_modality']}")
    if report['disagreement_detected']:
        print(f"⚠️  {report['recommendation']}")
```

---

## Interpretation Guide

### Confidence Scores

Represent how much each modality contributes to the final BPM estimate.

**Interpretation:**
```
[0.33, 0.33, 0.34]  ✓ All modalities equally trusted → HIGH CONFIDENCE
[0.40, 0.42, 0.18]  ⚠️  Audio downweighted → Audio sensor may be noisy
[0.50, 0.48, 0.02]  🔴 Audio failed → Do not use audio in this case
```

### Disagreement Score

Standard deviation of confidence scores (0 = perfect agreement, higher = more disagreement).

**Interpretation:**
```
disagreement_score < 0.10  ✓ Excellent modality agreement
disagreement_score 0.10-0.25  ~ Acceptable (normal variation)
disagreement_score 0.25-0.40  ⚠️  Moderate disagreement (check sensor calibration)
disagreement_score > 0.40    🔴 Critical disagreement (sensor malfunction likely)
```

### Modality Agreement Scores

Cosine similarity between pairwise modalities (-1 = opposite, 0 = uncorrelated, 1 = identical).

**Interpretation:**
```
depth_vs_flow: 0.85      ✓ Good agreement (respiratory motion captured similarly)
depth_vs_audio: 0.45     ~ Moderate (breath sounds and chest movement timing offset)
flow_vs_audio: -0.15     🔴 Negative correlation (flow and audio contradict each other)
```

### Signal Energy

Average squared magnitude of feature embeddings (detects silent/dead sensors).

**Interpretation:**
```
High energy (0.8+)   ✓ Active signal
Medium energy (0.3-0.8)  ~ Weak signal (low motion or noise)
Low energy (<0.1)    🔴 Dead sensor (no breathing motion detected)
```

---

## Diagnostic Report Example

```
{
  'sample_id': 0,
  'bpm_estimate': 42.5,
  'per_modality_bpm': {
    'depth': 43.2,
    'flow': 41.8,
    'audio': 65.4  ← Outlier
  },
  'modality_confidence': {
    'depth': 0.42,
    'flow': 0.43,
    'audio': 0.15   ← Downweighted
  },
  'most_reliable_modality': 'Optical Flow',
  'disagreement_detected': True,
  'disagreement_score': 0.38,
  
  'health_status': 'WARNING: Modalities Disagree',
  'recommendation': (
    'Modalities strongly disagree (score=0.38). '
    'Relying most on Optical Flow (confidence=43%). '
    'Verify sensor calibration if possible.'
  ),
  
  'modality_agreement_scores': {
    'depth_vs_flow': 0.82,       ✓ Good
    'depth_vs_audio': 0.28,      ~ Moderate
    'flow_vs_audio': -0.05       🔴 Poor
  },
  
  'signal_energy': {
    'depth': 0.75,     ✓ Active
    'flow': 0.73,      ✓ Active
    'audio': 0.12      🔴 Weak
  }
}
```

---

## Clinical Scenarios

### Scenario 1: Normal Operation
```
All modalities agree → Confidence [0.33, 0.34, 0.33]
→ BPM = 42.5 (highly reliable)
→ Status: GOOD
→ Action: Trust the measurement, proceed with diagnosis
```

### Scenario 2: Partial Occlusion (Cloth Over Camera)
```
Depth/Flow degraded but audio OK:
  - Depth BPM: 35 (low motion detected)
  - Flow BPM: 38 (optical flow fails on cloth)
  - Audio BPM: 48 (still clear)
→ Confidence: [0.20, 0.22, 0.58]
→ BPM = 45.5 (relies mostly on audio)
→ Status: WARNING - Modalities disagree
→ Action: Check camera for occlusions, recalibrate depth sensor
```

### Scenario 3: Microphone Failure
```
Depth/Flow normal but audio dies:
  - Depth BPM: 44 (normal)
  - Flow BPM: 43 (normal)
  - Audio BPM: 0 (no signal)
→ Confidence: [0.49, 0.50, 0.01]
→ BPM = 43.5 (ignores dead audio)
→ Status: WARNING - Modalities disagree
→ Action: Check microphone connections, replace if necessary
```

### Scenario 4: VLM Context Helps
```
Infant crying detected by VLM:
  - Without VLM: All modalities at normal weight
  - With VLM: Fusion transformer can upweight audio (crying sounds)
    and downweight optical flow (motion artifacts from movement)
→ Better diagnosis despite noisy signals
```

---

## Testing

Run the demo suite:

```bash
python src/core/test_modality_reliability.py
```

This will:
1. Test detector with synthetic data (agreement, disagreement, partial cases)
2. Test robust fusion wrapper with base transformer
3. Demonstrate robust weighted fusion mechanism
4. Integration test with RespiratoryController (if models available)

---

## Training / Calibration

The cross-modal anomaly detector trains end-to-end with the fusion transformer:

```python
# During training:
# 1. Forward pass through fusion transformer
# 2. Cross-modal detector runs
# 3. Confidence scores weight the modality predictions
# 4. Loss backpropagates through both networks

loss = MSE(robust_bpm, ground_truth_bpm)
loss.backward()  # Updates both fusion_model and anomaly_detector

# The detector learns which modality combinations are trustworthy!
```

**Key insight:** The detector automatically learns from the training data **when** each modality is reliable. No manual labeling needed.

---

## Limitations & Future Work

### Current Limitations
1. **Threshold tuning**: `threshold_disagreement=0.4` is empirical; optimal value depends on your data
2. **Synthetic failures only**: Tested on simulated sensor failures, not real hardware faults
3. **No temporal continuity**: Each frame assessed independently; could smooth disagreement scores over time

### Future Enhancements
1. **Temporal smoothing**: LSTM/Mamba to track disagreement trends (e.g., "audio degrading over time")
2. **Per-dataset calibration**: Learn threshold from your specific sensor hardware
3. **Meta-learning**: Few-shot adaptation when deploying to new camera types
4. **Hardware-in-the-loop testing**: Test with real sensor failures (simulated + actual hardware)
5. **Attention visualization**: Show clinicians *which video regions* caused disagreement detection

---

## References

- **Cross-modal fusion**: Baltrušaitis et al. (2018) "Multimodal Machine Learning: A Survey"
- **Robustness in medical AI**: Gianfrancesco et al. (2020) "Potential Biases in AI Systems for Chest Radiography"
- **Sensor fusion safety**: NASA Technical Reports on Multi-Modal Perception

---

## Questions?

For implementation questions, refer to:
- `modality_reliability.py` - Full source code with extensive comments
- `test_modality_reliability.py` - Usage examples for all scenarios
- `main_controller.py` - Integration into the full pipeline
