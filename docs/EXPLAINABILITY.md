# Explainable Diagnosis with Saliency Maps (Challenge #4)

## Overview

**Explainable Diagnosis** transforms your respiratory monitoring system from a "black box" into a **clinically interpretable decision-support tool**.

Instead of just saying "This infant has tachypnea", it shows:
- **WHERE** the model looked (spatial saliency maps)
- **WHEN** it was most certain (temporal attention)
- **WHY** it made the decision (reasoning and factors)
- **WHICH** sensors contributed most (modal attribution)

**Clinical Problem Solved:**
- Clinicians don't trust "magic box" AI (low adoption)
- Regulation requires interpretability (GDPR, FDA)
- Errors are hard to debug without explanations
- Different patient populations need different baselines

---

## Architecture

### Core Components

```
Respiratory Signal
    ↓
DiagnosisExplainer
    ├─ SaliencyGenerator: Which regions matter?
    ├─ TemporalExplainer: Which time steps matter?
    ├─ ModalExplainer: Which modalities contributed?
    └─ ExplanationReport: Synthesis with clinical reasoning
    ↓
ExplanationReport (Complete explanation)
    ├─ spatial_saliency: (H, W) importance map (0-1)
    ├─ temporal_saliency: (SeqLen,) frame importance
    ├─ hotspots: List of identified important regions
    ├─ modal_attribution: {depth: 0.6, flow: 0.3, audio: 0.1}
    ├─ reasoning: Human-readable clinical interpretation
    ├─ confidence_factors: Why are we confident/uncertain?
    └─ timestamp: When was this generated?
```

---

## Components in Detail

### 1. SaliencyGenerator (Spatial Attribution)

Identifies **which regions of the video** influenced the respiratory prediction.

**Method: Gradient-based with enhancement**

```python
generator = SaliencyGenerator(device="cuda")

# Compute pixel-level importance
saliency = generator.compute_spatial_saliency(
    input_tensor,      # (1, C, H, W) video frame
    model,             # Neural network
    target_output=0    # Which neuron to backprop from
)
# Returns: (H, W) normalized [0, 1]

# Enhance for visualization
enhanced = generator.enhance_saliency(saliency)

# Identify high-importance regions
hotspots = generator.identify_hotspots(
    enhanced,
    num_hotspots=3,    # Top-3 regions
    threshold=0.5
)
# Returns: [{"region": "chest", "importance": 0.95, ...}, ...]
```

**Clinical Use:**
- Chest center = good (normal breathing motion)
- Arms/face = unexpected (may indicate poor ROI)
- Edges = artifact artifacts (camera movement, shadows)

**Output Example:**
```
Saliency Map:
         ╔═══════════════════════╗
         ║   ╭──────────╮        ║
         ║   │ ▓▓▓▓▓▓▓▓ │  🔴 HOT(95%)
         ║   │ ▓░░░──── ║        ║
         ║   │ ▒▒░░──── ║        ║
         ║   │ ░░░░════ │        ║
         ║   ╰──────────╯        ║
         ╚═══════════════════════╝

Identified Regions (importance score):
1. Chest (upper body):      95%
2. Abdomen (center):        72%
3. Blanket/background:      20%
```

### 2. TemporalExplainer (Temporal Attribution)

Identifies **which frames (time steps) mattered most** in the sequence.

**Method: Attention weights extraction**

```python
temporal = TemporalExplainer(device="cuda")

# Extract temporal importance from attention layers
temporal_attn = temporal.get_temporal_attention(
    model,             # Should have attention layers
    input_sequence     # (1, SeqLen, Features) or (1, SeqLen, C, H, W)
)
# Returns: (SeqLen,) normalized importance for each frame

# Identify critical frames
critical = temporal.identify_critical_frames(
    temporal_attn,
    num_frames=3
)
# Returns: [{"frame_idx": 45, "importance": 0.18, "timing": "late"}, ...]
```

**Clinical Use:**
- **Recent frames (high importance)**: Model trusts latest data
- **All frames equal**: Model averaging over sequence (conservative)
- **Old frames (high importance)**: Model using baseline or sudden change detection

**Example:**
```
Video: 24 frames = 4 seconds at 6 fps
Frame importance over time:
  
  |
  |•         ••••
  | •       •••••
  |  •     •••••
  |   •   •••
  |    • •••
  |_________•_____
  0     12    24
  
Interpretation:
- Frames 0-8:  Low importance (baseline/warm-up)
- Frames 8-16: Medium (pattern detection)
- Frames 16-24: HIGH (recent status determines diagnosis)
```

### 3. ModalExplainer (Modality Attribution)

Determines **how much each modality (depth/flow/audio) contributed** to the final BPM prediction.

**Method: Permutation importance (ablation)**

```python
modal = ModalExplainer(device="cuda")

# Measure importance of each modality
attribution = modal.compute_modal_attribution(
    model,           # Fusion model
    depth_input,     # Depth signal tensor
    flow_input,      # Optical flow tensor
    audio_input      # Audio spectrogram tensor
)
# Returns: {"depth": 0.68, "flow": 0.22, "audio": 0.10}
```

**How it works:**
1. Get full model prediction with all modalities: BPM = 45.0
2. Drop modality A, get prediction without it: BPM = 42.5 (drop = 2.5)
3. Drop modality B, get prediction without it: BPM = 44.2 (drop = 0.8)
4. Drop modality C, get prediction without it: BPM = 44.7 (drop = 0.3)
5. Normalize: depth = 2.5/3.6 = 68%, flow = 0.8/3.6 = 22%, audio = 0.3/3.6 = 8%

**Clinical Use:**
- **Depth-dominant (>60%)**: Relying on chest motion. Good on clear RGB camera.
- **Flow-dominant (>40%)**: Relying on optical motion. Good when infant is moving.
- **Audio-dominant (>30%)**: Relying on breathing sounds. Good when visual fails.
- **Balanced**: Multiple cues agreeing. Most robust.

**Example Output:**
```
Patient: 6-month-old with ARI
Modal Attribution:
  Depth:  68% ████████████░░░░░░░
  Flow:   22% ████░░░░░░░░░░░░░░░
  Audio:  10% ██░░░░░░░░░░░░░░░░░

Interpretation: 
- Strong depth signal = Good quality video, chest visible
- Moderate flow = Some motion noise, but corroborates depth
- Low audio = No significant breathing sounds detected
- Verdict: Prediction is robust (multi-modal agreement)
```

### 4. DiagnosisExplainer (Complete Integration)

Combines all explainability sources with clinical reasoning.

```python
explainer = DiagnosisExplainer(device="cuda")

# Generate complete explanation
report = explainer.explain_prediction(
    bpm=45.2,
    anomaly_score=0.13,
    confidence=0.94,
    diagnosis="normal",
    spatial_saliency=saliency_map,      # Optional
    temporal_saliency=temporal_attn,    # Optional
    modal_attribution={"depth": 0.6, "flow": 0.3, "audio": 0.1},
    patient_age=6
)

# Access explanation
print(report.reasoning)  # Full clinical interpretation
print(report.confidence_factors)  # Why confident/uncertain
print(report.hotspots)  # Important regions
print(report.modal_attribution)  # Modality contributions
```

---

## ExplanationReport Structure

Complete report generated by `explain_prediction()`:

```python
ExplanationReport(
    # Core predictions
    bpm=45.2,
    anomaly_score=0.13,
    diagnosis="normal",
    confidence=0.94,
    
    # Saliency maps
    spatial_saliency=np.ndarray,  # (256, 256) importance map
    temporal_saliency=np.ndarray,  # (24,) sequence importance
    
    # Identified regions
    hotspots=[
        {
            "region": "chest",
            "coordinates": (y1, y2, x1, x2),
            "importance": 0.95,
            "pixel_count": 5240
        },
        ...
    ],
    
    # Modality breakdown
    modal_attribution={
        "depth": 0.68,
        "flow": 0.22,
        "audio": 0.10
    },
    
    # Clinical reasoning
    reasoning="Respiratory rate 45.2 is within normal range (35-55) for a 6-month-old. "
              "Very low anomaly score indicates regular breathing pattern...",
    
    # Confidence analysis
    confidence_factors=[
        "✓ High model confidence (>90%)",
        "✓ Consistent signals (normal BPM + low anomaly)",
        "⚠ Consider manual review of edge cases"
    ],
    
    timestamp=datetime(2026, 3, 5, 14, 30, 0)
)
```

---

## Clinical Decision Rules

The explainability system uses **age-based reference ranges**:

### Pediatric Respiratory Rates (Normal Ranges)

| Age | Normal BPM | Tachypnea | Critical |
|-----|-----------|-----------|----------|
| <3 months | 40-60 | >60 | >80 |
| 3-12 months | 35-55 | >55 | >75 |
| 1-3 years | 30-50 | >50 | >70 |

### Anomaly Score Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| 0.0-0.2 | Clear, regular breathing | Routine monitoring |
| 0.2-0.4 | Occasional irregularities | Monitor for trends |
| 0.4-0.7 | Frequent apnea events | Consider escalation |
| 0.7-1.0 | Severe/continuous apnea | **Urgent evaluation** |

---

## Usage Examples

### Basic Integration with RespiratoryController

```python
from src.core.main_controller import RespiratoryController

controller = RespiratoryController(
    config_path="src/config/test_infer.yaml",
    checkpoint_path="data/models/VIRENet_Coarse2fine_Body_best.pth",
    enable_transformer=True,
    enable_explainability=True  # ← Enable Challenge #4
)

# Process video
results = controller.process_video("video.mp4", age_months=6)

# Access explanation
if results.get("explanation_report"):
    report = results["explanation_report"]
    
    print("=" * 60)
    print("CLINICAL EXPLANATION")
    print("=" * 60)
    print(f"\nDiagnosis: {report.diagnosis.upper()}")
    print(f"Confidence: {report.confidence:.0%}")
    print(f"\nReasoning:\n{report.reasoning}")
    print(f"\nConfidence Factors:")
    for factor in report.confidence_factors:
        print(f"  {factor}")
    print(f"\nImportant Regions:")
    for hs in report.hotspots[:3]:
        print(f"  • {hs['region'].title()}: {hs['importance']:.0%} importance")
    print(f"\nModality Contributions:")
    for mod, contrib in report.modal_attribution.items():
        print(f"  {mod.title():6}: {contrib:.0%}")
```

### Direct Explanation Generation

```python
from src.core.explainability import explain_respiratory_prediction

report = explain_respiratory_prediction(
    bpm=48.5,
    anomaly_score=0.35,
    confidence=0.89,
    diagnosis="tachypnea",
    patient_age=6,
    device="cuda"
)

print(report.reasoning)
# Output: "Respiratory rate 48.5 is ABOVE normal range (35-55) for a 6-month-old. 
#          Tachypnea may indicate respiratory distress, fever, pain, or infection.
#          Moderate anomaly score suggests occasional apnea events..."

print(report.confidence_factors)
# Output: ['✓ Good model confidence (70-90%)', 
#          '⚠ Mixed signals - diagnosis may be ambiguous']
```

### Creating Visualizations

```python
import numpy as np
import cv2

# Generate report
report = explain_respiratory_prediction(
    bpm=45.0,
    anomaly_score=0.20,
    confidence=0.92,
    diagnosis="normal",
    patient_age=6
)

# Load a video frame
frame = cv2.imread("sample_frame.jpg")

# Create annotated visualization
from src.core.explainability import DiagnosisExplainer
explainer = DiagnosisExplainer()
annotated = explainer.visualize_explanation(
    report=report,
    frame=frame,
    save_path="explanation_visualization.jpg"
)

# Annotated frame shows:
# - Saliency heatmap overlay (which regions mattered)
# - Bounding boxes around hotspots
# - Diagnosis and confidence labels
# - Modal contribution breakdown
```

---

## Clinical Scenarios

### Scenario A: Healthy Infant (Interpretable Diagnosis)

```
Input:
  BPM: 42.0
  Anomaly Score: 0.10
  Confidence: 0.95
  Age: 6 months

Explanation:
  Reasoning: "Respiratory rate 42.0 is within normal range (35-55) for a 6-month-old. 
              Very low anomaly score indicates regular breathing pattern."
  
  Factors: ✓ High confidence (>90%)
           ✓ Consistent signals (normal BPM + low anomaly)
  
  Hotspots: Chest (94% importance) - Normal
  
  Modal Attribution:
    Depth:  68% (Primary contributor - clear video)
    Flow:   22% (Corroborating signal)
    Audio:  10% (Minimal breathing sounds)
  
  Verdict: NORMAL - Routine monitoring sufficient
```

### Scenario B: Tachypneic Infant (Ambiguous Signals)

```
Input:
  BPM: 65.0
  Anomaly Score: 0.25
  Confidence: 0.72
  Age: 6 months

Explanation:
  Reasoning: "Respiratory rate 65.0 is ABOVE normal range (35-55). 
              Tachypnea may indicate respiratory distress, fever, pain, or infection.
              Moderate anomaly score suggests occasional apnea events."
  
  Factors: ⚠ Lower confidence (70-90%)
           ⚠ Mixed signals (elevated BPM but not extreme anomaly)
  
  Hotspots: Abdomen (71% importance) - Compensatory breathing?
  
  Modal Attribution:
    Depth:  45% (Modest confidence - some motion noise)
    Flow:   35% (Strong signal in motion)
    Audio:  20% (Elevated noises detected)
  
  Verdict: TACHYPNEA - Assess for underlying causes
           Clinician should manually review for:
           - Fever, pain, distress
           - Poor video quality effects
           - Environmental factors
```

### Scenario C: Critical Apnea (Strong Evidence)

```
Input:
  BPM: 35.0
  Anomaly Score: 0.80
  Confidence: 0.94
  Age: 6 months

Explanation:
  Reasoning: "Respiratory rate 35.0 is BELOW normal range (35-55).
              High anomaly score (0.80) indicates FREQUENT or PROLONGED apnea events.
              CRITICAL: Combination of abnormal rate and significant apnea.
              Immediate intervention likely needed."
  
  Factors: ✓ Very high confidence (>90%)
           ✓ Consistent signals (slow BPM + high anomaly)
           ⚠ Extreme anomaly score may indicate data saturation
  
  Hotspots: Chest (89% importance) - Visible apnea pauses
  
  Modal Attribution:
    Depth:  72% (Clear pattern of pauses)
    Flow:   18% (Flow stops during apnea)
    Audio:  10% (Silence during events)
  
  Verdict: **CRITICAL** - Urgent action needed
           Forward to physician immediately
           Consider escalation to intensive monitoring
```

---

## Saliency Visualization

### What Saliency Maps Look Like

**Example: Chest-focused (Good)**
```
Model pays attention to the right place:

Video Frame              Saliency Map
┌─────────────┐        ┌─────────────┐
│             │        │░░░░░░░░░░░░│
│  ╭─────╮   │        │░▓▓▓▓▓▓▓▓▓░│
│  │█████│   │    →   │░▓████████░│
│  │█████│   │        │░▓████████░│
│  │█████│   │        │░▓████████░│
│  ╰─────╯   │        │░▓████████░│
│             │        │░░░░░░░░░░░░│
└─────────────┘        └─────────────┘

✓ Good: Focus on chest (center)
✓ Ignores: Blankets, background, camera artifacts
```

**Example: Peripheral Focus (Poor)**
```
Model paying attention to wrong things:

Video Frame              Saliency Map
┌─────────────┐        ┌─────────────┐
│      ▓▓▓▓  │        │▓▓▓▓▓▓▓▓▓▓▓│
│  ╭─────╮ █ │    →   │▓░░░░░░░░▓│
│  │     │ █ │        │▓▓▓▓▓▓▓▓▓▓│
│  │     │ █ │        │░░░░░░░░░░│
│  ╰─────╯░░ │        │░░░░░░░░░░│
│░░░░░░░░░░░│        │░░░░░░░░░░│
│             │        │░░░░░░░░░░│
└─────────────┘        └─────────────┘

⚠ Poor: Focus on edges and blanket
✗ Might indicate: Poor ROI, occlusions, artifacts
```

---

## Testing

Run the complete test suite:

```bash
python src/core/test_explainability.py
```

Tests verify:
- ✓ Spatial saliency map generation
- ✓ Hotspot identification
- ✓ Temporal attention extraction  
- ✓ Modal attribution analysis
- ✓ Diagnosis explanation quality
- ✓ Confidence factor identification
- ✓ Clinical scenario realism
- ✓ Visualization data preparation
- ✓ Controller integration
- ✓ Edge case handling

---

## Production Deployment

### Clinician Workflow

1. **Infant monitoring session** → Process video
2. **System generates diagnosis** → Display BPM + ARI status
3. **Clinician sees saliency map** → Verifies model looked at right place
4. **Reads explanation text** → Understands clinical reasoning
5. **Reviews confidence factors** → Knows when to double-check
6. **Checks modal attribution** → Identifies sensor problems
7. **Makes informed decision** → Combined with clinical judgment

### Integration Checklist

- [ ] Enable explainability: `enable_explainability=True`
- [ ] Provide frame to saliency generator
- [ ] Ensure model has attention layers for temporal analysis
- [ ] Calibrate reference BPM ranges per population
- [ ] Train modal attribution on your data
- [ ] Test with real clinical videos
- [ ] Get clinician feedback on explanations
- [ ] Iterate on reference ranges

### Regulatory Compliance

Explainability helps meet requirements:
- **GDPR Article 22**: Right to explanation ✓
- **FDA Software Validation**: Traceability ✓
- **Clinical Documentation**: Audit trail ✓
- **Liability Protection**: Documented reasoning ✓

---

## Files

- [src/core/explainability.py](src/core/explainability.py) - Core implementation (900 lines)
- `src/core/test_explainability.py` - Test suite (500 lines)
- `docs/EXPLAINABILITY.md` - This file
- `src/core/main_controller.py` - Integration code

---

## Questions?

See test suite for usage examples:
```bash
python src/core/test_explainability.py
```

Each test includes realistic scenarios showing:
- How to use the explainability system
- How to interpret results
- How to handle edge cases
- How to integrate with RespiratoryController
