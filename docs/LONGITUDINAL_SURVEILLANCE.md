# Longitudinal Disease Trajectory Modeling (Challenge #3)

## Overview

**Longitudinal Disease Trajectory Modeling** transforms your respiratory monitoring system from **point-in-time diagnosis** to **temporal disease tracking**.

Instead of asking "Is this infant sick right now?", it answers:
- "Is the infant getting worse?"
- "Will they need ICU admission in 24 hours?"
- "Should we escalate care now or wait?"

**Clinical Motivation:**
- BPM readings alone are unreliable (context matters)
- Trends are powerful: "BPM rising steadily" is worse than "BPM spiking once"
- Early intervention saves lives (catch deterioration before crisis)
- Clinicians need: trajectory prediction + actionable alerts

---

## Architecture

### Core Components

```
RespiratoryMeasurement (Single assessment)
    ↓
LongitudinalRespiratoryTracker (Keep history, compute trends)
    ├─ Store measurements over time
    ├─ Compute BPM slope (rate of change)
    ├─ Compute anomaly trends (apnea worsening?)
    └─ Extract sequences for ML
    ↓
TrajectoryPredictor (LSTM - what comes next?)
    ├─ Input: Sequence of (bpm, anomaly, confidence, age)
    ├─ LSTM layers: Capture temporal patterns
    └─ Output: Trajectory class + Risk score + Attention
    ↓
EarlyWarningSystem (Rule-based alerts)
    ├─ BPM trends (rising = concerning)
    ├─ Anomaly trends (more apnea = worsening)
    ├─ Measurement reliability
    ├─ ML predictions
    └─ Output: Color-coded alert level + Recommendations
```

### Data Flow

```
Video 1 → Inference → BPM=42, Anomaly=0.1
Video 2 → Inference → BPM=45, Anomaly=0.2
Video 3 → Inference → BPM=48, Anomaly=0.35
    ↓
[Trends: BPM slope=+3/hr, Anomaly slope=+0.125/hr]
    ↓
LSTM: "Deteriorating" (36% confidence)
    ↓
Alert: YELLOW (score 45/100)
    ↓
Recommendation: "Increase monitoring to every 15-30 minutes"
```

---

## Components in Detail

### 1. RespiratoryMeasurement

Single point-in-time assessment with:
- Timestamp
- BPM (respiratory rate)
- Anomaly score (0=normal, 1=critical apnea)
- Confidence (0-1, model certainty)
- Age (for contextualization)
- Diagnosis status (normal/tachypnea/apnea/critical)

```python
measurement = RespiratoryMeasurement(
    timestamp=datetime.now(),
    bpm=45.2,
    anomaly_score=0.25,
    confidence=0.92,
    age_months=6,
    diagnosis_status="tachypnea"
)
```

### 2. LongitudinalRespiratoryTracker

Maintains history and computes trends:

```python
tracker = LongitudinalRespiratoryTracker(max_history=100)  # Keep last 100 measurements

# Add measurements as videos are analyzed
for video_result in video_results:
    tracker.add_measurement(RespiratoryMeasurement(...))

# Analyze trends
bpm_trend = tracker.get_bpm_trend(window=5)
# Returns: mean, std, slope (rate of change), acceleration

anomaly_trend = tracker.get_anomaly_trend(window=5)
# Returns: mean, max, slope
```

**Key Metrics:**
- **BPM Slope**: Positive = worsening, Negative = improving
- **BPM Acceleration**: 2nd derivative (are things accelerating worse?)
- **Anomaly Slope**: Positive = more apnea events

### 3. TrajectoryPredictor (LSTM)

LSTM-based sequence model that predicts:

```python
model = TrajectoryPredictor(input_dim=4, hidden_dim=64, num_layers=2)

# Input: Recent measurement sequence
seq = torch.tensor([
    [40, 0.10, 0.95, 6],  # 6 hours ago
    [42, 0.15, 0.92, 6],
    [44, 0.22, 0.90, 6],
    [46, 0.30, 0.85, 6],
    [48, 0.40, 0.80, 6],  # 2 hours ago
], dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    output = model(seq)

# Output:
# - trajectory_class: 0=Improving, 1=Stable, 2=Deteriorating
# - bpm_next: Predicted next BPM
# - risk_score: Probability of intervention needed in 24h
# - attention_weights: Which time steps mattered most?
```

**Architecture:**
```
Input (B, SeqLen, 4)
    ↓
LSTM (64 hidden, 2 layers)
    ↓
Last hidden state (B, 64)
    ↓
├─ Trajectory head → (B, 3) logits → softmax → "Improving/Stable/Deteriorating"
├─ BPM head → (B, 1) → predicted next BPM
└─ Risk head → (B, 1) → sigmoid → intervention probability
```

### 4. EarlyWarningSystem

Rule-based + ML hybrid system:

```python
warning = early_warning.compute_warning_level(
    tracker,
    trajectory_output  # Optional ML prediction
)

# Returns:
#{
#    'level': 'GREEN' / 'YELLOW' / 'RED',
#    'score': 0-100,
#    'factors': ['Rising BPM', 'Increasing apnea', ...],
#    'recommendations': ['Increase monitoring', ...]
#}
```

**Factors Considered:**
1. **BPM trend** (slope > 2 = concern)
2. **BPM acceleration** (2nd derivative)
3. **Apnea trends** (max anomaly score)
4. **Measurement reliability** (confidence < 0.7)
5. **ML predictions** (intervention risk > threshold)
6. **Absolute status** (critical > apnea > tachypnea > normal)

**Risk Scoring (0-100):**
- **GREEN** (0-39): Routine monitoring
- **YELLOW** (40-69): Increased monitoring, prepare for intervention
- **RED** (70+): Urgent action needed

---

## Usage

### Basic Integration with RespiratoryController

```python
from src.core.main_controller import RespiratoryController

controller = RespiratoryController(
    config_path="src/config/test_infer.yaml",
    checkpoint_path="data/models/VIRENet_Coarse2fine_Body_best.pth",
    enable_transformer=True,
    enable_robust_fusion=True,
    enable_longitudinal_surveillance=True,  # ← Enable Challenge #3
)

# Process videos - measurements automatically added to tracker
results1 = controller.process_video("video1.mp4", age_months=6)
results2 = controller.process_video("video2.mp4", age_months=6)
results3 = controller.process_video("video3.mp4", age_months=6)

# Get comprehensive surveillance report
if controller.longitudinal_system:
    report = controller.longitudinal_system.get_report()
    
    print(f"Current status: {report['patient_snapshot']['bpm']} BPM")
    print(f"Trend: {report['trends']['bpm']['slope']:.2f} BPM/hour")
    print(f"Alert Level: {report['early_warning']['level']}")
    
    if report['clinical_action_needed']:
        print(f"ACTION NEEDED: {report['early_warning']['recommendations'][0]}")
```

### Direct Usage

```python
from src.core.longitudinal_surveillance import LongitudinalSurveillanceSystem

system = LongitudinalSurveillanceSystem(device="cuda")

# Simulate hourly monitoring
for hour in range(24):
    bpm, anomaly = get_measurement_from_video(video_path)
    system.add_measurement(
        bpm=bpm,
        anomaly_score=anomaly,
        confidence=0.95,
        age_months=6,
        diagnosis_status="normal"
    )
    
    # Check status every hour
    if hour % 4 == 0:  # Every 4 hours
        report = system.get_report()
        
        if report['clinical_action_needed']:
            print(f"ALERT: {report['early_warning']['level']}")
            for recommendation in report['early_warning']['recommendations']:
                print(f"  → {recommendation}")
```

---

## Output Report Structure

Complete surveillance report includes:

```python
report = {
    # Current snapshot
    'patient_snapshot': {
        'timestamp': '2026-03-05T14:30:00',
        'bpm': 48.5,
        'anomaly_score': 0.325,
        'status': 'tachypnea',
        'age_months': 6,
        'measurement_confidence': 0.92
    },
    
    # Historical trends
    'trends': {
        'bpm': {
            'mean': 46.0,      # Average over recent window
            'std': 2.5,        # Variability
            'slope': +2.1,     # Rising (positive = worsening)
            'acceleration': +0.3  # Accelerating worse
        },
        'anomaly': {
            'mean': 0.24,
            'max': 0.40,
            'slope': +0.05     # More apnea events
        },
        'measurements_count': 8
    },
    
    # ML trajectory prediction
    'trajectory': {
        'predicted_class': 'Deteriorating',
        'class_probabilities': {
            'improving': 0.20,
            'stable': 0.35,
            'deteriorating': 0.45
        },
        'predicted_next_bpm': 51.2,
        'intervention_risk': 0.63  # 63% chance of needing intervention in 24h
    },
    
    # Rule-based early warning
    'early_warning': {
        'level': 'YELLOW',  # GREEN / YELLOW / RED
        'score': 52,        # 0-100
        'factors': [
            'BPM consistently rising',
            'Apnea events increasing'
        ],
        'recommendations': [
            'Increase monitoring frequency (every 15-30 min)',
            'Check sensor calibration',
            'Reassess in 4 hours'
        ]
    },
    
    # Flag for clinical systems
    'clinical_action_needed': True,
    
    # Raw data for analysis
    'data_history': {
        'timestamps': [...],
        'bpms': [...],
        'anomaly_scores': [...],
        'statuses': [...]
    }
}
```

---

## Clinical Scenarios

### Scenario A: Healthy Infant

```
Measurements over 6 hours:
  12:00: BPM=40, Anomaly=0.08 → Normal
  13:00: BPM=41, Anomaly=0.09 → Normal
  14:00: BPM=39, Anomaly=0.07 → Normal
  15:00: BPM=42, Anomaly=0.10 → Normal
  16:00: BPM=40, Anomaly=0.09 → Normal
  17:00: BPM=40, Anomaly=0.08 → Normal

Trends:
  BPM slope: -0.1 (very stable)
  Anomaly slope: 0.0 (no worsening)

Alert: GREEN (0/100)
Recommendation: Routine monitoring (4-6 hourly)
```

### Scenario B: Worsening ARI (Acute Respiratory Infection)

```
Measurements over 6 hours:
  12:00: BPM=42, Anomaly=0.10 → Normal
  13:00: BPM=46, Anomaly=0.15 → Normal
  14:00: BPM=49, Anomaly=0.25 → Tachypnea
  15:00: BPM=52, Anomaly=0.38 → Tachypnea
  16:00: BPM=55, Anomaly=0.50 → Apnea
  17:00: BPM=57, Anomaly=0.62 → Critical

Trends:
  BPM slope: +3.8/hour (rapidly rising!)
  Anomaly slope: +0.10/hour (worsening)
  BPM acceleration: +1.2 (getting worse FASTER)

ML Prediction:
  Deteriorating (78% confidence)
  Intervention risk: 92%

Alert: RED (85/100)
Recommendations:
  [URGENT] Contact physician immediately
  Consider ICU admission
  Prepare for emergent intervention
```

### Scenario C: Recovery (Post-Treatment)

```
Measurements over 12 hours:
  Start (treatment initiated): BPM=52, Anomaly=0.55
  +2h:  BPM=50, Anomaly=0.45
  +4h:  BPM=48, Anomaly=0.35
  +6h:  BPM=46, Anomaly=0.25
  +8h:  BPM=44, Anomaly=0.15
  +10h: BPM=42, Anomaly=0.10
  +12h: BPM=40, Anomaly=0.08

Trends:
  BPM slope: -1.0/hour (improving)
  Anomaly slope: -0.04/hour (fewer apnea events)

ML Prediction:
  Improving (71% confidence)

Alert: GREEN (state improved to normal)
Recommendation: Continue current management, de-escalate monitoring frequency
```

---

## Training the Trajectory Predictor

For optimal performance, train on historical patient data:

```python
from src.core.longitudinal_surveillance import TrajectoryPredictor
import torch
from torch.utils.data import DataLoader, TensorDataset

# Prepare training data from your historical records
# sequences: (N, SeqLen, 4) - measurement sequences
# labels: (N,) - 0=Improving, 1=Stable, 2=Deteriorating

model = TrajectoryPredictor(hidden_dim=128, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    for seq_batch, label_batch in train_loader:
        seq_batch = seq_batch.to(device)
        label_batch = label_batch.to(device)
        
        output = model(seq_batch)
        loss = loss_fn(output['trajectory_logits'], label_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save trained model
torch.save({
    'model_state': model.state_dict(),
    'model_type': 'TrajectoryPredictor',
    'epoch': epoch
}, 'data/models/trajectory_predictor_trained.pth')

# Load during inference
system = LongitudinalSurveillanceSystem()
system.load_trajectory_model('data/models/trajectory_predictor_trained.pth')
```

---

## Early Warning Threshold Tuning

Calibrate thresholds based on your patient population:

```python
# High sensitivity (catch all at-risk patients)
system.warning_system = EarlyWarningSystem(
    bpm_threshold_multiplier=1.15,  # 15% above baseline
    anomaly_threshold=0.4,           # Lower threshold
    risk_score_threshold=0.5         # Lower threshold
)

# High specificity (reduce false alarms)
system.warning_system = EarlyWarningSystem(
    bpm_threshold_multiplier=1.35,  # 35% above baseline
    anomaly_threshold=0.7,           # Higher threshold
    risk_score_threshold=0.8         # Higher threshold
)
```

---

## Testing

Run the complete test suite:

```bash
$env:KMP_DUPLICATE_LIB_OK='TRUE'  # Windows OpenMP fix
python src/core/test_longitudinal_surveillance.py
```

Tests cover:
- ✓ Measurement tracking
- ✓ BPM & anomaly trend computation
- ✓ LSTM trajectory prediction
- ✓ Early warning scoring
- ✓ Clinical scenario realism
- ✓ Controller integration

---

## Clinical Impact

### Benefits

✓ **Catch deterioration early** - Before critical event
✓ **Reduce false positives** - Trends filter noise
✓ **Actionable alerts** - Recommendations for each level
✓ **Intervention timing** - Predict 24h ahead
✓ **Resource optimization** - Know when to escalate/de-escalate monitoring

### Metrics to Track

In production, monitor:
1. **Sensitivity**: % of true deteriorations caught
2. **Specificity**: % of false alarms reduced
3. **Time-to-alert**: How early are interventions flagged?
4. **Clinical outcomes**: Do flagged patients have better outcomes?
5. **Workload**: Are clinicians responding appropriately?

---

## Future Enhancements

1. **Temporal smoothing** - Moving average of alerts
2. **Patient-specific baselines** - Personalized thresholds
3. **Multi-organ integration** - Combine with vitals (SpO2, heart rate, temp)
4. **Uncertainty quantification** - Calibrated confidence intervals
5. **Decision support** - "What if" scenarios for interventions
6. **Explainability** - Which measurements drove the alert?

---

## Files

- `src/core/longitudinal_surveillance.py` - Core implementation (800 lines)
- `src/core/test_longitudinal_surveillance.py` - Comprehensive tests
- `docs/LONGITUDINAL_SURVEILLANCE.md` - This guide
- `src/core/main_controller.py` - Integration code

---

## Questions?

See test suite for practical examples:
```bash
python src/core/test_longitudinal_surveillance.py
```

Each test includes detailed output showing:
- How to use the system
- How to interpret results
- What medical scenarios look like
