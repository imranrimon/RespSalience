# Physiological State Classification (Challenge #1)

## Overview

**Physiological State Classification** solves a fundamental clinical problem: **respiratory rate is meaningless without behavioral context**.

### The Problem

The same exact BPM can be:
- **Abnormal/alarming** in one situation
- **Completely normal** in another

**Example:**
```
Infant's BPM = 60
├─ While sleeping → ABNORMAL (tachypnea, needs investigation)
└─ While crying   → NORMAL (expected physiologic elevation)
```

Without knowing the behavioral state, you can't diagnose. Without diagnosing correctly, clinicians don't trust the system.

### The Solution

This module:
1. **Detects behavioral state** (sleeping, awake, active, feeding, crying)
2. **Adjusts reference ranges** based on state
3. **Makes contextual diagnosis** (is THIS BPM normal for THIS state?)
4. **Explains the reasoning** (behavioral state + clinical thresholds)

---

## Clinical Rationale

### Why States Matter

Pediatric respiratory rate changes dramatically with behavioral state:

| State | Typical BPM | Mechanism |
|-------|-----------|-----------|
| **Sleeping** | 25-40 | Minimal metabolic demand, relaxed muscles |
| **Quiet awake** | 35-55 | Baseline activity, focused attention |
| **Active/playing** | 45-70 | Increased metabolism, fine motor activity |
| **Feeding** | 40-60 | Increased work, but coordinated effort |
| **Crying** | 70-110+ | Peak metabolic demand, greatest stress |

**Clinical Rule:** An elevated RR is normal IF the infant is doing something that requires elevated RR.

---

## Architecture

### Core Components

```
Vital Signs (BPM, Anomaly, Age)
    +
Video/Audio Signals (Motion, Sound)
    ↓
StateDetector
    ├─ Analyze multimodal cues
    ├─ Score each behavioral state
    └─ Return: Most likely state + confidence
    ↓
StateContextualizer  
    ├─ Look up norms for (age, state)
    ├─ Compare BPM to age/state-specific thresholds
    └─ Return: Contextual diagnosis
```

### StateDetector: Identifying Behavior

**Detection rules (heuristic scoring):**

**Sleeping:**
- ✓ Very low motion (optical flow < 0.2)
- ✓ No vocalization/sound (< 0.15)
- ✓ Eyes closed (if visible)
- ✓ Slow RR for observed state

**Crying/Distressed:**
- ✓ High sound level (> 0.7)
- ✓ High motion
- ✓ Eyes open

**Feeding:**
- ✓ Sucking motion detected
- ✓ Organized behavioral pattern
- ✓ Moderate motion + normal RR

**Active/Playing:**
- ✓ High motion (> 0.6)
- ✓ Eyes open
- ✓ Variable RR

**Quiet Awake (baseline):**
- ✓ Low-moderate motion
- ✓ Eyes open
- ✓ Normal RR
- ✓ Minimal vocalization

### RespiratorynormDatabase: Age-Specific Norms

Database stores normal/abnormal thresholds for each age × state combination:

```python
# Example: 6-month-old infant
6: {
    SLEEPING: {
        normal_min: 25,
        normal_max: 40,
        tachypnea_threshold: 40,    # Alert if > this
        critical_threshold: 65       # Emergency if > this
    },
    QUIET_AWAKE: {
        normal_min: 35,
        normal_max: 55,
        tachypnea_threshold: 55,
        critical_threshold: 75
    },
    ACTIVE: {
        normal_min: 45,
        normal_max: 70,
        tachypnea_threshold: 80,
        critical_threshold: 100
    },
    CRYING: {
        normal_min: 70,
        normal_max: 110,
        tachypnea_threshold: 110,   # Very high - hard to distinguish from distress
        critical_threshold: 150
    },
    ...
}
```

### ContextualDiagnoser: State-Aware Diagnosis

Makes diagnosis considering:
1. **Observed BPM** vs state-specific reference range
2. **Anomaly score** (overrides state if very high)
3. **Confidence modulation** (lower if state uncertain)
4. **Clinical caveats** (e.g., "reassess when calm if crying")

---

## Usage

### Basic Integration

```python
from src.core.main_controller import RespiratoryController

# Enable Challenge #1
controller = RespiratoryController(
    config_path="src/config/test_infer.yaml",
    checkpoint_path="data/models/VIRENet_Coarse2fine_Body_best.pth",
    enable_physiological_classification=True  # ← Enable
)

# Process video
results = controller.process_video("video.mp4", age_months=6)

# Get state-aware diagnosis
if results.get("physiological_state"):
    state_info = results["physiological_state"]
    
    print(f"Detected state: {state_info['state']} ({state_info['state_confidence']:.0%})")
    print(f"Expected BPM for this state: {state_info['expected_range']}")
    print(f"Contextual diagnosis: {state_info['contextual_status']}")
    print(f"Reasoning: {state_info['reasoning']}")
```

### Direct Usage

```python
from src.core.physiological_classifier import StateContextualizer

analyzer = StateContextualizer(device="cuda")

# Detect state and make contextual diagnosis
state_det, contextual_diag = analyzer.analyze(
    bpm=60.0,
    anomaly_score=0.10,
    motion_intensity=0.7,    # High motion (from optical flow)
    sound_level=0.05,        # Quiet (from audio)
    age_months=6,
    eyes_open=True
)

print(f"State: {state_det.state.value}")
print(f"Confidence: {state_det.confidence:.0%}")
print(f"Diagnosis: {contextual_diag.status}")
print(f"Expected range: {contextual_diag.expected_range}")
```

---

## Output Structure

### physiological_state Dict

```python
{
    'state': 'active',                    # Detected behavioral state
    'state_confidence': 0.85,             # 0-1: How sure about state?
    'state_factors': [                    # What cues indicated this state?
        'High body motion/activity',
        'Eyes open/alert',
        'Elevated RR (65 BPM) with activity'
    ],
    
    'contextual_status': 'normal',        # normal/tachypnea/bradypnea/critical
    'status_confidence': 0.92,            # 0-1: How sure about diagnosis?
    'expected_range': (45, 70),           # Normal BPM range for this state/age
    
    'reasoning': "Infant appears to be active/playing. RR of 65.0 BPM is within "
                 "expected range (45-70 BPM) for active infant. May indicate "
                 "increased work of breathing...",
    
    'caveats': [                          # Important limitations
        "⚠ State uncertainty reduces diagnostic confidence",
        "⚠ RR near upper normal limit - monitor trend"
    ]
}
```

---

## Clinical Scenarios

### Scenario A: Tachypnea Investigation

**Presentation:**
- Reported RR: 65 BPM
- Age: 6 months
- Clinical concern: "Is this normal?"

**With Challenge #1 Analysis:**

1. **Detect state:** Active/playing (high motion), eyes open
2. **Lookup norms:** Active infant, 6 months → 45-70 BPM normal
3. **Compare:** 65 BPM is within normal range
4. **Diagnosis:** **NORMAL** (reassure parents)
5. **Caveat:** "Reassess if RR remains elevated when quiet"

**Without Challenge #1:**
- Just see "65 BPM" against population-level norm (40-60)
- Classify as tachypnea
- Unnecessarily escalate, stress parents

---

### Scenario B: Sleeping Infant with Tachypnea

**Presentation:**
- Observed RR: 52 BPM
- Age: 6 months
- Observation: Sleeping peacefully

**With Challenge #1 Analysis:**

1. **Detect state:** Sleeping (low motion, eyes closed)
2. **Lookup norms:** Sleeping infant, 6 months → 25-40 BPM normal
3. **Compare:** 52 BPM is above normal range for sleeping
4. **Diagnosis:** **TACHYPNEA** (elevated even accounting for sleep)
5. **Action:** Investigate for underlying cause
   - Post-feeding (still elevated)?
   - Early fever?
   - Cardiovascular issue?

**Clinical Value:** Without state awareness, you might miss this—it looks "normal" without context.

---

### Scenario C: Infant Crying (Differential Diagnosis)

**Presentation:**
- Observed RR: 95 BPM
- Age: 6 months
- State: Loud crying/distressed

**With Challenge #1 Analysis:**

1. **Detect state:** Crying (high sound > 0.7)
2. **Lookup norms:** Crying infant, 6 months → 70-110 BPM normal
3. **Compare:** 95 BPM is within normal range
4. **Diagnosis:** **NORMAL** (expected for crying)
5. **Caveat:** "⚠ Diagnosis made during crying/distress - reassess when calm"

**Clinical Action:**
- Soothe infant, wait for calming
- Re-assess RR after infant relaxes
- If still elevated when calm → investigate

---

## Age-Based Normalization

Respiratory standards vary by age. Module automatically adjusts:

| Age | Sleeping | Quiet Awake | Active | Crying |
|-----|----------|------------|--------|--------|
| 1 month | 30-45 | 40-60 | 50-70 | 70-100 |
| 6 months | **25-40** | **35-55** | **45-70** | **70-110** |
| 12 months | 20-35 | 30-50 | 40-60 | 70-110 |

**Example:** 42 BPM is:
- 6mo sleeping: tachypnea (expected 25-40)
- 6mo awake: normal (expected 35-55)
- 12mo sleeping: normal (expected 20-35)

---

## State Detection Reliability

### High Confidence States

✅ **Sleeping:**
- Multiple corroborating cues (eyes closed, low motion, quiet)
- Lower false positive rate

✅ **Crying:**
- Obvious auditory cue (sound > 0.75)
- Hard to confuse with other states

### Challenging States

⚠️ **Active vs Quiet Awake:**
- Both have eyes open, some motion
- Threshold depends on motion intensity (0.5 cutoff)
- Can be ambiguous during transition

⚠️ **Feeding:**
- Requires sucking detection or characteristic pattern
- May be missed if sucking not visible

⚠️ **Indeterminate:**
- Conflicting signals
- System flags as uncertain, reduces diagnosis confidence

---

## Integration with Other Challenges

### Challenge #1 + #2 (State + Cross-Modal Robustness)

```python
# Cross-modal disagreement might depend on state
# E.g., optical flow dominated during active = expected
# But during sleeping = might indicate artifact
```

### Challenge #1 + #3 (State + Longitudinal Tracking)

```python
# Longitudinal abnormalities more concerning if consistent
# across states (vs appearing only when upset)
```

### Challenge #1 + #4 (State + Explainability)

```python
# Saliency maps might differ by state:
# - Sleeping: focus on subtle chest motion
# - Crying: focus on high-motion features
```

---

## Testing

Run comprehensive test suite:

```bash
python src/core/test_physiological_classifier.py
```

**Tests cover:**
- ✓ State detection accuracy (sleeping, crying, feeding, etc)
- ✓ Contextual diagnosis (same BPM → different status by state)
- ✓ Age-based norm adjustment
- ✓ Anomaly score overrides
- ✓ State uncertainty handling
- ✓ Clinical scenarios
- ✓ State transitions
- ✓ Missing visual data

**All 14 tests passing** ✅

---

## Production Deployment

### Clinician Workflow

1. **Video captured** of infant (state not controlled)
2. **System processes:** Detects RR, state, diagnosis
3. **Clinician sees:**
   - "BPM: 65"
   - "State: Active (85% confidence)"
   - "Expected for active infant: 45-70 BPM"
   - "Diagnosis: NORMAL"
   - "Note: Re-assess when quiet"
4. **Clinical decision:** Informed by behavioral context

### Configuration

```python
# Minimal setup - uses defaults
controller = RespiratoryController(
    ...,
    enable_physiological_classification=True
)

# Results automatically include physiological_state
# No additional configuration needed
```

### State Detection Signals

**Optical Flow:**
- Automatically extracted in process_video()
- Used to estimate motion_intensity

**Audio:**
- Automatically extracted in process_video()
- Used to estimate sound_level

**Face/Eyes:**
- Can be detected via face recognition (optional enhancement)
- Currently defaults to "visible" and "open"

---

## Limitations & Future Work

### Current Limitations

- ⚠️ State detection is heuristic (not ML-based)
- State confidence lower with missing visual data
- Doesn't distinguish "sleeping" from "sedated" (both low motion)
- Feeding state requires sucking detection

### Future Enhancements

1. **ML-based state detection** (train on labeled videos)
2. **Physiologic markers** (e.g., heart rate variability by state)
3. **Recurrent states** (model typical patterns for infant)
4. **Caregiver-reported state** (allow manual override)
5. **Personalized norms** (infant-specific baseline)

---

## Files

- [src/core/physiological_classifier.py](src/core/physiological_classifier.py) (1200+ lines)
  - StateDetector, ContextualDiagnoser, StateContextualizer
  - RespiratorynormDatabase with age/state norms
  - Convenience functions

- [src/core/test_physiological_classifier.py](src/core/test_physiological_classifier.py)
  - 14 comprehensive tests
  - All passing ✅

- [src/core/main_controller.py](src/core/main_controller.py)
  - Integration code (enabled via enable_physiological_classification=True)
  - Results include "physiological_state" field

- [docs/PHYSIOLOGICAL_STATES.md](docs/PHYSIOLOGICAL_STATES.md)
  - This documentation

---

## Questions?

See test suite for practical examples:
```bash
python src/core/test_physiological_classifier.py
```

Each test demonstrates:
- How to detect states
- How same BPM gets different diagnosis by behavioral state
- How confidence factors influence output
- Real clinical scenarios
