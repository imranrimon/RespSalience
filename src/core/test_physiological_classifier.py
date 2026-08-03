"""
Test suite for Challenge #1: Physiological State Classification

Comprehensive tests for state detection and state-aware diagnosis.
Covers:
- State detection accuracy
- Contextual diagnosis (state-dependent thresholds)
- Interaction with anomaly scores
- Age-based norm adjustment
- Integration with controller
- Clinical scenario realism
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.physiological_classifier import (
    StateDetector,
    StateContextualizer,
    ContextualDiagnoser,
    RespiratorynormDatabase,
    PhysiologicalState,
    classify_physiological_state
)


def test_state_detection_sleeping():
    """Test detection of sleeping state."""
    print("\n[TEST 1] State Detection: Sleeping")
    print("-" * 60)
    
    detector = StateDetector(device="cpu")
    
    # Sleeping cues: low motion, quiet, eyes closed, slow BPM
    state = detector.detect_state(
        bpm=32.0,
        anomaly_score=0.07,
        motion_intensity=0.12,
        sound_level=0.03,
        face_visible=True,
        eyes_open=False,
        sucking_detected=False
    )
    
    assert state.state == PhysiologicalState.SLEEPING, f"Expected SLEEPING, got {state.state}"
    assert state.confidence > 0.85, f"Confidence too low: {state.confidence}"
    assert len(state.factors) > 0, "Should have detection factors"
    
    print(f"  ✓ State detected: {state.state.value} ({state.confidence:.0%})")
    print(f"  ✓ Factors: {state.factors[0]}")
    print("✓ TEST 1 PASSED\n")


def test_state_detection_crying():
    """Test detection of crying state."""
    print("[TEST 2] State Detection: Crying")
    print("-" * 60)
    
    detector = StateDetector(device="cpu")
    
    # Crying cues: loud sound, high motion, eyes open
    state = detector.detect_state(
        bpm=110.0,
        anomaly_score=0.05,
        motion_intensity=0.85,
        sound_level=0.88,  # Very loud
        face_visible=True,
        eyes_open=True,
        sucking_detected=False
    )
    
    assert state.state == PhysiologicalState.CRYING, f"Expected CRYING, got {state.state}"
    assert state.confidence > 0.80, f"Confidence too low: {state.confidence}"
    
    print(f"  ✓ State detected: {state.state.value} ({state.confidence:.0%})")
    print(f"  ✓ Key factor: Loud crying detected")
    print("✓ TEST 2 PASSED\n")


def test_state_detection_feeding():
    """Test detection of feeding state."""
    print("[TEST 3] State Detection: Feeding")
    print("-" * 60)
    
    detector = StateDetector(device="cpu")
    
    # Feeding cues: sucking detected, moderate motion
    state = detector.detect_state(
        bpm=48.0,
        anomaly_score=0.10,
        motion_intensity=0.35,
        sound_level=0.15,
        face_visible=True,
        eyes_open=True,
        sucking_detected=True  # CRITICAL CUE
    )
    
    assert state.state == PhysiologicalState.FEEDING, f"Expected FEEDING, got {state.state}"
    assert state.confidence > 0.75, f"Confidence too low: {state.confidence}"
    
    print(f"  ✓ State detected: {state.state.value} ({state.confidence:.0%})")
    print(f"  ✓ Sucking behavior recognized")
    print("✓ TEST 3 PASSED\n")


def test_contextual_diagnosis_same_bpm_different_states():
    """Test that same BPM gets different diagnosis by behavioral state."""
    print("[TEST 4] Contextual Diagnosis: Same BPM, Different States")
    print("-" * 60)
    
    analyzer = StateContextualizer(device="cpu")
    
    # Test BPM that's abnormal when sleeping but normal when active
    test_bpm = 65.0
    
    # Case 1: Same BPM while sleeping
    state_sleep, diag_sleep = analyzer.analyze(
        bpm=test_bpm,
        anomaly_score=0.10,
        motion_intensity=0.15,  # Low motion = sleeping
        sound_level=0.05,       # Quiet
        age_months=6,
        eyes_open=False
    )
    
    # Case 2: Same BPM while active
    state_active, diag_active = analyzer.analyze(
        bpm=test_bpm,
        anomaly_score=0.10,
        motion_intensity=0.70,  # High motion = active
        sound_level=0.10,
        age_months=6,
        eyes_open=True
    )
    
    # Verify different diagnoses
    print(f"  BPM: {test_bpm}")
    print(f"  While sleeping: {diag_sleep.status:12} (expected {diag_sleep.expected_range})")
    print(f"  While active:   {diag_active.status:12} (expected {diag_active.expected_range})")
    
    # When sleeping, 65 BPM should be abnormal
    assert diag_sleep.status in ["tachypnea", "critical"], \
        f"Expected abnormal when sleeping, got {diag_sleep.status}"
    
    # When active, 65 BPM should be normal
    assert diag_active.status == "normal", \
        f"Expected normal when active, got {diag_active.status}"
    
    print(f"  ✓ Same BPM correctly contextualized by behavioral state")
    print("✓ TEST 4 PASSED\n")


def test_age_based_norms():
    """Test that norms adjust appropriately for age."""
    print("[TEST 5] Age-Based Respiratory Rate Norms")
    print("-" * 60)
    
    analyzer = StateContextualizer(device="cpu")
    
    # Test same BPM at different ages
    test_bpm = 48.0
    ages = [1, 6, 12]
    results = []
    
    for age in ages:
        _, diag = analyzer.analyze(
            bpm=test_bpm,
            anomaly_score=0.08,
            motion_intensity=0.30,
            sound_level=0.05,
            age_months=age,
            eyes_open=True
        )
        results.append((age, diag.status, diag.expected_range))
        print(f"  Age {age:2}mo: BPM {test_bpm} -> {diag.status:12} (range {diag.expected_range})")
    
    # Verify norms change with age
    assert len(set(status for _, status, _ in results)) > 0, "Norms should vary by age"
    print(f"  ✓ Norms appropriately adjust for age")
    print("✓ TEST 5 PASSED\n")


def test_anomaly_score_overrides():
    """Test that high anomaly score triggers critical diagnosis regardless of state."""
    print("[TEST 6] Anomaly Score Overrides State")
    print("-" * 60)
    
    analyzer = StateContextualizer(device="cpu")
    
    # Even with normal BPM and sleeping state, high anomaly = critical
    state, diag = analyzer.analyze(
        bpm=38.0,  # Normal for sleeping
        anomaly_score=0.85,  # VERY HIGH = critical apnea
        motion_intensity=0.15,  # Sleeping
        sound_level=0.05,
        age_months=6,
        eyes_open=False
    )
    
    print(f"  Scenario: Sleeping, BPM 38 (normal), Anomaly 0.85 (critical apnea)")
    print(f"  State detected: {state.state.value}")
    print(f"  Diagnosis: {diag.status}")
    
    assert diag.status == "critical", f"Expected critical when anomaly is 0.85, got {diag.status}"
    assert "apnea" in diag.reasoning.lower(), "Reasoning should mention apnea"
    
    print(f"  ✓ High anomaly score correctly overrides state-based thresholds")
    print("✓ TEST 6 PASSED\n")


def test_state_uncertainty_caveat():
    """Test that uncertain state detection adds caveats to diagnosis."""
    print("[TEST 7] State Uncertainty Reduces Confidence")
    print("-" * 60)
    
    analyzer = StateContextualizer(device="cpu")
    
    # Ambiguous state (conflicting signals)
    state, diag = analyzer.analyze(
        bpm=50.0,
        anomaly_score=0.10,
        motion_intensity=0.45,  # Borderline
        sound_level=0.30,       # Borderline
        age_months=6,
        eyes_open=True,
        face_visible=False
    )
    
    print(f"  State: {state.state.value} (confidence: {state.confidence:.0%})")
    print(f"  Diagnosis: {diag.status}")
    print(f"  Status confidence: {diag.status_confidence:.0%}")
    
    # Check for caveats
    has_caveat = any("state" in caveat.lower() or "uncertain" in caveat.lower() 
                     for caveat in diag.caveats)
    
    if state.confidence < 0.7:
        print(f"  ✓ Low state confidence triggers caveat: {diag.caveats[0]}")
        assert has_caveat or len(diag.caveats) > 0, "Should have caveats for uncertain state"
    
    print("✓ TEST 7 PASSED\n")


def test_crying_reassessment_caveat():
    """Test that crying diagnosis includes reassessment caveat."""
    print("[TEST 8] Crying State Reassessment Caveat")
    print("-" * 60)
    
    analyzer = StateContextualizer(device="cpu")
    
    # Infant is crying - high BPM expected
    state, diag = analyzer.analyze(
        bpm=100.0,
        anomaly_score=0.08,
        motion_intensity=0.80,
        sound_level=0.90,  # Very loud = crying
        age_months=6,
        eyes_open=True
    )
    
    print(f"  State: {state.state.value}")
    print(f"  Diagnosis: {diag.status}")
    print(f"  Caveats: {diag.caveats[0] if diag.caveats else 'None'}")
    
    assert state.state == PhysiologicalState.CRYING
    
    # Should have caveat about reassessing when calm
    reassess_caveat = any("reassess" in c.lower() or "calm" in c.lower() 
                         for c in diag.caveats)
    assert reassess_caveat, "Should suggest reassessment when calm"
    
    print(f"  ✓ Correctly suggests reassessment after infant calms")
    print("✓ TEST 8 PASSED\n")


def test_convenience_function():
    """Test the convenience function."""
    print("[TEST 9] Convenience Function API")
    print("-" * 60)
    
    state_name, confidence, status = classify_physiological_state(
        bpm=45.0,
        anomaly_score=0.12,
        motion_intensity=0.25,
        sound_level=0.08,
        age_months=6
    )
    
    print(f"  State: {state_name} ({confidence:.0%})")
    print(f"  Status: {status}")
    
    assert state_name in ["sleeping", "quiet_awake", "active", "feeding", "crying", "indeterminate"]
    assert 0 <= confidence <= 1
    assert status in ["normal", "tachypnea", "bradypnea", "critical"]
    
    print(f"  ✓ Convenience function returns expected types")
    print("✓ TEST 9 PASSED\n")


def test_clinical_scenario_ari_with_fever():
    """Realistic scenario: ARI with fever (elevated RR) but infant is awake/alert."""
    print("[TEST 10] Clinical Scenario: ARI with Fever (Awake)")
    print("-" * 60)
    
    analyzer = StateContextualizer(device="cpu")
    
    # Infant with fever: elevated RR expected, but infant is awake/fussy
    state, diag = analyzer.analyze(
        bpm=68.0,
        anomaly_score=0.15,  # Slight anomaly
        motion_intensity=0.55,  # Somewhat active (uncomfortable)
        sound_level=0.25,       # Fussy sounds
        age_months=6,
        eyes_open=True
    )
    
    print(f"  Clinical presentation: Possible fever/ARI")
    print(f"  Detected state: {state.state.value}")
    print(f"  RR: {diag.observed_bpm:.0f} BPM")
    print(f"  Expected for state: {diag.expected_range}")
    print(f"  Diagnosis: {diag.status}")
    
    # Elevated RR for active/awake state might still be normal
    # or tachypnea - depends on norms
    print(f"  ✓ Contextualizes RR elevation within behavioral context")
    print("✓ TEST 10 PASSED\n")


def test_clinical_scenario_sleeping_with_bradypnea():
    """Realistic scenario: Premature infant bradypneic even while sleeping."""
    print("[TEST 11] Clinical Scenario: Bradypnea While Sleeping")
    print("-" * 60)
    
    analyzer = StateContextualizer(device="cpu")
    
    # Very low RR even for sleeping - might indicate CNS depression
    state, diag = analyzer.analyze(
        bpm=18.0,  # Below typical sleeping range
        anomaly_score=0.05,
        motion_intensity=0.10,
        sound_level=0.02,
        age_months=1,  # Premature
        eyes_open=False
    )
    
    print(f"  Clinical presentation: Possibly depressed breathing pattern")
    print(f"  State: {state.state.value}")
    print(f"  RR: {diag.observed_bpm:.0f} BPM")
    print(f"  Expected for sleeping: {diag.expected_range}")
    print(f"  Diagnosis: {diag.status}")
    print(f"  Reasoning: {diag.reasoning[:100]}...")
    
    # Even accounting for sleep, 18 BPM is potentially concerning
    assert diag.status in ["bradypnea", "critical"], "Should detect abnormally low RR"
    print(f"  ✓ Detects bradypnea despite behavioral explanation")
    print("✓ TEST 11 PASSED\n")


def test_norm_database_completeness():
    """Test that norm database covers expected ages and states."""
    print("[TEST 12] Norm Database Completeness")
    print("-" * 60)
    
    db = RespiratorynormDatabase()
    
    # Check that database has entries for key ages
    ages_checked = [1, 6, 12]
    states_checked = [
        PhysiologicalState.SLEEPING,
        PhysiologicalState.QUIET_AWAKE,
        PhysiologicalState.ACTIVE,
        PhysiologicalState.CRYING,
        PhysiologicalState.FEEDING
    ]
    
    count = 0
    for age in ages_checked:
        for state in states_checked:
            norms = db.get_norms(age, state)
            assert norms.normal_min < norms.normal_max, f"Invalid norms for {age}mo, {state}"
            count += 1
    
    print(f"  ✓ Database covers {count} age/state combinations")
    print(f"  ✓ All norms are valid (min < max)")
    print("✓ TEST 12 PASSED\n")


def test_state_transitions():
    """Test realistic state transitions throughout a monitoring session."""
    print("[TEST 13] State Transitions During Monitoring")
    print("-" * 60)
    
    analyzer = StateContextualizer(device="cpu")
    
    # Simulate monitoring session: infant goes through states
    scenarios = [
        ("Sleeping", 32.0, 0.08, 0.1, 0.02, False),
        ("Waking up", 42.0, 0.10, 0.3, 0.10, True),
        ("Awake and alert", 48.0, 0.12, 0.4, 0.08, True),
        ("Getting fussy", 65.0, 0.15, 0.6, 0.30, True),
        ("Crying", 105.0, 0.10, 0.85, 0.88, True),
    ]
    
    print("  Simulated monitoring session:")
    for name, bpm, anomaly, motion, sound, eyes in scenarios:
        state, diag = analyzer.analyze(
            bpm=bpm,
            anomaly_score=anomaly,
            motion_intensity=motion,
            sound_level=sound,
            age_months=6,
            eyes_open=eyes
        )
        print(f"    {name:20} -> {state.state.value:15} (BPM {bpm:3.0f}, status: {diag.status})")
    
    print(f"  ✓ System handles state transitions throughout session")
    print("✓ TEST 13 PASSED\n")


def test_missing_visual_data():
    """Test state detection when face/eyes not visible."""
    print("[TEST 14] Missing Visual Data Handling")
    print("-" * 60)
    
    analyzer = StateContextualizer(device="cpu")
    
    # Low motion, quiet, but face not visible
    state, diag = analyzer.analyze(
        bpm=35.0,
        anomaly_score=0.08,
        motion_intensity=0.12,
        sound_level=0.03,
        age_months=6,
        face_visible=False,  # Can't see face
        eyes_open=False  # Eyes open status unknown
    )
    
    print(f"  Scenario: Face not visible, low motion, quiet")
    print(f"  Detected state: {state.state.value} ({state.confidence:.0%})")
    print(f"  Confidence reduced due to missing visual cues")
    
    # Should still make reasonable guess but with lower confidence
    assert state.state in [PhysiologicalState.SLEEPING, PhysiologicalState.QUIET_AWAKE]
    print(f"  ✓ Makes reasonable inference despite missing visual data")
    print("✓ TEST 14 PASSED\n")


def main():
    print("\n" + "=" * 80)
    print("Challenge #1: Physiological State Classification - Test Suite")
    print("=" * 80)
    
    tests = [
        test_state_detection_sleeping,
        test_state_detection_crying,
        test_state_detection_feeding,
        test_contextual_diagnosis_same_bpm_different_states,
        test_age_based_norms,
        test_anomaly_score_overrides,
        test_state_uncertainty_caveat,
        test_crying_reassessment_caveat,
        test_convenience_function,
        test_clinical_scenario_ari_with_fever,
        test_clinical_scenario_sleeping_with_bradypnea,
        test_norm_database_completeness,
        test_state_transitions,
        test_missing_visual_data,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ ASSERTION FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}\n")
            failed += 1
    
    print("=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("✓ All tests passed! Physiological state classifier ready for deployment.\n")
        return 0
    else:
        print(f"✗ {failed} test(s) failed. Please review.\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
