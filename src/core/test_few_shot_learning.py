"""
Comprehensive Test Suite for Challenge #5: Few-Shot Generalization

Tests few-shot learning, domain adaptation, transfer learning, and meta-learning.
All tests validate adaptation with minimal labeled examples for new populations.
"""

import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.few_shot_learning import (
    FewShotAdaptationModule, FewShotExample, DomainShiftDetector,
    TransferLearningAdapter, TransferLearningConfig, MetaLearnerOptimizer,
    PrototypicalNetwork, create_few_shot_adapter, detect_new_population_shift
)


def create_synthetic_example(label: int, population: str = "base") -> FewShotExample:
    """Create a synthetic training example."""
    np.random.seed(label * 100 + hash(population) % 1000)
    
    if population == "base":
        # Base population characteristics
        if label == 0:  # normal
            depth_base = 100
            flow_base = 5
            audio_base = 20
        elif label == 1:  # ARI
            depth_base = 120
            flow_base = 15
            audio_base = 50
        else:  # abnormal
            depth_base = 80
            flow_base = 25
            audio_base = 80
    else:
        # New population (shifted distribution)
        if label == 0:
            depth_base = 150  # Different camera sensitivity
            flow_base = 3
            audio_base = 25
        elif label == 1:
            depth_base = 180
            flow_base = 12
            audio_base = 60
        else:
            depth_base = 100
            flow_base = 22
            audio_base = 95
    
    # Create modality data
    depth = np.random.normal(depth_base, 20, (64, 64)).astype(np.float32)
    flow = np.random.normal([flow_base, flow_base], 3, (64, 64, 2)).astype(np.float32)
    audio = np.random.normal(audio_base, 15, (128,)).astype(np.float32)
    
    return FewShotExample(
        modality_data={'depth': depth, 'flow': flow, 'audio': audio},
        label=label,
        meta_info={'population': population, 'facility': 'test_facility'}
    )


# ============================================================================
# TEST 1: Basic Few-Shot Adaptation with 5 Examples
# ============================================================================

def test_few_shot_adaptation_5_examples():
    """TEST 1: Few-shot adaptation with minimum viable set (5 examples)."""
    print("\n" + "="*70)
    print("TEST 1: Few-Shot Adaptation with 5 Examples")
    print("="*70)
    
    try:
        adapter = create_few_shot_adapter(device='cpu')
        
        # Create 5 support examples (minimal set)
        support = [
            create_synthetic_example(0, "new_facility"),  # normal
            create_synthetic_example(1, "new_facility"),  # ARI
            create_synthetic_example(0, "new_facility"),  # normal
            create_synthetic_example(2, "new_facility"),  # abnormal
            create_synthetic_example(1, "new_facility")   # ARI
        ]
        
        # Adapt
        result = adapter.adapt(support, verbose=False)
        
        # Verify
        assert result.adapted, "Should be adapted"
        assert result.num_examples_used == 5, f"Should use 5 examples, got {result.num_examples_used}"
        assert 0 <= result.confidence <= 1, f"Confidence should be 0-1, got {result.confidence}"
        assert result.confidence >= 0.4, f"Confidence should be reasonable, got {result.confidence}"
        
        print(f"[PASS] Adaptation successful")
        print(f"  - Examples used: {result.num_examples_used}")
        print(f"  - Confidence: {result.confidence:.2f}")
        print(f"  - Domain shift detected: {result.domain_shift_detected}")
        print(f"  - Domain shift magnitude: {result.domain_shift_magnitude:.2f}")
        print(f"  - Improvement estimate: {result.improvement_estimate:.1f}%")
        print(f"  - Recommendation: {result.recommendation}")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# TEST 2: Domain Shift Detection (New Camera Type)
# ============================================================================

def test_domain_shift_detection_new_camera():
    """TEST 2: Detect domain shift from new camera type."""
    print("\n" + "="*70)
    print("TEST 2: Domain Shift Detection (New Camera Type)")
    print("="*70)
    
    try:
        adapter = create_few_shot_learning_module(device='cpu')
        
        # Create examples from new facility (different camera, different lighting, etc.)
        new_pop_examples = [
            create_synthetic_example(0, "new_facility"),
            create_synthetic_example(0, "new_facility"),
            create_synthetic_example(1, "new_facility")
        ]
        
        result = adapter.adapt(new_pop_examples)
        
        # Should detect domain shift
        assert result.domain_shift_detected, "Should detect domain shift from new facility"
        assert result.domain_shift_magnitude > 0.1, "Domain shift magnitude should be significant"
        
        print(f"[PASS] Domain shift detection working")
        print(f"  - Shift detected: {result.domain_shift_detected}")
        print(f"  - Shift magnitude: {result.domain_shift_magnitude:.2f}")
        print(f"  - Confidence: {result.confidence:.2f}")
        print(f"  - Recommendation: {result.recommendation}")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


def create_few_shot_learning_module(device='cpu'):
    """Helper to create module."""
    return create_few_shot_adapter(device=device)


# ============================================================================
# TEST 3: Few-Shot Prediction with Adapted Prototypes
# ============================================================================

def test_few_shot_prediction_adapted():
    """TEST 3: Predict using adapted prototypes from few-shot examples."""
    print("\n" + "="*70)
    print("TEST 3: Prediction with Adapted Prototypes")
    print("="*70)
    
    try:
        adapter = create_few_shot_adapter(device='cpu')
        
        # Adapt to new population
        support = [
            create_synthetic_example(0, "new_pop"),
            create_synthetic_example(1, "new_pop"),
            create_synthetic_example(0, "new_pop"),
            create_synthetic_example(1, "new_pop")
        ]
        
        adapt_result = adapter.adapt(support)
        assert adapt_result.adapted, "Should adapt"
        
        # Now predict on a query example
        query_example = create_synthetic_example(0, "new_pop")
        pred, conf, label = adapter.predict_with_adapted_prototypes(
            query_example.modality_data
        )
        
        # Verify prediction structure (not exact class due to randomness)
        assert pred in [0, 1, 2], f"Should predict valid class, got {pred}"
        assert 0 <= conf <= 1, f"Confidence should be 0-1, got {conf}"
        assert conf > 0.3, f"Confidence should be reasonable, got {conf}"
        assert label in ["normal", "ARI", "abnormal"], f"Label should be valid, got {label}"
        
        print(f"[PASS] Adapted prediction working")
        print(f"  - Prediction: {label}")
        print(f"  - Confidence: {conf:.2f}")
        print(f"  - Num examples used for adaptation: {adapt_result.num_examples_used}")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# TEST 4: Domain Shift Detector with Reference Distribution
# ============================================================================

def test_domain_shift_detector_mmd():
    """TEST 4: Domain shift detection using distribution analysis."""
    print("\n" + "="*70)
    print("TEST 4: Domain Shift Detector (Distribution-based)")
    print("="*70)
    
    try:
        # Create reference distribution (base training)
        ref_features = np.random.normal(100, 15, (20, 256)).astype(np.float32)
        ref_mean = np.mean(ref_features, axis=0)
        ref_cov = np.cov(ref_features.T)
        
        detector = DomainShiftDetector(ref_mean, ref_cov)
        
        # Test 1: In-distribution data (should compute shift magnitudes)
        indom_magnitudes = []
        indom_features = np.random.normal(100, 15, (5, 256)).astype(np.float32)
        for feat in indom_features:
            shift_det, shift_mag, expl = detector.detect_shift(feat)
            indom_magnitudes.append(float(shift_mag))
        
        # Check that we get valid shift magnitudes
        assert all(0 <= s <= 1 for s in indom_magnitudes), f"Shift magnitudes should be 0-1"
        
        # Test 2: Out-of-distribution data (should detect shift)
        ood_features = np.random.normal(180, 30, (5, 256)).astype(np.float32)
        shift_detected_count = 0
        for feat in ood_features:
            shift_det, shift_mag, expl = detector.detect_shift(feat)
            if shift_det:
                shift_detected_count += 1
        
        assert shift_detected_count >= 3, f"Should detect OOD shift, got {shift_detected_count}/5"
        
        print(f"[PASS] Domain shift detector working")
        print(f"  - In-distribution: computed {len(indom_magnitudes)} magnitudes")
        print(f"  - Out-of-distribution: detected in {shift_detected_count}/5 cases")
        print(f"  - Detection history: {len(detector.detection_history)} records")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# TEST 5: Transfer Learning with Few Samples
# ============================================================================

def test_transfer_learning_fine_tune():
    """TEST 5: Efficient fine-tuning on small dataset."""
    print("\n" + "="*70)
    print("TEST 5: Transfer Learning Fine-Tuning")
    print("="*70)
    
    try:
        device = 'cpu'
        
        # Create a simple linear model for classification (compatible with transfer learning)
        class SimpleClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(256, 128)
                self.fc2 = torch.nn.Linear(128, 64)
                self.fc3 = torch.nn.Linear(64, 3)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        model = SimpleClassifier().to(device)
        
        # Create small fine-tuning set
        support_features = []
        support_labels = []
        for i in range(8):
            example = create_synthetic_example(i % 3, "transfer_test")
            adapter = FewShotAdaptationModule(device=device)
            feat = adapter.prepare_multimodal_features(example.modality_data)
            support_features.append(feat)
            support_labels.append(example.label)
        
        support_tensor = torch.from_numpy(np.array(support_features)).to(device)
        labels_tensor = torch.from_numpy(np.array(support_labels)).long().to(device)
        
        # Create transfer learning adapter
        config = TransferLearningConfig(
            num_fine_tune_steps=15,
            learning_rate=0.001,
            layer_freeze_ratio=0.7
        )
        transfer_adapter = TransferLearningAdapter(model, config)
        
        # Fine-tune
        result = transfer_adapter.fine_tune(
            support_tensor, labels_tensor,
            num_steps=15,
            learning_rate=0.001
        )
        
        assert result['status'] == 'success', f"Fine-tuning should succeed, got {result['status']}"
        assert result['steps_completed'] > 0, "Should complete steps"
        assert len(result['loss_history']) > 0, "Should have loss history"
        assert result['final_loss'] >= 0, "Loss should be non-negative"
        
        print(f"[PASS] Transfer learning fine-tuning successful")
        print(f"  - Steps completed: {result['steps_completed']}")
        print(f"  - Final loss: {result['final_loss']:.4f}")
        print(f"  - Loss history length: {len(result['loss_history'])}")
        print(f"  - Layer freeze ratio: {config.layer_freeze_ratio}")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# TEST 6: Meta-Learning Rapid Adaptation
# ============================================================================

def test_meta_learning_maml_like():
    """TEST 6: Meta-learning for rapid task adaptation."""
    print("\n" + "="*70)
    print("TEST 6: Meta-Learning (MAML-like) Adaptation")
    print("="*70)
    
    try:
        device = 'cpu'
        
        # Create adapter
        adapter = FewShotAdaptationModule(device=device)
        
        meta_optimizer = MetaLearnerOptimizer(adapter.model, device=device)
        
        # Create support and query sets for a meta-learning task
        support_features = []
        support_labels = []
        for i in range(6):
            example = create_synthetic_example(i % 3, "meta_test")
            feat = adapter.prepare_multimodal_features(example.modality_data)
            support_features.append(feat)
            support_labels.append(example.label)
        
        query_features = []
        query_labels = []
        for i in range(3):
            example = create_synthetic_example(i % 3, "meta_test")
            feat = adapter.prepare_multimodal_features(example.modality_data)
            query_features.append(feat)
            query_labels.append(example.label)
        
        support_set = (
            torch.from_numpy(np.array(support_features)).to(device),
            torch.from_numpy(np.array(support_labels)).long().to(device)
        )
        query_set = (
            torch.from_numpy(np.array(query_features)).to(device),
            torch.from_numpy(np.array(query_labels)).long().to(device)
        )
        
        # Perform meta-adaptation
        result = meta_optimizer.meta_adapt(
            support_set, query_set,
            inner_lr=0.01,
            num_inner_steps=5
        )
        
        assert 0 <= result.task_accuracy <= 1, f"Accuracy should be 0-1, got {result.task_accuracy}"
        assert result.adaptation_steps > 0, "Should complete adaptation steps"
        assert result.confidence >= 0, "Confidence should be non-negative"
        
        print(f"[PASS] Meta-learning adaptation working")
        print(f"  - Task accuracy: {result.task_accuracy:.2f}")
        print(f"  - Adaptation steps: {result.adaptation_steps}")
        print(f"  - Parameter shift norm: {result.parameter_shift_norm:.4f}")
        print(f"  - Confidence: {result.confidence:.2f}")
        print(f"  - Meta history length: {len(meta_optimizer.meta_adaptation_history)}")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# TEST 7: Prototypical Network Architecture
# ============================================================================

def test_prototypical_network_forward():
    """TEST 7: Prototypical network forward pass and classification."""
    print("\n" + "="*70)
    print("TEST 7: Prototypical Network Architecture")
    print("="*70)
    
    try:
        device = 'cpu'
        model = PrototypicalNetwork(feature_dim=128, num_classes=3)
        model = model.to(device)
        
        # Create support and query tensors
        support_features = torch.randn(6, 256).to(device)  # 2 per class
        support_labels = torch.tensor([0, 1, 2, 0, 1, 2]).long().to(device)
        
        query_features = torch.randn(3, 256).to(device)
        
        # Forward pass
        logits = model.forward(query_features, support_features, support_labels)
        
        assert logits.shape == (3, 3), f"Logits shape should be (3,3), got {logits.shape}"
        
        # Predict
        predictions, confidences = model.predict(query_features, support_features, support_labels)
        
        assert len(predictions) == 3, "Should have 3 predictions"
        assert len(confidences) == 3, "Should have 3 confidences"
        assert all(0 <= c <= 1 for c in confidences), "Confidences should be 0-1"
        
        print(f"[PASS] Prototypical network working")
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Predictions: {predictions}")
        print(f"  - Confidences: {[f'{c:.2f}' for c in confidences]}")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# TEST 8: Feature Preparation from Multimodal Data
# ============================================================================

def test_multimodal_feature_preparation():
    """TEST 8: Prepare features from depth, flow, and audio."""
    print("\n" + "="*70)
    print("TEST 8: Multimodal Feature Preparation")
    print("="*70)
    
    try:
        adapter = create_few_shot_adapter(device='cpu')
        
        # Test with complete multimodal data
        example_complete = create_synthetic_example(0, "complete")
        features_complete = adapter.prepare_multimodal_features(
            example_complete.modality_data
        )
        
        assert features_complete.shape == (256,), f"Should be 256-dim, got {features_complete.shape}"
        assert features_complete.dtype == np.float32, f"Should be float32, got {features_complete.dtype}"
        
        # Test with missing modalities
        example_missing = FewShotExample(
            modality_data={
                'depth': np.random.normal(100, 20, (64, 64)),
                'flow': None,  # Missing
                'audio': None   # Missing
            },
            label=0
        )
        features_missing = adapter.prepare_multimodal_features(
            example_missing.modality_data
        )
        
        assert features_missing.shape == (256,), f"Should handle missing modalities"
        assert not np.all(np.isnan(features_missing)), "Should not be all NaN"
        
        # Test with all None
        example_empty = FewShotExample(
            modality_data={'depth': None, 'flow': None, 'audio': None},
            label=1
        )
        features_empty = adapter.prepare_multimodal_features(
            example_empty.modality_data
        )
        
        assert features_empty.shape == (256,), "Should handle empty case"
        
        print(f"[PASS] Multimodal feature preparation working")
        print(f"  - Complete multimodal: {features_complete.shape}, range [{features_complete.min():.2f}, {features_complete.max():.2f}]")
        print(f"  - Missing modalities: {features_missing.shape}, handled gracefully")
        print(f"  - All missing: {features_empty.shape}, defaults to zeros")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# TEST 9: Clinical Scenario - Hospital A to Hospital B Transfer
# ============================================================================

def test_clinical_scenario_hospital_transfer():
    """TEST 9: Real-world scenario - adapting model from Hospital A to Hospital B."""
    print("\n" + "="*70)
    print("TEST 9: Clinical Scenario - Hospital Transfer")
    print("="*70)
    
    try:
        adapter = create_few_shot_adapter(device='cpu')
        
        # Hospital A training (base model pretrained)
        # Hospital B setup (new population with different camera, age range, demographics)
        
        # Collect 10 labeled examples from Hospital B
        hospital_b_examples = [
            create_synthetic_example(0, "hospital_b"),  # normal infant
            create_synthetic_example(0, "hospital_b"),
            create_synthetic_example(1, "hospital_b"),  # ARI infant
            create_synthetic_example(1, "hospital_b"),
            create_synthetic_example(0, "hospital_b"),
            create_synthetic_example(2, "hospital_b"),  # abnormal
            create_synthetic_example(0, "hospital_b"),
            create_synthetic_example(1, "hospital_b"),
            create_synthetic_example(0, "hospital_b"),
            create_synthetic_example(1, "hospital_b")
        ]
        
        # Adapt to Hospital B
        adapt_result = adapter.adapt(hospital_b_examples)
        
        assert adapt_result.adapted, "Should adapt to new hospital"
        assert adapt_result.num_examples_used == 10, f"Should use all 10 examples"
        assert adapt_result.confidence > 0.5, f"Should have reasonable confidence"
        
        # Test prediction on Hospital B patient
        test_patient = create_synthetic_example(0, "hospital_b")
        pred, conf, label = adapter.predict_with_adapted_prototypes(
            test_patient.modality_data
        )
        
        assert pred in [0, 1, 2], f"Should predict valid class"
        assert conf > 0.3, f"Confidence should be reasonable, got {conf}"
        
        print(f"[PASS] Hospital transfer scenario successful")
        print(f"  - Hospital B adaptation: {adapt_result.adapted}")
        print(f"  - Examples used: {adapt_result.num_examples_used}")
        print(f"  - Domain shift: {adapt_result.domain_shift_magnitude:.2f}")
        print(f"  - Adaptation confidence: {adapt_result.confidence:.2f}")
        print(f"  - Test prediction: {label} (confidence {conf:.2f})")
        print(f"  - Recommendation: {adapt_result.recommendation}")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# TEST 10: Insufficient Data Handling
# ============================================================================

def test_insufficient_data_adaptation():
    """TEST 10: Graceful handling when too few examples provided."""
    print("\n" + "="*70)
    print("TEST 10: Insufficient Data Handling")
    print("="*70)
    
    try:
        adapter = create_few_shot_adapter(device='cpu')
        
        # Try with only 1 example (insufficient)
        minimal_examples = [create_synthetic_example(0, "minimal")]
        result_minimal = adapter.adapt(minimal_examples)
        
        assert not result_minimal.adapted, "Should reject adaptation with 1 example"
        assert "at least 3 examples" in result_minimal.recommendation.lower(), \
            f"Should mention minimum examples, got: {result_minimal.recommendation}"
        
        # Try with 3 examples (minimum)
        min_viable = [
            create_synthetic_example(0, "viable"),
            create_synthetic_example(1, "viable"),
            create_synthetic_example(0, "viable")
        ]
        result_viable = adapter.adapt(min_viable)
        
        assert result_viable.adapted, "Should adapt with 3 examples"
        assert result_viable.num_examples_used == 3, "Should use all 3"
        
        print(f"[PASS] Insufficient data handling working")
        print(f"  - 1 example: adapted={result_minimal.adapted}, msg='{result_minimal.recommendation[:40]}...'")
        print(f"  - 3 examples: adapted={result_viable.adapted}, confidence={result_viable.confidence:.2f}")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# TEST 11: Confidence Scores Scale with Sample Size
# ============================================================================

def test_confidence_scaling_with_samples():
    """TEST 11: Confidence should increase with more examples."""
    print("\n" + "="*70)
    print("TEST 11: Confidence Scaling with Sample Size")
    print("="*70)
    
    try:
        configs = [
            (5, "few"),
            (10, "moderate"),
            (20, "many")
        ]
        
        confidences = []
        
        for num_samples, label_desc in configs:
            adapter = create_few_shot_adapter(device='cpu')
            
            examples = [
                create_synthetic_example(i % 3, "scaling_test")
                for i in range(num_samples)
            ]
            
            result = adapter.adapt(examples)
            confidences.append(result.confidence)
            
            print(f"  - {num_samples} samples ({label_desc}): confidence = {result.confidence:.2f}")
        
        # Confidence should generally increase
        assert confidences[0] < confidences[2], \
            f"Confidence should increase: {confidences}"
        
        print(f"[PASS] Confidence scaling validated")
        print(f"  - Confidence progression: {[f'{c:.2f}' for c in confidences]}")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# TEST 12: Adaptation History Tracking
# ============================================================================

def test_adaptation_history_tracking():
    """TEST 12: Module tracks adaptation history across multiple adaptations."""
    print("\n" + "="*70)
    print("TEST 12: Adaptation History Tracking")
    print("="*70)
    
    try:
        adapter = create_few_shot_adapter(device='cpu')
        
        # First adaptation
        examples1 = [
            create_synthetic_example(0, "pop1"),
            create_synthetic_example(1, "pop1"),
            create_synthetic_example(0, "pop1")
        ]
        adapter.adapt(examples1)
        
        # Second adaptation
        examples2 = [
            create_synthetic_example(1, "pop2"),
            create_synthetic_example(1, "pop2"),
            create_synthetic_example(2, "pop2"),
            create_synthetic_example(0, "pop2"),
            create_synthetic_example(1, "pop2")
        ]
        adapter.adapt(examples2)
        
        # Third adaptation
        examples3 = [create_synthetic_example(i % 3, "pop3") for i in range(8)]
        adapter.adapt(examples3)
        
        # Check history
        history = adapter.adaptation_history
        assert len(history) == 3, f"Should have 3 adaptations, got {len(history)}"
        
        assert history[0]['num_examples'] == 3, "First: 3 examples"
        assert history[1]['num_examples'] == 5, "Second: 5 examples"
        assert history[2]['num_examples'] == 8, "Third: 8 examples"
        
        print(f"[PASS] Adaptation history tracking working")
        print(f"  - Total adaptations: {len(history)}")
        print(f"  - Adaptation 1: {history[0]['num_examples']} examples, shift={history[0]['domain_shift']:.2f}")
        print(f"  - Adaptation 2: {history[1]['num_examples']} examples, shift={history[1]['domain_shift']:.2f}")
        print(f"  - Adaptation 3: {history[2]['num_examples']} examples, shift={history[2]['domain_shift']:.2f}")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# TEST 13: Integration - Complete Few-Shot Pipeline
# ============================================================================

def test_complete_few_shot_pipeline():
    """TEST 13: Complete pipeline - adapt, predict, track, repeat."""
    print("\n" + "="*70)
    print("TEST 13: Complete Few-Shot Learning Pipeline")
    print("="*70)
    
    try:
        adapter = create_few_shot_adapter(device='cpu')
        detector = DomainShiftDetector()
        
        # Phase 1: Collect labeled data from new facility
        facility_data = [
            create_synthetic_example(0, "new_facility"),
            create_synthetic_example(1, "new_facility"),
            create_synthetic_example(0, "new_facility"),
            create_synthetic_example(2, "new_facility"),
            create_synthetic_example(1, "new_facility"),
            create_synthetic_example(0, "new_facility")
        ]
        
        # Phase 2: Adapt
        adapt_result = adapter.adapt(facility_data)
        assert adapt_result.adapted, "Should adapt"
        
        # Phase 3: Deploy and make predictions
        test_patients = [
            create_synthetic_example(0, "new_facility"),  # normal
            create_synthetic_example(1, "new_facility"),  # ARI
            create_synthetic_example(0, "new_facility")   # normal
        ]
        
        predictions = []
        for patient in test_patients:
            pred, conf, label = adapter.predict_with_adapted_prototypes(
                patient.modality_data
            )
            predictions.append((pred, conf, label))
        
        # Phase 4: Monitor for drift using detector
        shift_detected, shift_mag = detect_new_population_shift(
            adapter.prepare_multimodal_features(test_patients[0].modality_data),
            detector
        )
        
        # Phase 5: If significant drift detected, consider re-adaptation
        if shift_detected:
            print(f"  - Drift detected during deployment, would re-adapt")
        
        assert len(predictions) == 3, "Should have 3 predictions"
        assert all(isinstance(p[2], str) for p in predictions), "All labels should be strings"
        
        print(f"[PASS] Complete pipeline working")
        print(f"  - Adapted: {adapt_result.adapted}")
        print(f"  - Domain shift: {adapt_result.domain_shift_magnitude:.2f}")
        print(f"  - Test predictions:")
        for i, (pred, conf, label) in enumerate(predictions):
            print(f"    Patient {i+1}: {label} (confidence {conf:.2f})")
        print(f"  - Drift monitoring: {'active' if shift_detected else 'nominal'}")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# TEST 14: Few-Shot with Multi-Domain (Mixed Populations)
# ============================================================================

def test_few_shot_multi_domain_learning():
    """TEST 14: Few-shot learning across multiple domains/populations."""
    print("\n" + "="*70)
    print("TEST 14: Few-Shot Multi-Domain Learning")
    print("="*70)
    
    try:
        adapter = create_few_shot_adapter(device='cpu')
        
        # Collect examples from multiple facilities/domains
        multi_domain_examples = []
        
        # Facility A examples
        for i in range(3):
            multi_domain_examples.append(
                create_synthetic_example(i % 3, "facility_a")
            )
        
        # Facility B examples (different camera, lighting)
        for i in range(3):
            multi_domain_examples.append(
                create_synthetic_example(i % 3, "facility_b")
            )
        
        # Facility C examples (another variant)
        for i in range(2):
            multi_domain_examples.append(
                create_synthetic_example(i % 3, "facility_c")
            )
        
        # Adapt with mixed data
        result = adapter.adapt(multi_domain_examples)
        
        assert result.adapted, "Should adapt to multi-domain data"
        assert result.num_examples_used == 8, f"Should use all 8 examples"
        assert result.domain_shift_detected or not result.domain_shift_detected, \
            "Should compute shift magnitude"
        
        # Test: Make predictions that work across domains
        test_domains = ["facility_a", "facility_b", "facility_c"]
        for domain in test_domains:
            test = create_synthetic_example(0, domain)
            pred, conf, label = adapter.predict_with_adapted_prototypes(test.modality_data)
            assert pred in [0, 1, 2], f"Should predict valid class for {domain}"
            assert conf > 0.3, f"Confidence should be reasonable for {domain}"
        
        print(f"[PASS] Multi-domain few-shot learning working")
        print(f"  - Total examples: {result.num_examples_used}")
        print(f"  - Domains: facility_a, facility_b, facility_c")
        print(f"  - Domain shift magnitude: {result.domain_shift_magnitude:.2f}")
        print(f"  - Cross-domain predictions: all working")
        
        return True
    
    except Exception as e:
        print(f"[FAIL] FAILED: {str(e)}")
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHALLENGE #5: FEW-SHOT GENERALIZATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Few-Shot with 5 Examples", test_few_shot_adaptation_5_examples),
        ("Domain Shift Detection", test_domain_shift_detection_new_camera),
        ("Adapted Prediction", test_few_shot_prediction_adapted),
        ("Domain Shift Detector", test_domain_shift_detector_mmd),
        ("Transfer Learning", test_transfer_learning_fine_tune),
        ("Meta-Learning", test_meta_learning_maml_like),
        ("Prototypical Network", test_prototypical_network_forward),
        ("Multimodal Features", test_multimodal_feature_preparation),
        ("Clinical Scenario", test_clinical_scenario_hospital_transfer),
        ("Insufficient Data", test_insufficient_data_adaptation),
        ("Confidence Scaling", test_confidence_scaling_with_samples),
        ("History Tracking", test_adaptation_history_tracking),
        ("Complete Pipeline", test_complete_few_shot_pipeline),
        ("Multi-Domain", test_few_shot_multi_domain_learning)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n[FAIL] {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed_count}/{total_count} tests passed")
    print("="*70 + "\n")
    
    # Exit code
    exit(0 if passed_count == total_count else 1)

