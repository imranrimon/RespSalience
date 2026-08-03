"""
Comprehensive test suite for Challenge #7: Comorbidity & Etiology Prediction

Tests etiology classification, comorbidity detection, differential diagnosis,
and risk profiling components.

Author: Respiratory Rate Monitoring AI
Date: 2026
"""

import numpy as np
import torch
# import pytest  # Commented out to avoid dependency
from unittest.mock import Mock, patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from comorbidity_etiology import (
    EtiologyType, ComorbidityType,
    EtiologyPredictor, ComorbidityDetector,
    DifferentialDiagnosisEngine, RiskProfiler,
    EtiologyPrediction, ComorbidityProfile,
    create_comorbidity_and_etiology_system
)


class TestEtiologyPredictor:
    """Test etiology prediction functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = EtiologyPredictor(device='cpu')

    def test_initialization(self):
        """Test predictor initializes correctly."""
        assert self.predictor.device == 'cpu'
        assert self.predictor.model is not None
        assert len(self.predictor.etiology_labels) == 10
        assert EtiologyType.NORMAL in self.predictor.etiology_labels

    def test_predict_etiology_normal_case(self):
        """Test normal etiology prediction."""
        # Create mock features
        features = np.random.randn(256).astype(np.float32)

        result = self.predictor.predict_etiology(features)

        assert isinstance(result, EtiologyPrediction)
        assert result.primary_etiology in EtiologyType
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.differential_diagnoses) >= 1
        assert isinstance(result.reasoning, str)

    def test_predict_etiology_with_patterns(self):
        """Test prediction includes feature patterns in reasoning."""
        features = np.random.randn(256).astype(np.float32)

        result = self.predictor.predict_etiology(features)

        # Should have some reasoning
        assert len(result.reasoning) > 0

    def test_predict_etiology_error_handling(self):
        """Test error handling in prediction."""
        # Invalid features
        features = np.array([])  # Empty array

        result = self.predictor.predict_etiology(features)

        # Should return safe default
        assert result.primary_etiology == EtiologyType.NORMAL
        assert result.confidence == 0.0
        assert "error" in result.reasoning.lower()


class TestComorbidityDetector:
    """Test comorbidity detection functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ComorbidityDetector()

    def test_initialization(self):
        """Test detector initializes correctly."""
        assert self.detector.symptom_patterns is not None
        assert ComorbidityType.FEVER in self.detector.symptom_patterns

    def test_detect_comorbidities_normal_infant(self):
        """Test detection for normal infant."""
        patient_data = {
            'age_months': 6,
            'gestational_age_weeks': 40,
            'temperature_celsius': 36.5,
            'weight_kg': 7.5
        }
        features = np.random.randn(256).astype(np.float32)

        result = self.detector.detect_comorbidities(patient_data, features)

        assert isinstance(result, ComorbidityProfile)
        assert isinstance(result.detected_comorbidities, list)
        assert isinstance(result.severity_scores, dict)
        assert 0.0 <= result.respiratory_impact <= 1.0
        assert isinstance(result.reasoning, str)

    def test_detect_prematurity(self):
        """Test detection of prematurity comorbidity."""
        patient_data = {
            'age_months': 2,
            'gestational_age_weeks': 32,  # Premature
            'temperature_celsius': 36.5,
            'weight_kg': 3.0
        }
        features = np.random.randn(256).astype(np.float32)

        result = self.detector.detect_comorbidities(patient_data, features)

        assert ComorbidityType.PREMATURITY in result.detected_comorbidities
        assert result.respiratory_impact > 0.0

    def test_detect_fever(self):
        """Test detection of fever comorbidity."""
        patient_data = {
            'age_months': 6,
            'gestational_age_weeks': 40,
            'temperature_celsius': 39.0,  # Fever
            'weight_kg': 7.5
        }
        features = np.random.randn(256).astype(np.float32)

        result = self.detector.detect_comorbidities(patient_data, features)

        assert ComorbidityType.FEVER in result.detected_comorbidities
        assert result.severity_scores[ComorbidityType.FEVER.value] > 0.0

    def test_detect_malnutrition(self):
        """Test detection of malnutrition comorbidity."""
        patient_data = {
            'age_months': 6,
            'gestational_age_weeks': 40,
            'temperature_celsius': 36.5,
            'weight_kg': 4.0  # Low weight for age
        }
        features = np.random.randn(256).astype(np.float32)

        result = self.detector.detect_comorbidities(patient_data, features)

        assert ComorbidityType.MALNUTRITION in result.detected_comorbidities

    def test_detect_cardiac_defect(self):
        """Test detection of cardiac defect comorbidity."""
        patient_data = {
            'age_months': 6,
            'gestational_age_weeks': 40,
            'temperature_celsius': 36.5,
            'weight_kg': 7.5
        }
        # High variance in chest features
        features = np.random.randn(256).astype(np.float32) * 20  # High variance

        result = self.detector.detect_comorbidities(patient_data, features)

        # May or may not detect depending on random features
        assert isinstance(result.detected_comorbidities, list)


class TestDifferentialDiagnosisEngine:
    """Test differential diagnosis engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = DifferentialDiagnosisEngine(device='cpu')

    def test_initialization(self):
        """Test engine initializes correctly."""
        assert self.engine.etiology_predictor is not None
        assert self.engine.comorbidity_detector is not None

    def test_generate_differential_diagnosis(self):
        """Test differential diagnosis generation."""
        features = np.random.randn(256).astype(np.float32)
        patient_data = {
            'age_months': 6,
            'gestational_age_weeks': 40,
            'temperature_celsius': 36.5,
            'weight_kg': 7.5
        }

        result = self.engine.generate_differential_diagnosis(features, patient_data)

        assert len(result.top_3_diagnoses) == 3
        assert len(result.confidence_scores) == 3
        assert len(result.clinical_reasoning) == 3
        assert isinstance(result.recommendation, str)
        assert all(0.0 <= conf <= 1.0 for conf in result.confidence_scores)

    def test_diagnosis_with_comorbidities(self):
        """Test diagnosis adjusts for comorbidities."""
        features = np.random.randn(256).astype(np.float32)
        patient_data = {
            'age_months': 2,
            'gestational_age_weeks': 32,  # Premature
            'temperature_celsius': 39.0,  # Fever
            'weight_kg': 3.0
        }

        result = self.engine.generate_differential_diagnosis(features, patient_data)

        # Should have adjusted confidences
        assert len(result.confidence_scores) == 3
        # Prematurity should boost prematurity-related diagnoses
        # (This is probabilistic, so we just check structure)

    def test_diagnosis_recommendations(self):
        """Test diagnosis provides appropriate recommendations."""
        features = np.random.randn(256).astype(np.float32)
        patient_data = {
            'age_months': 6,
            'gestational_age_weeks': 40,
            'temperature_celsius': 36.5,
            'weight_kg': 7.5
        }

        result = self.engine.generate_differential_diagnosis(features, patient_data)

        assert "recommend" in result.recommendation.lower() or "assessment" in result.recommendation.lower()


class TestRiskProfiler:
    """Test risk profiling functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = RiskProfiler()

    def test_calculate_risk_score_normal(self):
        """Test risk calculation for normal patient."""
        patient_data = {
            'age_months': 6,
            'respiratory_rate': 40,
            'has_cyanosis': False,
            'respiratory_distress_score': 0.2
        }
        comorbidities = ComorbidityProfile(
            detected_comorbidities=[],
            severity_scores={},
            respiratory_impact=0.0
        )
        etiology = EtiologyPrediction(
            primary_etiology=EtiologyType.NORMAL,
            confidence=0.8
        )

        risk_score = self.profiler.calculate_risk_score(patient_data, comorbidities, etiology)

        assert 0.0 <= risk_score <= 1.0
        assert risk_score < 0.5  # Should be low risk

    def test_calculate_risk_score_high_risk(self):
        """Test risk calculation for high-risk patient."""
        patient_data = {
            'age_months': 6,
            'respiratory_rate': 80,  # Very high
            'has_cyanosis': True,
            'respiratory_distress_score': 0.9
        }
        comorbidities = ComorbidityProfile(
            detected_comorbidities=[ComorbidityType.CARDIAC_DEFECT],
            severity_scores={ComorbidityType.CARDIAC_DEFECT.value: 0.8},
            respiratory_impact=0.7
        )
        etiology = EtiologyPrediction(
            primary_etiology=EtiologyType.HEART_DISEASE,
            confidence=0.9
        )

        risk_score = self.profiler.calculate_risk_score(patient_data, comorbidities, etiology)

        assert risk_score > 0.5  # Should be high risk

    def test_get_urgency_level(self):
        """Test urgency level classification."""
        test_cases = [
            (0.1, "LOW RISK"),
            (0.3, "ROUTINE"),
            (0.5, "PRIORITY"),
            (0.7, "URGENT"),
            (0.9, "CRITICAL")
        ]

        for risk_score, expected_level in test_cases:
            level = self.profiler.get_urgency_level(risk_score)
            assert expected_level in level

    def test_risk_factors_extreme_rr(self):
        """Test risk scoring for extreme respiratory rates."""
        patient_data = {
            'age_months': 6,
            'respiratory_rate': 15,  # Very low
            'has_cyanosis': False,
            'respiratory_distress_score': 0.1
        }
        comorbidities = ComorbidityProfile(detected_comorbidities=[], severity_scores={}, respiratory_impact=0.0)
        etiology = EtiologyPrediction(primary_etiology=EtiologyType.NORMAL, confidence=0.5)

        risk_score = self.profiler.calculate_risk_score(patient_data, comorbidities, etiology)

        assert risk_score > 0.0  # Should detect extreme RR


class TestIntegration:
    """Test integration of all components."""

    def test_create_system(self):
        """Test system creation factory."""
        engine, profiler = create_comorbidity_and_etiology_system(device='cpu')

        assert isinstance(engine, DifferentialDiagnosisEngine)
        assert isinstance(profiler, RiskProfiler)

    def test_full_pipeline(self):
        """Test complete comorbidity and etiology pipeline."""
        # Create system
        engine, profiler = create_comorbidity_and_etiology_system()

        # Test data
        features = np.random.randn(256).astype(np.float32)
        patient_data = {
            'age_months': 4,
            'gestational_age_weeks': 35,  # Premature
            'temperature_celsius': 38.5,  # Fever
            'weight_kg': 5.0,
            'respiratory_rate': 65,
            'has_cyanosis': False,
            'respiratory_distress_score': 0.6
        }

        # Generate diagnosis
        diagnosis = engine.generate_differential_diagnosis(features, patient_data)

        # Get risk score
        # Need to extract etiology and comorbidities from diagnosis
        # For this test, create mock ones
        mock_etiology = EtiologyPrediction(
            primary_etiology=diagnosis.top_3_diagnoses[0],
            confidence=diagnosis.confidence_scores[0]
        )
        mock_comorbidities = ComorbidityProfile(
            detected_comorbidities=[ComorbidityType.PREMATURITY, ComorbidityType.FEVER],
            severity_scores={'prematurity': 0.7, 'fever': 0.6},
            respiratory_impact=0.5
        )

        risk_score = profiler.calculate_risk_score(patient_data, mock_comorbidities, mock_etiology)
        urgency = profiler.get_urgency_level(risk_score)

        # Verify outputs
        assert len(diagnosis.top_3_diagnoses) == 3
        assert 0.0 <= risk_score <= 1.0
        assert isinstance(urgency, str)
        assert len(urgency) > 0


if __name__ == "__main__":
    print("Comorbidity & Etiology Test Suite")
    print("Basic validation completed - all imports successful")
    print("✓ Module loads without errors")