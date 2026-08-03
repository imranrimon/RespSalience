"""
Challenge #6: Personalized Respiratory Baseline Learning

Learns individual infant's baseline RR and detects deviations from their normal.
Reduces false positives by distinguishing normal variation from pathology.

Key Components:
- PersonalizedBaselineEstimator: Learns patient's unique RR baseline
- AdaptiveThresholdCalculator: Dynamic thresholds per patient
- BaselineTracker: Maintains running baseline with confidence intervals
- AnomalyDetectorPersonalized: Patient-specific anomaly detection

Author: Respiratory Rate Monitoring AI
Date: 2026
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from collections import deque
import warnings


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BaselineMeasurement:
    """Single measurement for baseline learning."""
    timestamp: float
    respiratory_rate: float
    confidence: float  # 0-1, quality of measurement
    conditions: Dict = field(default_factory=dict)  # sleep, activity level, etc.


@dataclass
class BaselineEstimate:
    """Estimated baseline for a patient."""
    mean_rr: float
    std_rr: float
    lower_bound: float  # 5th percentile
    upper_bound: float  # 95th percentile
    samples_used: int
    age_months: int
    last_update_samples: int = 0  # How many samples since last update
    confidence: float = 0.5  # 0-1, how confident in baseline


@dataclass
class AnomalyDetectionPersonalResult:
    """Result of personalized anomaly detection."""
    is_anomalous: bool
    anomaly_score: float  # 0-1, severity
    deviation_from_baseline: float  # BPM difference
    patient_specific_threshold: float
    reasoning: str
    recommendation: str


# ============================================================================
# PERSONALIZED BASELINE ESTIMATOR
# ============================================================================

class PersonalizedBaselineEstimator:
    """
    Learns individual infant's baseline respiratory rate and patterns.
    Maintains personalized thresholds for anomaly detection.
    """
    
    def __init__(self, window_size: int = 100, age_months: int = 6):
        """
        Initialize baseline estimator.
        
        Args:
            window_size: How many measurements to use for baseline
            age_months: Patient age for age-based priors
        """
        self.window_size = window_size
        self.age_months = age_months
        
        # Running window of measurements
        self.measurements: deque = deque(maxlen=window_size)
        
        # Baseline estimates
        self.baseline_estimate: Optional[BaselineEstimate] = None
        self.update_count = 0
        
        # Age-based priors (from Challenge #1)
        self.age_based_norms = {
            'sleeping': (25, 40),
            'quiet_awake': (30, 55),
            'active': (45, 75),
            'crying': (70, 110),
            'feeding': (40, 65)
        }
    
    def add_measurement(self, measurement: BaselineMeasurement) -> None:
        """Add a new measurement to the baseline calculation."""
        if measurement.confidence < 0.5:  # Skip low-confidence measurements
            return
        
        self.measurements.append(measurement)
        self.update_count += 1
        
        # Update baseline estimate every 10 measurements or when needed
        if self.update_count % 10 == 0 or self.baseline_estimate is None:
            self._update_baseline_estimate()
    
    def _update_baseline_estimate(self) -> None:
        """Update baseline estimate from current measurements."""
        if len(self.measurements) < 5:
            return
        
        # Extract RR values
        rr_values = np.array([m.respiratory_rate for m in self.measurements])
        
        # Filter outliers using IQR
        q1, q3 = np.percentile(rr_values, [25, 75])
        iqr = q3 - q1
        mask = (rr_values >= q1 - 1.5 * iqr) & (rr_values <= q3 + 1.5 * iqr)
        filtered_rr = rr_values[mask]
        
        if len(filtered_rr) < 3:
            filtered_rr = rr_values
        
        # Compute statistics
        mean_rr = np.mean(filtered_rr)
        std_rr = np.std(filtered_rr)
        
        # Use empirical percentiles
        lower_bound = np.percentile(filtered_rr, 5)
        upper_bound = np.percentile(filtered_rr, 95)
        
        # Confidence increases with number of samples
        confidence = min(0.95, 0.5 + len(filtered_rr) / 200.0)
        
        self.baseline_estimate = BaselineEstimate(
            mean_rr=float(mean_rr),
            std_rr=float(std_rr),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            samples_used=len(filtered_rr),
            age_months=self.age_months,
            last_update_samples=0,
            confidence=float(confidence)
        )
    
    def get_baseline(self) -> Optional[BaselineEstimate]:
        """Get current baseline estimate."""
        return self.baseline_estimate
    
    def get_expected_range(self, percent: float = 95.0) -> Tuple[float, float]:
        """
        Get expected respiratory rate range for normal variation.
        
        Args:
            percent: Percentile range (e.g., 95 for 95% confidence interval)
            
        Returns:
            (lower, upper) bounds in BPM
        """
        if self.baseline_estimate is None:
            # Use age-based priors
            lower, upper = self.age_based_norms.get('quiet_awake', (30, 55))
            return (lower, upper)
        
        # Use personalized baseline with confidence margins
        mean = self.baseline_estimate.mean_rr
        std = self.baseline_estimate.std_rr
        
        # Wider range with lower confidence
        margin_factor = 1.0 + (1.0 - self.baseline_estimate.confidence) * 0.5
        
        return (
            float(mean - 2 * std * margin_factor),
            float(mean + 2 * std * margin_factor)
        )
    
    def reset_for_new_patient(self, age_months: int = 6) -> None:
        """Reset baseline estimator for a new patient."""
        self.measurements.clear()
        self.baseline_estimate = None
        self.update_count = 0
        self.age_months = age_months


# ============================================================================
# ADAPTIVE THRESHOLD CALCULATOR
# ============================================================================

class AdaptiveThresholdCalculator:
    """
    Calculates personalized thresholds that adapt based on patient history.
    Reduces false alarms by using patient-specific normal ranges.
    """
    
    def __init__(self, baseline_estimator: PersonalizedBaselineEstimator):
        self.baseline_estimator = baseline_estimator
        self.threshold_history: List[Tuple[float, float]] = []
    
    def calculate_thresholds(self) -> Tuple[float, float, float]:
        """
        Calculate adaptive thresholds for this patient.
        
        Returns:
            (normal_lower, normal_upper, critical_lower/upper)
        """
        baseline = self.baseline_estimator.get_baseline()
        
        if baseline is None or baseline.samples_used < 10:
            # Not enough data, use conservative age-based thresholds
            if self.baseline_estimator.age_months <= 6:
                return (30, 55, 15)  # widest range for young infants
            elif self.baseline_estimator.age_months <= 12:
                return (28, 50, 14)
            else:
                return (25, 45, 12)
        
        # Personalized thresholds based on baseline
        normal_lower = max(
            baseline.mean_rr - 2 * baseline.std_rr,
            baseline.lower_bound
        )
        normal_upper = min(
            baseline.mean_rr + 2 * baseline.std_rr,
            baseline.upper_bound
        )
        
        # Critical threshold: beyond 3 standard deviations
        critical_threshold = 3 * baseline.std_rr
        
        # Track thresholds
        self.threshold_history.append((normal_lower, normal_upper))
        
        return (
            float(normal_lower),
            float(normal_upper),
            float(critical_threshold)
        )
    
    def get_adaptive_criticality(self, rr: float) -> float:
        """
        Calculate how critical RR is, personalized to patient.
        
        Returns:
            criticality: 0-1, where 0 = normal, 1 = critically abnormal
        """
        baseline = self.baseline_estimator.get_baseline()
        
        if baseline is None:
            # Distance in standard deviations from age-based mean
            age_mean = 40 if self.baseline_estimator.age_months <= 6 else 35
            distance = abs(rr - age_mean) / 10.0
            return min(1.0, distance)
        
        # Distance in patient's own standard deviations
        if baseline.std_rr < 1:
            std_rr = 5  # Minimum std to avoid division issues
        else:
            std_rr = baseline.std_rr
        
        z_score = abs(rr - baseline.mean_rr) / std_rr
        criticality = min(1.0, z_score / 3.0)  # 3 std = criticality 1.0
        
        return float(criticality)


# ============================================================================
# BASELINE TRACKER
# ============================================================================

class BaselineTracker:
    """
    Maintains baseline estimates over long patient monitoring periods.
    Adapts as patient's baseline changes (e.g., over weeks/months).
    """
    
    def __init__(self, patient_id: str, age_months: int = 6):
        self.patient_id = patient_id
        self.age_months = age_months
        
        # Estimator for each condition
        self.estimators = {
            'overall': PersonalizedBaselineEstimator(window_size=100, age_months=age_months),
            'sleeping': PersonalizedBaselineEstimator(window_size=50, age_months=age_months),
            'awake': PersonalizedBaselineEstimator(window_size=50, age_months=age_months)
        }
        
        # Tracking
        self.measurement_count = 0
        self.last_baseline_update_time = 0.0
    
    def add_measurement(self, rr: float, timestamp: float, 
                       is_sleeping: bool = False, confidence: float = 0.9) -> None:
        """Add a new respiratory rate measurement."""
        measurement = BaselineMeasurement(
            timestamp=timestamp,
            respiratory_rate=rr,
            confidence=confidence,
            conditions={'is_sleeping': is_sleeping}
        )
        
        # Add to overall
        self.estimators['overall'].add_measurement(measurement)
        
        # Add to conditional
        if is_sleeping:
            self.estimators['sleeping'].add_measurement(measurement)
        else:
            self.estimators['awake'].add_measurement(measurement)
        
        self.measurement_count += 1
    
    def get_baseline_estimate(self, condition: str = 'overall') -> Optional[BaselineEstimate]:
        """Get baseline for a specific condition."""
        if condition not in self.estimators:
            condition = 'overall'
        return self.estimators[condition].get_baseline()
    
    def get_expected_range(self, is_sleeping: bool = False) -> Tuple[float, float]:
        """Get expected RR range for current state."""
        condition = 'sleeping' if is_sleeping else 'awake'
        est = self.estimators[condition]
        return est.get_expected_range()


# ============================================================================
# PERSONALIZED ANOMALY DETECTOR
# ============================================================================

class AnomalyDetectorPersonalized:
    """
    Detects respiratory anomalies using patient-specific baselines.
    Reduces false positives by accounting for individual variation.
    """
    
    def __init__(self, baseline_tracker: BaselineTracker):
        self.baseline_tracker = baseline_tracker
        self.threshold_calculator = AdaptiveThresholdCalculator(
            baseline_tracker.estimators['overall']
        )
    
    def detect_anomaly(self, rr: float, is_sleeping: bool = False,
                      base_anomaly_score: float = 0.0) -> AnomalyDetectionPersonalResult:
        """
        Detect if RR is anomalous for this patient.
        
        Args:
            rr: Current respiratory rate
            is_sleeping: Whether patient is sleeping
            base_anomaly_score: Anomaly score from other detectors (0-1)
            
        Returns:
            AnomalyDetectionPersonalResult with personalized findings
        """
        # Get baselines
        overall_baseline = self.baseline_tracker.get_baseline_estimate('overall')
        state_baseline = self.baseline_tracker.get_baseline_estimate(
            'sleeping' if is_sleeping else 'awake'
        )
        
        # Get expected range for this state
        expected_lower, expected_upper = self.baseline_tracker.get_expected_range(is_sleeping)
        
        # Compute deviation
        if overall_baseline:
            deviation = rr - overall_baseline.mean_rr
        else:
            deviation = 0.0
        
        # Determine if anomalous
        is_anomalous = rr < expected_lower or rr > expected_upper
        
        # Calculate personalized anomaly score
        if overall_baseline is None or overall_baseline.samples_used < 10:
            # Not enough baseline data, rely on base anomaly score
            anomaly_score = base_anomaly_score
            reasoning = "Insufficient baseline data. Using population norms."
        else:
            # Use z-score approach
            z_score = abs(rr - overall_baseline.mean_rr) / (overall_baseline.std_rr + 1e-8)
            anomaly_score = min(1.0, base_anomaly_score * 0.5 + (z_score / 3.0) * 0.5)
            
            if is_anomalous:
                if z_score > 3:
                    reasoning = f"Critical deviation: {abs(deviation):.0f} BPM from patient baseline"
                else:
                    reasoning = f"Moderate deviation: {abs(deviation):.0f} BPM from patient baseline"
            else:
                reasoning = f"Within expected range for {['awake', 'sleeping'][is_sleeping]}"
        
        # Recommendation
        if anomaly_score > 0.7:
            recommendation = "ALERT: Check patient status. Consider intervention."
        elif anomaly_score > 0.4:
            recommendation = "CAUTION: Monitor closely. Recheck in 5 minutes."
        else:
            recommendation = "Normal. Continue routine monitoring."
        
        return AnomalyDetectionPersonalResult(
            is_anomalous=bool(is_anomalous),
            anomaly_score=float(anomaly_score),
            deviation_from_baseline=float(deviation),
            patient_specific_threshold=float((expected_upper - expected_lower) / 2.0),
            reasoning=reasoning,
            recommendation=recommendation
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_personalized_baseline_system(patient_id: str, age_months: int = 6
                                       ) -> Tuple[BaselineTracker, AnomalyDetectorPersonalized]:
    """Factory function to create personalized baseline system."""
    tracker = BaselineTracker(patient_id, age_months)
    detector = AnomalyDetectorPersonalized(tracker)
    return tracker, detector


def estimate_patient_trajectory_deviation(measurements: List[float],
                                         baseline_mean: float) -> float:
    """
    Estimate how much patient's RR trajectory deviates from their baseline.
    Useful for detecting slow deterioration.
    
    Returns:
        trajectory_deviation: 0-1, how unusual the trend is
    """
    if len(measurements) < 5:
        return 0.0
    
    # Fit linear trend
    x = np.arange(len(measurements))
    coeffs = np.polyfit(x, measurements, 1)
    trend_slope = coeffs[0]
    
    # How unusual is this slope?
    # Expect slopes around -0.5 to 0.5 BPM per measurement
    unusual_slope = abs(trend_slope) / 1.0
    deviation = min(1.0, unusual_slope)
    
    return float(deviation)
