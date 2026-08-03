"""
Longitudinal Disease Trajectory Modeling (Challenge #3)

Tracks respiratory measurements over time to:
1. Detect disease progression (improving vs deteriorating)
2. Predict next-day status
3. Generate early warning scores
4. Identify intervention points

Clinical Motivation:
- A single BPM reading doesn't tell the full story
- Trends matter: "BPM rising over 6 hours" is worse than "steady BPM"
- Early warning scores help prevent ICU admission
- Clinicians need to know: "Will this baby need intervention in 24 hours?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import json


class RespiratoryMeasurement:
    """
    Single point-in-time respiratory measurement.
    Represents one video/patient assessment.
    """
    
    def __init__(self, timestamp: datetime, bpm: float, anomaly_score: float = 0.0,
                 confidence: float = 0.9, age_months: int = 6, 
                 diagnosis_status: str = "normal"):
        self.timestamp = timestamp
        self.bpm = bpm
        self.anomaly_score = anomaly_score  # 0 = normal, 1 = apnea
        self.confidence = confidence  # Model confidence in the measurement
        self.age_months = age_months
        self.status = diagnosis_status  # "normal", "tachypnea", "apnea", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'bpm': self.bpm,
            'anomaly_score': self.anomaly_score,
            'confidence': self.confidence,
            'age_months': self.age_months,
            'status': self.status
        }
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to feature vector for ML."""
        return torch.tensor([
            self.bpm,
            self.anomaly_score,
            self.confidence,
            float(self.age_months)
        ], dtype=torch.float32)


class LongitudinalRespiratoryTracker:
    """
    Maintains a history of respiratory measurements over time.
    Computes trends and prepares sequences for trajectory prediction.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Args:
            max_history: Maximum number of measurements to keep in memory
        """
        self.measurements: deque = deque(maxlen=max_history)
        self.max_history = max_history
    
    def add_measurement(self, measurement: RespiratoryMeasurement):
        """Add a new respiratory measurement to the history."""
        self.measurements.append(measurement)
    
    def get_bpm_trend(self, window: int = 5) -> Dict[str, float]:
        """
        Compute BPM trend over recent measurements.
        
        Returns:
            Dict with trend metrics:
            - bpm_mean: Average BPM in recent window
            - bpm_std: Variability in BPM
            - bpm_slope: Rate of change (positive = increasing)
            - bpm_acceleration: Second derivative (positive = worsening acceleration)
        """
        if len(self.measurements) < 2:
            return {
                'bpm_mean': 0.0,
                'bpm_std': 0.0,
                'bpm_slope': 0.0,
                'bpm_acceleration': 0.0
            }
        
        recent = list(self.measurements)[-window:]
        bpms = np.array([m.bpm for m in recent])
        
        mean_bpm = float(np.mean(bpms))
        std_bpm = float(np.std(bpms))
        
        # Linear regression slope (rate of change)
        x = np.arange(len(bpms))
        if np.std(x) > 0:
            slope = float(np.polyfit(x, bpms, 1)[0])
        else:
            slope = 0.0
        
        # Second derivative (acceleration)
        if len(bpms) >= 3:
            diffs = np.diff(bpms)
            acceleration = float(np.mean(np.diff(diffs)))
        else:
            acceleration = 0.0
        
        return {
            'bpm_mean': mean_bpm,
            'bpm_std': std_bpm,
            'bpm_slope': slope,
            'bpm_acceleration': acceleration
        }
    
    def get_anomaly_trend(self, window: int = 5) -> Dict[str, float]:
        """Compute anomaly score trends (apnea risk)."""
        if len(self.measurements) < 2:
            return {
                'anomaly_mean': 0.0,
                'anomaly_max': 0.0,
                'anomaly_slope': 0.0
            }
        
        recent = list(self.measurements)[-window:]
        anomalies = np.array([m.anomaly_score for m in recent])
        
        mean_anom = float(np.mean(anomalies))
        max_anom = float(np.max(anomalies))
        
        x = np.arange(len(anomalies))
        if np.std(x) > 0:
            slope = float(np.polyfit(x, anomalies, 1)[0])
        else:
            slope = 0.0
        
        return {
            'anomaly_mean': mean_anom,
            'anomaly_max': max_anom,
            'anomaly_slope': slope
        }
    
    def get_sequence(self, seq_len: int = 24) -> torch.Tensor:
        """
        Get last N measurements as a sequence tensor for model input.
        
        Returns:
            (seq_len, 4) tensor where each row is [bpm, anomaly, confidence, age]
        """
        recent = list(self.measurements)[-seq_len:]
        
        # Pad with zeros if not enough history
        if len(recent) < seq_len:
            padding = [RespiratoryMeasurement(
                timestamp=datetime.now(),
                bpm=0.0, anomaly_score=0.0, confidence=0.0, age_months=0
            )] * (seq_len - len(recent))
            recent = padding + recent
        
        tensors = [m.to_tensor() for m in recent]
        return torch.stack(tensors)  # (seq_len, 4)
    
    def get_history_dataframe(self) -> Dict[str, List]:
        """Convert history to dictionary format for logging."""
        return {
            'timestamps': [m.timestamp.isoformat() for m in self.measurements],
            'bpms': [m.bpm for m in self.measurements],
            'anomaly_scores': [m.anomaly_score for m in self.measurements],
            'statuses': [m.status for m in self.measurements],
        }


class TrajectoryPredictor(nn.Module):
    """
    LSTM-based model for predicting respiratory trajectory.
    
    Input: Sequence of (bpm, anomaly_score, confidence, age)
    Output: Trajectory class (improving, stable, deteriorating)
    
    Also predicts:
    - Next BPM
    - Probability of intervention in next 24h
    - Clinical risk score
    """
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Trajectory prediction head (improving, stable, deteriorating)
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),  # 3 classes
            nn.Softmax(dim=-1)
        )
        
        # BPM prediction head (continuous)
        self.bpm_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Predict next BPM
        )
        
        # Risk scoring head (intervention probability)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Probability in [0, 1]
        )
        
        # Attention weights for interpretability
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            sequence: (B, SeqLen, 4) tensor of measurements
            
        Returns:
            Dict with:
            - trajectory: (B, 3) logits [improving, stable, deteriorating]
            - trajectory_class: (B,) predicted class
            - bpm_next: (B, 1) predicted next BPM
            - risk_score: (B, 1) probability of intervention in 24h
            - attention_weights: (B, SeqLen) attention over time
        """
        B = sequence.shape[0]
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(sequence)  # lstm_out: (B, SeqLen, hidden_dim)
        
        # Use last hidden state for predictions
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_dim)
        
        # Trajectory prediction
        trajectory_logits = self.trajectory_head(last_hidden)  # (B, 3)
        trajectory_class = trajectory_logits.argmax(dim=-1)  # (B,)
        
        # BPM prediction for next measurement
        bpm_pred = self.bpm_head(last_hidden)  # (B, 1)
        
        # Intervention risk score
        risk_score = self.risk_head(last_hidden)  # (B, 1)
        
        # Compute attention weights over sequence for interpretability
        attn_out, attn_weights = self.attention(
            last_hidden.unsqueeze(1),  # (B, 1, hidden_dim) - query
            lstm_out,  # (B, SeqLen, hidden_dim) - key/value
            lstm_out
        )
        # attn_weights: (B, 1, SeqLen) -> squeeze to (B, SeqLen)
        attn_weights = attn_weights.squeeze(1)
        
        return {
            'trajectory_logits': trajectory_logits,
            'trajectory_class': trajectory_class,  # 0=improving, 1=stable, 2=deteriorating
            'bpm_next': bpm_pred,
            'risk_score': risk_score,
            'attention_weights': attn_weights
        }


class EarlyWarningSystem:
    """
    Rule-based + ML hybrid system for generating clinical alerts.
    
    Monitors:
    1. BPM trends (rising BPM = concern)
    2. Anomaly score (increasing apnea risk)
    3. Measurement confidence (unreliable measurements)
    4. Trajectory prediction (model forecast)
    5. Time since last measurement (gaps in data)
    """
    
    def __init__(self, bpm_threshold_multiplier: float = 1.2,
                 anomaly_threshold: float = 0.6,
                 risk_score_threshold: float = 0.7):
        """
        Args:
            bpm_threshold_multiplier: Consider at-risk if BPM > baseline × this
            anomaly_threshold: High apnea risk if anomaly > this
            risk_score_threshold: Alert if ML model risk > this
        """
        self.bpm_threshold_multiplier = bpm_threshold_multiplier
        self.anomaly_threshold = anomaly_threshold
        self.risk_score_threshold = risk_score_threshold
    
    def compute_warning_level(self, tracker: LongitudinalRespiratoryTracker,
                              trajectory_output: Dict = None) -> Dict[str, Any]:
        """
        Compute overall warning level (green/yellow/red).
        
        Returns:
            Dict with:
            - level: "GREEN" (normal), "YELLOW" (monitoring), "RED" (alert)
            - score: 0-100 cumulative risk score
            - factors: List of triggered risk factors
            - recommendations: Suggested actions
        """
        if len(tracker.measurements) == 0:
            return {'level': 'GRAY', 'score': 0, 'factors': ['No data'], 'recommendations': []}
        
        factors = []
        score = 0
        
        # Get latest measurement
        latest = list(tracker.measurements)[-1]
        
        # ─────────────────────────────────────────
        # 1. BPM Rising (concerning sign)
        # ─────────────────────────────────────────
        bpm_trend = tracker.get_bpm_trend(window=5)
        if bpm_trend['bpm_slope'] > 2.0:  # BPM rising > 2/measurement
            factors.append("BPM consistently rising")
            score += 20
        
        if bpm_trend['bpm_acceleration'] > 0.5:  # Acceleration worsening
            factors.append("Breathing rate acceleration detected")
            score += 15
        
        # ─────────────────────────────────────────
        # 2. Apnea Risk (anomaly scores high)
        # ─────────────────────────────────────────
        anomaly_trend = tracker.get_anomaly_trend(window=5)
        if anomaly_trend['anomaly_max'] > self.anomaly_threshold:
            factors.append(f"Apnea detected (anomaly={anomaly_trend['anomaly_max']:.2f})")
            score += 30
        
        if anomaly_trend['anomaly_slope'] > 0.05:  # Anomaly worsening
            factors.append("Apnea events increasing")
            score += 20
        
        # ─────────────────────────────────────────
        # 3. Measurement Reliability
        # ─────────────────────────────────────────
        if latest.confidence < 0.7:
            factors.append(f"Low measurement confidence ({latest.confidence:.0%})")
            score += 10
        
        # ─────────────────────────────────────────
        # 4. ML Trajectory Prediction
        # ─────────────────────────────────────────
        if trajectory_output:
            traj_class = trajectory_output['trajectory_class'].item()
            risk_score = trajectory_output['risk_score'].item()
            
            if traj_class == 2:  # Deteriorating
                factors.append("ML predicts deteriorating trajectory")
                score += 25
            elif traj_class == 1:  # Stable (neutral)
                pass
            else:  # Improving
                score -= 5  # Good sign
            
            if risk_score > self.risk_score_threshold:
                factors.append(f"ML intervention risk: {risk_score:.0%}")
                score += 20
        
        # ─────────────────────────────────────────
        # 5. Absolute Thresholds
        # ─────────────────────────────────────────
        if latest.status == "apnea":
            factors.append("Active apnea detected")
            score += 40
        elif latest.status == "critical":
            factors.append("Critical respiratory distress")
            score += 50
        elif latest.status == "tachypnea":
            factors.append("Tachypnea (fast breathing)")
            score += 15
        
        # ─────────────────────────────────────────
        # Determine Warning Level
        # ─────────────────────────────────────────
        score = max(0, min(100, score))  # Clamp to [0, 100]
        
        if score >= 70:
            level = "RED"
        elif score >= 40:
            level = "YELLOW"
        else:
            level = "GREEN"
        
        # ─────────────────────────────────────────
        # Generate Recommendations
        # ─────────────────────────────────────────
        recommendations = []
        
        if latest.status == "apnea" or latest.status == "critical":
            recommendations.append("URGENT: Contact physician immediately")
            recommendations.append("Consider ICU admission")
        elif score >= 70:
            recommendations.append("Increase monitoring frequency (every 15-30 min)")
            recommendations.append("Prepare for possible intervention")
            recommendations.append("Notify senior clinician")
        elif score >= 40:
            recommendations.append("Continue regular monitoring (hourly)")
            recommendations.append("Check sensor calibration")
            recommendations.append("Reassess in 4 hours")
        else:
            recommendations.append("Routine monitoring (4-6 hourly)")
        
        return {
            'level': level,
            'score': int(score),
            'factors': factors,
            'recommendations': recommendations,
            'timestamp': latest.timestamp.isoformat(),
            'latest_bpm': latest.bpm,
            'latest_anomaly': latest.anomaly_score
        }


class LongitudinalSurveillanceSystem:
    """
    Complete surveillance system combining:
    - Measurement tracking
    - Trajectory prediction
    - Early warning scoring
    - Clinical reporting
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.tracker = LongitudinalRespiratoryTracker(max_history=200)
        self.trajectory_model = TrajectoryPredictor(hidden_dim=64).to(device)
        self.trajectory_model.eval()
        self.warning_system = EarlyWarningSystem()
        self.device = device
    
    def add_measurement(self, bpm: float, anomaly_score: float = 0.0,
                       confidence: float = 0.9, age_months: int = 6,
                       diagnosis_status: str = "normal"):
        """Add a new respiratory measurement."""
        measurement = RespiratoryMeasurement(
            timestamp=datetime.now(),
            bpm=bpm,
            anomaly_score=anomaly_score,
            confidence=confidence,
            age_months=age_months,
            diagnosis_status=diagnosis_status
        )
        self.tracker.add_measurement(measurement)
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive surveillance report.
        
        Returns:
            Dict with:
            - patient_status: Current snapshot
            - trends: Historical trends
            - trajectory_prediction: ML-predicted future
            - early_warning: Risk assessment
            - clinical_action_needed: Boolean flag for alerts
        """
        if len(self.tracker.measurements) == 0:
            return {'error': 'No measurements yet'}
        
        latest = list(self.tracker.measurements)[-1]
        
        # Get trends
        bpm_trend = self.tracker.get_bpm_trend(window=5)
        anomaly_trend = self.tracker.get_anomaly_trend(window=5)
        
        # Get trajectory prediction
        trajectory_output = None
        if len(self.tracker.measurements) >= 5:
            with torch.no_grad():
                seq = self.tracker.get_sequence(seq_len=24).unsqueeze(0).to(self.device)
                trajectory_output = self.trajectory_model(seq)
                
                # Dequantify outputs
                traj_class = trajectory_output['trajectory_class'].cpu().item()
                risk_score = trajectory_output['risk_score'].cpu().item()
        
        # Get early warning
        warning = self.warning_system.compute_warning_level(self.tracker, trajectory_output)
        
        # Build report
        report = {
            'patient_snapshot': {
                'timestamp': latest.timestamp.isoformat(),
                'bpm': round(latest.bpm, 1),
                'anomaly_score': round(latest.anomaly_score, 4),
                'status': latest.status,
                'age_months': latest.age_months,
                'measurement_confidence': round(latest.confidence, 2)
            },
            
            'trends': {
                'bpm': {
                    'mean': round(bpm_trend['bpm_mean'], 1),
                    'std': round(bpm_trend['bpm_std'], 1),
                    'slope': round(bpm_trend['bpm_slope'], 2),  # Positive = worsening
                    'acceleration': round(bpm_trend['bpm_acceleration'], 3)
                },
                'anomaly': {
                    'mean': round(anomaly_trend['anomaly_mean'], 3),
                    'max': round(anomaly_trend['anomaly_max'], 3),
                    'slope': round(anomaly_trend['anomaly_slope'], 4)  # Positive = worsening
                },
                'measurements_count': len(self.tracker.measurements)
            },
            
            'early_warning': warning,
            
            'clinical_action_needed': warning['level'] in ['YELLOW', 'RED'],
        }
        
        # Add trajectory if available
        if trajectory_output:
            traj_names = ['Improving', 'Stable', 'Deteriorating']
            report['trajectory'] = {
                'predicted_class': traj_names[traj_class],
                'class_probabilities': {
                    'improving': round(trajectory_output['trajectory_logits'][0, 0].item(), 2),
                    'stable': round(trajectory_output['trajectory_logits'][0, 1].item(), 2),
                    'deteriorating': round(trajectory_output['trajectory_logits'][0, 2].item(), 2)
                },
                'predicted_next_bpm': round(trajectory_output['bpm_next'][0, 0].item(), 1),
                'intervention_risk': round(risk_score, 2)
            }
        
        report['data_history'] = self.tracker.get_history_dataframe()
        
        return report
    
    def load_trajectory_model(self, checkpoint_path: str):
        """Load pre-trained trajectory model."""
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            self.trajectory_model.load_state_dict(checkpoint['model_state'])
        else:
            self.trajectory_model.load_state_dict(checkpoint)
        
        print(f"[LongitudinalSurveillance] Loaded trajectory model from {checkpoint_path}")
    
    def save_trajectory_model(self, checkpoint_path: str):
        """Save trajectory model checkpoint."""
        torch.save({
            'model_state': self.trajectory_model.state_dict(),
            'model_type': 'TrajectoryPredictor',
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        print(f"[LongitudinalSurveillance] Saved trajectory model to {checkpoint_path}")


if __name__ == "__main__":
    # Quick test
    print("Testing LongitudinalSurveillanceSystem...")
    
    system = LongitudinalSurveillanceSystem(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Simulate measurements over time (deteriorating scenario)
    print("\n[Scenario: Infant's condition deteriorating over time]")
    timestamps = [datetime.now() - timedelta(hours=6-i) for i in range(6)]
    
    bpm_sequence = [38, 40, 42, 45, 48, 52]  # Rising BPM
    anomaly_sequence = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]  # Increasing apnea
    
    for i, (bpm, anomaly) in enumerate(zip(bpm_sequence, anomaly_sequence)):
        status = "normal"
        if bpm > 50: status = "tachypnea"
        if anomaly > 0.5: status = "apnea"
        
        system.add_measurement(
            bpm=bpm,
            anomaly_score=anomaly,
            confidence=0.95 if i < 5 else 0.7,  # Last measurement less confident
            age_months=6,
            diagnosis_status=status
        )
        print(f"  Hour {6-i}: BPM={bpm}, Anomaly={anomaly:.2f}")
    
    # Generate report
    report = system.get_report()
    
    print("\n[Surveillance Report]")
    print(f"Status: {report['early_warning']['level']}")
    print(f"Risk Score: {report['early_warning']['score']}/100")
    print(f"\nFactors:")
    for factor in report['early_warning']['factors']:
        print(f"  - {factor}")
    print(f"\nRecommendations:")
    for rec in report['early_warning']['recommendations']:
        print(f"  • {rec}")
    
    if 'trajectory' in report:
        print(f"\nTrajectory Prediction: {report['trajectory']['predicted_class']}")
        print(f"  Intervention Risk: {report['trajectory']['intervention_risk']:.0%}")
    
    print("\n[OK] System initialized successfully")
