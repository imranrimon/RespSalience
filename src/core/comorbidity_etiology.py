"""
Challenge #7: Comorbidity & Etiology Prediction

Predicts underlying cause of respiratory distress from multimodal data.
Distinguishes pneumonia, asthma, aspiration, heart disease, etc.

Key Components:
- EtiologyPredictor: Multi-class classification of respiratory condition types
- ComorbidityDetector: Detects concurrent health issues affecting respiration
- DifferentialDiagnosisEngine: Ranks likely diagnoses with confidence scores
- RiskProfiler: Identifies high-risk patients needing urgent referral

Author: Respiratory Rate Monitoring AI
Date: 2026
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import warnings


# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class EtiologyType(Enum):
    """Types of respiratory distress etiologies."""
    NORMAL = "normal"
    PNEUMONIA = "pneumonia"
    ASTHMA = "asthma"
    ASPIRATION = "aspiration"
    HEART_DISEASE = "heart_disease"
    PREMATURITY_RELATED = "prematurity_related"
    CROUP = "croup"
    BRONCHIOLITIS = "bronchiolitis"
    FOREIGN_BODY = "foreign_body"
    OTHER_INFECTION = "other_infection"


class ComorbidityType(Enum):
    """Types of comorbidities affecting respiration."""
    FEVER = "fever"
    MALNUTRITION = "malnutrition"
    DEVELOPMENTAL_DELAY = "developmental_delay"
    CARDIAC_DEFECT = "cardiac_defect"
    NEUROLOGICAL = "neurological"
    PREMATURITY = "prematurity"
    LOW_BIRTH_WEIGHT = "low_birth_weight"
    NONE = "none"


@dataclass
class EtiologyPrediction:
    """Prediction of respiratory condition etiology."""
    primary_etiology: EtiologyType
    confidence: float  # 0-1
    differential_diagnoses: List[Tuple[EtiologyType, float]] = field(
        default_factory=list
    )
    key_features: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class ComorbidityProfile:
    """Detected comorbidities for a patient."""
    detected_comorbidities: List[ComorbidityType] = field(default_factory=list)
    severity_scores: Dict[str, float] = field(default_factory=dict)  # {condition: severity}
    respiratory_impact: float = 0.0  # 0-1, how much affects respiration
    reasoning: str = ""


@dataclass
class DifferentialDiagnosisResult:
    """Result of differential diagnosis analysis."""
    top_3_diagnoses: List[EtiologyType]
    confidence_scores: List[float]
    clinical_reasoning: List[str]
    recommendation: str


# ============================================================================
# ETIOLOGY PREDICTOR
# ============================================================================

class EtiologyClassifier(nn.Module):
    """
    Neural network for classifying respiratory etiology.
    Takes multimodal features and predicts condition type.
    """
    
    def __init__(self, input_dim: int = 256, num_etiologies: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.num_etiologies = num_etiologies
        
        # Feature processing
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_etiologies)
        )
        
        # Uncertainty head (for confidence)
        self.uncertainty = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            logits: (batch, num_etiologies)
            confidence: (batch, 1) - model confidence in prediction
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        confidence = self.uncertainty(features)
        return logits, confidence


class EtiologyPredictor:
    """
    Predicts underlying causes of respiratory distress.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = EtiologyClassifier(input_dim=256, num_etiologies=10)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Etiology labels
        self.etiology_labels = [
            EtiologyType.NORMAL,
            EtiologyType.PNEUMONIA,
            EtiologyType.ASTHMA,
            EtiologyType.ASPIRATION,
            EtiologyType.HEART_DISEASE,
            EtiologyType.PREMATURITY_RELATED,
            EtiologyType.CROUP,
            EtiologyType.BRONCHIOLITIS,
            EtiologyType.FOREIGN_BODY,
            EtiologyType.OTHER_INFECTION
        ]
        
        # Feature importance for reasoning
        self.feature_patterns = {
            'pneumonia': [
                'abnormal breath sounds',
                'increased respiratory rate',
                'fever history',
                'chest asymmetry'
            ],
            'asthma': [
                'wheeze breath sounds',
                'rapid shallow breathing',
                'cough pattern',
                'activity-related distress'
            ],
            'aspiration': [
                'feeding history',
                'gag reflex abnormality',
                'wet breathing sounds',
                'unilateral involvement'
            ],
            'heart_disease': [
                'cyanosis indicators',
                'pulse irregularities',
                'murmur present',
                'orthopnea pattern'
            ]
        }
    
    def predict_etiology(self, features: np.ndarray) -> EtiologyPrediction:
        """
        Predict etiology from multimodal features.
        
        Args:
            features: (256,) feature vector
            
        Returns:
            EtiologyPrediction with confidence and differentials
        """
        try:
            # Convert to tensor
            features_tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
            features_tensor = features_tensor.to(self.device)
            
            with torch.no_grad():
                logits, confidence = self.model(features_tensor)
                probs = torch.softmax(logits, dim=1)
                
                # Get top 3 predictions
                top_k_probs, top_k_indices = torch.topk(probs, k=3, dim=1)
                
                primary_idx = int(top_k_indices[0, 0].cpu().numpy())
                primary_etiology = self.etiology_labels[primary_idx]
                primary_confidence = float(top_k_probs[0, 0].cpu().numpy())
                model_confidence = float(confidence[0, 0].cpu().numpy())
                
                # Adjust confidence (combination of softmax and model uncertainty)
                final_confidence = primary_confidence * (0.7 + model_confidence * 0.3)
                
                # Differential diagnoses
                differentials = [
                    (
                        self.etiology_labels[int(idx.cpu().numpy())],
                        float(prob.cpu().numpy())
                    )
                    for idx, prob in zip(
                        top_k_indices[0],
                        top_k_probs[0]
                    )
                ]
                
                # Generate reasoning
                reasoning = self._generate_reasoning(
                    primary_etiology,
                    final_confidence,
                    features
                )
                
                return EtiologyPrediction(
                    primary_etiology=primary_etiology,
                    confidence=final_confidence,
                    differential_diagnoses=differentials,
                    reasoning=reasoning
                )
        
        except Exception as e:
            return EtiologyPrediction(
                primary_etiology=EtiologyType.NORMAL,
                confidence=0.0,
                reasoning=f"Prediction error: {str(e)}"
            )
    
    def _generate_reasoning(self, etiology: EtiologyType, confidence: float,
                           features: np.ndarray) -> str:
        """Generate human-readable reasoning for diagnosis."""
        if confidence < 0.4:
            return f"Low confidence prediction for {etiology.value}. Clinical judgment needed."
        
        # Get key features for this etiology
        key_features = self.feature_patterns.get(etiology.value, [])
        
        if confidence > 0.7:
            severity = "high"
        elif confidence > 0.5:
            severity = "moderate"
        else:
            severity = "low"
        
        return (
            f"{severity.capitalize()} confidence ({confidence:.0%}) in {etiology.value}. "
            f"Key features: {', '.join(key_features[:2])}"
        )


# ============================================================================
# COMORBIDITY DETECTOR
# ============================================================================

class ComorbidityDetector:
    """
    Detects comorbidities that affect respiratory status.
    """
    
    def __init__(self):
        self.symptom_patterns = {
            ComorbidityType.FEVER: {
                'indicators': ['elevated_temperature', 'tachycardia'],
                'respiratory_impact': 0.6
            },
            ComorbidityType.MALNUTRITION: {
                'indicators': ['low_weight_for_age', 'weak_muscles'],
                'respiratory_impact': 0.5
            },
            ComorbidityType.PREMATURITY: {
                'indicators': ['early_birth', 'lung_immaturity'],
                'respiratory_impact': 0.8
            },
            ComorbidityType.CARDIAC_DEFECT: {
                'indicators': ['murmur', 'cyanosis'],
                'respiratory_impact': 0.9
            },
            ComorbidityType.NEUROLOGICAL: {
                'indicators': ['weak_cry', 'poor_muscle_tone'],
                'respiratory_impact': 0.7
            }
        }
    
    def detect_comorbidities(self, patient_data: Dict[str, any],
                            respiratory_features: np.ndarray) -> ComorbidityProfile:
        """
        Detect comorbidities from patient data and respiratory features.
        
        Args:
            patient_data: {age, birth_weight, symptoms, vital_signs, ...}
            respiratory_features: (256,) feature vector
            
        Returns:
            ComorbidityProfile with detected conditions
        """
        detected = []
        severity_scores = {}
        total_respiratory_impact = 0.0
        
        # Check for prematurity (high impact)
        if patient_data.get('gestational_age_weeks', 40) < 37:
            detected.append(ComorbidityType.PREMATURITY)
            severity_scores[ComorbidityType.PREMATURITY.value] = 0.8
            total_respiratory_impact += 0.7
        
        # Check for fever (check vital signs)
        if patient_data.get('temperature_celsius', 36.5) > 38.0:
            detected.append(ComorbidityType.FEVER)
            fever_severity = (patient_data.get('temperature_celsius', 36.5) - 37.5) / 2.0
            severity_scores[ComorbidityType.FEVER.value] = min(1.0, fever_severity)
            total_respiratory_impact += 0.5
        
        # Check for malnutrition (weight-for-age)
        if patient_data.get('weight_kg', 3) / max(patient_data.get('age_months', 6) / 12, 0.1) < 2.5:
            detected.append(ComorbidityType.MALNUTRITION)
            severity_scores[ComorbidityType.MALNUTRITION.value] = 0.6
            total_respiratory_impact += 0.4
        
        # Check for cardiac indicators (from respiratory features)
        # Features with high variance in chest wall might indicate cardiac issues
        chest_variance = np.var(respiratory_features[:50])  # First 50 features
        if chest_variance > 100:
            detected.append(ComorbidityType.CARDIAC_DEFECT)
            severity_scores[ComorbidityType.CARDIAC_DEFECT.value] = min(1.0, ((chest_variance - 100) / 100))
            total_respiratory_impact += 0.7
        
        # Cap total impact at 1.0
        total_respiratory_impact = min(1.0, total_respiratory_impact)
        
        reasoning = (
            f"Detected {len(detected)} comorbidities. "
            f"Total respiratory impact: {total_respiratory_impact:.0%}. "
            if detected else "No major comorbidities detected."
        )
        
        return ComorbidityProfile(
            detected_comorbidities=detected,
            severity_scores=severity_scores,
            respiratory_impact=total_respiratory_impact,
            reasoning=reasoning
        )


# ============================================================================
# DIFFERENTIAL DIAGNOSIS ENGINE
# ============================================================================

class DifferentialDiagnosisEngine:
    """
    Integrates etiology prediction and comorbidity detection
    to generate differential diagnoses with clinical reasoning.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.etiology_predictor = EtiologyPredictor(device)
        self.comorbidity_detector = ComorbidityDetector()
    
    def generate_differential_diagnosis(self,
                                       respiratory_features: np.ndarray,
                                       patient_data: Dict[str, any],
                                       base_diagnosis: str = "ARI") -> DifferentialDiagnosisResult:
        """
        Generate differential diagnosis with reasoning.
        
        Args:
            respiratory_features: (256,) multimodal features
            patient_data: Patient demographic and clinical data
            base_diagnosis: Base diagnosis from main system
            
        Returns:
            DifferentialDiagnosis with top 3 diagnoses and reasoning
        """
        # Get etiology prediction
        etiology = self.etiology_predictor.predict_etiology(respiratory_features)
        
        # Get comorbidities
        comorbidities = self.comorbidity_detector.detect_comorbidities(
            patient_data,
            respiratory_features
        )
        
        # Extract top 3
        top_3_diagnoses = [etiology.primary_etiology]
        confidence_scores = [etiology.confidence]
        
        # Add differentials
        for eti, conf in etiology.differential_diagnoses[1:3]:
            top_3_diagnoses.append(eti)
            confidence_scores.append(conf)
        
        # Adjust confidence scores based on comorbidities
        adjusted_scores = []
        for i, (diagnosis, base_conf) in enumerate(zip(top_3_diagnoses, confidence_scores)):
            # Comorbidities can increase/decrease likelihood
            if diagnosis == EtiologyType.HEART_DISEASE and ComorbidityType.CARDIAC_DEFECT in comorbidities.detected_comorbidities:
                adjusted_conf = min(1.0, base_conf * 1.3)
            elif diagnosis == EtiologyType.PREMATURITY_RELATED and ComorbidityType.PREMATURITY in comorbidities.detected_comorbidities:
                adjusted_conf = min(1.0, base_conf * 1.3)
            else:
                adjusted_conf = base_conf
            
            adjusted_scores.append(adjusted_conf)
        
        # Normalize to sum to 1.0
        total = sum(adjusted_scores)
        if total > 0:
            adjusted_scores = [s / total for s in adjusted_scores]
        
        # Generate reasoning
        clinical_reasoning = [
            f"{top_3_diagnoses[0].value}: {adjusted_scores[0]:.0%}",
            f"Comorbidities: {', '.join([c.value for c in comorbidities.detected_comorbidities]) or 'none'}",
            f"Respiratory impact from comorbidities: {comorbidities.respiratory_impact:.0%}"
        ]
        
        # Recommendation
        if confidence_scores[0] > 0.7:
            recommendation = f"High confidence in {top_3_diagnoses[0].value}. Recommend targeted management."
        elif any(c >= 0.6 for c in confidence_scores[:2]):
            recommendation = f"Consider both {top_3_diagnoses[0].value} and {top_3_diagnoses[1].value}. Further evaluation needed."
        else:
            recommendation = "Low confidence in any single etiology. Comprehensive clinical assessment recommended."
        
        return DifferentialDiagnosisResult(
            top_3_diagnoses=top_3_diagnoses,
            confidence_scores=adjusted_scores,
            clinical_reasoning=clinical_reasoning,
            recommendation=recommendation
        )


# ============================================================================
# RISK PROFILER
# ============================================================================

class RiskProfiler:
    """
    Identifies high-risk patients needing urgent referral.
    """
    
    def __init__(self):
        self.risk_criteria = {
            'critical_rr': lambda rr, age: rr > 70 or rr < 20,
            'cyanosis': lambda has_cyanosis: has_cyanosis,
            'severe_distress': lambda distress_score: distress_score > 0.8,
            'cardiac_disease': lambda has_it: has_it,
            'sepsis_indicators': lambda has_it: has_it
        }
    
    def calculate_risk_score(self, patient_data: Dict[str, any],
                            comorbidities: ComorbidityProfile,
                            etiology: EtiologyPrediction) -> float:
        """
        Calculate overall risk score (0-1) for patient.
        
        Returns:
            risk_score: 0-1, where 1.0 = highest risk
        """
        risk_factors = 0
        max_factors = 5
        
        # Rule 1: Extreme respiratory rates
        age = patient_data.get('age_months', 6)
        rr = patient_data.get('respiratory_rate', 40)
        if rr > 70 or rr < 15:
            risk_factors += 2
        elif rr > 60 or rr < 20:
            risk_factors += 1
        
        # Rule 2: Cyanosis or central abnormalities
        if patient_data.get('has_cyanosis', False):
            risk_factors += 2
        
        # Rule 3: Severe respiratory distress
        distress = patient_data.get('respiratory_distress_score', 0)
        if distress > 0.8:
            risk_factors += 2
        elif distress > 0.5:
            risk_factors += 1
        
        # Rule 4: Cardiac comorbidity
        if ComorbidityType.CARDIAC_DEFECT in comorbidities.detected_comorbidities:
            risk_factors += 1
        
        # Rule 5: Critical etiology predictions
        if etiology.primary_etiology in [
            EtiologyType.ASPIRATION,
            EtiologyType.FOREIGN_BODY,
            EtiologyType.HEART_DISEASE
        ]:
            risk_factors += 1
        
        risk_score = float(risk_factors) / float(max_factors)
        return min(1.0, risk_score)
    
    def get_urgency_level(self, risk_score: float) -> str:
        """Get urgency level based on risk score."""
        if risk_score >= 0.8:
            return "CRITICAL - Emergency referral needed immediately"
        elif risk_score >= 0.6:
            return "URGENT - Referral within 1 hour"
        elif risk_score >= 0.4:
            return "PRIORITY - Referral within 4 hours"
        elif risk_score >= 0.2:
            return "ROUTINE - Can be managed with close monitoring"
        else:
            return "LOW RISK - Continue routine monitoring"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_comorbidity_and_etiology_system(device: str = 'cpu'
                                          ) -> Tuple[DifferentialDiagnosisEngine, RiskProfiler]:
    """Factory to create comorbidity and etiology system."""
    engine = DifferentialDiagnosisEngine(device)
    profiler = RiskProfiler()
    return engine, profiler
