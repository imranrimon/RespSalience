from typing import Dict, Any

class ARIDiagnostic:
    """Diagnostic logic for Acute Respiratory Infections in children."""
    
    @staticmethod
    def classify_rr(bpm: float, age_months: int, anomaly_score: float = 0.0) -> Dict[str, Any]:
        """
        Classifies respiratory rate based on age.
        WHO criteria for fast breathing (Tachypnea):
        - < 2 months: >= 60 bpm
        - 2–12 months: >= 50 bpm
        - 12–60 months: >= 40 bpm
        Also includes Apnea detection based on SSL Reconstruction Anomaly Score.
        """
        is_fast = False
        threshold = 0
        
        if age_months < 2:
            threshold = 60
            if bpm >= 60:
                is_fast = True
        elif age_months < 12:
            threshold = 50
            if bpm >= 50:
                is_fast = True
        elif age_months <= 60:
            threshold = 40
            if bpm >= 40:
                is_fast = True
        else:
            # Over 5 years (adult/older child)
            threshold = 20 # General adult threshold
            if bpm >= 20: 
                is_fast = True
        
        
        # Apnea Detection (SSL Anomaly Score threshold is empirical, assuming 0.5 here for demo)
        is_apnea = anomaly_score > 0.5
        
        status_msg = "Normal"
        if is_apnea:
            status_msg = "Apnea/Abnormal Breathing Detected"
        elif is_fast:
            status_msg = "Tachypnea (Fast Breathing)"
            
        clinical_sign = "No fast/abnormal breathing detected"
        if is_apnea:
            clinical_sign = "High probability of Apnea or severely irregular breathing"
        elif is_fast:
            clinical_sign = "Possible Acute Respiratory Infection (ARI)"

        return {
            "bpm": bpm,
            "age_months": age_months,
            "is_tachypnea": is_fast,
            "is_apnea": is_apnea,
            "anomaly_score": round(anomaly_score, 4),
            "threshold": threshold,
            "status": status_msg,
            "clinical_sign": clinical_sign
        }

    @staticmethod
    def analyze_breathing_pattern(depth_signal: list) -> Dict[str, Any]:
        """
        Analyzes the depth signal for abnormal patterns like chest indrawing.
        Note: This is a placeholder for more advanced signal analysis.
        """
        # Placeholder for indrawing detection logic
        # In actual implementation, we'd look for exaggerated retractions.
        return {
            "indrawing_detected": False, # Placeholder
            "pattern_regularity": 0.95 # Placeholder
        }
