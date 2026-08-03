import joblib
import pandas as pd
import numpy as np
import os

class SupervisedFusion:
    """
    Inference engine for Supervised Fusion (Phase 12).
    Uses a trained Regressor (RandomForest/XGBoost) to predict GT BPM
    from noisy flow and depth signals.
    """
    
    def __init__(self, model_path="data/models/fusion_rf.pkl"):
        self.model = None
        self.output_dim = 1
        
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"[SupervisedFusion] Loaded model from {model_path}")
            except Exception as e:
                print(f"[SupervisedFusion] Error loading model: {e}")
        else:
            print(f"[SupervisedFusion] Model not found at {model_path}. Running in passthrough/fallback mode.")

    def predict(self, flow_bpm, depth_bpm, flow_conf, depth_conf):
        """
        Predicts fused BPM using the trained model.
        Returns: predicted_bpm (float) or None if model not loaded.
        """
        if self.model is None:
            return None
            
        # Prepare input dataframe (must match training columns)
        # feature_cols = ['flow_bpm', 'depth_bpm', 'flow_confidence', 'depth_confidence']
        
        X = pd.DataFrame([{
            'flow_bpm': flow_bpm,
            'depth_bpm': depth_bpm,
            'flow_confidence': flow_conf,
            'depth_confidence': depth_conf
        }])
        
        try:
            pred = self.model.predict(X)[0]
            return float(pred)
        except Exception as e:
            print(f"[SupervisedFusion] Inference Failed: {e}")
            return None
