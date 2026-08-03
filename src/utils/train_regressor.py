import sys
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.visualize_results import load_all_results

def train_fusion_model(output_path="data/models/fusion_rf.pkl"):
    print("Loading extracted features from ablation results...")
    
    # 1. Load Data
    try:
        df = load_all_results()
    except Exception as e:
        print(f"Error loading results: {e}")
        return
        
    print(f"Total records loaded: {len(df)}")
    
    # 2. Filter valid training data
    # We need rows that have 'flow_bpm', 'depth_bpm', 'gt_bpm'
    required_cols = ['flow_bpm', 'depth_bpm', 'flow_confidence', 'depth_confidence', 'gt_bpm']
    
    # Check if columns exist
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in data: {missing_cols}")
        print("Please run 'src/run_ablation_study.py --baseline Full-System' to generate features.")
        # Attempt to debug by showing head
        print(df.head())
        return

    # Drop NaNs
    train_df = df.dropna(subset=required_cols)
    print(f"Valid training records: {len(train_df)}")
    
    if len(train_df) < 10:
        print("Not enough data to train. Need at least 10 records.")
        return

    # 3. Prepare X, y
    feature_cols = ['flow_bpm', 'depth_bpm', 'flow_confidence', 'depth_confidence']
    X = train_df[feature_cols]
    y = train_df['gt_bpm']
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Train Model
    print(f"Training RandomForestRegressor on {len(X_train)} samples...")
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("-" * 40)
    print(f"Model Performance (Test Set N={len(X_test)}):")
    print(f"MAE:  {mae:.2f} BPM")
    print(f"RMSE: {rmse:.2f} BPM")
    print("-" * 40)
    
    # Compare with Unsupervised Baseline (Simple Average)
    # Check what the 'pred_bpm' (baseline) MAE is on this subset
    if 'pred_bpm' in train_df.columns:
        base_mae = mean_absolute_error(y_test, train_df.loc[X_test.index, 'pred_bpm'])
        print(f"Baseline (Unsupervised) MAE: {base_mae:.2f} BPM")
        print(f"Improvement: {base_mae - mae:.2f} BPM ({((base_mae - mae)/base_mae)*100:.1f}%)")
    
    # 7. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    train_fusion_model()
