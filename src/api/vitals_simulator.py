import pandas as pd
import requests
import time
import os

CSV_PATH = r"C:\Users\rimon\.cache\kagglehub\datasets\miadul\newborn-health-monitoring-dataset\versions\1\newborn_health_monitoring_with_risk.csv"
API_URL = "http://localhost:8000/vitals-check"

def run_simulation(limit=20):
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return
    
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Starting simulation (Processing first {limit} records)...")
    for index, row in df.head(limit).iterrows():
        # Age in months (CSV has age_days)
        age_months = int(row['age_days'] // 30)
        bpm = row['respiratory_rate_bpm']
        o2 = row['oxygen_saturation']
        baby_name = row['name']
        
        payload = {
            "age_months": age_months,
            "bpm": bpm,
            "oxygen_saturation": o2
        }
        
        print(f"[{index}] Sending data for {baby_name}: BPM={bpm}, O2={o2}%")
        try:
            response = requests.post(API_URL, data=payload)
            if response.status_code == 200:
                res_json = response.json()
                status = res_json['diagnosis']['status']
                rec = res_json['recommendation']
                warning = res_json.get('warning', '')
                
                print(f"  Result: {status} | Rec: {rec}")
                if warning:
                    print(f"  ⚠ WARNING: {warning}")
            else:
                print(f"  Error: API returned status {response.status_code}")
        except Exception as e:
            print(f"  Error: Could not connect to API: {e}")
            print("  Make sure telemedicine_api.py is running!")
            return
            
        time.sleep(1) # Slow down for visibility

if __name__ == "__main__":
    run_simulation()
