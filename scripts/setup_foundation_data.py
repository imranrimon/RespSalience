import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_synthetic_clinical_data(output_dir="data/foundation_data/synthetic", num_samples=5000):
    """
    Generates realistic 1D physiological waveforms perfectly paired with
    simulated clinical medical notes (Text) to train the CLISP model, and 
    raw tokens for RespMAE.
    
    This acts as a high-quality proxy dataset so we can train and benchmark
    the foundation models today, while waiting for PhysioNet (MIMIC-IV) approval.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "signals"), exist_ok=True)
    
    print(f"Generating {num_samples} synthetic Clinical Waveform + Text pairs...")
    
    metadata = []
    
    # Define clinical conditions to simulate
    conditions = [
        {"name": "Eupnea (Normal)", "bpm_range": (12, 20), "noise_level": 0.1, "amplitude": 1.0,
         "notes": ["Patient resting comfortably. Regular breathing pattern observed.",
                  "Respirations are unlabored and even.",
                  "Vitals stable, breathing at a normal rate.",
                  "Normal respiratory effort, no accessory muscle use."]},
                  
        {"name": "Tachypnea", "bpm_range": (25, 45), "noise_level": 0.2, "amplitude": 0.6,
         "notes": ["Patient exhibiting rapid, shallow breathing.",
                  "Tachypnea noted, patient appears mildly distressed.",
                  "Elevated respiratory rate, shallow tidal volume.",
                  "Breathing is fast and shallow due to suspected fever/pain."]},
                  
        {"name": "Bradypnea", "bpm_range": (6, 11), "noise_level": 0.05, "amplitude": 1.2,
         "notes": ["Slow respiratory rate observed, likely secondary to sedation.",
                  "Patient is bradypneic, monitor O2 sats closely.",
                  "Very slow, deep breathing pattern.",
                  "Respirations depressed, rate is below baseline."]},
                  
        {"name": "Apnea / Severe Occlusion", "bpm_range": (0, 0), "noise_level": 0.3, "amplitude": 0.0,
         "notes": ["CRITICAL: Apnea alarm triggered. No respiratory effort detected.",
                  "Patient stopped breathing for over 15 seconds.",
                  "Flatline respiratory signal, patient is apneic.",
                  "Sudden cessation of breathing."]},
                  
        {"name": "Cheyne-Stokes", "bpm_range": (15, 25), "noise_level": 0.15, "amplitude": "variable",
         "notes": ["Crescendo-decrescendo breathing pattern followed by apnea.",
                  "Classic Cheyne-Stokes respirations observed.",
                  "Periodic breathing with apneic pauses.",
                  "Waxing and waning tidal volumes."]}
    ]
    
    sr = 30  # 30 Hz sampling rate
    duration = 10  # 10 second windows
    seq_len = sr * duration
    t = np.linspace(0, duration, seq_len)
    
    for i in tqdm(range(num_samples)):
        pat_id = f"SYNTH_{i:05d}"
        
        # 1. Pick a random condition
        cond = np.random.choice(conditions, p=[0.4, 0.2, 0.1, 0.1, 0.2])
        
        # 2. Synthesize Signal
        if cond["name"] == "Apnea / Severe Occlusion":
            bpm = 0
            sig = np.random.randn(seq_len) * cond["noise_level"]
        elif cond["name"] == "Cheyne-Stokes":
            bpm = np.random.uniform(cond["bpm_range"][0], cond["bpm_range"][1])
            freq = bpm / 60.0
            base_wave = np.sin(2 * np.pi * freq * t)
            # Create the waxing/waning envelope
            envelope = np.sin(2 * np.pi * (0.05) * t)  # 20-second cycle
            envelope = np.clip(envelope, 0, 1) # Only positive half
            sig = base_wave * envelope + (np.random.randn(seq_len) * cond["noise_level"])
        else:
            bpm = np.random.uniform(cond["bpm_range"][0], cond["bpm_range"][1])
            freq = bpm / 60.0
            sig = np.sin(2 * np.pi * freq * t) * cond["amplitude"]
            
            # Add clinical artifacts (e.g., patient moving)
            if np.random.rand() > 0.7:
                sig += np.sin(2 * np.pi * 0.1 * t) * 1.5 # baseline wander
            if np.random.rand() > 0.8:
                spike_idx = np.random.randint(0, seq_len)
                sig[spike_idx:spike_idx+10] += 3.0 # motion spike
                
            sig += np.random.randn(seq_len) * cond["noise_level"]
            
        # Optional: Simulate complete signal loss (sensor disconnect) for Latent Diffusion training
        has_sensor_loss = False
        if np.random.rand() > 0.85:
            has_sensor_loss = True
            drop_start = np.random.randint(30, 240)
            drop_len = np.random.randint(30, 60)
            sig[drop_start:drop_start+drop_len] = np.random.randn(drop_len) * 5.0 # Massive artifact
        
        sig = sig.astype(np.float32)
        
        # 3. Save Signal
        sig_path = os.path.join(output_dir, "signals", f"{pat_id}.npz")
        np.savez_compressed(sig_path, signal=sig)
        
        # 4. Generate Clinical Text Note
        base_note = np.random.choice(cond["notes"])
        
        # Augment note with dynamic values to make it highly realistic
        if bpm > 0:
            bpm_text = f" RR is approx {int(bpm)} bpm."
        else:
            bpm_text = ""
            
        artifact_text = ""
        if has_sensor_loss:
            artifact_text = " Note: Transient sensor artifact/disconnect observed in recording."
            
        final_text = base_note + bpm_text + artifact_text
        
        metadata.append({
            "patient_id": pat_id,
            "condition": cond["name"],
            "true_bpm": float(bpm),
            "clinical_text": final_text,
            "has_sensor_loss": has_sensor_loss,
            "signal_path": os.path.abspath(sig_path)
        })
        
    # Save Metadata
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(output_dir, "clinical_notes_metadata.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*60)
    print("[v] Synthetic Clinical Dataset Generation Complete!")
    print(f"Total Samples: {num_samples}")
    print(f"Saved Metadata: {csv_path}")
    print("="*60)
    print("\n[INSTRUCTIONS FOR REAL MIMIC-IV DATA]")
    print("1. Create an account at https://physionet.org/")
    print("2. Complete the required CITI 'Data or Specimens Only Research' training course.")
    print("3. Sign the Data Use Agreement for 'MIMIC-IV Waveform Database'.")
    print("4. Use `wget` or the PhysioNet AWS bucket to download the matched waveforms and clinical notes.")
    print("   (Our CLISP model can easily swap this synthetic proxy data folder for the real MIMIC-IV folder).\n")

if __name__ == "__main__":
    generate_synthetic_clinical_data(num_samples=10000)
    
