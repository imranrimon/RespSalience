import numpy as np
import glob
import os

def check_data():
    files = glob.glob("data/ablation_results/signals/*.npz")
    print(f"Found {len(files)} files.")
    
    labels = []
    for f in files:
        try:
            d = np.load(f)
            if 'vlm_label' in d:
                labels.append(d['vlm_label'])
            else:
                print(f"File {f} missing vlm_label")
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not labels:
        print("No labels found.")
        return

    labels = np.array(labels)
    # Check if boolean or numeric
    print(f"Labels: True (Distress): {np.sum(labels==True)}, False (Normal): {np.sum(labels==False)}")
    print(f"Total: {len(labels)}")
    
    if np.sum(labels==True) == 0 or np.sum(labels==False) == 0:
        print("ALERT: MONO-CLASS DATASET DETECTED.")
        
if __name__ == "__main__":
    check_data()
