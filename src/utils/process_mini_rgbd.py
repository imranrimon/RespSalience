import cv2
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.depth_engine import DepthEngine

def process_subject(subject_id="01", data_root="data/MINI-RGBD/MINI-RGBD_web"):
    """
    Processes a MINI-RGBD subject folder and extracts respiration signal.
    """
    subject_path = os.path.join(data_root, subject_id, "depth")
    if not os.path.exists(subject_path):
        print(f"Error: Subject path {subject_path} not found.")
        return
    
    # Get all depth frames
    frames = sorted([f for f in os.listdir(subject_path) if f.endswith(".png")])
    print(f"Processing {len(frames)} frames for Subject {subject_id}...")
    
    # We don't have RGB for detector in this sub-folder (it's synthesized)
    # So we'll use a fixed ROI or a simple center crop for the chest
    # MINI-RGBD frames are 640x480 usually
    roi_box = np.array([100, 100, 300, 300]) # Dummy ROI for now
    
    engine = DepthEngine()
    all_depth_maps = []
    
    for i, frame_name in enumerate(frames):
        frame_path = os.path.join(subject_path, frame_name)
        # Depth frames are 16-bit or 8-bit PNGs. MINI-RGBD uses specific encoding.
        depth_map = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        all_depth_maps.append(depth_map)
        
        if (i+1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(frames)} frames...")
            
    print("Extracting signal from depth maps...")
    signals = engine.extract_signal_from_roi(np.array(all_depth_maps), roi_box)
    return signals

if __name__ == "__main__":
    subject = "01"
    signal = process_subject(subject)
    if signal is not None:
        print(f"Successfully extracted signal for Subject {subject}.")
        print(f"Signal shape: {signal.shape}")
        # Save signal
        output_dir = "data/outputs/MINI-RGBD"
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"subject_{subject}_signal.npy"), signal)
        print(f"Signal saved to {output_dir}")
