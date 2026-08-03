import cv2
import numpy as np
import sys
import os
import torch

# Add project root
sys.path.append(os.path.abspath("."))

from src.core.main_controller import RespiratoryController
from src.core.anatomical_stabilizer import AnatomicalStabilizer
from src.core.depth_engine import DepthEngine

def debug_video(video_path):
    print(f"DEBUGGING VIDEO: {video_path}")
    if not os.path.exists(video_path):
        print("Video not found!")
        return

    # 1. Check ROI
    stab = AnatomicalStabilizer()
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read video frame.")
        return
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi_xyxy = stab.get_stabilized_chest_roi(frame_rgb)
    print(f"ROI Result (XYXY expected): {roi_xyxy}")
    
    if roi_xyxy:
        x1, y1, x2, y2 = roi_xyxy
        print(f"ROI Stats: Width={x2-x1}, Height={y2-y1}")
        sliced = frame_rgb[y1:y2, x1:x2]
        print(f"Sliced ROI shape: {sliced.shape}")
        if sliced.size == 0:
            print("CRITICAL: Sliced ROI is empty!")
            
    # 2. Check Controller
    print("\nRunning Controller (Partial)...")
    ctrl = RespiratoryController(
        "src/config/test_infer.yaml",
        "data/models/VIRENet_Coarse2fine_Body_best.pth"
    )
    
    # Process video logic manually to see internals
    # (Or just call process_video and check logs)
    res = ctrl.process_video(video_path, age_months=6)
    print("\nController Result:")
    print(res)

if __name__ == "__main__":
    # Test on an AIR-400 video
    vid = "data/AIR_400/S01/1.mp4"
    if len(sys.argv) > 1:
        vid = sys.argv[1]
    debug_video(vid)
