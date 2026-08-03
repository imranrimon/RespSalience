from ultralytics import YOLO
import cv2
import numpy as np
import os

class AnatomicalStabilizer:
    """Uses YOLOv8-Pose landmarks to stabilize and track anatomical ROIs."""
    
    def __init__(self, model_path="data/yolov8n-pose.pt"):
        print(f"Initializing AnatomicalStabilizer with {model_path}...")
        self.model = YOLO(model_path)
        
    def get_stabilized_chest_roi(self, frame_rgb: np.ndarray):
        """
        Detects body keypoints and derives a chest ROI.
        Returns [x, y, w, h] or None if detection fails.
        """
        results = self.model(frame_rgb, verbose=False)
        
        if not results or len(results[0].keypoints) == 0:
            return None
            
        # Get keypoints for the first person detected
        # Keypoints: [0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear, 5: l_shoulder, 6: r_shoulder, 
        # 7: l_elbow, 8: r_elbow, 9: l_wrist, 10: r_wrist, 11: l_hip, 12: r_hip...]
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        
        # We need shoulders (5, 6) and hips (11, 12)
        if len(kpts) < 13:
            return None
            
        l_shoulder = kpts[5]
        r_shoulder = kpts[6]
        l_hip = kpts[11]
        r_hip = kpts[12]
        
        # Check if points are valid (not [0, 0])
        if np.all(l_shoulder == 0) or np.all(r_shoulder == 0):
            return None
            
        h, w, _ = frame_rgb.shape
        
        # Calculate chest center and dimensions
        shoulder_mid = [(l_shoulder[0] + r_shoulder[0])/2, (l_shoulder[1] + r_shoulder[1])/2]
        
        # If hips are missing, estimate height based on shoulder distance
        if np.all(l_hip == 0) or np.all(r_hip == 0):
            roi_h = abs(l_shoulder[0] - r_shoulder[0]) * 1.2
            center_y = shoulder_mid[1] + (roi_h * 0.4)
        else:
            hip_mid = [(l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2]
            center_y = (shoulder_mid[1] + hip_mid[1]) / 2
            roi_h = abs(shoulder_mid[1] - hip_mid[1]) * 0.8
            
        center_x = shoulder_mid[0]
        
        # Width is distance between shoulders with some padding
        roi_w = abs(l_shoulder[0] - r_shoulder[0]) * 1.6
        
        x = int(center_x - roi_w/2)
        y = int(center_y - roi_h/2)
        
        # Clamp to image bounds
        x = max(0, min(w-1, x))
        y = max(0, min(h-1, y))
        roi_w = max(10, min(w - x, int(roi_w)))
        roi_h = max(10, min(h - y, int(roi_h)))
        
        return [int(x), int(y), int(x + roi_w), int(y + roi_h)]
    
    def close(self):
        pass # YOLO model handles cleanup
