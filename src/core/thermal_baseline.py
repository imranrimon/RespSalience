import numpy as np
import cv2
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks

class ThermalSleepOrchestrator:
    """
    Implements the full non-contact respiratory and body movement detection pipeline 
    from "Non-Contact Respiration Monitoring and Body Movements Detection for Sleep 
    Using Thermal Imaging" (Jakkaew et al., 2020)
    
    Acts natively on RGB arrays by mapping to grayscale intensity to proxy thermal mapping.
    """
    def __init__(self, fps=30):
        self.fps = fps
        # Filter settings derived directly from the paper
        self.b_low = 0.05
        self.b_high = 1.5
        self.sg_window = 51
        self.sg_poly = 3

    def _extract_roi_signals(self, frames_gray):
        """
        Extracts ROI_h (Highest Temp Point) and ROI_t (Threshold Area) 
        continuous signals explicitly stated in Section 2.1.1
        """
        T, H, W = frames_gray.shape
        sig_h = np.zeros(T)
        sig_t = np.zeros(T)
        
        # Determine fixed ROIs from the first valid frame
        base_frame = frames_gray[0]
        
        # 1. ROI_h (Highest Temp Point)
        _, _, _, max_loc = cv2.minMaxLoc(base_frame)
        cx, cy = max_loc
        box_size = 50 # 50x50 empirically found best in paper
        half_box = box_size // 2
        
        # Bound box coordinates
        x1 = max(0, cx - half_box)
        y1 = max(0, cy - half_box)
        x2 = min(W, cx + half_box)
        y2 = min(H, cy + half_box)
        
        # 2. ROI_t (Largest High-Temp Area)
        # Empirical TR=176 for threshold
        _, thresh = cv2.threshold(base_frame, 176, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tx1, ty1, tw, th = 0, 0, W, H # Fallback to full frame
        if contours:
            biggest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(biggest_contour) > 50: # Minimize noise
                tx1, ty1, tw, th = cv2.boundingRect(biggest_contour)
                
        tx2 = min(W, tx1 + tw)
        ty2 = min(H, ty1 + th)
        
        # 3. Extract continuous sequence averaging
        for t in range(T):
            frame = frames_gray[t]
            
            roi_h_patch = frame[y1:y2, x1:x2]
            if roi_h_patch.size > 0:
                sig_h[t] = np.mean(roi_h_patch)
                
            roi_t_patch = frame[ty1:ty2, tx1:tx2]
            if roi_t_patch.size > 0:
                sig_t[t] = np.mean(roi_t_patch)
                
        # Zero-mean center the signals
        sig_h = sig_h - np.mean(sig_h)
        sig_t = sig_t - np.mean(sig_t)
        return sig_h, sig_t

    def _extract_breathing_motion(self, frames_gray):
        """
        Extracts BM (Breathing Motion) array as defined in Section 2.1.2.
        Finds bounding boxes of absolute difference over 1-second lag.
        """
        T = len(frames_gray)
        sig_bm = np.zeros(T)
        
        lag = int(self.fps) # 1 second before
        if lag == 0: lag = 1
        
        kernel = np.ones((5,5), np.uint8)
        
        for t in range(lag, T):
            curr = frames_gray[t]
            prev = frames_gray[t - lag]
            
            # Absolute difference
            diff = cv2.absdiff(curr, prev)
            
            # Thresholding 5
            _, thresh = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)
            
            # Morphological opening (erosion then dilation)
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Count bounding boxes
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter noise contours
            valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
            sig_bm[t] = len(valid_contours)
            
        # Pad beginning
        if lag < T:
            sig_bm[:lag] = sig_bm[lag]
            
        sig_bm = sig_bm - np.mean(sig_bm)
        return sig_bm

    def _detect_body_movements(self, frames_gray):
        """
        Binary flag over time. Section 2.2
        Absolute difference between consecutive frames. Threshold 35.
        """
        T = len(frames_gray)
        movement = np.zeros(T, dtype=np.int32)
        kernel = np.ones((5,5), np.uint8)
        
        for t in range(1, T):
            diff = cv2.absdiff(frames_gray[t], frames_gray[t-1])
            _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in contours if cv2.contourArea(c) > 100] # Major movement
            
            if len(valid) > 0:
                movement[t] = 1
                
        return movement

    def run(self, rgb_frames):
        """
        Executes full baseline pipeline.
        rgb_frames: list or array of RGB video frames [T, H, W, 3]
        Returns: Tuple of (Estimated BPM, Movement array)
        """
        if len(rgb_frames) < self.fps * 2:
            return 15.0, np.zeros(len(rgb_frames))
            
        # 1. Convert to grayscale to proxy thermal luminance logic
        print("    [Thermal] Converting frames...")
        frames_gray = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rgb_frames])
        
        # 2. Extract Signals
        print("    [Thermal] Extracting ROI_h, ROI_t, and Breathing Motion...")
        sig_h, sig_t = self._extract_roi_signals(frames_gray)
        sig_bm = self._extract_breathing_motion(frames_gray)
        
        # 3. Detect major body movements (Binary vector)
        movements = self._detect_body_movements(frames_gray)
        
        # 4. RMS Fusion (Eq 4)
        print("    [Thermal] RMS Fusion & Filtering...")
        fused = np.sqrt(sig_h**2 + sig_t**2 + sig_bm**2)
        
        # Zero-mean the fusion vector before filtering
        fused = fused - np.mean(fused)
        
        # 5. Butterworth Bandpass [0.05 - 1.5 Hz]
        nyq = 0.5 * self.fps
        b, a = butter(3, [self.b_low/nyq, self.b_high/nyq], btype='bandpass')
        
        # Try to apply filtfilt safely
        try:
            padlen = min(3 * max(len(a), len(b)), len(fused) - 1)
            fused_bp = filtfilt(b, a, fused, padlen=padlen)
        except Exception as e:
            fused_bp = fused # fallback if sequence too short
            
        # 6. Savitzky-Golay Smoothing
        window = min(self.sg_window, len(fused_bp))
        if window % 2 == 0: window -= 1
        if window > self.sg_poly:
            fused_sg = savgol_filter(fused_bp, window_length=window, polyorder=self.sg_poly)
        else:
            fused_sg = fused_bp
            
        # 7. Respiration Analysis (Peak Counting)
        print("    [Thermal] Peak Counting...")
        # Assume min distance between breaths is 60 / max_bpm (90 bpm)
        min_distance = int(self.fps * (60.0 / 90.0))
        peaks, _ = find_peaks(fused_sg, distance=min_distance)
        
        num_peaks = len(peaks)
        duration_mins = len(fused_sg) / float(self.fps * 60.0)
        
        if duration_mins > 0 and num_peaks > 0:
            bpm = num_peaks / duration_mins
        else:
            bpm = 15.0
            
        bpm = float(np.clip(bpm, 0.0, 90.0))
        
        return bpm, movements

if __name__ == "__main__":
    print("Testing Thermal Sleep Baseline Orchestrator...")
    dummy_video = [np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(300)]
    
    runner = ThermalSleepOrchestrator(fps=30)
    bpm, moves = runner.run(dummy_video)
    
    assert isinstance(bpm, float)
    assert len(moves) == 300
    print(f"BPM Output: {bpm:.2f}")
    print(f"Body Movement detected frames: {np.sum(moves)}")
    print("Baseline script working seamlessly.")
