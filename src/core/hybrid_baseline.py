import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.signal import savgol_filter, butter, filtfilt, welch, find_peaks
import scipy.stats

# ==============================================================================
# 1. Optical Flow & 6-ROI Extractor
# ==============================================================================

class MultiRegionExtractor:
    """
    Implements the 6-ROI Face/Chest detection and Farneback optical flow 
    motion extraction as defined in the PLoS One (2026) paper.
    ROIs: Forehead, Left Cheek, Right Cheek, Upper Chest, Left Shoulder, Right Shoulder.
    """
    def __init__(self, fps=30):
        self.fps = fps
        # Using Haar cascades for robust baseline face detection as per paper
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def _apply_clahe(self, frame):
        """Contrast Limited Adaptive Histogram Equalization"""
        # Convert to LAB space to equalize only luminance
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
    def extract_motion_signals(self, video_frames):
        """
        Extracts 6 channels of robust optical flow motion magnitudes.
        video_frames: List or numpy array of RGB frames [T, H, W, 3]
        Returns: np.ndarray [6, T] of raw motion signals.
        """
        T = len(video_frames)
        if T < 2:
            return np.zeros((6, T))
            
        signals = np.zeros((6, T-1))
        
        # 1. Initial face detection on first few frames to establish bounding box
        face_box = None
        for i in range(min(10, T)):
            gray = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                face_box = faces[0] # [x, y, w, h]
                break
                
        if face_box is None:
            # Fallback if no face detected: use generic center regions
            H, W = video_frames[0].shape[:2]
            face_box = [W//3, H//4, W//3, H//3]
            
        x, y, fw, fh = face_box
        
        # 2. Define the 6 anatomical ROIs as dynamic percentage offsets from the face box
        # [x_off_pct, y_off_pct, w_pct, h_pct]
        roi_defs = {
            'forehead': (0.2, -0.2, 0.6, 0.2), # Above face
            'l_cheek':  (-0.1, 0.4, 0.3, 0.3),  # Left of face center
            'r_cheek':  (0.8, 0.4, 0.3, 0.3),   # Right of face center
            'chest':    (-0.2, 1.2, 1.4, 0.8),  # Wide area below face
            'l_shoulder': (-0.8, 1.0, 0.5, 0.6),# Far left, below face
            'r_shoulder': (1.3, 1.0, 0.5, 0.6)  # Far right, below face
        }
        
        H, W = video_frames[0].shape[:2]
        rois = []
        for name, (ax, ay, aw, ah) in roi_defs.items():
            rx = int(np.clip(x + ax * fw, 0, W-1))
            ry = int(np.clip(y + ay * fh, 0, H-1))
            rw = int(np.clip(aw * fw, 1, W-rx))
            rh = int(np.clip(ah * fh, 1, H-ry))
            rois.append((rx, ry, rw, rh))
            
        # 3. Dense Optical Flow Calculation
        prev_gray = cv2.cvtColor(self._apply_clahe(video_frames[0]), cv2.COLOR_BGR2GRAY)
        
        for t in range(1, T):
            curr_gray = cv2.cvtColor(self._apply_clahe(video_frames[t]), cv2.COLOR_BGR2GRAY)
            
            # Farneback parameterized as per standard robust motion extraction
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate Euclidean magnitude
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Extract mean motion for each ROI
            for idx, (rx, ry, rw, rh) in enumerate(rois):
                roi_mag = mag[ry:ry+rh, rx:rx+rw]
                signals[idx, t-1] = np.mean(roi_mag)
                
            prev_gray = curr_gray
            
        # Pad first frame to maintain length T
        signals = np.pad(signals, ((0, 0), (1, 0)), mode='edge')
        return signals

# ==============================================================================
# 2. Advanced Signal Pre-processing & Filtering
# ==============================================================================

class SignalFilterPipeline:
    def __init__(self, fps=30):
        self.fps = fps
        
    def process(self, raw_signal):
        """
        Applies Savitzky-Golay, Butterworth bandpass, IQR outlier removal, 
        and Sigmoid compression.
        """
        if len(raw_signal) < 15: # Too short
            return raw_signal
            
        # 1. Temporally smooth (Savitzky-Golay window 5, poly 2)
        s1 = savgol_filter(raw_signal, window_length=5, polyorder=2)
        
        # 2. DC removal
        s2 = s1 - np.mean(s1)
        
        # 3. Butterworth Bandpass [0.1 - 0.8 Hz]
        # (Assuming 9-48 BPM clinically valid range)
        nyq = 0.5 * self.fps
        b, a = butter(2, [0.1/nyq, 0.8/nyq], btype='bandpass')
        s3 = filtfilt(b, a, s2)
        
        # 4. Outlier removal (IQR method)
        q1, q3 = np.percentile(s3, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        s4 = np.clip(s3, lower_bound, upper_bound)
        
        # 5. Normalization + Sigmoid bounded intensity mapping (Eq 7 & 8 from paper)
        std_val = np.std(s4) + 1e-8
        N_norm = (s4 - np.mean(s4)) / std_val
        motion_score = (2.0 / (1.0 + np.exp(-N_norm))) - 1.0 # Centered around 0
        
        return motion_score

    def compute_snr(self, filtered_signal):
        """Approximates the Signal-to-Noise Ratio (Eq 18)"""
        # A simple approximation: power of dominant frequency band / power of the rest
        try:
            f, pxx = welch(filtered_signal, fs=self.fps, nperseg=min(len(filtered_signal), 256))
            resp_band = (f >= 0.15) & (f <= 0.5)
            noise_band = ~resp_band
            signal_power = np.sum(pxx[resp_band]) + 1e-8
            noise_power = np.sum(pxx[noise_band]) + 1e-8
            snr = 10 * np.log10(signal_power / noise_power)
            return snr
        except Exception:
            return 0.0

# ==============================================================================
# 3. Exact 1D CNN Architecture (Table 3 implementation)
# ==============================================================================

class HybridCNN(nn.Module):
    """
    Proposed 1D CNN network architecture for RR estimation.
    Takes all 6 multi-region sequences directly.
    """
    def __init__(self, seq_len=300):
        super().__init__()
        
        # Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate flattened dimension
        ds_factor = 8 # Three MaxPool(2) layers = 2^3
        flat_dim = 256 * (seq_len // ds_factor)
        
        # Fully Connected Block
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Outputs a single continuous RR prediction (BPM)
        )
        
    def forward(self, x):
        # x shape: [Batch, 6, seq_len]
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        out = self.fc(c3)
        return out.squeeze(-1) # -> [Batch]

# ==============================================================================
# 4. Spectral Estimation & Adaptive Kalman Fusion
# ==============================================================================

class SpectralEstimator:
    def __init__(self, fps=30):
        self.fps = fps
        
    def estimate_bpm(self, signal):
        """Welch's Method for robust spectral peak finding"""
        if len(signal) < self.fps * 2: # Need at least 2 seconds
            return 15.0
            
        # 4-second window (or max available) with 75% overlap
        nperseg = min(len(signal), int(self.fps * 4.0)) 
        noverlap = int(nperseg * 0.75)
        
        f, Pxx = welch(signal, fs=self.fps, nperseg=nperseg, noverlap=noverlap, window='hann')
        
        # Restrict to physiological limits [0.15 - 0.5 Hz] -> [9 - 30 BPM]
        valid_idxs = np.where((f >= 0.15) & (f <= 0.5))[0]
        if len(valid_idxs) == 0:
            return 15.0
            
        valid_f = f[valid_idxs]
        valid_Pxx = Pxx[valid_idxs]
        
        peak_idx = np.argmax(valid_Pxx)
        dom_freq = valid_f[peak_idx]
        
        bpm = dom_freq * 60.0
        return np.clip(bpm, 9.0, 30.0)

class AdaptiveKalmanFilter:
    """Bounded 1D Kalman Filter to smooth predictions"""
    def __init__(self, initial_bpm=15.0):
        # State: [Position (BPM), Velocity (dBPM/dt)]
        self.x = np.array([[initial_bpm], [0.0]])
        self.P = np.array([[10.0, 0.0], [0.0, 10.0]]) # Initial uncertainty
        
        # Transition matrix (Assuming constant velocity model, dt=1)
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        
        # Measurement matrix (We only measure position/BPM)
        self.H = np.array([[1.0, 0.0]])
        
        # Process and Measurement Noise
        self.R = np.array([[1.5]]) # Measurement noise
        self.Q = np.array([[0.1, 0.0], [0.0, 0.01]]) # Process noise
        
    def update(self, measurement_bpm):
        # 1. Predict
        x_pred = np.dot(self.F, self.x)
        P_pred = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        # Rate-limiting constraint (Max change of 3 BPM between updates)
        if abs(measurement_bpm - x_pred[0, 0]) > 3.0:
            measurement_bpm = x_pred[0, 0] + np.sign(measurement_bpm - x_pred[0, 0]) * 3.0
            
        measurement_bpm = np.clip(measurement_bpm, 9.0, 30.0)
            
        # 2. Update
        y = np.array([[measurement_bpm]]) - np.dot(self.H, x_pred)
        S = np.dot(np.dot(self.H, P_pred), self.H.T) + self.R
        K = np.dot(np.dot(P_pred, self.H.T), np.linalg.inv(S))
        
        self.x = x_pred + np.dot(K, y)
        self.P = P_pred - np.dot(np.dot(K, self.H), P_pred)
        
        return float(self.x[0, 0])

class HybridFusionOrchestrator:
    """
    Executes the full pipeline:
    Extracts multi-region signals -> Filters them -> Gets Spectral + CNN BPMs
    -> Fuses via SNR weighting -> Smooths via Kalman.
    """
    def __init__(self, cnn_model_path=None, fps=30, device='cpu'):
        self.fps = fps
        self.device = device
        self.extractor = MultiRegionExtractor(fps=fps)
        self.filter_pipe = SignalFilterPipeline(fps=fps)
        self.spectral_est = SpectralEstimator(fps=fps)
        self.kalman = AdaptiveKalmanFilter()
        
        # Default sequence length assuming 10 second videos (300 frames)
        self.cnn = HybridCNN(seq_len=300).to(device)
        self.cnn.eval()
        
        if cnn_model_path and os.path.exists(cnn_model_path):
            try:
                self.cnn.load_state_dict(torch.load(cnn_model_path, map_location=device))
                print(f"[Hybrid CNN] Loaded weights from {cnn_model_path}")
            except Exception as e:
                print(f"[Hybrid CNN] Weight mismatch ({e}). Ignored, using initialization.")
        else:
            print("[Hybrid CNN] No weights provided. Using random init (Zero-shot / Untrained state).")
            
    def run(self, video_frames):
        """
        Main execution block taking a batch of RGB frames.
        Returns final stabilized BPM.
        """
        # 1. Extraction [6, T]
        raw_signals = self.extractor.extract_motion_signals(video_frames)
        
        # 2. Filtering & SNR calculation
        filtered_signals = np.zeros_like(raw_signals)
        snr_values = np.zeros(6)
        
        for i in range(6):
            filtered = self.filter_pipe.process(raw_signals[i])
            filtered_signals[i] = filtered
            snr_values[i] = self.filter_pipe.compute_snr(filtered)
            
        # Normalize SNR values to [0, 1] range
        min_snr, max_snr = np.min(snr_values), np.max(snr_values)
        if max_snr > min_snr:
            norm_snr = (snr_values - min_snr) / (max_snr - min_snr)
        else:
            norm_snr = np.ones(6) * 0.5
            
        # 3. Spectral Branch Computation (Weighted Average)
        spectral_bpms = []
        for i in range(6):
            spectral_bpms.append(self.spectral_est.estimate_bpm(filtered_signals[i]))
            
        # Weight spectral predictions by SNR
        r_spectral = np.average(spectral_bpms, weights=norm_snr)
        
        # 4. CNN Branch Computation
        # Truncate or Pad to fit 300 exactly for the fixed linear layer mapping
        target_len = 300
        H = filtered_signals.shape[1]
        
        if H > target_len:
            # Take last 300 frames
            model_input = filtered_signals[:, -target_len:]
        elif H < target_len:
            # Pad to 300 frames
            model_input = np.pad(filtered_signals, ((0, 0), (target_len - H, 0)), mode='edge')
        else:
            model_input = filtered_signals
            
        tensor_in = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(self.device) # [1, 6, 300]
        
        with torch.no_grad():
            r_cnn = self.cnn(tensor_in).item()
            
        # 5. SNR-weighted Fusion
        mean_norm_snr = np.mean(norm_snr)
        # Weight bounded strictly to [0.3, 0.7] as per paper
        w = np.clip(mean_norm_snr, 0.3, 0.7) 
        
        r_fused = (w * r_cnn) + ((1.0 - w) * r_spectral)
        
        # 6. Kalman Smoothing
        final_bpm = self.kalman.update(r_fused)
        
        return final_bpm

if __name__ == "__main__":
    print("Testing Hybrid CNN-Spectral Baseline Architecture Initialization...")
    
    # Simulate a dummy 10-second video at 30fps (300 frames) size 64x64
    dummy_video = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(300)]
    
    orchestrator = HybridFusionOrchestrator()
    predicted_bpm = orchestrator.run(dummy_video)
    
    print("\n[Hybrid Pipeline Modules Test]")
    print(f"Extracted 6-ROI Motion: Yes")
    print(f"Applied Filters (CLAHE, Sigmoid, Butter): Yes")
    print(f"CNN Branch Forward Pass: Yes")
    print(f"Spectral Branch Computation: Yes")
    print(f"SNR-Kalman Fusion Output: {predicted_bpm:.2f} BPM")
    print("\n✓ Baseline Architecture fully instantiated successfully.")
