import numpy as np
from vmdpy import VMD
from scipy.signal import butter, filtfilt

class SignalProcessor:
    """Advanced signal processing for respiratory waveforms."""
    
    @staticmethod
    def vmd_denoise(signal: np.ndarray, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7):
        """
        Decomposes signal using Variational Mode Decomposition (VMD).
        Isolates the respiratory mode (usually the one with the most periodic energy).
        """
        if len(signal) < 10:
            return signal
            
        # VMD decomposition
        u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
        
        # Identify the most "respiratory-like" mode
        # We look for the mode with the highest energy in the respiratory band (0.15 - 1.5 Hz)
        best_mode_idx = 0
        max_res_energy = -1
        
        # Estimate FPS if not known (default to 10.0 for clinical videos)
        fps = 10.0 
        
        for i in range(K):
            mode = u[i]
            n = len(mode)
            if n < 10:
                continue
                
            freqs = np.fft.rfftfreq(n, d=1/fps)
            fft_vals = np.abs(np.fft.rfft(mode))
            res_mask = (freqs >= 0.15) & (freqs <= 1.5)
            
            if np.any(res_mask):
                energy = np.sum(fft_vals[res_mask])
                if energy > max_res_energy:
                    max_res_energy = energy
                    best_mode_idx = i
                
        return u[best_mode_idx]

    @staticmethod
    def filter_signal(signal: np.ndarray, fps: float = 30.0, lowcut=0.15, highcut=1.5):
        """Applies a Butterworth bandpass filter to remove DC drift and high-freq noise."""
        if len(signal) < 15:
            return signal
            
        nyq = 0.5 * fps
        low = lowcut / nyq
        high = highcut / nyq
        
        if low <= 0 or high >= 1.0 or low >= high:
            return signal
            
        b, a = butter(2, [low, high], btype='band')
        
        # Pad signal to prevent edge artifacts (pad length must be less than signal length)
        padlen = min(len(signal) - 1, int(fps * 3)) 
        try:
            return filtfilt(b, a, signal, padlen=padlen)
        except Exception:
            return signal

    @staticmethod
    def detrend_signal(signal: np.ndarray, window_size=30):
        """Removes low-frequency baseline drift."""
        moving_avg = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        return signal - moving_avg

    @staticmethod
    def calculate_confidence(signal: np.ndarray, fps=30.0):
        """
        Calculates a confidence score (0-100) based on Signal-to-Noise Ratio (SNR).
        Uses FFT to compare the power of the dominant periodic peak vs. the noise floor.
        """
        n = len(signal)
        if n < 30: # Need at least 1 second of data
            return 0.0
            
        # Apply strict bandpass filter to stop spectral leakage from baseline drift
        clean_sig = SignalProcessor.filter_signal(signal, fps)
            
        # Compute FFT
        freqs = np.fft.rfftfreq(n, d=1/fps)
        fft_values = np.abs(np.fft.rfft(clean_sig))
        
        # Focus on respiratory range: 0.15 Hz to 1.5 Hz (9 to 90 BPM)
        res_mask = (freqs >= 0.15) & (freqs <= 1.5)
        if not np.any(res_mask):
            return 0.0
            
        # Find dominant peak in respiratory range
        res_indices = np.where(res_mask)[0]
        peak_idx = res_indices[np.argmax(fft_values[res_mask])]
        peak_power = fft_values[peak_idx]
        
        # Calculate background noise power (mean of absolute FFT outside the peak)
        # We consider a small neighborhood around the peak as part of the "signal"
        neighborhood = 2
        start = max(0, peak_idx - neighborhood)
        end = min(len(fft_values), peak_idx + neighborhood + 1)
        
        noise_mask = np.ones(len(fft_values), dtype=bool)
        noise_mask[start:end] = False
        noise_mask[0] = False # Ignore DC
        
        noise_power = np.mean(fft_values[noise_mask]) if np.any(noise_mask) else 1e-6
        
        # SNR calculation
        snr = peak_power / (noise_power + 1e-6)
        
        # Map SNR to 0-100 score
        # Relaxed threshold: Allow all signals to contribute
        confidence = min(100.0, snr * 10)
        
        return float(confidence)

    @staticmethod
    def get_bpm(signal: np.ndarray, fps=30.0):
        """Calculates BPM from signal using FFT."""
        n = len(signal)
        if n < 30: return 0.0
        
        # Detrend and Bandpass to absolutely prevent 1/f leakage hitting 0.15Hz boundary
        clean_sig = SignalProcessor.filter_signal(signal, fps)
        
        freqs = np.fft.rfftfreq(n, d=1/fps)
        fft_values = np.abs(np.fft.rfft(clean_sig))
        
        # Focus on respiratory range (9 - 90 BPM)
        res_mask = (freqs >= 0.15) & (freqs <= 1.5)
        if not np.any(res_mask): return 0.0
        
        res_indices = np.where(res_mask)[0]
        peak_idx = res_indices[np.argmax(fft_values[res_mask])]
        bpm = freqs[peak_idx] * 60
        return float(bpm)
