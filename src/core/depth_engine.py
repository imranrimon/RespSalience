import torch
import numpy as np
import cv2
import gc
from transformers import pipeline
from PIL import Image
from src.core.signal_processor import SignalProcessor

class DepthEngine:
    """Monocular depth estimation engine using DepthAnythingV2."""
    
    def __init__(
        self,
        model_id="depth-anything/Depth-Anything-V2-Small-hf",
        device=None,
        prefer_local_files=True,
        max_roi_side=384,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_id = model_id
        self.prefer_local_files = prefer_local_files
        self.max_roi_side = int(max_roi_side)
        self.pipe = None
        self.fallback_mode = False

        print(f"Initializing DepthEngine with {model_id} on {self.device}...")
        self._init_pipeline()

    def _init_pipeline(self):
        """
        Prefer the locally cached Hugging Face files first.
        This avoids network HEAD requests during offline benchmarking.
        """
        attempts = []
        if self.prefer_local_files:
            attempts.append(("local-cache", True))
        attempts.append(("hub", False))

        last_error = None
        for mode_name, local_only in attempts:
            try:
                self.pipe = pipeline(
                    task="depth-estimation",
                    model=self.model_id,
                    device=self.device,
                    local_files_only=local_only,
                )
                if local_only:
                    print("DepthEngine: loaded cached depth model in offline mode.")
                else:
                    print("DepthEngine: loaded depth model from Hugging Face hub.")
                self.fallback_mode = False
                return
            except Exception as exc:
                last_error = exc
                print(f"DepthEngine warning: {mode_name} init failed: {exc}")

        self.pipe = None
        self.fallback_mode = True
        print(
            "DepthEngine warning: depth model unavailable. "
            "Falling back to grayscale pseudo-depth features."
        )
        if last_error is not None:
            print(f"DepthEngine fallback reason: {last_error}")

    @staticmethod
    def _pseudo_depth_map(frame: np.ndarray) -> np.ndarray:
        """
        Offline-safe fallback when the depth model is unavailable.
        Uses luminance plus local contrast to create a pseudo-depth proxy.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
        contrast = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        pseudo = blur - 0.25 * contrast
        pseudo -= np.min(pseudo)
        denom = np.max(pseudo) - np.min(pseudo)
        if denom > 1e-6:
            pseudo = pseudo / denom
        return pseudo

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth for a single RGB frame (H, W, 3)."""
        if self.pipe is None:
            return self._pseudo_depth_map(frame)

        # Convert to PIL Image
        pil_img = Image.fromarray(frame)
        result = self.pipe(pil_img)
        depth_map = np.array(result["depth"])
        return depth_map

    @staticmethod
    def _is_memory_pressure_error(exc: BaseException) -> bool:
        text = str(exc).lower()
        return (
            isinstance(exc, MemoryError)
            or "out of memory" in text
            or "unable to allocate" in text
            or "cannot allocate" in text
            or "cuda out of memory" in text
        )

    def _resize_for_depth_inference(self, roi: np.ndarray) -> Image.Image:
        """Downscale large ROIs to keep depth pipeline preprocessing memory-bounded."""
        h, w = roi.shape[:2]
        max_side = max(h, w)
        if self.max_roi_side > 0 and max_side > self.max_roi_side:
            scale = float(self.max_roi_side) / float(max_side)
            new_w = max(16, int(round(w * scale)))
            new_h = max(16, int(round(h * scale)))
            roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return Image.fromarray(roi)

    def extract_signal_from_roi(self, depth_frames: np.ndarray, roi_box: list) -> np.ndarray:
        """
        Extract a 1D signal from a sequence of pre-computed depth maps.
        roi_box: [x, y, w, h]
        """
        x, y, w, h = map(int, roi_box)
        signals = []
        for d_map in depth_frames:
            roi = d_map[y:y+h, x:x+w]
            signals.append(np.mean(roi))
            
        signals = np.array(signals)
        
        # Apply VMD enhancement (Phase 11)
        if len(signals) > 20: # Ensure enough data for VMD
            print("  Applying VMD denoising...")
            signals = SignalProcessor.vmd_denoise(signals, K=2) # K=2 to split respiration vs noise
            
        signals = (signals - np.min(signals)) / (np.max(signals) - np.min(signals) + 1e-6)
        return signals

    def extract_signal_from_frames(self, frames_np, roi_box):
        """
        Extracts depth signal from a sequence of frames using batch processing.
        """
        x1, y1, x2, y2 = map(int, roi_box)
        
        # Prepare valid ROI images
        rois = []
        valid_indices = []
        H, W = frames_np[0].shape[:2]
        
        # Ensure ROI is within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(len(frames_np))

        for i, frame in enumerate(frames_np):
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                rois.append(self._resize_for_depth_inference(roi))
                valid_indices.append(i)
        
        if not rois:
            return np.zeros(len(frames_np))

        print(f"Depth analysis: Processing {len(rois)} frames (chunked) avoiding OOM...")

        if self.pipe is None:
            signal = np.zeros(len(frames_np), dtype=np.float32)
            for idx, frame in zip(valid_indices, [frames_np[i] for i in valid_indices]):
                roi = frame[y1:y2, x1:x2]
                pseudo_map = self._pseudo_depth_map(roi)
                signal[idx] = float(np.mean(pseudo_map))
            signal = signal - np.mean(signal)
            return signal
        
        # Process in chunks with adaptive batch size to survive tight RAM/VRAM conditions.
        chunk_size = 64
        batch_size = 16
        signal_values = []

        i = 0
        while i < len(rois):
            chunk = rois[i:i + chunk_size]
            try:
                chunk_results = self.pipe(chunk, batch_size=batch_size)
            except Exception as exc:
                if not self._is_memory_pressure_error(exc):
                    raise

                if chunk_size > 1:
                    chunk_size = max(1, chunk_size // 2)
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)

                print(
                    f"Depth analysis warning: memory pressure at frame {i}/{len(rois)}; "
                    f"retrying with chunk_size={chunk_size}, batch_size={batch_size}"
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue

            for res in chunk_results:
                d_img = res["depth"]
                signal_values.append(np.mean(np.array(d_img)))

            del chunk_results
            gc.collect()
            i += len(chunk)
            
        signal = np.zeros(len(frames_np))
        for idx, val in zip(valid_indices, signal_values):
            signal[idx] = val
            
        # Apply VMD enhancement (Phase 11)
        if len(signal) > 20:
             # Prevent noisy VMD on flat signals
             std = np.std(signal)
             if std > 1e-6:
                # print("  Applying VMD denoising...") # Optional log
                try:
                    signal = SignalProcessor.vmd_denoise(signal, K=2)
                except:
                    pass
            
        # Simple detrending / scaling
        signal = signal - np.mean(signal)
        return signal
