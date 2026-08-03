"""
Universal Modality-Agnostic Input Adapter
==========================================
Makes the respiratory rate estimation system work with ANY input type:
  - Video file (.mp4, .avi, .mov)
  - Image sequence (folder of .jpg/.png, or list of paths)
  - Single image (falls back to static depth analysis)
  - Audio file (.wav, .mp3, .flac)
  - Numpy signal file (.npz with pre-extracted signals)
  - Webcam / RTSP stream (url or int index)
  - Mixed: any combination of the above

The adapter auto-detects what's available and produces a canonical
ModalityBundle that the rest of the pipeline consumes.

Usage:
    from src.core.universal_adapter import UniversalAdapter
    
    adapter = UniversalAdapter()
    bundle  = adapter.load("path/to/anything")
    result  = adapter.estimate_rr(bundle)
    print(f"RR = {result['rr_bpm']:.1f} BPM  (modalities: {result['modalities_used']})")
"""
import os
import sys
import glob
import warnings
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Canonical modality bundle — all signals are 1-D float32 numpy arrays of
# arbitrary length. None means modality was not available/derivable.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModalityBundle:
    depth_signal:  Optional[np.ndarray] = None   # (T,)  chest expansion from depth
    flow_signal:   Optional[np.ndarray] = None   # (T,)  motion magnitude from optical flow
    audio_signal:  Optional[np.ndarray] = None   # (T,)  respiratory band audio
    frames:        Optional[np.ndarray] = None   # (N,H,W,3) raw RGB frames if available
    fps:           float = 30.0
    audio_sr:      int   = 22050
    source_path:   str   = ""
    source_type:   str   = ""   # 'video' | 'images' | 'audio' | 'npz' | 'stream'
    available:     List[str] = field(default_factory=list)  # which modalities are filled
    metadata:      Dict[str, Any] = field(default_factory=dict)


class ModalityDetector:
    """Detects input type and available modalities from a path or URL."""
    
    VIDEO_EXTS  = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv'}
    IMAGE_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    AUDIO_EXTS  = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    SIGNAL_EXTS = {'.npz', '.npy'}

    @staticmethod
    def detect(path: str) -> str:
        """Returns 'video' | 'images' | 'audio' | 'npz' | 'stream' | 'unknown'"""
        p = str(path)
        
        # Webcam or RTSP
        if isinstance(path, int) or p.startswith('rtsp://') or p.startswith('http'):
            return 'stream'
        
        path_obj = Path(p)
        ext = path_obj.suffix.lower()
        
        if ext in ModalityDetector.VIDEO_EXTS:
            return 'video'
        if ext in ModalityDetector.AUDIO_EXTS:
            return 'audio'
        if ext in ModalityDetector.SIGNAL_EXTS:
            return 'npz'
        if ext in ModalityDetector.IMAGE_EXTS:
            return 'images'  # single image
        if path_obj.is_dir():
            # Check for image sequence in directory
            imgs = []
            for e in ModalityDetector.IMAGE_EXTS:
                imgs.extend(glob.glob(str(path_obj / f'*{e}')))
            if imgs:
                return 'images'
            vids = []
            for e in ModalityDetector.VIDEO_EXTS:
                vids.extend(glob.glob(str(path_obj / f'*{e}')))
            if vids:
                return 'video'  # directory of videos → process first one
        
        return 'unknown'


class UniversalAdapter:
    """
    Converts any input to a ModalityBundle, then estimates RR.
    
    Design principle: degrade gracefully.
    Full quad-modal > tri-modal > bi-modal > uni-modal > fallback spectral.
    """

    def __init__(self, config_path: str = "configs/config.yaml",
                 checkpoint_path: str = "data/models/vlm_gated_fused.pth",
                 fps: float = 30.0):
        self.fps           = fps
        self.config_path   = config_path
        self.checkpoint_path = checkpoint_path
        self._depth_engine  = None   # lazy-loaded
        self._signal_proc   = None
        self._controller    = None

    # ── Lazy loaders ──────────────────────────────────────────────────────────
    def _get_depth_engine(self):
        if self._depth_engine is None:
            from src.core.depth_engine import DepthEngine
            self._depth_engine = DepthEngine()
        return self._depth_engine

    def _get_signal_processor(self):
        if self._signal_proc is None:
            from src.core.signal_processor import SignalProcessor
            self._signal_proc = SignalProcessor()
        return self._signal_proc

    def _get_controller(self):
        if self._controller is None:
            from src.core.main_controller import RespiratoryController
            if not os.path.exists(self.config_path):
                warnings.warn(f"Config not found: {self.config_path}. Using signal-only estimation.")
                return None
            self._controller = RespiratoryController(
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                enable_depth=True,
                enable_flow=True,
                enable_audio=True,
                enable_transformer=os.path.exists(self.checkpoint_path)
            )
        return self._controller

    # ── Public API ─────────────────────────────────────────────────────────────
    def load(self, source, extra_audio: str = None, fps: float = None) -> ModalityBundle:
        """
        Accept any source and return a filled ModalityBundle.
        
        Args:
            source: path to video/image/audio/npz/folder, or webcam int, or RTSP URL
            extra_audio: separate audio file to pair with a video-without-audio
            fps: override frame rate (for image sequences without metadata)
        """
        source_type = ModalityDetector.detect(source)
        fps = fps or self.fps
        
        loaders = {
            'video' : self._load_video,
            'images': self._load_images,
            'audio' : self._load_audio,
            'npz'   : self._load_npz,
            'stream': self._load_stream,
        }
        
        loader = loaders.get(source_type)
        if loader is None:
            raise ValueError(f"Cannot determine input type for: '{source}'. "
                             f"Supported: video, image folder, audio, .npz signal file, webcam int, RTSP URL.")
        
        bundle = loader(source, fps)
        bundle.source_path = str(source)
        bundle.source_type = source_type
        
        # Optionally inject a separate audio file
        if extra_audio and bundle.audio_signal is None:
            audio_bundle = self._load_audio(extra_audio, fps)
            bundle.audio_signal = audio_bundle.audio_signal
            bundle.audio_sr = audio_bundle.audio_sr
            if 'audio' not in bundle.available:
                bundle.available.append('audio')
        
        bundle.available = [m for m in ['depth', 'flow', 'audio']
                            if getattr(bundle, f'{m}_signal') is not None]
        
        print(f"[UniversalAdapter] Loaded '{source_type}' -> modalities: {bundle.available}")
        return bundle

    def estimate_rr(self, bundle: ModalityBundle) -> Dict[str, Any]:
        """
        Route to the best available estimator and return RR.
        Returns dict with: rr_bpm, confidence, modalities_used, method
        """
        avail = bundle.available

        # ── Route: best → worst ───────────────────────────────────────────────
        if len(avail) >= 2 and ('depth' in avail or 'flow' in avail):
            return self._estimate_multimodal(bundle)
        elif 'audio' in avail:
            return self._estimate_audio_only(bundle)
        elif 'depth' in avail:
            return self._estimate_signal_only(bundle.depth_signal, bundle.fps, 'depth')
        elif 'flow' in avail:
            return self._estimate_signal_only(bundle.flow_signal, bundle.fps, 'flow')
        else:
            raise RuntimeError("No valid modality in ModalityBundle. Cannot estimate RR.")

    # ── Input Loaders ──────────────────────────────────────────────────────────
    def _load_video(self, path: str, fps: float) -> ModalityBundle:
        """Extract depth + flow + audio from a video file."""
        import cv2
        cap = cv2.VideoCapture(str(path))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            raise ValueError(f"Could not read any frames from: {path}")
        
        print(f"  [video] {len(frames)} frames @ {actual_fps:.1f} fps")
        frames_arr = np.array(frames)  # (N, H, W, 3)

        # Extract depth signal (mean chest depth per frame)
        depth_signal = self._frames_to_depth_signal(frames)
        
        # Extract optical flow signal (mean motion magnitude)
        flow_signal  = self._frames_to_flow_signal(frames)
        
        # Extract audio
        audio_signal, audio_sr = self._extract_audio_from_video(str(path))
        
        return ModalityBundle(
            depth_signal=depth_signal,
            flow_signal=flow_signal,
            audio_signal=audio_signal,
            frames=frames_arr,
            fps=actual_fps,
            audio_sr=audio_sr or 22050,
        )

    def _load_images(self, path: str, fps: float) -> ModalityBundle:
        """Load a folder of images OR a single image as a sequence."""
        import cv2
        p = Path(path)
        
        if p.is_file():
            img_paths = [str(p)]
        else:
            img_paths = sorted(
                [str(f) for ext in ModalityDetector.IMAGE_EXTS
                 for f in p.glob(f'*{ext}')]
            )
        
        if not img_paths:
            raise ValueError(f"No images found in: {path}")
        
        frames = []
        for ip in img_paths:
            img = cv2.imread(ip)
            if img is not None:
                frames.append(img)
        
        if not frames:
            raise ValueError(f"Could not read images from: {path}")
        
        print(f"  [images] {len(frames)} images, treating as {fps:.1f} fps sequence")
        
        if len(frames) == 1:
            # Single image: only depth, no temporal signal possible → static estimate
            depth_signal = self._frames_to_depth_signal(frames)
            return ModalityBundle(
                depth_signal=depth_signal,
                frames=np.array(frames),
                fps=fps,
                metadata={'single_image': True,
                          'note': 'Single image: static depth only, RR is an estimate'}
            )
        else:
            depth_signal = self._frames_to_depth_signal(frames)
            flow_signal  = self._frames_to_flow_signal(frames)
            return ModalityBundle(
                depth_signal=depth_signal,
                flow_signal=flow_signal,
                frames=np.array(frames),
                fps=fps,
            )

    def _load_audio(self, path: str, fps: float) -> ModalityBundle:
        """
        Load an audio file and bandpass-filter to respiratory band (0.1–0.8 Hz).
        Works with WAV (always) and MP3 (requires librosa/pydub).
        Also handles ICBHI-style .txt annotation to derive GT RR.
        """
        audio_signal, sr = self._read_audio(str(path))
        resp_signal = self._audio_to_resp_signal(audio_signal, sr)
        
        # Check for ICBHI-style annotation file
        gt_rr = None
        txt_path = Path(path).with_suffix('.txt')
        if txt_path.exists():
            try:
                gt_rr = self._parse_icbhi_annotation(str(txt_path))
            except Exception:
                pass
        
        return ModalityBundle(
            audio_signal=resp_signal,
            fps=fps,
            audio_sr=sr,
            metadata={'gt_rr_bpm': gt_rr, 'raw_audio_sr': sr}
        )

    def _load_npz(self, path: str, fps: float) -> ModalityBundle:
        """Load a pre-extracted .npz signal file (our own format or ICBHI-derived)."""
        data = np.load(str(path), allow_pickle=True)
        
        # Support our format: keys = depth, flow, audio, gt_rr
        def load_key(*keys):
            for k in keys:
                if k in data:
                    arr = data[k]
                    return arr.astype(np.float32) if arr is not None else None
            return None
        
        depth_signal = load_key('depth', 'depth_signal')
        flow_signal  = load_key('flow',  'flow_signal')
        audio_signal = load_key('audio', 'audio_signal')
        gt_rr        = float(data['gt_rr']) if 'gt_rr' in data else None
        
        return ModalityBundle(
            depth_signal=depth_signal,
            flow_signal=flow_signal,
            audio_signal=audio_signal,
            fps=fps,
            metadata={'gt_rr_bpm': gt_rr, 'source': 'npz'}
        )

    def _load_stream(self, source, fps: float) -> ModalityBundle:
        """
        Load from webcam (int) or RTSP URL.
        Default: captures 30 seconds, then processes as video.
        """
        import cv2
        cap = cv2.VideoCapture(source)
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        n_frames = int(actual_fps * 30)  # 30 seconds of capture
        
        frames = []
        print(f"  [stream] Capturing {n_frames} frames ({30}s)...")
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames captured from stream: {source}")
        
        depth_signal = self._frames_to_depth_signal(frames)
        flow_signal  = self._frames_to_flow_signal(frames)
        return ModalityBundle(
            depth_signal=depth_signal,
            flow_signal=flow_signal,
            frames=np.array(frames),
            fps=actual_fps,
        )

    # ── Signal Extraction Helpers ──────────────────────────────────────────────
    def _frames_to_depth_signal(self, frames: list) -> Optional[np.ndarray]:
        """Run MiDaS depth estimation frame-by-frame, extract chest ROI mean."""
        try:
            depth_engine = self._get_depth_engine()
            signal = []
            for frame in frames:
                depth_map = depth_engine.estimate_depth(frame)
                h, w = depth_map.shape
                # Chest ROI: center vertical third, top 60% horizontal
                roi = depth_map[h//5 : 3*h//5, w//4 : 3*w//4]
                signal.append(float(np.mean(roi)))
            return np.array(signal, dtype=np.float32)
        except Exception as e:
            warnings.warn(f"Depth extraction failed: {e}. Depth modality unavailable.")
            return None

    def _frames_to_flow_signal(self, frames: list) -> Optional[np.ndarray]:
        """Lucas-Kanade sparse optical flow → mean motion magnitude per frame."""
        try:
            import cv2
            if len(frames) < 2:
                return None
            signal = [0.0]  # first frame has no flow
            prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            for frame in frames[1:]:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                signal.append(float(np.mean(magnitude)))
                prev_gray = curr_gray
            return np.array(signal, dtype=np.float32)
        except Exception as e:
            warnings.warn(f"Optical flow extraction failed: {e}. Flow modality unavailable.")
            return None

    def _extract_audio_from_video(self, video_path: str):
        """Extract audio track from video using moviepy or ffmpeg subprocess."""
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(video_path)
            if clip.audio is None:
                return None, None
            sr = 22050
            audio_arr = clip.audio.to_soundarray(fps=sr)
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1)  # stereo → mono
            clip.close()
            return audio_arr.astype(np.float32), sr
        except Exception:
            return None, None

    def _read_audio(self, path: str):
        """Read any audio file to (samples, sample_rate)."""
        try:
            import librosa
            audio, sr = librosa.load(path, sr=22050, mono=True)
            return audio, sr
        except Exception:
            pass
        try:
            import soundfile as sf
            audio, sr = sf.read(path, always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), sr
        except Exception:
            pass
        # Last resort: scipy for WAV only
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32), sr
        except Exception as e:
            raise RuntimeError(f"Could not read audio from {path}: {e}")

    def _audio_to_resp_signal(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Bandpass filter audio to respiratory frequency band (0.1–0.8 Hz)
        and downsample to 30 Hz to match video signals.
        """
        from scipy.signal import butter, filtfilt, resample
        # Bandpass: 0.1–0.8 Hz (6–48 BPM)
        low, high = 0.1 / (sr / 2), 0.8 / (sr / 2)
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, audio)
        # Downsample to 30 Hz
        target_len = int(len(filtered) * 30 / sr)
        resampled = resample(filtered, max(target_len, 1))
        return resampled.astype(np.float32)

    @staticmethod
    def _parse_icbhi_annotation(txt_path: str) -> float:
        """
        Parse ICBHI annotation file (start end crackle wheeze per line)
        and derive respiratory rate from cycle durations.
        """
        cycles = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        start, end = float(parts[0]), float(parts[1])
                        cycles.append(end - start)
                    except ValueError:
                        continue
        if not cycles:
            return None
        mean_cycle = np.mean(cycles)
        rr_bpm = 60.0 / mean_cycle
        return round(float(rr_bpm), 1)

    # ── Estimators ─────────────────────────────────────────────────────────────
    def _estimate_multimodal(self, bundle: ModalityBundle) -> Dict:
        """Use full pipeline via RespiratoryController."""
        try:
            ctrl = self._get_controller()
            if ctrl is not None and bundle.frames is not None:
                # Full pipeline via video frames
                # Write temp video and process
                result = self._run_controller_on_frames(ctrl, bundle)
                if result is not None:
                    return result
        except Exception as e:
            warnings.warn(f"Full controller failed: {e}. Falling back to signal-only.")
        
        # Fallback: SNR-weighted fusion of available 1D signals
        return self._estimate_snr_fusion(bundle)

    def _estimate_snr_fusion(self, bundle: ModalityBundle) -> Dict:
        """SNR-weighted fusion of available 1D signals."""
        from scipy.signal import welch
        votes = []
        for name in ['depth', 'flow', 'audio']:
            sig = getattr(bundle, f'{name}_signal')
            if sig is None or len(sig) < 2:
                continue
            rr, conf = self._signal_to_rr(sig, bundle.fps)
            if rr is not None:
                votes.append((rr, conf, name))
        
        if not votes:
            raise RuntimeError("All estimators failed.")
        
        total_conf = sum(c for _, c, _ in votes)
        weighted_rr = sum(rr * c for rr, c, _ in votes) / total_conf
        
        return {
            'rr_bpm': round(float(weighted_rr), 1),
            'confidence': round(float(max(c for _, c, _ in votes)), 3),
            'modalities_used': [n for _, _, n in votes],
            'method': 'snr_weighted_fusion',
            'per_modality': {n: {'rr': rr, 'conf': c} for rr, c, n in votes}
        }

    def _estimate_audio_only(self, bundle: ModalityBundle) -> Dict:
        rr, conf = self._signal_to_rr(bundle.audio_signal, bundle.fps)
        return {
            'rr_bpm': round(float(rr), 1),
            'confidence': round(float(conf), 3),
            'modalities_used': ['audio'],
            'method': 'audio_bandpass_fft'
        }

    def _estimate_signal_only(self, signal: np.ndarray, fps: float, name: str) -> Dict:
        rr, conf = self._signal_to_rr(signal, fps)
        return {
            'rr_bpm': round(float(rr), 1),
            'confidence': round(float(conf), 3),
            'modalities_used': [name],
            'method': f'{name}_fft'
        }

    def _signal_to_rr(self, signal: np.ndarray, fps: float):
        """FFT-based peak frequency in 0.1–0.8 Hz band → BPM."""
        from scipy.signal import welch
        if len(signal) < int(fps * 5):
            warnings.warn("Signal too short (<5s). RR estimate unreliable.")
        
        signal = signal - signal.mean()
        freqs, psd = welch(signal, fs=fps, nperseg=min(len(signal), int(fps * 30)))
        
        # Restrict to respiratory band: 0.1–0.8 Hz (6–48 BPM)
        mask = (freqs >= 0.1) & (freqs <= 0.8)
        if not mask.any():
            return None, 0.0
        
        peak_freq = freqs[mask][np.argmax(psd[mask])]
        rr_bpm = peak_freq * 60.0
        
        # Confidence: peak-to-band SNR
        peak_power = psd[mask].max()
        band_mean  = psd[mask].mean()
        confidence = float(np.clip(peak_power / (band_mean + 1e-9), 0, 20) / 20)
        
        return rr_bpm, confidence

    def _run_controller_on_frames(self, ctrl, bundle: ModalityBundle):
        """Attempt to pipe frames through main RespiratoryController."""
        # This integrates with the existing process_video flow
        # For now: skip if no temp file available — use signal fusion instead
        return None  # Override in subclass if needed


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Universal RR Estimator — any input type")
    parser.add_argument("source", help="Video/image/audio/npz/folder/webcam_int/rtsp_url")
    parser.add_argument("--audio", help="Separate audio file (for video-without-audio)")
    parser.add_argument("--fps",   type=float, default=30.0, help="Frame rate for image sequences")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="data/models/vlm_gated_fused.pth")
    args = parser.parse_args()

    adapter = UniversalAdapter(config_path=args.config,
                               checkpoint_path=args.checkpoint,
                               fps=args.fps)

    print(f"\n{'='*55}")
    print(f" Universal RR Estimator")
    print(f" Source: {args.source}")
    print(f"{'='*55}")

    source = int(args.source) if args.source.isdigit() else args.source
    bundle = adapter.load(source, extra_audio=args.audio, fps=args.fps)
    
    result = adapter.estimate_rr(bundle)
    
    print(f"\n{'='*55}")
    print(f" RR = {result['rr_bpm']:.1f} BPM")
    print(f" Confidence: {result['confidence']:.2f}")
    print(f" Method: {result['method']}")
    print(f" Modalities used: {result['modalities_used']}")
    if 'per_modality' in result:
        print(f" Per-modality:")
        for m, v in result['per_modality'].items():
            print(f"   {m}: {v['rr']:.1f} BPM (conf={v['conf']:.2f})")
    if bundle.metadata.get('gt_rr_bpm'):
        gt = bundle.metadata['gt_rr_bpm']
        print(f" GT (from annotation): {gt:.1f} BPM | Error: {abs(result['rr_bpm'] - gt):.1f} BPM")
    print(f"{'='*55}\n")
    print(json.dumps(result, indent=2))
