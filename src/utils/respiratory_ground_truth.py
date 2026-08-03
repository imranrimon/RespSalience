"""
Utility to load ground truth annotations from datasets:
  - AIR-400:  VIA JSON annotations
  - AIR-125:  HDF5 respiration waveforms
  - Pediatric-RR-332: Stata (.dta) file with manual respiratory rates
"""
import json
from pathlib import Path
import pandas as pd

from src.utils.dataset_paths import resolve_pediatric_rr_stata_path

# Cache the Pediatric-RR Stata dataframe so we only read it once
_pediatric_rr_cache = None

def _load_pediatric_rr_stata():
    """Lazily loads and caches the Pediatric-RR Stata file."""
    global _pediatric_rr_cache
    if _pediatric_rr_cache is None:
        dta_path = resolve_pediatric_rr_stata_path(base_dir="data")
        if dta_path and dta_path.exists():
            _pediatric_rr_cache = pd.read_stata(str(dta_path))
        else:
            _pediatric_rr_cache = pd.DataFrame()  # Empty fallback
    return _pediatric_rr_cache

def load_pediatric_rr_ground_truth(video_path):
    """
    Loads ground truth BPM for a Pediatric-RR-332 video.
    Args:
        video_path: Path to video file (e.g., '.../ID-1115/100-4616.mp4')
    Returns:
        dict with 'respiratory_rate', 'age_months', or None if not found
    """
    video_path = Path(video_path)
    # Extract child_id from parent folder name (e.g., "ID-1115" -> 1115)
    folder_name = video_path.parent.name  # "ID-1115"
    try:
        if folder_name.startswith("ID-"):
            child_id = int(folder_name.replace("ID-", ""))
        else:
            child_id = int("".join(ch for ch in video_path.stem if ch.isdigit())[:4])
    except ValueError:
        return None
    
    df = _load_pediatric_rr_stata()
    if df.empty:
        return None

    child_id_col = "child_id" if "child_id" in df.columns else None
    rr_col = "respiratory_rate" if "respiratory_rate" in df.columns else None
    age_col = "age" if "age" in df.columns else None
    if child_id_col is None or rr_col is None or age_col is None:
        return None

    row = df[df[child_id_col] == child_id]
    if row.empty:
        return None
    
    row = row.iloc[0]
    return {
        'respiratory_rate': float(row[rr_col]),
        'age_months': int(row[age_col]),
    }

def load_air400_ground_truth(video_path):
    """
    Loads ground truth BPM for an AIR-400 video.
    Args:
        video_path: Path to video file (e.g., 'data/AIR_400/S01/1.mp4')
    Returns:
        float: Ground truth BPM, or None if not found
    """
    video_path = Path(video_path)
    video_name = video_path.stem  # e.g., "1" from "1.mp4"
    subject_dir = video_path.parent
    
    # Find VIA annotation JSON
    via_files = list(subject_dir.glob("via_project_*.json"))
    if not via_files:
        return None
        
    # Load annotations
    with open(via_files[0], 'r') as f:
        annotations = json.load(f)
    
    # Find video ID
    video_id = None
    for fid, file_info in annotations.get('file', {}).items():
        if file_info['fname'] == f"{video_name}.mp4":
            video_id = fid
            break
    
    if not video_id:
        return None
    
    # Count breathing cycles (temporal segments)
    metadata = annotations.get('metadata', {})
    cycles = []
    for m in metadata.values():
        if m.get('vid') == video_id:
            z = m.get('z')
            if isinstance(z, list) and len(z) >= 2:
                cycles.append(m)
    
    if not cycles:
        return None
    
    last_cycle = max(cycles, key=lambda c: c['z'][1])
    duration_sec = last_cycle['z'][1]
    duration_min = duration_sec / 60.0
    
    num_breaths = len(cycles)
    bpm = num_breaths / duration_min if duration_min > 0 else None
    
    return round(bpm, 1) if bpm else None

def _air125_video_fps(video_path):
    """Return the AIR-125 video's true frame rate (the waveform's sampling rate).

    The HDF5 respiration/impulse arrays are stored one sample per video frame, so
    the correct sampling rate is the video fps (10 fps for S03, 15 fps for S05, ...),
    NOT a fixed 30 Hz. Returns None if it cannot be read.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return float(fps)
    except Exception:
        pass
    return None


def load_air125_ground_truth(video_path):
    """
    Loads ground truth BPM for an AIR-125 video from HDF5.

    Ground truth is the annotated breath count in the HDF5 'impulse' channel
    (one marker per breath), divided by the true recording duration. The waveform
    is sampled at the VIDEO frame rate (10/15 fps), so duration = N_samples / video_fps.
    Historically this used SignalProcessor.get_bpm(resp, fps=30.0), which assumed a
    fixed 30 Hz sampling rate and inflated every AIR-125 RR by 30/actual_fps
    (3x for 10 fps clips, 2x for 15 fps) — the source of the AIR-125 "catastrophe".
    """
    try:
        import h5py
        import numpy as np
        from src.core.signal_processor import SignalProcessor

        video_path = Path(video_path)
        h5_path = video_path.with_suffix('.hdf5')

        if not h5_path.exists():
            return None

        with h5py.File(h5_path, 'r') as f:
            resp_signal = f['respiration'][:] if 'respiration' in f else None
            impulse = f['impulse'][:] if 'impulse' in f else None

        if resp_signal is None:
            return None

        # True sampling rate = video fps (waveform is 1 sample/frame).
        fps = _air125_video_fps(video_path)
        if fps is None:
            fps = 30.0  # last-resort fallback; will be inflated — flag it loudly
            print(f"[WARN] AIR-125 fps unavailable for {h5_path.name}; "
                  f"falling back to 30 Hz (RR may be inflated).")

        n_samples = len(resp_signal)
        duration_sec = n_samples / fps if fps else 0.0

        # Preferred GT: annotated breath-event count / true duration.
        if impulse is not None and duration_sec > 0:
            n_breaths = int(np.count_nonzero(impulse))
            if n_breaths > 0:
                return round(n_breaths / duration_sec * 60.0, 1)

        # Fallback: spectral BPM at the correct sampling rate.
        return SignalProcessor.get_bpm(resp_signal, fps=fps)
    except Exception as e:
        print(f"Error loading HDF5 GT: {e}")
        return None

if __name__ == "__main__":
    # Test
    test_video = "data/AIR_400/S01/1.mp4"
    gt_bpm = load_air400_ground_truth(test_video)
    print(f"Ground truth for {test_video}: {gt_bpm} BPM")
    
    # Test Pediatric-RR
    test_ped = "data/Video Dataset for Respiratory Rate Counting in Children/Video Files/1. Institute of Child & Mother Health/ID-1115/100-4616.mp4"
    gt_ped = load_pediatric_rr_ground_truth(test_ped)
    print(f"Pediatric GT for {test_ped}: {gt_ped}")
