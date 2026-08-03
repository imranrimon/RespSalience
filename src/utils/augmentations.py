import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Respiratory event augmentations (apnea / motion artifacts)
# ---------------------------------------------------------------------------
# These synthesize the two failure modes the NICU/infant-respiration literature
# explicitly reports as unsolved (respiratory pauses/apnea and limb-motion
# corruption). They operate on 1D respiratory signals and return a binary event
# label + per-sample window mask, providing weak supervision for the apnea/motion
# detection heads in PrismBreathModel without needing hand-labeled events.


def _event_window(n: int, min_frac: float, max_frac: float, rng: np.random.Generator) -> Tuple[int, int, np.ndarray]:
    """Pick a contiguous window and return (start, length, per-sample mask)."""
    mask = np.zeros(n, dtype=np.float32)
    if n < 8:
        return 0, 0, mask
    frac = float(rng.uniform(min_frac, max_frac))
    win = max(4, int(frac * n))
    win = min(win, n)
    start = int(rng.integers(0, max(1, n - win + 1)))
    mask[start : start + win] = 1.0
    return start, win, mask


def apply_apnea_window(signal: np.ndarray, start: int, win: int, residual: float, rng: np.random.Generator) -> np.ndarray:
    """Collapse breathing amplitude inside a window to simulate a respiratory pause."""
    sig = np.asarray(signal, dtype=np.float32).copy()
    if win <= 0:
        return sig
    baseline = float(sig[max(0, start - 1)])
    sig[start : start + win] = baseline + residual * rng.standard_normal(win).astype(np.float32)
    return sig


def apply_motion_window(signal: np.ndarray, start: int, win: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Inject a large non-respiratory burst + baseline drift to simulate limb motion."""
    sig = np.asarray(signal, dtype=np.float32).copy()
    if win <= 0:
        return sig
    amp = float(strength) * (float(np.std(sig)) + 1e-3)
    burst = amp * rng.standard_normal(win).astype(np.float32)
    ramp = np.linspace(0.0, 1.0, win, dtype=np.float32)
    step = amp * float(rng.uniform(-1.0, 1.0)) * ramp
    sig[start : start + win] += burst + step
    return sig


def simulate_apnea(
    signal: np.ndarray,
    min_frac: float = 0.2,
    max_frac: float = 0.5,
    residual: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return (augmented_signal, apnea_label=1.0, window_mask)."""
    rng = rng or np.random.default_rng()
    n = int(np.asarray(signal).size)
    start, win, mask = _event_window(n, min_frac, max_frac, rng)
    return apply_apnea_window(signal, start, win, residual, rng), 1.0, mask


def simulate_motion_artifact(
    signal: np.ndarray,
    min_frac: float = 0.1,
    max_frac: float = 0.35,
    strength: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return (augmented_signal, motion_label=1.0, window_mask)."""
    rng = rng or np.random.default_rng()
    n = int(np.asarray(signal).size)
    start, win, mask = _event_window(n, min_frac, max_frac, rng)
    return apply_motion_window(signal, start, win, strength, rng), 1.0, mask


def apply_event_augmentations(
    signals: Dict[str, np.ndarray],
    apnea_prob: float = 0.15,
    motion_prob: float = 0.2,
    apnea_modalities: Sequence[str] = ("depth", "flow", "audio"),
    motion_modalities: Sequence[str] = ("depth", "flow", "audio"),
    reference: str = "depth",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Stochastically inject apnea and/or motion events across modalities.

    A single coherent window is shared across the affected modalities (a real
    apnea/motion event corrupts the depth and flow views simultaneously). Returns
    the possibly-augmented signals and an event dict with binary ``apnea``/``motion``
    labels and their per-sample masks (``None`` when the event was not applied).
    """
    rng = rng or np.random.default_rng()
    out = {key: np.asarray(value, dtype=np.float32).copy() for key, value in signals.items()}
    ref = out.get(reference)
    if ref is None:
        ref = next(iter(out.values()))
    n = int(ref.size)

    events: Dict[str, object] = {"apnea": 0.0, "motion": 0.0, "apnea_mask": None, "motion_mask": None}

    if rng.random() < float(apnea_prob):
        start, win, mask = _event_window(n, 0.2, 0.5, rng)
        for modality in apnea_modalities:
            if modality in out:
                out[modality] = apply_apnea_window(out[modality], start, win, residual=0.05, rng=rng)
        events["apnea"] = 1.0
        events["apnea_mask"] = mask

    if rng.random() < float(motion_prob):
        start, win, mask = _event_window(n, 0.1, 0.35, rng)
        for modality in motion_modalities:
            if modality in out:
                out[modality] = apply_motion_window(out[modality], start, win, strength=3.0, rng=rng)
        events["motion"] = 1.0
        events["motion_mask"] = mask

    return out, events


class TimeWarpAugmentation:
    """
    Physiological Time-Warping Augmentation (Equivalent to GNDN for Signals).
    Randomly warps the time-axis to simulate irregular breathing patterns
    while preserving the topological sequence of peaks/valleys.
    """
    def __init__(self, warp_sigma=0.2, num_knots=4):
        self.warp_sigma = warp_sigma
        self.num_knots = num_knots

    def __call__(self, signal_tensor):
        """
        Args:
            signal_tensor: (C, L) or (L,) torch tensor
        Returns:
            warped_signal: Same shape
        """
        is_1d = signal_tensor.dim() == 1
        if is_1d:
            signal_tensor = signal_tensor.unsqueeze(0) # (1, L)
            
        C, L = signal_tensor.shape
        device = signal_tensor.device
        
        # 1. Generate random warp grid
        # Anchor points at 0 and 1, perturb intermediates
        knots = torch.linspace(0, 1, self.num_knots + 2, device=device)
        noise = torch.randn(self.num_knots, device=device) * self.warp_sigma
        # Apply noise to inner knots
        knots[1:-1] += noise
        # Clamp to ensure monotonicity (simple sort)
        knots, _ = torch.sort(knots)
        knots[0] = 0.0
        knots[-1] = 1.0
        
        # 2. Interpolate to full length L
        orig_grid = torch.linspace(0, 1, L, device=device)
        # Linear interpolation of knots to get warp field
        # We want to map Orig -> Warped
        # Actually we need the INVERSE map for sampling: "Where do I look in source to get value at t?"
        # But for 1D, simple monotonic mapping works.
        
        # We use grid_sample? No, 1D interpolation is easier manually or via F.interpolate loop?
        # Let's use numpy.interp style logic or 1D grid sample.
        
        # Grid Sample requires (B, C, L, 1)?
        # Let's simple linear interp between knots for the warp field
        # T_warped = interp(knots_val, knots_idx, T_orig)
        
        # Simpler: Generate Flow Field
        # Flow is smooth deviation.
        
        # Let's use `grid_sample`.
        # Grid must be in [-1, 1].
        
        # Create Warped Grid in [-1, 1]
        # We define values at knots.
        knot_vals = torch.linspace(-1, 1, self.num_knots + 2, device=device)
        knot_vals[1:-1] += torch.randn(self.num_knots, device=device) * self.warp_sigma
        knot_vals, _ = torch.sort(knot_vals)
        knot_vals[0] = -1.0
        knot_vals[-1] = 1.0
        
        # Interpolate knot_vals to size L
        # (1, 1, num_knots) -> (1, 1, L)
        knot_tensor = knot_vals.view(1, 1, -1)
        grid_1d = F.interpolate(knot_tensor, size=L, mode='linear', align_corners=True)
        # grid_1d is (1, 1, L) -> values in [-1, 1]
        
        # Prepare signal for grid_sample (B, C, W, 1) -> W=L
        input_sig = signal_tensor.unsqueeze(0).unsqueeze(-1) # (1, C, L, 1)
        
        # Prepare Grid (1, L, 1, 2) ?
        # grid_sample(input, grid)
        # input: (N, C, H_in, W_in)
        # grid: (N, H_out, W_out, 2)
        # We treat L as "W". H=1.
        
        grid_expand = grid_1d.permute(0, 2, 1).unsqueeze(-1) # (1, L, 1, 1)
        # Last dim needs x, y coordinates.
        # x are the warped time points. y is 0 (constant).
        
        zeros = torch.zeros_like(grid_expand)
        grid_final = torch.cat([grid_expand, zeros], dim=-1) # (1, L, 1, 2)
        
        warped = F.grid_sample(input_sig, grid_final, mode='bilinear', padding_mode='reflection', align_corners=True)
        # warped: (1, C, L, 1)
        
        warped = warped.squeeze(0).squeeze(-1) # (C, L)
        
        if is_1d:
            warped = warped.squeeze(0)
            
        return warped

if __name__ == "__main__":
    # Test
    sig = torch.sin(torch.linspace(0, 10, 100))
    aug = TimeWarpAugmentation(warp_sigma=0.3)
    out = aug(sig)
    print(f"Augmented shape: {out.shape}")
