import glob
import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.respiratory_benchmark_dataset import signal_filename_from_entry
from src.utils.augmentations import TimeWarpAugmentation

class RespiratorySignalDataset(Dataset):
    """
    Dataset for SPMat Pretraining and Downstream Fine-tuning.
    Loads signals saved by run_ablation_study.py (--save-signals).
    """
    def __init__(self, data_dir, split='all', mode='pretrain', seq_len=300, manifest_path: Optional[str] = None):
        """
        Args:
            data_dir: Path to signal directory (e.g. data/ablation_results/signals)
            split: 'train', 'val', or 'all' (Simple split based on hashing subject ID)
            mode: 'pretrain' (returns pairs + surrogate) or 'finetune' (returns signal + gt)
            seq_len: Target length for padding/cutting
            manifest_path: Optional canonical benchmark manifest. When provided,
                the dataset uses the exact files referenced by that split.
        """
        self.mode = mode
        self.seq_len = seq_len
        self.manifest_path = manifest_path
        self.files = self._resolve_files(data_dir=data_dir, split=split, manifest_path=manifest_path)

        self.augmentor = TimeWarpAugmentation(warp_sigma=0.2)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        
        # Extract Signal (Depth preferred, or average of Flow+Depth?)
        # Let's use Depth as primary backbone input for now.
        # Or better: Provide BOTH (since Transformer uses both).
        # We'll enable "Early Fusion" by stacking or separate streams.
        # For this Foundation Model, let's treat "Depth" as the anchor.
        
        raw_depth = data['depth']
        raw_flow = data.get('flow', np.zeros_like(raw_depth))
        
        # Process to fixed length
        depth = self._process_signal(raw_depth)
        flow = self._process_signal(raw_flow)
        
        if self.mode == 'pretrain':
            # SPMat: Needs Two Views + Surrogate Label
            
            # View 1: "Clean" (or lightly augmented)
            # View 2: "Strongly Augmented" (Time Warped)
            
            v1_depth = torch.from_numpy(depth).float()
            v1_flow = torch.from_numpy(flow).float()
            
            # Augment View 2
            v2_depth = self.augmentor(v1_depth)
            v2_flow = self.augmentor(v1_flow) # Warp flow similarly? 
            # Ideally warp using SAME grid, but independent noise is also valid for "View Invariance".
            
            # Surrogate Label
            # 0 = Normal, 1 = Distress
            is_distress = data.get('vlm_label', False)
            label = 1.0 if is_distress else 0.0
            
            return {
                "v1_depth": v1_depth, "v1_flow": v1_flow,
                "v2_depth": v2_depth, "v2_flow": v2_flow,
                "label": torch.tensor(label).float(),
                "path": path
            }
            
        else:
            # Fine-tune: Signal + GT
            # Support both 'gt' and 'bpm' keys (legacy/new compatibility)
            gt = data.get('gt', data.get('bpm', 0.0))
            return {
                "depth": torch.from_numpy(depth).float(),
                "flow": torch.from_numpy(flow).float(),
                "gt": torch.tensor(float(gt)).float()
            }

    def _process_signal(self, sig):
        # Flatten
        sig = np.array(sig).flatten()
        # Truncate
        if len(sig) > self.seq_len:
            start = (len(sig) - self.seq_len) // 2
            sig = sig[start : start + self.seq_len]
        # Pad
        elif len(sig) < self.seq_len:
            pad = self.seq_len - len(sig)
            sig = np.pad(sig, (0, pad), mode='edge')
            
        # Normalize (Standardize)
        if np.std(sig) > 1e-6:
            sig = (sig - np.mean(sig)) / np.std(sig)
        else:
            sig = sig - np.mean(sig)
            
        return sig

    @staticmethod
    def _stable_split_fraction(text: str) -> float:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return int(digest, 16) / float(16 ** 16 - 1)

    def _resolve_files(self, data_dir: str, split: str, manifest_path: Optional[str]) -> List[str]:
        if manifest_path:
            manifest_file = Path(manifest_path)
            if not manifest_file.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_file}")
            signal_dir = Path(data_dir)
            with open(manifest_file, "r", encoding="utf-8") as handle:
                manifest_entries = json.load(handle)

            manifest_files = []
            for entry in manifest_entries:
                signal_path = signal_dir / signal_filename_from_entry(entry)
                if signal_path.exists():
                    manifest_files.append(str(signal_path))
            return sorted(manifest_files)

        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        train_files = []
        val_files = []
        for file_path in files:
            if self._stable_split_fraction(Path(file_path).stem) < 0.2:
                val_files.append(file_path)
            else:
                train_files.append(file_path)

        if split == 'train':
            return train_files
        if split == 'val':
            return val_files
        return files

class MaskedRespiratoryDataset(RespiratorySignalDataset):
    """
    Dataset for Masked Signal Modeling (MSM).
    Inherits preprocessing from RespiratorySignalDataset.
    Returns: Masked Signal (Input) + Original Signal (Target).
    """
    def __init__(self, data_dir, mask_ratio=0.5, block_size=30, **kwargs):
        super().__init__(data_dir, mode='pretrain', **kwargs)
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        
    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        
        # Process Raw
        depth = self._process_signal(data['depth'])
        flow = self._process_signal(data.get('flow', np.zeros_like(depth)))
        
        depth_ts = torch.from_numpy(depth).float()
        flow_ts = torch.from_numpy(flow).float()
        
        # Apply Mask
        masked_depth, mask_d = self._apply_mask(depth_ts)
        masked_flow, mask_f = self._apply_mask(flow_ts)
        
        return {
            "masked_depth": masked_depth, 
            "masked_flow": masked_flow,
            "target_depth": depth_ts,
            "target_flow": flow_ts,
            "mask_depth": mask_d,
            "mask_flow": mask_f,
            "path": path
        }
        
    def _apply_mask(self, signal):
        """Masks 50% of signal in random blocks."""
        L = len(signal)
        mask = torch.ones(L)
        masked_sig = signal.clone()
        
        # Number of blocks to achieve mask_ratio
        target_masked = int(L * self.mask_ratio)
        num_blocks = max(1, target_masked // self.block_size)
        
        for _ in range(num_blocks):
            start = np.random.randint(0, L - self.block_size - 1)
            # Find a valid start (simple random, overlaps allowed)
            mask[start : start + self.block_size] = 0
            masked_sig[start : start + self.block_size] = 0
            
        return masked_sig, mask
