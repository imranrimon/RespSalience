"""
Dataset Pre-processor
Extracts respiratory signals (Depth, Flow) from raw videos for training.
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.main_controller import RespiratoryController

def preprocess_dataset(manifest_path, output_dir):
    """
    Reads dataset manifest and processes each video to extract signals.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    print(f"Loaded manifest with {len(manifest)} videos")
    
    # Initialize Controller (Disable VLM/Visualization for speed)
    # Use existing weights if available, else standard config
    controller = RespiratoryController(
        config_path="src/config/test_infer.yaml",
        checkpoint_path="data/models/VIRENet_Coarse2fine_Body_best.pth",
        enable_depth=True,
        enable_flow=True,
        enable_vmd=False, # We can VMD during training or here. Let's do raw extraction here.
        enable_keypoint=True, # Critical for stabilization
        enable_transformer=False,
        enable_supervised=False
    )
    
    success_count = 0
    
    for entry in tqdm(manifest, desc="Processing Videos"):
        video_path = entry['full_path']
        video_id = entry['video_id']
        dataset = entry['dataset']
        subject_id = entry['subject_id']
        
        # Output filename: dataset_subject_videoid.npz
        out_name = f"{dataset}_{subject_id}_{video_id}.npz"
        out_path = output_dir / out_name
        
        if out_path.exists():
            continue # Skip existing
            
        try:
            # Process video
            # Assumed age=6 months for ROI priors if unknown
            results = controller.process_video(video_path, age_months=6)
            
            # Save relevant signals
            np.savez_compressed(
                out_path,
                depth=results.get('depth_signal', []),
                flow=results.get('flow_signal', []),
                bpm=results.get('bpm', 0.0),
                depth_conf=results.get('depth_confidence', 0.0),
                flow_conf=results.get('flow_confidence', 0.0),
                dataset=dataset,
                subject=subject_id,
                video_id=video_id
            )
            success_count += 1
            
        except Exception as e:
            print(f"Failed to process {video_path}: {e}")
            
    print(f"\nProcessing Complete. {success_count} new signals saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/organized/dataset_manifest.json")
    parser.add_argument("--output", default="data/processed_signals")
    args = parser.parse_args()
    
    preprocess_dataset(args.manifest, args.output)
