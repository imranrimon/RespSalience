"""
Dataset Organization Script
Merges and organizes all available respiratory datasets for training.
"""

import os
import sys
import shutil
import json
import csv
from pathlib import Path
from collections import defaultdict
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.dataset_paths import iter_pediatric_rr_videos, resolve_pediatric_rr_root


class DatasetOrganizer:
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.organized_dir = self.base_dir / "organized"
        self.organized_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets = {
            "AIR-125": self.base_dir / "AIR_125",
            "AIR-400": self.base_dir / "AIR_400",
            "MINI-RGBD": self.base_dir / "MINI-RGBD",
            "Pediatric-RR-332": resolve_pediatric_rr_root(self.base_dir),
        }
        
        self.manifest = []
        
    def scan_air_dataset(self, dataset_path, dataset_name):
        """Scan AIR dataset structure (S01, S02, etc.)"""
        print(f"\n[{dataset_name}] Scanning...")
        
        videos = []
        for subject_dir in sorted(dataset_path.glob("S*")):
            if not subject_dir.is_dir():
                continue
                
            subject_id = subject_dir.name
            for video_file in sorted(subject_dir.glob("*.mp4")):
                videos.append({
                    "dataset": dataset_name,
                    "subject_id": subject_id,
                    "video_id": video_file.stem,
                    "path": str(video_file.relative_to(self.base_dir)),
                    "full_path": str(video_file),
                    "format": "mp4"
                })
        
        print(f"  Found {len(videos)} videos across {len(list(dataset_path.glob('S*')))} subjects")
        return videos
    
    def scan_mini_rgbd(self):
        """Scan MINI-RGBD dataset"""
        print("\n[MINI-RGBD] Scanning...")
        dataset_path = self.datasets["MINI-RGBD"]
        
        if not dataset_path.exists():
            print("  Not found. Skipping.")
            return []
        
        videos = []
        # MINI-RGBD typically has RGB and Depth folders
        rgb_dir = dataset_path / "rgb"
        depth_dir = dataset_path / "depth"
        
        if rgb_dir.exists():
            for video_file in rgb_dir.glob("*.mp4"):
                videos.append({
                    "dataset": "MINI-RGBD",
                    "subject_id": "synthetic",
                    "video_id": video_file.stem,
                    "path": str(video_file.relative_to(self.base_dir)),
                    "full_path": str(video_file),
                    "format": "mp4",
                    "modality": "rgb",
                    "has_depth": (depth_dir / video_file.name).exists()
                })
        
        print(f"  Found {len(videos)} synthetic videos")
        return videos
    
    def scan_pediatric_rr(self):
        """Scan Pediatric RR 332 dataset"""
        print("\n[Pediatric-RR-332] Scanning...")
        dataset_path = self.datasets["Pediatric-RR-332"]
        
        if not dataset_path.exists():
            print("  Not found. Skipping.")
            return []
        
        videos = []
        metadata_file = dataset_path / "metadata.csv"
        for video_file in iter_pediatric_rr_videos(self.base_dir):
            subject_id = video_file.parent.name if video_file.parent.name.startswith("ID-") else video_file.stem
            entry = {
                "dataset": "Pediatric-RR-332",
                "subject_id": subject_id,
                "video_id": video_file.stem,
                "path": str(video_file.relative_to(self.base_dir)),
                "full_path": str(video_file),
                "format": "mp4",
            }
            if video_file.parent.parent != dataset_path and video_file.parent.parent.name:
                entry["site_id"] = video_file.parent.parent.name
            videos.append(entry)
        
        if metadata_file.exists():
            print(f"  Metadata file found: {metadata_file}")
        
        print(f"  Found {len(videos)} videos")
        return videos
    
    def organize_all(self):
        """Scan all datasets and create unified manifest"""
        print("="*60)
        print("DATASET ORGANIZATION")
        print("="*60)
        
        # Scan all datasets
        all_videos = []
        
        # AIR-125
        if self.datasets["AIR-125"].exists():
            all_videos.extend(self.scan_air_dataset(self.datasets["AIR-125"], "AIR-125"))
        
        # AIR-400
        if self.datasets["AIR-400"].exists():
            all_videos.extend(self.scan_air_dataset(self.datasets["AIR-400"], "AIR-400"))
        
        # MINI-RGBD
        all_videos.extend(self.scan_mini_rgbd())
        
        # Pediatric RR
        all_videos.extend(self.scan_pediatric_rr())
        
        self.manifest = all_videos
        
        # Save manifest
        manifest_path = self.organized_dir / "dataset_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(all_videos, f, indent=2)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Group by dataset
        by_dataset = defaultdict(list)
        for v in all_videos:
            by_dataset[v["dataset"]].append(v)
        
        total = 0
        for dataset_name, videos in sorted(by_dataset.items()):
            count = len(videos)
            total += count
            print(f"{dataset_name:20} {count:5} videos")
        
        print(f"{'TOTAL':20} {total:5} videos")
        print(f"\nManifest saved: {manifest_path}")
        
        return all_videos
    
    def create_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Create train/val/test splits"""
        print("\n" + "="*60)
        print("CREATING DATA SPLITS")
        print("="*60)
        
        random.seed(seed)
        
        # Group by dataset for stratified splitting
        by_dataset = defaultdict(list)
        for v in self.manifest:
            by_dataset[v["dataset"]].append(v)
        
        splits = {
            "train": [],
            "val": [],
            "test": []
        }
        
        for dataset_name, videos in by_dataset.items():
            random.shuffle(videos)
            
            n = len(videos)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            splits["train"].extend(videos[:train_end])
            splits["val"].extend(videos[train_end:val_end])
            splits["test"].extend(videos[val_end:])
            
            print(f"{dataset_name}:")
            print(f"  Train: {len(videos[:train_end])}")
            print(f"  Val:   {len(videos[train_end:val_end])}")
            print(f"  Test:  {len(videos[val_end:])}")
        
        print(f"\nTOTAL:")
        print(f"  Train: {len(splits['train'])}")
        print(f"  Val:   {len(splits['val'])}")
        print(f"  Test:  {len(splits['test'])}")
        
        # Save splits
        for split_name, split_videos in splits.items():
            split_path = self.organized_dir / f"{split_name}_split.json"
            with open(split_path, 'w') as f:
                json.dump(split_videos, f, indent=2)
            print(f"  {split_name}_split.json saved")
        
        return splits
    
    def generate_training_csv(self, splits):
        """Generate CSV files for training scripts"""
        print("\n" + "="*60)
        print("GENERATING TRAINING CSV FILES")
        print("="*60)
        
        for split_name, videos in splits.items():
            csv_path = self.organized_dir / f"{split_name}.csv"
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['video_path', 'dataset', 'subject_id'])
                writer.writeheader()
                
                for v in videos:
                    writer.writerow({
                        'video_path': v['full_path'],
                        'dataset': v['dataset'],
                        'subject_id': v['subject_id']
                    })
            
            print(f"  {csv_path.name} saved ({len(videos)} entries)")

def main():
    print("DATASET ORGANIZATION TOOL")
    
    organizer = DatasetOrganizer()
    
    # Step 1: Scan and organize
    all_videos = organizer.organize_all()
    
    if len(all_videos) == 0:
        print("\n❌ No videos found. Please check dataset paths.")
        return
    
    # Step 2: Create splits
    splits = organizer.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Step 3: Generate training CSVs
    organizer.generate_training_csv(splits)
    
    print("\n" + "="*60)
    print("✅ DATASET ORGANIZATION COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {organizer.organized_dir}")
    print("\nNext steps:")
    print("1. Review data/organized/ directory")
    print("2. Run training with expanded dataset")
    print("3. Compare results across datasets")

if __name__ == "__main__":
    main()
