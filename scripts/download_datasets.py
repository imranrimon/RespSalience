"""
Dataset Download Script for Respiratory Monitoring Project
Automates downloading of infant respiratory video datasets.
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import argparse

class DatasetDownloader:
    def __init__(self, base_dir="data/datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url, dest_path, desc="Downloading"):
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
                
    def extract_archive(self, archive_path, extract_to):
        """Extract zip or tar.gz archives."""
        print(f"Extracting {archive_path.name}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix == '.gz' and archive_path.stem.endswith('.tar'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unknown archive format: {archive_path}")
            
    def download_air400(self):
        """
        Download AIR-400 Dataset
        Note: This dataset may require manual download from the GitHub repository.
        """
        print("\n" + "="*60)
        print("AIR-400 Dataset")
        print("="*60)
        
        dataset_dir = self.base_dir / "AIR-400"
        
        print("""
AIR-400 Dataset Information:
- 400 videos of infant respiration
- Manual annotations included
- GitHub Repository: https://github.com/PHTemple/air-400

MANUAL DOWNLOAD REQUIRED:
1. Visit: https://github.com/PHTemple/air-400
2. Download the dataset (may require request access)
3. Extract to: {dataset_dir}
4. Run this script again to verify

Dataset structure should be:
{dataset_dir}/
  ├── videos/
  ├── annotations/
  └── README.md
""".format(dataset_dir=dataset_dir))
        
        if (dataset_dir / "videos").exists():
            print("✅ AIR-400 dataset found!")
            return True
        else:
            print("⚠️ AIR-400 not found. Please download manually.")
            return False
            
    def download_pediatric_rr_dataset(self):
        """
        Download 332-Video Pediatric Respiratory Rate Dataset
        """
        print("\n" + "="*60)
        print("Pediatric RR Dataset (332 Videos)")
        print("="*60)
        
        dataset_dir = self.base_dir / "Pediatric-RR-332"
        
        print("""
Dataset Information:
- 332 videos (~60 seconds each)
- Age range: 0-59 months
- Includes demographic and clinical metadata
- ResearchGate URL: https://www.researchgate.net/publication/[ID]

MANUAL DOWNLOAD REQUIRED:
This dataset is hosted on ResearchGate and requires:
1. ResearchGate account
2. Request access from authors
3. Download link will be provided via email

Steps:
1. Create ResearchGate account at: https://www.researchgate.net
2. Search for: "Dataset of Video Recordings for Respiratory Rate Counting in Children"
3. Click "Request full-text"
4. Extract downloaded files to: {dataset_dir}

Expected structure:
{dataset_dir}/
  ├── videos/
  ├── metadata.csv
  └── README.txt
""".format(dataset_dir=dataset_dir))
        
        if (dataset_dir / "videos").exists():
            print("✅ Pediatric RR dataset found!")
            return True
        else:
            print("⚠️ Dataset not found. Please download manually.")
            return False
            
    def download_rgb_thermal_dataset(self):
        """
        Download RGB + Thermal Video Dataset
        """
        print("\n" + "="*60)
        print("RGB + Thermal Video Dataset")
        print("="*60)
        
        dataset_dir = self.base_dir / "RGB-Thermal-Infant"
        
        print("""
Dataset Information:
- 43 hours of footage
- 17 infant subjects
- Dual modality: RGB + Thermal
- Research Paper: MDPI Sensors Journal

MANUAL DOWNLOAD REQUIRED:
This dataset may be available via:
1. Journal supplementary materials
2. Author's institutional repository
3. Direct request to authors

Paper Reference:
"Continuous Monitoring of Infant Respiration using RGB and Thermal Cameras"
MDPI Sensors

Steps:
1. Visit: https://www.mdpi.com/journal/sensors
2. Search for the paper title
3. Check "Supplementary Materials" section
4. OR contact authors for dataset access
5. Extract to: {dataset_dir}

Expected structure:
{dataset_dir}/
  ├── rgb_videos/
  ├── thermal_videos/
  ├── annotations.csv
  └── README.md
""".format(dataset_dir=dataset_dir))
        
        if (dataset_dir / "rgb_videos").exists():
            print("✅ RGB-Thermal dataset found!")
            return True
        else:
            print("⚠️ Dataset not found. Please download manually.")
            return False
            
    def download_mini_rgbd(self):
        """
        Download MINI-RGBD Dataset (Synthetic)
        """
        print("\n" + "="*60)
        print("MINI-RGBD Dataset (Synthetic)")
        print("="*60)
        
        dataset_dir = self.base_dir / "MINI-RGBD"
        
        print("""
Dataset Information:
- Synthetic RGB-D data for infant pose
- Useful for depth estimation validation
- GitHub/Project Page may have data

MANUAL DOWNLOAD:
1. Visit: https://github.com/ostadabbas/Infant-Posture-RGBD
   OR search "MINI-RGBD infant pose estimation"
2. Download dataset files
3. Extract to: {dataset_dir}

Note: This is primarily for pose estimation but depth data
can be useful for validating your DepthEngine.

Expected structure:
{dataset_dir}/
  ├── rgb/
  ├── depth/
  ├── poses/
  └── README.md
""".format(dataset_dir=dataset_dir))
        
        if (dataset_dir / "depth").exists():
            print("✅ MINI-RGBD dataset found!")
            return True
        else:
            print("⚠️ Dataset not found. Please download manually.")
            return False
            
    def verify_all_datasets(self):
        """Check which datasets are present."""
        print("\n" + "="*60)
        print("DATASET VERIFICATION SUMMARY")
        print("="*60)
        
        datasets = {
            "AIR-400": self.base_dir / "AIR-400" / "videos",
            "Pediatric-RR-332": self.base_dir / "Pediatric-RR-332" / "videos",
            "RGB-Thermal": self.base_dir / "RGB-Thermal-Infant" / "rgb_videos",
            "MINI-RGBD": self.base_dir / "MINI-RGBD" / "depth"
        }
        
        for name, path in datasets.items():
            status = "✅ FOUND" if path.exists() else "❌ NOT FOUND"
            print(f"{name:20} {status}")
            
        print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="Download respiratory monitoring datasets")
    parser.add_argument("--dataset", choices=["all", "air400", "pediatric", "thermal", "mini-rgbd"],
                       default="all", help="Which dataset to download")
    parser.add_argument("--verify", action="store_true", help="Verify existing datasets")
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.verify:
        downloader.verify_all_datasets()
        return
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║   RESPIRATORY MONITORING DATASET DOWNLOADER                  ║
╚══════════════════════════════════════════════════════════════╝

This script will guide you through downloading additional datasets
to expand your training data beyond the current 125 videos.

NOTE: Most research datasets require manual download due to:
  - Author permission requirements
  - Data use agreements
  - Authentication barriers

This script provides instructions and verifies downloads.
""")
    
    if args.dataset in ["all", "air400"]:
        downloader.download_air400()
        
    if args.dataset in ["all", "pediatric"]:
        downloader.download_pediatric_rr_dataset()
        
    if args.dataset in ["all", "thermal"]:
        downloader.download_rgb_thermal_dataset()
        
    if args.dataset in ["all", "mini-rgbd"]:
        downloader.download_mini_rgbd()
        
    print("\n")
    downloader.verify_all_datasets()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║   NEXT STEPS                                                 ║
╚══════════════════════════════════════════════════════════════╝

1. Follow the instructions above to manually download datasets
2. Run `python src/utils/download_datasets.py --verify` to check
3. Process datasets using appropriate scripts in src/utils/
4. Retrain models with expanded data

For questions or issues, refer to each dataset's documentation.
""")

if __name__ == "__main__":
    main()
