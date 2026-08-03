import os
import shutil
import glob
from pathlib import Path

def reorganize():
    print("Starting Project Reorganization...")
    
    # 1. Create Directories
    directories = ["scripts", "docs", "tests", "data/models"]
    for d in directories:
        os.makedirs(d, exist_ok=True)
        print(f"Ensured directory: {d}")

    # 2. Move Training Scripts to scripts/
    # We will rename them to remove the 'src' dependency if possible, or just move them
    # Actually, moving them out of src might break imports unless we adjust sys.path.
    # The 'standard' way is keeping specific modules in src, and execution scripts in scripts/ or root.
    # But usually scripts/ need to add project root to sys.path.
    
    scripts_to_move = [
        ("src/train_ssl_benchmark.py", "scripts/train_ssl_benchmark.py"),
        ("src/train_finetune.py", "scripts/train_finetune.py"),
        ("src/train_foundation.py", "scripts/train_foundation.py"),
        ("src/run_ablation_study.py", "scripts/run_ablation_study.py"),
        ("src/train_medgemma_demo.py", "scripts/train_medgemma_demo.py"),
        # Utils that are actually scripts
        ("src/utils/download_datasets.py", "scripts/download_datasets.py"),
        ("src/utils/organize_datasets.py", "scripts/organize_datasets.py"),
        ("src/utils/preprocess_dataset.py", "scripts/preprocess_dataset.py"),
        ("run_full_system.py", "scripts/run_full_system.py"),
        ("run_full_system.bat", "run_full_system.bat") # Keep bat in root
    ]
    
    for src, dst in scripts_to_move:
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {src} -> {dst}")
            
    # 3. Move Docs
    docs = [
        ("AUDIO_STRATEGY.md", "docs/AUDIO_STRATEGY.md"),
        ("2504.20112v1.pdf", "docs/reference_paper.pdf")
    ]
    for src, dst in docs:
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {src} -> {dst}")

    # 4. Cleanup
    # Remove empty directories
    empty_dirs = ["notebooks", "src/utils/__pycache__", "src/__pycache__"]
    for d in empty_dirs:
        if os.path.exists(d) and not os.listdir(d):
            os.rmdir(d)
            print(f"Removed empty dir: {d}")

if __name__ == "__main__":
    reorganize()
