import os
import shutil
import glob
import subprocess
from pathlib import Path

def cleanup():
    print(f"\n{'='*50}")
    print(" [RESET] Cleaning up previous experiments...")
    print(f"{'='*50}")
    
    # 1. Delete Experimental Models (Keep External/Pretrained Baselines)
    model_dir = Path("data/models")
    patterns_to_delete = [
        "ssl_benchmark_*.pth",
        "finetuned_*.pth", 
        "foundation_*.pth",
        "spmat_*.pth",
        "vlm_*.pth"
    ]
    
    deleted_count = 0
    if model_dir.exists():
        for pattern in patterns_to_delete:
            for file_path in model_dir.glob(pattern):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
    print(f"Deleted {deleted_count} model checkpoints.")

    # 2. Delete Results & Signals
    dirs_to_delete = [
        "data/ablation_results",          # Previous reports & semi-SL signals
        "data/processed_signals"          # SSL training data (forces re-extraction)
    ]
    
    for d in dirs_to_delete:
        path = Path(d)
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"Deleted Directory: {path}")
            except Exception as e:
                print(f"Error deleting {path}: {e}")

    print(f"\n[RESET] Cleanup Complete. Starting Full Pipeline...")
    print(f"{'='*50}\n")

def run_pipeline():
    try:
        # Run the main orchestrator
        subprocess.run(["python", "scripts/run_full_system.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline failed with exit code: {e.returncode}")
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline stopped by user.")

if __name__ == "__main__":
    cleanup()
    run_pipeline()
