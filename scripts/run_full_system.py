import subprocess
import os
import sys
import time
from pathlib import Path

def run_step(step_name, command, shell=False):
    print(f"\n==================================================")
    print(f"[TASK] {step_name}")
    print(f"==================================================")
    try:
        # Use shell=True if argument is a string
        if isinstance(command, str):
            shell = True
            
        result = subprocess.run(command, shell=shell, check=True)
        print(f"[SUCCESS] {step_name} complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {step_name} failed with exit code {e.returncode}")
        return False

def main():
    print("[INFO] Starting Full-System Experiment Pipeline (Python Orchestrator)...")
    
    # ══════════════════════════════════════════════════════════
    # STAGE 0: Signal Extraction (generates .npz files for training)
    # ══════════════════════════════════════════════════════════
    ssl_data_dir = Path("data/processed_signals")
    semi_sl_data_dir = Path("data/ablation_results/signals")
    
    # Check if we have enough signals for training
    if ssl_data_dir.exists() and len(list(ssl_data_dir.glob("*.npz"))) > 10:
        print(f"[INFO] Found {len(list(ssl_data_dir.glob('*.npz')))} processed signals. Skipping extraction.")
    else:
        print("[INFO] Signals not found. Running Preprocessing...")
        if not run_step("Signal Extraction", ["python", "scripts/preprocess_dataset.py"]):
            sys.exit(1)
    
    # If ablation signals with VLM labels exist, use them for semi-SL
    if not semi_sl_data_dir.exists() or len(list(semi_sl_data_dir.glob("*.npz"))) < 10:
        print("[INFO] Ablation signals not found. Running quick signal extraction pass...")
        run_step("Signal Extraction (with VLM labels)", 
                 f"python scripts/run_ablation_study.py --dataset AIR-400 --save-signals --baseline Flow-Only")

    # ══════════════════════════════════════════════════════════
    # STAGE 1: Self-Supervised Pretraining (3 methods)
    # Uses processed_signals (no labels needed)
    # ══════════════════════════════════════════════════════════
    ssl_dir = str(ssl_data_dir)
    if not Path("data/models/ssl_benchmark_msm.pth").exists():
        run_step("SSL: MSM Pretraining", f'python scripts/train_ssl_benchmark.py --method msm --epochs 100 --data_dir "{ssl_dir}"')
    else:
        print("[INFO] SSL: MSM model exists. Skipping.")

    if not Path("data/models/ssl_benchmark_sae.pth").exists():
        run_step("SSL: SAE Pretraining", f'python scripts/train_ssl_benchmark.py --method sae --epochs 100 --data_dir "{ssl_dir}"')
    else:
        print("[INFO] SSL: SAE model exists. Skipping.")

    if not Path("data/models/ssl_benchmark_contrast.pth").exists():
        run_step("SSL: Contrastive Pretraining", f'python scripts/train_ssl_benchmark.py --method contrast --epochs 100 --data_dir "{ssl_dir}"')
    else:
        print("[INFO] SSL: Contrastive model exists. Skipping.")

    # ══════════════════════════════════════════════════════════
    # STAGE 2: Semi-Supervised (VLM Surrogate Labels)
    # Uses ablation_results/signals (has vlm_label field)
    # ══════════════════════════════════════════════════════════
    semi_dir = str(semi_sl_data_dir)
    if not Path("data/models/spmat_bt.pth").exists():
        run_step("Semi-SL: SPMat-BT Pretraining", f'python scripts/train_vlm_fusion.py --stage pretrain --epochs 50 --data_dir "{semi_dir}"')
    else:
        print("[INFO] Semi-SL: SPMat-BT model exists. Skipping.")

    if not Path("data/models/vlm_gated_fused.pth").exists():
        run_step("Semi-SL: VLM-Gated Fine-tuning", f'python scripts/train_vlm_fusion.py --stage finetune --ft_epochs 30 --data_dir "{semi_dir}"')
    else:
        print("[INFO] Semi-SL: VLM-Gated model exists. Skipping.")

    # ══════════════════════════════════════════════════════════
    # STAGE 3: Supervised Fine-Tuning (ALL SSL backbones)
    # Each SSL pretrained model needs its regression head trained
    # ══════════════════════════════════════════════════════════
    ssl_finetune_configs = [
        ("data/models/ssl_benchmark_msm.pth",      "data/models/finetuned_msm.pth"),
        ("data/models/ssl_benchmark_sae.pth",       "data/models/finetuned_sae.pth"),
        ("data/models/ssl_benchmark_contrast.pth",  "data/models/finetuned_contrast.pth"),
    ]
    
    for pretrained_path, output_path in ssl_finetune_configs:
        backbone_name = Path(pretrained_path).stem.replace("ssl_benchmark_", "").upper()
        if Path(output_path).exists():
            print(f"[INFO] Finetuned {backbone_name} model exists. Skipping.")
            continue
        if os.path.exists(pretrained_path):
            run_step(f"Supervised: Fine-Tune ({backbone_name} backbone)", 
                     f'python scripts/train_finetune.py --pretrained "{pretrained_path}" --output "{output_path}" --epochs 100 --data_dir "{semi_dir}"')
        else:
            print(f"[WARN] SSL backbone {pretrained_path} not found. Skipping finetune.")
    
    # Also keep a copy as the "foundation" model (best SSL = MSM)
    if Path("data/models/finetuned_msm.pth").exists() and not Path("data/models/finetuned_foundation.pth").exists():
        import shutil
        shutil.copy2("data/models/finetuned_msm.pth", "data/models/finetuned_foundation.pth")
        print("[INFO] Copied finetuned_msm.pth → finetuned_foundation.pth")

    # ══════════════════════════════════════════════════════════
    # STAGE 4: Ablation Evaluation (All Paradigms × All Datasets)
    # ══════════════════════════════════════════════════════════
    run_step("Ablation: AIR-125", f"python scripts/run_ablation_study.py --dataset AIR-125 --save-signals")
    run_step("Ablation: AIR-400", f"python scripts/run_ablation_study.py --dataset AIR-400 --save-signals")
    run_step("Ablation: MINI-RGBD", f"python scripts/run_ablation_study.py --dataset MINI-RGBD")
    run_step("Ablation: Pediatric-RR", f"python scripts/run_ablation_study.py --dataset Pediatric-RR --save-signals")
    
    # Combined report
    run_step("Final Combined Report", f"python scripts/run_ablation_study.py")

    # ══════════════════════════════════════════════════════════
    # STAGE 5: Generate Results Tables (Markdown + LaTeX)
    # ══════════════════════════════════════════════════════════
    run_step("Generate Results Tables", f"python scripts/generate_results_tables.py --format both")

    # 5. Dashboard
    print("\n==================================================")
    print("[INFO] Pipeline Complete. Launching Dashboard...")
    print("==================================================")
    try:
        subprocess.Popen(["streamlit", "run", "src/ui/dashboard.py"])
        print("[SUCCESS] Dashboard launched.")
    except Exception as e:
        print(f"[ERROR] Failed to launch dashboard: {e}")

if __name__ == "__main__":
    main()
