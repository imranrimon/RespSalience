import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.vlm_trainer import VLMTrainer
from PIL import Image
import numpy as np

def run_demo():
    print("=== MedGemma Fine-tuning Demo ===")
    
    # 1. Setup
    output_dir = "data/models/medgemma_lora_demo"
    trainer = VLMTrainer(output_dir=output_dir)
    
    # 2. Create Dummy Data (Simulating AIR-400 frames)
    # in usage, we would scan 'data/AIR-400/*'
    print("Generating synthetic training data...")
    if not os.path.exists("data/demo_train"):
        os.makedirs("data/demo_train")
        
    img_paths = []
    prompts = []
    labels = []
    
    # Case 1: Normal
    img1_path = "data/demo_train/normal_01.jpg"
    Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(img1_path)
    img_paths.append(img1_path)
    prompts.append("Describe the infant's breathing.")
    labels.append("The infant exhibits normal breathing with no signs of distress.")
    
    # Case 2: Distress
    img2_path = "data/demo_train/distress_01.jpg"
    Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(img2_path)
    img_paths.append(img2_path)
    prompts.append("Describe the infant's breathing.")
    labels.append("Severe respiratory distress is present, characterized by deep subcostal retractions.")
    
    # 3. Prepare Dataset
    print("Preparing dataset...")
    dataset, collate_fn = trainer.prepare_dataset(img_paths, prompts, labels)
    
    # 4. Run Dry Run Training (1 Epoch, few steps)
    print("Starting dry-run training (this may take a moment to download base model config)...")
    try:
        # We use a very low learning rate and 1 epoch just to test the loop
        trainer.train(dataset, collate_fn, epochs=1, batch_size=1)
        print("\n✅ Fine-tuning dry run completed successfully!")
        print(f"Adapters saved to {output_dir}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        # Identify common errors
        if "Out of memory" in str(e):
            print("Tip: Try reducing batch_size or enabling gradient_checkpointing in VLM_trainer.")

if __name__ == "__main__":
    run_demo()
