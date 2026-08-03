import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import os
from PIL import Image

class VLMTrainer:
    """
    Fine-tunes PaliGemma/MedGemma using LoRA (Low-Rank Adaptation).
    Optimized for pediatric respiratory distress detection.
    """
    
    def __init__(self, model_id="google/paligemma-3b-pt-224", output_dir="data/models/medgemma_lora", device=None):
        self.model_id = model_id
        self.output_dir = output_dir
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing VLMTrainer for {model_id} on {self.device}...")
        
        # Load Processor
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        
    def prepare_dataset(self, image_paths, text_prompts, labels):
        """
        Creates a HuggingFace Dataset from lists of paths and text.
        Args:
            image_paths: List of file paths to images.
            text_prompts: List of input prompts (e.g. "Describe breathing effort").
            labels: List of target text responses (e.g. "Significant subcostal retractions present.").
        """
        def collate_fn(examples):
            images = [Image.open(path).convert("RGB") for path in examples['image_path']]
            texts = examples['text_prompt']
            targets = examples['label']
            
            # Process inputs
            model_inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
            
            # Process targets (labels)
            labels_tokens = self.processor(text=targets, return_tensors="pt", padding=True, truncation=True).input_ids
            
            # Replace padding token id with -100 for ignore_index in loss
            labels_tokens[labels_tokens == self.processor.tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels_tokens
            
            return model_inputs

        data = {
            'image_path': image_paths,
            'text_prompt': text_prompts,
            'label': labels
        }
        dataset = Dataset.from_dict(data)
        return dataset, collate_fn

    def train(self, dataset, collate_fn, epochs=3, batch_size=4, learning_rate=2e-4):
        """
        Runs the LoRA fine-tuning loop.
        """
        print("Loading base model...")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16, 
            device_map=self.device
        )
        
        # Configure LoRA
        print("Configuring LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] # Target attention layers
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Training Arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="epoch",
            fp16=False, # PaliGemma prefers bfloat16
            bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
            remove_unused_columns=False
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collate_fn
        )
        
        print("Starting training...")
        trainer.train()
        
        print(f"Saving LoRA adapters to {self.output_dir}...")
        model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        
if __name__ == "__main__":
    # Test stub
    # Create a dummy training run
    trainer = VLMTrainer(output_dir="data/models/test_lora")
    
    # Dummy data
    imgs = ["data/demo_frame.jpg"] * 2 # Placeholder, user needs to provide actual paths
    prompts = ["Describe the breathing."] * 2
    labels = ["Normal breathing, no retractions."] * 2
    
    # Create a dummy image if not exists
    if not os.path.exists("data/demo_frame.jpg"):
        from PIL import Image
        import numpy as np
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img.save("data/demo_frame.jpg")
        
    ds, collate = trainer.prepare_dataset(imgs, prompts, labels)
    
    # We won't actually run train() in the stub to avoid long execution, 
    # but we will print readiness.
    print("VLMTrainer initialized and ready for training data.")
