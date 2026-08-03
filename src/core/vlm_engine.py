import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import numpy as np

class VLMEngine:
    """Vision Language Model engine using MedGemma (gated) or BLIP (open) for clinical reasoning."""
    
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.models = {} # Lazy load models to save VRAM
        self.processors = {}
        print(f"VLMEngine initialized on {self.device}. Models will be lazy-loaded on demand.")

    def _load_model(self, model_id):
        if model_id not in self.models:
            print(f"Loading VLM model: {model_id}...")
            try:
                if "blip" in model_id.lower():
                    # Open Source Model (BLIP)
                    self.models[model_id] = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device).eval()
                    self.processors[model_id] = BlipProcessor.from_pretrained(model_id)
                else:
                    # Gated Model (PaliGemma)
                    self.models[model_id] = PaliGemmaForConditionalGeneration.from_pretrained(
                        model_id, 
                        torch_dtype=torch.bfloat16,
                        device_map=self.device
                    ).eval()
                    self.processors[model_id] = PaliGemmaProcessor.from_pretrained(model_id)
            except Exception as e:
                print(f"Error loading VLM {model_id}: {e}")
                if "blip" not in model_id.lower():
                    print("Tip: If this is a gated model, run `huggingface-cli login`.")
                self.models[model_id] = None # Mark as failed

    def extract_visual_features(self, frame_rgb: np.ndarray, model_id="Salesforce/blip-image-captioning-base"):
        """
        Extracts visual embedding vector from the VLM's vision encoder.
        Useful for fusing visual context into the time-series model.
        Returns: (1, 768) tensor or None
        """
        self._load_model(model_id)
        model = self.models.get(model_id)
        if model is None: return None
        
        processor = self.processors[model_id]
        image = Image.fromarray(frame_rgb)
        
        try:
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                if "blip" in model_id.lower():
                    # BLIP: Get image features from vision model
                    vision_outputs = model.vision_model(**inputs)
                    embedding = vision_outputs.pooler_output # (1, 768)
                else:
                    # PaliGemma: Get SigLIP embeddings
                    # Usually hidden_states[-1] or similar
                    outputs = model.vision_tower(inputs['pixel_values'])
                    embedding = outputs.last_hidden_state.mean(dim=1) # Global Average Pooling
            return embedding
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def analyze_clinical_signs(self, frame_rgb: np.ndarray, model_id="Salesforce/blip-image-captioning-base"):
        """
        Analyzes a single frame using the specified model.
        Falls back to heuristic analysis if VLM unavailable.
        """
        if model_id is None or model_id == "None":
            return self._heuristic_assessment(frame_rgb)
            
        self._load_model(model_id)
        model = self.models.get(model_id)
        
        if model is None:
            # Fallback: Heuristic-based assessment
            return self._heuristic_assessment(frame_rgb)
            
        processor = self.processors[model_id]
        
        image = Image.fromarray(frame_rgb)
        
        if "blip" in model_id.lower():
            # BLIP VQA Prompt
            prompt = "Question: Is the infant in respiratory distress? Answer:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        else:
            # MedGemma / PaliGemma Prompt
            prompt = (
                "Describe the infant's breathing effort. "
                "Are there visible signs of subcostal retractions, intercostal recession, or nasal flaring? "
                "Provide a clinical assessment of respiratory distress level."
            )
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        try:
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=100)
                
            description = processor.decode(output[0], skip_special_tokens=True)
            # Clean up the output to remove the prompt part if it repeats
            if "blip" not in model_id.lower():
                description = description.replace(prompt, "").strip()
            # For VQA result might just be "yes", "no", "severe"
            
            return {
                "model_id": model_id,
                "assessment": description,
                "distress_detected": "distress" in description.lower() or "retraction" in description.lower() or "yes" in description.lower()
            }
        except Exception as e:
            print(f"VLM inference error: {e}")
            return self._heuristic_assessment(frame_rgb)
    
    def _heuristic_assessment(self, frame_rgb: np.ndarray):
        """
        Fallback heuristic assessment when VLM is unavailable.
        Uses simple image statistics as proxies for clinical signs.
        """
        # Convert to grayscale for analysis
        gray = np.mean(frame_rgb, axis=2)
        
        # Basic heuristics (not medical-grade, just for graceful degradation)
        brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Simple rule: Low brightness + high variance might indicate poor lighting or motion
        assessment_parts = []
        signs_detected = False
        
        if brightness < 100:
            assessment_parts.append("Low lighting conditions detected.")
        
        if brightness_std > 60:
            assessment_parts.append("High motion or contrast variance observed.")
            signs_detected = True
        else:
            assessment_parts.append("Stable visual conditions.")
        
        assessment_parts.append("\n[Note: VLM unavailable - using basic heuristics. For full clinical assessment, authenticate with Hugging Face.]")
        
        return {
            "model_id": "heuristic-fallback",
            "assessment": " ".join(assessment_parts),
            "distress_detected": signs_detected
        }

if __name__ == "__main__":
    # Test block
    print("Testing VLMEngine initialization (Placeholder Base Model)...")
    try:
        engine = VLMEngine()
        # Create a dummy image for testing
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = engine.analyze_clinical_signs(dummy_img)
        print("Analysis Result:", result)
    except Exception as e:
        print(f"Error during test: {e}")
