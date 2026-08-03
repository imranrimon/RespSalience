import sys
import os
import gc
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
import logging

# Path Setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "air-400"
DATA_MODELS = PROJECT_ROOT / "data" / "models"
YOLO_PATH = PROJECT_ROOT / "data" / "yolov8n.pt"
LOCAL_YOLO_DIR = PROJECT_ROOT / ".ultralytics"

LOCAL_YOLO_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_YOLO_DIR))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

# Inject external path
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.append(str(EXTERNAL_ROOT))

# Now import from external
try:
    from infer import build_model, run_inference
    from dataloaders.inference_dataset import InferenceDataset
    from main import set_random_seeds, seed_worker
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"Failed to import external modules: {e}")
    raise


def _available_flow_methods():
    methods = {"farneback", "coarse2fine"}
    optflow = getattr(cv2, "optflow", None)
    if optflow is None:
        return methods

    creators = {
        "tvl1": "createOptFlow_DualTVL1",
        "pca": "createOptFlow_PCAFlow",
        "deep": "createOptFlow_DeepFlow",
    }
    for method_name, creator_name in creators.items():
        if hasattr(optflow, creator_name):
            methods.add(method_name)
    return methods


def _select_flow_method(model_name):
    available_methods = _available_flow_methods()
    preferred_methods = {
        "DeepPhys": ["deep", "farneback"],
        "TSCAN": ["pca", "farneback"],
        "EfficientPhys": ["coarse2fine", "farneback"],
        "VIRENet": ["coarse2fine", "farneback"],
    }.get(model_name, ["farneback"])

    for method_name in preferred_methods:
        if method_name in available_methods:
            return method_name
    return "farneback"


def _checkpoint_candidates(model_name, flow_method):
    suffixes_by_method = {
        "DeepPhys": {
            "deep": ["Deep_Body_best.pth", "Coarse2fine_Body_best.pth"],
            "farneback": ["Farneback_Body_best.pth", "Coarse2fine_Body_best.pth", "Deep_Body_best.pth"],
        },
        "TSCAN": {
            "pca": ["PCA_Body_best.pth", "Coarse2fine_Body_best.pth"],
            "farneback": ["Farneback_Body_best.pth", "Coarse2fine_Body_best.pth", "PCA_Body_best.pth"],
        },
        "EfficientPhys": {
            "coarse2fine": ["Coarse2fine_Body_best.pth", "Farneback_Body_best.pth"],
            "farneback": ["Farneback_Body_best.pth", "Coarse2fine_Body_best.pth"],
        },
        "VIRENet": {
            "coarse2fine": ["Coarse2fine_Body_best.pth", "Farneback_Body_best.pth"],
            "farneback": ["Farneback_Body_best.pth", "Coarse2fine_Body_best.pth"],
        },
    }
    suffixes = suffixes_by_method.get(model_name, {}).get(flow_method, ["Coarse2fine_Body_best.pth"])
    return [f"{model_name}_{suffix}" for suffix in suffixes]


class ExternalBenchmarkError(RuntimeError):
    """Raised when an external baseline cannot produce a valid prediction."""


class ExternalBenchmarkRunner:
    """Wrapper to run DeepPhys and TSCAN from the external repository."""

    def __init__(self, model_name="DeepPhys", checkpoint_path=None, flow_method=None):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.flow_method = flow_method or _select_flow_method(model_name)
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        
        # Base Config Template (derived from deepphys_deep_infer.yaml)
        self.config = {
            "NAME": f"{model_name.lower()}_infer",
            "DATA_PATH": {
                "OUTPUT_DIR": str(PROJECT_ROOT / "data" / "outputs" / "external"),
                "BODY_DETECTOR_PATH": str(YOLO_PATH),
                "FACE_DETECTOR_PATH": str(YOLO_PATH), # Fallback
            },
            "MODEL": {
                "NAME": model_name,
                "DROP_RATE": 0.2,
                "LOSS": "psd_mse",
                "FRAME_DEPTH": 10 if model_name == "DeepPhys" else 20, # TSCAN often uses 20
                "IN_CHANNELS": 3,
            },
            "TEST": {
                "DATA": {
                    "INFANT_FLAG": True,
                    "DATA_FORMAT": "NDCHW",
                    "PREPROCESS": {
                        "DO_DOWNSAMPLE_BEFORE_PREPROCESS": True,
                        "DOWNSAMPLE_SIZE_BEFORE_PREPROCESS": [320, 180],
                        "DATA_NORMALIZE_TYPE": "Standardized",
                        "LABEL_NORMALIZE_TYPE": "DiffNormalized",
                        "FLOW_NORMALIZE_TYPE": "Standardized",
                        "DO_CHUNK": True,
                        "CHUNK_LENGTH": 180,
                        "DO_CROP_INFANT_REGION": True,
                        "DYNAMIC_DETECTION": True,
                        "DYNAMIC_DETECTION_FREQUENCY": 10,
                        "INFANT_REGION": "Body",
                        "BODY_CONF_THRESHOLD": 0.25,
                        "FACE_CONF_THRESHOLD": 0.5,
                        "CHEST_SHIFT_ALPHA": 0.2,
                        "DO_OPTICAL_FLOW": True,
                        "OF_METHOD": self.flow_method,
                        "DO_DOWNSAMPLE_BEFORE_TRAINING": True,
                        "DOWNSAMPLE_SIZE_BEFORE_TRAINING": [96, 96]
                    }
                }
            },
            "INFERENCE": {
                # TSCAN uses FRAME_DEPTH=20 (2x DeepPhys) so its per-chunk tensors
                # are far larger; force batch size 1 to keep it inside memory on
                # long infant videos. Other models keep the reduced batch of 2.
                "BATCH_SIZE": 1 if model_name == "TSCAN" else 2,
                "EVALUATION_METHOD": "FFT",
                "USE_POST_PROCESS": True
            },
            "LOGGING": {"USE_WANDB": False}
        }

        # Adjust for TSCAN
        if model_name == "TSCAN":
             pass # Use standard preprocessing (96x96)
        
        # Setup Logger (Mock)
        self.logger = logging.getLogger(f"External_{model_name}")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
        if self.flow_method != "deep" and model_name == "DeepPhys":
            self.logger.warning("DeepPhys requested DeepFlow, but this OpenCV build does not provide it. Using '%s' instead.", self.flow_method)
        if self.flow_method != "pca" and model_name == "TSCAN":
            self.logger.warning("TSCAN requested PCAFlow, but this OpenCV build does not provide it. Using '%s' instead.", self.flow_method)

        # Build Model
        print(f"[{model_name}] Building model...")
        self.model = build_model(self.config, self.logger)
        
        # Load Checkpoint
        ckpt_path = self.checkpoint_path
        if ckpt_path is None:
            for ckpt_name in _checkpoint_candidates(model_name, self.flow_method):
                candidate = DATA_MODELS / ckpt_name
                if candidate.exists():
                    ckpt_path = candidate
                    break
        elif not ckpt_path.exists():
            raise FileNotFoundError(f"Explicit checkpoint not found for {model_name}: {ckpt_path}")

        if ckpt_path is not None:
            print(f"[{model_name}] Loading weights from {ckpt_path}...")
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
        else:
            print(f"[{model_name}] Warning: No compatible weights found for flow method '{self.flow_method}' in {DATA_MODELS}")

    def run(self, video_path):
        """Runs inference on a single video file."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        dataset = None
        loader = None
        try:
            # Create Dataset
            # InferenceDataset init(config, dataset_name_or_file)
            dataset = InferenceDataset(self.config, video_path)
            
            # Loader
            loader = DataLoader(
                dataset,
                batch_size=self.config['INFERENCE']['BATCH_SIZE'],
                shuffle=False,
                num_workers=0, # Avoid spawning issues in demo
                worker_init_fn=seed_worker
            )
            
            # Run
            metrics = run_inference(self.config, self.model, loader, self.logger)
            pred_rr_bpm = float(metrics.get('pred_rr_bpm', np.nan))
            if not np.isfinite(pred_rr_bpm):
                raise ExternalBenchmarkError(
                    f"{self.model_name} produced an invalid respiration rate for {video_path}: {pred_rr_bpm}"
                )
            return pred_rr_bpm

        except Exception as e:
            self.logger.exception("[%s] Inference Failed for %s", self.model_name, video_path)
            raise ExternalBenchmarkError(
                f"{self.model_name} inference failed for {video_path}: {e}"
            ) from e
        finally:
            del loader
            del dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    # Test
    runner = ExternalBenchmarkRunner("DeepPhys")
    bpm = runner.run("data/AIR_400/S01/1.mp4")
    print(f"DeepPhys BPM: {bpm}")
