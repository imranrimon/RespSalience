"""
Full-Scale Ablation Study Orchestrator
Evaluates all baseline configurations across available datasets
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import gc
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_local_yolo_dir = Path(".ultralytics").resolve()
_local_yolo_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_local_yolo_dir))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")


def _set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.visualize_results import generate_plots
from src.utils.respiratory_ground_truth import load_air400_ground_truth, load_air125_ground_truth, load_pediatric_rr_ground_truth
from src.core.ablation_manager import AblationManager
from src.core.main_controller import RespiratoryController
try:
    from src.core.external_runner import ExternalBenchmarkRunner, ExternalBenchmarkError
except ImportError:
    print("Warning: ExternalBenchmarkRunner not available (dependencies missing?)")
    ExternalBenchmarkRunner = None
    ExternalBenchmarkError = None

try:
    from src.core.hybrid_baseline import HybridFusionOrchestrator
except ImportError:
    print("Warning: HybridFusionOrchestrator not available.")
    HybridFusionOrchestrator = None

try:
    from src.core.thermal_baseline import ThermalSleepOrchestrator
except ImportError:
    print("Warning: ThermalSleepOrchestrator not available.")
    ThermalSleepOrchestrator = None

try:
    from src.core.rgb_baseline import RGBCameraOrchestrator
except ImportError:
    print("Warning: RGBCameraOrchestrator not available.")
    RGBCameraOrchestrator = None

try:
    from src.core.neonatal_baseline import NeonatalMSDOrchestrator
except ImportError:
    print("Warning: NeonatalMSDOrchestrator not available.")
    NeonatalMSDOrchestrator = None

try:
    from src.core.sleep_imu_baseline import SleepIMUOrchestrator
except ImportError:
    print("Warning: SleepIMUOrchestrator not available.")
    SleepIMUOrchestrator = None

try:
    from src.core.rapid_face_baseline import RapidFaceEVMOrchestrator
except ImportError:
    print("Warning: RapidFaceEVMOrchestrator not available.")
    RapidFaceEVMOrchestrator = None

class AblationOrchestrator:
    """Orchestrates comprehensive ablation study experiments."""
    
    def __init__(self, output_dir="data/ablation_results", seed=42):
        _set_global_seed(seed)
        self.seed = seed
        self.disable_vlm = False
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ablation_mgr = AblationManager()
        
        # Experiment configurations (tagged by learning paradigm)
        self.baselines = [
            # ── Unsupervised (Signal Processing) ──
            {"name": "Flow-Only", "paradigm": "Unsupervised", "use_depth": False, "use_vmd": False, "use_keypoint": False, "use_vlm": False},
            {"name": "Depth-Only", "paradigm": "Unsupervised", "use_depth": True, "use_flow": False, "use_vmd": False, "use_keypoint": False, "use_vlm": False},
            {"name": "Fused-SNR", "paradigm": "Unsupervised", "use_depth": True, "use_flow": True, "use_vmd": False, "use_keypoint": False, "use_vlm": False},
            {"name": "Fused-VMD", "paradigm": "Unsupervised", "use_depth": True, "use_flow": True, "use_vmd": True, "use_keypoint": False, "use_vlm": False},
            {"name": "Fused-DL-UNet", "paradigm": "Unsupervised", "use_depth": True, "use_flow": True, "use_dl_denoise": True, "dl_denoiser_type": "unet", "use_keypoint": False, "use_vlm": False},
            {"name": "Fused-DL-Transformer", "paradigm": "Unsupervised", "use_depth": True, "use_flow": True, "use_dl_denoise": True, "dl_denoiser_type": "transformer", "use_keypoint": False, "use_vlm": False},
            {"name": "Fused-DL-WaveNet", "paradigm": "Unsupervised", "use_depth": True, "use_flow": True, "use_dl_denoise": True, "dl_denoiser_type": "wavenet", "use_keypoint": False, "use_vlm": False},
            {"name": "Fused-DL-Diffusion", "paradigm": "Unsupervised", "use_depth": True, "use_flow": True, "use_dl_denoise": True, "dl_denoiser_type": "diffusion", "use_keypoint": False, "use_vlm": False},
            {"name": "Fused-DL-Mamba", "paradigm": "Unsupervised", "use_depth": True, "use_flow": True, "use_dl_denoise": True, "dl_denoiser_type": "mamba", "use_keypoint": False, "use_vlm": False},
            {"name": "Fused-DL-CNN-BiLSTM", "paradigm": "Unsupervised", "use_depth": True, "use_flow": True, "use_dl_denoise": True, "dl_denoiser_type": "cnn-bilstm", "use_keypoint": False, "use_vlm": False},
            {"name": "Fused-DL-TFMAE", "paradigm": "Unsupervised", "use_depth": True, "use_flow": True, "use_dl_denoise": True, "dl_denoiser_type": "tfmae", "use_keypoint": False, "use_vlm": False},
            {"name": "Fused-Keypoint", "paradigm": "Unsupervised", "use_depth": True, "use_flow": True, "use_vmd": False, "use_keypoint": True, "use_vlm": False, "use_supervised": False},
            {"name": "Fused-VMD-Keypoint", "paradigm": "Unsupervised", "use_depth": True, "use_flow": True, "use_vmd": True, "use_keypoint": True, "use_vlm": False, "use_supervised": False},
            
            # ── Self-Supervised (Pretrain → Fine-tune → Evaluate) ──
            # NOTE: These MUST point to the FINETUNED versions (with trained regression heads).
            # Raw SSL weights (ssl_benchmark_*.pth) have UNTRAINED regression heads → output ~0 BPM.
            {"name": "MSM-Finetune", "paradigm": "Self-Supervised", "use_depth": True, "use_flow": True, "use_transformer": True,
             "pretrained": "data/models/finetuned_msm.pth", "use_vlm": False, "use_supervised": False},
            {"name": "SAE-Finetune", "paradigm": "Self-Supervised", "use_depth": True, "use_flow": True, "use_transformer": True,
             "pretrained": "data/models/finetuned_sae.pth", "use_vlm": False, "use_supervised": False},
            {"name": "Contrastive-Finetune", "paradigm": "Self-Supervised", "use_depth": True, "use_flow": True, "use_transformer": True,
             "pretrained": "data/models/finetuned_contrast.pth", "use_vlm": False, "use_supervised": False},
            
            # ── Semi-Supervised (VLM Surrogate Labels) ──
            {"name": "SPMat-BT", "paradigm": "Semi-Supervised", "use_depth": True, "use_flow": True, "use_audio": False, "use_transformer": True,
             "pretrained": "data/models/spmat_bt.pth", "use_vlm": False, "use_supervised": False},
            {"name": "Tri-Modal VLM-Gated", "paradigm": "Semi-Supervised", "use_depth": True, "use_flow": True, "use_audio": False, "use_transformer": True,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False},
            {"name": "Quad-Modal Full (Ours)", "paradigm": "Semi-Supervised", "use_depth": True, "use_flow": True, "use_audio": True, "use_transformer": True,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False},
            {"name": "Quad-Modal Full + Robust Fusion", "paradigm": "Semi-Supervised", "use_depth": True, "use_flow": True, "use_audio": True, "use_transformer": True,
             "use_robust_fusion": True, "robust_disagreement_threshold": 0.4,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False},
            {"name": "Quad-Modal Full + Robust Fusion + Conformal Triage", "paradigm": "Semi-Supervised", "use_depth": True, "use_flow": True, "use_audio": True, "use_transformer": True,
             "use_robust_fusion": True, "robust_disagreement_threshold": 0.4,
             "use_conformal_triage": True, "conformal_condition_on_disagreement": True, "conformal_default_qhat": 6.0,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False},
            {"name": "Quad-Modal-AST", "paradigm": "Semi-Supervised", "use_depth": True, "use_flow": True, "use_audio": True, "use_ast_audio": True, "use_transformer": True,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False},
            {"name": "Quad-Modal-HTSAT", "paradigm": "Semi-Supervised", "use_depth": True, "use_flow": True, "use_audio": True, "use_htsat_audio": True, "use_transformer": True,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False},
            {"name": "Quad-Modal-PANNs", "paradigm": "Semi-Supervised", "use_depth": True, "use_flow": True, "use_audio": True, "use_panns_audio": True, "use_transformer": True,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False},
            {"name": "Quad-Modal-Perceiver", "paradigm": "Semi-Supervised", "use_depth": True, "use_flow": True, "use_audio": True, "use_perceiver": True,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False},
            {"name": "Quad-Modal-VideoMAE", "paradigm": "Semi-Supervised", "use_depth": False, "use_flow": False, "use_audio": True, "use_videomae": True, "use_transformer": True,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False},
            {"name": "Quad-Modal-ImageBind", "paradigm": "Semi-Supervised", "use_depth": True, "use_flow": True, "use_audio": True, "use_imagebind": True,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False},
            
            # ── Supervised ──
            {"name": "Transformer-Fusion", "paradigm": "Supervised", "use_depth": True, "use_flow": True, "use_audio": False, "use_transformer": True, "use_mamba": False, "use_vlm": False, "use_supervised": False},
            {"name": "Mamba-Fusion-Full", "paradigm": "Supervised", "use_depth": True, "use_flow": True, "use_audio": True, "use_transformer": False, "use_mamba": True, "use_vlm": False, "use_supervised": False},
            {"name": "Legacy-Supervised (XGB)", "paradigm": "Supervised", "use_depth": True, "use_flow": True, "use_audio": False, "use_vmd": True, "use_keypoint": True, "use_vlm": True, "use_supervised": True},
            
            # ── Robustness Testing (Missing Modality) ──
            {"name": "Quad-Modal-Full (Drop Audio)", "paradigm": "Robustness", "use_depth": True, "use_flow": True, "use_audio": True, "use_transformer": True,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False, "drop_modality": "audio"},
            {"name": "Quad-Modal-Full (Drop Flow)", "paradigm": "Robustness", "use_depth": True, "use_flow": True, "use_audio": True, "use_transformer": True,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False, "drop_modality": "flow"},
            {"name": "Quad-Modal-Full (Drop Depth)", "paradigm": "Robustness", "use_depth": True, "use_flow": True, "use_audio": True, "use_transformer": True,
             "pretrained": "data/models/vlm_gated_fused.pth", "use_vlm": True, "use_supervised": False, "drop_modality": "depth"},
        ]
        
        # Add External Baselines if available
        if ExternalBenchmarkRunner:
            self.baselines.append({"name": "DeepPhys", "paradigm": "Supervised (External)", "runner": "external", "model_name": "DeepPhys", "use_vlm": False})
            self.baselines.append({"name": "TSCAN", "paradigm": "Supervised (External)", "runner": "external", "model_name": "TSCAN", "use_vlm": False})
            self.baselines.append({"name": "DeepPhys-Official", "paradigm": "Supervised (Official AIR)", "runner": "external", "model_name": "DeepPhys", "use_vlm": False})
            self.baselines.append({"name": "MTTS-CAN-Official", "paradigm": "Supervised (Official AIR)", "runner": "external", "model_name": "TSCAN", "use_vlm": False})
            self.baselines.append({"name": "EfficientPhys-Official", "paradigm": "Supervised (Official AIR)", "runner": "external", "model_name": "EfficientPhys", "use_vlm": False})
            self.baselines.append({"name": "AIRFlowNet-Official", "paradigm": "Supervised (Official AIR)", "runner": "external", "model_name": "VIRENet", "use_vlm": False})
            
        if HybridFusionOrchestrator:
            self.baselines.append({"name": "Hybrid-CNN-Spectral (PLoS 2026)", "paradigm": "Literature Baseline (External)", "runner": "hybrid_baseline", "use_vlm": False})
            
        if ThermalSleepOrchestrator:
            self.baselines.append({"name": "Thermal-Sleep-Baseline (Jakkaew 2020)", "paradigm": "Literature Baseline (External)", "runner": "thermal_baseline", "use_vlm": False})

        if RGBCameraOrchestrator:
            self.baselines.append({"name": "RGB-Camera-Baseline (Romano 2021)", "paradigm": "Literature Baseline (External)", "runner": "rgb_baseline", "use_vlm": False})

        if NeonatalMSDOrchestrator:
            self.baselines.append({"name": "Neonatal-MSD-Baseline (Rossol 2020)", "paradigm": "Literature Baseline (External)", "runner": "neonatal_baseline", "use_vlm": False})

        if SleepIMUOrchestrator:
            self.baselines.append({"name": "Smart-Bedding-IMU-Baseline (Mejia 2026)", "paradigm": "Literature Baseline (External)", "runner": "sleep_imu", "use_vlm": False})

        if RapidFaceEVMOrchestrator:
            self.baselines.append({"name": "Rapid-Face-EVM-Baseline (Khan 2026)", "paradigm": "Literature Baseline (External)", "runner": "rapid_face", "use_vlm": False})
        
    def discover_datasets(self):
        """Scans data directory for available video datasets."""
        datasets = []
        
        # AIR-400 videos (in subject subdirectories)
        air400_dir = Path("data/AIR_400")
        if air400_dir.exists():
            # Scan each subject directory (S01, S02, ...)
            for subj_dir in sorted(air400_dir.glob("S*")):
                if not subj_dir.is_dir():
                    continue
                # Find all MP4 videos in this subject
                for video in sorted(subj_dir.glob("*.mp4")):
                    datasets.append({
                        "path": str(video),
                        "source": "AIR-400",
                        "subject": f"{subj_dir.name}/{video.stem}"
                    })
        
        # AIR-125 videos (in subject subdirectories S01, S02, ...)
        air125_dir = Path("data/AIR_125")
        if air125_dir.exists():
            for s_id in range(1, 20): 
                subj_folder = air125_dir / f"S{s_id:02d}"
                if subj_folder.exists():
                    for vid in subj_folder.glob("*.mp4"):
                        datasets.append({
                            "path": str(vid),
                            "source": "AIR-125",
                            "subject": f"AIR125-{subj_folder.name}-{vid.stem}"
                        })

        # MINI-RGBD (Image Sequences)
        mini_rgbd_dir = Path("data/MINI-RGBD/MINI-RGBD_web")
        if mini_rgbd_dir.exists():
            for i in range(1, 13): # 01 to 12
                seq_dir = mini_rgbd_dir / f"{i:02d}" / "rgb"
                if seq_dir.exists():
                    # Construct OpenCV image sequence path
                    # Using forward slashes for compatibility
                    seq_path = str(seq_dir).replace("\\", "/") + "/syn_%05d.png"
                    datasets.append({
                        "path": seq_path,
                        "source": "MINI-RGBD",
                        "subject": f"Syn-Infant-{i:02d}"
                    })
        
        # Pediatric-RR-332 (3 hospital sites)
        ped_rr_dir = Path("data/Video Dataset for Respiratory Rate Counting in Children/Video Files")
        if ped_rr_dir.exists():
            for site_dir in sorted(ped_rr_dir.iterdir()):
                if not site_dir.is_dir():
                    continue
                for id_dir in sorted(site_dir.iterdir()):
                    if not id_dir.is_dir() or not id_dir.name.startswith("ID-"):
                        continue
                    for vid in id_dir.glob("*.mp4"):
                        datasets.append({
                            "path": str(vid),
                            "source": "Pediatric-RR",
                            "subject": f"{id_dir.name}"
                        })
            
        print(f"Discovered {len(datasets)} videos across datasets.")
        return datasets

    @staticmethod
    def load_manifest_datasets(manifest_path):
        """Load canonical benchmark entries directly from a manifest JSON."""
        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        def _resolve_video_path(entry):
            """Resolve the video path robustly.

            Manifests store an absolute ``full_path`` that becomes stale if the
            repo moves (e.g. drive letter change). Fall back to ``relative_path``
            resolved against the ``data/`` root so benchmarks self-heal instead of
            silently feeding a missing path (which yields constant-fallback BPMs).
            """
            full_path = str(entry.get("full_path", "") or "")
            if full_path and os.path.exists(full_path):
                return full_path
            rel = str(entry.get("relative_path", "") or "").replace("\\", "/")
            if rel:
                candidate = os.path.join("data", rel)
                if os.path.exists(candidate):
                    return os.path.abspath(candidate)
            return ""

        datasets = []
        skipped = 0
        for entry in manifest:
            resolved = _resolve_video_path(entry)
            if not resolved:
                skipped += 1
                continue
            subject_id = str(entry.get("subject_id", "unknown_subject"))
            video_id = str(entry.get("video_id", Path(resolved).stem))
            datasets.append(
                {
                    "path": resolved,
                    "source": str(entry.get("dataset", "unknown")),
                    "subject": f"{subject_id}/{video_id}",
                    "subject_id": subject_id,
                    "video_id": video_id,
                    "gt_rr_bpm": entry.get("gt_rr_bpm"),
                    "age_months": entry.get("age_months", 6),
                }
            )

        print(f"Loaded {len(datasets)} videos from manifest: {manifest_file}"
              + (f" ({skipped} skipped: video not found)" if skipped else ""))
        return datasets
    
    def run_evaluation(self, limit=None, dataset_filter=None, manifest_path=None, explicit_datasets=None, generate_plots_flag=True):
        """
        Runs evaluation across all baselines and datasets.
        Args:
            limit: Maximum number of videos to process (for testing)
            dataset_filter: Filter for dataset source (e.g. "AIR-400")
        """
        from src.utils.respiratory_ground_truth import load_air400_ground_truth
        
        if explicit_datasets is not None:
            datasets = list(explicit_datasets)
        elif manifest_path:
            datasets = self.load_manifest_datasets(manifest_path)
        else:
            datasets = self.discover_datasets()
        
        if dataset_filter:
            datasets = [d for d in datasets if dataset_filter.lower() in d['source'].lower()]
            print(f"Filtered datasets to '{dataset_filter}': {len(datasets)} videos found.")
            
        if limit:
            datasets = datasets[:limit]
            print(f"Limiting to {limit} videos for quick test.")
            
        # Create directory for signals if needed
        self.signal_dir = self.output_dir / "signals"
        if hasattr(self, 'save_signals') and self.save_signals:
            self.signal_dir.mkdir(parents=True, exist_ok=True)
            
            # Optimization: When saving signals without a specific baseline requested,
            # only run ONE fast classical baseline (Fused-VMD preferred, else first Unsupervised).
            # Running untrained DL models here is a waste of time.
            if len(self.baselines) > 1 and not getattr(self, '_baseline_explicitly_set', False):
                print("\n[Optimization] --save-signals active. Using single classical baseline for extraction.")
                # Prefer Fused-VMD; fall back to first Unsupervised baseline
                signal_baseline = next(
                    (b for b in self.baselines if b['name'] == 'Fused-VMD'),
                    next((b for b in self.baselines if b.get('paradigm') == 'Unsupervised'), None)
                )
                if signal_baseline:
                    print(f"  -> Using baseline: '{signal_baseline['name']}'")
                    self.baselines = [signal_baseline]
                else:
                    print("  -> No classical baseline found; running all.")
        
        total_exps = len(self.baselines) * len(datasets)
        current = 0
        
        print(f"\n{'='*60}")
        print(f"Starting Ablation Study: {len(self.baselines)} baselines × {len(datasets)} videos")
        print(f"{'='*60}\n")
        
        for baseline in self.baselines:
            # Clear VRAM before starting new baseline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # CLEANUP PREVIOUS RESOURCES
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"\n--- Baseline: {baseline['name']} ---")
            
            # Initialize controller with this baseline's configuration
            controller = RespiratoryController(
                "src/config/test_infer.yaml",
                "data/models/VIRENet_Coarse2fine_Body_best.pth",
                enable_depth=baseline.get('use_depth', True),
                enable_flow=baseline.get('use_flow', True),
                enable_audio=baseline.get('use_audio', False),
                enable_vmd=baseline.get('use_vmd', False),
                enable_dl_denoise=baseline.get('use_dl_denoise', False),
                dl_denoiser_type=baseline.get('dl_denoiser_type', 'unet'),
                enable_keypoint=baseline.get('use_keypoint', True),
                enable_transformer=baseline.get('use_transformer', False),
                enable_mamba=baseline.get('use_mamba', False),
                enable_perceiver=baseline.get('use_perceiver', False),
                enable_imagebind=baseline.get('use_imagebind', False),
                enable_ast_audio=baseline.get('use_ast_audio', False),
                enable_htsat_audio=baseline.get('use_htsat_audio', False),
                enable_panns_audio=baseline.get('use_panns_audio', False),
                enable_videomae=baseline.get('use_videomae', False),
                enable_supervised=baseline.get('use_supervised', False),
                enable_robust_fusion=baseline.get('use_robust_fusion', False),
                robust_disagreement_threshold=baseline.get('robust_disagreement_threshold', 0.4),
                enable_conformal_triage=baseline.get('use_conformal_triage', False),
                conformal_calibration_csv=baseline.get('conformal_calibration_csv', None),
                conformal_default_qhat=baseline.get('conformal_default_qhat', 6.0),
                conformal_condition_on_disagreement=baseline.get('conformal_condition_on_disagreement', False),
                fusion_model_path=baseline.get('pretrained', None)
            ) if baseline.get("runner") != "external" else None
            
            # Setup External Runner
            ext_runner = None
            hybrid_runner = None
            thermal_runner = None
            rgb_runner = None
            neonatal_runner = None
            sleep_runner = None
            rapid_face_runner = None
            if baseline.get("runner") == "external":
                print(f"Initializing External Model: {baseline['model_name']}...")
                ext_runner = ExternalBenchmarkRunner(
                    baseline['model_name'],
                    checkpoint_path=baseline.get('checkpoint') or baseline.get('pretrained')
                )
            elif baseline.get("runner") == "hybrid_baseline":
                print("Initializing Hybrid CNN-Spectral Baseline (PLoS 2026)...")
                hybrid_runner = HybridFusionOrchestrator(cnn_model_path="data/models/baselines/hybrid_cnn_best.pth", fps=30)
            elif baseline.get("runner") == "thermal_baseline":
                print("Initializing Thermal Sleep Baseline (Jakkaew 2020)...")
                thermal_runner = ThermalSleepOrchestrator(fps=30)
            elif baseline.get("runner") == "rgb_baseline":
                print("Initializing RGB Camera Baseline (Romano 2021)...")
                rgb_runner = RGBCameraOrchestrator(fps=30)
            elif baseline.get("runner") == "neonatal_baseline":
                print("Initializing Neonatal MSD Baseline (Rossol 2020)...")
                neonatal_runner = NeonatalMSDOrchestrator(fps=30, block_size=16)
            elif baseline.get("runner") == "sleep_imu":
                print("Initializing Smart-Bedding IMU Baseline (Mejia 2026)...")
                sleep_runner = SleepIMUOrchestrator(fps=30)
            elif baseline.get("runner") == "rapid_face":
                print("Initializing Rapid-Face EVM Baseline (Khan 2026)...")
                rapid_face_runner = RapidFaceEVMOrchestrator(fps=30)
            
            for dataset in datasets:
                current += 1
                
                # Check for Resume (Phase 13 Optimization)
                if getattr(self, 'save_signals', False):
                    sig_path = self.signal_dir / f"{dataset['source']}_{dataset['subject'].replace('/','_')}_{baseline['name']}.npz"
                    if sig_path.exists():
                        print(f"[{current}/{total_exps}] Skipping {dataset['subject']} (Already processed: {sig_path.name})")
                        continue
                        
                print(f"[{current}/{total_exps}] Processing {dataset['subject']}...")
                
                # Load ground truth based on source
                gt_bpm = dataset.get('gt_rr_bpm', None)
                age_months = int(dataset.get('age_months', 6) or 6)
                if gt_bpm is None and dataset['source'] == 'AIR-400':
                    gt_bpm = load_air400_ground_truth(dataset['path'])
                elif gt_bpm is None and dataset['source'] == 'AIR-125':
                    gt_bpm = load_air125_ground_truth(dataset['path'])
                elif gt_bpm is None and dataset['source'] == 'Pediatric-RR':
                    ped_gt = load_pediatric_rr_ground_truth(dataset['path'])
                    if ped_gt:
                        gt_bpm = ped_gt['respiratory_rate']
                        age_months = ped_gt['age_months']
                
                # Validity check (Require GT for benchmarking, skip if missing)
                if dataset['source'] in ['AIR-400', 'AIR-125', 'Pediatric-RR'] and gt_bpm is None:
                    print(f"  ⚠ No GT found for {dataset['source']}, skipping...")
                    continue
                
                try:
                    # Run pipeline or external runner
                    if ext_runner:
                        bpm = ext_runner.run(dataset['path'])
                        result = {'bpm': bpm, 'confidence': 1.0}
                    elif hybrid_runner:
                        # Extract frames on the fly for the multi-ROI extractor
                        import cv2
                        cap = cv2.VideoCapture(dataset['path'])
                        frames = []
                        target_width = 320 # Resize to speed up dense flow
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            h, w = frame.shape[:2]
                            if w > target_width:
                                new_h = int(h * (target_width / w))
                                frame = cv2.resize(frame, (target_width, new_h))
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        cap.release()
                        
                        bpm = hybrid_runner.run(frames)
                        result = {'bpm': bpm, 'confidence': 1.0, 'vlm_assessment': 'N/A', 'vlm_distress': False}
                    elif thermal_runner:
                        # Extract frames identically for baseline consistency
                        import cv2
                        cap = cv2.VideoCapture(dataset['path'])
                        frames = []
                        target_width = 320 
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            h, w = frame.shape[:2]
                            if w > target_width:
                                new_h = int(h * (target_width / w))
                                frame = cv2.resize(frame, (target_width, new_h))
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        cap.release()
                        
                        bpm, movements = thermal_runner.run(frames)
                        result = {'bpm': bpm, 'confidence': 1.0, 'vlm_assessment': f'Movements Detected: {np.sum(movements)}', 'vlm_distress': False}
                    elif rgb_runner:
                        import cv2
                        cap = cv2.VideoCapture(dataset['path'])
                        frames = []
                        target_width = 320 
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            h, w = frame.shape[:2]
                            if w > target_width:
                                new_h = int(h * (target_width / w))
                                frame = cv2.resize(frame, (target_width, new_h))
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        cap.release()
                        
                        bpm, artifacts, apnea = rgb_runner.run(frames)
                        apn_str = 'Detected' if apnea else 'None'
                        result = {'bpm': bpm, 'confidence': 1.0, 'vlm_assessment': f'Artifacts: {np.sum(artifacts)} | Apnea: {apn_str}', 'vlm_distress': apnea}
                    elif neonatal_runner:
                        import cv2
                        cap = cv2.VideoCapture(dataset['path'])
                        frames = []
                        target_width = 320 
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            h, w = frame.shape[:2]
                            if w > target_width:
                                new_h = int(h * (target_width / w))
                                frame = cv2.resize(frame, (target_width, new_h))
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        cap.release()
                        
                        bpm, artifacts, apnea = neonatal_runner.run(frames)
                        apn_str = 'Detected' if apnea else 'None'
                        result = {'bpm': bpm, 'confidence': 1.0, 'vlm_assessment': f'Artifacts (Macro): {np.sum(artifacts)} | Apnea: {apn_str}', 'vlm_distress': apnea}
                    elif sleep_runner:
                        import cv2
                        cap = cv2.VideoCapture(dataset['path'])
                        frames = []
                        target_width = 320 
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            h, w = frame.shape[:2]
                            if w > target_width:
                                new_h = int(h * (target_width / w))
                                frame = cv2.resize(frame, (target_width, new_h))
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        cap.release()
                        
                        bpm, artifacts, apnea = sleep_runner.run(frames)
                        apn_str = 'Detected' if apnea else 'None'
                        result = {'bpm': bpm, 'confidence': 1.0, 'vlm_assessment': f'Macro Artifacts: {np.sum(artifacts)} | Apnea: {apn_str}', 'vlm_distress': apnea}
                    elif rapid_face_runner:
                        import cv2
                        cap = cv2.VideoCapture(dataset['path'])
                        frames = []
                        target_width = 320 
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret: break
                            h, w = frame.shape[:2]
                            if w > target_width:
                                new_h = int(h * (target_width / w))
                                frame = cv2.resize(frame, (target_width, new_h))
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        cap.release()
                        
                        bpm, artifacts, apnea = rapid_face_runner.run(frames)
                        result = {'bpm': bpm, 'confidence': 1.0, 'vlm_assessment': 'Analyzed via Face-EVM & Shoulder Flow.', 'vlm_distress': False}
                    else:
                        use_live_vlm = baseline.get('use_vlm') and (not self.disable_vlm)
                        result = controller.process_video(
                            dataset['path'],
                            age_months=age_months,
                            vlm_model_id="Salesforce/blip-image-captioning-base" if use_live_vlm else None,
                            drop_modality=baseline.get('drop_modality', None),
                            baseline_name=baseline.get('name'),
                            baseline_paradigm=baseline.get('paradigm'),
                        )
                    
                    # Log result ONLY if we have ground truth
                    if gt_bpm is not None:
                        # Extract features for supervised training
                        features = {k: v for k, v in result.items() if k not in ['roi', 'depth_signal', 'flow_signal']}
                        
                        self.ablation_mgr.log_experiment(
                            name=baseline['name'],
                            config={
                                "source": dataset['source'],
                                "subject": dataset['subject'],
                                **baseline
                            },
                            pred_bpm=result['bpm'],
                            gt_bpm=gt_bpm,
                            **features # Pass flow_bpm, depth_bpm, confidences
                        )
                        
                        # Save raw signals for training (Phase 13)
                        if getattr(self, 'save_signals', False) and 'depth_signal' in result:
                            sig_path = self.signal_dir / f"{dataset['source']}_{dataset['subject'].replace('/','_')}_{baseline['name']}.npz"
                            np.savez_compressed(sig_path, 
                                                depth=result['depth_signal'], 
                                                flow=result.get('flow_signal', []),
                                                audio=result.get('audio_signal', []),
                                                gt=gt_bpm,
                                                vlm_label=result.get('vlm_distress', False),
                                                vlm_text=result.get('vlm_assessment', "N/A"))
                            
                        print(f"  ✓ Pred: {result['bpm']:.1f} BPM | GT: {gt_bpm:.1f} BPM | Error: {abs(result['bpm'] - gt_bpm):.1f}")
                    else:
                        print(f"  ✓ Pred: {result['bpm']:.1f} BPM | GT: N/A (Qualitative Only)")
                    
                except Exception as e:
                    if ExternalBenchmarkError and isinstance(e, ExternalBenchmarkError):
                        print(f"  ⚠ Skipping failed external inference: {e}")
                    else:
                        print(f"  ✗ Error: {e}")
                        import traceback
                        traceback.print_exc()
                    continue
            
            # Cleanup Controller/Runner for this baseline
            if 'controller' in locals() and controller:
                del controller
            if 'ext_runner' in locals() and ext_runner:
                del ext_runner
            if 'hybrid_runner' in locals() and hybrid_runner:
                del hybrid_runner
            if 'thermal_runner' in locals() and thermal_runner:
                del thermal_runner
            if 'rgb_runner' in locals() and rgb_runner:
                del rgb_runner
            if 'neonatal_runner' in locals() and neonatal_runner:
                del neonatal_runner
            if 'sleep_runner' in locals() and sleep_runner:
                del sleep_runner
            if 'rapid_face_runner' in locals() and rapid_face_runner:
                del rapid_face_runner
            gc.collect()
            torch.cuda.empty_cache()
        
        # Generate and save report
        print(f"\n{'='*60}")
        print("Generating Report...")
        print(f"{'='*60}\n")
        
        report = self.ablation_mgr.generate_report()
        print(report)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"ablation_report_{timestamp}.csv"
        report.to_csv(report_path)
        print(f"\n✓ Report saved to: {report_path}")
        
        # Save raw results
        results_path = self.output_dir / f"ablation_raw_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(self.ablation_mgr.results, f, indent=2)
        print(f"✓ Raw results saved to: {results_path}")
        
        # Generate plots
        if generate_plots_flag:
            print(f"\n{'='*60}")
            print("Generating Performance Plots...")
            generate_plots()
        
        return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, help="Filter specific baseline (e.g., 'DeepPhys')")
    parser.add_argument("--dataset", type=str, help="Filter specific dataset (e.g., 'AIR-125')")
    parser.add_argument("--manifest", type=str, help="Optional canonical manifest JSON to evaluate instead of raw discovery")
    parser.add_argument("--limit", type=int, help="Limit number of videos per dataset")
    parser.add_argument("--save-signals", action="store_true", help="Save raw signals for deep learning training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--disable-vlm", action="store_true", help="Disable live VLM inference during evaluation")
    parser.add_argument("--skip-plots", action="store_true", help="Skip global plot regeneration at the end of the run")
    args = parser.parse_args()

    orchestrator = AblationOrchestrator(seed=args.seed)
    orchestrator.save_signals = args.save_signals
    orchestrator.disable_vlm = args.disable_vlm
    
    if args.baseline:
        # Exact-match first; fall back to case-insensitive substring if nothing found
        exact = [b for b in orchestrator.baselines if b['name'].lower() == args.baseline.lower()]
        matched = exact if exact else [b for b in orchestrator.baselines if args.baseline.lower() in b['name'].lower()]
        orchestrator.baselines = matched
        orchestrator._baseline_explicitly_set = True  # Skip save-signals optimization
        print(f"Running Only Baselines: {[b['name'] for b in orchestrator.baselines]}")
        if not orchestrator.baselines:
            print(f"ERROR: No baseline matched '{args.baseline}'.")
            print(f"Available: {[b['name'] for b in AblationOrchestrator().baselines]}")
            import sys; sys.exit(1)

    # Full-scale evaluation
    print("="*60)
    print("FULL-SCALE ABLATION STUDY")
    if args.dataset:
        print(f"Dataset Filter: {args.dataset}")
    print("Running selected baselines...")
    print("="*60)
    
    orchestrator.run_evaluation(
        limit=args.limit,
        dataset_filter=args.dataset,
        manifest_path=args.manifest,
        generate_plots_flag=not args.skip_plots,
    )
