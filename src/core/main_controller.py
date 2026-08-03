import sys
import os

# Add project root and external repos to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "external/air-400"))

import torch
import cv2
import yaml
import numpy as np
from processors.pre_processor import PreProcessor
from processors.post_processor import PostProcessor
from models.vire_net import VIRENet
from src.core.depth_engine import DepthEngine
from src.core.diagnosis import ARIDiagnostic
from src.core.vlm_engine import VLMEngine
from src.core.anatomical_stabilizer import AnatomicalStabilizer
from src.core.signal_processor import SignalProcessor
from src.core.multimodal_fusion_models import TemporalFusionTransformer, MambaFusionModel
from src.core.perceiver_fusion import PerceiverFusionModel
from src.core.imagebind_fusion import ImageBindLiteFusion
from src.core.supervised_fusion import SupervisedFusion
from src.core.tfmae_denoiser import TFMAE_Denoiser
from src.core.dl_denoiser import ResUNet1D, WaveNetAutoencoder, TransformerAutoencoder1D, ConditionalDiffusion1D, CNN_BiLSTM_Denoiser, MambaDenoiser
from src.core.video_encoders import VideoMAE_3D_Encoder
from src.core.few_shot_learning import FewShotAdaptationModule, DomainShiftDetector, create_few_shot_adapter
from src.core.personalized_baselines import BaselineTracker, AnomalyDetectorPersonalized, create_personalized_baseline_system
from src.core.comorbidity_etiology import DifferentialDiagnosisEngine, RiskProfiler, create_comorbidity_and_etiology_system
from src.core.modality_reliability import RobustFusionTransformer
from src.core.selective_triage import ConformalTriageHead, DisagreementConditionedConformal

class RespiratoryController:
    """Main controller to integrate VIRENet (Flow) and DepthEngine."""
    
    def __init__(self, config_path, checkpoint_path, 
                 enable_depth=True, enable_flow=True, enable_audio=True, enable_vmd=False, 
                 enable_dl_denoise=False, dl_denoiser_type="unet", enable_keypoint=True, 
                 enable_transformer=False, enable_mamba=False, enable_supervised=False, fusion_model_path=None,
                 enable_ast_audio=False, enable_htsat_audio=False, enable_panns_audio=False,
                 enable_perceiver=False, enable_imagebind=False, enable_videomae=False,
                 enable_few_shot_learning=False, enable_personalized_baselines=False, 
                 enable_comorbidity_etiology=False, enable_robust_fusion=False,
                 robust_disagreement_threshold=0.4,
                 enable_conformal_triage=False,
                 conformal_calibration_csv=None,
                 conformal_default_qhat=6.0,
                 conformal_condition_on_disagreement=False):
        """
        Initialize controller with configurable baseline components.
        Args:
            enable_depth: Use depth-based respiration estimation
            enable_flow: Use optical flow-based respiration estimation
            enable_audio: Use audio/spectrogram analysis
            enable_vmd: Apply VMD denoising to signals
            enable_keypoint: Use anatomical keypoint tracking for ROI
            enable_transformer: Use transformer fusion (requires trained weights)
            fusion_model_path: Path to trained fusion transformer weights
        """
        self.enable_depth = enable_depth
        self.enable_flow = enable_flow
        self.enable_audio = enable_audio
        self.enable_ast_audio = enable_ast_audio
        self.enable_htsat_audio = enable_htsat_audio
        self.enable_panns_audio = enable_panns_audio
        self.enable_vmd = enable_vmd
        self.enable_dl_denoise = enable_dl_denoise
        self.dl_denoiser_type = dl_denoiser_type
        self.enable_keypoint = enable_keypoint  
        self.enable_transformer = enable_transformer
        self.enable_mamba = enable_mamba
        self.enable_perceiver = enable_perceiver
        self.enable_imagebind = enable_imagebind
        self.enable_supervised = enable_supervised
        self.enable_videomae = enable_videomae
        self.enable_few_shot_learning = enable_few_shot_learning
        self.enable_personalized_baselines = enable_personalized_baselines
        self.enable_comorbidity_etiology = enable_comorbidity_etiology
        self.enable_robust_fusion = enable_robust_fusion
        self.robust_disagreement_threshold = robust_disagreement_threshold
        self.robust_fusion_model = None
        self.enable_conformal_triage = enable_conformal_triage
        self.conformal_condition_on_disagreement = conformal_condition_on_disagreement
        self.conformal_triage_head = ConformalTriageHead()
        self.conformal_qhat_selector = DisagreementConditionedConformal(
            calibration_csv=conformal_calibration_csv,
            default_qhat=conformal_default_qhat,
            disagreement_bins=(0.2, 0.4),
        )
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load VIRENet
        self.model = VIRENet(img_size=96).to(self.device) # Based on config
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        
        # Load DepthEngine
        self.depth_engine = DepthEngine(device=self.device)
        
        # Load VLMEngine (Optional/Lazy load to save VRAM if needed)
        self.vlm_engine = VLMEngine(device=self.device)
        
        # Load AnatomicalStabilizer
        self.stabilizer = AnatomicalStabilizer()
        
        if self.enable_transformer or self.enable_perceiver or self.enable_imagebind:
            # Must match training config (num_layers=4)
            # Check vlm_dim for VLM-Gated model
            vlm_dim = 2048 if fusion_model_path and "vlm_gated" in fusion_model_path else 0
            
            if self.enable_perceiver:
                self.fusion_model = PerceiverFusionModel(num_layers=4, vlm_dim=vlm_dim, use_ast_audio=self.enable_ast_audio)
            elif self.enable_imagebind:
                self.fusion_model = ImageBindLiteFusion(num_layers=4, vlm_dim=vlm_dim, use_ast_audio=self.enable_ast_audio)
            else:
                self.fusion_model = TemporalFusionTransformer(num_layers=4, vlm_dim=vlm_dim, 
                                                              use_ast_audio=self.enable_ast_audio,
                                                              use_htsat_audio=self.enable_htsat_audio,
                                                              use_panns_audio=self.enable_panns_audio)
            
            ckpt_path = fusion_model_path if fusion_model_path else "data/models/finetuned_model.pth"
            print(f"[RespiratoryController] Initializing Fusion Model from: {ckpt_path}")
            
            if os.path.exists(ckpt_path):
                # Load with strict=False to allow for vlm_proj mismatch if switching models
                state = torch.load(ckpt_path, map_location=self.device)
                if 'model' in state: state = state['model'] # Handle nested dict
                
                # Handling VLM dimension mismatch:
                # If current model has no VLM (vlm_dim=0) but checkpoint has vlm_proj, remove it.
                if vlm_dim == 0:
                    keys_to_remove = [k for k in state.keys() if 'vlm_proj' in k]
                    if keys_to_remove:
                        print(f"[RespiratoryController] Removing unused VLM keys for this baseline: {keys_to_remove}")
                        for k in keys_to_remove:
                            del state[k]

                try:
                    self.fusion_model.load_state_dict(state, strict=False)
                    print(f"[FusionTransformer] Loaded weights from {ckpt_path}")
                except RuntimeError as e:
                    print(f"[FusionTransformer] Error loading weights: {e}")
                    print("Attempting to load with strict=False and ignoring mismatch...")
                    # If specific keys fail, we might need more aggressive filtering, 
                    # but usually removing vlm_proj is enough for this specific error.
                    raise e
            else:
                print(f"[FusionTransformer] Warning: {ckpt_path} not found. Using random init.")
            self.fusion_model.to(self.device)
            
            # Challenge #2: Cross-Modal Anomaly Detection (robust fallback on conflict)
            if self.enable_robust_fusion:
                if self.enable_transformer and isinstance(self.fusion_model, TemporalFusionTransformer):
                    print("[RespiratoryController] Initializing Robust Fusion wrapper...")
                    self.robust_fusion_model = RobustFusionTransformer(
                        base_transformer=self.fusion_model,
                        d_model=self.fusion_model.d_model,
                        threshold_disagreement=self.robust_disagreement_threshold
                    ).to(self.device)
                    self.robust_fusion_model.eval()
                else:
                    print("[RespiratoryController] Warning: enable_robust_fusion requires TemporalFusionTransformer. Disabling robust fusion.")
                    self.enable_robust_fusion = False
        
        self.mamba_model = None
        if self.enable_mamba:
            print("[RespiratoryController] Initializing Mamba-1D SSM Model...")
            self.mamba_model = MambaFusionModel(
                use_ast_audio=self.enable_ast_audio,
                use_htsat_audio=self.enable_htsat_audio,
                use_panns_audio=self.enable_panns_audio
            ).to(self.device)
            # Weights loading logic could go here if pre-trained
            self.mamba_model.eval()

        # Load SupervisedFusion (Phase 12b - Regressor)
        self.supervised_model = SupervisedFusion()
        
        # Load Preprocessor
        self.preprocessor = PreProcessor(
            self.config['TEST']['DATA'][0]['PREPROCESS'],
            "data/yolov8m.pt",
            "data/yolov8n-face.pt"
        )
        self.post_processor = PostProcessor()
        
        # Load DL Denoiser (Phase 18)
        self.dl_denoiser = None
        
        # Load VideoMAE (Phase 20)
        self.videomae_encoder = None
        if self.enable_videomae:
            print("[RespiratoryController] Initializing VideoMAE 3D Encoder...")
            self.videomae_encoder = VideoMAE_3D_Encoder().to(self.device)
            self.videomae_encoder.eval()
            
        if self.enable_dl_denoise:
            if self.dl_denoiser_type == "unet":
                self.dl_denoiser = ResUNet1D().to(self.device)
            elif self.dl_denoiser_type == "transformer":
                self.dl_denoiser = TransformerAutoencoder1D().to(self.device)
            elif self.dl_denoiser_type == "wavenet":
                self.dl_denoiser = WaveNetAutoencoder().to(self.device)
            elif self.dl_denoiser_type == "diffusion":
                self.dl_denoiser = ConditionalDiffusion1D(timesteps=50).to(self.device)
            elif self.dl_denoiser_type == "cnn-bilstm":
                self.dl_denoiser = CNN_BiLSTM_Denoiser().to(self.device)
            elif self.dl_denoiser_type == "tfmae":
                self.dl_denoiser = TFMAE_Denoiser().to(self.device)
            elif self.dl_denoiser_type == "mamba":
                self.dl_denoiser = MambaDenoiser().to(self.device)
            else:
                raise ValueError(f"Unknown DL denoiser type: {self.dl_denoiser_type}")
                
            denoiser_path = f"data/models/denoisers/{self.dl_denoiser_type}_best.pth"
            if os.path.exists(denoiser_path):
                self.dl_denoiser.load_state_dict(torch.load(denoiser_path, map_location=self.device))
                print(f"[RespiratoryController] Loaded DL Denoiser ({self.dl_denoiser_type}) from {denoiser_path}")
            else:
                print(f"[RespiratoryController] Warning: DL Denoiser ({self.dl_denoiser_type}) weights not found. Using random init.")
            self.dl_denoiser.eval()
        
        # Challenge #5: Few-Shot Learning
        self.few_shot_adapter = None
        if self.enable_few_shot_learning:
            print("[RespiratoryController] Initializing Few-Shot Learning module...")
            self.few_shot_adapter = create_few_shot_adapter(device=self.device)
        
        # Challenge #6: Personalized Baselines
        self.baseline_tracker = None
        self.personalized_anomaly_detector = None
        if self.enable_personalized_baselines:
            print("[RespiratoryController] Initializing Personalized Baseline Learning...")
            self.baseline_tracker, self.personalized_anomaly_detector = create_personalized_baseline_system(
                patient_id="unknown", age_months=6
            )
        
        # Challenge #7: Comorbidity & Etiology
        self.differential_diagnosis_engine = None
        self.risk_profiler = None
        if self.enable_comorbidity_etiology:
            print("[RespiratoryController] Initializing Comorbidity & Etiology Prediction...")
            self.differential_diagnosis_engine, self.risk_profiler = create_comorbidity_and_etiology_system(
                device=self.device
            )

    def _infer_baseline_name(self):
        """Infer baseline name when explicit name is not provided."""
        if self.enable_transformer:
            if self.enable_robust_fusion:
                return "Quad-Modal Full + Robust Fusion"
            if self.enable_audio:
                return "Quad-Modal Full (Ours)"
            return "Tri-Modal VLM-Gated"
        if self.enable_mamba:
            return "Mamba-Fusion-Full"
        if self.enable_flow and not self.enable_depth:
            return "Flow-Only"
        if self.enable_depth and not self.enable_flow:
            return "Depth-Only"
        if self.enable_vmd:
            return "Fused-VMD"
        return "Fused-SNR"

    def process_video(
        self,
        video_path,
        age_months=None,
        vlm_model_id=None,
        drop_modality=None,
        baseline_name=None,
        baseline_paradigm=None,
    ):
        """Processes video for both Flow-based and Depth-based monitoring."""
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Strategy: Try High-Res (960px) for RTX 8000 quality, fallback to 640px if memory fails
        target_widths = [960, 640] 
        frames = []
        
        for target_width in target_widths:
            try:
                frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to start
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    h, w = frame.shape[:2]
                    if w > target_width:
                        scale = target_width / w
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (target_width, new_h))
                    
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Verify memory allocation by creating array immediately
                # (This triggers the OOM if it's going to happen)
                frames_test = np.array(frames) 
                
                # If successful, keep frames and break loop
                frames_np = frames_test
                print(f"Loaded video at {target_width}px width. Shape: {frames_np.shape}")
                break
                
            except (MemoryError, Exception) as e:
                # Catch specific memory errors (including numpy's custom ones)
                if "allocate" in str(e).lower() or isinstance(e, MemoryError):
                    print(f"Warning: Memory allocation failed at {target_width}px. Retrying with lower resolution...")
                    frames = [] # Clear memory
                    import gc
                    gc.collect()
                    continue
                else:
                    raise e # Re-raise non-memory errors

        cap.release()
        
        if len(frames) == 0:
            print("Error: Could not load video frames (OOM or empty).")
            return {
                "bpm": 0.0, "diagnosis": {"is_tachypnea": False}, "vlm_assessment": "Error: OOM/Empty", "vlm_distress": False
            }

        total_frames = len(frames_np)
        print(f"Total frames: {total_frames} at {fps} FPS")
        
        # 1. ROI Detection
        print("Detecting ROI...")
        # Try anatomical stabilizer first (Phase 11)
        sample_frame = frames_np[0]
        anatomical_roi = self.stabilizer.get_stabilized_chest_roi(sample_frame)
        
        if anatomical_roi:
            print("  Anatomical ROI detected via Keypoints.")
            roi_box = np.array(anatomical_roi)
        else:
            print("  Falling back to YOLO-based detection.")
            try:
                roi_box, _ = self.preprocessor.get_roi_box(frames_np)
            except Exception as e:
                print(f"  Warning: Preprocessor ROI detection failed: {e}")
                roi_box = None
            
            if roi_box is None:
                # Last resort: Center 50% Crop
                print("  Warning: All ROI detections failed. Using Center Crop.")
                h, w = frames_np[0].shape[:2]
                roi_box = np.array([w//4, h//4, 3*w//4, 3*h//4])
        
        # 2. Depth Signal Extraction (gated by enable_depth)
        if self.enable_depth:
            print("Extracting Depth Signal...")
            depth_signal = self.depth_engine.extract_signal_from_frames(frames_np, roi_box)
            if self.enable_dl_denoise and self.dl_denoiser is not None:
                # Apply DL Sequence Denoising
                t_in = torch.tensor(depth_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    t_out = self.dl_denoiser(t_in)
                depth_signal = t_out.squeeze().cpu().numpy()
            
            if drop_modality == "depth":
                print("  [Robustness Test] Dropping Depth signal...")
                depth_signal = np.zeros_like(depth_signal)
        else:
            print("Depth disabled for this baseline.")
            depth_signal = np.zeros(total_frames)
        
        # 4. Flow-based Monitoring (gated by enable_flow)
        if self.enable_flow:
            print("Extracting Flow Signal...")
            flow_signal = self._extract_optical_flow(frames_np, roi_box)
            if self.enable_dl_denoise and self.dl_denoiser is not None:
                # Apply DL Sequence Denoising
                t_in = torch.tensor(flow_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    t_out = self.dl_denoiser(t_in)
                flow_signal = t_out.squeeze().cpu().numpy()
            if drop_modality == "flow":
                print("  [Robustness Test] Dropping Flow signal...")
                flow_signal = np.zeros_like(flow_signal)
        else:
            print("Flow disabled for this baseline.")
            flow_signal = np.zeros(total_frames)
        
        # 4.5 Audio Signal Extraction (Phase 17 Quad-Modal)
        if self.enable_audio:
            print("Extracting Audio Signal (Spectrogram Features)...")
            audio_signal = self._extract_audio_features(video_path, total_frames)
            if drop_modality == "audio":
                print("  [Robustness Test] Dropping Audio signal...")
                audio_signal = np.zeros_like(audio_signal)
        else:
            print("Audio disabled for this baseline.")
            audio_signal = np.zeros(total_frames)
            
        # 4.6 VideoMAE Override (Phase 20)
        # If enabled, replacing Depth/Flow handcrafted signals with VideoMAE SSL projection.
        if self.enable_videomae and self.videomae_encoder is not None:
            print("Extracting 3D Spatio-Temporal VideoMAE Features...")
            # Sample 16 frames uniformly
            idx = np.linspace(0, len(frames_np) - 1, 16).astype(int)
            sampled_frames = frames_np[idx]
            # Convert to appropriate shape: (Batch=1, NumFrames=16, C=3, H=224, W=224)
            resized_frames = []
            for f in sampled_frames:
                resized_frames.append(cv2.resize(f, (224, 224)))
            video_tensor = torch.tensor(np.array(resized_frames), dtype=torch.float32).permute(0, 3, 1, 2)
            # Normalize 0-1
            video_tensor = video_tensor / 255.0
            video_tensor = video_tensor.unsqueeze(0).to(self.device) # (1, 16, 3, 224, 224)
            # Output from wrapper is (Batch, d_model, SeqLen) -> (1, 64, 16)
            # We want to map this to our 300 sequence length to mimic 1D waveforms
            with torch.no_grad():
                v_feats = self.videomae_encoder(video_tensor) # (1, 64, 16)
                v_feats = torch.nn.functional.interpolate(v_feats, size=len(depth_signal), mode='linear')
            
            # Since VideoMAE encodes both appearance and motion densely, we'll clone its robust representation 
            # across the modalities that feed into our fusion transformer so it naturally learns the representation.
            # (Just taking the first channels as proxy 1D signals for compatability)
            v_sig = v_feats[0].cpu().numpy() # (64, 300)
            depth_signal = v_sig[0] # Channel 0 proxy
            flow_signal = v_sig[1] # Channel 1 proxy
        
        # 5. Per-modality respiratory estimates.
        depth_bpm = SignalProcessor.get_bpm(depth_signal, fps)
        depth_confidence = SignalProcessor.calculate_confidence(depth_signal, fps)
        flow_bpm = SignalProcessor.get_bpm(flow_signal, fps)
        flow_confidence = SignalProcessor.calculate_confidence(flow_signal, fps)
        audio_bpm = SignalProcessor.get_bpm(audio_signal, fps)
        audio_confidence = SignalProcessor.calculate_confidence(audio_signal, fps)

        modality_names = ["depth", "flow", "audio"]
        modality_bpms = np.array([depth_bpm, flow_bpm, audio_bpm], dtype=np.float32)
        modality_confidences = np.array(
            [depth_confidence, flow_confidence, audio_confidence], dtype=np.float32
        )
        enabled_modalities = np.array(
            [float(self.enable_depth), float(self.enable_flow), float(self.enable_audio)],
            dtype=np.float32
        )
        if drop_modality in modality_names:
            drop_idx = modality_names.index(drop_modality)
            enabled_modalities[drop_idx] = 0.0
            modality_confidences[drop_idx] = 0.0
        
        def _normalized_weights(raw_weights, enabled_mask):
            weights = np.array(raw_weights, dtype=np.float32) * np.array(enabled_mask, dtype=np.float32)
            total = float(np.sum(weights))
            if total > 1e-8:
                return weights / total
            active_idx = np.where(np.array(enabled_mask) > 0)[0]
            if len(active_idx) == 0:
                return np.array([1.0 / len(weights)] * len(weights), dtype=np.float32)
            weights = np.zeros(len(weights), dtype=np.float32)
            weights[active_idx] = 1.0 / len(active_idx)
            return weights

        # 6. Adaptive Signal Fusion
        fusion_strategy = "SNR-Weighted (Robust)"
        snr_weights = _normalized_weights(modality_confidences, enabled_modalities)
        heuristic_bpm = float(np.dot(modality_bpms, snr_weights))
        total_conf = float(np.sum(modality_confidences * enabled_modalities))
        cross_modal_telemetry = None
        
        # Strategy A: Supervised Regressor (Highest Priority if trained & enabled)
        supervised_bpm = None
        if self.enable_supervised:
            supervised_bpm = self.supervised_model.predict(
                flow_bpm, depth_bpm, flow_confidence, depth_confidence
            )
        
        if supervised_bpm is not None:
            fused_bpm = supervised_bpm
            fusion_strategy = "Supervised Learning (XGB/RF)"
            # Skip logic below
        # Fallback Strategies
        # Strategy B: Temporal Cross-Attention Transformer (Phase 12)
        
        # Prepare inputs for Transformer
        # Signals need to be resampled/padded to model's seq_len (300)
        # For prototype, we'll just demonstrate the Tensor creation
        anomaly_score = 0.0
        try:
            target_len = 300
            d_tens = torch.tensor(depth_signal).float().unsqueeze(0) # (1, L)
            f_tens = torch.tensor(flow_signal).float().unsqueeze(0) # (1, L)
            a_tens = torch.tensor(audio_signal).float().unsqueeze(0) # (1, L)
            
            # Simple padding/truncation
            if d_tens.shape[1] > target_len:
                d_tens = d_tens[:, :target_len]
                f_tens = f_tens[:, :target_len]
                a_tens = a_tens[:, :target_len]
            elif d_tens.shape[1] < target_len:
                pad = target_len - d_tens.shape[1]
                d_tens = torch.nn.functional.pad(d_tens, (0, pad))
                f_tens = torch.nn.functional.pad(f_tens, (0, pad))
                a_tens = torch.nn.functional.pad(a_tens, (0, pad))
                
            d_tens = d_tens.to(self.device)
            f_tens = f_tens.to(self.device)
            a_tens = a_tens.to(self.device)
            
            if self.enable_transformer:
                transformer_bpm = self.fusion_model(d_tens, f_tens, audio_signal=a_tens).item()
                # Compute SSL Anomaly Score for Apnea detection
                anomaly_score_tensor = self.fusion_model.compute_anomaly_score(d_tens, f_tens, audio_signal=a_tens)
                anomaly_score = anomaly_score_tensor.item()
                robust_bpm = heuristic_bpm
                disagreement_detected = False

                if self.enable_robust_fusion and self.robust_fusion_model is not None:
                    try:
                        with torch.no_grad():
                            robust_output = self.robust_fusion_model(
                                d_tens, f_tens, audio_signal=a_tens, return_diagnostics=True
                            )

                        learned_confidence = robust_output["confidence"][0].detach().cpu().numpy()
                        learned_weights = _normalized_weights(learned_confidence, enabled_modalities)

                        # Blend learned agreement confidence with SNR confidence for stability.
                        robust_weights = _normalized_weights(
                            (0.4 * learned_weights) + (0.6 * snr_weights),
                            enabled_modalities
                        )
                        robust_bpm = float(np.dot(modality_bpms, robust_weights))

                        active_indices = np.where(enabled_modalities > 0)[0]
                        if len(active_indices) > 1:
                            active_bpms = modality_bpms[active_indices]
                            bpm_spread = float(np.std(active_bpms))
                            bpm_center = float(np.median(active_bpms))
                            bpm_disagreement_ratio = bpm_spread / max(abs(bpm_center), 1.0)
                        else:
                            bpm_disagreement_ratio = 0.0

                        model_disagreement_score = float(
                            robust_output["disagreement_score"][0].item()
                        )
                        model_disagreement_flag = bool(
                            robust_output["disagreement_flag"][0].item()
                        )
                        disagreement_detected = (
                            model_disagreement_flag
                            or bpm_disagreement_ratio > self.robust_disagreement_threshold
                        )

                        cross_modal_telemetry = {
                            "enabled": True,
                            "disagreement_detected": disagreement_detected,
                            "model_disagreement_score": round(model_disagreement_score, 4),
                            "bpm_disagreement_ratio": round(bpm_disagreement_ratio, 4),
                            "confidence_scores": {
                                "depth": round(float(robust_weights[0]), 4),
                                "flow": round(float(robust_weights[1]), 4),
                                "audio": round(float(robust_weights[2]), 4),
                            },
                            "fallback_bpm": round(robust_bpm, 2),
                        }
                    except Exception as robust_error:
                        cross_modal_telemetry = {
                            "enabled": True,
                            "disagreement_detected": False,
                            "error": str(robust_error),
                            "fallback_bpm": round(robust_bpm, 2),
                        }

                # Guardrail: if upstream signals are weak or output is implausible,
                # prefer robust classical estimate over unreliable transformer output.
                invalid_transformer = (
                    (not np.isfinite(transformer_bpm))
                    or transformer_bpm < 5.0
                    or transformer_bpm > 120.0
                    or total_conf < 0.15
                )

                if invalid_transformer:
                    fused_bpm = robust_bpm if self.enable_robust_fusion else heuristic_bpm
                    fusion_strategy = "Quad-Modal Transformer (Guarded Fallback)"
                    supervised_bpm = fused_bpm
                elif disagreement_detected:
                    fused_bpm = robust_bpm
                    fusion_strategy = "Quad-Modal Transformer + Cross-Modal Robust Fallback"
                    supervised_bpm = fused_bpm
                else:
                    fused_bpm = transformer_bpm
                    fusion_strategy = "Quad-Modal Transformer"
                    # Override supervised if transformer is enabled and reliable.
                    supervised_bpm = fused_bpm
                
            elif self.enable_mamba and self.mamba_model is not None:
                # Mamba needs inputs as (B, L)
                d_flat = d_tens.squeeze(0).squeeze(0)  # [L]
                f_flat = f_tens.squeeze(0).squeeze(0)
                a_flat = a_tens.squeeze(0).squeeze(0)
                
                # Reshape to (1, L)
                d_flat = d_flat.unsqueeze(0).to(self.device)
                f_flat = f_flat.unsqueeze(0).to(self.device)
                a_flat = a_flat.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    fused_bpm = self.mamba_model(d_flat, f_flat, a_flat).item()
                    
                fusion_strategy = "Quad-Modal Mamba-SSM"
                supervised_bpm = fused_bpm
                
        except Exception as e:
            # print(f"Transformer inference skipped: {e}")
            pass

        if supervised_bpm is None:
            # Strategy C: SNR-Weighted Heuristic (Fallback)
            fused_bpm = heuristic_bpm
        
        if vlm_model_id and vlm_model_id != "None":
            print(f"Performing VLM Clinical Assessment with {vlm_model_id}...")
            sample_frame = frames_np[0] # Using first frame for assessment
            vlm_results = self.vlm_engine.analyze_clinical_signs(sample_frame, model_id=vlm_model_id)
        else:
            # Skip VLM for speed/ablation
            vlm_results = {
                "assessment": "N/A (VLM Disabled)",
                "distress_detected": False
            }
        
        # 8. ARI Diagnosis
        print(
            f"Diagnosing with BPM={fused_bpm:.2f} "
            f"(Flow: {flow_bpm:.1f} [{flow_confidence:.1f}%], "
            f"Depth: {depth_bpm:.1f} [{depth_confidence:.1f}%], "
            f"Audio: {audio_bpm:.1f} [{audio_confidence:.1f}%])"
        )
        print(f"Apnea Anomaly Score: {anomaly_score:.4f}")
        diagnosis = ARIDiagnostic.classify_rr(fused_bpm, age_months, anomaly_score=anomaly_score)

        conformal_triage = None
        if self.enable_conformal_triage:
            disagreement_score = 0.0
            if isinstance(cross_modal_telemetry, dict):
                model_score = float(cross_modal_telemetry.get("model_disagreement_score", 0.0) or 0.0)
                bpm_ratio = float(cross_modal_telemetry.get("bpm_disagreement_ratio", 0.0) or 0.0)
                disagreement_score = max(model_score, bpm_ratio)

            selected_name = baseline_name if baseline_name else self._infer_baseline_name()
            qhat, qhat_meta = self.conformal_qhat_selector.select_qhat(
                model_name=selected_name,
                paradigm=baseline_paradigm,
                disagreement_score=disagreement_score,
                condition_on_disagreement=self.conformal_condition_on_disagreement,
            )
            conformal_triage = self.conformal_triage_head.triage(
                bpm=fused_bpm,
                age_months=age_months,
                qhat=qhat,
                disagreement_score=disagreement_score,
                anomaly_score=anomaly_score,
            )
            conformal_triage.update(qhat_meta)
            diagnosis["conformal_triage_status"] = conformal_triage["triage_status"]
            diagnosis["conformal_interval"] = [
                conformal_triage["interval_lower_bpm"],
                conformal_triage["interval_upper_bpm"],
            ]

        return {
            "bpm": round(fused_bpm, 1),
            "flow_bpm": round(flow_bpm, 1),
            "depth_bpm": round(depth_bpm, 1),
            "audio_bpm": round(audio_bpm, 1),
            "depth_confidence": round(depth_confidence, 1),
            "flow_confidence": round(flow_confidence, 1),
            "audio_confidence": round(audio_confidence, 1),
            "fusion_strategy": fusion_strategy,
            "cross_modal_telemetry": cross_modal_telemetry,
            "conformal_triage": conformal_triage,
            "diagnosis": diagnosis,
            "vlm_assessment": vlm_results['assessment'],
            "vlm_distress": vlm_results['distress_detected'],
            "roi": roi_box.tolist() if isinstance(roi_box, np.ndarray) else roi_box,
            "depth_signal": depth_signal.tolist(),
            "flow_signal": flow_signal.tolist(),
            "audio_signal": audio_signal.tolist(),
            "research_scope": {
                "supported_now": [
                    "bpm_estimation",
                    "cross_modal_conflict_detection"
                ],
                "future_work_label_gaps": [
                    "challenge_1_physiological_states_rem_nrem_feeding_distress",
                    "challenge_3_longitudinal_progression_outcomes",
                    "challenge_7_etiology_comorbidity_prediction"
                ]
            },
            "recommendation": (
                "Consult a pediatrician immediately."
                if (
                    diagnosis["is_tachypnea"]
                    or diagnosis.get("is_apnea", False)
                    or vlm_results['distress_detected']
                    or (
                        isinstance(conformal_triage, dict)
                        and conformal_triage.get("triage_status") in ["TACHYPNEA_ALERT", "CRITICAL_APNEA_ALERT"]
                    )
                )
                else (
                    "Manual review recommended due to conformal uncertainty."
                    if (
                        isinstance(conformal_triage, dict)
                        and conformal_triage.get("triage_status") == "UNCERTAIN_REVIEW"
                    )
                    else "Continue monitoring."
                )
            )
        }

    def _extract_optical_flow(self, frames, roi_box):
        """
        Extracts representative flow signal from ROI using Farneback Optical Flow.
        Returns: 1D numpy array of flow magnitude over time.
        """
        if len(frames) < 2:
            return np.zeros(len(frames))
            
        x1, y1, x2, y2 = map(int, roi_box)
        flow_signal = [0] # First frame has 0 flow
        
        # Ensure ROI is within frame bounds
        h, w = frames[0].shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(len(frames)) # Invalid ROI
        
        # Resize logic for Flow Efficiency & Memory Safety
        # Optical flow doesn't need 1080p to detect breathing motion.
        # Downscaling to 128x128 greatly reduces memory/compute.
        target_size = (128, 128) 
        
        # Force GC to clear any fragmented memory before dense allocation
        import gc
        gc.collect()

        try:
            # Prepare first frame
            prev_crop = frames[0][y1:y2, x1:x2]
            prev_gray = cv2.cvtColor(cv2.resize(prev_crop, target_size), cv2.COLOR_RGB2GRAY)
            
            for i in range(1, len(frames)):
                curr_crop = frames[i][y1:y2, x1:x2]
                curr_gray = cv2.cvtColor(cv2.resize(curr_crop, target_size), cv2.COLOR_RGB2GRAY)
                
                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                                   0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Compute magnitude of flow (movement intensity)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                avg_motion = np.mean(mag)
                flow_signal.append(avg_motion)
                
                prev_gray = curr_gray
                
                # Periodic GC during long loops if memory is tight
                if i % 100 == 0:
                    gc.collect()

        except Exception as e:
            print(f"[ERROR] Optical Flow Failed (Memory/Compute): {e}")
            print("Returning zero signal for Flow baseline.")
            return np.zeros(len(frames))
            
        flow_sig = np.array(flow_signal)
        
        # Safe normalization to prevent extreme DC offsets dominating the FFT
        if len(flow_sig) > 2:
            # Drop the first frame (which is forced to 0) to avoid artificial spikes
            flow_sig[0] = flow_sig[1] 
            
            # Detrend the global motion drift
            from src.core.signal_processor import SignalProcessor
            flow_sig = SignalProcessor.detrend_signal(flow_sig, window_size=int(len(frames)//3))
            
            _range = np.max(flow_sig) - np.min(flow_sig)
            if _range > 1e-6:
                flow_sig = (flow_sig - np.min(flow_sig)) / _range
                
        return flow_sig

    def _extract_audio_features(self, video_path, num_frames):
        """
        Extracts Audio Features (Log-Mel Spectrogram mean per frame window)
        to act as the fourth modality. Uses pydub/librosa if installed,
        else returns zero-padded signal (graceful degradation).
        """
        audio_signal = np.zeros(num_frames)
        # Attempt to read audio using external libs (stubbed for demo fallback)
        try:
            import librosa
            import moviepy.editor as mp
            
            clip = mp.VideoFileClip(video_path)
            if clip.audio is not None:
                # Save temp audio file
                temp_audio = "temp_audio.wav"
                clip.audio.write_audiofile(temp_audio, verbose=False, logger=None)
                
                # Load with librosa
                y, sr = librosa.load(temp_audio, sr=16000)
                # Compute Mel Spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
                S_DB = librosa.power_to_db(S, ref=np.max) # (mel_bins, time_frames)
                
                # Average power per time frame
                power_over_time = np.mean(S_DB, axis=0)
                
                # Interpolate to match video frames length seamlessly
                # We align the audio temporal dimension to the video temporal dimension
                x_old = np.linspace(0, 1, len(power_over_time))
                x_new = np.linspace(0, 1, num_frames)
                audio_signal = np.interp(x_new, x_old, power_over_time)
                
                os.remove(temp_audio)
                print(f"  Successfully extracted {len(audio_signal)} audio features via Librosa.")
            else:
                print("  No audio track found in video. Yielding zeros.")
        except ImportError:
            # print("  Librosa/MoviePy not found. Skipping genuine audio extraction.")
            pass
        except Exception as e:
            print(f"  Audio extraction failed: {e}. Yielding zeros.")
            
        return audio_signal

if __name__ == "__main__":
    # Example usage
    controller = RespiratoryController(
        "src/config/test_infer.yaml",
        "data/models/VIRENet_Coarse2fine_Body_best.pth"
    )
    result = controller.process_video("data/demo-air-400-s05-23.mp4", age_months=6)
    print("\n--- Pipeline Test Result ---")
    print(result)
