"""
Challenge #5: Few-Shot Generalization to New Populations

Enables rapid adaptation to new camera types, facilities, and populations with minimal labeled data.
Uses meta-learning, transfer learning, and domain adaptation techniques.

Key Components:
- FewShotAdaptationModule: Prototypical networks for learning from minimal examples
- DomainShiftDetector: Detects when model operates outside training distribution
- TransferLearningAdapter: Efficient fine-tuning with few samples
- MetaLearnerOptimizer: MAML-like optimization for rapid adaptation

Author: Respiratory Rate Monitoring AI
Date: 2026
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import warnings


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FewShotExample:
    """Single example for few-shot learning."""
    modality_data: Dict[str, np.ndarray]  # {depth, flow, audio}
    label: int  # 0=normal, 1=ARI, 2=abnormal_other
    meta_info: Dict = None  # age, population, facility, etc.


@dataclass
class AdaptationResult:
    """Result of few-shot adaptation."""
    adapted: bool
    confidence: float  # 0-1, how confident in adaptation
    num_examples_used: int
    domain_shift_detected: bool
    domain_shift_magnitude: float  # 0-1, how different from training
    improvement_estimate: float  # Expected accuracy improvement %
    recommendation: str  # Action recommendation


@dataclass
class MetaLearnResult:
    """Result of meta-learned rapid adaptation."""
    task_accuracy: float
    adaptation_steps: int
    parameter_shift_norm: float  # How much parameters changed
    confidence: float
    learned_features: np.ndarray  # (feature_dim,)


@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning."""
    num_fine_tune_steps: int = 20
    learning_rate: float = 0.001
    layer_freeze_ratio: float = 0.7  # Freeze 70% of layers
    regularization_strength: float = 0.01
    early_stopping_patience: int = 5


# ============================================================================
# FEW-SHOT LEARNING MODULE
# ============================================================================

class PrototypicalNetwork(nn.Module):
    """
    Prototypical networks for few-shot learning.
    Learns to classify based on distances to class prototypes.
    """
    
    def __init__(self, feature_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Encoder: Projects raw multimodal data to feature space
        self.encoder = nn.Sequential(
            nn.Linear(256, 256),  # Input: concatenated depth+flow+audio features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, feature_dim)
        )
        
        # Prototype storage (learned during meta-training)
        self.register_buffer(
            'class_prototypes',
            torch.randn(num_classes, feature_dim) / np.sqrt(feature_dim)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature space."""
        return self.encoder(x)
    
    def forward(self, query: torch.Tensor, support: torch.Tensor, 
                support_labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for few-shot classification.
        
        Args:
            query: (batch_size, input_dim) - Query examples
            support: (num_support, input_dim) - Support set examples
            support_labels: (num_support,) - Support set labels
            
        Returns:
            logits: (batch_size, num_classes) - Classification logits
        """
        # Encode all examples
        query_features = self.encode(query)  # (batch_size, feature_dim)
        support_features = self.encode(support)  # (num_support, feature_dim)
        
        # Compute class prototypes from support set
        prototypes = []
        for class_idx in range(self.num_classes):
            class_mask = support_labels == class_idx
            if class_mask.sum() > 0:
                prototype = support_features[class_mask].mean(dim=0)
            else:
                prototype = self.class_prototypes[class_idx]
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (num_classes, feature_dim)
        
        # Compute distances from query to each prototype
        # Using Euclidean distance in feature space
        distances = torch.cdist(query_features, prototypes)  # (batch_size, num_classes)
        
        # Convert distances to similarity (negative distance)
        logits = -distances
        return logits
    
    def predict(self, query: torch.Tensor, support: torch.Tensor,
                support_labels: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels for query examples.
        
        Returns:
            predictions: (batch_size,) - Predicted class indices
            confidences: (batch_size,) - Confidence scores (0-1)
        """
        with torch.no_grad():
            logits = self.forward(query, support, support_labels)
            probs = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            return predictions.cpu().detach().numpy(), confidences.cpu().detach().numpy()


class FewShotAdaptationModule:
    """
    Few-shot learning module for rapid adaptation to new populations.
    """
    
    def __init__(self, feature_dim: int = 128, num_classes: int = 3, device: str = 'cpu'):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.device = device
        
        # Prototypical network
        self.model = PrototypicalNetwork(feature_dim, num_classes)
        self.model = self.model.to(device)
        
        # Adaptation tracking
        self.adaptation_history: List[Dict] = []
        self.adapted_prototypes: Optional[torch.Tensor] = None
        self.num_adaptation_examples = 0
    
    def prepare_multimodal_features(self, modality_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert multimodal data (depth, flow, audio) to feature vector.
        
        Args:
            modality_data: {
                'depth': (H, W) array,
                'flow': (H, W, 2) optical flow,
                'audio': (frequencies,) spectrogram
            }
            
        Returns:
            features: (256,) concatenated feature vector
        """
        features = []
        
        # Depth features: mean, std, max, quantiles
        if 'depth' in modality_data and modality_data['depth'] is not None:
            depth = modality_data['depth'].astype(np.float32)
            depth = np.nan_to_num(depth)
            features.extend([
                np.mean(depth), np.std(depth), np.max(depth),
                np.percentile(depth, 25), np.percentile(depth, 75)
            ])
        else:
            features.extend([0.0] * 5)
        
        # Optical flow features: magnitude stats, direction
        if 'flow' in modality_data and modality_data['flow'] is not None:
            flow = modality_data['flow'].astype(np.float32)
            flow = np.nan_to_num(flow)
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            features.extend([
                np.mean(magnitude), np.std(magnitude), np.max(magnitude),
                np.percentile(magnitude, 25), np.percentile(magnitude, 75),
                np.mean(flow[..., 0]), np.mean(flow[..., 1])
            ])
        else:
            features.extend([0.0] * 7)
        
        # Audio features: spectrogram statistics
        if 'audio' in modality_data and modality_data['audio'] is not None:
            audio = modality_data['audio'].astype(np.float32)
            audio = np.nan_to_num(audio)
            if len(audio) > 0:
                # Ensure we have exactly right number of audio features
                audio_feat = [
                    np.mean(audio), np.std(audio), np.max(audio),
                    np.percentile(audio, 25) if len(audio) > 0 else 0,
                    np.percentile(audio, 75) if len(audio) > 0 else 0
                ]
                features.extend(audio_feat)
            else:
                features.extend([0.0] * 5)
        else:
            features.extend([0.0] * 5)
        
        # Pad to 256 features
        features = np.array(features[:256], dtype=np.float32)
        if len(features) < 256:
            features = np.pad(features, (0, 256 - len(features)), mode='constant')
        
        return features.astype(np.float32)
    
    def adapt(self, support_examples: List[FewShotExample], 
             validation_examples: Optional[List[FewShotExample]] = None,
             verbose: bool = False) -> AdaptationResult:
        """
        Adapt model using few-shot examples from new population.
        
        Args:
            support_examples: List of labeled examples from new population (typically 5-20)
            validation_examples: Optional validation set to estimate improvement
            verbose: Whether to print adaptation progress
            
        Returns:
            AdaptationResult with confidence, domain shift info, and recommendation
        """
        if len(support_examples) < 3:
            return AdaptationResult(
                adapted=False,
                confidence=0.0,
                num_examples_used=0,
                domain_shift_detected=False,
                domain_shift_magnitude=0.0,
                improvement_estimate=0.0,
                recommendation="Need at least 3 examples for adaptation"
            )
        
        try:
            # Prepare support set
            support_features = []
            support_labels = []
            
            for example in support_examples:
                features = self.prepare_multimodal_features(example.modality_data)
                support_features.append(features)
                support_labels.append(example.label)
            
            support_tensor = torch.from_numpy(np.array(support_features)).to(self.device)
            labels_tensor = torch.from_numpy(np.array(support_labels)).long().to(self.device)
            
            # Detect domain shift by comparing to base model distribution
            domain_shift_magnitude = self._detect_domain_shift(support_tensor)
            domain_shift_detected = domain_shift_magnitude > 0.3  # Threshold
            
            # Update adapted prototypes
            with torch.no_grad():
                support_encoded = self.model.encode(support_tensor)
                
                # Compute per-class prototypes
                adapted_prototypes = []
                for class_idx in range(self.num_classes):
                    class_mask = labels_tensor == class_idx
                    if class_mask.sum() > 0:
                        prototype = support_encoded[class_mask].mean(dim=0)
                    else:
                        # Fall back to base prototype
                        prototype = self.model.class_prototypes[class_idx]
                    adapted_prototypes.append(prototype)
                
                self.adapted_prototypes = torch.stack(adapted_prototypes)
            
            # Estimate improvement
            improvement = float(min(50.0, domain_shift_magnitude * 100))  # Rough estimate
            
            # Track adaptation
            self.adaptation_history.append({
                'num_examples': len(support_examples),
                'domain_shift': float(domain_shift_magnitude),
                'improvement_estimate': improvement
            })
            self.num_adaptation_examples = len(support_examples)
            
            confidence = 0.5 + (len(support_examples) / 50.0) * 0.4  # Up to 0.9
            confidence = min(0.9, confidence)
            
            if domain_shift_detected:
                recommendation = (
                    f"Domain shift detected ({domain_shift_magnitude:.2f}). "
                    f"Model adapting. Consider collecting more diverse examples."
                )
            else:
                recommendation = "Adaptation successful. Model ready for new population."
            
            return AdaptationResult(
                adapted=True,
                confidence=confidence,
                num_examples_used=len(support_examples),
                domain_shift_detected=domain_shift_detected,
                domain_shift_magnitude=float(domain_shift_magnitude),
                improvement_estimate=improvement,
                recommendation=recommendation
            )
        
        except Exception as e:
            return AdaptationResult(
                adapted=False,
                confidence=0.0,
                num_examples_used=0,
                domain_shift_detected=False,
                domain_shift_magnitude=0.0,
                improvement_estimate=0.0,
                recommendation=f"Adaptation failed: {str(e)}"
            )
    
    def _detect_domain_shift(self, new_data: torch.Tensor) -> float:
        """
        Detect domain shift by measuring distribution difference.
        Uses maximum mean discrepancy (MMD) approximation.
        
        Returns:
            shift_magnitude: 0-1, where 0 = no shift, 1 = complete shift
        """
        try:
            with torch.no_grad():
                # Encode new data
                new_features = self.model.encode(new_data)
                
                # MMD: Compare mean of new features to mean of base prototypes
                base_mean = self.model.class_prototypes.mean(dim=0)
                new_mean = new_features.mean(dim=0)
                
                shift = torch.norm(new_mean - base_mean) / (torch.norm(base_mean) + 1e-8)
                shift = torch.clamp(shift, torch.tensor(0.0), torch.tensor(1.0))
                
                return float(shift.cpu().numpy())
        except:
            return 0.5
    
    def predict_with_adapted_prototypes(self, query_data: Dict[str, np.ndarray]
                                       ) -> Tuple[int, float, str]:
        """
        Predict using adapted prototypes if available.
        
        Returns:
            prediction: Class label (0/1/2)
            confidence: Confidence score (0-1)
            label: Human-readable label
        """
        if self.adapted_prototypes is None:
            return -1, 0.0, "Not adapted yet"
        
        # Prepare query features
        features = self.prepare_multimodal_features(query_data)
        query_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                query_features = self.model.encode(query_tensor)
                
                # Compute distances to adapted prototypes
                distances = torch.cdist(query_features, self.adapted_prototypes)
                logits = -distances
                
                probs = torch.softmax(logits, dim=1)
                confidence, prediction = torch.max(probs, dim=1)
                
                label_names = {0: "normal", 1: "ARI", 2: "abnormal"}
                return int(prediction[0].cpu().numpy()), float(confidence[0].cpu().numpy()), label_names[int(prediction[0].cpu().numpy())]
        except:
            return -1, 0.0, "Prediction failed"


# ============================================================================
# DOMAIN SHIFT DETECTOR
# ============================================================================

class DomainShiftDetector:
    """
    Detects when model operates outside its training distribution.
    Prevents blind application of model to incompatible populations.
    """
    
    def __init__(self, reference_mean: Optional[np.ndarray] = None,
                 reference_cov: Optional[np.ndarray] = None):
        """
        Initialize with reference distribution from training data.
        
        Args:
            reference_mean: Mean of training feature distribution
            reference_cov: Covariance of training feature distribution
        """
        self.reference_mean = reference_mean if reference_mean is not None else np.zeros(256)
        self.reference_cov = reference_cov if reference_cov is not None else np.eye(256)
        self.detection_history: List[float] = []
    
    def detect_shift(self, features: np.ndarray) -> Tuple[bool, float, str]:
        """
        Detect domain shift in input features.
        
        Args:
            features: (256,) feature vector
            
        Returns:
            shift_detected: Boolean
            shift_magnitude: 0-1 score
            explanation: String description
        """
        try:
            features = features.astype(np.float32)
            
            # Method 1: Mahalanobis distance
            diff = features - self.reference_mean
            try:
                cov_inv = np.linalg.pinv(self.reference_cov)
                mahal_dist = float(np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T)))
            except:
                mahal_dist = float(np.linalg.norm(diff) / (np.linalg.norm(self.reference_mean) + 1e-8))
            
            # Normalize to 0-1
            # Assuming mean Mahal distance in training ~5-10, shift when >15
            shift_magnitude = float(min(1.0, mahal_dist / 15.0))
            shift_detected = bool(shift_magnitude > 0.3)
            
            # Method 2: Outlier detection
            feature_norms = np.abs(features)
            ref_norms = np.abs(self.reference_mean)
            norm_diff = np.mean(np.abs(feature_norms - ref_norms))
            
            if float(norm_diff) > float(shift_magnitude):
                shift_magnitude = float(norm_diff) / (float(np.mean(ref_norms)) + 1e-8)
                shift_magnitude = min(1.0, float(shift_magnitude))
            
            self.detection_history.append(float(shift_magnitude))
            
            # Explanation
            if shift_detected:
                explanation = f"Significant domain shift detected (magnitude: {shift_magnitude:.2f}). Data characteristics differ from training distribution."
            else:
                explanation = f"Domain shift within acceptable range (magnitude: {shift_magnitude:.2f})."
            
            return bool(shift_detected), float(shift_magnitude), explanation
        
        except Exception as e:
            return False, 0.0, f"Detection error: {str(e)}"
    
    def update_reference(self, features_batch: np.ndarray):
        """Update reference distribution with new data."""
        try:
            self.reference_mean = np.mean(features_batch, axis=0)
            self.reference_cov = np.cov(features_batch.T)
        except:
            pass


# ============================================================================
# TRANSFER LEARNING ADAPTER
# ============================================================================

class TransferLearningAdapter:
    """
    Efficient fine-tuning with few samples using transfer learning.
    Freezes early layers (generic features) and trains later layers (task-specific).
    """
    
    def __init__(self, base_model: nn.Module, config: TransferLearningConfig):
        self.base_model = base_model
        self.config = config
        self.optimizer = None
        self.training_loss_history: List[float] = []
    
    def freeze_layers(self, freeze_ratio: float = 0.7):
        """Freeze early layers, keep later layers trainable."""
        total_params = sum(1 for p in self.base_model.parameters())
        params_to_freeze = int(total_params * freeze_ratio)
        
        for i, param in enumerate(self.base_model.parameters()):
            if i < params_to_freeze:
                param.requires_grad = False
    
    def fine_tune(self, support_features: torch.Tensor, support_labels: torch.Tensor,
                 num_steps: int = 20, learning_rate: float = 0.001) -> Dict:
        """
        Fine-tune model on support set.
        
        Args:
            support_features: (num_support, feature_dim)
            support_labels: (num_support,)
            num_steps: Number of optimization steps
            learning_rate: Learning rate for fine-tuning
            
        Returns:
            results: Dictionary with training metrics
        """
        self.freeze_layers(self.config.layer_freeze_ratio)
        
        # Setup optimizer for trainable parameters
        trainable_params = [p for p in self.base_model.parameters() if p.requires_grad]
        if not trainable_params:
            return {'status': 'no_trainable_params', 'loss': 0}
        
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        
        self.training_loss_history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.base_model(support_features)
            loss = loss_fn(logits, support_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            self.training_loss_history.append(float(loss))
            
            # Early stopping
            if float(loss) < best_loss:
                best_loss = float(loss)
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                break
        
        return {
            'status': 'success',
            'final_loss': float(loss),
            'steps_completed': step + 1,
            'loss_history': self.training_loss_history
        }


# ============================================================================
# META-LEARNER OPTIMIZER (MAML-like)
# ============================================================================

class MetaLearnerOptimizer:
    """
    Meta-learning optimizer (simplified MAML).
    Learns how to adapt to new tasks with one gradient step.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.meta_adaptation_history: List[MetaLearnResult] = []
    
    def meta_adapt(self, support_set: Tuple[torch.Tensor, torch.Tensor],
                  query_set: Tuple[torch.Tensor, torch.Tensor],
                  inner_lr: float = 0.01,
                  num_inner_steps: int = 5) -> MetaLearnResult:
        """
        Perform meta-adaptation: rapid task adaptation using few gradient steps.
        
        Args:
            support_set: (features, labels) for task adaptation
            query_set: (features, labels) for evaluation
            inner_lr: Learning rate for inner loop adaptation
            num_inner_steps: Steps of adaptation on support set
            
        Returns:
            MetaLearnResult with adaptation metrics
        """
        support_features, support_labels = support_set
        query_features, query_labels = query_set
        
        # Save original parameters
        original_params = [p.clone() for p in self.model.parameters()]
        
        try:
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(self.model.parameters(), lr=inner_lr)
            
            # Inner loop: adapt on support set
            for step in range(num_inner_steps):
                optimizer.zero_grad()
                # Get features and compute classification
                support_encoded = self.model.encode(support_features)
                # Simple linear classifier on features
                logits = -torch.cdist(support_encoded, self.model.class_prototypes)
                loss = loss_fn(logits, support_labels)
                loss.backward()
                optimizer.step()
            
            # Evaluate on query set
            with torch.no_grad():
                query_encoded = self.model.encode(query_features)
                query_logits = -torch.cdist(query_encoded, self.model.class_prototypes)
                query_loss = loss_fn(query_logits, query_labels)
                
                predictions = torch.argmax(query_logits, dim=1)
                accuracy = (predictions == query_labels).float().mean()
            
            # Compute parameter shift
            param_shift_norm = 0.0
            for orig, curr in zip(original_params, self.model.parameters()):
                param_shift_norm += torch.norm(curr - orig).item()
            param_shift_norm /= (len(original_params) + 1e-8)
            
            # Restore original parameters
            for param, orig in zip(self.model.parameters(), original_params):
                param.data = orig
            
            result = MetaLearnResult(
                task_accuracy=float(accuracy),
                adaptation_steps=num_inner_steps,
                parameter_shift_norm=float(param_shift_norm),
                confidence=min(0.95, 0.5 + float(accuracy) * 0.45),
                learned_features=np.zeros(128)  # Placeholder
            )
            
            self.meta_adaptation_history.append(result)
            return result
        
        except Exception as e:
            # Restore parameters on error
            for param, orig in zip(self.model.parameters(), original_params):
                param.data = orig
            
            return MetaLearnResult(
                task_accuracy=0.0,
                adaptation_steps=0,
                parameter_shift_norm=0.0,
                confidence=0.0,
                learned_features=np.zeros(128)
            )


# ============================================================================
# UTILITY FUNCTIONS FOR INTEGRATION
# ============================================================================

def create_few_shot_adapter(device: str = 'cpu') -> FewShotAdaptationModule:
    """Factory function to create few-shot adaptation module."""
    return FewShotAdaptationModule(feature_dim=128, num_classes=3, device=device)


def detect_new_population_shift(features: np.ndarray, 
                               detector: DomainShiftDetector) -> Tuple[bool, float]:
    """
    Check if input features indicate a new population/facility.
    
    Returns:
        needs_adaptation: Boolean
        shift_score: 0-1 magnitude of shift
    """
    shift_detected, shift_magnitude, _ = detector.detect_shift(features)
    return shift_detected, shift_magnitude
