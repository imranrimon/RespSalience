import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedBarlowTwinsLoss(nn.Module):
    """
    Supervised Barlow Twins Loss (SPMat-BT).
    
    Ref: "Supervised Pretraining for Material Property Prediction" (2025).
    Adapts Barlow Twins to use Sample-Wise Cross-Correlation with Surrogate Labels.
    
    Objective:
    - Pull Same-Class samples to correlation +1.
    - Push Diff-Class samples to correlation -1.
    """
    def __init__(self, lambda_coeff=0.005):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def forward(self, z1, z2, labels):
        """
        Args:
            z1: (B, D) embeddings from View 1
            z2: (B, D) embeddings from View 2
            labels: (B) class labels (int or float)
            
        Returns:
            loss: scalar tensor
        """
        device = z1.device
        batch_size = z1.shape[0]
        
        # 1. Normalize embeddings (L2 norm per sample)
        # Result: Vectors on unit hypersphere. Dot product = Cosine Similarity.
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        # 2. Compute Sample-Wise Cross-Correlation (Cosine Similarity)
        # S_ij = z1_i . z2_j
        # Shape: (B, B)
        S = torch.mm(z1_norm, z2_norm.T)
        
        if labels is None:
            # Self-Supervised Mode (SSL)
            # Treat each sample as its own class -> M is Identity
            M = torch.eye(batch_size, device=device)
            
            # Term 1: Self-Similarity -> 1
            same_class_loss = (M * (1 - S).pow(2)).sum()
            
            # Term 2: Diff-Sample -> 0 (Orthogonal)
            # Note: Approaching -1 is impossible for N > 2 samples. 
            # So for SSL we target 0 (Decorrelation).
            diff_class_loss = ((1 - M) * (S).pow(2)).sum()
            
        else:
            # Supervised Mode (SPMat)
            labels = labels.view(-1, 1)
            M = torch.eq(labels, labels.T).float().to(device)
            
            # Term 1: Same Class -> Align to +1
            # Loss = (1 - S)^2
            same_class_loss = (M * (1 - S).pow(2)).sum()
            
            # Term 2: Diff Class -> Align to -1 (Anti-correlation)
            # Loss = (1 + S)^2. (Minimizes when S = -1)
            diff_class_loss = ((1 - M) * (1 + S).pow(2)).sum()
        
        # Normalize by batch size?
        # The paper sums over indices. But usually we mean over batch.
        # Eq 14 is just Sum. 
        # But for stability we might want to mean.
        # I'll stick to Sum but usually optimizers prefer Mean. 
        # Since B^2 terms, loss scales with B^2.
        # I'll Normalize by B^2 to keep LR stable.
        
        loss = same_class_loss + self.lambda_coeff * diff_class_loss
        loss = loss / (batch_size ** 2)
        
        return loss