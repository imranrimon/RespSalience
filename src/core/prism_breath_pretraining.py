"""Self-supervised pretraining utilities for PRISM-Breath encoders."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.losses import SupervisedBarlowTwinsLoss
from src.core.prism_breath_model import MODALITY_NAMES, PrismBreathConfig, RespiratorySignalEncoder


class PrismBreathPretrainer(nn.Module):
    """
    RespMAE-style masked pretraining for the same encoder family used downstream.

    The goal is not to match the legacy SSL backbone, but to pretrain the exact
    branch encoders that PRISM-Breath will later fine-tune.
    """

    def __init__(self, config: PrismBreathConfig, projection_dim: int | None = None):
        super().__init__()
        self.config = config
        self.modalities = MODALITY_NAMES
        proj_dim = int(projection_dim or config.d_model)

        encoder_kwargs = {
            "seq_len": config.seq_len,
            "d_model": config.d_model,
            "use_scan_block": config.use_scan_block,
            "use_spectral_features": config.use_spectral_features,
            "compact_encoder": config.compact_encoder,
        }
        self.branches = nn.ModuleDict({name: RespiratorySignalEncoder(**encoder_kwargs) for name in self.modalities})
        self.reconstruction_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv1d(config.d_model, config.d_model, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(config.d_model, max(16, config.d_model // 2), kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(max(16, config.d_model // 2), 1, kernel_size=3, padding=1),
                )
                for name in self.modalities
            }
        )
        self.projection_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(config.d_model, config.d_model),
                    nn.GELU(),
                    nn.Linear(config.d_model, proj_dim),
                )
                for name in self.modalities
            }
        )
        self.barlow = SupervisedBarlowTwinsLoss(lambda_coeff=0.01)

    def encode(self, signals: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {name: self.branches[name](signals[name], return_sequence=True) for name in self.modalities}

    def reconstruct(self, branch_outputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        recon = {}
        for name in self.modalities:
            seq = branch_outputs[name]["sequence"].transpose(1, 2)
            recon[name] = self.reconstruction_heads[name](seq).squeeze(1)
        return recon

    def pooled_projection(
        self,
        branch_outputs: Dict[str, Dict[str, torch.Tensor]],
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        projected = []
        for idx, name in enumerate(self.modalities):
            emb = self.projection_heads[name](branch_outputs[name]["embedding"])
            projected.append(emb * modality_mask[:, idx].unsqueeze(-1))
        stacked = torch.stack(projected, dim=1)
        denom = modality_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        fused = stacked.sum(dim=1) / denom
        return F.normalize(fused, dim=-1)

    def masked_reconstruction_loss(
        self,
        recon: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for idx, name in enumerate(self.modalities):
            sample_mask = modality_mask[:, idx].unsqueeze(-1)
            valid_mask = masks[name] * sample_mask
            denom = valid_mask.sum().clamp(min=1.0)
            losses.append((((recon[name] - targets[name]) ** 2) * valid_mask).sum() / denom)
        return torch.stack(losses).mean()

    def contrastive_loss(
        self,
        view_one: Dict[str, Dict[str, torch.Tensor]],
        view_two: Dict[str, Dict[str, torch.Tensor]],
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        z1 = self.pooled_projection(view_one, modality_mask=modality_mask)
        z2 = self.pooled_projection(view_two, modality_mask=modality_mask)
        return self.barlow(z1, z2, labels=None)

    def export_branch_state(self) -> Dict[str, torch.Tensor]:
        return self.branches.state_dict()
