"""
PRISM-Breath respiratory models and ablation variants.

Design goals:
- physiology-grounded signal summaries instead of black-box late fusion
- reliability-aware multimodal weighting under disagreement
- selective tachypnea triage with uncertainty intervals
- prototype and counterfactual evidence hooks for explainability
- configurable ablations for expert routing, scan blocks, spectral features,
  compact small-data baselines, and CodePhys-style clean-breathing memory
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITY_NAMES = ("depth", "flow", "audio")


def tensor_tachypnea_threshold(age_months: torch.Tensor) -> torch.Tensor:
    age = age_months.float()
    return torch.where(
        age < 2.0,
        torch.full_like(age, 60.0),
        torch.where(
            age < 12.0,
            torch.full_like(age, 50.0),
            torch.where(age <= 60.0, torch.full_like(age, 40.0), torch.full_like(age, 20.0)),
        ),
    )


@dataclass
class PrismBreathConfig:
    seq_len: int = 300
    d_model: int = 64
    num_experts: int = 3
    num_prototypes: int = 6
    codebook_size: int = 12
    model_variant: str = "prism_breath"
    use_expert_routing: bool = True
    use_scan_block: bool = True
    use_spectral_features: bool = True
    use_clean_codebook: bool = True
    use_prototypes: bool = True
    use_domain_adapter: bool = True
    compact_encoder: bool = False
    # Disagreement-gated frequency-forward fusion prior (#22): under channel disagreement,
    # favor the higher-frequency available modality. Monocular respiration proxies (esp. the
    # depth intensity proxy) fail by UNDER-counting fast shallow breaths far more than by
    # over-counting, so a lone high-reading channel is the trustworthy one. Default off.
    use_freq_forward_fusion: bool = False
    freq_forward_gamma: float = 3.0
    freq_forward_temp: float = 8.0
    # Heteroscedastic (aleatoric) uncertainty head: predict log-variance of the RR estimate so the
    # model can be UN-confident where it is wrong (esp. neonatal/high-RR). Enables uncertainty-based
    # selective deployment that the qhat/disagreement signals could not (model was confidently wrong).
    use_heteroscedastic: bool = False
    # SNR-gated fusion: ground the reliability weights in the COMPUTED respiratory-band spectral SNR
    # of each modality's input signal, instead of only the free-floating learned quality head (which
    # collapsed to a near-constant 0.49/0.51 on pedrr, so the "reliability" fusion never reweighted).
    use_snr_fusion: bool = False
    num_domains: int = 8
    active_modalities: List[str] = field(default_factory=lambda: list(MODALITY_NAMES))

    @classmethod
    def from_variant(cls, model_variant: str = "prism_breath", **overrides) -> "PrismBreathConfig":
        config = cls(model_variant=model_variant)
        variant = str(model_variant).strip().lower()

        if variant == "prism_shared_trunk":
            config.use_expert_routing = False
        elif variant == "prism_no_scan":
            config.use_scan_block = False
        elif variant == "prism_no_spectral":
            config.use_spectral_features = False
        elif variant == "prism_no_codebook":
            config.use_clean_codebook = False
        elif variant == "prism_no_prototypes":
            config.use_prototypes = False
        elif variant == "prism_lean":
            # Load-bearing audit (2026-08): the prototype bank, clean codebook, and het head are
            # decorative/inert on the clinical endpoint. Drop them ALL together; KEEP the load-bearing
            # scan block + spectral features + separate trunks + RR/tachy heads. Occam's lean model.
            config.use_clean_codebook = False
            config.use_prototypes = False
            config.use_heteroscedastic = False
        elif variant == "airflow_compact":
            config.use_expert_routing = False
            config.use_scan_block = False
            config.use_spectral_features = False
            config.use_clean_codebook = False
            config.use_prototypes = False
            config.use_domain_adapter = False
            config.compact_encoder = True
            config.num_experts = 1
            config.active_modalities = ["depth", "flow"]
        elif variant in ("cliffphys", "mamba"):
            # External-method baselines: keep the shared reliability-fusion +
            # triage protocol but disable PRISM-Breath-specific modules so the
            # comparison isolates the backbone encoder. The distinct temporal
            # backbone is supplied by the encoder swap in
            # src/core/baseline_wrappers.py (dispatched by the factory below).
            config.use_expert_routing = False
            config.use_scan_block = False
            config.use_clean_codebook = False
            config.use_prototypes = False
            config.use_domain_adapter = False
            if variant == "cliffphys":
                # CliffPhys couples a monocular-depth scalar field with an
                # optical-flow vector field; audio is out of scope for it.
                config.active_modalities = ["depth", "flow"]
        elif variant != "prism_breath":
            raise ValueError(
                f"Unknown model_variant={model_variant!r}. "
                "Expected one of: prism_breath, prism_shared_trunk, prism_no_scan, "
                "prism_no_spectral, prism_no_codebook, prism_no_prototypes, prism_lean, airflow_compact, "
                "cliffphys, mamba."
            )

        valid_keys = {item.name for item in fields(cls)}
        for key, value in overrides.items():
            if key in valid_keys and value is not None:
                setattr(config, key, value)
        return config

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["active_modalities"] = list(self.active_modalities)
        return payload

    @staticmethod
    def from_dict(payload: Optional[Dict]) -> "PrismBreathConfig":
        if payload is None:
            return PrismBreathConfig()
        valid_keys = {item.name for item in fields(PrismBreathConfig)}
        model_variant = payload.get("model_variant", "prism_breath")
        overrides = {key: payload[key] for key in valid_keys if key in payload and key != "model_variant"}
        if "use_domain_adapter" not in payload:
            overrides["use_domain_adapter"] = False
        return PrismBreathConfig.from_variant(model_variant=model_variant, **overrides)

    def model_name(self) -> str:
        mapping = {
            "prism_breath": "PRISM-Breath",
            "prism_shared_trunk": "PRISM-Breath-Shared",
            "prism_no_scan": "PRISM-Breath-NoScan",
            "prism_no_spectral": "PRISM-Breath-NoSpectral",
            "prism_no_codebook": "PRISM-Breath-NoCodebook",
            "prism_no_prototypes": "PRISM-Breath-NoPrototypes",
            "airflow_compact": "AIRFlow-Compact",
            "cliffphys": "CliffPhys-Signal",
            "mamba": "Mamba-RR",
        }
        return mapping.get(self.model_variant, self.model_variant)


class AdaptiveScanBlock(nn.Module):
    """A lightweight state-space-inspired scan for respiratory sequences."""

    def __init__(self, d_model: int):
        super().__init__()
        self.value_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        values = torch.tanh(self.value_proj(x))
        gates = torch.sigmoid(self.gate_proj(x))
        state = torch.zeros(batch_size, d_model, device=x.device, dtype=x.dtype)
        outputs = []
        for step in range(seq_len):
            gate_t = gates[:, step]
            state = gate_t * state + (1.0 - gate_t) * values[:, step]
            outputs.append(state)
        scanned = torch.stack(outputs, dim=1)
        return self.out_proj(scanned)


class RespiratorySignalEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int = 300,
        d_model: int = 64,
        fft_bins: int = 96,
        use_scan_block: bool = True,
        use_spectral_features: bool = True,
        compact_encoder: bool = False,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.fft_bins = int(fft_bins)
        self.stats_dim = 8
        self.use_scan_block = bool(use_scan_block)
        self.use_spectral_features = bool(use_spectral_features)

        hidden = max(16, d_model // 3) if compact_encoder else max(32, d_model // 2)
        kernel_sizes = [3, 7] if compact_encoder else [3, 7, 15]

        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=7 if compact_encoder else 9, padding=3 if compact_encoder else 4),
            nn.GELU(),
            nn.Conv1d(hidden, d_model, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.multi_scale = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.scan = AdaptiveScanBlock(d_model=d_model) if self.use_scan_block else nn.Identity()
        self.freq_proj = nn.Sequential(
            nn.Linear(self.fft_bins, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.stats_proj = nn.Sequential(
            nn.Linear(self.stats_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.embedding_proj = nn.Sequential(
            nn.Linear(d_model * 3 + hidden, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        head_dim = d_model + self.stats_dim
        self.bpm_head = nn.Sequential(nn.Linear(head_dim, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.scale_head = nn.Sequential(nn.Linear(head_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.quality_head = nn.Sequential(nn.Linear(head_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def _encode_sequence(self, signal: torch.Tensor) -> torch.Tensor:
        """Map a 1D signal to a temporal feature sequence ``(B, seq_len, d_model)``.

        This is the only architecture-specific step of the encoder. Baseline
        backbones (e.g. a selective state-space scan or Clifford geometric layers)
        subclass this encoder and override this method, inheriting the shared
        spectral/statistical heads so their per-modality output contract
        (``embedding``/``bpm``/``scale``/``quality``/``stats``) stays identical.
        """
        x = signal.unsqueeze(1)
        stem = self.stem(x)
        scale_features = sum(conv(stem) for conv in self.multi_scale) / float(len(self.multi_scale))
        seq = scale_features.transpose(1, 2)
        if self.use_scan_block:
            seq = seq + self.scan(seq)
        return seq

    def forward(self, signal: torch.Tensor, return_sequence: bool = False) -> Dict[str, torch.Tensor]:
        seq = self._encode_sequence(signal)

        pooled_mean = seq.mean(dim=1)
        pooled_max = seq.amax(dim=1)
        stats = self._signal_stats(signal)
        stats_embed = self.stats_proj(stats)

        if self.use_spectral_features:
            freq_features = self._spectral_features(signal)
            freq_embed = self.freq_proj(freq_features)
        else:
            freq_embed = torch.zeros_like(pooled_mean)

        embedding = self.embedding_proj(torch.cat([pooled_mean, pooled_max, freq_embed, stats_embed], dim=-1))
        head_input = torch.cat([embedding, stats], dim=-1)
        bpm = self.bpm_head(head_input).squeeze(-1)
        scale = F.softplus(self.scale_head(head_input).squeeze(-1)) + 1e-3
        quality = torch.sigmoid(self.quality_head(head_input).squeeze(-1))

        outputs = {
            "embedding": embedding,
            "bpm": bpm,
            "scale": scale,
            "quality": quality,
            "stats": stats,
        }
        if return_sequence:
            outputs["sequence"] = seq
        return outputs

    def _spectral_features(self, signal: torch.Tensor) -> torch.Tensor:
        centered = signal - signal.mean(dim=-1, keepdim=True)
        spectrum = torch.abs(torch.fft.rfft(centered, dim=-1))
        if spectrum.shape[-1] > 1:
            spectrum = spectrum[..., 1:]
        spectrum = torch.log1p(spectrum)
        if spectrum.shape[-1] >= self.fft_bins:
            spectrum = spectrum[..., : self.fft_bins]
        else:
            spectrum = F.pad(spectrum, (0, self.fft_bins - spectrum.shape[-1]))
        norm = spectrum.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return spectrum / norm

    def _signal_stats(self, signal: torch.Tensor) -> torch.Tensor:
        centered = signal - signal.mean(dim=-1, keepdim=True)
        std = centered.std(dim=-1, keepdim=True).clamp(min=1e-6)
        diff = centered[:, 1:] - centered[:, :-1]

        zero_cross = (centered[:, 1:] * centered[:, :-1] < 0).float().mean(dim=-1, keepdim=True)
        lag1 = (centered[:, :-1] * centered[:, 1:]).mean(dim=-1, keepdim=True) / (
            std.squeeze(-1).pow(2).unsqueeze(-1)
        )

        spectrum = torch.abs(torch.fft.rfft(centered, dim=-1))
        if spectrum.shape[-1] > 1:
            spectrum = spectrum[..., 1:]
        total_power = spectrum.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        spectral_prob = spectrum / total_power
        spectral_entropy = -(spectral_prob * torch.log(spectral_prob.clamp(min=1e-6))).sum(dim=-1, keepdim=True)
        spectral_entropy = spectral_entropy / math.log(max(2, spectrum.shape[-1]))
        dominant_idx = spectrum.argmax(dim=-1, keepdim=True).float() / max(1, spectrum.shape[-1] - 1)
        dominant_power = spectrum.amax(dim=-1, keepdim=True) / total_power

        return torch.cat(
            [
                centered.abs().mean(dim=-1, keepdim=True),
                std,
                diff.abs().mean(dim=-1, keepdim=True),
                zero_cross,
                lag1,
                spectral_entropy,
                dominant_idx,
                dominant_power,
            ],
            dim=-1,
        )


class PrototypeEvidenceBank(nn.Module):
    def __init__(self, d_model: int, num_prototypes: int = 6, temperature: float = 0.35):
        super().__init__()
        self.temperature = float(temperature)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))
        labels = [0] * (num_prototypes // 2) + [1] * (num_prototypes - num_prototypes // 2)
        self.register_buffer("prototype_labels", torch.tensor(labels, dtype=torch.long))

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        prototypes = F.normalize(self.prototypes, dim=-1)
        emb = F.normalize(embedding, dim=-1)
        similarity = emb @ prototypes.T
        weights = torch.softmax(similarity / self.temperature, dim=-1)
        context = weights @ prototypes
        return {"similarity": similarity, "weights": weights, "context": context}


class SourceAgeAdapter(nn.Module):
    """Lightweight domain adapter conditioned on source dataset and age."""

    def __init__(self, d_model: int, num_domains: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        embed_dim = max(8, d_model // 4)
        self.domain_embedding = nn.Embedding(num_domains, embed_dim)
        self.condition = nn.Sequential(
            nn.Linear(embed_dim + 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model * 2),
        )

    def forward(self, x: torch.Tensor, dataset_ids: torch.Tensor, age_months: torch.Tensor) -> torch.Tensor:
        dataset_ids = dataset_ids.long().clamp(min=0, max=self.domain_embedding.num_embeddings - 1)
        age_linear = age_months.float().div(60.0).clamp(min=0.0, max=5.0).unsqueeze(-1)
        age_log = torch.log1p(age_months.float()).div(math.log(301.0)).unsqueeze(-1)
        cond = torch.cat([self.domain_embedding(dataset_ids), age_linear, age_log], dim=-1)
        gamma, beta = self.condition(cond).chunk(2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)

        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return self.norm(x) * (1.0 + gamma) + beta


class CleanBreathingCodebook(nn.Module):
    """
    CodePhys-inspired denoising memory.

    The memory stores a compact bank of clean respiratory patterns and retrieves
    a reliability-conditioned context vector for each sample.
    """

    def __init__(self, d_model: int, num_codes: int = 12, temperature: float = 0.3):
        super().__init__()
        self.temperature = float(temperature)
        self.codes = nn.Parameter(torch.randn(num_codes, d_model))
        self.reliability_proj = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model + 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        embedding: torch.Tensor,
        quality_mean: torch.Tensor,
        disagreement: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        codes = F.normalize(self.codes, dim=-1)
        emb = F.normalize(embedding, dim=-1)
        reliability_context = torch.stack([quality_mean, 1.0 - disagreement], dim=-1)
        reliability_bias = self.reliability_proj(reliability_context)

        similarity = (emb + 0.1 * F.normalize(reliability_bias, dim=-1)) @ codes.T
        weights = torch.softmax(similarity / self.temperature, dim=-1)
        context = weights @ codes

        gate = torch.sigmoid(self.gate(torch.cat([embedding, reliability_context], dim=-1))).squeeze(-1)
        refined = embedding + gate.unsqueeze(-1) * (context - embedding)

        commitment = F.mse_loss(embedding, context.detach(), reduction="none").mean(dim=-1)
        codebook_loss = F.mse_loss(context, embedding.detach(), reduction="none").mean(dim=-1)
        top_val, top_idx = torch.max(weights, dim=-1)

        return {
            "context": context,
            "refined": refined,
            "weights": weights,
            "similarity": similarity,
            "gate": gate,
            "commitment_loss": commitment,
            "codebook_loss": codebook_loss,
            "top_idx": top_idx,
            "top_weight": top_val,
        }


class PrismBreathModel(nn.Module):
    """
    Physiology-grounded, reliability-aware, selective model for the paper-ready task.

    PRISM = Physiology-grounded Reliability-aware Interpretable Selective Model
    """

    def __init__(
        self,
        seq_len: int = 300,
        d_model: int = 64,
        num_experts: int = 3,
        num_prototypes: int = 6,
        codebook_size: int = 12,
        model_variant: str = "prism_breath",
        use_expert_routing: Optional[bool] = None,
        use_scan_block: Optional[bool] = None,
        use_spectral_features: Optional[bool] = None,
        use_clean_codebook: Optional[bool] = None,
        use_prototypes: Optional[bool] = None,
        compact_encoder: Optional[bool] = None,
        active_modalities: Optional[Sequence[str]] = None,
        config: Optional[PrismBreathConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = PrismBreathConfig.from_variant(
                model_variant=model_variant,
                seq_len=seq_len,
                d_model=d_model,
                num_experts=num_experts,
                num_prototypes=num_prototypes,
                codebook_size=codebook_size,
                use_expert_routing=use_expert_routing,
                use_scan_block=use_scan_block,
                use_spectral_features=use_spectral_features,
                use_clean_codebook=use_clean_codebook,
                use_prototypes=use_prototypes,
                compact_encoder=compact_encoder,
                active_modalities=list(active_modalities) if active_modalities is not None else None,
            )
        self.config = config
        self.seq_len = config.seq_len
        self.d_model = config.d_model
        self.modalities = MODALITY_NAMES

        encoder_kwargs = {
            "seq_len": config.seq_len,
            "d_model": config.d_model,
            "use_scan_block": config.use_scan_block,
            "use_spectral_features": config.use_spectral_features,
            "compact_encoder": config.compact_encoder,
        }
        self.branches = nn.ModuleDict({name: RespiratorySignalEncoder(**encoder_kwargs) for name in self.modalities})
        self.domain_adapter = (
            SourceAgeAdapter(d_model=config.d_model, num_domains=config.num_domains) if config.use_domain_adapter else None
        )

        active_mask = [1.0 if name in set(config.active_modalities) else 0.0 for name in self.modalities]
        self.register_buffer("active_modality_mask", torch.tensor(active_mask, dtype=torch.float32))

        self.meta_dim = len(self.modalities) * 5 + 2
        fusion_input_dim = len(self.modalities) * config.d_model + self.meta_dim

        if config.use_expert_routing:
            self.expert_gate = nn.Sequential(
                nn.Linear(self.meta_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.num_experts),
            )
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(fusion_input_dim, config.d_model),
                        nn.GELU(),
                        nn.Dropout(0.1),
                        nn.Linear(config.d_model, config.d_model),
                    )
                    for _ in range(config.num_experts)
                ]
            )
            self.shared_fusion = None
        else:
            self.expert_gate = None
            self.experts = None
            self.shared_fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, config.d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.d_model, config.d_model),
            )

        self.clean_codebook = (
            CleanBreathingCodebook(d_model=config.d_model, num_codes=config.codebook_size)
            if config.use_clean_codebook
            else None
        )
        self.prototype_bank = (
            PrototypeEvidenceBank(d_model=config.d_model, num_prototypes=config.num_prototypes)
            if config.use_prototypes
            else None
        )

        fused_dim = config.d_model + self.meta_dim
        hidden = max(16, config.d_model // 2)
        self.rr_correction_head = nn.Sequential(
            nn.Linear(fused_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        self.interval_head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Heteroscedastic aleatoric head: predicts log-variance of the RR estimate (always built so
        # checkpoints stay compatible; used by the Gaussian-NLL term only when het_weight > 0).
        self.rr_logvar_head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.tachy_head = nn.Sequential(
            nn.Linear(fused_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.review_head = nn.Sequential(
            nn.Linear(fused_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Respiratory-event heads: detect a respiratory pause/apnea and detect
        # motion-corrupted input. Conditioned on the fused state plus disagreement
        # and inverse-quality context (motion drives disagreement/low quality;
        # apnea drives low dominant spectral power captured in the fused state).
        self.apnea_head = nn.Sequential(
            nn.Linear(fused_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.motion_head = nn.Sequential(
            nn.Linear(fused_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def export_config(self) -> Dict:
        return self.config.to_dict()

    def model_name(self) -> str:
        return self.config.model_name()

    def _aux_placeholders(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = tensor.shape[0]
        device = tensor.device
        proto_shape = (batch_size, max(1, int(self.config.num_prototypes)))
        code_shape = (batch_size, max(1, int(self.config.codebook_size)))
        return {
            "prototype_similarity": torch.zeros(proto_shape, device=device, dtype=tensor.dtype),
            "prototype_weights": torch.zeros(proto_shape, device=device, dtype=tensor.dtype),
            "clean_code_similarity": torch.zeros(code_shape, device=device, dtype=tensor.dtype),
            "clean_code_weights": torch.zeros(code_shape, device=device, dtype=tensor.dtype),
            "clean_code_gate": torch.zeros(batch_size, device=device, dtype=tensor.dtype),
            "clean_commitment_loss": torch.zeros(batch_size, device=device, dtype=tensor.dtype),
            "clean_codebook_loss": torch.zeros(batch_size, device=device, dtype=tensor.dtype),
            "clean_code_idx": torch.zeros(batch_size, device=device, dtype=torch.long),
            "clean_code_weight": torch.zeros(batch_size, device=device, dtype=tensor.dtype),
        }

    def forward(
        self,
        depth: torch.Tensor,
        flow: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        age_months: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        dataset_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = depth.shape[0]
        device = depth.device
        if audio is None:
            audio = torch.zeros_like(depth)

        if age_months is None:
            age_months = torch.full((batch_size,), 6.0, device=device, dtype=depth.dtype)
        if dataset_ids is None:
            dataset_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            dataset_ids = dataset_ids.to(device=device, dtype=torch.long)

        if modality_mask is None:
            modality_mask = torch.ones(batch_size, len(self.modalities), device=device, dtype=depth.dtype)
        modality_mask = modality_mask.float() * self.active_modality_mask.to(device=device, dtype=depth.dtype)

        no_modality = modality_mask.sum(dim=-1, keepdim=True) <= 0
        if no_modality.any():
            modality_mask = modality_mask.clone()
            modality_mask[no_modality.expand_as(modality_mask)] = 0.0
            fallback_idx = 0
            for idx, name in enumerate(self.modalities):
                if name in set(self.config.active_modalities):
                    fallback_idx = idx
                    break
            modality_mask[no_modality.squeeze(-1), fallback_idx] = 1.0

        signals = {"depth": depth, "flow": flow, "audio": audio}
        branch_outputs = {name: self.branches[name](signals[name]) for name in self.modalities}
        embeddings = torch.stack([branch_outputs[name]["embedding"] for name in self.modalities], dim=1)
        if self.domain_adapter is not None:
            embeddings = self.domain_adapter(embeddings, dataset_ids=dataset_ids, age_months=age_months)
        bpm = torch.stack([branch_outputs[name]["bpm"] for name in self.modalities], dim=1)
        scales = torch.stack([branch_outputs[name]["scale"] for name in self.modalities], dim=1)
        qualities = torch.stack([branch_outputs[name]["quality"] for name in self.modalities], dim=1)
        stats = torch.stack([branch_outputs[name]["stats"] for name in self.modalities], dim=1)

        pairwise_diff = torch.abs(bpm.unsqueeze(2) - bpm.unsqueeze(1)) / 10.0
        agreement = torch.exp(-pairwise_diff)
        pair_mask = modality_mask.unsqueeze(2) * modality_mask.unsqueeze(1)
        eye = torch.eye(len(self.modalities), device=device, dtype=agreement.dtype).unsqueeze(0)
        agreement = agreement * pair_mask + eye * (1.0 - pair_mask)

        support = (agreement * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp(min=1.0)

        offdiag_mask = pair_mask * (1.0 - eye)
        agreement_sum = (agreement * offdiag_mask).sum(dim=(1, 2))
        agreement_count = offdiag_mask.sum(dim=(1, 2)).clamp(min=1.0)
        disagreement = 1.0 - agreement_sum / agreement_count

        reliability = qualities * support / scales.clamp(min=1e-3)
        if getattr(self.config, "use_snr_fusion", False):
            # Ground reliability in the COMPUTED respiratory-band spectral SNR of each input signal
            # (band-power 0.15-1.5 Hz / total power), so a channel whose signal is actually
            # respiratory gets up-weighted and a degenerate/broken channel gets down-weighted --
            # unlike the learned quality head, which collapsed to a constant ~0.5 on pedrr.
            def _band_snr(x):
                if x.dim() == 3:
                    x = x.mean(dim=1)               # (B,C,T) -> (B,T)
                x = x - x.mean(dim=-1, keepdim=True)
                T = x.shape[-1]
                w = torch.hann_window(T, device=x.device, dtype=x.dtype)
                P = torch.fft.rfft(x * w, dim=-1).abs() ** 2
                f = torch.fft.rfftfreq(T, d=1.0 / 25.0).to(x.device)
                band = (f >= 0.15) & (f <= 1.5)
                return (P[..., band].sum(-1) / P.sum(-1).clamp(min=1e-8)).clamp(1e-4, 1.0)
            snr = torch.stack([_band_snr(signals[name]) for name in self.modalities], dim=1)
            reliability = reliability * snr.clamp(min=1e-3)
        if getattr(self.config, "use_freq_forward_fusion", False):
            # #22: under disagreement, add a logit favoring the higher-frequency available
            # modality (soft, temperature-scaled), so a lone high-reading channel is trusted
            # instead of averaged down. Inert when channels agree (disagreement -> 0) or when
            # only one modality is available (softmax over a single finite entry is 1.0).
            bpm_avail = bpm.masked_fill(modality_mask <= 0, float("-inf"))
            hf_pref = torch.softmax(bpm_avail / float(self.config.freq_forward_temp), dim=-1)
            reliability = reliability + float(self.config.freq_forward_gamma) * disagreement.unsqueeze(-1) * hf_pref
        reliability = reliability.masked_fill(modality_mask <= 0, float("-inf"))
        fusion_weights = torch.softmax(reliability, dim=-1)
        quality_mean = (qualities * modality_mask).sum(dim=-1) / modality_mask.sum(dim=-1).clamp(min=1.0)

        flat_embeddings = torch.cat(
            [embeddings[:, idx] * modality_mask[:, idx].unsqueeze(-1) for idx in range(len(self.modalities))],
            dim=-1,
        )
        meta = torch.cat(
            [
                bpm,
                scales,
                qualities,
                modality_mask,
                fusion_weights,
                disagreement.unsqueeze(-1),
                quality_mean.unsqueeze(-1),
            ],
            dim=-1,
        )

        fusion_input = torch.cat([flat_embeddings, meta], dim=-1)
        if self.experts is not None and self.expert_gate is not None:
            expert_gates = torch.softmax(self.expert_gate(meta), dim=-1)
            expert_stack = torch.stack([expert(fusion_input) for expert in self.experts], dim=1)
            latent = (expert_gates.unsqueeze(-1) * expert_stack).sum(dim=1)
        else:
            expert_gates = torch.ones(batch_size, 1, device=device, dtype=depth.dtype)
            latent = self.shared_fusion(fusion_input)

        aux = self._aux_placeholders(depth)
        if self.clean_codebook is not None:
            clean = self.clean_codebook(latent, quality_mean=quality_mean, disagreement=disagreement)
            latent = clean["refined"]
            aux.update(
                {
                    "clean_code_similarity": clean["similarity"],
                    "clean_code_weights": clean["weights"],
                    "clean_code_gate": clean["gate"],
                    "clean_commitment_loss": clean["commitment_loss"],
                    "clean_codebook_loss": clean["codebook_loss"],
                    "clean_code_idx": clean["top_idx"],
                    "clean_code_weight": clean["top_weight"],
                }
            )

        if self.prototype_bank is not None:
            proto = self.prototype_bank(latent)
            latent = latent + proto["context"]
            aux.update(
                {
                    "prototype_similarity": proto["similarity"],
                    "prototype_weights": proto["weights"],
                }
            )

        fused_input = torch.cat([latent, meta], dim=-1)
        weighted_rr = (fusion_weights * bpm).sum(dim=-1)
        rr_pred = weighted_rr + self.rr_correction_head(fused_input).squeeze(-1)
        rr_logvar = self.rr_logvar_head(fused_input).squeeze(-1).clamp(-6.0, 8.0)  # aleatoric log-var
        qhat = (fusion_weights * scales).sum(dim=-1) + F.softplus(self.interval_head(fused_input).squeeze(-1))

        threshold = tensor_tachypnea_threshold(age_months)
        rr_margin = rr_pred - threshold
        tachy_logit = self.tachy_head(torch.cat([fused_input, rr_margin.unsqueeze(-1)], dim=-1)).squeeze(-1)
        review_logit = self.review_head(
            torch.cat([fused_input, torch.abs(rr_margin).unsqueeze(-1), disagreement.unsqueeze(-1)], dim=-1)
        ).squeeze(-1)

        event_context = torch.cat(
            [fused_input, disagreement.unsqueeze(-1), (1.0 - quality_mean).unsqueeze(-1)], dim=-1
        )
        apnea_logit = self.apnea_head(event_context).squeeze(-1)
        motion_logit = self.motion_head(event_context).squeeze(-1)

        interval_lower = rr_pred - qhat
        interval_upper = rr_pred + qhat
        triage_code = torch.full_like(rr_pred, 2, dtype=torch.long)
        triage_code = torch.where(interval_upper < threshold, torch.zeros_like(triage_code), triage_code)
        triage_code = torch.where(interval_lower >= threshold, torch.ones_like(triage_code), triage_code)

        return {
            "rr_pred": rr_pred,
            "rr_logvar": rr_logvar,
            "qhat": qhat,
            "interval_lower": interval_lower,
            "interval_upper": interval_upper,
            "threshold_bpm": threshold,
            "tachy_logit": tachy_logit,
            "tachy_prob": torch.sigmoid(tachy_logit),
            "review_logit": review_logit,
            "review_prob": torch.sigmoid(review_logit),
            "apnea_logit": apnea_logit,
            "apnea_prob": torch.sigmoid(apnea_logit),
            "motion_logit": motion_logit,
            "motion_prob": torch.sigmoid(motion_logit),
            "triage_code": triage_code,
            "disagreement_score": disagreement,
            "fusion_weights_tensor": fusion_weights,
            "modality_pred_tensor": bpm,
            "modality_scale_tensor": scales,
            "quality_tensor": qualities,
            "availability_tensor": modality_mask,
            "quality_mean": quality_mean,
            "expert_gates": expert_gates,
            "latent_state": latent,
            "modality_stats": stats,
            "model_name": self.model_name(),
            **aux,
        }

    @torch.no_grad()
    def counterfactual_report(
        self,
        depth: torch.Tensor,
        flow: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        age_months: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        dataset_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        if modality_mask is None:
            modality_mask = torch.ones(depth.shape[0], len(self.modalities), device=depth.device, dtype=depth.dtype)

        baseline = self.forward(
            depth=depth,
            flow=flow,
            audio=audio,
            age_months=age_months,
            modality_mask=modality_mask,
            dataset_ids=dataset_ids,
        )
        report: Dict[str, Dict[str, torch.Tensor]] = {"baseline": baseline}
        for idx, name in enumerate(self.modalities):
            masked = modality_mask.clone()
            masked[:, idx] = 0.0
            cf = self.forward(
                depth=depth,
                flow=flow,
                audio=audio,
                age_months=age_months,
                modality_mask=masked,
                dataset_ids=dataset_ids,
            )
            report[name] = {
                "rr_pred": cf["rr_pred"],
                "delta_rr": cf["rr_pred"] - baseline["rr_pred"],
                "triage_code": cf["triage_code"],
            }
        return report

    @staticmethod
    def triage_name(code: int) -> str:
        mapping = {0: "NORMAL_SAFE", 1: "TACHYPNEA_ALERT", 2: "UNCERTAIN_REVIEW"}
        return mapping.get(int(code), "UNKNOWN")


class PrismBreathLoss(nn.Module):
    def __init__(
        self,
        rr_weight: float = 1.0,
        uncertainty_weight: float = 0.25,
        tachy_weight: float = 0.35,
        quality_weight: float = 0.2,
        review_weight: float = 0.15,
        consistency_weight: float = 0.1,
        codebook_weight: float = 0.05,
        apnea_weight: float = 0.15,
        motion_weight: float = 0.15,
        het_weight: float = 0.0,
        decision_weight: float = 0.0,
        decision_refer_cost: float = 0.1,
    ):
        super().__init__()
        self.het_weight = het_weight
        self.decision_weight = decision_weight
        self.decision_refer_cost = decision_refer_cost
        self.rr_weight = rr_weight
        self.uncertainty_weight = uncertainty_weight
        self.tachy_weight = tachy_weight
        self.quality_weight = quality_weight
        self.review_weight = review_weight
        self.consistency_weight = consistency_weight
        self.codebook_weight = codebook_weight
        self.apnea_weight = apnea_weight
        self.motion_weight = motion_weight

    @staticmethod
    def _weighted_mean(values: torch.Tensor, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if sample_weight is None:
            return values.mean()
        weight = sample_weight.float().clamp(min=1e-6)
        return (values * weight).sum() / weight.sum()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_rr: torch.Tensor,
        age_months: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
        apnea_target: Optional[torch.Tensor] = None,
        motion_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        thresholds = tensor_tachypnea_threshold(age_months)
        tachy_target = (gt_rr >= thresholds).float()
        review_target = ((gt_rr - thresholds).abs() <= 3.0).float()

        rr_loss = self._weighted_mean(F.smooth_l1_loss(outputs["rr_pred"], gt_rr, reduction="none"), sample_weight)
        uncertainty_loss = self._weighted_mean(
            torch.abs(outputs["rr_pred"] - gt_rr) / outputs["qhat"].clamp(min=1e-3)
            + torch.log(outputs["qhat"].clamp(min=1e-3)),
            sample_weight,
        )
        tachy_loss = self._weighted_mean(
            F.binary_cross_entropy_with_logits(outputs["tachy_logit"], tachy_target, reduction="none"),
            sample_weight,
        )
        review_loss = self._weighted_mean(
            F.binary_cross_entropy_with_logits(outputs["review_logit"], review_target, reduction="none"),
            sample_weight,
        )

        quality_target = torch.exp(-torch.abs(outputs["modality_pred_tensor"] - gt_rr.unsqueeze(-1)) / 6.0)
        quality_target = quality_target * outputs["availability_tensor"]
        quality_num = (((outputs["quality_tensor"] - quality_target) ** 2) * outputs["availability_tensor"]).sum(dim=-1)
        quality_denom = outputs["availability_tensor"].sum(dim=-1).clamp(min=1.0)
        quality_loss = self._weighted_mean(quality_num / quality_denom, sample_weight)

        weighted_rr = (outputs["fusion_weights_tensor"].detach() * outputs["modality_pred_tensor"]).sum(dim=-1)
        consistency_loss = self._weighted_mean(torch.abs(outputs["rr_pred"] - weighted_rr), sample_weight)

        codebook_loss = self._weighted_mean(
            outputs.get("clean_commitment_loss", torch.zeros_like(outputs["rr_pred"])),
            sample_weight,
        )
        codebook_loss = codebook_loss + self._weighted_mean(
            outputs.get("clean_codebook_loss", torch.zeros_like(outputs["rr_pred"])),
            sample_weight,
        )

        # Respiratory-event losses only activate when weak event labels are
        # supplied (via event augmentation); otherwise they contribute nothing,
        # keeping existing callers/checkpoints unaffected.
        zero = torch.zeros((), device=outputs["rr_pred"].device, dtype=outputs["rr_pred"].dtype)
        apnea_loss = zero
        if apnea_target is not None and "apnea_logit" in outputs:
            apnea_loss = self._weighted_mean(
                F.binary_cross_entropy_with_logits(
                    outputs["apnea_logit"], apnea_target.to(outputs["apnea_logit"].dtype), reduction="none"
                ),
                sample_weight,
            )
        motion_loss = zero
        if motion_target is not None and "motion_logit" in outputs:
            motion_loss = self._weighted_mean(
                F.binary_cross_entropy_with_logits(
                    outputs["motion_logit"], motion_target.to(outputs["motion_logit"].dtype), reduction="none"
                ),
                sample_weight,
            )

        # Heteroscedastic Gaussian NLL: 0.5*(exp(-logvar)*(pred-gt)^2 + logvar). Lets the model raise
        # its predicted variance where it is wrong (neonatal/high-RR) -> uncertainty-based abstention.
        het_loss = zero
        if self.het_weight > 0 and "rr_logvar" in outputs:
            lv = outputs["rr_logvar"]
            het_loss = self._weighted_mean(
                0.5 * (torch.exp(-lv) * (outputs["rr_pred"] - gt_rr) ** 2 + lv), sample_weight
            )

        # Decision-focused net-benefit loss (Tier-2c): directly optimize the clinical decision curve
        # over the screening operating range, rather than only the regression/threshold surrogate. The
        # optional review-head gate lets the model learn a calibrated abstention (referral to escalation).
        # Off by default (decision_weight=0), so existing callers/checkpoints are unaffected.
        decision_loss = zero
        if self.decision_weight > 0:
            p = torch.sigmoid(outputs["tachy_logit"])
            keep = torch.ones_like(p)
            if self.decision_refer_cost > 0 and "review_logit" in outputs:
                keep = 1.0 - torch.sigmoid(outputs["review_logit"])   # retained = not referred to escalation
            n = float(max(p.shape[0], 1))
            nb = zero
            pt_grid = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)            # screening range (false negatives costly)
            for pt in pt_grid:
                d = torch.sigmoid(20.0 * (p - pt)) * keep            # soft screen decision on retained cases
                w = pt / (1.0 - pt)
                nb = nb + ((d * tachy_target).sum() - w * (d * (1.0 - tachy_target)).sum()) / n
            nb = nb / len(pt_grid)
            decision_loss = -nb + self.decision_refer_cost * (1.0 - keep).mean()

        total = (
            self.rr_weight * rr_loss
            + self.het_weight * het_loss
            + self.uncertainty_weight * uncertainty_loss
            + self.tachy_weight * tachy_loss
            + self.quality_weight * quality_loss
            + self.review_weight * review_loss
            + self.consistency_weight * consistency_loss
            + self.codebook_weight * codebook_loss
            + self.apnea_weight * apnea_loss
            + self.motion_weight * motion_loss
            + self.decision_weight * decision_loss
        )
        return {
            "loss": total,
            "decision_loss": decision_loss,
            "rr_loss": rr_loss,
            "het_loss": het_loss,
            "uncertainty_loss": uncertainty_loss,
            "tachy_loss": tachy_loss,
            "quality_loss": quality_loss,
            "review_loss": review_loss,
            "consistency_loss": consistency_loss,
            "codebook_loss": codebook_loss,
            "apnea_loss": apnea_loss,
            "motion_loss": motion_loss,
        }


BASELINE_VARIANTS = ("cliffphys", "mamba")


def build_prism_breath_model_from_config(payload: Optional[Dict]) -> PrismBreathModel:
    """Build the model for a saved/derived config.

    Both the training and evaluation scripts route model construction through
    this factory, so dispatching here transparently gives the external-method
    baselines (``cliffphys``/``mamba``) the exact same train/eval/conformal
    protocol as PRISM-Breath. The baseline classes subclass ``PrismBreathModel``
    and only swap the per-modality encoder.
    """
    config = PrismBreathConfig.from_dict(payload or {})
    variant = str(config.model_variant).strip().lower()
    if variant in BASELINE_VARIANTS:
        # Lazy import to avoid a circular dependency (baseline_wrappers imports
        # PrismBreathModel and this module).
        from src.core.baseline_wrappers import build_baseline_model

        return build_baseline_model(config)
    return PrismBreathModel(config=config)


def load_encoder_pretraining(model: PrismBreathModel, checkpoint_path: str, map_location: str | torch.device = "cpu") -> Dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    raw_state = checkpoint.get("branch_state_dict")
    if raw_state is None:
        raw_state = checkpoint.get("state_dict", {})

    branch_state: Dict[str, torch.Tensor] = {}
    model_state = model.state_dict()
    for key, value in raw_state.items():
        target_key = key if key.startswith("branches.") else f"branches.{key}"
        if target_key in model_state and model_state[target_key].shape == value.shape:
            branch_state[target_key] = value

    missing, unexpected = model.load_state_dict(branch_state, strict=False)
    return {
        "loaded_keys": len(branch_state),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
