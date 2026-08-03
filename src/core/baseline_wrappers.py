"""
External-method baseline backbones for the respiratory benchmark.

These wrappers let two literature architectures be trained and evaluated through
the exact same protocol as PRISM-Breath (reliability-aware fusion, age-conditioned
tachypnea triage, conformal calibration, counterfactual reports). They subclass
``PrismBreathModel`` and swap only the per-modality temporal backbone, so the full
output-dict contract used by ``PrismBreathLoss``, ``scripts/train_prism_breath.py``
and ``scripts/evaluate_prism_breath.py`` is inherited unchanged.

Two backbones are provided:

- ``MambaSignalEncoder`` — a genuine selective state-space (Mamba-style) scan with
  input-dependent (Δ, A, B, C) and a sequential recurrence. This is the efficient
  linear-time temporal model behind PhysMamba / RhythmMamba (arXiv:2409.12031,
  arXiv:2404.06483), which to date target only adult facial heart rate; here it is
  applied to infant respiration signals.

- ``CliffPhysSignalEncoder`` — Clifford (geometric-algebra) neural layers built on
  the Cl(2,0) geometric product, the core inductive bias of CliffPhys
  (ECCV 2024, air.unimi.it/2434/1125560), which fuses a monocular-depth scalar
  field with an optical-flow vector field for adult respiration. Because this
  repo's benchmark path operates on pre-extracted 1D respiratory signals rather
  than raw video frames, this is a signal-space adaptation ("CliffPhys-Signal"):
  the Clifford geometric-product mixing is preserved, while cross-modal
  depth/flow coupling is handled by the shared reliability fusion.

Both are honest, self-contained baselines (no external CUDA kernels) intended to
give the paper a current, reviewer-legible comparison point.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.prism_breath_model import (
    MODALITY_NAMES,
    PrismBreathConfig,
    PrismBreathModel,
    RespiratorySignalEncoder,
)


# ---------------------------------------------------------------------------
# Mamba-style selective state-space backbone
# ---------------------------------------------------------------------------
class SelectiveSSM(nn.Module):
    """A faithful (if unoptimized) selective state-space block.

    Implements input-dependent discretization of a diagonal SSM with a sequential
    recurrence, matching the Mamba formulation: Δ, B and C are projected from the
    (convolved) input, A is a learnable diagonal state matrix, and the hidden
    state is scanned across time. The scan is a Python loop over the sequence
    (O(L) and correct) rather than the fused CUDA associative scan; at seq_len 300
    this is inexpensive and avoids the "global-pooling approximation" shortcut of
    the legacy ``MambaBlock``.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, self.d_model // 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        # Projects the convolved input to (Δ_rank, B, C).
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        # A is stored as log for positivity of -exp(A_log); shape (d_inner, d_state).
        a_init = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        batch, length, _ = x.shape
        x_and_z = self.in_proj(x)
        x_in, z = x_and_z.chunk(2, dim=-1)  # each (B, L, d_inner)

        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :length].transpose(1, 2)
        x_conv = F.silu(x_conv)

        proj = self.x_proj(x_conv)
        dt, b_mat, c_mat = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        a = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # Discretize: dA = exp(Δ ⊙ A), dB = Δ ⊙ B (zero-order hold, B-simplified).
        d_a = torch.exp(dt.unsqueeze(-1) * a)               # (B, L, d_inner, d_state)
        d_bx = dt.unsqueeze(-1) * b_mat.unsqueeze(2)         # (B, L, d_inner, d_state)
        d_bx = d_bx * x_conv.unsqueeze(-1)                   # inject input

        state = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(length):
            state = d_a[:, t] * state + d_bx[:, t]
            y_t = torch.einsum("bds,bs->bd", state, c_mat[:, t])
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)                      # (B, L, d_inner)
        y = y + x_conv * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaSignalEncoder(RespiratorySignalEncoder):
    """Per-modality encoder whose temporal backbone is a selective SSM stack."""

    def __init__(self, *args, d_state: int = 16, num_ssm_layers: int = 2, **kwargs):
        # Force the parent scan off; this encoder brings its own temporal model.
        kwargs["use_scan_block"] = False
        super().__init__(*args, **kwargs)
        d_model = self.stem[-2].out_channels  # d_model produced by the shared conv stem
        self.ssm_layers = nn.ModuleList(
            [SelectiveSSM(d_model=d_model, d_state=d_state) for _ in range(max(1, num_ssm_layers))]
        )
        self.ssm_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(len(self.ssm_layers))])

    def _encode_sequence(self, signal: torch.Tensor) -> torch.Tensor:
        stem = self.stem(signal.unsqueeze(1))            # (B, d_model, L)
        seq = stem.transpose(1, 2)                       # (B, L, d_model)
        for layer, norm in zip(self.ssm_layers, self.ssm_norms):
            seq = seq + layer(norm(seq))                 # pre-norm residual scan
        return seq


# ---------------------------------------------------------------------------
# CliffPhys-style Clifford (geometric-algebra) backbone
# ---------------------------------------------------------------------------
def _cl2_structure_constants() -> torch.Tensor:
    """Structure constants T[i, j, k] of the Cl(2,0) geometric product.

    Basis order: [scalar(1), e1, e2, e12]. ``blade_i * blade_j = sum_k T[i,j,k] blade_k``.
    """
    t = torch.zeros(4, 4, 4)
    # scalar row
    t[0, 0, 0] = 1; t[0, 1, 1] = 1; t[0, 2, 2] = 1; t[0, 3, 3] = 1
    # e1 row
    t[1, 0, 1] = 1; t[1, 1, 0] = 1; t[1, 2, 3] = 1; t[1, 3, 2] = 1
    # e2 row
    t[2, 0, 2] = 1; t[2, 1, 3] = -1; t[2, 2, 0] = 1; t[2, 3, 1] = -1
    # e12 row
    t[3, 0, 3] = 1; t[3, 1, 2] = -1; t[3, 2, 1] = 1; t[3, 3, 0] = -1
    return t


class CliffordLinear(nn.Module):
    """Pointwise Clifford linear layer over multivector channels.

    Each output multivector channel is the sum of geometric products between the
    input multivector channels and learned weight multivectors, i.e. a genuine
    Clifford neural layer (the defining ingredient of CliffPhys) rather than a
    plain real-valued dense layer.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        weight = torch.randn(out_channels, in_channels, 4) * (1.0 / (in_channels ** 0.5))
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_channels, 4)) if bias else None
        self.register_buffer("structure", _cl2_structure_constants(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_channels, 4) -> (..., out_channels, 4)
        out = torch.einsum("...ni,onj,ijk->...ok", x, self.weight, self.structure)
        if self.bias is not None:
            out = out + self.bias
        return out


class CliffPhysSignalEncoder(RespiratorySignalEncoder):
    """Per-modality encoder built from Clifford geometric-algebra layers."""

    def __init__(self, *args, num_clifford_layers: int = 2, **kwargs):
        kwargs["use_scan_block"] = False
        super().__init__(*args, **kwargs)
        d_model = self.stem[-2].out_channels
        self.mv_channels = max(1, d_model // 4)
        self.clifford_dim = self.mv_channels * 4

        self.to_clifford = nn.Conv1d(d_model, self.clifford_dim, kernel_size=1)
        # Depthwise temporal mixing (per multivector component) before the
        # pointwise geometric-product channel mixing.
        self.temporal = nn.Conv1d(
            self.clifford_dim, self.clifford_dim, kernel_size=5, padding=2, groups=self.clifford_dim
        )
        self.clifford_layers = nn.ModuleList(
            [CliffordLinear(self.mv_channels, self.mv_channels) for _ in range(max(1, num_clifford_layers))]
        )
        self.from_clifford = nn.Conv1d(self.clifford_dim, d_model, kernel_size=1)

    def _encode_sequence(self, signal: torch.Tensor) -> torch.Tensor:
        stem = self.stem(signal.unsqueeze(1))            # (B, d_model, L)
        c = F.silu(self.temporal(self.to_clifford(stem)))  # (B, clifford_dim, L)
        batch, _, length = c.shape
        mv = c.view(batch, self.mv_channels, 4, length).permute(0, 3, 1, 2)  # (B, L, mv, 4)
        for idx, layer in enumerate(self.clifford_layers):
            mv = layer(mv)
            if idx < len(self.clifford_layers) - 1:
                mv = F.gelu(mv)
        c_out = mv.permute(0, 2, 3, 1).reshape(batch, self.clifford_dim, length)
        seq = self.from_clifford(c_out).transpose(1, 2)  # (B, L, d_model)
        return seq


# ---------------------------------------------------------------------------
# Model wrappers + factory
# ---------------------------------------------------------------------------
class _EncoderSwapModel(PrismBreathModel):
    """PrismBreathModel with the per-modality branch encoder replaced."""

    encoder_cls: type = RespiratorySignalEncoder

    def __init__(self, config: PrismBreathConfig):
        super().__init__(config=config)
        encoder_kwargs = {
            "seq_len": config.seq_len,
            "d_model": config.d_model,
            "use_scan_block": config.use_scan_block,
            "use_spectral_features": config.use_spectral_features,
            "compact_encoder": config.compact_encoder,
        }
        self.branches = nn.ModuleDict(
            {name: self.encoder_cls(**encoder_kwargs) for name in MODALITY_NAMES}
        )


class MambaRespirationBaseline(_EncoderSwapModel):
    encoder_cls = MambaSignalEncoder


class CliffPhysBaseline(_EncoderSwapModel):
    encoder_cls = CliffPhysSignalEncoder


def build_baseline_model(config: PrismBreathConfig) -> PrismBreathModel:
    variant = str(config.model_variant).strip().lower()
    if variant == "mamba":
        return MambaRespirationBaseline(config)
    if variant == "cliffphys":
        return CliffPhysBaseline(config)
    raise ValueError(
        f"build_baseline_model received unsupported variant={config.model_variant!r}; "
        "expected one of: cliffphys, mamba."
    )
