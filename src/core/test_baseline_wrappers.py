"""
Standalone smoke test for the CliffPhys/Mamba baseline wrappers.

Run directly from the project root:
    python src/core/test_baseline_wrappers.py

Verifies that each baseline:
- builds through the shared factory (the path train/eval use),
- produces the full PRISM-Breath output-dict contract,
- is trainable with PrismBreathLoss (one backward step),
- supports counterfactual reports, and
- round-trips through export_config -> factory -> load_state_dict (the eval path).
"""

import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.prism_breath_model import (  # noqa: E402
    PrismBreathConfig,
    PrismBreathLoss,
    build_prism_breath_model_from_config,
)
from src.core.baseline_wrappers import CliffPhysBaseline, MambaRespirationBaseline  # noqa: E402

REQUIRED_KEYS = [
    "rr_pred", "qhat", "interval_lower", "interval_upper", "threshold_bpm",
    "tachy_logit", "tachy_prob", "review_logit", "review_prob", "triage_code",
    "disagreement_score", "fusion_weights_tensor", "modality_pred_tensor",
    "quality_tensor", "availability_tensor",
]

EXPECTED_CLASS = {"cliffphys": CliffPhysBaseline, "mamba": MambaRespirationBaseline}


def _fake_batch(batch_size=4, seq_len=64):
    torch.manual_seed(0)
    return {
        "depth": torch.randn(batch_size, seq_len),
        "flow": torch.randn(batch_size, seq_len),
        "audio": torch.randn(batch_size, seq_len),
        "gt_rr": torch.rand(batch_size) * 40 + 20,
        "age_months": torch.rand(batch_size) * 24,
        "modality_mask": torch.ones(batch_size, 3),
        "dataset_id": torch.randint(0, 4, (batch_size,)),
    }


def _check_variant(variant: str) -> int:
    print(f"\n=== {variant} ===")
    seq_len = 64
    config = PrismBreathConfig.from_variant(model_variant=variant, seq_len=seq_len, d_model=32)
    model = build_prism_breath_model_from_config(config.to_dict())

    assert isinstance(model, EXPECTED_CLASS[variant]), f"factory returned {type(model).__name__}"
    print(f"  factory -> {type(model).__name__} ('{model.model_name()}')")

    batch = _fake_batch(seq_len=seq_len)
    out = model(
        depth=batch["depth"], flow=batch["flow"], audio=batch["audio"],
        age_months=batch["age_months"], modality_mask=batch["modality_mask"],
        dataset_ids=batch["dataset_id"],
    )
    missing = [k for k in REQUIRED_KEYS if k not in out]
    assert not missing, f"missing output keys: {missing}"
    assert out["rr_pred"].shape == (batch["depth"].shape[0],), out["rr_pred"].shape
    assert torch.isfinite(out["rr_pred"]).all(), "non-finite rr_pred"
    print(f"  forward OK: {len(out)} keys, rr_pred mean={out['rr_pred'].mean().item():.2f}")

    criterion = PrismBreathLoss()
    losses = criterion(out, gt_rr=batch["gt_rr"], age_months=batch["age_months"])
    assert torch.isfinite(losses["loss"]), "non-finite loss"
    losses["loss"].backward()
    grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad > 0, "no gradients flowed"
    print(f"  loss={losses['loss'].item():.4f}, backward OK (grad-mass={grad:.2f})")

    report = model.counterfactual_report(
        depth=batch["depth"], flow=batch["flow"], audio=batch["audio"],
        age_months=batch["age_months"], modality_mask=batch["modality_mask"],
        dataset_ids=batch["dataset_id"],
    )
    assert "baseline" in report and "depth" in report, "counterfactual report incomplete"
    print(f"  counterfactual report OK: {sorted(report.keys())}")

    # Eval round-trip: rebuild from exported config and load weights (strict).
    rebuilt = build_prism_breath_model_from_config(model.export_config())
    rebuilt.load_state_dict(model.state_dict())
    print("  export_config -> factory -> load_state_dict OK (strict)")
    return 0


def main() -> int:
    failures = 0
    for variant in ("cliffphys", "mamba"):
        try:
            _check_variant(variant)
        except AssertionError as exc:  # noqa: PERF203
            failures += 1
            print(f"  FAIL: {exc}")
    print("\n" + ("ALL BASELINE WRAPPER CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
