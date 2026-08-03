"""Self-supervised pretraining for PRISM-Breath encoders."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.prism_breath_model import PrismBreathConfig
from src.core.prism_breath_pretraining import PrismBreathPretrainer
from src.core.respiratory_benchmark_dataset import RespiratoryBenchmarkDataset
from src.utils.augmentations import TimeWarpAugmentation


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_block_mask(signal: torch.Tensor, mask_ratio: float, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    masked = signal.clone()
    mask = torch.zeros_like(signal)
    batch_size, seq_len = signal.shape
    blocks = max(1, int(seq_len * mask_ratio) // max(1, block_size))
    for idx in range(batch_size):
        for _ in range(blocks):
            start = random.randint(0, max(0, seq_len - block_size))
            stop = min(seq_len, start + block_size)
            masked[idx, start:stop] = 0.0
            mask[idx, start:stop] = 1.0
    return masked, mask


def augment_batch(signal: torch.Tensor, warp: TimeWarpAugmentation) -> torch.Tensor:
    augmented = []
    for row in signal:
        warped = warp(row.detach().cpu()).to(signal.device)
        scale = 1.0 + random.uniform(-0.08, 0.08)
        noise = 0.02 * torch.randn_like(warped)
        augmented.append(scale * warped + noise)
    return torch.stack(augmented, dim=0)


def run_epoch(
    model: PrismBreathPretrainer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    mask_ratio: float,
    block_size: int,
    contrastive_weight: float,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    warp = TimeWarpAugmentation(warp_sigma=0.2)
    total_loss = 0.0
    total_recon = 0.0
    total_contrast = 0.0
    total_items = 0

    for batch in loader:
        signals = {
            "depth": batch["depth"].to(device),
            "flow": batch["flow"].to(device),
            "audio": batch["audio"].to(device),
        }
        modality_mask = batch["modality_mask"].to(device)

        masked_signals = {}
        masks = {}
        for name, tensor in signals.items():
            masked_signals[name], masks[name] = apply_block_mask(tensor, mask_ratio=mask_ratio, block_size=block_size)

        encoded = model.encode(masked_signals)
        recon = model.reconstruct(encoded)
        recon_loss = model.masked_reconstruction_loss(recon, signals, masks, modality_mask=modality_mask)

        contrastive_loss = torch.zeros((), device=device)
        if contrastive_weight > 0.0:
            view_one = {name: augment_batch(tensor, warp=warp) for name, tensor in signals.items()}
            view_two = {name: augment_batch(tensor, warp=warp) for name, tensor in signals.items()}
            emb_one = model.encode(view_one)
            emb_two = model.encode(view_two)
            contrastive_loss = model.contrastive_loss(emb_one, emb_two, modality_mask=modality_mask)

        loss = recon_loss + contrastive_weight * contrastive_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        batch_size = signals["depth"].shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_recon += float(recon_loss.detach().cpu()) * batch_size
        total_contrast += float(contrastive_loss.detach().cpu()) * batch_size
        total_items += batch_size

    denom = max(1, total_items)
    return {
        "loss": total_loss / denom,
        "reconstruction_loss": total_recon / denom,
        "contrastive_loss": total_contrast / denom,
    }


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="RespMAE-style masked pretraining for PRISM-Breath encoders.")
    parser.add_argument("--signal-dir", default="data/processed_signals")
    parser.add_argument("--train-manifest", default="data/organized/respiratory_benchmark_train.json")
    parser.add_argument("--val-manifest", default="data/organized/respiratory_benchmark_val.json")
    parser.add_argument("--output-dir", default="outputs/prism_breath_pretrain")
    parser.add_argument("--model-variant", default="prism_breath")
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=30)
    parser.add_argument("--contrastive-weight", type=float, default=0.15)
    parser.add_argument("--disable-source-balanced-sampling", action="store_true")
    parser.add_argument("--air125-sampling-boost", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = PrismBreathConfig.from_variant(model_variant=args.model_variant, seq_len=args.seq_len, d_model=args.d_model)

    train_ds = RespiratoryBenchmarkDataset(
        signal_dir=args.signal_dir,
        manifest_path=args.train_manifest,
        seq_len=args.seq_len,
    )
    val_ds = RespiratoryBenchmarkDataset(
        signal_dir=args.signal_dir,
        manifest_path=args.val_manifest,
        seq_len=args.seq_len,
    )
    train_sampler = None
    if not args.disable_source_balanced_sampling:
        sample_weights = train_ds.source_balanced_sample_weights(
            power=1.0,
            source_boosts={"AIR-125": args.air125_sampling_boost},
        )
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    print(f"[Pretrain Split] dataset counts: {train_ds.dataset_counts()}")
    print(
        "[Pretrain Split] source-balanced sampling: "
        f"{not args.disable_source_balanced_sampling} "
        f"(AIR-125 boost={args.air125_sampling_boost:.2f})"
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = PrismBreathPretrainer(config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "prism_breath_pretrained.pt"

    best_score = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            mask_ratio=args.mask_ratio,
            block_size=args.block_size,
            contrastive_weight=args.contrastive_weight,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device=device,
                optimizer=None,
                mask_ratio=args.mask_ratio,
                block_size=args.block_size,
                contrastive_weight=args.contrastive_weight,
            )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[Epoch {epoch:02d}] "
            f"train_recon={train_metrics['reconstruction_loss']:.4f} "
            f"val_recon={val_metrics['reconstruction_loss']:.4f} "
            f"val_contrast={val_metrics['contrastive_loss']:.4f}"
        )

        if val_metrics["loss"] < best_score:
            best_score = val_metrics["loss"]
            torch.save(
                {
                    "branch_state_dict": model.export_branch_state(),
                    "state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "best_score": best_score,
                },
                checkpoint_path,
            )

    save_json(output_dir / "pretraining_history.json", {"history": history, "best_score": best_score})
    print(f"[OK] Saved encoder pretraining checkpoint to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
