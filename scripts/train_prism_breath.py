"""
Train PRISM-Breath on the respiratory benchmark.

Expected usage:
1) Build manifests with scripts/build_respiratory_benchmark_manifest.py
2) Extract signals from those manifests with scripts/extract_benchmark_signals.py
3) Train with this script on the resulting `.npz` signal directory
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.respiratory_benchmark_dataset import RespiratoryBenchmarkDataset, canonical_dataset_name
from src.core.prism_breath_model import (
    PrismBreathConfig,
    PrismBreathLoss,
    build_prism_breath_model_from_config,
    load_encoder_pretraining,
)


def seed_everything(seed: int) -> None:
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic kernels so a 3-seed std reflects seed variance, not GPU nondeterminism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker so augmentation RNG is reproducible across runs."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def triage_metrics(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "review_rate": float("nan"),
        }

    gt = np.asarray([row["gt_rr"] for row in rows], dtype=np.float32)
    pred = np.asarray([row["pred_rr"] for row in rows], dtype=np.float32)
    triage = np.asarray([row["triage_code"] for row in rows], dtype=np.int64)
    tachy = np.asarray([row["gt_tachy"] for row in rows], dtype=np.int64)

    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    review_rate = float(np.mean(triage == 2))

    positives = tachy == 1
    negatives = tachy == 0
    sensitivity = float(np.mean(triage[positives] == 1)) if positives.any() else float("nan")
    specificity = float(np.mean(triage[negatives] == 0)) if negatives.any() else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "review_rate": review_rate,
    }


def collect_rows(batch: Dict, outputs: Dict[str, torch.Tensor]) -> List[Dict]:
    gt_rr = batch["gt_rr"].detach().cpu().numpy()
    age = batch["age_months"].detach().cpu().numpy()
    pred_rr = outputs["rr_pred"].detach().cpu().numpy()
    qhat = outputs["qhat"].detach().cpu().numpy()
    triage = outputs["triage_code"].detach().cpu().numpy()
    thresholds = outputs["threshold_bpm"].detach().cpu().numpy()
    rows: List[Dict] = []
    for idx in range(len(gt_rr)):
        rows.append(
            {
                "dataset": batch["dataset"][idx],
                "subject_id": batch["subject_id"][idx],
                "video_id": batch["video_id"][idx],
                "gt_rr": float(gt_rr[idx]),
                "pred_rr": float(pred_rr[idx]),
                "qhat": float(qhat[idx]),
                "triage_code": int(triage[idx]),
                "threshold_bpm": float(thresholds[idx]),
                "age_months": float(age[idx]),
                "gt_tachy": int(gt_rr[idx] >= thresholds[idx]),
            }
        )
    return rows


def apply_modality_dropout(modality_mask: torch.Tensor, dropout_prob: float) -> torch.Tensor:
    if dropout_prob <= 0.0:
        return modality_mask
    keep = modality_mask.clone()
    for idx in range(keep.shape[0]):
        available = torch.nonzero(keep[idx] > 0, as_tuple=False).flatten()
        if available.numel() <= 1:
            continue
        for modality_idx in available.tolist():
            if random.random() < dropout_prob and keep[idx].sum() > 1:
                keep[idx, modality_idx] = 0.0
    return keep


def build_sample_weight(
    dataset_names: List[str],
    device: torch.device,
    dtype: torch.dtype,
    air125_loss_weight: float,
) -> torch.Tensor:
    weights = [
        float(air125_loss_weight) if canonical_dataset_name(name) == "AIR-125" else 1.0
        for name in dataset_names
    ]
    return torch.tensor(weights, device=device, dtype=dtype)


def run_epoch(
    model,
    loader: DataLoader,
    criterion: PrismBreathLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    modality_dropout_prob: float = 0.0,
    air125_loss_weight: float = 1.0,
    supervise_events: bool = False,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    rows: List[Dict] = []

    for batch in loader:
        depth = batch["depth"].to(device)
        flow = batch["flow"].to(device)
        audio = batch["audio"].to(device)
        age = batch["age_months"].to(device)
        gt = batch["gt_rr"].to(device)
        dataset_ids = batch["dataset_id"].to(device)
        modality_mask = batch["modality_mask"].to(device)
        if is_train and modality_dropout_prob > 0.0:
            modality_mask = apply_modality_dropout(modality_mask, dropout_prob=modality_dropout_prob)

        outputs = model(
            depth=depth,
            flow=flow,
            audio=audio,
            age_months=age,
            modality_mask=modality_mask,
            dataset_ids=dataset_ids,
        )
        sample_weight = build_sample_weight(
            batch["dataset"],
            device=device,
            dtype=gt.dtype,
            air125_loss_weight=air125_loss_weight,
        )
        apnea_target = batch["apnea_target"].to(device) if supervise_events and "apnea_target" in batch else None
        motion_target = batch["motion_target"].to(device) if supervise_events and "motion_target" in batch else None
        losses = criterion(
            outputs,
            gt_rr=gt,
            age_months=age,
            sample_weight=sample_weight,
            apnea_target=apnea_target,
            motion_target=motion_target,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        batch_size = depth.shape[0]
        total_loss += float(losses["loss"].detach().cpu()) * batch_size
        total_items += batch_size
        rows.extend(collect_rows(batch, outputs))

    metrics = triage_metrics(rows)
    metrics["loss"] = total_loss / max(1, total_items)
    return metrics


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PRISM-Breath for respiratory-rate estimation and safe triage.")
    parser.add_argument("--signal-dir", default="data/processed_signals", help="Directory containing extracted signal .npz files")
    parser.add_argument("--train-manifest", default="data/organized/respiratory_benchmark_train.json", help="Training manifest JSON")
    parser.add_argument("--val-manifest", default="data/organized/respiratory_benchmark_val.json", help="Validation manifest JSON")
    parser.add_argument("--output-dir", default="outputs/prism_breath", help="Output directory")
    parser.add_argument(
        "--model-variant",
        default="prism_breath",
        choices=[
            "prism_breath",
            "prism_shared_trunk",
            "prism_no_scan",
            "prism_no_spectral",
            "prism_no_codebook",
            "prism_no_prototypes",
            "prism_lean",
            "airflow_compact",
            "cliffphys",
            "mamba",
        ],
    )
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--num-prototypes", type=int, default=6)
    parser.add_argument("--codebook-size", type=int, default=12)
    parser.add_argument("--use-freq-forward-fusion", action="store_true",
                        help="#22: disagreement-gated fusion prior that trusts the higher-frequency channel")
    parser.add_argument("--freq-forward-gamma", type=float, default=3.0)
    parser.add_argument("--freq-forward-temp", type=float, default=8.0)
    parser.add_argument("--use-snr-fusion", action="store_true",
                        help="#C: ground reliability weights in computed spectral SNR (de-collapse the inert fusion)")
    parser.add_argument("--het-weight", type=float, default=0.0,
                        help="heteroscedastic Gaussian-NLL weight (>0 enables the aleatoric log-var head)")
    parser.add_argument("--synth-highrr-prob", type=float, default=0.0,
                        help="per-sample prob of synth high-RR time-compression augmentation (attacks neonatal scarcity)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--encoder-pretrained", default="", help="Optional encoder checkpoint from scripts/pretrain_prism_breath.py")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0)
    parser.add_argument("--modality-dropout-prob", type=float, default=0.1)
    parser.add_argument("--disable-source-balanced-sampling", action="store_true")
    parser.add_argument("--air125-sampling-boost", type=float, default=1.25)
    parser.add_argument("--air125-loss-weight", type=float, default=2.0)
    parser.add_argument("--disable-air125-augmentation", action="store_true")
    parser.add_argument("--general-augment-prob", type=float, default=0.12)
    parser.add_argument("--air125-augment-prob", type=float, default=0.55)
    parser.add_argument("--air125-augment-strength", type=float, default=1.75)
    parser.add_argument("--disable-domain-adapter", action="store_true")
    parser.add_argument(
        "--enable-event-augmentation",
        action="store_true",
        help="Inject synthetic apnea/motion events and supervise the apnea/motion detection heads.",
    )
    parser.add_argument("--apnea-prob", type=float, default=0.15, help="Per-sample probability of injecting an apnea event")
    parser.add_argument("--motion-prob", type=float, default=0.2, help="Per-sample probability of injecting a motion artifact")
    parser.add_argument("--apnea-loss-weight", type=float, default=0.15)
    parser.add_argument("--motion-loss-weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--file-glob", default="*.npz", help="Glob used only when manifest paths are omitted")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = RespiratoryBenchmarkDataset(
        signal_dir=args.signal_dir,
        manifest_path=args.train_manifest,
        seq_len=args.seq_len,
        file_glob=args.file_glob,
        training=True,
        enable_augmentation=not args.disable_air125_augmentation,
        general_augment_prob=args.general_augment_prob,
        air125_augment_prob=args.air125_augment_prob,
        air125_augment_strength=args.air125_augment_strength,
        enable_event_augmentation=args.enable_event_augmentation,
        apnea_prob=args.apnea_prob,
        motion_prob=args.motion_prob,
        synth_highrr_prob=args.synth_highrr_prob,
    )
    val_ds = RespiratoryBenchmarkDataset(
        signal_dir=args.signal_dir,
        manifest_path=args.val_manifest,
        seq_len=args.seq_len,
        file_glob=args.file_glob,
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
    print(f"[Train Split] dataset counts: {train_ds.dataset_counts()}")
    print(
        "[Train Split] AIR-125 mitigation: "
        f"balanced_sampling={not args.disable_source_balanced_sampling} "
        f"sampling_boost={args.air125_sampling_boost:.2f} "
        f"loss_weight={args.air125_loss_weight:.2f} "
        f"domain_adapter={not args.disable_domain_adapter} "
        f"augmented={not args.disable_air125_augmentation}"
    )

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=loader_gen,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, worker_init_fn=seed_worker)

    model_config = PrismBreathConfig.from_variant(
        model_variant=args.model_variant,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_experts=args.num_experts,
        num_prototypes=args.num_prototypes,
        codebook_size=args.codebook_size,
        use_freq_forward_fusion=args.use_freq_forward_fusion,
        use_snr_fusion=args.use_snr_fusion,
        freq_forward_gamma=args.freq_forward_gamma,
        freq_forward_temp=args.freq_forward_temp,
        use_heteroscedastic=(args.het_weight > 0.0),
        use_domain_adapter=None
        if args.model_variant in ("airflow_compact", "cliffphys", "mamba")
        else (not args.disable_domain_adapter),
    )
    model = build_prism_breath_model_from_config(model_config.to_dict()).to(device)
    criterion = PrismBreathLoss(apnea_weight=args.apnea_loss_weight, motion_weight=args.motion_loss_weight,
                                het_weight=args.het_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pretrain_info = None
    if args.encoder_pretrained:
        pretrain_info = load_encoder_pretraining(model, args.encoder_pretrained, map_location=device)
        print(f"[Pretraining] loaded encoder weights: {pretrain_info}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_score = math.inf
    history: List[Dict[str, float]] = []
    checkpoint_path = output_dir / "prism_breath_best.pt"

    for epoch in range(1, args.epochs + 1):
        freeze_encoder = epoch <= int(args.freeze_encoder_epochs)
        for param_name, param in model.named_parameters():
            if param_name.startswith("branches."):
                param.requires_grad = not freeze_encoder

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            modality_dropout_prob=args.modality_dropout_prob,
            air125_loss_weight=args.air125_loss_weight,
            supervise_events=args.enable_event_augmentation,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                modality_dropout_prob=0.0,
                air125_loss_weight=args.air125_loss_weight,
                supervise_events=args.enable_event_augmentation,
            )

        composite = float(val_metrics["mae"]) + 5.0 * float(val_metrics["review_rate"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_sensitivity": val_metrics["sensitivity"],
                "val_specificity": val_metrics["specificity"],
                "val_review_rate": val_metrics["review_rate"],
                "selection_score": composite,
                "freeze_encoder": freeze_encoder,
            }
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_mae={val_metrics['mae']:.3f} "
            f"val_review={val_metrics['review_rate']:.3f}"
        )

        if composite < best_score:
            best_score = composite
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        **vars(args),
                        "model_name": model.model_name(),
                        "model_config": model.export_config(),
                        "pretraining": pretrain_info,
                    },
                    "best_score": best_score,
                },
                checkpoint_path,
            )

    save_json(
        output_dir / "training_history.json",
        {
            "history": history,
            "best_score": best_score,
            "model_config": model.export_config(),
            "pretraining": pretrain_info,
        },
    )
    print(f"[OK] Saved best checkpoint to {checkpoint_path}")
    print(f"[OK] Saved training history to {output_dir / 'training_history.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
