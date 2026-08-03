"""Fold the pedrr-trained learned extractor into PRISM as the flow channel.

For every Pediatric-RR video, run the trained EfficientPhys (Plan B best.pt) over its
cached coarse2fine tensors, aggregate the per-frame respiratory prediction into one
displacement waveform per video, and write a NEW signal set:

    pedrr_signals_learned/Pediatric-RR_{subject}_{video}.npz
      = learned FLOW waveform  +  v3 DEPTH + v3 AUDIO (copied unchanged)  +  gt/age/...

PRISM then re-runs unchanged (it z-norms + center-crops flow to seq_len=300), so this is a
clean drop-in test of "does the better extractor move the clinical endpoints?" vs pedrr_v3.

Reuses the Plan-B training machinery (scripts/train_learned_extractor_pedrr.py) by import:
same config/model/cache, same PedRRPostProcessor band (0.20-1.70 Hz), same worker cap.
Cache at data/pedrr_learned_cache/full is reused; any un-cached val clips are preprocessed
once (coarse2fine) -- so DON'T run this while another coarse2fine job holds the CPU.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))          # to import the Plan-B trainer module

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402

import train_learned_extractor_pedrr as T           # noqa: E402  (applies cpu_count cap on import)
from trainers.base_trainer import BaseTrainer        # noqa: E402
from loss.psdmse import PSDMSE                       # noqa: E402

V3_DIR = ROOT / "data" / "pedrr_signals_v3"
OUT_DIR = ROOT / "data" / "pedrr_signals_learned"
CKPT = ROOT / "outputs" / "paper_runs" / "pedrr_learned" / "full" / "model" / \
    "EfficientPhys" / "efficientphys_c2f_body_pedrr_full" / "EfficientPhys_best.pth"


def extract_split(trainer, loader, subject_gt, split, overwrite):
    """Aggregate per-subject learned waveform; write learned-flow npz. Returns (saved, misses, rr_rows)."""
    if loader is None:
        return 0, [], []
    dev = trainer.device
    preds, fs_map = defaultdict(list), {}
    trainer.model.eval()
    with torch.no_grad():
        for batch in loader:
            data, labels, subj_idx, _chunk, ds_names, fs_list = batch
            data, labels = data.to(dev), labels.to(dev)
            if trainer.model_name.lower() not in ["deepphys", "tscan"]:
                data = (data[:, :, :trainer.in_channels, ...] if trainer.do_optical_flow
                        else data[:, :, trainer.in_channels:, ...])
            N, D, C, H, W = data.shape
            data = data.view(N * D, C, H, W)
            labels = labels.view(-1, 1)
            valid = (data.shape[0] // trainer.base_len) * trainer.base_len
            data, labels = data[:valid], labels[:valid]
            if trainer.model_name.lower().startswith("efficientphys"):
                last = torch.unsqueeze(data[-1], 0).repeat(trainer.num_of_gpu, 1, 1, 1)
                data = torch.cat((data, last), 0)
            pred = trainer.model(data)
            sp, _sl, _dm, fm = trainer._aggregate_batch_by_subject(
                pred, labels, subj_idx, ds_names, fs_list)
            for s in sp:
                preds[s].extend(sp[s]); fs_map[s] = fm[s]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved, misses, rr_rows = 0, [], []
    for s in preds:
        pred_subj = torch.cat(preds[s], 0).detach().cpu().numpy().astype(np.float64).flatten()
        # Flow waveform = the band-passed displacement the model actually reads to get RR
        # (post_process: cumsum(diff pred) -> detrend -> band-pass 0.20-1.70 Hz). This is the
        # exact signal behind the 4.88 test MAE AND matches the v3 hand-crafted flow's
        # band-passed character; raw cumsum without band-pass leaves low-freq drift that
        # corrupts the dominant frequency. post_process returns (processed_waveform, rr).
        processed, rr = trainer.post_processor.post_process(
            torch.from_numpy(pred_subj), fs=fs_map[s], diff_flag=True,
            use_bandpass=True, eval_method="FFT")
        wave = np.asarray(processed, dtype=np.float32).flatten()
        subj_id, vid = s.rsplit("_", 1)
        gt = subject_gt.get(s.rsplit("_", 1)[0], subject_gt.get(subj_id, float("nan")))
        rr_rows.append((s, float(rr), float(gt)))

        npz = V3_DIR / f"Pediatric-RR_{subj_id}_{vid}.npz"
        if not npz.exists():                                     # fallback: fuzzy match
            cand = list(V3_DIR.glob(f"Pediatric-RR_{subj_id}_*{vid}*.npz")) or \
                   list(V3_DIR.glob(f"*{subj_id}*{vid}*.npz"))
            npz = cand[0] if cand else None
        if npz is None:
            misses.append(s); continue
        out = OUT_DIR / npz.name
        if out.exists() and not overwrite:
            saved += 1; continue
        z = dict(np.load(npz, allow_pickle=True))
        z["flow"] = wave                                          # <-- the only change
        np.savez_compressed(out, **z)
        saved += 1
    return saved, misses, rr_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    T.logger.info("build_learned_flow_signals: ckpt=%s", CKPT)
    T.air_main.set_random_seeds(100)

    p = dict(T.FULL_DEFAULTS)
    p["train_limit"], p["val_limit"], p["test_limit"] = args.train_limit, args.val_limit, args.test_limit
    cache_dir = str(ROOT / "data" / "pedrr_learned_cache" / "full")     # reuse training cache
    out_run = str(ROOT / "outputs" / "paper_runs" / "pedrr_learned" / "flow_extract")

    combined_manifest, split_subjects, subject_gt = T.build_combined_manifest(False)
    config = T.build_config(p, combined_manifest, split_subjects, cache_dir, out_run, "flow_extract")
    T.air_main.listify_datasets(config)
    datasets = T.air_main.get_split_datasets(config, {T.DATASET_NAME: T.PedRRDataset})
    train_loader, val_loader, test_loader = T.build_loaders(config, datasets, int(p["seed"]))

    model = T.build_model(config)
    T.load_init_checkpoint(model, str(CKPT))
    trainer = BaseTrainer(model=model, config=config,
                          train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    trainer.criterion[T.DATASET_NAME] = PSDMSE(high_pass=T.PEDRR_HIGH_HZ, low_pass=T.PEDRR_LOW_HZ)
    trainer.post_processor = T.PedRRPostProcessor(T.PEDRR_LOW_HZ, T.PEDRR_HIGH_HZ)

    total_saved, all_miss = 0, []
    for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        saved, misses, rr_rows = extract_split(trainer, loader, subject_gt, split, args.overwrite)
        total_saved += saved; all_miss += misses
        if rr_rows:
            err = np.array([abs(r - g) for _, r, g in rr_rows if np.isfinite(g)])
            print(f"[{split}] saved={saved} miss={len(misses)} "
                  f"RR_MAE={err.mean():.2f} (n={err.size})", flush=True)
    print(f"\nTOTAL saved={total_saved} -> {OUT_DIR}", flush=True)
    if all_miss:
        print(f"UNMATCHED ({len(all_miss)}): {all_miss[:10]}", flush=True)


if __name__ == "__main__":
    main()
