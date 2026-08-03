"""TSCAN-flow variant of build_learned_flow_signals: swap PRISM's flow backbone EfficientPhys->TSCAN.

Trains TSCAN (air-400) on the SAME default pedrr train split used to train the EfficientPhys flow
extractor, then aggregates its per-subject band-passed displacement waveform and writes a NEW signal
set that is IDENTICAL to pedrr_signals_learned except the flow channel:

    pedrr_signals_tscan/Pediatric-RR_{subject}_{video}.npz
      = TSCAN flow waveform  +  pedrr_signals_learned DEPTH + AUDIO (copied unchanged)  +  gt/age/...

PRISM then re-runs unchanged on these signals -> a clean "does the flow extractor backbone matter?"
robustness test vs the EfficientPhys-flow main run. Restricted to the 298 cached-coarse2fine subjects
(hang-free); PRISM's dataset skips the 34 missing signals automatically (sig_path.exists() guard).

Reuses train_learned_extractor_pedrr machinery (config/loaders/cache/PedRRPostProcessor) + air-400's
dispatching build_model(MODEL.NAME). extract_split mirrors build_learned_flow_signals but sources the
copied depth/audio from pedrr_signals_learned and handles the TSCAN forward (no channel split, no
EfficientPhys last-frame repeat) exactly as run_extractor_cv does.
"""
from __future__ import annotations

import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402

import train_learned_extractor_pedrr as T           # noqa: E402  (applies cpu_count cap on import)
import infer as air_infer                            # noqa: E402  (dispatching build_model by MODEL.NAME)
from trainers.base_trainer import BaseTrainer        # noqa: E402
from loss.psdmse import PSDMSE                       # noqa: E402

SRC_DIR = ROOT / "data" / "pedrr_signals_learned"    # copy depth+audio from the main-run signals
OUT_DIR = ROOT / "data" / "pedrr_signals_tscan"
CACHE = str(ROOT / "data" / "pedrr_learned_cache" / "full")


def cached_subjects():
    subs = set()
    for f in glob.glob(os.path.join(CACHE, "PEDRR", "*", "*.npy")):
        m = re.match(r"(ID-\d+)", os.path.basename(f))
        if m:
            subs.add(m.group(1))
    return subs


def extract_split(trainer, loader, subject_gt, overwrite):
    """Aggregate per-subject TSCAN waveform; write tscan-flow npz. Returns (saved, misses, rr_rows)."""
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
        processed, rr = trainer.post_processor.post_process(
            torch.from_numpy(pred_subj), fs=fs_map[s], diff_flag=True,
            use_bandpass=True, eval_method="FFT")
        wave = np.asarray(processed, dtype=np.float32).flatten()
        subj_id, vid = s.rsplit("_", 1)
        gt = subject_gt.get(s.rsplit("_", 1)[0], subject_gt.get(subj_id, float("nan")))
        rr_rows.append((s, float(rr), float(gt)))

        npz = SRC_DIR / f"Pediatric-RR_{subj_id}_{vid}.npz"
        if not npz.exists():                                     # fallback: fuzzy match
            cand = list(SRC_DIR.glob(f"Pediatric-RR_{subj_id}_*{vid}*.npz")) or \
                   list(SRC_DIR.glob(f"*{subj_id}*{vid}*.npz"))
            npz = cand[0] if cand else None
        if npz is None:
            misses.append(s); continue
        out = OUT_DIR / npz.name
        if out.exists() and not overwrite:
            saved += 1; continue
        z = dict(np.load(npz, allow_pickle=False))
        z["flow"] = wave                                          # <-- the only change vs the main signals
        np.savez_compressed(out, **z)
        saved += 1
    return saved, misses, rr_rows


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("tscan_flow")
    T.air_main.set_random_seeds(100)

    combined_manifest, split_subjects, subject_gt = T.build_combined_manifest(False)
    cached = cached_subjects()
    split_subjects = {k: [s for s in v if s in cached] for k, v in split_subjects.items()}
    logger.info("cached-restricted split: train=%d val=%d test=%d",
                len(split_subjects["train"]), len(split_subjects["val"]), len(split_subjects["test"]))

    p = dict(T.FULL_DEFAULTS)
    p["train_limit"] = 0; p["val_limit"] = 0; p["test_limit"] = 0
    out_run = str(ROOT / "outputs" / "paper_runs" / "pedrr_tscan" / "flow_extract")
    config = T.build_config(p, combined_manifest, split_subjects, CACHE, out_run, "tscan_flow_extract")
    config["MODEL"]["NAME"] = "TSCAN"
    T.air_main.listify_datasets(config)
    datasets = T.air_main.get_split_datasets(config, {T.DATASET_NAME: T.PedRRDataset})
    train_loader, val_loader, test_loader = T.build_loaders(config, datasets, int(p["seed"]))

    model = air_infer.build_model(config, logger)           # dispatch -> TSCAN
    trainer = BaseTrainer(model=model, config=config,
                          train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    trainer.criterion[T.DATASET_NAME] = PSDMSE(high_pass=T.PEDRR_HIGH_HZ, low_pass=T.PEDRR_LOW_HZ)
    trainer.post_processor = T.PedRRPostProcessor(T.PEDRR_LOW_HZ, T.PEDRR_HIGH_HZ)

    print("\n===== training TSCAN flow extractor on default pedrr train split =====", flush=True)
    trainer.train()

    total_saved, all_miss = 0, []
    for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        saved, misses, rr_rows = extract_split(trainer, loader, subject_gt, overwrite=True)
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
