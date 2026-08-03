"""Leakage-free TSCAN-flow signals: per PRISM CV fold, train TSCAN ONLY on that fold's train
subjects and extract ONLY its held-out test subjects. The 5 test folds partition all 298 cached
subjects exactly once (verified), so every subject's flow waveform is produced by a TSCAN that never
saw it -> zero flow-extractor leakage. Confirms whether the leaky-build TSCAN-flow win (MAE 5.23,
AUROC 0.911) survives when the extractor is held out.

Reuses build_tscan_flow_signals.extract_split (writes flow into pedrr_signals_learned depth/audio)
by monkeypatching its OUT_DIR to a clean dir. Same TSCAN forward / post_process as run_extractor_cv.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np                              # noqa: E402
import train_learned_extractor_pedrr as T       # noqa: E402
import infer as air_infer                        # noqa: E402
import build_tscan_flow_signals as BT             # noqa: E402  (extract_split, cached_subjects, SRC_DIR)
from trainers.base_trainer import BaseTrainer    # noqa: E402
from loss.psdmse import PSDMSE                    # noqa: E402

KF = ROOT / "data" / "organized" / "kfold_learned"
CACHE = str(ROOT / "data" / "pedrr_learned_cache" / "full")
BT.OUT_DIR = ROOT / "data" / "pedrr_signals_tscan_clean"     # redirect extract_split's writes
NFOLDS = 5


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("tscan_clean")
    combined_manifest, _bs, subject_gt = T.build_combined_manifest(False)
    cached = BT.cached_subjects()
    BT.OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for k in range(NFOLDS):
        T.air_main.set_random_seeds(100)
        tr = [r for r in json.load(open(KF / f"fold{k}" / "train.json")) if r["subject_id"] in cached]
        va = [r for r in json.load(open(KF / f"fold{k}" / "val.json")) if r["subject_id"] in cached]
        te = [r for r in json.load(open(KF / f"fold{k}" / "test.json")) if r["subject_id"] in cached]
        ss = {"train": [str(r["subject_id"]) for r in tr], "val": [str(r["subject_id"]) for r in va],
              "test": [str(r["subject_id"]) for r in te]}
        p = dict(T.FULL_DEFAULTS); p["train_limit"] = 0; p["val_limit"] = 0; p["test_limit"] = 0
        out_run = str(ROOT / "outputs" / "paper_runs" / "pedrr_tscan_clean" / f"fold{k}")
        config = T.build_config(p, combined_manifest, ss, CACHE, out_run, f"tscan_clean_fold{k}")
        config["MODEL"]["NAME"] = "TSCAN"
        T.air_main.listify_datasets(config)
        datasets = T.air_main.get_split_datasets(config, {T.DATASET_NAME: T.PedRRDataset})
        train_loader, val_loader, test_loader = T.build_loaders(config, datasets, int(p["seed"]))
        model = air_infer.build_model(config, logger)
        trainer = BaseTrainer(model=model, config=config,
                              train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
        trainer.criterion[T.DATASET_NAME] = PSDMSE(high_pass=T.PEDRR_HIGH_HZ, low_pass=T.PEDRR_LOW_HZ)
        trainer.post_processor = T.PedRRPostProcessor(T.PEDRR_LOW_HZ, T.PEDRR_HIGH_HZ)
        print(f"\n===== [tscan-clean fold{k}] train={len(tr)} -> extract test={len(te)} =====", flush=True)
        trainer.train()
        # extract ONLY the held-out test subjects (leakage-free)
        saved, misses, rr_rows = BT.extract_split(trainer, test_loader, subject_gt, overwrite=True)
        total += saved
        if rr_rows:
            err = np.array([abs(r - g) for _, r, g in rr_rows if np.isfinite(g)])
            print(f"[fold{k}] saved={saved} miss={len(misses)} RR_MAE={err.mean():.2f} (n={err.size})", flush=True)
    print(f"\nTOTAL leakage-free tscan signals saved={total} -> {BT.OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
