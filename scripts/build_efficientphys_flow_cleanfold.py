"""B2: Leakage-free EfficientPhys-flow control. Per PRISM CV fold, train EfficientPhys ONLY on that
fold's train subjects and extract ONLY its held-out test subjects, so every subject's flow waveform
is produced by an extractor that never saw it -> zero flow-extractor leakage. Quantifies the
~0.014-AUROC leakage baked into the headline (which uses a full-split EfficientPhys). Mirror of
build_tscan_flow_cleanfold.py but for EfficientPhys, using build_learned_flow_signals.extract_split
(which handles the EfficientPhys last-frame-repeat forward).

RUN in an unrestricted terminal (GPU + multiprocessing; sandbox blocks it):
    set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
    python scripts/build_efficientphys_flow_cleanfold.py
Then run PRISM k-fold on data/pedrr_signals_learned_clean and compare AUROC vs the leaky headline.
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np                                  # noqa: E402
import train_learned_extractor_pedrr as T           # noqa: E402
import infer as air_infer                            # noqa: E402
import build_learned_flow_signals as BL              # noqa: E402  (EfficientPhys-aware extract_split, OUT_DIR, V3_DIR)
import build_tscan_flow_signals as BT                # noqa: E402  (cached_subjects)
from trainers.base_trainer import BaseTrainer        # noqa: E402
from loss.psdmse import PSDMSE                        # noqa: E402

KF = ROOT / "data" / "organized" / "kfold_learned"
CACHE = str(ROOT / "data" / "pedrr_learned_cache" / "full")
BL.OUT_DIR = ROOT / "data" / "pedrr_signals_learned_clean"      # redirect extract_split's writes
NFOLDS = 5


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Leakage-free EfficientPhys-flow build (full 5-fold). "
                                 "Resumable: reuses the coarse2fine cache and skips completed folds.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="resume a hung/partial run: skip folds with a .extracted.done marker and don't "
                         "re-extract subjects whose signal already exists (still a full 5-fold run)")
    ap.add_argument("--only-folds", default="",
                    help="optional comma list, e.g. '4' (default: all 5 = full run)")
    a = ap.parse_args()
    folds = [int(x) for x in a.only_folds.split(",") if x.strip()] or list(range(NFOLDS))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("effphys_clean")
    combined_manifest, _bs, subject_gt = T.build_combined_manifest(False)
    cached = BT.cached_subjects()
    BL.OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    for k in folds:
        done_marker = ROOT / "outputs" / "paper_runs" / "pedrr_effphys_clean" / f"fold{k}" / ".extracted.done"
        if a.skip_existing and done_marker.exists():
            print(f"[skip] fold{k}: already extracted ({done_marker})", flush=True); continue
        T.air_main.set_random_seeds(100)
        tr = [r for r in json.load(open(KF / f"fold{k}" / "train.json")) if r["subject_id"] in cached]
        va = [r for r in json.load(open(KF / f"fold{k}" / "val.json")) if r["subject_id"] in cached]
        te = [r for r in json.load(open(KF / f"fold{k}" / "test.json")) if r["subject_id"] in cached]
        ss = {"train": [str(r["subject_id"]) for r in tr], "val": [str(r["subject_id"]) for r in va],
              "test": [str(r["subject_id"]) for r in te]}
        p = dict(T.FULL_DEFAULTS); p["train_limit"] = 0; p["val_limit"] = 0; p["test_limit"] = 0
        out_run = str(ROOT / "outputs" / "paper_runs" / "pedrr_effphys_clean" / f"fold{k}")
        config = T.build_config(p, combined_manifest, ss, CACHE, out_run, f"effphys_clean_fold{k}")
        config["MODEL"]["NAME"] = "EfficientPhys"
        config["SYSTEM"]["PIN_MEMORY"] = False   # host RAM is tight (shared w/ bdsl_graph); avoid page-locked RAM
        T.air_main.listify_datasets(config)
        datasets = T.air_main.get_split_datasets(config, {T.DATASET_NAME: T.PedRRDataset})
        train_loader, val_loader, test_loader = T.build_loaders(config, datasets, int(p["seed"]))
        model = air_infer.build_model(config, logger)
        trainer = BaseTrainer(model=model, config=config,
                              train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
        trainer.criterion[T.DATASET_NAME] = PSDMSE(high_pass=T.PEDRR_HIGH_HZ, low_pass=T.PEDRR_LOW_HZ)
        trainer.post_processor = T.PedRRPostProcessor(T.PEDRR_LOW_HZ, T.PEDRR_HIGH_HZ)
        print(f"\n===== [effphys-clean fold{k}] train={len(tr)} -> extract test={len(te)} =====", flush=True)
        trainer.train()
        # extract ONLY the held-out test subjects (leakage-free); BL handles the EfficientPhys forward
        saved, misses, rr_rows = BL.extract_split(trainer, test_loader, subject_gt, "test",
                                                  overwrite=not a.skip_existing)
        total += saved
        if rr_rows:
            err = np.array([abs(r - g) for _, r, g in rr_rows if np.isfinite(g)])
            print(f"[fold{k}] saved={saved} miss={len(misses)} RR_MAE={err.mean():.2f} (n={err.size})", flush=True)
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.write_text("done")   # resume marker: this fold's test extraction finished
    print(f"\nTOTAL leakage-free EfficientPhys signals saved={total} -> {BL.OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
