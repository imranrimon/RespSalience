"""Which saved checkpoint actually produced a published baseline's committed predictions?

The re-selection sweep's fidelity check found that re-running `<Net>_best.pth` does NOT reproduce the
committed predictions CSV. Before any re-selected number can be trusted, we need to know which weights
the published run actually evaluated. The likely cause is visible in `run_extractor_cv.py`:

    trainer.train()
    rr_by_subj = extract_test_rr(trainer, test_loader)

The extraction uses whatever weights are resident after `train()` returns -- which is the LAST epoch
unless the trainer restores its best state. If that is what happened, the published baselines were
never checkpoint-selected at all, and that materially changes how the comparison must be described.

This script runs EVERY saved epoch checkpoint (plus `_best.pth`) over the test fold and reports the
max absolute RR difference against the committed CSV. Exactly one should match to ~0.

    python scripts/identify_baseline_checkpoint.py --model-name EfficientPhys --fold 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402

import train_learned_extractor_pedrr as T       # noqa: E402
import infer as air_infer                        # noqa: E402
from trainers.base_trainer import BaseTrainer    # noqa: E402
from loss.psdmse import PSDMSE                    # noqa: E402

from reselect_baseline_checkpoints import (      # noqa: E402
    KF, CACHE, auroc_of, best_checkpoint, cached_subjects, epoch_checkpoints,
    extract_rr, load_weights, mae_of, rows_for,
)


def main() -> int:
    import logging
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True,
                    choices=["DeepPhys", "TSCAN", "VIRENet", "EfficientPhys"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("identify")
    T.air_main.set_random_seeds(100)
    combined_manifest, _bs, _sg = T.build_combined_manifest(False)

    net = args.model_name
    SRC = ROOT / "outputs" / "paper_runs" / f"pedrr_kfold_{net.lower()}{args.tag}"
    k = args.fold
    sfd = SRC / f"fold{k}"
    comm_path = sfd / "test_eval" / "prism_breath_predictions.csv"
    if not comm_path.exists():
        raise SystemExit(f"no committed predictions at {comm_path}")
    want = pd.read_csv(comm_path)

    cached = cached_subjects()
    tr = [r for r in json.load(open(KF / f"fold{k}" / "train.json")) if r["subject_id"] in cached]
    va = [r for r in json.load(open(KF / f"fold{k}" / "val.json")) if r["subject_id"] in cached]
    te = [r for r in json.load(open(KF / f"fold{k}" / "test.json")) if r["subject_id"] in cached]
    ss = {"train": [str(r["subject_id"]) for r in tr], "val": [str(r["subject_id"]) for r in va],
          "test": [str(r["subject_id"]) for r in te]}
    p = dict(T.FULL_DEFAULTS); p["train_limit"] = 0; p["val_limit"] = 0; p["test_limit"] = 0
    config = T.build_config(p, combined_manifest, ss, CACHE,
                            str(ROOT / "outputs" / "runtime_tmp" / f"identify_{net}_f{k}"),
                            f"{net}_fold{k}")
    config["MODEL"]["NAME"] = net
    T.air_main.listify_datasets(config)
    datasets = T.air_main.get_split_datasets(config, {T.DATASET_NAME: T.PedRRDataset})
    train_loader, val_loader, test_loader = T.build_loaders(config, datasets, int(p["seed"]))
    model = air_infer.build_model(config, logger)
    trainer = BaseTrainer(model=model, config=config, train_loader=train_loader,
                          val_loader=val_loader, test_loader=test_loader)
    trainer.criterion[T.DATASET_NAME] = PSDMSE(high_pass=T.PEDRR_HIGH_HZ, low_pass=T.PEDRR_LOW_HZ)
    trainer.post_processor = T.PedRRPostProcessor(T.PEDRR_LOW_HZ, T.PEDRR_HIGH_HZ)

    cands = list(epoch_checkpoints(sfd, net))
    bp = best_checkpoint(sfd, net)
    if bp:
        cands.append((-1, bp))                   # -1 denotes "_best.pth"

    committed_auroc = auroc_of(want.to_dict("records"))
    print(f"{net}{args.tag} fold{k}: committed n={len(want)} AUROC={committed_auroc:.4f} "
          f"MAE={float((want.pred_rr_bpm - want.gt_rr_bpm).abs().mean()):.2f}")
    print(f"\n{'checkpoint':<14}{'max|dRR|':>12}{'mean|dRR|':>12}{'test AUROC':>12}{'test MAE':>10}")
    results = []
    for ep, path in cands:
        load_weights(trainer.model, path)
        rr = extract_rr(trainer, test_loader)
        got = pd.DataFrame(rows_for(te, rr))
        j = want.merge(got, on="subject_id", suffixes=("_c", "_n"))
        d = (j.pred_rr_bpm_c - j.pred_rr_bpm_n).abs()
        name = "_best.pth" if ep < 0 else f"Epoch{ep}"
        rec = {"ckpt": name, "max": float(d.max()), "mean": float(d.mean()),
               "auroc": auroc_of(got.to_dict("records")), "mae": mae_of(got.to_dict("records"))}
        results.append(rec)
        print(f"{name:<14}{rec['max']:>12.4f}{rec['mean']:>12.4f}{rec['auroc']:>12.4f}{rec['mae']:>10.2f}")

    hit = min(results, key=lambda r: r["max"])
    print(f"\nClosest match: {hit['ckpt']} (max |dRR| = {hit['max']:.3e})")
    if hit["max"] < 1e-6:
        print(f"=> The committed predictions come from {hit['ckpt']}.")
        if hit["ckpt"] != "_best.pth":
            print("   The published run therefore did NOT use the validation-selected checkpoint; it")
            print("   evaluated the weights resident after train() returned. The baselines are")
            print("   last-epoch models, not loss-selected ones -- state this when comparing.")
    else:
        print("=> No checkpoint reproduces the committed predictions exactly. The extraction path")
        print("   itself is not deterministic (loader order/chunking), so re-selected numbers must be")
        print("   reported as a fresh matched sweep, with BOTH arms recomputed by this same harness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
