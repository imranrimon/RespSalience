"""Run 2: additional published extractors (DeepPhys / TSCAN / VIRENet) under OUR 5-fold CV.

Generalization of run_efficientphys_cv.py: per fold, train the chosen air-400 backbone on the fold's
cached-subject train set (reusing coarse2fine tensors), extract held-out-fold RR, write clinical-
compatible predictions. Uses air-400's dispatching build_model (MODEL.NAME). Trains from scratch
(no matched pretrained init for these). Restricted to the 298 cached subjects (hang-free).

Usage: python scripts/run_extractor_cv.py --model-name DeepPhys
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
import torch                                    # noqa: E402
import train_learned_extractor_pedrr as T       # noqa: E402  (sets up air-400 path + worker cap)
import infer as air_infer                        # noqa: E402  (dispatching build_model by MODEL.NAME)
from trainers.base_trainer import BaseTrainer    # noqa: E402
from loss.psdmse import PSDMSE                    # noqa: E402

KF = ROOT / "data" / "organized" / "kfold_learned"
CACHE = str(ROOT / "data" / "pedrr_learned_cache" / "full")
NFOLDS = 5


def _who_thr(age):
    return 60.0 if age < 2 else (50.0 if age < 12 else 40.0)


def cached_subjects():
    subs = set()
    for f in glob.glob(os.path.join(CACHE, "PEDRR", "*", "*.npy")):
        m = re.match(r"(ID-\d+)", os.path.basename(f))
        if m:
            subs.add(m.group(1))
    return subs


def extract_test_rr(trainer, loader):
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
            data = data.view(N * D, C, H, W); labels = labels.view(-1, 1)
            valid = (data.shape[0] // trainer.base_len) * trainer.base_len
            data, labels = data[:valid], labels[:valid]
            if trainer.model_name.lower().startswith("efficientphys"):
                last = torch.unsqueeze(data[-1], 0).repeat(trainer.num_of_gpu, 1, 1, 1)
                data = torch.cat((data, last), 0)
            pred = trainer.model(data)
            sp, _sl, _dm, fm = trainer._aggregate_batch_by_subject(pred, labels, subj_idx, ds_names, fs_list)
            for s in sp:
                preds[s].extend(sp[s]); fs_map[s] = fm[s]
    out = {}
    for s in preds:
        ps = torch.cat(preds[s], 0).detach().cpu().numpy().astype(np.float64).flatten()
        _p, rr = trainer.post_processor.post_process(torch.from_numpy(ps), fs=fs_map[s],
                                                     diff_flag=True, use_bandpass=True, eval_method="FFT")
        out[s] = float(rr)
    return out


def main():
    import logging
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True, choices=["DeepPhys", "TSCAN", "VIRENet", "EfficientPhys"])
    ap.add_argument("--epochs", type=int, default=None, help="override FULL_DEFAULTS epochs (matched-budget check)")
    ap.add_argument("--tag", default="", help="suffix for output dir, e.g. _100ep")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("extractor_cv")
    T.air_main.set_random_seeds(100)
    combined_manifest, _bs, _sg = T.build_combined_manifest(False)
    OUT = ROOT / "outputs" / "paper_runs" / f"pedrr_kfold_{args.model_name.lower()}{args.tag}"
    OUT.mkdir(parents=True, exist_ok=True)
    cached = cached_subjects()
    print(f"[extractor_cv] model={args.model_name}  cached_subjects={len(cached)}", flush=True)

    for k in range(NFOLDS):
        fd = OUT / f"fold{k}"; (fd / "test_eval").mkdir(parents=True, exist_ok=True)
        outcsv = fd / "test_eval" / "prism_breath_predictions.csv"
        if outcsv.exists():
            print(f"[skip] {args.model_name} fold{k}", flush=True); continue
        tr = [r for r in json.load(open(KF / f"fold{k}" / "train.json")) if r["subject_id"] in cached]
        va = [r for r in json.load(open(KF / f"fold{k}" / "val.json")) if r["subject_id"] in cached]
        te = [r for r in json.load(open(KF / f"fold{k}" / "test.json")) if r["subject_id"] in cached]
        ss = {"train": [str(r["subject_id"]) for r in tr], "val": [str(r["subject_id"]) for r in va],
              "test": [str(r["subject_id"]) for r in te]}
        p = dict(T.FULL_DEFAULTS); p["train_limit"] = 0; p["val_limit"] = 0; p["test_limit"] = 0
        if args.epochs:
            p["epochs"] = int(args.epochs)
        config = T.build_config(p, combined_manifest, ss, CACHE, str(fd / "run"), f"{args.model_name}_fold{k}")
        config["MODEL"]["NAME"] = args.model_name
        T.air_main.listify_datasets(config)
        datasets = T.air_main.get_split_datasets(config, {T.DATASET_NAME: T.PedRRDataset})
        train_loader, val_loader, test_loader = T.build_loaders(config, datasets, int(p["seed"]))
        model = air_infer.build_model(config, logger)          # dispatch by MODEL.NAME
        trainer = BaseTrainer(model=model, config=config,
                              train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
        trainer.criterion[T.DATASET_NAME] = PSDMSE(high_pass=T.PEDRR_HIGH_HZ, low_pass=T.PEDRR_LOW_HZ)
        trainer.post_processor = T.PedRRPostProcessor(T.PEDRR_LOW_HZ, T.PEDRR_HIGH_HZ)
        print(f"\n===== [{args.model_name} fold{k}] train={len(tr)} test={len(te)} =====", flush=True)
        try:
            trainer.train()
            rr_by_subj = extract_test_rr(trainer, test_loader)
        except Exception as e:
            print(f"[FAIL] {args.model_name} fold{k}: {type(e).__name__}: {e}", flush=True); continue
        rr_map = {}
        for ak, v in rr_by_subj.items():           # aggregation key is "subject_video"
            rr_map[ak] = v
            rr_map[ak.rsplit("_", 1)[0]] = v
        rows = []
        for r in te:
            key = f"{r['subject_id']}_{r['video_id']}"
            rr = float(rr_map.get(key, rr_map.get(str(r["subject_id"]), float("nan"))))
            gt = float(r["gt_rr_bpm"]); age = float(r.get("age_months", 6)); thr = _who_thr(age); mg = rr - thr
            rows.append({"subject_id": str(r["subject_id"]), "video_id": r["video_id"], "age_months": age,
                         "gt_rr_bpm": gt, "pred_rr_bpm": rr, "gt_bpm": gt, "pred_bpm": rr,
                         "threshold_bpm": thr, "gt_tachy": int(gt >= thr),
                         "tachy_prob": float(1.0 / (1.0 + np.exp(-mg / 5.0)))})
        pd.DataFrame(rows).to_csv(outcsv, index=False)
        mae = np.nanmean([abs(x["pred_rr_bpm"] - x["gt_rr_bpm"]) for x in rows])
        print(f"[{args.model_name} fold{k}] wrote {len(rows)} preds, RR-MAE={mae:.2f}", flush=True)
    print(f"DONE: {args.model_name} CV -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
