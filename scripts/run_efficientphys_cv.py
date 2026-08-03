"""Fair SOTA baseline: EfficientPhys+Coarse2Fine+Body under OUR 5-fold subject-disjoint CV.

Per fold: fine-tune the AIR-pretrained EfficientPhys on the fold's TRAIN subjects (reusing the
cached coarse2fine tensors -> no re-preprocessing, no ID-1322 hang), then extract RR on the held-out
TEST fold. Writes clinical-compatible predictions per fold so clinical_endpoint_eval + compare_clinical_models
score it under the IDENTICAL protocol as PRISM (tachypnea by WHO band, bootstrap CIs, DeLong/McNemar).

Reuses train_learned_extractor_pedrr (build_config/build_model/build_loaders/PedRRPostProcessor) by
overriding split_subjects with the CV fold assignment. Resume-safe (skips folds already extracted).
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
import torch                                    # noqa: E402
import train_learned_extractor_pedrr as T       # noqa: E402
from trainers.base_trainer import BaseTrainer    # noqa: E402
from loss.psdmse import PSDMSE                    # noqa: E402

KF = ROOT / "data" / "organized" / "kfold_learned"
CACHE = str(ROOT / "data" / "pedrr_learned_cache" / "full")
INIT_CKPT = str(ROOT / "data" / "models" / "EfficientPhys_Coarse2fine_Body_best.pth")  # SOTA init
OUT = ROOT / "outputs" / "paper_runs" / "pedrr_kfold_efficientphys"
NFOLDS = 5


def _who_thr(age):
    return 60.0 if age < 2 else (50.0 if age < 12 else 40.0)


def cached_subjects(cache_root):
    """Subjects whose coarse2fine tensors are already cached (avoids re-preprocessing the 34 uncached
    clips, which include the pathological ID-1322 hang). Comparison runs on this ~298-subject cohort;
    compare_clinical_models aligns PRISM to the same subjects automatically (clip-key intersection)."""
    import glob, re
    subs = set()
    for f in glob.glob(os.path.join(cache_root, "PEDRR", "*", "*.npy")):
        m = re.match(r"(ID-\d+)", os.path.basename(f))
        if m:
            subs.add(m.group(1))
    return subs


def extract_test_rr(trainer, loader):
    """Per-subject post-processed RR on the test fold (mirrors build_learned_flow_signals.extract_split)."""
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
        _proc, rr = trainer.post_processor.post_process(
            torch.from_numpy(ps), fs=fs_map[s], diff_flag=True, use_bandpass=True, eval_method="FFT")
        out[s] = float(rr)
    return out


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    T.air_main.set_random_seeds(100)
    combined_manifest, _base_split, subject_gt = T.build_combined_manifest(False)
    OUT.mkdir(parents=True, exist_ok=True)
    cached = cached_subjects(CACHE)
    print(f"[cache] {len(cached)} cached subjects (uncached clips are dropped to avoid the coarse2fine hang)", flush=True)

    for k in range(NFOLDS):
        fd = OUT / f"fold{k}"; (fd / "test_eval").mkdir(parents=True, exist_ok=True)
        outcsv = fd / "test_eval" / "prism_breath_predictions.csv"
        if outcsv.exists():
            print(f"[skip] fold{k} already extracted", flush=True); continue
        # restrict to CACHED subjects only (hang-free, already preprocessed)
        tr = [r for r in json.load(open(KF / f"fold{k}" / "train.json")) if r["subject_id"] in cached]
        va = [r for r in json.load(open(KF / f"fold{k}" / "val.json")) if r["subject_id"] in cached]
        te = [r for r in json.load(open(KF / f"fold{k}" / "test.json")) if r["subject_id"] in cached]
        split_subjects = {"train": [str(r["subject_id"]) for r in tr],
                          "val": [str(r["subject_id"]) for r in va],
                          "test": [str(r["subject_id"]) for r in te]}
        p = dict(T.FULL_DEFAULTS); p["train_limit"] = 0; p["val_limit"] = 0; p["test_limit"] = 0
        config = T.build_config(p, combined_manifest, split_subjects, CACHE, str(fd / "run"), f"eff_fold{k}")
        T.air_main.listify_datasets(config)
        datasets = T.air_main.get_split_datasets(config, {T.DATASET_NAME: T.PedRRDataset})
        train_loader, val_loader, test_loader = T.build_loaders(config, datasets, int(p["seed"]))
        model = T.build_model(config)
        if os.path.exists(INIT_CKPT):
            T.load_init_checkpoint(model, INIT_CKPT)
        trainer = BaseTrainer(model=model, config=config,
                              train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
        trainer.criterion[T.DATASET_NAME] = PSDMSE(high_pass=T.PEDRR_HIGH_HZ, low_pass=T.PEDRR_LOW_HZ)
        trainer.post_processor = T.PedRRPostProcessor(T.PEDRR_LOW_HZ, T.PEDRR_HIGH_HZ)
        print(f"\n===== [efficientphys fold{k}] train={len(split_subjects['train'])} "
              f"test={len(split_subjects['test'])} =====", flush=True)
        trainer.train()
        rr_by_subj = extract_test_rr(trainer, test_loader)
        rr_map = {}
        for ak, v in rr_by_subj.items():           # aggregation key is "subject_video"
            rr_map[ak] = v
            rr_map[ak.rsplit("_", 1)[0]] = v        # also index by subject (pedrr = 1 clip/subject)

        rows = []
        for r in te:
            key = f"{r['subject_id']}_{r['video_id']}"
            rr = float(rr_map.get(key, rr_map.get(str(r["subject_id"]), float("nan"))))
            gt = float(r["gt_rr_bpm"]); age = float(r.get("age_months", 6))
            thr = _who_thr(age); margin = rr - thr
            rows.append({"subject_id": str(r["subject_id"]), "video_id": r["video_id"], "age_months": age,
                         "gt_rr_bpm": gt, "pred_rr_bpm": rr, "gt_bpm": gt, "pred_bpm": rr,
                         "threshold_bpm": thr, "gt_tachy": int(gt >= thr),
                         "tachy_prob": float(1.0 / (1.0 + np.exp(-margin / 5.0)))})
        pd.DataFrame(rows).to_csv(outcsv, index=False)
        mae = np.nanmean([abs(x["pred_rr_bpm"] - x["gt_rr_bpm"]) for x in rows])
        print(f"[fold{k}] wrote {len(rows)} preds, test RR-MAE={mae:.2f}", flush=True)
    print(f"DONE: EfficientPhys CV -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
