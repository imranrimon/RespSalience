"""Give DeepPhys a fair fold3 number: it collapsed under seed 100 (near-constant ~16 BPM, a
from-scratch degenerate low-freq attractor). Re-train fold3 from scratch under 3 alternate seeds,
report each seed's RR-MAE, and install the MEDIAN-MAE seed as the canonical fold3 predictions
(backing up the collapsed original). The other 4 folds trained stably and are left untouched.

Reuses run_extractor_cv (extract_test_rr, cached_subjects, _who_thr) + train_learned_extractor_pedrr
machinery, exactly as the main SOTA CV — only the seed and the single fold differ.
"""
from __future__ import annotations
import json, logging, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
import train_learned_extractor_pedrr as T       # noqa: E402
import infer as air_infer                        # noqa: E402
import run_extractor_cv as RX                     # noqa: E402  (extract_test_rr, cached_subjects, _who_thr)
from trainers.base_trainer import BaseTrainer    # noqa: E402
from loss.psdmse import PSDMSE                    # noqa: E402

KF = ROOT / "data" / "organized" / "kfold_learned"
CACHE = str(ROOT / "data" / "pedrr_learned_cache" / "full")
OUT = ROOT / "outputs" / "paper_runs" / "pedrr_kfold_deepphys"
MODEL_NAME = "DeepPhys"
FOLD = 3
SEEDS = [101, 202, 303]


def run_seed(seed, combined_manifest, cached, logger):
    T.air_main.set_random_seeds(seed)
    tr = [r for r in json.load(open(KF / f"fold{FOLD}" / "train.json")) if r["subject_id"] in cached]
    va = [r for r in json.load(open(KF / f"fold{FOLD}" / "val.json")) if r["subject_id"] in cached]
    te = [r for r in json.load(open(KF / f"fold{FOLD}" / "test.json")) if r["subject_id"] in cached]
    ss = {"train": [str(r["subject_id"]) for r in tr], "val": [str(r["subject_id"]) for r in va],
          "test": [str(r["subject_id"]) for r in te]}
    p = dict(T.FULL_DEFAULTS); p["seed"] = seed
    p["train_limit"] = 0; p["val_limit"] = 0; p["test_limit"] = 0
    seed_dir = OUT / "fold3_reseed" / f"s{seed}"; (seed_dir / "test_eval").mkdir(parents=True, exist_ok=True)
    config = T.build_config(p, combined_manifest, ss, CACHE, str(seed_dir / "run"), f"{MODEL_NAME}_fold3_s{seed}")
    config["MODEL"]["NAME"] = MODEL_NAME
    T.air_main.listify_datasets(config)
    datasets = T.air_main.get_split_datasets(config, {T.DATASET_NAME: T.PedRRDataset})
    train_loader, val_loader, test_loader = T.build_loaders(config, datasets, int(p["seed"]))
    model = air_infer.build_model(config, logger)
    trainer = BaseTrainer(model=model, config=config,
                          train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    trainer.criterion[T.DATASET_NAME] = PSDMSE(high_pass=T.PEDRR_HIGH_HZ, low_pass=T.PEDRR_LOW_HZ)
    trainer.post_processor = T.PedRRPostProcessor(T.PEDRR_LOW_HZ, T.PEDRR_HIGH_HZ)
    print(f"\n===== [{MODEL_NAME} fold3 seed={seed}] train={len(tr)} test={len(te)} =====", flush=True)
    trainer.train()
    rr_by_subj = RX.extract_test_rr(trainer, test_loader)
    rr_map = {}
    for ak, v in rr_by_subj.items():
        rr_map[ak] = v; rr_map[ak.rsplit("_", 1)[0]] = v
    rows = []
    for r in te:
        key = f"{r['subject_id']}_{r['video_id']}"
        rr = float(rr_map.get(key, rr_map.get(str(r["subject_id"]), float("nan"))))
        gt = float(r["gt_rr_bpm"]); age = float(r.get("age_months", 6)); thr = RX._who_thr(age); mg = rr - thr
        rows.append({"subject_id": str(r["subject_id"]), "video_id": r["video_id"], "age_months": age,
                     "gt_rr_bpm": gt, "pred_rr_bpm": rr, "gt_bpm": gt, "pred_bpm": rr,
                     "threshold_bpm": thr, "gt_tachy": int(gt >= thr),
                     "tachy_prob": float(1.0 / (1.0 + np.exp(-mg / 5.0)))})
    csv = seed_dir / "test_eval" / "prism_breath_predictions.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    pr = np.array([x["pred_rr_bpm"] for x in rows]); g = np.array([x["gt_rr_bpm"] for x in rows])
    mae = float(np.nanmean(np.abs(pr - g))); pinned = int((np.abs(pr - np.median(pr)) < 2).sum())
    print(f"[{MODEL_NAME} fold3 seed={seed}] RR-MAE={mae:.2f}  pred_med={np.median(pr):.1f}  "
          f"pinned@med={pinned}/{len(pr)}", flush=True)
    return {"seed": seed, "mae": mae, "csv": csv, "pred_med": float(np.median(pr))}


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("reseed")
    combined_manifest, _bs, _sg = T.build_combined_manifest(False)
    cached = RX.cached_subjects()
    results = [run_seed(s, combined_manifest, cached, logger) for s in SEEDS]

    results_sorted = sorted(results, key=lambda d: d["mae"])
    median = results_sorted[len(results_sorted) // 2]                 # median-MAE seed (3 seeds -> middle)
    print("\n===== fold3 reseed summary =====", flush=True)
    for d in results:
        tag = "  <- MEDIAN (install)" if d["seed"] == median["seed"] else ""
        print(f"  seed {d['seed']}: MAE={d['mae']:.2f}  pred_med={d['pred_med']:.1f}{tag}", flush=True)
    print(f"  (original collapsed seed 100: MAE=22.88, pred_med=16.1)", flush=True)

    canonical = OUT / "fold3" / "test_eval" / "prism_breath_predictions.csv"
    backup = OUT / "fold3" / "test_eval" / "prism_breath_predictions.collapsed_seed100.csv"
    if canonical.exists() and not backup.exists():
        shutil.copy2(canonical, backup); print(f"  backed up collapsed fold3 -> {backup.name}", flush=True)
    shutil.copy2(median["csv"], canonical)
    print(f"  installed seed {median['seed']} (MAE {median['mae']:.2f}) as canonical fold3", flush=True)
    print("DONE reseed_deepphys_fold3", flush=True)


if __name__ == "__main__":
    main()
