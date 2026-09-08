"""Re-select published-extractor checkpoints on validation AUROC instead of validation loss.

Why this exists
---------------
The paper's checkpoint-selection experiment showed that choosing PRISM's weights by validation
regression error costs tachypnea AUROC. That comparison is only fair if the BASELINES are given the
same opportunity: their checkpoints were also selected by a regression criterion (PSD-MSE validation
loss), so a decision-selected PRISM against loss-selected baselines would be a rigged fight.

Every training epoch was checkpointed for the four published extractors
(`fold*/run/model/<Net>/<exp>/<Net>_Epoch*.pth`), so the fair arm needs no retraining -- only
inference. For each fold this script:

  1. runs every saved epoch checkpoint over the VALIDATION fold and scores tachypnea AUROC,
  2. keeps the epoch with the best validation AUROC,
  3. runs that checkpoint over the TEST fold and writes predictions in the standard schema.

Baseline `tachy_prob` is `sigmoid((rr - who_threshold)/5)` -- a strictly monotone function of the
margin -- so AUROC ranks on the margin exactly as in the published runs. Nothing about the scoring
rule changes; only which epoch is kept.

Fidelity check (`--verify-published`): re-runs the ORIGINAL `<Net>_best.pth` through this same
extraction path and compares against the committed predictions CSV. If that does not reproduce, the
harness is not faithful and the new numbers must not be trusted.

    python scripts/reselect_baseline_checkpoints.py --model-name EfficientPhys --verify-published
    python scripts/reselect_baseline_checkpoints.py --model-name DeepPhys --tag _100ep
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
import torch                                    # noqa: E402
from sklearn.metrics import roc_auc_score        # noqa: E402

import train_learned_extractor_pedrr as T       # noqa: E402  (sets up air-400 path + worker cap)
import infer as air_infer                        # noqa: E402  (dispatching build_model by MODEL.NAME)
from trainers.base_trainer import BaseTrainer    # noqa: E402
from loss.psdmse import PSDMSE                    # noqa: E402

KF = ROOT / "data" / "organized" / "kfold_learned"
CACHE = str(ROOT / "data" / "pedrr_learned_cache" / "full")
NFOLDS = 5


def who_thr(age: float) -> float:
    return 60.0 if age < 2 else (50.0 if age < 12 else 40.0)


def cached_subjects() -> set:
    subs = set()
    for f in glob.glob(os.path.join(CACHE, "PEDRR", "*", "*.npy")):
        m = re.match(r"(ID-\d+)", os.path.basename(f))
        if m:
            subs.add(m.group(1))
    return subs


def extract_rr(trainer, loader) -> dict:
    """Per-subject post-processed RR. Mirrors run_extractor_cv.extract_test_rr exactly."""
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


def rows_for(records, rr_by_subj) -> list:
    """Standard prediction schema; identical construction to the published drivers."""
    rr_map = {}
    for ak, v in rr_by_subj.items():            # aggregation key is "subject_video"
        rr_map[ak] = v
        rr_map[ak.rsplit("_", 1)[0]] = v
    rows = []
    for r in records:
        key = f"{r['subject_id']}_{r['video_id']}"
        rr = float(rr_map.get(key, rr_map.get(str(r["subject_id"]), float("nan"))))
        gt = float(r["gt_rr_bpm"]); age = float(r.get("age_months", 6))
        thr = who_thr(age); mg = rr - thr
        rows.append({"subject_id": str(r["subject_id"]), "video_id": r["video_id"], "age_months": age,
                     "gt_rr_bpm": gt, "pred_rr_bpm": rr, "gt_bpm": gt, "pred_bpm": rr,
                     "threshold_bpm": thr, "gt_tachy": int(gt >= thr),
                     "tachy_prob": float(1.0 / (1.0 + np.exp(-mg / 5.0)))})
    return rows


def auroc_of(rows) -> float:
    y = np.array([r["gt_tachy"] for r in rows], int)
    s = np.array([r["tachy_prob"] for r in rows], float)
    ok = np.isfinite(s)
    if ok.sum() < len(s):
        y, s = y[ok], s[ok]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def mae_of(rows) -> float:
    return float(np.nanmean([abs(r["pred_rr_bpm"] - r["gt_rr_bpm"]) for r in rows]))


def load_weights(model, path: str) -> None:
    """Load a checkpoint saved from a possibly DataParallel-wrapped model into `model`."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith("module.") for k in sd):
        sd = sd["state_dict"]
    tgt = model.module if hasattr(model, "module") else model
    stripped = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    tgt.load_state_dict(stripped, strict=True)


def epoch_checkpoints(fold_dir: Path, net: str) -> list:
    """[(epoch:int, path:str)] sorted by epoch, from fold*/run/model/<Net>/<exp>/<Net>_Epoch<N>.pth."""
    pats = glob.glob(str(fold_dir / "run" / "model" / "*" / "*" / f"{net}_Epoch*.pth"))
    out = []
    for p in pats:
        m = re.search(rf"{re.escape(net)}_Epoch(\d+)\.pth$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def best_checkpoint(fold_dir: Path, net: str):
    hits = glob.glob(str(fold_dir / "run" / "model" / "*" / "*" / f"{net}_best.pth"))
    return hits[0] if hits else None


def main() -> int:
    import logging
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True,
                    choices=["DeepPhys", "TSCAN", "VIRENet", "EfficientPhys"])
    ap.add_argument("--tag", default="", help="suffix of the SOURCE run dir, e.g. _100ep")
    ap.add_argument("--out-suffix", default="_aurocsel", help="suffix for the re-selected output run")
    ap.add_argument("--folds", default="", help="comma-separated fold indices; default all five")
    ap.add_argument("--verify-published", action="store_true",
                    help="also re-run the original *_best.pth and compare to the committed CSV")
    ap.add_argument("--max-epochs-scanned", type=int, default=0,
                    help="cap epochs scanned per fold to the LAST N (0 = all)")
    ap.add_argument("--scan-budget", type=int, default=0,
                    help="give every model the SAME number of candidate checkpoints by striding "
                         "evenly across its saved epochs (0 = off). DeepPhys stored 100 epochs where "
                         "the others stored 15; letting it search 100 and them 15 would hand it a "
                         "larger selection budget, which is a confound, not a fair arm.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("reselect")
    T.air_main.set_random_seeds(100)
    combined_manifest, _bs, _sg = T.build_combined_manifest(False)

    net = args.model_name
    SRC = ROOT / "outputs" / "paper_runs" / f"pedrr_kfold_{net.lower()}{args.tag}"
    DST = ROOT / "outputs" / "paper_runs" / f"pedrr_kfold_{net.lower()}{args.tag}{args.out_suffix}"
    if not SRC.is_dir():
        raise SystemExit(f"source run not found: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)
    cached = cached_subjects()
    folds = [int(x) for x in args.folds.split(",") if x.strip()] if args.folds else list(range(NFOLDS))
    print(f"[reselect] {net}{args.tag}  src={SRC.name}  dst={DST.name}  folds={folds}  "
          f"cached_subjects={len(cached)}", flush=True)

    summary = []
    for k in folds:
        sfd, dfd = SRC / f"fold{k}", DST / f"fold{k}"
        (dfd / "test_eval").mkdir(parents=True, exist_ok=True)
        outcsv = dfd / "test_eval" / "prism_breath_predictions.csv"
        report = dfd / "selection_report.json"
        if outcsv.exists() and report.exists():
            print(f"[skip] fold{k} already re-selected", flush=True)
            summary.append(json.loads(report.read_text()))
            continue

        ckpts = epoch_checkpoints(sfd, net)
        if not ckpts:
            print(f"[FAIL] fold{k}: no epoch checkpoints under {sfd}", flush=True)
            continue
        if args.max_epochs_scanned:
            ckpts = ckpts[-args.max_epochs_scanned:]
        if args.scan_budget and len(ckpts) > args.scan_budget:
            idx = np.unique(np.linspace(0, len(ckpts) - 1, args.scan_budget).round().astype(int))
            ckpts = [ckpts[i] for i in idx]

        tr = [r for r in json.load(open(KF / f"fold{k}" / "train.json")) if r["subject_id"] in cached]
        va = [r for r in json.load(open(KF / f"fold{k}" / "val.json")) if r["subject_id"] in cached]
        te = [r for r in json.load(open(KF / f"fold{k}" / "test.json")) if r["subject_id"] in cached]
        ss = {"train": [str(r["subject_id"]) for r in tr], "val": [str(r["subject_id"]) for r in va],
              "test": [str(r["subject_id"]) for r in te]}
        p = dict(T.FULL_DEFAULTS); p["train_limit"] = 0; p["val_limit"] = 0; p["test_limit"] = 0
        config = T.build_config(p, combined_manifest, ss, CACHE, str(dfd / "run"), f"{net}_fold{k}")
        config["MODEL"]["NAME"] = net
        T.air_main.listify_datasets(config)
        datasets = T.air_main.get_split_datasets(config, {T.DATASET_NAME: T.PedRRDataset})
        train_loader, val_loader, test_loader = T.build_loaders(config, datasets, int(p["seed"]))
        model = air_infer.build_model(config, logger)
        trainer = BaseTrainer(model=model, config=config, train_loader=train_loader,
                              val_loader=val_loader, test_loader=test_loader)
        trainer.criterion[T.DATASET_NAME] = PSDMSE(high_pass=T.PEDRR_HIGH_HZ, low_pass=T.PEDRR_LOW_HZ)
        trainer.post_processor = T.PedRRPostProcessor(T.PEDRR_LOW_HZ, T.PEDRR_HIGH_HZ)

        print(f"\n===== [{net} fold{k}] scanning {len(ckpts)} epoch checkpoints on "
              f"val n={len(va)} =====", flush=True)

        # --- fidelity check: does the published best.pth reproduce the committed CSV? ---
        verify = None
        if args.verify_published:
            # The published drivers call extract_test_rr(trainer, ...) straight after train(), so the
            # committed predictions come from the LAST epoch resident in memory, not from best.pth
            # (verified bit-exact for EfficientPhys fold0 by identify_baseline_checkpoint.py).
            bp = ckpts[-1][1] if ckpts else best_checkpoint(sfd, net)
            comm = sfd / "test_eval" / "prism_breath_predictions.csv"
            if bp and comm.exists():
                load_weights(trainer.model, bp)
                rr = extract_rr(trainer, test_loader)
                got = pd.DataFrame(rows_for(te, rr)).sort_values("subject_id").reset_index(drop=True)
                want = pd.read_csv(comm).sort_values("subject_id").reset_index(drop=True)
                j = want.merge(got, on="subject_id", suffixes=("_c", "_n"))
                d = (j.pred_rr_bpm_c - j.pred_rr_bpm_n).abs()
                verify = {"n": int(len(j)), "max_abs_diff": float(d.max()),
                          "mean_abs_diff": float(d.mean()),
                          "committed_auroc": auroc_of(want.to_dict("records")),
                          "reproduced_auroc": auroc_of(got.to_dict("records"))}
                flag = "FAITHFUL" if verify["max_abs_diff"] < 1e-6 else "MISMATCH"
                print(f"  [verify] published last-epoch weights reproduce committed preds: {flag} "
                      f"(max |dRR| = {verify['max_abs_diff']:.3e}, "
                      f"AUROC {verify['committed_auroc']:.4f} -> {verify['reproduced_auroc']:.4f})",
                      flush=True)

        # --- scan every epoch on validation ---
        scan = []
        for ep, path in ckpts:
            load_weights(trainer.model, path)
            rr = extract_rr(trainer, val_loader)
            vrows = rows_for(va, rr)
            a, m = auroc_of(vrows), mae_of(vrows)
            scan.append({"epoch": ep, "val_auroc": a, "val_mae": m, "path": path})
            print(f"  epoch {ep:>3}  val AUROC {a:.4f}  val MAE {m:6.2f}", flush=True)

        usable = [s for s in scan if np.isfinite(s["val_auroc"])]
        if not usable:
            print(f"[FAIL] fold{k}: no finite validation AUROC (val fold may be single-class)", flush=True)
            continue
        pick = max(usable, key=lambda s: s["val_auroc"])
        pick_mae = min(usable, key=lambda s: s["val_mae"])
        print(f"  -> AUROC-selected epoch {pick['epoch']} (val AUROC {pick['val_auroc']:.4f}); "
              f"MAE-selected would be epoch {pick_mae['epoch']} (val MAE {pick_mae['val_mae']:.2f})",
              flush=True)

        # --- evaluate the chosen checkpoint on TEST ---
        load_weights(trainer.model, pick["path"])
        rr = extract_rr(trainer, test_loader)
        trows = rows_for(te, rr)
        pd.DataFrame(trows).to_csv(outcsv, index=False)
        rec = {"fold": k, "model": net, "selected_epoch": pick["epoch"],
               "val_auroc": pick["val_auroc"], "val_mae": pick["val_mae"],
               "mae_selected_epoch": pick_mae["epoch"],
               "test_auroc": auroc_of(trows), "test_mae": mae_of(trows),
               "n_test": len(trows), "scan": [{kk: v for kk, v in s.items() if kk != "path"}
                                              for s in scan],
               "verify_published": verify}
        report.write_text(json.dumps(rec, indent=2))
        summary.append(rec)
        print(f"[{net} fold{k}] test AUROC {rec['test_auroc']:.4f}  MAE {rec['test_mae']:.2f}  "
              f"-> {outcsv}", flush=True)

    if summary:
        print(f"\n===== {net}{args.tag} summary =====")
        print(f"{'fold':<6}{'epoch':>7}{'val AUROC':>11}{'test AUROC':>12}{'test MAE':>10}")
        for r in summary:
            print(f"{r['fold']:<6}{r['selected_epoch']:>7}{r['val_auroc']:>11.4f}"
                  f"{r['test_auroc']:>12.4f}{r['test_mae']:>10.2f}")
    print(f"\nDONE -> {DST}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
