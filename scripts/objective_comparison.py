# RUN: conda run -n resp_env python scripts/objective_comparison.py --signal-dir data/pedrr_signals_learned
#      (per-fold frozen encoder defaults to outputs/paper_runs/pedrr_kfold_learned/fold{fold}/train/prism_breath_best.pt)
"""
Objective-comparison harness  (reviewer #4 -- THE pivotal experiment).

HYPOTHESIS
----------
The classification / distributional framing of tachypnea screening beats
regression-then-threshold because of the training OBJECTIVE, not because of a
different learned REPRESENTATION. If we FREEZE one PRISM encoder (so every head
consumes the *identical* fused representation, on the *identical* subjects and
5-fold subject-disjoint split) and vary ONLY the prediction head + loss, then a
classification-style objective (salience / ordinal / direct-BCE) should still
win the clinical endpoint (tachypnea AUROC / Sens / Spec) over the L1 / Huber
regression objectives -- while regression may still win raw RR-MAE. That
dissociation (better clinical decision, not-better RR-MAE) is the reviewer-#4
result: the framing gain is objective-driven, representation-controlled.

DESIGN (representation-controlled)
----------------------------------
* Frozen encoder: for fold k we load that fold's own checkpoint
  (`pedrr_kfold_learned/fold{k}/train/prism_breath_best.pt`), which never saw
  fold k's TEST subjects -> the frozen features stay subject-disjoint from the
  test set. (Pass a single --checkpoint without `{fold}` to force one encoder
  for all folds; that re-introduces representation leakage on 4/5 folds and is
  only for a quick sanity pass -- a warning is printed.)
* Frozen feature = EXACTLY the vector PRISM's own heads consume:
  concat(latent_state [d_model], meta [17]) reconstructed from the model's
  exposed output tensors (bpm/scales/qualities/mask/fusion_weights/disagreement/
  quality_mean). Standardized per-fold with TRAIN statistics.
* Every head shares the SAME MLP trunk capacity (Linear->GELU->Dropout->Linear);
  only the output dimension, the loss, and the decode differ. So the ONLY thing
  that changes between rows is the objective.

HEADS (objectives)
------------------
  l1              L1 regression                      -> RR; tachy score = margin
  huber           Huber (smooth-L1) regression       -> RR; tachy score = margin
  gaussian_nll    mean+logvar Gaussian NLL           -> RR=mean; score=P(RR>=thr)
  quantile        pinball over quantile grid         -> RR=median; score=1-CDF(thr)
  ordinal         cumulative-link ordered RR bins    -> RR=E[bin]; score=P(RR>thr)
  salience        RespSalience Gaussian-target grid  -> RR=grid-mean; score=mass>=thr
  bce             direct BCE tachypnea               -> score=sigmoid (no RR)
  focal           focal BCE tachypnea (gamma)        -> score=sigmoid (no RR)
  calib_margin    Huber RR + Platt-calibrated margin -> RR; score=sigmoid(a*margin+b)

For every head we report BOTH RR-MAE (NaN for the RR-less bce/focal heads) and
tachypnea AUROC / AUPRC / Sens / Spec / Se@Sp90 on the same pooled test clips,
with subject-bootstrap 95% CIs.

RESUME / EXECUTION
------------------
Single-process, num_workers=0. GPU-preferred (encoder forward + head training),
runs on CPU too (features are tiny). Resume-safe: per-fold feature caches under
outputs/objective_comparison/features/ and per-head `.done` markers; re-running
skips completed heads unless --overwrite. THIS BOX CANNOT TRAIN (WinError 5) --
launch on the RTX 8000.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ndtr  # standard-normal CDF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.core.respiratory_benchmark_dataset import RespiratoryBenchmarkDataset  # noqa: E402
from src.core.prism_breath_model import (  # noqa: E402
    build_prism_breath_model_from_config,
    tensor_tachypnea_threshold,
)

try:
    from compare_clinical_models import delong_test  # noqa: E402
    _HAVE_DELONG = True
except Exception:
    _HAVE_DELONG = False

KFOLD = ROOT / "data" / "organized" / "kfold_learned"
HEADS = ["l1", "huber", "gaussian_nll", "quantile", "ordinal", "salience", "bce", "focal", "calib_margin"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Frozen encoder -> fixed feature vector
# --------------------------------------------------------------------------- #
def load_frozen_model(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    config = ckpt.get("config", {})
    model_config = config.get("model_config", config)
    model = build_prism_breath_model_from_config(model_config).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    seq_len = int(model_config.get("seq_len", config.get("seq_len", 300)))
    return model, seq_len


@torch.no_grad()
def frozen_features(model, batch) -> torch.Tensor:
    """Reconstruct PRISM's own fused head-input = concat(latent_state, meta)."""
    out = model(
        depth=batch["depth"].to(DEVICE),
        flow=batch["flow"].to(DEVICE),
        audio=batch["audio"].to(DEVICE),
        age_months=batch["age_months"].to(DEVICE),
        modality_mask=batch["modality_mask"].to(DEVICE),
        dataset_ids=batch["dataset_id"].to(DEVICE),
    )
    meta = torch.cat(
        [
            out["modality_pred_tensor"],
            out["modality_scale_tensor"],
            out["quality_tensor"],
            out["availability_tensor"],
            out["fusion_weights_tensor"],
            out["disagreement_score"].unsqueeze(-1),
            out["quality_mean"].unsqueeze(-1),
        ],
        dim=-1,
    )
    return torch.cat([out["latent_state"], meta], dim=-1).detach().cpu()


def extract_split(model, seq_len, signal_dir, manifest, fold, batch_size) -> Dict[str, np.ndarray]:
    ds = RespiratoryBenchmarkDataset(signal_dir=signal_dir, manifest_path=str(manifest), seq_len=seq_len)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, gt, age, subj, clip = [], [], [], [], []
    for batch in loader:
        feats.append(frozen_features(model, batch).numpy())
        gt.append(batch["gt_rr"].numpy())
        age.append(batch["age_months"].numpy())
        subj.extend([str(s) for s in batch["subject_id"]])
        clip.extend([f"{s}_{v}" for s, v in zip(batch["subject_id"], batch["video_id"])])
    gt = np.concatenate(gt).astype(np.float64)
    age = np.concatenate(age).astype(np.float64)
    thr = tensor_tachypnea_threshold(torch.tensor(age)).numpy().astype(np.float64)
    return {
        "feats": np.concatenate(feats).astype(np.float32),
        "gt": gt,
        "age": age,
        "thr": thr,
        "gt_tachy": (gt >= thr).astype(np.int64),
        "subject": np.asarray(subj),
        "clip": np.asarray(clip),
        "fold": np.full(len(gt), fold, dtype=np.int64),
    }


def get_fold_features(model, seq_len, signal_dir, fold, cache_dir, batch_size, refresh) -> Dict[str, Dict]:
    cache = cache_dir / f"fold{fold}.npz"
    if cache.exists() and not refresh:
        z = np.load(cache, allow_pickle=True)
        return {sp: {k[len(sp) + 1:]: z[k] for k in z.files if k.startswith(sp + "|")} for sp in ("train", "val", "test")}
    splits = {}
    flat = {}
    for sp in ("train", "val", "test"):
        d = extract_split(model, seq_len, signal_dir, KFOLD / f"fold{fold}" / f"{sp}.json", fold, batch_size)
        splits[sp] = d
        for k, v in d.items():
            flat[f"{sp}|{k}"] = v
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **flat)
    return splits


# --------------------------------------------------------------------------- #
# Shared head trunk (identical capacity for every objective)
# --------------------------------------------------------------------------- #
class Head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def focal_bce(logit, target, gamma):
    p = torch.sigmoid(logit)
    pt = torch.where(target > 0.5, p, 1.0 - p).clamp(1e-6, 1 - 1e-6)
    return (-((1.0 - pt) ** gamma) * torch.log(pt)).mean()


def gauss_grid_target(gt: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((grid[None, :] - gt[:, None]) / sigma) ** 2).astype(np.float32)  # peak-1 CREPE-style


# --------------------------------------------------------------------------- #
# Train one head on frozen features; return per-clip test predictions
# --------------------------------------------------------------------------- #
def run_head(head: str, tr, va, te, args, grid, quantiles, ord_bins) -> Dict[str, np.ndarray]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mu = tr["feats"].mean(0, keepdims=True)
    sd = tr["feats"].std(0, keepdims=True) + 1e-6

    def T(d):  # standardized feature tensor
        return torch.tensor((d["feats"] - mu) / sd, dtype=torch.float32, device=DEVICE)

    Xtr, Xva, Xte = T(tr), T(va), T(te)
    in_dim = Xtr.shape[1]

    # Standardize the RR target for the regression-family heads so L1/Huber/pinball/NLL
    # are well-conditioned (raw-bpm targets ~16-96 make sign-gradient descent converge far
    # too slowly, which would handicap regression vs the O(1)-target classification heads and
    # confound the objective comparison). Classification heads (ordinal/salience/bce/focal)
    # keep bpm targets. Decode un-normalizes.
    gmean = float(tr["gt"].mean())
    gstd = float(tr["gt"].std() + 1e-6)

    gtr = torch.tensor(tr["gt"], dtype=torch.float32, device=DEVICE)
    gva = torch.tensor(va["gt"], dtype=torch.float32, device=DEVICE)
    ttr = torch.tensor(tr["thr"], dtype=torch.float32, device=DEVICE)
    tva = torch.tensor(va["thr"], dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(tr["gt_tachy"], dtype=torch.float32, device=DEVICE)
    yva = torch.tensor(va["gt_tachy"], dtype=torch.float32, device=DEVICE)

    grid_t = torch.tensor(grid, dtype=torch.float32, device=DEVICE)
    qs_t = torch.tensor(quantiles, dtype=torch.float32, device=DEVICE)
    ob_t = torch.tensor(ord_bins, dtype=torch.float32, device=DEVICE)

    out_dim = {
        "l1": 1, "huber": 1, "calib_margin": 1,
        "gaussian_nll": 2,
        "quantile": len(quantiles),
        "ordinal": len(ord_bins),
        "salience": len(grid),
        "bce": 1, "focal": 1,
    }[head]
    net = Head(in_dim, out_dim, args.hidden, args.dropout).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def loss_fn(x, gt, thr, y):
        pred = net(x)
        gtn = (gt - gmean) / gstd  # standardized RR target for regression-family heads
        if head in ("l1",):
            return F.l1_loss(pred.squeeze(-1), gtn)
        if head in ("huber", "calib_margin"):
            return F.smooth_l1_loss(pred.squeeze(-1), gtn)
        if head == "gaussian_nll":
            mean, lv = pred[:, 0], pred[:, 1].clamp(-6.0, 8.0)
            return (0.5 * (torch.exp(-lv) * (mean - gtn) ** 2 + lv)).mean()
        if head == "quantile":
            e = gtn.unsqueeze(-1) - pred  # (B,Q) in standardized space
            return torch.maximum(qs_t * e, (qs_t - 1.0) * e).mean()
        if head == "ordinal":
            tgt = (gt.unsqueeze(-1) > ob_t.unsqueeze(0)).float()  # P(rr > b_j)
            return F.binary_cross_entropy_with_logits(pred, tgt)
        if head == "salience":
            tgt = torch.exp(-0.5 * ((grid_t.unsqueeze(0) - gt.unsqueeze(-1)) / args.salience_sigma) ** 2)
            return F.binary_cross_entropy_with_logits(pred, tgt)
        if head == "bce":
            return F.binary_cross_entropy_with_logits(pred.squeeze(-1), y)
        if head == "focal":
            return focal_bce(pred.squeeze(-1), y, args.focal_gamma)
        raise ValueError(head)

    best_val, best_state, since = math.inf, None, 0
    idx = np.arange(len(gtr))
    for ep in range(args.epochs):
        net.train()
        np.random.shuffle(idx)
        for i in range(0, len(idx), args.batch_size):
            b = idx[i:i + args.batch_size]
            opt.zero_grad()
            loss_fn(Xtr[b], gtr[b], ttr[b], ytr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(Xva, gva, tva, yva))
        if vl < best_val - 1e-5:
            best_val, best_state, since = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            since += 1
            if args.head_patience and since >= args.head_patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    # ---- decode test predictions --------------------------------------- #
    net.eval()
    thr_te = te["thr"]
    with torch.no_grad():
        raw = net(Xte).cpu().numpy()

    has_rr = head not in ("bce", "focal")
    pred_rr = np.full(len(thr_te), np.nan)
    score = np.zeros(len(thr_te))

    if head in ("l1", "huber"):
        pred_rr = raw.squeeze(-1) * gstd + gmean  # un-normalize
        score = pred_rr - thr_te  # margin
        decision = (pred_rr >= thr_te).astype(int)
    elif head == "calib_margin":
        pred_rr = raw.squeeze(-1) * gstd + gmean
        # Platt-calibrate the TRAIN margin -> tachy, then map the test margin.
        with torch.no_grad():
            rr_tr = net(Xtr).cpu().numpy().squeeze(-1) * gstd + gmean
        m_tr = (rr_tr - tr["thr"]).reshape(-1, 1)
        from sklearn.linear_model import LogisticRegression
        try:
            clf = LogisticRegression(max_iter=2000).fit(m_tr, tr["gt_tachy"])
            score = clf.predict_proba((pred_rr - thr_te).reshape(-1, 1))[:, 1]
        except Exception:  # single-class train fold -> fall back to margin ranking
            score = pred_rr - thr_te
        decision = (score >= 0.5).astype(int)
    elif head == "gaussian_nll":
        mean_n, lv = raw[:, 0], np.clip(raw[:, 1], -6.0, 8.0)
        mean = mean_n * gstd + gmean  # un-normalize mean
        sigma = np.exp(0.5 * lv) * gstd  # un-normalize std
        pred_rr = mean
        score = 1.0 - ndtr((thr_te - mean) / np.maximum(sigma, 1e-3))  # P(RR>=thr)
        decision = (mean >= thr_te).astype(int)
    elif head == "quantile":
        qv = raw * gstd + gmean  # un-normalize each quantile (N,Q)
        med = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
        pred_rr = qv[:, med]
        for i in range(len(thr_te)):
            xp = np.sort(qv[i])  # quantile values (ascending)
            tau_at = np.interp(thr_te[i], xp, quantiles)  # CDF at threshold
            score[i] = 1.0 - float(np.clip(tau_at, 0.0, 1.0))
        decision = (pred_rr >= thr_te).astype(int)
    elif head == "ordinal":
        surv = 1.0 / (1.0 + np.exp(-raw))  # P(rr > b_j)
        surv = np.minimum.accumulate(surv, axis=1)  # enforce monotone non-increasing
        step = float(ord_bins[1] - ord_bins[0]) if len(ord_bins) > 1 else 1.0
        pred_rr = float(ord_bins[0]) + step * surv.sum(axis=1)  # expected value
        for i in range(len(thr_te)):
            score[i] = float(np.interp(thr_te[i], ord_bins, surv[i]))
        decision = (score >= 0.5).astype(int)
    elif head == "salience":
        sal = 1.0 / (1.0 + np.exp(-raw))
        norm = sal / (sal.sum(1, keepdims=True) + 1e-8)
        pred_rr = (norm * grid[None, :]).sum(1)
        for i in range(len(thr_te)):
            score[i] = float(norm[i, grid >= thr_te[i]].sum())
        decision = (score >= 0.5).astype(int)
    elif head in ("bce", "focal"):
        score = 1.0 / (1.0 + np.exp(-raw.squeeze(-1)))
        decision = (score >= 0.5).astype(int)
    else:
        raise ValueError(head)

    return {
        "clip": te["clip"], "subject": te["subject"], "fold": te["fold"],
        "gt_rr": te["gt"], "threshold": thr_te, "age": te["age"], "gt_tachy": te["gt_tachy"],
        "pred_rr": np.asarray(pred_rr, dtype=np.float64),
        "tachy_score": np.asarray(score, dtype=np.float64),
        "decision": np.asarray(decision, dtype=np.int64),
        "has_rr": has_rr, "best_val_loss": best_val,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def compute_metrics(rows: Dict[str, np.ndarray]) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_curve
    y = rows["gt_tachy"].astype(int)
    s = rows["tachy_score"].astype(float)
    dec = rows["decision"].astype(int)
    o: Dict[str, float] = {"n": int(len(y)), "pos": int(y.sum())}
    if rows["has_rr"]:
        e = rows["pred_rr"] - rows["gt_rr"]
        o["MAE"] = float(np.nanmean(np.abs(e)))
        o["RMSE"] = float(np.sqrt(np.nanmean(e ** 2)))
    else:
        o["MAE"] = float("nan")
        o["RMSE"] = float("nan")
    if len(np.unique(y)) == 2:
        o["AUROC"] = roc_auc(y, s)
        o["AUPRC"] = float(average_precision_score(y, s))
        fpr, tpr, _ = roc_curve(y, s)
        m = fpr <= 0.10
        o["Se@Sp90"] = float(np.max(tpr[m])) if m.any() else 0.0
        hs = tpr >= 0.90
        o["Sp@Se90"] = float(np.max(1.0 - fpr[hs])) if hs.any() else 0.0
    tp = int(((dec == 1) & (y == 1)).sum()); fn = int(((dec == 0) & (y == 1)).sum())
    tn = int(((dec == 0) & (y == 0)).sum()); fp = int(((dec == 1) & (y == 0)).sum())
    o["Sens"] = tp / (tp + fn) if (tp + fn) else float("nan")
    o["Spec"] = tn / (tn + fp) if (tn + fp) else float("nan")
    return o


def subject_bootstrap(rows: Dict[str, np.ndarray], B: int, seed: int) -> Dict[str, Tuple[float, float]]:
    rng = np.random.RandomState(seed)
    subs = np.unique(rows["subject"])
    idxby = {s: np.where(rows["subject"] == s)[0] for s in subs}
    acc: Dict[str, List[float]] = {}
    for _ in range(B):
        take = np.concatenate([idxby[s] for s in rng.choice(subs, len(subs), replace=True)])
        bd = {k: (v[take] if isinstance(v, np.ndarray) else v) for k, v in rows.items()}
        if np.unique(bd["gt_tachy"]).size < 2:
            continue
        for k, v in compute_metrics(bd).items():
            if k in ("n", "pos"):
                continue
            acc.setdefault(k, []).append(v)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in acc.items() if v}


def save_predictions(path: Path, rows: Dict[str, np.ndarray]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["clip", "subject", "fold", "gt_rr", "threshold", "age", "gt_tachy", "pred_rr", "tachy_score", "decision"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for i in range(len(rows["clip"])):
            w.writerow([rows["clip"][i], rows["subject"][i], int(rows["fold"][i]),
                        f"{rows['gt_rr'][i]:.3f}", f"{rows['threshold'][i]:.1f}", f"{rows['age'][i]:.1f}",
                        int(rows["gt_tachy"][i]),
                        ("" if np.isnan(rows["pred_rr"][i]) else f"{rows['pred_rr'][i]:.3f}"),
                        f"{rows['tachy_score'][i]:.5f}", int(rows["decision"][i])])


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal-dir", default="data/pedrr_signals_learned")
    ap.add_argument("--checkpoint",
                    default="outputs/paper_runs/pedrr_kfold_learned/fold{fold}/train/prism_breath_best.pt",
                    help="Frozen-encoder checkpoint. Use `{fold}` for the leakage-safe per-fold encoder "
                         "(default); a path without `{fold}` forces ONE encoder for all folds (prints a "
                         "representation-leakage warning).")
    ap.add_argument("--output-dir", default="outputs/objective_comparison")
    ap.add_argument("--heads", default=",".join(HEADS), help="comma-separated subset of: " + ",".join(HEADS))
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--head-patience", type=int, default=40, help="early-stop head on val loss (0=off)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--extract-batch-size", type=int, default=32)
    ap.add_argument("--rr-min", type=float, default=6.0)
    ap.add_argument("--rr-max", type=float, default=100.0, help="grid top; pedrr max GT is ~96.5")
    ap.add_argument("--rr-step", type=float, default=0.5)
    ap.add_argument("--salience-sigma", type=float, default=2.0)
    ap.add_argument("--ordinal-bin", type=float, default=2.0, help="ordinal cumulative-link bin width (bpm)")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh-features", action="store_true", help="re-extract frozen features (ignore cache)")
    ap.add_argument("--overwrite", action="store_true", help="recompute heads even if a .done marker exists")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    out = ROOT / args.output_dir
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "heads").mkdir(parents=True, exist_ok=True)

    grid = np.arange(args.rr_min, args.rr_max + 1e-6, args.rr_step).astype(np.float64)
    quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    ord_bins = np.arange(args.rr_min, args.rr_max + 1e-6, args.ordinal_bin).astype(np.float64)

    per_fold_ckpt = "{fold}" in args.checkpoint
    if not per_fold_ckpt:
        print("[WARN] single --checkpoint for all folds: the frozen representation saw test subjects "
              "on 4/5 folds (representation leakage). Use the default `{fold}` pattern for the clean result.")

    print(f"device={DEVICE}  heads={heads}  folds={folds}  signal_dir={args.signal_dir}")

    # ---- per-fold feature extraction ---------------------------------- #
    fold_feats: Dict[int, Dict[str, Dict]] = {}
    for k in folds:
        ckpt = Path(args.checkpoint.format(fold=k)) if per_fold_ckpt else Path(args.checkpoint)
        ckpt = ckpt if ckpt.is_absolute() else ROOT / ckpt
        if not ckpt.exists():
            print(f"[skip fold{k}] missing checkpoint {ckpt}")
            continue
        cache = out / "features" / f"fold{k}.npz"
        if cache.exists() and not args.refresh_features:
            print(f"[fold{k}] using cached features {cache}")
        else:
            print(f"[fold{k}] extracting frozen features from {ckpt}")
        model, seq_len = (None, None)
        if not cache.exists() or args.refresh_features:
            model, seq_len = load_frozen_model(ckpt)
        fold_feats[k] = get_fold_features(model, seq_len, args.signal_dir, k, out / "features",
                                          args.extract_batch_size, args.refresh_features)
        del model

    if not fold_feats:
        print("[abort] no folds had usable checkpoints/features.")
        return 1

    # ---- per-head training across folds -> pooled test predictions ---- #
    summary_rows = []
    pooled_scores: Dict[str, Dict[str, np.ndarray]] = {}
    for head in heads:
        done = out / "heads" / f"{head}.done"
        pred_csv = out / f"predictions_{head}.csv"
        if done.exists() and pred_csv.exists() and not args.overwrite:
            print(f"[head {head}] done marker present -> skipping (use --overwrite to recompute)")
            metrics = json.load(open(out / "heads" / f"{head}_metrics.json"))
            summary_rows.append(metrics)
            continue

        print(f"\n=== head: {head} ===")
        parts = {k: [] for k in ("clip", "subject", "fold", "gt_rr", "threshold", "age", "gt_tachy",
                                  "pred_rr", "tachy_score", "decision")}
        has_rr = True
        val_losses = []
        for k in folds:
            if k not in fold_feats:
                continue
            sp = fold_feats[k]
            res = run_head(head, sp["train"], sp["val"], sp["test"], args, grid, quantiles, ord_bins)
            has_rr = res["has_rr"]
            val_losses.append(res["best_val_loss"])
            for key in parts:
                parts[key].append(res[key])
            print(f"  fold{k}: n_test={len(res['clip'])}  best_val_loss={res['best_val_loss']:.4f}")

        rows = {key: np.concatenate(parts[key]) for key in parts}
        rows["has_rr"] = has_rr
        save_predictions(pred_csv, rows)
        pooled_scores[head] = rows

        pt = compute_metrics(rows)
        ci = subject_bootstrap(rows, args.bootstrap, args.seed)
        metrics = {"head": head, "has_rr": has_rr, "mean_val_loss": float(np.mean(val_losses))}
        metrics.update(pt)
        for k, (lo, hi) in ci.items():
            metrics[f"{k}_lo"], metrics[f"{k}_hi"] = lo, hi
        json.dump(metrics, open(out / "heads" / f"{head}_metrics.json", "w"), indent=2)
        done.write_text("ok", encoding="utf-8")
        summary_rows.append(metrics)
        print(f"  POOLED  MAE={pt.get('MAE', float('nan')):.3f}  AUROC={pt.get('AUROC', float('nan')):.3f}  "
              f"Sens={pt.get('Sens', float('nan')):.3f}  Spec={pt.get('Spec', float('nan')):.3f}")

    # ---- optional DeLong of each classification head vs the L1 regression head ---- #
    delong = {}
    if _HAVE_DELONG and "l1" in pooled_scores:
        base = pooled_scores["l1"]
        order = np.argsort(base["clip"])
        yb = base["gt_tachy"][order].astype(int)
        sb = base["tachy_score"][order]
        for head, rows in pooled_scores.items():
            if head == "l1":
                continue
            o2 = np.argsort(rows["clip"])
            if not np.array_equal(rows["clip"][o2], base["clip"][order]):
                continue
            aA, aB, p = delong_test(yb, sb, rows["tachy_score"][o2])
            delong[head] = {"auroc_l1": float(aA), "auroc_head": float(aB), "delong_p_vs_l1": float(p)}

    # ---- write summary table ------------------------------------------ #
    cols = ["MAE", "RMSE", "AUROC", "AUPRC", "Se@Sp90", "Sp@Se90", "Sens", "Spec"]
    summ_path = out / "objective_comparison_summary.csv"
    with open(summ_path, "w", encoding="utf-8") as fh:
        fh.write("head,has_rr," + ",".join(cols) + ",AUROC_95CI,delong_p_vs_l1\n")
        for m in summary_rows:
            h = m["head"]
            ci = f"[{m.get('AUROC_lo', float('nan')):.3f},{m.get('AUROC_hi', float('nan')):.3f}]"
            dp = delong.get(h, {}).get("delong_p_vs_l1", "")
            fh.write(f"{h},{m.get('has_rr')}," +
                     ",".join(f"{m.get(c, float('nan')):.4f}" for c in cols) +
                     f",{ci},{dp}\n")
    json.dump({"summary": summary_rows, "delong_vs_l1": delong}, open(out / "objective_comparison_summary.json", "w"), indent=2)

    print("\n=== OBJECTIVE COMPARISON (pooled test clips, subject-bootstrap CI on AUROC) ===")
    print("{:<14}{:>8}{:>8}{:>8}{:>8}{:>8}".format("head", "MAE", "AUROC", "AUPRC", "Sens", "Spec"))
    for m in summary_rows:
        print("{:<14}{:>8}{:>8.3f}{:>8.3f}{:>8.3f}{:>8.3f}".format(
            m["head"],
            ("  n/a" if np.isnan(m.get("MAE", float("nan"))) else f"{m['MAE']:.2f}"),
            m.get("AUROC", float("nan")), m.get("AUPRC", float("nan")),
            m.get("Sens", float("nan")), m.get("Spec", float("nan"))))
    print(f"\nwrote: {summ_path}")
    print(f"       {out / 'objective_comparison_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
