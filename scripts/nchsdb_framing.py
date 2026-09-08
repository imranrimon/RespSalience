"""External replication: does the framing effect hold on an independent neonatal cohort?

Everything the paper claims about how the clinical decision should be formed rests on one video cohort,
one population and one reference standard. This tests the same claim on NCH Sleep DataBank neonates,
breaking four confounds at once: different continent, different reference (airflow transducers rather
than video review), different sensor (contact thoracic belt rather than a camera), and a cohort that
actually contains fast neonatal breathing.

The comparison is deliberately tight. One encoder architecture, one training budget, one set of
subject-disjoint folds, identical inputs. The ONLY difference is the head and the decision rule:

  regression framing      scalar rate, L1 loss           -> decide  pred_rr >= 60
  classification framing  distribution over rate bins,   -> decide  mass above 60 >= t
                          Gaussian-smoothed BCE target

A clean contact belt is an easier signal than a camera, so testing on it raw would answer a question
nobody asked. We therefore sweep a camera-realistic degradation grid -- additive noise, low-pass
band-limiting, and amplitude compression -- and report where the framing advantage appears and where it
vanishes. That boundary is the transferable result.

    python scripts/nchsdb_framing.py
    python scripts/nchsdb_framing.py --epochs 40 --grid full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal as ss
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FS = 32.0
BINS = np.arange(6.0, 90.0 + 1e-6, 0.5).astype(np.float32)
WHO_NEONATE = 60.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gauss_target(rr, s=2.0):
    return np.exp(-0.5 * ((BINS - rr) / s) ** 2).astype(np.float32)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(ci, co, k):
            return nn.Sequential(nn.Conv1d(ci, co, k, padding=k // 2), nn.BatchNorm1d(co),
                                 nn.ReLU(), nn.MaxPool1d(2))
        self.net = nn.Sequential(blk(1, 32, 65), blk(32, 64, 33), blk(64, 128, 17),
                                 nn.Conv1d(128, 128, 9, padding=4), nn.BatchNorm1d(128), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class RegHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(128, 1)

    def forward(self, z):
        return self.fc(self.pool(z).squeeze(-1)).squeeze(-1)


class ClsHead(nn.Module):
    def __init__(self, nb=len(BINS)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(128, nb)

    def forward(self, z):
        return torch.sigmoid(self.fc(self.pool(z).squeeze(-1)))


def degrade(X, noise=0.0, lowpass_hz=0.0, compress=1.0, rng=None):
    """Camera-realistic degradation of a clean contact-belt signal."""
    rng = rng or np.random.default_rng(0)
    Y = X.copy()
    if lowpass_hz and lowpass_hz < FS / 2:
        b, a = ss.butter(3, lowpass_hz / (FS / 2), btype="low")
        Y = ss.filtfilt(b, a, Y, axis=1)
    if compress != 1.0:                       # amplitude compression, as a camera loses dynamic range
        Y = np.sign(Y) * np.abs(Y) ** compress
    if noise > 0:
        Y = Y + rng.normal(0.0, noise * Y.std(axis=1, keepdims=True), Y.shape)
    m = Y.mean(axis=1, keepdims=True); s = Y.std(axis=1, keepdims=True) + 1e-6
    return ((Y - m) / s).astype(np.float32)


def train_eval(Xtr, ytr_rr, Xte, epochs, seed):
    """Train both framings on identical data and return their test scores."""
    torch.manual_seed(seed); np.random.seed(seed)
    out = {}
    xt = torch.tensor(Xtr, device=DEV).unsqueeze(1)
    xe = torch.tensor(Xte, device=DEV).unsqueeze(1)

    # -- regression framing -------------------------------------------------
    enc, head = Encoder().to(DEV), RegHead().to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), 1e-3, weight_decay=1e-4)
    yt = torch.tensor(ytr_rr, device=DEV, dtype=torch.float32)
    lf = nn.L1Loss(); idx = np.arange(len(xt))
    for _ in range(epochs):
        enc.train(); head.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 32):
            b = idx[i:i + 32]; opt.zero_grad()
            lf(head(enc(xt[b])), yt[b]).backward(); opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        out["reg_rr"] = head(enc(xe)).cpu().numpy()

    # -- classification framing --------------------------------------------
    enc, head = Encoder().to(DEV), ClsHead().to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), 1e-3, weight_decay=1e-4)
    T = torch.tensor(np.stack([gauss_target(r) for r in ytr_rr]), device=DEV)
    lf = nn.BCELoss()
    for _ in range(epochs):
        enc.train(); head.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 32):
            b = idx[i:i + 32]; opt.zero_grad()
            lf(head(enc(xt[b])), T[b]).backward(); opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        sal = head(enc(xe))
        sal = sal / (sal.sum(1, keepdim=True) + 1e-8)
        bt = torch.tensor(BINS, device=DEV)
        out["cls_mass"] = sal[:, bt >= WHO_NEONATE].sum(1).cpu().numpy()
        out["cls_rr"] = (sal * bt).sum(1).cpu().numpy()
    return out


def subject_auroc(y, s, groups):
    """Subject-level AUROC: aggregate windows to one score and one label per infant."""
    df = pd.DataFrame({"g": groups, "y": y, "s": s})
    a = df.groupby("g").agg(y=("y", "max"), s=("s", "mean"))
    return roc_auc_score(a.y, a.s) if a.y.nunique() > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5,
                    help="repeat the whole CV this many times; single runs are dominated by training "
                         "variance (observed sign flips at fixed nominal seed)")
    ap.add_argument("--grid", choices=["core", "full"], default="core")
    ap.add_argument("--out", default="outputs/nchsdb_framing.json")
    a = ap.parse_args()

    z = np.load(ROOT / "outputs" / "nchsdb_waveforms.npz", allow_pickle=True)
    X0, rec, rr = z["X"], z["record"].astype(str), z["gt_rr"].astype(float)
    y = (rr >= WHO_NEONATE).astype(int)
    print(f"windows {len(X0)} | infants {len(set(rec))} | tachypneic windows {int(y.sum())} "
          f"({y.mean()*100:.1f}%) across {len(set(rec[y == 1]))} infants")

    subs = np.array(sorted(set(rec)))
    rs = np.random.default_rng(a.seed); rs.shuffle(subs)
    fold_of = {s: i % a.folds for i, s in enumerate(subs)}
    folds = np.array([fold_of[s] for s in rec])

    grid = ([{"noise": n, "lowpass_hz": 0.0, "compress": 1.0} for n in (0.0, 0.5, 1.0, 2.0)]
            if a.grid == "core" else
            [{"noise": n, "lowpass_hz": lp, "compress": c}
             for n in (0.0, 0.5, 1.0, 2.0) for lp in (0.0, 3.0) for c in (1.0, 0.5)])

    rows = []
    for cfg in grid:
        Xd = degrade(X0, rng=np.random.default_rng(a.seed), **cfg)
        per_seed, keep_scores = [], None
        for s_i in range(a.seeds):
            reg_s = np.full(len(X0), np.nan); cls_s = np.full(len(X0), np.nan)
            for k in range(a.folds):
                te = folds == k; tr = ~te
                if y[tr].sum() == 0 or te.sum() == 0:
                    continue
                o = train_eval(Xd[tr], rr[tr], Xd[te], a.epochs, 1000 * s_i + a.seed + k)
                reg_s[te] = o["reg_rr"]      # regression framing ranks by the estimated rate
                cls_s[te] = o["cls_mass"]    # classification framing ranks by mass above the cutoff
            m = np.isfinite(reg_s) & np.isfinite(cls_s)
            per_seed.append((subject_auroc(y[m], reg_s[m], rec[m]),
                             subject_auroc(y[m], cls_s[m], rec[m])))
            if keep_scores is None:
                keep_scores = (reg_s.copy(), cls_s.copy())
            print(f"    seed {s_i}: reg {per_seed[-1][0]:.3f}  cls {per_seed[-1][1]:.3f}  "
                  f"delta {per_seed[-1][1]-per_seed[-1][0]:+.3f}", flush=True)
        arr = np.array(per_seed)
        deltas = arr[:, 1] - arr[:, 0]
        from scipy.stats import ttest_1samp
        tp = float(ttest_1samp(deltas, 0.0).pvalue) if len(deltas) > 1 else float("nan")
        ar, ac = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        reg_s, cls_s = keep_scores
        ok = np.isfinite(reg_s) & np.isfinite(cls_s)
        # subject bootstrap on the paired difference
        df = pd.DataFrame({"g": rec[ok], "y": y[ok], "r": reg_s[ok], "c": cls_s[ok]})
        agg = df.groupby("g").agg(y=("y", "max"), r=("r", "mean"), c=("c", "mean")).reset_index()
        rng = np.random.default_rng(0); d = []
        for _ in range(2000):
            i = rng.integers(0, len(agg), len(agg))
            if 0 < agg.y.values[i].sum() < len(i):
                d.append(roc_auc_score(agg.y.values[i], agg.c.values[i])
                         - roc_auc_score(agg.y.values[i], agg.r.values[i]))
            lo, hi = (np.percentile(d, [2.5, 97.5]) if d else (np.nan, np.nan))
        # Persist per-window scores so the aggregation choice can be audited without retraining.
        pd.DataFrame({"record": rec, "window_idx": np.arange(len(rec)), "gt_rr": rr, "y": y,
                      "reg_score": reg_s, "cls_score": cls_s,
                      "noise": cfg["noise"], "lowpass_hz": cfg["lowpass_hz"],
                      "compress": cfg["compress"]}).to_csv(
            ROOT / "outputs" / f"nchsdb_scores_n{cfg['noise']}_lp{cfg['lowpass_hz']}_c{cfg['compress']}.csv",
            index=False)
        rows.append({**cfg, "auroc_reg": ar, "auroc_cls": ac, "delta": ac - ar,
                     "delta_sd_across_seeds": float(deltas.std(ddof=1)) if len(deltas) > 1 else None,
                     "delta_min": float(deltas.min()), "delta_max": float(deltas.max()),
                     "t_test_p": tp, "n_seeds": int(len(deltas)),
                     "ci_lo": float(lo), "ci_hi": float(hi),
                     "p_gain_gt0": float(np.mean(np.array(d) > 0)) if d else np.nan,
                     "n_subjects": int(len(agg))})
        print(f"  noise {cfg['noise']:.1f} lp {cfg['lowpass_hz']:.0f} comp {cfg['compress']:.1f} | "
              f"reg {ar:.3f}  cls {ac:.3f}  delta {ac-ar:+.3f} "
              f"(SD {deltas.std(ddof=1):.3f}, range {deltas.min():+.3f}..{deltas.max():+.3f}, "
              f"t-test p={tp:.3f}) over {len(deltas)} seeds\n")

    Path(a.out).write_text(json.dumps(rows, indent=2))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
