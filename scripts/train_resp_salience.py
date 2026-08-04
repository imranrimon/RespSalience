"""RespSalience: respiratory rate as pitch-salience (CREPE/SPICE-inspired) for infant tachypnea.

Reframes RR estimation as classification of a SALIENCE distribution over a BPM grid, which
structurally removes the high-rate (neonatal) regression ceiling. RR = salience peak; tachypnea
probability = salience mass above the WHO age threshold (a naturally calibratable probability).

Trains per subject-disjoint fold (reusing the kfold_learned split + signal paths + labels), writes
standard-schema predictions so the existing clinical-endpoint / calibration tooling applies unchanged.

Usage: python scripts/train_resp_salience.py --seed 0 [--sincnet] [--channel flow]
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd
from scipy import signal as ss
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
KFOLD = ROOT / "outputs" / "paper_runs" / "pedrr_kfold_learned"
FS, N = 25.0, 1440
BINS = np.arange(6.0, 90.0 + 1e-6, 0.5); NB = len(BINS)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gauss_target(rr, sigma=2.0):
    return np.exp(-0.5 * ((BINS - rr) / sigma) ** 2).astype(np.float32)   # peak-1 (CREPE-style)


def prep(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 64: x = np.pad(x, (0, 64 - len(x)))
    x = ss.detrend(x)
    b, a = ss.butter(3, [0.1 / (FS / 2), 1.7 / (FS / 2)], btype="band"); x = ss.filtfilt(b, a, x)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return (x[:N] if len(x) >= N else np.pad(x, (0, N - len(x)))).astype(np.float32)


class SincConv(nn.Module):
    """Learnable band-pass filterbank front-end (SincNet, Ravanelli & Bengio 2018)."""
    def __init__(self, out_ch=32, kernel=129, fs=FS):
        super().__init__()
        self.kernel = kernel + (kernel % 2 == 0); self.fs = fs
        lo = np.linspace(0.1, 1.6, out_ch).astype(np.float32)
        self.f_lo = nn.Parameter(torch.tensor(lo)); self.bw = nn.Parameter(torch.full((out_ch,), 0.1))
        n = (self.kernel - 1) / 2.0
        self.register_buffer("t", torch.arange(-n, n + 1).float() / fs)
        self.register_buffer("win", torch.hamming_window(self.kernel))

    def forward(self, x):
        lo = torch.clamp(self.f_lo, 0.05, self.fs / 2 - 0.1)
        hi = torch.clamp(lo + torch.abs(self.bw), 0.06, self.fs / 2)
        filts = []
        for i in range(len(lo)):
            def sinc(fc):
                z = 2 * np.pi * fc * self.t
                return torch.where(self.t == 0, torch.tensor(2 * fc, device=x.device), torch.sin(z) / (self.t + 1e-9) / np.pi)
            band = (2 * hi[i] * sinc(hi[i]) - 2 * lo[i] * sinc(lo[i])) * self.win
            filts.append(band / (band.abs().max() + 1e-9))
        w = torch.stack(filts).unsqueeze(1)
        return torch.relu(torch.nn.functional.conv1d(x, w, padding=self.kernel // 2))


class SalienceNet(nn.Module):
    def __init__(self, nb=NB, sincnet=False):
        super().__init__()
        def blk(ci, co, k): return nn.Sequential(nn.Conv1d(ci, co, k, padding=k // 2), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(2))
        self.front = SincConv(32) if sincnet else None
        c0 = 32 if sincnet else 1
        self.f = nn.Sequential(blk(c0, 32, 65), blk(32, 64, 33), blk(64, 128, 17), blk(128, 128, 9), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(128, nb)

    def forward(self, x):
        x = x.unsqueeze(1)
        if self.front is not None: x = self.front(x)
        return torch.sigmoid(self.head(self.f(x).squeeze(-1)))


def load_folds(channel):
    rows = []
    for k in range(5):
        d = pd.read_csv(KFOLD / f"fold{k}" / "test_eval" / "prism_breath_predictions.csv")
        d["subject_id"] = d["subject_id"].astype(str); d["fold"] = k
        rows.append(d[["subject_id", "video_id", "signal_path", "gt_rr_bpm", "threshold_bpm", "gt_tachy", "age_months", "fold"]])
    D = pd.concat(rows, ignore_index=True)
    X, keep = [], []
    for i, r in D.iterrows():
        p = r["signal_path"]
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            if channel in z.files:
                X.append(prep(z[channel])); keep.append(i)
    D = D.loc[keep].reset_index(drop=True)
    return np.stack(X), D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sincnet", action="store_true")
    ap.add_argument("--channel", default="flow")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    X, D = load_folds(args.channel)
    print(f"[salience] samples={len(X)} subjects={D['subject_id'].nunique()} sincnet={args.sincnet} seed={args.seed}", flush=True)
    Tgt = np.stack([gauss_target(r) for r in D["gt_rr_bpm"]])
    binsT = torch.tensor(BINS, dtype=torch.float32, device=DEV)
    pred_rr = np.full(len(D), np.nan); pred_prob = np.full(len(D), np.nan)
    for k in range(5):
        tr = (D["fold"] != k).values; te = (D["fold"] == k).values
        Xtr = torch.tensor(X[tr], device=DEV); Ytr = torch.tensor(Tgt[tr], device=DEV); Xte = torch.tensor(X[te], device=DEV)
        net = SalienceNet(sincnet=args.sincnet).to(DEV)
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4); lf = nn.BCELoss()
        idx = np.arange(len(Xtr))
        for ep in range(args.epochs):
            net.train(); np.random.shuffle(idx)
            for i in range(0, len(idx), 16):
                b = idx[i:i + 16]; opt.zero_grad(); lf(net(Xtr[b]), Ytr[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            sal = net(Xte); saln = sal / (sal.sum(1, keepdim=True) + 1e-8)
            pred_rr[te] = (saln * binsT).sum(1).cpu().numpy()
            thr = torch.tensor(D["threshold_bpm"].values[te], device=DEV)
            pred_prob[te] = torch.stack([saln[j, binsT >= thr[j]].sum() for j in range(len(saln))]).cpu().numpy()
        print(f"  fold{k} done", flush=True)
    D["pred_rr_bpm"] = pred_rr; D["pred_bpm"] = pred_rr; D["gt_bpm"] = D["gt_rr_bpm"]; D["tachy_prob"] = pred_prob
    for k in range(5):
        out = ROOT / "outputs" / "paper_runs" / f"pedrr_kfold_salience{args.tag}_seed{args.seed}" / f"fold{k}" / "test_eval"
        out.mkdir(parents=True, exist_ok=True)
        D[D["fold"] == k].to_csv(out / "prism_breath_predictions.csv", index=False)
    from sklearn.metrics import roc_auc_score
    g = D.groupby("subject_id").agg(t=("gt_tachy", "max"), p=("tachy_prob", "mean")).dropna()
    print(f"[salience] seed{args.seed} subject AUROC={roc_auc_score(g['t'], g['p']):.3f}", flush=True)


if __name__ == "__main__":
    main()
