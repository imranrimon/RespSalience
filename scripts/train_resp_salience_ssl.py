"""RespSalience + self-supervised pretraining (masked-signal reconstruction).

The one lever a random-kernel baseline (ROCKET) cannot use: pretrain the salience encoder on
unlabeled respiration signals (mask spans -> reconstruct), then fine-tune the salience head.
Flow channel; saves standard-schema predictions. Usage: --seed S [--no-ssl] [--tag ...]
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd
from scipy import signal as ss
import torch, torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]
KFOLD = ROOT / "outputs" / "paper_runs" / "pedrr_kfold_learned"   # overridden by --kfold
AGE_REF = ROOT / "outputs" / "paper_runs" / "pedrr_kfold_flowonly_agefix"  # carries corrected ages
FS, N = 25.0, 1440
BINS = np.arange(6.0, 90.0 + 1e-6, 0.5); NB = len(BINS)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def gauss(rr, s=2.0): return np.exp(-0.5*((BINS-rr)/s)**2).astype(np.float32)
def prep(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 64: x = np.pad(x, (0, 64-len(x)))
    x = ss.detrend(x); b, a = ss.butter(3, [0.1/(FS/2), 1.7/(FS/2)], btype="band"); x = ss.filtfilt(b, a, x)
    x = (x-x.mean())/(x.std()+1e-6)
    return (x[:N] if len(x) >= N else np.pad(x, (0, N-len(x)))).astype(np.float32)

class Encoder(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(ci, co, k): return nn.Sequential(nn.Conv1d(ci, co, k, padding=k//2), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(2))
        s.net = nn.Sequential(blk(1,32,65), blk(32,64,33), blk(64,128,17), nn.Conv1d(128,128,9,padding=4), nn.BatchNorm1d(128), nn.ReLU())
    def forward(s, x): return s.net(x)                 # (B,128,N/8)

class Decoder(nn.Module):
    def __init__(s):
        super().__init__()
        def up(ci, co): return nn.Sequential(nn.ConvTranspose1d(ci, co, 4, stride=2, padding=1), nn.BatchNorm1d(co), nn.ReLU())
        s.net = nn.Sequential(up(128,128), up(128,64), up(64,32), nn.Conv1d(32,1,9,padding=4))
    def forward(s, z): return s.net(z)

class SalHead(nn.Module):
    def __init__(s, nb=NB):
        super().__init__(); s.pool = nn.AdaptiveAvgPool1d(1); s.fc = nn.Linear(128, nb)
    def forward(s, z): return torch.sigmoid(s.fc(s.pool(z).squeeze(-1)))

def who_thr(a):
    return 60.0 if a < 2 else (50.0 if a < 12 else 40.0)


def corrected_ages():
    """subject_id -> age_months from the reference run that carries the age correction."""
    m = {}
    for k in range(5):
        p = AGE_REF / f"fold{k}" / "test_eval" / "prism_breath_predictions.csv"
        if p.exists():
            d = pd.read_csv(p)
            m.update(dict(zip(d["subject_id"].astype(str), d["age_months"].astype(float))))
    return m


def load(age_fix=True):
    rows = []
    for k in range(5):
        d = pd.read_csv(KFOLD / f"fold{k}" / "test_eval" / "prism_breath_predictions.csv")
        d["subject_id"] = d["subject_id"].astype(str); d["fold"] = k
        rows.append(d[["subject_id","video_id","signal_path","gt_rr_bpm","threshold_bpm","gt_tachy","age_months","fold"]])
    D = pd.concat(rows, ignore_index=True)
    if age_fix:
        am = corrected_ages()
        if am:
            n0 = int((D["gt_tachy"] != (D["gt_rr_bpm"] >= D["threshold_bpm"]).astype(int)).sum())
            D["age_months"] = D["subject_id"].map(am).fillna(D["age_months"])
            D["threshold_bpm"] = D["age_months"].map(who_thr)
            D["gt_tachy"] = (D["gt_rr_bpm"] >= D["threshold_bpm"]).astype(int)
            print(f"[age-fix] applied corrected ages to {len(am)} subjects; "
                  f"positives now {int(D['gt_tachy'].sum())}", flush=True)
    X, keep = [], []
    for i, r in D.iterrows():
        if os.path.exists(r["signal_path"]):
            z = np.load(r["signal_path"], allow_pickle=True)
            if "flow" in z.files: X.append(prep(z["flow"])); keep.append(i)
    return np.stack(X), D.loc[keep].reset_index(drop=True)

def pretrain(enc, dec, Xtr, epochs=40):
    opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), 1e-3); mse = nn.MSELoss()
    idx = np.arange(len(Xtr))
    for ep in range(epochs):
        enc.train(); dec.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), 32):
            b = Xtr[idx[i:i+32]]
            mask = (torch.rand_like(b) > 0.4).float()          # keep 60%
            z = enc((b*mask).unsqueeze(1)); rec = dec(z).squeeze(1)
            rec = rec[:, :b.shape[1]] if rec.shape[1] >= b.shape[1] else nn.functional.pad(rec, (0, b.shape[1]-rec.shape[1]))
            opt.zero_grad(); (mse(rec*(1-mask), b*(1-mask))).backward(); opt.step()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-ssl", action="store_true")
    ap.add_argument("--epochs", type=int, default=70); ap.add_argument("--tag", default="_ssl")
    ap.add_argument("--kfold", default="pedrr_kfold_learned",
                    help="run root supplying fold assignment and signal paths; use a 332-subject run "
                         "(e.g. pedrr_kfold_main332) to train on the full cohort")
    ap.add_argument("--no-age-fix", dest="age_fix", action="store_false", default=True)
    args = ap.parse_args()
    global KFOLD
    KFOLD = ROOT / "outputs" / "paper_runs" / args.kfold
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    X, D = load(args.age_fix)
    print(f"[ssl-salience] kfold={args.kfold} samples={len(X)} ssl={not args.no_ssl} "
          f"seed={args.seed}", flush=True)
    Tgt = np.stack([gauss(r) for r in D["gt_rr_bpm"]]); binsT = torch.tensor(BINS, device=DEV)
    pr = np.full(len(D), np.nan); pb = np.full(len(D), np.nan)
    for k in range(5):
        tr = (D["fold"] != k).values; te = (D["fold"] == k).values
        Xtr = torch.tensor(X[tr], device=DEV); Ytr = torch.tensor(Tgt[tr], device=DEV); Xte = torch.tensor(X[te], device=DEV)
        enc = Encoder().to(DEV); head = SalHead().to(DEV)
        if not args.no_ssl:
            pretrain(enc, Decoder().to(DEV), Xtr)                  # SSL on this fold's train signals only
        opt = torch.optim.Adam(list(enc.parameters())+list(head.parameters()), 1e-3, weight_decay=1e-4); lf = nn.BCELoss()
        idx = np.arange(len(Xtr))
        for ep in range(args.epochs):
            enc.train(); head.train(); np.random.shuffle(idx)
            for i in range(0, len(idx), 16):
                b = idx[i:i+16]; opt.zero_grad(); lf(head(enc(Xtr[b].unsqueeze(1))), Ytr[b]).backward(); opt.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            sal = head(enc(Xte.unsqueeze(1))); saln = sal/(sal.sum(1, keepdim=True)+1e-8)
            pr[te] = (saln*binsT).sum(1).cpu().numpy()
            thr = torch.tensor(D["threshold_bpm"].values[te], device=DEV)
            pb[te] = torch.stack([saln[j, binsT >= thr[j]].sum() for j in range(len(saln))]).cpu().numpy()
        print(f"  fold{k} done", flush=True)
    D["pred_rr_bpm"] = pr; D["pred_bpm"] = pr; D["gt_bpm"] = D["gt_rr_bpm"]; D["tachy_prob"] = pb
    for k in range(5):
        out = ROOT / "outputs" / "paper_runs" / f"pedrr_kfold_salience{args.tag}_seed{args.seed}" / f"fold{k}" / "test_eval"
        out.mkdir(parents=True, exist_ok=True); D[D["fold"] == k].to_csv(out / "prism_breath_predictions.csv", index=False)
    from sklearn.metrics import roc_auc_score
    g = D.groupby("subject_id").agg(t=("gt_tachy","max"), p=("tachy_prob","mean")).dropna()
    print(f"[ssl-salience] seed{args.seed} AUROC={roc_auc_score(g['t'], g['p']):.3f}", flush=True)

if __name__ == "__main__":
    main()
