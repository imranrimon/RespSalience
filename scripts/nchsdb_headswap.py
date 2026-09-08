"""Faithful replication of the E2 head-swap on the NCHSDB neonatal cohort.

`scripts/nchsdb_framing.py` compared a learned regression head against a learned salience head and found
no classification advantage. That is one implementation; the paper's framing result was established
with a DIFFERENT one -- ROCKET random-kernel classification on the waveform, against the extractor's own
rate thresholded at the WHO cutoff (scripts/framing_headswap.py). Before concluding that the effect
fails to replicate, we repeat the original method here.

This version is also deterministic: ROCKET kernels are seeded and the classifier is logistic
regression, so there is none of the training-seed instability that made the learned comparison swing by
0.17 AUROC between runs at fixed nominal seed.

  regression framing      the belt's own FFT rate, ranked by margin against the WHO cutoff
  classification framing  ROCKET features on the same 60 s waveform -> P(tachypnea)

Same subjects, same windows, same subject-grouped folds. Only the read-out differs.

    python scripts/nchsdb_headswap.py
    python scripts/nchsdb_headswap.py --noise 0.5 1.0 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FS = 32.0
WHO_NEONATE = 60.0
dev = "cuda" if torch.cuda.is_available() else "cpu"


def rocket(X, seed=0):
    """Random convolutional kernels; PPV and max pooling. Mirrors scripts/framing_headswap.py."""
    torch.manual_seed(seed)
    xt = torch.tensor(X, device=dev, dtype=torch.float32).unsqueeze(1)
    feats = []
    for ks, dil in [(9, 1), (9, 2), (9, 4), (11, 1), (11, 4), (7, 2), (5, 1), (13, 2)]:
        w = torch.randn(96, 1, ks, device=dev); b = torch.randn(96, device=dev)
        c = torch.nn.functional.conv1d(xt, w, padding=(ks // 2) * dil, dilation=dil)
        feats.append((c > b[None, :, None]).float().mean(-1).cpu().numpy())
        feats.append(c.max(-1).values.cpu().numpy())
    return np.concatenate(feats, 1)


def cv_prob(F, y, grp, folds=5):
    uniq = sorted(set(grp)); rng = np.random.RandomState(0)
    up = [u for u in uniq if y[grp == u].sum() > 0]
    un = [u for u in uniq if y[grp == u].sum() == 0]
    rng.shuffle(up); rng.shuffle(un)
    fold = {}
    for i, u in enumerate(up): fold[u] = i % folds
    for i, u in enumerate(un): fold[u] = i % folds
    fk = np.array([fold[g] for g in grp]); prob = np.full(len(y), np.nan)
    for k in range(folds):
        tr, te = fk != k, fk == k
        if te.sum() == 0:
            continue
        if y[tr].sum() < 2 or (~y[tr].astype(bool)).sum() < 2:
            prob[te] = y[tr].mean(); continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.2))
        clf.fit(F[tr], y[tr]); prob[te] = clf.predict_proba(F[te])[:, 1]
    return prob


def degrade(X, noise=0.0, lowpass_hz=0.0, seed=0):
    rng = np.random.default_rng(seed)
    Y = X.copy()
    if lowpass_hz and lowpass_hz < FS / 2:
        b, a = ss.butter(3, lowpass_hz / (FS / 2), btype="low")
        Y = ss.filtfilt(b, a, Y, axis=1)
    if noise > 0:
        Y = Y + rng.normal(0.0, noise * Y.std(axis=1, keepdims=True), Y.shape)
    m = Y.mean(axis=1, keepdims=True); s = Y.std(axis=1, keepdims=True) + 1e-6
    return ((Y - m) / s).astype(np.float32)


def fft_rate(X, fs=FS, lo=0.1, hi=1.5):
    """Regression framing: the rate the signal itself yields, with no training."""
    out = np.empty(len(X))
    for i, x in enumerate(X):
        x = ss.detrend(np.asarray(x, float))
        try:
            b, a = ss.butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
            x = ss.filtfilt(b, a, x)
        except Exception:
            out[i] = np.nan; continue
        f, P = ss.welch(x, fs=fs, nperseg=min(len(x), int(fs * 60)))
        m = (f >= lo) & (f <= hi)
        out[i] = f[m][np.argmax(P[m])] * 60.0 if m.any() else np.nan
    return out


def subj(y, s, g, how="mean"):
    d = pd.DataFrame({"g": g, "y": y, "s": s}).dropna()
    a = d.groupby("g").agg(y=("y", "max"), s=("s", how))
    return (roc_auc_score(a.y, a.s) if a.y.nunique() > 1 else np.nan), a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--lowpass", type=float, nargs="*", default=[0.0])
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", default="outputs/nchsdb_headswap.csv")
    a = ap.parse_args()

    z = np.load(ROOT / "outputs" / "nchsdb_waveforms.npz", allow_pickle=True)
    X0 = z["X"]; rec = z["record"].astype(str); rr = z["gt_rr"].astype(float)
    y = (rr >= WHO_NEONATE).astype(int)
    print(f"windows {len(X0)} | infants {len(set(rec))} | tachypneic {int(y.sum())} "
          f"({y.mean()*100:.1f}%) across {len(set(rec[y==1]))} infants\n")

    rows = []
    print(f"{'noise':>6}{'lp':>5}{'reg AUROC':>11}{'cls AUROC':>11}{'delta':>9}{'95% CI':>20}"
          f"{'P(gain>0)':>11}")
    for lp in a.lowpass:
        for nz in a.noise:
            Xd = degrade(X0, noise=nz, lowpass_hz=lp)
            reg = fft_rate(Xd) - WHO_NEONATE           # margin against the WHO cutoff
            cls = cv_prob(rocket(Xd), y, rec)
            ar, agr = subj(y, reg, rec)
            ac, agc = subj(y, cls, rec)
            m = agr.join(agc, rsuffix="_c", how="inner")
            rng = np.random.default_rng(0); d = []
            for _ in range(a.boot):
                i = rng.integers(0, len(m), len(m))
                yy = m.y.values[i]
                if 0 < yy.sum() < len(yy):
                    d.append(roc_auc_score(yy, m.s_c.values[i]) - roc_auc_score(yy, m.s.values[i]))
            lo_, hi_ = np.percentile(d, [2.5, 97.5])
            pg = float(np.mean(np.array(d) > 0))
            rows.append({"noise": nz, "lowpass_hz": lp, "auroc_reg": ar, "auroc_cls": ac,
                         "delta": ac - ar, "ci_lo": lo_, "ci_hi": hi_, "p_gain_gt0": pg,
                         "n_subjects": len(m)})
            print(f"{nz:>6.1f}{lp:>5.0f}{ar:>11.3f}{ac:>11.3f}{ac-ar:>+9.3f}"
                  f"{f'[{lo_:+.3f},{hi_:+.3f}]':>20}{pg:>11.2f}")

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\n-> {a.out}")
    print("reg = the belt's own FFT rate thresholded at the WHO cutoff; cls = ROCKET on the same")
    print("waveform. Deterministic: no training-seed instability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
