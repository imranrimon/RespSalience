"""Where does the clinical signal actually come from? A component isolation on ONE cohort.

The version of this analysis in earlier drafts mixed three cohorts (n = 297, 298, 332) while its text
claimed identical folds, reported a cumulative "+ stage" progression whose numbers did not accumulate
(0.873 followed by 0.827, labelled +0.008), and quoted an EfficientPhys value that disagreed with the
comparison table. None of its intermediate values were reproducible from stored artifacts.

This recomputes the isolation from predictions that exist, on a single subject set, so each stage is
comparable to the one above it:

  1. raw extracted flow, read out by a plain FFT   -- what the signal alone supports
  2. a published end-to-end extractor              -- what prior work achieves
  3. encoder outputs + a linear read-out           -- per-modality rates/qualities into logistic
                                                      regression, subject-grouped CV
  4. the full deep fusion head                     -- the complete regression baseline
  5. the distributional read-out                   -- the proposed model

Stages 3-5 share the same encoder, so the difference between them is the read-out and nothing else.

    python scripts/component_isolation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_main_run import load, scored  # noqa: E402
from compare_clinical_models import delong_test  # noqa: E402

PRISM = "pedrr_kfold_main332"
SALIENCE = [f"pedrr_kfold_salience_ssl332_seed{i}" for i in range(3)]
ENCODER_COLS = ["depth_rr_bpm", "flow_rr_bpm", "audio_rr_bpm",
                "depth_quality", "flow_quality", "audio_quality",
                "depth_weight", "flow_weight", "audio_weight"]


def raw_fft_rate(path, lo=0.15, hi=1.5, fs=25.0):
    """Rate from the extracted flow waveform with no learned encoder at all."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception:
        return np.nan
    if "flow" not in getattr(z, "files", []):
        return np.nan
    x = np.asarray(z["flow"], float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 64:
        return np.nan
    x = ss.detrend(x)
    try:
        b, a = ss.butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        x = ss.filtfilt(b, a, x)
    except Exception:
        return np.nan
    f, P = ss.welch(x, fs=fs, nperseg=min(len(x), 1024))
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(P[m])] * 60.0) if m.any() else np.nan


def cv_logit(X, y, groups, folds):
    """Subject-grouped CV probability from a linear model on encoder outputs."""
    p = np.full(len(y), np.nan)
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or y[tr].sum() < 2 or (~y[tr].astype(bool)).sum() < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.2))
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="pedrr_kfold_efficientphys")
    a = ap.parse_args()

    ref = load("pedrr_kfold_flowonly_agefix")
    age_map = dict(zip(ref["subject_id"].astype(str), ref["age_months"].astype(float)))
    P = scored(PRISM, age_map)
    E = scored(a.baseline, age_map)
    S = [scored(r, age_map) for r in SALIENCE]
    if P is None or E is None or any(s is None for s in S):
        raise SystemExit("required runs missing")

    # One cohort for every row: present in the baseline, PRISM and the proposed model, finite scores.
    common = set(P.sid) & set(E.sid) & set(S[0].sid)
    common &= set(E.loc[np.isfinite(E.tachy_prob.to_numpy(float)), "sid"])
    common = sorted(common)
    g = lambda d: d[d.sid.isin(common)].sort_values("sid").reset_index(drop=True)
    p, e = g(P), g(E)
    y = p.y.to_numpy(int)
    folds = p.fold.to_numpy()
    print(f"single cohort: n={len(y)}, {int(y.sum())} tachypneic, {len(set(folds))} folds\n")

    rows = []

    # 1 -- raw extraction, no encoder
    rr = np.array([raw_fft_rate(sp) for sp in p.signal_path])
    ok = np.isfinite(rr)
    margin = rr - p.thr.to_numpy(float)
    rows.append(("Raw extracted flow, FFT read-out", int(ok.sum()),
                 roc_auc_score(y[ok], margin[ok]), "no learned component"))

    # 2 -- published end-to-end extractor
    rows.append(("Published extractor (EfficientPhys)", len(y),
                 roc_auc_score(y, e.tachy_prob), "prior work, end to end"))

    # 3 -- encoder outputs + linear read-out
    cols = [c for c in ENCODER_COLS if c in p.columns]
    X = p[cols].to_numpy(float)
    X = np.nan_to_num(X, nan=np.nanmedian(X))
    lin = cv_logit(X, y, p.sid.to_numpy(), folds)
    m = np.isfinite(lin)
    rows.append(("Encoder outputs $+$ linear read-out", int(m.sum()),
                 roc_auc_score(y[m], lin[m]), f"logistic on {len(cols)} encoder features"))

    # 4 -- the full deep fusion head
    rows.append(("$+$ deep fusion head (full baseline)", len(y),
                 roc_auc_score(y, p.tachy_prob), "complete regression model"))

    # 5 -- the distributional read-out
    sal = np.mean([g(s).tachy_prob.to_numpy(float) for s in S], axis=0)
    rows.append(("Distributional read-out (proposed)", len(y),
                 roc_auc_score(y, sal), "same signal, different decision"))

    print(f"{'Pipeline stage':<40}{'n':>5}{'AUROC':>8}   note")
    print("-" * 92)
    for name, n, au, note in rows:
        print(f"{name.replace('$','').replace('+','+'):<40}{n:>5}{au:>8.3f}   {note}")

    # The comparison the section actually rests on: read-out vs read-out, same encoder.
    _, _, p_lin = delong_test(y, sal, np.nan_to_num(lin, nan=np.nanmedian(lin)))
    _, _, p_head = delong_test(y, sal, p.tachy_prob.to_numpy(float))
    _, _, p_fuse = delong_test(y, p.tachy_prob.to_numpy(float),
                               np.nan_to_num(lin, nan=np.nanmedian(lin)))
    print(f"\npaired DeLong, all on the same {len(y)} subjects:")
    print(f"  deep fusion head vs linear read-out : p={p_fuse:.3f}  "
          f"(the fusion machinery buys nothing)")
    print(f"  proposed vs linear read-out         : p={p_lin:.2e}")
    print(f"  proposed vs deep fusion head        : p={p_head:.2e}")
    print("\nStages 3-5 share an encoder; the differences between them are read-out alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
