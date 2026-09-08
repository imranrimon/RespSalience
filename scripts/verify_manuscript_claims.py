"""Re-derive every checkable quantitative claim in the manuscript from stored artifacts.

The paper has been rewritten repeatedly and several numbers were carried between drafts by hand. This
script is the guard against that: each claim below names the value printed in the manuscript and the
computation that should reproduce it. Anything that does not match is either a stale number or a wrong
one, and both must be fixed before submission.

It deliberately re-derives rather than re-reads: nothing here trusts a cached summary CSV where the
underlying predictions are available.

    python scripts/verify_manuscript_claims.py
    python scripts/verify_manuscript_claims.py --tol 0.002
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_main_run import load, scored  # noqa: E402
from compare_clinical_models import delong_test  # noqa: E402

MS = ROOT / "manuscript" / "prism_breath.tex"
SALIENCE = [f"pedrr_kfold_salience_ssl332_seed{i}" for i in range(3)]
PRISM = "pedrr_kfold_main332"


def band(a):
    return "<2mo" if a < 2 else ("2-12mo" if a < 12 else "12-60mo")


def ensemble(runs, age_map):
    ds = [scored(r, age_map) for r in runs]
    if any(d is None for d in ds):
        return None
    sid = sorted(ds[0].sid)
    base = ds[0].set_index("sid").loc[sid].copy()
    base["tachy_prob"] = np.mean([d.set_index("sid").loc[sid].tachy_prob.to_numpy(float) for d in ds], axis=0)
    base["pred_rr_bpm"] = np.mean([d.set_index("sid").loc[sid].pred_rr_bpm.to_numpy(float) for d in ds], axis=0)
    return base.reset_index()


class Check:
    def __init__(self, tol):
        self.tol = tol
        self.rows = []

    def add(self, name, claimed, measured, tol=None, note=""):
        t = self.tol if tol is None else tol
        if measured is None or (isinstance(measured, float) and not np.isfinite(measured)):
            ok = None
        else:
            ok = abs(float(claimed) - float(measured)) <= t
        self.rows.append((name, claimed, measured, ok, note))

    def report(self):
        w = max(len(r[0]) for r in self.rows) + 2
        print(f"{'claim':<{w}}{'in paper':>10}{'measured':>11}   verdict")
        print("-" * (w + 34))
        bad = 0
        for name, c, m, ok, note in self.rows:
            ms = "  n/a" if m is None else f"{float(m):.3f}"
            v = "MATCH" if ok else ("UNCHECKED" if ok is None else "MISMATCH")
            if ok is False:
                bad += 1
            print(f"{name:<{w}}{float(c):>10.3f}{ms:>11}   {v}{('  ' + note) if note else ''}")
        n_ok = sum(1 for r in self.rows if r[3] is True)
        n_un = sum(1 for r in self.rows if r[3] is None)
        print(f"\n{n_ok} match, {bad} MISMATCH, {n_un} unchecked, of {len(self.rows)} claims")
        return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.0015)
    a = ap.parse_args()
    ck = Check(a.tol)

    ref = load("pedrr_kfold_flowonly_agefix")
    if ref is None:
        raise SystemExit("age reference run missing")
    age_map = dict(zip(ref["subject_id"].astype(str), ref["age_months"].astype(float)))

    # ---------------- tab:headline -- recommended model vs regression baseline ----------------
    P = scored(PRISM, age_map); S = ensemble(SALIENCE, age_map)
    if P is not None and S is not None:
        P["band"] = P.age.map(band); S["band"] = S.age.map(band)
        sid = sorted(set(P.sid) & set(S.sid))
        p = P[P.sid.isin(sid)].sort_values("sid").reset_index(drop=True)
        s = S[S.sid.isin(sid)].sort_values("sid").reset_index(drop=True)
        y = p.y.to_numpy(int)
        ck.add("tab:headline PRISM overall", 0.827, roc_auc_score(y, p.tachy_prob))
        ck.add("tab:headline Salience overall", 0.936, roc_auc_score(y, s.tachy_prob))
        for b, cl_p, cl_s in (("<2mo", 0.650, 0.957), ("2-12mo", 0.864, 0.958), ("12-60mo", 0.915, 0.962)):
            m = (p.band == b).to_numpy()
            ck.add(f"tab:headline PRISM {b}", cl_p, roc_auc_score(y[m], p.tachy_prob.to_numpy()[m]))
            ck.add(f"tab:headline Salience {b}", cl_s, roc_auc_score(y[m], s.tachy_prob.to_numpy()[m]))
        ck.add("tab:headline PRISM MAE", 6.22, (p.pred_rr_bpm - p.gt_rr_bpm).abs().mean(), tol=0.02)
        # per-seed MAE mean, as the text reports it
        maes = [ (scored(r, age_map).pred_rr_bpm - scored(r, age_map).gt_rr_bpm).abs().mean() for r in SALIENCE ]
        ck.add("tab:headline Salience MAE", 5.60, float(np.mean(maes)), tol=0.02)
        aus = [roc_auc_score(scored(r, age_map).y, scored(r, age_map).tachy_prob) for r in SALIENCE]
        ck.add("Salience per-seed mean AUROC", 0.907, float(np.mean(aus)), tol=0.002)
        ck.add("Salience per-seed SD", 0.036, float(np.std(aus, ddof=1)), tol=0.002)
        dec_p = (p.pred_rr_bpm >= p.thr).astype(int)
        ck.add("tab:headline PRISM Sens", 0.55,
               ((dec_p == 1) & (p.y == 1)).sum() / max(1, (p.y == 1).sum()), tol=0.01)

    # The manuscript scores every head-to-head on ONE cohort: subjects present, with a finite score,
    # in PRISM and all five published extractors. Build it once and use it everywhere below.
    COMMON = None
    if P is not None:
        COMMON = set(P.sid)
        for _n, _t, _c in [("EfficientPhys", "pedrr_kfold_efficientphys", 0),
                           ("TS-CAN", "pedrr_kfold_tscan", 0), ("VIRENet", "pedrr_kfold_virenet", 0),
                           ("DeepPhys", "pedrr_kfold_deepphys_100ep", 0),
                           ("PhysFormer", "pedrr_kfold_physformer", 0)]:
            _d = scored(_t, age_map)
            if _d is not None:
                COMMON &= set(_d.loc[np.isfinite(_d.tachy_prob.to_numpy(float)), "sid"])
        print(f"[cohort] all-model common set: n={len(COMMON)}\n")

    def au_common(d):
        if d is None or COMMON is None:
            return None
        g = d[d.sid.isin(COMMON)].sort_values("sid")
        return roc_auc_score(g.y, g.tachy_prob)

    # ---------------- tab:sota -- five published extractors ----------------
    BASE = [("EfficientPhys", "pedrr_kfold_efficientphys", 0.755),
            ("TS-CAN", "pedrr_kfold_tscan", 0.726),
            ("VIRENet", "pedrr_kfold_virenet", 0.760),
            ("DeepPhys", "pedrr_kfold_deepphys_100ep", 0.777),
            ("PhysFormer", "pedrr_kfold_physformer", 0.734)]
    if P is not None:
        for name, tag, claimed in BASE:
            d = scored(tag, age_map)
            if d is None:
                ck.add(f"tab:sota {name}", claimed, None, note="run missing"); continue
            ck.add(f"tab:sota {name}", claimed, au_common(d))
        g = P[P.sid.isin(COMMON)].sort_values("sid")
        ck.add("tab:sota PRISM (matched 298)", 0.826, roc_auc_score(g.y, g.tachy_prob))
        ck.add("tab:sota PRISM MAE (matched)", 6.42, (g.pred_rr_bpm - g.gt_rr_bpm).abs().mean(), tol=0.02)

    # ---------------- tab:main -- recommended model vs prior work, one cohort ----------------
    if S is not None and COMMON is not None:
        sg = S[S.sid.isin(COMMON)].sort_values("sid").reset_index(drop=True)
        yy = sg.y.to_numpy(int)
        ck.add("tab:main RespSalience AUROC", 0.935, roc_auc_score(yy, sg.tachy_prob))
        ck.add("tab:main RespSalience MAE", 5.21,
               (sg.pred_rr_bpm - sg.gt_rr_bpm).abs().mean(), tol=0.02)
        for nm, tg, cl_mae in (("DeepPhys", "pedrr_kfold_deepphys_100ep", 5.62),
                               ("VIRENet", "pedrr_kfold_virenet", 5.91),
                               ("EfficientPhys", "pedrr_kfold_efficientphys", 6.10),
                               ("PhysFormer", "pedrr_kfold_physformer", 6.65),
                               ("TS-CAN", "pedrr_kfold_tscan", 6.15)):
            d = scored(tg, age_map)
            if d is None:
                continue
            g2 = d[d.sid.isin(COMMON)].sort_values("sid")
            ck.add(f"tab:main {nm} MAE", cl_mae,
                   (g2.pred_rr_bpm - g2.gt_rr_bpm).abs().mean(), tol=0.02)

    # ---------------- ablation of the recommended model (SSL on/off) ----------------
    NOSSL = [f"pedrr_kfold_salience_nossl332_seed{i}" for i in range(3)]
    N = ensemble(NOSSL, age_map)
    if S is not None and N is not None:
        sid2 = sorted(set(S.sid) & set(N.sid))
        s2 = S[S.sid.isin(sid2)].sort_values("sid"); n2 = N[N.sid.isin(sid2)].sort_values("sid")
        ck.add("ablation no-SSL ensemble", 0.918, roc_auc_score(s2.y, n2.tachy_prob))
        aus_n = [roc_auc_score(scored(r, age_map).y, scored(r, age_map).tachy_prob) for r in NOSSL]
        ck.add("ablation no-SSL per-seed mean", 0.898, float(np.mean(aus_n)), tol=0.002)
        maes_n = [(scored(r, age_map).pred_rr_bpm - scored(r, age_map).gt_rr_bpm).abs().mean() for r in NOSSL]
        ck.add("ablation no-SSL MAE", 5.48, float(np.mean(maes_n)), tol=0.02)

    # ---------------- shrinkage of the regressed rate at high true rates ----------------
    # The paper states this for the full cohort; on the 298-subject head-to-head subset it is -17.4.
    if P is not None:
        hi = P.gt_rr_bpm >= 60
        ck.add("regression bias at RR>=60 (n=332)", -16.9,
               float((P.pred_rr_bpm - P.gt_rr_bpm)[hi].mean()), tol=0.15)

    # ---------------- tab:selmatrix -- selection rule x model family ----------------
    SEL = [("EfficientPhys", "pedrr_kfold_efficientphys_aurocsel", 0.772),
           ("TS-CAN", "pedrr_kfold_tscan_aurocsel", 0.716),
           ("VIRENet", "pedrr_kfold_virenet_aurocsel", 0.755),
           ("DeepPhys", "pedrr_kfold_deepphys_100ep_aurocsel", 0.778)]
    for name, tag, claimed in SEL:
        ck.add(f"tab:selmatrix {name} sel", claimed, au_common(scored(tag, age_map)))
    Pa = scored("pedrr_kfold_main332auroc", age_map)
    if Pa is not None:
        ck.add("tab:selmatrix PRISM sel (full 332)", 0.850, roc_auc_score(Pa.y, Pa.tachy_prob))
        ck.add("tab:selmatrix PRISM sel (common)", 0.848, au_common(Pa))
        ck.add("tab:selmatrix PRISM sel MAE", 6.90,
               (Pa.pred_rr_bpm - Pa.gt_rr_bpm).abs().mean(), tol=0.02)
        gg = Pa[Pa.sid.isin(COMMON)]
        ck.add("matched head-to-head PRISM MAE", 7.11,
               (gg.pred_rr_bpm - gg.gt_rr_bpm).abs().mean(), tol=0.02)

    # ---------------- external replication ----------------
    hp = ROOT / "outputs" / "nchsdb_headswap.csv"
    if hp.exists():
        h = pd.read_csv(hp).sort_values("noise")
        for claimed, (_, r) in zip((0.014, 0.026, -0.056, -0.014), h.iterrows()):
            ck.add(f"sec:extval ROCKET delta @noise {r.noise:.1f}", claimed, r.delta, tol=0.002)
    wp = ROOT / "outputs" / "nchsdb_windows.csv"
    if wp.exists():
        w = pd.read_csv(wp)
        ck.add("sec:extval n windows", 1645, len(w), tol=0.5)
        ck.add("sec:extval n records", 70, w.record.nunique(), tol=0.5)
        ck.add("sec:extval n tachypneic", 234, int((w.gt_rr >= 60).sum()), tol=0.5)

    # ---------------- radar ----------------
    rp = ROOT / "outputs" / "radar_children.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        ck.add("tab:radar children MAE", 1.50, r["mae"], tol=0.01)
        ck.add("tab:radar children r", 0.97, r["pearson_r"], tol=0.005)

    # ---------------- objective head-swap (frozen encoder, L1 vs BCE) ----------------
    oc = ROOT / "outputs" / "objective_comparison" / "objective_comparison_summary.csv"
    if oc.exists():
        # This artifact is malformed: 12 header names for 13 columns, so every field is shifted one
        # place left of its label. Read positionally rather than by name -- the true AUROC is field 4
        # (0-based) after the head name. Verified against the raw file: l1 -> 0.8325, bce -> 0.8799.
        raw = [ln.split(",") for ln in oc.read_text().strip().splitlines()[1:] if ln.strip()]
        obj = {r[0]: float(r[4]) for r in raw if len(r) > 4}
        if "l1" in obj:
            ck.add("tab:objective L1 regression", 0.833, obj["l1"], tol=0.002)
        if "bce" in obj:
            ck.add("tab:objective BCE classification", 0.880, obj["bce"], tol=0.002)

    # ---------------- MAE vs AUROC rank correlation among the five extractors ----------------
    if P is not None and COMMON is not None:
        from scipy.stats import spearmanr
        mm, aa = [], []
        for _n, _t, _c in BASE:
            d = scored(_t, age_map)
            if d is None:
                continue
            g = d[d.sid.isin(COMMON)].sort_values("sid")
            mm.append(float((g.pred_rr_bpm - g.gt_rr_bpm).abs().mean()))
            aa.append(roc_auc_score(g.y, g.tachy_prob))
        if len(mm) == 5:
            rho, pv = spearmanr(mm, aa)
            ck.add("extractors MAE-vs-AUROC rho", -0.90, float(rho), tol=0.02)
            ck.add("extractors MAE-vs-AUROC p", 0.037, float(pv), tol=0.005)

    # ---------------- reference-standard ceilings (sensitivity analysis, seeded) ----------------
    if P is not None:
        rr = P["gt_rr_bpm"].to_numpy(float); thr = P["thr"].to_numpy(float)
        rng = np.random.default_rng(0)
        au_c, se_c = [], []
        for _ in range(400):
            true_rr = rr - rng.normal(0.0, 1.35, size=len(rr))
            y_true = (true_rr >= thr).astype(int)
            y_obs = ((true_rr + rng.normal(0.0, 1.35, size=len(rr))) >= thr).astype(int)
            if y_obs.sum() in (0, len(y_obs)):
                continue
            au_c.append(roc_auc_score(y_obs, true_rr - thr))
            tp = ((y_true == 1) & (y_obs == 1)).sum(); fn = ((y_true == 0) & (y_obs == 1)).sum()
            se_c.append(tp / max(1, tp + fn))
        ck.add("sec:ceiling AUROC ceiling", 0.998, float(np.mean(au_c)), tol=0.003)
        ck.add("sec:ceiling Sens ceiling", 0.953, float(np.mean(se_c)), tol=0.006)

    # ---------------- selective age-conditional read-out ----------------
    hr = ROOT / "outputs" / "hybrid_readout_summary.json"
    if hr.exists():
        h = json.loads(hr.read_text())
        ck.add("sec:age <2mo sens (selective)", 0.80, h.get("neonatal_sens_hybrid"), tol=0.01)

    bad = ck.report()
    if bad:
        print("\nEvery MISMATCH is a number in the manuscript that its own artifacts do not support.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
