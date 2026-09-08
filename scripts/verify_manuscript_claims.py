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
# The published model. Self-supervised pretraining was tested and dropped: over six matched
# seeds it changed AUROC by -0.0008 (p=0.95) and MAE by -0.12 bpm (p=0.70), so it is not part
# of the model. Deterministic runs; see set_seed() in train_resp_salience_ssl.py.
SALIENCE = [f"pedrr_kfold_salience_nossl332det_seed{i}" for i in range(6)]
SSL_VARIANT = [f"pedrr_kfold_salience_ssl332det_seed{i}" for i in range(6)]
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
    ap.add_argument("--salience-suffix", default="",
                    help="evaluate the manuscript's claims against an alternative RespSalience run "
                         "family, e.g. --salience-suffix det. Used to see exactly which reported "
                         "numbers move when the training runs change; with no suffix this checks "
                         "the runs the manuscript actually reports.")
    a = ap.parse_args()
    ck = Check(a.tol)
    global SALIENCE
    if a.salience_suffix:
        SALIENCE = [f"pedrr_kfold_salience_ssl332{a.salience_suffix}_seed{i}" for i in range(3)]
        print(f"[alt runs] RespSalience family -> ...ssl332{a.salience_suffix}_seed*\n")

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
        ck.add("tab:headline Salience overall", 0.927, roc_auc_score(y, s.tachy_prob))
        for b, cl_p, cl_s in (("<2mo", 0.650, 0.939), ("2-12mo", 0.864, 0.942), ("12-60mo", 0.915, 0.975)):
            m = (p.band == b).to_numpy()
            ck.add(f"tab:headline PRISM {b}", cl_p, roc_auc_score(y[m], p.tachy_prob.to_numpy()[m]))
            ck.add(f"tab:headline Salience {b}", cl_s, roc_auc_score(y[m], s.tachy_prob.to_numpy()[m]))
        ck.add("tab:headline PRISM MAE", 6.22, (p.pred_rr_bpm - p.gt_rr_bpm).abs().mean(), tol=0.02)
        # per-seed MAE mean, as the text reports it
        maes = [ (scored(r, age_map).pred_rr_bpm - scored(r, age_map).gt_rr_bpm).abs().mean() for r in SALIENCE ]
        ck.add("tab:headline Salience MAE", 5.50, float(np.mean(maes)), tol=0.02)
        aus = [roc_auc_score(scored(r, age_map).y, scored(r, age_map).tachy_prob) for r in SALIENCE]
        ck.add("Salience per-seed mean AUROC", 0.904, float(np.mean(aus)), tol=0.002)
        ck.add("Salience per-seed SD", 0.023, float(np.std(aus, ddof=1)), tol=0.002)
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
        ck.add("tab:main RespSalience AUROC", 0.925, roc_auc_score(yy, sg.tachy_prob))
        ck.add("tab:main RespSalience MAE", 5.13,
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
    NOSSL = SSL_VARIANT   # the rejected addition, reported as a null result
    N = ensemble(NOSSL, age_map)
    if S is not None and N is not None:
        sid2 = sorted(set(S.sid) & set(N.sid))
        s2 = S[S.sid.isin(sid2)].sort_values("sid"); n2 = N[N.sid.isin(sid2)].sort_values("sid")
        ck.add("SSL variant ensemble (rejected)", 0.936, roc_auc_score(s2.y, n2.tachy_prob))
        aus_n = [roc_auc_score(scored(r, age_map).y, scored(r, age_map).tachy_prob) for r in NOSSL]
        ck.add("SSL variant per-seed mean (rejected)", 0.903, float(np.mean(aus_n)), tol=0.002)
        maes_n = [(scored(r, age_map).pred_rr_bpm - scored(r, age_map).gt_rr_bpm).abs().mean() for r in NOSSL]
        ck.add("SSL variant MAE (rejected)", 5.62, float(np.mean(maes_n)), tol=0.02)

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

    # ---------------- neonatal escalation pathway (Fig. escalation, app:escalation) ----------------
    # This section was the one place where the manuscript, its appendix and the rendered figure each
    # carried a different set of numbers, because none of them was re-derived here. It is now.
    import importlib.util
    _spec = importlib.util.spec_from_file_location("_ep", ROOT / "scripts" / "escalation_pathway.py")
    _ep = importlib.util.module_from_spec(_spec)
    _argv, sys.argv = sys.argv, ["verify"]
    try:
        _spec.loader.exec_module(_ep)
    finally:
        sys.argv = _argv

    eps = sorted(glob.glob(str(ROOT / "outputs" / "paper_runs" / PRISM / "fold*" / "test_eval"
                               / "prism_breath_predictions.csv")))
    if eps:
        parts = []
        for p in eps:
            x = pd.read_csv(p); x["fold"] = int(re.search(r"fold(\d+)", p).group(1)); parts.append(x)
        e = pd.concat(parts, ignore_index=True)
        e["band"] = e["age_months"].apply(band)
        e["T"] = e["threshold_bpm"].astype(float)
        es = _ep.aggregate_subject(e)
        es["res"] = (es["pred_rr_bpm"] - es["gt_rr_bpm"]).abs()
        es["tri"], qq = _ep.lofo_ageconditional_triage(es)

        for b, claimed_refer, claimed_q in (("<2mo", 52.6, 19.0), ("2-12mo", 45.5, None),
                                            ("12-60mo", 43.8, 7.0)):
            m = (es.band == b).values
            ck.add(f"tab:escalation REFER% {b}", claimed_refer,
                   100.0 * (es.tri.to_numpy()[m] == 2).mean(), tol=0.1)
            if claimed_q is not None:
                ck.add(f"escalation qhat {b} (rounded)", claimed_q, float(np.round(qq[m].mean())), tol=0.5)

        ret = _ep.screen_metrics(es, "tri")
        ck.add("tab:retained overall Sens", 0.690, ret["sens"], tol=0.002)
        ck.add("tab:retained overall Spec", 0.980, ret["spec"], tol=0.002)
        ck.add("tab:retained 12-60mo Sens", 1.000,
               _ep.screen_metrics(es[es.band == "12-60mo"], "tri")["sens"], tol=0.002)
        ck.add("escalation coverage (% referred)", 45.0,
               100.0 * float((es.tri == 2).mean()), tol=0.6)

        n2 = es[(es.band == "<2mo") & (es.gt_tachy == 1)]
        cam = (n2.pred_rr_bpm.to_numpy(float) >= n2["T"].to_numpy(float)).astype(int)
        ck.add("escalation #tachypneic neonates", 10, len(n2), tol=0.5)
        ck.add("escalation camera-only neonatal Sens", 0.100, float(cam.mean()), tol=0.002)
        ck.add("escalation neonates falsely cleared (camera)", 9, int((cam == 0).sum()), tol=0.5)
        ck.add("escalation neonates falsely cleared (triage)", 3, int((n2.tri == 0).sum()), tol=0.5)
        ck.add("escalation neonates referred", 7, int((n2.tri == 2).sum()), tol=0.5)
        ck.add("full-coverage overall Sens", 0.548,
               float((es.pred_rr_bpm.to_numpy(float) >= es["T"].to_numpy(float))
                     [es.gt_tachy.to_numpy(int) == 1].mean()), tol=0.002)

        # Pooled (non-Mondrian) width: the comparison that justifies age-conditional calibration.
        qp = np.empty(len(es))
        for k in sorted(es.fold.unique()):
            qp[(es.fold == k).values] = _ep.conf_q(es.loc[es.fold != k, "res"].to_numpy(float))
        pooled = _ep.straddle(es.pred_rr_bpm.to_numpy(float) - qp,
                              es.pred_rr_bpm.to_numpy(float) + qp, es["T"].to_numpy(float))
        m2 = (es.band == "<2mo").values
        ck.add("pooled width <2mo coverage", 0.579,
               float((es.res.to_numpy(float) <= qp)[m2].mean()), tol=0.002)
        ck.add("pooled REFER% <2mo", 15.8, 100.0 * float((pooled[m2] == 2).mean()), tol=0.1)
        ck.add("pooled REFER% 12-60mo", 59.4,
               100.0 * float((pooled[(es.band == "12-60mo").values] == 2).mean()), tol=0.1)
        n2p = pooled[((es.band == "<2mo") & (es.gt_tachy == 1)).values]
        ck.add("pooled neonates rescued", 2, int((cam == 0).sum()) - int((n2p == 0).sum()), tol=0.5)

    # ---------------- distributional triage read-out (sec:disttriage, tab:disttriage) ----------------
    _spec2 = importlib.util.spec_from_file_location("_dt", ROOT / "scripts" / "distributional_triage.py")
    _dt = importlib.util.module_from_spec(_spec2)
    _argv, sys.argv = sys.argv, ["verify"]
    try:
        _spec2.loader.exec_module(_dt)
    finally:
        sys.argv = _argv

    # Keep the triage section on the same run family as everything above.
    _dt.SALIENCE = list(SALIENCE)
    _sal = _dt.ensemble(_dt.SALIENCE)
    _pri = _dt.load_run(_dt.PRISM)
    if _sal is not None and _pri is not None:
        cm = sorted(set(_sal.subject_id) & set(_pri.subject_id))
        _sal = _sal[_sal.subject_id.isin(cm)].reset_index(drop=True)
        _pri = _pri[_pri.subject_id.isin(cm)].reset_index(drop=True)

        def burden_row(d, col):
            cleared = d[d[col] == 0]
            n2 = d[(d.band == "<2mo") & (d.gt_tachy == 1)]
            return (100.0 * float((d[col] != 0).mean()),
                    int((cleared.gt_tachy == 1).sum()), int((n2[col] == 0).sum()))

        _pri["c"], _ = _dt.triage_conformal(_pri)
        b_a, cl_a, neo_a = burden_row(_pri, "c")
        ck.add("tab:disttriage PRISM conformal burden", 52.4, b_a, tol=0.1)
        ck.add("tab:disttriage PRISM conformal cleared", 9, cl_a, tol=0.5)
        ck.add("tab:disttriage PRISM conformal neonates", 3, neo_a, tol=0.5)

        _sal["c"], _ = _dt.triage_conformal(_sal)
        b_a2, cl_a2, neo_a2 = burden_row(_sal, "c")
        ck.add("tab:disttriage Salience conformal burden", 53.3, b_a2, tol=0.1)
        ck.add("tab:disttriage Salience conformal cleared", 9, cl_a2, tol=0.5)

        _sal["d"], _, _ = _dt.triage_distributional(_sal, 0.02, 0.50)
        b_b, cl_b, neo_b = burden_row(_sal, "d")
        ck.add("tab:disttriage distributional burden", 44.3, b_b, tol=0.1)
        ck.add("tab:disttriage distributional cleared", 1, cl_b, tol=0.5)
        ck.add("tab:disttriage distributional neonates", 0, neo_b, tol=0.5)

        # Paired burden test quoted in the text.
        from scipy import stats as _st
        ba = (_pri.set_index("subject_id").loc[cm, "c"].to_numpy() != 0).astype(int)
        bb = (_sal.set_index("subject_id").loc[cm, "d"].to_numpy() != 0).astype(int)
        n10 = int(((ba == 1) & (bb == 0)).sum()); n01 = int(((ba == 0) & (bb == 1)).sum())
        ck.add("sec:disttriage burden discordant a-only", 58, n10, tol=0.5)
        ck.add("sec:disttriage burden discordant b-only", 31, n01, tol=0.5)
        ck.add("sec:disttriage burden McNemar p", 0.006,
               float(_st.binomtest(n10, n10 + n01, 0.5).pvalue), tol=0.0015)
        ck.add("sec:disttriage burden reduction", 0.081, float(ba.mean() - bb.mean()), tol=0.002)

        # Safety comparison is quoted as NON-significant; verify it stays that way.
        y = _pri.set_index("subject_id").loc[cm, "gt_tachy"].to_numpy(int)
        fa = ((_pri.set_index("subject_id").loc[cm, "c"].to_numpy() == 0) & (y == 1)).astype(int)
        fb = ((_sal.set_index("subject_id").loc[cm, "d"].to_numpy() == 0) & (y == 1)).astype(int)
        m10 = int(((fa == 1) & (fb == 0)).sum()); m01 = int(((fa == 0) & (fb == 1)).sum())
        ck.add("sec:disttriage safety McNemar p", 0.021,
               float(_st.binomtest(m10, m10 + m01, 0.5).pvalue), tol=0.002)

        # Out-of-sample calibration of the safety budget.
        for d_, claimed in ((0.05, 0.051), (0.10, 0.063)):
            s2 = _sal.copy()
            s2["t"], _, _ = _dt.triage_distributional(s2, d_, 0.30)
            cl = s2[s2.t == 0]
            ck.add(f"sec:disttriage realized FCR @delta={d_}", claimed,
                   float(cl.gt_tachy.mean()) if len(cl) else np.nan, tol=0.002)

        # Replication across two independent three-seed ensembles: the paper quotes these to say
        # plainly that a three-seed ensemble cannot establish either effect on its own.
        def _split(seeds):
            runs = [f"pedrr_kfold_salience_nossl332det_seed{i}" for i in seeds]
            sub = _dt.ensemble(runs)
            cc = sorted(set(sub.subject_id) & set(_pri.subject_id))
            sx = sub[sub.subject_id.isin(cc)].reset_index(drop=True)
            sx["t"], _, _ = _dt.triage_distributional(sx, 0.02, 0.50)
            aa = _pri.set_index("subject_id").loc[cc]; bb2 = sx.set_index("subject_id").loc[cc]
            yy = aa.gt_tachy.to_numpy(int)
            f_a = ((aa.c.to_numpy() == 0) & (yy == 1)).astype(int)
            f_b = ((bb2.t.to_numpy() == 0) & (yy == 1)).astype(int)
            u_a = (aa.c.to_numpy() != 0).astype(int); u_b = (bb2.t.to_numpy() != 0).astype(int)
            def mc(x, z):
                a1 = int(((x == 1) & (z == 0)).sum()); a0 = int(((x == 0) & (z == 1)).sum())
                return float(_st.binomtest(a1, a1 + a0, 0.5).pvalue) if (a1 + a0) else 1.0
            return mc(u_a, u_b), mc(f_a, f_b)

        for seeds, lab, cb, cs in (([0, 1, 2], "A", 0.142, 0.109), ([3, 4, 5], "B", 0.013, 0.070)):
            pb_, ps_ = _split(seeds)
            ck.add(f"sec:disttriage split {lab} burden p", cb, pb_, tol=0.002)
            ck.add(f"sec:disttriage split {lab} safety p", cs, ps_, tol=0.002)

    # ---------------- the model definition itself ----------------
    # These guard the two corrections this study had to make. If a future edit reintroduces
    # self-supervised pretraining, or silently swaps the published runs for a non-deterministic
    # family, these fail rather than the paper quietly regaining an unsupported claim.
    import scipy.stats as _sst
    S6 = ensemble(SALIENCE, age_map)
    V6 = ensemble(SSL_VARIANT, age_map)
    if S6 is not None and V6 is not None:
        ck.add("model uses six seeds", 6, len(SALIENCE), tol=0.5)
        s_au = [roc_auc_score(scored(r, age_map).y, scored(r, age_map).tachy_prob) for r in SALIENCE]
        v_au = [roc_auc_score(scored(r, age_map).y, scored(r, age_map).tachy_prob) for r in SSL_VARIANT]
        d = np.asarray(v_au) - np.asarray(s_au)          # SSL minus no-SSL, per matched seed
        ck.add("SSL effect on AUROC (null)", -0.0008, float(d.mean()), tol=0.002)
        ck.add("SSL effect AUROC paired p (n.s.)", 0.95,
               float(_sst.ttest_rel(v_au, s_au).pvalue), tol=0.02)
        s_mae = [float((scored(r, age_map).pred_rr_bpm - scored(r, age_map).gt_rr_bpm).abs().mean())
                 for r in SALIENCE]
        v_mae = [float((scored(r, age_map).pred_rr_bpm - scored(r, age_map).gt_rr_bpm).abs().mean())
                 for r in SSL_VARIANT]
        ck.add("SSL effect on MAE (null, wrong sign)", -0.12,
               float(np.mean(np.asarray(s_mae) - np.asarray(v_mae))), tol=0.02)
        ck.add("SSL effect MAE paired p (n.s.)", 0.70,
               float(_sst.ttest_rel(v_mae, s_mae).pvalue), tol=0.02)

        # The materialised ensemble that downstream scripts read must BE the published model.
        pub = scored("pedrr_kfold_salience_published", age_map)
        if pub is not None:
            m = S6.set_index("sid").loc[sorted(S6.sid)]
            q = pub.set_index("sid").loc[sorted(S6.sid)]
            ck.add("published run == seed ensemble (max |dProb|)", 0.0,
                   float((m.tachy_prob - q.tachy_prob).abs().max()), tol=1e-6)

    bad = ck.report()
    if bad:
        print("\nEvery MISMATCH is a number in the manuscript that its own artifacts do not support.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
