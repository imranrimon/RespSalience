"""Regenerate supplementary Tables SI-SIII from a single, consistent label set.

The committed tables were assembled across two label sets: when the age
correction landed (9 neonates carried a stale ``age_months=6`` and so were
scored against the wrong WHO threshold), the label-dependent cells were updated
but the MAE and per-fold AUROC columns were not. The result is a table whose
``Mean+-SD`` row is the mean of the pre-fix per-fold AUROCs while its pooled
cell reports the post-fix value.

This script recomputes *every* cell from one designated artifact and one
designated label set, so the tables are reproducible from the repository.

Age labels: WHO thresholds are applied to the corrected ages. The correction is
read from a reference run rather than hard-coded, so it stays traceable.

Run from the repository root:

    python scripts/regenerate_supplementary_tables.py
    python scripts/regenerate_supplementary_tables.py --prism-run pedrr_kfold_lean
    python scripts/regenerate_supplementary_tables.py --no-age-fix   # reproduce the pre-fix numbers
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUN_DIR = Path("outputs/paper_runs")
# Run carrying the corrected ages for all 332 subjects.
AGE_REFERENCE_RUN = "pedrr_kfold_flowonly_agefix"
BASELINES = [
    ("EfficientPhys", "pedrr_kfold_efficientphys"),
    ("DeepPhys", "pedrr_kfold_deepphys"),
    ("TS-CAN", "pedrr_kfold_tscan"),
    ("VIRENet", "pedrr_kfold_virenet"),
]


def who_threshold(age_months: float) -> float:
    if age_months < 2:
        return 60.0
    if age_months < 12:
        return 50.0
    if age_months <= 60:
        return 40.0
    return 20.0


def load_run(run: str) -> Optional[pd.DataFrame]:
    frames = []
    for path in sorted(glob.glob(str(RUN_DIR / run / "fold*" / "test_eval" / "prism_breath_predictions.csv"))):
        df = pd.read_csv(path)
        df["fold"] = int(os.path.normpath(path).split(os.sep)[-3].replace("fold", ""))
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def corrected_age_map() -> Dict[str, float]:
    ref = load_run(AGE_REFERENCE_RUN)
    if ref is None:
        raise SystemExit(f"Age reference run {AGE_REFERENCE_RUN} not found; cannot apply the age correction.")
    return dict(zip(ref["subject_id"].astype(str), ref["age_months"].astype(float)))


def apply_labels(df: pd.DataFrame, age_fix: bool) -> pd.DataFrame:
    out = df.copy()
    if age_fix:
        fix = corrected_age_map()
        out["age_months"] = out["subject_id"].astype(str).map(fix).fillna(out["age_months"])
    out["threshold_bpm"] = out["age_months"].map(who_threshold)
    out["gt_tachy"] = (out["gt_rr_bpm"] >= out["threshold_bpm"]).astype(int)
    return out


def fold_metrics(g: pd.DataFrame) -> Dict[str, float]:
    """All cells from one label set. The decision rule is the WHO threshold on
    the predicted rate, as the supplementary text states."""
    decision = (g["pred_rr_bpm"] >= g["threshold_bpm"]).astype(int)
    y = g["gt_tachy"].to_numpy(dtype=int)
    tp = int(((decision == 1) & (y == 1)).sum())
    fn = int(((decision == 0) & (y == 1)).sum())
    tn = int(((decision == 0) & (y == 0)).sum())
    fp = int(((decision == 1) & (y == 0)).sum())
    return {
        "n": len(g),
        "pos": int(y.sum()),
        "mae": float((g["pred_rr_bpm"] - g["gt_rr_bpm"]).abs().mean()),
        "sens": tp / (tp + fn) if (tp + fn) else float("nan"),
        "spec": tn / (tn + fp) if (tn + fp) else float("nan"),
        "auroc": float(roc_auc_score(y, g["tachy_prob"])) if len(np.unique(y)) > 1 else float("nan"),
    }


def per_fold_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"fold": f, **fold_metrics(g)} for f, g in df.groupby("fold")]
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def latex_table_s1(per_fold: pd.DataFrame, pooled: Dict[str, float]) -> str:
    lines = [
        r"\begin{table}[!t]",
        r"\caption{PRISM-Breath per-fold clinical performance (subject-level).}",
        r"\label{tab:s1}\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccccc@{}}\toprule",
        r"Fold & $n$ & pos & RR-MAE & Sens & Spec & AUROC\\\midrule",
    ]
    for _, r in per_fold.iterrows():
        lines.append(f"Fold {int(r['fold'])} & {int(r['n'])} & {int(r['pos'])} & {r['mae']:.2f} & "
                     f"{r['sens']:.2f} & {r['spec']:.2f} & {r['auroc']:.3f} \\\\")
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Pooled}} & {int(pooled['n'])} & {int(pooled['pos'])} & {pooled['mae']:.2f} & "
                 f"{pooled['sens']:.2f} & {pooled['spec']:.2f} & {pooled['auroc']:.3f} \\\\")
    mae_sd = per_fold["mae"].std(ddof=1)
    auc_sd = per_fold["auroc"].std(ddof=1)
    lines.append(f"\\textbf{{Mean$\\pm$SD}} & --- & --- & ${per_fold['mae'].mean():.2f}{{\\pm}}{mae_sd:.2f}$ & "
                 f"--- & --- & ${per_fold['auroc'].mean():.3f}{{\\pm}}{auc_sd:.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_matrix_table(rows: Dict[str, List[float]], caption: str, label: str,
                       fmt: str, bold_first: bool) -> str:
    lines = [
        r"\begin{table}[!t]",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}\\centering\\small",
        r"\begin{tabular}{@{}lccccc@{}}\toprule",
        r"Method & Fold 0 & Fold 1 & Fold 2 & Fold 3 & Fold 4\\\midrule",
    ]
    for i, (name, vals) in enumerate(rows.items()):
        cells = " & ".join(f"\\textbf{{{v:{fmt}}}}" if (bold_first and i == 0) else f"{v:{fmt}}" for v in vals)
        label_txt = f"\\textbf{{{name}}}" if (bold_first and i == 0) else name
        lines.append(f"{label_txt} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate supplementary Tables SI-SIII from one label set.")
    ap.add_argument("--prism-run", default="pedrr_kfold_v3")
    ap.add_argument("--no-age-fix", action="store_true", help="Score against the stale ages (pre-fix numbers)")
    ap.add_argument("--out", default="manuscript/supplementary_tables_regenerated.tex")
    ap.add_argument("--compare-runs", default="",
                    help="Comma-separated extra runs to score under the same labels, to aid selection")
    args = ap.parse_args()
    age_fix = not args.no_age_fix

    prism = load_run(args.prism_run)
    if prism is None:
        raise SystemExit(f"Run not found: {args.prism_run}")
    prism = apply_labels(prism, age_fix)

    pf = per_fold_table(prism)
    pooled = fold_metrics(prism)
    print(f"PRISM run     : {args.prism_run}")
    print(f"Label set     : {'age-corrected' if age_fix else 'stale ages (pre-fix)'}")
    print(f"Subjects      : {len(prism)}   positives: {int(prism.gt_tachy.sum())}\n")
    print(pf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nPOOLED  n={pooled['n']} pos={pooled['pos']} MAE={pooled['mae']:.2f} "
          f"Sens={pooled['sens']:.2f} Spec={pooled['spec']:.2f} AUROC={pooled['auroc']:.3f}")
    print(f"Mean+-SD (ddof=1)  MAE {pf['mae'].mean():.2f}+-{pf['mae'].std(ddof=1):.2f}   "
          f"AUROC {pf['auroc'].mean():.3f}+-{pf['auroc'].std(ddof=1):.3f}")

    # Baselines under the identical label set.
    auroc_rows: Dict[str, List[float]] = {"PRISM-Breath": pf["auroc"].tolist()}
    mae_rows: Dict[str, List[float]] = {"PRISM-Breath": pf["mae"].tolist()}
    print("\nBaselines (scored under the same label set):")
    for name, run in BASELINES:
        b = load_run(run)
        if b is None:
            print(f"  {name:<15} MISSING")
            continue
        b = apply_labels(b, age_fix)
        bpf = per_fold_table(b)
        auroc_rows[name] = bpf["auroc"].tolist()
        mae_rows[name] = bpf["mae"].tolist()
        bp = fold_metrics(b)
        print(f"  {name:<15} n={bp['n']}  pos={bp['pos']}  MAE={bp['mae']:.2f}  AUROC={bp['auroc']:.3f}")

    if args.compare_runs:
        print("\nCandidate PRISM configurations under the identical label set:")
        print(f"  {'run':<34}{'n':>5}{'pos':>5}{'MAE':>8}{'AUROC':>8}")
        for run in [r.strip() for r in args.compare_runs.split(",") if r.strip()]:
            c = load_run(run)
            if c is None:
                print(f"  {run:<34}  MISSING")
                continue
            c = apply_labels(c, age_fix)
            m = fold_metrics(c)
            print(f"  {run:<34}{m['n']:>5}{m['pos']:>5}{m['mae']:>8.2f}{m['auroc']:>8.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks = [
        "% Regenerated by scripts/regenerate_supplementary_tables.py",
        f"% PRISM run: {args.prism_run}   labels: {'age-corrected' if age_fix else 'stale (pre-fix)'}",
        "% Every cell recomputed from a single label set.",
        "",
        latex_table_s1(pf, pooled),
        "",
        latex_matrix_table(auroc_rows,
                           f"Per-fold tachypnea AUROC, all methods ({len(prism)} subjects for PRISM; "
                           "baselines on their matched subset).",
                           "tab:s2", ".3f", True),
        "",
        latex_matrix_table(mae_rows,
                           "Per-fold RR mean-absolute-error (breaths/min), all methods.",
                           "tab:s3", ".2f", False),
    ]
    out.write_text("\n".join(blocks), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
