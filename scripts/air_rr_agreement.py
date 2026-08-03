"""#23 cross-dataset RR agreement (AIR has no tachypnea -> RR-MAE/Bland-Altman only).

Reports per-dataset (AIR-125, AIR-400) RR-MAE, RMSE, Bland-Altman bias + 95% LoA, Pearson r,
with subject-level bootstrap 95% CI on MAE. Pass two prediction CSVs to contrast (e.g. with-fps
vs without-fps) so the fps-normalization effect is explicit.

Usage:
  python scripts/air_rr_agreement.py --pred with_fps.csv --label with-fps [--pred2 no_fps.csv --label2 no-fps]
"""
import argparse
import numpy as np
import pandas as pd


def _dscol(df):
    for c in ("source_dataset", "dataset"):
        if c in df.columns:
            return c
    df["dataset"] = "AIR"
    return "dataset"


def report(path, label):
    df = pd.read_csv(path)
    dsc = _dscol(df)
    scol = "subject_id" if "subject_id" in df.columns else dsc
    rng = np.random.default_rng(0)
    print(f"\n===== {label}  ({path}) =====")
    print(f"{'dataset':<10}{'n':>4}{'MAE [95% CI]':>22}{'RMSE':>8}{'BA bias':>9}{'95% LoA':>18}{'r':>7}")
    for ds, g in df.groupby(dsc):
        gt = g["gt_rr_bpm"].to_numpy(float); pr = g["pred_rr_bpm"].to_numpy(float)
        d = pr - gt
        mae = np.mean(np.abs(d)); rmse = np.sqrt(np.mean(d ** 2))
        bias = np.mean(d); lo, hi = bias - 1.96 * np.std(d), bias + 1.96 * np.std(d)
        r = float(np.corrcoef(gt, pr)[0, 1]) if np.std(pr) > 1e-6 and len(pr) > 2 else float("nan")
        subs = g[scol].to_numpy(); uniq = np.unique(subs)
        s2i = {u: np.where(subs == u)[0] for u in uniq}
        boot = []
        for _ in range(4000):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([s2i[u] for u in pick])
            boot.append(np.mean(np.abs(pr[idx] - gt[idx])))
        clo, chi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
        print(f"{str(ds):<10}{len(g):>4}{f'{mae:.2f} [{clo:.2f},{chi:.2f}]':>22}"
              f"{rmse:>8.2f}{bias:>+9.2f}{f'[{lo:+.1f},{hi:+.1f}]':>18}{r:>+7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--label", default="pred")
    ap.add_argument("--pred2", default=""); ap.add_argument("--label2", default="pred2")
    args = ap.parse_args()
    report(args.pred, args.label)
    if args.pred2:
        report(args.pred2, args.label2)
        print("\n(If with-fps MAE << without-fps, the fps-normalization fix is what enables cross-dataset transfer.)")


if __name__ == "__main__":
    raise SystemExit(main())
