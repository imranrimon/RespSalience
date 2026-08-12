"""(#16) Conformal subgroup coverage for PRISM on the full pedrr-332 (corrected join).

The cached prism_breath_predictions.csv carry conformal RR-interval columns interval_lower_bpm /
interval_upper_bpm (plus qhat). Per subgroup (overall, age band, sex, site) reports n, empirical
coverage = P(interval_lower <= gt_rr <= interval_upper), and mean interval width (bpm). If those
columns are absent it reports which columns exist instead. CPU-only, read-only cached predictions.

Run:  python scripts/conformal_subgroup.py
"""
import numpy as np
import _corrected_common as C

RUN = "outputs/paper_runs/pedrr_kfold_learned"
SMALL_N = 30


def cover_row(d):
    inside = ((d["gt_rr"].values >= d["ilo"].values) & (d["gt_rr"].values <= d["ihi"].values))
    width = (d["ihi"].values - d["ilo"].values)
    return len(d), inside.mean(), width.mean()


if __name__ == "__main__":
    d = C.load_corrected(RUN)
    if "ilo" not in d.columns or "ihi" not in d.columns:
        import pandas as pd, glob, os
        cols = list(pd.read_csv(glob.glob(os.path.join(RUN, "fold*", "test_eval", "prism_breath_predictions.csv"))[0]).columns)
        print("Conformal interval columns NOT present in cached CSVs. Columns available:")
        print(cols)
        raise SystemExit(0)

    tgt = 0.80  # nominal conformal level used in the study (pooled alpha=0.20)
    print(f"PRISM full n={len(d)}  conformal cols: interval_lower_bpm, interval_upper_bpm, qhat  (nominal ~{tgt:.2f})\n")

    def line(label, sub):
        n, cov, w = cover_row(sub)
        flag = "  small n" if n < SMALL_N else ""
        print(f"{label:>24}{n:>5}{cov:>10.3f}{w:>12.2f}{flag}")

    print(f"{'group':>24}{'n':>5}{'coverage':>10}{'mean_width':>12}")
    line("OVERALL", d)
    print("  -- age band --")
    for b in ["<2mo", "2-12mo", "12-60mo"]:
        line(b, d[d.band == b])
    print("  -- sex --")
    for s in [1, 2]:
        line(C.SEX_LABEL[s], d[d.sex == s])
    print("  -- site --")
    for fc in [1, 2, 3]:
        line(C.FAC_LABEL[fc], d[d.fac == fc])
