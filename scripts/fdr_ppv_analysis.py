"""Top-tier rigor add-ons (CPU, cached predictions):
  (1) Benjamini-Hochberg FDR correction across the DeLong test family (PRISM vs each baseline).
  (2) PPV / NPV / number-needed-to-screen (NNS) at a sensitivity-first operating point across a
      deployment-prevalence sweep -- a screen's real-world value depends on base rate, not the
      26% cohort prevalence.
Reuses full_metric_comparison.load_run + compare_clinical_models.delong_test on the matched 298 set.
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_metric_comparison import load_run, RUNS
from compare_clinical_models import delong_test
from sklearn.metrics import roc_curve

# use DeepPhys at its matched 100-epoch budget (its strongest), consistent with tab:sota
RUNS = dict(RUNS); RUNS["DeepPhys"] = "outputs/paper_runs/pedrr_kfold_deepphys_100ep"

# ---- common matched set (same as the comparison) ----
data = {m: load_run(r) for m, r in RUNS.items()}
common = set.intersection(*[set(d["clip_key"]) for d in data.values()])
data = {m: d[d["clip_key"].isin(common)].sort_values("clip_key").reset_index(drop=True) for m, d in data.items()}
P = data["PRISM-Breath"]
y = P["gt_tachy"].values.astype(int); pp = P["tachy_prob"].values
print(f"matched common set n={len(y)}  tachypnea prevalence={y.mean():.3f}\n")

# ---- (1) BH-FDR over the DeLong family ----
print("=== (1) DeLong p-values vs PRISM, Benjamini-Hochberg FDR-corrected ===")
tests = []
for m in RUNS:
    if m == "PRISM-Breath": continue
    _, aB, p = delong_test(y, pp, data[m]["tachy_prob"].values)
    tests.append((m, aB, p))
tests.sort(key=lambda t: t[2])                       # ascending p for BH
K = len(tests)
print(f"{'baseline':14s}{'AUROC':>8}{'raw p':>12}{'BH q':>12}  sig@FDR.05")
# BH: q_i = p_i * K / rank, enforce monotonicity from the largest rank down
qs = [None] * K
prev_q = 1.0
for i in range(K - 1, -1, -1):
    q = tests[i][2] * K / (i + 1)
    prev_q = min(prev_q, q)
    qs[i] = prev_q
for (m, aB, p), q in zip(tests, qs):
    print(f"{m:14s}{aB:>8.3f}{p:>12.2e}{q:>12.2e}  {'YES' if q < 0.05 else 'no'}")
print("  -> all five survive FDR correction (family-wise control) if every 'YES'.\n")

# ---- (2) PPV/NPV/NNS at a sensitivity-first operating point, prevalence sweep ----
def op_point(target_se):
    fpr, tpr, thr = roc_curve(y, pp)
    ok = np.where(tpr >= target_se)[0]
    i = ok[0]                                        # smallest threshold reaching target sensitivity
    return float(tpr[i]), float(1 - fpr[i]), float(thr[i])

for target in (0.90, 0.80):
    se, sp, t = op_point(target)
    print(f"=== (2) Operating point: sensitivity-first Se>={target:.2f}  ->  Se={se:.3f} Sp={sp:.3f} (thr={t:.3f}) ===")
    print(f"{'prevalence':>11}{'PPV':>8}{'NPV':>8}{'NNS':>8}   (NNS = screens per true case detected)")
    for prev in (0.02, 0.05, 0.10, 0.15, 0.20, round(float(y.mean()), 3)):
        ppv = se * prev / (se * prev + (1 - sp) * (1 - prev)) if (se*prev + (1-sp)*(1-prev)) > 0 else float("nan")
        npv = sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev) if (sp*(1-prev)+(1-se)*prev) > 0 else float("nan")
        nns = 1.0 / (prev * se) if prev * se > 0 else float("nan")   # people screened to detect one true case
        print(f"{prev:>11.3f}{ppv:>8.3f}{npv:>8.3f}{nns:>8.1f}")
    print()
