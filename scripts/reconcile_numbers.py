"""Reconcile the number conventions across paper tables. Two subject sets exist: 332 (PRISM-family,
covers all pedrr) vs 298 (matched set where baselines have predictions). And two 'regression' scores:
PRISM's tachy_prob head vs the pure rate-margin (pred_rr - WHO threshold). This prints the canonical
numbers for each convention so every table can be made consistent + clearly labeled with N."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_metric_comparison import RUNS, load_run, point_metrics

if __name__=="__main__":
    data={m:load_run(r) for m,r in RUNS.items()}
    common=set.intersection(*[set(d["clip_key"]) for d in data.values()])
    P=data["PRISM-Breath"]
    Pfull=P; Pmatch=P[P["clip_key"].isin(common)]
    print(f"PRISM-Breath: full N={len(Pfull)} (332-basis) | matched N={len(Pmatch)} (298-basis, shared w/ baselines)\n")

    def show(tag, d):
        m=point_metrics(d)
        print(f"{tag:22s} AUROC_head={m.get('AUROC',float('nan')):.3f}  AUROC_margin={m.get('AUROC_margin',float('nan')):.3f}  "
              f"AUROC_rawrr={m.get('AUROC_rawrr',float('nan')):.3f}  AUPRC={m.get('AUPRC',float('nan')):.3f}  "
              f"Sp@Se90={m.get('Sp@Se90',float('nan')):.3f}  MAE={m['MAE']:.2f}")

    print("=== PRISM on each subject basis (pick ONE per table; label N) ===")
    show("PRISM (332 full)", Pfull)
    show("PRISM (298 matched)", Pmatch)
    print("\n  'regression->threshold' score = AUROC_margin (pure rate rule). 'tachy_prob head' = AUROC_head")
    print("  (learned classifier on fused features; not pure regression). 'signal classifier' = ROCKET, reported separately.\n")

    print("=== baselines (298 matched basis only; they lack 332 coverage) ===")
    for m in RUNS:
        if m=="PRISM-Breath": continue
        show(m+" (298)", data[m][data[m]["clip_key"].isin(common)])

    print("\n=== CONVENTION ===")
    print("  * Any table comparing PRISM vs baselines -> 298 matched (paired, fair).")
    print("  * PRISM-only tables (ablations, component isolation) -> 332 full, labeled.")
    print("  * 'Regression' baseline for the framing thesis = rate-margin (AUROC_margin), NOT the tachy_prob head.")
