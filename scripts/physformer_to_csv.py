"""B3 glue: turn PhysFormer per-subject RR predictions into the standard k-fold prediction CSVs so
full_metric_comparison.py can include PhysFormer as a modern (2022) baseline. PhysFormer is an
RGB-video transformer with its own trainer/loss and does NOT fit the frame-wise air-400 harness, so
you produce the RR externally (rPPG-Toolbox's PhysFormer on the pedrr videos, per docs/REPRODUCIBILITY.md
B3); this converts them to the exact schema the other baselines use (same margin->tachy_prob rule).

Input  --preds FILE : JSON {"ID-1234_100-5678": rr, ...} or {"ID-1234": rr, ...}, OR a CSV with
                       columns [subject_id(,video_id),pred_rr_bpm].
Output : outputs/paper_runs/pedrr_kfold_physformer/fold*/test_eval/prism_breath_predictions.csv
Then add  "PhysFormer": "outputs/paper_runs/pedrr_kfold_physformer"  to full_metric_comparison.RUNS and re-run.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KF = ROOT / "data" / "organized" / "kfold_learned"
OUT = ROOT / "outputs" / "paper_runs" / "pedrr_kfold_physformer"

def who_thr(age): return 60.0 if age < 2 else (50.0 if age < 12 else 40.0)

def load_preds(path):
    if path.endswith(".json"):
        return {str(k): float(v) for k, v in json.load(open(path)).items()}
    df = pd.read_csv(path)
    col = "pred_rr_bpm" if "pred_rr_bpm" in df else ("pred_bpm" if "pred_bpm" in df else df.columns[-1])
    d = {}
    for _, r in df.iterrows():
        d[str(r["subject_id"])] = float(r[col])
        if "video_id" in df: d[f"{r['subject_id']}_{r['video_id']}"] = float(r[col])
    return d

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="outputs/physformer_preds.json",
                    help="PhysFormer RR predictions (JSON dict or CSV)")
    a = ap.parse_args()
    if not os.path.exists(a.preds):
        print(f"[physformer_to_csv] predictions not found: {a.preds}\nProduce them first (external GPU):")
        print("  1) git clone https://github.com/ubicomp-lab/rPPG-Toolbox  (has PhysFormer + PhysFormerTrainer)")
        print("  2) preprocess the pedrr videos to rPPG-Toolbox RGB chunks; train PhysFormer per fold on")
        print("     the kfold_learned train subjects; infer RR (FFT of the rPPG output) for held-out test subjects")
        print("  3) save {subject_id_or_subject_video: pred_rr_bpm} -> outputs/physformer_preds.json")
        print("  Then: python scripts/physformer_to_csv.py")
        sys.exit()
    preds = load_preds(a.preds)
    total = 0
    for k in range(5):
        te = json.load(open(KF / f"fold{k}" / "test.json"))
        fd = OUT / f"fold{k}" / "test_eval"; fd.mkdir(parents=True, exist_ok=True)
        rows = []
        for r in te:
            key = f"{r['subject_id']}_{r['video_id']}"
            rr = float(preds.get(key, preds.get(str(r["subject_id"]), float("nan"))))
            gt = float(r["gt_rr_bpm"]); age = float(r.get("age_months", 6)); thr = who_thr(age); mg = rr - thr
            rows.append({"subject_id": str(r["subject_id"]), "video_id": r["video_id"], "age_months": age,
                         "gt_rr_bpm": gt, "pred_rr_bpm": rr, "gt_bpm": gt, "pred_bpm": rr,
                         "threshold_bpm": thr, "gt_tachy": int(gt >= thr),
                         "tachy_prob": float(1.0 / (1.0 + np.exp(-mg / 5.0)))})
        pd.DataFrame(rows).to_csv(fd / "prism_breath_predictions.csv", index=False)
        miss = sum(1 for x in rows if not np.isfinite(x["pred_rr_bpm"])); total += len(rows) - miss
        print(f"[fold{k}] wrote {len(rows)} rows ({miss} missing pred)", flush=True)
    print(f"DONE: {total} PhysFormer preds -> {OUT}\n"
          f'Add  "PhysFormer": "outputs/paper_runs/pedrr_kfold_physformer"  to full_metric_comparison.RUNS, then re-run it.', flush=True)
