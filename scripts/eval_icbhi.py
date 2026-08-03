"""
ICBHI Audio-Only Evaluation
Processes all WAV files in data/ICBHI_final_database/ using UniversalAdapter.
Ground truth RR is derived from the paired .txt annotation files.
Output: data/ablation_results/icbhi_audio_results.json
"""
import sys, os, glob, json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.universal_adapter import UniversalAdapter

wavs = glob.glob("data/ICBHI_final_database/*.wav")
if not wavs:
    print("No WAV files found in data/ICBHI_final_database/")
    sys.exit(0)

print(f"Processing {len(wavs)} ICBHI recordings...")
adapter  = UniversalAdapter()
results, errors = [], 0

for wav in wavs:
    try:
        bundle = adapter.load(wav)
        res    = adapter.estimate_rr(bundle)
        gt     = bundle.metadata.get("gt_rr_bpm")
        if gt:
            mae = abs(res["rr_bpm"] - gt)
            results.append({"file": os.path.basename(wav),
                             "pred": round(res["rr_bpm"], 1),
                             "gt":   round(gt, 1),
                             "mae":  round(mae, 2)})
    except Exception as e:
        errors += 1

if results:
    mae_mean = float(np.mean([r["mae"] for r in results]))
    mae_std  = float(np.std ([r["mae"] for r in results]))
    ioa2     = float(np.mean([r["mae"] <= 2.0 for r in results])) * 100
    ioa3     = float(np.mean([r["mae"] <= 3.0 for r in results])) * 100
    summary  = {"mae": round(mae_mean, 2), "mae_std": round(mae_std, 2),
                 "ioa_2bpm": round(ioa2, 1), "ioa_3bpm": round(ioa3, 1),
                 "n": len(results), "errors": errors}
    print(f"ICBHI Audio-Only | MAE={mae_mean:.2f} BPM  "
          f"IoA@2={ioa2:.1f}%  IoA@3={ioa3:.1f}%  N={len(results)}  Errors={errors}")
    os.makedirs("data/ablation_results", exist_ok=True)
    with open("data/ablation_results/icbhi_audio_results.json", "w") as f:
        json.dump({**summary, "per_file": results}, f, indent=2)
    print("Saved: data/ablation_results/icbhi_audio_results.json")
else:
    print("No annotated files found (check .txt annotation files exist alongside .wav files).")
