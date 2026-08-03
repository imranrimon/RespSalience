"""#23 helper: build the combined AIR eval manifest + a no-fps copy of the learned signals.
Run AFTER build_air_learned_signals.py. Uses os.path.join to avoid path-escaping issues."""
import glob, json, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORG = os.path.join(REPO, "data", "organized")
SRC = os.path.join(REPO, "data", "air_signals_learned")
DST = os.path.join(REPO, "data", "air_signals_learned_nofps")

# 1) combined eval manifest (both AIR sets)
recs = json.load(open(os.path.join(ORG, "respiratory_benchmark_lodo_air-125_test.json"))) + \
       json.load(open(os.path.join(ORG, "respiratory_benchmark_lodo_air-400_test.json")))
json.dump(recs, open(os.path.join(ORG, "air_learned_eval.json"), "w"))
print(f"[prep] eval manifest: {len(recs)} records -> data/organized/air_learned_eval.json")

# 2) no-fps copy (strip the fps key so PRISM assumes 25 fps -> demonstrates the fps-fix contrast)
os.makedirs(DST, exist_ok=True)
fs = glob.glob(os.path.join(SRC, "*.npz"))
n = 0
for f in fs:
    z = np.load(f, allow_pickle=True)
    np.savez_compressed(os.path.join(DST, os.path.basename(f)), **{k: z[k] for k in z.files if k != "fps"})
    n += 1
print(f"[prep] no-fps copies: {n} -> data/air_signals_learned_nofps")
