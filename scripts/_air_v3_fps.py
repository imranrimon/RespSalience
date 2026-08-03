"""Backfill the fps key into a copy of air_signals_v3 (from the matched air_signals_learned npz),
so the v3-trained model can be fps-normalized at load for the cross-dataset eval."""
import glob, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "data", "air_signals_v3")
LRN = os.path.join(REPO, "data", "air_signals_learned")
DST = os.path.join(REPO, "data", "air_signals_v3_fps")
os.makedirs(DST, exist_ok=True)
n = 0
for f in glob.glob(os.path.join(SRC, "*.npz")):
    name = os.path.basename(f)
    z = dict(np.load(f, allow_pickle=True))
    lp = os.path.join(LRN, name)
    if os.path.exists(lp):
        lz = np.load(lp, allow_pickle=True)
        if "fps" in lz.files:
            z["fps"] = lz["fps"]
    np.savez_compressed(os.path.join(DST, name), **z)
    n += 1
print(f"[v3-fps] {n} copies -> data/air_signals_v3_fps")
