"""Improvement #1: spatial-coherence source separation for high-RR extraction.

Both prior extractor attempts (intensity + real depth) failed the SAME way: ROI aggregation locks
onto the strongest in-band oscillation, which is a ~10 BPM lighting/global-motion CONFOUND, not the
subtle fast breathing. That is a source-separation failure. Fix: decompose the pixel x time matrix
(PCA/ICA) and SELECT the component that is (a) in the respiratory band AND (b) spatially COHERENT/
localized (breathing = a contiguous chest/abdomen region moving together; the confound = global,
spatially diffuse loading). Then read RR (harmonic-product-spectrum) from that component.

Signal-level A/B on the 4 known high-RR false-negative clips + low-RR controls. CPU, no retrain.
Usage: python scripts/ab_spatial_decomp.py [--source intensity]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from scipy.signal import detrend, get_window
from sklearn.decomposition import PCA, FastICA
from src.core.upgraded_respiration_extractor import _decode_gray, _roi_slice, harmonic_rr, DEFAULT_LOW_HZ, DEFAULT_HIGH_HZ

FN = [("ID-1177", "100-4680", 63.5), ("ID-1330", "100-4927", 65.5),
      ("ID-1405", "100-5063", 61.0), ("ID-2205", "100-5318", 58.5)]
MANIFESTS = ["data/organized/pedrr_indist_train.json", "data/organized/pedrr_indist_val.json",
             "data/organized/pedrr_indist_test.json"]


def inband_frac(sig, fps, lo, hi):
    n = len(sig)
    S = np.abs(np.fft.rfft((sig - sig.mean()) * get_window("hann", n))) ** 2
    f = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (f >= lo) & (f <= hi)
    return float(S[band].sum() / (S[1:].sum() + 1e-12))


def spatial_concentration(loading):
    """Fraction of loading energy in the top 20% of pixels (localized breathing -> high; global -> low)."""
    e = np.abs(loading) ** 2
    e = np.sort(e)[::-1]
    k = max(1, int(0.20 * len(e)))
    return float(e[:k].sum() / (e.sum() + 1e-12))


def decomp_rr(stack, fps, lo, hi, method, k=8):
    T = stack.shape[0]
    P = detrend(stack.reshape(T, -1).astype(np.float64), axis=0)
    # band-limit each pixel so components live in the respiratory band
    F = np.fft.rfft(P, axis=0); f = np.fft.rfftfreq(T, d=1.0 / fps)
    F[~((f >= lo) & (f <= hi))] = 0
    Pb = np.fft.irfft(F, n=T, axis=0)
    # keep only pixels with some variance
    var = Pb.var(0); keep = var > np.percentile(var, 50)
    Pb = Pb[:, keep]
    k = min(k, Pb.shape[1], T)
    if method == "ica":
        dec = FastICA(n_components=k, max_iter=400, random_state=0, whiten="unit-variance")
    else:
        dec = PCA(n_components=k, random_state=0)
    try:
        scores = dec.fit_transform(Pb)          # T x k temporal components
        loads = dec.components_                  # k x N spatial loadings
    except Exception:
        return float("nan")
    best, best_score = None, -1
    for j in range(scores.shape[1]):
        s = scores[:, j]
        ib = inband_frac(s, fps, lo, hi)
        conc = spatial_concentration(loads[j])
        score = ib * conc                        # in-band AND spatially localized
        if score > best_score:
            best_score, best = score, s
    return harmonic_rr(best, fps, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="intensity")
    ap.add_argument("--n-controls", type=int, default=4)
    args = ap.parse_args()
    idx = {}
    for m in MANIFESTS:
        for r in json.load(open(m)):
            idx[(r["subject_id"], r["video_id"])] = r
    fn_keys = {(s, v) for s, v, _ in FN}
    controls = sorted(((r["gt_rr_bpm"], k, r) for k, r in idx.items() if k not in fn_keys))[:args.n_controls]
    rows = [(s, v, gt, "FN") for s, v, gt in FN] + [(k[0], k[1], gt, "ctrl") for gt, k, r in controls]
    lo, hi = DEFAULT_LOW_HZ, DEFAULT_HIGH_HZ
    print(f"[spatial-decomp] band={lo}-{hi}Hz\n{'clip':<20}{'grp':<5}{'gt':>6}{'PCA-RR':>10}{'ICA-RR':>10}")
    err = {"PCA": [], "ICA": []}
    for s, v, gt, grp in rows:
        rec = idx.get((s, v)); p = rec and rec.get("full_path")
        if not (p and Path(p).exists()):
            print(f"{s}_{v:<12} no video"); continue
        try:
            stack, fps = _decode_gray(p)
            ys, xs = _roi_slice(stack.shape[1], stack.shape[2], "abdomen")
            roi = stack[:, ys, xs]
            rp = decomp_rr(roi, fps, lo, hi, "pca"); ri = decomp_rr(roi, fps, lo, hi, "ica")
        except Exception as e:
            print(f"{s}_{v:<12} FAIL {type(e).__name__}: {e}"); continue
        err["PCA"].append(abs(rp - gt)); err["ICA"].append(abs(ri - gt))
        print(f"{s+'_'+v:<20}{grp:<5}{gt:>6.1f}{rp:>10.1f}{ri:>10.1f}")
    print(f"\n[MAE]  PCA={np.mean(err['PCA']):.2f}  ICA={np.mean(err['ICA']):.2f}  (proxy baseline was 24.3)")
    print("Read: does spatial-coherence decomposition pull the FN clips (gt 58-65) to ~gt where mean/")
    print("weighted aggregation collapsed to ~10 BPM? That would separate breathing from the confound.")


if __name__ == "__main__":
    raise SystemExit(main())
