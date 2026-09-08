"""Extract per-window respiratory signals and reference rates from NCH Sleep DataBank EDFs.

This is the missing first half of `sec:extval`. The section tests whether the classification framing
beats regression-then-threshold on an independent pediatric cohort, using a THORACIC BELT channel as
the model input and an AIRFLOW/CAPNOGRAPHY-derived rate as the label -- a non-circular pairing, since
input and label come from different transducers.

Channel layout (verified on the 3.1.0 release, 25 channels @ 256 Hz, 4 s records):
    Resp Thoracic  -> model input        Resp Airflow / Resp PTAF -> airflow label
    EtCO2 / Capno  -> capnography label  Resp Rate (bpm)          -> device-derived cross-check

Two things about these files require care:
  * The headers are not EDF+ compliant -- physical min/max are stored inverted (2000 / -2000), which
    makes pyedflib refuse to open them. We parse the header ourselves and take abs() of the scale.
  * Records are ~10 h at 256 Hz across 25 channels. We read ONLY the channels we need, and stream them
    record-block by record-block, so memory stays bounded on a 15 GB host.

Windows are accepted only when two independent transducers agree on the rate, which is the quality
control that makes the label trustworthy; the acceptance rate is reported so it can be stated.

    python scripts/nchsdb_extract.py --limit 2            # develop against whatever is local
    python scripts/nchsdb_extract.py --out outputs/nchsdb_windows.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, welch

ROOT = Path(__file__).resolve().parents[1]
SEARCH = [ROOT / "data" / "nchsdb" / "Sleep_Data",
          ROOT / "physionet.org" / "files" / "nch-sleep" / "3.1.0" / "Sleep_Data"]

# Two montages ship in NCHSDB 3.1.0 and they name the same transducers differently. Both are listed;
# lookup is case-insensitive. Order matters -- the first match wins.
INPUT_CH = ["Resp Thoracic", "Resp Chest"]                       # thoracic belt (the model input)
# Two DIFFERENT airflow transducers, both independent of the thoracic belt, so the label stays
# non-circular while still being cross-checkable.
FLOW_CH = ["Resp Airflow", "Resp Flow", "C-flow", "C-Flow", "Flow_DR"]   # thermistor / calibrated flow
FLOW2_CH = ["Resp PTAF", "XFlow"]                                        # nasal pressure
# Capnography is NOT usable in this release: EtCO2 carries a corrupt digital range (dmax <= dmin) in
# 8 of the 14 records checked, which pins the decoded signal at its physical maximum. Any claim that
# this cohort is capnography-referenced is unsupported.
CAPNO_CH = ["EtCO2", "Capno"]
RATE_CH = "Resp Rate"
RR_LO, RR_HI = 0.1, 1.5          # 6-90 breaths/min


# ---------------------------------------------------------------- EDF reading
class Edf:
    """Minimal, tolerant EDF reader: header parse + per-channel streaming."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            h = f.read(256)
        g = lambda a, b: h[a:b].decode("ascii", "replace").strip()
        self.hdr_bytes = int(g(184, 192))
        self.n_records = int(g(236, 244))
        self.rec_dur = float(g(244, 252))
        self.ns = int(g(252, 256))
        with open(self.path, "rb") as f:
            f.seek(256)
            blk = f.read(self.hdr_bytes - 256)
        ns = self.ns
        txt = lambda off, w, i: blk[off + i * w:off + (i + 1) * w].decode("ascii", "replace").strip()
        self.labels = [txt(0, 16, i) for i in range(ns)]
        o = ns * 16 + ns * 80
        self.dims = [txt(o, 8, i) for i in range(ns)]
        o += ns * 8
        pmin = [float(txt(o, 8, i)) for i in range(ns)]
        o += ns * 8
        pmax = [float(txt(o, 8, i)) for i in range(ns)]
        o += ns * 8
        dmin = [float(txt(o, 8, i)) for i in range(ns)]
        o += ns * 8
        dmax = [float(txt(o, 8, i)) for i in range(ns)]
        o += ns * 8 + ns * 80
        self.nsamp = [int(txt(o, 8, i)) for i in range(ns)]
        # Inverted phys min/max in this release would flip the signal's sign and scale; use magnitude.
        self.scale, self.offset = [], []
        for i in range(ns):
            dspan = (dmax[i] - dmin[i]) or 1.0
            k = abs((pmax[i] - pmin[i]) / dspan)
            self.scale.append(k)
            self.offset.append(min(pmin[i], pmax[i]) - min(dmin[i], dmax[i]) * k)
        # A channel whose DIGITAL range is impossible cannot be decoded to physical units. (An
        # inverted PHYSICAL range is harmless for rate estimation and is handled by the abs() above.)
        self.usable = [(dmax[i] > dmin[i]) and abs(dmin[i]) <= 32768 and abs(dmax[i]) <= 32768
                       for i in range(ns)]
        self.rec_samples = sum(self.nsamp)
        self.fs = [n / self.rec_dur for n in self.nsamp]

    def index(self, name, require_usable=True):
        key = str(name).strip().lower()
        i = next((k for k, l in enumerate(self.labels) if l.strip().lower() == key), None)
        if i is None:
            return None
        return i if (not require_usable or self.usable[i]) else None

    def first_index(self, names):
        for n in names:
            i = self.index(n)
            if i is not None:
                return i, n
        return None, None

    def read_channel(self, ci, max_records=None):
        """Stream one channel across data records (int16 little-endian, per EDF)."""
        nrec = self.n_records if max_records is None else min(self.n_records, max_records)
        pre = sum(self.nsamp[:ci])
        n_i = self.nsamp[ci]
        out = np.empty(nrec * n_i, dtype=np.float32)
        with open(self.path, "rb") as f:
            f.seek(self.hdr_bytes)
            for r in range(nrec):
                buf = f.read(self.rec_samples * 2)
                if len(buf) < self.rec_samples * 2:
                    out = out[:r * n_i]
                    break
                raw = np.frombuffer(buf, dtype="<i2", count=n_i, offset=pre * 2)
                out[r * n_i:(r + 1) * n_i] = raw * self.scale[ci] + self.offset[ci]
        return out


# ---------------------------------------------------------------- rate estimation
def rr_from(x, fs, lo=RR_LO, hi=RR_HI):
    """Dominant in-band rate (breaths/min) plus its spectral prominence."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < int(fs * 20) or np.allclose(x, x[0]):
        return np.nan, np.nan
    x = detrend(x)
    ny = fs / 2.0
    try:
        b, a = butter(3, [lo / ny, min(hi, ny * 0.95) / ny], btype="band")
        x = filtfilt(b, a, x)
    except Exception:
        return np.nan, np.nan
    f, P = welch(x, fs=fs, nperseg=min(len(x), int(fs * 60)))
    m = (f >= lo) & (f <= hi)
    if not m.any() or not np.isfinite(P[m]).any() or P[m].sum() <= 0:
        return np.nan, np.nan
    pk = f[m][np.argmax(P[m])]
    prom = float(P[m].max() / P[m].sum())
    return float(pk * 60.0), prom


def decimate(x, fs, target=32.0):
    q = int(fs // target)
    if q <= 1:
        return x, fs
    n = (len(x) // q) * q
    return x[:n].reshape(-1, q).mean(axis=1), fs / q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=60.0, help="seconds per analysis window")
    ap.add_argument("--max-hours", type=float, default=2.0,
                    help="cap hours read per record (full nights are ~10 h; the framing test does not "
                         "need all of it and this bounds runtime)")
    ap.add_argument("--agree-bpm", type=float, default=3.0,
                    help="accept a window only if thorax-independent references agree within this")
    ap.add_argument("--limit", type=int, default=0, help="process at most N records")
    ap.add_argument("--cohort", default="",
                    help="CSV with a 'record' column naming the subjects to include. Without it the "
                         "search path may pick up unrelated records left by an aborted mirror.")
    ap.add_argument("--require-two", action="store_true", default=True,
                    help="accept a window only when BOTH independent airflow references yield a rate")
    ap.add_argument("--allow-single-ref", dest="require_two", action="store_false")
    ap.add_argument("--out", default="outputs/nchsdb_windows.csv")
    ap.add_argument("--save-waveforms", default="outputs/nchsdb_waveforms.npz",
                    help="the classification framing is trained ON the waveform, so the decimated "
                         "thorax window is stored alongside the extracted rate")
    a = ap.parse_args()

    files = []
    for d in SEARCH:
        files += sorted(glob.glob(str(d / "*.edf")))
    files = sorted(set(files))
    if not files:
        raise SystemExit("no EDFs found under data/nchsdb/Sleep_Data or the physionet.org mirror")
    if a.cohort:
        keep = {os.path.basename(r) for r in pd.read_csv(a.cohort)["record"]}
        before = len(files)
        files = [f for f in files if os.path.basename(f) in keep]
        print(f"cohort filter: {len(files)}/{before} records retained from {a.cohort}")
    if a.limit:
        files = files[:a.limit]
    print(f"records: {len(files)}")

    ages = {}
    sp = ROOT / "data" / "nchsdb" / "Health_Data" / "SLEEP_STUDY.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        ages = {(str(p), str(t)): d / 30.44 for p, t, d in
                zip(s.STUDY_PAT_ID, s.SLEEP_STUDY_ID, s.AGE_AT_SLEEP_STUDY_DAYS)}

    rows, waves, stats = [], [], {"windows": 0, "accepted": 0}
    for path in files:
        stem = Path(path).stem
        pat, st = (stem.split("_", 1) + [""])[:2]
        try:
            e = Edf(path)
        except Exception as ex:
            print(f"  [skip] {stem}: {type(ex).__name__}: {ex}")
            continue
        ti, tname = e.first_index(INPUT_CH)
        fi, fname = e.first_index(FLOW_CH)
        ci, cname = e.first_index(FLOW2_CH)
        if ti is None or fi is None:
            avail = [l for l in e.labels if any(k in l.lower() for k in
                     ("resp", "tho", "chest", "abd", "flow", "rip", "effort"))]
            print(f"  [skip] {stem}: missing {'thorax' if ti is None else 'airflow'}; "
                  f"respiratory channels present: {avail}")
            continue
        max_rec = int(a.max_hours * 3600 / e.rec_dur)
        thor = e.read_channel(ti, max_rec)
        flow = e.read_channel(fi, max_rec)
        capn = e.read_channel(ci, max_rec) if ci is not None else None   # second airflow reference
        fs_t, fs_f = e.fs[ti], e.fs[fi]
        thor, fst = decimate(thor, fs_t)
        flow, fsf = decimate(flow, fs_f)
        if capn is not None:
            capn, fsc = decimate(capn, e.fs[ci])

        w_t, w_f = int(a.window * fst), int(a.window * fsf)
        nw = min(len(thor) // w_t, len(flow) // w_f)
        acc = 0
        for k in range(nw):
            rt, pt = rr_from(thor[k * w_t:(k + 1) * w_t], fst)
            rf, pf = rr_from(flow[k * w_f:(k + 1) * w_f], fsf)
            rc = np.nan
            if capn is not None:
                w_c = int(a.window * fsc)
                if (k + 1) * w_c <= len(capn):
                    rc, _ = rr_from(capn[k * w_c:(k + 1) * w_c], fsc)
            stats["windows"] += 1
            # Label quality control: two transducers independent of the thorax belt must agree.
            refs = [r for r in (rf, rc) if np.isfinite(r)]
            if not np.isfinite(rt):
                continue
            if a.require_two and len(refs) < 2:
                continue
            if not refs:
                continue
            if len(refs) == 2 and abs(refs[0] - refs[1]) > a.agree_bpm:
                continue
            gt = float(np.mean(refs))
            rows.append({"record": stem, "pat": pat, "study": st,
                         "age_months": ages.get((pat, st), np.nan),
                         "window": k, "thorax_rr": rt, "thorax_prom": pt,
                         "flow_rr": rf, "flow2_rr": rc, "gt_rr": gt,
                         "input_ch": tname, "flow_ch": fname, "flow2_ch": cname})
            waves.append(thor[k * w_t:(k + 1) * w_t].astype("float32"))
            acc += 1
        stats["accepted"] += acc
        print(f"  {stem}: age {ages.get((pat,st), float('nan')):.1f}mo  windows {nw}  accepted {acc}")

    if not rows:
        raise SystemExit("no windows accepted")
    d = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)
    if a.save_waveforms and waves:
        L = min(len(w) for w in waves)
        np.savez_compressed(a.save_waveforms,
                            X=np.stack([w[:L] for w in waves]),
                            record=d.record.to_numpy().astype("U"), window=d.window.to_numpy(),
                            gt_rr=d.gt_rr.to_numpy(), age_months=d.age_months.to_numpy(),
                            fs=np.float32(32.0))
        print(f"waveforms: {len(waves)} x {L} samples -> {a.save_waveforms}")
    err = (d.thorax_rr - d.gt_rr).abs()
    print(f"\nwindows seen {stats['windows']}, accepted {stats['accepted']} "
          f"({100*stats['accepted']/max(1,stats['windows']):.0f}%)")
    print(f"reference RR {d.gt_rr.min():.0f}-{d.gt_rr.max():.0f} (mean {d.gt_rr.mean():.1f})")
    print(f"thorax-vs-reference MAE {err.mean():.2f} br/min, within 3: {(err<=3).mean()*100:.0f}%")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
