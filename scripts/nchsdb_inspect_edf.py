"""Inspect an NCH Sleep DataBank EDF and map its channels to the roles sec:extval needs.

The external-validation section compares a classification framing against a regression framing on
polysomnography, using a thoracic-belt channel as the INPUT and an airflow/capnography-derived rate as
the LABEL. Channel naming in PSG is not standardised, so before any of that can be written we need to
see what a real record actually contains.

Run this on the first downloaded record. It prints every channel with its sampling rate and dynamic
range, then proposes a role mapping and says plainly which roles it could not fill -- a missing airflow
or CO2 channel would mean the section's stated label source is not available in this release, which is
something to discover now rather than after a 24 GB download.

    python scripts/nchsdb_inspect_edf.py data/nchsdb/Sleep_Data/238_1279.edf
    python scripts/nchsdb_inspect_edf.py --any     # inspect whichever record has arrived
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Role -> ordered regex preferences. First match wins, so put the most specific patterns first.
ROLES = {
    "thorax (model input)": [r"\bthor", r"\bchest", r"rip.?thor", r"\btho\b", r"abdo?men|\babd\b"],
    "airflow (label)":      [r"nasal.?pressure", r"\bp?flow\b", r"airflow", r"\bcannula", r"\bptaf\b",
                             r"therm"],
    "capnography (label)":  [r"capno", r"etco2", r"co2", r"\bpco2\b"],
    "SpO2 (context)":       [r"spo2", r"sao2", r"\bsat\b"],
    "effort abdomen":       [r"abdo", r"\babd\b"],
}


def classify(labels):
    used, mapping = set(), {}
    for role, pats in ROLES.items():
        for p in pats:
            hit = next((i for i, l in enumerate(labels)
                        if i not in used and re.search(p, l, re.I)), None)
            if hit is not None:
                mapping[role] = (hit, labels[hit], p)
                used.add(hit)
                break
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="")
    ap.add_argument("--any", action="store_true", help="use any EDF found under data/nchsdb")
    ap.add_argument("--seconds", type=float, default=120.0, help="window to summarise")
    a = ap.parse_args()

    path = a.path
    if not path or a.any:
        hits = sorted(glob.glob(str(ROOT / "data" / "nchsdb" / "**" / "*.edf"), recursive=True))
        if not hits:
            raise SystemExit("no EDF found under data/nchsdb -- the download has not landed yet")
        path = hits[0]
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"not found: {p}")

    import pyedflib

    size_gb = p.stat().st_size / 1e9
    f = pyedflib.EdfReader(str(p))
    try:
        n = f.signals_in_file
        labels = [f.getLabel(i).strip() for i in range(n)]
        fss = [float(f.getSampleFrequency(i)) for i in range(n)]
        dur = f.getFileDuration()
        print(f"file      : {p.name}  ({size_gb:.2f} GB)")
        print(f"duration  : {dur/3600:.2f} h    channels: {n}\n")
        print(f"{'#':>3} {'label':<28}{'fs (Hz)':>9}{'units':>10}{'min':>10}{'max':>10}")
        for i in range(n):
            try:
                k = int(min(fss[i] * a.seconds, f.getNSamples()[i]))
                x = f.readSignal(i, 0, k) if k > 0 else np.array([])
                lo, hi = (float(np.nanmin(x)), float(np.nanmax(x))) if x.size else (np.nan, np.nan)
            except Exception:
                lo = hi = float("nan")
            print(f"{i:>3} {labels[i]:<28}{fss[i]:>9.1f}{f.getPhysicalDimension(i).strip():>10}"
                  f"{lo:>10.2f}{hi:>10.2f}")

        print("\nproposed role mapping")
        m = classify(labels)
        for role in ROLES:
            if role in m:
                i, lab, pat = m[role]
                print(f"  {role:<22} -> ch {i:>2}  '{lab}'   (matched /{pat}/, fs {fss[i]:.1f} Hz)")
            else:
                print(f"  {role:<22} -> NOT FOUND")
        missing = [r for r in ("thorax (model input)", "airflow (label)") if r not in m]
        if missing:
            print(f"\nBLOCKER: {', '.join(missing)} absent. sec:extval as written needs a thoracic "
                  "belt input and an airflow/capnography label; if this record is representative, the "
                  "section's stated design cannot be run on this release.")
        else:
            print("\nBoth required roles are present -- the extraction pipeline can be written against "
                  "this layout.")
            if "capnography (label)" not in m:
                print("Note: no CO2 channel. The paper describes the label as 'airflow+capnography'; "
                      "on this record only airflow is available, so that wording needs narrowing.")
    finally:
        f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
