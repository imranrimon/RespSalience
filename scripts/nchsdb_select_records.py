"""Select the NCH Sleep DataBank records for the external-validation cohort, and emit the download plan.

Why this exists
---------------
`sec:extval` reports a cross-cohort framing test on NCHSDB (n=70). Neither the analysis code nor the
result artifact survives in this repository, and the polysomnography signals are gone: `data/nchsdb/`
holds only the 25 MB `Health_Data` metadata and a `RECORDS` index naming 3984 EDF files under a
`Sleep_Data/` directory that does not exist locally. The section therefore cannot be regenerated here.

This script does the half that is reproducible from what we have: it selects the target records
deterministically from the metadata and writes the download list, so that recovering the section is one
credentialed download plus one analysis pass rather than a reconstruction from memory.

Selection is age-driven, because the framing effect the section tests has an age gradient: the cohort is
drawn from the youngest available studies, where respiratory rates are highest.

    python scripts/nchsdb_select_records.py --n 70 --out outputs/nchsdb_cohort.csv

The download itself needs PhysioNet credentialed access and must be run by the data owner, e.g.
    wget -r -N -c -np --user <USER> --ask-password \\
         -P data/nchsdb/ https://physionet.org/files/nch-sleep/1.0.0/Sleep_Data/<file>
Run it with `--print-commands` to get one line per selected record.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NCH = ROOT / "data" / "nchsdb"
BASE_URL = "https://physionet.org/files/nch-sleep/1.0.0/"
DAYS_PER_MONTH = 30.44


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=70, help="cohort size to select")
    ap.add_argument("--max-age-months", type=float, default=12.0,
                    help="restrict to studies below this age; the framing gain is largest where "
                         "respiratory rates are highest")
    ap.add_argument("--out", default="outputs/nchsdb_cohort.csv")
    ap.add_argument("--print-commands", action="store_true")
    a = ap.parse_args()

    records_path = NCH / "RECORDS"
    study_path = NCH / "Health_Data" / "SLEEP_STUDY.csv"
    if not records_path.exists() or not study_path.exists():
        raise SystemExit(f"NCHSDB metadata not found under {NCH}")

    records = [ln.strip() for ln in records_path.read_text().splitlines() if ln.strip()]
    # RECORDS entries are "Sleep_Data/<patient>_<study>.edf"
    idx = {}
    for r in records:
        stem = Path(r).stem
        if "_" in stem:
            pat, study = stem.split("_", 1)
            idx[(pat, study)] = r

    s = pd.read_csv(study_path)
    s["age_months"] = s["AGE_AT_SLEEP_STUDY_DAYS"] / DAYS_PER_MONTH
    s["record"] = [idx.get((str(p), str(t))) for p, t in
                   zip(s["STUDY_PAT_ID"], s["SLEEP_STUDY_ID"])]
    have = s[s["record"].notna()].copy()

    present = sum((NCH / r).exists() for r in have["record"])
    print(f"records indexed: {len(records)}   matched to metadata: {len(have)}   "
          f"present on disk: {present}")
    if present == 0:
        print("=> No signal files are present locally. sec:extval cannot be regenerated until they are "
              "downloaded (PhysioNet credentialed access).")

    cohort = (have[have["age_months"] < a.max_age_months]
              .sort_values(["age_months", "record"])          # deterministic
              .head(a.n))
    if len(cohort) < a.n:
        print(f"[warn] only {len(cohort)} studies under {a.max_age_months} months; "
              "raise --max-age-months for a larger cohort")

    band = pd.cut(cohort["age_months"], [-0.1, 2, 12, 60], labels=["<2mo", "2-12mo", "12-60mo"])
    print(f"\nselected {len(cohort)} studies, age {cohort.age_months.min():.1f}"
          f"-{cohort.age_months.max():.1f} months")
    print(band.value_counts().reindex(["<2mo", "2-12mo", "12-60mo"]).to_string())

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cohort[["STUDY_PAT_ID", "SLEEP_STUDY_ID", "age_months", "record"]].to_csv(out, index=False)
    print(f"\n-> {out}")

    total_gb_hint = len(cohort) * 0.35
    print(f"Estimated download: ~{total_gb_hint:.0f} GB ({len(cohort)} full-night PSG EDFs).")
    if a.print_commands:
        print("\n# run these yourself; they need your PhysioNet credentials")
        for r in cohort["record"]:
            print(f'wget -c -N -np --user <USER> --ask-password -P data/nchsdb/Sleep_Data/ '
                  f'"{BASE_URL}{r}"')
    else:
        print("Re-run with --print-commands to emit the per-record wget lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
