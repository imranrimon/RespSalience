"""Measure reference-standard uncertainty directly, once reader-level VEP assessments are available.

`scripts/label_noise_ceiling.py` estimates what an imperfect reference costs by SIMULATION, because
per-video reader values are not distributed with Pediatric-RR-332. This script replaces that simulation
with measurement the moment reader-level data arrives, however roughly it is formatted.

It answers five questions that no camera-based respiratory-rate paper has reported:

  1. How far apart are the readers, overall and near the WHO threshold?
  2. How often does the WHO fast-breathing LABEL itself flip between readers?
  3. How often was adjudication triggered, and on which cases?
  4. Does reader disagreement concentrate at the clinical boundary, as our near-threshold
     stratification implies it should?
  5. Does our model perform worse exactly where the readers disagree -- i.e. is residual error shared
     with human uncertainty, or independent of it?

Expected input: any delimited file (csv/tsv/xlsx) with a video or child identifier and at least two
reader rates. Column names are matched case- and separator-insensitively against these aliases, so a
basic spreadsheet works without editing:

  video_id | child_id | filename ....... identifier
  reader1_rr .. reader4_rr ............. individual reader rates (r1_rr, rater1, reader_1 also match)
  final_rr ............................. released consensus rate (falls back to the Stata value)
  adjudicated .......................... bool/0-1/yes-no
  reader1_id, reader2_id, final_pair ... anonymous reader identifiers
  interpretable ........................ bool/0-1/yes-no

    python scripts/reader_level_analysis.py --readers <file>
    python scripts/reader_level_analysis.py --readers <file> --model-run pedrr_kfold_salience_ssl332_seed0
    python scripts/reader_level_analysis.py --template          # writes a blank CSV to send onward
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_main_run import load, scored  # noqa: E402

DTA = Path("data/Video Dataset for Respiratory Rate Counting in Children/Stata File/"
           "Data Respiratory Rate Counting from Video Recording.dta")

ALIASES = {
    "video_id": ["video_id", "videoid", "video", "child_id", "childid", "id", "filename", "file"],
    "reader1_rr": ["reader1_rr", "reader1", "r1_rr", "r1", "rater1", "rater1_rr", "reader_1_rr",
                   "reader_1", "rr1", "obs1"],
    "reader2_rr": ["reader2_rr", "reader2", "r2_rr", "r2", "rater2", "rater2_rr", "reader_2_rr",
                   "reader_2", "rr2", "obs2"],
    "reader3_rr": ["reader3_rr", "reader3", "r3_rr", "r3", "rater3", "rater3_rr", "reader_3_rr",
                   "reader_3", "rr3", "obs3"],
    "reader4_rr": ["reader4_rr", "reader4", "r4_rr", "r4", "rater4", "rater4_rr", "reader_4_rr",
                   "reader_4", "rr4", "obs4"],
    "final_rr": ["final_rr", "finalrr", "consensus_rr", "consensus", "vep_rr", "reference_rr",
                 "respiratory_rate", "rr_final"],
    "adjudicated": ["adjudicated", "adjudication", "was_adjudicated", "third_reader", "escalated"],
    "reader1_id": ["reader1_id", "r1_id", "rater1_id", "reader_1_id"],
    "reader2_id": ["reader2_id", "r2_id", "rater2_id", "reader_2_id"],
    "final_pair": ["final_pair", "agreeing_pair", "pair", "consensus_pair"],
    "interpretable": ["interpretable", "is_interpretable", "usable", "quality_ok", "readable"],
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def map_columns(df):
    lookup = {norm(c): c for c in df.columns}
    out, unmatched = {}, []
    for canon, alts in ALIASES.items():
        hit = next((lookup[norm(a)] for a in alts if norm(a) in lookup), None)
        if hit is not None:
            out[canon] = hit
        elif canon in ("video_id", "reader1_rr", "reader2_rr"):
            unmatched.append(canon)
    return out, unmatched


def to_bool(s):
    if s is None:
        return None
    m = {"1": True, "0": False, "true": True, "false": False, "yes": True, "no": False,
         "y": True, "n": False, "t": True, "f": False}
    return s.astype(str).str.strip().str.lower().map(m)


def who_thr(age):
    return 60.0 if age < 2 else (50.0 if age < 12 else 40.0)


def read_any(path):
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    for sep in (None, ",", "\t", ";", "|"):
        try:
            df = pd.read_csv(p, sep=sep, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise SystemExit(f"could not parse {p}")


def write_template(path):
    cols = ["video_id", "reader1_rr", "reader2_rr", "reader3_rr", "reader4_rr", "adjudicated",
            "final_rr", "reader1_id", "reader2_id", "final_pair", "interpretable"]
    pd.DataFrame(columns=cols).to_csv(path, index=False)
    print(f"blank template -> {path}")
    print("One row per video. Leave reader3/reader4 empty where adjudication was not triggered.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", default="", help="reader-level file (csv/tsv/xlsx)")
    ap.add_argument("--template", action="store_true", help="write a blank CSV template and exit")
    ap.add_argument("--model-run", default="pedrr_kfold_salience_ssl332_seed0",
                    help="run root whose predictions are compared against consensus strength")
    ap.add_argument("--out", default="outputs/reader_level_analysis.csv")
    a = ap.parse_args()

    if a.template or not a.readers:
        write_template(a.out.replace(".csv", "_TEMPLATE.csv"))
        if not a.readers:
            print("\n(no --readers file given; wrote the template only)")
            return 0

    raw = read_any(a.readers)
    cols, missing = map_columns(raw)
    if missing:
        raise SystemExit(f"required column(s) not found: {missing}\nsaw: {list(raw.columns)}")
    print(f"parsed {len(raw)} rows; mapped columns: "
          + ", ".join(f"{k}<-{v}" for k, v in cols.items()))

    d = pd.DataFrame({"video_id": raw[cols["video_id"]].astype(str).str.strip()})
    for k in ("reader1_rr", "reader2_rr", "reader3_rr", "reader4_rr", "final_rr"):
        d[k] = pd.to_numeric(raw[cols[k]], errors="coerce") if k in cols else np.nan
    for k in ("adjudicated", "interpretable"):
        d[k] = to_bool(raw[cols[k]]) if k in cols else None
    for k in ("reader1_id", "reader2_id", "final_pair"):
        d[k] = raw[cols[k]].astype(str) if k in cols else None

    # Ages and released rates come from the cohort's own metadata; readers supply only the ratings.
    # Identifiers vary between sources, so resolve whatever arrives to a canonical subject id.
    if DTA.exists():
        meta = pd.read_stata(DTA, convert_categoricals=False)
        meta["_child"] = meta["child_id"].astype(str).str.strip()
        meta["_file"] = meta["filename"].astype(str).str.strip()
        age_by, rr_by, sid_by = {}, {}, {}
        for _, r in meta.iterrows():
            canon = f"ID-{r['_child']}"
            for k in {r["_child"], canon, r["_file"]}:
                age_by[k] = float(r["age"]); rr_by[k] = float(r["respiratory_rate"])
                sid_by[k] = canon
        stem = d.video_id.str.replace(r"\.(mp4|avi|mov)$", "", regex=True).str.strip()
        d["age"] = stem.map(age_by)
        d["released_rr"] = stem.map(rr_by)
        d["subject_id"] = stem.map(sid_by)
        matched = int(d["age"].notna().sum())
        print(f"  linked {matched}/{len(d)} rows to cohort metadata "
              f"(accepts child_id, ID-child_id or filename)")
        if matched == 0:
            print("[warn] no identifier overlap with the Stata metadata; ages unavailable")
    d["final_rr"] = d["final_rr"].fillna(d.get("released_rr"))
    d = d[d["age"].notna()] if "age" in d and d["age"].notna().any() else d
    if d.empty:
        raise SystemExit("no rows could be linked to the cohort metadata")
    d["thr"] = d["age"].map(who_thr)

    # ---- 1. reader disagreement -------------------------------------------------------------
    d["reader_diff"] = (d.reader1_rr - d.reader2_rr).abs()
    ok = d.reader_diff.notna()
    print(f"\n=== 1. Reader disagreement (n={int(ok.sum())} with two readers) ===")
    print(f"  |r1-r2|: mean {d.reader_diff[ok].mean():.2f}, median {d.reader_diff[ok].median():.2f}, "
          f"90th pct {d.reader_diff[ok].quantile(0.9):.2f} breaths/min")
    print(f"  within 2 br/min: {(d.reader_diff[ok] <= 2).mean()*100:.1f}%")
    print(f"  implied single-reader SD: {d.reader_diff[ok].std(ddof=1)/np.sqrt(2):.2f} br/min "
          "(compare the 1.35 assumed in the sensitivity analysis)")

    # ---- 2. WHO label flips between readers -------------------------------------------------
    l1 = (d.reader1_rr >= d.thr); l2 = (d.reader2_rr >= d.thr)
    flip = (l1 != l2) & ok
    print(f"\n=== 2. WHO label instability (the quantity we could only simulate) ===")
    print(f"  labels that differ between reader 1 and reader 2: {int(flip.sum())}/{int(ok.sum())} "
          f"({flip.sum()/max(1,ok.sum())*100:.1f}%)")
    print("  the sensitivity analysis assumed ~2.2% at SD 1.35 -- this is the measured value")

    # ---- 3. adjudication --------------------------------------------------------------------
    if d["adjudicated"].notna().any():
        adj = d["adjudicated"].fillna(False).astype(bool)
        print(f"\n=== 3. Adjudication ===")
        print(f"  triggered on {int(adj.sum())}/{len(d)} ({adj.mean()*100:.1f}%)")
        if ok.any():
            print(f"  mean |r1-r2| when adjudicated {d.reader_diff[adj & ok].mean():.2f} vs "
                  f"{d.reader_diff[~adj & ok].mean():.2f} otherwise")
    if d["interpretable"].notna().any():
        it = d["interpretable"].fillna(True).astype(bool)
        print(f"  interpretable: {int(it.sum())}/{len(d)} ({it.mean()*100:.1f}%)")

    # ---- 4. does disagreement concentrate at the boundary? -----------------------------------
    d["dist"] = (d.final_rr - d.thr).abs()
    print(f"\n=== 4. Disagreement vs distance from the WHO cutoff ===")
    print(f"  {'stratum':<14}{'n':>5}{'mean |r1-r2|':>14}{'label flips':>13}")
    for lab, m in (("<=2 br/min", d.dist <= 2), ("2-5 br/min", (d.dist > 2) & (d.dist <= 5)),
                   (">5 br/min", d.dist > 5)):
        mm = m & ok
        if not mm.any():
            continue
        print(f"  {lab:<14}{int(mm.sum()):>5}{d.reader_diff[mm].mean():>14.2f}"
              f"{flip[mm].mean()*100:>12.1f}%")

    # ---- 5. model error vs consensus strength ------------------------------------------------
    ref = load("pedrr_kfold_flowonly_agefix")
    if ref is not None:
        age_map = dict(zip(ref["subject_id"].astype(str), ref["age_months"].astype(float)))
        pred = scored(a.model_run, age_map)
        if pred is not None:
            pred["k"] = pred["sid"].astype(str)
            left_key = d["subject_id"] if "subject_id" in d.columns else \
                d.video_id.str.replace(r"\.(mp4|avi|mov)$", "", regex=True)
            j = d.assign(_jk=left_key).merge(
                pred[["k", "pred_rr_bpm", "gt_rr_bpm", "tachy_prob", "y", "thr"]],
                left_on="_jk", right_on="k", how="inner", suffixes=("", "_m"))
            if len(j):
                j["abs_err"] = (j.pred_rr_bpm - j.gt_rr_bpm).abs()
                j["dec"] = (j.pred_rr_bpm >= j.thr_m).astype(int)
                print(f"\n=== 5. Model performance vs consensus strength (n={len(j)}) ===")
                print(f"  {'readers agree':<16}{'n':>5}{'model MAE':>11}{'model correct':>15}")
                for lab, m in (("within 1", j.reader_diff <= 1), ("1-2", (j.reader_diff > 1) & (j.reader_diff <= 2)),
                               ("> 2", j.reader_diff > 2)):
                    if not m.any():
                        continue
                    corr = (j.dec[m] == j.y[m]).mean()
                    print(f"  {lab:<16}{int(m.sum()):>5}{j.abs_err[m].mean():>11.2f}{corr*100:>14.1f}%")
                lo, hi = j.reader_diff <= 2, j.reader_diff > 2
                if lo.any() and hi.any():
                    b = binomtest(int((j.dec[hi] == j.y[hi]).sum()), int(hi.sum()),
                                  float((j.dec[lo] == j.y[lo]).mean()))
                    print(f"  model accuracy where readers disagree vs agree: p={b.pvalue:.4f}")
                    print("  a significant drop means our residual error is SHARED with human")
                    print("  uncertainty -- the cases are intrinsically hard, not model-specific.")
            else:
                print("\n[warn] no identifier overlap with model predictions; skipped section 5")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)
    print(f"\n-> {a.out}")
    print("If this ran, replace the simulated ceiling in Sec. sec:ceiling with these measured values")
    print("and drop the 'sensitivity analysis' framing for the measured quantities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
