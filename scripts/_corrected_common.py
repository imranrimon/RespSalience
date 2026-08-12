"""Shared corrected-label loader for the CPU post-hoc analyses (near-threshold, calibration,
fairness, lean non-inferiority, conformal coverage).

Reproduces the age-newborn fix EXACTLY as scripts/canonical_recompute.py: load fold*/test_eval/
prism_breath_predictions.csv, join authoritative age/sex/facility from the Stata .dta via
video_id == filename, recompute the WHO threshold and gt_tachy. Also surfaces conformal columns
(interval_lower_bpm, interval_upper_bpm, qhat) when present. CPU-only, read-only on cached CSVs.
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import full_metric_comparison as F  # reuse point_metrics / subj_boot / RUNS

DTA = "data/Video Dataset for Respiratory Rate Counting in Children/Stata File/Data Respiratory Rate Counting from Video Recording.dta"
_dta = pd.read_stata(DTA, convert_categoricals=False)
_dta["filename"] = _dta["filename"].astype(str)
AGE = dict(zip(_dta["filename"], _dta["age"].astype(int)))
SEX = dict(zip(_dta["filename"], _dta["sex"].astype(int)))       # 1=Male, 2=Female
FAC = dict(zip(_dta["filename"], _dta["facility_code"].astype(int)))  # 1=ICMH, 2=Upazila HC, 3=Community Clinic

SEX_LABEL = {1: "Male", 2: "Female"}
FAC_LABEL = {1: "ICMH", 2: "Upazila Health Complex", 3: "Community Clinic"}


def who(a):
    return 60.0 if a < 2 else (50.0 if a < 12 else 40.0)


def band(a):
    return "<2mo" if a < 2 else ("2-12mo" if a < 12 else "12-60mo")


def load_corrected(run_root):
    """Return one row per clip with corrected age/threshold/gt_tachy + conformal columns if present."""
    paths = sorted(glob.glob(os.path.join(run_root, "fold*", "test_eval", "prism_breath_predictions.csv")))
    if not paths:
        raise FileNotFoundError(f"no fold*/test_eval predictions under {run_root}")
    frames = [pd.read_csv(p).assign(fold=i) for i, p in enumerate(paths)]
    df = pd.concat(frames, ignore_index=True)
    df["video_id"] = df["video_id"].astype(str)
    df["clip_key"] = df["subject_id"].astype(str) + "_" + df["video_id"]
    agg = dict(
        subject_id=("subject_id", "first"), video_id=("video_id", "first"), fold=("fold", "first"),
        gt_rr=("gt_rr_bpm", "mean"), pred_rr=("pred_rr_bpm", "mean"),
        threshold_old=("threshold_bpm", "first"), gt_tachy_old=("gt_tachy", "first"),
        tachy_prob=("tachy_prob", "mean"),
    )
    for col, key in [("ilo", "interval_lower_bpm"), ("ihi", "interval_upper_bpm"), ("qhat", "qhat")]:
        if key in df.columns:
            agg[col] = (key, "mean")
    g = df.groupby("clip_key").agg(**agg).reset_index()
    g["age"] = g["video_id"].map(AGE)
    g["sex"] = g["video_id"].map(SEX)
    g["fac"] = g["video_id"].map(FAC)
    g["threshold"] = g["age"].apply(who)                        # CORRECTED WHO threshold
    g["gt_tachy"] = (g["gt_rr"] >= g["threshold"]).astype(int)  # CORRECTED label
    g["band"] = g["age"].apply(band)
    return g
