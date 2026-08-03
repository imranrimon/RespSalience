"""
Conformal inference heads for clinically safe respiratory monitoring.

This module adds two architectural components:
1) ConformalTriageHead: Converts point BPM + conformal interval into triage decisions.
2) DisagreementConditionedConformal: Selects qhat optionally conditioned on cross-modal disagreement.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def age_based_tachypnea_threshold(age_months: int) -> int:
    """WHO-style tachypnea threshold by age."""
    if age_months < 2:
        return 60
    if age_months < 12:
        return 50
    if age_months <= 60:
        return 40
    return 20


class ConformalTriageHead:
    """
    Decision head over conformal intervals.

    Triage states:
    - NORMAL_SAFE: upper interval is below tachypnea threshold.
    - TACHYPNEA_ALERT: lower interval is above threshold.
    - UNCERTAIN_REVIEW: interval overlaps threshold (abstain/escalate).
    - CRITICAL_APNEA_ALERT: anomaly-driven override.
    """

    def triage(
        self,
        bpm: float,
        age_months: int,
        qhat: float,
        disagreement_score: float = 0.0,
        anomaly_score: float = 0.0,
    ) -> Dict[str, Any]:
        threshold = age_based_tachypnea_threshold(int(age_months))
        qhat = float(max(0.0, qhat))
        lower = float(bpm - qhat)
        upper = float(bpm + qhat)
        overlaps_threshold = bool(lower < threshold <= upper)

        if anomaly_score > 0.5:
            status = "CRITICAL_APNEA_ALERT"
            rationale = "High anomaly score overrides interval-based tachypnea decision."
        elif upper < threshold:
            status = "NORMAL_SAFE"
            rationale = "Entire interval below tachypnea threshold."
        elif lower >= threshold:
            status = "TACHYPNEA_ALERT"
            rationale = "Entire interval above tachypnea threshold."
        else:
            status = "UNCERTAIN_REVIEW"
            rationale = "Interval crosses threshold; manual review recommended."

        return {
            "triage_status": status,
            "threshold_bpm": float(threshold),
            "point_bpm": float(bpm),
            "interval_lower_bpm": round(lower, 3),
            "interval_upper_bpm": round(upper, 3),
            "qhat": round(qhat, 3),
            "overlaps_threshold": overlaps_threshold,
            "disagreement_score": round(float(disagreement_score), 4),
            "rationale": rationale,
        }


class DisagreementConditionedConformal:
    """
    qhat selector with optional disagreement-conditioned lookup.
    """

    def __init__(
        self,
        calibration_csv: Optional[str] = None,
        default_qhat: float = 6.0,
        disagreement_bins: Tuple[float, float] = (0.2, 0.4),
    ):
        self.default_qhat = float(default_qhat)
        self.disagreement_bins = tuple(sorted([float(x) for x in disagreement_bins]))[:2]
        self.calibration_df: Optional[pd.DataFrame] = None

        if calibration_csv:
            path = Path(calibration_csv)
            if path.exists():
                try:
                    self.calibration_df = pd.read_csv(path)
                except Exception as exc:
                    print(f"[Conformal] Failed to load calibration CSV '{path}': {exc}")
            else:
                print(f"[Conformal] Calibration CSV not found: {path}")

    def disagreement_bin(self, score: float) -> str:
        low_hi = self.disagreement_bins
        s = float(score)
        if s <= low_hi[0]:
            return "low"
        if s <= low_hi[1]:
            return "mid"
        return "high"

    @staticmethod
    def _pick_qhat_value(df: pd.DataFrame) -> float:
        for col in ["qhat", "qhat_mean"]:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if not series.empty:
                    return float(series.iloc[0])
        return float("nan")

    def select_qhat(
        self,
        model_name: str,
        paradigm: Optional[str] = None,
        source: Optional[str] = None,
        disagreement_score: float = 0.0,
        condition_on_disagreement: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Returns:
            qhat, metadata dict
        """
        meta = {
            "qhat_source": "default",
            "qhat_model_name": model_name,
            "qhat_paradigm": paradigm if paradigm is not None else "",
            "qhat_source_dataset": source if source is not None else "",
            "qhat_disagreement_bin": self.disagreement_bin(disagreement_score),
        }

        if self.calibration_df is None or self.calibration_df.empty:
            return self.default_qhat, meta

        df = self.calibration_df.copy()
        if "name" not in df.columns:
            return self.default_qhat, meta

        candidates = df[df["name"].astype(str) == str(model_name)]
        if candidates.empty:
            return self.default_qhat, meta

        if paradigm is not None and "paradigm" in candidates.columns:
            cand2 = candidates[candidates["paradigm"].astype(str) == str(paradigm)]
            if not cand2.empty:
                candidates = cand2

        if source is not None and "source" in candidates.columns:
            cand_source = candidates[candidates["source"].astype(str) == str(source)]
            if not cand_source.empty:
                candidates = cand_source
                meta["qhat_source"] = "calibration_source_level"

        if condition_on_disagreement and "disagreement_bin" in candidates.columns:
            b = self.disagreement_bin(disagreement_score)
            cand3 = candidates[candidates["disagreement_bin"].astype(str) == b]
            if not cand3.empty:
                candidates = cand3
                prefix = "calibration_source" if meta["qhat_source"] == "calibration_source_level" else "calibration"
                meta["qhat_source"] = f"{prefix}_disagreement_bin"
            else:
                if meta["qhat_source"] == "calibration_source_level":
                    meta["qhat_source"] = "calibration_source_level_fallback"
                else:
                    meta["qhat_source"] = "calibration_model_level_fallback"
        else:
            if meta["qhat_source"] == "default":
                meta["qhat_source"] = "calibration_model_level"

        qhat = self._pick_qhat_value(candidates)
        if np.isnan(qhat):
            return self.default_qhat, meta
        return float(max(0.0, qhat)), meta
