import pandas as pd
import numpy as np
from typing import Dict, List


class AblationManager:
    """Manages multi-step ablation studies and baseline comparisons."""

    def __init__(self, metrics=["MAE", "RMSE", "Pearson_r", "IoA@2BPM", "IoA@3BPM"]):
        self.metrics = metrics
        self.results = []

    def log_experiment(self, name: str, config: Dict, pred_bpm: float, gt_bpm: float, **kwargs):
        """Logs a single ablation experiment result."""
        error = abs(pred_bpm - gt_bpm)
        sq_error = (pred_bpm - gt_bpm) ** 2
        result = {
            "name":         name,
            "pred_bpm":     pred_bpm,
            "gt_bpm":       gt_bpm,
            "abs_error":    error,
            "squared_error": sq_error,
            **config,
            **kwargs,
        }
        self.results.append(result)

    def generate_report(self) -> pd.DataFrame:
        """
        Aggregates results and computes all paper-ready metrics:
          MAE, RMSE, Pearson ρ, IoA@2BPM, IoA@3BPM, N
        IoA (Index of Agreement) @ ±2/3 BPM is the clinical acceptability
        threshold for neonatal/pediatric respiratory rate monitoring.
        """
        df = pd.DataFrame(self.results)
        if df.empty:
            return df

        group_cols = ["name"]
        if "paradigm" in df.columns:
            group_cols = ["paradigm"] + group_cols
        if "source" in df.columns:
            group_cols.insert(1 if "paradigm" in group_cols else 0, "source")

        def pearson_r(g):
            if len(g) < 2 or g["gt_bpm"].std() < 1e-6:
                return float("nan")
            return float(np.corrcoef(g["pred_bpm"], g["gt_bpm"])[0, 1])

        rows = []
        for keys, g in df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            row["MAE"]       = round(float(g["abs_error"].mean()), 2)
            row["RMSE"]      = round(float(np.sqrt(g["squared_error"].mean())), 2)
            row["Pearson_r"] = round(pearson_r(g), 3)
            row["IoA@2BPM"]  = round(100.0 * (g["abs_error"] <= 2.0).mean(), 1)
            row["IoA@3BPM"]  = round(100.0 * (g["abs_error"] <= 3.0).mean(), 1)
            row["N"]         = len(g)
            rows.append(row)

        result_df = pd.DataFrame(rows)
        if group_cols:
            result_df = result_df.set_index(group_cols)
        return result_df

    def generate_latex_table(self, caption: str = "Ablation Study Results",
                              label: str = "tab:ablation") -> str:
        """
        Generates a LaTeX table ready to paste into a paper.
        Best row per metric is bolded automatically.
        """
        df = self.generate_report().reset_index()
        if df.empty:
            return "% No results to render."

        metrics = ["MAE", "RMSE", "Pearson_r", "IoA@2BPM", "IoA@3BPM"]
        available = [m for m in metrics if m in df.columns]

        lines = [
            r"\begin{table}[!t]",
            r"\centering",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\resizebox{\columnwidth}{!}{",
            r"\begin{tabular}{l" + "c" * len(available) + r"}",
            r"\hline",
        ]

        # Header
        header_map = {
            "MAE": r"MAE$\downarrow$",
            "RMSE": r"RMSE$\downarrow$",
            "Pearson_r": r"$\rho\uparrow$",
            "IoA@2BPM": r"IoA@2\uparrow",
            "IoA@3BPM": r"IoA@3\uparrow",
        }
        header = "Method & " + " & ".join(header_map.get(m, m) for m in available) + r" \\"
        lines += [header, r"\hline"]

        # Find best per metric (lower=better for MAE/RMSE, higher=better for rest)
        lower_better = {"MAE", "RMSE"}
        bests = {}
        for m in available:
            col = df[m].dropna()
            if m in lower_better:
                bests[m] = col.min()
            else:
                bests[m] = col.max()

        # Paradigm grouping separator
        last_paradigm = None
        for _, row in df.iterrows():
            paradigm = row.get("paradigm", "")
            if paradigm and paradigm != last_paradigm:
                if last_paradigm is not None:
                    lines.append(r"\hline")
                lines.append(r"\multicolumn{" + str(len(available) + 1) +
                              r"}{l}{\textit{" + str(paradigm) + r"}} \\")
                last_paradigm = paradigm

            name = str(row.get("name", row.name if hasattr(row, "name") else ""))
            cells = []
            for m in available:
                val = row.get(m, float("nan"))
                if pd.isna(val):
                    cells.append("—")
                    continue
                fmt = f"{val:.1f}" if m in ("IoA@2BPM", "IoA@3BPM") else f"{val:.2f}"
                if val == bests.get(m):
                    fmt = r"\textbf{" + fmt + "}"
                cells.append(fmt)
            lines.append(name + " & " + " & ".join(cells) + r" \\")

        lines += [
            r"\hline",
            r"\end{tabular}",
            r"}",  # end resizebox
            r"\end{table}",
        ]
        return "\n".join(lines)

    @staticmethod
    def get_ablation_configs():
        """Returns standard ablation configurations for the study."""
        return [
            {"name": "Base (Flow Only)",       "config": {"modality": "flow",  "vlm": "none"}},
            {"name": "Depth Only",             "config": {"modality": "depth", "vlm": "none"}},
            {"name": "SOTA Hybrid",            "config": {"modality": "fused", "vlm": "none"}},
            {"name": "Hybrid + VLM Analysis",  "config": {"modality": "fused", "vlm": "medgemma"}},
        ]


if __name__ == "__main__":
    # Smoke test
    np.random.seed(42)
    am = AblationManager()
    baselines = [
        ("Flow-Only",           "Unsupervised", 8.0),
        ("Fused-VMD",           "Unsupervised", 5.4),
        ("Fused-DL-UNet",       "Unsupervised", 4.1),
        ("Quad-Modal Full (Ours)", "Semi-Supervised", 2.8),
    ]
    for name, paradigm, noise_std in baselines:
        for _ in range(50):
            gt  = 30 + np.random.randn() * 5
            pred = gt + np.random.randn() * noise_std
            am.log_experiment(name, {"paradigm": paradigm}, pred, gt)

    print("\n--- Ablation Report ---")
    print(am.generate_report().to_string())
    print("\n--- LaTeX Table ---")
    print(am.generate_latex_table())
