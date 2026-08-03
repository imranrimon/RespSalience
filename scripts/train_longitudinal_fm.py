"""
Foundation-model benchmark for longitudinal respiratory forecasting.

Builds pseudo-longitudinal trajectories from ablation raw outputs, then evaluates:
1) TabPFN (tabular FM) on sliding-window forecasting
2) TimesFM (time-series FM) zero-shot next-step forecasting
3) Chronos / Chronos-2 (time-series FM) zero-shot next-step forecasting

Input expected:
- One or more ablation_raw_*.json files from run_ablation_study / camera-ready runs.
"""

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Sample:
    patient_id: str
    source: str
    context: np.ndarray  # shape [context_len]
    target: float
    current: float


def _timesfm_python_supported() -> bool:
    """TimesFM currently targets Python 3.10/3.11 in this project setup."""
    return sys.version_info.major == 3 and sys.version_info.minor in (10, 11)


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None


def _load_raw_records(raw_paths: Sequence[Path]) -> List[dict]:
    all_records: List[dict] = []
    for path in raw_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                all_records.extend(data)
            else:
                print(f"[WARN] Skipping non-list JSON: {path}")
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}")
    return all_records


def _extract_patient_and_step(subject: str, source: str) -> Tuple[str, int]:
    subject = str(subject or "")
    source = str(source or "")

    # AIR-125 format: AIR125-S04-025
    if source == "AIR-125":
        m = re.match(r"^AIR125-(S\d+)-(\d+)$", subject)
        if m:
            return m.group(1), int(m.group(2))

    # AIR-400 format: S07/01
    if source == "AIR-400":
        m = re.match(r"^(S\d+)[/\\-](\d+)$", subject)
        if m:
            return m.group(1), int(m.group(2))

    # Pediatric often looks like ID-4008
    m = re.match(r"^([A-Za-z]+)[-_]?(\d+)$", subject)
    if m:
        return m.group(1), int(m.group(2))

    # Generic fallback: last integer token as time index
    ints = re.findall(r"\d+", subject)
    step = int(ints[-1]) if ints else 0
    patient = re.sub(r"[\d/_-]+$", "", subject).strip("_-/")
    if not patient:
        patient = "unknown"
    return patient, step


def _build_patient_sequences(
    records: Sequence[dict],
    baseline_name: str,
    sources_filter: Optional[Sequence[str]] = None,
) -> Dict[Tuple[str, str], List[float]]:
    rows: List[Tuple[str, str, int, float]] = []
    source_set = set(sources_filter or [])

    for rec in records:
        if rec.get("name") != baseline_name:
            continue
        source = str(rec.get("source", ""))
        if source_set and source not in source_set:
            continue
        gt = _safe_float(rec.get("gt_bpm"))
        if gt is None:
            continue
        subject = str(rec.get("subject", ""))
        patient, step = _extract_patient_and_step(subject, source)
        rows.append((source, patient, step, gt))

    grouped: Dict[Tuple[str, str], Dict[int, List[float]]] = {}
    for source, patient, step, gt in rows:
        key = (source, patient)
        grouped.setdefault(key, {}).setdefault(step, []).append(gt)

    sequences: Dict[Tuple[str, str], List[float]] = {}
    for key, step_map in grouped.items():
        ordered_steps = sorted(step_map.keys())
        seq = [float(np.mean(step_map[s])) for s in ordered_steps]
        sequences[key] = seq

    return sequences


def _build_samples(
    sequences: Dict[Tuple[str, str], List[float]],
    context_len: int,
    horizon: int,
    min_seq_len: int,
) -> List[Sample]:
    samples: List[Sample] = []
    for (source, patient), seq in sequences.items():
        if len(seq) < max(min_seq_len, context_len + horizon):
            continue
        for i in range(context_len, len(seq) - horizon + 1):
            ctx = np.array(seq[i - context_len : i], dtype=np.float32)
            target = float(seq[i + horizon - 1])
            current = float(seq[i - 1])
            samples.append(
                Sample(
                    patient_id=f"{source}:{patient}",
                    source=source,
                    context=ctx,
                    target=target,
                    current=current,
                )
            )
    return samples


def _split_by_patient(samples: Sequence[Sample], train_frac: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    patient_ids = sorted({s.patient_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    n_train = max(1, int(len(patient_ids) * train_frac))
    train_patients = set(patient_ids[:n_train])
    train = [s for s in samples if s.patient_id in train_patients]
    test = [s for s in samples if s.patient_id not in train_patients]
    return train, test


def _binary_labels(samples: Sequence[Sample], delta: float) -> np.ndarray:
    return np.array([1 if (s.target - s.current) >= delta else 0 for s in samples], dtype=np.int64)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true == y_pred).astype(np.float32)))


def _f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _precision_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def _recall_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def _balanced_acc_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return float((tpr + tnr) / 2.0)


def _evaluate_forecasts(
    samples_test: Sequence[Sample],
    preds_next: np.ndarray,
    deterioration_delta: float,
) -> Dict[str, float]:
    y_true = np.array([s.target for s in samples_test], dtype=np.float32)
    y_cls_true = _binary_labels(samples_test, deterioration_delta)
    y_cls_pred = np.array(
        [1 if (float(preds_next[i]) - float(samples_test[i].current)) >= deterioration_delta else 0 for i in range(len(samples_test))],
        dtype=np.int64,
    )
    tp = int(np.sum((y_cls_true == 1) & (y_cls_pred == 1)))
    fp = int(np.sum((y_cls_true == 0) & (y_cls_pred == 1)))
    fn = int(np.sum((y_cls_true == 1) & (y_cls_pred == 0)))
    tn = int(np.sum((y_cls_true == 0) & (y_cls_pred == 0)))
    return {
        "mae_next_bpm": _mae(y_true, preds_next),
        "rmse_next_bpm": _rmse(y_true, preds_next),
        "acc_deterioration": _acc(y_cls_true, y_cls_pred),
        "balanced_acc_deterioration": _balanced_acc_binary(y_cls_true, y_cls_pred),
        "precision_deterioration": _precision_binary(y_cls_true, y_cls_pred),
        "recall_deterioration": _recall_binary(y_cls_true, y_cls_pred),
        "f1_deterioration": _f1_binary(y_cls_true, y_cls_pred),
        "true_positive_rate": float(np.mean(y_cls_true)),
        "pred_positive_rate": float(np.mean(y_cls_pred)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_test_samples": int(len(samples_test)),
    }


def _run_tabpfn(
    samples_train: Sequence[Sample],
    samples_test: Sequence[Sample],
    deterioration_delta: float,
    device: str,
) -> Dict[str, float]:
    try:
        from tabpfn import TabPFNRegressor
    except Exception as exc:
        raise RuntimeError(f"TabPFN import failed. Install with `pip install tabpfn`. Error: {exc}") from exc

    x_train = np.stack([s.context for s in samples_train], axis=0)
    y_train = np.array([s.target for s in samples_train], dtype=np.float32)
    x_test = np.stack([s.context for s in samples_test], axis=0)

    reg = TabPFNRegressor(device=device)
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test).astype(np.float32)
    return _evaluate_forecasts(samples_test, pred, deterioration_delta)


def _run_timesfm(
    samples_test: Sequence[Sample],
    deterioration_delta: float,
    device: str,
    model_id: str,
) -> Dict[str, float]:
    try:
        import timesfm
    except Exception as exc:
        raise RuntimeError(
            "TimesFM import failed. Install with `pip install timesfm` and a working `torch` in Python 3.10/3.11."
        ) from exc

    contexts = [s.context.astype(np.float32) for s in samples_test]
    max_ctx = max(len(c) for c in contexts)
    ctx_len = int(math.ceil(max_ctx / 32.0) * 32)
    backend = "gpu" if device == "cuda" else "cpu"

    # Keep hparams aligned with TimesFM defaults to avoid checkpoint shape mismatches.
    # In particular, model dimensions/layer count differ across released checkpoints.
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=min(32, len(contexts)),
            horizon_len=1,
            context_len=max(32, ctx_len),
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_id),
    )

    point, _ = tfm.forecast(contexts, freq=[0] * len(contexts))
    point = np.asarray(point, dtype=np.float32)
    pred = point[:, 0] if point.ndim > 1 else point
    return _evaluate_forecasts(samples_test, pred, deterioration_delta)


def _run_chronos(
    samples_test: Sequence[Sample],
    deterioration_delta: float,
    device: str,
    model_id: str,
    num_samples: int,
) -> Dict[str, float]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for Chronos.") from exc

    # Prefer legacy ChronosPipeline for direct array predict, fallback to Chronos2Pipeline dataframe API.
    pred_values: List[float] = []
    try:
        from chronos import ChronosPipeline

        pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )

        for s in samples_test:
            forecast = pipeline.predict(
                context=torch.tensor(s.context, dtype=torch.float32),
                prediction_length=1,
                num_samples=num_samples,
            )
            arr = forecast[0].detach().cpu().numpy()  # [num_samples, horizon]
            pred_values.append(float(np.median(arr[:, -1])))
    except Exception:
        try:
            import pandas as pd
            from chronos import Chronos2Pipeline
        except Exception as exc:
            raise RuntimeError(
                "Chronos import failed. Install with `pip install chronos-forecasting` in Python >=3.10."
            ) from exc

        pipeline = Chronos2Pipeline.from_pretrained(
            model_id,
            device_map=device,
        )

        rows = []
        for i, s in enumerate(samples_test):
            for t, y in enumerate(s.context.tolist()):
                rows.append(
                    {
                        "id": f"s{i}",
                        "timestamp": t,
                        "target": float(y),
                    }
                )
        context_df = pd.DataFrame(rows)
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=1,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        grouped = pred_df.groupby("id", as_index=False)["predictions"].median()
        pred_values = grouped.sort_values("id")["predictions"].astype(float).tolist()

    pred = np.array(pred_values, dtype=np.float32)
    if len(pred) != len(samples_test):
        raise RuntimeError(f"Chronos produced {len(pred)} predictions for {len(samples_test)} samples.")
    return _evaluate_forecasts(samples_test, pred, deterioration_delta)


def _discover_raw_paths(raw_glob: str) -> List[Path]:
    found = [p for p in Path(".").glob(raw_glob) if p.is_file()]
    return sorted(found)


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate TabPFN/TimesFM/Chronos on pseudo-longitudinal respiratory trajectories")
    parser.add_argument(
        "--raw-jsons",
        type=str,
        default="",
        help="Comma-separated ablation_raw JSON paths. If omitted, --raw-glob is used.",
    )
    parser.add_argument(
        "--raw-glob",
        type=str,
        default="outputs/**/ablation_raw_*.json",
        help="Glob pattern used when --raw-jsons is omitted.",
    )
    parser.add_argument("--baseline", type=str, default="Quad-Modal Full (Ours)")
    parser.add_argument(
        "--sources",
        type=str,
        default="",
        help="Comma-separated sources to include (e.g., AIR-125,AIR-400). Empty = use all discovered sources.",
    )
    parser.add_argument("--context-len", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-seq-len", type=int, default=6)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--deterioration-delta", type=float, default=3.0)
    parser.add_argument(
        "--deterioration-deltas",
        type=str,
        default="",
        help="Optional comma-separated deterioration thresholds (e.g., '1.0,1.5,2.0,3.0'). Overrides --deterioration-delta.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default="tabpfn,timesfm,chronos")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--timesfm-model-id", type=str, default="google/timesfm-1.0-200m-pytorch")
    parser.add_argument("--chronos-model-id", type=str, default="amazon/chronos-bolt-small")
    parser.add_argument("--chronos-num-samples", type=int, default=20)
    parser.add_argument("--output-json", type=str, default="outputs/exp3_longitudinal/fm_results.json")
    parser.add_argument("--output-csv", type=str, default="outputs/exp3_longitudinal/fm_results.csv")
    args = parser.parse_args()

    if args.raw_jsons.strip():
        raw_paths = [Path(p) for p in _parse_csv_list(args.raw_jsons)]
    else:
        raw_paths = _discover_raw_paths(args.raw_glob)

    if not raw_paths:
        raise SystemExit("No ablation raw JSON files found. Provide --raw-jsons or adjust --raw-glob.")

    print(f"Using {len(raw_paths)} raw files.")
    records = _load_raw_records(raw_paths)
    if not records:
        raise SystemExit("No records loaded from raw JSON files.")
    print(f"Loaded {len(records)} records.")

    if args.sources.strip():
        sources = _parse_csv_list(args.sources)
    else:
        discovered_sources = sorted({str(r.get("source", "")).strip() for r in records if str(r.get("source", "")).strip()})
        if not discovered_sources:
            raise SystemExit("No source labels found in records; pass --sources explicitly.")
        sources = discovered_sources
        print(f"[INFO] --sources not provided; using all discovered sources: {sources}")

    sequences = _build_patient_sequences(records, baseline_name=args.baseline, sources_filter=sources)
    print(f"Built {len(sequences)} patient sequences for baseline='{args.baseline}' and sources={sources}.")

    samples = _build_samples(
        sequences=sequences,
        context_len=args.context_len,
        horizon=args.horizon,
        min_seq_len=args.min_seq_len,
    )
    if len(samples) < 10:
        raise SystemExit(f"Too few samples ({len(samples)}). Try lowering --context-len/--min-seq-len or using more raw JSON files.")
    print(f"Generated {len(samples)} forecasting samples.")

    train, test = _split_by_patient(samples, train_frac=args.train_frac, seed=args.seed)
    if not train or not test:
        raise SystemExit(f"Invalid split. train={len(train)} test={len(test)}")
    print(f"Split by patient: train={len(train)} test={len(test)}")

    if args.deterioration_deltas.strip():
        deterioration_deltas = [float(x) for x in _parse_csv_list(args.deterioration_deltas)]
        if not deterioration_deltas:
            raise SystemExit("--deterioration-deltas was provided but no valid values were parsed.")
    else:
        deterioration_deltas = [float(args.deterioration_delta)]

    results = []
    model_names = _parse_csv_list(args.models.lower())

    for model in model_names:
        for deterioration_delta in deterioration_deltas:
            run_label = model if len(deterioration_deltas) == 1 else f"{model}@d{deterioration_delta:g}"
            print(f"\n[RUN] {run_label}")
            try:
                if model == "tabpfn":
                    metrics = _run_tabpfn(train, test, deterioration_delta, args.device)
                elif model == "timesfm":
                    if not _timesfm_python_supported():
                        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                        msg = (
                            "Skipped TimesFM: requires Python 3.10/3.11 in this setup "
                            f"(current: {py_ver}). Use a separate env (e.g., timesfm_env) for --models timesfm."
                        )
                        print(f"  [SKIP] {msg}")
                        results.append({"model": run_label, "deterioration_delta": deterioration_delta, "error": msg})
                        continue
                    metrics = _run_timesfm(test, deterioration_delta, args.device, args.timesfm_model_id)
                elif model == "chronos":
                    metrics = _run_chronos(
                        test,
                        deterioration_delta,
                        args.device,
                        args.chronos_model_id,
                        args.chronos_num_samples,
                    )
                else:
                    print(f"[WARN] Unknown model '{model}', skipping.")
                    continue

                row = {"model": run_label, "base_model": model, "deterioration_delta": deterioration_delta, **metrics}
                results.append(row)
                print(
                    f"  MAE={metrics['mae_next_bpm']:.3f} | RMSE={metrics['rmse_next_bpm']:.3f} "
                    f"| ACC={metrics['acc_deterioration']:.3f} | BACC={metrics['balanced_acc_deterioration']:.3f} "
                    f"| F1={metrics['f1_deterioration']:.3f} | Pos(T/P)={metrics['true_positive_rate']:.3f}/{metrics['pred_positive_rate']:.3f}"
                )
            except Exception as exc:
                print(f"  [ERROR] {run_label} failed: {exc}")
                results.append({"model": run_label, "base_model": model, "deterioration_delta": deterioration_delta, "error": str(exc)})

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "baseline": args.baseline,
            "sources": sources,
            "context_len": args.context_len,
            "horizon": args.horizon,
            "min_seq_len": args.min_seq_len,
            "train_frac": args.train_frac,
            "deterioration_deltas": deterioration_deltas,
            "seed": args.seed,
            "device": args.device,
            "timesfm_model_id": args.timesfm_model_id,
            "chronos_model_id": args.chronos_model_id,
            "chronos_num_samples": args.chronos_num_samples,
            "n_raw_files": len(raw_paths),
            "n_records": len(records),
            "n_sequences": len(sequences),
            "n_samples": len(samples),
            "n_train": len(train),
            "n_test": len(test),
        },
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_rows = []
    for row in results:
        csv_rows.append(
            {
                "model": row.get("model"),
                "base_model": row.get("base_model", row.get("model")),
                "deterioration_delta": row.get("deterioration_delta"),
                "mae_next_bpm": row.get("mae_next_bpm"),
                "rmse_next_bpm": row.get("rmse_next_bpm"),
                "acc_deterioration": row.get("acc_deterioration"),
                "balanced_acc_deterioration": row.get("balanced_acc_deterioration"),
                "precision_deterioration": row.get("precision_deterioration"),
                "recall_deterioration": row.get("recall_deterioration"),
                "f1_deterioration": row.get("f1_deterioration"),
                "true_positive_rate": row.get("true_positive_rate"),
                "pred_positive_rate": row.get("pred_positive_rate"),
                "tp": row.get("tp"),
                "fp": row.get("fp"),
                "fn": row.get("fn"),
                "tn": row.get("tn"),
                "n_test_samples": row.get("n_test_samples"),
                "error": row.get("error", ""),
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "base_model",
                "deterioration_delta",
                "mae_next_bpm",
                "rmse_next_bpm",
                "acc_deterioration",
                "balanced_acc_deterioration",
                "precision_deterioration",
                "recall_deterioration",
                "f1_deterioration",
                "true_positive_rate",
                "pred_positive_rate",
                "tp",
                "fp",
                "fn",
                "tn",
                "n_test_samples",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved CSV : {out_csv}")


if __name__ == "__main__":
    main()

