"""
Respiratory benchmark registry for the paper-ready task.

This module narrows the benchmark to contactless respiratory-rate estimation
with safe tachypnea triage. It builds canonical manifests across AIR-125,
AIR-400, and Pediatric-RR, including subject-disjoint splits and
leave-one-dataset-out evaluation files.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from src.utils.dataset_paths import iter_pediatric_rr_videos
from src.utils.respiratory_ground_truth import (
    load_air125_ground_truth,
    load_air400_ground_truth,
    load_pediatric_rr_ground_truth,
)


# Pediatric-RR-332 (real-world Bangladeshi under-5 clinical cohort) is held out
# in its entirety as the external, out-of-distribution test set by default, so
# cross-population generalization is measured rather than assumed. Pass an empty
# external set (``--pool-external``) to revert to pooling every dataset.
DEFAULT_EXTERNAL_TEST_DATASETS = ("Pediatric-RR",)

_EXTERNAL_DATASET_ALIASES = {
    "pediatric-rr": "Pediatric-RR",
    "pediatric-rr-332": "Pediatric-RR",
    "pediatric_rr_332": "Pediatric-RR",
    "pediatric_rr": "Pediatric-RR",
}


def _canonical_external_name(name: str) -> str:
    text = str(name or "").strip()
    return _EXTERNAL_DATASET_ALIASES.get(text.lower(), text)


def _external_dataset_set(external_test_datasets: Iterable[str]) -> set:
    return {_canonical_external_name(name) for name in (external_test_datasets or ()) if str(name).strip()}


def _stable_fraction(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    value = int(digest, 16)
    return value / float(16 ** 16 - 1)


def _subject_split_map(subject_ids: List[str], dataset_name: str, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, str]:
    ordered = sorted(subject_ids, key=lambda sid: hashlib.sha256(f"{dataset_name}|{sid}".encode("utf-8")).hexdigest())
    n_subjects = len(ordered)
    if n_subjects == 0:
        return {}
    if n_subjects == 1:
        return {ordered[0]: "train"}
    if n_subjects == 2:
        return {ordered[0]: "train", ordered[1]: "test"}

    train_n = max(1, int(round(n_subjects * train_ratio)))
    val_n = max(1, int(round(n_subjects * val_ratio)))
    if train_n + val_n >= n_subjects:
        overflow = train_n + val_n - (n_subjects - 1)
        train_n = max(1, train_n - overflow)
    test_n = n_subjects - train_n - val_n
    if test_n < 1:
        train_n = max(1, train_n - 1)
        test_n = n_subjects - train_n - val_n
    if test_n < 1:
        val_n = max(1, val_n - 1)
        test_n = n_subjects - train_n - val_n

    split_map: Dict[str, str] = {}
    for idx, subject_id in enumerate(ordered):
        if idx < train_n:
            split_map[subject_id] = "train"
        elif idx < train_n + val_n:
            split_map[subject_id] = "val"
        else:
            split_map[subject_id] = "test"
    return split_map


def _safe_relative(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except Exception:
        return str(path)


def _safe_gt(loader, path: str):
    try:
        return loader(path)
    except Exception:
        return None


def _scan_air_dataset(base_dir: Path, dataset_name: str, dataset_root: Path) -> List[Dict]:
    entries: List[Dict] = []
    gt_loader = load_air125_ground_truth if dataset_name == "AIR-125" else load_air400_ground_truth
    gt_source = "resp_waveform_hdf5" if dataset_name == "AIR-125" else "via_cycle_annotations"

    for subject_dir in sorted(dataset_root.glob("S*")):
        if not subject_dir.is_dir():
            continue
        for video_file in sorted(subject_dir.glob("*.mp4")):
            entries.append(
                {
                    "dataset": dataset_name,
                    "subject_id": subject_dir.name,
                    "video_id": video_file.stem,
                    "site_id": dataset_name,
                    "relative_path": _safe_relative(video_file, base_dir),
                    "full_path": str(video_file.resolve()),
                    "age_months": 6,
                    "age_source": "default_infant_prior",
                    "gt_rr_bpm": _safe_gt(gt_loader, str(video_file)),
                    "gt_source": gt_source,
                    "protocol_split": "",
                    "paper_scope": "respiratory_benchmark_rr_triage",
                }
            )
    return entries


def _scan_pediatric_rr(base_dir: Path) -> List[Dict]:
    entries: List[Dict] = []
    for video_file in iter_pediatric_rr_videos(base_dir):
        ped_gt = _safe_gt(load_pediatric_rr_ground_truth, str(video_file)) or {}
        subject_id = video_file.parent.name if video_file.parent.name.startswith("ID-") else video_file.stem
        site_id = video_file.parent.parent.name if video_file.parent.parent.name else "Pediatric-RR"
        entries.append(
            {
                "dataset": "Pediatric-RR",
                "subject_id": subject_id,
                "video_id": video_file.stem,
                "site_id": site_id,
                "relative_path": _safe_relative(video_file, base_dir),
                "full_path": str(video_file.resolve()),
                "age_months": int(ped_gt.get("age_months", 6)),
                "age_source": "stata_metadata" if ped_gt else "default_infant_prior",
                "gt_rr_bpm": ped_gt.get("respiratory_rate"),
                "gt_source": "stata_manual_count",
                "protocol_split": "",
                "paper_scope": "respiratory_benchmark_rr_triage",
            }
        )
    return entries


def _apply_protocol_splits(manifest: List[Dict], external_test_datasets: Iterable[str] = ()) -> List[Dict]:
    external = _external_dataset_set(external_test_datasets)

    # External-test datasets are excluded from the per-dataset subject split maps
    # so they never leak into the in-distribution train/val/test partitions.
    by_dataset_subjects = defaultdict(set)
    for entry in manifest:
        dataset_name = str(entry.get("dataset", ""))
        if dataset_name in external:
            continue
        by_dataset_subjects[dataset_name].add(str(entry.get("subject_id", "")))

    split_maps = {
        dataset_name: _subject_split_map(sorted(subject_ids), dataset_name)
        for dataset_name, subject_ids in by_dataset_subjects.items()
    }

    for entry in manifest:
        dataset_name = str(entry.get("dataset", ""))
        subject_id = str(entry.get("subject_id", ""))
        if dataset_name in external:
            entry["protocol_split"] = "external_test"
        else:
            entry["protocol_split"] = split_maps.get(dataset_name, {}).get(subject_id, "train")
    return manifest


def build_respiratory_benchmark_manifest(
    base_dir: str | Path = "data",
    external_test_datasets: Iterable[str] = DEFAULT_EXTERNAL_TEST_DATASETS,
) -> List[Dict]:
    base = Path(base_dir)
    manifest: List[Dict] = []

    air125 = base / "AIR_125"
    if air125.exists():
        manifest.extend(_scan_air_dataset(base, "AIR-125", air125))

    air400 = base / "AIR_400"
    if air400.exists():
        manifest.extend(_scan_air_dataset(base, "AIR-400", air400))

    manifest.extend(_scan_pediatric_rr(base))
    manifest = _apply_protocol_splits(manifest, external_test_datasets=external_test_datasets)

    manifest = sorted(
        manifest,
        key=lambda item: (
            str(item.get("dataset", "")),
            str(item.get("subject_id", "")),
            str(item.get("video_id", "")),
        ),
    )
    return manifest


def _write_json(path: Path, payload: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(payload), handle, indent=2)


def write_respiratory_benchmark_manifests(
    manifest: List[Dict],
    output_dir: str | Path = "data/organized",
    external_test_datasets: Iterable[str] = DEFAULT_EXTERNAL_TEST_DATASETS,
) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    manifest_path = out_dir / "respiratory_benchmark_manifest.json"
    _write_json(manifest_path, manifest)
    outputs["manifest"] = manifest_path

    split_map = defaultdict(list)
    for entry in manifest:
        split_map[str(entry.get("protocol_split", "train"))].append(entry)

    # In-distribution partitions (external-test datasets carry protocol_split
    # "external_test" and are therefore excluded from these three).
    for split_name in ["train", "val", "test"]:
        split_path = out_dir / f"respiratory_benchmark_{split_name}.json"
        _write_json(split_path, split_map.get(split_name, []))
        outputs[split_name] = split_path

    # Dedicated external out-of-distribution test manifest (e.g. Pediatric-RR-332).
    external_entries = split_map.get("external_test", [])
    external_path = out_dir / "respiratory_benchmark_external_test.json"
    _write_json(external_path, external_entries)
    outputs["external_test"] = external_path

    datasets = sorted({str(entry.get("dataset", "")) for entry in manifest})
    for dataset_name in datasets:
        slug = dataset_name.lower().replace(" ", "_").replace("/", "_")
        train_entries = [entry for entry in manifest if str(entry.get("dataset")) != dataset_name]
        test_entries = [entry for entry in manifest if str(entry.get("dataset")) == dataset_name]
        train_path = out_dir / f"respiratory_benchmark_lodo_{slug}_train.json"
        test_path = out_dir / f"respiratory_benchmark_lodo_{slug}_test.json"
        _write_json(train_path, train_entries)
        _write_json(test_path, test_entries)
        outputs[f"lodo_train_{slug}"] = train_path
        outputs[f"lodo_test_{slug}"] = test_path

    dataset_summary_path = out_dir / "respiratory_benchmark_dataset_summary.csv"
    split_summary_path = out_dir / "respiratory_benchmark_split_summary.csv"

    by_dataset = Counter(str(entry.get("dataset", "")) for entry in manifest)
    with open(dataset_summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "num_videos"])
        writer.writeheader()
        for dataset_name, count in sorted(by_dataset.items()):
            writer.writerow({"dataset": dataset_name, "num_videos": count})
    outputs["dataset_summary"] = dataset_summary_path

    split_counts = defaultdict(int)
    for entry in manifest:
        key = (str(entry.get("dataset", "")), str(entry.get("protocol_split", "")))
        split_counts[key] += 1
    with open(split_summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "num_videos"])
        writer.writeheader()
        for (dataset_name, split_name), count in sorted(split_counts.items()):
            writer.writerow({"dataset": dataset_name, "split": split_name, "num_videos": count})
    outputs["split_summary"] = split_summary_path

    return outputs
