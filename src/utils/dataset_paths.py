"""
Dataset path helpers shared across raw-data utilities.

The repository has historically referred to Pediatric-RR using two layouts:
1) data/Video Dataset for Respiratory Rate Counting in Children/...
2) data/datasets/Pediatric-RR-332/...

This module resolves those layouts into a single canonical interface so that
journal-scope manifests and legacy scripts do not silently diverge.
"""

from pathlib import Path
from typing import Iterator, Optional


# Canonical dataset name for the Pediatric-RR-332 cohort. This is the real-world
# Bangladeshi under-5 clinical corpus used as the external, out-of-distribution
# test set for the respiratory benchmark (see respiratory_benchmark_registry).
PEDIATRIC_RR_CANONICAL_NAME = "Pediatric-RR"


def pediatric_rr_available(base_dir: str | Path = "data") -> bool:
    """True when the Pediatric-RR-332 external cohort is present on disk."""
    return resolve_pediatric_rr_video_dir(base_dir=base_dir) is not None


def resolve_pediatric_rr_root(base_dir: str | Path = "data") -> Optional[Path]:
    """Return the existing Pediatric-RR root directory, if found."""
    base = Path(base_dir)
    candidates = [
        base / "Video Dataset for Respiratory Rate Counting in Children",
        base / "datasets" / "Pediatric-RR-332",
    ]
    for candidate in candidates:
        if (candidate / "Video Files").exists() or (candidate / "videos").exists():
            return candidate
    return None


def resolve_pediatric_rr_video_dir(base_dir: str | Path = "data") -> Optional[Path]:
    """Return the existing Pediatric-RR video directory across supported layouts."""
    root = resolve_pediatric_rr_root(base_dir=base_dir)
    if root is None:
        return None
    for candidate in [root / "Video Files", root / "videos"]:
        if candidate.exists():
            return candidate
    return None


def resolve_pediatric_rr_stata_path(base_dir: str | Path = "data") -> Optional[Path]:
    """Return the Pediatric-RR Stata annotation file when available."""
    root = resolve_pediatric_rr_root(base_dir=base_dir)
    if root is None:
        return None
    candidates = [
        root / "Stata File" / "Data Respiratory Rate Counting from Video Recording.dta",
        root / "metadata.dta",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def iter_pediatric_rr_videos(base_dir: str | Path = "data") -> Iterator[Path]:
    """
    Yield Pediatric-RR videos across both known layouts.

    Supported patterns:
    - root/Video Files/<site>/ID-1234/*.mp4
    - root/videos/*.mp4
    """
    video_dir = resolve_pediatric_rr_video_dir(base_dir=base_dir)
    if video_dir is None:
        return

    if video_dir.name.lower() == "videos":
        for video in sorted(video_dir.glob("*.mp4")):
            yield video
        return

    for site_dir in sorted(video_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        child_dirs = sorted(site_dir.glob("ID-*"))
        if child_dirs:
            for child_dir in child_dirs:
                if not child_dir.is_dir():
                    continue
                for video in sorted(child_dir.glob("*.mp4")):
                    yield video
            continue
        for video in sorted(site_dir.glob("*.mp4")):
            yield video
