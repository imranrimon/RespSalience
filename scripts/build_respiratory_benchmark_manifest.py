"""Build canonical manifests for the paper-ready respiratory benchmark."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.respiratory_benchmark_registry import (
    DEFAULT_EXTERNAL_TEST_DATASETS,
    build_respiratory_benchmark_manifest,
    write_respiratory_benchmark_manifests,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build respiratory benchmark manifests for RR + safe tachypnea triage.")
    parser.add_argument("--base-dir", default="data", help="Dataset root directory")
    parser.add_argument("--output-dir", default="data/organized", help="Output directory for manifest JSON/CSV files")
    parser.add_argument(
        "--external-test-datasets",
        nargs="*",
        default=list(DEFAULT_EXTERNAL_TEST_DATASETS),
        help="Datasets held out entirely as the external OOD test set "
        "(default: Pediatric-RR). Written to respiratory_benchmark_external_test.json "
        "and excluded from train/val/test.",
    )
    parser.add_argument(
        "--pool-external",
        action="store_true",
        help="Disable external hold-out and pool every dataset into train/val/test (legacy behavior).",
    )
    args = parser.parse_args()

    external = () if args.pool_external else tuple(args.external_test_datasets)

    manifest = build_respiratory_benchmark_manifest(base_dir=args.base_dir, external_test_datasets=external)
    outputs = write_respiratory_benchmark_manifests(
        manifest, output_dir=args.output_dir, external_test_datasets=external
    )

    external_count = sum(1 for entry in manifest if str(entry.get("protocol_split")) == "external_test")
    print("[Respiratory Benchmark Manifest]")
    print(f"  Entries: {len(manifest)}")
    print(f"  External hold-out: {list(external) if external else 'none (pooled)'} "
          f"({external_count} external-test videos)")
    for key, path in sorted(outputs.items()):
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
