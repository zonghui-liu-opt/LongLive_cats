#!/usr/bin/env python3
"""Validate all MP4 outputs produced from a prepared Stage-1 causal testset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage1_causal_validation import validate_causal_testset_outputs  # noqa: E402
from utils.stage1_io import atomic_write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--minimum-first-frame-psnr-db", type=float, default=12.0)
    parser.add_argument("--minimum-frame-std", type=float, default=5.0)
    parser.add_argument("--minimum-temporal-abs-diff", type=float, default=0.05)
    parser.add_argument("--report-path")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_causal_testset_outputs(
        args.prepared_manifest,
        minimum_first_frame_psnr_db=args.minimum_first_frame_psnr_db,
        minimum_frame_std=args.minimum_frame_std,
        minimum_temporal_abs_diff=args.minimum_temporal_abs_diff,
    )
    if args.report_path:
        atomic_write_json(args.report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
