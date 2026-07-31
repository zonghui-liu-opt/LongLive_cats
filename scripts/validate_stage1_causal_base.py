#!/usr/bin/env python3
"""Audit a converted Stage-1 causal base against its manifest and source."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage1_causal_validation import validate_converted_causal_base  # noqa: E402
from utils.stage1_io import atomic_write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint",
        default=os.environ.get("LONG_LIVE_STAGE1_BASE_CHECKPOINT"),
        required="LONG_LIVE_STAGE1_BASE_CHECKPOINT" not in os.environ,
    )
    parser.add_argument(
        "--base-manifest",
        default=os.environ.get("LONG_LIVE_STAGE1_BASE_MANIFEST"),
        required="LONG_LIVE_STAGE1_BASE_MANIFEST" not in os.environ,
    )
    parser.add_argument(
        "--source-checkpoint",
        default=os.environ.get("LONG_LIVE_STAGE1_SOURCE_CHECKPOINT"),
        help="When supplied, re-hash every current DiffSynth source shard.",
    )
    parser.add_argument("--num-frame-per-block", type=int, default=8)
    parser.add_argument("--skip-finite-check", action="store_true")
    parser.add_argument("--report-path")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_converted_causal_base(
        args.base_checkpoint,
        args.base_manifest,
        source_checkpoint=args.source_checkpoint,
        expected_num_frame_per_block=args.num_frame_per_block,
        check_finite=not args.skip_finite_check,
    )
    if args.report_path:
        atomic_write_json(args.report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
