#!/usr/bin/env python3
"""Run strict Stage-2 Generator-EMA batch inference under torchrun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage2_inference_config import (  # noqa: E402 - isolated CLI bootstrap
    load_stage2_inference_config,
)
from utils.stage2_inference_runtime import (  # noqa: E402 - isolated CLI bootstrap
    run_stage2_inference,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "infer_i2v_stage2_baseline.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the strict Stage-2 single/two-action EMA inference matrix. "
            "Launch with torchrun; profiles are selected only by the strict config."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=(
            "Stage-2 inference YAML. Any canonical rollout profile is accepted "
            "by the strict resolver; the release shell uses the baseline config."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_stage2_inference_config(args.config)
    result = run_stage2_inference(config)
    if result["rank"] == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
