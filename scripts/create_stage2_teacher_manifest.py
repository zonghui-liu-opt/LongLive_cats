#!/usr/bin/env python3
"""Create a strict provenance sidecar for the Stage-2 real-score teacher."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage1_io import atomic_write_json  # noqa: E402
from utils.stage2_role_manifest import build_stage2_teacher_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architecture-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint-format",
        required=True,
        choices=("longlive_wrapper_pt", "wan_native_transformer"),
    )
    parser.add_argument(
        "--state-dict-selector",
        required=True,
        choices=("generator", "real_score", "model", "root"),
    )
    parser.add_argument("--source-kind", required=True)
    parser.add_argument("--source-identifier", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument(
        "--conversion-command",
        action="append",
        required=True,
        help="Repeat for each argv item; use one value 'none' for a direct checkpoint.",
    )
    parser.add_argument(
        "--attest-cat-domain-bidirectional-ti2v",
        action="store_true",
        help="Operator attests this is the frozen cat-domain bidirectional TI2V teacher.",
    )
    parser.add_argument(
        "--attest-video-global-flow",
        action="store_true",
        help="Operator attests the teacher uses video-global flow prediction.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = build_stage2_teacher_manifest(
        checkpoint_path=args.checkpoint,
        architecture_root=args.architecture_root,
        checkpoint_format=args.checkpoint_format,
        state_dict_selector=args.state_dict_selector,
        source_kind=args.source_kind,
        source_identifier=args.source_identifier,
        source_sha256=args.source_sha256,
        conversion_command=tuple(args.conversion_command),
        attest_cat_domain_bidirectional_ti2v=(
            args.attest_cat_domain_bidirectional_ti2v
        ),
        attest_video_global_flow=args.attest_video_global_flow,
    )
    atomic_write_json(output, manifest)
    print(f"Wrote {output}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
