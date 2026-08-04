#!/usr/bin/env python3
"""Merge step-3750 EMA and run the fixed Stage-1 continuation experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_lora_generator import merge_stage1_ema_checkpoint  # noqa: E402
from scripts.run_stage1_training_checkpoints_validation import (  # noqa: E402
    _env,
    _load_merge_config,
    _required_env,
)
from utils.stage1_causal_validation import (  # noqa: E402
    validate_converted_causal_base,
)
from utils.stage1_checkpoint import checkpoint_step, validate_checkpoint  # noqa: E402
from utils.stage1_continuation_report import (  # noqa: E402
    write_continuation_comparison_html,
)
from utils.stage1_continuation_validation import (  # noqa: E402
    load_continuation_metadata,
    prepare_continuation_inference,
    validate_continuation_outputs,
)
from utils.stage1_io import atomic_write_json  # noqa: E402

REPORT_SCHEMA = "longlive_stage1_continuation_validation"
REPORT_SCHEMA_VERSION = 1
LOCKED_CHECKPOINT_STEP = 3750
FORMAL_METADATA = (
    PROJECT_ROOT
    / "testsets"
    / "metadata_8cases_two_actions_continuation_480x832_253frames.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed Stage-1 step-3750 A/HOLD/B continuation matrix "
            "for sink=0 and sink=1."
        )
    )
    parser.add_argument(
        "--training-checkpoint",
        help=(
            "Exact checkpoint_model_003750 directory. When omitted, resolves "
            "it under LONG_LIVE_STAGE1_TRAIN_DIR."
        ),
    )
    parser.add_argument("--metadata", default=os.fspath(FORMAL_METADATA))
    parser.add_argument("--work-dir", required=True)
    _required_env(parser, "--base-checkpoint", "LONG_LIVE_STAGE1_BASE_CHECKPOINT")
    _required_env(parser, "--base-manifest", "LONG_LIVE_STAGE1_BASE_MANIFEST")
    parser.add_argument(
        "--source-checkpoint",
        default=_env("LONG_LIVE_STAGE1_SOURCE_CHECKPOINT"),
        help="Optional source checkpoint for converted-base provenance validation.",
    )
    _required_env(parser, "--architecture-root", "LONG_LIVE_STAGE1_ARCHITECTURE_ROOT")
    _required_env(parser, "--t5-checkpoint", "LONG_LIVE_STAGE1_T5_CHECKPOINT")
    _required_env(parser, "--tokenizer-dir", "LONG_LIVE_STAGE1_TOKENIZER_DIR")
    _required_env(parser, "--vae-checkpoint", "LONG_LIVE_STAGE1_VAE_CHECKPOINT")
    parser.add_argument("--merge-device", default="cpu")
    parser.add_argument("--minimum-first-frame-psnr-db", type=float, default=12.0)
    parser.add_argument("--minimum-frame-std", type=float, default=5.0)
    parser.add_argument("--minimum-temporal-abs-diff", type=float, default=0.05)
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Retain the reconstructable full BF16 merged checkpoint after success.",
    )
    parser.add_argument(
        "--skip-base-finite-check",
        action="store_true",
        help="Skip only the converted-base full BF16 finite scan.",
    )
    return parser


def _resolve_training_checkpoint(args: argparse.Namespace, *, base_sha256: str) -> Path:
    raw_checkpoint = args.training_checkpoint
    if not raw_checkpoint:
        training_root = _env("LONG_LIVE_STAGE1_TRAIN_DIR")
        if not training_root:
            raise ValueError(
                "Provide --training-checkpoint or set LONG_LIVE_STAGE1_TRAIN_DIR"
            )
        raw_checkpoint = os.fspath(
            Path(training_root).expanduser()
            / f"checkpoint_model_{LOCKED_CHECKPOINT_STEP:06d}"
        )
    checkpoint = Path(raw_checkpoint).expanduser().resolve()
    step = checkpoint_step(checkpoint)
    if step != LOCKED_CHECKPOINT_STEP:
        raise ValueError(
            f"continuation experiment requires optimizer step {LOCKED_CHECKPOINT_STEP}, "
            f"got {step}"
        )
    validate_checkpoint(
        checkpoint,
        require_resumable=False,
        expected_base_sha256=base_sha256,
    )
    return checkpoint


def run_validation(
    args: argparse.Namespace,
    *,
    base_validator: Callable[..., dict[str, Any]] = validate_converted_causal_base,
    merge_fn: Callable[..., dict[str, Any]] = merge_stage1_ema_checkpoint,
    prepare_fn: Callable[..., dict[str, Any]] = prepare_continuation_inference,
    output_validator: Callable[..., dict[str, Any]] = validate_continuation_outputs,
    html_writer: Callable[..., Path] = write_continuation_comparison_html,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    work_dir = Path(args.work_dir).expanduser().resolve()
    if work_dir.exists() and not work_dir.is_dir():
        raise FileExistsError(f"work path exists but is not a directory: {work_dir}")
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"Continuation work directory must be empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "validation_report.json"
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "initializing",
        "work_dir": os.fspath(work_dir),
        "checkpoint": None,
    }
    atomic_write_json(report_path, report)

    checkpoint_root: Path | None = None
    merged_path: Path | None = None
    try:
        metadata_path = Path(args.metadata).expanduser().resolve()
        records = load_continuation_metadata(metadata_path)
        base_report = base_validator(
            args.base_checkpoint,
            args.base_manifest,
            source_checkpoint=args.source_checkpoint,
            expected_num_frame_per_block=8,
            check_finite=not args.skip_base_finite_check,
        )
        checkpoint = _resolve_training_checkpoint(
            args,
            base_sha256=str(base_report["output_sha256"]),
        )
        step = checkpoint_step(checkpoint)
        checkpoint_root = work_dir / f"checkpoint_model_{step:06d}"
        checkpoint_root.mkdir()
        merged_path = checkpoint_root / "stage1_causal_ema_merged.pt"
        merge_manifest_path = checkpoint_root / "merge_manifest.json"
        prepared_root = checkpoint_root / "prepared_continuation"
        video_root = checkpoint_root / "continuation"
        output_report_path = checkpoint_root / "output_validation_report.json"

        checkpoint_report: dict[str, Any] = {
            "optimizer_step": step,
            "training_checkpoint": os.fspath(checkpoint),
            "status": "merging",
            "merged_checkpoint_retained": False,
        }
        report.update(
            {
                "status": "running",
                "base": base_report,
                "metadata": os.fspath(metadata_path),
                "sampling": {
                    "solver": "unipc",
                    "sampling_steps": 50,
                    "guidance_scale": 5.0,
                    "seed": 1,
                    "sink_sizes": [0, 1],
                },
                "checkpoint": checkpoint_report,
            }
        )
        atomic_write_json(report_path, report)

        merge_manifest = merge_fn(
            base_checkpoint=args.base_checkpoint,
            training_checkpoint=checkpoint,
            output_path=merged_path,
            output_manifest_path=merge_manifest_path,
            config=_load_merge_config(checkpoint, args.architecture_root),
            device=args.merge_device,
        )
        checkpoint_report.update(
            {
                "merge_manifest": os.fspath(merge_manifest_path),
                "merged_checkpoint_sha256": merge_manifest["output"]["sha256"],
                "merged_checkpoint_retained": True,
                "status": "preparing",
            }
        )
        atomic_write_json(report_path, report)

        prepared = prepare_fn(
            metadata_path=metadata_path,
            output_root=prepared_root,
            video_root=video_root,
            base_checkpoint=merged_path,
            architecture_root=args.architecture_root,
            t5_checkpoint=args.t5_checkpoint,
            tokenizer_dir=args.tokenizer_dir,
            vae_checkpoint=args.vae_checkpoint,
            sampling_steps=50,
            guidance_scale=5.0,
            seed=1,
        )
        prepared_manifest_path = prepared_root / "prepared_manifest.json"
        checkpoint_report["prepared_manifest"] = os.fspath(prepared_manifest_path)
        checkpoint_report["status"] = "inferencing"
        atomic_write_json(report_path, report)

        for bucket in prepared["buckets"]:
            checkpoint_report["active_bucket"] = bucket["bucket_id"]
            atomic_write_json(report_path, report)
            command = [
                sys.executable,
                os.fspath(PROJECT_ROOT / "inference.py"),
                "--config_path",
                bucket["config_path"],
            ]
            print(
                f"[stage1-continuation-validation] bucket={bucket['bucket_id']}: "
                f"{' '.join(command)}"
            )
            command_runner(command, cwd=PROJECT_ROOT, check=True)
        checkpoint_report.pop("active_bucket", None)
        checkpoint_report["status"] = "validating_outputs"
        atomic_write_json(report_path, report)

        output_report = output_validator(
            prepared_manifest_path,
            minimum_first_frame_psnr_db=args.minimum_first_frame_psnr_db,
            minimum_frame_std=args.minimum_frame_std,
            minimum_temporal_abs_diff=args.minimum_temporal_abs_diff,
        )
        atomic_write_json(output_report_path, output_report)
        comparison_path = work_dir / "comparison.html"
        html_writer(
            comparison_path,
            records,
            output_report["samples"],
            work_dir=work_dir,
        )

        checkpoint_report.update(
            {
                "status": "pass",
                "output_report": os.fspath(output_report_path),
                "sample_count": int(output_report["sample_count"]),
                "samples": output_report["samples"],
                "merged_checkpoint_retained": bool(args.keep_merged),
            }
        )
        report.update(
            {
                "status": "pass",
                "comparison_html": os.fspath(comparison_path),
                "sample_count": int(output_report["sample_count"]),
            }
        )
        atomic_write_json(report_path, report)
        if not args.keep_merged:
            merged_path.unlink()
        return report
    except Exception as exc:
        checkpoint_report = report.get("checkpoint")
        if isinstance(checkpoint_report, dict):
            checkpoint_report.pop("active_bucket", None)
            checkpoint_report["status"] = "failed"
            checkpoint_report["error"] = f"{type(exc).__name__}: {exc}"
            checkpoint_report["merged_checkpoint_retained"] = bool(
                merged_path is not None and merged_path.is_file()
            )
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(report_path, report)
        raise


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = run_validation(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
