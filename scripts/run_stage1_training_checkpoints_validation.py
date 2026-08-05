#!/usr/bin/env python3
"""Merge and validate Stage-1 EMA checkpoints on the repository testsets."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable
from urllib.parse import quote

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_lora_generator import merge_stage1_ema_checkpoint  # noqa: E402
from utils.config import DEFAULT_NEGATIVE_PROMPT, normalize_config  # noqa: E402
from utils.stage1_causal_validation import (  # noqa: E402
    discover_stage1_training_checkpoints,
    load_causal_testset_records,
    prepare_causal_testsets,
    validate_causal_testset_outputs,
    validate_converted_causal_base,
)
from utils.stage1_checkpoint import checkpoint_step, validate_checkpoint  # noqa: E402
from utils.stage1_io import atomic_write_bytes, atomic_write_json  # noqa: E402


REPORT_SCHEMA_VERSION = 1
NUM_FRAME_PER_BLOCK = 8


def _env(name: str) -> str | None:
    return os.environ.get(name)


def _required_env(parser: argparse.ArgumentParser, flag: str, env_name: str) -> None:
    value = _env(env_name)
    parser.add_argument(flag, default=value, required=value is None)


def _parse_steps(value: str | None) -> list[int] | None:
    if value is None:
        return None
    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise argparse.ArgumentTypeError("--steps must be a comma-separated list such as 75,300,750")
    try:
        steps = [int(piece) for piece in pieces]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--steps values must be integers") from exc
    if any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("--steps values must be positive")
    if len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("--steps must not contain duplicates")
    return steps


def _parse_num_latent_frames(value: str | int) -> int:
    try:
        num_latent_frames = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("--num-latent-frames must be an integer") from exc
    if num_latent_frames <= 0 or num_latent_frames % NUM_FRAME_PER_BLOCK:
        raise argparse.ArgumentTypeError(
            "--num-latent-frames must be positive and divisible by 8"
        )
    return num_latent_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="testsets/metadata_6cases_480x832.csv")
    parser.add_argument(
        "--num-latent-frames",
        type=_parse_num_latent_frames,
        default=24,
        help="Generated latent frames; must be positive and divisible by 8.",
    )
    parser.add_argument(
        "--allow-repeated-input-images",
        action="store_true",
        help="Allow multiple metadata rows to reuse the same validated input image.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--training-root",
        help=(
            "Directory containing checkpoint_model_XXXXXX directories. Defaults to "
            "LONG_LIVE_STAGE1_TRAIN_DIR when no explicit checkpoint is supplied."
        ),
    )
    selection.add_argument(
        "--training-checkpoint",
        action="append",
        help="Explicit checkpoint directory; repeat this flag to compare multiple checkpoints.",
    )
    parser.add_argument(
        "--steps",
        type=_parse_steps,
        help="Optional comma-separated optimizer steps selected from --training-root.",
    )
    parser.add_argument("--work-dir", required=True)
    _required_env(parser, "--base-checkpoint", "LONG_LIVE_STAGE1_BASE_CHECKPOINT")
    _required_env(parser, "--base-manifest", "LONG_LIVE_STAGE1_BASE_MANIFEST")
    parser.add_argument(
        "--source-checkpoint",
        default=_env("LONG_LIVE_STAGE1_SOURCE_CHECKPOINT"),
        help="Optional DiffSynth source path for full converted-base provenance audit.",
    )
    _required_env(parser, "--architecture-root", "LONG_LIVE_STAGE1_ARCHITECTURE_ROOT")
    _required_env(parser, "--t5-checkpoint", "LONG_LIVE_STAGE1_T5_CHECKPOINT")
    _required_env(parser, "--tokenizer-dir", "LONG_LIVE_STAGE1_TOKENIZER_DIR")
    _required_env(parser, "--vae-checkpoint", "LONG_LIVE_STAGE1_VAE_CHECKPOINT")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--merge-device", default="cpu")
    parser.add_argument("--minimum-first-frame-psnr-db", type=float, default=12.0)
    parser.add_argument("--minimum-frame-std", type=float, default=5.0)
    parser.add_argument("--minimum-temporal-abs-diff", type=float, default=0.05)
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Retain each full BF16 merged checkpoint after successful inference.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed checkpoint and continue validating later selected steps.",
    )
    parser.add_argument(
        "--skip-base-finite-check",
        action="store_true",
        help="Skip only the converted-base full BF16 finite scan; merge remains strict.",
    )
    return parser


def _resolve_checkpoints(args: argparse.Namespace, base_sha256: str) -> list[Path]:
    explicit = args.training_checkpoint
    if explicit:
        if args.steps is not None:
            raise ValueError("--steps is only valid with --training-root discovery")
        by_step: dict[int, Path] = {}
        for raw_path in explicit:
            path = Path(raw_path).expanduser().resolve()
            step = checkpoint_step(path)
            if step in by_step:
                raise ValueError(f"Stage-1 optimizer step {step} was selected more than once")
            validate_checkpoint(
                path,
                require_resumable=False,
                expected_base_sha256=base_sha256,
            )
            by_step[step] = path
        return [by_step[step] for step in sorted(by_step)]

    training_root = args.training_root or _env("LONG_LIVE_STAGE1_TRAIN_DIR")
    if not training_root:
        raise ValueError(
            "Provide --training-root/--training-checkpoint or set LONG_LIVE_STAGE1_TRAIN_DIR"
        )
    return discover_stage1_training_checkpoints(
        training_root,
        steps=args.steps,
        expected_base_sha256=base_sha256,
    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    atomic_write_json(path, report)


def _load_merge_config(checkpoint: Path, architecture_root: str):
    config_path = checkpoint / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = normalize_config(OmegaConf.load(config_path))
    if getattr(config, "model_kwargs", None) is None:
        raise ValueError(f"Resolved Stage-1 config has no model_kwargs: {config_path}")
    # Auxiliary/model files may be mounted at a different absolute path from
    # the training host.  This path-only override does not change architecture
    # or adapter semantics; init_weights=False keeps the converted base as the
    # sole source of generator weights.
    config.model_kwargs.architecture_root = os.fspath(
        Path(architecture_root).expanduser().resolve()
    )
    config.model_kwargs.init_weights = False
    return config


def _comparison_html(
    report: dict[str, Any],
    *,
    records,
    work_dir: Path,
) -> str:
    passed = [item for item in report.get("checkpoints", []) if item.get("status") == "pass"]
    title = "Stage-1 testsets checkpoint comparison"
    style = """
body{font-family:system-ui,sans-serif;margin:24px;color:#202124;background:#fafafa}
table{border-collapse:collapse;background:white}th,td{border:1px solid #ddd;padding:10px;vertical-align:top}
th{position:sticky;top:0;background:#f1f3f4;z-index:1}.case{min-width:240px;max-width:320px}
video{display:block;max-width:300px;max-height:300px;background:#111}.metrics{font-size:12px;color:#555;margin-top:6px}
.failed{color:#b3261e}code{font-size:12px}
"""
    lines = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        f"<title>{html.escape(title)}</title><style>{style}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<p>所有视频使用相同 testsets、seed 与采样参数。自动指标只做技术门禁；请人工比较动作、身份一致性和 block 边界。</p>",
        "<table><thead><tr><th>test case</th>",
    ]
    for item in passed:
        lines.append(f"<th>step {int(item['optimizer_step']):06d}<br>EMA</th>")
    lines.append("</tr></thead><tbody>")
    samples_by_step = {
        int(item["optimizer_step"]): {
            int(sample["row_id"]): sample for sample in item.get("samples", [])
        }
        for item in passed
    }
    for record in records:
        lines.append(
            "<tr><td class=\"case\"><strong>row "
            f"{record.row_id}</strong><br>{html.escape(record.bucket_id)}"
            f"<p>{html.escape(record.prompt)}</p></td>"
        )
        for item in passed:
            sample = samples_by_step[int(item["optimizer_step"])].get(record.row_id)
            if sample is None:
                lines.append('<td class="failed">missing output</td>')
                continue
            video_path = Path(sample["output_video"]).resolve()
            try:
                relative = video_path.relative_to(work_dir).as_posix()
            except ValueError:
                relative = os.fspath(video_path)
            source = quote(relative, safe="/._-")
            metrics = sample["metrics"]
            lines.append(
                "<td>"
                f'<video controls loop preload="metadata" src="{html.escape(source)}"></video>'
                '<div class="metrics">'
                f"PSNR {float(metrics['first_frame_psnr_db']):.2f} dB · "
                f"std {float(metrics['mean_frame_std']):.2f} · "
                f"temporal Δ {float(metrics['mean_temporal_abs_diff']):.3f}"
                "</div></td>"
            )
        lines.append("</tr>")
    lines.append("</tbody></table>")
    failed = [item for item in report.get("checkpoints", []) if item.get("status") == "failed"]
    if failed:
        lines.append('<h2 class="failed">Failed checkpoints</h2><ul>')
        for item in failed:
            lines.append(
                f"<li>step {int(item['optimizer_step']):06d}: "
                f"{html.escape(str(item.get('error', 'unknown error')))}</li>"
            )
        lines.append("</ul>")
    lines.append("</body></html>")
    return "".join(lines)


def run_validation(
    args: argparse.Namespace,
    *,
    base_validator: Callable[..., dict[str, Any]] = validate_converted_causal_base,
    merge_fn: Callable[..., dict[str, Any]] = merge_stage1_ema_checkpoint,
    prepare_fn: Callable[..., dict[str, Any]] = prepare_causal_testsets,
    output_validator: Callable[..., dict[str, Any]] = validate_causal_testset_outputs,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    try:
        num_latent_frames = _parse_num_latent_frames(
            getattr(args, "num_latent_frames", 24)
        )
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc
    allow_repeated_input_images = bool(
        getattr(args, "allow_repeated_input_images", False)
    )
    work_dir = Path(args.work_dir).expanduser().resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(
            f"Validation work directory must be empty to reject stale videos: {work_dir}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "validation_report.json"
    report: dict[str, Any] = {
        "schema": "longlive_stage1_training_checkpoint_validation",
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "initializing",
        "work_dir": os.fspath(work_dir),
        "checkpoints": [],
    }
    _write_report(report_path, report)

    try:
        base_report = base_validator(
            args.base_checkpoint,
            args.base_manifest,
            source_checkpoint=args.source_checkpoint,
            expected_num_frame_per_block=8,
            check_finite=not args.skip_base_finite_check,
        )
        base_sha256 = str(base_report["output_sha256"])
        checkpoints = _resolve_checkpoints(args, base_sha256)
        records = load_causal_testset_records(
            args.metadata,
            allow_repeated_input_images=allow_repeated_input_images,
        )
        report.update(
            {
                "status": "running",
                "base": base_report,
                "metadata": os.fspath(Path(args.metadata).expanduser().resolve()),
                "sampling": {
                    "solver": "unipc",
                    "sampling_steps": args.sampling_steps,
                    "guidance_scale": args.guidance_scale,
                    "seed": args.seed,
                    "negative_prompt": args.negative_prompt,
                },
                "checkpoint_count": len(checkpoints),
                "checkpoints": [
                    {
                        "optimizer_step": checkpoint_step(path),
                        "training_checkpoint": os.fspath(path),
                        "status": "pending",
                    }
                    for path in checkpoints
                ],
            }
        )
        _write_report(report_path, report)

        for checkpoint, item in zip(checkpoints, report["checkpoints"], strict=True):
            step = int(item["optimizer_step"])
            checkpoint_work_dir = work_dir / f"checkpoint_model_{step:06d}"
            checkpoint_work_dir.mkdir()
            merged_path = checkpoint_work_dir / "stage1_causal_ema_merged.pt"
            merge_manifest_path = checkpoint_work_dir / "merge_manifest.json"
            prepared_root = checkpoint_work_dir / "prepared"
            output_report_path = checkpoint_work_dir / "output_validation_report.json"
            try:
                item["status"] = "merging"
                _write_report(report_path, report)
                merge_manifest = merge_fn(
                    base_checkpoint=args.base_checkpoint,
                    training_checkpoint=checkpoint,
                    output_path=merged_path,
                    output_manifest_path=merge_manifest_path,
                    config=_load_merge_config(checkpoint, args.architecture_root),
                    device=args.merge_device,
                )
                item["merge_manifest"] = os.fspath(merge_manifest_path)
                item["merged_checkpoint_sha256"] = merge_manifest["output"]["sha256"]

                item["status"] = "preparing"
                _write_report(report_path, report)
                prepared = prepare_fn(
                    metadata_path=args.metadata,
                    output_root=prepared_root,
                    base_checkpoint=merged_path,
                    architecture_root=args.architecture_root,
                    t5_checkpoint=args.t5_checkpoint,
                    tokenizer_dir=args.tokenizer_dir,
                    vae_checkpoint=args.vae_checkpoint,
                    num_latent_frames=num_latent_frames,
                    num_frame_per_block=NUM_FRAME_PER_BLOCK,
                    minimum_source_frames=97,
                    sampling_steps=args.sampling_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed,
                    negative_prompt=args.negative_prompt,
                    allow_repeated_input_images=allow_repeated_input_images,
                )
                item["prepared_manifest"] = os.fspath(prepared_root / "prepared_manifest.json")

                item["status"] = "inferencing"
                for bucket in prepared["buckets"]:
                    item["active_bucket"] = bucket["bucket_id"]
                    _write_report(report_path, report)
                    command = [
                        sys.executable,
                        os.fspath(PROJECT_ROOT / "inference.py"),
                        "--config_path",
                        bucket["config_path"],
                    ]
                    print(
                        f"[stage1-checkpoint-validation] step={step:06d} "
                        f"bucket={bucket['bucket_id']}: {' '.join(command)}"
                    )
                    command_runner(command, cwd=PROJECT_ROOT, check=True)
                item.pop("active_bucket", None)

                item["status"] = "validating_outputs"
                _write_report(report_path, report)
                output_report = output_validator(
                    prepared_root / "prepared_manifest.json",
                    minimum_first_frame_psnr_db=args.minimum_first_frame_psnr_db,
                    minimum_frame_std=args.minimum_frame_std,
                    minimum_temporal_abs_diff=args.minimum_temporal_abs_diff,
                )
                atomic_write_json(output_report_path, output_report)
                item.update(
                    {
                        "status": "pass",
                        "output_report": os.fspath(output_report_path),
                        "sample_count": int(output_report["sample_count"]),
                        "samples": output_report["samples"],
                    }
                )
                if args.keep_merged:
                    item["merged_checkpoint"] = os.fspath(merged_path)
                    item["merged_checkpoint_retained"] = True
                else:
                    merged_path.unlink()
                    item["merged_checkpoint_retained"] = False
                _write_report(report_path, report)
            except Exception as exc:
                item.pop("active_bucket", None)
                item["status"] = "failed"
                item["error"] = f"{type(exc).__name__}: {exc}"
                item["merged_checkpoint_retained"] = merged_path.is_file()
                _write_report(report_path, report)
                if not args.continue_on_error:
                    raise

        failed_count = sum(item["status"] == "failed" for item in report["checkpoints"])
        passed_count = sum(item["status"] == "pass" for item in report["checkpoints"])
        report["passed_checkpoint_count"] = passed_count
        report["failed_checkpoint_count"] = failed_count
        report["status"] = "pass" if failed_count == 0 else "failed"
        comparison_path = work_dir / "comparison.html"
        comparison_html = _comparison_html(report, records=records, work_dir=work_dir)
        atomic_write_bytes(
            comparison_path,
            comparison_html.encode("utf-8"),
        )
        report["comparison_html"] = os.fspath(comparison_path)
        _write_report(report_path, report)
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(report_path, report)
        raise


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = run_validation(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
