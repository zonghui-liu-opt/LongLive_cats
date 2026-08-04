#!/usr/bin/env python3
"""Merge and validate Stage-1 EMA checkpoints on the repository testsets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
PROMPT_STYLE_ORDER = ("absolute_timeline", "sequential", "phase_relative")
PROMPT_STYLE_REVIEW_COLUMNS = (
    "input_image",
    "prompt",
    "height",
    "width",
    "bucket",
    "case_group",
    "prompt_style",
    "cat_id",
    "action_order",
)
PROMPT_STYLE_ACTION_ORDERS = {"jump_then_toy", "toy_then_jump"}


@dataclass(frozen=True)
class PromptStyleReviewRecord:
    row_id: int
    input_image: str
    prompt: str
    height: int
    width: int
    bucket: str
    case_group: str
    prompt_style: str
    cat_id: str
    action_order: str


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


def _load_prompt_style_review_records(
    metadata_path: str | os.PathLike[str],
) -> list[PromptStyleReviewRecord]:
    """Load and validate the review-only metadata used by prompt-style HTML."""

    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [name for name in PROMPT_STYLE_REVIEW_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(
                f"Prompt-style metadata is missing required columns {missing}: {metadata_path}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"Prompt-style metadata is empty: {metadata_path}")

    records: list[PromptStyleReviewRecord] = []
    grouped: dict[str, list[PromptStyleReviewRecord]] = {}
    for row_id, raw_row in enumerate(rows):
        if None in raw_row:
            raise ValueError(f"row {row_id}: prompt-style metadata has unexpected extra columns")
        row = {str(key): "" if value is None else str(value) for key, value in raw_row.items()}

        empty = [name for name in PROMPT_STYLE_REVIEW_COLUMNS if not row[name].strip()]
        if empty:
            raise ValueError(f"row {row_id}: prompt-style metadata has empty fields {empty}")
        try:
            height = int(row["height"])
            width = int(row["width"])
        except ValueError as exc:
            raise ValueError(f"row {row_id}: height/width must be integers") from exc
        if height <= 0 or width <= 0:
            raise ValueError(f"row {row_id}: height/width must be positive")

        prompt_style = row["prompt_style"].strip()
        if prompt_style not in PROMPT_STYLE_ORDER:
            raise ValueError(
                f"row {row_id}: unsupported prompt_style {prompt_style!r}; "
                f"expected one of {list(PROMPT_STYLE_ORDER)}"
            )
        action_order = row["action_order"].strip()
        if action_order not in PROMPT_STYLE_ACTION_ORDERS:
            raise ValueError(
                f"row {row_id}: unsupported action_order {action_order!r}; "
                f"expected one of {sorted(PROMPT_STYLE_ACTION_ORDERS)}"
            )
        cat_id = row["cat_id"].strip()
        case_group = row["case_group"].strip()
        expected_case_group = f"{cat_id}_{action_order}"
        if case_group != expected_case_group:
            raise ValueError(
                f"row {row_id}: case_group {case_group!r} must equal "
                f"{expected_case_group!r}"
            )

        image_path = Path(row["input_image"].strip()).expanduser()
        if not image_path.is_absolute():
            image_path = metadata_path.parent / image_path
        record = PromptStyleReviewRecord(
            row_id=row_id,
            input_image=os.fspath(image_path.resolve()),
            prompt=row["prompt"].strip(),
            height=height,
            width=width,
            bucket=row["bucket"].strip().lower(),
            case_group=case_group,
            prompt_style=prompt_style,
            cat_id=cat_id,
            action_order=action_order,
        )
        records.append(record)
        grouped.setdefault(case_group, []).append(record)

    expected_styles = set(PROMPT_STYLE_ORDER)
    for case_group, group_records in grouped.items():
        observed_styles = [record.prompt_style for record in group_records]
        if len(group_records) != len(PROMPT_STYLE_ORDER) or set(observed_styles) != expected_styles:
            raise ValueError(
                f"case_group {case_group!r} must contain exactly one row for each "
                f"prompt_style {list(PROMPT_STYLE_ORDER)}; got {observed_styles}"
            )
        group_contracts = {
            (
                record.input_image,
                record.height,
                record.width,
                record.bucket,
                record.cat_id,
                record.action_order,
            )
            for record in group_records
        }
        if len(group_contracts) != 1:
            raise ValueError(
                f"case_group {case_group!r} must use one input image, geometry, "
                "bucket, cat_id, and action_order"
            )
    return records


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
    parser.add_argument(
        "--comparison-mode",
        choices=("checkpoint", "prompt-style"),
        default="checkpoint",
        help="Render checkpoints as columns or compare two prompt styles per case group.",
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


def _prompt_style_comparison_html(
    report: dict[str, Any],
    *,
    review_records: list[PromptStyleReviewRecord],
    work_dir: Path,
) -> str:
    checkpoints = report.get("checkpoints", [])
    if len(checkpoints) != 1:
        raise ValueError("prompt-style comparison requires exactly one checkpoint")
    checkpoint = checkpoints[0]
    title = "Stage-1 two-action prompt-style comparison"
    style = """
body{font-family:system-ui,sans-serif;margin:24px;color:#202124;background:#fafafa}
table{border-collapse:collapse;background:white;width:100%}th,td{border:1px solid #ddd;padding:10px;vertical-align:top}
th{position:sticky;top:0;background:#f1f3f4;z-index:1}.case{min-width:210px;max-width:280px}
video{display:block;max-width:360px;max-height:360px;background:#111}.prompt{max-width:520px;white-space:pre-wrap}
.metrics{font-size:12px;color:#555;margin-top:6px}.failed{color:#b3261e}code{font-size:12px}
"""
    lines = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        f"<title>{html.escape(title)}</title><style>{style}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p>step {int(checkpoint['optimizer_step']):06d} EMA。自动指标只做技术门禁；视觉效果需人工判断。</p>",
    ]
    if checkpoint.get("status") != "pass":
        lines.append(
            '<p class="failed">checkpoint failed: '
            f"{html.escape(str(checkpoint.get('error', 'unknown error')))}</p></body></html>"
        )
        return "".join(lines)

    samples_by_row: dict[int, dict[str, Any]] = {}
    for sample in checkpoint.get("samples", []):
        row_id = int(sample["row_id"])
        if row_id in samples_by_row:
            raise RuntimeError(f"duplicate output row_id in prompt-style comparison: {row_id}")
        samples_by_row[row_id] = sample
    expected_row_ids = {record.row_id for record in review_records}
    if len(expected_row_ids) != len(review_records):
        raise RuntimeError("prompt-style review metadata contains duplicate row_id values")
    if set(samples_by_row) != expected_row_ids:
        raise RuntimeError(
            "prompt-style output row_id set mismatch: "
            f"missing={sorted(expected_row_ids - set(samples_by_row))}, "
            f"unexpected={sorted(set(samples_by_row) - expected_row_ids)}"
        )

    groups: dict[str, dict[str, PromptStyleReviewRecord]] = {}
    for record in review_records:
        groups.setdefault(record.case_group, {})[record.prompt_style] = record
    prompt_style_headers = "".join(
        f"<th>{html.escape(prompt_style)}</th>" for prompt_style in PROMPT_STYLE_ORDER
    )
    lines.append(
        "<table><thead><tr><th>case group</th>"
        f"{prompt_style_headers}</tr></thead><tbody>"
    )
    work_dir = work_dir.resolve()
    for case_group, styles in groups.items():
        first = styles[PROMPT_STYLE_ORDER[0]]
        lines.append(
            '<tr><td class="case"><strong>'
            f"{html.escape(case_group)}</strong><br>cat: {html.escape(first.cat_id)}"
            f"<br>action order: {html.escape(first.action_order)}</td>"
        )
        for prompt_style in PROMPT_STYLE_ORDER:
            record = styles[prompt_style]
            sample = samples_by_row[record.row_id]
            video_path = Path(sample["output_video"]).resolve()
            try:
                relative = video_path.relative_to(work_dir).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"prompt-style output video must be inside work dir: {video_path}"
                ) from exc
            source = quote(relative, safe="/._-")
            metrics = sample.get("metrics", {})
            metrics_html = ""
            if {
                "first_frame_psnr_db",
                "mean_frame_std",
                "mean_temporal_abs_diff",
            }.issubset(metrics):
                metrics_html = (
                    '<div class="metrics">'
                    f"PSNR {float(metrics['first_frame_psnr_db']):.2f} dB · "
                    f"std {float(metrics['mean_frame_std']):.2f} · "
                    f"temporal Δ {float(metrics['mean_temporal_abs_diff']):.3f}"
                    "</div>"
                )
            lines.append(
                "<td>"
                f"<strong>{html.escape(prompt_style)}</strong> · row {record.row_id}"
                f'<video controls loop preload="metadata" src="{html.escape(source)}"></video>'
                f'<p class="prompt">{html.escape(record.prompt)}</p>'
                f"{metrics_html}</td>"
            )
        lines.append("</tr>")
    lines.append("</tbody></table></body></html>")
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
    comparison_mode = getattr(args, "comparison_mode", "checkpoint")
    if comparison_mode not in {"checkpoint", "prompt-style"}:
        raise ValueError(f"unsupported comparison mode: {comparison_mode!r}")

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
        if comparison_mode == "prompt-style" and len(checkpoints) != 1:
            raise ValueError("prompt-style comparison requires exactly one checkpoint")
        records = load_causal_testset_records(
            args.metadata,
            allow_repeated_input_images=allow_repeated_input_images,
        )
        review_records = (
            _load_prompt_style_review_records(args.metadata)
            if comparison_mode == "prompt-style"
            else None
        )
        if review_records is not None:
            if len(review_records) != len(records):
                raise RuntimeError(
                    "prompt-style review metadata row count differs from causal metadata"
                )
            for record, review_record in zip(records, review_records, strict=True):
                if (
                    record.row_id != review_record.row_id
                    or record.input_image != review_record.input_image
                    or record.prompt != review_record.prompt
                    or record.height != review_record.height
                    or record.width != review_record.width
                    or record.bucket != review_record.bucket
                ):
                    raise RuntimeError(
                        f"prompt-style review row {review_record.row_id} differs from causal metadata"
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
        if comparison_mode == "prompt-style":
            assert review_records is not None
            comparison_html = _prompt_style_comparison_html(
                report,
                review_records=review_records,
                work_dir=work_dir,
            )
        else:
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
