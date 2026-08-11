#!/usr/bin/env python3
"""Infer a pre-merged Stage-1 checkpoint and pair it with a reference run.

The reference is a ``checkpoint_model_XXXXXX`` directory produced by
``run_stage1_training_checkpoints_validation.py``.  Its prepared carrier
videos and inference configs are reused so that the generator checkpoint and
output directory are the only intentional config changes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable
from urllib.parse import quote

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import normalize_config  # noqa: E402
from utils.stage1_causal_validation import (  # noqa: E402
    CAUSAL_TESTSET_SCHEMA_VERSION,
    load_causal_testset_records,
    probe_video,
    validate_causal_testset_outputs,
)
from utils.stage1_io import (  # noqa: E402
    atomic_output_path,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)

REPORT_SCHEMA_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-checkpoint", required=True)
    parser.add_argument(
        "--reference-checkpoint-dir",
        required=True,
        help=(
            "Existing checkpoint_model_XXXXXX result directory created by "
            "run_stage1_training_checkpoints_validation.py."
        ),
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument(
        "--work-dir",
        required=True,
        help="A new or empty directory. Existing results are never overwritten.",
    )
    parser.add_argument("--minimum-first-frame-psnr-db", type=float, default=12.0)
    parser.add_argument("--minimum-frame-std", type=float, default=5.0)
    parser.add_argument("--minimum-temporal-abs-diff", type=float, default=0.05)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _assert_reference_manifest_integrity(manifest: dict[str, Any]) -> None:
    if int(manifest.get("schema_version", -1)) != CAUSAL_TESTSET_SCHEMA_VERSION:
        raise RuntimeError("Unsupported reference prepared-manifest schema version")
    recorded_hash = manifest.get("manifest_sha256")
    unhashed = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    actual_hash = canonical_json_sha256(unhashed)
    if recorded_hash != actual_hash:
        raise RuntimeError(
            "Reference prepared manifest does not match its recorded SHA256"
        )


def _assert_metadata_matches_reference(
    metadata_path: Path,
    reference_manifest: dict[str, Any],
) -> list[Any]:
    metadata = reference_manifest.get("metadata", {})
    actual_sha256 = sha256_file(metadata_path)
    if metadata.get("sha256") != actual_sha256:
        raise RuntimeError(
            "Requested metadata CSV differs from the reference inference metadata: "
            f"expected {metadata.get('sha256')}, got {actual_sha256}"
        )
    records = load_causal_testset_records(metadata_path)
    if int(metadata.get("record_count", -1)) != len(records):
        raise RuntimeError(
            "Reference metadata record count differs from the current CSV"
        )

    current_by_row = {record.row_id: record for record in records}
    reference_by_row: dict[int, dict[str, Any]] = {}
    for bucket in reference_manifest.get("buckets", []):
        for record in bucket.get("records", []):
            row_id = int(record["row_id"])
            if row_id in reference_by_row:
                raise RuntimeError(f"Reference row {row_id} appears more than once")
            reference_by_row[row_id] = record
    if set(reference_by_row) != set(current_by_row):
        raise RuntimeError(
            "Reference rows differ from the requested metadata rows: "
            f"reference={sorted(reference_by_row)}, current={sorted(current_by_row)}"
        )
    for row_id, record in current_by_row.items():
        reference = reference_by_row[row_id]
        expected = {
            "row_sha256": record.row_sha256,
            "image_sha256": record.image_sha256,
            "height": record.height,
            "width": record.width,
            "bucket_id": record.bucket_id,
        }
        wrong = {
            key: {"expected": value, "actual": reference.get(key)}
            for key, value in expected.items()
            if reference.get(key) != value
        }
        if wrong:
            raise RuntimeError(f"Reference metadata row {row_id} mismatch: {wrong}")
    return records


def _assert_reference_config_contract(
    config_path: Path,
    *,
    bucket: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    config = normalize_config(OmegaConf.load(config_path))
    if getattr(config, "adapter", None) is not None:
        raise RuntimeError(
            f"Reference config unexpectedly enables a LoRA adapter: {config_path}"
        )
    if getattr(config, "lora_ckpt", None):
        raise RuntimeError(
            f"Reference config unexpectedly loads LoRA weights: {config_path}"
        )
    if bool(getattr(config, "use_ema", False)):
        raise RuntimeError(
            f"Reference config must load a full merged generator: {config_path}"
        )
    if int(getattr(config, "num_samples", -1)) != 1:
        raise RuntimeError(f"Reference config num_samples must be 1: {config_path}")
    if not bool(getattr(config, "save_with_index", False)):
        raise RuntimeError(
            f"Reference config must use indexed filenames: {config_path}"
        )

    sampling = manifest["sampling"]
    frame_policy = manifest["frame_policy"]
    expected = {
        "sampling_steps": int(sampling["sampling_steps"]),
        "guidance_scale": float(sampling["guidance_scale"]),
        "seed": int(sampling["seed"]),
        "negative_prompt": str(sampling["negative_prompt"]),
        "num_output_frames": int(frame_policy["num_latent_frames"]),
        "data_path": os.fspath(Path(bucket["data_root"]).resolve()),
    }
    actual = {
        "sampling_steps": int(config.sampling_steps),
        "guidance_scale": float(config.guidance_scale),
        "seed": int(config.seed),
        "negative_prompt": str(config.negative_prompt),
        "num_output_frames": int(config.num_output_frames),
        "data_path": os.fspath(Path(config.data_path).resolve()),
    }
    wrong = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    if wrong:
        raise RuntimeError(f"Reference inference config/manifest mismatch: {wrong}")


def clone_reference_preparation(
    *,
    reference_manifest_path: str | os.PathLike[str],
    merged_checkpoint: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Clone reference configs while changing only checkpoint/output paths."""

    reference_manifest_path = Path(reference_manifest_path).expanduser().resolve()
    merged_checkpoint = Path(merged_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not reference_manifest_path.is_file():
        raise FileNotFoundError(reference_manifest_path)
    if not merged_checkpoint.is_file():
        raise FileNotFoundError(merged_checkpoint)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Merged inference directory must be empty: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    reference = _load_json(reference_manifest_path)
    _assert_reference_manifest_integrity(reference)
    cloned = deepcopy(reference)
    cloned.pop("manifest_sha256", None)
    cloned["reference_prepared_manifest"] = os.fspath(reference_manifest_path)
    cloned["model_paths"]["base_checkpoint"] = os.fspath(merged_checkpoint)

    for bucket in cloned["buckets"]:
        bucket_id = str(bucket["bucket_id"])
        source_config_path = Path(bucket["config_path"]).expanduser().resolve()
        data_root = Path(bucket["data_root"]).expanduser().resolve()
        if not source_config_path.is_file():
            raise FileNotFoundError(source_config_path)
        if not data_root.is_dir():
            raise FileNotFoundError(data_root)
        for record in bucket["records"]:
            for key in ("carrier_video", "caption_json"):
                artifact = Path(record[key]).expanduser().resolve()
                if not artifact.is_file():
                    raise FileNotFoundError(artifact)

        _assert_reference_config_contract(
            source_config_path,
            bucket=bucket,
            manifest=reference,
        )
        config = OmegaConf.load(source_config_path)
        if config.get("adapter", None) is not None:
            raise RuntimeError(
                f"Refusing to clone adapter-enabled config: {source_config_path}"
            )
        if config.get("checkpoints", None) is None:
            config.checkpoints = OmegaConf.create({})
        config.checkpoints.generator_ckpt = os.fspath(merged_checkpoint)
        config.checkpoints.pop("lora_ckpt", None)
        config.generator_ckpt = os.fspath(merged_checkpoint)
        config.pop("lora_ckpt", None)
        config.use_ema = False
        config.num_samples = 1
        config.save_with_index = True

        output_dir = output_root / "videos" / bucket_id
        output_dir.mkdir(parents=True, exist_ok=True)
        config.output_folder = os.fspath(output_dir)
        if config.get("inference", None) is not None:
            config.inference.output_folder = os.fspath(output_dir)
            config.inference.use_ema = False

        config_path = output_root / "configs" / f"{bucket_id}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_text = OmegaConf.to_yaml(config, resolve=False, sort_keys=False)
        atomic_write_bytes(config_path, config_text.encode("utf-8"))
        bucket["reference_config_path"] = os.fspath(source_config_path)
        bucket["config_path"] = os.fspath(config_path)
        bucket["output_dir"] = os.fspath(output_dir)

    cloned["manifest_sha256"] = canonical_json_sha256(cloned)
    atomic_write_json(output_root / "prepared_manifest.json", cloned)
    return cloned


def write_side_by_side_video(
    left_video: str | os.PathLike[str],
    right_video: str | os.PathLike[str],
    output_video: str | os.PathLike[str],
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    video_probe: Callable[[str | os.PathLike[str]], dict[str, Any]] = probe_video,
) -> dict[str, Any]:
    """Write one MP4 with reference on the left and merged output on the right."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create side-by-side videos")
    left_video = Path(left_video).expanduser().resolve()
    right_video = Path(right_video).expanduser().resolve()
    output_video = Path(output_video).expanduser().resolve()
    left = video_probe(left_video)
    right = video_probe(right_video)
    required_equal = ("width", "height", "frame_count", "fps")
    mismatch = {
        key: {"left": left[key], "right": right[key]}
        for key in required_equal
        if left[key] != right[key]
    }
    if mismatch:
        raise RuntimeError(
            f"Cannot pair videos with different stream properties: {mismatch}"
        )

    with atomic_output_path(output_video, suffix=".mp4") as temporary:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            os.fspath(left_video),
            "-i",
            os.fspath(right_video),
            "-filter_complex",
            (
                "[0:v]setpts=PTS-STARTPTS,setsar=1[left];"
                "[1:v]setpts=PTS-STARTPTS,setsar=1[right];"
                "[left][right]hstack=inputs=2:shortest=1[paired]"
            ),
            "-map",
            "[paired]",
            "-frames:v",
            str(int(left["frame_count"])),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            os.fspath(temporary),
        ]
        command_runner(command, check=True)

    paired = video_probe(output_video)
    expected = {
        "width": 2 * int(left["width"]),
        "height": int(left["height"]),
        "frame_count": int(left["frame_count"]),
        "fps": float(left["fps"]),
    }
    wrong = {
        key: {"expected": value, "actual": paired[key]}
        for key, value in expected.items()
        if paired[key] != value
    }
    if wrong:
        raise RuntimeError(f"Side-by-side output stream mismatch: {wrong}")
    return paired


def _comparison_html(
    *,
    samples: list[dict[str, Any]],
    work_dir: Path,
) -> str:
    style = """
body{font-family:system-ui,sans-serif;margin:24px;color:#202124;background:#fafafa}
.case{margin:0 0 28px;padding:16px;background:#fff;border:1px solid #ddd;border-radius:8px}
video{display:block;max-width:100%;max-height:640px;background:#111}.meta{font-size:13px;color:#555}
"""
    lines = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        f"<title>Stage-1 merged checkpoint comparison</title><style>{style}</style></head><body>",
        "<h1>Stage-1 merged checkpoint comparison</h1>",
        "<p><strong>左侧：</strong>原 infer_stage1 结果；<strong>右侧：</strong>当前 merged checkpoint 结果。</p>",
    ]
    for sample in samples:
        video_path = Path(sample["comparison_video"]).resolve()
        relative = video_path.relative_to(work_dir).as_posix()
        source = quote(relative, safe="/._-")
        lines.extend(
            [
                '<section class="case">',
                f"<h2>row {int(sample['row_id'])} · {html.escape(sample['bucket_id'])}</h2>",
                f'<video controls loop preload="metadata" src="{html.escape(source)}"></video>',
                '<p class="meta">left = infer_stage1 reference · right = merged checkpoint</p>',
                "</section>",
            ]
        )
    lines.append("</body></html>")
    return "".join(lines)


def run_comparison(
    args: argparse.Namespace,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    output_validator: Callable[..., dict[str, Any]] = validate_causal_testset_outputs,
    pair_writer: Callable[..., dict[str, Any]] = write_side_by_side_video,
) -> dict[str, Any]:
    merged_checkpoint = Path(args.merged_checkpoint).expanduser().resolve()
    reference_dir = Path(args.reference_checkpoint_dir).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    reference_manifest_path = reference_dir / "prepared" / "prepared_manifest.json"
    merged_inference_root = work_dir / "merged_inference"
    report_path = work_dir / "comparison_report.json"

    if not merged_checkpoint.is_file():
        raise FileNotFoundError(merged_checkpoint)
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if work_dir == reference_dir or work_dir.is_relative_to(reference_dir):
        raise ValueError(
            "Work directory must be outside the immutable reference result"
        )
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(
            f"Comparison work directory must be empty to reject stale outputs: {work_dir}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "longlive_stage1_premerged_reference_comparison",
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "initializing",
        "work_dir": os.fspath(work_dir),
        "metadata": os.fspath(metadata_path),
        "merged_checkpoint": os.fspath(merged_checkpoint),
        "reference_checkpoint_dir": os.fspath(reference_dir),
        "side_by_side_order": {
            "left": "infer_stage1_reference",
            "right": "premerged_checkpoint",
        },
    }
    atomic_write_json(report_path, report)

    try:
        reference_manifest = _load_json(reference_manifest_path)
        _assert_reference_manifest_integrity(reference_manifest)
        records = _assert_metadata_matches_reference(
            metadata_path,
            reference_manifest,
        )
        report["sampling"] = reference_manifest["sampling"]
        report["frame_policy"] = reference_manifest["frame_policy"]
        report["status"] = "validating_reference"
        atomic_write_json(report_path, report)

        validator_kwargs = {
            "minimum_first_frame_psnr_db": args.minimum_first_frame_psnr_db,
            "minimum_frame_std": args.minimum_frame_std,
            "minimum_temporal_abs_diff": args.minimum_temporal_abs_diff,
        }
        reference_outputs = output_validator(
            reference_manifest_path,
            **validator_kwargs,
        )
        report["reference_output_validation"] = os.fspath(
            work_dir / "reference_output_validation.json"
        )
        atomic_write_json(report["reference_output_validation"], reference_outputs)

        report["status"] = "preparing_merged_inference"
        atomic_write_json(report_path, report)
        cloned_manifest = clone_reference_preparation(
            reference_manifest_path=reference_manifest_path,
            merged_checkpoint=merged_checkpoint,
            output_root=merged_inference_root,
        )

        report["status"] = "inferencing_merged_checkpoint"
        atomic_write_json(report_path, report)
        for bucket in cloned_manifest["buckets"]:
            report["active_bucket"] = bucket["bucket_id"]
            atomic_write_json(report_path, report)
            command = [
                sys.executable,
                os.fspath(PROJECT_ROOT / "inference.py"),
                "--config_path",
                bucket["config_path"],
            ]
            print(
                f"[stage1-merged-comparison] bucket={bucket['bucket_id']}: "
                f"{' '.join(command)}"
            )
            command_runner(command, cwd=PROJECT_ROOT, check=True)
        report.pop("active_bucket", None)

        report["status"] = "validating_merged_outputs"
        atomic_write_json(report_path, report)
        cloned_manifest_path = merged_inference_root / "prepared_manifest.json"
        merged_outputs = output_validator(cloned_manifest_path, **validator_kwargs)
        merged_output_validation_path = work_dir / "merged_output_validation.json"
        atomic_write_json(merged_output_validation_path, merged_outputs)
        report["merged_output_validation"] = os.fspath(merged_output_validation_path)

        reference_by_row = {
            int(sample["row_id"]): sample for sample in reference_outputs["samples"]
        }
        merged_by_row = {
            int(sample["row_id"]): sample for sample in merged_outputs["samples"]
        }
        expected_rows = {record.row_id for record in records}
        if (
            set(reference_by_row) != expected_rows
            or set(merged_by_row) != expected_rows
        ):
            raise RuntimeError(
                "Validated output row sets are incomplete: "
                f"expected={sorted(expected_rows)}, "
                f"reference={sorted(reference_by_row)}, merged={sorted(merged_by_row)}"
            )

        report["status"] = "pairing_videos"
        atomic_write_json(report_path, report)
        comparisons_dir = work_dir / "side_by_side"
        samples = []
        for record in records:
            left = Path(reference_by_row[record.row_id]["output_video"])
            right = Path(merged_by_row[record.row_id]["output_video"])
            output = comparisons_dir / (
                f"row{record.row_id:04d}_{record.bucket_id}_"
                "infer-stage1-left_merged-right.mp4"
            )
            stream = pair_writer(
                left,
                right,
                output,
                command_runner=command_runner,
            )
            samples.append(
                {
                    "row_id": record.row_id,
                    "bucket_id": record.bucket_id,
                    "prompt": record.prompt,
                    "left_reference_video": os.fspath(left.resolve()),
                    "right_merged_video": os.fspath(right.resolve()),
                    "comparison_video": os.fspath(output.resolve()),
                    "comparison_sha256": sha256_file(output),
                    "stream": stream,
                }
            )

        comparison_html_path = work_dir / "comparison.html"
        atomic_write_bytes(
            comparison_html_path,
            _comparison_html(samples=samples, work_dir=work_dir).encode("utf-8"),
        )
        report.update(
            {
                "status": "pass",
                "sample_count": len(samples),
                "samples": samples,
                "comparison_html": os.fspath(comparison_html_path),
            }
        )
        atomic_write_json(report_path, report)
        return report
    except Exception as exc:
        report.pop("active_bucket", None)
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(report_path, report)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_comparison(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
