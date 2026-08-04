"""Single-GPU orchestration for one prepared Stage-1 continuation bucket."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from pipeline.causal_diffusion_continuation import tensor_identity_sha256
from utils.inference_utils import save_video
from utils.stage1_continuation_validation import (
    CONTINUATION_BLOCK_SIZE,
    CONTINUATION_EXPECTED_PIXEL_FRAMES,
    CONTINUATION_SINK_SIZES,
    CONTINUATION_TOTAL_LATENT_FRAMES,
    ContinuationMetadataRecord,
    load_prepared_continuation_manifest,
)
from utils.stage1_i2v_data import load_stage1_input_image
from utils.stage1_io import atomic_output_path, atomic_write_json


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _clear_vae_cache(pipeline) -> None:
    model = getattr(pipeline.vae, "model", None)
    clear_cache = getattr(model, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


def _failure_trace(
    *,
    session,
    exc: BaseException,
    initial_latent: torch.Tensor,
    noise_plan: torch.Tensor,
) -> dict[str, Any]:
    if session is not None:
        trace = session.trace_snapshot()
    else:
        trace = {
            "schema": "longlive_stage1_continuation_session",
            "schema_version": 1,
            "status": "failed",
            "cursor_frames": 0,
            "locked": {},
            "initial_latent_sha256": tensor_identity_sha256(initial_latent),
            "noise_identity_sha256": tensor_identity_sha256(noise_plan),
            "noise_slices": [],
            "negative_prompt": None,
            "segments": [],
            "blocks": [],
            "anchors": [],
            "failure": None,
        }
    trace["status"] = "failed"
    trace["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    return trace


def _enrich_trace(
    trace: dict[str, Any],
    *,
    record: ContinuationMetadataRecord,
    entry: Mapping[str, Any],
    sink_size: int,
    manifest: Mapping[str, Any],
    output_video: Path,
    session_trace: Path,
) -> dict[str, Any]:
    trace.update(
        {
            "metadata": {
                "row_id": record.row_id,
                "row_sha256": record.row_sha256,
                "image_sha256": record.image_sha256,
                "case_group": record.case_group,
                "cat_id": record.cat_id,
                "action_order": record.action_order,
                "block_schedule": list(record.block_schedule),
                "soft_reanchor": record.soft_reanchor,
            },
            "sampling": {
                "solver": manifest["sampling"]["solver"],
                "sampling_steps": manifest["sampling"]["sampling_steps"],
                "guidance_scale": manifest["sampling"]["guidance_scale"],
                "seed": manifest["sampling"]["seed"],
                "negative_prompt_sha256": manifest["sampling"][
                    "negative_prompt_sha256"
                ],
                "sink_size": sink_size,
            },
            "frame_policy": dict(manifest["frame_policy"]),
            "output": {
                "output_video": os.fspath(output_video),
                "session_trace": os.fspath(session_trace),
            },
            "technical_validation": {"status": "pending"},
        }
    )
    return trace


def _validate_bucket_runtime(
    pipeline,
    config,
    *,
    bucket: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    if int(_config_value(config, "num_samples", -1)) != 1:
        raise ValueError("continuation runtime requires num_samples=1")
    if int(_config_value(config, "num_output_frames", -1)) != 64:
        raise ValueError("continuation runtime requires num_output_frames=64")
    if not bool(_config_value(config, "i2v", False)):
        raise ValueError("continuation runtime requires i2v=true")
    expected_shape = [
        1,
        CONTINUATION_TOTAL_LATENT_FRAMES,
        48,
        int(bucket["latent_height"]),
        int(bucket["latent_width"]),
    ]
    configured_shape = [
        int(value) for value in _config_value(config, "image_or_video_shape", [])
    ]
    if configured_shape != expected_shape:
        raise ValueError(
            f"continuation runtime shape mismatch: expected {expected_shape}, "
            f"got {configured_shape}"
        )
    if int(pipeline.num_frame_per_block) != CONTINUATION_BLOCK_SIZE:
        raise ValueError("continuation runtime requires 8-latent blocks")
    if int(pipeline.frame_seq_length) != (
        int(bucket["latent_height"]) * int(bucket["latent_width"]) // 4
    ):
        raise ValueError("pipeline frame_seq_length differs from bucket geometry")
    if (
        str(pipeline.sample_solver).lower() != "unipc"
        or int(pipeline.sampling_steps) != 50
        or float(pipeline.guidance_scale) != 5.0
        or float(pipeline.shift) != 5.0
        or int(manifest["sampling"]["seed"]) != 1
    ):
        raise ValueError("continuation runtime sampling contract is not locked")


def run_prepared_continuation_bucket(
    pipeline,
    config,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    """Generate both sink variants for every row in one prepared geometry."""

    continuation_config = _config_value(config, "continuation")
    if continuation_config is None or not bool(
        _config_value(continuation_config, "enabled", False)
    ):
        raise ValueError("continuation config branch is not enabled")
    manifest_path = (
        Path(str(_config_value(continuation_config, "manifest_path", "")))
        .expanduser()
        .resolve()
    )
    bucket_id = str(_config_value(continuation_config, "bucket_id", ""))
    manifest, records = load_prepared_continuation_manifest(manifest_path)
    buckets = [
        bucket for bucket in manifest["buckets"] if bucket["bucket_id"] == bucket_id
    ]
    if len(buckets) != 1:
        raise ValueError(f"continuation manifest has no unique bucket {bucket_id!r}")
    bucket = buckets[0]
    _validate_bucket_runtime(
        pipeline,
        config,
        bucket=bucket,
        manifest=manifest,
    )

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("formal continuation inference requires one CUDA GPU")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise ValueError(
            "continuation inference does not support distributed execution"
        )

    records_by_row = {record.row_id: record for record in records}
    entries_by_row = {int(entry["row_id"]): entry for entry in manifest["records"]}
    bucket_rows = [int(row_id) for row_id in bucket["row_ids"]]
    samples = []
    for row_id in bucket_rows:
        record = records_by_row[row_id]
        entry = entries_by_row[row_id]
        print(
            f"[stage1-continuation] bucket={bucket_id} "
            f"case={record.case_group}: encoding input image once"
        )
        _clear_vae_cache(pipeline)
        image = load_stage1_input_image(record).to(
            device=device,
            dtype=torch.bfloat16,
        )
        initial_latent = pipeline.vae.encode_to_latent(image).to(
            device=device,
            dtype=torch.bfloat16,
        )
        expected_initial_shape = (
            1,
            1,
            48,
            record.height // 16,
            record.width // 16,
        )
        if tuple(initial_latent.shape) != expected_initial_shape:
            raise RuntimeError(
                f"VAE initial latent shape mismatch for {record.case_group}: "
                f"expected {expected_initial_shape}, got {tuple(initial_latent.shape)}"
            )
        if not torch.isfinite(initial_latent).all().item():
            raise RuntimeError(
                f"VAE initial latent contains NaN/Inf for {record.case_group}"
            )

        generator = torch.Generator(device=device)
        generator.manual_seed(int(manifest["sampling"]["seed"]))
        noise_plan = torch.randn(
            (
                1,
                CONTINUATION_TOTAL_LATENT_FRAMES,
                48,
                record.height // 16,
                record.width // 16,
            ),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        noise_identity = tensor_identity_sha256(noise_plan)
        initial_identity = tensor_identity_sha256(initial_latent)

        for output in entry["outputs"]:
            sink_size = int(output["sink_size"])
            if sink_size not in CONTINUATION_SINK_SIZES:
                raise RuntimeError("prepared output contains an unsupported sink")
            output_video = Path(output["output_video"]).resolve()
            session_trace = Path(output["session_trace"]).resolve()
            if output_video.exists() or session_trace.exists():
                raise FileExistsError(
                    f"continuation output already exists for {record.case_group}/sink{sink_size}"
                )
            session = None
            trace: dict[str, Any] | None = None
            try:
                print(
                    f"[stage1-continuation] case={record.case_group} "
                    f"sink={sink_size}: generating A/HOLD/B"
                )
                session = pipeline.begin_session(
                    initial_latent=initial_latent.clone(),
                    sink_size=sink_size,
                    noise_plan=noise_plan.clone(),
                )
                session.generate_segment(
                    record.action_a_prompt,
                    noise=noise_plan[:, 0:24].clone(),
                    segment_name="action_a",
                )
                session.generate_segment(
                    record.hold_prompt,
                    noise=noise_plan[:, 24:40].clone(),
                    segment_name="hold",
                )
                session.generate_segment(
                    record.action_b_prompt,
                    noise=noise_plan[:, 40:64].clone(),
                    carry_last_latent_as_anchor=True,
                    segment_name="action_b",
                )
                result = session.finish()
                expected_latent_shape = (
                    1,
                    64,
                    48,
                    record.height // 16,
                    record.width // 16,
                )
                expected_pixel_shape = (
                    1,
                    CONTINUATION_EXPECTED_PIXEL_FRAMES,
                    3,
                    record.height,
                    record.width,
                )
                if tuple(result.latents.shape) != expected_latent_shape:
                    raise RuntimeError(
                        f"continuation latent shape mismatch: {tuple(result.latents.shape)}"
                    )
                if tuple(result.video.shape) != expected_pixel_shape:
                    raise RuntimeError(
                        f"continuation pixel shape mismatch: {tuple(result.video.shape)}"
                    )
                trace = result.trace
                if (
                    trace.get("noise_identity_sha256") != noise_identity
                    or trace.get("initial_latent_sha256") != initial_identity
                ):
                    raise RuntimeError(
                        "session changed the shared initial latent/noise identity"
                    )
                trace["result"] = {
                    "latent_shape": list(result.latents.shape),
                    "pixel_shape": list(result.video.shape),
                    "decode_calls": 1,
                    "decode_mode": "single_full_sequence",
                    "latent_boundaries": manifest["frame_policy"]["latent_boundaries"],
                    "pixel_boundaries": manifest["frame_policy"]["pixel_boundaries"],
                }
                _enrich_trace(
                    trace,
                    record=record,
                    entry=entry,
                    sink_size=sink_size,
                    manifest=manifest,
                    output_video=output_video,
                    session_trace=session_trace,
                )
                with atomic_output_path(output_video, suffix=".mp4") as temporary:
                    save_video(result.video, temporary, fps=24)
                atomic_write_json(session_trace, trace)
                samples.append(
                    {
                        "row_id": row_id,
                        "case_group": record.case_group,
                        "sink_size": sink_size,
                        "output_video": os.fspath(output_video),
                        "session_trace": os.fspath(session_trace),
                        "noise_identity_sha256": noise_identity,
                        "initial_latent_sha256": initial_identity,
                    }
                )
                del result
            except Exception as exc:
                failure = _failure_trace(
                    session=session,
                    exc=exc,
                    initial_latent=initial_latent,
                    noise_plan=noise_plan,
                )
                _enrich_trace(
                    failure,
                    record=record,
                    entry=entry,
                    sink_size=sink_size,
                    manifest=manifest,
                    output_video=output_video,
                    session_trace=session_trace,
                )
                atomic_write_json(session_trace, failure)
                raise
            finally:
                _clear_vae_cache(pipeline)

        del image, initial_latent, noise_plan
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "bucket_id": bucket_id,
        "row_count": len(bucket_rows),
        "sample_count": len(samples),
        "samples": samples,
        "prepared_manifest": os.fspath(manifest_path),
    }
