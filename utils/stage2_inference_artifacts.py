"""Technical-only Stage-2 inference traces, manifests, and review index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Callable, Mapping, Sequence
from html import escape
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from pipeline.stage2_rollout_profile import (
    STAGE2_ROLLOUT_PROFILE_NAMES,
    Stage2RolloutSpec,
    resolve_stage2_rollout_profile,
    resolve_stage2_shift5_schedule,
)
from utils.stage1_causal_validation import probe_video
from utils.stage1_io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from utils.stage2_inference import (
    STAGE2_INFERENCE_FPS,
    STAGE2_INFERENCE_TRACE_SCHEMA,
    STAGE2_NOISE_STREAM_POLICY,
    STAGE2_OUTPUT_PIXEL_FRAMES_PER_EPISODE,
)
from utils.stage2_inference_assets import validate_stage2_runtime_asset_identity
from utils.stage2_inference_batch import (
    STAGE2_SINGLE_DATASET,
    STAGE2_TWO_ACTION_DATASET,
    Stage2InferenceSample,
)
from utils.stage2_inference_config import (
    STAGE2_INFERENCE_CONFIG_SCHEMA,
    ResolvedStage2InferenceConfig,
)
from utils.stage2_inference_sweep_config import (
    STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA,
    STAGE2_INFERENCE_SWEEP_MAX_PROFILES,
    ResolvedStage2InferenceSweepConfig,
)
from utils.stage2_inference_timing_report import validate_stage2_timing

STAGE2_SAMPLE_TRACE_SCHEMA = "longlive_stage2_sample_trace/v2"
STAGE2_INFERENCE_MANIFEST_SCHEMA = "longlive_stage2_inference_manifest/v2"
STAGE2_INFERENCE_CONFIG_IDENTITY_SCHEMA = "longlive_stage2_inference_config_identity/v2"
STAGE2_INFERENCE_MANIFEST_NAME = "manifest.json"
STAGE2_REVIEW_INDEX_NAME = "index.html"
_FORBIDDEN_QUALITY_FIELDS = {
    "psnr",
    "ssim",
    "lpips",
    "clip_score",
    "temporal_difference",
    "frame_std",
    "quality_score",
}

_CHECKPOINT_KEYS = {
    "directory",
    "manifest_sha256",
    "completed_generator_updates",
    "contract_hash",
    "generator_ema_sha256",
}
_RESOLVED_INFERENCE_CONFIG_KEYS = {
    "schema",
    "stage2_checkpoint",
    "architecture_root",
    "t5_checkpoint",
    "tokenizer_dir",
    "vae_checkpoint",
    "source_cache_manifest",
    "single_metadata",
    "two_action_metadata",
    "output_root",
    "profiles",
    "seeds",
    "dtype",
    "cfg_scale",
    "fps",
    "merge_ema_lora",
    "batch_size_per_device",
}
_RESOLVED_INFERENCE_SWEEP_CONFIG_KEYS = {
    *_RESOLVED_INFERENCE_CONFIG_KEYS,
    "base_config_path",
    "base_config_sha256",
    "evaluation_mode",
    "single_row_ids",
    "two_action_row_ids",
    "profile_set_sha256",
}
_ROLLOUT_RESULT_KEYS = {
    "exit_step",
    "requires_grad",
    "scheduler_timesteps",
    "scheduler_sigmas",
    "chunk_timesteps",
    "chunk_sigmas",
    "cache_audit",
    "rollout_mode",
    "chunk_trace",
    "initial_latent_sha256_per_sample",
    "generated_latent_sha256_per_sample",
}
_CACHE_AUDIT_KEYS = {
    "layers",
    "capacity_frames",
    "capacity_tokens",
    "global_end_index",
    "local_end_index",
    "persistent_kv_detached",
    "conditional_cache_branches",
    "cross_kv_active",
    "cross_kv_initialized",
    "profile",
    "scheduler_instances",
    "solver_update_calls",
    "terminal_x0_direct",
    "noisy_forward_calls",
    "clean_recache_forward_calls",
    "sink_preload_forward_calls",
    "generator_forward_calls",
    "logical_query_tokens",
}
_CHUNK_TRACE_KEYS = {
    "profile_name",
    "episode_index",
    "chunk_index",
    "generated_frame_start",
    "generated_frame_end_exclusive",
    "rope_frame_start",
    "rope_frame_end_exclusive",
    "timestep_count",
    "exit_step",
    "denoising_forward_calls",
    "solver_update_calls",
    "clean_recache_forward_calls",
    "rollout_mode",
    "timesteps",
    "sigmas",
    "terminal_sigma",
    "fresh_scheduler",
    "noisy_self_kv_commits",
    "clean_self_kv_commits",
    "cache_before_global_end_index",
    "cache_before_local_end_index",
    "cache_after_global_end_index",
    "cache_after_local_end_index",
    "cache_capacity_frames",
    "cache_capacity_tokens",
    "sink_frames",
    "initial_latent_sha256_per_sample",
    "clean_latent_sha256_per_sample",
    "captured_prefix_sink_frames",
    "captured_prefix_clean_latent_sha256_per_sample",
}


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _rollout_topology_key(spec: Stage2RolloutSpec) -> tuple[Any, ...]:
    return (
        int(spec.generated_episode_frames),
        int(spec.chunk_frames),
        int(spec.local_window_frames),
        int(spec.global_sink_frames),
        int(spec.num_denoising_steps),
        str(spec.solver),
        float(spec.timestep_shift),
    )


def _canonical_rollout_key(spec: Stage2RolloutSpec) -> tuple[Any, ...]:
    return (*_rollout_topology_key(spec), spec.name)


def _canonical_int_list(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not minimum <= item <= maximum
        for item in value
    ):
        raise ValueError(
            f"{label} must contain plain integers in [{minimum}, {maximum}]"
        )
    if value != sorted(value) or len(value) != len(set(value)):
        raise ValueError(f"{label} must be sorted and unique")
    return list(value)


def _validate_sweep_resolved_config(resolved: Mapping[str, Any]) -> None:
    _sha256(resolved.get("base_config_sha256"), "config.base_config_sha256")
    profiles = resolved.get("profiles")
    if (
        not isinstance(profiles, list)
        or not profiles
        or len(profiles) > STAGE2_INFERENCE_SWEEP_MAX_PROFILES
        or any(not isinstance(item, str) for item in profiles)
        or len(profiles) != len(set(profiles))
    ):
        raise ValueError("Stage-2 resolved inference sweep profiles are invalid")
    specs = tuple(resolve_stage2_rollout_profile(name) for name in profiles)
    if any(spec.global_sink_frames != 1 for spec in specs):
        raise ValueError("Stage-2 inference sweep profiles must all use S=1")
    if specs != tuple(sorted(specs, key=_canonical_rollout_key)):
        raise ValueError("Stage-2 inference sweep profiles are not canonical")
    topology_keys = [_rollout_topology_key(spec) for spec in specs]
    if len(topology_keys) != len(set(topology_keys)):
        raise ValueError("Stage-2 inference sweep repeats a rollout topology")
    profile_set_sha256 = _sha256(
        resolved.get("profile_set_sha256"),
        "config.profile_set_sha256",
    )
    expected_profile_set_sha256 = canonical_json_sha256(
        [spec.to_dict() for spec in specs]
    )
    if profile_set_sha256 != expected_profile_set_sha256:
        raise RuntimeError("Stage-2 inference sweep profile set SHA-256 mismatch")

    seeds = _canonical_int_list(
        resolved.get("seeds"),
        "Stage-2 resolved inference sweep seeds",
        minimum=0,
        maximum=(1 << 63) - 1,
    )
    single_row_ids = _canonical_int_list(
        resolved.get("single_row_ids"),
        "Stage-2 resolved inference sweep single row ids",
        minimum=0,
        maximum=5,
    )
    two_action_row_ids = _canonical_int_list(
        resolved.get("two_action_row_ids"),
        "Stage-2 resolved inference sweep two-action row ids",
        minimum=0,
        maximum=7,
    )
    evaluation_mode = resolved.get("evaluation_mode")
    if evaluation_mode not in {"formal", "quick"}:
        raise ValueError("Stage-2 inference sweep evaluation mode is invalid")
    if evaluation_mode == "formal" and (
        seeds != [1, 2, 3, 4]
        or single_row_ids != list(range(6))
        or two_action_row_ids != list(range(8))
    ):
        raise ValueError(
            "formal Stage-2 inference sweep requires seeds 1..4, "
            "single rows 0..5, and two-action rows 0..7"
        )


def validate_stage2_inference_config_identity(
    identity: Any,
) -> dict[str, Any]:
    """Validate the exact resolved config identity embedded in every artifact."""

    expected_keys = {
        "schema",
        "resolved",
        "runtime_assets",
        "resolved_contract_hash",
        "resolved_launch_hash",
        "runtime_contract_hash",
        "runtime_launch_hash",
    }
    if not isinstance(identity, Mapping) or set(identity) != expected_keys:
        raise ValueError("Stage-2 inference config identity schema mismatch")
    if identity.get("schema") != STAGE2_INFERENCE_CONFIG_IDENTITY_SCHEMA:
        raise ValueError("Stage-2 inference config identity version mismatch")
    raw_resolved = identity.get("resolved")
    if not isinstance(raw_resolved, Mapping):
        raise TypeError("Stage-2 resolved inference config must be a mapping")
    resolved = dict(raw_resolved)
    resolved_schema = resolved.get("schema")
    if resolved_schema == STAGE2_INFERENCE_CONFIG_SCHEMA:
        if set(resolved) != _RESOLVED_INFERENCE_CONFIG_KEYS:
            raise ValueError("Stage-2 resolved inference config schema mismatch")
        profiles = resolved.get("profiles")
        if (
            not isinstance(profiles, list)
            or not profiles
            or any(not isinstance(item, str) for item in profiles)
            or len(profiles) != len(set(profiles))
        ):
            raise ValueError("Stage-2 resolved inference profiles are invalid")
        if any(profile not in STAGE2_ROLLOUT_PROFILE_NAMES for profile in profiles):
            raise ValueError(
                "formal Stage-2 resolved config requires a frozen named profile"
            )
        for profile in profiles:
            resolve_stage2_rollout_profile(profile)
        seeds = resolved.get("seeds")
        if seeds != [1, 2, 3, 4] or any(type(seed) is not int for seed in seeds):
            raise ValueError("Stage-2 resolved inference seeds are invalid")
    elif resolved_schema == STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA:
        if set(resolved) != _RESOLVED_INFERENCE_SWEEP_CONFIG_KEYS:
            raise ValueError("Stage-2 resolved inference sweep config schema mismatch")
        _validate_sweep_resolved_config(resolved)
    else:
        raise ValueError("Stage-2 resolved inference config version mismatch")

    path_fields = [
        "stage2_checkpoint",
        "architecture_root",
        "t5_checkpoint",
        "tokenizer_dir",
        "vae_checkpoint",
        "source_cache_manifest",
        "single_metadata",
        "two_action_metadata",
        "output_root",
    ]
    if resolved_schema == STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA:
        path_fields.append("base_config_path")
    for field in path_fields:
        value = resolved.get(field)
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or not Path(value).is_absolute()
        ):
            raise ValueError(
                f"Stage-2 resolved inference config {field} must be an absolute path"
            )
    if resolved.get("dtype") != "bfloat16":
        raise ValueError("Stage-2 resolved inference dtype is invalid")
    cfg_scale = resolved.get("cfg_scale")
    if type(cfg_scale) is not float:
        raise TypeError("Stage-2 resolved inference CFG scale is invalid")
    if not math.isfinite(float(cfg_scale)) or float(cfg_scale) != 1.0:
        raise ValueError("Stage-2 resolved inference CFG scale is invalid")
    if resolved.get("fps") != 24 or type(resolved.get("fps")) is not int:
        raise ValueError("Stage-2 resolved inference FPS is invalid")
    if resolved.get("merge_ema_lora") is not True:
        raise ValueError("Stage-2 resolved inference EMA merge mode is invalid")
    if (
        resolved.get("batch_size_per_device") != 1
        or type(resolved.get("batch_size_per_device")) is not int
    ):
        raise ValueError("Stage-2 resolved inference batch size is invalid")
    try:
        canonical = json.loads(json.dumps(resolved, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Stage-2 resolved inference config is not strict JSON"
        ) from exc
    runtime_assets = validate_stage2_runtime_asset_identity(
        identity.get("runtime_assets")
    )
    expected_asset_paths = {
        "t5_checkpoint": canonical["t5_checkpoint"],
        "tokenizer_dir": canonical["tokenizer_dir"],
        "vae_checkpoint": canonical["vae_checkpoint"],
        "architecture_config": str(
            Path(canonical["architecture_root"]) / "config.json"
        ),
    }
    if runtime_assets["checkpoint"]["directory"] != canonical["stage2_checkpoint"]:
        raise RuntimeError(
            "Stage-2 runtime checkpoint path differs from the resolved config"
        )
    if runtime_assets["source_manifest"]["path"] != canonical["source_cache_manifest"]:
        raise RuntimeError(
            "Stage-2 runtime source manifest path differs from the resolved config"
        )
    for name, expected_path in expected_asset_paths.items():
        if runtime_assets["assets"][name]["path"] != expected_path:
            raise RuntimeError(
                f"Stage-2 runtime asset {name} path differs from the resolved config"
            )
    contract_value = dict(canonical)
    contract_value.pop("output_root")
    expected_resolved_contract_hash = canonical_json_sha256(contract_value)
    expected_resolved_launch_hash = canonical_json_sha256(canonical)
    expected_runtime_contract_hash = canonical_json_sha256(
        {"resolved": contract_value, "runtime_assets": runtime_assets}
    )
    expected_runtime_launch_hash = canonical_json_sha256(
        {"resolved": canonical, "runtime_assets": runtime_assets}
    )
    resolved_contract_hash = _sha256(
        identity.get("resolved_contract_hash"),
        "config.resolved_contract_hash",
    )
    resolved_launch_hash = _sha256(
        identity.get("resolved_launch_hash"),
        "config.resolved_launch_hash",
    )
    runtime_contract_hash = _sha256(
        identity.get("runtime_contract_hash"),
        "config.runtime_contract_hash",
    )
    runtime_launch_hash = _sha256(
        identity.get("runtime_launch_hash"),
        "config.runtime_launch_hash",
    )
    if resolved_contract_hash != expected_resolved_contract_hash:
        raise RuntimeError("Stage-2 resolved inference contract hash mismatch")
    if resolved_launch_hash != expected_resolved_launch_hash:
        raise RuntimeError("Stage-2 resolved inference launch hash mismatch")
    if runtime_contract_hash != expected_runtime_contract_hash:
        raise RuntimeError("Stage-2 runtime inference contract hash mismatch")
    if runtime_launch_hash != expected_runtime_launch_hash:
        raise RuntimeError("Stage-2 runtime inference launch hash mismatch")
    return {
        "schema": STAGE2_INFERENCE_CONFIG_IDENTITY_SCHEMA,
        "resolved": canonical,
        "runtime_assets": runtime_assets,
        "resolved_contract_hash": resolved_contract_hash,
        "resolved_launch_hash": resolved_launch_hash,
        "runtime_contract_hash": runtime_contract_hash,
        "runtime_launch_hash": runtime_launch_hash,
    }


def build_stage2_inference_config_identity(
    config: ResolvedStage2InferenceConfig | ResolvedStage2InferenceSweepConfig,
    *,
    runtime_assets: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a canonical, self-verifying identity for a resolved launch."""

    if not isinstance(
        config,
        (ResolvedStage2InferenceConfig, ResolvedStage2InferenceSweepConfig),
    ):
        raise TypeError(
            "Stage-2 inference config identity requires a resolved strict config"
        )
    resolved = json.loads(json.dumps(config.to_dict(), allow_nan=False, sort_keys=True))
    validated_assets = validate_stage2_runtime_asset_identity(runtime_assets)
    contract_resolved = dict(resolved)
    contract_resolved.pop("output_root")
    return validate_stage2_inference_config_identity(
        {
            "schema": STAGE2_INFERENCE_CONFIG_IDENTITY_SCHEMA,
            "resolved": resolved,
            "runtime_assets": validated_assets,
            "resolved_contract_hash": config.contract_hash(),
            "resolved_launch_hash": config.launch_hash(),
            "runtime_contract_hash": canonical_json_sha256(
                {"resolved": contract_resolved, "runtime_assets": validated_assets}
            ),
            "runtime_launch_hash": canonical_json_sha256(
                {"resolved": resolved, "runtime_assets": validated_assets}
            ),
        }
    )


def _hash_list(value: Any, label: str, *, count: int = 1) -> list[str]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label} must contain exactly {count} hashes")
    return [_sha256(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _forbidden_quality_key(value: Any, path: str = "trace") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_QUALITY_FIELDS:
                return f"{path}.{key}"
            found = _forbidden_quality_key(child, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            found = _forbidden_quality_key(child, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _validate_checkpoint_identity(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != _CHECKPOINT_KEYS:
        raise ValueError("Stage-2 inference checkpoint identity schema mismatch")
    directory = checkpoint["directory"]
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("Stage-2 checkpoint directory must be a non-empty string")
    completed = _plain_int(
        checkpoint["completed_generator_updates"],
        "completed_generator_updates",
    )
    if completed < 40:
        raise RuntimeError(
            "Stage-2 inference requires initialized Generator EMA (G>=40)"
        )
    for field in ("manifest_sha256", "contract_hash", "generator_ema_sha256"):
        _sha256(checkpoint[field], f"checkpoint.{field}")
    return dict(checkpoint)


def _expected_timetable(steps: int) -> tuple[int, ...]:
    return resolve_stage2_shift5_schedule(steps).timesteps


def _frame_tokens(sample: Stage2InferenceSample) -> int:
    latent_height = sample.height // 16
    latent_width = sample.width // 16
    if sample.height % 16 or sample.width % 16 or latent_height % 2 or latent_width % 2:
        raise ValueError("Stage-2 inference resolution is not compatible with patching")
    return (latent_height // 2) * (latent_width // 2)


def _raw_prompt_sha256(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Stage-2 raw prompts must be non-empty strings")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _relative_artifact(root: Path, path: Path, *, expected: Path) -> str:
    root = root.expanduser().resolve()
    path = path.expanduser().resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Stage-2 artifact escapes output root: {path}") from exc
    if relative != expected:
        raise RuntimeError(
            f"Stage-2 artifact path mismatch: expected={expected}, actual={relative}"
        )
    return relative.as_posix()


def _expected_frames(sample: Stage2InferenceSample) -> int:
    if sample.dataset == STAGE2_SINGLE_DATASET:
        return STAGE2_OUTPUT_PIXEL_FRAMES_PER_EPISODE
    if sample.dataset == STAGE2_TWO_ACTION_DATASET:
        return 2 * STAGE2_OUTPUT_PIXEL_FRAMES_PER_EPISODE
    raise ValueError(f"unknown Stage-2 inference dataset {sample.dataset!r}")


def _validate_sample_config_scope(
    sample: Stage2InferenceSample,
    resolved_config: Mapping[str, Any],
) -> None:
    if sample.profile not in resolved_config["profiles"]:
        raise RuntimeError("Stage-2 sample trace profile is absent from its config")
    if resolved_config["schema"] != STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA:
        return
    if sample.seed not in resolved_config["seeds"] or isinstance(sample.seed, bool):
        raise RuntimeError("Stage-2 sweep sample seed is absent from its config")
    row_field = (
        "single_row_ids"
        if sample.dataset == STAGE2_SINGLE_DATASET
        else "two_action_row_ids"
    )
    if sample.row_id not in resolved_config[row_field] or isinstance(
        sample.row_id, bool
    ):
        raise RuntimeError("Stage-2 sweep sample row is absent from its config")


def _expected_sample_coordinates(
    resolved_config: Mapping[str, Any],
) -> list[tuple[str, str, int, int]]:
    if resolved_config["schema"] == STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA:
        single_rows = resolved_config["single_row_ids"]
        two_action_rows = resolved_config["two_action_row_ids"]
    else:
        single_rows = list(range(6))
        two_action_rows = list(range(8))
    coordinates: list[tuple[str, str, int, int]] = []
    for profile in resolved_config["profiles"]:
        for row_id in single_rows:
            for seed in resolved_config["seeds"]:
                coordinates.append((profile, STAGE2_SINGLE_DATASET, row_id, seed))
        for row_id in two_action_rows:
            for seed in resolved_config["seeds"]:
                coordinates.append((profile, STAGE2_TWO_ACTION_DATASET, row_id, seed))
    return coordinates


def validate_stage2_video_artifact(
    sample: Stage2InferenceSample,
    video_path: str | os.PathLike[str],
    *,
    probe_fn: Callable[[str | os.PathLike[str]], Mapping[str, Any]] = probe_video,
) -> dict[str, Any]:
    """Validate only the task-authorized technical MP4 properties."""

    path = Path(video_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Stage-2 output is not a non-empty regular file: {path}")
    probe = dict(probe_fn(path))
    expected_probe = {
        "width": sample.width,
        "height": sample.height,
        "frame_count": _expected_frames(sample),
    }
    for key, expected in expected_probe.items():
        actual = probe.get(key)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual != expected
        ):
            raise RuntimeError(
                f"Stage-2 video {key} mismatch for {sample.sample_key}: "
                f"expected={expected}, actual={actual}"
            )
    fps = probe.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise TypeError("Stage-2 ffprobe fps must be numeric")
    if abs(float(fps) - STAGE2_INFERENCE_FPS) > 1e-6:
        raise RuntimeError(f"Stage-2 video fps mismatch: expected=24, actual={fps}")
    return {
        **expected_probe,
        "fps": float(fps),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_generation_trace(
    sample: Stage2InferenceSample,
    generation: Mapping[str, Any],
    profile: Stage2RolloutSpec,
) -> None:
    if not isinstance(generation, Mapping):
        raise TypeError("Stage-2 generation trace must be an object")
    expected_mode = (
        "single_action" if sample.dataset == STAGE2_SINGLE_DATASET else "two_action"
    )
    expected_keys = {
        "schema",
        "mode",
        "seeds",
        "noise_stream",
        "initial_latent_sha256",
        "noise_plan_sha256",
        "noise_episode_sha256",
        "prompt_embedding_sha256",
        "episodes",
        "vae_events",
        "output_pixel_frames",
    }
    if expected_mode == "two_action":
        expected_keys.add("reset_events")
    if "timing" in generation:
        expected_keys.add("timing")
        timing = validate_stage2_timing(generation["timing"])
        episode1_spec = (
            resolve_stage2_rollout_profile("baseline_c8w16k4s1")
            if profile.global_sink_frames > 1
            else profile
        )
        expected_dit_calls = episode1_spec.fresh_deploy_dit_calls
        expected_vae_calls = 1
        # Each decode normalizes pixels; two-action additionally concatenates,
        # and the runtime saves the finished video exactly once.
        expected_postprocess_calls = 2
        if expected_mode == "two_action":
            expected_dit_calls += profile.fresh_deploy_dit_calls - 1
            expected_vae_calls = 2
            expected_postprocess_calls = 4
        if (
            timing["dit_calls"] != expected_dit_calls
            or timing["vae_decode_calls"] != expected_vae_calls
            or timing["video_postprocess_calls"] != expected_postprocess_calls
        ):
            raise ValueError("Stage-2 timing call counts differ from its rollout")
    if set(generation) != expected_keys:
        raise ValueError("Stage-2 generation trace schema mismatch")
    if generation.get("schema") != STAGE2_INFERENCE_TRACE_SCHEMA:
        raise ValueError("Stage-2 generation trace schema mismatch")
    if generation.get("mode") != expected_mode:
        raise ValueError("Stage-2 generation trace mode differs from its dataset")
    if generation.get("seeds") != [sample.seed]:
        raise ValueError("Stage-2 generation trace seed differs from its sample key")
    expected_episodes = 1 if expected_mode == "single_action" else 2
    expected_noise_stream = {
        "policy": STAGE2_NOISE_STREAM_POLICY,
        "rng_initializations_per_sample": 1,
        "episode_slots": 24,
        "episode_order": ["single"] if expected_episodes == 1 else ["A", "B"],
    }
    if generation.get("noise_stream") != expected_noise_stream:
        raise RuntimeError(
            "Stage-2 noise trace must encode one per-sample RNG and contiguous "
            "episode slots"
        )
    initial_hashes = _hash_list(
        generation.get("initial_latent_sha256"),
        "initial_latent_sha256",
    )
    _hash_list(generation.get("noise_plan_sha256"), "noise_plan_sha256")
    episodes = generation.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        raise ValueError("Stage-2 generation trace episode count mismatch")
    if generation.get("output_pixel_frames") != _expected_frames(sample):
        raise ValueError("Stage-2 generation trace pixel frame count mismatch")
    noise_hashes = generation.get("noise_episode_sha256")
    if (
        not isinstance(noise_hashes, list)
        or len(noise_hashes) != expected_episodes
        or any(not isinstance(item, list) or len(item) != 1 for item in noise_hashes)
    ):
        raise ValueError("Stage-2 generation trace noise hash layout mismatch")
    for index, item in enumerate(noise_hashes):
        _hash_list(item, f"noise_episode_sha256[{index}]")
    if expected_episodes == 2 and noise_hashes[0][0] == noise_hashes[1][0]:
        raise RuntimeError("Stage-2 A/B initial noise hashes must differ")
    prompt_hashes = generation.get("prompt_embedding_sha256")
    if (
        not isinstance(prompt_hashes, list)
        or len(prompt_hashes) != expected_episodes
        or any(not isinstance(item, list) or len(item) != 1 for item in prompt_hashes)
    ):
        raise ValueError("Stage-2 prompt embedding hash layout mismatch")
    for index, item in enumerate(prompt_hashes):
        _hash_list(item, f"prompt_embedding_sha256[{index}]")
    frame_tokens = _frame_tokens(sample)
    for episode_index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping) or set(episode) != _ROLLOUT_RESULT_KEYS:
            raise ValueError("Stage-2 rollout result trace schema mismatch")
        actual_profile = profile
        if profile.global_sink_frames > 1 and episode_index == 0:
            actual_profile = resolve_stage2_rollout_profile("baseline_c8w16k4s1")
        if episode.get("requires_grad") is not False:
            raise RuntimeError("Stage-2 deployment episodes must run without gradients")
        if episode.get("rollout_mode") != "full_denoising":
            raise RuntimeError("Stage-2 deployment episode must use full_denoising")
        if episode.get("exit_step") != actual_profile.num_denoising_steps - 1:
            raise RuntimeError("Stage-2 deployment episode did not execute full K")
        timetable = episode.get("scheduler_timesteps")
        reference_schedule = resolve_stage2_shift5_schedule(
            actual_profile.num_denoising_steps
        )
        expected_timetable = list(reference_schedule.timesteps)
        if timetable != expected_timetable:
            raise RuntimeError("Stage-2 deployment UniPC timetable mismatch")
        sigmas = episode.get("scheduler_sigmas")
        if (
            not isinstance(sigmas, list)
            or len(sigmas) != actual_profile.num_denoising_steps + 1
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in sigmas
            )
            or any(float(left) < float(right) for left, right in pairwise(sigmas))
            or float(sigmas[0]) <= 0.0
            or abs(float(sigmas[-1])) > 1.0e-12
        ):
            raise RuntimeError("Stage-2 deployment UniPC sigma schedule is invalid")
        sigma_fp32_bits = tuple(
            struct.pack(">f", float(value)).hex() for value in sigmas
        )
        if sigma_fp32_bits != reference_schedule.sigma_fp32_bits:
            raise RuntimeError("Stage-2 deployment UniPC sigma schedule drifted")
        chunks = actual_profile.num_chunks
        if episode.get("chunk_timesteps") != [expected_timetable] * chunks:
            raise RuntimeError("Stage-2 chunk timesteps differ from the fresh schedule")
        if episode.get("chunk_sigmas") != [sigmas[:-1]] * chunks:
            raise RuntimeError("Stage-2 chunk sigmas differ from the fresh schedule")
        if episode.get("initial_latent_sha256_per_sample") != initial_hashes:
            raise RuntimeError("Stage-2 rollout initial sink hash changed")
        _hash_list(
            episode.get("generated_latent_sha256_per_sample"),
            f"episodes[{episode_index}].generated_latent_sha256_per_sample",
        )
        cache = episode.get("cache_audit")
        if not isinstance(cache, Mapping) or set(cache) != _CACHE_AUDIT_KEYS:
            raise TypeError("Stage-2 deployment cache audit schema mismatch")
        _plain_int(cache.get("layers"), "cache_audit.layers", minimum=1)
        expected_preload = int(episode_index == 0)
        expected_noisy_calls = chunks * actual_profile.num_denoising_steps
        expected_recache_calls = chunks
        expected_calls = (
            expected_preload + expected_noisy_calls + expected_recache_calls
        )
        expected_capacity = actual_profile.physical_kv_capacity_frames
        expected_cache = {
            "capacity_frames": expected_capacity,
            "capacity_tokens": expected_capacity * frame_tokens,
            "global_end_index": (actual_profile.global_sink_frames + 24) * frame_tokens,
            "local_end_index": expected_capacity * frame_tokens,
            "persistent_kv_detached": True,
            "conditional_cache_branches": 1,
            "cross_kv_active": True,
            "cross_kv_initialized": True,
            "profile": actual_profile.to_dict(),
            "scheduler_instances": chunks,
            "solver_update_calls": chunks * (actual_profile.num_denoising_steps - 1),
            "terminal_x0_direct": True,
            "noisy_forward_calls": expected_noisy_calls,
            "clean_recache_forward_calls": expected_recache_calls,
            "sink_preload_forward_calls": expected_preload,
            "generator_forward_calls": expected_calls,
            "logical_query_tokens": (
                expected_preload * actual_profile.global_sink_frames * frame_tokens
                + (expected_noisy_calls + expected_recache_calls)
                * actual_profile.chunk_frames
                * frame_tokens
            ),
        }
        for key, expected in expected_cache.items():
            if cache.get(key) != expected:
                raise RuntimeError(
                    f"Stage-2 cache audit {key} mismatch: "
                    f"expected={expected}, actual={cache.get(key)}"
                )
        chunk_trace = episode.get("chunk_trace")
        if not isinstance(chunk_trace, list) or len(chunk_trace) != chunks:
            raise RuntimeError("Stage-2 deployment chunk trace count mismatch")
        for chunk_index, chunk in enumerate(chunk_trace):
            if not isinstance(chunk, Mapping) or set(chunk) != _CHUNK_TRACE_KEYS:
                raise ValueError("Stage-2 deployment chunk trace schema mismatch")
            start = chunk_index * actual_profile.chunk_frames
            before_frames = min(
                expected_capacity,
                actual_profile.global_sink_frames + start,
            )
            after_frames = min(
                expected_capacity,
                actual_profile.global_sink_frames + start + actual_profile.chunk_frames,
            )
            capture_frames = (
                profile.global_sink_frames
                if expected_mode == "two_action"
                and profile.global_sink_frames > 1
                and episode_index == 0
                and chunk_index == 0
                else 0
            )
            expected_chunk = {
                "profile_name": actual_profile.name,
                "episode_index": episode_index,
                "chunk_index": chunk_index,
                "generated_frame_start": start,
                "generated_frame_end_exclusive": start + actual_profile.chunk_frames,
                "rope_frame_start": actual_profile.global_sink_frames + start,
                "rope_frame_end_exclusive": actual_profile.global_sink_frames
                + start
                + actual_profile.chunk_frames,
                "timestep_count": actual_profile.num_denoising_steps,
                "exit_step": actual_profile.num_denoising_steps - 1,
                "denoising_forward_calls": actual_profile.num_denoising_steps,
                "solver_update_calls": actual_profile.num_denoising_steps - 1,
                "clean_recache_forward_calls": 1,
                "rollout_mode": "full_denoising",
                "timesteps": expected_timetable,
                "sigmas": sigmas[:-1],
                "terminal_sigma": sigmas[-1],
                "fresh_scheduler": True,
                "noisy_self_kv_commits": 0,
                "clean_self_kv_commits": 1,
                "cache_before_global_end_index": (
                    actual_profile.global_sink_frames + start
                )
                * frame_tokens,
                "cache_before_local_end_index": before_frames * frame_tokens,
                "cache_after_global_end_index": (
                    actual_profile.global_sink_frames
                    + start
                    + actual_profile.chunk_frames
                )
                * frame_tokens,
                "cache_after_local_end_index": after_frames * frame_tokens,
                "cache_capacity_frames": expected_capacity,
                "cache_capacity_tokens": expected_capacity * frame_tokens,
                "sink_frames": actual_profile.global_sink_frames,
                "initial_latent_sha256_per_sample": initial_hashes,
                "captured_prefix_sink_frames": capture_frames,
            }
            for key, expected in expected_chunk.items():
                if chunk.get(key) != expected:
                    raise RuntimeError(
                        f"Stage-2 chunk trace {key} mismatch at chunk "
                        f"{chunk_index}: expected={expected}, actual={chunk.get(key)}"
                    )
            _hash_list(
                chunk.get("clean_latent_sha256_per_sample"),
                f"episodes[{episode_index}].chunk_trace[{chunk_index}]"
                ".clean_latent_sha256_per_sample",
            )
            captured_hashes = chunk.get(
                "captured_prefix_clean_latent_sha256_per_sample"
            )
            if capture_frames:
                _hash_list(
                    captured_hashes,
                    "captured_prefix_clean_latent_sha256_per_sample",
                )
            elif captured_hashes != []:
                raise RuntimeError("uncaptured Stage-2 prefix unexpectedly has a hash")
    if expected_mode == "two_action":
        resets = generation.get("reset_events")
        if not isinstance(resets, list) or len(resets) != 1:
            raise ValueError("Stage-2 two-action trace requires one reset event")
        reset = resets[0]
        if not isinstance(reset, Mapping) or set(reset) != {
            "after_episode",
            "retained_sink_frames",
            "cleared_non_sink_self_kv",
            "cleared_cross_kv",
            "next_future_start_frame",
            "prefix_snapshot_restored",
        }:
            raise ValueError("Stage-2 reset trace schema mismatch")
        required_reset = {
            "after_episode": "A",
            "retained_sink_frames": profile.global_sink_frames,
            "cleared_non_sink_self_kv": True,
            "cleared_cross_kv": True,
            "next_future_start_frame": profile.global_sink_frames,
            "prefix_snapshot_restored": profile.global_sink_frames > 1,
        }
        for key, expected in required_reset.items():
            if reset.get(key) != expected:
                raise RuntimeError(
                    f"Stage-2 reset trace {key} mismatch: "
                    f"expected={expected}, actual={reset.get(key)}"
                )
    vae_events = generation.get("vae_events")
    if not isinstance(vae_events, list) or len(vae_events) != expected_episodes:
        raise ValueError("Stage-2 VAE trace event count mismatch")
    expected_labels = ["single"] if expected_episodes == 1 else ["A", "B"]
    for event, label in zip(vae_events, expected_labels):
        if (
            not isinstance(event, Mapping)
            or set(event)
            != {
                "episode",
                "cache_cleared",
                "decode_input_latents",
                "decoded_pixel_frames_with_sink",
                "dropped_pixel_frame_indices",
                "output_pixel_frames",
            }
            or event.get("episode") != label
            or event.get("cache_cleared") is not True
            or event.get("decode_input_latents") != 25
            or event.get("decoded_pixel_frames_with_sink") != 97
            or event.get("dropped_pixel_frame_indices") != [0]
            or event.get("output_pixel_frames") != 96
        ):
            raise RuntimeError("Stage-2 VAE decode/drop trace mismatch")


def build_stage2_sample_trace(
    *,
    sample: Stage2InferenceSample,
    generation_trace: Mapping[str, Any],
    output_root: str | os.PathLike[str],
    video_path: str | os.PathLike[str],
    checkpoint: Mapping[str, Any],
    inference_config: Mapping[str, Any],
    code_version: Mapping[str, Any] | None = None,
    probe_fn: Callable[[str | os.PathLike[str]], Mapping[str, Any]] = probe_video,
) -> dict[str, Any]:
    """Bind one technically validated video to all deterministic inputs."""

    profile = resolve_stage2_rollout_profile(sample.profile)
    _validate_generation_trace(sample, generation_trace, profile)
    checkpoint_identity = _validate_checkpoint_identity(checkpoint)
    config_identity = validate_stage2_inference_config_identity(inference_config)
    root = Path(output_root)
    if root.expanduser().resolve() != Path(config_identity["resolved"]["output_root"]):
        raise RuntimeError("Stage-2 sample trace output root differs from its config")
    _validate_sample_config_scope(sample, config_identity["resolved"])
    path = Path(video_path)
    relative_video = _relative_artifact(
        root,
        path,
        expected=sample.output_relative_path,
    )
    technical = validate_stage2_video_artifact(sample, path, probe_fn=probe_fn)
    profile_value = profile.to_dict()
    profile_value["profile_sha256"] = canonical_json_sha256(profile_value)
    payload = {
        "schema": STAGE2_SAMPLE_TRACE_SCHEMA,
        "status": "pass",
        "sample": sample.to_manifest_source(),
        "raw_prompt_sha256": [_raw_prompt_sha256(item) for item in sample.prompts],
        "checkpoint": checkpoint_identity,
        "inference_config": config_identity,
        "code_version": {} if code_version is None else code_version,
        "profile": profile_value,
        "generation": dict(generation_trace),
        "output": {
            "relative_path": relative_video,
            **technical,
        },
        "quality_metrics": None,
    }
    payload["trace_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_stage2_sample_trace(
    trace: Mapping[str, Any],
    *,
    sample: Stage2InferenceSample,
    inference_config: Mapping[str, Any] | None = None,
    code_version: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate content and inputs; code_version is historical metadata only."""

    expected_keys = {
        "schema",
        "status",
        "sample",
        "raw_prompt_sha256",
        "checkpoint",
        "inference_config",
        "profile",
        "generation",
        "output",
        "quality_metrics",
        "trace_sha256",
    }
    if not isinstance(trace, Mapping) or set(trace) - {"code_version"} != expected_keys:
        raise ValueError("Stage-2 sample trace top-level schema mismatch")
    if (
        trace.get("schema") != STAGE2_SAMPLE_TRACE_SCHEMA
        or trace.get("status") != "pass"
    ):
        raise ValueError("Stage-2 sample trace schema/status mismatch")
    if trace.get("sample") != sample.to_manifest_source():
        raise RuntimeError("Stage-2 sample trace source differs from its sample key")
    if trace.get("raw_prompt_sha256") != [
        _raw_prompt_sha256(item) for item in sample.prompts
    ]:
        raise RuntimeError("Stage-2 raw prompt hash mismatch")
    if trace.get("quality_metrics") is not None:
        raise RuntimeError("Stage-2 automatic visual quality metrics are forbidden")
    forbidden = _forbidden_quality_key(
        {key: value for key, value in trace.items() if key != "code_version"}
    )
    if forbidden is not None:
        raise RuntimeError(
            f"Stage-2 trace contains a forbidden quality metric key: {forbidden}"
        )
    without_hash = dict(trace)
    claimed = without_hash.pop("trace_sha256")
    _sha256(claimed, "trace_sha256")
    if claimed != canonical_json_sha256(without_hash):
        raise RuntimeError("Stage-2 sample trace self hash mismatch")
    profile = resolve_stage2_rollout_profile(sample.profile)
    expected_profile = profile.to_dict()
    expected_profile = {
        **expected_profile,
        "profile_sha256": canonical_json_sha256(expected_profile),
    }
    if trace.get("profile") != expected_profile:
        raise RuntimeError("Stage-2 sample trace rollout profile mismatch")
    _validate_checkpoint_identity(trace.get("checkpoint"))
    config_identity = validate_stage2_inference_config_identity(
        trace.get("inference_config")
    )
    if inference_config is not None and config_identity != (
        validate_stage2_inference_config_identity(inference_config)
    ):
        raise RuntimeError("Stage-2 sample trace inference config mismatch")
    _validate_sample_config_scope(sample, config_identity["resolved"])
    _validate_generation_trace(sample, trace["generation"], profile)
    output = trace.get("output")
    if not isinstance(output, Mapping) or set(output) != {
        "relative_path",
        "width",
        "height",
        "frame_count",
        "fps",
        "size",
        "sha256",
    }:
        raise ValueError("Stage-2 sample trace output schema mismatch")
    expected_output = {
        "relative_path": sample.output_relative_path.as_posix(),
        "width": sample.width,
        "height": sample.height,
        "frame_count": _expected_frames(sample),
        "fps": float(STAGE2_INFERENCE_FPS),
    }
    for key, expected in expected_output.items():
        if output.get(key) != expected:
            raise RuntimeError(
                f"Stage-2 sample trace output {key} mismatch: "
                f"expected={expected}, actual={output.get(key)}"
            )
    _plain_int(output.get("size"), "output.size", minimum=1)
    _sha256(output.get("sha256"), "output.sha256")
    return dict(trace)


def write_stage2_sample_trace(
    output_root: str | os.PathLike[str],
    *,
    sample: Stage2InferenceSample,
    trace: Mapping[str, Any],
) -> Path:
    validated = validate_stage2_sample_trace(trace, sample=sample)
    path = Path(output_root) / sample.trace_relative_path
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    atomic_write_json(path, validated)
    return path


def _artifact_entry(
    output_root: Path,
    *,
    sample: Stage2InferenceSample,
    trace: Mapping[str, Any],
    trace_path: Path,
) -> dict[str, Any]:
    relative_trace = _relative_artifact(
        output_root,
        trace_path,
        expected=sample.trace_relative_path,
    )
    if trace_path.is_symlink() or not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    try:
        disk_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid committed Stage-2 trace: {trace_path}") from exc
    if disk_trace != trace:
        raise RuntimeError(
            f"committed Stage-2 trace differs from validated memory state: {trace_path}"
        )
    video_path = output_root / sample.output_relative_path
    if video_path.is_symlink() or not video_path.is_file():
        raise FileNotFoundError(video_path)
    video = trace["output"]
    if (
        video_path.stat().st_size != video["size"]
        or sha256_file(video_path) != video["sha256"]
    ):
        raise RuntimeError(
            f"committed Stage-2 video differs from its trace: {video_path}"
        )
    return {
        "sample_key": sample.sample_key,
        "dataset": sample.dataset,
        "row_id": sample.row_id,
        "seed": sample.seed,
        "profile": sample.profile,
        "video": dict(trace["output"]),
        "trace": {
            "relative_path": relative_trace,
            "size": trace_path.stat().st_size,
            "sha256": sha256_file(trace_path),
            "content_sha256": trace["trace_sha256"],
        },
    }


def _artifact_files(root: Path, directory: str) -> set[str]:
    base = root / directory
    if base.is_symlink():
        raise RuntimeError(f"Stage-2 artifact directory may not be a symlink: {base}")
    if not base.exists():
        return set()
    if not base.is_dir():
        raise RuntimeError(f"Stage-2 artifact root is not a directory: {base}")
    result: set[str] = set()
    for child in base.rglob("*"):
        if child.is_symlink():
            raise RuntimeError(f"Stage-2 artifact may not be a symlink: {child}")
        if child.is_file():
            result.add(child.relative_to(root).as_posix())
        elif not child.is_dir():
            raise RuntimeError(f"unsupported Stage-2 artifact type: {child}")
    return result


def validate_stage2_inference_artifact_set(
    output_root: str | os.PathLike[str],
    *,
    samples: Sequence[Stage2InferenceSample],
) -> None:
    """Require exactly the planned MP4 and JSON trace files, with no extras."""

    root = Path(output_root).expanduser().resolve()
    expected_videos = {sample.output_relative_path.as_posix() for sample in samples}
    expected_traces = {sample.trace_relative_path.as_posix() for sample in samples}
    actual_videos = _artifact_files(root, "videos")
    actual_traces = _artifact_files(root, "traces")
    if actual_videos != expected_videos:
        raise RuntimeError(
            "Stage-2 video artifact set mismatch: "
            f"missing={sorted(expected_videos - actual_videos)}, "
            f"extra={sorted(actual_videos - expected_videos)}"
        )
    if actual_traces != expected_traces:
        raise RuntimeError(
            "Stage-2 trace artifact set mismatch: "
            f"missing={sorted(expected_traces - actual_traces)}, "
            f"extra={sorted(actual_traces - expected_traces)}"
        )


def _validate_metadata_identity(
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if set(metadata) != {STAGE2_SINGLE_DATASET, STAGE2_TWO_ACTION_DATASET}:
        raise ValueError("Stage-2 inference metadata identity set mismatch")
    validated: dict[str, dict[str, Any]] = {}
    for dataset, identity in metadata.items():
        if not isinstance(identity, Mapping) or set(identity) != {"path", "sha256"}:
            raise ValueError(f"Stage-2 {dataset} metadata identity schema mismatch")
        path_value = identity["path"]
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"Stage-2 {dataset} metadata path is invalid")
        source = Path(path_value).expanduser()
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(source)
        path = source.resolve()
        claimed = _sha256(identity["sha256"], f"metadata.{dataset}.sha256")
        if sha256_file(path) != claimed:
            raise RuntimeError(f"Stage-2 {dataset} metadata SHA-256 mismatch")
        validated[dataset] = {"path": os.fspath(path), "sha256": claimed}
    return validated


def _safe_manifest_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} must stay under the Stage-2 output root")
    return value


def validate_stage2_inference_manifest(
    manifest: Mapping[str, Any],
    *,
    inference_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "status",
        "checkpoint",
        "inference_config",
        "metadata",
        "seeds",
        "profiles",
        "expected_sample_count",
        "samples",
        "quality_metrics",
        "manifest_sha256",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) - {"code_version"} != expected_keys
    ):
        raise ValueError("Stage-2 inference manifest top-level schema mismatch")
    if (
        manifest.get("schema") != STAGE2_INFERENCE_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Stage-2 inference manifest schema/status mismatch")
    if manifest.get("quality_metrics") is not None:
        raise RuntimeError("Stage-2 manifest may not contain automatic quality metrics")
    forbidden = _forbidden_quality_key(
        {key: value for key, value in manifest.items() if key != "code_version"}
    )
    if forbidden is not None:
        raise RuntimeError(
            f"Stage-2 manifest contains a forbidden quality metric key: {forbidden}"
        )
    _validate_checkpoint_identity(manifest.get("checkpoint"))
    config_identity = validate_stage2_inference_config_identity(
        manifest.get("inference_config")
    )
    if inference_config is not None and config_identity != (
        validate_stage2_inference_config_identity(inference_config)
    ):
        raise RuntimeError("Stage-2 manifest inference config mismatch")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != {
        STAGE2_SINGLE_DATASET,
        STAGE2_TWO_ACTION_DATASET,
    }:
        raise ValueError("Stage-2 inference manifest metadata schema mismatch")
    for dataset, identity in metadata.items():
        if not isinstance(identity, Mapping) or set(identity) != {"path", "sha256"}:
            raise ValueError(f"Stage-2 {dataset} metadata identity schema mismatch")
        if not isinstance(identity["path"], str) or not identity["path"]:
            raise ValueError(f"Stage-2 {dataset} metadata path is invalid")
        _sha256(identity["sha256"], f"metadata.{dataset}.sha256")
    seeds = manifest.get("seeds")
    expected_seeds = config_identity["resolved"]["seeds"]
    if (
        not isinstance(seeds, list)
        or seeds != expected_seeds
        or any(type(seed) is not int for seed in seeds)
    ):
        raise ValueError("Stage-2 inference manifest seed set mismatch")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Stage-2 inference manifest profiles are missing")
    profile_names: list[str] = []
    for item in profiles:
        if not isinstance(item, Mapping) or "name" not in item:
            raise ValueError("Stage-2 inference manifest profile schema mismatch")
        spec = resolve_stage2_rollout_profile(item["name"])
        expected = spec.to_dict()
        expected = {**expected, "profile_sha256": canonical_json_sha256(expected)}
        if item != expected:
            raise RuntimeError(f"Stage-2 profile identity mismatch: {spec.name}")
        profile_names.append(spec.name)
    if len(profile_names) != len(set(profile_names)):
        raise RuntimeError("Stage-2 inference manifest repeats a profile")
    if profile_names != config_identity["resolved"]["profiles"]:
        raise RuntimeError("Stage-2 inference manifest profiles differ from its config")
    expected_coordinates = _expected_sample_coordinates(config_identity["resolved"])
    count = _plain_int(
        manifest.get("expected_sample_count"),
        "expected_sample_count",
        minimum=1,
    )
    if count != len(expected_coordinates):
        raise RuntimeError(
            "Stage-2 inference manifest count differs from its resolved sample matrix"
        )
    entries = manifest.get("samples")
    if not isinstance(entries, list) or len(entries) != count:
        raise RuntimeError("Stage-2 inference manifest sample count mismatch")
    keys: list[str] = []
    coordinates: list[tuple[str, str, int, int]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or set(entry) != {
            "sample_key",
            "dataset",
            "row_id",
            "seed",
            "profile",
            "video",
            "trace",
        }:
            raise ValueError(f"Stage-2 manifest samples[{index}] schema mismatch")
        sample_key = entry["sample_key"]
        if not isinstance(sample_key, str) or not sample_key:
            raise ValueError(f"Stage-2 manifest samples[{index}].sample_key is invalid")
        keys.append(sample_key)
        if entry["dataset"] not in {STAGE2_SINGLE_DATASET, STAGE2_TWO_ACTION_DATASET}:
            raise ValueError(f"Stage-2 manifest samples[{index}].dataset is invalid")
        row_id = _plain_int(entry["row_id"], f"samples[{index}].row_id")
        if entry["seed"] not in expected_seeds or isinstance(entry["seed"], bool):
            raise ValueError(f"Stage-2 manifest samples[{index}].seed is invalid")
        if config_identity["resolved"]["schema"] == (
            STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA
        ):
            row_field = (
                "single_row_ids"
                if entry["dataset"] == STAGE2_SINGLE_DATASET
                else "two_action_row_ids"
            )
            if row_id not in config_identity["resolved"][row_field]:
                raise ValueError(
                    f"Stage-2 manifest samples[{index}].row_id is outside its config"
                )
        if entry["profile"] not in profile_names:
            raise RuntimeError(f"Stage-2 manifest samples[{index}] profile is unknown")
        coordinate = (
            entry["profile"],
            entry["dataset"],
            row_id,
            entry["seed"],
        )
        coordinates.append(coordinate)
        expected_sample_key = (
            f"{entry['dataset']}/row{row_id:03d}/"
            f"seed{entry['seed']:04d}/{entry['profile']}"
        )
        if sample_key != expected_sample_key:
            raise RuntimeError(
                f"Stage-2 manifest samples[{index}].sample_key is non-canonical"
            )
        video = entry["video"]
        if not isinstance(video, Mapping) or set(video) != {
            "relative_path",
            "width",
            "height",
            "frame_count",
            "fps",
            "size",
            "sha256",
        }:
            raise ValueError(f"Stage-2 manifest samples[{index}].video schema mismatch")
        video_path = _safe_manifest_relative_path(
            video["relative_path"], f"samples[{index}].video.relative_path"
        )
        if not video_path.startswith("videos/"):
            raise ValueError("Stage-2 manifest video must be under videos/")
        _plain_int(video["width"], f"samples[{index}].video.width", minimum=1)
        _plain_int(video["height"], f"samples[{index}].video.height", minimum=1)
        _plain_int(
            video["frame_count"], f"samples[{index}].video.frame_count", minimum=1
        )
        fps = video["fps"]
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or float(fps) != float(STAGE2_INFERENCE_FPS)
        ):
            raise ValueError(f"Stage-2 manifest samples[{index}].video.fps mismatch")
        _plain_int(video["size"], f"samples[{index}].video.size", minimum=1)
        _sha256(video["sha256"], f"samples[{index}].video.sha256")
        trace = entry["trace"]
        if not isinstance(trace, Mapping) or set(trace) != {
            "relative_path",
            "size",
            "sha256",
            "content_sha256",
        }:
            raise ValueError(f"Stage-2 manifest samples[{index}].trace schema mismatch")
        trace_path = _safe_manifest_relative_path(
            trace["relative_path"], f"samples[{index}].trace.relative_path"
        )
        if not trace_path.startswith("traces/"):
            raise ValueError("Stage-2 manifest trace must be under traces/")
        _plain_int(trace["size"], f"samples[{index}].trace.size", minimum=1)
        _sha256(trace["sha256"], f"samples[{index}].trace.sha256")
        _sha256(
            trace["content_sha256"],
            f"samples[{index}].trace.content_sha256",
        )
    if len(keys) != len(set(keys)):
        raise RuntimeError("Stage-2 inference manifest repeats a sample key")
    if coordinates != expected_coordinates:
        raise RuntimeError(
            "Stage-2 inference manifest samples differ from its resolved Cartesian plan"
        )
    without_hash = dict(manifest)
    claimed = without_hash.pop("manifest_sha256")
    _sha256(claimed, "manifest_sha256")
    if claimed != canonical_json_sha256(without_hash):
        raise RuntimeError("Stage-2 inference manifest self hash mismatch")
    return dict(manifest)


def validate_stage2_inference_manifest_artifacts(
    output_root: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    *,
    expected_checkpoint: Mapping[str, Any] | None = None,
    expected_resolved_config: Mapping[str, Any] | None = None,
    expected_samples: Sequence[Stage2InferenceSample] | None = None,
) -> dict[str, Any]:
    """Re-authenticate every committed file referenced by a final manifest."""

    candidate = Path(output_root).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(
            f"Stage-2 inference output root is not a regular directory: {candidate}"
        )
    root = candidate.resolve()
    validated = validate_stage2_inference_manifest(manifest)
    config_identity = validate_stage2_inference_config_identity(
        validated["inference_config"]
    )
    if expected_resolved_config is not None:
        try:
            expected_config = json.loads(
                json.dumps(expected_resolved_config, allow_nan=False, sort_keys=True)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "expected Stage-2 inference config is not strict JSON"
            ) from exc
        if config_identity["resolved"] != expected_config:
            raise RuntimeError(
                "Stage-2 manifest resolved config differs from current config"
            )
    if Path(config_identity["resolved"]["output_root"]).resolve() != root:
        raise RuntimeError("Stage-2 manifest output root differs from committed root")
    checkpoint_identity = _validate_checkpoint_identity(validated["checkpoint"])
    if expected_checkpoint is not None and checkpoint_identity != (
        _validate_checkpoint_identity(expected_checkpoint)
    ):
        raise RuntimeError(
            "Stage-2 manifest checkpoint differs from expected checkpoint"
        )
    _validate_metadata_identity(validated["metadata"])

    samples: list[Stage2InferenceSample] = []
    trace_by_key: dict[str, dict[str, Any]] = {}
    entry_by_key: dict[str, Mapping[str, Any]] = {}
    for entry in validated["samples"]:
        trace_path = root / entry["trace"]["relative_path"]
        if trace_path.is_symlink() or not trace_path.is_file():
            raise RuntimeError(
                f"Stage-2 committed trace is not a regular file: {trace_path}"
            )
        if (
            trace_path.stat().st_size != entry["trace"]["size"]
            or sha256_file(trace_path) != entry["trace"]["sha256"]
        ):
            raise RuntimeError(
                f"Stage-2 committed trace hash/size mismatch: {trace_path}"
            )
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid committed Stage-2 trace: {trace_path}"
            ) from exc
        source = trace.get("sample")
        if not isinstance(source, Mapping):
            raise TypeError("Stage-2 committed trace has no sample source")
        sample = Stage2InferenceSample(
            dataset=source["dataset"],
            row_id=source["row_id"],
            seed=source["seed"],
            profile=source["profile"],
            input_image=Path(source["input_image"]),
            image_sha256=source["image_sha256"],
            row_sha256=source["row_sha256"],
            height=source["height"],
            width=source["width"],
            bucket=source["bucket"],
            prompts=tuple(source["prompts"]),
            case_group=source["case_group"],
        )
        if sample.sample_key != entry["sample_key"]:
            raise RuntimeError("Stage-2 trace sample key differs from final manifest")
        checked_trace = validate_stage2_sample_trace(
            trace,
            sample=sample,
            inference_config=config_identity,
        )
        if checked_trace["checkpoint"] != checkpoint_identity:
            raise RuntimeError("Stage-2 trace checkpoint differs from final manifest")
        if checked_trace["trace_sha256"] != entry["trace"]["content_sha256"]:
            raise RuntimeError("Stage-2 trace content hash differs from final manifest")
        if checked_trace["output"] != entry["video"]:
            raise RuntimeError("Stage-2 trace video claim differs from final manifest")
        samples.append(sample)
        trace_by_key[sample.sample_key] = checked_trace
        entry_by_key[sample.sample_key] = entry

    if expected_samples is not None:
        planned = tuple(expected_samples)
        if [sample.to_manifest_source() for sample in samples] != [
            sample.to_manifest_source() for sample in planned
        ]:
            raise RuntimeError(
                "Stage-2 manifest sample plan differs from current metadata/config"
            )
        samples = list(planned)

    validate_stage2_inference_artifact_set(root, samples=samples)
    for sample in samples:
        entry = entry_by_key[sample.sample_key]
        video = root / sample.output_relative_path
        if video.is_symlink() or not video.is_file():
            raise RuntimeError(
                f"Stage-2 committed video is not a regular file: {video}"
            )
        if (
            video.stat().st_size != entry["video"]["size"]
            or sha256_file(video) != entry["video"]["sha256"]
        ):
            raise RuntimeError(f"Stage-2 committed video hash/size mismatch: {video}")
        if trace_by_key[sample.sample_key]["output"] != entry["video"]:
            raise RuntimeError("Stage-2 committed video differs from its trace")

    index = root / STAGE2_REVIEW_INDEX_NAME
    if index.is_symlink() or not index.is_file():
        raise RuntimeError("Stage-2 review index is missing or invalid")
    expected_index = _render_stage2_review_index(validated).encode("utf-8")
    if index.read_bytes() != expected_index:
        raise RuntimeError("Stage-2 review index differs from final manifest")
    return validated


def build_stage2_inference_manifest(
    *,
    output_root: str | os.PathLike[str],
    samples: Sequence[Stage2InferenceSample],
    traces: Mapping[str, Mapping[str, Any]],
    trace_paths: Mapping[str, str | os.PathLike[str]],
    checkpoint: Mapping[str, Any],
    inference_config: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
    code_version: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete manifest only after every expected trace is committed."""

    root = Path(output_root).expanduser().resolve()
    sample_keys = [sample.sample_key for sample in samples]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("Stage-2 inference manifest received duplicate sample keys")
    if set(traces) != set(sample_keys) or set(trace_paths) != set(sample_keys):
        raise RuntimeError(
            "Stage-2 inference manifest trace set is incomplete or extra"
        )
    validate_stage2_inference_artifact_set(root, samples=samples)
    checkpoint_identity = _validate_checkpoint_identity(checkpoint)
    config_identity = validate_stage2_inference_config_identity(inference_config)
    if root != Path(config_identity["resolved"]["output_root"]):
        raise RuntimeError("Stage-2 manifest output root differs from its config")
    metadata_identity = _validate_metadata_identity(metadata)
    entries = []
    for sample in samples:
        trace = validate_stage2_sample_trace(
            traces[sample.sample_key],
            sample=sample,
            inference_config=config_identity,
        )
        if trace["checkpoint"] != checkpoint_identity:
            raise RuntimeError(
                f"Stage-2 sample {sample.sample_key} uses a different checkpoint"
            )
        entries.append(
            _artifact_entry(
                root,
                sample=sample,
                trace=trace,
                trace_path=Path(trace_paths[sample.sample_key]),
            )
        )
    profile_names = tuple(dict.fromkeys(sample.profile for sample in samples))
    if list(profile_names) != config_identity["resolved"]["profiles"]:
        raise RuntimeError("Stage-2 manifest profiles differ from its config")
    profiles = []
    for name in profile_names:
        value = resolve_stage2_rollout_profile(name).to_dict()
        profiles.append({**value, "profile_sha256": canonical_json_sha256(value)})
    payload = {
        "schema": STAGE2_INFERENCE_MANIFEST_SCHEMA,
        "status": "complete",
        "code_version": {} if code_version is None else code_version,
        "checkpoint": checkpoint_identity,
        "inference_config": config_identity,
        "metadata": {
            key: dict(value) for key, value in sorted(metadata_identity.items())
        },
        "seeds": list(config_identity["resolved"]["seeds"]),
        "profiles": profiles,
        "expected_sample_count": len(samples),
        "samples": entries,
        "quality_metrics": None,
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return validate_stage2_inference_manifest(payload)


def write_stage2_inference_manifest(
    output_root: str | os.PathLike[str],
    *,
    manifest: Mapping[str, Any],
) -> Path:
    validated = validate_stage2_inference_manifest(manifest)
    path = Path(output_root) / STAGE2_INFERENCE_MANIFEST_NAME
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    atomic_write_json(path, validated)
    return path


def _render_stage2_review_index(manifest: Mapping[str, Any]) -> str:
    validated = validate_stage2_inference_manifest(manifest)
    rows = []
    for entry in validated["samples"]:
        video = entry["video"]["relative_path"]
        trace = entry["trace"]["relative_path"]
        rows.append(
            "<tr>"
            f"<td>{escape(entry['profile'])}</td>"
            f"<td>{escape(entry['dataset'])}</td>"
            f"<td>{entry['row_id']}</td><td>{entry['seed']}</td>"
            f"<td><video controls preload='metadata' src='{quote(video)}'></video></td>"
            f"<td><a href='{quote(trace)}'>trace</a></td>"
            "</tr>"
        )
    style = (
        "body{font-family:system-ui;margin:24px;background:#111;color:#eee}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #444;"
        "padding:8px;vertical-align:top}video{width:min(360px,32vw);height:auto}"
        "a{color:#8cf}"
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Stage-2 inference review</title>"
        f"<style>{style}</style></head><body>"
        "<h1>Stage-2 inference review</h1>"
        "<p>仅展示技术产物；动作、身份、颜色、毛发和跨动作污染由人工判断。</p>"
        "<table><thead><tr><th>profile</th><th>dataset</th><th>row</th>"
        "<th>seed</th><th>video</th><th>trace</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


def write_stage2_review_index(
    output_root: str | os.PathLike[str],
    *,
    manifest: Mapping[str, Any],
) -> Path:
    root = Path(output_root)
    html = _render_stage2_review_index(manifest)
    path = root / STAGE2_REVIEW_INDEX_NAME
    atomic_write_bytes(path, html.encode("utf-8"))
    return path


__all__ = [
    "STAGE2_INFERENCE_CONFIG_IDENTITY_SCHEMA",
    "STAGE2_INFERENCE_MANIFEST_NAME",
    "STAGE2_INFERENCE_MANIFEST_SCHEMA",
    "STAGE2_REVIEW_INDEX_NAME",
    "STAGE2_SAMPLE_TRACE_SCHEMA",
    "build_stage2_inference_config_identity",
    "build_stage2_inference_manifest",
    "build_stage2_sample_trace",
    "validate_stage2_inference_artifact_set",
    "validate_stage2_inference_config_identity",
    "validate_stage2_inference_manifest",
    "validate_stage2_inference_manifest_artifacts",
    "validate_stage2_sample_trace",
    "validate_stage2_video_artifact",
    "write_stage2_inference_manifest",
    "write_stage2_review_index",
    "write_stage2_sample_trace",
]
