from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.stage2_rollout_profile import (
    Stage2RolloutSpec,
    build_stage2_deployment_rollout_spec,
    resolve_stage2_rollout_profile,
    resolve_stage2_shift5_schedule,
)
from utils.stage1_io import canonical_json_sha256, sha256_file
from utils.stage2_inference import (
    STAGE2_INFERENCE_TRACE_SCHEMA,
    STAGE2_NOISE_STREAM_POLICY,
)
from utils.stage2_inference_artifacts import (
    build_stage2_inference_config_identity,
    build_stage2_inference_manifest,
    build_stage2_sample_trace,
    validate_stage2_inference_artifact_set,
    validate_stage2_inference_config_identity,
    validate_stage2_inference_manifest,
    validate_stage2_inference_manifest_artifacts,
    validate_stage2_sample_trace,
    validate_stage2_video_artifact,
    write_stage2_inference_manifest,
    write_stage2_review_index,
    write_stage2_sample_trace,
)
from utils.stage2_inference_assets import STAGE2_RUNTIME_ASSET_IDENTITY_SCHEMA
from utils.stage2_inference_batch import (
    STAGE2_BASELINE_PROFILE,
    STAGE2_SINGLE_DATASET,
    STAGE2_TWO_ACTION_DATASET,
    build_stage2_inference_samples,
)
from utils.stage2_inference_config import ResolvedStage2InferenceConfig
from utils.stage2_inference_sweep_config import (
    STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA,
    ResolvedStage2InferenceSweepConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_METADATA = PROJECT_ROOT / "testsets" / "metadata_6cases_480x832.csv"
TWO_METADATA = (
    PROJECT_ROOT
    / "testsets"
    / "metadata_8cases_two_actions_continuation_480x832_253frames.csv"
)
CHECKPOINT = {
    "directory": "/checkpoints/checkpoint_stage2_g000280",
    "manifest_sha256": "1" * 64,
    "completed_generator_updates": 280,
    "contract_hash": "2" * 64,
    "generator_ema_sha256": "3" * 64,
}
CODE_VERSION = {"stage2_source_sha256": "a" * 64}


def _runtime_asset_identity(
    config: ResolvedStage2InferenceConfig | ResolvedStage2InferenceSweepConfig,
) -> dict[str, object]:
    paths = {
        "t5_checkpoint": config.t5_checkpoint,
        "tokenizer_dir": config.tokenizer_dir,
        "vae_checkpoint": config.vae_checkpoint,
        "architecture_config": str(Path(config.architecture_root) / "config.json"),
        "generator_base": "/models/generator-stage1-step3075.pt",
    }
    body = {
        "schema": STAGE2_RUNTIME_ASSET_IDENTITY_SCHEMA,
        "checkpoint": {
            "directory": config.stage2_checkpoint,
            "manifest_sha256": CHECKPOINT["manifest_sha256"],
            "generator_recorded_asset_sha256": "4" * 64,
        },
        "source_manifest": {
            "path": config.source_cache_manifest,
            "manifest_sha256": "5" * 64,
            "file_sha256": "6" * 64,
            "size": 1,
        },
        "assets": {
            name: {
                "path": path,
                "aggregate_sha256": f"{index:x}" * 64,
                "file_count": 1,
                "total_size": 1,
            }
            for index, (name, path) in enumerate(paths.items(), start=7)
        },
    }
    return {**body, "identity_sha256": canonical_json_sha256(body)}


def _resolved_inference_config(
    output_root: Path,
    *,
    profiles: tuple[str, ...] = (STAGE2_BASELINE_PROFILE,),
    **overrides: object,
) -> ResolvedStage2InferenceConfig:
    config = ResolvedStage2InferenceConfig(
        schema="longlive_stage2_inference/v1",
        stage2_checkpoint=CHECKPOINT["directory"],
        source_cache_manifest="/source/stage2-cache-manifest.json",
        architecture_root="/models/wan",
        t5_checkpoint="/models/t5.pth",
        tokenizer_dir="/models/tokenizer",
        vae_checkpoint="/models/vae.pth",
        single_metadata=str(SINGLE_METADATA.resolve()),
        two_action_metadata=str(TWO_METADATA.resolve()),
        output_root=str(output_root.resolve()),
        profiles=profiles,
        seeds=(1, 2, 3, 4),
        dtype="bfloat16",
        cfg_scale=1.0,
        fps=24,
        merge_ema_lora=True,
        batch_size_per_device=1,
    )
    return replace(config, **overrides)


def _inference_config(
    output_root: Path,
    *,
    profiles: tuple[str, ...] = (STAGE2_BASELINE_PROFILE,),
    **overrides: object,
) -> dict[str, object]:
    config = _resolved_inference_config(
        output_root,
        profiles=profiles,
        **overrides,
    )
    return build_stage2_inference_config_identity(
        config,
        runtime_assets=_runtime_asset_identity(config),
    )


def _resolved_sweep_config(
    output_root: Path,
    *,
    specs: tuple[Stage2RolloutSpec, ...],
    seeds: tuple[int, ...] = (9,),
    evaluation_mode: str = "quick",
    single_row_ids: tuple[int, ...] = (0,),
    two_action_row_ids: tuple[int, ...] = (0,),
) -> ResolvedStage2InferenceSweepConfig:
    base = _resolved_inference_config(output_root)
    return ResolvedStage2InferenceSweepConfig(
        schema=STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA,
        base_config_path="/configs/stage2_inference.yaml",
        base_config_sha256="d" * 64,
        stage2_checkpoint=base.stage2_checkpoint,
        source_cache_manifest=base.source_cache_manifest,
        architecture_root=base.architecture_root,
        t5_checkpoint=base.t5_checkpoint,
        tokenizer_dir=base.tokenizer_dir,
        vae_checkpoint=base.vae_checkpoint,
        single_metadata=base.single_metadata,
        two_action_metadata=base.two_action_metadata,
        output_root=base.output_root,
        profiles=tuple(spec.name for spec in specs),
        seeds=seeds,
        dtype=base.dtype,
        cfg_scale=base.cfg_scale,
        fps=base.fps,
        merge_ema_lora=base.merge_ema_lora,
        batch_size_per_device=base.batch_size_per_device,
        evaluation_mode=evaluation_mode,
        single_row_ids=single_row_ids,
        two_action_row_ids=two_action_row_ids,
        profile_set_sha256=canonical_json_sha256([spec.to_dict() for spec in specs]),
        _rollout_specs=specs,
    )


def _sweep_inference_config(
    output_root: Path,
    *,
    specs: tuple[Stage2RolloutSpec, ...],
    seeds: tuple[int, ...] = (9,),
) -> dict[str, object]:
    config = _resolved_sweep_config(output_root, specs=specs, seeds=seeds)
    return build_stage2_inference_config_identity(
        config,
        runtime_assets=_runtime_asset_identity(config),
    )


def _rehash_config_identity(identity: dict[str, object]) -> None:
    resolved = identity["resolved"]
    assert isinstance(resolved, dict)
    runtime_assets = identity["runtime_assets"]
    contract_resolved = dict(resolved)
    contract_resolved.pop("output_root")
    identity["resolved_contract_hash"] = canonical_json_sha256(contract_resolved)
    identity["resolved_launch_hash"] = canonical_json_sha256(resolved)
    identity["runtime_contract_hash"] = canonical_json_sha256(
        {"resolved": contract_resolved, "runtime_assets": runtime_assets}
    )
    identity["runtime_launch_hash"] = canonical_json_sha256(
        {"resolved": resolved, "runtime_assets": runtime_assets}
    )


def _samples():
    return build_stage2_inference_samples(
        single_metadata=SINGLE_METADATA,
        two_action_metadata=TWO_METADATA,
    )


def _generation(sample, *, profile_name=None, profile_spec=None):
    profile = profile_spec or resolve_stage2_rollout_profile(
        profile_name or sample.profile
    )
    count = 1 if sample.dataset == STAGE2_SINGLE_DATASET else 2
    episodes = []
    frame_tokens = (sample.height // 32) * (sample.width // 32)
    for episode_index in range(count):
        actual_profile = profile
        if profile.global_sink_frames > 1 and episode_index == 0:
            actual_profile = resolve_stage2_rollout_profile("baseline_c8w16k4s1")
        steps = actual_profile.num_denoising_steps
        schedule = resolve_stage2_shift5_schedule(steps)
        timetable = list(schedule.timesteps)
        sigmas = list(schedule.sigmas)
        chunks = actual_profile.num_chunks
        preload = int(episode_index == 0)
        noisy_calls = chunks * steps
        recache_calls = chunks
        capacity = actual_profile.physical_kv_capacity_frames
        chunk_trace = []
        for chunk_index in range(chunks):
            start = chunk_index * actual_profile.chunk_frames
            before_frames = min(capacity, actual_profile.global_sink_frames + start)
            after_frames = min(
                capacity,
                actual_profile.global_sink_frames + start + actual_profile.chunk_frames,
            )
            capture_frames = (
                profile.global_sink_frames
                if count == 2
                and profile.global_sink_frames > 1
                and episode_index == 0
                and chunk_index == 0
                else 0
            )
            chunk_trace.append(
                {
                    "profile_name": actual_profile.name,
                    "episode_index": episode_index,
                    "chunk_index": chunk_index,
                    "generated_frame_start": start,
                    "generated_frame_end_exclusive": start
                    + actual_profile.chunk_frames,
                    "rope_frame_start": actual_profile.global_sink_frames + start,
                    "rope_frame_end_exclusive": actual_profile.global_sink_frames
                    + start
                    + actual_profile.chunk_frames,
                    "timestep_count": steps,
                    "exit_step": steps - 1,
                    "denoising_forward_calls": steps,
                    "solver_update_calls": steps - 1,
                    "clean_recache_forward_calls": 1,
                    "rollout_mode": "full_denoising",
                    "timesteps": timetable,
                    "sigmas": sigmas[:-1],
                    "terminal_sigma": 0.0,
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
                    "cache_capacity_frames": capacity,
                    "cache_capacity_tokens": capacity * frame_tokens,
                    "sink_frames": actual_profile.global_sink_frames,
                    "initial_latent_sha256_per_sample": ["4" * 64],
                    "clean_latent_sha256_per_sample": ["5" * 64],
                    "captured_prefix_sink_frames": capture_frames,
                    "captured_prefix_clean_latent_sha256_per_sample": (
                        ["c" * 64] if capture_frames else []
                    ),
                }
            )
        episodes.append(
            {
                "exit_step": steps - 1,
                "requires_grad": False,
                "scheduler_timesteps": timetable,
                "scheduler_sigmas": sigmas,
                "chunk_timesteps": [timetable] * chunks,
                "chunk_sigmas": [sigmas[:-1]] * chunks,
                "cache_audit": {
                    "layers": 40,
                    "capacity_frames": capacity,
                    "capacity_tokens": capacity * frame_tokens,
                    "global_end_index": (actual_profile.global_sink_frames + 24)
                    * frame_tokens,
                    "local_end_index": capacity * frame_tokens,
                    "persistent_kv_detached": True,
                    "conditional_cache_branches": 1,
                    "cross_kv_active": True,
                    "cross_kv_initialized": True,
                    "profile": actual_profile.to_dict(),
                    "scheduler_instances": chunks,
                    "solver_update_calls": chunks * (steps - 1),
                    "terminal_x0_direct": True,
                    "noisy_forward_calls": noisy_calls,
                    "clean_recache_forward_calls": recache_calls,
                    "sink_preload_forward_calls": preload,
                    "generator_forward_calls": preload + noisy_calls + recache_calls,
                    "logical_query_tokens": (
                        preload * actual_profile.global_sink_frames * frame_tokens
                        + (noisy_calls + recache_calls)
                        * actual_profile.chunk_frames
                        * frame_tokens
                    ),
                },
                "rollout_mode": "full_denoising",
                "chunk_trace": chunk_trace,
                "initial_latent_sha256_per_sample": ["4" * 64],
                "generated_latent_sha256_per_sample": ["5" * 64],
            }
        )
    trace = {
        "schema": STAGE2_INFERENCE_TRACE_SCHEMA,
        "mode": (
            "single_action" if sample.dataset == STAGE2_SINGLE_DATASET else "two_action"
        ),
        "seeds": [sample.seed],
        "noise_stream": {
            "policy": STAGE2_NOISE_STREAM_POLICY,
            "rng_initializations_per_sample": 1,
            "episode_slots": 24,
            "episode_order": ["single"] if count == 1 else ["A", "B"],
        },
        "initial_latent_sha256": ["4" * 64],
        "noise_plan_sha256": ["6" * 64],
        "noise_episode_sha256": [
            [("7" if index == 0 else "8") * 64] for index in range(count)
        ],
        "prompt_embedding_sha256": [
            [("9" if index == 0 else "a") * 64] for index in range(count)
        ],
        "episodes": episodes,
        "vae_events": [
            {
                "episode": ("single" if count == 1 else ("A" if index == 0 else "B")),
                "cache_cleared": True,
                "decode_input_latents": 25,
                "decoded_pixel_frames_with_sink": 97,
                "dropped_pixel_frame_indices": [0],
                "output_pixel_frames": 96,
            }
            for index in range(count)
        ],
        "output_pixel_frames": 96 * count,
    }
    if count == 2:
        trace["reset_events"] = [
            {
                "after_episode": "A",
                "retained_sink_frames": profile.global_sink_frames,
                "cleared_non_sink_self_kv": True,
                "cleared_cross_kv": True,
                "next_future_start_frame": profile.global_sink_frames,
                "prefix_snapshot_restored": profile.global_sink_frames > 1,
            }
        ]
    return trace


def _probe(sample):
    return {
        "width": sample.width,
        "height": sample.height,
        "frame_count": 96 if sample.dataset == STAGE2_SINGLE_DATASET else 192,
        "fps": 24.0,
    }


def _write_video(root, sample):
    path = root / sample.output_relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"video:{sample.sample_key}".encode())
    return path


def test_sample_trace_binds_video_prompt_checkpoint_profile_and_no_quality_metrics(
    tmp_path,
):
    sample = _samples()[0]
    video = _write_video(tmp_path, sample)
    inference_config = _inference_config(tmp_path)
    trace = build_stage2_sample_trace(
        sample=sample,
        generation_trace=_generation(sample),
        output_root=tmp_path,
        video_path=video,
        checkpoint=CHECKPOINT,
        inference_config=inference_config,
        code_version=CODE_VERSION,
        probe_fn=lambda _: _probe(sample),
    )
    assert trace["sample"]["sample_key"] == sample.sample_key
    assert trace["output"]["frame_count"] == 96
    assert trace["output"]["sha256"] == sha256_file(video)
    assert trace["quality_metrics"] is None
    assert (
        validate_stage2_sample_trace(
            trace,
            sample=sample,
            inference_config=inference_config,
        )
        == trace
    )
    path = write_stage2_sample_trace(tmp_path, sample=sample, trace=trace)
    assert json.loads(path.read_text()) == trace


def test_config_identity_is_self_verifying_and_trace_rejects_resolved_drift(
    tmp_path: Path,
) -> None:
    sample = _samples()[0]
    video = _write_video(tmp_path, sample)
    inference_config = _inference_config(tmp_path)
    assert (
        validate_stage2_inference_config_identity(inference_config) == inference_config
    )
    trace = build_stage2_sample_trace(
        sample=sample,
        generation_trace=_generation(sample),
        output_root=tmp_path,
        video_path=video,
        checkpoint=CHECKPOINT,
        inference_config=inference_config,
        code_version=CODE_VERSION,
        probe_fn=lambda _: _probe(sample),
    )

    changed = _inference_config(
        tmp_path,
        t5_checkpoint="/models/t5-v2.pth",
    )
    with pytest.raises(RuntimeError, match="inference config mismatch"):
        validate_stage2_sample_trace(
            trace,
            sample=sample,
            inference_config=changed,
        )

    tampered = json.loads(json.dumps(inference_config))
    tampered["resolved"]["vae_checkpoint"] = "/models/vae-v2.pth"
    with pytest.raises(RuntimeError, match="path differs|contract hash mismatch"):
        validate_stage2_inference_config_identity(tampered)


def test_trace_rejects_code_version_and_bit_inexact_sigma_drift(
    tmp_path: Path,
) -> None:
    sample = _samples()[0]
    video = _write_video(tmp_path, sample)
    trace = build_stage2_sample_trace(
        sample=sample,
        generation_trace=_generation(sample),
        output_root=tmp_path,
        video_path=video,
        checkpoint=CHECKPOINT,
        inference_config=_inference_config(tmp_path),
        code_version=CODE_VERSION,
        probe_fn=lambda _: _probe(sample),
    )

    with pytest.raises(RuntimeError, match="code version mismatch"):
        validate_stage2_sample_trace(
            trace,
            sample=sample,
            code_version={"stage2_source_sha256": "b" * 64},
        )

    drifted = deepcopy(trace)
    sigmas = [0.9, 0.6, 0.3, 0.1, 0.0]
    episode = drifted["generation"]["episodes"][0]
    episode["scheduler_sigmas"] = sigmas
    episode["chunk_sigmas"] = [sigmas[:-1]] * len(episode["chunk_sigmas"])
    for chunk in episode["chunk_trace"]:
        chunk["sigmas"] = sigmas[:-1]
    body = dict(drifted)
    body.pop("trace_sha256")
    drifted["trace_sha256"] = canonical_json_sha256(body)
    with pytest.raises(RuntimeError, match="sigma schedule drifted"):
        validate_stage2_sample_trace(drifted, sample=sample)


def test_formal_config_identity_cannot_smuggle_a_dynamic_sweep_profile(
    tmp_path: Path,
) -> None:
    identity = _inference_config(tmp_path)
    spec = build_stage2_deployment_rollout_spec(
        chunk_frames=6,
        local_window_frames=12,
        num_denoising_steps=3,
    )
    forged = json.loads(json.dumps(identity))
    forged["resolved"]["profiles"] = [spec.name]
    with pytest.raises(ValueError, match="frozen named profile"):
        validate_stage2_inference_config_identity(forged)


def test_config_identity_separates_resolved_and_runtime_hash_semantics(
    tmp_path: Path,
) -> None:
    config = _resolved_inference_config(tmp_path / "output")
    runtime_assets = _runtime_asset_identity(config)
    identity = build_stage2_inference_config_identity(
        config,
        runtime_assets=runtime_assets,
    )
    assert identity["resolved_contract_hash"] == config.contract_hash()
    assert identity["resolved_launch_hash"] == config.launch_hash()

    moved_config = replace(config, output_root=str((tmp_path / "moved").resolve()))
    moved = build_stage2_inference_config_identity(
        moved_config,
        runtime_assets=runtime_assets,
    )
    assert moved["resolved_contract_hash"] == identity["resolved_contract_hash"]
    assert moved["runtime_contract_hash"] == identity["runtime_contract_hash"]
    assert moved["resolved_launch_hash"] != identity["resolved_launch_hash"]
    assert moved["runtime_launch_hash"] != identity["runtime_launch_hash"]

    changed_assets = deepcopy(runtime_assets)
    changed_assets.pop("identity_sha256")
    changed_assets["assets"]["t5_checkpoint"]["aggregate_sha256"] = "f" * 64
    changed_assets["identity_sha256"] = canonical_json_sha256(changed_assets)
    changed = build_stage2_inference_config_identity(
        config,
        runtime_assets=changed_assets,
    )
    assert changed["resolved_contract_hash"] == identity["resolved_contract_hash"]
    assert changed["resolved_launch_hash"] == identity["resolved_launch_hash"]
    assert changed["runtime_contract_hash"] != identity["runtime_contract_hash"]
    assert changed["runtime_launch_hash"] != identity["runtime_launch_hash"]


def test_config_identity_v2_rejects_ambiguous_legacy_hash_keys(
    tmp_path: Path,
) -> None:
    identity = _inference_config(tmp_path)
    legacy = dict(identity)
    legacy["contract_hash"] = legacy.pop("resolved_contract_hash")
    legacy["launch_hash"] = legacy.pop("resolved_launch_hash")
    with pytest.raises(ValueError, match="config identity schema mismatch"):
        validate_stage2_inference_config_identity(legacy)


def test_formal_config_identity_v2_hash_golden_is_unchanged() -> None:
    config = _resolved_inference_config(
        Path("/artifacts/stage2-formal"),
        single_metadata="/metadata/single.csv",
        two_action_metadata="/metadata/two.csv",
    )
    identity = build_stage2_inference_config_identity(
        config,
        runtime_assets=_runtime_asset_identity(config),
    )
    assert identity["resolved"] == json.loads(json.dumps(config.to_dict()))
    assert identity["resolved_contract_hash"] == (
        "01471d5c3c147203f503cd15fc7890928a724bf7025cc31209339071e69507da"
    )
    assert identity["resolved_launch_hash"] == (
        "873d99ed2518e8afc0ca62c33a4b4cdb5de61d841b4f3ad9e9f58bd25e96b7bc"
    )
    assert identity["runtime_contract_hash"] == (
        "63e96f72bcdcfac1ee1281c320828dffd425b7f798fdac089796688022283d8f"
    )
    assert identity["runtime_launch_hash"] == (
        "cb9b001cf35240946524002f86487d82bae8033dd09ada6845b6294151bc03f2"
    )


def test_sweep_config_identity_recomputes_profile_set_and_validates_evaluation(
    tmp_path: Path,
) -> None:
    spec = build_stage2_deployment_rollout_spec(
        chunk_frames=6,
        local_window_frames=12,
        num_denoising_steps=3,
    )
    identity = _sweep_inference_config(tmp_path, specs=(spec,))
    assert validate_stage2_inference_config_identity(identity) == identity
    assert identity["resolved"]["profile_set_sha256"] == canonical_json_sha256(
        [spec.to_dict()]
    )

    bad_profile_sha = deepcopy(identity)
    bad_profile_sha["resolved"]["profile_set_sha256"] = "f" * 64
    _rehash_config_identity(bad_profile_sha)
    with pytest.raises(RuntimeError, match="profile set SHA-256 mismatch"):
        validate_stage2_inference_config_identity(bad_profile_sha)

    bad_base_sha = deepcopy(identity)
    bad_base_sha["resolved"]["base_config_sha256"] = "D" * 64
    _rehash_config_identity(bad_base_sha)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_stage2_inference_config_identity(bad_base_sha)

    duplicate_seed = deepcopy(identity)
    duplicate_seed["resolved"]["seeds"] = [9, 9]
    _rehash_config_identity(duplicate_seed)
    with pytest.raises(ValueError, match="sorted and unique"):
        validate_stage2_inference_config_identity(duplicate_seed)

    invalid_formal_evaluation = deepcopy(identity)
    invalid_formal_evaluation["resolved"]["evaluation_mode"] = "formal"
    _rehash_config_identity(invalid_formal_evaluation)
    with pytest.raises(ValueError, match="formal Stage-2 inference sweep"):
        validate_stage2_inference_config_identity(invalid_formal_evaluation)

    extra_key = deepcopy(identity)
    extra_key["resolved"]["unexpected"] = None
    _rehash_config_identity(extra_key)
    with pytest.raises(ValueError, match="sweep config schema mismatch"):
        validate_stage2_inference_config_identity(extra_key)


def test_sweep_config_identity_rejects_noncanonical_or_forged_profiles(
    tmp_path: Path,
) -> None:
    low = build_stage2_deployment_rollout_spec(
        chunk_frames=4,
        local_window_frames=8,
        num_denoising_steps=3,
    )
    high = build_stage2_deployment_rollout_spec(
        chunk_frames=8,
        local_window_frames=16,
        num_denoising_steps=5,
    )
    noncanonical = _resolved_sweep_config(tmp_path, specs=(high, low))
    with pytest.raises(ValueError, match="profiles are not canonical"):
        build_stage2_inference_config_identity(
            noncanonical,
            runtime_assets=_runtime_asset_identity(noncanonical),
        )

    baseline = resolve_stage2_rollout_profile("baseline_c8w16k4s1")
    dynamic_baseline = build_stage2_deployment_rollout_spec(
        chunk_frames=8,
        local_window_frames=16,
        num_denoising_steps=4,
    )
    duplicate = _resolved_sweep_config(
        tmp_path,
        specs=(baseline, dynamic_baseline),
    )
    with pytest.raises(ValueError, match="repeats a rollout topology"):
        build_stage2_inference_config_identity(
            duplicate,
            runtime_assets=_runtime_asset_identity(duplicate),
        )

    identity = _sweep_inference_config(tmp_path, specs=(low,))
    forged = deepcopy(identity)
    profile_name = forged["resolved"]["profiles"][0]
    replacement = "0" if profile_name[-1] != "0" else "1"
    forged["resolved"]["profiles"][0] = profile_name[:-1] + replacement
    _rehash_config_identity(forged)
    with pytest.raises(ValueError, match="canonical name mismatch"):
        validate_stage2_inference_config_identity(forged)


def test_dynamic_sweep_profile_trace_and_manifest_roundtrip(tmp_path: Path) -> None:
    spec = build_stage2_deployment_rollout_spec(
        chunk_frames=6,
        local_window_frames=12,
        num_denoising_steps=3,
    )
    inference_config = _sweep_inference_config(tmp_path, specs=(spec,))
    samples = build_stage2_inference_samples(
        single_metadata=SINGLE_METADATA,
        two_action_metadata=TWO_METADATA,
        seeds=(9,),
        profiles=(spec.name,),
        single_row_ids=(0,),
        two_action_row_ids=(0,),
    )
    sample = samples[0]
    video = _write_video(tmp_path, sample)
    trace = build_stage2_sample_trace(
        sample=sample,
        generation_trace=_generation(sample, profile_spec=spec),
        output_root=tmp_path,
        video_path=video,
        checkpoint=CHECKPOINT,
        inference_config=inference_config,
        code_version=CODE_VERSION,
        probe_fn=lambda _: _probe(sample),
    )
    assert trace["profile"] == {
        **spec.to_dict(),
        "profile_sha256": canonical_json_sha256(spec.to_dict()),
    }
    assert trace["generation"]["episodes"][0]["scheduler_timesteps"] == list(
        resolve_stage2_shift5_schedule(3).timesteps
    )
    assert (
        validate_stage2_sample_trace(
            trace,
            sample=sample,
            inference_config=inference_config,
        )
        == trace
    )
    trace_path = write_stage2_sample_trace(tmp_path, sample=sample, trace=trace)
    metadata = {
        STAGE2_SINGLE_DATASET: {
            "path": str(SINGLE_METADATA),
            "sha256": sha256_file(SINGLE_METADATA),
        },
        STAGE2_TWO_ACTION_DATASET: {
            "path": str(TWO_METADATA),
            "sha256": sha256_file(TWO_METADATA),
        },
    }
    with pytest.raises(RuntimeError, match="resolved sample matrix"):
        build_stage2_inference_manifest(
            output_root=tmp_path,
            samples=(sample,),
            traces={sample.sample_key: trace},
            trace_paths={sample.sample_key: trace_path},
            checkpoint=CHECKPOINT,
            inference_config=inference_config,
            metadata=metadata,
            code_version=CODE_VERSION,
        )

    traces = {sample.sample_key: trace}
    trace_paths = {sample.sample_key: trace_path}
    second = samples[1]
    second_video = _write_video(tmp_path, second)
    second_trace = build_stage2_sample_trace(
        sample=second,
        generation_trace=_generation(second, profile_spec=spec),
        output_root=tmp_path,
        video_path=second_video,
        checkpoint=CHECKPOINT,
        inference_config=inference_config,
        code_version=CODE_VERSION,
        probe_fn=lambda _: _probe(second),
    )
    traces[second.sample_key] = second_trace
    trace_paths[second.sample_key] = write_stage2_sample_trace(
        tmp_path,
        sample=second,
        trace=second_trace,
    )
    manifest = build_stage2_inference_manifest(
        output_root=tmp_path,
        samples=samples,
        traces=traces,
        trace_paths=trace_paths,
        checkpoint=CHECKPOINT,
        inference_config=inference_config,
        metadata=metadata,
        code_version=CODE_VERSION,
    )
    assert manifest["seeds"] == [9]
    assert manifest["profiles"] == [trace["profile"]]
    assert manifest["samples"][0]["seed"] == 9
    assert validate_stage2_inference_manifest(manifest) == manifest
    write_stage2_inference_manifest(tmp_path, manifest=manifest)
    write_stage2_review_index(tmp_path, manifest=manifest)
    assert (
        validate_stage2_inference_manifest_artifacts(
            tmp_path,
            manifest,
            expected_checkpoint=CHECKPOINT,
            expected_resolved_config=inference_config["resolved"],
            expected_samples=samples,
        )
        == manifest
    )

    wrong_entry_seed = deepcopy(manifest)
    wrong_entry_seed["samples"][0]["seed"] = 1
    body = dict(wrong_entry_seed)
    body.pop("manifest_sha256")
    wrong_entry_seed["manifest_sha256"] = canonical_json_sha256(body)
    with pytest.raises(ValueError, match=r"samples\[0\]\.seed is invalid"):
        validate_stage2_inference_manifest(wrong_entry_seed)


def test_technical_video_gate_rejects_wrong_frames_fps_or_resolution(tmp_path):
    sample = _samples()[0]
    video = _write_video(tmp_path, sample)
    for field, value in (("frame_count", 97), ("fps", 23.0), ("width", 1)):
        probe = _probe(sample)
        probe[field] = value
        with pytest.raises(RuntimeError, match=field):
            validate_stage2_video_artifact(sample, video, probe_fn=lambda _, p=probe: p)


def test_trace_rejects_reused_a_b_noise_and_automatic_quality_metric(tmp_path):
    sample = next(
        item for item in _samples() if item.dataset == STAGE2_TWO_ACTION_DATASET
    )
    video = _write_video(tmp_path, sample)
    generation = _generation(sample)
    generation["noise_episode_sha256"][1] = generation["noise_episode_sha256"][0]
    with pytest.raises(RuntimeError, match="noise hashes must differ"):
        build_stage2_sample_trace(
            sample=sample,
            generation_trace=generation,
            output_root=tmp_path,
            video_path=video,
            checkpoint=CHECKPOINT,
            inference_config=_inference_config(tmp_path),
            code_version=CODE_VERSION,
            probe_fn=lambda _: _probe(sample),
        )

    generation = _generation(sample)
    trace = build_stage2_sample_trace(
        sample=sample,
        generation_trace=generation,
        output_root=tmp_path,
        video_path=video,
        checkpoint=CHECKPOINT,
        inference_config=_inference_config(tmp_path),
        code_version=CODE_VERSION,
        probe_fn=lambda _: _probe(sample),
    )
    trace["quality_metrics"] = {"psnr": 99.0}
    without = dict(trace)
    without.pop("trace_sha256")
    trace["trace_sha256"] = canonical_json_sha256(without)
    with pytest.raises(RuntimeError, match="quality metrics are forbidden"):
        validate_stage2_sample_trace(trace, sample=sample)


def test_trace_records_one_contiguous_rng_stream_without_prompt_false_positive(
    tmp_path,
):
    sample = replace(_samples()[0], prompts=("A PSNR sign is visible in the scene",))
    video = _write_video(tmp_path, sample)
    generation = _generation(sample)
    trace = build_stage2_sample_trace(
        sample=sample,
        generation_trace=generation,
        output_root=tmp_path,
        video_path=video,
        checkpoint=CHECKPOINT,
        inference_config=_inference_config(tmp_path),
        code_version=CODE_VERSION,
        probe_fn=lambda _: _probe(sample),
    )
    assert trace["generation"]["noise_stream"] == {
        "policy": STAGE2_NOISE_STREAM_POLICY,
        "rng_initializations_per_sample": 1,
        "episode_slots": 24,
        "episode_order": ["single"],
    }
    assert validate_stage2_sample_trace(trace, sample=sample) == trace

    generation["noise_stream"]["rng_initializations_per_sample"] = 2
    with pytest.raises(RuntimeError, match="one per-sample RNG"):
        build_stage2_sample_trace(
            sample=sample,
            generation_trace=generation,
            output_root=tmp_path,
            video_path=video,
            checkpoint=CHECKPOINT,
            inference_config=_inference_config(tmp_path),
            code_version=CODE_VERSION,
            probe_fn=lambda _: _probe(sample),
        )


@pytest.mark.parametrize("profile_name", ["c8w16k4s4", "c8w16k4s8", "c4w8k2s1"])
def test_compression_and_multi_sink_trace_uses_native_profile_contract(
    tmp_path, profile_name
):
    sample = next(
        item
        for item in build_stage2_inference_samples(
            single_metadata=SINGLE_METADATA,
            two_action_metadata=TWO_METADATA,
            profiles=(profile_name,),
        )
        if item.dataset == STAGE2_TWO_ACTION_DATASET
    )
    video = _write_video(tmp_path, sample)
    trace = build_stage2_sample_trace(
        sample=sample,
        generation_trace=_generation(sample),
        output_root=tmp_path,
        video_path=video,
        checkpoint=CHECKPOINT,
        inference_config=_inference_config(
            tmp_path,
            profiles=(profile_name,),
        ),
        code_version=CODE_VERSION,
        probe_fn=lambda _: _probe(sample),
    )
    assert validate_stage2_sample_trace(trace, sample=sample) == trace
    if profile_name.endswith(("s4", "s8")):
        assert (
            trace["generation"]["episodes"][0]["cache_audit"]["capacity_frames"] == 17
        )
        assert trace["generation"]["episodes"][1]["cache_audit"]["capacity_frames"] in {
            20,
            24,
        }
    else:
        assert trace["generation"]["episodes"][0]["scheduler_timesteps"] == [
            999,
            833,
        ]


def test_complete_56_sample_manifest_and_static_review_index(tmp_path):
    samples = _samples()
    inference_config = _inference_config(tmp_path)
    traces = {}
    trace_paths = {}
    for sample in samples:
        video = _write_video(tmp_path, sample)
        trace = build_stage2_sample_trace(
            sample=sample,
            generation_trace=_generation(sample),
            output_root=tmp_path,
            video_path=video,
            checkpoint=CHECKPOINT,
            inference_config=inference_config,
            code_version=CODE_VERSION,
            probe_fn=lambda _, item=sample: _probe(item),
        )
        trace_path = write_stage2_sample_trace(tmp_path, sample=sample, trace=trace)
        traces[sample.sample_key] = trace
        trace_paths[sample.sample_key] = trace_path
    metadata = {
        STAGE2_SINGLE_DATASET: {
            "path": str(SINGLE_METADATA),
            "sha256": sha256_file(SINGLE_METADATA),
        },
        STAGE2_TWO_ACTION_DATASET: {
            "path": str(TWO_METADATA),
            "sha256": sha256_file(TWO_METADATA),
        },
    }
    manifest = build_stage2_inference_manifest(
        output_root=tmp_path,
        samples=samples,
        traces=traces,
        trace_paths=trace_paths,
        checkpoint=CHECKPOINT,
        inference_config=inference_config,
        metadata=metadata,
        code_version={"stage2_source_sha256": "a" * 64},
    )
    assert manifest["status"] == "complete"
    assert manifest["inference_config"] == inference_config
    assert manifest["expected_sample_count"] == 56
    assert len(manifest["samples"]) == 56
    assert len({item["sample_key"] for item in manifest["samples"]}) == 56
    without = dict(manifest)
    claimed = without.pop("manifest_sha256")
    assert claimed == canonical_json_sha256(without)
    assert validate_stage2_inference_manifest(manifest) == manifest
    bool_seed_manifest = json.loads(json.dumps(manifest))
    bool_seed_manifest["seeds"][0] = True
    bool_seed_body = {
        key: value
        for key, value in bool_seed_manifest.items()
        if key != "manifest_sha256"
    }
    bool_seed_manifest["manifest_sha256"] = canonical_json_sha256(bool_seed_body)
    with pytest.raises(ValueError, match="manifest seed set mismatch"):
        validate_stage2_inference_manifest(bool_seed_manifest)
    with pytest.raises(RuntimeError, match="manifest inference config mismatch"):
        validate_stage2_inference_manifest(
            manifest,
            inference_config=_inference_config(
                tmp_path,
                tokenizer_dir="/models/tokenizer-v2",
            ),
        )
    validate_stage2_inference_artifact_set(tmp_path, samples=samples)
    manifest_path = write_stage2_inference_manifest(tmp_path, manifest=manifest)
    assert json.loads(manifest_path.read_text()) == manifest
    with pytest.raises(FileExistsError):
        write_stage2_inference_manifest(tmp_path, manifest=manifest)
    index = write_stage2_review_index(tmp_path, manifest=manifest)
    html = index.read_text()
    assert html.count("<video ") == 56
    assert "PSNR" not in html and "SSIM" not in html
    assert (
        validate_stage2_inference_manifest_artifacts(
            tmp_path,
            manifest,
            expected_checkpoint=CHECKPOINT,
            expected_resolved_config=inference_config["resolved"],
            expected_samples=samples,
        )
        == manifest
    )
    wrong_checkpoint = {**CHECKPOINT, "manifest_sha256": "9" * 64}
    with pytest.raises(RuntimeError, match="differs from expected checkpoint"):
        validate_stage2_inference_manifest_artifacts(
            tmp_path,
            manifest,
            expected_checkpoint=wrong_checkpoint,
        )
    wrong_resolved = {**inference_config["resolved"], "profiles": ["c4w8k2s1"]}
    with pytest.raises(RuntimeError, match="differs from current config"):
        validate_stage2_inference_manifest_artifacts(
            tmp_path,
            manifest,
            expected_resolved_config=wrong_resolved,
        )
    with pytest.raises(RuntimeError, match="sample plan differs"):
        validate_stage2_inference_manifest_artifacts(
            tmp_path,
            manifest,
            expected_samples=samples[:-1],
        )

    original_index = index.read_bytes()
    index.write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="review index differs"):
        validate_stage2_inference_manifest_artifacts(tmp_path, manifest)
    index.write_bytes(original_index)

    first_video = tmp_path / samples[0].output_relative_path
    original_video = first_video.read_bytes()
    first_video.write_bytes(b"x" * len(original_video))
    with pytest.raises(RuntimeError, match="video hash/size mismatch"):
        validate_stage2_inference_manifest_artifacts(tmp_path, manifest)
    first_video.write_bytes(original_video)

    first_trace = Path(trace_paths[samples[0].sample_key])
    original_trace = first_trace.read_bytes()
    first_trace.write_bytes(b"x" * len(original_trace))
    with pytest.raises(RuntimeError, match="trace hash/size mismatch"):
        validate_stage2_inference_manifest_artifacts(tmp_path, manifest)
    first_trace.write_bytes(original_trace)

    missing = dict(traces)
    missing.pop(samples[0].sample_key)
    with pytest.raises(RuntimeError, match="incomplete or extra"):
        build_stage2_inference_manifest(
            output_root=tmp_path,
            samples=samples,
            traces=missing,
            trace_paths=trace_paths,
            checkpoint=CHECKPOINT,
            inference_config=inference_config,
            metadata=metadata,
            code_version={"stage2_source_sha256": "a" * 64},
        )

    trace_path = Path(trace_paths[samples[0].sample_key])
    trace_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from validated memory state"):
        build_stage2_inference_manifest(
            output_root=tmp_path,
            samples=samples,
            traces=traces,
            trace_paths=trace_paths,
            checkpoint=CHECKPOINT,
            inference_config=inference_config,
            metadata=metadata,
            code_version={"stage2_source_sha256": "a" * 64},
        )

    extra = tmp_path / "videos" / "unexpected.mp4"
    extra.write_bytes(b"extra")
    with pytest.raises(RuntimeError, match="video artifact set mismatch"):
        validate_stage2_inference_artifact_set(tmp_path, samples=samples)
