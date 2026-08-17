from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from utils.config import DEFAULT_NEGATIVE_PROMPT, normalize_config
from utils.stage2_config import (
    STAGE2_CONFIG_SCHEMA,
    STAGE2_METRICS_SCHEMA,
    STAGE2_NEGATIVE_PROMPT_SHA256,
    STAGE2_PROFILE,
    STAGE2_TRAINER,
    load_stage2_config,
    resolve_stage2_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_i2v_stage2_600cats.yaml"


def _config() -> dict:
    loaded = OmegaConf.load(CONFIG_PATH)
    plain = OmegaConf.to_container(loaded, resolve=True)
    assert isinstance(plain, dict)
    return plain


def _set_path(config: dict, path: str, value) -> None:
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def _delete_path(config: dict, path: str) -> None:
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        target = target[part]
    del target[parts[-1]]


def test_release_stage2_config_resolves_locked_baseline_contract():
    resolved = load_stage2_config(CONFIG_PATH)

    assert resolved.config_schema == STAGE2_CONFIG_SCHEMA
    assert resolved.profile == STAGE2_PROFILE
    assert resolved.trainer == STAGE2_TRAINER
    assert resolved.initialization_mode == "init_from_stage1"
    assert (
        resolved.expected_world_size,
        resolved.data_parallel_size,
        resolved.sequence_parallel_size,
    ) == (8, 8, 1)
    assert resolved.expected_nodes == 1
    assert (resolved.fsdp_backend, resolved.sharding_strategy) == ("fsdp2", "full")
    assert (resolved.forward_dtype, resolved.master_dtype) == (
        "bfloat16",
        "float32",
    )
    assert resolved.latent_patch_size == (1, 2, 2)
    assert resolved.temporal_compression_ratio == 4
    assert resolved.saved_tensor_cpu_offload_scope == "generator_grad_exit_only"
    assert resolved.architecture_root.endswith("Wan2.2-TI2V-5B")
    assert resolved.generator_stage1_step == 3075
    assert resolved.init_generator_checkpoint is not None
    assert resolved.init_generator_manifest is not None
    assert resolved.init_real_score_checkpoint is not None
    assert resolved.init_real_score_manifest is not None
    assert resolved.resume_stage2_checkpoint is None

    assert resolved.generator_trainable == "adapter_only"
    assert resolved.generator_conditioning_mode == "conditional_only"
    assert resolved.generator_forward_mode == "single_conditional"
    assert resolved.generator_cfg_formula == "conditional_only"
    assert resolved.generator_self_kv_cache_branches == 1
    assert resolved.generator_cross_kv_cache_branches == 1
    assert resolved.real_score_trainable == "frozen"
    assert resolved.real_score_conditioning_mode == "standard_cfg"
    assert resolved.real_score_forward_mode == "sequential_cond_uncond"
    assert (
        resolved.real_score_cfg_formula == "uncond_plus_scale_times_cond_minus_uncond"
    )
    assert resolved.real_score_self_kv_cache_branches == 0
    assert resolved.real_score_cross_kv_cache_branches == 0
    assert resolved.fake_score_trainable == "adapter_only"
    assert resolved.fake_score_conditioning_mode == "conditional_only"
    assert resolved.fake_score_forward_mode == "single_conditional"
    assert resolved.fake_score_cfg_formula == "conditional_only"
    assert resolved.fake_score_self_kv_cache_branches == 0
    assert resolved.fake_score_cross_kv_cache_branches == 0

    assert resolved.generated_episode_frames == 24
    assert resolved.chunk_frames == 8
    assert resolved.num_chunks == 3
    assert resolved.local_window_frames == 16
    assert resolved.history_frames == 8
    assert resolved.global_sink_frames == 1
    assert resolved.physical_kv_capacity_frames == 17
    assert resolved.num_denoising_steps == 4
    assert resolved.solver == "unipc"
    assert resolved.exit_sampling == "stratified_uniform"
    assert resolved.rollout_timestep_shift == 5.0

    assert resolved.video_latent_frames == 25
    assert resolved.initial_latent_frames == 1
    assert resolved.future_latent_frames == 24
    assert resolved.score_input_frames == 25
    assert resolved.orientation_patch_tokens == (390, 390)
    assert resolved.patch_tokens_per_frame == 390
    assert resolved.sink_query_tokens == 390
    assert resolved.future_query_tokens == 9_360
    assert resolved.score_seq_len == 9_750
    assert resolved.decoded_pixel_frames_with_sink == 97
    assert resolved.output_pixel_frames_single_action == 96
    assert resolved.output_pixel_frames_two_actions == 192
    assert resolved.baseline_deploy_dit_calls == 16

    assert resolved.rollout_cfg_scale == 1.0
    assert resolved.score_fake_cfg_scale == 1.0
    assert resolved.score_real_cfg_scale == 5.0
    assert resolved.score_timestep_shift == 5.0
    assert resolved.score_timestep_mode == "video_global"
    assert resolved.score_timestep_sampling == "shifted_uniform_integer"
    assert resolved.score_sigma_mode == "continuous"
    assert resolved.score_timestep_min == 20.0
    assert resolved.score_timestep_max == 980.0

    assert resolved.expected_num_samples == 600
    assert resolved.expected_num_actions == 3
    assert dict(resolved.expected_action_counts) == {
        "head_tilt_and_wink": 198,
        "jump": 202,
        "play_with_a_cat_wand": 200,
    }
    assert resolved.metadata_path.endswith("metadata_600clips_480x832_buckets.csv")
    assert resolved.cache_dir.endswith("stage2_i2v_600_bf16")
    assert resolved.source_cache_manifest.endswith(
        "stage2_f25_cache_manifest.attested.json"
    )
    assert resolved.action_label_source_policy == "manifest_first_sidecar_fallback"
    assert resolved.action_labels_path is None
    assert resolved.negative_conditioning_manifest.endswith(
        "stage2_negative_conditioning_manifest.json"
    )
    assert resolved.latent_channels == 48
    assert resolved.cache_dtype == "bfloat16"
    assert resolved.allowed_latent_spatial_shapes == ((30, 52), (52, 30))
    assert resolved.prompt_padding_value == 0.0
    assert resolved.num_workers == 2
    assert resolved.per_action_batch_counts == (22, 21, 21)
    assert resolved.global_batch_size == 64
    assert resolved.microbatch_size_per_device == 2
    assert resolved.gradient_accumulation_steps == 4
    assert resolved.effective_global_batch == 64
    assert resolved.candidate_batch_profiles == ((2, 4), (1, 8))
    assert resolved.candidate_profile_global_batches == (64, 64)

    assert resolved.generator_updates_per_epoch == 10
    assert resolved.fake_updates_per_generator_update == 5
    assert resolved.phase_b_mode == "dmd_dfd"
    assert resolved.phase_b_dfd_probability_max == 0.25
    assert resolved.reset_optimizers_between_phases is False
    assert resolved.nonfinite_max_attempts_per_update == 2
    assert resolved.phase_a_generator_updates == 240
    assert resolved.phase_a_fake_updates == 1_200
    assert resolved.phase_b_generator_updates == 40
    assert resolved.phase_b_fake_updates == 200
    assert resolved.total_generator_updates == 280
    assert resolved.total_fake_updates == 1_400
    assert resolved.total_cycles == 280
    assert resolved.total_optimizer_substeps == 1_680
    assert resolved.milestone_generator_updates == (
        80,
        120,
        160,
        200,
        240,
        250,
        260,
        270,
        280,
    )
    assert resolved.checkpoint_interval_generator_updates == 10

    assert resolved.ema_decay == 0.99
    assert resolved.ema_role == "generator"
    assert resolved.ema_target == "generator_adapter"
    assert resolved.ema_initialize_at_completed_generator_update == 40
    assert resolved.ema_first_decay_completed_generator_update == 41

    generator = resolved.generator_adapter
    fake_score = resolved.fake_score_adapter
    assert (generator.rank, generator.alpha) == (32, 32)
    assert (generator.dropout, generator.bias, generator.modules_to_save) == (
        0.0,
        "none",
        (),
    )
    assert generator.expected_target_modules == 180
    assert generator.expected_trainable_parameters == 57_016_320
    assert generator.expected_adapter_tensors == 360
    assert (fake_score.rank, fake_score.alpha) == (64, 64)
    assert fake_score.expected_target_modules == 180
    assert fake_score.expected_trainable_parameters == 114_032_640
    assert fake_score.expected_adapter_tensors == 360
    assert (
        generator.target_patterns
        == fake_score.target_patterns
        == (
            r"^blocks\.[0-9]+\.self_attn\.(q|k|v|o)$",
            r"^blocks\.[0-9]+\.ffn\.(0|2)$",
        )
    )

    generator_optimizer = resolved.generator_optimizer
    fake_optimizer = resolved.fake_score_optimizer
    assert generator_optimizer.optimizer_type == "adamw"
    assert generator_optimizer.learning_rate == 2.0e-6
    assert generator_optimizer.betas == (0.0, 0.999)
    assert generator_optimizer.max_grad_norm == 10.0
    assert fake_optimizer.optimizer_type == "adamw"
    assert fake_optimizer.learning_rate == 4.0e-7

    assert resolved.metrics_schema == STAGE2_METRICS_SCHEMA
    assert resolved.jsonl_path == "metrics/stage2_train_metrics.jsonl"
    assert resolved.jsonl_every_steps == 1
    assert resolved.fsync_every_steps == 1
    assert resolved.disable_wandb is True
    assert resolved.keep_last_resumable == 2
    assert resolved.atomic_success_marker is True
    assert resolved.negative_prompt_sha256 == STAGE2_NEGATIVE_PROMPT_SHA256


def test_phase_b1_probability_has_exact_ten_point_endpoints():
    probabilities = load_stage2_config(CONFIG_PATH).phase_b1_dfd_probabilities
    assert len(probabilities) == 10
    assert probabilities[0] == 0.0
    assert probabilities[-1] == 0.25
    assert probabilities == tuple(0.25 * index / 9 for index in range(10))


def test_micro1_acc8_is_the_only_baseline_fallback_and_preserves_global64():
    config = _config()
    config["training"]["microbatch_size_per_device"] = 1
    config["training"]["gradient_accumulation_steps"] = 8
    resolved = resolve_stage2_config(config)
    assert resolved.effective_global_batch == 64
    assert (
        resolved.microbatch_size_per_device,
        resolved.gradient_accumulation_steps,
    ) == (
        1,
        8,
    )


def test_saved_tensor_cpu_offload_is_only_available_to_fallback_candidate():
    invalid = _config()
    invalid["infra"]["saved_tensor_cpu_offload"] = True
    with pytest.raises(ValueError, match="saved_tensor_cpu_offload"):
        resolve_stage2_config(invalid)

    fallback = _config()
    fallback["training"]["microbatch_size_per_device"] = 1
    fallback["training"]["gradient_accumulation_steps"] = 8
    fallback["infra"]["saved_tensor_cpu_offload"] = True
    assert resolve_stage2_config(fallback).saved_tensor_cpu_offload is True


def test_safe_compile_candidate_seed_and_fsync_are_recorded_not_silently_locked():
    config = _config()
    config["infra"]["torch_compile"] = True
    config["training"]["seed"] = 20260808
    config["logging"]["fsync_every_steps"] = 10
    resolved = resolve_stage2_config(config)
    assert resolved.torch_compile is True
    assert resolved.torch_compile_mode == "max-autotune-no-cudagraphs"
    assert resolved.training_seed == 20260808
    assert resolved.fsync_every_steps == 10


def test_phase_b_zero_resolves_a_only_counts_without_phase_b_markers():
    config = _config()
    config["training"]["phase_b_epochs"] = 0
    config["training"]["phase_b_mode"] = "disabled"
    config["training"]["phase_b_dfd_probability_max"] = 0.0
    resolved = resolve_stage2_config(config)
    assert resolved.phase_b_generator_updates == 0
    assert resolved.phase_b_fake_updates == 0
    assert resolved.total_generator_updates == 240
    assert resolved.total_fake_updates == 1_200
    assert resolved.phase_b1_dfd_probabilities == ()
    assert resolved.milestone_generator_updates == (80, 120, 160, 200, 240)


def test_matched_phase_b_pure_dmd_control_preserves_update_budget():
    config = _config()
    config["training"]["phase_b_mode"] = "dmd_only"
    config["training"]["phase_b_dfd_probability_max"] = 0.0
    resolved = resolve_stage2_config(config)
    assert resolved.phase_b_epochs == 4
    assert resolved.phase_b_mode == "dmd_only"
    assert resolved.phase_b_generator_updates == 40
    assert resolved.phase_b_fake_updates == 200
    assert resolved.total_generator_updates == 280
    assert resolved.total_fake_updates == 1_400
    assert resolved.phase_b1_dfd_probabilities == (0.0,) * 10


@pytest.mark.parametrize(
    ("epochs", "mode", "probability"),
    [
        (0, "dmd_dfd", 0.0),
        (0, "disabled", 0.25),
        (4, "disabled", 0.0),
        (4, "dmd_only", 0.25),
        (4, "dmd_dfd", 0.0),
    ],
)
def test_phase_b_schedule_combinations_fail_fast(epochs, mode, probability):
    config = _config()
    config["training"]["phase_b_epochs"] = epochs
    config["training"]["phase_b_mode"] = mode
    config["training"]["phase_b_dfd_probability_max"] = probability
    with pytest.raises(ValueError, match="phase_b"):
        resolve_stage2_config(config)


def test_resolver_is_stable_idempotent_and_does_not_mutate_input():
    config = _config()
    original = copy.deepcopy(config)
    first = resolve_stage2_config(config)
    second = resolve_stage2_config(copy.deepcopy(config))
    assert config == original
    assert first.to_dict() == second.to_dict()
    assert first.launch_hash() == second.launch_hash()
    assert first.contract_hash() == second.contract_hash()
    assert first.resolved_hash() == first.launch_hash()
    assert len(first.launch_hash()) == len(first.contract_hash()) == 64
    int(first.launch_hash(), 16)
    int(first.contract_hash(), 16)
    json.dumps(first.to_dict(), allow_nan=False)


def test_resume_changes_launch_hash_but_preserves_static_contract_hash():
    initial = resolve_stage2_config(_config())
    resume_config = _config()
    resume_config["checkpoints"]["init_from_stage1"] = None
    resume_config["checkpoints"]["resume_stage2"] = "/path/to/stage2_resume"
    resume_config["logging"]["jsonl_path"] = "/path/to/resumed_metrics.jsonl"
    resumed = resolve_stage2_config(resume_config)
    assert resumed.initialization_mode == "resume_stage2"
    assert resumed.init_generator_checkpoint is None
    assert resumed.init_generator_manifest is None
    assert resumed.init_real_score_checkpoint is None
    assert resumed.init_real_score_manifest is None
    assert resumed.resume_stage2_checkpoint == "/path/to/stage2_resume"
    assert resumed.launch_hash() != initial.launch_hash()
    assert resumed.contract_hash() == initial.contract_hash()


def test_operator_locations_do_not_change_contract_but_seed_does():
    baseline = resolve_stage2_config(_config())
    relocated_config = _config()
    relocated_config["model_kwargs"]["architecture_root"] = "/mnt/model"
    relocated_config["checkpoints"]["init_from_stage1"][
        "generator_checkpoint"
    ] = "/mnt/generator.pt"
    relocated_config["checkpoints"]["init_from_stage1"][
        "generator_manifest"
    ] = "/mnt/generator.json"
    relocated_config["checkpoints"]["init_from_stage1"][
        "real_score_checkpoint"
    ] = "/mnt/real.pt"
    relocated_config["checkpoints"]["init_from_stage1"][
        "real_score_manifest"
    ] = "/mnt/real.json"
    relocated_config["data"]["metadata_path"] = "/mnt/metadata.csv"
    relocated_config["data"]["cache_dir"] = "/mnt/cache"
    relocated_config["data"]["negative_conditioning"][
        "artifact_manifest"
    ] = "/mnt/negative.json"
    relocated_config["logging"]["jsonl_path"] = "/mnt/metrics.jsonl"
    relocated = resolve_stage2_config(relocated_config)
    assert relocated.launch_hash() != baseline.launch_hash()
    assert relocated.contract_hash() == baseline.contract_hash()

    new_seed_config = _config()
    new_seed_config["training"]["seed"] += 1
    assert (
        resolve_stage2_config(new_seed_config).contract_hash()
        != baseline.contract_hash()
    )


def test_equivalent_numeric_spellings_have_identical_semantic_hashes():
    baseline = resolve_stage2_config(_config())
    equivalent_config = _config()
    equivalent_config["algorithm"]["score_real_cfg_scale"] = 5
    equivalent_config["algorithm"]["score_fake_cfg_scale"] = 1
    equivalent_config["rollout"]["timestep_shift"] = 5
    equivalent = resolve_stage2_config(equivalent_config)
    assert equivalent.launch_hash() == baseline.launch_hash()
    assert equivalent.contract_hash() == baseline.contract_hash()


def test_negative_prompt_utf8_sha256_is_locked_to_repository_constant():
    expected = hashlib.sha256(DEFAULT_NEGATIVE_PROMPT.encode("utf-8")).hexdigest()
    assert expected == STAGE2_NEGATIVE_PROMPT_SHA256
    assert (
        expected == "ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e"
    )


def test_manifest_first_action_labels_accept_null_and_explicit_sidecar():
    without_sidecar = resolve_stage2_config(_config())
    assert without_sidecar.action_labels_path is None

    config = _config()
    config["data"]["action_labels_path"] = "/path/to/operator_confirmed.csv"
    with_sidecar = resolve_stage2_config(config)
    assert with_sidecar.action_labels_path == "/path/to/operator_confirmed.csv"
    assert with_sidecar.launch_hash() != without_sidecar.launch_hash()
    assert with_sidecar.contract_hash() == without_sidecar.contract_hash()


def test_release_data_paths_support_h100_environment_overrides(monkeypatch):
    monkeypatch.delenv("LONG_LIVE_STAGE2_METADATA_PATH", raising=False)
    monkeypatch.delenv("LONG_LIVE_STAGE2_ACTION_LABELS_PATH", raising=False)
    monkeypatch.delenv("LONG_LIVE_STAGE2_SOURCE_MANIFEST", raising=False)
    baseline = load_stage2_config(CONFIG_PATH)

    assert baseline.metadata_path == (
        "training_sets/metadata_600clips_480x832_buckets.csv"
    )
    assert baseline.action_labels_path is None
    assert baseline.source_cache_manifest == (
        "/path/to/stage2_f25_cache_manifest.attested.json"
    )
    assert (
        baseline.contract_hash()
        == "a7365f2ec45f74c3918ec05725b5d19b488fa4447a409cc6b5db4ccb114dd6c6"
    )

    monkeypatch.setenv("LONG_LIVE_STAGE2_METADATA_PATH", "/mnt/stage2/metadata_600.csv")
    monkeypatch.setenv(
        "LONG_LIVE_STAGE2_ACTION_LABELS_PATH", "/mnt/stage2/action_labels.csv"
    )
    monkeypatch.setenv(
        "LONG_LIVE_STAGE2_SOURCE_MANIFEST",
        "/mnt/stage2/stage2_f25_cache_manifest.attested.json",
    )
    overridden = load_stage2_config(CONFIG_PATH)

    assert overridden.metadata_path == "/mnt/stage2/metadata_600.csv"
    assert overridden.action_labels_path == "/mnt/stage2/action_labels.csv"
    assert overridden.source_cache_manifest == (
        "/mnt/stage2/stage2_f25_cache_manifest.attested.json"
    )
    assert overridden.contract_hash() == baseline.contract_hash()
    assert overridden.launch_hash() != baseline.launch_hash()


def test_matched_phase_b_arms_share_only_the_a24_parent_contract():
    b1_config = OmegaConf.load(CONFIG_PATH)
    b0_config = OmegaConf.create(OmegaConf.to_container(b1_config, resolve=False))
    b0_config.training.phase_b_mode = "dmd_only"
    b0_config.training.phase_b_dfd_probability_max = 0.0

    b1 = resolve_stage2_config(b1_config)
    b0 = resolve_stage2_config(b0_config)

    assert b0.contract_hash() == b1.contract_hash()
    assert b0.launch_hash() != b1.launch_hash()

    different_budget = OmegaConf.create(
        OmegaConf.to_container(b1_config, resolve=False)
    )
    different_budget.training.phase_b_epochs = 0
    different_budget.training.phase_b_mode = "disabled"
    different_budget.training.phase_b_dfd_probability_max = 0.0
    a_only = resolve_stage2_config(different_budget)
    assert a_only.contract_hash() != b1.contract_hash()


def test_derived_values_and_unipc_timetable_are_not_yaml_sources_of_truth():
    config = _config()
    forbidden_keys = {
        "chunks",
        "num_chunks",
        "history_frames",
        "physical_kv_capacity_frames",
        "score_input_frames",
        "patch_tokens_per_frame",
        "score_seq_len",
        "effective_global_batch",
        "generator_updates_per_epoch",
        "total_generator_updates",
        "total_fake_updates",
        "timesteps",
        "denoising_step_list",
    }

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key not in forbidden_keys
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(config)


def test_repository_unipc_schedule_characterization_for_future_step7_gate():
    # Read-only characterization only. Step 7 must add the production startup
    # assertion and a scheduler-drift negative test around the actual rollout.
    from wan_5b.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(4, device="cpu", shift=5)
    assert [int(value) for value in scheduler.timesteps.tolist()] == [
        999,
        937,
        833,
        624,
    ]
    assert float(scheduler.sigmas[-1]) == 0.0


@pytest.mark.parametrize(
    ("path", "value", "error_path"),
    [
        ("typo", True, "config.typo"),
        ("rollout.chunk_fames", 8, "rollout.chunk_fames"),
        ("adapter.generator.alhpa", 32, "adapter.generator.alhpa"),
        ("score", {"seq_len": 9_750}, "config.score"),
        ("training.total_generator_updates", 280, "training.total_generator_updates"),
    ],
)
def test_unknown_keys_fail_with_full_paths(path, value, error_path):
    config = _config()
    _set_path(config, path, value)
    with pytest.raises(ValueError, match=re.escape(error_path)):
        resolve_stage2_config(config)


def test_non_string_keys_fail_with_full_section_path():
    config = _config()
    config["rollout"][7] = "invalid"
    with pytest.raises(ValueError, match=re.escape("rollout contains non-string keys")):
        resolve_stage2_config(config)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("model_kwargs.local_attn_size", 32),
        ("algorithm.all_causal", True),
        ("algorithm.real_guidance_scale", 4.0),
        ("algorithm.fake_guidance_scale", 0.0),
        ("algorithm.backward_simulation", True),
        ("algorithm.teacher_forcing", False),
        ("training.dfake_gen_update_ratio", 5),
        ("training.max_iters", 5_000),
        ("training.num_training_frames", 32),
        ("training.slice_last_frames", 32),
        ("training.ema_start_step", 40),
        ("adapter.rank", 128),
        ("adapter.apply_to_critic", True),
        ("inference", {"sink_size": 0}),
    ],
)
def test_legacy_dmd_fields_are_rejected_even_when_numerically_similar(path, value):
    config = _config()
    _set_path(config, path, value)
    error_path = f"config.{path}" if "." not in path else path
    with pytest.raises(ValueError, match=re.escape(error_path)):
        resolve_stage2_config(config)


@pytest.mark.parametrize(
    "path",
    [
        "profile",
        "infra.expected_world_size",
        "algorithm.score_real_cfg_scale",
        "rollout.generated_episode_frames",
        "adapter.fake_score",
        "training.phase_a_epochs",
    ],
)
def test_missing_required_keys_never_receive_silent_defaults(path):
    config = _config()
    _delete_path(config, path)
    with pytest.raises(ValueError, match=re.escape(path.split(".")[-1])):
        resolve_stage2_config(config)


@pytest.mark.parametrize(
    "path",
    [
        "rollout.generated_episode_frames",
        "adapter.fake_score",
        "algorithm.score_real_cfg_scale",
    ],
)
def test_required_null_values_fail_instead_of_receiving_defaults(path):
    config = _config()
    _set_path(config, path, None)
    with pytest.raises(ValueError, match=re.escape(path)):
        resolve_stage2_config(config)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("rollout.generated_episode_frames", 23, "generated_episode_frames"),
        ("rollout.generated_episode_frames", 25, "generated_episode_frames"),
        ("rollout.chunk_frames", 0, "chunk_frames"),
        ("rollout.local_window_frames", 4, "local_window_frames"),
        ("rollout.local_window_frames", 12, "local_window_frames"),
        ("rollout.global_sink_frames", 0, "global_sink_frames"),
        ("rollout.num_denoising_steps", 2, "num_denoising_steps"),
        ("rollout.solver", "euler", "solver"),
        ("rollout.timestep_shift", 4.0, "timestep_shift"),
        ("model_contract.latent_patch_size", [1, 4, 2], "latent_patch_size"),
        ("data.video_latent_frames", 24, "video_latent_frames"),
        ("data.latent_channels", 16, "latent_channels"),
        ("data.cache_dtype", "float16", "cache_dtype"),
        (
            "data.allowed_latent_spatial_shapes",
            [[44, 80], [80, 44]],
            "allowed_latent_spatial_shapes",
        ),
        ("data.expected_num_samples", 599, "expected_num_samples"),
        ("infra.expected_world_size", 7, "expected_world_size"),
        ("infra.sequence_parallel_size", 2, "sequence_parallel_size"),
        (
            "infra.generator_activation_checkpointing",
            True,
            "generator_activation_checkpointing",
        ),
        (
            "infra.saved_tensor_cpu_offload_scope",
            "all_saved_tensors",
            "saved_tensor_cpu_offload_scope",
        ),
        ("roles.real_score.trainable", "adapter_only", "roles.real_score.trainable"),
        (
            "roles.generator.conditioning_mode",
            "standard_cfg",
            "roles.generator.conditioning_mode",
        ),
        (
            "roles.generator.cross_kv_cache_branches",
            2,
            "roles.generator.cross_kv_cache_branches",
        ),
        ("adapter.generator.rank", 64, "adapter.generator.rank"),
        (
            "adapter.fake_score.target_patterns",
            [r"^blocks\\..*$"],
            "adapter.fake_score.target_patterns",
        ),
        ("training.fake_updates_per_generator_update", 4, "fake_updates"),
        ("training.phase_a_epochs", 23, "phase_a_epochs"),
        ("training.phase_b_epochs", 2, "phase_b_epochs"),
        ("training.reset_optimizers_between_phases", True, "reset_optimizers"),
        ("training.optimizers.generator.lr", 4.0e-6, "generator.lr"),
        ("training.ema.role", "fake_score", "training.ema.role"),
        ("training.ema.target", "fake_score_adapter", "training.ema.target"),
        ("algorithm.rollout_cfg_scale", 2.0, "rollout_cfg_scale"),
        ("algorithm.score_fake_cfg_scale", 0.0, "score_fake_cfg_scale"),
        ("algorithm.score_real_cfg_scale", 4.0, "score_real_cfg_scale"),
        (
            "algorithm.score_timestep_sampling",
            "uniform_continuous",
            "score_timestep_sampling",
        ),
        ("training.ema.decay", 0.999, "decay"),
        (
            "training.ema.initialize_after_completed_generator_updates",
            39,
            "initialize_after_completed_generator_updates",
        ),
        ("logging.jsonl_every_steps", 2, "jsonl_every_steps"),
        (
            "data.negative_conditioning.artifact_manifest",
            " /path/to/manifest.json",
            "leading or trailing whitespace",
        ),
        ("training.microbatch_size_per_device", 4, "candidate"),
        ("training.gradient_accumulation_steps", 3, "candidate"),
        ("rollout.chunk_frames", True, "integer"),
        ("algorithm.denominator_clamp", float("nan"), "finite"),
        ("algorithm.score_timestep_max", float("inf"), "finite"),
        ("algorithm.denominator_clamp", 10**400, "finite"),
    ],
)
def test_baseline_value_shape_topology_and_numeric_drift_fail_fast(path, value, match):
    config = _config()
    _set_path(config, path, value)
    with pytest.raises(ValueError, match=match):
        resolve_stage2_config(config)


@pytest.mark.parametrize(
    ("microbatch", "accumulation"),
    [(2, 3), (1, 4), (4, 2)],
)
def test_noncandidate_batch_profiles_fail_even_if_one_happens_to_total_64(
    microbatch, accumulation
):
    config = _config()
    config["training"]["microbatch_size_per_device"] = microbatch
    config["training"]["gradient_accumulation_steps"] = accumulation
    with pytest.raises(ValueError, match="candidate"):
        resolve_stage2_config(config)


def test_init_from_stage1_and_resume_stage2_are_exactly_one_mode():
    both = _config()
    both["checkpoints"]["resume_stage2"] = "/path/to/checkpoint"
    with pytest.raises(ValueError, match="Exactly one"):
        resolve_stage2_config(both)

    neither = _config()
    neither["checkpoints"]["init_from_stage1"] = None
    with pytest.raises(ValueError, match="Exactly one"):
        resolve_stage2_config(neither)

    resume = _config()
    resume["checkpoints"]["init_from_stage1"] = None
    resume["checkpoints"]["resume_stage2"] = "/path/to/checkpoint"
    assert resolve_stage2_config(resume).initialization_mode == "resume_stage2"


def test_real_score_cannot_gain_an_adapter_node():
    config = _config()
    config["adapter"]["real_score"] = copy.deepcopy(config["adapter"]["fake_score"])
    with pytest.raises(ValueError, match="adapter.real_score"):
        resolve_stage2_config(config)


def test_preflight_profiles_are_exact_and_preserve_global_batch():
    config = _config()
    config["preflight"]["candidate_profiles"].reverse()
    with pytest.raises(ValueError, match="candidate_profiles"):
        resolve_stage2_config(config)


def test_stage2_must_route_raw_config_before_legacy_normalization():
    """Characterize the integration boundary; Batch 1 does not edit train.py."""

    config = OmegaConf.load(CONFIG_PATH)
    normalized = normalize_config(config)
    assert normalized.trainer == STAGE2_TRAINER
    assert normalized.config_schema == STAGE2_CONFIG_SCHEMA
    assert normalized.metrics_schema == STAGE2_METRICS_SCHEMA
    assert "schema" not in normalized
    for key in (
        "real_guidance_scale",
        "fake_guidance_scale",
        "all_causal",
        "backward_simulation",
        "teacher_forcing",
    ):
        assert key not in normalized
    with pytest.raises(ValueError, match=re.escape("config contains unknown keys")):
        resolve_stage2_config(normalized)


def test_stage2_config_import_is_pure_and_does_not_load_torch_or_models():
    program = """
import sys
import utils.stage2_config
for name in ('torch', 'model', 'trainer', 'pipeline', 'diffusers'):
    assert name not in sys.modules, (name, sorted(sys.modules))
print('stage2-config-import-ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stage2-config-import-ok"


def test_config_cli_hash_only_matches_library():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "utils.stage2_config",
            "--config",
            str(CONFIG_PATH),
            "--hash-only",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == load_stage2_config(CONFIG_PATH).resolved_hash()

    contract_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "utils.stage2_config",
            "--config",
            str(CONFIG_PATH),
            "--contract-hash-only",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert contract_result.returncode == 0, contract_result.stderr
    assert (
        contract_result.stdout.strip()
        == load_stage2_config(CONFIG_PATH).contract_hash()
    )
