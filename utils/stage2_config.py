"""Strict, side-effect-free resolver for the Stage-2 release baseline.

This module deliberately does not import torch, model code, distributed code,
or CUDA-facing libraries.  It validates the primary configuration contract and
derives every count that would otherwise become a second source of truth.
Asset contents, model construction, and runtime scheduler checks belong to
later Stage-2 gates.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from utils.config import DEFAULT_NEGATIVE_PROMPT, wan_default_config
from utils.stage2_action_contract import STAGE2_EXPECTED_ACTION_COUNTS

STAGE2_CONFIG_SCHEMA = "longlive_stage2_train/v1"
STAGE2_METRICS_SCHEMA = "longlive_stage2_metrics/v1"
STAGE2_TRAINER = "stage2_distillation"
STAGE2_PROFILE = "baseline"
STAGE2_H100_LONGRUN_PROFILE = "h100_micro1_acc8_longrun"
STAGE2_H100_C4W16S1_LONGRUN_PROFILE = "h100_c4w16s1_micro1_acc8_longrun"
STAGE2_NEGATIVE_PROMPT_SHA256 = hashlib.sha256(
    DEFAULT_NEGATIVE_PROMPT.encode("utf-8")
).hexdigest()

_CORE_TARGET_PATTERNS = (
    r"^blocks\.[0-9]+\.self_attn\.(q|k|v|o)$",
    r"^blocks\.[0-9]+\.ffn\.(0|2)$",
)
_CANDIDATE_BATCH_PROFILES = ((2, 4), (1, 8))


@dataclass(frozen=True)
class _Stage2ProfileSpec:
    rollout_profile: str
    training_batch_profiles: tuple[tuple[int, int], ...]
    phase_a_epochs: int
    phase_b_epochs: int
    generator_lr: float
    fake_score_lr: float
    checkpoint_every_generator_epochs: int
    keep_last_resumable: int
    phase_a_milestone_epochs: tuple[int, ...]
    phase_b_milestone_epochs: tuple[int, ...]


_STAGE2_PROFILE_SPECS = {
    STAGE2_PROFILE: _Stage2ProfileSpec(
        rollout_profile="baseline_c8w16k4s1",
        training_batch_profiles=_CANDIDATE_BATCH_PROFILES,
        phase_a_epochs=24,
        phase_b_epochs=4,
        generator_lr=2.0e-6,
        fake_score_lr=4.0e-7,
        checkpoint_every_generator_epochs=1,
        keep_last_resumable=2,
        phase_a_milestone_epochs=(8, 12, 16, 20, 24),
        phase_b_milestone_epochs=(1, 2, 3, 4),
    ),
    STAGE2_H100_LONGRUN_PROFILE: _Stage2ProfileSpec(
        rollout_profile="baseline_c8w16k4s1",
        training_batch_profiles=((1, 8),),
        phase_a_epochs=360,
        phase_b_epochs=40,
        generator_lr=1.0e-5,
        fake_score_lr=2.0e-6,
        checkpoint_every_generator_epochs=4,
        keep_last_resumable=2,
        phase_a_milestone_epochs=(
            4,
            8,
            12,
            16,
            20,
            24,
            40,
            60,
            80,
            120,
            160,
            200,
            240,
            280,
            320,
            360,
        ),
        phase_b_milestone_epochs=(1, 2, 3, 4, 10, 20, 30, 40),
    ),
    STAGE2_H100_C4W16S1_LONGRUN_PROFILE: _Stage2ProfileSpec(
        rollout_profile="c4w16k4s1",
        training_batch_profiles=((1, 8),),
        phase_a_epochs=360,
        phase_b_epochs=40,
        generator_lr=5.0e-6,
        fake_score_lr=1.0e-6,
        checkpoint_every_generator_epochs=1,
        keep_last_resumable=400,
        phase_a_milestone_epochs=(),
        phase_b_milestone_epochs=(),
    ),
}
_CONTRACT_LOCAL_DERIVED_FIELDS = {
    "initialization_mode",
    "architecture_root",
    "init_generator_checkpoint",
    "init_generator_manifest",
    "init_real_score_checkpoint",
    "init_real_score_manifest",
    "resume_stage2_checkpoint",
    "metadata_path",
    "source_cache_manifest",
    "cache_dir",
    "action_labels_path",
    "negative_conditioning_manifest",
    "jsonl_path",
    # These are the two matched-arm choices after the resolved Phase-A boundary.
    # They must be excluded from both halves of the contract payload: the
    # canonical config view below and the duplicated resolved/derived view.
    "phase_b_mode",
    "phase_b_dfd_probability_max",
    "phase_b1_dfd_probabilities",
}

_TOP_LEVEL_KEYS = {
    "config_schema",
    "profile",
    "infra",
    "model_kwargs",
    "model_contract",
    "checkpoints",
    "roles",
    "algorithm",
    "rollout",
    "adapter",
    "data",
    "sampler",
    "training",
    "checkpointing",
    "evaluation",
    "logging",
    "preflight",
}
_INFRA_KEYS = {
    "expected_nodes",
    "expected_world_size",
    "data_parallel_size",
    "sequence_parallel_size",
    "fsdp_backend",
    "sharding_strategy",
    "mixed_precision",
    "forward_dtype",
    "master_dtype",
    "generator_activation_checkpointing",
    "real_score_activation_checkpointing",
    "fake_score_activation_checkpointing",
    "cpu_model_offload",
    "saved_tensor_cpu_offload",
    "saved_tensor_cpu_offload_scope",
    "model_quant",
    "torch_compile",
    "torch_compile_mode",
    "load_text_encoder",
    "load_vae",
}
_MODEL_KEYS = {"model_name", "architecture_root", "init_weights"}
_MODEL_CONTRACT_KEYS = {"latent_patch_size", "temporal_compression_ratio"}
_CHECKPOINT_KEYS = {"init_from_stage1", "resume_stage2"}
_INIT_CHECKPOINT_KEYS = {
    "generator_stage1_step",
    "generator_checkpoint",
    "generator_manifest",
    "real_score_checkpoint",
    "real_score_manifest",
}
_ROLE_NAMES = {"generator", "real_score", "fake_score"}
_ROLE_KEYS = {
    "backbone",
    "base_source",
    "trainable",
    "conditioning_mode",
    "forward_mode",
    "cfg_formula",
    "self_kv_cache_branches",
    "cross_kv_cache_branches",
}
_ALGORITHM_KEYS = {
    "trainer",
    "objective",
    "i2v",
    "num_train_timesteps",
    "score_timestep_mode",
    "score_timestep_sampling",
    "score_sigma_mode",
    "score_math_dtype",
    "loss_reduction_dtype",
    "score_timestep_shift",
    "score_timestep_min",
    "score_timestep_max",
    "score_real_cfg_scale",
    "score_fake_cfg_scale",
    "rollout_cfg_scale",
    "ts_schedule",
    "ts_schedule_max",
    "denominator_clamp",
    "sink_excluded_from_loss",
}
_ROLLOUT_KEYS = {
    "generated_episode_frames",
    "chunk_frames",
    "local_window_frames",
    "global_sink_frames",
    "num_denoising_steps",
    "solver",
    "timestep_shift",
    "exit_sampling",
}
_ADAPTER_KEYS = {"generator", "fake_score"}
_ADAPTER_ROLE_KEYS = {
    "type",
    "rank",
    "alpha",
    "dropout",
    "bias",
    "modules_to_save",
    "target_patterns",
    "expected_target_modules",
    "expected_trainable_parameters",
    "expected_adapter_tensors",
}
_DATA_KEYS = {
    "backend",
    "metadata_path",
    "source_cache_manifest",
    "action_label_source_policy",
    "action_labels_path",
    "cache_dir",
    "expected_num_samples",
    "expected_num_actions",
    "expected_action_counts",
    "video_latent_frames",
    "initial_latent_frames",
    "future_latent_frames",
    "latent_channels",
    "cache_dtype",
    "allowed_latent_spatial_shapes",
    "prompt_padding_value",
    "negative_conditioning",
    "num_workers",
    "deterministic",
}
_NEGATIVE_CONDITIONING_KEYS = {"artifact_manifest"}
_SAMPLER_KEYS = {
    "per_action_batch_counts",
    "rotate_extra_slot",
    "independent_role_streams",
    "deterministic",
}
_TRAINING_KEYS = {
    "seed",
    "global_batch_size",
    "microbatch_size_per_device",
    "gradient_accumulation_steps",
    "fake_updates_per_generator_update",
    "phase_a_epochs",
    "phase_b_epochs",
    "phase_b_mode",
    "phase_b_dfd_probability_max",
    "reset_optimizers_between_phases",
    "nonfinite_max_attempts_per_update",
    "optimizers",
    "ema",
}
_OPTIMIZER_ROLE_NAMES = {"generator", "fake_score"}
_OPTIMIZER_KEYS = {
    "type",
    "lr",
    "betas",
    "eps",
    "weight_decay",
    "max_grad_norm",
    "schedule",
}
_EMA_KEYS = {
    "enabled",
    "role",
    "target",
    "decay",
    "initialize_after_completed_generator_updates",
    "dtype",
    "device",
    "trainable_only",
}
_CHECKPOINTING_KEYS = {
    "every_generator_epochs",
    "keep_last_resumable",
    "phase_a_milestone_epochs",
    "phase_b_milestone_epochs",
    "atomic_success_marker",
}
_EVALUATION_KEYS = {"interval"}
_LOGGING_KEYS = {
    "backend",
    "metrics_schema",
    "jsonl_path",
    "jsonl_every_steps",
    "fsync_every_steps",
    "disable_wandb",
}
_PREFLIGHT_KEYS = {
    "candidate_profiles",
    "max_allocated_fraction",
    "max_reserved_fraction",
    "min_free_gib",
    "min_free_fraction",
    "max_live_allocated_growth_gib",
    "max_straggler_ratio",
}
_PREFLIGHT_PROFILE_KEYS = {
    "microbatch_size_per_device",
    "gradient_accumulation_steps",
}


@dataclass(frozen=True)
class Stage2AdapterSpec:
    role: str
    rank: int
    alpha: int
    dropout: float
    bias: str
    modules_to_save: tuple[str, ...]
    target_patterns: tuple[str, ...]
    expected_target_modules: int
    expected_trainable_parameters: int
    expected_adapter_tensors: int


@dataclass(frozen=True)
class Stage2OptimizerSpec:
    role: str
    optimizer_type: str
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    max_grad_norm: float
    schedule: str


@dataclass(frozen=True)
class Stage2ResolvedConfig:
    """Canonical baseline inputs plus values derived from those inputs."""

    _launch_config_json: str = field(repr=False)
    _contract_config_json: str = field(repr=False)
    config_schema: str
    profile: str
    trainer: str
    initialization_mode: str
    expected_nodes: int
    expected_world_size: int
    data_parallel_size: int
    sequence_parallel_size: int
    fsdp_backend: str
    sharding_strategy: str
    forward_dtype: str
    master_dtype: str
    generator_activation_checkpointing: bool
    real_score_activation_checkpointing: bool
    fake_score_activation_checkpointing: bool
    saved_tensor_cpu_offload: bool
    saved_tensor_cpu_offload_scope: str
    torch_compile: bool
    torch_compile_mode: str
    model_name: str
    architecture_root: str
    latent_patch_size: tuple[int, int, int]
    temporal_compression_ratio: int
    generator_stage1_step: int
    init_generator_checkpoint: str | None
    init_generator_manifest: str | None
    init_real_score_checkpoint: str | None
    init_real_score_manifest: str | None
    resume_stage2_checkpoint: str | None
    generator_trainable: str
    generator_conditioning_mode: str
    generator_forward_mode: str
    generator_cfg_formula: str
    generator_self_kv_cache_branches: int
    generator_cross_kv_cache_branches: int
    real_score_trainable: str
    real_score_conditioning_mode: str
    real_score_forward_mode: str
    real_score_cfg_formula: str
    real_score_self_kv_cache_branches: int
    real_score_cross_kv_cache_branches: int
    fake_score_trainable: str
    fake_score_conditioning_mode: str
    fake_score_forward_mode: str
    fake_score_cfg_formula: str
    fake_score_self_kv_cache_branches: int
    fake_score_cross_kv_cache_branches: int
    num_train_timesteps: int
    score_timestep_mode: str
    score_timestep_sampling: str
    score_sigma_mode: str
    score_math_dtype: str
    loss_reduction_dtype: str
    denominator_clamp: float
    sink_excluded_from_loss: bool
    generated_episode_frames: int
    chunk_frames: int
    num_chunks: int
    local_window_frames: int
    history_frames: int
    global_sink_frames: int
    physical_kv_capacity_frames: int
    num_denoising_steps: int
    solver: str
    exit_sampling: str
    rollout_timestep_shift: float
    score_timestep_shift: float
    score_timestep_min: float
    score_timestep_max: float
    rollout_cfg_scale: float
    score_real_cfg_scale: float
    score_fake_cfg_scale: float
    video_latent_frames: int
    initial_latent_frames: int
    future_latent_frames: int
    score_input_frames: int
    orientation_patch_tokens: tuple[int, ...]
    patch_tokens_per_frame: int
    sink_query_tokens: int
    future_query_tokens: int
    score_seq_len: int
    decoded_pixel_frames_with_sink: int
    output_pixel_frames_single_action: int
    output_pixel_frames_two_actions: int
    baseline_deploy_dit_calls: int
    expected_num_samples: int
    expected_num_actions: int
    expected_action_counts: tuple[tuple[str, int], ...]
    metadata_path: str
    source_cache_manifest: str
    cache_dir: str
    action_label_source_policy: str
    action_labels_path: str | None
    negative_conditioning_manifest: str
    latent_channels: int
    cache_dtype: str
    allowed_latent_spatial_shapes: tuple[tuple[int, int], ...]
    prompt_padding_value: float
    num_workers: int
    per_action_batch_counts: tuple[int, ...]
    training_seed: int
    global_batch_size: int
    microbatch_size_per_device: int
    gradient_accumulation_steps: int
    effective_global_batch: int
    candidate_batch_profiles: tuple[tuple[int, int], ...]
    candidate_profile_global_batches: tuple[int, ...]
    generator_updates_per_epoch: int
    fake_updates_per_generator_update: int
    phase_a_epochs: int
    phase_b_epochs: int
    phase_b_mode: str
    phase_b_dfd_probability_max: float
    reset_optimizers_between_phases: bool
    nonfinite_max_attempts_per_update: int
    phase_a_generator_updates: int
    phase_a_fake_updates: int
    phase_b_generator_updates: int
    phase_b_fake_updates: int
    total_generator_updates: int
    total_fake_updates: int
    total_cycles: int
    total_optimizer_substeps: int
    phase_b1_dfd_probabilities: tuple[float, ...]
    milestone_generator_updates: tuple[int, ...]
    ema_decay: float
    ema_role: str
    ema_target: str
    ema_initialize_at_completed_generator_update: int
    ema_first_decay_completed_generator_update: int
    generator_adapter: Stage2AdapterSpec
    fake_score_adapter: Stage2AdapterSpec
    generator_optimizer: Stage2OptimizerSpec
    fake_score_optimizer: Stage2OptimizerSpec
    negative_prompt_sha256: str
    metrics_schema: str
    jsonl_path: str
    jsonl_every_steps: int
    fsync_every_steps: int
    disable_wandb: bool
    checkpoint_interval_generator_updates: int
    keep_last_resumable: int
    atomic_success_marker: bool
    preflight_max_allocated_fraction: float
    preflight_max_reserved_fraction: float
    preflight_min_free_gib: float
    preflight_min_free_fraction: float
    preflight_max_live_allocated_growth_gib: float
    preflight_max_straggler_ratio: float

    @property
    def canonical_config(self) -> dict[str, Any]:
        return json.loads(self._launch_config_json)

    def _derived_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("_launch_config_json")
        payload.pop("_contract_config_json")
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.canonical_config,
            "derived": self._derived_dict(),
        }

    @staticmethod
    def _hash_payload(payload: dict[str, Any]) -> str:
        payload = _semantic_json_value(payload)
        payload = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def contract_hash(self) -> str:
        derived = self._derived_dict()
        for field_name in _CONTRACT_LOCAL_DERIVED_FIELDS:
            derived.pop(field_name)
        return self._hash_payload(
            {
                "config": json.loads(self._contract_config_json),
                "derived": derived,
            }
        )

    def launch_hash(self) -> str:
        return self._hash_payload(self.to_dict())

    def resolved_hash(self) -> str:
        """Backward-compatible name for the launch-specific fingerprint."""

        return self.launch_hash()


def _as_plain_mapping(config: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError:  # pragma: no cover - the project requires OmegaConf.
        OmegaConf = None

    if OmegaConf is not None and OmegaConf.is_config(config):
        plain = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, Mapping):
        plain = copy.deepcopy(dict(config))
    else:
        raise TypeError(
            "Stage-2 config must be a mapping or OmegaConf config, "
            f"got {type(config).__name__}."
        )
    if not isinstance(plain, dict):
        raise TypeError("Stage-2 config root must be a mapping.")
    return plain


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping, got {value!r}.")
    return dict(value)


def _exact_keys(value: Any, path: str, expected: set[str]) -> dict[str, Any]:
    mapping = _mapping(value, path)
    non_string_keys = [key for key in mapping if not isinstance(key, str)]
    if non_string_keys:
        rendered = sorted((repr(key) for key in non_string_keys))
        raise ValueError(f"{path} contains non-string keys: {rendered}.")
    actual = set(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected, key=repr)
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}.")
    if unknown:
        full_paths = [f"{path}.{key}" for key in unknown]
        raise ValueError(f"{path} contains unknown keys: {full_paths}.")
    return mapping


def _required_section(
    config: Mapping[str, Any], key: str, expected: set[str]
) -> dict[str, Any]:
    if key not in config:
        raise ValueError(f"config is missing required key: {key}.")
    return _exact_keys(config[key], key, expected)


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string, got {value!r}.")
    if value != value.strip():
        raise ValueError(
            f"{path} must not contain leading or trailing whitespace, got {value!r}."
        )
    return value


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer, got {value!r}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be >= {minimum}, got {value}.")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number, got {value!r}.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{path} must be a finite number, got {value!r}.") from error
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite, got {value!r}.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{path} must be >= {minimum}, got {result}.")
    return result


def _semantic_json_value(value: Any) -> Any:
    """Normalize equivalent numeric spellings before semantic hashing."""

    if isinstance(value, Mapping):
        return {key: _semantic_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_semantic_json_value(child) for child in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean, got {value!r}.")
    return value


def _sequence(value: Any, path: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence, got {value!r}.")
    return list(value)


def _locked(path: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(
            f"{path} is locked to {expected!r} for the Stage-2 baseline, "
            f"got {actual!r}."
        )


def _locked_string(
    mapping: Mapping[str, Any], key: str, path: str, expected: str
) -> str:
    result = _string(mapping[key], f"{path}.{key}")
    _locked(f"{path}.{key}", result, expected)
    return result


def _locked_integer(
    mapping: Mapping[str, Any], key: str, path: str, expected: int
) -> int:
    result = _integer(mapping[key], f"{path}.{key}")
    _locked(f"{path}.{key}", result, expected)
    return result


def _locked_number(
    mapping: Mapping[str, Any], key: str, path: str, expected: float
) -> float:
    result = _number(mapping[key], f"{path}.{key}")
    if not math.isclose(result, expected, rel_tol=0.0, abs_tol=0.0):
        _locked(f"{path}.{key}", result, expected)
    return result


def _locked_boolean(
    mapping: Mapping[str, Any], key: str, path: str, expected: bool
) -> bool:
    result = _boolean(mapping[key], f"{path}.{key}")
    _locked(f"{path}.{key}", result, expected)
    return result


def _contract_config_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove operator-local locations from the static research contract.

    The launch hash still covers the complete resolved YAML. This narrower view
    lets an initialization launch and a later resume prove that their research
    contract is identical even though checkpoint and output paths necessarily
    differ. Step 2 will extend this view with verified artifact-manifest hashes.
    """

    contract = copy.deepcopy(dict(config))
    contract["model_kwargs"]["architecture_root"] = "<runtime-asset>"
    contract["checkpoints"] = {"initialization": "<runtime-checkpoint>"}
    contract["data"]["metadata_path"] = "<runtime-asset>"
    # This field was added after the static contract was published.  It names
    # an already-hashed runtime asset and is removed (rather than replaced) so
    # the released research-contract digest stays byte-for-byte stable.
    contract["data"].pop("source_cache_manifest", None)
    contract["data"]["cache_dir"] = "<runtime-asset>"
    # The manifest-first policy is the research contract. Selecting an optional
    # operator-confirmed sidecar is a launch-local asset choice, just like the
    # metadata and cache locations, so it must not change the contract hash.
    contract["data"]["action_labels_path"] = None
    contract["data"]["negative_conditioning"]["artifact_manifest"] = "<runtime-asset>"
    contract["logging"]["jsonl_path"] = "<runtime-output>"
    # Phase B has one task-book-authorized matched-control fork: the B1 arm
    # mixes DMD/DFD up to p=0.25, while B0 remains DMD-only for the same four
    # epochs.  Both children must be able to resume the *same* A24 checkpoint,
    # so only these two post-A24 choices are excluded from the parent research
    # contract.  ``phase_b_epochs`` and every other training field remain
    # covered and therefore fail closed on drift.
    contract["training"]["phase_b_mode"] = "<matched-phase-b-arm>"
    contract["training"][
        "phase_b_dfd_probability_max"
    ] = "<matched-phase-b-probability>"
    return contract


def _parse_adapter(
    value: Any,
    role: str,
    *,
    expected_rank: int,
    expected_parameters: int,
) -> Stage2AdapterSpec:
    path = f"adapter.{role}"
    config = _exact_keys(value, path, _ADAPTER_ROLE_KEYS)
    _locked_string(config, "type", path, "lora")
    rank = _locked_integer(config, "rank", path, expected_rank)
    alpha = _locked_integer(config, "alpha", path, expected_rank)
    dropout = _locked_number(config, "dropout", path, 0.0)
    bias = _locked_string(config, "bias", path, "none")
    modules_to_save = _sequence(config["modules_to_save"], f"{path}.modules_to_save")
    _locked(f"{path}.modules_to_save", modules_to_save, [])
    patterns = tuple(
        _string(pattern, f"{path}.target_patterns[{index}]")
        for index, pattern in enumerate(
            _sequence(config["target_patterns"], f"{path}.target_patterns")
        )
    )
    _locked(f"{path}.target_patterns", patterns, _CORE_TARGET_PATTERNS)
    target_modules = _locked_integer(config, "expected_target_modules", path, 180)
    trainable_parameters = _locked_integer(
        config,
        "expected_trainable_parameters",
        path,
        expected_parameters,
    )
    adapter_tensors = _locked_integer(config, "expected_adapter_tensors", path, 360)
    return Stage2AdapterSpec(
        role=role,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        bias=bias,
        modules_to_save=tuple(modules_to_save),
        target_patterns=patterns,
        expected_target_modules=target_modules,
        expected_trainable_parameters=trainable_parameters,
        expected_adapter_tensors=adapter_tensors,
    )


def _validate_optimizer(
    value: Any,
    role: str,
    *,
    expected_lr: float,
) -> Stage2OptimizerSpec:
    path = f"training.optimizers.{role}"
    config = _exact_keys(value, path, _OPTIMIZER_KEYS)
    optimizer_type = _locked_string(config, "type", path, "adamw")
    learning_rate = _locked_number(config, "lr", path, expected_lr)
    betas = tuple(
        _number(beta, f"{path}.betas[{index}]")
        for index, beta in enumerate(_sequence(config["betas"], f"{path}.betas"))
    )
    _locked(f"{path}.betas", betas, (0.0, 0.999))
    eps = _locked_number(config, "eps", path, 1.0e-8)
    weight_decay = _locked_number(config, "weight_decay", path, 0.0)
    max_grad_norm = _locked_number(config, "max_grad_norm", path, 10.0)
    schedule = _locked_string(config, "schedule", path, "constant")
    return Stage2OptimizerSpec(
        role=role,
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        schedule=schedule,
    )


def resolve_stage2_config(config: Any) -> Stage2ResolvedConfig:
    """Resolve and audit the locked Stage-2 baseline without runtime side effects.

    The current public profile is deliberately release-specific. Compression
    profiles introduced in Step 13 must use an explicit profile instead of
    weakening these baseline assertions.
    """

    raw = _as_plain_mapping(config)
    raw = _exact_keys(raw, "config", _TOP_LEVEL_KEYS)

    config_schema = _string(raw["config_schema"], "config_schema")
    _locked("config_schema", config_schema, STAGE2_CONFIG_SCHEMA)
    profile = _string(raw["profile"], "profile")
    profile_spec = _STAGE2_PROFILE_SPECS.get(profile)
    if profile_spec is None:
        raise ValueError(
            f"profile must be one of {tuple(_STAGE2_PROFILE_SPECS)}, got {profile!r}"
        )
    from pipeline.stage2_rollout_profile import resolve_stage2_rollout_profile

    rollout_profile = resolve_stage2_rollout_profile(profile_spec.rollout_profile)
    if not rollout_profile.training_allowed:
        raise AssertionError(
            f"Stage-2 training profile selected deployment-only rollout "
            f"{rollout_profile.name!r}"
        )

    infra = _required_section(raw, "infra", _INFRA_KEYS)
    expected_nodes = _locked_integer(infra, "expected_nodes", "infra", 1)
    world_size = _integer(
        infra["expected_world_size"], "infra.expected_world_size", minimum=1
    )
    if world_size not in {4, 8}:
        raise ValueError("infra.expected_world_size must be 4 or 8.")
    data_parallel_size = _integer(
        infra["data_parallel_size"], "infra.data_parallel_size", minimum=1
    )
    if data_parallel_size != world_size:
        raise ValueError(
            "infra.data_parallel_size must equal infra.expected_world_size."
        )
    sequence_parallel_size = _locked_integer(
        infra, "sequence_parallel_size", "infra", 1
    )
    fsdp_backend = _locked_string(infra, "fsdp_backend", "infra", "fsdp2")
    sharding_strategy = _locked_string(infra, "sharding_strategy", "infra", "full")
    _locked_boolean(infra, "mixed_precision", "infra", True)
    forward_dtype = _locked_string(infra, "forward_dtype", "infra", "bfloat16")
    master_dtype = _locked_string(infra, "master_dtype", "infra", "float32")
    generator_activation_checkpointing = _locked_boolean(
        infra, "generator_activation_checkpointing", "infra", False
    )
    real_score_activation_checkpointing = _locked_boolean(
        infra, "real_score_activation_checkpointing", "infra", False
    )
    fake_score_activation_checkpointing = _locked_boolean(
        infra, "fake_score_activation_checkpointing", "infra", True
    )
    for key in (
        "cpu_model_offload",
        "model_quant",
        "load_text_encoder",
        "load_vae",
    ):
        _locked_boolean(infra, key, "infra", False)
    saved_tensor_cpu_offload = _boolean(
        infra["saved_tensor_cpu_offload"], "infra.saved_tensor_cpu_offload"
    )
    saved_tensor_cpu_offload_scope = _locked_string(
        infra,
        "saved_tensor_cpu_offload_scope",
        "infra",
        "generator_grad_exit_only",
    )
    torch_compile = _boolean(infra["torch_compile"], "infra.torch_compile")
    torch_compile_mode = _locked_string(
        infra,
        "torch_compile_mode",
        "infra",
        "max-autotune-no-cudagraphs",
    )
    if data_parallel_size * sequence_parallel_size != world_size:
        raise ValueError(
            "infra.data_parallel_size * infra.sequence_parallel_size must equal "
            "infra.expected_world_size."
        )

    model = _required_section(raw, "model_kwargs", _MODEL_KEYS)
    model_name = _locked_string(model, "model_name", "model_kwargs", "Wan2.2-TI2V-5B")
    architecture_root = _string(
        model["architecture_root"], "model_kwargs.architecture_root"
    )
    _locked_boolean(model, "init_weights", "model_kwargs", False)

    model_contract = _required_section(raw, "model_contract", _MODEL_CONTRACT_KEYS)
    raw_patch_size = _sequence(
        model_contract["latent_patch_size"], "model_contract.latent_patch_size"
    )
    latent_patch_size = tuple(
        _integer(value, f"model_contract.latent_patch_size[{index}]", minimum=1)
        for index, value in enumerate(raw_patch_size)
    )
    _locked("model_contract.latent_patch_size", latent_patch_size, (1, 2, 2))
    temporal_compression_ratio = _locked_integer(
        model_contract,
        "temporal_compression_ratio",
        "model_contract",
        4,
    )
    _locked(
        "model_contract.temporal_compression_ratio",
        temporal_compression_ratio,
        int(wan_default_config[model_name]["temporal_compression_ratio"]),
    )

    checkpoints = _required_section(raw, "checkpoints", _CHECKPOINT_KEYS)
    init_config = checkpoints["init_from_stage1"]
    resume_config = checkpoints["resume_stage2"]
    has_init = init_config is not None
    has_resume = resume_config is not None
    if has_init == has_resume:
        raise ValueError(
            "Exactly one of checkpoints.init_from_stage1 and "
            "checkpoints.resume_stage2 must be configured."
        )
    generator_stage1_step = 3075
    init_generator_checkpoint = None
    init_generator_manifest = None
    init_real_score_checkpoint = None
    init_real_score_manifest = None
    resume_stage2_checkpoint = None
    if has_init:
        init_mapping = _exact_keys(
            init_config, "checkpoints.init_from_stage1", _INIT_CHECKPOINT_KEYS
        )
        generator_stage1_step = _locked_integer(
            init_mapping,
            "generator_stage1_step",
            "checkpoints.init_from_stage1",
            3075,
        )
        init_generator_checkpoint = _string(
            init_mapping["generator_checkpoint"],
            "checkpoints.init_from_stage1.generator_checkpoint",
        )
        init_generator_manifest = _string(
            init_mapping["generator_manifest"],
            "checkpoints.init_from_stage1.generator_manifest",
        )
        init_real_score_checkpoint = _string(
            init_mapping["real_score_checkpoint"],
            "checkpoints.init_from_stage1.real_score_checkpoint",
        )
        init_real_score_manifest = _string(
            init_mapping["real_score_manifest"],
            "checkpoints.init_from_stage1.real_score_manifest",
        )
        initialization_mode = "init_from_stage1"
    else:
        resume_stage2_checkpoint = _string(resume_config, "checkpoints.resume_stage2")
        initialization_mode = "resume_stage2"

    roles = _required_section(raw, "roles", _ROLE_NAMES)
    role_contracts = {
        "generator": {
            "backbone": "causal",
            "base_source": "stage1_step3075_ema_merged",
            "trainable": "adapter_only",
            "conditioning_mode": "conditional_only",
            "forward_mode": "single_conditional",
            "cfg_formula": "conditional_only",
            "self_kv_cache_branches": 1,
            "cross_kv_cache_branches": 1,
        },
        "real_score": {
            "backbone": "bidirectional_ti2v",
            "base_source": "real_score_checkpoint",
            "trainable": "frozen",
            "conditioning_mode": "standard_cfg",
            "forward_mode": "sequential_cond_uncond",
            "cfg_formula": "uncond_plus_scale_times_cond_minus_uncond",
            "self_kv_cache_branches": 0,
            "cross_kv_cache_branches": 0,
        },
        "fake_score": {
            "backbone": "bidirectional_ti2v",
            "base_source": "real_score_checkpoint",
            "trainable": "adapter_only",
            "conditioning_mode": "conditional_only",
            "forward_mode": "single_conditional",
            "cfg_formula": "conditional_only",
            "self_kv_cache_branches": 0,
            "cross_kv_cache_branches": 0,
        },
    }
    for role, expected_fields in role_contracts.items():
        role_config = _exact_keys(roles[role], f"roles.{role}", _ROLE_KEYS)
        for key, expected_value in expected_fields.items():
            if isinstance(expected_value, int):
                _locked_integer(role_config, key, f"roles.{role}", expected_value)
            else:
                _locked_string(role_config, key, f"roles.{role}", expected_value)
    generator_trainable = role_contracts["generator"]["trainable"]
    generator_conditioning_mode = role_contracts["generator"]["conditioning_mode"]
    generator_forward_mode = role_contracts["generator"]["forward_mode"]
    generator_cfg_formula = role_contracts["generator"]["cfg_formula"]
    generator_self_kv_cache_branches = role_contracts["generator"][
        "self_kv_cache_branches"
    ]
    generator_cross_kv_cache_branches = role_contracts["generator"][
        "cross_kv_cache_branches"
    ]
    real_score_trainable = role_contracts["real_score"]["trainable"]
    real_score_conditioning_mode = role_contracts["real_score"]["conditioning_mode"]
    real_score_forward_mode = role_contracts["real_score"]["forward_mode"]
    real_score_cfg_formula = role_contracts["real_score"]["cfg_formula"]
    real_score_self_kv_cache_branches = role_contracts["real_score"][
        "self_kv_cache_branches"
    ]
    real_score_cross_kv_cache_branches = role_contracts["real_score"][
        "cross_kv_cache_branches"
    ]
    fake_score_trainable = role_contracts["fake_score"]["trainable"]
    fake_score_conditioning_mode = role_contracts["fake_score"]["conditioning_mode"]
    fake_score_forward_mode = role_contracts["fake_score"]["forward_mode"]
    fake_score_cfg_formula = role_contracts["fake_score"]["cfg_formula"]
    fake_score_self_kv_cache_branches = role_contracts["fake_score"][
        "self_kv_cache_branches"
    ]
    fake_score_cross_kv_cache_branches = role_contracts["fake_score"][
        "cross_kv_cache_branches"
    ]

    algorithm = _required_section(raw, "algorithm", _ALGORITHM_KEYS)
    trainer = _locked_string(algorithm, "trainer", "algorithm", STAGE2_TRAINER)
    _locked_string(algorithm, "objective", "algorithm", "dmd_dfd")
    _locked_boolean(algorithm, "i2v", "algorithm", True)
    num_train_timesteps = _locked_integer(
        algorithm, "num_train_timesteps", "algorithm", 1000
    )
    score_timestep_mode = _locked_string(
        algorithm, "score_timestep_mode", "algorithm", "video_global"
    )
    score_timestep_sampling = _locked_string(
        algorithm,
        "score_timestep_sampling",
        "algorithm",
        "shifted_uniform_integer",
    )
    score_sigma_mode = _locked_string(
        algorithm, "score_sigma_mode", "algorithm", "continuous"
    )
    score_math_dtype = _locked_string(
        algorithm, "score_math_dtype", "algorithm", "float32"
    )
    loss_reduction_dtype = _locked_string(
        algorithm, "loss_reduction_dtype", "algorithm", "float32"
    )
    score_timestep_shift = _locked_number(
        algorithm, "score_timestep_shift", "algorithm", 5.0
    )
    score_timestep_min = _locked_number(
        algorithm, "score_timestep_min", "algorithm", 20.0
    )
    score_timestep_max = _locked_number(
        algorithm, "score_timestep_max", "algorithm", 980.0
    )
    if score_timestep_min >= score_timestep_max:
        raise ValueError(
            "algorithm.score_timestep_min must be less than "
            "algorithm.score_timestep_max."
        )
    score_real_cfg_scale = _locked_number(
        algorithm, "score_real_cfg_scale", "algorithm", 5.0
    )
    score_fake_cfg_scale = _locked_number(
        algorithm, "score_fake_cfg_scale", "algorithm", 1.0
    )
    rollout_cfg_scale = _locked_number(algorithm, "rollout_cfg_scale", "algorithm", 1.0)
    _locked_boolean(algorithm, "ts_schedule", "algorithm", False)
    _locked_boolean(algorithm, "ts_schedule_max", "algorithm", False)
    denominator_clamp = _locked_number(
        algorithm, "denominator_clamp", "algorithm", 1.0e-6
    )
    sink_excluded_from_loss = _locked_boolean(
        algorithm, "sink_excluded_from_loss", "algorithm", True
    )

    rollout = _required_section(raw, "rollout", _ROLLOUT_KEYS)
    generated_episode_frames = _integer(
        rollout["generated_episode_frames"],
        "rollout.generated_episode_frames",
        minimum=1,
    )
    chunk_frames = _integer(rollout["chunk_frames"], "rollout.chunk_frames", minimum=1)
    local_window_frames = _integer(
        rollout["local_window_frames"],
        "rollout.local_window_frames",
        minimum=1,
    )
    global_sink_frames = _integer(
        rollout["global_sink_frames"],
        "rollout.global_sink_frames",
        minimum=1,
    )
    num_denoising_steps = _integer(
        rollout["num_denoising_steps"],
        "rollout.num_denoising_steps",
        minimum=1,
    )
    if generated_episode_frames % chunk_frames:
        raise ValueError(
            "rollout.generated_episode_frames must be divisible by "
            "rollout.chunk_frames."
        )
    if local_window_frames < chunk_frames or local_window_frames % chunk_frames:
        raise ValueError(
            "rollout.local_window_frames must be a multiple of and at least "
            "rollout.chunk_frames."
        )
    _locked(
        "rollout.generated_episode_frames",
        generated_episode_frames,
        rollout_profile.generated_episode_frames,
    )
    _locked("rollout.chunk_frames", chunk_frames, rollout_profile.chunk_frames)
    _locked(
        "rollout.local_window_frames",
        local_window_frames,
        rollout_profile.local_window_frames,
    )
    _locked(
        "rollout.global_sink_frames",
        global_sink_frames,
        rollout_profile.global_sink_frames,
    )
    _locked(
        "rollout.num_denoising_steps",
        num_denoising_steps,
        rollout_profile.num_denoising_steps,
    )
    solver = _locked_string(rollout, "solver", "rollout", "unipc")
    rollout_timestep_shift = _locked_number(rollout, "timestep_shift", "rollout", 5.0)
    exit_sampling = _locked_string(
        rollout, "exit_sampling", "rollout", "stratified_uniform"
    )

    adapters = _required_section(raw, "adapter", _ADAPTER_KEYS)
    generator_adapter = _parse_adapter(
        adapters["generator"],
        "generator",
        expected_rank=32,
        expected_parameters=57_016_320,
    )
    fake_score_adapter = _parse_adapter(
        adapters["fake_score"],
        "fake_score",
        expected_rank=64,
        expected_parameters=114_032_640,
    )

    data = _required_section(raw, "data", _DATA_KEYS)
    _locked_string(data, "backend", "data", "stage2_i2v_cache")
    metadata_path = _string(data["metadata_path"], "data.metadata_path")
    source_cache_manifest = _string(
        data["source_cache_manifest"], "data.source_cache_manifest"
    )
    cache_dir = _string(data["cache_dir"], "data.cache_dir")
    action_label_source_policy = _locked_string(
        data,
        "action_label_source_policy",
        "data",
        "manifest_first_sidecar_fallback",
    )
    action_labels_path = data["action_labels_path"]
    if action_labels_path is not None:
        _string(action_labels_path, "data.action_labels_path")
    expected_num_samples = _locked_integer(data, "expected_num_samples", "data", 600)
    expected_num_actions = _locked_integer(data, "expected_num_actions", "data", 3)
    action_populations = _exact_keys(
        data["expected_action_counts"],
        "data.expected_action_counts",
        set(STAGE2_EXPECTED_ACTION_COUNTS),
    )
    expected_action_counts = tuple(
        (
            action_id,
            _locked_integer(
                action_populations,
                action_id,
                "data.expected_action_counts",
                expected_count,
            ),
        )
        for action_id, expected_count in STAGE2_EXPECTED_ACTION_COUNTS.items()
    )
    if len(expected_action_counts) != expected_num_actions:
        raise ValueError(
            "data.expected_action_counts must contain data.expected_num_actions "
            "entries."
        )
    if sum(count for _, count in expected_action_counts) != expected_num_samples:
        raise ValueError(
            "data.expected_action_counts must sum to data.expected_num_samples."
        )
    video_latent_frames = _locked_integer(data, "video_latent_frames", "data", 25)
    initial_latent_frames = _locked_integer(data, "initial_latent_frames", "data", 1)
    future_latent_frames = _locked_integer(data, "future_latent_frames", "data", 24)
    if initial_latent_frames + future_latent_frames != video_latent_frames:
        raise ValueError(
            "data.initial_latent_frames + data.future_latent_frames must equal "
            "data.video_latent_frames."
        )
    latent_channels = _locked_integer(data, "latent_channels", "data", 48)
    cache_dtype = _locked_string(data, "cache_dtype", "data", "bfloat16")
    raw_shapes = _sequence(
        data["allowed_latent_spatial_shapes"],
        "data.allowed_latent_spatial_shapes",
    )
    shapes: list[tuple[int, int]] = []
    for shape_index, raw_shape in enumerate(raw_shapes):
        dims = _sequence(
            raw_shape,
            f"data.allowed_latent_spatial_shapes[{shape_index}]",
        )
        if len(dims) != 2:
            raise ValueError(
                "data.allowed_latent_spatial_shapes entries must have two dimensions."
            )
        shapes.append(
            tuple(
                _integer(
                    dim,
                    f"data.allowed_latent_spatial_shapes[{shape_index}][{dim_index}]",
                    minimum=1,
                )
                for dim_index, dim in enumerate(dims)
            )
        )
    _locked(
        "data.allowed_latent_spatial_shapes",
        tuple(shapes),
        ((30, 52), (52, 30)),
    )
    prompt_padding_value = _locked_number(data, "prompt_padding_value", "data", 0.0)
    negative = _exact_keys(
        data["negative_conditioning"],
        "data.negative_conditioning",
        _NEGATIVE_CONDITIONING_KEYS,
    )
    negative_sha256 = STAGE2_NEGATIVE_PROMPT_SHA256
    negative_conditioning_manifest = _string(
        negative["artifact_manifest"],
        "data.negative_conditioning.artifact_manifest",
    )
    num_workers = _integer(data["num_workers"], "data.num_workers", minimum=0)
    _locked_boolean(data, "deterministic", "data", True)

    sampler = _required_section(raw, "sampler", _SAMPLER_KEYS)
    action_counts = tuple(
        _integer(count, f"sampler.per_action_batch_counts[{index}]", minimum=1)
        for index, count in enumerate(
            _sequence(
                sampler["per_action_batch_counts"],
                "sampler.per_action_batch_counts",
            )
        )
    )
    _locked("sampler.per_action_batch_counts", action_counts, (22, 21, 21))
    for key in ("rotate_extra_slot", "independent_role_streams", "deterministic"):
        _locked_boolean(sampler, key, "sampler", True)

    training = _required_section(raw, "training", _TRAINING_KEYS)
    training_seed = _integer(training["seed"], "training.seed", minimum=0)
    global_batch_size = _locked_integer(training, "global_batch_size", "training", 64)
    microbatch_size = _integer(
        training["microbatch_size_per_device"],
        "training.microbatch_size_per_device",
        minimum=1,
    )
    accumulation = _integer(
        training["gradient_accumulation_steps"],
        "training.gradient_accumulation_steps",
        minimum=1,
    )
    training_batch_profiles = tuple(
        (microbatch, profile_accumulation * 8 // world_size)
        for microbatch, profile_accumulation in profile_spec.training_batch_profiles
    )
    if (microbatch_size, accumulation) not in training_batch_profiles:
        raise ValueError(
            f"training microbatch/accumulation for profile {profile!r} must be "
            f"one of the candidate profiles {training_batch_profiles}, got "
            f"{(microbatch_size, accumulation)}."
        )
    if saved_tensor_cpu_offload and (microbatch_size, accumulation) != (
        1,
        64 // world_size,
    ):
        raise ValueError(
            "infra.saved_tensor_cpu_offload is only a candidate after the "
            "micro1 global64 profile is selected."
        )
    effective_global_batch = world_size * microbatch_size * accumulation
    if effective_global_batch != global_batch_size:
        raise ValueError(
            "infra.expected_world_size * training.microbatch_size_per_device * "
            "training.gradient_accumulation_steps must equal "
            "training.global_batch_size."
        )
    if accumulation % num_denoising_steps:
        raise ValueError(
            "training.gradient_accumulation_steps must be divisible by "
            "rollout.num_denoising_steps."
        )
    if sum(action_counts) != global_batch_size:
        raise ValueError(
            "sampler.per_action_batch_counts must sum to training.global_batch_size."
        )
    fake_updates_per_generator_update = _locked_integer(
        training,
        "fake_updates_per_generator_update",
        "training",
        5,
    )
    phase_a_epochs = _locked_integer(
        training,
        "phase_a_epochs",
        "training",
        profile_spec.phase_a_epochs,
    )
    phase_b_epochs = _integer(training["phase_b_epochs"], "training.phase_b_epochs")
    if phase_b_epochs not in (0, profile_spec.phase_b_epochs):
        raise ValueError(
            "training.phase_b_epochs must be 0 for an A-only run or "
            f"{profile_spec.phase_b_epochs} for a "
            f"matched Phase-B run, got {phase_b_epochs}."
        )
    phase_b_mode = _string(training["phase_b_mode"], "training.phase_b_mode")
    phase_b_probability_max = _number(
        training["phase_b_dfd_probability_max"],
        "training.phase_b_dfd_probability_max",
        minimum=0.0,
    )
    if phase_b_epochs == 0:
        _locked("training.phase_b_mode", phase_b_mode, "disabled")
        _locked("training.phase_b_dfd_probability_max", phase_b_probability_max, 0.0)
    elif phase_b_mode == "dmd_dfd":
        _locked("training.phase_b_dfd_probability_max", phase_b_probability_max, 0.25)
    elif phase_b_mode == "dmd_only":
        _locked("training.phase_b_dfd_probability_max", phase_b_probability_max, 0.0)
    else:
        raise ValueError(
            "training.phase_b_mode must be 'dmd_dfd' or 'dmd_only' when "
            f"training.phase_b_epochs is {profile_spec.phase_b_epochs}."
        )
    reset_optimizers_between_phases = _locked_boolean(
        training,
        "reset_optimizers_between_phases",
        "training",
        False,
    )
    nonfinite_max_attempts = _locked_integer(
        training, "nonfinite_max_attempts_per_update", "training", 2
    )

    optimizers = _exact_keys(
        training["optimizers"],
        "training.optimizers",
        _OPTIMIZER_ROLE_NAMES,
    )
    generator_optimizer = _validate_optimizer(
        optimizers["generator"],
        "generator",
        expected_lr=profile_spec.generator_lr,
    )
    fake_score_optimizer = _validate_optimizer(
        optimizers["fake_score"],
        "fake_score",
        expected_lr=profile_spec.fake_score_lr,
    )

    ema = _exact_keys(training["ema"], "training.ema", _EMA_KEYS)
    _locked_boolean(ema, "enabled", "training.ema", True)
    ema_role = _locked_string(ema, "role", "training.ema", "generator")
    ema_target = _locked_string(ema, "target", "training.ema", "generator_adapter")
    ema_decay = _locked_number(ema, "decay", "training.ema", 0.99)
    ema_initialize = _locked_integer(
        ema,
        "initialize_after_completed_generator_updates",
        "training.ema",
        40,
    )
    _locked_string(ema, "dtype", "training.ema", "float32")
    _locked_string(ema, "device", "training.ema", "cpu")
    _locked_boolean(ema, "trainable_only", "training.ema", True)

    checkpointing = _required_section(raw, "checkpointing", _CHECKPOINTING_KEYS)
    checkpoint_epochs = _locked_integer(
        checkpointing,
        "every_generator_epochs",
        "checkpointing",
        profile_spec.checkpoint_every_generator_epochs,
    )
    keep_last_resumable = _locked_integer(
        checkpointing,
        "keep_last_resumable",
        "checkpointing",
        profile_spec.keep_last_resumable,
    )
    phase_a_milestones = tuple(
        _integer(value, f"checkpointing.phase_a_milestone_epochs[{index}]", minimum=1)
        for index, value in enumerate(
            _sequence(
                checkpointing["phase_a_milestone_epochs"],
                "checkpointing.phase_a_milestone_epochs",
            )
        )
    )
    phase_b_milestones = tuple(
        _integer(value, f"checkpointing.phase_b_milestone_epochs[{index}]", minimum=1)
        for index, value in enumerate(
            _sequence(
                checkpointing["phase_b_milestone_epochs"],
                "checkpointing.phase_b_milestone_epochs",
            )
        )
    )
    _locked(
        "checkpointing.phase_a_milestone_epochs",
        phase_a_milestones,
        profile_spec.phase_a_milestone_epochs,
    )
    _locked(
        "checkpointing.phase_b_milestone_epochs",
        phase_b_milestones,
        profile_spec.phase_b_milestone_epochs,
    )
    atomic_success_marker = _locked_boolean(
        checkpointing, "atomic_success_marker", "checkpointing", True
    )

    evaluation = _required_section(raw, "evaluation", _EVALUATION_KEYS)
    _locked_integer(evaluation, "interval", "evaluation", 0)

    logging = _required_section(raw, "logging", _LOGGING_KEYS)
    _locked_string(logging, "backend", "logging", "jsonl")
    metrics_schema = _locked_string(
        logging, "metrics_schema", "logging", STAGE2_METRICS_SCHEMA
    )
    jsonl_path = _string(logging["jsonl_path"], "logging.jsonl_path")
    jsonl_every_steps = _locked_integer(logging, "jsonl_every_steps", "logging", 1)
    fsync_every_steps = _integer(
        logging["fsync_every_steps"], "logging.fsync_every_steps", minimum=1
    )
    disable_wandb = _locked_boolean(logging, "disable_wandb", "logging", True)

    preflight = _required_section(raw, "preflight", _PREFLIGHT_KEYS)
    raw_profiles = _sequence(
        preflight["candidate_profiles"], "preflight.candidate_profiles"
    )
    profiles: list[tuple[int, int]] = []
    for profile_index, raw_profile in enumerate(raw_profiles):
        profile_path = f"preflight.candidate_profiles[{profile_index}]"
        profile_config = _exact_keys(raw_profile, profile_path, _PREFLIGHT_PROFILE_KEYS)
        profiles.append(
            (
                _integer(
                    profile_config["microbatch_size_per_device"],
                    f"{profile_path}.microbatch_size_per_device",
                    minimum=1,
                ),
                _integer(
                    profile_config["gradient_accumulation_steps"],
                    f"{profile_path}.gradient_accumulation_steps",
                    minimum=1,
                ),
            )
        )
    expected_candidate_profiles = tuple(
        (microbatch, accumulation * 8 // world_size)
        for microbatch, accumulation in _CANDIDATE_BATCH_PROFILES
    )
    _locked(
        "preflight.candidate_profiles",
        tuple(profiles),
        expected_candidate_profiles,
    )
    max_allocated_fraction = _locked_number(
        preflight, "max_allocated_fraction", "preflight", 0.85
    )
    max_reserved_fraction = _locked_number(
        preflight, "max_reserved_fraction", "preflight", 0.90
    )
    min_free_gib = _locked_number(preflight, "min_free_gib", "preflight", 8.0)
    min_free_fraction = _locked_number(
        preflight, "min_free_fraction", "preflight", 0.10
    )
    max_live_allocated_growth_gib = _locked_number(
        preflight, "max_live_allocated_growth_gib", "preflight", 1.0
    )
    max_straggler_ratio = _locked_number(
        preflight, "max_straggler_ratio", "preflight", 1.15
    )

    num_chunks = generated_episode_frames // chunk_frames
    history_frames = local_window_frames - chunk_frames
    physical_kv_capacity_frames = global_sink_frames + local_window_frames
    score_input_frames = initial_latent_frames + generated_episode_frames
    if generated_episode_frames != future_latent_frames:
        raise ValueError(
            "rollout.generated_episode_frames must equal data.future_latent_frames."
        )
    patch_tokens = []
    _, patch_height, patch_width = latent_patch_size
    for height, width in shapes:
        if height % patch_height or width % patch_width:
            raise ValueError(
                "data.allowed_latent_spatial_shapes must be divisible by the "
                "locked model_contract spatial patch size."
            )
        patch_tokens.append((height // patch_height) * (width // patch_width))
    if len(set(patch_tokens)) != 1:
        raise ValueError(
            "All data.allowed_latent_spatial_shapes must produce the same "
            "patch token count."
        )
    patch_tokens_per_frame = patch_tokens[0]
    _locked("derived.patch_tokens_per_frame", patch_tokens_per_frame, 390)
    sink_query_tokens = initial_latent_frames * patch_tokens_per_frame
    future_query_tokens = generated_episode_frames * patch_tokens_per_frame
    score_seq_len = score_input_frames * patch_tokens_per_frame
    if sink_query_tokens + future_query_tokens != score_seq_len:
        raise AssertionError("Internal Stage-2 token accounting drifted.")

    decoded_with_sink = 1 + (score_input_frames - 1) * temporal_compression_ratio
    output_single = decoded_with_sink - 1
    output_two = 2 * output_single
    baseline_deploy_dit_calls = global_sink_frames + num_chunks * (
        num_denoising_steps + 1
    )

    generator_updates_per_epoch = math.ceil(expected_num_samples / global_batch_size)
    phase_a_generator_updates = phase_a_epochs * generator_updates_per_epoch
    phase_a_fake_updates = phase_a_generator_updates * fake_updates_per_generator_update
    phase_b_generator_updates = phase_b_epochs * generator_updates_per_epoch
    phase_b_fake_updates = phase_b_generator_updates * fake_updates_per_generator_update
    total_generator_updates = phase_a_generator_updates + phase_b_generator_updates
    total_fake_updates = phase_a_fake_updates + phase_b_fake_updates
    if phase_b_epochs and phase_b_mode == "dmd_dfd":
        b1_probabilities = tuple(
            phase_b_probability_max * index / (generator_updates_per_epoch - 1)
            for index in range(generator_updates_per_epoch)
        )
    elif phase_b_epochs:
        b1_probabilities = (0.0,) * generator_updates_per_epoch
    else:
        b1_probabilities = ()
    milestones = tuple(
        epoch * generator_updates_per_epoch for epoch in phase_a_milestones
    )
    if phase_b_epochs:
        milestones += tuple(
            phase_a_generator_updates + epoch * generator_updates_per_epoch
            for epoch in phase_b_milestones
        )

    profile_global_batches = tuple(
        world_size * micro * accum for micro, accum in profiles
    )
    if any(batch != global_batch_size for batch in profile_global_batches):
        raise ValueError(
            "Every preflight candidate profile must preserve the locked global batch."
        )
    if any(accum % num_denoising_steps for _, accum in profiles):
        raise ValueError(
            "Every preflight accumulation count must be divisible by the number "
            "of denoising steps."
        )

    launch_config_json = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    contract_config_json = json.dumps(
        _contract_config_view(raw),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return Stage2ResolvedConfig(
        _launch_config_json=launch_config_json,
        _contract_config_json=contract_config_json,
        config_schema=config_schema,
        profile=profile,
        trainer=trainer,
        initialization_mode=initialization_mode,
        expected_nodes=expected_nodes,
        expected_world_size=world_size,
        data_parallel_size=data_parallel_size,
        sequence_parallel_size=sequence_parallel_size,
        fsdp_backend=fsdp_backend,
        sharding_strategy=sharding_strategy,
        forward_dtype=forward_dtype,
        master_dtype=master_dtype,
        generator_activation_checkpointing=generator_activation_checkpointing,
        real_score_activation_checkpointing=real_score_activation_checkpointing,
        fake_score_activation_checkpointing=fake_score_activation_checkpointing,
        saved_tensor_cpu_offload=saved_tensor_cpu_offload,
        saved_tensor_cpu_offload_scope=saved_tensor_cpu_offload_scope,
        torch_compile=torch_compile,
        torch_compile_mode=torch_compile_mode,
        model_name=model_name,
        architecture_root=architecture_root,
        latent_patch_size=latent_patch_size,
        temporal_compression_ratio=temporal_compression_ratio,
        generator_stage1_step=generator_stage1_step,
        init_generator_checkpoint=init_generator_checkpoint,
        init_generator_manifest=init_generator_manifest,
        init_real_score_checkpoint=init_real_score_checkpoint,
        init_real_score_manifest=init_real_score_manifest,
        resume_stage2_checkpoint=resume_stage2_checkpoint,
        generator_trainable=generator_trainable,
        generator_conditioning_mode=generator_conditioning_mode,
        generator_forward_mode=generator_forward_mode,
        generator_cfg_formula=generator_cfg_formula,
        generator_self_kv_cache_branches=generator_self_kv_cache_branches,
        generator_cross_kv_cache_branches=generator_cross_kv_cache_branches,
        real_score_trainable=real_score_trainable,
        real_score_conditioning_mode=real_score_conditioning_mode,
        real_score_forward_mode=real_score_forward_mode,
        real_score_cfg_formula=real_score_cfg_formula,
        real_score_self_kv_cache_branches=real_score_self_kv_cache_branches,
        real_score_cross_kv_cache_branches=real_score_cross_kv_cache_branches,
        fake_score_trainable=fake_score_trainable,
        fake_score_conditioning_mode=fake_score_conditioning_mode,
        fake_score_forward_mode=fake_score_forward_mode,
        fake_score_cfg_formula=fake_score_cfg_formula,
        fake_score_self_kv_cache_branches=fake_score_self_kv_cache_branches,
        fake_score_cross_kv_cache_branches=fake_score_cross_kv_cache_branches,
        num_train_timesteps=num_train_timesteps,
        score_timestep_mode=score_timestep_mode,
        score_timestep_sampling=score_timestep_sampling,
        score_sigma_mode=score_sigma_mode,
        score_math_dtype=score_math_dtype,
        loss_reduction_dtype=loss_reduction_dtype,
        denominator_clamp=denominator_clamp,
        sink_excluded_from_loss=sink_excluded_from_loss,
        generated_episode_frames=generated_episode_frames,
        chunk_frames=chunk_frames,
        num_chunks=num_chunks,
        local_window_frames=local_window_frames,
        history_frames=history_frames,
        global_sink_frames=global_sink_frames,
        physical_kv_capacity_frames=physical_kv_capacity_frames,
        num_denoising_steps=num_denoising_steps,
        solver=solver,
        exit_sampling=exit_sampling,
        rollout_timestep_shift=rollout_timestep_shift,
        score_timestep_shift=score_timestep_shift,
        score_timestep_min=score_timestep_min,
        score_timestep_max=score_timestep_max,
        rollout_cfg_scale=rollout_cfg_scale,
        score_real_cfg_scale=score_real_cfg_scale,
        score_fake_cfg_scale=score_fake_cfg_scale,
        video_latent_frames=video_latent_frames,
        initial_latent_frames=initial_latent_frames,
        future_latent_frames=future_latent_frames,
        score_input_frames=score_input_frames,
        orientation_patch_tokens=tuple(patch_tokens),
        patch_tokens_per_frame=patch_tokens_per_frame,
        sink_query_tokens=sink_query_tokens,
        future_query_tokens=future_query_tokens,
        score_seq_len=score_seq_len,
        decoded_pixel_frames_with_sink=decoded_with_sink,
        output_pixel_frames_single_action=output_single,
        output_pixel_frames_two_actions=output_two,
        baseline_deploy_dit_calls=baseline_deploy_dit_calls,
        expected_num_samples=expected_num_samples,
        expected_num_actions=expected_num_actions,
        expected_action_counts=expected_action_counts,
        metadata_path=metadata_path,
        source_cache_manifest=source_cache_manifest,
        cache_dir=cache_dir,
        action_label_source_policy=action_label_source_policy,
        action_labels_path=action_labels_path,
        negative_conditioning_manifest=negative_conditioning_manifest,
        latent_channels=latent_channels,
        cache_dtype=cache_dtype,
        allowed_latent_spatial_shapes=tuple(shapes),
        prompt_padding_value=prompt_padding_value,
        num_workers=num_workers,
        per_action_batch_counts=action_counts,
        training_seed=training_seed,
        global_batch_size=global_batch_size,
        microbatch_size_per_device=microbatch_size,
        gradient_accumulation_steps=accumulation,
        effective_global_batch=effective_global_batch,
        candidate_batch_profiles=tuple(profiles),
        candidate_profile_global_batches=profile_global_batches,
        generator_updates_per_epoch=generator_updates_per_epoch,
        fake_updates_per_generator_update=fake_updates_per_generator_update,
        phase_a_epochs=phase_a_epochs,
        phase_b_epochs=phase_b_epochs,
        phase_b_mode=phase_b_mode,
        phase_b_dfd_probability_max=phase_b_probability_max,
        reset_optimizers_between_phases=reset_optimizers_between_phases,
        nonfinite_max_attempts_per_update=nonfinite_max_attempts,
        phase_a_generator_updates=phase_a_generator_updates,
        phase_a_fake_updates=phase_a_fake_updates,
        phase_b_generator_updates=phase_b_generator_updates,
        phase_b_fake_updates=phase_b_fake_updates,
        total_generator_updates=total_generator_updates,
        total_fake_updates=total_fake_updates,
        total_cycles=total_generator_updates,
        total_optimizer_substeps=total_generator_updates + total_fake_updates,
        phase_b1_dfd_probabilities=b1_probabilities,
        milestone_generator_updates=milestones,
        ema_decay=ema_decay,
        ema_role=ema_role,
        ema_target=ema_target,
        ema_initialize_at_completed_generator_update=ema_initialize,
        ema_first_decay_completed_generator_update=ema_initialize + 1,
        generator_adapter=generator_adapter,
        fake_score_adapter=fake_score_adapter,
        generator_optimizer=generator_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        negative_prompt_sha256=negative_sha256,
        metrics_schema=metrics_schema,
        jsonl_path=jsonl_path,
        jsonl_every_steps=jsonl_every_steps,
        fsync_every_steps=fsync_every_steps,
        disable_wandb=disable_wandb,
        checkpoint_interval_generator_updates=(
            checkpoint_epochs * generator_updates_per_epoch
        ),
        keep_last_resumable=keep_last_resumable,
        atomic_success_marker=atomic_success_marker,
        preflight_max_allocated_fraction=max_allocated_fraction,
        preflight_max_reserved_fraction=max_reserved_fraction,
        preflight_min_free_gib=min_free_gib,
        preflight_min_free_fraction=min_free_fraction,
        preflight_max_live_allocated_growth_gib=max_live_allocated_growth_gib,
        preflight_max_straggler_ratio=max_straggler_ratio,
    )


def load_stage2_config(path: str | Path) -> Stage2ResolvedConfig:
    """Load an OmegaConf YAML and resolve the strict Stage-2 baseline."""

    from omegaconf import OmegaConf

    return resolve_stage2_config(OmegaConf.load(Path(path)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve and print the static LongLive Stage-2 baseline contract."
    )
    parser.add_argument("--config", required=True, help="Path to the Stage-2 YAML.")
    hash_group = parser.add_mutually_exclusive_group()
    hash_group.add_argument(
        "--hash-only",
        action="store_true",
        help="Print only the launch-specific resolved SHA256.",
    )
    hash_group.add_argument(
        "--contract-hash-only",
        action="store_true",
        help="Print only the path-independent research-contract SHA256.",
    )
    args = parser.parse_args(argv)
    resolved = load_stage2_config(args.config)
    if args.contract_hash_only:
        print(resolved.contract_hash())
    elif args.hash_only:
        print(resolved.resolved_hash())
    else:
        payload = resolved.to_dict()
        payload["contract_sha256"] = resolved.contract_hash()
        payload["launch_sha256"] = resolved.launch_hash()
        print(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
