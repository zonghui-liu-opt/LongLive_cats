"""Strict configuration for deployment-only Stage-2 rollout sweeps."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

from pipeline.stage2_rollout_profile import (
    STAGE2_ROLLOUT_PROFILE_NAMES,
    Stage2RolloutSpec,
    build_stage2_deployment_rollout_spec,
    resolve_stage2_rollout_profile,
)
from utils.stage1_io import canonical_json_sha256, sha256_file
from utils.stage2_inference_config import (
    ResolvedStage2InferenceConfig,
    load_stage2_inference_config,
)

STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA = "longlive_stage2_inference_sweep/v1"
STAGE2_INFERENCE_SWEEP_MAX_PROFILES = 32
STAGE2_BASELINE_ROLLOUT_PROFILE = "baseline_c8w16k4s1"

_ROOT_KEYS = {"schema", "base_config", "output_root", "profile_set", "evaluation"}
_PROFILE_SET_KEYS = {"named", "grids", "cases"}
_PROFILE_FIELDS = {
    "chunk_frames",
    "local_window_frames",
    "num_denoising_steps",
}
_EVALUATION_KEYS = {"mode", "seeds", "single_row_ids", "two_action_row_ids"}
_FORMAL_SEEDS = (1, 2, 3, 4)
_FORMAL_SINGLE_ROW_IDS = tuple(range(6))
_FORMAL_TWO_ACTION_ROW_IDS = tuple(range(8))


def _plain_mapping(value: Any, label: str) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError:  # pragma: no cover - project dependency
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _exact_mapping(value: Any, label: str, keys: set[str]) -> dict[str, Any]:
    mapping = _plain_mapping(value, label)
    if set(mapping) != keys:
        raise ValueError(
            f"{label} schema mismatch: missing={sorted(keys - set(mapping))}, "
            f"extra={sorted(set(mapping) - keys)}"
        )
    return mapping


def _plain_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical path string")
    return os.fspath(Path(value).expanduser().resolve())


def _plain_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive plain integer")
    return value


def _plain_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return list(value)


def _topology_key(spec: Stage2RolloutSpec) -> tuple[Any, ...]:
    return (
        int(spec.generated_episode_frames),
        int(spec.chunk_frames),
        int(spec.local_window_frames),
        int(spec.global_sink_frames),
        int(spec.num_denoising_steps),
        str(spec.solver),
        float(spec.timestep_shift),
    )


def _canonical_spec_key(spec: Stage2RolloutSpec) -> tuple[Any, ...]:
    return (*_topology_key(spec), spec.name)


def _s1_named_profiles() -> frozenset[str]:
    return frozenset(
        name
        for name in STAGE2_ROLLOUT_PROFILE_NAMES
        if resolve_stage2_rollout_profile(name).global_sink_frames == 1
    )


def _profile_axis(value: Any, label: str) -> tuple[int, ...]:
    values = _plain_list(value, label)
    if not values:
        raise ValueError(f"{label} must be non-empty")
    return tuple(
        _plain_positive_int(item, f"{label}[{index}]")
        for index, item in enumerate(values)
    )


def _build_profile_specs(value: Any) -> tuple[Stage2RolloutSpec, ...]:
    profile_set = _exact_mapping(value, "profile_set", _PROFILE_SET_KEYS)
    named = _plain_list(profile_set["named"], "profile_set.named")
    grids = _plain_list(profile_set["grids"], "profile_set.grids")
    cases = _plain_list(profile_set["cases"], "profile_set.cases")
    allowed_named = _s1_named_profiles()

    specs: list[Stage2RolloutSpec] = []
    for index, name in enumerate(named):
        if not isinstance(name, str) or name not in allowed_named:
            raise ValueError(
                f"profile_set.named[{index}] must be an existing S1 named profile"
            )
        specs.append(resolve_stage2_rollout_profile(name))

    grid_axes: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    expanded_count = len(specs) + len(cases)
    for index, raw_grid in enumerate(grids):
        label = f"profile_set.grids[{index}]"
        grid = _exact_mapping(raw_grid, label, _PROFILE_FIELDS)
        axes = (
            _profile_axis(grid["chunk_frames"], f"{label}.chunk_frames"),
            _profile_axis(grid["local_window_frames"], f"{label}.local_window_frames"),
            _profile_axis(grid["num_denoising_steps"], f"{label}.num_denoising_steps"),
        )
        expanded_count += len(axes[0]) * len(axes[1]) * len(axes[2])
        if expanded_count > STAGE2_INFERENCE_SWEEP_MAX_PROFILES:
            raise ValueError(
                "Stage-2 inference sweep expands beyond the maximum of "
                f"{STAGE2_INFERENCE_SWEEP_MAX_PROFILES} profiles"
            )
        grid_axes.append(axes)

    if expanded_count > STAGE2_INFERENCE_SWEEP_MAX_PROFILES:
        raise ValueError(
            "Stage-2 inference sweep expands beyond the maximum of "
            f"{STAGE2_INFERENCE_SWEEP_MAX_PROFILES} profiles"
        )

    for chunks, windows, steps in grid_axes:
        for chunk_frames, local_window_frames, num_denoising_steps in product(
            chunks, windows, steps
        ):
            specs.append(
                build_stage2_deployment_rollout_spec(
                    chunk_frames=chunk_frames,
                    local_window_frames=local_window_frames,
                    num_denoising_steps=num_denoising_steps,
                )
            )

    for index, raw_case in enumerate(cases):
        label = f"profile_set.cases[{index}]"
        case = _exact_mapping(raw_case, label, _PROFILE_FIELDS)
        specs.append(
            build_stage2_deployment_rollout_spec(
                chunk_frames=_plain_positive_int(
                    case["chunk_frames"], f"{label}.chunk_frames"
                ),
                local_window_frames=_plain_positive_int(
                    case["local_window_frames"], f"{label}.local_window_frames"
                ),
                num_denoising_steps=_plain_positive_int(
                    case["num_denoising_steps"], f"{label}.num_denoising_steps"
                ),
            )
        )

    if not specs:
        raise ValueError("profile_set must resolve at least one rollout profile")
    if len(specs) > STAGE2_INFERENCE_SWEEP_MAX_PROFILES:
        raise ValueError(
            "Stage-2 inference sweep expands beyond the maximum of "
            f"{STAGE2_INFERENCE_SWEEP_MAX_PROFILES} profiles"
        )

    specs.sort(key=_canonical_spec_key)
    topology_keys = [_topology_key(spec) for spec in specs]
    if len(topology_keys) != len(set(topology_keys)):
        raise ValueError("profile_set contains duplicate rollout topologies")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise RuntimeError("canonical Stage-2 sweep profile names collided")
    return tuple(specs)


def _canonical_int_set(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    values = _plain_list(value, label)
    if not values and not allow_empty:
        raise ValueError(f"{label} must be non-empty")
    normalized: list[int] = []
    for index, item in enumerate(values):
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not minimum <= item <= maximum
        ):
            raise ValueError(
                f"{label}[{index}] must be a plain integer in [{minimum}, {maximum}]"
            )
        normalized.append(item)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must contain unique values")
    return tuple(sorted(normalized))


def _resolve_evaluation(
    value: Any,
) -> tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    evaluation = _exact_mapping(value, "evaluation", _EVALUATION_KEYS)
    mode = evaluation["mode"]
    if mode not in {"formal", "quick"}:
        raise ValueError("evaluation.mode must be 'formal' or 'quick'")
    seeds = _canonical_int_set(
        evaluation["seeds"],
        "evaluation.seeds",
        minimum=0,
        maximum=(1 << 63) - 1,
    )
    single_rows = _canonical_int_set(
        evaluation["single_row_ids"],
        "evaluation.single_row_ids",
        minimum=0,
        maximum=5,
    )
    two_action_rows = _canonical_int_set(
        evaluation["two_action_row_ids"],
        "evaluation.two_action_row_ids",
        minimum=0,
        maximum=7,
        allow_empty=True,
    )
    if mode == "formal" and (
        seeds != _FORMAL_SEEDS
        or single_rows != _FORMAL_SINGLE_ROW_IDS
        or two_action_rows not in ((), _FORMAL_TWO_ACTION_ROW_IDS)
    ):
        raise ValueError(
            "formal evaluation requires seeds 1..4, single rows 0..5, "
            "and two-action rows 0..7 or [] to disable two-action generation"
        )
    return mode, seeds, single_rows, two_action_rows


def _resolve_base_config(
    raw_path: Any,
    *,
    sweep_source: Path,
) -> tuple[str, str, ResolvedStage2InferenceConfig]:
    if not isinstance(raw_path, str) or not raw_path or raw_path != raw_path.strip():
        raise ValueError("base_config must be a non-empty path string")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = sweep_source.parent / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    before_sha256 = sha256_file(candidate)
    base = load_stage2_inference_config(candidate)
    after_sha256 = sha256_file(candidate)
    if after_sha256 != before_sha256:
        raise RuntimeError("base_config changed while resolving the Stage-2 sweep")
    if base.profiles != (STAGE2_BASELINE_ROLLOUT_PROFILE,):
        raise ValueError(
            "Stage-2 inference sweep base_config must contain only the formal "
            f"baseline profile {STAGE2_BASELINE_ROLLOUT_PROFILE!r}"
        )
    return os.fspath(candidate.resolve()), before_sha256, base


@dataclass(frozen=True, slots=True)
class ResolvedStage2InferenceSweepConfig:
    """Canonical sweep plus the inherited strict inference runtime fields."""

    schema: str
    base_config_path: str
    base_config_sha256: str
    stage2_checkpoint: str
    source_cache_manifest: str
    architecture_root: str
    t5_checkpoint: str
    tokenizer_dir: str
    vae_checkpoint: str
    single_metadata: str
    two_action_metadata: str
    output_root: str
    profiles: tuple[str, ...]
    seeds: tuple[int, ...]
    dtype: str
    cfg_scale: float
    fps: int
    merge_ema_lora: bool
    batch_size_per_device: int
    evaluation_mode: str
    single_row_ids: tuple[int, ...]
    two_action_row_ids: tuple[int, ...]
    profile_set_sha256: str
    _rollout_specs: tuple[Stage2RolloutSpec, ...] = field(repr=False, compare=False)

    @property
    def rollout_specs(self) -> tuple[Stage2RolloutSpec, ...]:
        return self._rollout_specs

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "base_config_path": self.base_config_path,
            "base_config_sha256": self.base_config_sha256,
            "stage2_checkpoint": self.stage2_checkpoint,
            "source_cache_manifest": self.source_cache_manifest,
            "architecture_root": self.architecture_root,
            "t5_checkpoint": self.t5_checkpoint,
            "tokenizer_dir": self.tokenizer_dir,
            "vae_checkpoint": self.vae_checkpoint,
            "single_metadata": self.single_metadata,
            "two_action_metadata": self.two_action_metadata,
            "output_root": self.output_root,
            "profiles": list(self.profiles),
            "seeds": list(self.seeds),
            "dtype": self.dtype,
            "cfg_scale": self.cfg_scale,
            "fps": self.fps,
            "merge_ema_lora": self.merge_ema_lora,
            "batch_size_per_device": self.batch_size_per_device,
            "evaluation_mode": self.evaluation_mode,
            "single_row_ids": list(self.single_row_ids),
            "two_action_row_ids": list(self.two_action_row_ids),
            "profile_set_sha256": self.profile_set_sha256,
        }

    def contract_hash(self) -> str:
        value = self.to_dict()
        value.pop("output_root")
        return canonical_json_sha256(value)

    def launch_hash(self) -> str:
        return canonical_json_sha256(self.to_dict())


def resolve_stage2_inference_sweep_config(
    value: Any,
    *,
    source_path: str | os.PathLike[str],
) -> ResolvedStage2InferenceSweepConfig:
    """Resolve one sweep relative to the YAML file that declared it."""

    raw = _exact_mapping(value, "Stage-2 inference sweep config root", _ROOT_KEYS)
    if raw["schema"] != STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA:
        raise ValueError("Stage-2 inference sweep config schema mismatch")
    sweep_source = Path(source_path).expanduser().resolve()
    base_path, base_sha256, base = _resolve_base_config(
        raw["base_config"], sweep_source=sweep_source
    )
    specs = _build_profile_specs(raw["profile_set"])
    evaluation_mode, seeds, single_rows, two_action_rows = _resolve_evaluation(
        raw["evaluation"]
    )
    profile_payload = [spec.to_dict() for spec in specs]
    profile_set_sha256 = canonical_json_sha256(profile_payload)
    resolved = ResolvedStage2InferenceSweepConfig(
        schema=STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA,
        base_config_path=base_path,
        base_config_sha256=base_sha256,
        stage2_checkpoint=base.stage2_checkpoint,
        source_cache_manifest=base.source_cache_manifest,
        architecture_root=base.architecture_root,
        t5_checkpoint=base.t5_checkpoint,
        tokenizer_dir=base.tokenizer_dir,
        vae_checkpoint=base.vae_checkpoint,
        single_metadata=base.single_metadata,
        two_action_metadata=base.two_action_metadata,
        output_root=_plain_path(raw["output_root"], "output_root"),
        profiles=tuple(spec.name for spec in specs),
        seeds=seeds,
        dtype=base.dtype,
        cfg_scale=base.cfg_scale,
        fps=base.fps,
        merge_ema_lora=base.merge_ema_lora,
        batch_size_per_device=base.batch_size_per_device,
        evaluation_mode=evaluation_mode,
        single_row_ids=single_rows,
        two_action_row_ids=two_action_rows,
        profile_set_sha256=profile_set_sha256,
        _rollout_specs=specs,
    )
    json.dumps(resolved.to_dict(), allow_nan=False, sort_keys=True)
    return resolved


def load_stage2_inference_sweep_config(
    path: str | os.PathLike[str],
) -> ResolvedStage2InferenceSweepConfig:
    from omegaconf import OmegaConf

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    source = candidate.resolve()
    return resolve_stage2_inference_sweep_config(
        OmegaConf.load(source),
        source_path=source,
    )


__all__ = [
    "STAGE2_BASELINE_ROLLOUT_PROFILE",
    "STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA",
    "STAGE2_INFERENCE_SWEEP_MAX_PROFILES",
    "ResolvedStage2InferenceSweepConfig",
    "load_stage2_inference_sweep_config",
    "resolve_stage2_inference_sweep_config",
]
