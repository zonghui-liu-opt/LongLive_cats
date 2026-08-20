#!/usr/bin/env python3
"""Apply cumulative Stage-2 runtime fixes without requiring Git.

The transformer recognizes exact legacy/current source fragments, prepares the
entire multi-file result in memory, compiles and audits it, creates
content-addressed backups, and only then atomically replaces runtime sources.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

HOTFIX_API_VERSION = "longlive_stage2_dmd_runtime/v2"
PARAMETER_NAME_API_VERSION = "longlive_stage2_parameter_names/v1"
LORA_LOAD_API_VERSION = "longlive_stage2_lora_load/v1"
_PARAMETER_NAMES_SOURCE = '''"""Fail-closed parameter-name mapping across transparent module wrappers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

STAGE2_PARAMETER_NAME_API_VERSION = "longlive_stage2_parameter_names/v1"


def _validated_names(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be an iterable of parameter names")
    names = tuple(values)
    invalid = [
        name
        for name in names
        if not isinstance(name, str) or not name or name != name.strip()
    ]
    if invalid:
        raise ValueError(f"{label} contains invalid parameter names: {invalid}")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate parameter names")
    return names


def map_parameter_names_to_expected(
    actual_names: Iterable[str],
    expected_names: Iterable[str],
    *,
    label: str,
    require_complete: bool = True,
) -> OrderedDict[str, str]:
    """Map runtime FQNs to one immutable namespace without guessing.

    A runtime name may equal an expected name or add one or more complete
    dotted wrapper segments in front of it.  Zero matches, multiple matches,
    and two runtime names resolving to the same expected name are rejected.
    """

    actual = _validated_names(actual_names, label=f"{label} runtime names")
    expected = _validated_names(expected_names, label=f"{label} expected names")
    if not expected:
        raise ValueError(f"{label} expected parameter names are empty")
    expected_set = set(expected)

    mapped: dict[str, str] = {}
    owners: dict[str, str] = {}
    for actual_name in actual:
        suffixes = [actual_name]
        suffixes.extend(
            actual_name[index + 1 :]
            for index, character in enumerate(actual_name)
            if character == "."
        )
        matches = [suffix for suffix in suffixes if suffix in expected_set]
        if len(matches) != 1:
            raise ValueError(
                f"{label} parameter {actual_name!r} did not map uniquely to the "
                f"expected namespace: matches={matches}"
            )
        expected_name = matches[0]
        if expected_name in owners:
            raise ValueError(
                f"{label} parameter-name collision for {expected_name!r}: "
                f"{owners[expected_name]!r} and {actual_name!r}"
            )
        mapped[actual_name] = expected_name
        owners[expected_name] = actual_name

    if require_complete and set(owners) != expected_set:
        raise ValueError(
            f"{label} parameter-name mapping is incomplete: "
            f"missing={sorted(expected_set - set(owners))}, "
            f"extra={sorted(set(owners) - expected_set)}"
        )
    return OrderedDict((name, mapped[name]) for name in sorted(mapped))


__all__ = [
    "STAGE2_PARAMETER_NAME_API_VERSION",
    "map_parameter_names_to_expected",
]
'''


class HotfixError(RuntimeError):
    """Raised before writing when the target is unsafe or unrecognized."""


@dataclass(frozen=True)
class TransformResult:
    source: str
    changed: bool
    changed_units: tuple[str, ...]


@dataclass(frozen=True)
class HotfixResult:
    status: str
    target_path: Path
    backup_path: Path | None
    changed_units: tuple[str, ...]
    target_paths: tuple[Path, ...]
    backup_paths: tuple[Path, ...]
    changed_files: tuple[str, ...]


@dataclass(frozen=True)
class _PatchUnit:
    name: str
    legacy: str
    current: str


_PATCH_UNITS = (
    _PatchUnit(
        "callable_import",
        "from collections.abc import Iterable, Mapping",
        "from collections.abc import Callable, Iterable, Mapping",
    ),
    _PatchUnit(
        "runtime_api_version",
        '''class Stage2DMD(nn.Module):
    """Three owned roles plus explicit, gradient-audited loss reductions."""

    def __init__(
''',
        '''class Stage2DMD(nn.Module):
    """Three owned roles plus explicit, gradient-audited loss reductions."""

    RUNTIME_API_VERSION = "longlive_stage2_dmd_runtime/v2"

    def __init__(
''',
    ),
    _PatchUnit(
        "generator_timing_signature",
        """    def generator_distribution_matching_loss_from_models(
        self,
        *,
        branch: str,
        generated_future: torch.Tensor,
        noised_score: Stage2NoisedScoreInput | Stage2NoisedScorePair,
        conditional_dict: Mapping[str, torch.Tensor],
        real_unconditional_dict: Mapping[str, torch.Tensor],
    ) -> Stage2GeneratorLossOutput:
""",
        """    def generator_distribution_matching_loss_from_models(
        self,
        *,
        branch: str,
        generated_future: torch.Tensor,
        noised_score: Stage2NoisedScoreInput | Stage2NoisedScorePair,
        conditional_dict: Mapping[str, torch.Tensor],
        real_unconditional_dict: Mapping[str, torch.Tensor],
        timing_callback: Callable[[str, Callable[[], object]], object] | None = None,
    ) -> Stage2GeneratorLossOutput:
""",
    ),
    _PatchUnit(
        "generator_timing_calls",
        """        # These teachers are inference-only for a Generator update.  no_grad
        # also prevents an expensive x_hat->score input Jacobian from forming.
        with torch.no_grad():
            fake_raw_flow, _ = self.fake_score.forward_score(
                noisy_image_or_video=noisy_fake_score,
                conditional_dict=conditional_dict,
                frame_timestep=frame_timestep,
            )
            real_cond_raw_flow, _ = self.real_score.forward_score(
                noisy_image_or_video=real_teacher_input,
                conditional_dict=conditional_dict,
                frame_timestep=frame_timestep,
            )
            real_uncond_raw_flow, _ = self.real_score.forward_score(
                noisy_image_or_video=real_teacher_input,
                conditional_dict=real_unconditional_dict,
                frame_timestep=frame_timestep,
            )
""",
        """        # These teachers are inference-only for a Generator update.  no_grad
        # also prevents an expensive x_hat->score input Jacobian from forming.
        def measured(label: str, callback: Callable[[], object]) -> object:
            return (
                callback()
                if timing_callback is None
                else timing_callback(label, callback)
            )

        with torch.no_grad():
            fake_raw_flow, _ = measured(
                "fake_score",
                lambda: self.fake_score.forward_score(
                    noisy_image_or_video=noisy_fake_score,
                    conditional_dict=conditional_dict,
                    frame_timestep=frame_timestep,
                ),
            )
            real_cond_raw_flow, _ = measured(
                "real_cond",
                lambda: self.real_score.forward_score(
                    noisy_image_or_video=real_teacher_input,
                    conditional_dict=conditional_dict,
                    frame_timestep=frame_timestep,
                ),
            )
            real_uncond_raw_flow, _ = measured(
                "real_uncond",
                lambda: self.real_score.forward_score(
                    noisy_image_or_video=real_teacher_input,
                    conditional_dict=real_unconditional_dict,
                    frame_timestep=frame_timestep,
                ),
            )
""",
    ),
    _PatchUnit(
        "fake_score_timing_signature",
        """    def fake_score_flow_dsm_loss_from_model(
        self,
        *,
        generated_future: torch.Tensor,
        noised_fake_score: Stage2NoisedScoreInput,
        conditional_dict: Mapping[str, torch.Tensor],
    ) -> Stage2FakeScoreLossOutput:
""",
        """    def fake_score_flow_dsm_loss_from_model(
        self,
        *,
        generated_future: torch.Tensor,
        noised_fake_score: Stage2NoisedScoreInput,
        conditional_dict: Mapping[str, torch.Tensor],
        timing_callback: Callable[[str, Callable[[], object]], object] | None = None,
    ) -> Stage2FakeScoreLossOutput:
""",
    ),
    _PatchUnit(
        "fake_score_timing_call",
        """        fake_raw_flow, _ = self.fake_score.forward_score(
            noisy_image_or_video=noisy_fake_score,
            conditional_dict=conditional_dict,
            frame_timestep=noising.frame_timestep,
        )
""",
        """        def callback():
            return self.fake_score.forward_score(
                noisy_image_or_video=noisy_fake_score,
                conditional_dict=conditional_dict,
                frame_timestep=noising.frame_timestep,
            )

        fake_raw_flow, _ = (
            callback()
            if timing_callback is None
            else timing_callback("fake_score", callback)
        )
""",
    ),
)


_TRAINER_PATCH_UNITS = (
    _PatchUnit(
        "timing_runtime_fields",
        """_STAGE2_DMD_RUNTIME_REPAIR = (
    "Run scripts/apply_stage2_innernet_hotfix.py from the project root, or "
    "deploy one complete stage-2 source snapshot; do not mix trainer and model files."
)
_STAGE2_DMD_RUNTIME_METHODS = {
""",
        """_STAGE2_DMD_RUNTIME_REPAIR = (
    "Run scripts/apply_stage2_innernet_hotfix.py from the project root, or "
    "deploy one complete stage-2 source snapshot; do not mix trainer and model files."
)
_STAGE2_TIMING_RUNTIME_FIELDS = (
    "data_seconds_max",
    "h2d_seconds_max",
    "rollout_seconds_max",
    "fake_score_seconds_max",
    "real_cond_seconds_max",
    "real_uncond_seconds_max",
    "loss_build_seconds_max",
    "backward_seconds_max",
    "clip_optimizer_seconds_max",
    "orchestration_seconds_max",
    "ema_seconds_max",
)
_STAGE2_DMD_RUNTIME_METHODS = {
""",
    ),
    _PatchUnit(
        "timing_runtime_audit",
        """        methods[method_name] = str(signature)
    return {
        "api_version": actual_version,
        "model_type": f"{model_type.__module__}.{model_type.__qualname__}",
        "source_file": str(Path(source_file).expanduser().resolve()),
        "methods": methods,
    }
""",
        """        methods[method_name] = str(signature)
    from utils.stage2_metrics import STAGE2_TIMING_FIELDS

    actual_timing_fields = tuple(STAGE2_TIMING_FIELDS)
    if actual_timing_fields != _STAGE2_TIMING_RUNTIME_FIELDS:
        raise RuntimeError(
            "Stage-2 timing runtime API mismatch: "
            f"expected={list(_STAGE2_TIMING_RUNTIME_FIELDS)}, "
            f"actual={list(actual_timing_fields)}. {_STAGE2_DMD_RUNTIME_REPAIR}"
        )
    return {
        "api_version": actual_version,
        "model_type": f"{model_type.__module__}.{model_type.__qualname__}",
        "source_file": str(Path(source_file).expanduser().resolve()),
        "methods": methods,
        "timing_fields": actual_timing_fields,
    }
""",
    ),
    _PatchUnit(
        "timing_summary_category",
        """            "backward",
            "clip_optimizer",
            "ema",
""",
        """            "backward",
            "clip_optimizer",
            "orchestration",
            "ema",
""",
    ),
    _PatchUnit(
        "timing_summary_field",
        """            "backward_seconds_max": selected["backward"],
            "clip_optimizer_seconds_max": selected["clip_optimizer"],
            "compute_seconds_max": compute,
""",
        """            "backward_seconds_max": selected["backward"],
            "clip_optimizer_seconds_max": selected["clip_optimizer"],
            "orchestration_seconds_max": selected["orchestration"],
            "compute_seconds_max": compute,
""",
    ),
    _PatchUnit(
        "attempt_wall_start",
        """        from utils.distributed import fsdp2_accumulation

        module = getattr(self.model, role)
""",
        """        from utils.distributed import fsdp2_accumulation

        attempt_started = time.perf_counter()
        module = getattr(self.model, role)
""",
    ),
    _PatchUnit(
        "attempt_orchestration_measurement",
        """        numerator, count = self._reduce_loss(local_numerator, local_count)
        merged = self._reduce_diagnostics(diagnostics)
        return {
""",
        """        numerator, count = self._reduce_loss(local_numerator, local_count)
        merged = self._reduce_diagnostics(diagnostics)
        torch.cuda.synchronize(self.device)
        attempt_seconds = time.perf_counter() - attempt_started
        classified_attempt_seconds = sum(phase_timings.values()) + optimizer_seconds
        attempt_orchestration_seconds = max(
            0.0, attempt_seconds - classified_attempt_seconds
        )
        return {
""",
    ),
    _PatchUnit(
        "attempt_orchestration_result",
        """            "compute_seconds": compute_seconds,
            "optimizer_seconds": optimizer_seconds,
            "phase_timings": phase_timings,
""",
        """            "compute_seconds": compute_seconds,
            "optimizer_seconds": optimizer_seconds,
            "attempt_seconds": attempt_seconds,
            "attempt_orchestration_seconds": attempt_orchestration_seconds,
            "phase_timings": phase_timings,
""",
    ),
    _PatchUnit(
        "pre_attempt_control_start",
        """            h2d_seconds = time.perf_counter() - h2d_started
            exits = self.exit_rng.draw(
""",
        """            h2d_seconds = time.perf_counter() - h2d_started
            control_started = time.perf_counter()
            exits = self.exit_rng.draw(
""",
    ),
    _PatchUnit(
        "pre_attempt_control_elapsed",
        """            branch = (
                self._draw_branch(probability) if role == "generator" else "flow_dsm"
            )
            result = self._run_update_attempt(
""",
        """            branch = (
                self._draw_branch(probability) if role == "generator" else "flow_dsm"
            )
            control_seconds = time.perf_counter() - control_started
            result = self._run_update_attempt(
""",
    ),
    _PatchUnit(
        "post_attempt_control_start",
        """            result = self._run_update_attempt(
                role=role, batches=batches, exits=exits, branch=branch
            )
            torch.cuda.synchronize(self.device)
""",
        """            result = self._run_update_attempt(
                role=role, batches=batches, exits=exits, branch=branch
            )
            post_attempt_control_started = time.perf_counter()
            torch.cuda.synchronize(self.device)
""",
    ),
    _PatchUnit(
        "ema_control_split",
        """            if role == "generator":
                next_completed_g = self.state.completed_g + 1
                ema_started = time.perf_counter()
                ema_action = self._runtime_world_checked(
""",
        """            if role == "generator":
                next_completed_g = self.state.completed_g + 1
                ema_started = time.perf_counter()
                control_seconds += ema_started - post_attempt_control_started
                ema_action = self._runtime_world_checked(
""",
    ),
    _PatchUnit(
        "post_ema_control_restart",
        """                )
                ema_seconds = time.perf_counter() - ema_started
                self._runtime_world_checked(
                    "commit generator clock",
""",
        """                )
                ema_seconds = time.perf_counter() - ema_started
                post_attempt_control_started = time.perf_counter()
                self._runtime_world_checked(
                    "commit generator clock",
""",
    ),
    _PatchUnit(
        "post_attempt_control_elapsed",
        """            self._assert_state_consensus()
            torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
""",
        """            self._assert_state_consensus()
            torch.cuda.synchronize(self.device)
            control_seconds += time.perf_counter() - post_attempt_control_started
            elapsed = time.perf_counter() - started
""",
    ),
    _PatchUnit(
        "logical_substep_orchestration_category",
        """                    ),
                    "clip_optimizer": result["optimizer_seconds"],
                    "ema": ema_seconds,
""",
        """                    ),
                    "clip_optimizer": result["optimizer_seconds"],
                    "orchestration": result.get("attempt_orchestration_seconds", 0.0)
                    + control_seconds,
                    "ema": ema_seconds,
        """,
    ),
    _PatchUnit(
        "generator_ema_schema_names",
        """        self.generator_ema = TrainableShardedEMA(
            self.model.generator,
            decay=self.resolved.ema_decay,
            start_step=self.resolved.ema_initialize_at_completed_generator_update,
            topology={
""",
        """        self.generator_ema = TrainableShardedEMA(
            self.model.generator,
            decay=self.resolved.ema_decay,
            start_step=self.resolved.ema_initialize_at_completed_generator_update,
            expected_parameter_names=tuple(
                spec.raw_parameter_name
                for spec in self.lora_schemas["generator"].values()
            ),
            topology={
""",
    ),
)


_METRICS_PATCH_UNITS = (
    _PatchUnit(
        "orchestration_timing_field",
        """    "backward_seconds_max",
    "clip_optimizer_seconds_max",
    "ema_seconds_max",
""",
        """    "backward_seconds_max",
    "clip_optimizer_seconds_max",
    "orchestration_seconds_max",
    "ema_seconds_max",
""",
    ),
)


_PLOT_PATCH_UNITS = (
    _PatchUnit(
        "orchestration_plot_label",
        """        "backward_seconds_max": "Backward",
        "clip_optimizer_seconds_max": "Clip + optimizer",
        "ema_seconds_max": "EMA",
""",
        """        "backward_seconds_max": "Backward",
        "clip_optimizer_seconds_max": "Clip + optimizer",
        "orchestration_seconds_max": "Runtime orchestration + audits",
        "ema_seconds_max": "EMA",
""",
    ),
)


_DISTRIBUTED_PATCH_UNITS = (
    _PatchUnit(
        "parameter_name_import",
        """import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
""",
        """import torch
import torch.distributed as dist
from utils.parameter_names import map_parameter_names_to_expected
from torch.distributed.fsdp import (
""",
    ),
    _PatchUnit(
        "ema_expected_name_signature",
        """        require_lora_only: bool = True,
        topology: Mapping[str, Any] | None = None,
    ):
""",
        """        require_lora_only: bool = True,
        topology: Mapping[str, Any] | None = None,
        expected_parameter_names: Sequence[str] | None = None,
    ):
""",
    ),
    _PatchUnit(
        "ema_expected_name_state",
        """        self.require_lora_only = bool(require_lora_only)
        self.topology = dict(topology or {})
        self.shadow: dict[str, torch.Tensor] = {}
""",
        """        self.require_lora_only = bool(require_lora_only)
        self.topology = dict(topology or {})
        self.expected_parameter_names = (
            tuple(expected_parameter_names)
            if expected_parameter_names is not None
            else None
        )
        self.shadow: dict[str, torch.Tensor] = {}
""",
    ),
    _PatchUnit(
        "ema_runtime_to_schema_mapping",
        """                raise ValueError(
                    "TrainableShardedEMA only accepts LoRA trainables; "
                    f"found {non_lora}"
                )
        return dict(sorted(parameters.items()))

    def _validate_local_topology(
""",
        '''                raise ValueError(
                    "TrainableShardedEMA only accepts LoRA trainables; "
                    f"found {non_lora}"
                )
        if self.expected_parameter_names is not None:
            mapping = map_parameter_names_to_expected(
                parameters,
                self.expected_parameter_names,
                label="TrainableShardedEMA",
            )
            parameters = {
                mapping[name]: parameter for name, parameter in parameters.items()
            }
        return dict(sorted(parameters.items()))

    def _normalize_state_parameter_names(
        self, state_dict: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Upgrade legacy wrapper-prefixed EMA keys before strict validation."""

        if self.expected_parameter_names is None:
            return state_dict
        local_shapes = state_dict.get("local_shapes")
        if not isinstance(local_shapes, Mapping):
            return state_dict
        mapping = map_parameter_names_to_expected(
            local_shapes,
            self.expected_parameter_names,
            label="EMA checkpoint",
        )
        runtime_names = set(mapping)
        normalized = dict(state_dict)
        for field in ("local_shapes", "global_shapes", "shard_metadata"):
            values = state_dict.get(field)
            if not isinstance(values, Mapping) or set(values) != runtime_names:
                raise ValueError(
                    f"EMA checkpoint {field} names differ from local_shapes"
                )
            normalized[field] = {mapping[name]: value for name, value in values.items()}
        shadows = state_dict.get("shadow")
        if isinstance(shadows, Mapping):
            if shadows and set(shadows) != runtime_names:
                raise ValueError("EMA checkpoint shadow names differ from local_shapes")
            normalized["shadow"] = {
                mapping[name]: value for name, value in shadows.items()
            }
        return normalized

    def _validate_local_topology(
''',
    ),
    _PatchUnit(
        "ema_load_name_normalization",
        """    def load_state_dict(
        self, state_dict: Mapping[str, Any], module: torch.nn.Module
    ) -> None:
        rank, world_size = self._rank_and_world_size()
""",
        """    def load_state_dict(
        self, state_dict: Mapping[str, Any], module: torch.nn.Module
    ) -> None:
        state_dict = self._normalize_state_parameter_names(state_dict)
        rank, world_size = self._rank_and_world_size()
""",
    ),
)


_LORA_PATCH_UNITS = (
    _PatchUnit(
        "lora_load_api_version",
        """_LORA_PARAMETER_MARKERS = (".lora_A.", ".lora_B.")
_CANONICAL_LORA_KEY_MARKERS = (".lora_A.weight", ".lora_B.weight")


@dataclass(frozen=True)
""",
        """_LORA_PARAMETER_MARKERS = (".lora_A.", ".lora_B.")
_CANONICAL_LORA_KEY_MARKERS = (".lora_A.weight", ".lora_B.weight")
STAGE2_LORA_LOAD_API_VERSION = "longlive_stage2_lora_load/v1"


@dataclass(frozen=True)
""",
    ),
    _PatchUnit(
        "distributed_safe_lora_load_doc",
        '''    """Strictly validate and load a complete canonical PEFT adapter state."""
''',
        '''    """Strictly load canonical default-adapter LoRA A/B tensors.

    PEFT's generic loader probes Hugging Face tensor parallelism whenever a
    distributed process group exists, even when this model uses only FSDP2.
    That optional probe makes checkpoint resume depend on a Transformers
    integration which is irrelevant to this project.  Stage-2 has a narrower
    contract: one complete default LoRA adapter is loaded before FSDP wrapping.
    Resolve that exact canonical/runtime bijection here and use PyTorch's
    native partial state load while retaining the existing value audit.
    """
''',
    ),
    _PatchUnit(
        "distributed_safe_lora_load",
        """    incompatible = peft.set_peft_model_state_dict(lora_model, validated)
    mismatched = list(getattr(incompatible, "mismatched_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if mismatched or unexpected:
        raise ValueError(
            "PEFT rejected adapter tensors: "
            f"mismatched={mismatched}, unexpected={unexpected}"
        )
""",
        """    runtime_by_canonical: dict[str, str] = {}
    for runtime_name, _parameter in lora_model.named_parameters():
        if not _is_lora_parameter_name(runtime_name):
            continue
        canonical_key = _canonical_key_from_parameter_name(runtime_name)
        previous = runtime_by_canonical.get(canonical_key)
        if previous is not None:
            raise ValueError(
                "canonical LoRA key maps to multiple runtime parameters: "
                f"key={canonical_key!r}, parameters={[previous, runtime_name]}"
            )
        runtime_by_canonical[canonical_key] = runtime_name

    expected_keys = set(validated)
    runtime_keys = set(runtime_by_canonical)
    if runtime_keys != expected_keys:
        raise ValueError(
            "canonical/runtime LoRA parameter mapping is incomplete: "
            f"missing={sorted(expected_keys - runtime_keys)}, "
            f"extra={sorted(runtime_keys - expected_keys)}"
        )
    runtime_state = OrderedDict(
        (runtime_by_canonical[key], validated[key]) for key in sorted(validated)
    )
    incompatible = lora_model.load_state_dict(runtime_state, strict=False)
    missing_adapter = sorted(set(incompatible.missing_keys).intersection(runtime_state))
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing_adapter or unexpected:
        raise ValueError(
            "PyTorch rejected canonical LoRA adapter tensors: "
            f"missing={missing_adapter}, unexpected={unexpected}"
        )
""",
    ),
    _PatchUnit(
        "parameter_name_import",
        """from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

_LORA_PARAMETER_MARKERS = (".lora_A.", ".lora_B.")
""",
        """from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from utils.parameter_names import map_parameter_names_to_expected

_LORA_PARAMETER_MARKERS = (".lora_A.", ".lora_B.")
""",
    ),
    _PatchUnit(
        "canonical_schema_mapping",
        """def _match_expected_adapter_key(
    candidate: str,
    expected_schema: Mapping[str, LoraTensorSpec],
) -> str:
    matches = [
        key
        for key in expected_schema
        if candidate == key or candidate.endswith(f".{key}")
    ]
    if len(matches) != 1:
        raise ValueError(
            "FSDP LoRA parameter did not map uniquely to the pre-FSDP schema: "
            f"parameter={candidate!r}, matches={matches}"
        )
    return matches[0]
""",
        """def _match_expected_adapter_key(
    candidate: str,
    expected_schema: Mapping[str, LoraTensorSpec],
) -> str:
    return map_parameter_names_to_expected(
        (candidate,),
        expected_schema,
        label="FSDP LoRA canonical key",
        require_complete=False,
    )[candidate]
""",
    ),
    _PatchUnit(
        "raw_schema_mapping",
        """        cleaned_raw_name = _clean_fsdp_parameter_name(raw_name)
        if not (
            cleaned_raw_name == spec.raw_parameter_name
            or cleaned_raw_name.endswith(f".{spec.raw_parameter_name}")
        ):
            raise ValueError(
                "post-FSDP parameter name differs from pre-FSDP schema: "
                f"current={cleaned_raw_name!r}, expected={spec.raw_parameter_name!r}"
            )
        if parameter.dtype != spec.dtype:
""",
        """        cleaned_raw_name = _clean_fsdp_parameter_name(raw_name)
        map_parameter_names_to_expected(
            (cleaned_raw_name,),
            (spec.raw_parameter_name,),
            label="post-FSDP LoRA raw name",
        )
        if parameter.dtype != spec.dtype:
""",
    ),
)


_STAGE2_FSDP2_PATCH_UNITS = (
    _PatchUnit(
        "parameter_name_import",
        """from utils.lora_utils import LoraTensorSpec

STAGE2_FSDP2_WORLD_SIZE = 8
""",
        """from utils.lora_utils import LoraTensorSpec
from utils.parameter_names import map_parameter_names_to_expected

STAGE2_FSDP2_WORLD_SIZE = 8
""",
    ),
    _PatchUnit(
        "post_fsdp_schema_map",
        """    frozen_tensor_count = 0
    global_frozen_parameters = 0
    canonical_keys: list[str] = []
    for name, parameter in named_parameters:
""",
        """    frozen_tensor_count = 0
    global_frozen_parameters = 0
    canonical_keys: list[str] = []
    runtime_to_raw: Mapping[str, str] = {}
    raw_to_key: dict[str, str] = {}
    if expected_schema is not None:
        raw_to_key = {
            spec.raw_parameter_name: key for key, spec in expected_schema.items()
        }
        if len(raw_to_key) != len(expected_schema):
            raise ValueError(f"Stage-2 {role} schema has duplicate raw parameter names")
        runtime_to_raw = map_parameter_names_to_expected(
            (name for name, _ in trainable),
            raw_to_key,
            label=f"Stage-2 {role} post-FSDP LoRA",
        )
    for name, parameter in named_parameters:
""",
    ),
    _PatchUnit(
        "post_fsdp_schema_lookup",
        """        assert expected_schema is not None
        matches = [
            key
            for key, spec in expected_schema.items()
            if name.endswith(spec.raw_parameter_name)
            or name.endswith(f".{spec.raw_parameter_name}")
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Stage-2 {role} post-FSDP parameter does not map to schema: {name}"
            )
        key = matches[0]
""",
        """        assert expected_schema is not None
        key = raw_to_key[runtime_to_raw[name]]
""",
    ),
)


_STAGE2_CHECKPOINT_PATCH_UNITS = (
    _PatchUnit(
        "parameter_name_import",
        """from utils.lora_utils import LocalLoraShard, LoraTensorSpec
from utils.stage2_code_version import capture_stage2_source_version
""",
        """from utils.lora_utils import LocalLoraShard, LoraTensorSpec
from utils.parameter_names import map_parameter_names_to_expected
from utils.stage2_code_version import capture_stage2_source_version
""",
    ),
    _PatchUnit(
        "schema_raw_name_helpers",
        """def _is_lora_name(name: str) -> bool:
    return any(marker in name for marker in ("lora_A", "lora_B"))


def _validate_stage2_optimizer_param_groups(
""",
        """def _is_lora_name(name: str) -> bool:
    return any(marker in name for marker in ("lora_A", "lora_B"))


def _schema_raw_parameter_specs(
    schema: Mapping[str, LoraTensorSpec],
    *,
    role: str,
) -> OrderedDict[str, LoraTensorSpec]:
    raw = OrderedDict(
        (spec.raw_parameter_name, spec) for _, spec in sorted(schema.items())
    )
    if len(raw) != len(schema) or any(
        not isinstance(name, str) or not name or name != name.strip() for name in raw
    ):
        raise ValueError(
            f"Stage-2 {role} schema has duplicate or invalid raw parameter names"
        )
    return raw


def _map_parameter_names_to_schema_raw(
    names: Iterable[str],
    schema: Mapping[str, LoraTensorSpec],
    *,
    role: str,
    label: str,
) -> OrderedDict[str, str]:
    raw_specs = _schema_raw_parameter_specs(schema, role=role)
    return map_parameter_names_to_expected(
        names,
        raw_specs,
        label=f"Stage-2 {role} {label}",
    )


def _validate_stage2_optimizer_param_groups(
""",
    ),
    _PatchUnit(
        "optimizer_runtime_schema_audit",
        """    if expected_schema is not None:
        mapped: set[str] = set()
        for name, parameter in named.items():
            matches = [
                key
                for key, spec in expected_schema.items()
                if name == spec.raw_parameter_name
                or name.endswith(f".{spec.raw_parameter_name}")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Stage-2 {role} optimizer parameter {name!r} does not map "
                    f"uniquely to its role schema: {matches}"
                )
            spec = expected_schema[matches[0]]
            if tuple(parameter.shape) != tuple(spec.global_shape):
                raise ValueError(
                    f"Stage-2 {role} optimizer parameter shape drift: {name}"
                )
            mapped.add(matches[0])
        if mapped != set(expected_schema):
            raise ValueError(f"Stage-2 {role} optimizer role schema is incomplete")
""",
        """    mapped_names: Mapping[str, str] | None = None
    if expected_schema is not None:
        mapped_names = _map_parameter_names_to_schema_raw(
            named,
            expected_schema,
            role=role,
            label="optimizer runtime names",
        )
        raw_specs = _schema_raw_parameter_specs(expected_schema, role=role)
        for name, parameter in named.items():
            spec = raw_specs[mapped_names[name]]
            if tuple(parameter.shape) != tuple(spec.global_shape):
                raise ValueError(
                    f"Stage-2 {role} optimizer parameter shape drift: {name}"
                )
""",
    ),
    _PatchUnit(
        "optimizer_audit_canonical_return",
        """    return tuple(named)


def _validate_adam_values(
""",
        """    return tuple(
        mapped_names[name] if mapped_names is not None else name for name in named
    )


def _validate_adam_values(
""",
    ),
    _PatchUnit(
        "optimizer_state_name_transforms",
        """    return state


def _dcp_optimizer_apis():
""",
        '''    return state


def _rename_stage2_optimizer_state_parameters(
    state: Mapping[str, Any],
    mapping: Mapping[str, str],
    *,
    role: str,
) -> Mapping[str, Any]:
    """Rename both DCP moment keys and param-group FQNs atomically."""

    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
        raise ValueError(f"Stage-2 {role} optimizer state has invalid containers")
    moments = state["state"]
    groups = state["param_groups"]
    if (
        not isinstance(moments, Mapping)
        or isinstance(groups, (str, bytes))
        or not isinstance(groups, Sequence)
    ):
        raise TypeError(f"Stage-2 {role} optimizer state has invalid containers")
    source_names = set(mapping)
    if set(moments) != source_names:
        raise ValueError(
            f"Stage-2 {role} optimizer moment names differ from param groups"
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"Stage-2 {role} optimizer parameter rename collides")
    renamed_groups: list[dict[str, Any]] = []
    grouped_names: list[str] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise TypeError(f"Stage-2 {role} optimizer param group is invalid")
        params = group.get("params")
        if isinstance(params, (str, bytes)) or not isinstance(params, Sequence):
            raise TypeError(f"Stage-2 {role} optimizer params must be FQNs")
        if any(name not in mapping for name in params):
            raise ValueError(
                f"Stage-2 {role} optimizer param groups contain unknown names"
            )
        grouped_names.extend(params)
        renamed_groups.append(
            {
                **dict(group),
                "params": [mapping[name] for name in params],
            }
        )
    if (
        len(grouped_names) != len(set(grouped_names))
        or set(grouped_names) != source_names
    ):
        raise ValueError(
            f"Stage-2 {role} optimizer param groups/moments are inconsistent"
        )
    if all(source == target for source, target in mapping.items()):
        return state
    return {
        "state": OrderedDict(
            (mapping[name], moments[name]) for name in sorted(source_names)
        ),
        "param_groups": renamed_groups,
    }


def _canonicalize_stage2_optimizer_state(
    state: Mapping[str, Any],
    schema: Mapping[str, LoraTensorSpec],
    *,
    role: str,
) -> Mapping[str, Any]:
    names = _optimizer_names(state)
    mapping = _map_parameter_names_to_schema_raw(
        names,
        schema,
        role=role,
        label="optimizer checkpoint names",
    )
    return _rename_stage2_optimizer_state_parameters(state, mapping, role=role)


def _optimizer_state_for_runtime(
    state: Mapping[str, Any],
    module: torch.nn.Module,
    schema: Mapping[str, LoraTensorSpec],
    *,
    role: str,
) -> Mapping[str, Any]:
    canonical = _canonicalize_stage2_optimizer_state(state, schema, role=role)
    runtime_names = tuple(
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    )
    runtime_to_raw = _map_parameter_names_to_schema_raw(
        runtime_names,
        schema,
        role=role,
        label="optimizer restore runtime names",
    )
    raw_to_runtime = {raw: runtime for runtime, raw in runtime_to_raw.items()}
    if len(raw_to_runtime) != len(runtime_to_raw):
        raise ValueError(f"Stage-2 {role} optimizer runtime name mapping collides")
    return _rename_stage2_optimizer_state_parameters(
        canonical,
        raw_to_runtime,
        role=role,
    )


def _dcp_optimizer_apis():
''',
    ),
    _PatchUnit(
        "optimizer_gather_canonicalization",
        """    local_error: Exception | None = None
    if rank == 0:
        try:
            validate_stage2_full_optimizer_state(
                full_state,
                role=role,
""",
        """    local_error: Exception | None = None
    canonical_state: Mapping[str, Any] | None = None
    if rank == 0:
        try:
            canonical_state = _canonicalize_stage2_optimizer_state(
                full_state,
                expected_schema,
                role=role,
            )
            validate_stage2_full_optimizer_state(
                canonical_state,
                role=role,
""",
    ),
    _PatchUnit(
        "optimizer_gather_canonical_return",
        """    ops.barrier()
    return full_state if rank == 0 else None


def restore_stage2_optimizer_state(
""",
        """    ops.barrier()
    return canonical_state if rank == 0 else None


def restore_stage2_optimizer_state(
""",
    ),
    _PatchUnit(
        "optimizer_restore_name_transform",
        """    local_error: Exception | None = None
    if rank == 0:
        try:
            if optimizer_state is None:
                raise RuntimeError(f"rank0 is missing {role} optimizer state")
            validate_stage2_full_optimizer_state(
                optimizer_state,
                role=role,
                expected_parameter_names=names,
                expected_completed_updates=expected_completed_updates,
            )
""",
        """    local_error: Exception | None = None
    runtime_optimizer_state: Mapping[str, Any] | None = None
    if rank == 0:
        try:
            if optimizer_state is None:
                raise RuntimeError(f"rank0 is missing {role} optimizer state")
            canonical_optimizer_state = _canonicalize_stage2_optimizer_state(
                optimizer_state,
                expected_schema,
                role=role,
            )
            validate_stage2_full_optimizer_state(
                canonical_optimizer_state,
                role=role,
                expected_parameter_names=names,
                expected_completed_updates=expected_completed_updates,
                expected_optimizer_spec=expected_optimizer_spec,
            )
            runtime_optimizer_state = _optimizer_state_for_runtime(
                canonical_optimizer_state,
                module,
                expected_schema,
                role=role,
            )
""",
    ),
    _PatchUnit(
        "optimizer_restore_runtime_payload",
        """            optimizer_state if rank == 0 else {},
            options=options,
""",
        """            runtime_optimizer_state if rank == 0 else {},
            options=options,
""",
    ),
    _PatchUnit(
        "prepared_optimizer_return_types",
        """    OrderedDict[str, torch.Tensor],
    OrderedDict[str, torch.Tensor],
    OrderedDict[str, torch.Tensor] | None,
    dict[str, Any],
]:
""",
        """    OrderedDict[str, torch.Tensor],
    OrderedDict[str, torch.Tensor],
    OrderedDict[str, torch.Tensor] | None,
    Mapping[str, Any],
    Mapping[str, Any],
    dict[str, Any],
]:
""",
    ),
    _PatchUnit(
        "prepared_optimizer_canonicalization",
        """    generator_names = _optimizer_names_for_schema(
        generator_optimizer_state, generator_schema, role="generator"
    )
    fake_names = _optimizer_names_for_schema(
        fake_score_optimizer_state, fake_score_schema, role="fake_score"
    )
    validate_stage2_full_optimizer_state(
        generator_optimizer_state,
""",
        """    canonical_generator_optimizer = _canonicalize_stage2_optimizer_state(
        generator_optimizer_state,
        generator_schema,
        role="generator",
    )
    canonical_fake_optimizer = _canonicalize_stage2_optimizer_state(
        fake_score_optimizer_state,
        fake_score_schema,
        role="fake_score",
    )
    generator_names = _optimizer_names_for_schema(
        canonical_generator_optimizer, generator_schema, role="generator"
    )
    fake_names = _optimizer_names_for_schema(
        canonical_fake_optimizer, fake_score_schema, role="fake_score"
    )
    validate_stage2_full_optimizer_state(
        canonical_generator_optimizer,
""",
    ),
    _PatchUnit(
        "prepared_fake_optimizer_validation",
        """    validate_stage2_full_optimizer_state(
        fake_score_optimizer_state,
        role="fake_score",
""",
        """    validate_stage2_full_optimizer_state(
        canonical_fake_optimizer,
        role="fake_score",
""",
    ),
    _PatchUnit(
        "prepared_optimizer_return",
        """    return raw_g, raw_f, ema, _validate_topology(topology)


def _optimizer_names_for_schema(
""",
        """    return (
        raw_g,
        raw_f,
        ema,
        canonical_generator_optimizer,
        canonical_fake_optimizer,
        _validate_topology(topology),
    )


def _optimizer_names_for_schema(
""",
    ),
    _PatchUnit(
        "optimizer_payload_schema_mapping",
        """    names = _optimizer_names(optimizer_state)
    mapped: set[str] = set()
    for name in names:
        matches = [
            key
            for key, spec in schema.items()
            if name == spec.raw_parameter_name
            or name.endswith(f".{spec.raw_parameter_name}")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Stage-2 {role} optimizer parameter {name!r} does not map "
                f"uniquely to the role schema: {matches}"
            )
        mapped.add(matches[0])
    if mapped != set(schema) or len(names) != len(schema):
        raise ValueError(f"Stage-2 {role} optimizer role/schema mapping is incomplete")
    return names
""",
        """    names = _optimizer_names(optimizer_state)
    mapping = _map_parameter_names_to_schema_raw(
        names,
        schema,
        role=role,
        label="optimizer payload names",
    )
    return tuple(mapping[name] for name in names)
""",
    ),
    _PatchUnit(
        "prepared_optimizer_unpack",
        """        raw_g, raw_f, ema, audited_topology = _validate_prepared_payloads(
""",
        """        (
            raw_g,
            raw_f,
            ema,
            generator_optimizer_state,
            fake_score_optimizer_state,
            audited_topology,
        ) = _validate_prepared_payloads(
""",
    ),
)


_EXPECTED_METHODS = {
    "fake_score_flow_dsm_loss_from_model": (
        "generated_future",
        "noised_fake_score",
        "conditional_dict",
        "timing_callback",
    ),
    "generator_distribution_matching_loss_from_models": (
        "branch",
        "generated_future",
        "noised_score",
        "conditional_dict",
        "real_unconditional_dict",
        "timing_callback",
    ),
}


def _apply_exact_unit(source: str, unit: _PatchUnit) -> tuple[str, bool]:
    legacy_count = source.count(unit.legacy)
    current_count = source.count(unit.current)
    if legacy_count == 1 and current_count == 0:
        replacement = unit.current
        if unit.name == "optimizer_restore_name_transform":
            optimizer_spec_signature = """    expected_completed_updates: int,
    expected_optimizer_spec: Any | None = None,
    collectives: Stage2CollectiveOps | None = None,
"""
            optimizer_spec_argument = (
                "                expected_optimizer_spec=expected_optimizer_spec,\n"
            )
            if optimizer_spec_signature not in source:
                if replacement.count(optimizer_spec_argument) != 1:
                    raise HotfixError(
                        "optimizer restore hotfix has an invalid optimizer-spec fragment"
                    )
                replacement = replacement.replace(optimizer_spec_argument, "", 1)
        return source.replace(unit.legacy, replacement, 1), True
    if legacy_count == 0 and current_count == 1:
        return source, False
    raise HotfixError(
        f"unrecognized or partial Stage-2 source at {unit.name}: "
        f"legacy_matches={legacy_count}, current_matches={current_count}"
    )


def _validate_current_ast(source: str) -> None:
    try:
        tree = ast.parse(source, filename="model/stage2_dmd.py")
        compile(source, "model/stage2_dmd.py", "exec")
    except (SyntaxError, ValueError) as error:
        raise HotfixError(
            f"transformed Stage-2 source does not compile: {error}"
        ) from error

    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Stage2DMD"
    ]
    if len(classes) != 1:
        raise HotfixError(f"expected one Stage2DMD class, found {len(classes)}")
    model_class = classes[0]
    version_values = [
        statement.value.value
        for statement in model_class.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "RUNTIME_API_VERSION"
        and isinstance(statement.value, ast.Constant)
    ]
    if version_values != [HOTFIX_API_VERSION]:
        raise HotfixError(
            "Stage2DMD runtime API version is missing or unexpected: "
            f"actual={version_values!r}"
        )

    methods = {
        statement.name: statement
        for statement in model_class.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for method_name, expected_names in _EXPECTED_METHODS.items():
        method = methods.get(method_name)
        if method is None:
            raise HotfixError(f"Stage2DMD lacks {method_name}")
        positional_names = tuple(
            argument.arg for argument in (*method.args.posonlyargs, *method.args.args)
        )
        keyword_names = tuple(argument.arg for argument in method.args.kwonlyargs)
        if positional_names != ("self",) or keyword_names != expected_names:
            raise HotfixError(
                f"Stage2DMD {method_name} signature mismatch: "
                f"positional={positional_names}, keyword_only={keyword_names}"
            )
        if method.args.vararg is not None or method.args.kwarg is not None:
            raise HotfixError(f"Stage2DMD {method_name} must not use *args/**kwargs")
        defaults = method.args.kw_defaults
        if any(value is not None for value in defaults[:-1]) or not (
            isinstance(defaults[-1], ast.Constant) and defaults[-1].value is None
        ):
            raise HotfixError(f"Stage2DMD {method_name} defaults mismatch")


def transform_stage2_dmd_source(source: str) -> TransformResult:
    """Return a fully validated v2 source or reject before any disk write."""

    if not isinstance(source, str) or not source:
        raise HotfixError("model/stage2_dmd.py is empty or unreadable")
    transformed = source
    changed_units: list[str] = []
    for unit in _PATCH_UNITS:
        transformed, changed = _apply_exact_unit(transformed, unit)
        if changed:
            changed_units.append(unit.name)
    _validate_current_ast(transformed)
    return TransformResult(
        source=transformed,
        changed=bool(changed_units),
        changed_units=tuple(changed_units),
    )


def _transform_runtime_source(
    relative_path: str,
    source: str,
    patch_units: tuple[_PatchUnit, ...],
) -> TransformResult:
    if not isinstance(source, str) or not source:
        raise HotfixError(f"{relative_path} is empty or unreadable")
    transformed = source
    changed_units: list[str] = []
    for unit in patch_units:
        transformed, changed = _apply_exact_unit(transformed, unit)
        if changed:
            changed_units.append(f"{relative_path}:{unit.name}")
    try:
        ast.parse(transformed, filename=relative_path)
        compile(transformed, relative_path, "exec")
    except (SyntaxError, ValueError) as error:
        raise HotfixError(
            f"transformed {relative_path} does not compile: {error}"
        ) from error
    return TransformResult(
        source=transformed,
        changed=bool(changed_units),
        changed_units=tuple(changed_units),
    )


def _transform_parameter_names_source(source: str) -> TransformResult:
    if source == _PARAMETER_NAMES_SOURCE:
        changed = False
    elif source == "":
        changed = True
    else:
        raise HotfixError("unrecognized or partial utils/parameter_names.py")
    try:
        tree = ast.parse(_PARAMETER_NAMES_SOURCE, filename="utils/parameter_names.py")
        compile(tree, "utils/parameter_names.py", "exec")
    except (SyntaxError, ValueError) as error:
        raise HotfixError(
            f"bundled parameter-name resolver is invalid: {error}"
        ) from error
    return TransformResult(
        source=_PARAMETER_NAMES_SOURCE,
        changed=changed,
        changed_units=("utils/parameter_names.py:create",) if changed else (),
    )


def transform_stage2_runtime_sources(
    sources: Mapping[str, str],
) -> dict[str, TransformResult]:
    """Validate and transform every cumulative Stage-2 runtime target in memory."""

    required = {
        "model/stage2_dmd.py",
        "trainer/stage2_distillation.py",
        "utils/distributed.py",
        "utils/lora_utils.py",
        "utils/parameter_names.py",
        "utils/stage2_checkpoint.py",
        "utils/stage2_fsdp2.py",
        "utils/stage2_metrics.py",
        "scripts/plot_stage2_training.py",
    }
    missing = sorted(required - set(sources))
    unexpected = sorted(set(sources) - required)
    if missing or unexpected:
        raise HotfixError(
            f"runtime source set mismatch: missing={missing}, unexpected={unexpected}"
        )
    return {
        "model/stage2_dmd.py": transform_stage2_dmd_source(
            sources["model/stage2_dmd.py"]
        ),
        "trainer/stage2_distillation.py": _transform_runtime_source(
            "trainer/stage2_distillation.py",
            sources["trainer/stage2_distillation.py"],
            _TRAINER_PATCH_UNITS,
        ),
        "utils/distributed.py": _transform_runtime_source(
            "utils/distributed.py",
            sources["utils/distributed.py"],
            _DISTRIBUTED_PATCH_UNITS,
        ),
        "utils/lora_utils.py": _transform_runtime_source(
            "utils/lora_utils.py",
            sources["utils/lora_utils.py"],
            _LORA_PATCH_UNITS,
        ),
        "utils/parameter_names.py": _transform_parameter_names_source(
            sources["utils/parameter_names.py"]
        ),
        "utils/stage2_checkpoint.py": _transform_runtime_source(
            "utils/stage2_checkpoint.py",
            sources["utils/stage2_checkpoint.py"],
            _STAGE2_CHECKPOINT_PATCH_UNITS,
        ),
        "utils/stage2_fsdp2.py": _transform_runtime_source(
            "utils/stage2_fsdp2.py",
            sources["utils/stage2_fsdp2.py"],
            _STAGE2_FSDP2_PATCH_UNITS,
        ),
        "utils/stage2_metrics.py": _transform_runtime_source(
            "utils/stage2_metrics.py",
            sources["utils/stage2_metrics.py"],
            _METRICS_PATCH_UNITS,
        ),
        "scripts/plot_stage2_training.py": _transform_runtime_source(
            "scripts/plot_stage2_training.py",
            sources["scripts/plot_stage2_training.py"],
            _PLOT_PATCH_UNITS,
        ),
    }


def _write_backup(target: Path, original: bytes) -> Path:
    digest = hashlib.sha256(original).hexdigest()[:16]
    backup = target.with_name(f"{target.name}.pre_hotfix_{digest}.bak")
    try:
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if backup.read_bytes() != original:
            raise HotfixError(f"existing backup has unexpected contents: {backup}")
        return backup
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        backup.unlink(missing_ok=True)
        raise
    return backup


def _atomic_replace(target: Path, replacement: bytes, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.hotfix.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(replacement)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            directory_descriptor = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_runtime(project_root: Path, target: Path) -> str:
    probe = r"""
import inspect
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected_source = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(root))
from model.stage2_dmd import Stage2DMD
from trainer.stage2_distillation import _audit_stage2_dmd_runtime_api
from utils.distributed import TrainableShardedEMA
from utils.lora_utils import (
    STAGE2_LORA_LOAD_API_VERSION,
    strict_load_lora_state_dict,
)
from utils.parameter_names import (
    STAGE2_PARAMETER_NAME_API_VERSION,
    map_parameter_names_to_expected,
)
from utils.stage2_checkpoint import (
    _canonicalize_stage2_optimizer_state,
    _optimizer_state_for_runtime,
)

actual_source = Path(inspect.getsourcefile(Stage2DMD) or "<unknown>").resolve()
if actual_source != expected_source:
    raise RuntimeError(
        f"Stage2DMD loaded from unexpected source: {actual_source} != {expected_source}"
    )
audit = _audit_stage2_dmd_runtime_api(Stage2DMD)
if STAGE2_PARAMETER_NAME_API_VERSION != "longlive_stage2_parameter_names/v1":
    raise RuntimeError("Stage-2 parameter-name API version mismatch")
mapping = map_parameter_names_to_expected(
    ("model.base_model.model.block.lora_A.default.weight",),
    ("base_model.model.block.lora_A.default.weight",),
    label="Stage-2 hotfix probe",
)
if mapping != {
    "model.base_model.model.block.lora_A.default.weight":
        "base_model.model.block.lora_A.default.weight"
}:
    raise RuntimeError("Stage-2 parameter-name resolver probe failed")
if "expected_parameter_names" not in inspect.signature(TrainableShardedEMA).parameters:
    raise RuntimeError("TrainableShardedEMA lacks schema-aware names")
if not callable(_canonicalize_stage2_optimizer_state) or not callable(
    _optimizer_state_for_runtime
):
    raise RuntimeError("Stage-2 optimizer name transforms are unavailable")
if STAGE2_LORA_LOAD_API_VERSION != "longlive_stage2_lora_load/v1":
    raise RuntimeError("Stage-2 LoRA load API version mismatch")
if "peft.set_peft_model_state_dict" in inspect.getsource(
    strict_load_lora_state_dict
):
    raise RuntimeError("Stage-2 LoRA loader still depends on PEFT distributed TP")
print(
    "STAGE2_DMD_RUNTIME_API=PASS "
    f"version={audit['api_version']} source={audit['source_file']}"
)
print(
    "STAGE2_PARAMETER_NAMES_API=PASS "
    f"version={STAGE2_PARAMETER_NAME_API_VERSION}"
)
print(
    "STAGE2_LORA_LOAD_API=PASS "
    f"version={STAGE2_LORA_LOAD_API_VERSION}"
)
"""
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    with tempfile.TemporaryDirectory(prefix="stage2_hotfix_pycache_") as pycache:
        environment["PYTHONPYCACHEPREFIX"] = pycache
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-X",
                f"pycache_prefix={pycache}",
                "-c",
                probe,
                str(project_root),
                str(target),
            ],
            cwd=project_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise HotfixError(
            "isolated Stage-2 runtime verification failed"
            + (f":\n{details}" if details else "")
        )
    return completed.stdout.strip()


def apply_stage2_innernet_hotfix(
    project_root: str | Path,
    *,
    check_only: bool = False,
    verify_runtime: bool = True,
) -> HotfixResult:
    root = Path(project_root).expanduser().resolve()
    relative_paths = (
        "model/stage2_dmd.py",
        "trainer/stage2_distillation.py",
        "utils/distributed.py",
        "utils/lora_utils.py",
        "utils/parameter_names.py",
        "utils/stage2_checkpoint.py",
        "utils/stage2_fsdp2.py",
        "utils/stage2_metrics.py",
        "scripts/plot_stage2_training.py",
    )
    targets = {relative: root / relative for relative in relative_paths}
    originals: dict[str, bytes] = {}
    sources: dict[str, str] = {}
    modes: dict[str, int] = {}
    for relative, target in targets.items():
        if not target.is_file():
            if (
                relative != "utils/parameter_names.py"
                or target.exists()
                or target.is_symlink()
            ):
                raise HotfixError(f"Stage-2 runtime source does not exist: {target}")
            originals[relative] = b""
            sources[relative] = ""
            modes[relative] = 0o644
            continue
        original = target.read_bytes()
        try:
            source = original.decode("utf-8")
        except UnicodeDecodeError as error:
            raise HotfixError(
                f"Stage-2 runtime source is not UTF-8: {target}"
            ) from error
        originals[relative] = original
        sources[relative] = source
        modes[relative] = target.stat().st_mode & 0o7777

    transformed = transform_stage2_runtime_sources(sources)
    changed_files = tuple(
        relative for relative in relative_paths if transformed[relative].changed
    )
    changed_units = tuple(
        unit
        for relative in relative_paths
        for unit in transformed[relative].changed_units
    )
    primary_target = targets["model/stage2_dmd.py"]

    if changed_files and check_only:
        return HotfixResult(
            status="NEEDS_PATCH",
            target_path=primary_target,
            backup_path=None,
            changed_units=changed_units,
            target_paths=tuple(targets.values()),
            backup_paths=(),
            changed_files=changed_files,
        )

    backup_by_file: dict[str, Path] = {}
    status = "ALREADY_APPLIED"
    written: list[str] = []
    if changed_files:
        for relative in changed_files:
            if originals[relative]:
                backup_by_file[relative] = _write_backup(
                    targets[relative], originals[relative]
                )
        try:
            for relative in changed_files:
                _atomic_replace(
                    targets[relative],
                    transformed[relative].source.encode("utf-8"),
                    modes[relative],
                )
                written.append(relative)
            runtime_message = (
                _verify_runtime(root, primary_target) if verify_runtime else ""
            )
        except Exception as error:
            rollback_failures = []
            for relative in reversed(written):
                try:
                    if originals[relative]:
                        _atomic_replace(
                            targets[relative], originals[relative], modes[relative]
                        )
                    else:
                        targets[relative].unlink()
                except OSError as rollback_error:
                    rollback_failures.append(f"{relative}: {rollback_error}")
            details = (
                f"; rollback_failures={rollback_failures}"
                if rollback_failures
                else "; all written files rolled back"
            )
            raise HotfixError(
                f"cumulative Stage-2 hotfix failed: {error}{details}"
            ) from error
        status = "PATCHED"
    else:
        runtime_message = (
            _verify_runtime(root, primary_target) if verify_runtime else ""
        )
    if runtime_message:
        print(runtime_message)
    return HotfixResult(
        status=status,
        target_path=primary_target,
        backup_path=backup_by_file.get("model/stage2_dmd.py"),
        changed_units=changed_units,
        target_paths=tuple(targets.values()),
        backup_paths=tuple(backup_by_file.values()),
        changed_files=changed_files,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely apply cumulative no-Git Stage-2 runtime hotfixes."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="LongLive project root (default: parent of this script's scripts dir)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate only; exit 2 if the recognized legacy source needs patching",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = apply_stage2_innernet_hotfix(
            args.project_root,
            check_only=args.check,
            verify_runtime=True,
        )
    except (HotfixError, OSError) as error:
        print(f"STAGE2_INNERNET_HOTFIX=FAIL error={error}", file=sys.stderr)
        return 1

    units = ",".join(result.changed_units) or "none"
    backups = ",".join(str(path) for path in result.backup_paths) or "none"
    files = ",".join(result.changed_files) or "none"
    print(
        f"STAGE2_INNERNET_HOTFIX={result.status} "
        f"targets={len(result.target_paths)} changed_files={files} "
        f"backups={backups} units={units}"
    )
    return 2 if result.status == "NEEDS_PATCH" else 0


if __name__ == "__main__":
    raise SystemExit(main())
