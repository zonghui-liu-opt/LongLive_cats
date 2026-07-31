"""Atomic artifact/resume checkpoint schema for Stage-1 adapter training."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
from typing import Any, TypeVar

import numpy as np
import torch

from utils.stage1_io import (
    atomic_output_path,
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)


CHECKPOINT_SCHEMA = "longlive_stage1_lora_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
TRAINER_STATE_SCHEMA = "longlive_stage1_trainer_state"
TRAINER_STATE_SCHEMA_VERSION = 1
STAGE1_WORLD_SIZE = 6
STAGE1_SEQUENCE_PARALLEL_SIZE = 3
STAGE1_DATA_PARALLEL_SIZE = 2
STAGE1_EXPECTED_ADAPTER_TENSORS = 360
STAGE1_EXPECTED_TRAINABLE_NUMEL = 57_016_320
CHECKPOINT_DIRECTORY_PATTERN = re.compile(r"checkpoint_model_([0-9]{6})")
ARTIFACT_REQUIRED_FILES = {
    "adapter_raw.safetensors",
    "adapter_ema.safetensors",
    "resolved_config.yaml",
    "base_reference.json",
    "checkpoint_manifest.json",
    "_SUCCESS",
}


_T = TypeVar("_T")


@dataclass(frozen=True)
class Stage1CollectiveOps:
    """Injectable WORLD collective surface used by checkpoint workflows.

    Production callers should omit this argument and use the initialized
    process group.  Tests may inject deterministic callbacks, but must still
    report the locked six-rank topology; this prevents a single-rank code path
    from accidentally becoming the production implementation.
    """

    get_rank: Callable[[], int]
    get_world_size: Callable[[], int]
    barrier: Callable[[], None]
    consensus: Callable[[bool], bool]


@dataclass(frozen=True)
class Stage1AdapterPair:
    """Rank-0 canonical raw/EMA adapters; both are ``None`` elsewhere."""

    raw: OrderedDict[str, torch.Tensor] | None
    ema: OrderedDict[str, torch.Tensor] | None


@dataclass(frozen=True)
class Stage1ResumePayload:
    """Per-rank resume payload with canonical-per-SP error-buffer state."""

    trainer_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any] | None
    ema_state: Mapping[str, Any]
    rng_state: Mapping[str, Any]
    error_buffer_state: Any


def _default_collective_ops() -> Stage1CollectiveOps:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Stage-1 checkpoint collectives require an initialized process group"
        )

    def consensus(local_success: bool) -> bool:
        backend = str(dist.get_backend()).lower()
        if "nccl" in backend:
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL checkpoint consensus requires CUDA")
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
        flag = torch.tensor(
            1 if local_success else 0,
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    return Stage1CollectiveOps(
        get_rank=dist.get_rank,
        get_world_size=dist.get_world_size,
        barrier=dist.barrier,
        consensus=consensus,
    )


def _stage1_collectives(
    collectives: Stage1CollectiveOps | None,
) -> Stage1CollectiveOps:
    value = _default_collective_ops() if collectives is None else collectives
    world_size = int(value.get_world_size())
    rank = int(value.get_rank())
    if world_size != STAGE1_WORLD_SIZE:
        raise RuntimeError(
            f"Stage-1 checkpoint requires WORLD_SIZE=6, got {world_size}"
        )
    if rank < 0 or rank >= world_size:
        raise RuntimeError(f"invalid Stage-1 checkpoint rank {rank}/{world_size}")
    return value


def _consensus_call(
    label: str,
    callback: Callable[[], _T],
    collectives: Stage1CollectiveOps,
) -> _T:
    """Run local work and make its success/failure a WORLD decision."""

    result: _T | None = None
    local_error: Exception | None = None
    try:
        result = callback()
    except Exception as exc:
        local_error = exc
    if not collectives.consensus(local_error is None):
        raise RuntimeError(f"Stage-1 checkpoint stage failed: {label}") from local_error
    assert local_error is None
    return result  # type: ignore[return-value]


def _schema_contract(
    expected_schema: Mapping[str, Any],
    *,
    expected_adapter_tensors: int,
    expected_global_numel: int,
    expected_dtype: torch.dtype,
) -> OrderedDict[str, Any]:
    if not isinstance(expected_schema, Mapping):
        raise TypeError("pre-FSDP LoRA schema must be a mapping")
    schema = OrderedDict((key, expected_schema[key]) for key in sorted(expected_schema))
    if len(schema) != int(expected_adapter_tensors):
        raise ValueError(
            "pre-FSDP LoRA tensor count mismatch: "
            f"expected={expected_adapter_tensors}, actual={len(schema)}"
        )
    raw_names: set[str] = set()
    total = 0
    for key, spec in schema.items():
        if not isinstance(key, str) or not key:
            raise TypeError("pre-FSDP LoRA schema keys must be non-empty strings")
        if not (key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")):
            raise ValueError(f"non-canonical LoRA key in pre-FSDP schema: {key!r}")
        try:
            shape = tuple(int(value) for value in spec.global_shape)
            dtype = spec.dtype
            raw_name = str(spec.raw_parameter_name)
        except AttributeError as exc:
            raise TypeError(f"invalid LoRA tensor spec for {key!r}") from exc
        if not shape or any(value <= 0 for value in shape):
            raise ValueError(f"invalid global LoRA shape for {key}: {shape}")
        if dtype != expected_dtype:
            raise TypeError(
                f"pre-FSDP LoRA dtype mismatch for {key}: "
                f"expected={expected_dtype}, actual={dtype}"
            )
        if not raw_name or raw_name in raw_names:
            raise ValueError(f"duplicate/empty raw LoRA parameter name: {raw_name!r}")
        raw_names.add(raw_name)
        total += int(torch.Size(shape).numel())
    if total != int(expected_global_numel):
        raise ValueError(
            "pre-FSDP LoRA global numel mismatch: "
            f"expected={expected_global_numel}, actual={total}"
        )
    return schema


def validate_stage1_adapter_state(
    adapter_state: Mapping[str, torch.Tensor],
    *,
    expected_schema: Mapping[str, Any],
    expected_adapter_tensors: int = STAGE1_EXPECTED_ADAPTER_TENSORS,
    expected_global_numel: int = STAGE1_EXPECTED_TRAINABLE_NUMEL,
    expected_dtype: torch.dtype = torch.float32,
) -> OrderedDict[str, torch.Tensor]:
    """Validate a rank-0 canonical adapter against the immutable pre-FSDP schema."""

    schema = _schema_contract(
        expected_schema,
        expected_adapter_tensors=expected_adapter_tensors,
        expected_global_numel=expected_global_numel,
        expected_dtype=expected_dtype,
    )
    if not isinstance(adapter_state, Mapping):
        raise TypeError("canonical adapter state must be a mapping")
    if set(adapter_state) != set(schema):
        raise ValueError(
            "canonical adapter key mismatch: "
            f"missing={sorted(set(schema) - set(adapter_state))}, "
            f"extra={sorted(set(adapter_state) - set(schema))}"
        )
    canonical: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, spec in schema.items():
        tensor = adapter_state[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"canonical adapter value {key!r} is not a Tensor")
        if tuple(tensor.shape) != tuple(spec.global_shape):
            raise ValueError(
                f"canonical adapter shape mismatch for {key}: "
                f"expected={tuple(spec.global_shape)}, actual={tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"canonical adapter dtype mismatch for {key}: "
                f"expected={expected_dtype}, actual={tensor.dtype}"
            )
        cpu_tensor = tensor.detach().to(device="cpu").contiguous()
        if cpu_tensor.numel() and not bool(torch.isfinite(cpu_tensor).all().item()):
            raise ValueError(f"canonical adapter tensor {key!r} contains non-finite values")
        canonical[key] = cpu_tensor
    return canonical


def gather_stage1_raw_and_ema_adapters(
    module: torch.nn.Module,
    ema: Any,
    *,
    expected_schema: Mapping[str, Any],
    authoritative_shard_group: Any,
    replica_group: Any,
    expected_adapter_tensors: int = STAGE1_EXPECTED_ADAPTER_TENSORS,
    expected_global_numel: int = STAGE1_EXPECTED_TRAINABLE_NUMEL,
    collectives: Stage1CollectiveOps | None = None,
    get_local_shards_fn: Callable[..., Mapping[str, Any]] | None = None,
    gather_fn: Callable[..., OrderedDict[str, torch.Tensor] | None] | None = None,
) -> Stage1AdapterPair:
    """Collect raw then EMA adapters in one all-rank, restore-safe workflow.

    This function deliberately has no model-state-dict callback.  The only
    extraction callback accepted is the selective local-LoRA API.  EMA is
    installed through ``swap_into`` and its ``finally`` path is WORLD-audited
    before this function returns or re-raises.
    """

    if get_local_shards_fn is None or gather_fn is None:
        from utils.lora_utils import (
            gather_fsdp2_lora_state_dict,
            get_lora_sharded_state_dict,
        )

        get_local_shards_fn = get_local_shards_fn or get_lora_sharded_state_dict
        gather_fn = gather_fn or gather_fsdp2_lora_state_dict
    ops = _stage1_collectives(collectives)
    rank = int(ops.get_rank())
    schema = _consensus_call(
        "pre-FSDP LoRA schema contract",
        lambda: _schema_contract(
            expected_schema,
            expected_adapter_tensors=expected_adapter_tensors,
            expected_global_numel=expected_global_numel,
            expected_dtype=torch.float32,
        ),
        ops,
    )

    def local_shards() -> Mapping[str, Any]:
        return get_local_shards_fn(
            module,
            cpu_offload=True,
            expected_schema=schema,
            expected_adapter_tensors=expected_adapter_tensors,
            expected_dtype=torch.float32,
        )

    def directed_gather(shards: Mapping[str, Any]):
        return gather_fn(
            shards,
            authoritative_shard_group=authoritative_shard_group,
            replica_group=replica_group,
            dst_global_rank=0,
            expected_schema=schema,
        )

    ops.barrier()
    raw_local = _consensus_call("raw selective local copy", local_shards, ops)
    raw = _consensus_call(
        "raw selective directed gather", lambda: directed_gather(raw_local), ops
    )
    if rank == 0:
        raw = _consensus_call(
            "raw canonical schema validation",
            lambda: validate_stage1_adapter_state(
                raw,
                expected_schema=schema,
                expected_adapter_tensors=expected_adapter_tensors,
                expected_global_numel=expected_global_numel,
            ),
            ops,
        )
    else:
        # Rank 0 is the only materialization destination, but every rank enters
        # exactly one consensus for this validation stage.
        if not ops.consensus(True):
            raise RuntimeError("Stage-1 raw canonical validation failed on rank 0")
        raw = None
    ops.barrier()

    swap_context = ema.swap_into(module)
    entered = False
    enter_error: Exception | None = None
    try:
        swap_context.__enter__()
        entered = True
    except Exception as exc:
        enter_error = exc
    if not ops.consensus(enter_error is None):
        restore_error: Exception | None = None
        if entered:
            try:
                swap_context.__exit__(None, None, None)
            except Exception as exc:
                restore_error = exc
        if not ops.consensus(restore_error is None):
            raise RuntimeError("EMA swap entry failed and raw restore also failed") from (
                restore_error or enter_error
            )
        raise RuntimeError("EMA local swap failed on at least one rank") from enter_error

    ema_result: OrderedDict[str, torch.Tensor] | None = None
    body_error: Exception | None = None
    try:
        ops.barrier()
        ema_local = _consensus_call("EMA selective local copy", local_shards, ops)
        ema_result = _consensus_call(
            "EMA selective directed gather", lambda: directed_gather(ema_local), ops
        )
        if rank == 0:
            ema_result = _consensus_call(
                "EMA canonical schema validation",
                lambda: validate_stage1_adapter_state(
                    ema_result,
                    expected_schema=schema,
                    expected_adapter_tensors=expected_adapter_tensors,
                    expected_global_numel=expected_global_numel,
                ),
                ops,
            )
        else:
            if not ops.consensus(True):
                raise RuntimeError("Stage-1 EMA canonical validation failed on rank 0")
            ema_result = None
    except Exception as exc:
        body_error = exc
    finally:
        restore_error: Exception | None = None
        try:
            swap_context.__exit__(
                type(body_error) if body_error is not None else None,
                body_error,
                body_error.__traceback__ if body_error is not None else None,
            )
        except Exception as exc:
            restore_error = exc
        if not ops.consensus(restore_error is None):
            raise RuntimeError(
                "EMA adapter gather did not restore every raw local shard"
            ) from restore_error
        ops.barrier()
    if body_error is not None:
        raise body_error
    return Stage1AdapterPair(raw=raw if rank == 0 else None, ema=ema_result)


def atomic_save_stage1_adapter(
    path: str | os.PathLike[str],
    adapter_state: Mapping[str, torch.Tensor],
    *,
    expected_schema: Mapping[str, Any],
    kind: str,
    completed_step: int,
    expected_adapter_tensors: int = STAGE1_EXPECTED_ADAPTER_TENSORS,
    expected_global_numel: int = STAGE1_EXPECTED_TRAINABLE_NUMEL,
) -> OrderedDict[str, torch.Tensor]:
    """Validate and atomically write one canonical FP32 safetensors adapter."""

    from safetensors.torch import save_file

    if kind not in {"raw", "ema"}:
        raise ValueError(f"adapter kind must be 'raw' or 'ema', got {kind!r}")
    step = int(completed_step)
    if isinstance(completed_step, bool) or step != completed_step or step <= 0:
        raise ValueError("completed_step must be a positive integer")
    canonical = validate_stage1_adapter_state(
        adapter_state,
        expected_schema=expected_schema,
        expected_adapter_tensors=expected_adapter_tensors,
        expected_global_numel=expected_global_numel,
    )
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": str(CHECKPOINT_SCHEMA_VERSION),
        "kind": kind,
        "completed_step": str(step),
        "tensor_count": str(len(canonical)),
        "global_numel": str(expected_global_numel),
        "dtype": "float32",
    }
    with atomic_output_path(path) as temporary:
        save_file(dict(canonical), str(temporary), metadata=metadata)
    return canonical


def write_stage1_adapter_pair(
    directory: str | os.PathLike[str],
    adapters: Stage1AdapterPair,
    *,
    expected_schema: Mapping[str, Any],
    completed_step: int,
    expected_adapter_tensors: int = STAGE1_EXPECTED_ADAPTER_TENSORS,
    expected_global_numel: int = STAGE1_EXPECTED_TRAINABLE_NUMEL,
    collectives: Stage1CollectiveOps | None = None,
) -> tuple[Path, Path] | None:
    """Rank 0 atomically writes raw/EMA files after a WORLD consensus."""

    ops = _stage1_collectives(collectives)
    rank = int(ops.get_rank())
    directory = Path(directory)
    ops.barrier()

    def write_on_rank_zero() -> tuple[Path, Path] | None:
        if rank != 0:
            if adapters.raw is not None or adapters.ema is not None:
                raise RuntimeError("nonzero rank unexpectedly materialized canonical adapters")
            return None
        if adapters.raw is None or adapters.ema is None:
            raise RuntimeError("rank 0 did not receive both canonical adapters")
        directory.mkdir(parents=True, exist_ok=True)
        raw_path = directory / "adapter_raw.safetensors"
        ema_path = directory / "adapter_ema.safetensors"
        atomic_save_stage1_adapter(
            raw_path,
            adapters.raw,
            expected_schema=expected_schema,
            kind="raw",
            completed_step=completed_step,
            expected_adapter_tensors=expected_adapter_tensors,
            expected_global_numel=expected_global_numel,
        )
        atomic_save_stage1_adapter(
            ema_path,
            adapters.ema,
            expected_schema=expected_schema,
            kind="ema",
            completed_step=completed_step,
            expected_adapter_tensors=expected_adapter_tensors,
            expected_global_numel=expected_global_numel,
        )
        return raw_path, ema_path

    result = _consensus_call("atomic raw/EMA adapter writes", write_on_rank_zero, ops)
    ops.barrier()
    return result if rank == 0 else None


def _match_trainable_to_schema(
    name: str,
    parameter: torch.Tensor,
    expected_schema: Mapping[str, Any] | None,
) -> str | None:
    if expected_schema is None:
        return None
    matches = [
        key
        for key, spec in expected_schema.items()
        if name == str(spec.raw_parameter_name)
        or name.endswith(f".{spec.raw_parameter_name}")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"optimizer LoRA parameter {name!r} did not map uniquely to schema: {matches}"
        )
    spec = expected_schema[matches[0]]
    if tuple(parameter.shape) != tuple(spec.global_shape):
        raise ValueError(
            f"optimizer parameter global shape mismatch for {name}: "
            f"schema={tuple(spec.global_shape)}, actual={tuple(parameter.shape)}"
        )
    return matches[0]


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor
    except (ImportError, AttributeError):
        DTensor = ()  # type: ignore[assignment]
    if isinstance(value, DTensor):
        return value.detach().to_local()
    return value.detach()


def _audit_moment_mapping(
    state: Mapping[Any, Any],
    *,
    expected_parameters: set[Any] | set[str],
    require_initialized_moments: bool,
    require_cpu: bool,
    label: str,
    expected_completed_step: int | None = None,
) -> None:
    state_keys = set(state)
    if not state_keys.issubset(expected_parameters):
        raise ValueError(f"{label} has state for non-LoRA parameters")
    if require_initialized_moments and state_keys != expected_parameters:
        raise ValueError(
            f"{label} AdamW state is incomplete: "
            f"missing={len(expected_parameters - state_keys)}"
        )
    moment_names = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
    for parameter_name, values in state.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"{label} state for {parameter_name!r} must be a mapping")
        if require_initialized_moments:
            missing = [name for name in ("exp_avg", "exp_avg_sq") if name not in values]
            if missing:
                raise ValueError(
                    f"{label} state for {parameter_name!r} is missing moments {missing}"
                )
            if "step" not in values:
                raise ValueError(
                    f"{label} state for {parameter_name!r} is missing AdamW step"
                )
        if "step" in values:
            step_value = values["step"]
            if isinstance(step_value, torch.Tensor):
                local_step = _local_tensor(step_value)
                if local_step.numel() != 1 or not bool(
                    torch.isfinite(local_step).all().item()
                ):
                    raise ValueError(
                        f"{label} AdamW step for {parameter_name!r} must be one finite scalar"
                    )
                step_value = local_step.item()
            if isinstance(step_value, bool) or not isinstance(step_value, (int, float)):
                raise TypeError(
                    f"{label} AdamW step for {parameter_name!r} is not numeric"
                )
            step = int(step_value)
            if float(step_value) != float(step) or step < 0:
                raise ValueError(
                    f"{label} AdamW step for {parameter_name!r} is invalid: {step_value}"
                )
            if expected_completed_step is not None and step != int(
                expected_completed_step
            ):
                raise ValueError(
                    f"{label} AdamW step mismatch for {parameter_name!r}: "
                    f"checkpoint={step}, expected={int(expected_completed_step)}"
                )
        for name in moment_names:
            if name not in values:
                continue
            moment = values[name]
            if not isinstance(moment, torch.Tensor):
                raise TypeError(f"{label} {name} for {parameter_name!r} is not a Tensor")
            local = _local_tensor(moment)
            if local.dtype != torch.float32:
                raise TypeError(
                    f"{label} {name} for {parameter_name!r} must be FP32, "
                    f"got {local.dtype}"
                )
            if require_cpu and local.device.type != "cpu":
                raise TypeError(
                    f"{label} {name} for {parameter_name!r} must be CPU, "
                    f"got {local.device}"
                )
            if local.numel() and not bool(torch.isfinite(local).all().item()):
                raise ValueError(
                    f"{label} {name} for {parameter_name!r} contains non-finite values"
                )


def audit_stage1_lora_optimizer(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_schema: Mapping[str, Any] | None = None,
    require_initialized_moments: bool = False,
    expected_completed_step: int | None = None,
) -> tuple[str, ...]:
    """Assert optimizer groups are exactly the FP32 trainable LoRA parameters."""

    named = OrderedDict(
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )
    if not named:
        raise ValueError("Stage-1 optimizer model has no trainable parameters")
    invalid = [
        (name, parameter.dtype)
        for name, parameter in named.items()
        if parameter.dtype != torch.float32
        or not (".lora_A." in name or ".lora_B." in name)
    ]
    if invalid:
        raise TypeError(f"Stage-1 optimizer accepts only FP32 LoRA parameters: {invalid}")
    if expected_schema is not None:
        schema = OrderedDict(
            (key, expected_schema[key]) for key in sorted(expected_schema)
        )
        mapped = {
            _match_trainable_to_schema(name, parameter, schema)
            for name, parameter in named.items()
        }
        if mapped != set(schema):
            raise ValueError(
                "optimizer trainables differ from pre-FSDP schema: "
                f"missing={sorted(set(schema) - mapped)}"
            )

    expected_ids = {id(parameter) for parameter in named.values()}
    group_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    ]
    group_ids = [id(parameter) for parameter in group_parameters]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("Stage-1 optimizer parameter groups contain duplicate identities")
    if set(group_ids) != expected_ids:
        raise ValueError(
            "Stage-1 optimizer parameter identities do not exactly match trainable LoRA"
        )
    parameter_by_id = {id(parameter): parameter for parameter in named.values()}
    optimizer_state_by_parameter = {
        parameter_by_id[id(parameter)]: values
        for parameter, values in optimizer.state.items()
        if id(parameter) in parameter_by_id
    }
    if len(optimizer_state_by_parameter) != len(optimizer.state):
        raise ValueError("Stage-1 optimizer state contains a non-LoRA parameter")
    _audit_moment_mapping(
        optimizer_state_by_parameter,
        expected_parameters=set(named.values()),
        require_initialized_moments=require_initialized_moments,
        require_cpu=False,
        label="local optimizer",
        expected_completed_step=expected_completed_step,
    )
    return tuple(named)


def validate_stage1_full_optimizer_state(
    optimizer_state: Mapping[str, Any],
    *,
    expected_parameter_names: Sequence[str],
    require_initialized_moments: bool = True,
    expected_completed_step: int | None = None,
) -> Mapping[str, Any]:
    """Validate the DCP full optimizer-only CPU representation on rank 0."""

    if not isinstance(optimizer_state, Mapping):
        raise TypeError("DCP optimizer state must be a mapping")
    if set(optimizer_state) != {"state", "param_groups"}:
        raise ValueError(
            f"DCP optimizer state keys must be state/param_groups, got {sorted(optimizer_state)}"
        )
    state = optimizer_state["state"]
    groups = optimizer_state["param_groups"]
    if not isinstance(state, Mapping) or not isinstance(groups, Sequence):
        raise TypeError("DCP optimizer state has invalid state/param_groups containers")
    expected = set(str(name) for name in expected_parameter_names)
    group_names: list[str] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or "params" not in group:
            raise TypeError(f"DCP optimizer param group {index} is invalid")
        params = group["params"]
        if not isinstance(params, Sequence) or isinstance(params, (str, bytes)):
            raise TypeError(f"DCP optimizer group {index} params must be a sequence")
        if not all(isinstance(name, str) for name in params):
            raise TypeError(f"DCP optimizer group {index} contains non-FQN params")
        group_names.extend(params)
    if len(group_names) != len(set(group_names)) or set(group_names) != expected:
        raise ValueError(
            "DCP optimizer param groups differ from LoRA FQNs: "
            f"missing={sorted(expected - set(group_names))}, "
            f"extra={sorted(set(group_names) - expected)}"
        )
    _audit_moment_mapping(
        state,
        expected_parameters=expected,
        require_initialized_moments=require_initialized_moments,
        require_cpu=True,
        label="full optimizer",
        expected_completed_step=expected_completed_step,
    )
    return optimizer_state


def _dcp_optimizer_apis():
    try:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_optimizer_state_dict,
            set_optimizer_state_dict,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Stage-1 FSDP2 optimizer checkpointing requires the PyTorch 2.8 "
            "torch.distributed.checkpoint state_dict APIs"
        ) from exc
    return get_optimizer_state_dict, set_optimizer_state_dict, StateDictOptions


def gather_stage1_optimizer_state(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_schema: Mapping[str, Any] | None = None,
    expected_completed_step: int | None = None,
    collectives: Stage1CollectiveOps | None = None,
    get_optimizer_state_dict_fn: Callable[..., Mapping[str, Any]] | None = None,
    options_factory: Callable[..., Any] | None = None,
) -> Mapping[str, Any] | None:
    """All-rank DCP gather of full *optimizer-only* state to CPU rank 0."""

    ops = _stage1_collectives(collectives)
    rank = int(ops.get_rank())
    default_get, _, default_options = _dcp_optimizer_apis()
    get_fn = get_optimizer_state_dict_fn or default_get
    make_options = options_factory or default_options
    expected_names = _consensus_call(
        "LoRA optimizer identity audit before save",
        lambda: audit_stage1_lora_optimizer(
            module,
            optimizer,
            expected_schema=expected_schema,
            require_initialized_moments=True,
            expected_completed_step=expected_completed_step,
        ),
        ops,
    )
    ops.barrier()
    options = make_options(full_state_dict=True, cpu_offload=True)
    full_state = _consensus_call(
        "DCP optimizer full-state gather",
        lambda: get_fn(module, optimizer, options=options),
        ops,
    )
    if rank == 0:
        full_state = _consensus_call(
            "rank-0 optimizer-only state validation",
            lambda: validate_stage1_full_optimizer_state(
                full_state,
                expected_parameter_names=expected_names,
                require_initialized_moments=True,
                expected_completed_step=expected_completed_step,
            ),
            ops,
        )
    else:
        if not ops.consensus(True):
            raise RuntimeError("rank-0 optimizer-only validation failed")
        full_state = None
    ops.barrier()
    return full_state if rank == 0 else None


def restore_stage1_optimizer_state(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any] | None,
    *,
    expected_schema: Mapping[str, Any] | None = None,
    expected_completed_step: int | None = None,
    collectives: Stage1CollectiveOps | None = None,
    set_optimizer_state_dict_fn: Callable[..., None] | None = None,
    options_factory: Callable[..., Any] | None = None,
) -> None:
    """Broadcast and install rank-0 DCP optimizer state on every FSDP2 rank."""

    ops = _stage1_collectives(collectives)
    rank = int(ops.get_rank())
    _, default_set, default_options = _dcp_optimizer_apis()
    set_fn = set_optimizer_state_dict_fn or default_set
    make_options = options_factory or default_options
    expected_names = _consensus_call(
        "LoRA optimizer identity audit before restore",
        lambda: audit_stage1_lora_optimizer(
            module,
            optimizer,
            expected_schema=expected_schema,
            require_initialized_moments=False,
        ),
        ops,
    )
    local_validation_error: Exception | None = None
    if rank == 0:
        try:
            if optimizer_state is None:
                raise RuntimeError("rank 0 did not load optimizer state")
            validate_stage1_full_optimizer_state(
                optimizer_state,
                expected_parameter_names=expected_names,
                require_initialized_moments=True,
                expected_completed_step=expected_completed_step,
            )
        except Exception as exc:
            local_validation_error = exc
    elif optimizer_state not in (None, {}):
        local_validation_error = RuntimeError(
            "only rank 0 may supply a full optimizer state for restore"
        )
    if not ops.consensus(local_validation_error is None):
        raise RuntimeError("optimizer state validation failed before broadcast") from (
            local_validation_error
        )
    ops.barrier()
    options = make_options(
        full_state_dict=True,
        cpu_offload=True,
        broadcast_from_rank0=True,
    )
    rank_state = optimizer_state if rank == 0 else {}
    _consensus_call(
        "DCP optimizer state broadcast/restore",
        lambda: set_fn(module, optimizer, rank_state, options=options),
        ops,
    )
    _consensus_call(
        "LoRA optimizer FP32 moment audit after restore",
        lambda: audit_stage1_lora_optimizer(
            module,
            optimizer,
            expected_schema=expected_schema,
            require_initialized_moments=True,
            expected_completed_step=expected_completed_step,
        ),
        ops,
    )
    ops.barrier()


def build_stage1_trainer_state(
    *,
    optimizer_state: Mapping[str, Any],
    completed_step: int,
    global_epoch: int,
    committed_microbatch_cursor_in_epoch: int,
    phase_derivation: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
    dataloader_generator_state: torch.Tensor,
    next_attempt_index: int,
    nonfinite_attempt_count: int,
    resolved_config_sha256: str,
) -> dict[str, Any]:
    """Build the exact rank-0 trainer payload; schedules remain stateless."""

    value = {
        "schema": TRAINER_STATE_SCHEMA,
        "schema_version": TRAINER_STATE_SCHEMA_VERSION,
        "optimizer_state": optimizer_state,
        "completed_step": int(completed_step),
        "next_update_index": int(completed_step),
        "global_epoch": int(global_epoch),
        "committed_microbatch_cursor_in_epoch": int(
            committed_microbatch_cursor_in_epoch
        ),
        "phase_derivation": dict(phase_derivation),
        "sampler_state": dict(sampler_state),
        "dataloader_generator_state": dataloader_generator_state,
        "next_attempt_index": int(next_attempt_index),
        "nonfinite_attempt_count": int(nonfinite_attempt_count),
        "resolved_config_sha256": str(resolved_config_sha256),
    }
    validate_stage1_trainer_state(value)
    return value


def validate_stage1_trainer_state(
    state: Mapping[str, Any],
    *,
    expected_completed_step: int | None = None,
    expected_resolved_config_sha256: str | None = None,
) -> Mapping[str, Any]:
    required = {
        "schema",
        "schema_version",
        "optimizer_state",
        "completed_step",
        "next_update_index",
        "global_epoch",
        "committed_microbatch_cursor_in_epoch",
        "phase_derivation",
        "sampler_state",
        "dataloader_generator_state",
        "next_attempt_index",
        "nonfinite_attempt_count",
        "resolved_config_sha256",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        actual = set(state) if isinstance(state, Mapping) else set()
        raise ValueError(
            "trainer state key mismatch: "
            f"missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )
    if state["schema"] != TRAINER_STATE_SCHEMA or int(state["schema_version"]) != (
        TRAINER_STATE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported Stage-1 trainer state schema")
    completed_step = int(state["completed_step"])
    if completed_step <= 0 or int(state["next_update_index"]) != completed_step:
        raise ValueError("trainer completed_step/next_update_index is inconsistent")
    if expected_completed_step is not None and completed_step != int(
        expected_completed_step
    ):
        raise ValueError(
            f"trainer completed step mismatch: expected={expected_completed_step}, "
            f"actual={completed_step}"
        )
    for key in (
        "global_epoch",
        "committed_microbatch_cursor_in_epoch",
        "next_attempt_index",
        "nonfinite_attempt_count",
    ):
        value = int(state[key])
        if value < 0:
            raise ValueError(f"trainer state {key} must be non-negative")
    if not isinstance(state["phase_derivation"], Mapping) or not isinstance(
        state["sampler_state"], Mapping
    ):
        raise TypeError("trainer phase_derivation/sampler_state must be mappings")
    generator_state = state["dataloader_generator_state"]
    if not isinstance(generator_state, torch.Tensor) or generator_state.device.type != "cpu":
        raise TypeError("dataloader generator state must be a CPU Tensor")
    config_hash = str(state["resolved_config_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", config_hash) is None:
        raise ValueError("resolved_config_sha256 must be a lowercase SHA-256 digest")
    if (
        expected_resolved_config_sha256 is not None
        and config_hash != expected_resolved_config_sha256
    ):
        raise ValueError(
            "resolved config hash mismatch: "
            f"expected={expected_resolved_config_sha256}, actual={config_hash}"
        )
    if not isinstance(state["optimizer_state"], Mapping):
        raise TypeError("trainer optimizer_state must be a mapping")
    return state


def _trainer_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return small WORLD metadata without duplicating full optimizer state."""

    validate_stage1_trainer_state(state)
    return {key: value for key, value in state.items() if key != "optimizer_state"}


def _validate_trainer_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_completed_step: int,
    expected_resolved_config_sha256: str | None,
) -> None:
    if not isinstance(metadata, Mapping) or "optimizer_state" in metadata:
        raise ValueError("broadcast trainer metadata must not contain optimizer state")
    probe = dict(metadata)
    probe["optimizer_state"] = {}
    validate_stage1_trainer_state(
        probe,
        expected_completed_step=expected_completed_step,
        expected_resolved_config_sha256=expected_resolved_config_sha256,
    )


def write_stage1_resume_payloads(
    directory: str | os.PathLike[str],
    *,
    trainer_state: Mapping[str, Any] | None,
    ema_state: Mapping[str, Any],
    rng_state: Mapping[str, Any],
    error_buffer_state: Any,
    collectives: Stage1CollectiveOps | None = None,
    torch_save_fn: Callable[[str | os.PathLike[str], Any], None] = atomic_torch_save,
) -> tuple[str, ...]:
    """Write rank-owned heavy state; ranks 0/1/2 own canonical SP buffers."""

    ops = _stage1_collectives(collectives)
    rank = int(ops.get_rank())
    directory = Path(directory)
    ops.barrier()

    def write_local() -> tuple[str, ...]:
        directory.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        if rank == 0:
            if trainer_state is None:
                raise ValueError("rank 0 requires trainer_state")
            validate_stage1_trainer_state(trainer_state)
            torch_save_fn(directory / "trainer_state.pt", dict(trainer_state))
            written.append("trainer_state.pt")
        elif trainer_state is not None:
            raise ValueError("nonzero ranks must not carry the full trainer_state")
        if not isinstance(ema_state, Mapping):
            raise TypeError("local EMA state must be a mapping")
        if not isinstance(rng_state, Mapping):
            raise TypeError("local RNG state must be a mapping")
        ema_name = f"ema_local_rank{rank:05d}.pt"
        rng_name = f"rng_state_rank{rank:05d}.pt"
        torch_save_fn(directory / ema_name, dict(ema_state))
        torch_save_fn(directory / rng_name, dict(rng_state))
        written.extend((ema_name, rng_name))
        if rank < STAGE1_SEQUENCE_PARALLEL_SIZE:
            if error_buffer_state is None:
                raise ValueError(f"authoritative rank {rank} requires error-buffer state")
            buffer_name = f"error_buffer_sp{rank}.pt"
            torch_save_fn(directory / buffer_name, error_buffer_state)
            written.append(buffer_name)
        return tuple(written)

    written = _consensus_call("rank-local heavy resume writes", write_local, ops)
    ops.barrier()
    return written


def _torch_load_cpu(path: str | os.PathLike[str]) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_stage1_resume_payloads(
    directory: str | os.PathLike[str],
    *,
    replica_group: Any,
    expected_base_sha256: str | None = None,
    expected_resolved_config_sha256: str | None = None,
    collectives: Stage1CollectiveOps | None = None,
    torch_load_fn: Callable[[str | os.PathLike[str]], Any] = _torch_load_cpu,
    broadcast_object_list_fn: Callable[..., None] | None = None,
    get_process_group_ranks_fn: Callable[[Any], Sequence[int]] | None = None,
) -> Stage1ResumePayload:
    """Load resume state with rank-0 hash scan and two scoped broadcasts.

    Rank 0 broadcasts only epoch/cursor/attempt/schedule metadata to WORLD;
    the full optimizer state remains rank-0-only for the later DCP broadcast.
    Canonical error buffers use the global authoritative source ranks 0/1/2
    within replica groups ``(0,3)``, ``(1,4)``, and ``(2,5)``.
    """

    import torch.distributed as dist

    ops = _stage1_collectives(collectives)
    rank = int(ops.get_rank())
    directory = Path(directory)
    sp_position = rank % STAGE1_SEQUENCE_PARALLEL_SIZE
    broadcast_fn = broadcast_object_list_fn or dist.broadcast_object_list
    group_ranks_fn = get_process_group_ranks_fn or dist.get_process_group_ranks
    ops.barrier()

    def validate_and_load_rank_zero():
        if rank == 0:
            validate_checkpoint(
                directory,
                require_resumable=True,
                expected_base_sha256=expected_base_sha256,
                expected_topology=(
                    STAGE1_WORLD_SIZE,
                    STAGE1_SEQUENCE_PARALLEL_SIZE,
                    STAGE1_DATA_PARALLEL_SIZE,
                ),
            )
            trainer = torch_load_fn(directory / "trainer_state.pt")
            validate_stage1_trainer_state(
                trainer,
                expected_completed_step=checkpoint_step(directory),
                expected_resolved_config_sha256=expected_resolved_config_sha256,
            )
            return trainer
        return None

    full_trainer = _consensus_call(
        "rank-0 checkpoint hash validation and trainer load",
        validate_and_load_rank_zero,
        ops,
    )
    metadata_holder = [
        _trainer_metadata(full_trainer) if rank == 0 else None
    ]
    _consensus_call(
        "WORLD trainer metadata broadcast",
        lambda: broadcast_fn(metadata_holder, src=0),
        ops,
    )
    metadata = metadata_holder[0]
    _consensus_call(
        "WORLD trainer metadata validation",
        lambda: _validate_trainer_metadata(
            metadata,
            expected_completed_step=checkpoint_step(directory),
            expected_resolved_config_sha256=expected_resolved_config_sha256,
        ),
        ops,
    )

    def load_rank_owned():
        ema_value = torch_load_fn(directory / f"ema_local_rank{rank:05d}.pt")
        rng_value = torch_load_fn(directory / f"rng_state_rank{rank:05d}.pt")
        if not isinstance(ema_value, Mapping) or not isinstance(rng_value, Mapping):
            raise TypeError("loaded EMA/RNG resume payload must be mappings")
        buffer_value = (
            torch_load_fn(directory / f"error_buffer_sp{sp_position}.pt")
            if rank < STAGE1_SEQUENCE_PARALLEL_SIZE
            else None
        )
        expected_replica_ranks = (sp_position, sp_position + STAGE1_SEQUENCE_PARALLEL_SIZE)
        actual_replica_ranks = tuple(int(value) for value in group_ranks_fn(replica_group))
        if actual_replica_ranks != expected_replica_ranks:
            raise RuntimeError(
                "canonical error-buffer replica group mismatch: "
                f"expected={expected_replica_ranks}, actual={actual_replica_ranks}"
            )
        return ema_value, rng_value, buffer_value

    ema_value, rng_value, buffer_value = _consensus_call(
        "rank-owned EMA/RNG/error-buffer resume load",
        load_rank_owned,
        ops,
    )
    holder = [buffer_value]

    def broadcast_buffer() -> None:
        broadcast_fn(holder, src=sp_position, group=replica_group)

    _consensus_call(
        "canonical-per-SP error-buffer broadcast",
        broadcast_buffer,
        ops,
    )
    def validate_received_buffer() -> None:
        if holder[0] is None:
            raise RuntimeError(
                f"SP position {sp_position} received no error-buffer state"
            )

    _consensus_call(
        "canonical-per-SP error-buffer receive validation",
        validate_received_buffer,
        ops,
    )
    ops.barrier()
    return Stage1ResumePayload(
        trainer_state=metadata,
        optimizer_state=(full_trainer["optimizer_state"] if rank == 0 else None),
        ema_state=ema_value,
        rng_state=rng_value,
        error_buffer_state=holder[0],
    )


def heavy_required_files(world_size: int = 6, sp_size: int = 3) -> set[str]:
    if world_size <= 0 or sp_size <= 0:
        raise ValueError("world_size and sp_size must be positive.")
    return {
        "trainer_state.pt",
        *{f"ema_local_rank{rank:05d}.pt" for rank in range(world_size)},
        *{f"rng_state_rank{rank:05d}.pt" for rank in range(world_size)},
        *{f"error_buffer_sp{rank}.pt" for rank in range(sp_size)},
        "_RESUMABLE_SUCCESS",
    }


def checkpoint_directory(root: str | os.PathLike[str], completed_step: int) -> Path:
    step = int(completed_step)
    if isinstance(completed_step, bool) or step != completed_step or step <= 0:
        raise ValueError("completed_step must be a positive integer.")
    return Path(root) / f"checkpoint_model_{step:06d}"


def checkpoint_step(path: str | os.PathLike[str]) -> int:
    match = CHECKPOINT_DIRECTORY_PATTERN.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Not a Stage-1 checkpoint directory name: {path}")
    return int(match.group(1))


def capture_rng_state(*, include_cuda: bool = True) -> dict[str, Any]:
    cuda_available = bool(include_cuda and torch.cuda.is_available())
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_device_index = torch.cuda.current_device() if cuda_available else None
    state: dict[str, Any] = {
        "schema_version": 2,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "cuda_device_count": cuda_device_count,
        "cuda_device_index": cuda_device_index,
        "torch_cuda": None,
    }
    if cuda_available:
        # Each torchrun process owns only LOCAL_RANK. Reading all visible CUDA
        # generators would unnecessarily initialize/touch the other five GPUs.
        state["torch_cuda"] = torch.cuda.get_rng_state(cuda_device_index).cpu()
    return state


def restore_rng_state(state: Mapping[str, Any], *, require_cuda_topology: bool = True) -> None:
    if int(state.get("schema_version", -1)) != 2:
        raise RuntimeError(f"Unsupported RNG state schema: {state.get('schema_version')}")
    required = {
        "python",
        "numpy",
        "torch_cpu",
        "cuda_device_count",
        "cuda_device_index",
        "torch_cuda",
    }
    if set(state) != required | {"schema_version"}:
        raise RuntimeError(
            f"RNG state keys mismatch: expected {sorted(required | {'schema_version'})}, "
            f"got {sorted(state)}."
        )
    saved_cuda_count = int(state["cuda_device_count"])
    saved_cuda_index = state["cuda_device_index"]
    current_cuda_count = torch.cuda.device_count()
    if require_cuda_topology and saved_cuda_count != current_cuda_count:
        raise RuntimeError(
            f"CUDA RNG topology mismatch: checkpoint={saved_cuda_count}, current={current_cuda_count}."
        )
    if saved_cuda_count:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable.")
        if saved_cuda_index is None:
            raise RuntimeError("CUDA RNG state is missing its device index.")
        saved_cuda_index = int(saved_cuda_index)
        if saved_cuda_index < 0 or saved_cuda_index >= saved_cuda_count:
            raise RuntimeError(
                "CUDA RNG device index is outside the saved topology: "
                f"device={saved_cuda_index}, count={saved_cuda_count}."
            )
        if require_cuda_topology and torch.cuda.current_device() != saved_cuda_index:
            raise RuntimeError(
                "CUDA RNG local device mismatch: "
                f"checkpoint={saved_cuda_index}, current={torch.cuda.current_device()}."
            )
        if not isinstance(state["torch_cuda"], torch.Tensor):
            raise TypeError("CUDA RNG state must be a tensor.")
    elif saved_cuda_index is not None or state["torch_cuda"] is not None:
        raise RuntimeError("CPU-only RNG state contains unexpected CUDA state.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if saved_cuda_count:
        torch.cuda.set_rng_state(state["torch_cuda"], device=saved_cuda_index)


def write_success_marker(directory: str | os.PathLike[str], *, resumable: bool) -> Path:
    name = "_RESUMABLE_SUCCESS" if resumable else "_SUCCESS"
    path = Path(directory) / name
    atomic_write_bytes(path, b"")
    return path


def write_base_reference(
    directory: str | os.PathLike[str],
    *,
    base_checkpoint: str | os.PathLike[str],
    base_manifest: str | os.PathLike[str],
    base_sha256: str | None = None,
) -> dict[str, Any]:
    base_checkpoint = Path(base_checkpoint).expanduser().resolve()
    base_manifest = Path(base_manifest).expanduser().resolve()
    value = {
        "schema_version": 1,
        "base_checkpoint": str(base_checkpoint),
        "base_sha256": base_sha256 or sha256_file(base_checkpoint),
        "base_manifest": str(base_manifest),
        "base_manifest_sha256": sha256_file(base_manifest),
    }
    atomic_write_json(Path(directory) / "base_reference.json", value)
    return value


def build_checkpoint_manifest(
    directory: str | os.PathLike[str],
    *,
    completed_step: int,
    world_size: int,
    sequence_parallel_size: int,
    data_parallel_size: int,
    resumable: bool,
    files: Iterable[str] | None = None,
) -> dict[str, Any]:
    directory = Path(directory)
    step = checkpoint_step(directory)
    if step != int(completed_step):
        raise ValueError(
            f"Checkpoint directory step {step} does not match completed_step {completed_step}."
        )
    if world_size != sequence_parallel_size * data_parallel_size:
        raise ValueError("world_size must equal SP×DP.")
    if files is None:
        ignored = {"checkpoint_manifest.json", "_SUCCESS", "_RESUMABLE_SUCCESS"}
        files = sorted(
            path.name for path in directory.iterdir() if path.is_file() and path.name not in ignored
        )
    entries = []
    for name in sorted(set(files)):
        if Path(name).name != name:
            raise ValueError(f"Checkpoint manifest file names must be local: {name!r}")
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append({"name": name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "completed_step": int(completed_step),
        "next_update_index": int(completed_step),
        "topology": {
            "world_size": int(world_size),
            "sequence_parallel_size": int(sequence_parallel_size),
            "data_parallel_size": int(data_parallel_size),
        },
        "resumable": bool(resumable),
        "error_buffer_resume_policy": "canonical_per_sp_position",
        "files": entries,
    }
    value["manifest_sha256"] = canonical_json_sha256(value)
    return value


def write_checkpoint_manifest(directory: str | os.PathLike[str], **kwargs) -> dict[str, Any]:
    value = build_checkpoint_manifest(directory, **kwargs)
    atomic_write_json(Path(directory) / "checkpoint_manifest.json", value)
    return value


def _read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "checkpoint_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    expected = value.pop("manifest_sha256", None)
    actual = canonical_json_sha256(value)
    value["manifest_sha256"] = expected
    if expected != actual:
        raise RuntimeError(
            f"Checkpoint manifest hash mismatch at {path}: expected {expected}, got {actual}."
        )
    return value


def validate_checkpoint(
    directory: str | os.PathLike[str],
    *,
    require_resumable: bool = False,
    expected_base_sha256: str | None = None,
    expected_topology: tuple[int, int, int] | None = None,
    verify_file_hashes: bool = True,
) -> dict[str, Any]:
    directory = Path(directory)
    required = set(ARTIFACT_REQUIRED_FILES)
    if require_resumable:
        if expected_topology is None:
            world, sp = 6, 3
        else:
            world, sp, _ = expected_topology
        required.update(heavy_required_files(world, sp))
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        raise RuntimeError(f"Checkpoint {directory} is incomplete; missing {missing}.")
    manifest = _read_manifest(directory)
    if manifest.get("schema") != CHECKPOINT_SCHEMA or int(
        manifest.get("schema_version", -1)
    ) != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported checkpoint schema in {directory}.")
    step = checkpoint_step(directory)
    if int(manifest.get("completed_step", -1)) != step or int(
        manifest.get("next_update_index", -1)
    ) != step:
        raise RuntimeError(f"Checkpoint step metadata mismatch in {directory}.")
    if require_resumable and not manifest.get("resumable", False):
        raise RuntimeError(f"Checkpoint manifest is artifact-only: {directory}")
    topology = manifest.get("topology", {})
    if expected_topology is not None:
        expected = {
            "world_size": int(expected_topology[0]),
            "sequence_parallel_size": int(expected_topology[1]),
            "data_parallel_size": int(expected_topology[2]),
        }
        if topology != expected:
            raise RuntimeError(
                f"Checkpoint topology mismatch: checkpoint={topology}, current={expected}."
            )
    if verify_file_hashes:
        for entry in manifest.get("files", []):
            path = directory / entry["name"]
            if not path.is_file():
                raise RuntimeError(f"Manifest-listed checkpoint file is missing: {path}")
            actual = sha256_file(path)
            if actual != entry["sha256"] or path.stat().st_size != int(entry["size"]):
                raise RuntimeError(f"Checkpoint file hash/size mismatch: {path}")
    with (directory / "base_reference.json").open("r", encoding="utf-8") as handle:
        base_reference = json.load(handle)
    if expected_base_sha256 is not None and base_reference.get("base_sha256") != expected_base_sha256:
        raise RuntimeError(
            f"Base checkpoint hash mismatch: expected {expected_base_sha256}, "
            f"got {base_reference.get('base_sha256')}."
        )
    return manifest


def find_latest_resumable_checkpoint(
    root: str | os.PathLike[str],
    *,
    expected_topology: tuple[int, int, int] = (6, 3, 2),
    expected_base_sha256: str | None = None,
) -> Path | None:
    """Return the latest valid marker, rejecting every corrupted candidate.

    A damaged newer checkpoint must never make callers silently resume an
    older state. Artifact-only checkpoints are valid but are not candidates.
    """

    root = Path(root)
    if not root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for marker in sorted(root.rglob("_RESUMABLE_SUCCESS")):
        path = marker.parent
        if path.parent.resolve() != root.resolve():
            raise RuntimeError(f"unexpected nested resumable marker: {marker}")
        try:
            step = checkpoint_step(path)
        except ValueError as exc:
            raise RuntimeError(
                f"resumable marker is not inside checkpoint_model_XXXXXX: {marker}"
            ) from exc
        validate_checkpoint(
            path,
            require_resumable=True,
            expected_base_sha256=expected_base_sha256,
            expected_topology=expected_topology,
        )
        candidates.append((step, path))

    for path in sorted(root.iterdir()):
        if (
            not path.is_dir()
            or CHECKPOINT_DIRECTORY_PATTERN.fullmatch(path.name) is None
            or (path / "_RESUMABLE_SUCCESS").is_file()
        ):
            continue
        if not (path / "_SUCCESS").is_file():
            raise RuntimeError(f"incomplete uncommitted Stage-1 checkpoint: {path}")
        manifest = validate_checkpoint(
            path,
            require_resumable=False,
            expected_base_sha256=expected_base_sha256,
            expected_topology=expected_topology,
        )
        if bool(manifest.get("resumable", False)):
            raise RuntimeError(
                "checkpoint advertises resumable state but lacks its final marker: "
                f"{path}"
            )
    return max(candidates, default=(None, None))[1]


def remove_heavy_resume_state(directory: str | os.PathLike[str]) -> None:
    """Downgrade a checkpoint to artifact-only while preserving all adapters."""
    directory = Path(directory)
    manifest = validate_checkpoint(directory, require_resumable=False)
    marker = directory / "_RESUMABLE_SUCCESS"
    if marker.exists():
        marker.unlink()
    heavy_patterns = (
        re.compile(r"trainer_state\.pt"),
        re.compile(r"ema_local_rank[0-9]{5}\.pt"),
        re.compile(r"rng_state_rank[0-9]{5}\.pt"),
        re.compile(r"error_buffer_sp[0-9]+\.pt"),
    )
    heavy_names = {
        entry["name"]
        for entry in manifest["files"]
        if any(pattern.fullmatch(entry["name"]) for pattern in heavy_patterns)
    }
    # First atomically remove heavy files from the recognized manifest. If the
    # process dies during unlink, stale unlisted files are harmless artifacts.
    manifest["files"] = [entry for entry in manifest["files"] if entry["name"] not in heavy_names]
    manifest["resumable"] = False
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(directory / "checkpoint_manifest.json", manifest)
    for name in sorted(heavy_names):
        path = directory / name
        if path.exists():
            path.unlink()


def apply_resume_retention(
    root: str | os.PathLike[str],
    *,
    keep_last: int,
    keep_steps: Iterable[int] = (),
) -> list[int]:
    if keep_last < 0:
        raise ValueError("keep_last must be non-negative.")
    root = Path(root)
    candidates = sorted(
        (checkpoint_step(path), path)
        for path in root.iterdir()
        if path.is_dir()
        and CHECKPOINT_DIRECTORY_PATTERN.fullmatch(path.name)
        and (path / "_RESUMABLE_SUCCESS").is_file()
    ) if root.is_dir() else []
    keep = {int(value) for value in keep_steps}
    keep.update(step for step, _ in candidates[-keep_last:] if keep_last)
    removed = []
    for step, path in candidates:
        if step in keep:
            continue
        remove_heavy_resume_state(path)
        removed.append(step)
    return removed


def apply_stage1_resume_retention_collective(
    root: str | os.PathLike[str],
    *,
    keep_last: int = 2,
    keep_steps: Iterable[int] = (300,),
    collectives: Stage1CollectiveOps | None = None,
) -> list[int] | None:
    """Run retention on rank 0 while all ranks bracket it with consensus."""

    ops = _stage1_collectives(collectives)
    rank = int(ops.get_rank())
    ops.barrier()
    removed = _consensus_call(
        "rank-0 resume-heavy retention",
        lambda: apply_resume_retention(
            root,
            keep_last=keep_last,
            keep_steps=keep_steps,
        )
        if rank == 0
        else None,
        ops,
    )
    ops.barrier()
    return removed if rank == 0 else None


def finalize_stage1_checkpoint(
    directory: str | os.PathLike[str],
    *,
    completed_step: int,
    resumable: bool,
    collectives: Stage1CollectiveOps | None = None,
) -> Mapping[str, Any] | None:
    """Commit manifest and markers only after every required file is complete.

    Expected call order is adapter pair -> trainer/rank-local heavy payloads ->
    this function.  No marker is written if required-file or hash validation
    fails.  ``_RESUMABLE_SUCCESS`` is always the final filesystem mutation.
    """

    ops = _stage1_collectives(collectives)
    rank = int(ops.get_rank())
    directory = Path(directory)
    ops.barrier()

    def preflight_rank_zero() -> None:
        if rank != 0:
            return
        if checkpoint_step(directory) != int(completed_step):
            raise ValueError("checkpoint directory does not match completed_step")
        existing_markers = [
            name
            for name in ("_SUCCESS", "_RESUMABLE_SUCCESS")
            if (directory / name).exists()
        ]
        if existing_markers:
            raise RuntimeError(
                f"refusing to overwrite committed checkpoint markers: {existing_markers}"
            )
        required = {
            "adapter_raw.safetensors",
            "adapter_ema.safetensors",
            "resolved_config.yaml",
            "base_reference.json",
        }
        if resumable:
            required.update(
                heavy_required_files(
                    STAGE1_WORLD_SIZE, STAGE1_SEQUENCE_PARALLEL_SIZE
                )
                - {"_RESUMABLE_SUCCESS"}
            )
        missing = sorted(name for name in required if not (directory / name).is_file())
        if missing:
            raise RuntimeError(
                f"checkpoint preflight failed before markers; missing {missing}"
            )

    _consensus_call("checkpoint marker preflight", preflight_rank_zero, ops)

    def manifest_rank_zero() -> Mapping[str, Any] | None:
        if rank != 0:
            return None
        manifest_path = directory / "checkpoint_manifest.json"
        if manifest_path.exists():
            raise RuntimeError("refusing to overwrite an existing checkpoint manifest")
        value = write_checkpoint_manifest(
            directory,
            completed_step=completed_step,
            world_size=STAGE1_WORLD_SIZE,
            sequence_parallel_size=STAGE1_SEQUENCE_PARALLEL_SIZE,
            data_parallel_size=STAGE1_DATA_PARALLEL_SIZE,
            resumable=resumable,
        )
        # Validate every listed byte before publishing either success marker.
        loaded = _read_manifest(directory)
        if loaded != value:
            raise RuntimeError("checkpoint manifest changed during atomic publication")
        for entry in loaded["files"]:
            path = directory / entry["name"]
            if (
                not path.is_file()
                or path.stat().st_size != int(entry["size"])
                or sha256_file(path) != entry["sha256"]
            ):
                raise RuntimeError(f"checkpoint manifest verification failed: {path}")
        return value

    manifest = _consensus_call(
        "atomic checkpoint manifest publication", manifest_rank_zero, ops
    )
    ops.barrier()

    def artifact_marker_rank_zero() -> None:
        if rank == 0:
            write_success_marker(directory, resumable=False)

    try:
        _consensus_call(
            "artifact success marker publication", artifact_marker_rank_zero, ops
        )
    except Exception:
        cleanup_error: Exception | None = None
        if rank == 0:
            try:
                for name in ("_SUCCESS", "_RESUMABLE_SUCCESS"):
                    marker = directory / name
                    if marker.exists():
                        marker.unlink()
            except Exception as exc:
                cleanup_error = exc
        if not ops.consensus(cleanup_error is None):
            raise RuntimeError("failed to roll back checkpoint markers") from cleanup_error
        raise
    ops.barrier()

    if resumable:
        def resume_marker_rank_zero() -> None:
            if rank == 0:
                write_success_marker(directory, resumable=True)

        try:
            _consensus_call(
                "resumable success marker publication", resume_marker_rank_zero, ops
            )
        except Exception:
            cleanup_error: Exception | None = None
            if rank == 0:
                try:
                    for name in ("_RESUMABLE_SUCCESS", "_SUCCESS"):
                        marker = directory / name
                        if marker.exists():
                            marker.unlink()
                except Exception as exc:
                    cleanup_error = exc
            if not ops.consensus(cleanup_error is None):
                raise RuntimeError("failed to roll back checkpoint markers") from cleanup_error
            raise
        ops.barrier()
    return manifest if rank == 0 else None
