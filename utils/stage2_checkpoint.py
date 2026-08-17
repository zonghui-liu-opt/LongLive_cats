"""Cycle-boundary, fail-closed checkpoints for Stage-2 DMD/DFD training.

The checkpoint contains only the two trainable LoRA roles and their training
state.  It never asks FSDP for a full model state dict, so the three immutable
5B bases are referenced by verified provenance instead of being copied into
every checkpoint.

Publication is deliberately stricter than a collection of atomic files:
everything is written below a same-parent hidden directory, every byte is
hashed into a self-hashed manifest, that directory is atomically renamed, and
``_SUCCESS`` is written last. Discovery ignores a final-name directory without
the marker (the only expected crash window), while a damaged checkpoint that
claims completion still fails closed instead of silently resuming older work.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch

from utils.lora_utils import LocalLoraShard, LoraTensorSpec
from utils.stage2_code_version import capture_stage2_source_version
from utils.stage1_checkpoint import capture_rng_state, restore_rng_state
from utils.stage1_io import (
    atomic_output_path,
    atomic_torch_save,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)

STAGE2_CHECKPOINT_SCHEMA = "longlive_stage2_checkpoint"
STAGE2_CHECKPOINT_SCHEMA_VERSION = 2
STAGE2_TRAINER_STATE_SCHEMA = "longlive_stage2_trainer_state"
STAGE2_TRAINER_STATE_SCHEMA_VERSION = 1
STAGE2_RNG_STATE_SCHEMA = "longlive_stage2_rank_rng"
STAGE2_RNG_STATE_SCHEMA_VERSION = 1
STAGE2_PHASE_SCHEDULE_VERSION = "longlive_stage2_phase_schedule/v1"
STAGE2_WORLD_SIZE = 8
STAGE2_EMA_START_GENERATOR_UPDATE = 40
STAGE2_GENERATOR_UPDATES_PER_EPOCH = 10
STAGE2_PHASE_A_GENERATOR_UPDATES = 240
STAGE2_PHASE_B_GENERATOR_UPDATES = 40
STAGE2_TOTAL_GENERATOR_UPDATES = 280
STAGE2_CHECKPOINT_MILESTONES = frozenset((80, 120, 160, 200, 240, 250, 260, 270, 280))
STAGE2_CHECKPOINT_PATTERN = re.compile(r"checkpoint_stage2_g([0-9]{6})")
STAGE2_RANK0_CONTROL_RNG_NAMES = frozenset(
    ("generator_exit", "fake_score_exit", "dfd_branch")
)
STAGE2_DEDICATED_RNG_NAMES = frozenset(
    (
        "fake_score_loader",
        "generator_loader",
        "fake_score_rollout",
        "generator_rollout",
        "fake_score_timestep",
        "generator_timestep",
        "fake_score_noise",
        "generator_noise",
    )
)
STAGE2_PENDING_STATE_KEYS = frozenset(
    ("gradients", "batch", "branch", "kv", "optimizer_step")
)
STAGE2_ROLE_NAMES = ("fake_score", "generator")
STAGE2_PROVENANCE_SCHEMA = "longlive_stage2_checkpoint_provenance"
STAGE2_PROVENANCE_SCHEMA_VERSION = 2

_STAGE2_OPTIMIZER_CONTRACT = {
    "generator": {
        "lr": 2e-6,
        "betas": (0.0, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
    },
    "fake_score": {
        "lr": 4e-7,
        "betas": (0.0, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
    },
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def _exact_mapping(
    value: Any, *, label: str, expected_keys: Iterable[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result = dict(value)
    expected = set(expected_keys)
    missing = sorted(expected - set(result))
    extra = sorted(set(result) - expected)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")
    return result


def validate_stage2_provenance(
    provenance: Mapping[str, Any],
    *,
    add_code_version: bool = False,
) -> dict[str, Any]:
    """Normalize and validate the exact checkpoint provenance schema."""

    if not isinstance(provenance, Mapping):
        raise TypeError("Stage-2 checkpoint provenance must be a mapping")
    value = dict(provenance)
    raw_keys = {"assets", "data", "lineage", "smoke_probe"}
    normalized_keys = {
        "schema",
        "schema_version",
        "code_version",
        *raw_keys,
    }
    if set(value) == raw_keys:
        if not add_code_version:
            raise ValueError("Stage-2 provenance is missing schema/code_version")
        value = {
            "schema": STAGE2_PROVENANCE_SCHEMA,
            "schema_version": STAGE2_PROVENANCE_SCHEMA_VERSION,
            "code_version": capture_stage2_source_version(),
            **value,
        }
    value = _exact_mapping(
        value,
        label="Stage-2 checkpoint provenance",
        expected_keys=normalized_keys,
    )
    if (
        value["schema"] != STAGE2_PROVENANCE_SCHEMA
        or type(value["schema_version"]) is not int
        or value["schema_version"] != STAGE2_PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported Stage-2 checkpoint provenance schema")
    code = _exact_mapping(
        value["code_version"],
        label="Stage-2 code_version",
        expected_keys={"stage2_source_sha256"},
    )
    _sha256(code["stage2_source_sha256"], "code_version.stage2_source_sha256")

    assets = _exact_mapping(
        value["assets"],
        label="Stage-2 provenance assets",
        expected_keys={"generator", "real_score", "fake_score"},
    )
    asset_hashes: dict[str, str] = {}
    for role, asset in assets.items():
        if not isinstance(asset, Mapping):
            raise TypeError(f"Stage-2 provenance asset {role} must be a mapping")
        asset_hashes[role] = _sha256(
            asset.get("checkpoint_sha256"),
            f"Stage-2 provenance assets.{role}.checkpoint_sha256",
        )
    if asset_hashes["real_score"] != asset_hashes["fake_score"]:
        raise ValueError("Stage-2 real/fake immutable base hashes differ")

    data = _exact_mapping(
        value["data"],
        label="Stage-2 provenance data",
        expected_keys={
            "stage2_manifest_sha256",
            "source_manifest_sha256",
            "negative_manifest_sha256",
            "negative_artifact_sha256",
        },
    )
    for name, sha256 in data.items():
        _sha256(sha256, f"Stage-2 provenance data.{name}")
    lineage = _exact_mapping(
        value["lineage"],
        label="Stage-2 provenance lineage",
        expected_keys={
            "parent_checkpoint",
            "parent_checkpoint_manifest_sha256",
        },
    )
    parent = lineage["parent_checkpoint"]
    parent_sha = lineage["parent_checkpoint_manifest_sha256"]
    if parent is None:
        if parent_sha is not None:
            raise ValueError("cold Stage-2 lineage cannot contain a parent hash")
    else:
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError("Stage-2 lineage parent_checkpoint must be a path string")
        _sha256(parent_sha, "lineage.parent_checkpoint_manifest_sha256")
    smoke_probe = value["smoke_probe"]
    if smoke_probe is not None and not isinstance(smoke_probe, Mapping):
        raise TypeError("Stage-2 provenance smoke_probe must be a mapping or None")
    # Prove all nested values are canonical finite JSON before any filesystem
    # mutation. This also rejects tensors or other non-portable provenance.
    canonical_json_sha256(value)
    return value


def _uint8_rng_state_tensor(value: Any, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.uint8
        or value.ndim != 1
    ):
        raise TypeError(f"{label} must be a one-dimensional CPU uint8 Tensor")
    return value


def _cpu_uint8_rng_state(value: Any, label: str) -> torch.Tensor:
    value = _uint8_rng_state_tensor(value, label)
    candidate = torch.Generator(device="cpu")
    try:
        candidate.set_state(value.clone())
    except RuntimeError as exc:
        raise ValueError(f"{label} is not a valid generator state") from exc
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class Stage2CollectiveOps:
    """Small injectable WORLD collective surface used by checkpoint code."""

    get_rank: Callable[[], int]
    get_world_size: Callable[[], int]
    barrier: Callable[[], None]
    consensus: Callable[[bool], bool]
    broadcast_object: Callable[[Any, int], Any]

    def validate(self) -> "Stage2CollectiveOps":
        world_size = self.get_world_size()
        rank = self.get_rank()
        if type(world_size) is not int or type(rank) is not int:
            raise TypeError("Stage-2 collective rank/world_size must be integers")
        if world_size != STAGE2_WORLD_SIZE:
            raise RuntimeError(
                f"Stage-2 checkpoint requires WORLD_SIZE=8, got {world_size}"
            )
        if rank < 0 or rank >= world_size:
            raise RuntimeError(f"invalid Stage-2 checkpoint rank {rank}/{world_size}")
        return self


def _default_collective_ops() -> Stage2CollectiveOps:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Stage-2 collective checkpointing requires an initialized process group"
        )

    def consensus(local_success: bool) -> bool:
        backend = str(dist.get_backend()).lower()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if "nccl" in backend
            else torch.device("cpu")
        )
        flag = torch.tensor(1 if local_success else 0, device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def broadcast_object(value: Any, src: int) -> Any:
        holder = [value if dist.get_rank() == src else None]
        dist.broadcast_object_list(holder, src=src)
        return holder[0]

    return Stage2CollectiveOps(
        get_rank=dist.get_rank,
        get_world_size=dist.get_world_size,
        barrier=dist.barrier,
        consensus=consensus,
        broadcast_object=broadcast_object,
    ).validate()


def _collectives(value: Stage2CollectiveOps | None) -> Stage2CollectiveOps:
    return (_default_collective_ops() if value is None else value).validate()


def _consensus_call(
    label: str, callback: Callable[[], _T], operations: Stage2CollectiveOps
) -> _T:
    result: _T | None = None
    local_error: Exception | None = None
    try:
        result = callback()
    except Exception as exc:  # all ranks must reach the same consensus
        local_error = exc
    if not operations.consensus(local_error is None):
        raise RuntimeError(f"Stage-2 checkpoint stage failed: {label}") from local_error
    assert local_error is None
    return result  # type: ignore[return-value]


def validate_stage2_cycle_boundary(
    *,
    completed_generator_updates: int,
    completed_fake_updates: int,
    completed_cycles: int,
    next_substep: str,
    accumulation_cursor: int,
    pending_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that a checkpoint is exactly after ``5F -> G -> EMA``."""

    completed_g = _plain_int(
        completed_generator_updates, "completed_generator_updates", minimum=1
    )
    completed_f = _plain_int(
        completed_fake_updates, "completed_fake_updates", minimum=1
    )
    cycles = _plain_int(completed_cycles, "completed_cycles", minimum=1)
    if completed_f != 5 * completed_g:
        raise RuntimeError(
            "cycle boundary requires completed_fake_updates="
            "5*completed_generator_updates"
        )
    if cycles != completed_g:
        raise RuntimeError(
            "completed_cycles must equal completed_generator_updates at a boundary"
        )
    if next_substep != "F1":
        raise RuntimeError("checkpoint next_substep must be F1")
    if _plain_int(accumulation_cursor, "accumulation_cursor") != 0:
        raise RuntimeError("checkpoint accumulation_cursor must be zero")
    pending = _exact_mapping(
        pending_state,
        label="pending_state",
        expected_keys=STAGE2_PENDING_STATE_KEYS,
    )
    if any(type(value) is not bool for value in pending.values()) or any(
        pending.values()
    ):
        raise RuntimeError(
            "checkpoint pending_state must contain only false gradients/batch/"
            "branch/KV/optimizer flags"
        )
    return {
        "completed_generator_updates": completed_g,
        "completed_fake_updates": completed_f,
        "completed_cycles": cycles,
        "next_substep": "F1",
        "accumulation_cursor": 0,
        "pending_state": pending,
    }


def _milestone_name(completed_g: int) -> str | None:
    if completed_g in (80, 120, 160, 200, 240):
        return f"A{completed_g // 10}"
    if completed_g in (250, 260, 270, 280):
        return f"B{(completed_g - 240) // 10}"
    return None


def derive_stage2_phase_state(
    completed_generator_updates: int,
    *,
    phase_a_generator_updates: int = STAGE2_PHASE_A_GENERATOR_UPDATES,
    phase_b_generator_updates: int = STAGE2_PHASE_B_GENERATOR_UPDATES,
    phase_b_mode: str = "dmd_dfd",
) -> dict[str, Any]:
    """Derive the next phase/DFD position from the successful G clock only."""

    completed = _plain_int(completed_generator_updates, "completed_generator_updates")
    phase_a = _plain_int(
        phase_a_generator_updates, "phase_a_generator_updates", minimum=1
    )
    phase_b = _plain_int(phase_b_generator_updates, "phase_b_generator_updates")
    if phase_a % 10 or phase_b % 10:
        raise ValueError("Stage-2 phase update counts must be whole 10-update epochs")
    if phase_b == 0:
        if phase_b_mode != "disabled":
            raise ValueError(
                "an A-only Stage-2 checkpoint schedule requires phase_b_mode=disabled"
            )
    elif phase_b_mode not in {"dmd_dfd", "dmd_only"}:
        raise ValueError(f"unsupported Stage-2 phase_b_mode {phase_b_mode!r}")
    total = phase_a + phase_b
    if completed > total:
        raise RuntimeError(
            f"completed G updates exceed the resolved schedule: {completed}>{total}"
        )
    cursor = completed % STAGE2_GENERATOR_UPDATES_PER_EPOCH
    if completed < phase_a:
        phase = "A"
        phase_epoch = completed // 10
        next_probability: float | None = 0.0
    elif completed < total:
        phase = "B"
        phase_epoch = (completed - phase_a) // 10
        if phase_b_mode == "dmd_only":
            next_probability = 0.0
        elif phase_epoch == 0:
            next_probability = 0.25 * cursor / 9
        else:
            next_probability = 0.25
    else:
        phase = "complete"
        phase_epoch = phase_b // 10
        next_probability = None
    return {
        "schedule_version": STAGE2_PHASE_SCHEDULE_VERSION,
        "phase_a_generator_updates": phase_a,
        "phase_b_generator_updates": phase_b,
        "phase_b_mode": phase_b_mode,
        "total_generator_updates": total,
        "phase": phase,
        "phase_epoch_index": phase_epoch,
        "completed_generator_epochs": completed // 10,
        "generator_update_cursor_in_epoch": cursor,
        "next_dfd_probability": next_probability,
        "milestone": _milestone_name(completed),
    }


_TRAINER_STATE_KEYS = {
    "schema",
    "schema_version",
    "completed_generator_updates",
    "completed_fake_updates",
    "completed_cycles",
    "cycle",
    "next_substep",
    "accumulation_cursor",
    "pending_state",
    "successful_attempts",
    "nonfinite_attempts",
    "next_logical_substep_id",
    "phase_state",
    "sampler_state",
    "dataloader_generator_states",
    "jsonl_lineage",
    "contract_hash",
    "launch_hash",
    "cache_audit_launch_hash",
}


def build_stage2_trainer_state(
    *,
    completed_generator_updates: int,
    completed_fake_updates: int,
    successful_attempts: Mapping[str, int],
    nonfinite_attempts: Mapping[str, int],
    sampler_state: Mapping[str, Any],
    dataloader_generator_states: Mapping[str, torch.Tensor],
    run_id: str,
    next_attempt_index: int,
    contract_hash: str,
    launch_hash: str,
    cache_audit_launch_hash: str | None = None,
    phase_a_generator_updates: int = STAGE2_PHASE_A_GENERATOR_UPDATES,
    phase_b_generator_updates: int = STAGE2_PHASE_B_GENERATOR_UPDATES,
    phase_b_mode: str = "dmd_dfd",
) -> dict[str, Any]:
    """Build the only legal rank-0 state: the next action is always F1."""

    boundary = validate_stage2_cycle_boundary(
        completed_generator_updates=completed_generator_updates,
        completed_fake_updates=completed_fake_updates,
        completed_cycles=completed_generator_updates,
        next_substep="F1",
        accumulation_cursor=0,
        pending_state={key: False for key in STAGE2_PENDING_STATE_KEYS},
    )
    completed_g = boundary["completed_generator_updates"]
    state = {
        "schema": STAGE2_TRAINER_STATE_SCHEMA,
        "schema_version": STAGE2_TRAINER_STATE_SCHEMA_VERSION,
        **boundary,
        "cycle": completed_g,
        "successful_attempts": dict(successful_attempts),
        "nonfinite_attempts": dict(nonfinite_attempts),
        "next_logical_substep_id": 6 * completed_g,
        "phase_state": derive_stage2_phase_state(
            completed_g,
            phase_a_generator_updates=phase_a_generator_updates,
            phase_b_generator_updates=phase_b_generator_updates,
            phase_b_mode=phase_b_mode,
        ),
        "sampler_state": dict(sampler_state),
        "dataloader_generator_states": dict(dataloader_generator_states),
        "jsonl_lineage": {
            "run_id": run_id,
            "next_attempt_index": next_attempt_index,
        },
        "contract_hash": contract_hash,
        "launch_hash": launch_hash,
        "cache_audit_launch_hash": (
            launch_hash if cache_audit_launch_hash is None else cache_audit_launch_hash
        ),
    }
    validate_stage2_trainer_state(state)
    return state


def validate_stage2_trainer_state(
    state: Mapping[str, Any],
    *,
    expected_contract_hash: str | None = None,
) -> Mapping[str, Any]:
    """Validate counters and every redundant resume cursor before mutation."""

    value = _exact_mapping(
        state, label="Stage-2 trainer state", expected_keys=_TRAINER_STATE_KEYS
    )
    if (
        value["schema"] != STAGE2_TRAINER_STATE_SCHEMA
        or type(value["schema_version"]) is not int
        or value["schema_version"] != STAGE2_TRAINER_STATE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported Stage-2 trainer state schema")
    boundary = validate_stage2_cycle_boundary(
        completed_generator_updates=value["completed_generator_updates"],
        completed_fake_updates=value["completed_fake_updates"],
        completed_cycles=value["completed_cycles"],
        next_substep=value["next_substep"],
        accumulation_cursor=value["accumulation_cursor"],
        pending_state=value["pending_state"],
    )
    completed_g = boundary["completed_generator_updates"]
    completed_f = boundary["completed_fake_updates"]
    if _plain_int(value["cycle"], "cycle", minimum=1) != completed_g:
        raise RuntimeError("trainer cycle differs from completed_cycles")
    if (
        _plain_int(value["next_logical_substep_id"], "next_logical_substep_id")
        != 6 * completed_g
    ):
        raise RuntimeError("next_logical_substep_id differs from the cycle clock")

    successful = _exact_mapping(
        value["successful_attempts"],
        label="successful_attempts",
        expected_keys=STAGE2_ROLE_NAMES,
    )
    expected_successful = {"generator": completed_g, "fake_score": completed_f}
    if successful != expected_successful:
        raise RuntimeError(
            f"successful_attempts mismatch: {successful} != {expected_successful}"
        )
    nonfinite = _exact_mapping(
        value["nonfinite_attempts"],
        label="nonfinite_attempts",
        expected_keys=STAGE2_ROLE_NAMES,
    )
    for role, count in nonfinite.items():
        _plain_int(count, f"nonfinite_attempts.{role}")

    phase = _exact_mapping(
        value["phase_state"],
        label="phase_state",
        expected_keys={
            "schedule_version",
            "phase_a_generator_updates",
            "phase_b_generator_updates",
            "phase_b_mode",
            "total_generator_updates",
            "phase",
            "phase_epoch_index",
            "completed_generator_epochs",
            "generator_update_cursor_in_epoch",
            "next_dfd_probability",
            "milestone",
        },
    )
    expected_phase = derive_stage2_phase_state(
        completed_g,
        phase_a_generator_updates=phase["phase_a_generator_updates"],
        phase_b_generator_updates=phase["phase_b_generator_updates"],
        phase_b_mode=phase["phase_b_mode"],
    )
    if phase != expected_phase:
        raise RuntimeError(
            f"trainer phase_state is not derived from completed G: {phase} != "
            f"{expected_phase}"
        )

    from utils.stage2_sampler import validate_stage2_sampler_streams_state

    validate_stage2_sampler_streams_state(
        value["sampler_state"],
        expected_completed_batches={
            "fake_score": completed_f,
            "generator": completed_g,
        },
    )

    loader_states = _exact_mapping(
        value["dataloader_generator_states"],
        label="dataloader_generator_states",
        expected_keys=STAGE2_ROLE_NAMES,
    )
    for role, generator_state in loader_states.items():
        _cpu_uint8_rng_state(generator_state, f"DataLoader {role} generator state")

    lineage = _exact_mapping(
        value["jsonl_lineage"],
        label="jsonl_lineage",
        expected_keys={"run_id", "next_attempt_index"},
    )
    if (
        not isinstance(lineage["run_id"], str)
        or not lineage["run_id"]
        or lineage["run_id"] != lineage["run_id"].strip()
    ):
        raise ValueError("jsonl_lineage.run_id must be a non-empty clean string")
    _plain_int(lineage["next_attempt_index"], "jsonl_lineage.next_attempt_index")
    contract_hash = _sha256(value["contract_hash"], "contract_hash")
    _sha256(value["launch_hash"], "launch_hash")
    _sha256(value["cache_audit_launch_hash"], "cache_audit_launch_hash")
    if expected_contract_hash is not None and contract_hash != _sha256(
        expected_contract_hash, "expected_contract_hash"
    ):
        raise RuntimeError(
            "Stage-2 trainer contract_hash differs from the current config"
        )
    return state


def _generator_snapshot(generator: torch.Generator, label: str) -> dict[str, Any]:
    if not isinstance(generator, torch.Generator):
        raise TypeError(f"{label} must be a torch.Generator")
    device = str(generator.device)
    state = generator.get_state().detach().to(device="cpu").clone()
    _uint8_rng_state_tensor(state, f"{label} state")
    return {"device": device, "state": state}


def capture_stage2_rng_state(
    *,
    rank: int,
    dedicated_generators: Mapping[str, torch.Generator],
    rank0_control_generators: Mapping[str, torch.Generator] | None,
    include_cuda: bool = True,
) -> dict[str, Any]:
    """Capture default and every explicit Stage-2 RNG stream for one rank."""

    rank = _plain_int(rank, "rank")
    if rank >= STAGE2_WORLD_SIZE:
        raise ValueError(f"rank must be below {STAGE2_WORLD_SIZE}")
    if not isinstance(dedicated_generators, Mapping) or not dedicated_generators:
        raise ValueError("dedicated_generators must be a non-empty mapping")
    names = sorted(dedicated_generators)
    if any(
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or name in STAGE2_RANK0_CONTROL_RNG_NAMES
        for name in names
    ):
        raise ValueError("dedicated RNG names must be clean and non-control names")
    controls = dict(rank0_control_generators or {})
    if rank == 0 and set(controls) != STAGE2_RANK0_CONTROL_RNG_NAMES:
        raise ValueError(
            "rank0 control RNGs must contain generator_exit/fake_score_exit/dfd_branch"
        )
    if rank != 0 and controls:
        raise ValueError("only rank0 may save control RNG streams")
    return {
        "schema": STAGE2_RNG_STATE_SCHEMA,
        "schema_version": STAGE2_RNG_STATE_SCHEMA_VERSION,
        "rank": rank,
        "world_size": STAGE2_WORLD_SIZE,
        "general": capture_rng_state(include_cuda=include_cuda),
        "dedicated": {
            name: _generator_snapshot(dedicated_generators[name], name)
            for name in names
        },
        "rank0_control": {
            name: _generator_snapshot(controls[name], name) for name in sorted(controls)
        },
    }


def _validate_generator_snapshots(
    snapshots: Any,
    generators: Mapping[str, torch.Generator] | None,
    *,
    label: str,
    expected_names: set[str] | None = None,
    expected_cuda_index: int | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(snapshots, Mapping):
        raise TypeError(f"{label} snapshots must be a mapping")
    value = dict(snapshots)
    names = set(value)
    if expected_names is not None and names != expected_names:
        raise RuntimeError(
            f"{label} names mismatch: expected={sorted(expected_names)}, "
            f"actual={sorted(names)}"
        )
    if generators is not None and names != set(generators):
        raise RuntimeError(
            f"{label} names differ from runtime generators: "
            f"checkpoint={sorted(names)}, runtime={sorted(generators)}"
        )
    validated: dict[str, dict[str, Any]] = {}
    for name in sorted(value):
        snapshot = _exact_mapping(
            value[name],
            label=f"{label}.{name}",
            expected_keys={"device", "state"},
        )
        if not isinstance(snapshot["device"], str) or not snapshot["device"]:
            raise TypeError(f"{label}.{name}.device must be a string")
        try:
            declared_device = torch.device(snapshot["device"])
        except (RuntimeError, TypeError) as exc:
            raise RuntimeError(f"invalid {label}.{name} device") from exc
        if declared_device.type not in {"cpu", "cuda"}:
            raise RuntimeError(
                f"{label}.{name} must use a CPU or CUDA generator device"
            )
        if declared_device.type == "cuda" and (
            declared_device.index is None
            or (
                expected_cuda_index is not None
                and declared_device.index != expected_cuda_index
            )
        ):
            raise RuntimeError(
                f"{label}.{name} CUDA device must be explicit and rank-local"
            )
        _uint8_rng_state_tensor(snapshot["state"], f"{label}.{name}.state")
        if (
            generators is not None
            and str(generators[name].device) != snapshot["device"]
        ):
            raise RuntimeError(
                f"{label}.{name} device mismatch: checkpoint={snapshot['device']}, "
                f"runtime={generators[name].device}"
            )
        # CPU bytes are always safe to validate. CUDA bytes are validated only
        # against an explicitly supplied *local* runtime generator; rank 0 must
        # never initialize cuda:1..7 merely to inspect foreign-rank payloads.
        if declared_device.type == "cpu" or generators is not None:
            candidate_device = (
                str(generators[name].device)
                if generators is not None
                else snapshot["device"]
            )
            try:
                candidate = torch.Generator(device=candidate_device)
                candidate.set_state(snapshot["state"].clone())
            except (RuntimeError, TypeError) as exc:
                raise RuntimeError(f"invalid {label}.{name} generator state") from exc
        validated[name] = snapshot
    return validated


def _validate_general_rng_state(state: Any) -> Mapping[str, Any]:
    value = _exact_mapping(
        state,
        label="Stage-2 general RNG state",
        expected_keys={
            "schema_version",
            "python",
            "numpy",
            "torch_cpu",
            "cuda_device_count",
            "cuda_device_index",
            "torch_cuda",
        },
    )
    if type(value["schema_version"]) is not int or value["schema_version"] != 2:
        raise RuntimeError("unsupported Stage-2 general RNG schema")
    try:
        random.Random().setstate(value["python"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid Stage-2 Python RNG state") from exc
    try:
        np.random.RandomState().set_state(value["numpy"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid Stage-2 NumPy RNG state") from exc
    _cpu_uint8_rng_state(value["torch_cpu"], "Stage-2 torch CPU RNG state")
    cuda_count = _plain_int(value["cuda_device_count"], "cuda_device_count")
    cuda_index = value["cuda_device_index"]
    cuda_state = value["torch_cuda"]
    if cuda_count == 0:
        if cuda_index is not None or cuda_state is not None:
            raise RuntimeError("CPU-only Stage-2 RNG state contains CUDA data")
    else:
        if (
            isinstance(cuda_index, bool)
            or not isinstance(cuda_index, int)
            or cuda_index < 0
            or cuda_index >= cuda_count
        ):
            raise RuntimeError("Stage-2 CUDA RNG device index is invalid")
        # CUDA generator bytes are device-specific.  Structural publication
        # validation checks their portable CPU representation without touching
        # or initializing that rank's CUDA device on rank 0.
        _uint8_rng_state_tensor(cuda_state, "Stage-2 torch CUDA RNG state")
    return value


def validate_stage2_rng_state(
    state: Mapping[str, Any],
    *,
    expected_rank: int,
    expected_dedicated_names: set[str] | None = None,
) -> Mapping[str, Any]:
    value = _exact_mapping(
        state,
        label="Stage-2 rank RNG state",
        expected_keys={
            "schema",
            "schema_version",
            "rank",
            "world_size",
            "general",
            "dedicated",
            "rank0_control",
        },
    )
    if (
        value["schema"] != STAGE2_RNG_STATE_SCHEMA
        or type(value["schema_version"]) is not int
        or value["schema_version"] != STAGE2_RNG_STATE_SCHEMA_VERSION
    ):
        raise RuntimeError("unsupported Stage-2 rank RNG schema")
    if (
        type(value["rank"]) is not int
        or value["rank"] != expected_rank
        or type(value["world_size"]) is not int
        or value["world_size"] != STAGE2_WORLD_SIZE
    ):
        raise RuntimeError("Stage-2 rank RNG topology mismatch")
    general = _validate_general_rng_state(value["general"])
    if general["cuda_device_count"] and general["cuda_device_index"] != expected_rank:
        raise RuntimeError("Stage-2 CUDA RNG state is not rank-local")
    dedicated = _validate_generator_snapshots(
        value["dedicated"],
        None,
        label="dedicated RNG",
        expected_names=expected_dedicated_names,
        expected_cuda_index=expected_rank,
    )
    if not dedicated:
        raise RuntimeError("Stage-2 checkpoint is missing dedicated RNG streams")
    for loader_name in ("fake_score_loader", "generator_loader"):
        if loader_name in dedicated and dedicated[loader_name]["device"] != "cpu":
            raise RuntimeError(f"Stage-2 {loader_name} RNG must be CPU")
    controls = _validate_generator_snapshots(
        value["rank0_control"],
        None,
        label="rank0 control RNG",
        expected_names=(
            set(STAGE2_RANK0_CONTROL_RNG_NAMES) if expected_rank == 0 else set()
        ),
        expected_cuda_index=expected_rank,
    )
    if expected_rank != 0 and controls:
        raise RuntimeError("nonzero rank contains rank0 control RNG state")
    if any(snapshot["device"] != "cpu" for snapshot in controls.values()):
        raise RuntimeError("Stage-2 rank0 control RNG streams must be CPU")
    return state


def restore_stage2_rng_state(
    state: Mapping[str, Any],
    *,
    rank: int,
    dedicated_generators: Mapping[str, torch.Generator],
    rank0_control_generators: Mapping[str, torch.Generator] | None,
    require_cuda_topology: bool = True,
) -> None:
    """Validate all RNG bytes first, then restore them as the final resume step."""

    validate_stage2_rng_state(state, expected_rank=rank)
    dedicated = _validate_generator_snapshots(
        state["dedicated"],
        dedicated_generators,
        label="dedicated RNG",
    )
    controls_runtime = dict(rank0_control_generators or {})
    controls = _validate_generator_snapshots(
        state["rank0_control"],
        controls_runtime,
        label="rank0 control RNG",
        expected_names=(set(STAGE2_RANK0_CONTROL_RNG_NAMES) if rank == 0 else set()),
    )
    # General restore is deliberately last among construction operations.  The
    # explicit generators are independent objects and cannot consume defaults.
    restore_rng_state(state["general"], require_cuda_topology=require_cuda_topology)
    for name, snapshot in dedicated.items():
        dedicated_generators[name].set_state(snapshot["state"].clone())
    for name, snapshot in controls.items():
        controls_runtime[name].set_state(snapshot["state"].clone())


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor
    except (ImportError, AttributeError):
        DTensor = ()  # type: ignore[assignment]
    return value.detach().to_local() if isinstance(value, DTensor) else value.detach()


def _is_lora_name(name: str) -> bool:
    return any(marker in name for marker in ("lora_A", "lora_B"))


def _validate_stage2_optimizer_param_groups(
    groups: Any,
    *,
    role: str,
) -> Sequence[Mapping[str, Any]]:
    if role not in _STAGE2_OPTIMIZER_CONTRACT:
        raise ValueError(f"invalid Stage-2 optimizer role {role!r}")
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise TypeError(f"Stage-2 {role} optimizer param_groups must be a sequence")
    if len(groups) != 1 or not isinstance(groups[0], Mapping):
        raise ValueError(
            f"Stage-2 {role} optimizer must contain exactly one param group"
        )
    group = groups[0]
    expected = _STAGE2_OPTIMIZER_CONTRACT[role]
    for key in ("lr", "eps", "weight_decay"):
        value = group.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Stage-2 {role} AdamW {key} must be numeric")
        if float(value) != expected[key]:
            raise ValueError(
                f"Stage-2 {role} AdamW {key} mismatch: "
                f"checkpoint/runtime={value}, expected={expected[key]}"
            )
    betas = group.get("betas")
    if (
        isinstance(betas, (str, bytes))
        or not isinstance(betas, Sequence)
        or len(betas) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in betas
        )
        or tuple(float(value) for value in betas) != expected["betas"]
    ):
        raise ValueError(
            f"Stage-2 {role} AdamW betas mismatch: "
            f"checkpoint/runtime={betas}, expected={expected['betas']}"
        )
    return groups


def audit_stage2_lora_optimizer(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    role: str,
    expected_schema: Mapping[str, LoraTensorSpec] | None = None,
    require_initialized_moments: bool = False,
    expected_completed_updates: int | None = None,
) -> tuple[str, ...]:
    """Assert one optimizer owns exactly one role's FP32 LoRA parameters."""

    if role not in STAGE2_ROLE_NAMES:
        raise ValueError(f"invalid Stage-2 optimizer role {role!r}")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError(f"Stage-2 {role} optimizer must be torch.optim.AdamW")
    named = OrderedDict(
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )
    if not named:
        raise ValueError(f"Stage-2 {role} optimizer model has no trainables")
    invalid = [
        (name, parameter.dtype)
        for name, parameter in named.items()
        if parameter.dtype != torch.float32 or not _is_lora_name(name)
    ]
    if invalid:
        raise TypeError(
            f"Stage-2 {role} optimizer accepts only FP32 LoRA A/B: {invalid[:8]}"
        )
    if expected_schema is not None:
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

    expected_ids = {id(parameter) for parameter in named.values()}
    group_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    ]
    group_ids = [id(parameter) for parameter in group_parameters]
    if len(group_ids) != len(set(group_ids)) or set(group_ids) != expected_ids:
        raise ValueError(
            f"Stage-2 {role} optimizer parameter identities do not exactly match "
            "the role LoRA"
        )
    if any(id(parameter) not in expected_ids for parameter in optimizer.state):
        raise ValueError(f"Stage-2 {role} optimizer state contains another role")
    _validate_stage2_optimizer_param_groups(optimizer.param_groups, role=role)
    if require_initialized_moments:
        if set(map(id, optimizer.state)) != expected_ids:
            raise ValueError(f"Stage-2 {role} optimizer moments are incomplete")
        for name, parameter in named.items():
            values = optimizer.state[parameter]
            _validate_adam_values(
                values,
                label=f"local {role} optimizer {name}",
                expected_completed_updates=expected_completed_updates,
                require_cpu=False,
            )
    return tuple(named)


def _validate_adam_values(
    values: Any,
    *,
    label: str,
    expected_completed_updates: int | None,
    require_cpu: bool,
) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} state must be a mapping")
    missing = sorted({"step", "exp_avg", "exp_avg_sq"} - set(values))
    if missing:
        raise ValueError(f"{label} is missing AdamW values {missing}")
    step_value = values["step"]
    if isinstance(step_value, torch.Tensor):
        local = _local_tensor(step_value)
        if local.numel() != 1 or not bool(torch.isfinite(local).all().item()):
            raise ValueError(f"{label} AdamW step must be one finite scalar")
        step_value = local.item()
    if (
        isinstance(step_value, bool)
        or not isinstance(step_value, (int, float))
        or not math.isfinite(float(step_value))
        or int(step_value) != float(step_value)
        or int(step_value) < 0
    ):
        raise ValueError(f"{label} AdamW step is invalid: {step_value!r}")
    if expected_completed_updates is not None and int(step_value) != int(
        expected_completed_updates
    ):
        raise ValueError(
            f"{label} AdamW step mismatch: checkpoint={int(step_value)}, "
            f"expected={int(expected_completed_updates)}"
        )
    for name in ("exp_avg", "exp_avg_sq"):
        tensor = values[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} {name} is not a Tensor")
        local = _local_tensor(tensor)
        if local.dtype != torch.float32:
            raise TypeError(f"{label} {name} must be FP32")
        if require_cpu and local.device.type != "cpu":
            raise TypeError(f"{label} {name} must be on CPU")
        if local.numel() and not bool(torch.isfinite(local).all().item()):
            raise ValueError(f"{label} {name} contains non-finite values")


def validate_stage2_full_optimizer_state(
    state: Mapping[str, Any],
    *,
    role: str,
    expected_parameter_names: Sequence[str],
    expected_completed_updates: int,
) -> Mapping[str, Any]:
    """Validate one rank-0 DCP full optimizer-only CPU state."""

    if role not in STAGE2_ROLE_NAMES:
        raise ValueError(f"invalid Stage-2 optimizer role {role!r}")
    value = _exact_mapping(
        state,
        label=f"Stage-2 {role} full optimizer state",
        expected_keys={"state", "param_groups"},
    )
    moments = value["state"]
    groups = value["param_groups"]
    if not isinstance(moments, Mapping) or not isinstance(groups, Sequence):
        raise TypeError(f"Stage-2 {role} optimizer containers are invalid")
    _validate_stage2_optimizer_param_groups(groups, role=role)
    expected = set(expected_parameter_names)
    if not expected or len(expected) != len(tuple(expected_parameter_names)):
        raise ValueError(f"Stage-2 {role} expected parameter names are invalid")
    grouped: list[str] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or "params" not in group:
            raise TypeError(f"Stage-2 {role} optimizer group {index} is invalid")
        params = group["params"]
        if (
            isinstance(params, (str, bytes))
            or not isinstance(params, Sequence)
            or not all(isinstance(name, str) for name in params)
        ):
            raise TypeError(
                f"Stage-2 {role} optimizer group {index} params must be FQNs"
            )
        grouped.extend(params)
    if len(grouped) != len(set(grouped)) or set(grouped) != expected:
        raise ValueError(
            f"Stage-2 {role} optimizer parameters differ from its role schema"
        )
    if set(moments) != expected:
        raise ValueError(
            f"Stage-2 {role} optimizer moment keys differ from its role parameters"
        )
    for name in sorted(expected):
        _validate_adam_values(
            moments[name],
            label=f"full {role} optimizer {name}",
            expected_completed_updates=expected_completed_updates,
            require_cpu=True,
        )
    return state


def _dcp_optimizer_apis():
    try:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_optimizer_state_dict,
            set_optimizer_state_dict,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Stage-2 FSDP2 optimizer resume requires PyTorch 2.8 DCP state-dict APIs"
        ) from exc
    return get_optimizer_state_dict, set_optimizer_state_dict, StateDictOptions


def gather_stage2_optimizer_state(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    role: str,
    expected_schema: Mapping[str, LoraTensorSpec],
    expected_completed_updates: int,
    collectives: Stage2CollectiveOps | None = None,
    get_optimizer_state_dict_fn: Callable[..., Mapping[str, Any]] | None = None,
    options_factory: Callable[..., Any] | None = None,
) -> Mapping[str, Any] | None:
    """Collectively gather one role's optimizer-only state to rank 0."""

    ops = _collectives(collectives)
    rank = int(ops.get_rank())
    default_get, _, default_options = _dcp_optimizer_apis()
    get_fn = get_optimizer_state_dict_fn or default_get
    make_options = options_factory or default_options
    names = _consensus_call(
        f"{role} optimizer identity audit",
        lambda: audit_stage2_lora_optimizer(
            module,
            optimizer,
            role=role,
            expected_schema=expected_schema,
            require_initialized_moments=True,
            expected_completed_updates=expected_completed_updates,
        ),
        ops,
    )
    ops.barrier()
    full_state = _consensus_call(
        f"{role} DCP optimizer gather",
        lambda: get_fn(
            module,
            optimizer,
            options=make_options(full_state_dict=True, cpu_offload=True),
        ),
        ops,
    )
    local_error: Exception | None = None
    if rank == 0:
        try:
            validate_stage2_full_optimizer_state(
                full_state,
                role=role,
                expected_parameter_names=names,
                expected_completed_updates=expected_completed_updates,
            )
        except Exception as exc:
            local_error = exc
    if not ops.consensus(local_error is None):
        raise RuntimeError(
            f"Stage-2 {role} rank-0 optimizer validation failed"
        ) from local_error
    ops.barrier()
    return full_state if rank == 0 else None


def restore_stage2_optimizer_state(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_state: Mapping[str, Any] | None,
    *,
    role: str,
    expected_schema: Mapping[str, LoraTensorSpec],
    expected_completed_updates: int,
    collectives: Stage2CollectiveOps | None = None,
    set_optimizer_state_dict_fn: Callable[..., None] | None = None,
    options_factory: Callable[..., Any] | None = None,
) -> None:
    """Validate on rank 0, DCP-broadcast, install, then audit every local shard."""

    ops = _collectives(collectives)
    rank = int(ops.get_rank())
    _, default_set, default_options = _dcp_optimizer_apis()
    set_fn = set_optimizer_state_dict_fn or default_set
    make_options = options_factory or default_options
    names = _consensus_call(
        f"{role} optimizer pre-restore identity audit",
        lambda: audit_stage2_lora_optimizer(
            module,
            optimizer,
            role=role,
            expected_schema=expected_schema,
        ),
        ops,
    )
    local_error: Exception | None = None
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
        except Exception as exc:
            local_error = exc
    elif optimizer_state not in (None, {}):
        local_error = RuntimeError(
            f"nonzero rank must not materialize full {role} optimizer state"
        )
    if not ops.consensus(local_error is None):
        raise RuntimeError(
            f"Stage-2 {role} optimizer state failed validation"
        ) from local_error
    ops.barrier()
    options = make_options(
        full_state_dict=True,
        cpu_offload=True,
        broadcast_from_rank0=True,
    )
    _consensus_call(
        f"{role} DCP optimizer restore",
        lambda: set_fn(
            module,
            optimizer,
            optimizer_state if rank == 0 else {},
            options=options,
        ),
        ops,
    )
    _consensus_call(
        f"{role} optimizer post-restore audit",
        lambda: audit_stage2_lora_optimizer(
            module,
            optimizer,
            role=role,
            expected_schema=expected_schema,
            require_initialized_moments=True,
            expected_completed_updates=expected_completed_updates,
        ),
        ops,
    )
    ops.barrier()


def _chunk_size_and_offset(
    dim_size: int, chunks: int, coordinate: int
) -> tuple[int, int]:
    chunk_size = (dim_size + chunks - 1) // chunks
    offset = min(coordinate * chunk_size, dim_size)
    return min(chunk_size, dim_size - offset), offset


def consolidate_stage2_lora_shards(
    shards_by_rank: Sequence[Mapping[str, LocalLoraShard]],
    *,
    expected_schema: Mapping[str, LoraTensorSpec],
) -> OrderedDict[str, torch.Tensor]:
    """Reconstruct canonical LoRA tensors from the 1D world8 shard mesh."""

    if len(shards_by_rank) != STAGE2_WORLD_SIZE:
        raise ValueError("Stage-2 selective LoRA gather requires exactly 8 ranks")
    expected_keys = set(expected_schema)
    if not expected_keys:
        raise ValueError("Stage-2 selective LoRA schema is empty")
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    ranks = tuple(range(STAGE2_WORLD_SIZE))
    for index, mapping in enumerate(shards_by_rank):
        if set(mapping) != expected_keys:
            raise ValueError(f"Stage-2 rank {index} LoRA shard keys drifted")
    for key in sorted(expected_schema):
        spec = expected_schema[key]
        records = sorted(
            (mapping[key] for mapping in shards_by_rank),
            key=lambda record: record.shard_rank,
        )
        if [record.shard_rank for record in records] != list(ranks):
            raise ValueError(f"Stage-2 {key} does not have one shard per rank")
        global_numel = int(torch.Size(spec.global_shape).numel())
        output = torch.empty(global_numel, dtype=spec.dtype, device="cpu")
        covered = torch.zeros(global_numel, dtype=torch.bool)
        for rank, record in enumerate(records):
            exact_topology = (
                record.is_dtensor
                and record.global_shape == tuple(spec.global_shape)
                and record.mesh_shape == (8,)
                and record.mesh_dim_names == ("shard",)
                and record.placements == ("shard:0",)
                and record.mesh_coordinate == (rank,)
                and record.mesh_ranks == (ranks,)
                and record.shard_world_size == 8
                and record.shard_group_ranks == ranks
                and record.replica_group_ranks == (rank,)
                and record.authoritative_shard_group_ranks == ranks
                and record.shard_dim == 0
                and record.global_rank == rank
            )
            if not exact_topology:
                raise ValueError(
                    f"Stage-2 {key} is not one-dimensional world8 Shard(0)"
                )
            rows, offset = _chunk_size_and_offset(
                int(spec.global_shape[0]), STAGE2_WORLD_SIZE, rank
            )
            expected_shape = (rows, *tuple(spec.global_shape[1:]))
            local = record.tensor.detach().to(device="cpu").contiguous()
            if (
                tuple(local.shape) != expected_shape
                or record.local_shape != expected_shape
                or record.shard_offset != offset
                or local.dtype != spec.dtype
            ):
                raise ValueError(f"Stage-2 {key} local shard geometry/dtype drifted")
            if local.numel() and not bool(torch.isfinite(local).all().item()):
                raise ValueError(f"Stage-2 {key} local shard is non-finite")
            start = offset * int(torch.Size(spec.global_shape[1:]).numel())
            end = start + local.numel()
            if record.intra_param_start != start or record.intra_param_end != end:
                raise ValueError(f"Stage-2 {key} flattened shard offset drifted")
            if local.numel() and bool(covered[start:end].any().item()):
                raise ValueError(f"Stage-2 {key} shards overlap")
            output[start:end].copy_(local.reshape(-1))
            covered[start:end] = True
        if not bool(covered.all().item()):
            raise ValueError(f"Stage-2 {key} shards leave values uncovered")
        result[key] = output.reshape(spec.global_shape)
    return result


def gather_stage2_lora_state_dict(
    module: torch.nn.Module,
    *,
    expected_schema: Mapping[str, LoraTensorSpec],
    role: str,
    shard_group: Any = None,
    collectives: Stage2CollectiveOps | None = None,
    local_shards_fn: Callable[..., Mapping[str, LocalLoraShard]] | None = None,
    gather_object_fn: Callable[..., None] | None = None,
) -> OrderedDict[str, torch.Tensor] | None:
    """Gather only local LoRA DTensor shards; frozen 5B weights are untouched."""

    if role not in STAGE2_ROLE_NAMES:
        raise ValueError(f"invalid Stage-2 LoRA role {role!r}")
    import torch.distributed as dist
    from utils.lora_utils import get_lora_sharded_state_dict

    ops = _collectives(collectives)
    rank = int(ops.get_rank())
    local_fn = local_shards_fn or get_lora_sharded_state_dict
    gather_fn = gather_object_fn or dist.gather_object
    local = _consensus_call(
        f"{role} selective local LoRA copy",
        lambda: local_fn(
            module,
            cpu_offload=True,
            expected_schema=expected_schema,
            expected_adapter_tensors=len(expected_schema),
            expected_dtype=torch.float32,
            expected_mesh_shape=(8,),
            expected_mesh_ranks=(tuple(range(8)),),
            expected_mesh_dim_names=("shard",),
        ),
        ops,
    )
    gathered: list[Any] | None = [None] * 8 if rank == 0 else None
    _consensus_call(
        f"{role} directed selective LoRA gather",
        lambda: gather_fn(
            local,
            object_gather_list=gathered,
            dst=0,
            group=shard_group,
        ),
        ops,
    )
    local_error: Exception | None = None
    canonical = None
    if rank == 0:
        try:
            assert gathered is not None
            canonical = consolidate_stage2_lora_shards(
                gathered, expected_schema=expected_schema
            )
        except Exception as exc:
            local_error = exc
    if not ops.consensus(local_error is None):
        raise RuntimeError(f"Stage-2 {role} LoRA consolidation failed") from local_error
    ops.barrier()
    return canonical if rank == 0 else None


def validate_stage2_adapter_state(
    state: Mapping[str, torch.Tensor],
    *,
    expected_schema: Mapping[str, LoraTensorSpec],
    role: str,
) -> OrderedDict[str, torch.Tensor]:
    if role not in STAGE2_ROLE_NAMES and role != "generator_ema":
        raise ValueError(f"invalid Stage-2 adapter role {role!r}")
    if not isinstance(state, Mapping) or set(state) != set(expected_schema):
        actual = set(state) if isinstance(state, Mapping) else set()
        raise ValueError(
            f"Stage-2 {role} adapter key mismatch: "
            f"missing={sorted(set(expected_schema) - actual)}, "
            f"extra={sorted(actual - set(expected_schema))}"
        )
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in sorted(expected_schema):
        tensor = state[key]
        spec = expected_schema[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Stage-2 {role} adapter {key} is not a Tensor")
        if tuple(tensor.shape) != tuple(spec.global_shape):
            raise ValueError(f"Stage-2 {role} adapter shape drift for {key}")
        if tensor.dtype != torch.float32:
            raise TypeError(f"Stage-2 {role} adapter {key} must be FP32")
        cpu = tensor.detach().to(device="cpu").contiguous()
        if cpu.numel() and not bool(torch.isfinite(cpu).all().item()):
            raise ValueError(f"Stage-2 {role} adapter {key} is non-finite")
        result[key] = cpu
    return result


def checkpoint_directory(
    root: str | os.PathLike[str], completed_generator_updates: int
) -> Path:
    completed = _plain_int(
        completed_generator_updates, "completed_generator_updates", minimum=1
    )
    if completed > 999_999:
        raise ValueError("completed_generator_updates exceeds checkpoint name width")
    return Path(root) / f"checkpoint_stage2_g{completed:06d}"


def checkpoint_generator_updates(path: str | os.PathLike[str]) -> int:
    match = STAGE2_CHECKPOINT_PATTERN.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"not a Stage-2 checkpoint directory name: {path}")
    return int(match.group(1))


def _validate_topology(value: Mapping[str, Any]) -> dict[str, Any]:
    topology = _exact_mapping(
        value,
        label="Stage-2 checkpoint topology",
        expected_keys={
            "world_size",
            "nodes",
            "fsdp_backend",
            "sharding_strategy",
            "mesh_shape",
            "mesh_dim_names",
            "rank_layout",
            "microbatch_size_per_device",
            "gradient_accumulation_steps",
            "global_batch_size",
        },
    )
    expected_fixed = {
        "world_size": 8,
        "nodes": 1,
        "fsdp_backend": "fsdp2",
        "sharding_strategy": "FULL_SHARD",
        "mesh_shape": [8],
        "mesh_dim_names": ["shard"],
        "rank_layout": list(range(8)),
        "global_batch_size": 64,
    }
    for key, expected in expected_fixed.items():
        if topology[key] != expected or type(topology[key]) is not type(expected):
            raise RuntimeError(
                f"Stage-2 checkpoint topology {key} mismatch: "
                f"{topology[key]!r} != {expected!r}"
            )
    profile = (
        _plain_int(
            topology["microbatch_size_per_device"],
            "topology.microbatch_size_per_device",
            minimum=1,
        ),
        _plain_int(
            topology["gradient_accumulation_steps"],
            "topology.gradient_accumulation_steps",
            minimum=1,
        ),
    )
    if profile not in {(2, 4), (1, 8)}:
        raise RuntimeError(f"unsupported Stage-2 checkpoint batch profile {profile}")
    return topology


def _validate_ema_state(
    state: Mapping[str, Any],
    *,
    rank: int,
    completed_g: int,
    expected_parameter_names: set[str] | None = None,
) -> Mapping[str, Any]:
    expected_initialized = completed_g >= STAGE2_EMA_START_GENERATOR_UPDATE
    if (
        not isinstance(state, Mapping)
        or type(state.get("initialized")) is not bool
        or state["initialized"] is not expected_initialized
    ):
        raise RuntimeError(
            "Stage-2 EMA initialization state disagrees with the completed G clock"
        )
    from utils.distributed import validate_trainable_sharded_ema_state_dict

    validate_trainable_sharded_ema_state_dict(
        state,
        expected_rank=rank,
        expected_world_size=STAGE2_WORLD_SIZE,
        expected_decay=0.99,
        expected_start_step=STAGE2_EMA_START_GENERATOR_UPDATE,
        expected_completed_step=completed_g,
        expected_initialized=expected_initialized,
        expected_topology={
            "rank_layout": tuple(range(STAGE2_WORLD_SIZE)),
            "mesh_dim_names": ("shard",),
        },
        expected_parameter_names=expected_parameter_names,
    )
    for name, metadata in state["shard_metadata"].items():
        if (
            metadata["kind"] != "dtensor"
            or metadata["mesh_device_type"] != "cuda"
            or tuple(metadata["mesh_shape"]) != (STAGE2_WORLD_SIZE,)
            or tuple(tuple(row) for row in metadata["rank_layout"])
            != tuple((mesh_rank,) for mesh_rank in range(STAGE2_WORLD_SIZE))
            or tuple(metadata["coordinate"]) != (rank,)
            or tuple(metadata["placements"]) != ("S(0)",)
        ):
            raise RuntimeError(
                f"Stage-2 EMA shard topology is not 1D FULL_SHARD for {name!r}"
            )
    return state


def _atomic_save_safetensors(
    path: Path,
    state: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, str],
) -> None:
    from safetensors.torch import save_file

    with atomic_output_path(path, suffix=".safetensors.tmp") as temporary:
        save_file(dict(state), str(temporary), metadata=dict(metadata))


def _torch_load_cpu(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_safetensors(path: Path) -> OrderedDict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return OrderedDict(sorted(load_file(str(path), device="cpu").items()))


def _load_authenticated_safetensors_snapshot(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> OrderedDict[str, torch.Tensor]:
    """Authenticate and deserialize one immutable in-memory byte snapshot."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"canonical Generator EMA is not a regular file: {path}")
    expected_size = _plain_int(expected_size, "canonical Generator EMA size")
    expected_sha256 = _sha256(
        expected_sha256,
        "canonical Generator EMA SHA-256",
    )
    snapshot = path.read_bytes()
    actual_sha256 = hashlib.sha256(snapshot).hexdigest()
    if len(snapshot) != expected_size or actual_sha256 != expected_sha256:
        raise RuntimeError(
            "canonical Generator EMA snapshot hash/size mismatch: "
            f"expected=({expected_size},{expected_sha256}), "
            f"actual=({len(snapshot)},{actual_sha256})"
        )

    # Deserialize the exact bytes authenticated above. Reopening ``path`` here
    # would reintroduce a hash-check/load TOCTOU window.
    from safetensors.torch import load

    return OrderedDict(sorted(load(snapshot).items()))


def _required_payload_names(ema_initialized: bool) -> set[str]:
    result = {
        "generator_raw.safetensors",
        "fake_score_raw.safetensors",
        "metrics_lineage.jsonl",
        "optimizer_generator.pt",
        "optimizer_fake_score.pt",
        "trainer_state.pt",
        "resolved_config.json",
        "provenance.json",
        *(f"ema_state_rank{rank:05d}.pt" for rank in range(8)),
        *(f"rng_state_rank{rank:05d}.pt" for rank in range(8)),
    }
    if ema_initialized:
        result.add("generator_ema.safetensors")
    return result


def _validate_metrics_lineage_snapshot(
    path: Path,
    *,
    trainer_state: Mapping[str, Any],
) -> bytes:
    """Validate the exact pre-checkpoint-event JSONL prefix bound to a save."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Stage-2 metrics lineage is not a regular file: {path}")
    snapshot = path.read_bytes()
    if not snapshot or not snapshot.endswith(b"\n"):
        raise RuntimeError(
            "Stage-2 metrics lineage must be a non-empty, newline-terminated "
            "JSONL snapshot"
        )
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(snapshot.splitlines(), start=1):
        if not raw_line.strip():
            raise RuntimeError(
                f"Stage-2 metrics lineage contains a blank line at {line_number}"
            )
        try:
            record = json.loads(raw_line.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Stage-2 metrics lineage contains invalid JSON at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(
                f"Stage-2 metrics lineage line {line_number} is not an object"
            )
        records.append(record)

    from utils.jsonl_logger import (
        latest_run_id,
        lineage_run_ids,
        next_attempt_index,
        records_for_latest_lineage,
    )

    lineage = trainer_state["jsonl_lineage"]
    run_id = lineage["run_id"]
    try:
        owners = lineage_run_ids(records, run_id=run_id)
        train_records = records_for_latest_lineage(
            records,
            record_type="train_step",
            run_id=run_id,
            step_key="logical_substep_id",
        )
        snapshot_next_attempt = next_attempt_index(records, run_id=run_id)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Stage-2 metrics lineage graph is invalid") from exc
    if latest_run_id(records) != run_id:
        raise RuntimeError(
            "Stage-2 metrics lineage checkpoint owner is not the latest run"
        )
    run_start_counts = {
        owner: sum(
            record.get("record_type") == "run_start" and record.get("run_id") == owner
            for record in records
        )
        for owner in owners
    }
    if any(count != 1 for count in run_start_counts.values()):
        raise RuntimeError(
            "Stage-2 metrics lineage requires exactly one run_start per ancestor"
        )
    owner_set = set(owners)
    attempts = [
        record["attempt_index"]
        for record in records
        if "attempt_index" in record and record.get("run_id") in owner_set
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in attempts
    ) or attempts != sorted(set(attempts)):
        raise RuntimeError(
            "Stage-2 metrics lineage attempt indices must be unique and increasing"
        )
    expected_logical = int(trainer_state["next_logical_substep_id"])
    logical_ids = [int(record["logical_substep_id"]) for record in train_records]
    if logical_ids != list(range(expected_logical)):
        raise RuntimeError(
            "Stage-2 metrics lineage does not contain the exact committed "
            f"logical prefix 0..{expected_logical - 1}"
        )
    # The checkpoint is published before its checkpoint_event. The stored
    # cursor reserves exactly that one future attempt index.
    if int(lineage["next_attempt_index"]) != snapshot_next_attempt + 1:
        raise RuntimeError(
            "Stage-2 metrics lineage attempt cursor does not reserve exactly "
            "one checkpoint_event"
        )
    return snapshot


def _file_entries(directory: Path, names: Iterable[str]) -> list[dict[str, Any]]:
    entries = []
    for name in sorted(set(names)):
        if Path(name).name != name or not name:
            raise ValueError(f"checkpoint manifest file name is not local: {name!r}")
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"checkpoint payload is missing/not regular: {path}")
        entries.append(
            {"name": name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return entries


def _build_manifest(
    directory: Path,
    *,
    trainer_state: Mapping[str, Any],
    topology: Mapping[str, Any],
    ema_initialized: bool,
    payload_names: Iterable[str],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    completed_g = int(trainer_state["completed_generator_updates"])
    payload = {
        "schema": STAGE2_CHECKPOINT_SCHEMA,
        "schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
        "completed_generator_updates": completed_g,
        "completed_fake_updates": int(trainer_state["completed_fake_updates"]),
        "completed_cycles": int(trainer_state["completed_cycles"]),
        "next_substep": "F1",
        "topology": dict(topology),
        "config": {
            "contract_hash": trainer_state["contract_hash"],
            "launch_hash": trainer_state["launch_hash"],
            "cache_audit_launch_hash": trainer_state["cache_audit_launch_hash"],
        },
        "ema": {
            "initialized": ema_initialized,
            "canonical_artifact": (
                "generator_ema.safetensors" if ema_initialized else None
            ),
            "last_completed_generator_update": completed_g,
        },
        "provenance_sha256": canonical_json_sha256(dict(provenance)),
        "files": _file_entries(directory, payload_names),
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def _validate_rank_state_maps(
    *,
    rank_ema_states: Mapping[int, Mapping[str, Any]],
    rank_rng_states: Mapping[int, Mapping[str, Any]],
    completed_g: int,
    expected_ema_parameter_names: set[str],
    trainer_state: Mapping[str, Any],
) -> None:
    expected_ranks = set(range(STAGE2_WORLD_SIZE))
    if set(rank_ema_states) != expected_ranks or set(rank_rng_states) != expected_ranks:
        raise RuntimeError("Stage-2 checkpoint requires EMA/RNG state for all 8 ranks")
    for rank in range(STAGE2_WORLD_SIZE):
        _validate_ema_state(
            rank_ema_states[rank],
            rank=rank,
            completed_g=completed_g,
            expected_parameter_names=expected_ema_parameter_names,
        )
        validate_stage2_rng_state(
            rank_rng_states[rank],
            expected_rank=rank,
            expected_dedicated_names=set(STAGE2_DEDICATED_RNG_NAMES),
        )
    reference_global_shapes = {
        name: tuple(shape)
        for name, shape in rank_ema_states[0]["global_shapes"].items()
    }
    for rank in range(1, STAGE2_WORLD_SIZE):
        candidate = {
            name: tuple(shape)
            for name, shape in rank_ema_states[rank]["global_shapes"].items()
        }
        if candidate != reference_global_shapes:
            raise RuntimeError("Stage-2 EMA global shapes differ across ranks")
    for name, global_shape in reference_global_shapes.items():
        local_numel = sum(
            math.prod(tuple(rank_ema_states[rank]["local_shapes"][name]))
            for rank in range(STAGE2_WORLD_SIZE)
        )
        if local_numel != math.prod(global_shape):
            raise RuntimeError(
                f"Stage-2 EMA local shards do not partition {name!r}: "
                f"local_numel={local_numel}, global_numel={math.prod(global_shape)}"
            )
    rank0_dedicated = rank_rng_states[0]["dedicated"]
    loader_states = trainer_state["dataloader_generator_states"]
    for role in STAGE2_ROLE_NAMES:
        checkpoint_loader = rank0_dedicated[f"{role}_loader"]["state"]
        if not torch.equal(checkpoint_loader, loader_states[role]):
            raise RuntimeError(f"Stage-2 rank0 {role} DataLoader RNG copies disagree")


def _validate_prepared_payloads(
    *,
    trainer_state: Mapping[str, Any],
    generator_raw: Mapping[str, torch.Tensor],
    fake_score_raw: Mapping[str, torch.Tensor],
    generator_ema: Mapping[str, torch.Tensor] | None,
    generator_schema: Mapping[str, LoraTensorSpec],
    fake_score_schema: Mapping[str, LoraTensorSpec],
    generator_optimizer_state: Mapping[str, Any],
    fake_score_optimizer_state: Mapping[str, Any],
    rank_ema_states: Mapping[int, Mapping[str, Any]],
    rank_rng_states: Mapping[int, Mapping[str, Any]],
    topology: Mapping[str, Any],
) -> tuple[
    OrderedDict[str, torch.Tensor],
    OrderedDict[str, torch.Tensor],
    OrderedDict[str, torch.Tensor] | None,
    dict[str, Any],
]:
    validate_stage2_trainer_state(trainer_state)
    completed_g = int(trainer_state["completed_generator_updates"])
    completed_f = int(trainer_state["completed_fake_updates"])
    raw_g = validate_stage2_adapter_state(
        generator_raw, expected_schema=generator_schema, role="generator"
    )
    raw_f = validate_stage2_adapter_state(
        fake_score_raw, expected_schema=fake_score_schema, role="fake_score"
    )
    expected_ema = completed_g >= STAGE2_EMA_START_GENERATOR_UPDATE
    if expected_ema != (generator_ema is not None):
        raise RuntimeError(
            "Generator EMA canonical artifact presence disagrees with G40 initialization"
        )
    ema = (
        validate_stage2_adapter_state(
            generator_ema,
            expected_schema=generator_schema,
            role="generator_ema",
        )
        if generator_ema is not None
        else None
    )
    generator_names = _optimizer_names_for_schema(
        generator_optimizer_state, generator_schema, role="generator"
    )
    fake_names = _optimizer_names_for_schema(
        fake_score_optimizer_state, fake_score_schema, role="fake_score"
    )
    validate_stage2_full_optimizer_state(
        generator_optimizer_state,
        role="generator",
        expected_parameter_names=generator_names,
        expected_completed_updates=completed_g,
    )
    validate_stage2_full_optimizer_state(
        fake_score_optimizer_state,
        role="fake_score",
        expected_parameter_names=fake_names,
        expected_completed_updates=completed_f,
    )
    _validate_rank_state_maps(
        rank_ema_states=rank_ema_states,
        rank_rng_states=rank_rng_states,
        completed_g=completed_g,
        expected_ema_parameter_names={
            spec.raw_parameter_name for spec in generator_schema.values()
        },
        trainer_state=trainer_state,
    )
    return raw_g, raw_f, ema, _validate_topology(topology)


def _optimizer_names_for_schema(
    optimizer_state: Mapping[str, Any],
    schema: Mapping[str, LoraTensorSpec],
    *,
    role: str,
) -> tuple[str, ...]:
    names = _optimizer_names(optimizer_state)
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


def save_stage2_checkpoint_from_payloads(
    root: str | os.PathLike[str],
    *,
    trainer_state: Mapping[str, Any],
    metrics_lineage_snapshot: bytes | None = None,
    generator_raw: Mapping[str, torch.Tensor],
    fake_score_raw: Mapping[str, torch.Tensor],
    generator_ema: Mapping[str, torch.Tensor] | None,
    generator_schema: Mapping[str, LoraTensorSpec],
    fake_score_schema: Mapping[str, LoraTensorSpec],
    generator_optimizer_state: Mapping[str, Any],
    fake_score_optimizer_state: Mapping[str, Any],
    rank_ema_states: Mapping[int, Mapping[str, Any]],
    rank_rng_states: Mapping[int, Mapping[str, Any]],
    resolved_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    topology: Mapping[str, Any],
    io_include_cuda: bool = True,
    failure_injector: Callable[[str], None] | None = None,
) -> Path:
    """Publish already-gathered payloads with directory-level atomicity.

    This rank-0 core is also useful for deterministic CPU tests.  Distributed
    trainers should call :func:`save_stage2_checkpoint`, which performs the
    selective adapter and DCP optimizer gathers first.
    """

    io_rng = capture_rng_state(include_cuda=io_include_cuda)
    temporary: Path | None = None
    renamed = False
    quarantined_uncommitted: Path | None = None
    try:
        raw_g, raw_f, ema, audited_topology = _validate_prepared_payloads(
            trainer_state=trainer_state,
            generator_raw=generator_raw,
            fake_score_raw=fake_score_raw,
            generator_ema=generator_ema,
            generator_schema=generator_schema,
            fake_score_schema=fake_score_schema,
            generator_optimizer_state=generator_optimizer_state,
            fake_score_optimizer_state=fake_score_optimizer_state,
            rank_ema_states=rank_ema_states,
            rank_rng_states=rank_rng_states,
            topology=topology,
        )
        if not isinstance(resolved_config, Mapping):
            raise TypeError("resolved_config must be a mapping")
        provenance = validate_stage2_provenance(
            provenance,
            add_code_version=True,
        )
        if not isinstance(metrics_lineage_snapshot, bytes):
            raise TypeError("metrics_lineage_snapshot must be exact bytes")
        # Encode now: invalid/non-finite JSON must fail before a directory exists.
        canonical_json_sha256(dict(resolved_config))
        canonical_json_sha256(dict(provenance))
        completed_g = int(trainer_state["completed_generator_updates"])
        root_path = Path(root).expanduser()
        if root_path.is_symlink() or (root_path.exists() and not root_path.is_dir()):
            raise RuntimeError(
                f"Stage-2 checkpoint root is not a regular directory: {root_path}"
            )
        root_path.mkdir(parents=True, exist_ok=True)
        if root_path.is_symlink():
            raise RuntimeError(f"Stage-2 checkpoint root became a symlink: {root_path}")
        destination = checkpoint_directory(root_path.resolve(), completed_g)
        if destination.exists() or destination.is_symlink():
            marker = destination / "_SUCCESS"
            if destination.is_symlink() or not destination.is_dir():
                raise RuntimeError(
                    f"Stage-2 checkpoint destination is not a regular directory: {destination}"
                )
            if marker.exists() or marker.is_symlink():
                # Complete (or marker-corrupt) checkpoints are immutable.
                raise FileExistsError(destination)
            quarantine_root = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.uncommitted.", dir=destination.parent
                )
            )
            quarantined_uncommitted = quarantine_root / "payload"
            os.replace(destination, quarantined_uncommitted)
            _fsync_directory(destination.parent)
        parent = destination.parent
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
        if failure_injector is not None:
            failure_injector("after_temp")

        _atomic_save_safetensors(
            temporary / "generator_raw.safetensors",
            raw_g,
            metadata={
                "role": "generator",
                "kind": "raw",
                "completed_g": str(completed_g),
            },
        )
        _atomic_save_safetensors(
            temporary / "fake_score_raw.safetensors",
            raw_f,
            metadata={
                "role": "fake_score",
                "kind": "raw",
                "completed_f": str(5 * completed_g),
            },
        )
        if ema is not None:
            _atomic_save_safetensors(
                temporary / "generator_ema.safetensors",
                ema,
                metadata={
                    "role": "generator",
                    "kind": "ema",
                    "completed_g": str(completed_g),
                },
            )
        atomic_torch_save(
            temporary / "optimizer_generator.pt", dict(generator_optimizer_state)
        )
        atomic_torch_save(
            temporary / "optimizer_fake_score.pt", dict(fake_score_optimizer_state)
        )
        atomic_torch_save(temporary / "trainer_state.pt", dict(trainer_state))
        atomic_write_bytes(
            temporary / "metrics_lineage.jsonl",
            metrics_lineage_snapshot,
        )
        for rank in range(STAGE2_WORLD_SIZE):
            atomic_torch_save(
                temporary / f"ema_state_rank{rank:05d}.pt",
                dict(rank_ema_states[rank]),
            )
            atomic_torch_save(
                temporary / f"rng_state_rank{rank:05d}.pt",
                dict(rank_rng_states[rank]),
            )
        atomic_write_json(temporary / "resolved_config.json", dict(resolved_config))
        atomic_write_json(temporary / "provenance.json", dict(provenance))
        _validate_metrics_lineage_snapshot(
            temporary / "metrics_lineage.jsonl",
            trainer_state=trainer_state,
        )
        if failure_injector is not None:
            failure_injector("after_files")

        ema_initialized = ema is not None
        payload_names = _required_payload_names(ema_initialized)
        manifest = _build_manifest(
            temporary,
            trainer_state=trainer_state,
            topology=audited_topology,
            ema_initialized=ema_initialized,
            payload_names=payload_names,
            provenance=provenance,
        )
        atomic_write_json(temporary / "checkpoint_manifest.json", manifest)
        if failure_injector is not None:
            failure_injector("after_manifest")
        _validate_manifest_and_files(
            temporary,
            expected_completed_g=completed_g,
            require_success_marker=False,
        )
        _fsync_directory(temporary)
        if failure_injector is not None:
            failure_injector("before_rename")
        os.replace(temporary, destination)
        renamed = True
        _fsync_directory(parent)
        if failure_injector is not None:
            failure_injector("before_success_marker")
        atomic_write_bytes(destination / "_SUCCESS", b"")
        _fsync_directory(destination)
        _fsync_directory(parent)
        if failure_injector is not None:
            failure_injector("after_success_marker")
        validate_stage2_checkpoint(destination)
        if quarantined_uncommitted is not None:
            shutil.rmtree(quarantined_uncommitted.parent)
            _fsync_directory(parent)
        return destination
    finally:
        if temporary is not None and not renamed and temporary.exists():
            shutil.rmtree(temporary)
        # Checkpoint serialization and failure injection are transparent to the
        # next training random draw.
        restore_rng_state(io_rng, require_cuda_topology=io_include_cuda)


_MANIFEST_KEYS = {
    "schema",
    "schema_version",
    "completed_generator_updates",
    "completed_fake_updates",
    "completed_cycles",
    "next_substep",
    "topology",
    "config",
    "ema",
    "provenance_sha256",
    "files",
    "manifest_sha256",
}


def _read_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"checkpoint manifest is missing/not regular: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    manifest = _exact_mapping(
        value, label="Stage-2 checkpoint manifest", expected_keys=_MANIFEST_KEYS
    )
    expected = manifest["manifest_sha256"]
    body = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    actual = canonical_json_sha256(body)
    if expected != actual:
        raise RuntimeError(
            f"Stage-2 checkpoint manifest self hash mismatch: {expected} != {actual}"
        )
    return manifest


def _validate_manifest_and_files(
    directory: Path,
    *,
    expected_completed_g: int,
    require_success_marker: bool,
) -> dict[str, Any]:
    manifest = _read_manifest(directory / "checkpoint_manifest.json")
    if (
        manifest["schema"] != STAGE2_CHECKPOINT_SCHEMA
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != STAGE2_CHECKPOINT_SCHEMA_VERSION
    ):
        raise RuntimeError("unsupported Stage-2 checkpoint manifest schema")
    completed_g = _plain_int(
        manifest["completed_generator_updates"],
        "manifest.completed_generator_updates",
        minimum=1,
    )
    if completed_g != expected_completed_g:
        raise RuntimeError("checkpoint directory and manifest G clocks disagree")
    completed_f = _plain_int(
        manifest["completed_fake_updates"],
        "manifest.completed_fake_updates",
        minimum=1,
    )
    completed_cycles = _plain_int(
        manifest["completed_cycles"],
        "manifest.completed_cycles",
        minimum=1,
    )
    if completed_f != 5 * completed_g or completed_cycles != completed_g:
        raise RuntimeError("checkpoint manifest F/G/cycle clocks disagree")
    if manifest["next_substep"] != "F1":
        raise RuntimeError("checkpoint manifest next_substep must be F1")
    _validate_topology(manifest["topology"])
    config = _exact_mapping(
        manifest["config"],
        label="checkpoint config hashes",
        expected_keys={
            "contract_hash",
            "launch_hash",
            "cache_audit_launch_hash",
        },
    )
    _sha256(config["contract_hash"], "checkpoint config contract_hash")
    _sha256(config["launch_hash"], "checkpoint config launch_hash")
    _sha256(
        config["cache_audit_launch_hash"],
        "checkpoint config cache_audit_launch_hash",
    )
    _sha256(manifest["provenance_sha256"], "checkpoint provenance_sha256")
    ema = _exact_mapping(
        manifest["ema"],
        label="checkpoint EMA manifest",
        expected_keys={
            "initialized",
            "canonical_artifact",
            "last_completed_generator_update",
        },
    )
    expected_initialized = completed_g >= STAGE2_EMA_START_GENERATOR_UPDATE
    if (
        type(ema["initialized"]) is not bool
        or ema["initialized"] != expected_initialized
        or ema["canonical_artifact"]
        != ("generator_ema.safetensors" if expected_initialized else None)
        or _plain_int(
            ema["last_completed_generator_update"],
            "manifest.ema.last_completed_generator_update",
            minimum=1,
        )
        != completed_g
    ):
        raise RuntimeError("checkpoint EMA manifest disagrees with completed G")
    entries = manifest["files"]
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("checkpoint manifest files must be a sequence")
    expected_names = _required_payload_names(expected_initialized)
    names: list[str] = []
    for entry in entries:
        item = _exact_mapping(
            entry,
            label="checkpoint file entry",
            expected_keys={"name", "size", "sha256"},
        )
        name = item["name"]
        if not isinstance(name, str) or Path(name).name != name or not name:
            raise RuntimeError(
                f"checkpoint manifest contains unsafe file name {name!r}"
            )
        _plain_int(item["size"], f"checkpoint file {name} size")
        _sha256(item["sha256"], f"checkpoint file {name} sha256")
        names.append(name)
    if len(names) != len(set(names)) or set(names) != expected_names:
        raise RuntimeError(
            f"checkpoint manifest file set mismatch: expected={sorted(expected_names)}, "
            f"actual={sorted(names)}"
        )
    marker_names = {"checkpoint_manifest.json"}
    if require_success_marker:
        marker_names.add("_SUCCESS")
    actual_names: set[str] = set()
    for child in directory.iterdir():
        if child.is_symlink() or not child.is_file():
            raise RuntimeError(
                f"checkpoint file set contains non-regular entry: {child}"
            )
        actual_names.add(child.name)
    expected_actual = expected_names | marker_names
    if actual_names != expected_actual:
        raise RuntimeError(
            f"checkpoint directory file set mismatch: expected={sorted(expected_actual)}, "
            f"actual={sorted(actual_names)}"
        )
    for item in entries:
        path = directory / item["name"]
        if path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"checkpoint file hash/size mismatch: {path}")
    if require_success_marker:
        marker = directory / "_SUCCESS"
        if marker.is_symlink() or not marker.is_file() or marker.stat().st_size != 0:
            raise RuntimeError(f"checkpoint success marker is invalid: {marker}")
    return manifest


def validate_stage2_checkpoint(
    directory: str | os.PathLike[str],
    *,
    expected_contract_hash: str | None = None,
    expected_topology: Mapping[str, Any] | None = None,
    expected_phase_b_mode: str | None = None,
    expected_generator_parameter_names: Sequence[str] | None = None,
    expected_fake_score_parameter_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Hash-scan and cross-check a committed checkpoint before loading tensors."""

    candidate = Path(directory).expanduser()
    # Check before ``resolve``: resolving first would silently turn a symlinked
    # checkpoint into its target and defeat the fail-closed directory contract.
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(
            f"Stage-2 checkpoint is not a regular directory: {candidate}"
        )
    path = candidate.resolve()
    completed_g = checkpoint_generator_updates(path)
    if not (path / "_SUCCESS").is_file():
        raise RuntimeError(f"incomplete uncommitted Stage-2 checkpoint: {path}")
    manifest = _validate_manifest_and_files(
        path, expected_completed_g=completed_g, require_success_marker=True
    )
    if expected_contract_hash is not None and manifest["config"][
        "contract_hash"
    ] != _sha256(expected_contract_hash, "expected_contract_hash"):
        raise RuntimeError("Stage-2 checkpoint contract hash mismatch")
    if expected_topology is not None and manifest["topology"] != _validate_topology(
        expected_topology
    ):
        raise RuntimeError("Stage-2 checkpoint topology differs from this launch")

    trainer = _torch_load_cpu(path / "trainer_state.pt")
    validate_stage2_trainer_state(
        trainer,
        expected_contract_hash=(
            expected_contract_hash or manifest["config"]["contract_hash"]
        ),
    )
    _validate_metrics_lineage_snapshot(
        path / "metrics_lineage.jsonl",
        trainer_state=trainer,
    )
    if expected_phase_b_mode is not None:
        if expected_phase_b_mode not in {"disabled", "dmd_dfd", "dmd_only"}:
            raise ValueError("expected_phase_b_mode is invalid")
        checkpoint_mode = trainer["phase_state"]["phase_b_mode"]
        if (
            completed_g > STAGE2_PHASE_A_GENERATOR_UPDATES
            and checkpoint_mode != expected_phase_b_mode
        ):
            raise RuntimeError(
                "Stage-2 post-A24 checkpoint belongs to a different Phase-B arm"
            )
    if (
        trainer["completed_generator_updates"] != completed_g
        or trainer["completed_fake_updates"] != manifest["completed_fake_updates"]
        or trainer["launch_hash"] != manifest["config"]["launch_hash"]
        or trainer["cache_audit_launch_hash"]
        != manifest["config"]["cache_audit_launch_hash"]
    ):
        raise RuntimeError("trainer state and checkpoint manifest clocks/hashes differ")
    provenance_path = path / "provenance.json"
    with provenance_path.open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    if (
        not isinstance(provenance, dict)
        or canonical_json_sha256(provenance) != manifest["provenance_sha256"]
    ):
        raise RuntimeError("checkpoint provenance self binding mismatch")
    validate_stage2_provenance(provenance)
    generator_state = _torch_load_cpu(path / "optimizer_generator.pt")
    fake_state = _torch_load_cpu(path / "optimizer_fake_score.pt")
    generator_names = tuple(
        expected_generator_parameter_names or _optimizer_names(generator_state)
    )
    fake_names = tuple(
        expected_fake_score_parameter_names or _optimizer_names(fake_state)
    )
    rank_ema_states = {
        rank: _torch_load_cpu(path / f"ema_state_rank{rank:05d}.pt")
        for rank in range(STAGE2_WORLD_SIZE)
    }
    rank_rng_states = {
        rank: _torch_load_cpu(path / f"rng_state_rank{rank:05d}.pt")
        for rank in range(STAGE2_WORLD_SIZE)
    }
    _validate_rank_state_maps(
        rank_ema_states=rank_ema_states,
        rank_rng_states=rank_rng_states,
        completed_g=completed_g,
        expected_ema_parameter_names=set(generator_names),
        trainer_state=trainer,
    )
    validate_stage2_full_optimizer_state(
        generator_state,
        role="generator",
        expected_parameter_names=generator_names,
        expected_completed_updates=completed_g,
    )
    validate_stage2_full_optimizer_state(
        fake_state,
        role="fake_score",
        expected_parameter_names=fake_names,
        expected_completed_updates=5 * completed_g,
    )
    return manifest


def _optimizer_names(state: Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(state, Mapping) or not isinstance(
        state.get("param_groups"), Sequence
    ):
        raise TypeError("checkpoint optimizer state has no param_groups")
    names: list[str] = []
    for group in state["param_groups"]:
        if not isinstance(group, Mapping) or not isinstance(
            group.get("params"), Sequence
        ):
            raise TypeError("checkpoint optimizer param group is invalid")
        names.extend(group["params"])
    return tuple(names)


@dataclass(frozen=True)
class Stage2CheckpointPayload:
    """CPU payload returned after every file hash has been validated."""

    directory: Path
    manifest: Mapping[str, Any]
    trainer_state: Mapping[str, Any]
    generator_raw: OrderedDict[str, torch.Tensor]
    fake_score_raw: OrderedDict[str, torch.Tensor]
    generator_ema: OrderedDict[str, torch.Tensor] | None
    generator_optimizer_state: Mapping[str, Any]
    fake_score_optimizer_state: Mapping[str, Any]
    rank_ema_states: Mapping[int, Mapping[str, Any]]
    rank_rng_states: Mapping[int, Mapping[str, Any]]
    resolved_config: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class Stage2GeneratorEMACheckpointPayload:
    """Inference-only view of a committed, initialized Generator EMA.

    The loader for this view verifies every manifest-listed byte, but it never
    deserializes optimizer, rank-local EMA, trainer, or RNG pickle payloads.
    This keeps the inference gate memory-bounded even for a full 8-rank H100
    checkpoint while retaining the same fail-closed publication envelope.
    """

    directory: Path
    manifest: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    provenance: Mapping[str, Any]
    generator_ema: OrderedDict[str, torch.Tensor]


@dataclass(frozen=True)
class Stage2DistributedCheckpointPayload:
    """Per-rank resume view produced by collective checkpoint loading.

    Raw G/F adapters are available on every rank for strict pre-FSDP loading.
    Full optimizer states and the canonical EMA artifact remain rank-0-only;
    DCP later broadcasts optimizer state.  EMA/RNG state is rank-local.
    """

    directory: Path
    manifest: Mapping[str, Any]
    trainer_state: Mapping[str, Any]
    generator_raw: OrderedDict[str, torch.Tensor]
    fake_score_raw: OrderedDict[str, torch.Tensor]
    generator_ema_rank0: OrderedDict[str, torch.Tensor] | None
    generator_optimizer_state_rank0: Mapping[str, Any] | None
    fake_score_optimizer_state_rank0: Mapping[str, Any] | None
    local_ema_state: Mapping[str, Any]
    local_rng_state: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    provenance: Mapping[str, Any]


def load_stage2_generator_ema_checkpoint(
    directory: str | os.PathLike[str],
    *,
    expected_contract_hash: str,
    expected_launch_hash: str,
    expected_topology: Mapping[str, Any] | None = None,
    expected_resolved_config: Mapping[str, Any] | None = None,
    expected_generator_schema: Mapping[str, LoraTensorSpec] | None = None,
) -> Stage2GeneratorEMACheckpointPayload:
    """Load only the canonical Generator EMA behind a complete checkpoint.

    ``expected_contract_hash`` and ``expected_launch_hash`` bind the caller's
    resolved experiment to the manifest. ``resolved_config.json`` and all
    other payload bytes are, in turn, bound to that same self-hashed manifest.
    Passing ``expected_resolved_config`` adds an exact canonical-JSON equality
    check. Optimizer, trainer, per-rank EMA, and RNG files are hash-scanned as
    ordinary bytes and are deliberately never passed to :func:`torch.load`.
    """

    candidate = Path(directory).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(
            f"Stage-2 checkpoint is not a regular directory: {candidate}"
        )
    path = candidate.resolve()
    completed_g = checkpoint_generator_updates(path)
    if not (path / "_SUCCESS").is_file():
        raise RuntimeError(f"incomplete uncommitted Stage-2 checkpoint: {path}")
    manifest = _validate_manifest_and_files(
        path,
        expected_completed_g=completed_g,
        require_success_marker=True,
    )
    expected_contract = _sha256(expected_contract_hash, "expected_contract_hash")
    expected_launch = _sha256(expected_launch_hash, "expected_launch_hash")
    if manifest["config"]["contract_hash"] != expected_contract:
        raise RuntimeError("Stage-2 checkpoint contract hash mismatch")
    if manifest["config"]["launch_hash"] != expected_launch:
        raise RuntimeError("Stage-2 checkpoint launch hash mismatch")
    if expected_topology is not None and manifest["topology"] != _validate_topology(
        expected_topology
    ):
        raise RuntimeError("Stage-2 checkpoint topology differs from this launch")
    if (
        completed_g < STAGE2_EMA_START_GENERATOR_UPDATE
        or manifest["ema"]["initialized"] is not True
        or manifest["ema"]["canonical_artifact"] != "generator_ema.safetensors"
    ):
        raise RuntimeError(
            "Stage-2 Generator EMA inference requires an initialized G>=40 checkpoint"
        )

    with (path / "resolved_config.json").open("r", encoding="utf-8") as handle:
        resolved_config = json.load(handle)
    if not isinstance(resolved_config, Mapping):
        raise TypeError("Stage-2 resolved_config.json must contain a mapping")
    # Reject non-canonical/non-finite JSON even though its raw bytes were
    # already authenticated by the manifest.
    canonical_json_sha256(resolved_config)
    if expected_resolved_config is not None and canonical_json_sha256(
        dict(resolved_config)
    ) != canonical_json_sha256(dict(expected_resolved_config)):
        raise RuntimeError("Stage-2 resolved config differs from this launch")

    with (path / "provenance.json").open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    if (
        not isinstance(provenance, Mapping)
        or canonical_json_sha256(dict(provenance)) != manifest["provenance_sha256"]
    ):
        raise RuntimeError("checkpoint provenance self binding mismatch")
    provenance = validate_stage2_provenance(provenance)

    generator_ema_entry = next(
        item
        for item in manifest["files"]
        if item["name"] == "generator_ema.safetensors"
    )
    generator_ema = _load_authenticated_safetensors_snapshot(
        path / "generator_ema.safetensors",
        expected_size=generator_ema_entry["size"],
        expected_sha256=generator_ema_entry["sha256"],
    )
    if expected_generator_schema is not None:
        generator_ema = validate_stage2_adapter_state(
            generator_ema,
            expected_schema=expected_generator_schema,
            role="generator_ema",
        )
    else:
        if not generator_ema:
            raise RuntimeError("Stage-2 canonical Generator EMA is empty")
        for name, tensor in generator_ema.items():
            if not isinstance(name, str) or not name or name != name.strip():
                raise RuntimeError("Stage-2 canonical Generator EMA has an invalid key")
            if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                raise TypeError(
                    f"Stage-2 Generator EMA {name} must be canonical CPU FP32"
                )
            if tensor.numel() and not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"Stage-2 Generator EMA {name} is non-finite")

    return Stage2GeneratorEMACheckpointPayload(
        directory=path,
        manifest=manifest,
        resolved_config=dict(resolved_config),
        provenance=provenance,
        generator_ema=generator_ema,
    )


def load_stage2_checkpoint(
    directory: str | os.PathLike[str],
    *,
    expected_contract_hash: str | None = None,
    expected_world_size: int = STAGE2_WORLD_SIZE,
    expected_topology: Mapping[str, Any] | None = None,
    expected_phase_b_mode: str | None = None,
    expected_generator_parameter_names: Sequence[str] | None = None,
    expected_fake_score_parameter_names: Sequence[str] | None = None,
) -> Stage2CheckpointPayload:
    """Load complete CPU state only after marker, manifest, and byte validation."""

    if expected_world_size != STAGE2_WORLD_SIZE:
        raise RuntimeError(
            f"Stage-2 resume requires expected_world_size=8, got {expected_world_size}"
        )
    candidate = Path(directory).expanduser()
    manifest = validate_stage2_checkpoint(
        candidate,
        expected_contract_hash=expected_contract_hash,
        expected_topology=expected_topology,
        expected_phase_b_mode=expected_phase_b_mode,
        expected_generator_parameter_names=expected_generator_parameter_names,
        expected_fake_score_parameter_names=expected_fake_score_parameter_names,
    )
    path = candidate.resolve()
    with (path / "resolved_config.json").open("r", encoding="utf-8") as handle:
        resolved_config = json.load(handle)
    with (path / "provenance.json").open("r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    return Stage2CheckpointPayload(
        directory=path,
        manifest=manifest,
        trainer_state=_torch_load_cpu(path / "trainer_state.pt"),
        generator_raw=_load_safetensors(path / "generator_raw.safetensors"),
        fake_score_raw=_load_safetensors(path / "fake_score_raw.safetensors"),
        generator_ema=(
            _load_safetensors(path / "generator_ema.safetensors")
            if manifest["ema"]["initialized"]
            else None
        ),
        generator_optimizer_state=_torch_load_cpu(path / "optimizer_generator.pt"),
        fake_score_optimizer_state=_torch_load_cpu(path / "optimizer_fake_score.pt"),
        rank_ema_states={
            rank: _torch_load_cpu(path / f"ema_state_rank{rank:05d}.pt")
            for rank in range(8)
        },
        rank_rng_states={
            rank: _torch_load_cpu(path / f"rng_state_rank{rank:05d}.pt")
            for rank in range(8)
        },
        resolved_config=resolved_config,
        provenance=provenance,
    )


def load_stage2_checkpoint_collective(
    directory: str | os.PathLike[str],
    *,
    expected_contract_hash: str,
    expected_phase_b_mode: str,
    expected_world_size: int = STAGE2_WORLD_SIZE,
    expected_topology: Mapping[str, Any] | None = None,
    expected_generator_parameter_names: Sequence[str] | None = None,
    expected_fake_score_parameter_names: Sequence[str] | None = None,
    collectives: Stage2CollectiveOps | None = None,
    io_include_cuda: bool = True,
) -> Stage2DistributedCheckpointPayload:
    """Rank-0 validate once, then construct the minimal exact view per rank."""

    ops = _collectives(collectives)
    rank = int(ops.get_rank())
    if expected_world_size != STAGE2_WORLD_SIZE:
        raise RuntimeError(
            f"Stage-2 collective resume requires expected_world_size=8, "
            f"got {expected_world_size}"
        )
    io_rng = capture_rng_state(include_cuda=io_include_cuda)
    try:
        full_rank0 = _consensus_call(
            "rank0 full checkpoint validation/load",
            lambda: (
                load_stage2_checkpoint(
                    directory,
                    expected_contract_hash=expected_contract_hash,
                    expected_world_size=expected_world_size,
                    expected_topology=expected_topology,
                    expected_phase_b_mode=expected_phase_b_mode,
                    expected_generator_parameter_names=expected_generator_parameter_names,
                    expected_fake_score_parameter_names=expected_fake_score_parameter_names,
                )
                if rank == 0
                else None
            ),
            ops,
        )
        small = (
            {
                "directory": str(full_rank0.directory),
                "manifest": dict(full_rank0.manifest),
                "trainer_state": dict(full_rank0.trainer_state),
                "resolved_config": dict(full_rank0.resolved_config),
                "provenance": dict(full_rank0.provenance),
            }
            if rank == 0
            else None
        )
        small = ops.broadcast_object(small, 0)
        if not isinstance(small, Mapping):
            raise RuntimeError("rank0 did not broadcast Stage-2 resume metadata")
        path = Path(small["directory"])
        manifest = small["manifest"]
        file_entries = {entry["name"]: entry for entry in manifest["files"]}

        def load_rank_view():
            for name in (
                "generator_raw.safetensors",
                "fake_score_raw.safetensors",
                f"ema_state_rank{rank:05d}.pt",
                f"rng_state_rank{rank:05d}.pt",
            ):
                entry = file_entries[name]
                candidate = path / name
                if (
                    candidate.is_symlink()
                    or not candidate.is_file()
                    or candidate.stat().st_size != entry["size"]
                    or sha256_file(candidate) != entry["sha256"]
                ):
                    raise RuntimeError(
                        f"Stage-2 resume file changed after rank0 validation: {candidate}"
                    )
            local_ema = (
                full_rank0.rank_ema_states[rank]
                if rank == 0
                else _torch_load_cpu(path / f"ema_state_rank{rank:05d}.pt")
            )
            local_rng = (
                full_rank0.rank_rng_states[rank]
                if rank == 0
                else _torch_load_cpu(path / f"rng_state_rank{rank:05d}.pt")
            )
            completed_g = int(small["trainer_state"]["completed_generator_updates"])
            _validate_ema_state(local_ema, rank=rank, completed_g=completed_g)
            validate_stage2_rng_state(
                local_rng,
                expected_rank=rank,
                expected_dedicated_names=set(STAGE2_DEDICATED_RNG_NAMES),
            )
            raw_g = (
                full_rank0.generator_raw
                if rank == 0
                else _load_safetensors(path / "generator_raw.safetensors")
            )
            raw_f = (
                full_rank0.fake_score_raw
                if rank == 0
                else _load_safetensors(path / "fake_score_raw.safetensors")
            )
            return raw_g, raw_f, local_ema, local_rng

        raw_g, raw_f, local_ema, local_rng = _consensus_call(
            "rank-local Stage-2 resume view load", load_rank_view, ops
        )
        ops.barrier()
        return Stage2DistributedCheckpointPayload(
            directory=path,
            manifest=manifest,
            trainer_state=small["trainer_state"],
            generator_raw=raw_g,
            fake_score_raw=raw_f,
            generator_ema_rank0=(full_rank0.generator_ema if rank == 0 else None),
            generator_optimizer_state_rank0=(
                full_rank0.generator_optimizer_state if rank == 0 else None
            ),
            fake_score_optimizer_state_rank0=(
                full_rank0.fake_score_optimizer_state if rank == 0 else None
            ),
            local_ema_state=local_ema,
            local_rng_state=local_rng,
            resolved_config=small["resolved_config"],
            provenance=small["provenance"],
        )
    finally:
        # Loading is transparent until the trainer explicitly performs the
        # final restore_stage2_rng_state call after iterator construction.
        restore_rng_state(io_rng, require_cuda_topology=io_include_cuda)


def _checkpoint_candidates(root: Path) -> list[tuple[int, Path]]:
    if not root.exists():
        if root.is_symlink():
            raise RuntimeError(
                f"Stage-2 checkpoint root is not a regular directory: {root}"
            )
        return []
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(
            f"Stage-2 checkpoint root is not a regular directory: {root}"
        )
    root = root.resolve()
    candidates: list[tuple[int, Path]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        match = STAGE2_CHECKPOINT_PATTERN.fullmatch(child.name)
        if match is None:
            continue
        if child.is_symlink() or not child.is_dir():
            raise RuntimeError(
                f"Stage-2 checkpoint candidate is not a directory: {child}"
            )
        marker = child / "_SUCCESS"
        if not marker.exists() and not marker.is_symlink():
            # A crash after atomic rename but before marker publication leaves
            # this exact shape. It is not a checkpoint and must not prevent
            # recovery from the previous complete cycle.
            continue
        step = int(match.group(1))
        validate_stage2_checkpoint(child)
        candidates.append((step, child))
    return candidates


def find_latest_stage2_checkpoint(
    root: str | os.PathLike[str],
) -> Path | None:
    """Return newest valid checkpoint; any damaged candidate fails closed."""

    candidates = _checkpoint_candidates(Path(root).expanduser())
    return max(candidates, default=(None, None))[1]


def apply_stage2_checkpoint_retention(
    root: str | os.PathLike[str],
    *,
    keep_last: int = 2,
    milestone_updates: Iterable[int] = STAGE2_CHECKPOINT_MILESTONES,
) -> list[int]:
    """Delete old non-milestones only after validating every candidate first."""

    keep_last = _plain_int(keep_last, "keep_last")
    root_path = Path(root).expanduser()
    candidates = _checkpoint_candidates(root_path)
    keep = {
        _plain_int(value, "milestone update", minimum=1) for value in milestone_updates
    }
    if keep_last:
        keep.update(step for step, _ in candidates[-keep_last:])
    remove = [(step, path) for step, path in candidates if step not in keep]
    # The full validation above occurs before the first destructive mutation.
    removed: list[int] = []
    for step, path in remove:
        shutil.rmtree(path)
        removed.append(step)
    if candidates:
        _fsync_directory(root_path.resolve())
    return removed


def _gather_stage2_ema_adapter(
    generator_module: torch.nn.Module,
    generator_ema: Any,
    *,
    expected_schema: Mapping[str, LoraTensorSpec],
    shard_group: Any,
    collectives: Stage2CollectiveOps,
    local_shards_fn: Callable[..., Mapping[str, LocalLoraShard]] | None,
    gather_object_fn: Callable[..., None] | None,
) -> OrderedDict[str, torch.Tensor] | None:
    """Gather canonical EMA while proving raw shards are restored on all ranks."""

    context = generator_ema.swap_into(generator_module)
    entered = False
    enter_error: Exception | None = None
    try:
        context.__enter__()
        entered = True
    except Exception as exc:
        enter_error = exc
    if not collectives.consensus(enter_error is None):
        restore_error: Exception | None = None
        if entered:
            try:
                context.__exit__(None, None, None)
            except Exception as exc:
                restore_error = exc
        if not collectives.consensus(restore_error is None):
            raise RuntimeError(
                "Stage-2 EMA swap entry and raw restore both failed"
            ) from (restore_error or enter_error)
        raise RuntimeError(
            "Stage-2 EMA swap failed on at least one rank"
        ) from enter_error

    result = None
    body_error: Exception | None = None
    try:
        result = gather_stage2_lora_state_dict(
            generator_module,
            expected_schema=expected_schema,
            role="generator",
            shard_group=shard_group,
            collectives=collectives,
            local_shards_fn=local_shards_fn,
            gather_object_fn=gather_object_fn,
        )
    except Exception as exc:
        body_error = exc
    exit_error: Exception | None = None
    try:
        context.__exit__(
            type(body_error) if body_error is not None else None,
            body_error,
            body_error.__traceback__ if body_error is not None else None,
        )
    except Exception as exc:
        exit_error = exc
    if not collectives.consensus(exit_error is None):
        raise RuntimeError(
            "Stage-2 EMA gather did not restore raw shards"
        ) from exit_error
    if body_error is not None:
        raise body_error
    collectives.barrier()
    return result


def save_stage2_checkpoint(
    root: str | os.PathLike[str],
    *,
    trainer_state: Mapping[str, Any],
    metrics_lineage_snapshot: bytes | None = None,
    resolved_config: Mapping[str, Any] | Any,
    generator_module: torch.nn.Module,
    fake_score_module: torch.nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    fake_score_optimizer: torch.optim.Optimizer,
    generator_ema: Any,
    generator_schema: Mapping[str, LoraTensorSpec],
    fake_score_schema: Mapping[str, LoraTensorSpec],
    dedicated_generators: Mapping[str, torch.Generator],
    rank0_control_generators: Mapping[str, torch.Generator] | None,
    provenance: Mapping[str, Any],
    topology: Mapping[str, Any],
    shard_group: Any = None,
    collectives: Stage2CollectiveOps | None = None,
    local_shards_fn: Callable[..., Mapping[str, LocalLoraShard]] | None = None,
    gather_lora_object_fn: Callable[..., None] | None = None,
    gather_rank_object_fn: Callable[..., None] | None = None,
    io_include_cuda: bool = True,
    keep_last: int = 2,
    apply_retention: bool = True,
    failure_injector: Callable[[str], None] | None = None,
) -> Path:
    """Collectively gather, publish, validate, and retain one checkpoint.

    Every rank calls this function in the same order.  Only selective LoRA
    shards and optimizer-only DCP state are gathered; immutable base weights
    are never materialized.  Rank 0 performs directory publication after all
    eight rank-local EMA/RNG payloads have arrived, then every rank receives
    the committed path or the same non-zero failure.
    """

    import torch.distributed as dist

    ops = _collectives(collectives)
    rank = int(ops.get_rank())
    io_rng = capture_rng_state(include_cuda=io_include_cuda)
    local_rng_state: Mapping[str, Any] | None = None
    try:
        # Persist the exact entry state before any LoRA/DCP/EMA gather. The same
        # snapshot is restored in ``finally`` so checkpoint I/O is transparent
        # to both default and explicit RNG streams.
        local_rng_state = _consensus_call(
            "rank-local Stage-2 entry RNG capture",
            lambda: capture_stage2_rng_state(
                rank=rank,
                dedicated_generators=dedicated_generators,
                rank0_control_generators=rank0_control_generators,
                include_cuda=io_include_cuda,
            ),
            ops,
        )
        validate_stage2_rng_state(
            local_rng_state,
            expected_rank=rank,
            expected_dedicated_names=set(STAGE2_DEDICATED_RNG_NAMES),
        )
        validate_stage2_trainer_state(trainer_state)
        completed_g = int(trainer_state["completed_generator_updates"])
        completed_f = int(trainer_state["completed_fake_updates"])
        _validate_topology(topology)

        def audit_runtime_boundary() -> None:
            pending_gradients = [
                f"{role}.{name}"
                for role, module in (
                    ("generator", generator_module),
                    ("fake_score", fake_score_module),
                )
                for name, parameter in module.named_parameters()
                if parameter.grad is not None
            ]
            if pending_gradients:
                raise RuntimeError(
                    "Stage-2 checkpoint boundary contains pending gradients: "
                    f"{pending_gradients[:8]}"
                )
            loader_states = trainer_state["dataloader_generator_states"]
            for role in STAGE2_ROLE_NAMES:
                generator = dedicated_generators.get(f"{role}_loader")
                if generator is None or not torch.equal(
                    generator.get_state().detach().cpu(), loader_states[role]
                ):
                    raise RuntimeError(
                        f"Stage-2 {role} DataLoader RNG copies disagree at save entry"
                    )

        _consensus_call("live cycle-boundary audit", audit_runtime_boundary, ops)
        raw_generator = gather_stage2_lora_state_dict(
            generator_module,
            expected_schema=generator_schema,
            role="generator",
            shard_group=shard_group,
            collectives=ops,
            local_shards_fn=local_shards_fn,
            gather_object_fn=gather_lora_object_fn,
        )
        raw_fake_score = gather_stage2_lora_state_dict(
            fake_score_module,
            expected_schema=fake_score_schema,
            role="fake_score",
            shard_group=shard_group,
            collectives=ops,
            local_shards_fn=local_shards_fn,
            gather_object_fn=gather_lora_object_fn,
        )
        generator_optimizer_state = gather_stage2_optimizer_state(
            generator_module,
            generator_optimizer,
            role="generator",
            expected_schema=generator_schema,
            expected_completed_updates=completed_g,
            collectives=ops,
        )
        fake_score_optimizer_state = gather_stage2_optimizer_state(
            fake_score_module,
            fake_score_optimizer,
            role="fake_score",
            expected_schema=fake_score_schema,
            expected_completed_updates=completed_f,
            collectives=ops,
        )
        expected_ema_initialized = completed_g >= STAGE2_EMA_START_GENERATOR_UPDATE

        def audit_local_ema_clock() -> Mapping[str, Any]:
            state = generator_ema.state_dict()
            return _validate_ema_state(
                state,
                rank=rank,
                completed_g=completed_g,
                expected_parameter_names={
                    spec.raw_parameter_name for spec in generator_schema.values()
                },
            )

        local_ema_state = _consensus_call(
            "rank-local Generator EMA clock audit", audit_local_ema_clock, ops
        )
        ema_adapter = (
            _gather_stage2_ema_adapter(
                generator_module,
                generator_ema,
                expected_schema=generator_schema,
                shard_group=shard_group,
                collectives=ops,
                local_shards_fn=local_shards_fn,
                gather_object_fn=gather_lora_object_fn,
            )
            if expected_ema_initialized
            else None
        )
        gather_fn = gather_rank_object_fn or dist.gather_object
        gathered_rank_payloads: list[Any] | None = [None] * 8 if rank == 0 else None
        _consensus_call(
            "rank-local EMA/RNG payload gather",
            lambda: gather_fn(
                (rank, local_ema_state, local_rng_state),
                object_gather_list=gathered_rank_payloads,
                dst=0,
            ),
            ops,
        )
        ops.barrier()

        def publish_rank_zero() -> str | None:
            if rank != 0:
                return None
            if not isinstance(metrics_lineage_snapshot, bytes):
                raise TypeError(
                    "rank0 requires the exact Stage-2 metrics lineage snapshot"
                )
            assert gathered_rank_payloads is not None
            by_rank: dict[int, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
            for payload in gathered_rank_payloads:
                if (
                    not isinstance(payload, tuple)
                    or len(payload) != 3
                    or isinstance(payload[0], bool)
                    or not isinstance(payload[0], int)
                ):
                    raise RuntimeError(
                        "invalid gathered Stage-2 rank checkpoint payload"
                    )
                payload_rank, ema_state, rng_state = payload
                if payload_rank in by_rank:
                    raise RuntimeError("duplicate gathered Stage-2 checkpoint rank")
                by_rank[payload_rank] = (ema_state, rng_state)
            if set(by_rank) != set(range(8)):
                raise RuntimeError("gathered Stage-2 checkpoint ranks are incomplete")
            if raw_generator is None or raw_fake_score is None:
                raise RuntimeError("rank0 is missing canonical raw adapters")
            if generator_optimizer_state is None or fake_score_optimizer_state is None:
                raise RuntimeError("rank0 is missing full optimizer state")
            config_value = (
                resolved_config.to_dict()
                if callable(getattr(resolved_config, "to_dict", None))
                else resolved_config
            )
            if not isinstance(config_value, Mapping):
                raise TypeError("resolved Stage-2 config must serialize to a mapping")
            destination = save_stage2_checkpoint_from_payloads(
                root,
                trainer_state=trainer_state,
                metrics_lineage_snapshot=metrics_lineage_snapshot,
                generator_raw=raw_generator,
                fake_score_raw=raw_fake_score,
                generator_ema=ema_adapter,
                generator_schema=generator_schema,
                fake_score_schema=fake_score_schema,
                generator_optimizer_state=generator_optimizer_state,
                fake_score_optimizer_state=fake_score_optimizer_state,
                rank_ema_states={key: value[0] for key, value in by_rank.items()},
                rank_rng_states={key: value[1] for key, value in by_rank.items()},
                resolved_config=dict(config_value),
                provenance=provenance,
                topology=topology,
                io_include_cuda=io_include_cuda,
                failure_injector=failure_injector,
            )
            if apply_retention:
                apply_stage2_checkpoint_retention(root, keep_last=keep_last)
            return str(destination)

        destination_value = _consensus_call(
            "rank0 atomic Stage-2 checkpoint publication", publish_rank_zero, ops
        )
        destination_value = ops.broadcast_object(destination_value, 0)
        if not isinstance(destination_value, str) or not destination_value:
            raise RuntimeError("rank0 did not broadcast the committed checkpoint path")
        ops.barrier()
        return Path(destination_value)
    finally:
        # Includes collective gathering and rank-local serialization, not just
        # rank-0 filesystem writes.
        if local_rng_state is None:
            restore_rng_state(io_rng, require_cuda_topology=io_include_cuda)
        else:
            restore_stage2_rng_state(
                local_rng_state,
                rank=rank,
                dedicated_generators=dedicated_generators,
                rank0_control_generators=rank0_control_generators,
                require_cuda_topology=io_include_cuda,
            )


__all__ = [
    "STAGE2_CHECKPOINT_MILESTONES",
    "STAGE2_CHECKPOINT_SCHEMA",
    "STAGE2_DEDICATED_RNG_NAMES",
    "STAGE2_EMA_START_GENERATOR_UPDATE",
    "STAGE2_PROVENANCE_SCHEMA",
    "Stage2CheckpointPayload",
    "Stage2CollectiveOps",
    "Stage2DistributedCheckpointPayload",
    "Stage2GeneratorEMACheckpointPayload",
    "apply_stage2_checkpoint_retention",
    "audit_stage2_lora_optimizer",
    "build_stage2_trainer_state",
    "capture_stage2_rng_state",
    "checkpoint_directory",
    "checkpoint_generator_updates",
    "consolidate_stage2_lora_shards",
    "derive_stage2_phase_state",
    "find_latest_stage2_checkpoint",
    "gather_stage2_lora_state_dict",
    "gather_stage2_optimizer_state",
    "load_stage2_checkpoint",
    "load_stage2_checkpoint_collective",
    "load_stage2_generator_ema_checkpoint",
    "restore_stage2_optimizer_state",
    "restore_stage2_rng_state",
    "save_stage2_checkpoint",
    "save_stage2_checkpoint_from_payloads",
    "validate_stage2_adapter_state",
    "validate_stage2_checkpoint",
    "validate_stage2_cycle_boundary",
    "validate_stage2_full_optimizer_state",
    "validate_stage2_provenance",
    "validate_stage2_rng_state",
    "validate_stage2_trainer_state",
]
