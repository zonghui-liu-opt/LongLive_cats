"""Stage-2-only one-dimensional FSDP2 FULL_SHARD helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
import socket
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from utils.lora_utils import LoraTensorSpec
from utils.parameter_names import map_parameter_names_to_expected

STAGE2_FSDP2_WORLD_SIZES = (4, 8)
# Retained only so the cumulative pre-Phase-35 innernet hotfix can leave an
# older world8 checkout runnable; production logic below is topology-derived.
STAGE2_FSDP2_WORLD_SIZE = 8
STAGE2_FSDP2_MESH_SHAPE = (8,)
STAGE2_FSDP2_MESH_DIM_NAMES = ("shard",)
STAGE2_MIN_H100_MEMORY_BYTES = 75 * 1024**3


@dataclass(frozen=True)
class Stage2RuntimeDescriptor:
    rank: int
    local_rank: int
    hostname: str
    visible_device_count: int
    device_name: str
    total_memory: int
    bf16_supported: bool


def validate_stage2_fsdp2_topology(
    *,
    world_size: int,
    sequence_parallel_size: int,
    data_parallel_size: int,
    mesh_shape: Sequence[int],
    mesh_dim_names: Sequence[str],
) -> dict[str, Any]:
    actual = {
        "world_size": int(world_size),
        "sequence_parallel_size": int(sequence_parallel_size),
        "data_parallel_size": int(data_parallel_size),
        "mesh_shape": tuple(int(value) for value in mesh_shape),
        "mesh_dim_names": tuple(str(value) for value in mesh_dim_names),
    }
    if actual["world_size"] not in STAGE2_FSDP2_WORLD_SIZES:
        raise ValueError(
            "Stage-2 FSDP2 world_size must be one of "
            f"{STAGE2_FSDP2_WORLD_SIZES}, got {actual['world_size']}"
        )
    expected = {
        "world_size": actual["world_size"],
        "sequence_parallel_size": 1,
        "data_parallel_size": actual["world_size"],
        "mesh_shape": (actual["world_size"],),
        "mesh_dim_names": STAGE2_FSDP2_MESH_DIM_NAMES,
    }
    wrong = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if wrong:
        raise ValueError(f"Stage-2 FSDP2 FULL_SHARD topology mismatch: {wrong}")
    return {
        **actual,
        "backend": "fsdp2",
        "sharding_strategy": "FULL_SHARD",
        "rank_layout": tuple(range(actual["world_size"])),
    }


def audit_stage2_runtime_descriptors(
    descriptors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    world_size = len(descriptors)
    if world_size not in STAGE2_FSDP2_WORLD_SIZES:
        raise RuntimeError(
            "Stage-2 runtime audit requires exactly 4 or 8 rank descriptors"
        )
    ordered = sorted(descriptors, key=lambda item: int(item["rank"]))
    ranks = tuple(int(item["rank"]) for item in ordered)
    local_ranks = tuple(sorted(int(item["local_rank"]) for item in ordered))
    expected_ranks = tuple(range(world_size))
    if ranks != expected_ranks or local_ranks != expected_ranks:
        raise RuntimeError(
            f"Stage-2 rank/local-rank layout mismatch: ranks={ranks}, "
            f"local_ranks={local_ranks}"
        )
    hosts = {str(item["hostname"]) for item in ordered}
    if len(hosts) != 1:
        raise RuntimeError(f"Stage-2 requires a single host, got {sorted(hosts)}")
    failures: list[str] = []
    for item in ordered:
        rank = int(item["rank"])
        if int(item["visible_device_count"]) != world_size:
            failures.append(
                f"rank{rank}: visible GPUs={item['visible_device_count']} "
                f"expected={world_size}"
            )
        if "H100" not in str(item["device_name"]).upper():
            failures.append(f"rank{rank}: device is not H100 ({item['device_name']!r})")
        if int(item["total_memory"]) < STAGE2_MIN_H100_MEMORY_BYTES:
            failures.append(
                f"rank{rank}: H100 memory={item['total_memory']} below "
                f"{STAGE2_MIN_H100_MEMORY_BYTES}"
            )
        if not bool(item["bf16_supported"]):
            failures.append(f"rank{rank}: CUDA BF16 is unavailable")
    if failures:
        raise RuntimeError("Stage-2 H100 runtime audit failed: " + "; ".join(failures))
    return {
        "single_host": next(iter(hosts)),
        "world_size": world_size,
        "device_names": tuple(str(item["device_name"]) for item in ordered),
        "total_memory_bytes": tuple(int(item["total_memory"]) for item in ordered),
        "bf16_supported": True,
    }


def _fsdp2_api() -> dict[str, Any]:
    # Reuse the already version-gated public FSDP2 import surface without
    # importing or applying the Stage-1 2x3 topology.
    from utils.distributed import _import_fsdp2_api, validate_stage1_fsdp2_api

    validate_stage1_fsdp2_api()
    return _import_fsdp2_api()


def validate_stage2_fsdp2_runtime() -> dict[str, Any]:
    _fsdp2_api()
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Stage-2 FSDP2 requires an initialized process group")
    if not dist.is_nccl_available() or "nccl" not in str(dist.get_backend()).lower():
        raise RuntimeError("Stage-2 FSDP2 requires the NCCL backend")
    validate_stage2_fsdp2_topology(
        world_size=dist.get_world_size(),
        sequence_parallel_size=1,
        data_parallel_size=dist.get_world_size(),
        mesh_shape=(dist.get_world_size(),),
        mesh_dim_names=STAGE2_FSDP2_MESH_DIM_NAMES,
    )
    rank = dist.get_rank()
    descriptor = None
    error = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        for key, actual in (("WORLD_SIZE", dist.get_world_size()), ("RANK", rank)):
            if key not in os.environ or int(os.environ[key]) != int(actual):
                raise RuntimeError(f"torchrun {key} disagrees with the process group")
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("torchrun LOCAL_RANK is missing")
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank != torch.cuda.current_device():
            raise RuntimeError(
                f"current CUDA device {torch.cuda.current_device()} != LOCAL_RANK {local_rank}"
            )
        properties = torch.cuda.get_device_properties(local_rank)
        descriptor = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": socket.gethostname(),
            "visible_device_count": torch.cuda.device_count(),
            "device_name": properties.name,
            "total_memory": int(properties.total_memory),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    except Exception as exc:  # turn rank-local failures into one WORLD failure
        error = f"{type(exc).__name__}: {exc}"
    statuses: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(
        statuses,
        {"rank": rank, "descriptor": descriptor, "error": error},
    )
    errors = [
        f"rank{item['rank']}: {item['error']}"
        for item in statuses
        if item is not None and item["error"] is not None
    ]
    if errors or any(item is None for item in statuses):
        raise RuntimeError(f"Stage-2 rank-local runtime audit failed: {errors}")
    return audit_stage2_runtime_descriptors(
        [item["descriptor"] for item in statuses if item is not None]
    )


def build_stage2_fsdp2_device_mesh():
    runtime = validate_stage2_fsdp2_runtime()
    world_size = int(runtime["world_size"])
    mesh = _fsdp2_api()["init_device_mesh"](
        "cuda",
        (world_size,),
        mesh_dim_names=STAGE2_FSDP2_MESH_DIM_NAMES,
    )
    ranks = tuple(int(value) for value in mesh.mesh.detach().cpu().tolist())
    if ranks != tuple(range(world_size)):
        raise RuntimeError(f"Stage-2 DeviceMesh rank layout mismatch: {ranks}")
    if tuple(mesh.mesh_dim_names or ()) != STAGE2_FSDP2_MESH_DIM_NAMES:
        raise RuntimeError("Stage-2 DeviceMesh dim names mismatch")
    return mesh, runtime


def _mesh_contract(mesh: Any) -> None:
    tensor = mesh.mesh.detach().to(device="cpu")
    if tensor.ndim != 1:
        raise RuntimeError("Stage-2 FSDP2 requires a 1D world4/world8 DeviceMesh")
    ranks = tuple(int(value) for value in tensor.tolist())
    if len(ranks) not in STAGE2_FSDP2_WORLD_SIZES or ranks != tuple(range(len(ranks))):
        raise RuntimeError("Stage-2 FSDP2 requires a 1D world4/world8 DeviceMesh")
    if tuple(mesh.mesh_dim_names or ()) != STAGE2_FSDP2_MESH_DIM_NAMES:
        raise RuntimeError("Stage-2 FSDP2 DeviceMesh must be named ('shard',)")


def _audit_unsharded_parameter_contract(
    module: torch.nn.Module,
    *,
    role: str,
    expected_trainable_tensors: int,
    expected_trainable_parameters: int,
) -> tuple[str, ...]:
    trainable = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if len(trainable) != int(expected_trainable_tensors):
        raise RuntimeError(
            f"Stage-2 {role} trainable tensor count mismatch: "
            f"expected={expected_trainable_tensors}, actual={len(trainable)}"
        )
    numel = sum(parameter.numel() for _, parameter in trainable)
    if numel != int(expected_trainable_parameters):
        raise RuntimeError(
            f"Stage-2 {role} trainable parameter count mismatch: "
            f"expected={expected_trainable_parameters}, actual={numel}"
        )
    invalid_trainable = [
        (name, str(parameter.dtype))
        for name, parameter in trainable
        if parameter.dtype != torch.float32
        or not ("lora_A" in name or "lora_B" in name)
    ]
    if invalid_trainable:
        raise TypeError(
            f"Stage-2 {role} trainables must be FP32 LoRA A/B: "
            f"{invalid_trainable[:8]}"
        )
    invalid_frozen = [
        (name, str(parameter.dtype))
        for name, parameter in module.named_parameters()
        if not parameter.requires_grad
        and parameter.is_floating_point()
        and parameter.dtype != torch.bfloat16
    ]
    if invalid_frozen:
        raise TypeError(
            f"Stage-2 {role} frozen base must be BF16: {invalid_frozen[:8]}"
        )
    if role == "real_score" and trainable:
        raise RuntimeError("Stage-2 real_score must remain fully frozen")
    return tuple(name for name, _ in trainable)


def fsdp2_wrap_stage2_role(
    module: torch.nn.Module,
    *,
    transformer: torch.nn.Module,
    mesh: Any,
    role: str,
    block_class_name: str,
    expected_trainable_tensors: int,
    expected_trainable_parameters: int,
    expected_blocks: int = 30,
    fsdp_api: Mapping[str, Any] | None = None,
):
    """Wrap one role independently, 30 blocks bottom-up then its own root."""

    if role not in {"generator", "real_score", "fake_score"}:
        raise ValueError(f"unknown Stage-2 FSDP role: {role!r}")
    _mesh_contract(mesh)
    names_before = _audit_unsharded_parameter_contract(
        module,
        role=role,
        expected_trainable_tensors=expected_trainable_tensors,
        expected_trainable_parameters=expected_trainable_parameters,
    )
    blocks = tuple(
        child
        for child in transformer.modules()
        if child.__class__.__name__ == block_class_name
    )
    if len(blocks) != int(expected_blocks) or len(
        {id(block) for block in blocks}
    ) != len(blocks):
        raise RuntimeError(
            f"Stage-2 {role} expected {expected_blocks} {block_class_name} blocks, "
            f"found {len(blocks)}"
        )
    api = dict(fsdp_api or _fsdp2_api())
    block_policy = api["MixedPrecisionPolicy"](
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    # The Stage2DiTRole root receives semantic FP32 tensors (score/rollout
    # timesteps and exact UniPC sigmas).  Casting the whole pytree here would
    # round values such as 999 -> 1000 in BF16 before the model can build its
    # time embedding.  Latent/prompt compute tensors are cast explicitly by
    # the role adapter; transformer blocks may still autocast their activations.
    root_policy = api["MixedPrecisionPolicy"](
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=False,
    )
    fully_shard = api["fully_shard"]
    for block in blocks:
        fully_shard(
            block,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=block_policy,
        )
    # All three 5B roots reshard after forward.  This init-only batch never
    # performs a forward; the setting prevents a full teacher root lingering
    # in later sequential real CFG passes.
    fully_shard(
        module,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=root_policy,
    )
    trainable_after = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if tuple(sorted(trainable_after)) != tuple(sorted(names_before)):
        raise RuntimeError(f"Stage-2 {role} FSDP2 changed trainable FQNs")
    wrong_master = [
        (name, str(parameter.dtype))
        for name, parameter in trainable_after.items()
        if parameter.dtype != torch.float32
    ]
    if wrong_master:
        raise TypeError(
            f"Stage-2 {role} FSDP2 did not preserve FP32 LoRA masters: "
            f"{wrong_master[:8]}"
        )
    return module


def audit_stage2_fsdp2_role(
    module: torch.nn.Module,
    *,
    role: str,
    expected_schema: Mapping[str, LoraTensorSpec] | None,
    expected_fsdp_modules: int = 31,
    fsdp_module_type: type | None = None,
) -> dict[str, Any]:
    """Audit every post-wrap parameter and every independent FSDP2 group.

    The frozen 5B base is part of this gate: checking only the trainable LoRA
    shards would allow a no-op/root-only wrapper to be reported as FULL_SHARD.
    This routine deliberately reads DTensor metadata only and never gathers a
    parameter or materializes a full state dict.
    """

    if role not in {"generator", "real_score", "fake_score"}:
        raise ValueError(f"unknown Stage-2 post-FSDP role: {role!r}")
    if fsdp_module_type is None:
        fsdp_module_type = _fsdp2_api()["FSDPModule"]
    fsdp_modules = tuple(
        child for child in module.modules() if isinstance(child, fsdp_module_type)
    )
    if not isinstance(module, fsdp_module_type):
        raise RuntimeError(f"Stage-2 {role} root is not an FSDPModule")
    if len(fsdp_modules) != int(expected_fsdp_modules):
        raise RuntimeError(
            f"Stage-2 {role} FSDPModule count mismatch: "
            f"expected={expected_fsdp_modules}, actual={len(fsdp_modules)}"
        )

    named_parameters = tuple(module.named_parameters())
    if not named_parameters:
        raise RuntimeError(f"Stage-2 {role} has no parameters after FSDP2")
    trainable = [
        (name, value) for name, value in named_parameters if value.requires_grad
    ]
    if role == "real_score" and (trainable or expected_schema):
        raise RuntimeError("Stage-2 real_score post-FSDP must have no trainables")
    if role != "real_score" and not expected_schema:
        raise ValueError(f"Stage-2 {role} requires the pre-FSDP LoRA schema")
    if expected_schema is not None and len(trainable) != len(expected_schema):
        raise RuntimeError(f"Stage-2 {role} post-FSDP trainable count mismatch")

    frozen_tensor_count = 0
    global_frozen_parameters = 0
    canonical_keys: list[str] = []
    audited_mesh_shape: tuple[int, ...] | None = None
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
        if not isinstance(parameter, DTensor):
            raise RuntimeError(f"Stage-2 {role} parameter is not DTensor: {name}")
        if parameter.is_meta:
            raise RuntimeError(f"Stage-2 {role} parameter remained meta: {name}")
        mesh = parameter.device_mesh
        mesh_shape = tuple(int(value) for value in mesh.shape)
        if (
            len(mesh_shape) != 1
            or mesh_shape[0] not in STAGE2_FSDP2_WORLD_SIZES
            or (audited_mesh_shape is not None and mesh_shape != audited_mesh_shape)
        ):
            raise RuntimeError(f"Stage-2 {role} parameter has wrong mesh shape: {name}")
        audited_mesh_shape = mesh_shape
        if tuple(mesh.mesh_dim_names or ()) != STAGE2_FSDP2_MESH_DIM_NAMES:
            raise RuntimeError(f"Stage-2 {role} parameter has wrong mesh names: {name}")
        placements = tuple(parameter.placements)
        if not (
            len(placements) == 1
            and isinstance(placements[0], Shard)
            and int(placements[0].dim) == 0
        ):
            raise RuntimeError(
                f"Stage-2 {role} parameter must use one-dimensional Shard(0): {name}"
            )
        if not parameter.requires_grad:
            if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
                raise TypeError(
                    f"Stage-2 {role} frozen DTensor must remain BF16: "
                    f"{name}={parameter.dtype}"
                )
            frozen_tensor_count += 1
            global_frozen_parameters += parameter.numel()
            continue
        if parameter.dtype != torch.float32:
            raise TypeError(f"Stage-2 {role} LoRA DTensor must remain FP32: {name}")
        assert expected_schema is not None
        key = raw_to_key[runtime_to_raw[name]]
        if tuple(parameter.shape) != tuple(expected_schema[key].global_shape):
            raise RuntimeError(f"Stage-2 {role} global LoRA shape mismatch: {key}")
        canonical_keys.append(key)
    if expected_schema is not None and set(canonical_keys) != set(expected_schema):
        raise RuntimeError(f"Stage-2 {role} post-FSDP LoRA keys drifted")
    return {
        "role": role,
        "trainable_tensor_count": len(canonical_keys),
        "global_trainable_parameters": sum(
            int(torch.Size(spec.global_shape).numel())
            for spec in (expected_schema or {}).values()
        ),
        "canonical_keys": tuple(sorted(canonical_keys)),
        "frozen_tensor_count": frozen_tensor_count,
        "global_frozen_parameters": global_frozen_parameters,
        "all_parameter_tensor_count": len(named_parameters),
        "all_parameters_are_dtensor": True,
        "fsdp_module_count": len(fsdp_modules),
        "root_and_30_blocks_independently_wrapped": True,
        "mesh_shape": audited_mesh_shape,
        "mesh_dim_names": STAGE2_FSDP2_MESH_DIM_NAMES,
        "placements": ("shard:0",),
    }
