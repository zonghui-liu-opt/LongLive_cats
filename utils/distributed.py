from datetime import timedelta
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
import inspect
import os
import re
import socket
from typing import Any
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy


STAGE1_FSDP1_GATE0_MESSAGE = (
    "Stage-1 Gate 0 failed: the locked FSDP1 size-wrapped model places BF16 "
    "frozen base parameters and FP32 LoRA master parameters in the same "
    "FlatParamHandle, but FSDP1 requires one dtype per handle. PyTorch's "
    "FSDP1 SHARDED_STATE_DICT path also unshards the frozen flat parameter "
    "before ignore_frozen_params filtering. FSDP1 Stage-1 is disabled instead "
    "of falling back to a frozen 5B gather; use the validated Stage-1-only "
    "FSDP2 fully_shard + 2D HSDP backend. Legacy non-Stage-1 FSDP1 remains "
    "available."
)


STAGE1_FSDP2_MESH_DIM_NAMES = ("replicate", "shard")
STAGE1_FSDP2_RANK_LAYOUT = ((0, 1, 2), (3, 4, 5))
STAGE1_MIN_H100_MEMORY_BYTES = 75 * 1024**3


@dataclass(frozen=True)
class Stage1FSDP2Topology:
    """Locked Stage-1 HSDP topology.

    Rows are the two data-parallel replicas and are sharded across the three
    sequence-parallel ranks. Columns contain corresponding shards that are
    replicated across the two data-parallel samples.
    """

    rank_layout: tuple[tuple[int, ...], ...] = STAGE1_FSDP2_RANK_LAYOUT
    mesh_dim_names: tuple[str, str] = STAGE1_FSDP2_MESH_DIM_NAMES

    @property
    def replicate_size(self) -> int:
        return len(self.rank_layout)

    @property
    def shard_size(self) -> int:
        return len(self.rank_layout[0])

    @property
    def world_size(self) -> int:
        return self.replicate_size * self.shard_size

    @property
    def mesh_shape(self) -> tuple[int, int]:
        return self.replicate_size, self.shard_size


STAGE1_FSDP2_TOPOLOGY = Stage1FSDP2Topology()


def _torch_release_tuple(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(version))
    if match is None:
        raise RuntimeError(f"cannot parse PyTorch version {version!r}")
    return tuple(int(value or 0) for value in match.groups())


def _import_fsdp2_api() -> dict[str, Any]:
    """Import FSDP2 lazily so older legacy-FSDP1 jobs can still import here."""

    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Stage-1 requires the public PyTorch FSDP2 API introduced before "
            "the locked torch==2.8 runtime (fully_shard, FSDPModule, and "
            "MixedPrecisionPolicy). Legacy FSDP1 remains available."
        ) from exc
    return {
        "fully_shard": fully_shard,
        "FSDPModule": FSDPModule,
        "MixedPrecisionPolicy": MixedPrecisionPolicy,
        "init_device_mesh": init_device_mesh,
    }


def validate_stage1_fsdp2_api(*, torch_version: str | None = None) -> None:
    """Fail fast unless the installed API satisfies the PyTorch 2.8 contract."""

    version = torch.__version__ if torch_version is None else torch_version
    if _torch_release_tuple(version) < (2, 8, 0):
        raise RuntimeError(
            f"Stage-1 FSDP2 requires PyTorch >=2.8.0, found {version!r}"
        )
    api = _import_fsdp2_api()
    fully_shard_parameters = inspect.signature(api["fully_shard"]).parameters
    required_fully_shard_parameters = {
        "module",
        "mesh",
        "reshard_after_forward",
        "mp_policy",
    }
    missing = sorted(required_fully_shard_parameters - set(fully_shard_parameters))
    if missing:
        raise RuntimeError(
            f"installed fully_shard API is incompatible with PyTorch 2.8: missing {missing}"
        )
    policy_parameters = inspect.signature(api["MixedPrecisionPolicy"]).parameters
    missing_policy = sorted(
        {"param_dtype", "reduce_dtype"} - set(policy_parameters)
    )
    if missing_policy:
        raise RuntimeError(
            "installed MixedPrecisionPolicy API is incompatible with PyTorch 2.8: "
            f"missing {missing_policy}"
        )
    module_api = api["FSDPModule"]
    required_methods = {
        "set_is_last_backward",
        "set_requires_gradient_sync",
        "set_reshard_after_backward",
    }
    missing_methods = sorted(
        method for method in required_methods if not hasattr(module_api, method)
    )
    if missing_methods:
        raise RuntimeError(
            f"installed FSDPModule API is incompatible with PyTorch 2.8: missing {missing_methods}"
        )


def validate_stage1_fsdp2_topology(
    *,
    world_size: int,
    rank: int,
    topology: Stage1FSDP2Topology = STAGE1_FSDP2_TOPOLOGY,
) -> None:
    """Validate the exact DP2 x SP3 row-major rank topology."""

    rows = topology.rank_layout
    if not rows or any(len(row) != topology.shard_size for row in rows):
        raise RuntimeError(f"Stage-1 FSDP2 rank layout is not rectangular: {rows}")
    flat_ranks = tuple(rank_value for row in rows for rank_value in row)
    expected_ranks = tuple(range(topology.world_size))
    if flat_ranks != expected_ranks:
        raise RuntimeError(
            "Stage-1 FSDP2 rank layout must be row-major contiguous ranks; "
            f"expected {expected_ranks}, got {flat_ranks}"
        )
    if topology.mesh_dim_names != STAGE1_FSDP2_MESH_DIM_NAMES:
        raise RuntimeError(
            "Stage-1 FSDP2 mesh dimensions must be ('replicate', 'shard'), "
            f"got {topology.mesh_dim_names}"
        )
    if int(world_size) != topology.world_size:
        raise RuntimeError(
            f"Stage-1 FSDP2 requires WORLD_SIZE={topology.world_size}, got {world_size}"
        )
    if int(rank) not in expected_ranks:
        raise RuntimeError(
            f"Stage-1 FSDP2 rank must be in {expected_ranks}, got {rank}"
        )


def _validate_stage1_fsdp2_rank_descriptors(
    descriptors: list[Mapping[str, Any]],
    *,
    topology: Stage1FSDP2Topology,
) -> None:
    """Audit all six ranks after a WORLD collective capability exchange."""

    if len(descriptors) != topology.world_size:
        raise RuntimeError(
            "Stage-1 FSDP2 cross-rank audit returned the wrong record count: "
            f"expected={topology.world_size}, actual={len(descriptors)}"
        )
    ordered = sorted(descriptors, key=lambda item: int(item["rank"]))
    ranks = tuple(int(item["rank"]) for item in ordered)
    if ranks != tuple(range(topology.world_size)):
        raise RuntimeError(f"Stage-1 FSDP2 cross-rank audit has invalid ranks: {ranks}")
    hosts = {str(item["hostname"]) for item in ordered}
    if len(hosts) != 1:
        raise RuntimeError(
            f"Stage-1 is locked to one machine, but ranks span hosts={sorted(hosts)}"
        )
    local_ranks = tuple(sorted(int(item["local_rank"]) for item in ordered))
    if local_ranks != tuple(range(topology.world_size)):
        raise RuntimeError(
            "Stage-1 single-node LOCAL_RANK topology mismatch: "
            f"expected={tuple(range(topology.world_size))}, actual={local_ranks}"
        )
    failures: list[str] = []
    for descriptor in ordered:
        rank = int(descriptor["rank"])
        expected_coordinate = (rank // topology.shard_size, rank % topology.shard_size)
        coordinate = tuple(int(value) for value in descriptor["coordinate"])
        if coordinate != expected_coordinate:
            failures.append(
                f"rank{rank}: coordinate={coordinate}, expected={expected_coordinate}"
            )
        visible_device_count = int(descriptor["visible_device_count"])
        if visible_device_count != topology.world_size:
            failures.append(
                f"rank{rank}: visible CUDA device count={visible_device_count}, "
                f"expected={topology.world_size}"
            )
        device_name = str(descriptor["device_name"])
        if "H100" not in device_name.upper():
            failures.append(f"rank{rank}: device is not H100 ({device_name!r})")
        total_memory = int(descriptor["total_memory"])
        if total_memory < STAGE1_MIN_H100_MEMORY_BYTES:
            failures.append(
                f"rank{rank}: CUDA memory={total_memory} bytes, "
                f"minimum={STAGE1_MIN_H100_MEMORY_BYTES}"
            )
        if not bool(descriptor["bf16_supported"]):
            failures.append(f"rank{rank}: CUDA BF16 is not supported")
    if failures:
        raise RuntimeError(
            "Stage-1 FSDP2 six-rank H100 capability audit failed: "
            + "; ".join(failures)
        )


def validate_stage1_fsdp2_runtime(
    *, topology: Stage1FSDP2Topology = STAGE1_FSDP2_TOPOLOGY
) -> None:
    """Validate CUDA/NCCL/BF16 and launcher state before creating a mesh."""

    validate_stage1_fsdp2_api()
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Stage-1 FSDP2 requires an initialized process group")
    if not dist.is_nccl_available():
        raise RuntimeError("Stage-1 FSDP2 requires an NCCL-enabled PyTorch build")
    backend = str(dist.get_backend()).lower()
    if "nccl" not in backend:
        raise RuntimeError(f"Stage-1 FSDP2 requires NCCL, found backend={backend!r}")
    validate_stage1_fsdp2_topology(
        world_size=dist.get_world_size(), rank=dist.get_rank(), topology=topology
    )
    rank = dist.get_rank()
    local_descriptor = None
    local_error = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Stage-1 FSDP2 requires CUDA")
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("Stage-1 FSDP2 requires torchrun LOCAL_RANK")
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside visible CUDA devices "
                f"(count={torch.cuda.device_count()})"
            )
        if torch.cuda.current_device() != local_rank:
            raise RuntimeError(
                "Stage-1 FSDP2 current CUDA device must equal LOCAL_RANK: "
                f"current={torch.cuda.current_device()}, local_rank={local_rank}"
            )
        for environment_name, actual in (
            ("WORLD_SIZE", dist.get_world_size()),
            ("RANK", rank),
        ):
            if environment_name not in os.environ:
                raise RuntimeError(
                    f"Stage-1 FSDP2 requires torchrun {environment_name}"
                )
            if int(os.environ[environment_name]) != int(actual):
                raise RuntimeError(
                    f"{environment_name} disagrees with the process group: "
                    f"env={os.environ[environment_name]}, distributed={actual}"
                )
        properties = torch.cuda.get_device_properties(local_rank)
        local_descriptor = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": socket.gethostname(),
            "coordinate": (
                rank // topology.shard_size,
                rank % topology.shard_size,
            ),
            "visible_device_count": torch.cuda.device_count(),
            "device_name": properties.name,
            "total_memory": int(properties.total_memory),
            "capability": (int(properties.major), int(properties.minor)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    statuses: list[dict[str, Any] | None] = [None] * topology.world_size
    dist.all_gather_object(
        statuses,
        {"rank": rank, "error": local_error, "descriptor": local_descriptor},
    )
    missing = [index for index, status in enumerate(statuses) if status is None]
    errors = [
        f"rank{status['rank']}: {status['error']}"
        for status in statuses
        if status is not None and status["error"] is not None
    ]
    if missing or errors:
        raise RuntimeError(
            "Stage-1 FSDP2 rank-local runtime audit failed: "
            f"missing={missing}, errors={errors}"
        )
    descriptors = [status["descriptor"] for status in statuses if status is not None]
    _validate_stage1_fsdp2_rank_descriptors(
        descriptors,
        topology=topology,
    )


def build_stage1_fsdp2_device_mesh(
    *, topology: Stage1FSDP2Topology = STAGE1_FSDP2_TOPOLOGY
):
    """Create the locked CUDA HSDP mesh ``[[0,1,2],[3,4,5]]``."""

    validate_stage1_fsdp2_runtime(topology=topology)
    init_device_mesh = _import_fsdp2_api()["init_device_mesh"]
    mesh = init_device_mesh(
        "cuda",
        topology.mesh_shape,
        mesh_dim_names=topology.mesh_dim_names,
    )
    actual_layout = tuple(
        tuple(int(rank) for rank in row)
        for row in mesh.mesh.detach().cpu().tolist()
    )
    if actual_layout != topology.rank_layout:
        raise RuntimeError(
            "Stage-1 FSDP2 DeviceMesh rank layout mismatch: "
            f"expected={topology.rank_layout}, actual={actual_layout}"
        )
    if tuple(mesh.mesh_dim_names or ()) != topology.mesh_dim_names:
        raise RuntimeError(
            "Stage-1 FSDP2 DeviceMesh names mismatch: "
            f"expected={topology.mesh_dim_names}, actual={mesh.mesh_dim_names}"
        )
    shard_group = mesh.get_group("shard")
    replicate_group = mesh.get_group("replicate")
    if dist.get_world_size(shard_group) != topology.shard_size:
        raise RuntimeError("Stage-1 FSDP2 shard process group has the wrong size")
    if dist.get_world_size(replicate_group) != topology.replicate_size:
        raise RuntimeError("Stage-1 FSDP2 replicate process group has the wrong size")
    return mesh


def validate_stage1_fsdp1_gate0() -> None:
    """Reject FSDP1 only for Stage-1; legacy FSDP1 jobs remain untouched."""

    raise RuntimeError(STAGE1_FSDP1_GATE0_MESSAGE)


def _stage1_fsdp2_blocks(
    transformer: torch.nn.Module,
    *,
    expected_blocks: int,
    block_class_name: str,
) -> tuple[torch.nn.Module, ...]:
    blocks = tuple(
        module
        for _, module in transformer.named_modules()
        if module.__class__.__name__ == block_class_name
    )
    if len(blocks) != int(expected_blocks):
        raise RuntimeError(
            "Stage-1 FSDP2 must wrap the locked Wan blocks bottom-up: "
            f"expected {expected_blocks} {block_class_name} modules, found {len(blocks)}"
        )
    if len({id(block) for block in blocks}) != len(blocks):
        raise RuntimeError("Stage-1 FSDP2 block discovery returned duplicate modules")
    return blocks


def _audit_stage1_fsdp2_parameter_contract(
    module: torch.nn.Module,
    *,
    expected_trainable_tensors: int,
    expected_trainable_numel: int,
) -> tuple[str, ...]:
    trainable: list[tuple[str, torch.nn.Parameter]] = []
    invalid_frozen_dtype: list[tuple[str, str]] = []
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            trainable.append((name, parameter))
        elif parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            invalid_frozen_dtype.append((name, str(parameter.dtype)))
    if invalid_frozen_dtype:
        raise TypeError(
            "Stage-1 frozen base storage must be BF16; mismatches="
            f"{invalid_frozen_dtype[:8]}"
        )
    if len(trainable) != int(expected_trainable_tensors):
        raise RuntimeError(
            "Stage-1 FSDP2 trainable tensor count mismatch: "
            f"expected={expected_trainable_tensors}, actual={len(trainable)}"
        )
    trainable_numel = sum(parameter.numel() for _, parameter in trainable)
    if trainable_numel != int(expected_trainable_numel):
        raise RuntimeError(
            "Stage-1 FSDP2 trainable parameter count mismatch: "
            f"expected={expected_trainable_numel}, actual={trainable_numel}"
        )
    invalid_trainables = [
        (name, str(parameter.dtype))
        for name, parameter in trainable
        if parameter.dtype != torch.float32
        or not (".lora_A." in name or ".lora_B." in name)
    ]
    if invalid_trainables:
        raise TypeError(
            "Stage-1 FSDP2 trainables must be FP32 LoRA master parameters; "
            f"mismatches={invalid_trainables[:8]}"
        )
    return tuple(name for name, _ in trainable)


def fsdp2_wrap_stage1(
    module: torch.nn.Module,
    *,
    transformer: torch.nn.Module,
    mesh,
    expected_blocks: int = 30,
    block_class_name: str = "CausalWanAttentionBlock",
    expected_trainable_tensors: int = 360,
    expected_trainable_numel: int = 57_016_320,
):
    """Apply Stage-1-only FSDP2 HSDP bottom-up without touching legacy FSDP1.

    The original parameter dtype remains the optimizer/master dtype (FP32 for
    LoRA), while FSDP2 all-gathers BF16 parameters for forward/backward and
    reduces gradients in FP32. Frozen BF16 parameters and FP32 LoRA parameters
    may coexist because FSDP2 shards them per parameter instead of flattening
    them into a single mixed-dtype FlatParameter.
    """

    validate_stage1_fsdp2_api()
    if mesh is None:
        raise ValueError("Stage-1 FSDP2 requires the explicit 2D DeviceMesh")
    actual_layout = tuple(
        tuple(int(rank) for rank in row)
        for row in mesh.mesh.detach().cpu().tolist()
    )
    if actual_layout != STAGE1_FSDP2_RANK_LAYOUT:
        raise RuntimeError(
            "Stage-1 FSDP2 received the wrong DeviceMesh layout: "
            f"expected={STAGE1_FSDP2_RANK_LAYOUT}, actual={actual_layout}"
        )
    if tuple(mesh.mesh_dim_names or ()) != STAGE1_FSDP2_MESH_DIM_NAMES:
        raise RuntimeError(
            "Stage-1 FSDP2 received the wrong DeviceMesh dimension names: "
            f"{mesh.mesh_dim_names}"
        )
    trainable_names_before = _audit_stage1_fsdp2_parameter_contract(
        module,
        expected_trainable_tensors=expected_trainable_tensors,
        expected_trainable_numel=expected_trainable_numel,
    )
    blocks = _stage1_fsdp2_blocks(
        transformer,
        expected_blocks=expected_blocks,
        block_class_name=block_class_name,
    )
    api = _import_fsdp2_api()
    policy = api["MixedPrecisionPolicy"](
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    fully_shard = api["fully_shard"]
    for block in blocks:
        fully_shard(
            block,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=policy,
        )
    # Root last: parameters owned by the block groups are automatically
    # excluded, while patch/text/time embeddings and the head form the root
    # communication group. Keep the root unsharded between forward/backward as
    # recommended by the public FSDP2 contract.
    fully_shard(
        module,
        mesh=mesh,
        reshard_after_forward=False,
        mp_policy=policy,
    )
    trainable_after = {
        name: parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    if tuple(sorted(trainable_after)) != tuple(sorted(trainable_names_before)):
        raise RuntimeError(
            "FSDP2 changed Stage-1 trainable FQNs; FSDP2 must preserve names"
        )
    invalid_master_dtypes = [
        (name, str(parameter.dtype))
        for name, parameter in trainable_after.items()
        if parameter.dtype != torch.float32
    ]
    if invalid_master_dtypes:
        raise TypeError(
            "FSDP2 did not preserve FP32 LoRA optimizer/master shards: "
            f"{invalid_master_dtypes[:8]}"
        )
    return module


@contextmanager
def stage1_fsdp2_accumulation(
    module: torch.nn.Module,
    *,
    sync_gradients: bool,
    reshard_after_backward: bool = True,
):
    """Configure one FSDP2 microbatch's gradient synchronization.

    Set ``sync_gradients=False`` for non-final accumulation microbatches and
    ``True`` for the final microbatch. With the Stage-1 FP32 reduce policy,
    unsynchronized gradients accumulate in FP32. State is restored on exit so
    an exception cannot leak no-sync behavior into the next attempt.
    """

    api = _import_fsdp2_api()
    if not isinstance(module, api["FSDPModule"]):
        raise TypeError("stage1_fsdp2_accumulation requires an FSDP2 root module")
    sync_gradients = bool(sync_gradients)
    module.set_requires_gradient_sync(sync_gradients, recurse=True)
    module.set_reshard_after_backward(bool(reshard_after_backward), recurse=True)
    module.set_is_last_backward(sync_gradients)
    try:
        yield module
    finally:
        module.set_requires_gradient_sync(True, recurse=True)
        module.set_reshard_after_backward(True, recurse=True)
        module.set_is_last_backward(True)


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(
    module,
    sharding_strategy="full",
    mixed_precision=False,
    wrap_strategy="size",
    min_num_params=int(5e7),
    transformer_module=None,
    cpu_offload=False,
):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    # Use a long timeout so that slow collectives during checkpoint saving
    # (e.g. FSDP.optim_state_dict all-gather + rank0-only disk write for a
    # multi-GB full optimizer state) do not trip the NCCL watchdog on other
    # ranks while they wait at the post-save barrier.
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=60))
    torch.cuda.set_device(local_rank)


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @staticmethod
    def _clean_param_name(name: str) -> str:
        """Remove FSDP wrapper prefixes from parameter names."""
        return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                # Clean the parameter name to remove FSDP prefixes
                # This ensures shadow keys are compatible with unwrapped models for inference
                cleaned_name = self._clean_param_name(n)
                self.shadow[cleaned_name] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                cleaned_name = self._clean_param_name(n)
                if cleaned_name in self.shadow:
                    self.shadow[cleaned_name].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        # Return shadow dict directly - keys are already cleaned during init/update
        # This makes the state_dict directly usable for inference with unwrapped models
        return self.shadow            # picklable

    def load_state_dict(self, sd):
        # Handle both cases: with or without FSDP prefixes
        # This ensures backward compatibility and flexibility
        cleaned_sd = {}
        for k, v in sd.items():
            # Remove FSDP prefixes if present to match internal naming convention
            cleaned_key = self._clean_param_name(k)
            cleaned_sd[cleaned_key] = v.clone()
        self.shadow = cleaned_sd

    def copy_to(self, fsdp_module):
        # load EMA weights into an (unwrapped) copy of the generator
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                cleaned_name = self._clean_param_name(n)
                if cleaned_name in self.shadow:
                    p.data.copy_(self.shadow[cleaned_name].to(p.dtype, device=p.device))


def _is_dtensor_parameter(parameter: torch.Tensor) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except (ImportError, AttributeError):
        return False
    return isinstance(parameter, DTensor)


def _local_parameter_tensor(
    parameter: torch.Tensor, *, writable: bool = False
) -> torch.Tensor:
    """Return a rank-local Tensor for Tensor or FSDP2 DTensor parameters."""

    if _is_dtensor_parameter(parameter):
        # DTensor.to_local() is an autograd view. Writes must start from .data
        # or PyTorch rejects the in-place operation as a custom-Function view.
        return parameter.data.to_local() if writable else parameter.detach().to_local()
    return parameter.data if writable else parameter.detach()


def _parameter_shard_metadata(parameter: torch.Tensor) -> dict[str, Any]:
    local_tensor = _local_parameter_tensor(parameter)
    metadata: dict[str, Any] = {
        "kind": "dtensor" if _is_dtensor_parameter(parameter) else "tensor",
        "global_shape": tuple(parameter.shape),
        "local_shape": tuple(local_tensor.shape),
        "dtype": str(parameter.dtype),
    }
    if not _is_dtensor_parameter(parameter):
        return metadata
    mesh = parameter.device_mesh
    coordinate = mesh.get_coordinate()
    metadata.update(
        {
            "mesh_device_type": str(mesh.device_type),
            "mesh_dim_names": tuple(mesh.mesh_dim_names or ()),
            "mesh_shape": tuple(mesh.mesh.shape),
            "rank_layout": tuple(
                tuple(int(rank) for rank in row)
                if isinstance(row, list)
                else (int(row),)
                for row in mesh.mesh.detach().cpu().tolist()
            ),
            "coordinate": None
            if coordinate is None
            else tuple(int(index) for index in coordinate),
            "placements": tuple(str(placement) for placement in parameter.placements),
        }
    )
    return metadata


class TrainableShardedEMA:
    """CPU FP32 EMA over only the local trainable LoRA parameter shards.

    Unlike :class:`EMA_FSDP`, this class never summons full parameters.  It is
    intended for an FSDP2 per-parameter DTensor model; empty local shards are
    valid and are retained in the topology metadata.

    ``update_after_step`` encodes the Stage-1 convention directly: after
    ``start_step`` completes, EMA is initialized from the raw LoRA shards; later
    completed optimizer steps apply decay.
    """

    schema_version = 2

    def __init__(
        self,
        module: torch.nn.Module,
        decay: float = 0.99,
        start_step: int = 75,
        *,
        require_lora_only: bool = True,
        topology: Mapping[str, Any] | None = None,
    ):
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        if int(start_step) < 1:
            raise ValueError(f"EMA start_step must be >= 1, got {start_step}")
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.require_lora_only = bool(require_lora_only)
        self.topology = dict(topology or {})
        self.shadow: dict[str, torch.Tensor] = {}
        self.local_shapes: dict[str, tuple[int, ...]] = {}
        self.global_shapes: dict[str, tuple[int, ...]] = {}
        self.shard_metadata: dict[str, dict[str, Any]] = {}
        self.last_completed_step: int | None = None
        self.initialized = False

        # Validate names and capture the expected post-FSDP local topology, but
        # do not create shadows before the configured completed optimizer step.
        parameters = self._trainable_parameters(module)
        self.local_shapes = {
            name: tuple(_local_parameter_tensor(parameter).shape)
            for name, parameter in parameters.items()
        }
        self.global_shapes = {
            name: tuple(parameter.shape) for name, parameter in parameters.items()
        }
        self.shard_metadata = {
            name: _parameter_shard_metadata(parameter)
            for name, parameter in parameters.items()
        }

    @staticmethod
    def _clean_param_name(name: str) -> str:
        return EMA_FSDP._clean_param_name(name)

    @staticmethod
    def _is_lora_parameter_name(name: str) -> bool:
        return ".lora_A." in name or ".lora_B." in name

    @staticmethod
    def _rank_and_world_size() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _trainable_parameters(self, module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
        parameters: dict[str, torch.nn.Parameter] = {}
        raw_names: dict[str, str] = {}
        for raw_name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            name = self._clean_param_name(raw_name)
            if name in parameters:
                raise ValueError(
                    "FSDP prefix cleanup produced duplicate trainable parameter name "
                    f"{name!r}: {raw_names[name]!r} and {raw_name!r}"
                )
            parameters[name] = parameter
            raw_names[name] = raw_name

        if not parameters:
            raise ValueError("TrainableShardedEMA found no trainable local parameters")
        if self.require_lora_only:
            non_lora = sorted(
                name for name in parameters if not self._is_lora_parameter_name(name)
            )
            if non_lora:
                raise ValueError(
                    "TrainableShardedEMA only accepts LoRA trainables; "
                    f"found {non_lora}"
                )
        return dict(sorted(parameters.items()))

    def _validate_local_topology(
        self, parameters: Mapping[str, torch.nn.Parameter]
    ) -> None:
        expected_names = set(self.local_shapes)
        actual_names = set(parameters)
        if expected_names != actual_names:
            raise ValueError(
                "EMA local trainable names changed: "
                f"missing={sorted(expected_names - actual_names)}, "
                f"extra={sorted(actual_names - expected_names)}"
            )
        mismatched = {
            name: (
                self.local_shapes[name],
                tuple(_local_parameter_tensor(parameters[name]).shape),
            )
            for name in sorted(expected_names)
            if self.local_shapes[name]
            != tuple(_local_parameter_tensor(parameters[name]).shape)
        }
        if mismatched:
            raise ValueError(f"EMA local shard shapes changed: {mismatched}")
        actual_global_shapes = {
            name: tuple(parameter.shape) for name, parameter in parameters.items()
        }
        if actual_global_shapes != self.global_shapes:
            raise ValueError(
                "EMA global parameter shapes changed: "
                f"expected={self.global_shapes}, actual={actual_global_shapes}"
            )
        actual_metadata = {
            name: _parameter_shard_metadata(parameter)
            for name, parameter in parameters.items()
        }
        if actual_metadata != self.shard_metadata:
            raise ValueError(
                "EMA FSDP2 shard/mesh topology changed: "
                f"expected={self.shard_metadata}, actual={actual_metadata}"
            )

    @staticmethod
    def _fp32_cpu_copy(name: str, parameter: torch.Tensor) -> torch.Tensor:
        value = (
            _local_parameter_tensor(parameter)
            .to(device="cpu", dtype=torch.float32)
            .clone()
        )
        if value.numel() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"non-finite local LoRA shard cannot update EMA: {name}")
        return value

    @torch.no_grad()
    def initialize(self, module: torch.nn.Module, *, completed_step: int) -> None:
        completed_step = int(completed_step)
        if self.initialized:
            raise RuntimeError("TrainableShardedEMA is already initialized")
        if completed_step != self.start_step:
            raise ValueError(
                "EMA must initialize exactly at its configured completed step: "
                f"expected {self.start_step}, got {completed_step}"
            )
        if self.last_completed_step is not None and completed_step <= self.last_completed_step:
            raise ValueError(
                f"EMA completed steps must increase: previous={self.last_completed_step}, "
                f"got={completed_step}"
            )
        parameters = self._trainable_parameters(module)
        self._validate_local_topology(parameters)
        self.shadow = {
            name: self._fp32_cpu_copy(name, parameter)
            for name, parameter in parameters.items()
        }
        self.initialized = True
        self.last_completed_step = completed_step

    @torch.no_grad()
    def update_after_step(self, module: torch.nn.Module, completed_step: int) -> str:
        """Advance EMA after one successful optimizer step.

        Returns ``"skipped"``, ``"initialized"``, or ``"updated"`` for
        diagnostics. Duplicate/out-of-order steps fail rather than double-decay.
        """

        completed_step = int(completed_step)
        if completed_step < 1:
            raise ValueError(f"completed_step must be >= 1, got {completed_step}")
        if self.last_completed_step is not None and completed_step <= self.last_completed_step:
            raise ValueError(
                f"EMA completed steps must increase: previous={self.last_completed_step}, "
                f"got={completed_step}"
            )
        if (
            self.initialized
            and self.last_completed_step is not None
            and completed_step != self.last_completed_step + 1
        ):
            raise ValueError(
                "EMA decay steps must be consecutive after initialization: "
                f"previous={self.last_completed_step}, got={completed_step}"
            )

        if completed_step < self.start_step:
            if self.initialized:
                raise RuntimeError("EMA cannot be initialized before start_step")
            self.last_completed_step = completed_step
            return "skipped"
        if completed_step == self.start_step:
            self.initialize(module, completed_step=completed_step)
            return "initialized"
        if not self.initialized:
            raise RuntimeError(
                "EMA start step was skipped; load the saved EMA state before continuing "
                f"or call at completed_step={self.start_step}"
            )

        parameters = self._trainable_parameters(module)
        self._validate_local_topology(parameters)
        if set(parameters) != set(self.shadow):
            raise ValueError("EMA shadow keys do not match current local trainable shards")
        # Stage all device-to-CPU copies and finite/shape checks before mutating
        # any shadow. A later non-finite shard must not partially decay earlier
        # keys, since the same completed step may then be retried.
        current_values = {
            name: self._fp32_cpu_copy(name, parameter)
            for name, parameter in parameters.items()
        }
        for name, current in current_values.items():
            shadow = self.shadow[name]
            if shadow.shape != current.shape:
                raise ValueError(
                    f"EMA shadow shape mismatch for {name}: "
                    f"expected {tuple(shadow.shape)}, got {tuple(current.shape)}"
                )
        one_minus_decay = 1.0 - self.decay
        for name, current in current_values.items():
            shadow = self.shadow[name]
            shadow.mul_(self.decay).add_(current, alpha=one_minus_decay)
        self.last_completed_step = completed_step
        return "updated"

    # Conventional alias for trainer integrations.
    update = update_after_step

    def state_dict(self) -> dict[str, Any]:
        rank, world_size = self._rank_and_world_size()
        return {
            "schema_version": self.schema_version,
            "decay": self.decay,
            "start_step": self.start_step,
            "last_completed_step": self.last_completed_step,
            "initialized": self.initialized,
            "rank": rank,
            "world_size": world_size,
            "topology": dict(self.topology),
            "local_shapes": {name: tuple(shape) for name, shape in self.local_shapes.items()},
            "global_shapes": {
                name: tuple(shape) for name, shape in self.global_shapes.items()
            },
            "shard_metadata": {
                name: dict(metadata) for name, metadata in self.shard_metadata.items()
            },
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], module: torch.nn.Module) -> None:
        required = {
            "schema_version",
            "decay",
            "start_step",
            "last_completed_step",
            "initialized",
            "rank",
            "world_size",
            "topology",
            "local_shapes",
            "global_shapes",
            "shard_metadata",
            "shadow",
        }
        missing = sorted(required - set(state_dict))
        extra = sorted(set(state_dict) - required)
        if missing or extra:
            raise ValueError(f"EMA state key mismatch: missing={missing}, extra={extra}")
        if int(state_dict["schema_version"]) != self.schema_version:
            raise ValueError(
                f"unsupported EMA schema version: {state_dict['schema_version']}"
            )
        if float(state_dict["decay"]) != self.decay:
            raise ValueError(
                f"EMA decay mismatch: checkpoint={state_dict['decay']}, current={self.decay}"
            )
        if int(state_dict["start_step"]) != self.start_step:
            raise ValueError(
                "EMA start_step mismatch: "
                f"checkpoint={state_dict['start_step']}, current={self.start_step}"
            )
        rank, world_size = self._rank_and_world_size()
        if int(state_dict["rank"]) != rank or int(state_dict["world_size"]) != world_size:
            raise ValueError(
                "EMA distributed topology mismatch: "
                f"checkpoint=(rank={state_dict['rank']}, world={state_dict['world_size']}), "
                f"current=(rank={rank}, world={world_size})"
            )
        checkpoint_topology = dict(state_dict["topology"])
        if checkpoint_topology != self.topology:
            raise ValueError(
                f"EMA logical topology mismatch: checkpoint={checkpoint_topology}, "
                f"current={self.topology}"
            )

        parameters = self._trainable_parameters(module)
        self._validate_local_topology(parameters)
        checkpoint_shapes = {
            str(name): tuple(shape) for name, shape in dict(state_dict["local_shapes"]).items()
        }
        if checkpoint_shapes != self.local_shapes:
            raise ValueError(
                f"EMA local shard topology mismatch: checkpoint={checkpoint_shapes}, "
                f"current={self.local_shapes}"
            )
        checkpoint_global_shapes = {
            str(name): tuple(shape)
            for name, shape in dict(state_dict["global_shapes"]).items()
        }
        if checkpoint_global_shapes != self.global_shapes:
            raise ValueError(
                "EMA global shape topology mismatch: "
                f"checkpoint={checkpoint_global_shapes}, current={self.global_shapes}"
            )
        checkpoint_shard_metadata = {
            str(name): dict(metadata)
            for name, metadata in dict(state_dict["shard_metadata"]).items()
        }
        if checkpoint_shard_metadata != self.shard_metadata:
            raise ValueError(
                "EMA FSDP2 shard/mesh topology mismatch: "
                f"checkpoint={checkpoint_shard_metadata}, current={self.shard_metadata}"
            )

        initialized = bool(state_dict["initialized"])
        shadows = dict(state_dict["shadow"])
        expected_shadow_names = set(self.local_shapes) if initialized else set()
        if set(shadows) != expected_shadow_names:
            raise ValueError(
                "EMA shadow key mismatch: "
                f"missing={sorted(expected_shadow_names - set(shadows))}, "
                f"extra={sorted(set(shadows) - expected_shadow_names)}"
            )
        validated_shadows: dict[str, torch.Tensor] = {}
        for name, value in shadows.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"EMA shadow {name!r} is not a tensor")
            if value.device.type != "cpu" or value.dtype != torch.float32:
                raise TypeError(
                    f"EMA shadow {name!r} must be CPU FP32, got {value.device}/{value.dtype}"
                )
            if tuple(value.shape) != self.local_shapes[name]:
                raise ValueError(
                    f"EMA shadow shape mismatch for {name}: "
                    f"expected {self.local_shapes[name]}, got {tuple(value.shape)}"
                )
            if value.numel() and not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"EMA shadow {name!r} contains non-finite values")
            validated_shadows[name] = value.detach().clone()

        last_completed_step = state_dict["last_completed_step"]
        if last_completed_step is not None:
            last_completed_step = int(last_completed_step)
        if initialized and (
            last_completed_step is None or last_completed_step < self.start_step
        ):
            raise ValueError("initialized EMA state has an invalid last_completed_step")
        if not initialized and last_completed_step is not None:
            if last_completed_step >= self.start_step:
                raise ValueError("uninitialized EMA state reached or passed start_step")

        self.shadow = validated_shadows
        self.initialized = initialized
        self.last_completed_step = last_completed_step

    @torch.no_grad()
    def copy_to(self, module: torch.nn.Module) -> None:
        """Copy local CPU EMA shadows into the matching local trainable shards."""

        if not self.initialized:
            raise RuntimeError("TrainableShardedEMA is not initialized")
        parameters = self._trainable_parameters(module)
        self._validate_local_topology(parameters)
        if set(parameters) != set(self.shadow):
            raise ValueError("EMA shadow keys do not match current local trainable shards")
        for name, parameter in parameters.items():
            local_parameter = _local_parameter_tensor(parameter, writable=True)
            local_parameter.copy_(
                self.shadow[name].to(
                    device=local_parameter.device, dtype=local_parameter.dtype
                )
            )

    @contextmanager
    def swap_into(self, module: torch.nn.Module):
        """Temporarily swap EMA into local shards and always restore raw values."""

        if not self.initialized:
            raise RuntimeError("TrainableShardedEMA is not initialized")
        parameters = self._trainable_parameters(module)
        self._validate_local_topology(parameters)
        raw = {
            name: _local_parameter_tensor(parameter).clone()
            for name, parameter in parameters.items()
        }
        try:
            self.copy_to(module)
            yield module
        finally:
            # Restore every local shard even when selective gather or validation
            # raises inside the context.
            for name, parameter in parameters.items():
                _local_parameter_tensor(parameter, writable=True).copy_(raw[name])
