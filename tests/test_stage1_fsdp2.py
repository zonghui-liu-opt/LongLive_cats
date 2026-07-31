from __future__ import annotations

import os

import pytest
import torch
from torch import nn

import utils.distributed as distributed_utils
from utils.distributed import (
    STAGE1_MIN_H100_MEMORY_BYTES,
    STAGE1_FSDP2_MESH_DIM_NAMES,
    STAGE1_FSDP2_RANK_LAYOUT,
    TrainableShardedEMA,
    build_stage1_fsdp2_device_mesh,
    fsdp2_wrap_stage1,
    stage1_fsdp2_accumulation,
    validate_stage1_fsdp2_api,
    validate_stage1_fsdp2_topology,
)


class FakeMesh:
    def __init__(self):
        self.mesh = torch.tensor(STAGE1_FSDP2_RANK_LAYOUT)
        self.mesh_dim_names = STAGE1_FSDP2_MESH_DIM_NAMES
        self.shard_group = object()
        self.replicate_group = object()

    def get_group(self, name):
        return {
            "shard": self.shard_group,
            "replicate": self.replicate_group,
        }[name]


def test_stage1_fsdp2_api_and_locked_topology_fail_fast():
    validate_stage1_fsdp2_api()
    with pytest.raises(RuntimeError, match="requires PyTorch >=2.8.0"):
        validate_stage1_fsdp2_api(torch_version="2.7.1")

    validate_stage1_fsdp2_topology(world_size=6, rank=5)
    with pytest.raises(RuntimeError, match="WORLD_SIZE=6"):
        validate_stage1_fsdp2_topology(world_size=3, rank=0)
    with pytest.raises(RuntimeError, match="rank must be"):
        validate_stage1_fsdp2_topology(world_size=6, rank=6)


def _h100_rank_descriptors():
    return [
        {
            "rank": rank,
            "local_rank": rank,
            "hostname": "single-h100-host",
            "coordinate": (rank // 3, rank % 3),
            "visible_device_count": 6,
            "device_name": "NVIDIA H100 80GB HBM3",
            "total_memory": STAGE1_MIN_H100_MEMORY_BYTES,
            "capability": (9, 0),
            "bf16_supported": True,
        }
        for rank in range(6)
    ]


def test_cross_rank_h100_80g_capability_audit_is_strict():
    descriptors = _h100_rank_descriptors()
    distributed_utils._validate_stage1_fsdp2_rank_descriptors(
        descriptors, topology=distributed_utils.STAGE1_FSDP2_TOPOLOGY
    )

    descriptors[4]["device_name"] = "NVIDIA A100-SXM4-80GB"
    descriptors[5]["total_memory"] = STAGE1_MIN_H100_MEMORY_BYTES - 1
    with pytest.raises(RuntimeError, match="rank4: device is not H100") as error:
        distributed_utils._validate_stage1_fsdp2_rank_descriptors(
            descriptors, topology=distributed_utils.STAGE1_FSDP2_TOPOLOGY
        )
    assert "rank5: CUDA memory" in str(error.value)

    descriptors = _h100_rank_descriptors()
    descriptors[0]["visible_device_count"] = 8
    with pytest.raises(RuntimeError, match="visible CUDA device count=8"):
        distributed_utils._validate_stage1_fsdp2_rank_descriptors(
            descriptors, topology=distributed_utils.STAGE1_FSDP2_TOPOLOGY
        )

    descriptors = _h100_rank_descriptors()
    descriptors[5]["hostname"] = "second-host"
    with pytest.raises(RuntimeError, match="locked to one machine"):
        distributed_utils._validate_stage1_fsdp2_rank_descriptors(
            descriptors, topology=distributed_utils.STAGE1_FSDP2_TOPOLOGY
        )


def test_build_stage1_mesh_uses_exact_row_major_layout(monkeypatch):
    fake_mesh = FakeMesh()
    captured = {}

    def fake_init_device_mesh(device_type, shape, *, mesh_dim_names):
        captured.update(
            device_type=device_type,
            shape=shape,
            mesh_dim_names=mesh_dim_names,
        )
        return fake_mesh

    monkeypatch.setattr(
        distributed_utils, "validate_stage1_fsdp2_runtime", lambda **kwargs: None
    )
    monkeypatch.setattr(
        distributed_utils,
        "_import_fsdp2_api",
        lambda: {"init_device_mesh": fake_init_device_mesh},
    )
    monkeypatch.setattr(
        distributed_utils.dist,
        "get_world_size",
        lambda group=None: 3 if group is fake_mesh.shard_group else 2,
    )

    assert build_stage1_fsdp2_device_mesh() is fake_mesh
    assert captured == {
        "device_type": "cuda",
        "shape": (2, 3),
        "mesh_dim_names": ("replicate", "shard"),
    }


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(
            torch.zeros(2, 2, dtype=torch.bfloat16), requires_grad=False
        )
        self.lora_A = nn.ParameterDict(
            {"default": nn.Parameter(torch.ones(2, 1, dtype=torch.float32))}
        )
        self.lora_B = nn.ParameterDict(
            {"default": nn.Parameter(torch.zeros(1, 2, dtype=torch.float32))}
        )


class TinyStage1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Parameter(
            torch.zeros(1, dtype=torch.bfloat16), requires_grad=False
        )
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])


class FakeMixedPrecisionPolicy:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_fsdp2_wrap_is_bottom_up_and_preserves_fp32_lora(monkeypatch):
    model = TinyStage1Model()
    calls = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))
        return module

    monkeypatch.setattr(
        distributed_utils, "validate_stage1_fsdp2_api", lambda **kwargs: None
    )
    monkeypatch.setattr(
        distributed_utils,
        "_import_fsdp2_api",
        lambda: {
            "fully_shard": fake_fully_shard,
            "MixedPrecisionPolicy": FakeMixedPrecisionPolicy,
        },
    )

    result = fsdp2_wrap_stage1(
        model,
        transformer=model,
        mesh=FakeMesh(),
        expected_blocks=2,
        block_class_name="TinyBlock",
        expected_trainable_tensors=4,
        expected_trainable_numel=8,
    )

    assert result is model
    assert [module for module, _ in calls] == [*model.blocks, model]
    assert [kwargs["reshard_after_forward"] for _, kwargs in calls] == [
        True,
        True,
        False,
    ]
    for _, kwargs in calls:
        policy = kwargs["mp_policy"]
        assert policy.param_dtype == torch.bfloat16
        assert policy.reduce_dtype == torch.float32
        assert policy.cast_forward_inputs is True
    assert all(
        parameter.dtype == torch.float32
        for parameter in model.parameters()
        if parameter.requires_grad
    )


class FakeFSDP2Root:
    def __init__(self):
        self.calls = []

    def set_requires_gradient_sync(self, value, *, recurse):
        self.calls.append(("sync", value, recurse))

    def set_reshard_after_backward(self, value, *, recurse):
        self.calls.append(("reshard", value, recurse))

    def set_is_last_backward(self, value):
        self.calls.append(("last", value))


def test_fsdp2_accumulation_controls_and_restores_state(monkeypatch):
    monkeypatch.setattr(
        distributed_utils,
        "_import_fsdp2_api",
        lambda: {"FSDPModule": FakeFSDP2Root},
    )
    module = FakeFSDP2Root()
    with pytest.raises(RuntimeError, match="microbatch failed"):
        with stage1_fsdp2_accumulation(module, sync_gradients=False):
            raise RuntimeError("microbatch failed")
    assert module.calls == [
        ("sync", False, True),
        ("reshard", True, True),
        ("last", False),
        ("sync", True, True),
        ("reshard", True, True),
        ("last", True),
    ]


class TinyMixedDtypeFSDP2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(
            torch.eye(4, dtype=torch.bfloat16), requires_grad=False
        )
        self.lora_A = nn.ParameterDict(
            {"default": nn.Parameter(torch.ones(4, 2, dtype=torch.float32))}
        )
        self.lora_B = nn.ParameterDict(
            {"default": nn.Parameter(torch.zeros(2, 4, dtype=torch.float32))}
        )

    def forward(self, inputs):
        delta = self.lora_A["default"] @ self.lora_B["default"]
        return inputs @ (self.base + delta)


def _run_two_rank_fsdp2_mixed_dtype(rank, rendezvous):
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=f"file://{rendezvous}",
    )
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        from torch.distributed.tensor import DTensor

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("shard",))
        model = TinyMixedDtypeFSDP2Model()
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert all(
            isinstance(parameter, DTensor) and parameter.dtype == torch.float32
            for parameter in trainable
        )
        optimizer = torch.optim.AdamW(trainable, lr=1e-2)
        optimizer.zero_grad()
        with stage1_fsdp2_accumulation(model, sync_gradients=False):
            model(torch.ones(2, 4, dtype=torch.bfloat16)).float().sum().backward()
        with stage1_fsdp2_accumulation(model, sync_gradients=True):
            model(torch.ones(2, 4, dtype=torch.bfloat16)).float().sum().backward()
        optimizer.step()
        assert all(parameter.dtype == torch.float32 for parameter in trainable)
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_gloo_fsdp2_accepts_bf16_base_fp32_lora_and_accumulation(
    tmp_path,
):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is required for the tiny FSDP2 test")
    if torch.distributed.is_initialized():
        pytest.skip("test requires ownership of the process group lifecycle")
    rendezvous = tmp_path / "fsdp2_two_rank_rendezvous"
    torch.multiprocessing.start_processes(
        _run_two_rank_fsdp2_mixed_dtype,
        args=(str(rendezvous),),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    if rendezvous.exists():
        os.unlink(rendezvous)


def test_trainable_sharded_ema_uses_dtensor_local_shards(tmp_path):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is required for the CPU DTensor test")
    if torch.distributed.is_initialized():
        pytest.skip("test requires ownership of the process group lifecycle")

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    rendezvous = tmp_path / "dtensor_rendezvous"
    torch.distributed.init_process_group(
        "gloo",
        rank=0,
        world_size=1,
        init_method=f"file://{rendezvous}",
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("shard",))
        module = nn.Module()
        module.layer = nn.Module()
        module.layer.lora_A = nn.ParameterDict(
            {
                "default": nn.Parameter(
                    distribute_tensor(
                        torch.full((4, 3), 2.0), mesh, placements=(Shard(0),)
                    )
                )
            }
        )
        ema = TrainableShardedEMA(module, decay=0.5, start_step=1)
        assert ema.update_after_step(module, 1) == "initialized"
        name, parameter = next(module.named_parameters())
        assert ema.local_shapes[name] == tuple(parameter.to_local().shape)
        assert ema.global_shapes[name] == (4, 3)
        assert ema.shard_metadata[name]["kind"] == "dtensor"
        assert ema.shard_metadata[name]["mesh_dim_names"] == ("shard",)
        assert ema.shard_metadata[name]["placements"] == ("S(0)",)

        parameter.data.to_local().fill_(6.0)
        with ema.swap_into(module):
            assert torch.equal(
                parameter.to_local(), torch.full_like(parameter.to_local(), 2.0)
            )
        assert torch.equal(
            parameter.to_local(), torch.full_like(parameter.to_local(), 6.0)
        )

        state = ema.state_dict()
        assert state["schema_version"] == 2
        assert state["shard_metadata"] == ema.shard_metadata
    finally:
        torch.distributed.destroy_process_group()
        if rendezvous.exists():
            os.unlink(rendezvous)
