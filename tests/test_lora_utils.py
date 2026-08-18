from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from utils.lora_utils import (
    LocalLoraShard,
    LoraTensorSpec,
    assert_lora_b_weights_zero,
    audit_fsdp2_lora_dtensor_topology,
    audit_lora_model,
    build_lora_shard_schema,
    consolidate_lora_shards,
    configure_lora_for_model,
    gather_fsdp2_lora_state_dict,
    get_canonical_lora_state_dict,
    get_lora_sharded_state_dict,
    load_lora_safetensors_strict,
    resolve_lora_target_modules,
    save_lora_safetensors_strict,
    strict_load_lora_state_dict,
    validate_lora_replica_shards,
)


class TinySelfAttention(nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.o = nn.Linear(width, width)


class TinyCrossAttention(nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.q = nn.Linear(width, width)


class TinyBlock(nn.Module):
    def __init__(self, width: int = 4, hidden: int = 8):
        super().__init__()
        self.self_attn = TinySelfAttention(width)
        self.cross_attn = TinyCrossAttention(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )


class TinyTransformer(nn.Module):
    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock() for _ in range(num_blocks)])
        self.head = nn.Linear(4, 4)

    def forward(self, value):
        return self.head(value)


class CausalWanAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 4)
        self.k = nn.Linear(4, 4)


class LegacyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = CausalWanAttentionBlock()
        self.unrelated = nn.Linear(4, 4)


def exact_config(**overrides):
    # Per block: four 4->4 attention adapters (4 * 16 params) plus
    # 4->8 and 8->4 FFN adapters (24 params each) = 112 params.
    config = {
        "type": "lora",
        "rank": 2,
        "alpha": 2,
        "dropout": 0.0,
        "bias": "none",
        "modules_to_save": [],
        "target_patterns": [
            r"^blocks\.[0-9]+\.self_attn\.(q|k|v|o)$",
            r"^blocks\.[0-9]+\.ffn\.(0|2)$",
        ],
        "expected_target_modules": 12,
        "expected_trainable_parameters": 224,
        "expected_adapter_tensors": 24,
    }
    config.update(overrides)
    return config


def build_exact_lora(**overrides):
    model = TinyTransformer()
    model.requires_grad_(False)
    return configure_lora_for_model(
        model,
        "generator",
        exact_config(**overrides),
        is_main_process=False,
    )


def clone_state(state):
    return OrderedDict((key, value.detach().clone()) for key, value in state.items())


def _fsdp2_lora_worker(rank: int, init_file: str):
    dist.init_process_group(
        "gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=6,
    )
    original_to_local = DTensor.to_local
    try:
        torch.manual_seed(1234)
        model = build_exact_lora()
        expected = clone_state(get_canonical_lora_state_dict(model))
        schema = build_lora_shard_schema(model)
        for parameter in model.parameters():
            if not parameter.requires_grad:
                parameter.data = parameter.data.to(torch.bfloat16)

        mesh = DeviceMesh(
            "cpu",
            torch.arange(6, dtype=torch.int32).reshape(2, 3),
            mesh_dim_names=("replicate", "shard"),
        )
        fully_shard(model, mesh=mesh)
        assert all(
            parameter.dtype == torch.bfloat16
            for parameter in model.parameters()
            if not parameter.requires_grad
        )
        assert all(
            parameter.dtype == torch.float32
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        # A frozen base DTensor must never even expose its local tensor to this
        # adapter-only path. This catches accidental whole-model enumeration.
        def guarded_to_local(self, *args, **kwargs):
            if not self.requires_grad:
                raise AssertionError("frozen base DTensor was materialized")
            return original_to_local(self, *args, **kwargs)

        DTensor.to_local = guarded_to_local
        audit = audit_fsdp2_lora_dtensor_topology(
            model,
            expected_schema=schema,
        )
        assert audit["adapter_tensor_count"] == 24
        assert audit["mesh_shape"] == (2, 3)
        assert audit["placements"] == ("replicate", "shard:0")
        local_shards = get_lora_sharded_state_dict(
            model,
            expected_schema=schema,
            expected_adapter_tensors=24,
        )
        DTensor.to_local = original_to_local

        assert len(local_shards) == 24
        assert all(record.is_dtensor for record in local_shards.values())
        assert all(record.mesh_shape == (2, 3) for record in local_shards.values())
        assert all(
            record.mesh_coordinate == (rank // 3, rank % 3)
            for record in local_shards.values()
        )
        result = gather_fsdp2_lora_state_dict(
            local_shards,
            authoritative_shard_group=mesh.get_group(1),
            replica_group=mesh.get_group(0),
            expected_schema=schema,
        )
        if rank == 0:
            assert result is not None
            assert tuple(result) == tuple(expected)
            assert all(
                torch.equal(result[key], expected[key].cpu()) for key in expected
            )
        else:
            assert result is None
    finally:
        DTensor.to_local = original_to_local
        if dist.is_initialized():
            dist.destroy_process_group()


def _synthetic_dtensor_shard(replica: int, shard: int) -> LocalLoraShard:
    global_tensor = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    row_sizes = (2, 2, 1)
    row_offsets = (0, 2, 4)
    offset = row_offsets[shard]
    local = global_tensor[offset : offset + row_sizes[shard]].clone()
    return LocalLoraShard(
        tensor=local,
        global_shape=(5, 2),
        intra_param_start=offset * 2,
        shard_rank=shard,
        shard_world_size=3,
        shard_group_ranks=(0, 1, 2) if replica == 0 else (3, 4, 5),
        fsdp_unit_fingerprint="same-schema",
        local_shape=tuple(local.shape),
        mesh_shape=(2, 3),
        mesh_dim_names=("replicate", "shard"),
        placements=("replicate", "shard:0"),
        mesh_coordinate=(replica, shard),
        mesh_ranks=((0, 1, 2), (3, 4, 5)),
        replica_group_ranks=(shard, shard + 3),
        authoritative_shard_group_ranks=(0, 1, 2),
        shard_dim=0,
        shard_offset=offset,
        global_rank=replica * 3 + shard,
        is_dtensor=True,
    )


def test_exact_patterns_are_full_name_sorted_and_exclude_cross_attention():
    base = TinyTransformer()
    names = resolve_lora_target_modules(base, "generator", exact_config())

    assert names == tuple(sorted(names))
    assert len(names) == 12
    assert all("cross_attn" not in name for name in names)
    assert all("head" not in name for name in names)

    lora_model = build_exact_lora()
    audit = audit_lora_model(
        lora_model,
        target_module_names=names,
        expected_target_modules=12,
        expected_trainable_parameters=224,
        expected_adapter_tensors=24,
        require_lora_only=True,
        check_b_zero=True,
    )
    assert audit["target_module_count"] == 12
    assert audit["trainable_parameter_count"] == 224
    assert audit["adapter_tensor_count"] == 24
    assert all("cross_attn" not in name for name in audit["trainable_parameter_names"])


def test_exact_patterns_fail_on_unmatched_or_non_linear_and_wrong_audit():
    model = TinyTransformer()
    with pytest.raises(ValueError, match="matched no modules"):
        resolve_lora_target_modules(
            model,
            "generator",
            exact_config(target_patterns=[r"^blocks\.99\.self_attn\.q$"]),
        )
    with pytest.raises(TypeError, match="non-linear"):
        resolve_lora_target_modules(
            model,
            "generator",
            exact_config(target_patterns=[r"^blocks\.0\.ffn\.1$"]),
        )
    with pytest.raises(ValueError, match="unexpected LoRA target modules"):
        build_exact_lora(expected_target_modules=11)


def test_legacy_scan_without_patterns_is_preserved():
    names = resolve_lora_target_modules(
        LegacyTransformer(),
        "generator",
        {"type": "lora", "rank": 2},
    )
    assert names == ("attn.k", "attn.q")


def test_lora_b_zero_audit_detects_mutation():
    model = build_exact_lora()
    b_names = assert_lora_b_weights_zero(model)
    assert len(b_names) == 12
    first_b = next(
        parameter for name, parameter in model.named_parameters() if ".lora_B." in name
    )
    first_b.data.flatten()[0] = 1.0
    with pytest.raises(ValueError, match="nonzero tensors"):
        assert_lora_b_weights_zero(model)


def test_strict_safetensors_roundtrip(tmp_path):
    source = build_exact_lora()
    generator = torch.Generator().manual_seed(123)
    for parameter in source.parameters():
        if parameter.requires_grad:
            parameter.data.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            )

    path = tmp_path / "adapter.safetensors"
    saved = save_lora_safetensors_strict(
        source,
        path,
        metadata={"schema": 1, "kind": "raw"},
    )
    assert path.is_file()
    assert all(value.device.type == "cpu" for value in saved.values())
    assert all(value.dtype == torch.float32 for value in saved.values())

    target = build_exact_lora()
    loaded = load_lora_safetensors_strict(target, path)
    extracted = get_canonical_lora_state_dict(target)
    assert tuple(loaded) == tuple(extracted)
    for key in loaded:
        assert torch.equal(loaded[key], extracted[key].cpu())


def test_strict_load_does_not_import_optional_transformers_tensor_parallel(
    monkeypatch,
):
    source = build_exact_lora()
    generator = torch.Generator().manual_seed(987)
    for parameter in source.parameters():
        if parameter.requires_grad:
            parameter.data.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            )
    state = clone_state(get_canonical_lora_state_dict(source))
    target = build_exact_lora()

    import builtins

    original_import = builtins.__import__

    def reject_optional_tensor_parallel(name, *args, **kwargs):
        if name == "transformers.integrations.tensor_parallel":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(builtins, "__import__", reject_optional_tensor_parallel)

    strict_load_lora_state_dict(target, state, expected_dtype=torch.float32)

    loaded = get_canonical_lora_state_dict(target)
    assert tuple(loaded) == tuple(state)
    for key, expected in state.items():
        assert torch.equal(loaded[key], expected)


@pytest.mark.parametrize("failure", ["missing", "extra", "shape", "dtype", "nan"])
def test_strict_adapter_validation_rejects_schema_and_value_errors(failure):
    model = build_exact_lora()
    state = clone_state(get_canonical_lora_state_dict(model))
    first_key = next(iter(state))

    if failure == "missing":
        del state[first_key]
        match = "adapter key mismatch"
    elif failure == "extra":
        state["base_model.model.extra.lora_A.weight"] = torch.zeros(2, 2)
        match = "adapter key mismatch"
    elif failure == "shape":
        state[first_key] = torch.zeros(1, dtype=state[first_key].dtype)
        match = "shape mismatch"
    elif failure == "dtype":
        state[first_key] = state[first_key].to(torch.bfloat16)
        match = "dtype mismatch"
    else:
        state[first_key].flatten()[0] = torch.nan
        match = "non-finite"

    with pytest.raises((ValueError, TypeError), match=match):
        strict_load_lora_state_dict(model, state)


def test_selective_state_returns_only_canonical_trainable_lora_tensors(monkeypatch):
    model = build_exact_lora()

    # The Stage-1 API must not touch the legacy FULL_STATE_DICT context.
    def forbidden_full_state(*args, **kwargs):
        raise AssertionError("full FSDP state dict must not be used")

    monkeypatch.setattr(
        "utils.lora_utils.FSDP.state_dict_type",
        forbidden_full_state,
    )
    expected = get_canonical_lora_state_dict(model)
    selective = get_lora_sharded_state_dict(
        model,
        expected_adapter_tensors=24,
        expected_keys=expected.keys(),
        expected_shapes={key: value.shape for key, value in expected.items()},
    )
    assert tuple(selective) == tuple(expected)
    assert all("lora_" in key for key in selective)
    assert all(value.dtype == torch.float32 for value in selective.values())


def test_selective_state_fails_if_non_lora_parameter_is_trainable():
    model = build_exact_lora()
    base_parameter = next(
        parameter
        for name, parameter in model.named_parameters()
        if "base_layer.weight" in name
    )
    base_parameter.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable non-LoRA"):
        get_lora_sharded_state_dict(model)


def test_selective_state_rejects_inner_module_with_fsdp_local_shards():
    model = build_exact_lora()
    first_trainable = next(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    first_trainable._fsdp_flattened = True
    with pytest.raises(RuntimeError, match="pass the outer FSDP root"):
        get_lora_sharded_state_dict(model)


def test_replica_validation_and_authoritative_consolidation_are_strict():
    key = "base_model.model.block.lora_A.weight"
    replica_zero = [{key: _synthetic_dtensor_shard(0, shard)} for shard in range(3)]
    replica_one = [{key: _synthetic_dtensor_shard(1, shard)} for shard in range(3)]
    for shard in range(3):
        summary = validate_lora_replica_shards(
            (replica_zero[shard], replica_one[shard])
        )
        assert summary["shard_rank"] == shard

    schema = OrderedDict(
        ((key, LoraTensorSpec("block.lora_A.default.weight", (5, 2), torch.float32)),)
    )
    consolidated = consolidate_lora_shards(replica_zero, expected_schema=schema)
    assert torch.equal(
        consolidated[key],
        torch.arange(10, dtype=torch.float32).reshape(5, 2),
    )

    changed = replace(
        replica_one[1][key],
        tensor=replica_one[1][key].tensor + 1,
    )
    with pytest.raises(ValueError, match="values differ"):
        validate_lora_replica_shards((replica_zero[1], {key: changed}))

    with pytest.raises(ValueError, match="authoritative"):
        consolidate_lora_shards(replica_one, expected_schema=schema)


def test_fsdp2_mixed_precision_selective_gather_never_materializes_base(tmp_path):
    init_file = tmp_path / "fsdp2_lora_gloo_init"
    mp.spawn(
        _fsdp2_lora_worker,
        args=(str(init_file),),
        nprocs=6,
        join=True,
    )
