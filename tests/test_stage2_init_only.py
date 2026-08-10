from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from utils.stage2_fsdp2 import (
    STAGE2_FSDP2_MESH_DIM_NAMES,
    STAGE2_FSDP2_MESH_SHAPE,
    audit_stage2_fsdp2_role,
    audit_stage2_runtime_descriptors,
    fsdp2_wrap_stage2_role,
    validate_stage2_fsdp2_topology,
)
from utils.stage2_role_init import stage2_init_only_side_effect_guard


def _descriptors():
    return [
        {
            "rank": rank,
            "local_rank": rank,
            "hostname": "one-host",
            "visible_device_count": 8,
            "device_name": "NVIDIA H100 80GB HBM3",
            "total_memory": 80 * 1024**3,
            "bf16_supported": True,
        }
        for rank in range(8)
    ]


def test_stage2_topology_is_single_node_world8_one_dimensional_full_shard():
    audit = validate_stage2_fsdp2_topology(
        world_size=8,
        sequence_parallel_size=1,
        data_parallel_size=8,
        mesh_shape=STAGE2_FSDP2_MESH_SHAPE,
        mesh_dim_names=STAGE2_FSDP2_MESH_DIM_NAMES,
    )
    assert audit["sharding_strategy"] == "FULL_SHARD"
    assert audit["mesh_shape"] == (8,)
    assert audit["mesh_dim_names"] == ("shard",)
    audit_stage2_runtime_descriptors(_descriptors())


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda rows: rows[7].__setitem__("hostname", "other"), "single host"),
        (lambda rows: rows[0].__setitem__("device_name", "A100"), "H100"),
        (lambda rows: rows[0].__setitem__("bf16_supported", False), "BF16"),
        (
            lambda rows: rows[0].__setitem__("total_memory", 40 * 1024**3),
            "memory",
        ),
    ],
)
def test_runtime_descriptor_audit_rejects_wrong_hardware(mutation, match):
    rows = _descriptors()
    mutation(rows)
    with pytest.raises(RuntimeError, match=match):
        audit_stage2_runtime_descriptors(rows)


class _CausalWanAttentionBlock(nn.Module):
    pass


class _Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_CausalWanAttentionBlock() for _ in range(30)])
        self.base = nn.Parameter(
            torch.zeros(1, dtype=torch.bfloat16), requires_grad=False
        )
        self.lora_A = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(2, dtype=torch.float32))


class _Mesh:
    mesh = torch.arange(8)
    mesh_dim_names = ("shard",)


def test_fsdp_role_wrap_is_bottom_up_root_last_and_preserves_trainables():
    calls = []

    def fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    root = _Root()
    wrapped = fsdp2_wrap_stage2_role(
        root,
        transformer=root,
        mesh=_Mesh(),
        role="generator",
        block_class_name="_CausalWanAttentionBlock",
        expected_trainable_tensors=2,
        expected_trainable_parameters=4,
        fsdp_api={
            "fully_shard": fully_shard,
            "MixedPrecisionPolicy": lambda **kwargs: kwargs,
        },
    )
    assert wrapped is root
    assert [module for module, _ in calls[:-1]] == list(root.blocks)
    assert calls[-1][0] is root
    assert all(call[1]["reshard_after_forward"] is True for call in calls)
    assert all(
        call[1]["mp_policy"]["cast_forward_inputs"] is True for call in calls[:-1]
    )
    assert calls[-1][1]["mp_policy"]["cast_forward_inputs"] is False
    assert calls[-1][1]["mp_policy"]["reduce_dtype"] is torch.float32


def test_post_fsdp_audit_does_not_claim_frozen_base_is_sharded_without_dtensor():
    root = _Root().requires_grad_(False)
    with pytest.raises(RuntimeError, match="parameter is not DTensor"):
        audit_stage2_fsdp2_role(
            root,
            role="real_score",
            expected_schema=None,
            expected_fsdp_modules=1,
            fsdp_module_type=_Root,
        )


def test_init_only_side_effect_guard_is_measured_and_restored():
    layer = nn.Linear(2, 2)
    with stage2_init_only_side_effect_guard() as clean:
        pass
    assert clean["forward_calls"] == 0
    assert clean["optimizer_created"] is False
    assert clean["text_encoder_created"] is False
    assert clean["vae_created"] is False
    assert clean["dataloader_created"] is False
    assert layer(torch.ones(1, 2)).shape == (1, 2)

    with pytest.raises(RuntimeError, match="module forward"):
        with stage2_init_only_side_effect_guard() as attempted:
            layer(torch.ones(1, 2))
    assert attempted["forward_calls"] == 1

    with pytest.raises(RuntimeError, match="direct module forward"):
        with stage2_init_only_side_effect_guard() as attempted_direct:
            layer.forward(torch.ones(1, 2))
    assert attempted_direct["forward_calls"] == 1


def test_init_only_guard_blocks_every_forbidden_constructor():
    import torch.utils.data
    import utils.distributed as distributed_utils
    import utils.wan_5b_wrapper as wan_wrapper

    cases = (
        (lambda: torch.optim.Optimizer([], {}), "optimizer_created", "optimizer"),
        (lambda: distributed_utils.EMA_FSDP(), "ema_created", "EMA_FSDP"),
        (
            lambda: wan_wrapper.WanTextEncoder(),
            "text_encoder_created",
            "WanTextEncoder",
        ),
        (lambda: wan_wrapper.WanVAEWrapper(), "vae_created", "WanVAEWrapper"),
        (lambda: torch.utils.data.DataLoader([]), "dataloader_created", "DataLoader"),
    )
    for invoke, field, match in cases:
        with pytest.raises(RuntimeError, match=match):
            with stage2_init_only_side_effect_guard() as audit:
                invoke()
        assert audit[field] is True


def test_init_only_cli_help_is_lazy_and_does_not_import_training_stack():
    root = Path(__file__).resolve().parents[1]
    program = """
import runpy
import sys
sys.argv = ['preflight_stage2_roles.py', '--help']
try:
    runpy.run_path('scripts/preflight_stage2_roles.py', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
for name in ('trainer.distillation', 'model.dmd', 'wan_5b.modules.t5', 'wan_5b.modules.vae2_2'):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout


def test_init_only_cli_bootstraps_project_root_outside_repo(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "preflight_stage2_roles.py"),
            "--help",
        ],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout
