"""Real six-rank CPU parity gate for the locked Stage-1 FSDP2 topology."""

from __future__ import annotations

from datetime import timedelta
import os

import pytest
import torch
from torch import nn

from utils.distributed import stage1_fsdp2_accumulation
from utils.stage1_loss import loss_for_backward


_WORLD_SIZE = 6
_SP_SIZE = 3
_DP_SIZE = 2
_ACCUMULATION_STEPS = 2
_LOCAL_COUNT = 2
_GLOBAL_COUNT_PER_SAMPLE = _SP_SIZE * _LOCAL_COUNT
_MESH_LAYOUT = ((0, 1, 2), (3, 4, 5))
_MESH_DIM_NAMES = ("replicate", "shard")


class _TinyStage1Parameter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([-0.35, 0.20, 0.45, -0.10, 0.30, -0.25])
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weight


def _sample_sp_chunk(sample_index: int, sp_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two deterministic observations owned by one logical SP rank."""

    rows = []
    targets = []
    for local_index in range(_LOCAL_COUNT):
        observation = sample_index * _GLOBAL_COUNT_PER_SAMPLE + sp_rank * 2 + local_index
        rows.append(
            [
                1.0 + 0.07 * observation,
                -0.4 + 0.03 * sample_index,
                0.2 * (sp_rank + 1),
                (-1.0) ** observation * 0.35,
                0.05 * (observation + 2),
                0.6 - 0.04 * local_index,
            ]
        )
        targets.append(-0.3 + 0.11 * observation + 0.05 * sample_index)
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(
        targets, dtype=torch.float32
    )


def _reference_gradient_and_update() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model = _TinyStage1Parameter()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-2,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    numerator = torch.zeros((), dtype=torch.float32)
    for sample_index in range(_DP_SIZE * _ACCUMULATION_STEPS):
        for sp_rank in range(_SP_SIZE):
            features, targets = _sample_sp_chunk(sample_index, sp_rank)
            numerator = numerator + (model(features) - targets).square().sum()
    loss = numerator / (
        (_DP_SIZE * _ACCUMULATION_STEPS) * _GLOBAL_COUNT_PER_SAMPLE
    )
    loss.backward()
    preclip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    gradient = model.weight.grad.detach().clone()
    optimizer.step()
    return preclip_norm.detach().clone(), gradient, model.weight.detach().clone()


def _run_six_rank_hsdp_parity(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo",
        rank=rank,
        world_size=_WORLD_SIZE,
        init_method=f"file://{rendezvous}",
        timeout=timedelta(seconds=90),
    )
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        from torch.distributed.tensor import DTensor, Replicate, Shard

        mesh = init_device_mesh(
            "cpu", (_DP_SIZE, _SP_SIZE), mesh_dim_names=_MESH_DIM_NAMES
        )
        assert tuple(tuple(row) for row in mesh.mesh.tolist()) == _MESH_LAYOUT
        assert tuple(mesh.mesh_dim_names) == _MESH_DIM_NAMES

        model = _TinyStage1Parameter()
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
            ),
        )
        assert isinstance(model.weight, DTensor)
        assert model.weight.placements == (Replicate(), Shard(0))
        assert tuple(model.weight.to_local().shape) == (2,)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-2,
            betas=(0.0, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
        optimizer.zero_grad()

        dp_rank = rank // _SP_SIZE
        sp_rank = rank % _SP_SIZE
        local_sample_ids = []
        for micro_step in range(_ACCUMULATION_STEPS):
            sample_index = micro_step * _DP_SIZE + dp_rank
            local_sample_ids.append(sample_index)
            features, targets = _sample_sp_chunk(sample_index, sp_rank)
            local_numerator = (model(features) - targets).square().sum()
            backward_loss = loss_for_backward(
                local_numerator,
                global_valid_count_per_sp_sample=_GLOBAL_COUNT_PER_SAMPLE,
                sequence_parallel_size=_SP_SIZE,
                gradient_accumulation_steps=_ACCUMULATION_STEPS,
            )
            with stage1_fsdp2_accumulation(
                model,
                sync_gradients=micro_step == _ACCUMULATION_STEPS - 1,
            ):
                backward_loss.backward()

        # Prove that rows represent DP samples and columns represent their SP
        # contributions for both accumulation microsteps.
        gathered_sample_ids: list[tuple[int, int] | None] = [None] * _WORLD_SIZE
        torch.distributed.all_gather_object(
            gathered_sample_ids, tuple(local_sample_ids)
        )
        assert gathered_sample_ids == [
            (0, 2),
            (0, 2),
            (0, 2),
            (1, 3),
            (1, 3),
            (1, 3),
        ]

        reference_norm, reference_gradient, reference_weight = (
            _reference_gradient_and_update()
        )
        assert isinstance(model.weight.grad, DTensor)
        distributed_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=0.5
        )
        if isinstance(distributed_norm, DTensor):
            distributed_norm = distributed_norm.full_tensor()
        torch.testing.assert_close(
            distributed_norm,
            reference_norm,
            rtol=2e-5,
            atol=2e-6,
        )
        distributed_gradient = model.weight.grad.full_tensor()
        torch.testing.assert_close(
            distributed_gradient,
            reference_gradient,
            rtol=2e-5,
            atol=2e-6,
        )

        optimizer.step()
        distributed_weight = model.weight.full_tensor()
        torch.testing.assert_close(
            distributed_weight,
            reference_weight,
            rtol=2e-5,
            atol=2e-6,
        )
    finally:
        torch.distributed.destroy_process_group()


def test_real_six_rank_gloo_hsdp_gradient_and_adamw_update_parity(tmp_path):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is required for the six-rank FSDP2 parity gate")
    if torch.distributed.is_initialized():
        pytest.skip("test requires ownership of the process-group lifecycle")
    rendezvous = tmp_path / "six_rank_hsdp_rendezvous"
    try:
        torch.multiprocessing.start_processes(
            _run_six_rank_hsdp_parity,
            args=(str(rendezvous),),
            nprocs=_WORLD_SIZE,
            join=True,
            # `fork` is unsafe after another test initializes Apple's MPS/
            # Objective-C runtime. `spawn` also matches torchrun's clean
            # interpreter semantics and keeps this gate order-independent.
            start_method="spawn",
        )
    finally:
        if rendezvous.exists():
            os.unlink(rendezvous)


def test_omitting_sp_multiplier_makes_gradient_exactly_three_times_too_small():
    # Two micros × DP2 × SP3 local derivative contributions. FSDP/HSDP averages
    # across all six ranks, so omitting the explicit SP3 factor retains an
    # unwanted division by three even after accumulation scaling.
    contributions = torch.arange(1, 13, dtype=torch.float64)
    reference = contributions.sum() / (
        (_DP_SIZE * _ACCUMULATION_STEPS) * _GLOBAL_COUNT_PER_SAMPLE
    )
    without_sp_multiplier = contributions.sum() / (
        _WORLD_SIZE * _ACCUMULATION_STEPS * _GLOBAL_COUNT_PER_SAMPLE
    )
    with_sp_multiplier = without_sp_multiplier * _SP_SIZE
    assert with_sp_multiplier.item() == pytest.approx(reference.item())
    assert without_sp_multiplier.item() == pytest.approx(
        reference.item() / _SP_SIZE
    )
