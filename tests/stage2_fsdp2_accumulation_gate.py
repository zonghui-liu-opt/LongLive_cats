"""Real FSDP2 accumulation parity gate for torchrun.

Local/CI seam (closest real):
    torchrun --standalone --nproc-per-node=2 \
        tests/stage2_fsdp2_accumulation_gate.py

Release H100 gate (exact Stage-2 topology and both locked profiles):
    torchrun --standalone --nproc-per-node=8 \
        tests/stage2_fsdp2_accumulation_gate.py --require-h100
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402 - isolated torchrun bootstrap
import torch.distributed as dist  # noqa: E402 - isolated torchrun bootstrap

from utils.distributed import (  # noqa: E402 - isolated torchrun bootstrap
    fsdp2_accumulation,
)
from utils.stage2_train_state import (  # noqa: E402 - isolated torchrun bootstrap
    stage2_global_mean_loss_for_backward,
)


class _TinyFSDP2Role(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weight.t()


def _full_tensor(value: torch.Tensor) -> torch.Tensor:
    full_tensor = getattr(value, "full_tensor", None)
    return full_tensor().detach() if callable(full_tensor) else value.detach()


def _global_data(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.arange(64 * 5, device=device, dtype=torch.float32).reshape(64, 5)
    features = (features.remainder(29) - 14.0) / 11.0
    targets = torch.arange(64 * 8, device=device, dtype=torch.float32).reshape(64, 8)
    targets = (targets.remainder(23) - 11.0) / 9.0
    return features, targets


def _reference(
    weight: torch.Tensor, features: torch.Tensor, targets: torch.Tensor, *, lr: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate = weight.clone().requires_grad_(True)
    error = features @ candidate.t() - targets
    loss = error.square().mean()
    loss.backward()
    assert candidate.grad is not None
    return (
        loss.detach(),
        candidate.grad.detach(),
        candidate.detach() - lr * candidate.grad,
    )


def _run_variant(
    *,
    mesh,
    device: torch.device,
    rank: int,
    world_size: int,
    microbatch: int,
    accumulation: int,
    sync_every_microbatch: bool,
    initial_weight: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    lr: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from torch.distributed._composable.fsdp import fully_shard

    module = _TinyFSDP2Role(initial_weight).to(device)
    fully_shard(module, mesh=mesh, reshard_after_forward=True)
    optimizer = torch.optim.SGD(module.parameters(), lr=lr)
    optimizer.zero_grad(set_to_none=True)
    per_rank = microbatch * accumulation
    rank_start = rank * per_rank
    total_numerator = torch.zeros((), device=device, dtype=torch.float32)
    global_count = targets.numel()
    for micro_step in range(accumulation):
        start = rank_start + micro_step * microbatch
        stop = start + microbatch
        error = module(features[start:stop]) - targets[start:stop]
        numerator = error.square().sum()
        total_numerator += numerator.detach()
        synchronize = sync_every_microbatch or micro_step == accumulation - 1
        with fsdp2_accumulation(module, sync_gradients=synchronize):
            stage2_global_mean_loss_for_backward(
                numerator,
                global_count=global_count,
                world_size=world_size,
            ).backward()
    dist.all_reduce(total_numerator, op=dist.ReduceOp.SUM)
    gradient = _full_tensor(module.weight.grad).cpu()
    optimizer.step()
    updated = _full_tensor(module.weight).cpu()
    return (total_numerator / global_count).cpu(), gradient, updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-h100", action="store_true")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if args.require_h100:
        if world_size != 8 or not torch.cuda.is_available():
            raise RuntimeError(
                "Stage-2 release accumulation gate requires 8 CUDA ranks"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
        name = torch.cuda.get_device_name(local_rank)
        if "H100" not in name.upper():
            raise RuntimeError(f"Stage-2 release gate requires H100, got {name!r}")
    else:
        if world_size < 2:
            raise RuntimeError("local FSDP2 parity seam requires at least two ranks")
        device = torch.device("cpu")
        backend = "gloo"
        # PyTorch nightlies may query this optional MPS hook even for a CPU mesh.
        if hasattr(torch, "mps") and not hasattr(torch.mps, "is_initialized"):
            torch.mps.is_initialized = lambda: False  # type: ignore[attr-defined]

    dist.init_process_group(backend=backend)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(
            device.type,
            (world_size,),
            mesh_dim_names=("shard",),
        )
        features, targets = _global_data(device)
        initial_weight = torch.arange(
            8 * 5, device=device, dtype=torch.float32
        ).reshape(8, 5)
        initial_weight = (initial_weight.remainder(13) - 6.0) / 17.0
        learning_rate = 0.03125
        reference_loss, reference_gradient, reference_updated = _reference(
            initial_weight,
            features,
            targets,
            lr=learning_rate,
        )
        for microbatch in (2, 1):
            if 64 % (world_size * microbatch):
                raise RuntimeError("global64 is not divisible by this gate topology")
            accumulation = 64 // (world_size * microbatch)
            if args.require_h100:
                expected = 4 if microbatch == 2 else 8
                if accumulation != expected:
                    raise RuntimeError("H100 gate did not resolve the locked profile")
            synchronized = _run_variant(
                mesh=mesh,
                device=device,
                rank=rank,
                world_size=world_size,
                microbatch=microbatch,
                accumulation=accumulation,
                sync_every_microbatch=True,
                initial_weight=initial_weight,
                features=features,
                targets=targets,
                lr=learning_rate,
            )
            no_sync = _run_variant(
                mesh=mesh,
                device=device,
                rank=rank,
                world_size=world_size,
                microbatch=microbatch,
                accumulation=accumulation,
                sync_every_microbatch=False,
                initial_weight=initial_weight,
                features=features,
                targets=targets,
                lr=learning_rate,
            )
            for actual in (synchronized, no_sync):
                torch.testing.assert_close(
                    actual[0], reference_loss.cpu(), rtol=2e-5, atol=2e-6
                )
                torch.testing.assert_close(
                    actual[1], reference_gradient.cpu(), rtol=2e-5, atol=2e-6
                )
                torch.testing.assert_close(
                    actual[2], reference_updated.cpu(), rtol=2e-5, atol=2e-6
                )
            for left, right in zip(synchronized, no_sync):
                torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)
        if rank == 0:
            mode = "H100-world8" if args.require_h100 else f"CPU-world{world_size}"
            print(f"STAGE2_FSDP2_ACCUMULATION_GATE=PASS mode={mode}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
