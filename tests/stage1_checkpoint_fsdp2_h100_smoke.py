"""Real six-H100 checkpoint collective smoke (not collected by pytest).

Run on the target node with a fresh output directory:

    torchrun --standalone --nproc_per_node=6 \
      tests/stage1_checkpoint_fsdp2_h100_smoke.py \
      --output /local_nvme/longlive_checkpoint_smoke

This entry exercises NCCL, the locked 2x3 HSDP mesh, selective raw/EMA
gathers, DCP optimizer get/set, rank-local heavy files, canonical-per-SP
buffer broadcast, and marker finalization.  CPU/mock unit tests do not claim
that this target-hardware test has passed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from utils.distributed import (
    TrainableShardedEMA,
    build_stage1_fsdp2_device_mesh,
)
from utils.lora_utils import LoraTensorSpec
from utils.stage1_checkpoint import (
    build_stage1_trainer_state,
    capture_rng_state,
    checkpoint_directory,
    finalize_stage1_checkpoint,
    gather_stage1_optimizer_state,
    gather_stage1_raw_and_ema_adapters,
    load_stage1_resume_payloads,
    restore_stage1_optimizer_state,
    validate_checkpoint,
    write_stage1_adapter_pair,
    write_stage1_resume_payloads,
)
from utils.stage1_io import atomic_write_bytes, atomic_write_json


class TinyHSDPLora(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_weight = nn.Parameter(
            torch.randn(4, 4, dtype=torch.bfloat16), requires_grad=False
        )
        self.block = nn.Module()
        self.block.lora_A = nn.ModuleDict(
            {"default": nn.Linear(4, 3, bias=False, dtype=torch.float32)}
        )
        self.block.lora_B = nn.ModuleDict(
            {"default": nn.Linear(3, 4, bias=False, dtype=torch.float32)}
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        adapter = (
            self.block.lora_B["default"].weight
            @ self.block.lora_A["default"].weight
        )
        weight = self.base_weight + adapter.to(torch.bfloat16)
        return torch.nn.functional.linear(value, weight)


SCHEMA = {
    "block.lora_A.weight": LoraTensorSpec(
        "block.lora_A.default.weight", (3, 4), torch.float32
    ),
    "block.lora_B.weight": LoraTensorSpec(
        "block.lora_B.default.weight", (4, 3), torch.float32
    ),
}
ADAPTER_NUMEL = 24


def optimizer_update(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    value = torch.arange(16, device="cuda", dtype=torch.bfloat16).reshape(4, 4)
    loss = model(value).float().square().mean()
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("tiny HSDP loss is non-finite")
    loss.backward()
    optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    try:
        mesh = build_stage1_fsdp2_device_mesh()
        torch.manual_seed(1234)
        model = TinyHSDPLora().cuda()
        fully_shard(
            model,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
            reshard_after_forward=False,
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        optimizer_update(model, optimizer)
        ema = TrainableShardedEMA(
            model,
            decay=0.5,
            start_step=1,
            topology={"mesh_shape": (2, 3), "mesh_dim_names": ("replicate", "shard")},
        )
        ema.update_after_step(model, 1)
        optimizer_update(model, optimizer)
        ema.update_after_step(model, 2)

        directory = checkpoint_directory(args.output, 2)
        setup_status = [None]
        if rank == 0:
            try:
                if directory.exists() and any(directory.iterdir()):
                    raise RuntimeError(
                        f"smoke checkpoint directory is not empty: {directory}"
                    )
                directory.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(directory / "resolved_config.yaml", b"smoke: true\n")
                atomic_write_json(
                    directory / "base_reference.json",
                    {"base_sha256": "h100-smoke-base"},
                )
                setup_status[0] = (True, None)
            except Exception as exc:
                setup_status[0] = (False, str(exc))
        dist.broadcast_object_list(setup_status, src=0)
        if not setup_status[0][0]:
            raise RuntimeError(f"rank-0 smoke output setup failed: {setup_status[0][1]}")
        dist.barrier()

        adapters = gather_stage1_raw_and_ema_adapters(
            model,
            ema,
            expected_schema=SCHEMA,
            authoritative_shard_group=mesh.get_group("shard"),
            replica_group=mesh.get_group("replicate"),
            expected_adapter_tensors=2,
            expected_global_numel=ADAPTER_NUMEL,
        )
        write_stage1_adapter_pair(
            directory,
            adapters,
            expected_schema=SCHEMA,
            completed_step=2,
            expected_adapter_tensors=2,
            expected_global_numel=ADAPTER_NUMEL,
        )
        optimizer_state = gather_stage1_optimizer_state(
            model,
            optimizer,
            expected_schema=SCHEMA,
            expected_completed_step=2,
        )
        trainer_state = (
            build_stage1_trainer_state(
                optimizer_state=optimizer_state,
                completed_step=2,
                global_epoch=0,
                committed_microbatch_cursor_in_epoch=4,
                phase_derivation={"phase": "smoke"},
                sampler_state={"epoch": 0},
                dataloader_generator_state=torch.Generator().manual_seed(7).get_state(),
                next_attempt_index=2,
                nonfinite_attempt_count=0,
                resolved_config_sha256="a" * 64,
            )
            if rank == 0
            else None
        )
        write_stage1_resume_payloads(
            directory,
            trainer_state=trainer_state,
            ema_state=ema.state_dict(),
            rng_state=capture_rng_state(include_cuda=True),
            error_buffer_state={"schema": 1, "sp_position": rank} if rank < 3 else None,
        )
        finalize_stage1_checkpoint(directory, completed_step=2, resumable=True)

        payload = load_stage1_resume_payloads(
            directory,
            replica_group=mesh.get_group("replicate"),
            expected_base_sha256="h100-smoke-base",
            expected_resolved_config_sha256="a" * 64,
        )
        if payload.trainer_state["completed_step"] != 2:
            raise RuntimeError("WORLD trainer metadata broadcast mismatch")
        if payload.error_buffer_state["sp_position"] != rank % 3:
            raise RuntimeError("canonical-per-SP buffer broadcast mismatch")

        optimizer.state.clear()
        restore_stage1_optimizer_state(
            model,
            optimizer,
            payload.optimizer_state,
            expected_schema=SCHEMA,
            expected_completed_step=2,
        )
        restored_ema = TrainableShardedEMA(
            model,
            decay=0.5,
            start_step=1,
            topology={"mesh_shape": (2, 3), "mesh_dim_names": ("replicate", "shard")},
        )
        restored_ema.load_state_dict(payload.ema_state, model)
        dist.barrier()
        if rank == 0:
            validate_checkpoint(
                directory,
                require_resumable=True,
                expected_base_sha256="h100-smoke-base",
                expected_topology=(6, 3, 2),
            )
            print(json.dumps({"status": "passed", "checkpoint": str(directory)}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
