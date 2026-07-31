"""Loss accounting primitives for SP3 × DP2 Stage-1 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LocalLossMetrics:
    numerator: torch.Tensor
    count: torch.Tensor
    block_numerators: torch.Tensor
    block_counts: torch.Tensor

    def detached(self) -> "LocalLossMetrics":
        return LocalLossMetrics(
            numerator=self.numerator.detach(),
            count=self.count.detach(),
            block_numerators=self.block_numerators.detach(),
            block_counts=self.block_counts.detach(),
        )


def local_loss_metrics(
    per_frame_loss: torch.Tensor,
    loss_mask: torch.Tensor | None,
    *,
    num_frame_per_block: int,
    global_frame_offset: int = 0,
    total_blocks: int | None = None,
) -> LocalLossMetrics:
    """Compute detached-ready numerators/counts from the existing loss tensor."""
    if per_frame_loss.ndim != 2:
        raise ValueError(
            f"per_frame_loss must have shape [B,F], got {tuple(per_frame_loss.shape)}."
        )
    block_size = int(num_frame_per_block)
    frame_offset = int(global_frame_offset)
    if block_size <= 0 or frame_offset < 0:
        raise ValueError("num_frame_per_block must be positive and offset non-negative.")
    if frame_offset % block_size:
        raise ValueError(
            f"global_frame_offset ({frame_offset}) must align to block size ({block_size})."
        )
    if loss_mask is None:
        mask = torch.ones_like(per_frame_loss)
    else:
        if tuple(loss_mask.shape) != tuple(per_frame_loss.shape):
            raise ValueError(
                f"loss_mask shape {tuple(loss_mask.shape)} must match per-frame "
                f"loss {tuple(per_frame_loss.shape)}."
            )
        mask = loss_mask.to(device=per_frame_loss.device, dtype=per_frame_loss.dtype)
    if not torch.isfinite(per_frame_loss).all():
        # Do not hide the failure inside a reduction. The trainer still performs
        # a WORLD consensus before deciding whether to retry the cached update.
        pass
    weighted = per_frame_loss * mask
    numerator = weighted.sum()
    count = mask.sum()

    local_frames = per_frame_loss.shape[1]
    global_end = frame_offset + local_frames
    if total_blocks is None:
        total_blocks = (global_end + block_size - 1) // block_size
    total_blocks = int(total_blocks)
    if total_blocks <= 0 or global_end > total_blocks * block_size:
        raise ValueError(
            f"Local frame range [{frame_offset}, {global_end}) does not fit "
            f"{total_blocks} blocks of {block_size}."
        )
    block_numerators = torch.zeros(
        total_blocks, device=per_frame_loss.device, dtype=per_frame_loss.dtype
    )
    block_counts = torch.zeros_like(block_numerators)
    for block_index in range(total_blocks):
        overlap_start = max(frame_offset, block_index * block_size)
        overlap_end = min(global_end, (block_index + 1) * block_size)
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - frame_offset
        local_end = overlap_end - frame_offset
        block_numerators[block_index] = weighted[:, local_start:local_end].sum()
        block_counts[block_index] = mask[:, local_start:local_end].sum()
    return LocalLossMetrics(numerator, count, block_numerators, block_counts)


def loss_for_backward(
    local_numerator: torch.Tensor,
    *,
    global_valid_count_per_sp_sample: torch.Tensor | float | int,
    sequence_parallel_size: int,
    gradient_accumulation_steps: int,
) -> torch.Tensor:
    """Apply the P0 SP/FSDP scaling without detaching the numerator graph."""
    sp_size = int(sequence_parallel_size)
    accumulation = int(gradient_accumulation_steps)
    if sp_size <= 0 or accumulation <= 0:
        raise ValueError("SP size and gradient accumulation must be positive.")
    denominator = torch.as_tensor(
        global_valid_count_per_sp_sample,
        device=local_numerator.device,
        dtype=local_numerator.dtype,
    )
    if denominator.numel() != 1 or not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError(
            "global_valid_count_per_sp_sample must be one positive finite scalar."
        )
    return local_numerator / denominator * (sp_size / accumulation)


def aggregate_loss_metrics(
    numerator: torch.Tensor,
    count: torch.Tensor,
    block_numerators: torch.Tensor,
    block_counts: torch.Tensor,
) -> dict[str, Any]:
    """Convert already WORLD-reduced sums to the canonical JSON-ready shape."""
    if numerator.numel() != 1 or count.numel() != 1:
        raise ValueError("Total numerator/count must be scalar tensors.")
    if block_numerators.ndim != 1 or block_counts.shape != block_numerators.shape:
        raise ValueError("Block numerator/count tensors must be matching vectors.")
    total_count = float(count.item())
    if total_count <= 0:
        raise ValueError("Reduced valid frame count must be positive.")
    blocks = {}
    for index, (block_num, block_count) in enumerate(
        zip(block_numerators.tolist(), block_counts.tolist())
    ):
        block_count = float(block_count)
        if block_count <= 0:
            raise ValueError(f"Reduced block{index} valid frame count must be positive.")
        blocks[f"block{index}"] = {
            "numerator": float(block_num),
            "count": block_count,
            "loss": float(block_num) / block_count,
        }
    return {
        "loss_numerator": float(numerator.item()),
        "loss_count": total_count,
        "loss_total": float(numerator.item()) / total_count,
        "blocks": blocks,
    }
