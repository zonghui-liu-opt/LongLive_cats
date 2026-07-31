import torch
import pytest

from utils.stage1_loss import (
    aggregate_loss_metrics,
    local_loss_metrics,
    loss_for_backward,
)


def test_three_sp_chunks_reduce_to_locked_counts_and_weighted_loss():
    total_num = torch.tensor(0.0)
    total_count = torch.tensor(0.0)
    block_num = torch.zeros(3)
    block_count = torch.zeros(3)
    # Four logical samples/update: DP2 × two accumulation micro-steps.
    for _logical_sample in range(4):
        for sp_rank in range(3):
            values = torch.full((1, 8), float(sp_rank + 1))
            mask = torch.ones_like(values)
            if sp_rank == 0:
                mask[:, 0] = 0
            metrics = local_loss_metrics(
                values,
                mask,
                num_frame_per_block=8,
                global_frame_offset=sp_rank * 8,
                total_blocks=3,
            )
            total_num += metrics.numerator
            total_count += metrics.count
            block_num += metrics.block_numerators
            block_count += metrics.block_counts
    result = aggregate_loss_metrics(total_num, total_count, block_num, block_count)
    assert result["loss_count"] == 92
    assert [result["blocks"][f"block{i}"]["count"] for i in range(3)] == [28, 32, 32]
    assert result["loss_total"] == (28 * 1 + 32 * 2 + 32 * 3) / 92
    assert result["loss_total"] != sum([1.0, 2.0, 3.0]) / 3


def test_sp3_dp2_accum2_backward_matches_four_sample_reference():
    # Each rank owns a sequence contribution. FSDP averages the six rank
    # gradients after each micro-step. The explicit ×SP/accum must recover the
    # mean of four complete logical samples.
    sample_contributions = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]
    reference = sum(sum(sample) for sample in sample_contributions) / (4 * 23)

    # micro0 uses samples 0/1 across the two DP replicas; micro1 uses 2/3.
    accumulated_rank_average = 0.0
    for micro in range(2):
        rank_losses = []
        for dp_rank in range(2):
            sample = sample_contributions[micro * 2 + dp_rank]
            for sp_rank in range(3):
                numerator = torch.tensor(sample[sp_rank], requires_grad=True)
                rank_losses.append(
                    loss_for_backward(
                        numerator,
                        global_valid_count_per_sp_sample=23,
                        sequence_parallel_size=3,
                        gradient_accumulation_steps=2,
                    ).item()
                )
        accumulated_rank_average += sum(rank_losses) / 6
    assert accumulated_rank_average == pytest.approx(reference, rel=1e-6)
