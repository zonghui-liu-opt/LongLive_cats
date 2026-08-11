from __future__ import annotations

import pytest
import torch

import utils.distributed as distributed_utils
from utils.distributed import fsdp2_accumulation, stage1_fsdp2_accumulation
from utils.stage2_train_state import stage2_global_mean_loss_for_backward


def _dataset():
    generator = torch.Generator(device="cpu").manual_seed(20260811)
    features = torch.randn(64, 5, generator=generator, dtype=torch.float64)
    targets = torch.randn(64, 3, generator=generator, dtype=torch.float64)
    return features, targets


def _reference_gradient(features, targets, weight):
    candidate = weight.detach().clone().requires_grad_(True)
    prediction = features @ candidate
    loss = (prediction - targets).square().mean()
    loss.backward()
    return loss.detach(), candidate.grad.detach()


def _distributed_profile_gradient(
    features, targets, weight, *, microbatch, accumulation
):
    world_size = 8
    global_count = targets.numel()
    rank_gradients = []
    total_numerator = torch.zeros((), dtype=torch.float64)
    for rank in range(world_size):
        candidate = weight.detach().clone().requires_grad_(True)
        for micro_step in range(accumulation):
            start = (micro_step * world_size + rank) * microbatch
            stop = start + microbatch
            local_error = features[start:stop] @ candidate - targets[start:stop]
            local_numerator = local_error.square().sum()
            total_numerator += local_numerator.detach()
            stage2_global_mean_loss_for_backward(
                local_numerator,
                global_count=global_count,
                world_size=world_size,
            ).backward()
        rank_gradients.append(candidate.grad.detach())
    # FSDP's reduce-scatter averages the accumulated local gradients over WORLD.
    return total_numerator / global_count, torch.stack(rank_gradients).mean(dim=0)


@pytest.mark.parametrize(("microbatch", "accumulation"), [(2, 4), (1, 8)])
def test_stage2_accumulation_matches_one_global_batch_loss_and_gradient(
    microbatch, accumulation
):
    features, targets = _dataset()
    weight = torch.randn(5, 3, dtype=torch.float64)
    reference_loss, reference_gradient = _reference_gradient(features, targets, weight)
    actual_loss, actual_gradient = _distributed_profile_gradient(
        features,
        targets,
        weight,
        microbatch=microbatch,
        accumulation=accumulation,
    )

    torch.testing.assert_close(actual_loss, reference_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        actual_gradient, reference_gradient, rtol=1e-12, atol=1e-12
    )


def test_global_mean_scaling_supports_unequal_local_counts():
    weight = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    numerators = [weight * 1.0, weight * 3.0, weight * 2.0]
    counts = [1, 4, 2]
    world_size = 3
    global_count = sum(counts)
    local_gradients = []
    for numerator in numerators:
        local_weight = weight.detach().clone().requires_grad_(True)
        local_numerator = local_weight * float(numerator.detach() / weight.detach())
        stage2_global_mean_loss_for_backward(
            local_numerator,
            global_count=global_count,
            world_size=world_size,
        ).backward()
        local_gradients.append(local_weight.grad)
    actual = torch.stack(local_gradients).mean()
    expected = torch.tensor((1.0 + 3.0 + 2.0) / global_count, dtype=torch.float64)
    torch.testing.assert_close(actual, expected)


class FakeFSDP2Root:
    def __init__(self):
        self.calls = []

    def set_requires_gradient_sync(self, value, *, recurse):
        self.calls.append(("sync", value, recurse))

    def set_reshard_after_backward(self, value, *, recurse):
        self.calls.append(("reshard", value, recurse))

    def set_is_last_backward(self, value):
        self.calls.append(("last", value))


def test_generic_fsdp2_accumulation_and_stage1_alias_have_identical_semantics(
    monkeypatch,
):
    monkeypatch.setattr(
        distributed_utils,
        "_import_fsdp2_api",
        lambda: {"FSDPModule": FakeFSDP2Root},
    )
    generic = FakeFSDP2Root()
    legacy = FakeFSDP2Root()

    with fsdp2_accumulation(generic, sync_gradients=False):
        pass
    with stage1_fsdp2_accumulation(legacy, sync_gradients=False):
        pass

    assert generic.calls == legacy.calls
    assert generic.calls == [
        ("sync", False, True),
        ("reshard", True, True),
        ("last", False),
        ("sync", True, True),
        ("reshard", True, True),
        ("last", True),
    ]


@pytest.mark.parametrize("bad_count", [0, -1, float("nan"), float("inf")])
def test_global_mean_scaling_rejects_invalid_denominator(bad_count):
    with pytest.raises(ValueError, match="global_count"):
        stage2_global_mean_loss_for_backward(
            torch.tensor(1.0, requires_grad=True),
            global_count=bad_count,
            world_size=8,
        )


def test_global_mean_scaling_rejects_invalid_world_size():
    with pytest.raises(ValueError, match="world_size"):
        stage2_global_mean_loss_for_backward(
            torch.tensor(1.0, requires_grad=True),
            global_count=64,
            world_size=0,
        )
