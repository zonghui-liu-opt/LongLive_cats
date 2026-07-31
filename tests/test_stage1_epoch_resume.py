from types import SimpleNamespace

import torch

from trainer.diffusion import Trainer


class _EpochSampler(torch.utils.data.Sampler[int]):
    def __init__(self, size: int, seed: int = 42):
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def _runtime(*, generator, sampler, epoch, cursor, step):
    runtime = Trainer.__new__(Trainer)
    runtime.stage1_dataloader_generator = generator
    runtime.stage1_sampler = sampler
    runtime.stage1_dataloader = torch.utils.data.DataLoader(
        list(range(len(sampler))),
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    runtime.stage1_epoch = epoch
    runtime.stage1_microbatch_cursor = cursor
    runtime.stage1_schedule = SimpleNamespace(total_updates=750)
    runtime.step = step
    runtime.stage1_iterator = None
    return runtime


def test_mid_epoch_resume_skips_exact_cursor_and_preserves_next_epoch_rng():
    original_generator = torch.Generator().manual_seed(10_042)
    original_sampler = _EpochSampler(300)
    original = _runtime(
        generator=original_generator,
        sampler=original_sampler,
        epoch=0,
        cursor=0,
        step=0,
    )
    original.stage1_iterator = iter(original.stage1_dataloader)
    saved_generator_state = original_generator.get_state().clone()
    for _ in range(150):
        next(original.stage1_iterator)
    expected_next = next(original.stage1_iterator)

    resumed_generator = torch.Generator().manual_seed(999)
    resumed = _runtime(
        generator=resumed_generator,
        sampler=_EpochSampler(300),
        epoch=0,
        cursor=150,
        step=75,
    )
    resumed._stage1_rebuild_iterator_at_committed_cursor(saved_generator_state)

    assert torch.equal(next(resumed.stage1_iterator), expected_next)
    assert torch.equal(resumed_generator.get_state(), saved_generator_state)


def test_epoch_boundary_resume_advances_loader_generator_once():
    continuous_generator = torch.Generator().manual_seed(10_042)
    continuous_sampler = _EpochSampler(300)
    continuous = _runtime(
        generator=continuous_generator,
        sampler=continuous_sampler,
        epoch=0,
        cursor=0,
        step=0,
    )
    continuous.stage1_iterator = iter(continuous.stage1_dataloader)
    saved_generator_state = continuous_generator.get_state().clone()
    continuous_sampler.set_epoch(1)
    next_epoch_iterator = iter(continuous.stage1_dataloader)
    expected_generator_state = continuous_generator.get_state().clone()
    expected_next = next(next_epoch_iterator)

    resumed_generator = torch.Generator().manual_seed(123)
    resumed = _runtime(
        generator=resumed_generator,
        sampler=_EpochSampler(300),
        epoch=1,
        cursor=0,
        step=150,
    )
    resumed._stage1_rebuild_iterator_at_committed_cursor(saved_generator_state)

    assert torch.equal(next(resumed.stage1_iterator), expected_next)
    assert torch.equal(resumed_generator.get_state(), expected_generator_state)


def test_completed_schedule_does_not_construct_an_unused_iterator():
    generator = torch.Generator().manual_seed(10_042)
    saved = generator.get_state().clone()
    runtime = _runtime(
        generator=generator,
        sampler=_EpochSampler(300),
        epoch=5,
        cursor=0,
        step=750,
    )
    runtime._stage1_rebuild_iterator_at_committed_cursor(saved)
    assert runtime.stage1_iterator is None
    assert torch.equal(generator.get_state(), saved)
