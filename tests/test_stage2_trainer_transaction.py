from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from utils.stage2_train_transaction import (
    Stage2NonfiniteAttempt,
    Stage2RetryExhausted,
    Stage2SubstepTransaction,
)


class TinyStatefulStream:
    def __init__(self):
        self.cursor = 0

    def draw(self) -> int:
        value = self.cursor
        self.cursor += 1
        return value

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = int(state["cursor"])


def _rng_bundle(stream, generators):
    return {
        "batch": stream.draw(),
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch": float(torch.rand(())),
        "exit": torch.rand(4, generator=generators["exit"]).clone(),
        "branch": torch.rand(4, generator=generators["branch"]).clone(),
        "timestep": torch.rand(4, generator=generators["timestep"]).clone(),
        "noise": torch.rand(4, generator=generators["noise"]).clone(),
        "loader": torch.rand(4, generator=generators["loader"]).clone(),
    }


def _assert_bundles_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if torch.is_tensor(left[key]):
            assert torch.equal(left[key], right[key])
        else:
            assert left[key] == right[key]


def test_nonfinite_retry_replays_batch_and_every_rng_exactly_then_commits_once():
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    stream = TinyStatefulStream()
    generators = {
        "exit": torch.Generator(device="cpu").manual_seed(4),
        "branch": torch.Generator(device="cpu").manual_seed(5),
        "timestep": torch.Generator(device="cpu").manual_seed(6),
        "noise": torch.Generator(device="cpu").manual_seed(7),
        "loader": torch.Generator(device="cpu").manual_seed(8),
    }
    attempts = []
    zero_grad_calls = []
    nonfinite_events = []

    transaction = Stage2SubstepTransaction(
        logical_substep_id="cycle000001/F1",
        stateful_streams={"fake_sampler": stream},
        generators=generators,
        max_attempts=2,
        include_cuda_rng=False,
    )

    def attempt(attempt_number):
        bundle = _rng_bundle(stream, generators)
        attempts.append(bundle)
        if attempt_number == 1:
            raise Stage2NonfiniteAttempt("non-finite fake-score gradient")
        return bundle

    result = transaction.run(
        attempt,
        zero_grad=lambda: zero_grad_calls.append("zero"),
        on_nonfinite=lambda event: nonfinite_events.append(event),
    )

    assert result.attempts == 2
    assert result.nonfinite_attempts == 1
    _assert_bundles_equal(attempts[0], attempts[1])
    _assert_bundles_equal(result.value, attempts[1])
    assert stream.cursor == 1
    assert zero_grad_calls == ["zero"]
    assert [event.attempt_number for event in nonfinite_events] == [1]
    assert nonfinite_events[0].logical_substep_id == "cycle000001/F1"


def test_two_nonfinite_attempts_fail_fast_and_restore_every_committed_stream():
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    stream = TinyStatefulStream()
    generators = {
        "exit": torch.Generator(device="cpu").manual_seed(14),
        "branch": torch.Generator(device="cpu").manual_seed(15),
        "timestep": torch.Generator(device="cpu").manual_seed(16),
        "noise": torch.Generator(device="cpu").manual_seed(17),
        "loader": torch.Generator(device="cpu").manual_seed(18),
    }
    before_global = (
        copy.deepcopy(random.getstate()),
        copy.deepcopy(np.random.get_state()),
        torch.get_rng_state().clone(),
    )
    before_generators = {
        name: generator.get_state().clone() for name, generator in generators.items()
    }
    attempts = []
    zero_grad_calls = []

    transaction = Stage2SubstepTransaction(
        logical_substep_id="cycle000001/G",
        stateful_streams={"generator_sampler": stream},
        generators=generators,
        max_attempts=2,
        include_cuda_rng=False,
    )

    def always_nonfinite(_attempt_number):
        attempts.append(_rng_bundle(stream, generators))
        raise Stage2NonfiniteAttempt("non-finite generator loss")

    with pytest.raises(Stage2RetryExhausted, match="after 2 attempts") as error:
        transaction.run(
            always_nonfinite,
            zero_grad=lambda: zero_grad_calls.append("zero"),
        )

    assert error.value.attempts == 2
    _assert_bundles_equal(attempts[0], attempts[1])
    assert zero_grad_calls == ["zero", "zero"]
    assert stream.cursor == 0
    assert random.getstate() == before_global[0]
    current_numpy = np.random.get_state()
    assert current_numpy[0] == before_global[1][0]
    assert np.array_equal(current_numpy[1], before_global[1][1])
    assert current_numpy[2:] == before_global[1][2:]
    assert torch.equal(torch.get_rng_state(), before_global[2])
    for name, generator in generators.items():
        assert torch.equal(generator.get_state(), before_generators[name])


def test_unexpected_exception_is_not_retried_but_restores_and_clears_gradients():
    stream = TinyStatefulStream()
    generator = torch.Generator(device="cpu").manual_seed(99)
    before = generator.get_state().clone()
    zero_grad_calls = []
    transaction = Stage2SubstepTransaction(
        logical_substep_id="cycle000003/F4",
        stateful_streams={"sampler": stream},
        generators={"noise": generator},
        include_cuda_rng=False,
    )

    def broken(_attempt_number):
        stream.draw()
        torch.rand((), generator=generator)
        raise KeyError("programming bug")

    with pytest.raises(KeyError, match="programming bug"):
        transaction.run(broken, zero_grad=lambda: zero_grad_calls.append("zero"))

    assert stream.cursor == 0
    assert torch.equal(generator.get_state(), before)
    assert zero_grad_calls == ["zero"]


def test_transaction_rejects_ambiguous_or_mutable_registration():
    generator = torch.Generator(device="cpu")
    with pytest.raises(ValueError, match="logical_substep_id"):
        Stage2SubstepTransaction(logical_substep_id="")
    with pytest.raises(ValueError, match="max_attempts"):
        Stage2SubstepTransaction(logical_substep_id="F1", max_attempts=0)
    with pytest.raises(ValueError, match="registered twice"):
        Stage2SubstepTransaction(
            logical_substep_id="F1",
            stateful_streams={"shared": TinyStatefulStream()},
            generators={"shared": generator},
        )
