"""Exact replay transaction for one Stage-2 F/G optimizer substep.

Only randomness and stateful input streams belong in this transaction.  The
attempt callback must perform all finite gates before calling ``optimizer.step``;
an optimizer or EMA mutation is deliberately not presented as rollback-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from collections.abc import Callable, Mapping
import random
from typing import Any, Generic, TypeVar

import numpy as np
import torch

T = TypeVar("T")


class Stage2NonfiniteAttempt(RuntimeError):
    """Signal that an attempt failed a pre-optimizer finite gate."""


class Stage2RetryExhausted(RuntimeError):
    """Raised after the configured number of exact non-finite attempts."""

    def __init__(
        self,
        *,
        logical_substep_id: str,
        attempts: int,
        last_error: Stage2NonfiniteAttempt,
    ) -> None:
        self.logical_substep_id = logical_substep_id
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Stage-2 substep {logical_substep_id!r} remained non-finite "
            f"after {attempts} attempts: {last_error}"
        )


@dataclass(frozen=True)
class Stage2NonfiniteEvent:
    logical_substep_id: str
    attempt_number: int
    max_attempts: int
    attempts_remaining: int
    reason: str


@dataclass(frozen=True)
class Stage2TransactionResult(Generic[T]):
    value: T
    attempts: int
    nonfinite_attempts: int


@dataclass(frozen=True)
class Stage2SubstepSnapshot:
    python_rng_state: object
    numpy_rng_state: tuple[Any, ...]
    torch_cpu_rng_state: torch.Tensor
    cuda_device: int | None
    torch_cuda_rng_state: torch.Tensor | None
    stateful_stream_states: dict[str, Any]
    generator_states: dict[str, torch.Tensor]


def _clone_state(value: Any) -> Any:
    return copy.deepcopy(value)


class Stage2SubstepTransaction:
    """Snapshot and exactly replay all registered input/RNG streams."""

    def __init__(
        self,
        *,
        logical_substep_id: str,
        stateful_streams: Mapping[str, Any] | None = None,
        generators: Mapping[str, torch.Generator] | None = None,
        max_attempts: int = 2,
        include_cuda_rng: bool = True,
    ) -> None:
        if (
            not isinstance(logical_substep_id, str)
            or not logical_substep_id
            or logical_substep_id != logical_substep_id.strip()
        ):
            raise ValueError(
                "logical_substep_id must be a non-empty string without whitespace padding"
            )
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts must be an integer >= 1")
        if not isinstance(include_cuda_rng, bool):
            raise TypeError("include_cuda_rng must be bool")
        self.logical_substep_id = logical_substep_id
        self.max_attempts = max_attempts
        self.include_cuda_rng = include_cuda_rng
        self.stateful_streams = dict(stateful_streams or {})
        self.generators = dict(generators or {})
        duplicate_names = sorted(set(self.stateful_streams) & set(self.generators))
        if duplicate_names:
            raise ValueError(f"stream names registered twice: {duplicate_names}")
        if any(not isinstance(name, str) or not name for name in self.stateful_streams):
            raise ValueError("stateful stream names must be non-empty strings")
        if any(not isinstance(name, str) or not name for name in self.generators):
            raise ValueError("generator names must be non-empty strings")
        seen_objects: dict[int, str] = {}
        for name, stream in self.stateful_streams.items():
            if not callable(getattr(stream, "state_dict", None)) or not callable(
                getattr(stream, "load_state_dict", None)
            ):
                raise TypeError(
                    f"stateful stream {name!r} must implement state_dict/load_state_dict"
                )
            previous = seen_objects.setdefault(id(stream), name)
            if previous != name:
                raise ValueError(
                    f"the same stateful stream is registered as {previous!r} and {name!r}"
                )
        for name, generator in self.generators.items():
            if not isinstance(generator, torch.Generator):
                raise TypeError(f"generator {name!r} is not a torch.Generator")
            previous = seen_objects.setdefault(id(generator), name)
            if previous != name:
                raise ValueError(
                    f"the same generator is registered as {previous!r} and {name!r}"
                )
        self._running = False

    def snapshot(self) -> Stage2SubstepSnapshot:
        """Capture the pre-attempt committed stream position."""

        stream_states = {
            name: _clone_state(stream.state_dict())
            for name, stream in sorted(self.stateful_streams.items())
        }
        generator_states = {
            name: generator.get_state().detach().cpu().clone()
            for name, generator in sorted(self.generators.items())
        }
        cuda_device: int | None = None
        cuda_state: torch.Tensor | None = None
        if self.include_cuda_rng and torch.cuda.is_available():
            cuda_device = int(torch.cuda.current_device())
            cuda_state = torch.cuda.get_rng_state(cuda_device).detach().cpu().clone()
        return Stage2SubstepSnapshot(
            python_rng_state=_clone_state(random.getstate()),
            numpy_rng_state=_clone_state(np.random.get_state()),
            torch_cpu_rng_state=torch.get_rng_state().detach().cpu().clone(),
            cuda_device=cuda_device,
            torch_cuda_rng_state=cuda_state,
            stateful_stream_states=stream_states,
            generator_states=generator_states,
        )

    def restore(self, snapshot: Stage2SubstepSnapshot) -> None:
        """Restore a trusted snapshot without leaving RNG changes from loaders."""

        if not isinstance(snapshot, Stage2SubstepSnapshot):
            raise TypeError("snapshot must be a Stage2SubstepSnapshot")
        if set(snapshot.stateful_stream_states) != set(self.stateful_streams):
            raise ValueError("snapshot stateful stream names do not match transaction")
        if set(snapshot.generator_states) != set(self.generators):
            raise ValueError("snapshot generator names do not match transaction")
        for name, stream in sorted(self.stateful_streams.items()):
            stream.load_state_dict(_clone_state(snapshot.stateful_stream_states[name]))
        for name, generator in sorted(self.generators.items()):
            generator.set_state(snapshot.generator_states[name].detach().cpu().clone())
        random.setstate(_clone_state(snapshot.python_rng_state))
        np.random.set_state(_clone_state(snapshot.numpy_rng_state))
        torch.set_rng_state(snapshot.torch_cpu_rng_state.detach().cpu().clone())
        if snapshot.torch_cuda_rng_state is not None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "cannot restore a Stage-2 CUDA RNG snapshot without CUDA"
                )
            current_device = int(torch.cuda.current_device())
            if snapshot.cuda_device != current_device:
                raise RuntimeError(
                    "Stage-2 CUDA RNG device changed during a substep: "
                    f"snapshot={snapshot.cuda_device}, current={current_device}"
                )
            torch.cuda.set_rng_state(
                snapshot.torch_cuda_rng_state.detach().cpu().clone(),
                device=current_device,
            )

    def _rollback(
        self,
        snapshot: Stage2SubstepSnapshot,
        *,
        zero_grad: Callable[[], None] | None,
        event: Stage2NonfiniteEvent | None = None,
        on_nonfinite: Callable[[Stage2NonfiniteEvent], None] | None = None,
    ) -> None:
        try:
            if zero_grad is not None:
                zero_grad()
            if event is not None and on_nonfinite is not None:
                on_nonfinite(event)
        finally:
            # Restore last so logging/cleanup code cannot perturb the retry RNG.
            self.restore(snapshot)

    def run(
        self,
        attempt: Callable[[int], T],
        *,
        zero_grad: Callable[[], None] | None = None,
        on_nonfinite: Callable[[Stage2NonfiniteEvent], None] | None = None,
    ) -> Stage2TransactionResult[T]:
        """Run one logical substep, retrying only explicit non-finite failures."""

        if not callable(attempt):
            raise TypeError("attempt must be callable")
        if zero_grad is not None and not callable(zero_grad):
            raise TypeError("zero_grad must be callable")
        if on_nonfinite is not None and not callable(on_nonfinite):
            raise TypeError("on_nonfinite must be callable")
        if self._running:
            raise RuntimeError("Stage2SubstepTransaction cannot run re-entrantly")
        self._running = True
        try:
            snapshot = self.snapshot()
            nonfinite_attempts = 0
            for attempt_number in range(1, self.max_attempts + 1):
                try:
                    value = attempt(attempt_number)
                except Stage2NonfiniteAttempt as error:
                    nonfinite_attempts += 1
                    event = Stage2NonfiniteEvent(
                        logical_substep_id=self.logical_substep_id,
                        attempt_number=attempt_number,
                        max_attempts=self.max_attempts,
                        attempts_remaining=self.max_attempts - attempt_number,
                        reason=str(error),
                    )
                    self._rollback(
                        snapshot,
                        zero_grad=zero_grad,
                        event=event,
                        on_nonfinite=on_nonfinite,
                    )
                    if attempt_number == self.max_attempts:
                        raise Stage2RetryExhausted(
                            logical_substep_id=self.logical_substep_id,
                            attempts=attempt_number,
                            last_error=error,
                        ) from error
                    continue
                except BaseException:
                    self._rollback(snapshot, zero_grad=zero_grad)
                    raise
                return Stage2TransactionResult(
                    value=value,
                    attempts=attempt_number,
                    nonfinite_attempts=nonfinite_attempts,
                )
        finally:
            self._running = False
        raise AssertionError("unreachable Stage-2 transaction state")


__all__ = [
    "Stage2NonfiniteAttempt",
    "Stage2NonfiniteEvent",
    "Stage2RetryExhausted",
    "Stage2SubstepSnapshot",
    "Stage2SubstepTransaction",
    "Stage2TransactionResult",
]
