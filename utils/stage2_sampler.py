"""Stateful, role-independent balanced sampling for Stage-2 updates.

Each successful optimizer update consumes one DP-global batch of 64 samples.
The three actions receive 22/21/21 samples and the extra slot rotates between
actions.  Fake-score (F) and Generator (G) own separate instances, generators,
queues, cursors, and checkpoint state; consuming five F batches cannot perturb
the next G batch.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import product
import math
from typing import Any

import torch

from utils.stage1_io import canonical_json_sha256

STAGE2_SAMPLER_STATE_SCHEMA = "longlive_stage2_balanced_sampler/v2"
STAGE2_SAMPLER_STREAMS_SCHEMA = "longlive_stage2_sampler_streams/v2"
STAGE2_SAMPLER_ROLES = ("fake_score", "generator")
STAGE2_PER_ACTION_BATCH_COUNTS = (22, 21, 21)
STAGE2_GLOBAL_BATCH_SIZE = 64
STAGE2_NUM_SAMPLES = 600
STAGE2_ALLOWED_SPATIAL_SHAPES = ((30, 52), (52, 30))
STAGE2_BATCHES_PER_STREAM_EPOCH = math.ceil(
    STAGE2_NUM_SAMPLES / STAGE2_GLOBAL_BATCH_SIZE
)

_STATE_KEYS = {
    "schema",
    "role",
    "base_seed",
    "stream_seed",
    "dataset_action_spatial_sha256",
    "action_order",
    "spatial_shape_order",
    "microbatch_size_per_device",
    "global_batch_size",
    "per_action_batch_counts",
    "batches_per_stream_epoch",
    "completed_batches",
    "stream_epoch",
    "batch_cursor",
    "extra_slot_cursor",
    "action_states",
    "generator_state",
}
_ACTION_STATE_KEYS = {"groups"}
_GROUP_STATE_KEYS = {"queue_epoch", "cursor", "order"}


def _require_plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}, got {value!r}.")
    return value


def _require_exact_mapping(
    value: Any, *, label: str, expected_keys: set[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping.")
    mapping = dict(value)
    missing = sorted(expected_keys - set(mapping))
    extra = sorted(set(mapping) - expected_keys)
    if missing or extra:
        raise RuntimeError(f"{label} keys mismatch: missing={missing}, extra={extra}.")
    return mapping


def _derive_stream_seed(base_seed: int, role: str) -> int:
    payload = f"longlive-stage2-balanced-sampler\0{base_seed}\0{role}".encode()
    # torch.Generator.manual_seed accepts signed 64-bit-compatible positive
    # values on every supported backend.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _normalize_spatial_shape(value: Any, label: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a two-integer latent spatial shape.")
    values = tuple(value)
    if len(values) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in values
    ):
        raise ValueError(f"{label} must be a two-integer latent spatial shape.")
    shape = (int(values[0]), int(values[1]))
    if shape not in STAGE2_ALLOWED_SPATIAL_SHAPES:
        raise ValueError(
            f"{label} must be one of {STAGE2_ALLOWED_SPATIAL_SHAPES}, got {shape}."
        )
    return shape


def _shape_key(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def _dataset_action_spatial_hash(
    action_ids: Sequence[str], spatial_shapes: Sequence[tuple[int, int]]
) -> str:
    return canonical_json_sha256(
        [
            {
                "index": index,
                "action_id": action_id,
                "latent_spatial_shape": list(spatial_shapes[index]),
            }
            for index, action_id in enumerate(action_ids)
        ]
    )


def rotating_action_batch_counts(completed_batches: int) -> tuple[int, int, int]:
    """Return 22/21/21 with the 22 slot rotated after every committed batch."""

    completed_batches = _require_plain_int(
        completed_batches, "completed_batches", minimum=0
    )
    extra_index = completed_batches % 3
    return tuple(22 if index == extra_index else 21 for index in range(3))


class Stage2BalancedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Deterministic stateful DP-global batch sampler for one model role.

    Call :meth:`next_global_batch` only when beginning an optimizer attempt and
    snapshot :meth:`state_dict` immediately beforehand.  A non-finite retry can
    restore that snapshot to replay the exact same batch and RNG state.
    """

    def __init__(
        self,
        action_ids: Sequence[str],
        spatial_shapes: Sequence[Sequence[int]],
        *,
        action_order: Sequence[str],
        base_seed: int,
        role: str,
        microbatch_size_per_device: int = 2,
    ) -> None:
        super().__init__()
        if role not in STAGE2_SAMPLER_ROLES:
            raise ValueError(
                f"role must be one of {STAGE2_SAMPLER_ROLES}, got {role!r}."
            )
        if (
            isinstance(base_seed, bool)
            or not isinstance(base_seed, int)
            or base_seed < 0
        ):
            raise ValueError("base_seed must be a non-negative integer.")
        self.role = role
        self.base_seed = base_seed
        self.stream_seed = _derive_stream_seed(base_seed, role)
        self.action_order = tuple(action_order)
        if len(self.action_order) != 3 or len(set(self.action_order)) != 3:
            raise ValueError(
                "action_order must contain exactly three unique action ids."
            )
        if any(
            not isinstance(action_id, str)
            or not action_id
            or action_id != action_id.strip()
            for action_id in self.action_order
        ):
            raise ValueError(
                "action_order values must be non-empty strings without surrounding whitespace."
            )

        self.action_ids = tuple(action_ids)
        if len(self.action_ids) != STAGE2_NUM_SAMPLES:
            raise ValueError(
                f"Stage-2 balanced sampling requires exactly {STAGE2_NUM_SAMPLES} "
                f"samples, got {len(self.action_ids)}."
            )
        unknown = sorted(set(self.action_ids) - set(self.action_order))
        if unknown:
            raise ValueError(f"Dataset contains unknown action ids: {unknown}.")
        counts = Counter(self.action_ids)
        if set(counts) != set(self.action_order):
            raise ValueError(
                "Stage-2 dataset must contain every configured action; got counts "
                f"{dict(counts)}."
            )
        self.action_populations = {
            action_id: counts[action_id] for action_id in self.action_order
        }

        if len(spatial_shapes) != STAGE2_NUM_SAMPLES:
            raise ValueError(
                "Stage-2 balanced sampling requires one spatial shape for every "
                f"sample, got {len(spatial_shapes)} for {STAGE2_NUM_SAMPLES} samples."
            )
        self.spatial_shapes = tuple(
            _normalize_spatial_shape(value, f"spatial_shapes[{index}]")
            for index, value in enumerate(spatial_shapes)
        )
        present_shapes = set(self.spatial_shapes)
        self.spatial_shape_order = tuple(
            shape for shape in STAGE2_ALLOWED_SPATIAL_SHAPES if shape in present_shapes
        )
        if (
            isinstance(microbatch_size_per_device, bool)
            or not isinstance(microbatch_size_per_device, int)
            or microbatch_size_per_device not in (1, 2)
        ):
            raise ValueError(
                "microbatch_size_per_device must be 1 or 2 for the locked Stage-2 "
                "global-batch profiles."
            )
        self.microbatch_size_per_device = microbatch_size_per_device
        self.dataset_action_spatial_sha256 = _dataset_action_spatial_hash(
            self.action_ids, self.spatial_shapes
        )
        self._base_indices = {
            (action_id, shape): tuple(
                index
                for index, (sample_action, sample_shape) in enumerate(
                    zip(self.action_ids, self.spatial_shapes)
                )
                if sample_action == action_id and sample_shape == shape
            )
            for action_id in self.action_order
            for shape in self.spatial_shape_order
        }
        self._base_indices = {
            group: indices for group, indices in self._base_indices.items() if indices
        }
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.stream_seed)
        self._orders: dict[tuple[str, tuple[int, int]], list[int]] = {
            group: [] for group in self._base_indices
        }
        self._queue_epochs = {group: -1 for group in self._base_indices}
        self._cursors = {group: 0 for group in self._base_indices}
        self.completed_batches = 0
        for action_id in self.action_order:
            for shape in self.spatial_shape_order:
                group = (action_id, shape)
                if group in self._base_indices:
                    self._reshuffle_group(group, avoid=frozenset())

        # Feasibility depends on the fixed action/shape population and the three
        # possible 22/21/21 rotations, not on queue order.  Reject a dataset that
        # can never form shape-homogeneous rank-local microbatches before training.
        for rotation in range(len(self.action_order)):
            self._choose_shape_allocations(rotating_action_batch_counts(rotation))

    @property
    def stream_epoch(self) -> int:
        return self.completed_batches // STAGE2_BATCHES_PER_STREAM_EPOCH

    @property
    def batch_cursor(self) -> int:
        return self.completed_batches % STAGE2_BATCHES_PER_STREAM_EPOCH

    @property
    def extra_slot_cursor(self) -> int:
        return self.completed_batches % len(self.action_order)

    def _reshuffle_group(
        self,
        group: tuple[str, tuple[int, int]],
        *,
        avoid: frozenset[int],
    ) -> None:
        base = self._base_indices[group]
        permutation = torch.randperm(
            len(base), generator=self._generator, device="cpu"
        ).tolist()
        order = [base[position] for position in permutation]
        if avoid:
            # A queue wrap can occur in the middle of a global batch.  Keep the
            # new epoch a full deterministic permutation while moving any
            # already selected rows behind unseen rows, preventing duplicate
            # samples inside that one batch.
            order = [index for index in order if index not in avoid] + [
                index for index in order if index in avoid
            ]
        self._orders[group] = order
        self._cursors[group] = 0
        self._queue_epochs[group] += 1

    def _take_group(self, group: tuple[str, tuple[int, int]], count: int) -> list[int]:
        if count > len(self._base_indices[group]):
            action_id, shape = group
            raise RuntimeError(
                f"Cannot draw {count} unique {action_id!r}/{shape} samples from a "
                f"population of {len(self._base_indices[group])}."
            )
        selected: list[int] = []
        while len(selected) < count:
            cursor = self._cursors[group]
            order = self._orders[group]
            if cursor == len(order):
                self._reshuffle_group(group, avoid=frozenset(selected))
                cursor = 0
                order = self._orders[group]
            remaining = count - len(selected)
            take = min(remaining, len(order) - cursor)
            selected.extend(order[cursor : cursor + take])
            self._cursors[group] = cursor + take
        if len(selected) != len(set(selected)):
            action_id, shape = group
            raise RuntimeError(
                "Internal sampler error: duplicated "
                f"{action_id!r}/{shape} sample in one batch."
            )
        return selected

    def _group_consumed(self, action_id: str, shape: tuple[int, int]) -> int:
        group = (action_id, shape)
        if group not in self._base_indices:
            return 0
        return (
            self._queue_epochs[group] * len(self._base_indices[group])
            + self._cursors[group]
        )

    def _action_allocation_candidates(
        self, action_id: str, count: int
    ) -> list[tuple[tuple[int, ...], int]]:
        """Return unique-draw allocations ranked by long-run shape fairness."""

        candidates: list[tuple[tuple[int, ...], int]] = []

        def visit(shape_index: int, remaining: int, values: list[int]) -> None:
            if shape_index == len(self.spatial_shape_order):
                if remaining == 0:
                    allocation = tuple(values)
                    consumed = [
                        self._group_consumed(action_id, shape)
                        for shape in self.spatial_shape_order
                    ]
                    projected_total = sum(consumed) + count
                    # Integer-only proportional-deficit score.  This keeps each
                    # action's two orientation streams close to its immutable
                    # source population while the global divisibility constraint
                    # decides only between equally valid nearby allocations.
                    score = 0
                    for index, shape in enumerate(self.spatial_shape_order):
                        population = len(self._base_indices.get((action_id, shape), ()))
                        score += abs(
                            (consumed[index] + allocation[index])
                            * self.action_populations[action_id]
                            - projected_total * population
                        )
                    candidates.append((allocation, score))
                return

            shape = self.spatial_shape_order[shape_index]
            population = len(self._base_indices.get((action_id, shape), ()))
            upper = min(remaining, population)
            for value in range(upper + 1):
                visit(shape_index + 1, remaining - value, [*values, value])

        visit(0, count, [])
        return candidates

    def _choose_shape_allocations(
        self, counts: Sequence[int]
    ) -> dict[str, tuple[int, ...]]:
        per_action = [
            self._action_allocation_candidates(action_id, int(count))
            for action_id, count in zip(self.action_order, counts)
        ]
        if any(not candidates for candidates in per_action):
            raise RuntimeError(
                "Stage-2 action/orientation populations cannot supply one unique "
                "22/21/21 global batch. Use a corrected cache manifest or the "
                "micro1 x accumulation8 fallback."
            )

        best_key: tuple[Any, ...] | None = None
        best_choice: tuple[tuple[tuple[int, ...], int], ...] | None = None
        for choice in product(*per_action):
            shape_totals = tuple(
                sum(candidate[0][shape_index] for candidate in choice)
                for shape_index in range(len(self.spatial_shape_order))
            )
            if any(
                total % self.microbatch_size_per_device != 0 for total in shape_totals
            ):
                continue
            allocations = tuple(value for candidate in choice for value in candidate[0])
            key = (sum(candidate[1] for candidate in choice), allocations)
            if best_key is None or key < best_key:
                best_key = key
                best_choice = choice

        if best_choice is None:
            populations = {
                action_id: {
                    _shape_key(shape): len(
                        self._base_indices.get((action_id, shape), ())
                    )
                    for shape in self.spatial_shape_order
                }
                for action_id in self.action_order
            }
            raise RuntimeError(
                "Stage-2 cannot satisfy both global 22/21/21 action balance and "
                f"shape-homogeneous micro{self.microbatch_size_per_device} batches "
                f"for populations={populations}. Select microbatch_size_per_device=1 "
                "with gradient_accumulation_steps=8, or correct the action/orientation "
                "dataset distribution."
            )
        return {
            action_id: best_choice[index][0]
            for index, action_id in enumerate(self.action_order)
        }

    def next_global_batch(self) -> list[int]:
        counts = rotating_action_batch_counts(self.completed_batches)
        allocations = self._choose_shape_allocations(counts)
        by_shape: dict[tuple[int, int], list[int]] = {
            shape: [] for shape in self.spatial_shape_order
        }
        for action_id in self.action_order:
            for shape_index, shape in enumerate(self.spatial_shape_order):
                count = allocations[action_id][shape_index]
                if count:
                    by_shape[shape].extend(self._take_group((action_id, shape), count))

        micro_groups: list[list[int]] = []
        for shape in self.spatial_shape_order:
            indices = by_shape[shape]
            if len(indices) % self.microbatch_size_per_device:
                raise RuntimeError(
                    "Internal sampler error: shape allocation is not divisible by "
                    "the rank-local microbatch size."
                )
            permutation = torch.randperm(
                len(indices), generator=self._generator, device="cpu"
            ).tolist()
            shuffled = [indices[position] for position in permutation]
            micro_groups.extend(
                shuffled[offset : offset + self.microbatch_size_per_device]
                for offset in range(0, len(shuffled), self.microbatch_size_per_device)
            )
        group_order = torch.randperm(
            len(micro_groups), generator=self._generator, device="cpu"
        ).tolist()
        batch = [index for position in group_order for index in micro_groups[position]]
        if len(batch) != STAGE2_GLOBAL_BATCH_SIZE:
            raise RuntimeError(
                f"Internal sampler error: built batch of {len(batch)}, expected 64."
            )
        if len(batch) != len(set(batch)):
            raise RuntimeError(
                "Internal sampler error: global batch contains duplicates."
            )
        observed = Counter(self.action_ids[index] for index in batch)
        expected = {
            action_id: count for action_id, count in zip(self.action_order, counts)
        }
        if dict(observed) != expected:
            raise RuntimeError(
                f"Internal sampler error: action composition {dict(observed)} != {expected}."
            )
        for offset in range(0, len(batch), self.microbatch_size_per_device):
            microbatch = batch[offset : offset + self.microbatch_size_per_device]
            shapes = {self.spatial_shapes[index] for index in microbatch}
            if len(shapes) != 1:
                raise RuntimeError(
                    "Internal sampler error: rank-local microbatch mixes latent shapes."
                )
        self.completed_batches += 1
        return batch

    def __iter__(self) -> Iterator[list[int]]:
        # Yield only the unfinished portion of this sampler-local 10-batch
        # stream epoch.  G's stream epoch is the research "generator epoch";
        # F advances independently and never changes G's epoch/cursor.
        remaining = STAGE2_BATCHES_PER_STREAM_EPOCH - self.batch_cursor
        for _ in range(remaining):
            yield self.next_global_batch()

    def __len__(self) -> int:
        return STAGE2_BATCHES_PER_STREAM_EPOCH - self.batch_cursor

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": STAGE2_SAMPLER_STATE_SCHEMA,
            "role": self.role,
            "base_seed": self.base_seed,
            "stream_seed": self.stream_seed,
            "dataset_action_spatial_sha256": self.dataset_action_spatial_sha256,
            "action_order": list(self.action_order),
            "spatial_shape_order": [list(shape) for shape in self.spatial_shape_order],
            "microbatch_size_per_device": self.microbatch_size_per_device,
            "global_batch_size": STAGE2_GLOBAL_BATCH_SIZE,
            "per_action_batch_counts": list(STAGE2_PER_ACTION_BATCH_COUNTS),
            "batches_per_stream_epoch": STAGE2_BATCHES_PER_STREAM_EPOCH,
            "completed_batches": self.completed_batches,
            "stream_epoch": self.stream_epoch,
            "batch_cursor": self.batch_cursor,
            "extra_slot_cursor": self.extra_slot_cursor,
            "action_states": {
                action_id: {
                    "groups": {
                        _shape_key(shape): {
                            "queue_epoch": self._queue_epochs[(action_id, shape)],
                            "cursor": self._cursors[(action_id, shape)],
                            "order": list(self._orders[(action_id, shape)]),
                        }
                        for shape in self.spatial_shape_order
                        if (action_id, shape) in self._base_indices
                    }
                }
                for action_id in self.action_order
            },
            "generator_state": self._generator.get_state().clone(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        state = _require_exact_mapping(
            state_dict, label="Stage-2 sampler state", expected_keys=_STATE_KEYS
        )
        locked_values = {
            "schema": STAGE2_SAMPLER_STATE_SCHEMA,
            "role": self.role,
            "base_seed": self.base_seed,
            "stream_seed": self.stream_seed,
            "dataset_action_spatial_sha256": self.dataset_action_spatial_sha256,
            "action_order": list(self.action_order),
            "spatial_shape_order": [list(shape) for shape in self.spatial_shape_order],
            "microbatch_size_per_device": self.microbatch_size_per_device,
            "global_batch_size": STAGE2_GLOBAL_BATCH_SIZE,
            "per_action_batch_counts": list(STAGE2_PER_ACTION_BATCH_COUNTS),
            "batches_per_stream_epoch": STAGE2_BATCHES_PER_STREAM_EPOCH,
        }
        for key, expected in locked_values.items():
            if state[key] != expected or type(state[key]) is not type(expected):
                raise RuntimeError(
                    f"Stage-2 sampler state {key} mismatch: expected {expected!r}, "
                    f"got {state[key]!r}."
                )

        completed = _require_plain_int(
            state["completed_batches"], "completed_batches", minimum=0
        )
        expected_derived = {
            "stream_epoch": completed // STAGE2_BATCHES_PER_STREAM_EPOCH,
            "batch_cursor": completed % STAGE2_BATCHES_PER_STREAM_EPOCH,
            "extra_slot_cursor": completed % len(self.action_order),
        }
        for key, expected in expected_derived.items():
            actual = _require_plain_int(state[key], key, minimum=0)
            if actual != expected:
                raise RuntimeError(
                    f"Stage-2 sampler state {key} is inconsistent with "
                    f"completed_batches: {actual} != {expected}."
                )

        action_states = _require_exact_mapping(
            state["action_states"],
            label="Stage-2 sampler action_states",
            expected_keys=set(self.action_order),
        )
        candidate_orders: dict[tuple[str, tuple[int, int]], list[int]] = {}
        candidate_epochs: dict[tuple[str, tuple[int, int]], int] = {}
        candidate_cursors: dict[tuple[str, tuple[int, int]], int] = {}
        for action_id in self.action_order:
            action_state = _require_exact_mapping(
                action_states[action_id],
                label=f"Stage-2 sampler action state {action_id!r}",
                expected_keys=_ACTION_STATE_KEYS,
            )
            expected_group_keys = {
                _shape_key(shape)
                for shape in self.spatial_shape_order
                if (action_id, shape) in self._base_indices
            }
            groups = _require_exact_mapping(
                action_state["groups"],
                label=f"Stage-2 sampler action groups {action_id!r}",
                expected_keys=expected_group_keys,
            )
            for shape in self.spatial_shape_order:
                group = (action_id, shape)
                if group not in self._base_indices:
                    continue
                key = _shape_key(shape)
                group_state = _require_exact_mapping(
                    groups[key],
                    label=f"Stage-2 sampler group {action_id!r}/{key}",
                    expected_keys=_GROUP_STATE_KEYS,
                )
                epoch = _require_plain_int(
                    group_state["queue_epoch"],
                    f"{action_id}/{key}.queue_epoch",
                    minimum=0,
                )
                cursor = _require_plain_int(
                    group_state["cursor"],
                    f"{action_id}/{key}.cursor",
                    minimum=0,
                )
                order = group_state["order"]
                if not isinstance(order, list) or any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in order
                ):
                    raise RuntimeError(
                        f"{action_id}/{key}.order must be a list of integer indices."
                    )
                if len(order) != len(self._base_indices[group]) or set(order) != set(
                    self._base_indices[group]
                ):
                    raise RuntimeError(
                        f"{action_id}/{key}.order must be an exact permutation of "
                        "its action/shape rows."
                    )
                if cursor > len(order):
                    raise RuntimeError(
                        f"{action_id}/{key}.cursor {cursor} exceeds queue length "
                        f"{len(order)}."
                    )
                candidate_orders[group] = list(order)
                candidate_epochs[group] = epoch
                candidate_cursors[group] = cursor

            action_index = self.action_order.index(action_id)
            full_rotations, remainder = divmod(completed, len(self.action_order))
            expected_consumed = full_rotations * STAGE2_GLOBAL_BATCH_SIZE + sum(
                rotating_action_batch_counts(update)[action_index]
                for update in range(remainder)
            )
            actual_consumed = sum(
                candidate_epochs[(action_id, shape)]
                * len(self._base_indices[(action_id, shape)])
                + candidate_cursors[(action_id, shape)]
                for shape in self.spatial_shape_order
                if (action_id, shape) in self._base_indices
            )
            if actual_consumed != expected_consumed:
                raise RuntimeError(
                    f"Stage-2 sampler {action_id!r} queue epoch/cursor state consumed "
                    f"{actual_consumed} samples, expected {expected_consumed} from "
                    f"completed_batches={completed}."
                )

        generator_state = state["generator_state"]
        if (
            not isinstance(generator_state, torch.Tensor)
            or generator_state.dtype != torch.uint8
            or generator_state.device.type != "cpu"
            or generator_state.ndim != 1
        ):
            raise RuntimeError(
                "Stage-2 sampler generator_state must be a one-dimensional CPU uint8 tensor."
            )
        candidate_generator = torch.Generator(device="cpu")
        try:
            candidate_generator.set_state(generator_state.clone())
        except RuntimeError as exc:
            raise RuntimeError("Invalid Stage-2 sampler generator_state.") from exc

        # Commit only after every field has passed; a malformed checkpoint
        # cannot leave a half-restored sampler behind.
        self.completed_batches = completed
        self._orders = candidate_orders
        self._queue_epochs = candidate_epochs
        self._cursors = candidate_cursors
        self._generator = candidate_generator


def partition_stage2_global_batch(
    global_batch: Sequence[int],
    *,
    rank: int,
    world_size: int = 8,
    microbatch_size_per_device: int = 2,
    gradient_accumulation_steps: int = 4,
    spatial_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Split one committed 64-row schedule into rank-local microbatches."""

    for value, label, minimum in (
        (rank, "rank", 0),
        (world_size, "world_size", 1),
        (microbatch_size_per_device, "microbatch_size_per_device", 1),
        (gradient_accumulation_steps, "gradient_accumulation_steps", 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}.")
    if rank >= world_size:
        raise ValueError(f"rank must be less than world_size, got {rank}/{world_size}.")
    expected = world_size * microbatch_size_per_device * gradient_accumulation_steps
    if expected != STAGE2_GLOBAL_BATCH_SIZE:
        raise ValueError(
            "Stage-2 partition must preserve effective global batch 64; got "
            f"world_size*microbatch*accumulation={expected}."
        )
    if len(global_batch) != STAGE2_GLOBAL_BATCH_SIZE:
        raise ValueError(
            f"global_batch must contain exactly 64 indices, got {len(global_batch)}."
        )
    if any(
        isinstance(index, bool) or not isinstance(index, int) for index in global_batch
    ):
        raise ValueError("global_batch indices must be integers.")
    if any(index < 0 or index >= STAGE2_NUM_SAMPLES for index in global_batch):
        raise ValueError("global_batch indices must be in the Stage-2 range [0, 600).")
    if len(set(global_batch)) != len(global_batch):
        raise ValueError("global_batch must not contain duplicate indices.")
    if len(spatial_shapes) != STAGE2_NUM_SAMPLES:
        raise ValueError(
            "spatial_shapes must contain exactly one shape for each of the 600 "
            "Stage-2 rows."
        )
    normalized_shapes = tuple(
        _normalize_spatial_shape(value, f"spatial_shapes[{index}]")
        for index, value in enumerate(spatial_shapes)
    )

    microbatches: list[tuple[int, ...]] = []
    offset = 0
    for _ in range(gradient_accumulation_steps):
        global_microbatch_size = world_size * microbatch_size_per_device
        block = global_batch[offset : offset + global_microbatch_size]
        start = rank * microbatch_size_per_device
        local = tuple(block[start : start + microbatch_size_per_device])
        if len(local) != microbatch_size_per_device:
            raise RuntimeError("Internal Stage-2 partitioning error.")
        if len({normalized_shapes[index] for index in local}) != 1:
            raise ValueError(
                "Stage-2 rank-local microbatch mixes latent spatial shapes. Build the "
                "global batch with a sampler configured for this microbatch size, or "
                "use microbatch_size_per_device=1 with accumulation=8."
            )
        microbatches.append(local)
        offset += global_microbatch_size
    return tuple(microbatches)


@dataclass
class Stage2RoleSamplerStreams:
    """The two checkpointed streams; neither shares mutable state with the other."""

    fake_score: Stage2BalancedBatchSampler
    generator: Stage2BalancedBatchSampler

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": STAGE2_SAMPLER_STREAMS_SCHEMA,
            "fake_score": self.fake_score.state_dict(),
            "generator": self.generator.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        state = _require_exact_mapping(
            state_dict,
            label="Stage-2 sampler streams state",
            expected_keys={"schema", "fake_score", "generator"},
        )
        if state["schema"] != STAGE2_SAMPLER_STREAMS_SCHEMA:
            raise RuntimeError("Unsupported Stage-2 sampler streams schema.")
        before_fake = self.fake_score.state_dict()
        before_generator = self.generator.state_dict()
        try:
            self.fake_score.load_state_dict(state["fake_score"])
            self.generator.load_state_dict(state["generator"])
        except Exception:
            self.fake_score.load_state_dict(before_fake)
            self.generator.load_state_dict(before_generator)
            raise


def build_stage2_role_samplers(
    action_ids: Sequence[str],
    spatial_shapes: Sequence[Sequence[int]],
    *,
    action_order: Sequence[str],
    base_seed: int,
    microbatch_size_per_device: int = 2,
) -> Stage2RoleSamplerStreams:
    return Stage2RoleSamplerStreams(
        fake_score=Stage2BalancedBatchSampler(
            action_ids,
            spatial_shapes,
            action_order=action_order,
            base_seed=base_seed,
            role="fake_score",
            microbatch_size_per_device=microbatch_size_per_device,
        ),
        generator=Stage2BalancedBatchSampler(
            action_ids,
            spatial_shapes,
            action_order=action_order,
            base_seed=base_seed,
            role="generator",
            microbatch_size_per_device=microbatch_size_per_device,
        ),
    )


__all__ = [
    "STAGE2_ALLOWED_SPATIAL_SHAPES",
    "STAGE2_BATCHES_PER_STREAM_EPOCH",
    "STAGE2_GLOBAL_BATCH_SIZE",
    "STAGE2_PER_ACTION_BATCH_COUNTS",
    "Stage2BalancedBatchSampler",
    "Stage2RoleSamplerStreams",
    "build_stage2_role_samplers",
    "partition_stage2_global_batch",
    "rotating_action_batch_counts",
]
