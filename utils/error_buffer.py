# Adopted from https://github.com/vita-epfl/Stable-Video-Infinity
# SPDX-License-Identifier: Apache-2.0

import random
from collections import Counter

import torch


ERROR_BUFFER_STATE_SCHEMA = "longlive_error_buffer"
ERROR_BUFFER_STATE_VERSION = 2


class ErrorBuffer:
    """Bucketed ring buffer for storing prediction errors on CPU.

    Two layouts are supported:

      * **1D (timestep-only)** — when ``num_blocks <= 0``. Buckets are keyed by
        the diffusion timestep. This is the original SVI behavior.

      * **2D (position × timestep)** — when ``num_blocks > 0``. Each entry is
        keyed by both the global block position along the sequence and the
        timestep. Inject paths can then choose:
          - ``sample(pos, t)``: match BOTH position and timestep
            (E_vid / E_noise — noise-level dependent errors)
          - ``sample_pos_any_t(pos)``: match position, sample uniformly across
            timesteps (E_img — position-dependent context corruption that is
            agnostic to the current denoising step)
          - ``sample_global()``: legacy fallback, samples uniformly everywhere

    The 2D layout encodes the teacher-forcing insight that ``noisy_suffix[i]``
    looks at clean_prefix[0..i] during training but at model rollouts during
    inference; storing prediction errors per-position therefore lets later
    blocks self-feed larger errors without any manual position ramp.

    **Sharded timestep buckets** (``shard_size > 1``):
    Each rank only allocates the timestep buckets it owns
    (``t_bucket % shard_size == shard_rank``), reducing per-rank CPU memory
    by ~``shard_size`` times.  Typically ``shard_rank/shard_size`` are set to
    ``sp_rank/sp_size`` so that sharding is per-SP-rank and saving follows
    the same per-SP-rank pattern as the 2D position split.  On ``add()``,
    non-owned buckets are silently skipped; on ``sample()``, non-owned buckets
    are remapped to the nearest owned one.
    """

    def __init__(
        self,
        num_buckets=40,
        max_size_per_bucket=50,
        num_train_timesteps=1000,
        modulate_factor=0.3,
        replacement_strategy="random",
        num_blocks=0,
        global_block_offset=0,
        shard_rank=0,
        shard_size=1,
    ):
        self.num_buckets = int(num_buckets)
        self.max_size = int(max_size_per_bucket)
        self.num_train_timesteps = int(num_train_timesteps)
        self.modulate_factor = float(modulate_factor)
        self.replacement_strategy = replacement_strategy
        if self.num_buckets <= 0:
            raise ValueError("num_buckets must be positive.")
        if self.max_size <= 0:
            raise ValueError("max_size_per_bucket must be positive.")
        if self.num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive.")
        self.bucket_width = self.num_train_timesteps / self.num_buckets
        self.num_blocks = int(num_blocks) if num_blocks else 0
        # ``global_block_offset`` is only used for stats / debug display so
        # users can tell which absolute positions of the full sequence this
        # buffer covers (the LAST SP rank carries the highest accumulated
        # error positions). It does NOT participate in bucket keying.
        self.global_block_offset = int(global_block_offset)

        self.shard_rank = int(shard_rank)
        self.shard_size = max(int(shard_size), 1)
        if not 0 <= self.shard_rank < self.shard_size:
            raise ValueError(
                f"shard_rank must be in [0, {self.shard_size}), got "
                f"{self.shard_rank}."
            )
        self._owned_t_buckets = sorted(
            t
            for t in range(self.num_buckets)
            if t % self.shard_size == self.shard_rank
        )
        if not self._owned_t_buckets:
            raise ValueError(
                f"shard_rank={self.shard_rank}/{self.shard_size} owns no timestep "
                f"buckets out of num_buckets={self.num_buckets}."
            )

        if self.num_blocks > 0:
            self.buckets = {
                (p, t): []
                for p in range(self.num_blocks)
                for t in self._owned_t_buckets
            }
        else:
            self.buckets = {t: [] for t in self._owned_t_buckets}
        self.total_added = 0

    # ------------------------------------------------------------------ keys
    def _t_bucket(self, timestep_index):
        b = int(timestep_index / self.bucket_width)
        return max(0, min(b, self.num_buckets - 1))

    def _is_owned_t(self, t_bucket):
        return self.shard_size <= 1 or (t_bucket % self.shard_size == self.shard_rank)

    def _nearest_owned_t(self, t_bucket):
        """Remap ``t_bucket`` to the closest owned timestep bucket."""
        if self.shard_size <= 1 or self._is_owned_t(t_bucket):
            return t_bucket
        fwd = (self.shard_rank - t_bucket % self.shard_size) % self.shard_size
        bwd = self.shard_size - fwd
        t_up, t_down = t_bucket + fwd, t_bucket - bwd
        up_ok = 0 <= t_up < self.num_buckets
        down_ok = 0 <= t_down < self.num_buckets
        if up_ok and down_ok:
            return t_up if fwd <= bwd else t_down
        return t_up if up_ok else t_down

    def _make_key(self, t_bucket, block_pos):
        if self.num_blocks > 0:
            assert block_pos is not None, "block_pos required when num_blocks>0"
            p = max(0, min(int(block_pos), self.num_blocks - 1))
            return (p, t_bucket)
        return t_bucket

    @staticmethod
    def _normalize_spatial_shape(expected_spatial_shape):
        if expected_spatial_shape is None:
            return None
        shape = tuple(int(value) for value in expected_spatial_shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError(
                "expected_spatial_shape must contain positive (height, width), "
                f"got {expected_spatial_shape!r}."
            )
        return shape

    @classmethod
    def _filter_spatial_shape(cls, entries, expected_spatial_shape):
        expected = cls._normalize_spatial_shape(expected_spatial_shape)
        if expected is None:
            return entries
        return [
            entry
            for entry in entries
            if torch.is_tensor(entry)
            and entry.ndim >= 2
            and tuple(entry.shape[-2:]) == expected
        ]

    # ------------------------------------------------------------------ add
    def add(self, error_block, timestep_index, block_pos=None):
        """Store a single block error into the matching bucket.

        Args:
            error_block: (block_size, C, H, W) tensor
            timestep_index: int, raw index in [0, num_train_timesteps)
            block_pos: int, global block position; required iff num_blocks>0
        """
        t = self._t_bucket(timestep_index)
        if not self._is_owned_t(t):
            return
        key = self._make_key(t, block_pos)
        # Store in the source dtype on CPU to match SVI (which keeps bf16),
        # cutting buffer memory in half vs. casting to fp32.
        entry = error_block.detach().to("cpu", copy=True)

        buf = self.buckets[key]
        if len(buf) < self.max_size:
            buf.append(entry)
        else:
            if self.replacement_strategy == "fifo":
                buf.pop(0)
                buf.append(entry)
            elif self.replacement_strategy == "l2":
                # A base bucket may contain both orientations.  Its capacity is
                # still shared, but L2 distance is only defined for identical
                # tensor shapes.  If no comparable entry exists, replace one
                # slot randomly instead of stacking incompatible tensors.
                comparable_indices = [
                    index
                    for index, candidate in enumerate(buf)
                    if tuple(candidate.shape) == tuple(entry.shape)
                ]
                if comparable_indices:
                    stacked = torch.stack(
                        [buf[index] for index in comparable_indices]
                    )
                    dists = (
                        stacked.float() - entry.unsqueeze(0).float()
                    ).flatten(1).norm(dim=1)
                    closest = comparable_indices[torch.argmin(dists).item()]
                    buf[closest] = entry
                else:
                    buf[random.randint(0, len(buf) - 1)] = entry
            else:  # "random" (default)
                idx = random.randint(0, self.max_size - 1)
                buf[idx] = entry
        self.total_added += 1

    # ------------------------------------------------------------------ sample
    def sample(
        self,
        timestep_index,
        device,
        dtype,
        block_pos=None,
        expected_spatial_shape=None,
    ):
        """Sample one entry matching (block_pos, timestep_index) when 2D, or
        just timestep_index when 1D.  Non-owned timestep buckets are
        transparently remapped to the nearest owned one.  Returns None if the
        (remapped) bucket is empty."""
        t = self._nearest_owned_t(self._t_bucket(timestep_index))
        key = self._make_key(t, block_pos)
        buf = self._filter_spatial_shape(
            self.buckets[key], expected_spatial_shape
        )
        if not buf:
            return None
        err = random.choice(buf)
        return self._modulate(err).to(device=device, dtype=dtype)

    def sample_pos_any_t(
        self, block_pos, device, dtype, expected_spatial_shape=None
    ):
        """For 2D buffers: sample at the given position, with random timestep.

        This is the natural choice for context (E_img) injection — the clean
        prefix is the result of a full ODE rollout so its accumulated error
        could have originated at any timestep, but its magnitude scales with
        position along the sequence.

        Falls back to ``sample_global`` when the buffer is 1D.
        Only owned timestep buckets are scanned.
        """
        if self.num_blocks <= 0:
            return self.sample_global(
                device, dtype, expected_spatial_shape=expected_spatial_shape
            )
        p = max(0, min(int(block_pos), self.num_blocks - 1))
        all_entries = []
        for t in self._owned_t_buckets:
            all_entries.extend(self.buckets[(p, t)])
        all_entries = self._filter_spatial_shape(
            all_entries, expected_spatial_shape
        )
        if not all_entries:
            return None
        err = random.choice(all_entries)
        return self._modulate(err).to(device=device, dtype=dtype)

    def sample_global(self, device, dtype, expected_spatial_shape=None):
        """Sample one entry uniformly from ALL buckets (legacy SVI E_img)."""
        all_entries = []
        for buf in self.buckets.values():
            all_entries.extend(buf)
        all_entries = self._filter_spatial_shape(
            all_entries, expected_spatial_shape
        )
        if not all_entries:
            return None
        err = random.choice(all_entries)
        return self._modulate(err).to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ misc
    def _modulate(self, err):
        if self.modulate_factor > 0:
            lo = 1.0 - self.modulate_factor
            hi = 1.0 + self.modulate_factor
            err = err * random.uniform(lo, hi)
        return err

    def is_empty(self):
        return self.total_added == 0

    def has_pos(self, block_pos):
        """Whether ANY owned timestep bucket at ``block_pos`` has samples (2D only)."""
        if self.num_blocks <= 0:
            return not self.is_empty()
        p = max(0, min(int(block_pos), self.num_blocks - 1))
        return any(len(self.buckets[(p, t)]) > 0 for t in self._owned_t_buckets)

    def stats(self):
        filled = sum(1 for b in self.buckets.values() if len(b) > 0)
        total = sum(len(b) for b in self.buckets.values())
        num_owned_t = len(self._owned_t_buckets)
        denom = self.num_blocks * num_owned_t if self.num_blocks > 0 else num_owned_t
        out = {
            "total_added": self.total_added,
            "filled_buckets": f"{filled}/{denom}",
            "total_entries": total,
        }
        shape_counts = Counter(
            tuple(entry.shape[-2:])
            for bucket in self.buckets.values()
            for entry in bucket
            if torch.is_tensor(entry) and entry.ndim >= 2
        )
        out["entries_by_spatial_shape"] = {
            f"{height}x{width}": count
            for (height, width), count in sorted(shape_counts.items())
        }
        if self.shard_size > 1:
            out["shard"] = f"shard_rank={self.shard_rank}/{self.shard_size} ({num_owned_t}/{self.num_buckets} t-buckets)"
        if self.num_blocks > 0:
            lo = self.global_block_offset
            hi = self.global_block_offset + self.num_blocks
            out["global_block_range"] = f"[{lo},{hi})"
        return out

    def state_dict(self):
        # Keys are tuples (pos, t) when 2D — torch.save handles them fine
        # via pickle. We serialize the bucket layout so loaders can validate.
        return {
            "state_schema": ERROR_BUFFER_STATE_SCHEMA,
            "state_version": ERROR_BUFFER_STATE_VERSION,
            "buckets": {k: list(v) for k, v in self.buckets.items()},
            "total_added": self.total_added,
            "num_blocks": self.num_blocks,
            "num_buckets": self.num_buckets,
            "max_size_per_bucket": self.max_size,
            "num_train_timesteps": self.num_train_timesteps,
            "modulate_factor": self.modulate_factor,
            "replacement_strategy": self.replacement_strategy,
            "global_block_offset": self.global_block_offset,
            "shard_rank": self.shard_rank,
            "shard_size": self.shard_size,
        }

    def load_state_dict(self, state, strict_offset=True, strict_schema=False):
        """Restore buckets from a serialized state.

        Args:
            state: dict produced by ``state_dict``.
            strict_offset: when True (default) and the buffer is 2D,
                refuse to load if the saved ``global_block_offset`` does
                not match the current one. This prevents the silent
                position-misalignment bug under SP, where a checkpoint
                saved by SP rank 0 (covering global blocks ``[0, B)``)
                would otherwise be loaded into SP rank 1 (which expects
                ``[B, 2B)``) and corrupt position-bucketed sampling.
                Pass ``strict_offset=False`` only for backward-compat
                with checkpoints saved before this field existed.
        """
        if strict_schema:
            self._validate_strict_state(state)

        if self.num_blocks > 0 and strict_offset:
            saved_off = state.get("global_block_offset", None)
            if saved_off is None:
                raise RuntimeError(
                    "Refusing to load: this is a 2D position-bucketed buffer "
                    "but the checkpoint has no `global_block_offset` field. "
                    "Pass strict_offset=False if you accept the misalignment risk."
                )
            if int(saved_off) != self.global_block_offset:
                raise RuntimeError(
                    f"Refusing to load: checkpoint covers global blocks "
                    f"starting at {saved_off}, but this rank covers blocks "
                    f"starting at {self.global_block_offset}. Make sure each "
                    f"SP rank loads its own per-rank checkpoint file."
                )
        # Shard check: warn but don't crash if shard layout changed (e.g.
        # resuming a non-sharded checkpoint into a sharded buffer is fine —
        # we just load whichever buckets overlap).
        saved_shard_size = int(state.get("shard_size", state.get("dp_size", 1)))
        saved_shard_rank = int(state.get("shard_rank", state.get("dp_rank", 0)))
        if (
            not strict_schema
            and (
                saved_shard_size != self.shard_size
                or saved_shard_rank != self.shard_rank
            )
        ):
            import logging
            logging.warning(
                f"[ErrorBuffer] Shard layout changed: checkpoint was "
                f"shard_rank={saved_shard_rank}/{saved_shard_size}, current is "
                f"shard_rank={self.shard_rank}/{self.shard_size}. "
                f"Loading overlapping buckets only."
            )

        saved = state["buckets"]
        # Lenient match: ignore keys that don't exist in the current layout.
        for k in self.buckets:
            if k in saved:
                self.buckets[k] = saved[k]
            elif isinstance(k, tuple):
                # Try string-form key from older serializations
                continue
            elif str(k) in saved:
                self.buckets[k] = saved[str(k)]
        self.total_added = int(state.get("total_added", 0))

    def _validate_strict_state(self, state):
        if state.get("state_schema") != ERROR_BUFFER_STATE_SCHEMA:
            raise RuntimeError(
                "ErrorBuffer strict load requires state_schema="
                f"{ERROR_BUFFER_STATE_SCHEMA!r}, got "
                f"{state.get('state_schema')!r}."
            )
        if int(state.get("state_version", -1)) != ERROR_BUFFER_STATE_VERSION:
            raise RuntimeError(
                "Unsupported ErrorBuffer state_version="
                f"{state.get('state_version')!r}; expected "
                f"{ERROR_BUFFER_STATE_VERSION}."
            )

        expected_fields = {
            "num_blocks": self.num_blocks,
            "num_buckets": self.num_buckets,
            "max_size_per_bucket": self.max_size,
            "num_train_timesteps": self.num_train_timesteps,
            "modulate_factor": self.modulate_factor,
            "replacement_strategy": self.replacement_strategy,
            "global_block_offset": self.global_block_offset,
            "shard_rank": self.shard_rank,
            "shard_size": self.shard_size,
        }
        for field, expected in expected_fields.items():
            if field not in state or state[field] != expected:
                raise RuntimeError(
                    f"ErrorBuffer strict load mismatch for {field}: "
                    f"checkpoint={state.get(field)!r}, current={expected!r}."
                )

        saved_buckets = state.get("buckets")
        if not isinstance(saved_buckets, dict):
            raise RuntimeError("ErrorBuffer strict load requires a buckets mapping.")
        if set(saved_buckets) != set(self.buckets):
            missing = sorted(set(self.buckets) - set(saved_buckets), key=repr)
            unexpected = sorted(set(saved_buckets) - set(self.buckets), key=repr)
            raise RuntimeError(
                "ErrorBuffer strict bucket-key mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )

        entry_count = 0
        for key, entries in saved_buckets.items():
            if not isinstance(entries, (list, tuple)):
                raise RuntimeError(
                    f"ErrorBuffer bucket {key!r} must contain a list/tuple."
                )
            if len(entries) > self.max_size:
                raise RuntimeError(
                    f"ErrorBuffer bucket {key!r} has {len(entries)} entries, "
                    f"exceeding capacity {self.max_size}."
                )
            for entry in entries:
                if not torch.is_tensor(entry) or entry.ndim < 2:
                    raise RuntimeError(
                        f"ErrorBuffer bucket {key!r} contains an invalid entry."
                    )
            entry_count += len(entries)

        total_added = int(state.get("total_added", -1))
        if total_added < entry_count:
            raise RuntimeError(
                f"ErrorBuffer total_added={total_added} is smaller than the "
                f"{entry_count} serialized entries."
            )


def build_error_buffer(config, num_blocks=0, global_block_offset=0,
                       shard_rank=0, shard_size=1):
    """Build an ErrorBuffer from an OmegaConf/dict config node.

    When ``num_blocks > 0`` the buffer becomes 2D (position × timestep),
    enabling teacher-forcing-aware position-dependent error injection.
    Pass ``global_block_offset`` so logs can identify which absolute slice
    of the full sequence this rank's buffer covers (e.g. the last SP rank
    is responsible for the most error-accumulated tail blocks).

    ``shard_rank`` / ``shard_size`` shard timestep buckets: each rank only
    allocates the buckets it owns, reducing per-rank CPU memory by
    ~``shard_size`` times.  Typically set to ``(sp_rank, sp_size)``.
    """
    cfg = config if isinstance(config, dict) else dict(config)
    return ErrorBuffer(
        num_buckets=cfg.get("num_buckets", 40),
        max_size_per_bucket=cfg.get("buffer_size_per_bucket", 50),
        num_train_timesteps=cfg.get("num_train_timesteps", 1000),
        modulate_factor=cfg.get("modulate_factor", 0.3),
        replacement_strategy=cfg.get("replacement_strategy", "random"),
        num_blocks=num_blocks,
        global_block_offset=global_block_offset,
        shard_rank=shard_rank,
        shard_size=shard_size,
    )
