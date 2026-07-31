from collections import defaultdict
from collections.abc import Hashable, Iterator, Sequence

import torch


class ResolutionAwareDistributedSampler(torch.utils.data.Sampler[int]):
    """Deterministic Stage-1 sampler over DP-global micro-batches.

    The locked Stage-1 layout has two DP replicas with local batch size one.
    Samples are shuffled without replacement inside each spatial bucket, paired
    into DP-global micro-batches, and then split so each DP replica receives one
    member of every pair.  SP ranks that pass the same DP rank therefore see the
    exact same stream.

    Odd bucket tails are paired across buckets only after all same-bucket pairs.
    This keeps the configured warmup prefix shape-compatible for cross-DP error
    gathering without dropping or duplicating the tail samples.
    """

    def __init__(
        self,
        dataset,
        *,
        seed: int,
        rank: int,
        num_replicas: int,
        spatial_shapes: Sequence[Hashable] | None = None,
        local_batch_size: int = 1,
        warmup_microbatches: int = 0,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.seed = int(seed)
        self.rank = int(rank)
        self.num_replicas = int(num_replicas)
        self.local_batch_size = int(local_batch_size)
        self.warmup_microbatches = int(warmup_microbatches)
        self.epoch = 0

        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas}), got {self.rank}."
            )
        if self.local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive.")
        if self.warmup_microbatches < 0:
            raise ValueError("warmup_microbatches must be non-negative.")

        global_microbatch_size = self.num_replicas * self.local_batch_size
        if self.num_replicas != 2 or self.local_batch_size != 1:
            raise ValueError(
                "The Stage-1 resolution-aware sampler requires exactly two DP "
                "replicas with local_batch_size=1; got "
                f"num_replicas={self.num_replicas}, "
                f"local_batch_size={self.local_batch_size}."
            )

        if spatial_shapes is None:
            spatial_shapes = getattr(dataset, "spatial_shapes", None)
        if spatial_shapes is None:
            raise ValueError(
                "spatial_shapes must be provided, or the dataset must expose "
                "a spatial_shapes sequence."
            )
        if len(spatial_shapes) != len(dataset):
            raise ValueError(
                f"spatial_shapes length {len(spatial_shapes)} must match dataset "
                f"length {len(dataset)}."
            )
        if len(dataset) == 0:
            raise ValueError("Resolution-aware sampling requires a non-empty dataset.")
        if len(dataset) % global_microbatch_size != 0:
            raise ValueError(
                f"Dataset length {len(dataset)} is not divisible by DP-global "
                f"micro-batch size {global_microbatch_size}; refusing to drop or "
                "oversample records."
            )

        normalized_shapes = []
        for index, shape in enumerate(spatial_shapes):
            if isinstance(shape, list):
                shape = tuple(shape)
            if not isinstance(shape, Hashable):
                raise TypeError(
                    f"spatial_shapes[{index}] must be hashable, got {type(shape).__name__}."
                )
            normalized_shapes.append(shape)
        self.spatial_shapes = tuple(normalized_shapes)

        # Validate warmup feasibility immediately instead of failing after the
        # DataLoader has already started workers.
        self._global_microbatches_for_epoch(self.epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _stable_bucket_sort_key(key: Hashable) -> tuple[str, str]:
        return type(key).__qualname__, repr(key)

    def _global_microbatches_for_epoch(self, epoch: int) -> list[tuple[int, int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(epoch))

        bucket_to_indices: dict[Hashable, list[int]] = defaultdict(list)
        for index, shape in enumerate(self.spatial_shapes):
            bucket_to_indices[shape].append(index)

        same_bucket_pairs: list[tuple[int, int]] = []
        odd_tails: list[tuple[Hashable, int]] = []
        for bucket in sorted(bucket_to_indices, key=self._stable_bucket_sort_key):
            indices = bucket_to_indices[bucket]
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[position] for position in order]
            pair_end = len(shuffled) - (len(shuffled) % 2)
            same_bucket_pairs.extend(
                (shuffled[position], shuffled[position + 1])
                for position in range(0, pair_end, 2)
            )
            if pair_end != len(shuffled):
                odd_tails.append((bucket, shuffled[-1]))

        if len(odd_tails) % 2 != 0:
            raise RuntimeError(
                "Internal sampler error: an even dataset produced an odd number "
                "of per-bucket tail samples."
            )

        if same_bucket_pairs:
            pair_order = torch.randperm(
                len(same_bucket_pairs), generator=generator
            ).tolist()
            same_bucket_pairs = [same_bucket_pairs[i] for i in pair_order]

        cross_bucket_pairs = [
            (odd_tails[position][1], odd_tails[position + 1][1])
            for position in range(0, len(odd_tails), 2)
        ]
        if cross_bucket_pairs:
            cross_order = torch.randperm(
                len(cross_bucket_pairs), generator=generator
            ).tolist()
            cross_bucket_pairs = [cross_bucket_pairs[i] for i in cross_order]

        if len(same_bucket_pairs) < self.warmup_microbatches:
            raise ValueError(
                f"Only {len(same_bucket_pairs)} same-resolution global "
                f"micro-batches are available, fewer than the requested warmup "
                f"prefix of {self.warmup_microbatches}."
            )

        microbatches = same_bucket_pairs + cross_bucket_pairs
        expected = len(self.dataset) // 2
        if len(microbatches) != expected:
            raise RuntimeError(
                f"Internal sampler error: built {len(microbatches)} global "
                f"micro-batches, expected {expected}."
            )
        flattened = [index for pair in microbatches for index in pair]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(
            range(len(self.dataset))
        ):
            raise RuntimeError(
                "Internal sampler error: epoch schedule is not an exact "
                "permutation of the dataset."
            )
        return microbatches

    def global_microbatches(self) -> tuple[tuple[int, int], ...]:
        """Return the deterministic DP-global schedule for preflight/tests."""
        return tuple(self._global_microbatches_for_epoch(self.epoch))

    def __iter__(self) -> Iterator[int]:
        # Stage-1 is locked to DP2 x local_batch1, so each rank selects one
        # member from every DP-global pair.
        return iter(pair[self.rank] for pair in self.global_microbatches())

    def __len__(self) -> int:
        return len(self.dataset) // self.num_replicas


def build_training_sampler(
    dataset,
    *,
    seed,
    rank=None,
    num_replicas=None,
    resolution_aware=False,
    spatial_shapes=None,
    local_batch_size=1,
    warmup_microbatches=0,
):
    """Build a shared-seed training sampler.

    The default remains the legacy ``DistributedSampler`` path.  Stage-1 must
    opt into the exact, no-drop resolution-aware schedule explicitly.
    """
    if (rank is None) != (num_replicas is None):
        raise ValueError("rank and num_replicas must be provided together.")

    if resolution_aware:
        if rank is None:
            raise ValueError(
                "resolution_aware sampling requires explicit DP rank and "
                "num_replicas."
            )
        return ResolutionAwareDistributedSampler(
            dataset,
            seed=seed,
            rank=rank,
            num_replicas=num_replicas,
            spatial_shapes=spatial_shapes,
            local_batch_size=local_batch_size,
            warmup_microbatches=warmup_microbatches,
        )

    sampler_kwargs = {}
    if rank is not None:
        sampler_kwargs.update(rank=rank, num_replicas=num_replicas)

    return torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=True,
        drop_last=True,
        seed=int(seed),
        **sampler_kwargs,
    )
