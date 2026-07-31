import unittest

from utils.sampler import (
    ResolutionAwareDistributedSampler,
    build_training_sampler,
)


LANDSCAPE = (480, 832)
PORTRAIT = (832, 480)


class _ShapeDataset:
    def __init__(self, spatial_shapes):
        self.spatial_shapes = list(spatial_shapes)

    def __len__(self):
        return len(self.spatial_shapes)


def _build(dataset, rank, *, seed=42, warmup=0):
    return build_training_sampler(
        dataset,
        seed=seed,
        rank=rank,
        num_replicas=2,
        resolution_aware=True,
        warmup_microbatches=warmup,
    )


class Stage1ResolutionAwareSamplerTest(unittest.TestCase):
    def test_epoch_is_exact_and_odd_cross_orientation_pair_is_after_warmup(self):
        # Both buckets have one odd tail, matching the hardest locked-data case.
        dataset = _ShapeDataset([LANDSCAPE] * 301 + [PORTRAIT] * 299)
        rank0 = _build(dataset, 0, warmup=150)
        rank1 = _build(dataset, 1, warmup=150)

        global_batches = rank0.global_microbatches()
        self.assertEqual(global_batches, rank1.global_microbatches())
        self.assertEqual(len(global_batches), 300)
        self.assertEqual(list(rank0), [pair[0] for pair in global_batches])
        self.assertEqual(list(rank1), [pair[1] for pair in global_batches])

        flattened = [index for pair in global_batches for index in pair]
        self.assertEqual(len(flattened), 600)
        self.assertEqual(len(set(flattened)), 600)
        self.assertEqual(set(flattened), set(range(600)))

        for first, second in global_batches[:150]:
            self.assertEqual(
                dataset.spatial_shapes[first], dataset.spatial_shapes[second]
            )
        cross_orientation_positions = [
            position
            for position, (first, second) in enumerate(global_batches)
            if dataset.spatial_shapes[first] != dataset.spatial_shapes[second]
        ]
        self.assertEqual(len(cross_orientation_positions), 1)
        self.assertGreaterEqual(cross_orientation_positions[0], 150)

    def test_epoch_shuffle_changes_and_is_reproducible(self):
        dataset = _ShapeDataset([LANDSCAPE] * 300 + [PORTRAIT] * 300)
        sampler = _build(dataset, 0, seed=123, warmup=150)
        epoch0 = sampler.global_microbatches()
        sampler.set_epoch(1)
        epoch1 = sampler.global_microbatches()

        replica = _build(dataset, 0, seed=123, warmup=150)
        replica.set_epoch(1)
        self.assertNotEqual(epoch0, epoch1)
        self.assertEqual(epoch1, replica.global_microbatches())

    def test_sp_ranks_with_same_dp_rank_share_sampler_stream(self):
        dataset = _ShapeDataset([LANDSCAPE] * 12 + [PORTRAIT] * 12)
        streams = []
        sp_size = 3
        for global_rank in range(6):
            dp_rank = global_rank // sp_size
            streams.append(list(_build(dataset, dp_rank, seed=9, warmup=4)))

        self.assertEqual(streams[0], streams[1])
        self.assertEqual(streams[1], streams[2])
        self.assertEqual(streams[3], streams[4])
        self.assertEqual(streams[4], streams[5])
        self.assertNotEqual(streams[0], streams[3])
        self.assertTrue(set(streams[0]).isdisjoint(streams[3]))
        self.assertEqual(set(streams[0]) | set(streams[3]), set(range(24)))

    def test_impossible_warmup_fails_fast(self):
        dataset = _ShapeDataset([LANDSCAPE, PORTRAIT])
        with self.assertRaisesRegex(ValueError, "same-resolution"):
            _build(dataset, 0, warmup=1)

    def test_sampler_refuses_drop_or_oversample(self):
        dataset = _ShapeDataset([LANDSCAPE, LANDSCAPE, PORTRAIT])
        with self.assertRaisesRegex(ValueError, "refusing to drop or oversample"):
            _build(dataset, 0)

    def test_sampler_requires_locked_dp2_batch1_layout(self):
        dataset = _ShapeDataset([LANDSCAPE] * 4)
        with self.assertRaisesRegex(ValueError, "exactly two DP replicas"):
            build_training_sampler(
                dataset,
                seed=1,
                rank=0,
                num_replicas=1,
                local_batch_size=2,
                resolution_aware=True,
            )

    def test_public_class_matches_builder(self):
        dataset = _ShapeDataset([LANDSCAPE] * 4)
        sampler = _build(dataset, 0)
        self.assertIsInstance(sampler, ResolutionAwareDistributedSampler)


if __name__ == "__main__":
    unittest.main()
