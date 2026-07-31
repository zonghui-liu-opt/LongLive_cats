import unittest

import torch

from utils.error_buffer import ErrorBuffer


class ErrorBufferSpatialShapeTest(unittest.TestCase):
    def _buffer(self, *, capacity=4, replacement="random"):
        return ErrorBuffer(
            num_buckets=1,
            max_size_per_bucket=capacity,
            num_train_timesteps=1000,
            modulate_factor=0.0,
            replacement_strategy=replacement,
            num_blocks=1,
        )

    def test_all_sampling_paths_filter_exact_spatial_shape(self):
        buffer = self._buffer()
        landscape = torch.ones(2, 3, 3, 5, dtype=torch.bfloat16)
        portrait = torch.full((2, 3, 5, 3), 2.0, dtype=torch.bfloat16)
        buffer.add(landscape, 10, block_pos=0)
        buffer.add(portrait, 10, block_pos=0)

        exact = buffer.sample(
            10,
            "cpu",
            torch.bfloat16,
            block_pos=0,
            expected_spatial_shape=(3, 5),
        )
        any_t = buffer.sample_pos_any_t(
            0,
            "cpu",
            torch.bfloat16,
            expected_spatial_shape=(5, 3),
        )
        global_sample = buffer.sample_global(
            "cpu", torch.bfloat16, expected_spatial_shape=(3, 5)
        )

        self.assertTrue(torch.equal(exact, landscape))
        self.assertTrue(torch.equal(any_t, portrait))
        self.assertTrue(torch.equal(global_sample, landscape))
        self.assertIsNone(
            buffer.sample(
                10,
                "cpu",
                torch.bfloat16,
                block_pos=0,
                expected_spatial_shape=(7, 7),
            )
        )

    def test_mixed_shapes_share_one_base_bucket_capacity(self):
        buffer = self._buffer(capacity=3)
        for index in range(12):
            shape = (3, 5) if index % 2 == 0 else (5, 3)
            buffer.add(torch.full((2, 3, *shape), float(index)), 10, block_pos=0)

        stats = buffer.stats()
        self.assertEqual(stats["total_entries"], 3)
        self.assertEqual(sum(stats["entries_by_spatial_shape"].values()), 3)
        self.assertEqual(len(buffer.buckets[(0, 0)]), 3)

    def test_l2_replacement_is_safe_for_mixed_shapes(self):
        buffer = self._buffer(capacity=2, replacement="l2")
        buffer.add(torch.zeros(2, 3, 3, 5), 10, block_pos=0)
        buffer.add(torch.zeros(2, 3, 5, 3), 10, block_pos=0)

        # Comparable entries use L2; a third, entirely new shape takes the
        # explicit safe fallback. Neither path may stack incompatible tensors.
        buffer.add(torch.ones(2, 3, 3, 5), 10, block_pos=0)
        buffer.add(torch.ones(2, 3, 4, 4), 10, block_pos=0)
        self.assertEqual(len(buffer.buckets[(0, 0)]), 2)

    def test_strict_schema_roundtrip_and_mismatch_detection(self):
        source = self._buffer(capacity=3)
        entry = torch.randn(2, 3, 3, 5, dtype=torch.bfloat16)
        source.add(entry, 10, block_pos=0)
        state = source.state_dict()

        restored = self._buffer(capacity=3)
        restored.load_state_dict(state, strict_schema=True)
        sampled = restored.sample(
            10,
            "cpu",
            torch.bfloat16,
            block_pos=0,
            expected_spatial_shape=(3, 5),
        )
        self.assertTrue(torch.equal(sampled, entry))

        incompatible = self._buffer(capacity=4)
        with self.assertRaisesRegex(RuntimeError, "max_size_per_bucket"):
            incompatible.load_state_dict(state, strict_schema=True)

        legacy_state = dict(state)
        legacy_state.pop("state_schema")
        legacy_state.pop("state_version")
        legacy = self._buffer(capacity=3)
        legacy.load_state_dict(legacy_state, strict_schema=False)
        self.assertEqual(legacy.stats()["total_entries"], 1)
        with self.assertRaisesRegex(RuntimeError, "state_schema"):
            legacy.load_state_dict(legacy_state, strict_schema=True)

    def test_invalid_expected_spatial_shape_fails(self):
        buffer = self._buffer()
        with self.assertRaisesRegex(ValueError, "expected_spatial_shape"):
            buffer.sample_global(
                "cpu", torch.float32, expected_spatial_shape=(3,)
            )


if __name__ == "__main__":
    unittest.main()
