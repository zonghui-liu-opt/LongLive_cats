from collections import Counter
import copy

import pytest
import torch

from utils.stage2_sampler import (
    STAGE2_ALLOWED_SPATIAL_SHAPES,
    STAGE2_BATCHES_PER_STREAM_EPOCH,
    STAGE2_GLOBAL_BATCH_SIZE,
    Stage2BalancedBatchSampler,
    build_stage2_role_samplers,
    partition_stage2_global_batch,
    rotating_action_batch_counts,
)
from utils.stage2_i2v_data import stage2_i2v_cache_collate

ACTIONS = ("head_tilt", "jump", "toy_play")
ACTION_IDS = tuple(ACTIONS[index % 3] for index in range(600))
SPATIAL_SHAPES = tuple(STAGE2_ALLOWED_SPATIAL_SHAPES[index % 2] for index in range(600))


def _sampler(role="generator", seed=17, microbatch=2, spatial_shapes=SPATIAL_SHAPES):
    return Stage2BalancedBatchSampler(
        ACTION_IDS,
        spatial_shapes,
        action_order=ACTIONS,
        base_seed=seed,
        role=role,
        microbatch_size_per_device=microbatch,
    )


def _counts(batch):
    return Counter(ACTION_IDS[index] for index in batch)


def _assert_states_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key])
        elif isinstance(left[key], dict):
            _assert_states_equal(left[key], right[key])
        else:
            assert left[key] == right[key]


def test_rotating_22_21_21_composition_is_exact_and_long_term_equal():
    sampler = _sampler()
    totals = Counter()
    for update in range(30):
        batch = sampler.next_global_batch()
        assert len(batch) == STAGE2_GLOBAL_BATCH_SIZE
        assert len(set(batch)) == STAGE2_GLOBAL_BATCH_SIZE
        expected_tuple = rotating_action_batch_counts(update)
        assert tuple(_counts(batch)[action] for action in ACTIONS) == expected_tuple
        totals.update(_counts(batch))
    assert totals == Counter({action: 640 for action in ACTIONS})


def test_action_queues_are_deterministically_shuffled_and_reproducible():
    left = _sampler(seed=123)
    right = _sampler(seed=123)
    batches_left = [left.next_global_batch() for _ in range(14)]
    batches_right = [right.next_global_batch() for _ in range(14)]
    assert batches_left == batches_right
    assert left.state_dict()["action_states"] == right.state_dict()["action_states"]

    different = _sampler(seed=124)
    assert batches_left[0] != different.next_global_batch()


def test_five_fake_batches_do_not_advance_or_perturb_generator_stream():
    streams = build_stage2_role_samplers(
        ACTION_IDS, SPATIAL_SHAPES, action_order=ACTIONS, base_seed=9
    )
    untouched_generator_state = copy.deepcopy(streams.generator.state_dict())
    for _ in range(5):
        streams.fake_score.next_global_batch()
    _assert_states_equal(streams.generator.state_dict(), untouched_generator_state)

    pristine = build_stage2_role_samplers(
        ACTION_IDS, SPATIAL_SHAPES, action_order=ACTIONS, base_seed=9
    )
    assert (
        streams.generator.next_global_batch() == pristine.generator.next_global_batch()
    )
    assert streams.fake_score.completed_batches == 5
    assert streams.generator.completed_batches == 1
    assert streams.fake_score.stream_seed != streams.generator.stream_seed


def test_sampler_state_dict_resume_repeats_exact_cross_epoch_future():
    source = _sampler(seed=33)
    for _ in range(11):
        source.next_global_batch()
    checkpoint = copy.deepcopy(source.state_dict())
    assert checkpoint["stream_epoch"] == 1
    assert checkpoint["batch_cursor"] == 1
    assert checkpoint["extra_slot_cursor"] == 2
    expected_future = [source.next_global_batch() for _ in range(12)]
    expected_final_state = source.state_dict()

    resumed = _sampler(seed=33)
    resumed.load_state_dict(checkpoint)
    actual_future = [resumed.next_global_batch() for _ in range(12)]
    assert actual_future == expected_future
    _assert_states_equal(resumed.state_dict(), expected_final_state)


def test_snapshot_restore_replays_nonfinite_attempt_batch_and_rng():
    sampler = _sampler(seed=44)
    for _ in range(3):
        sampler.next_global_batch()
    before_attempt = copy.deepcopy(sampler.state_dict())
    failed_attempt_batch = sampler.next_global_batch()
    failed_attempt_post_state = copy.deepcopy(sampler.state_dict())

    sampler.load_state_dict(before_attempt)
    retry_batch = sampler.next_global_batch()
    assert retry_batch == failed_attempt_batch
    _assert_states_equal(sampler.state_dict(), failed_attempt_post_state)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state.__setitem__("role", "fake_score"),
        lambda state: state.__setitem__("batch_cursor", 9),
        lambda state: state["action_states"][ACTIONS[0]]["groups"]["30x52"].__setitem__(
            "cursor", 101
        ),
        lambda state: state["action_states"][ACTIONS[0]]["groups"]["30x52"][
            "order"
        ].__setitem__(
            0,
            state["action_states"][ACTIONS[0]]["groups"]["30x52"]["order"][1],
        ),
        lambda state: state.__setitem__(
            "generator_state", state["generator_state"].to(torch.int64)
        ),
    ],
)
def test_sampler_resume_rejects_role_cursor_order_and_rng_corruption(mutate):
    sampler = _sampler()
    sampler.next_global_batch()
    before = copy.deepcopy(sampler.state_dict())
    bad = copy.deepcopy(before)
    mutate(bad)
    with pytest.raises(RuntimeError):
        sampler.load_state_dict(bad)
    _assert_states_equal(sampler.state_dict(), before)


def test_pair_state_resume_is_exact_and_validated_atomically():
    streams = build_stage2_role_samplers(
        ACTION_IDS, SPATIAL_SHAPES, action_order=ACTIONS, base_seed=55
    )
    for _ in range(7):
        streams.fake_score.next_global_batch()
    for _ in range(2):
        streams.generator.next_global_batch()
    checkpoint = copy.deepcopy(streams.state_dict())
    expected_f = streams.fake_score.next_global_batch()
    expected_g = streams.generator.next_global_batch()

    restored = build_stage2_role_samplers(
        ACTION_IDS, SPATIAL_SHAPES, action_order=ACTIONS, base_seed=55
    )
    restored.load_state_dict(checkpoint)
    assert restored.fake_score.next_global_batch() == expected_f
    assert restored.generator.next_global_batch() == expected_g

    before = copy.deepcopy(restored.state_dict())
    bad = copy.deepcopy(checkpoint)
    bad["generator"]["dataset_action_spatial_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="dataset_action_spatial_sha256"):
        restored.load_state_dict(bad)
    _assert_states_equal(restored.state_dict(), before)


@pytest.mark.parametrize(
    ("microbatch", "accumulation"),
    [(2, 4), (1, 8)],
)
def test_global_batch_partitions_exactly_across_eight_ranks(microbatch, accumulation):
    batch = _sampler(microbatch=microbatch).next_global_batch()
    rank_microbatches = [
        partition_stage2_global_batch(
            batch,
            rank=rank,
            world_size=8,
            microbatch_size_per_device=microbatch,
            gradient_accumulation_steps=accumulation,
            spatial_shapes=SPATIAL_SHAPES,
        )
        for rank in range(8)
    ]
    assert all(len(value) == accumulation for value in rank_microbatches)
    assert all(
        len(micro) == microbatch for value in rank_microbatches for micro in value
    )
    flattened = [
        index for value in rank_microbatches for micro in value for index in micro
    ]
    assert sorted(flattened) == sorted(batch)


def _collate_item(index):
    height, width = SPATIAL_SHAPES[index]
    return {
        "sample_id": index,
        "action_id": ACTION_IDS[index],
        "action_index": ACTIONS.index(ACTION_IDS[index]),
        "initial_latent": torch.empty(1, 48, height, width, dtype=torch.uint8),
        "real_future": torch.empty(24, 48, height, width, dtype=torch.uint8),
        "prompt_embeds": torch.empty(1, 1),
        "prompt_mask": torch.ones(1, dtype=torch.bool),
        "height": height * 16,
        "width": width * 16,
        "bucket": "landscape" if width > height else "portrait",
    }


def test_micro2_sampler_partition_and_dataset_collate_are_shape_homogeneous():
    batch = _sampler(microbatch=2).next_global_batch()
    for rank in range(8):
        local_microbatches = partition_stage2_global_batch(
            batch,
            rank=rank,
            microbatch_size_per_device=2,
            gradient_accumulation_steps=4,
            spatial_shapes=SPATIAL_SHAPES,
        )
        for local in local_microbatches:
            collated = stage2_i2v_cache_collate(
                [_collate_item(index) for index in local]
            )
            assert collated["initial_latent"].shape[0] == 2
            assert collated["real_future"].shape[0] == 2


def test_sampler_requires_shape_for_every_row_and_binds_shape_mapping_on_resume():
    with pytest.raises(ValueError, match="one spatial shape for every sample"):
        Stage2BalancedBatchSampler(
            ACTION_IDS,
            SPATIAL_SHAPES[:-1],
            action_order=ACTIONS,
            base_seed=1,
            role="generator",
        )

    source = _sampler(seed=77)
    source.next_global_batch()
    checkpoint = copy.deepcopy(source.state_dict())
    changed_shapes = list(SPATIAL_SHAPES)
    changed_shapes[0], changed_shapes[1] = changed_shapes[1], changed_shapes[0]
    changed = _sampler(seed=77, spatial_shapes=changed_shapes)
    with pytest.raises(RuntimeError, match="dataset_action_spatial_sha256"):
        changed.load_state_dict(checkpoint)


def test_infeasible_micro2_population_fails_fast_but_micro1_remains_supported():
    fixed_shapes = tuple(
        (30, 52) if action_id in ACTIONS[:2] else (52, 30) for action_id in ACTION_IDS
    )
    with pytest.raises(RuntimeError, match="microbatch_size_per_device=1"):
        _sampler(microbatch=2, spatial_shapes=fixed_shapes)

    sampler = _sampler(microbatch=1, spatial_shapes=fixed_shapes)
    batch = sampler.next_global_batch()
    assert tuple(_counts(batch)[action] for action in ACTIONS) == (22, 21, 21)
    for rank in range(8):
        assert (
            len(
                partition_stage2_global_batch(
                    batch,
                    rank=rank,
                    microbatch_size_per_device=1,
                    gradient_accumulation_steps=8,
                    spatial_shapes=fixed_shapes,
                )
            )
            == 8
        )


def test_partition_rejects_a_shape_mixed_micro2_layout():
    mixed = list(range(64))
    with pytest.raises(ValueError, match="mixes latent spatial shapes"):
        partition_stage2_global_batch(
            mixed,
            rank=0,
            microbatch_size_per_device=2,
            gradient_accumulation_steps=4,
            spatial_shapes=SPATIAL_SHAPES,
        )


def test_stream_epoch_is_ten_successful_batches_and_iterator_resumes_cursor():
    sampler = _sampler()
    for _ in range(3):
        sampler.next_global_batch()
    assert sampler.stream_epoch == 0
    assert sampler.batch_cursor == 3
    assert len(sampler) == 7
    assert len(list(iter(sampler))) == 7
    assert sampler.stream_epoch == 1
    assert sampler.batch_cursor == 0
    assert STAGE2_BATCHES_PER_STREAM_EPOCH == 10


def test_sampler_rejects_non_600_or_non_200_200_200_action_mapping():
    with pytest.raises(ValueError, match="exactly 600"):
        Stage2BalancedBatchSampler(
            ACTION_IDS[:-1],
            SPATIAL_SHAPES[:-1],
            action_order=ACTIONS,
            base_seed=1,
            role="generator",
        )
    unbalanced = list(ACTION_IDS)
    unbalanced[0] = ACTIONS[1]
    with pytest.raises(ValueError, match="action counts"):
        Stage2BalancedBatchSampler(
            unbalanced,
            SPATIAL_SHAPES,
            action_order=ACTIONS,
            base_seed=1,
            role="generator",
        )


def test_partition_rejects_wrong_effective_batch_or_duplicates():
    batch = _sampler().next_global_batch()
    with pytest.raises(ValueError, match="effective global batch 64"):
        partition_stage2_global_batch(
            batch,
            rank=0,
            microbatch_size_per_device=1,
            gradient_accumulation_steps=4,
            spatial_shapes=SPATIAL_SHAPES,
        )
    bad = list(batch)
    bad[-1] = bad[0]
    with pytest.raises(ValueError, match="duplicate"):
        partition_stage2_global_batch(bad, rank=0, spatial_shapes=SPATIAL_SHAPES)
