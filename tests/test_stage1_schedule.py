from types import SimpleNamespace

import pytest

from utils.stage1_schedule import resolve_stage1_schedule


def _config(ramp_fraction=0.4):
    return SimpleNamespace(
        training=SimpleNamespace(
            gradient_accumulation_steps=2,
            phases=[
                {
                    "name": "phase_a_teacher_forcing",
                    "epochs": 2,
                    "lr": {"schedule": "constant", "start": 1e-5, "end": 1e-5},
                    "error_recycling": {"mode": "collect_only"},
                },
                {
                    "name": "phase_b_error_recycling",
                    "epochs": 3,
                    "transition": {
                        "schedule": "smoothstep",
                        "ramp_fraction": ramp_fraction,
                    },
                    "lr": {"start": 1e-5, "end": 5e-6},
                    "error_recycling": {
                        "mode": "collect_and_inject",
                        "max_active_prob": 0.5,
                        "effective_context_prob": 0.30,
                        "effective_latent_prob": 0.10,
                        "effective_noise_prob": 0.0,
                    },
                },
            ],
        ),
        checkpointing=SimpleNamespace(every_epochs=0.5),
    )


def test_locked_stage1_counts_and_boundaries():
    schedule = resolve_stage1_schedule(_config(), dataloader_length=300)
    assert schedule.updates_per_epoch == 150
    assert schedule.phases[0].updates == 300
    assert schedule.phases[1].ramp_updates == 180
    assert schedule.phases[1].updates == 450
    assert schedule.total_updates == 750
    assert schedule.checkpoint_interval_updates == 75

    at_299 = schedule.values_at(299)
    assert at_299.phase_index == 0
    assert at_299.lr == pytest.approx(1e-5)
    assert at_299.active_probability == 0.0

    at_300 = schedule.values_at(300)
    assert at_300.phase_index == 1
    assert at_300.ramp_u == 0.0
    assert at_300.lr == pytest.approx(1e-5)
    assert at_300.effective_context_probability == 0.0

    at_479 = schedule.values_at(479)
    assert at_479.ramp_u == 1.0
    assert at_479.ramp_s == 1.0
    assert at_479.lr == pytest.approx(5e-6)
    assert at_479.active_probability == pytest.approx(0.5)
    assert at_479.context_probability_given_active == pytest.approx(0.6)
    assert at_479.latent_probability_given_active == pytest.approx(0.2)
    assert at_479.effective_context_probability == pytest.approx(0.3)
    assert at_479.effective_latent_probability == pytest.approx(0.1)

    at_480 = schedule.values_at(480)
    at_749 = schedule.values_at(749)
    assert at_480.ramp_s == at_749.ramp_s == 1.0
    assert at_480.lr == at_749.lr == pytest.approx(5e-6)
    assert schedule.is_checkpoint_step(75)
    assert schedule.is_checkpoint_step(300)
    assert schedule.is_checkpoint_step(750)


def test_ramp_of_one_is_explicitly_supported():
    cfg = _config(ramp_fraction=1 / 450)
    schedule = resolve_stage1_schedule(cfg, dataloader_length=300)
    assert schedule.phases[1].ramp_updates == 1
    assert schedule.values_at(300).ramp_u == 1.0
    assert schedule.values_at(300).lr == pytest.approx(5e-6)


@pytest.mark.parametrize("length", [0, 301])
def test_invalid_dataloader_length_fails_fast(length):
    with pytest.raises(ValueError):
        resolve_stage1_schedule(_config(), dataloader_length=length)


def test_non_integer_checkpoint_interval_fails_fast():
    cfg = _config()
    cfg.checkpointing.every_epochs = 0.333
    with pytest.raises(ValueError, match="integer"):
        resolve_stage1_schedule(cfg, dataloader_length=300)


def test_schedule_hash_is_stable_and_bounds_are_strict():
    one = resolve_stage1_schedule(_config(), dataloader_length=300)
    two = resolve_stage1_schedule(_config(), dataloader_length=300)
    assert one.resolved_hash() == two.resolved_hash()
    with pytest.raises(IndexError):
        one.values_at(-1)
    with pytest.raises(IndexError):
        one.values_at(750)


def test_real_dataloader_topology_is_cross_checked():
    cfg = _config()
    cfg.infra = SimpleNamespace(data_parallel_size=2)
    cfg.data = SimpleNamespace(expected_num_samples=600, batch_size=1)
    assert resolve_stage1_schedule(cfg, 300).updates_per_epoch == 150
    with pytest.raises(ValueError, match=r"len\(dataloader\)"):
        resolve_stage1_schedule(cfg, 298)


def test_phase_schedule_and_legacy_max_iters_are_mutually_exclusive():
    cfg = _config()
    cfg.training.max_iters = 750
    with pytest.raises(ValueError, match="source of truth"):
        resolve_stage1_schedule(cfg, 300)
