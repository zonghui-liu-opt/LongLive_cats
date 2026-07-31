from utils.stage1_error_recycling import sample_error_recycling_gate
from utils.stage1_schedule import ScheduleValues


class _Rng:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


def _values(mode="collect_and_inject", active=0.5):
    return ScheduleValues(
        update_index=480,
        optimizer_step=481,
        phase_index=1,
        phase_name="b",
        lr=5e-6,
        ramp_u=1.0,
        ramp_s=1.0,
        error_recycling_mode=mode,
        active_probability=active,
        context_probability_given_active=0.6,
        latent_probability_given_active=0.2,
        noise_probability_given_active=0.0,
        effective_context_probability=0.3,
        effective_latent_probability=0.1,
        effective_noise_probability=0.0,
    )


def test_phase_a_collects_without_any_injection_randomness():
    gate = sample_error_recycling_gate(
        _values(mode="collect_only"), clean_buffer_update_prob=0.1, rng=_Rng([])
    )
    assert gate.to_dict() == {
        "active": False,
        "context": False,
        "latent": False,
        "noise": False,
        "update_buffer": True,
    }


def test_active_gate_samples_context_and_latent_conditionally():
    gate = sample_error_recycling_gate(
        _values(),
        clean_buffer_update_prob=0.1,
        # active, context, latent, noise
        rng=_Rng([0.49, 0.59, 0.19, 0.5]),
    )
    assert gate.active and gate.context and gate.latent
    assert not gate.noise
    assert gate.update_buffer


def test_inactive_clean_branch_has_separate_ten_percent_buffer_gate():
    updated = sample_error_recycling_gate(
        _values(), clean_buffer_update_prob=0.1, rng=_Rng([0.9, 0.09])
    )
    skipped = sample_error_recycling_gate(
        _values(), clean_buffer_update_prob=0.1, rng=_Rng([0.9, 0.11])
    )
    assert not updated.active and updated.update_buffer
    assert not skipped.active and not skipped.update_buffer
