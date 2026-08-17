from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

import scripts.plot_stage2_training as plot_module
from scripts.plot_stage2_training import plot_stage2_training
from trainer.stage2_distillation import Trainer
from utils.stage2_metrics import (
    STAGE2_TIMING_FIELDS,
    Stage2MetricsLogger,
    load_stage2_metrics,
    stage2_records_for_latest_lineage,
)
from utils.stage2_train_state import Stage2TrainingSchedule, Stage2TrainingState


def _canonical_sha256(value):
    def semantic_json_value(item):
        if isinstance(item, dict):
            return {key: semantic_json_value(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [semantic_json_value(child) for child in item]
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item

    payload = json.dumps(
        semantic_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_metadata(*, cycles=2, dry_run=False, phase_a_updates=1):
    phase_b_updates = cycles - phase_a_updates
    if phase_a_updates < 1 or phase_b_updates < 0:
        raise ValueError("test metadata requires 1 <= phase_a_updates <= cycles")
    resolved_config = {
        "config": {
            "algorithm": {"score_timestep_min": 20.0},
            "logging": {"metrics_schema": "longlive_stage2_metrics/v1"},
            "training": {
                "phase_a_generator_updates": phase_a_updates,
                "phase_b_generator_updates": phase_b_updates,
            },
        },
        "derived": {
            "metrics_schema": "longlive_stage2_metrics/v1",
            "expected_world_size": 8,
            "fsdp_backend": "fsdp2",
            "sharding_strategy": "FULL_SHARD",
            "microbatch_size_per_device": 2,
            "gradient_accumulation_steps": 4,
            "global_batch_size": 64,
            "fake_updates_per_generator_update": 5,
            "phase_a_generator_updates": phase_a_updates,
            "phase_a_fake_updates": phase_a_updates * 5,
            "phase_b_generator_updates": phase_b_updates,
            "phase_b_fake_updates": phase_b_updates * 5,
            "phase_b_epochs": int(phase_b_updates > 0),
            "total_fake_updates": cycles * 5,
            "total_generator_updates": cycles,
            "total_cycles": cycles,
            "patch_tokens_per_frame": 390,
            "score_input_frames": 25,
            "future_latent_frames": 24,
            "global_sink_frames": 1,
            "generated_episode_frames": 24,
        },
    }
    result = {
        "config_contract_sha256": "a" * 64,
        "config_launch_sha256": _canonical_sha256(resolved_config),
        "resolved_config": resolved_config,
        "world_size": 8,
        "topology": {
            "fsdp_backend": "fsdp2",
            "sharding_strategy": "FULL_SHARD",
            "microbatch_size_per_device": 2,
            "gradient_accumulation_steps": 4,
            "global_batch_size": 64,
        },
        "role_hashes": {
            "generator": "b" * 64,
            "real_score": "c" * 64,
            "fake_score": "d" * 64,
        },
        "terminal_counts": {
            "fake_score_updates": cycles * 5,
            "generator_updates": cycles,
            "cycles": cycles,
        },
        "phase_boundaries": [
            {"label": "Phase A end", "generator_update": phase_a_updates},
            *(
                [{"label": "Phase B end", "generator_update": cycles}]
                if phase_b_updates
                else []
            ),
        ],
        "workload": {
            "patch_tokens_per_frame": 390,
            "score_input_frames": 25,
            "loss_future_frames": 24,
            "sink_in_score_compute": True,
            "sink_in_loss": False,
        },
        "dry_run": dry_run,
        "smoke_mode": "C0" if dry_run else None,
    }
    return result


def _timing_fields(*, role):
    values = {
        "data_seconds_max": 0.10,
        "h2d_seconds_max": 0.05,
        "rollout_seconds_max": 0.55,
        "fake_score_seconds_max": 0.30,
        "real_cond_seconds_max": 0.20 if role == "generator" else 0.0,
        "real_uncond_seconds_max": 0.20 if role == "generator" else 0.0,
        "loss_build_seconds_max": 0.05 if role == "generator" else 0.50,
        "backward_seconds_max": 0.25,
        "clip_optimizer_seconds_max": 0.15,
        "orchestration_seconds_max": 0.05,
        "ema_seconds_max": 0.05 if role == "generator" else 0.0,
    }
    values["timing_closure_error_seconds"] = 2.0 - sum(values.values())
    return values


def _train_fields(
    *,
    role,
    fake,
    generator,
    substep,
    loss,
    branch=None,
    dry_run=False,
):
    logical = fake + generator - 1
    fields = {
        "logical_substep_id": logical,
        "role": role,
        "completed_fake_updates": fake,
        "completed_generator_updates": generator,
        "completed_cycles": generator,
        "cycle_substep": substep,
        "phase": "phase_a" if generator <= 1 else "phase_b",
        "branch": "flow_dsm" if role == "fake_score" else branch,
        "loss": loss,
        "loss_numerator": loss * 100.0,
        "loss_count": 100,
        "learning_rate": 4e-7 if role == "fake_score" else 2e-6,
        "preclip_grad_norm": 1.0,
        "step_seconds_max": 2.0,
        "step_seconds_mean": 1.8,
        "straggler_ratio": 2.0 / 1.8,
        **_timing_fields(role=role),
        "samples_per_second": 32.0,
        "generated_latents_per_second": 768.0,
        "generator_forward_calls": 16,
        "generator_logical_query_tokens": 160_000,
        "generator_tokens_per_second": 80_000.0,
        "fake_score_forward_calls": 4,
        "fake_score_logical_query_tokens": 40_000,
        "fake_score_tokens_per_second": 20_000.0,
        "gpu_memory_allocated_gib_max": 50.0,
        "gpu_memory_reserved_gib_max": 60.0,
        "gpu_memory_free_gib_min": 18.0,
        "score_timestep_min": 20.0,
        "score_timestep_mean": 400.0,
        "score_timestep_max": 980.0,
        "score_timestep_histogram": {"[20,500)": 32, "[500,980]": 32},
        "score_timestep_edge_mass": 0.0,
        "exit_step_mean": 1.5,
        "exit_histogram": {"0": 1, "1": 1, "2": 1, "3": 1},
        "dfd_probability": 0.0 if branch != "dfd" else 0.25,
        "branch_is_dfd": int(branch == "dfd"),
        "dry_run": dry_run,
    }
    if role == "fake_score":
        fields.update(
            {
                "target_flow_l2": 0.9,
                "prediction_l2": 0.8,
                # Zeros in inapplicable G-only panels are intentionally legal.
                "real_cond_seconds_max": 0.0,
                "real_uncond_seconds_max": 0.0,
                "ema_seconds_max": 0.0,
            }
        )
    else:
        fields.update(
            {
                "denominator_min": 0.5,
                "denominator_mean": 1.0,
                "denominator_max": 1.5,
                "raw_score_difference_l2": 0.7,
                "fake_x0_l2": 0.8,
                "real_x0_l2": 0.9,
                "ema_action": "decayed",
                "real_score_forward_calls": 8,
                "real_score_logical_query_tokens": 80_000,
                "real_score_tokens_per_second": 40_000.0,
            }
        )
    # A role-specific overwrite above can change the timing sum.
    fields["timing_closure_error_seconds"] = 2.0 - sum(
        fields[key] for key in STAGE2_TIMING_FIELDS
    )
    return fields


def _nonfinite_fields(*, cycle_index=0, attempt=1, dry_run=False):
    fake = cycle_index * 5
    generator = cycle_index
    return {
        "logical_substep_id": fake + generator,
        "completed_fake_updates": fake,
        "completed_generator_updates": generator,
        "completed_cycles": generator,
        "cycle_substep": "F1",
        "role": "fake_score",
        "attempt_number_for_substep": attempt,
        "reason": "injected nonfinite loss",
        "elapsed_seconds": 0.5,
        "batch_identity": "c" * 64,
        "nonfinite": True,
        "skipped": True,
        "gpu_memory_allocated_gib_max": 50.0,
        "gpu_memory_reserved_gib_max": 60.0,
        "gpu_memory_free_gib_min": 18.0,
        "dry_run": dry_run,
    }


def _append_cycle(logger, *, cycle_index, branch, dry_run=False, loss_scale=1.0):
    if cycle_index == 0:
        logger.append("nonfinite_attempt", _nonfinite_fields(dry_run=dry_run))
    for substep_index in range(1, 6):
        fake = cycle_index * 5 + substep_index
        logger.append(
            "train_step",
            _train_fields(
                role="fake_score",
                fake=fake,
                generator=cycle_index,
                substep=f"F{substep_index}",
                loss=loss_scale / fake,
                dry_run=dry_run,
            ),
        )
    completed_generator = cycle_index + 1
    completed_fake = completed_generator * 5
    logger.append(
        "train_step",
        _train_fields(
            role="generator",
            fake=completed_fake,
            generator=completed_generator,
            substep="G",
            loss=loss_scale * 0.25 / completed_generator,
            branch=branch,
            dry_run=dry_run,
        ),
    )
    logger.append(
        "cycle_summary",
        {
            "logical_substep_id": completed_fake + completed_generator - 1,
            "completed_fake_updates": completed_fake,
            "completed_generator_updates": completed_generator,
            "completed_cycles": completed_generator,
            "cycle_substep": "G",
            "cycle_seconds": 12.0,
            "fake_score_seconds": 10.0,
            "generator_seconds": 2.0,
            "fake_score_time_fraction": 10.0 / 12.0,
            "generator_time_fraction": 2.0 / 12.0,
            "cycle_samples_per_second": 32.0,
            "fake_score_samples_per_second": 32.0,
            "generator_samples_per_second": 32.0,
            "successful_substeps": 6,
            "nonfinite_attempts": int(cycle_index == 0),
            "dry_run": dry_run,
        },
    )


def _append_run_end(logger, *, cycles=2, status="complete", dry_run=False):
    logger.append(
        "run_end",
        {
            "logical_substep_id": cycles * 6 - 1,
            "completed_fake_updates": cycles * 5,
            "completed_generator_updates": cycles,
            "completed_cycles": cycles,
            "cycle_substep": "G",
            "status": status,
            "dry_run": dry_run,
        },
    )


def _produce_run(
    path,
    *,
    cycles=2,
    run_id="run",
    parent_run_id=None,
    dry_run=False,
    append_run_end=True,
):
    with Stage2MetricsLogger(
        path,
        experiment_id="stage2-test",
        run_id=run_id,
        parent_run_id=parent_run_id,
        run_metadata=_run_metadata(cycles=cycles, dry_run=dry_run),
    ) as logger:
        for cycle_index in range(cycles):
            _append_cycle(
                logger,
                cycle_index=cycle_index,
                branch="dmd" if cycle_index == 0 else "dfd",
                dry_run=dry_run,
            )
        if append_run_end:
            _append_run_end(
                logger,
                cycles=cycles,
                status="smoke_complete" if dry_run else "complete",
                dry_run=dry_run,
            )


def test_trainer_metric_producer_drives_real_logger_and_plotter_contract(
    tmp_path, monkeypatch
):
    path = tmp_path / "trainer-produced.jsonl"
    schedule = Stage2TrainingSchedule(
        generator_updates_per_epoch=2,
        fake_updates_per_generator_update=5,
        phase_a_generator_updates=2,
        phase_b_generator_updates=0,
        total_generator_updates=2,
        phase_b_mode="disabled",
        phase_b_dfd_probability_max=0.0,
        ema_initialize_at_completed_generator_update=1,
        ema_first_decay_completed_generator_update=2,
        nonfinite_max_attempts_per_update=2,
    )
    resolved = SimpleNamespace(
        total_generator_updates=2,
        nonfinite_max_attempts_per_update=2,
        gradient_accumulation_steps=1,
        num_denoising_steps=4,
        exit_sampling="stratified_uniform",
        generator_optimizer=SimpleNamespace(learning_rate=2.0e-6),
        fake_score_optimizer=SimpleNamespace(learning_rate=4.0e-7),
        score_timestep_min=20.0,
        score_timestep_max=980.0,
        global_batch_size=64,
        generated_episode_frames=24,
        checkpoint_interval_generator_updates=10,
    )
    state = Stage2TrainingState()
    trainer = Trainer.__new__(Trainer)
    trainer.resolved = resolved
    trainer.schedule = schedule
    trainer.state = state
    trainer.options = SimpleNamespace(
        dry_run=False,
        smoke_mode=None,
        no_save=True,
    )
    trainer.device = torch.device("cpu")
    trainer.world_size = 1
    trainer.is_main_process = True
    trainer.exit_rng = SimpleNamespace(draw=lambda *_args, **_kwargs: (0,))
    trainer.model = SimpleNamespace(generator=object())
    trainer.generator_ema = SimpleNamespace(
        update_after_step=lambda _model, completed_g: schedule.expected_ema_action(
            completed_g
        )
    )
    trainer._snapshot_attempt = lambda: {"snapshot": True}
    trainer._restore_attempt = lambda _snapshot: None
    trainer._materialize_batches = lambda _role: [
        {"sample_id": torch.tensor([7], dtype=torch.long)}
    ]
    trainer._to_device = lambda batch: batch
    trainer._draw_branch = lambda _probability: "dmd"
    trainer._memory_fields = lambda: {
        "gpu_memory_allocated_gib_max": 50.0,
        "gpu_memory_reserved_gib_max": 60.0,
        "gpu_memory_free_gib_min": 18.0,
    }

    def update_attempt(*, role, branch, **_kwargs):
        generator = role == "generator"
        diagnostics = (
            {
                "denominator_min": 0.5,
                "denominator_mean": 1.0,
                "denominator_max": 1.5,
                "raw_score_difference_l2": 0.7,
                "fake_x0_l2": 0.8,
                "real_x0_l2": 0.9,
            }
            if generator
            else {"target_flow_l2": 0.9, "prediction_l2": 0.8}
        )
        return {
            "success": True,
            "loss": 1.0,
            "loss_numerator": 100.0,
            "loss_count": 100,
            "preclip_grad_norm": 1.0,
            "diagnostics": diagnostics,
            "rollout_forward_calls": 16,
            "rollout_tokens": 160_000,
            "score_forward_calls": 3 if generator else 1,
            "score_tokens": 120_000 if generator else 40_000,
            "fake_score_forward_calls": 1,
            "fake_score_tokens": 40_000,
            "real_score_forward_calls": 2 if generator else 0,
            "real_score_tokens": 80_000 if generator else 0,
            "exit_values": [0],
            "timestep_values": [100.0],
            "compute_seconds": 1.4,
            "optimizer_seconds": 0.15,
            "phase_timings": {
                "rollout": 0.60,
                "fake_score": 0.30,
                "real_cond": 0.20 if generator else 0.0,
                "real_uncond": 0.20 if generator else 0.0,
                "loss_build": 0.05 if generator else 0.50,
                "backward": 0.25,
            },
        }

    trainer._run_update_attempt = update_attempt

    def timing_summary(_elapsed, categories):
        role = "generator" if categories.get("real_cond", 0.0) else "fake_score"
        return {
            "step_seconds_max": 2.0,
            "step_seconds_mean": 2.0,
            "straggler_ratio": 1.0,
            **_timing_fields(role=role),
        }

    trainer._timing_summary = timing_summary
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda *_args, **_kwargs: None
    )

    with Stage2MetricsLogger(
        path,
        experiment_id="trainer-producer-contract",
        run_metadata=_run_metadata(cycles=2, phase_a_updates=2),
    ) as logger:
        trainer.logger = logger
        trainer._train_loop()
        trainer._append_metric(
            "run_end",
            {
                **trainer._metric_clock("G", state.successful_attempts - 1),
                "status": "complete",
                "dry_run": False,
                "smoke_mode": None,
            },
        )

    records = load_stage2_metrics(path)
    assert [record["record_type"] for record in records].count("train_step") == 12
    assert [record["record_type"] for record in records].count("cycle_summary") == 2
    outputs = plot_stage2_training(
        path, tmp_path / "trainer-plots", require_complete=True
    )
    assert len(outputs) == 19


def _checkpoint_metrics_payload(directory, snapshot):
    source = directory / "metrics_lineage.jsonl"
    directory.mkdir(parents=True)
    source.write_bytes(snapshot)
    return SimpleNamespace(
        directory=directory,
        manifest={
            "files": [
                {
                    "name": "metrics_lineage.jsonl",
                    "size": len(snapshot),
                    "sha256": hashlib.sha256(snapshot).hexdigest(),
                }
            ]
        },
    )


@pytest.mark.parametrize("branch", ["dmd", "dfd"])
def test_external_a24_branch_imports_authenticated_metrics_and_plots_full_lineage(
    tmp_path, branch
):
    parent_path = tmp_path / "parent" / "metrics.jsonl"
    metadata = _run_metadata(cycles=2)
    with Stage2MetricsLogger(
        parent_path,
        experiment_id="stage2-test",
        run_id="a24-parent",
        run_metadata=metadata,
    ) as parent:
        _append_cycle(parent, cycle_index=0, branch="dmd")
        checkpoint_next_attempt = parent.next_attempt_index + 1
        snapshot = parent_path.read_bytes()

    checkpoint = _checkpoint_metrics_payload(
        tmp_path / "checkpoint_stage2_g000001",
        snapshot,
    )
    child_path = tmp_path / branch / "metrics.jsonl"
    trainer = Trainer.__new__(Trainer)
    trainer.resolved = SimpleNamespace(jsonl_path=str(child_path))
    trainer.options = SimpleNamespace(output_dir=tmp_path / branch)
    trainer._prepare_metrics_lineage(checkpoint)
    assert child_path.read_bytes() == snapshot

    with Stage2MetricsLogger(
        child_path,
        experiment_id="stage2-test",
        run_id=f"{branch}-child",
        parent_run_id="a24-parent",
        resume_from_logical_substep=6,
        checkpoint_next_attempt_index=checkpoint_next_attempt,
        run_metadata=metadata,
    ) as child:
        _append_cycle(child, cycle_index=1, branch=branch)
        _append_run_end(child, cycles=2)

    records = load_stage2_metrics(child_path)
    selected = stage2_records_for_latest_lineage(records, run_id=f"{branch}-child")
    assert [record["logical_substep_id"] for record in selected] == list(range(12))
    outputs = plot_stage2_training(
        child_path,
        tmp_path / branch / "plots",
        require_complete=True,
    )
    assert len(outputs) == 19


def test_terminal_checkpoint_restart_can_append_only_complete_run_end(tmp_path):
    path = tmp_path / "terminal-resume.jsonl"
    metadata = _run_metadata(cycles=1)
    with Stage2MetricsLogger(
        path,
        experiment_id="stage2-test",
        run_id="terminal-parent",
        run_metadata=metadata,
    ) as parent:
        _append_cycle(parent, cycle_index=0, branch="dmd")
        checkpoint_next_attempt = parent.next_attempt_index + 1

    with Stage2MetricsLogger(
        path,
        experiment_id="stage2-test",
        run_id="terminal-child",
        parent_run_id="terminal-parent",
        resume_from_logical_substep=6,
        checkpoint_next_attempt_index=checkpoint_next_attempt,
        run_metadata=metadata,
    ) as child:
        _append_run_end(child, cycles=1)

    records = load_stage2_metrics(path)
    assert records[-1]["record_type"] == "run_end"
    assert records[-1]["run_id"] == "terminal-child"
    assert records[-1]["status"] == "complete"
    outputs = plot_stage2_training(
        path,
        tmp_path / "terminal-plots",
        require_complete=True,
    )
    assert len(outputs) == 19


@pytest.mark.parametrize(
    ("cycles", "resume_boundary", "record_type", "fields"),
    [
        pytest.param(
            2,
            6,
            "run_end",
            {
                "logical_substep_id": 5,
                "completed_fake_updates": 5,
                "completed_generator_updates": 1,
                "completed_cycles": 1,
                "cycle_substep": "G",
                "status": "complete",
                "dry_run": False,
            },
            id="nonterminal-complete",
        ),
        pytest.param(
            1,
            6,
            "run_end",
            {
                "logical_substep_id": 5,
                "completed_fake_updates": 5,
                "completed_generator_updates": 1,
                "completed_cycles": 1,
                "cycle_substep": "G",
                "status": "failed",
                "dry_run": False,
            },
            id="terminal-failed-status",
        ),
        pytest.param(
            1,
            6,
            "cycle_summary",
            {
                "logical_substep_id": 5,
                "completed_fake_updates": 5,
                "completed_generator_updates": 1,
                "completed_cycles": 1,
                "cycle_substep": "G",
                "cycle_seconds": 12.0,
                "fake_score_seconds": 10.0,
                "generator_seconds": 2.0,
                "fake_score_time_fraction": 10.0 / 12.0,
                "generator_time_fraction": 2.0 / 12.0,
                "cycle_samples_per_second": 32.0,
                "fake_score_samples_per_second": 32.0,
                "generator_samples_per_second": 32.0,
                "successful_substeps": 6,
                "nonfinite_attempts": 0,
                "dry_run": False,
            },
            id="terminal-stale-cycle-summary",
        ),
    ],
)
def test_terminal_finalize_exception_does_not_allow_other_stale_records(
    tmp_path, cycles, resume_boundary, record_type, fields
):
    path = tmp_path / f"{record_type}.jsonl"
    metadata = _run_metadata(cycles=cycles)
    with Stage2MetricsLogger(
        path,
        experiment_id="stage2-test",
        run_id="root",
        run_metadata=metadata,
    ):
        pass
    with Stage2MetricsLogger(
        path,
        experiment_id="stage2-test",
        run_id="child",
        parent_run_id="root",
        resume_from_logical_substep=resume_boundary,
        run_metadata=metadata,
    ) as child:
        with pytest.raises(ValueError, match="before this run's resume boundary"):
            child.append(record_type, fields)


def test_real_stage2_logger_producer_drives_all_semantic_plots_and_complete_html(
    tmp_path,
):
    path = tmp_path / "metrics.jsonl"
    _produce_run(path)
    outputs = plot_stage2_training(
        path,
        tmp_path / "plots",
        rolling_window=2,
        require_complete=True,
    )

    assert len(outputs) == 19
    assert all(item.is_file() and item.stat().st_size > 0 for item in outputs)
    html_text = (tmp_path / "plots" / "index.html").read_text(encoding="utf-8")
    assert "Status: <strong>complete</strong>" in html_text
    assert "&quot;run_end_status&quot;: &quot;complete&quot;" in html_text
    assert "metrics_jsonl_sha256" in html_text
    assert html_text.count("<tr><td><a") == 18
    assert html_text.count("<figure>") == 9

    semantic_labels = {
        "generator_loss.svg": (
            "DMD surrogate",
            "DFD surrogate",
            "Holistic denominator",
            "Raw score-difference L2",
        ),
        "fake_score_loss.svg": ("Fake-score raw-flow MSE", "Target flow L2"),
        "optimization.svg": (
            "Generator optimization (independent update axis)",
            "Fake-score optimization (independent update axis)",
        ),
        "generator_throughput.svg": (
            "Generator update seconds",
            "Samples/s",
            "Generated latents/s",
            "Generator logical query tokens/s",
            "Fake-score logical query tokens/s",
            "Real-score logical query tokens/s",
        ),
        "fake_score_throughput.svg": (
            "Fake-score update seconds",
            "Generator logical query tokens/s",
            "Fake-score logical query tokens/s",
        ),
        "cycle_throughput.svg": (
            "5F -&gt; 1G cycle seconds",
            "F samples/s",
            "G samples/s",
            "F role time fraction",
            "G role time fraction",
        ),
        "time_breakdown.svg": (
            "Data wait raw",
            "Data wait",
            "H2D",
            "Rollout",
            "Fake score",
            "Real cond",
            "Real uncond",
            "Loss build",
            "Backward",
            "Clip + optimizer",
            "EMA",
            "Wall-time closure error",
        ),
        "memory_straggler.svg": (
            "Max allocated GiB",
            "Max reserved GiB",
            "Min free GiB",
            "Cross-rank max/mean step-time ratio",
        ),
        "timestep_exit_phase.svg": (
            "Score timestep histogram",
            "Random-exit coverage histogram",
            "Cumulative DMD",
            "Cumulative DFD",
            "Nonfinite attempt",
            "Phase A end",
            "Phase B end",
        ),
    }
    for name, labels in semantic_labels.items():
        svg = (tmp_path / "plots" / name).read_text(encoding="utf-8")
        for label in labels:
            assert label in svg, (name, label)


def test_html_hash_uses_the_exact_jsonl_byte_snapshot_that_was_plotted(
    tmp_path, monkeypatch
):
    path = tmp_path / "metrics.jsonl"
    _produce_run(path, cycles=1)
    plotted_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    original_write_html = plot_module._write_html_index

    def mutate_source_before_html(**kwargs):
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        return original_write_html(**kwargs)

    monkeypatch.setattr(plot_module, "_write_html_index", mutate_source_before_html)
    outputs = plot_module.plot_stage2_training(path, tmp_path / "plots")
    html_text = outputs[-1].read_text(encoding="utf-8")
    mutated_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    assert plotted_sha256 != mutated_sha256
    assert plotted_sha256 in html_text
    assert mutated_sha256 not in html_text


def test_terminal_counts_without_latest_complete_run_end_remain_partial(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _produce_run(path, append_run_end=False)

    outputs = plot_stage2_training(path, tmp_path / "partial", rolling_window=2)
    assert "Status: <strong>partial</strong>" in outputs[-1].read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=r"run_end\.status=None"):
        plot_stage2_training(path, tmp_path / "fail", require_complete=True)


def test_run_end_cannot_claim_complete_before_resolved_terminal_counts(tmp_path):
    path = tmp_path / "false-complete.jsonl"
    with Stage2MetricsLogger(
        path,
        experiment_id="false-complete",
        run_metadata=_run_metadata(cycles=2),
    ) as logger:
        _append_cycle(logger, cycle_index=0, branch="dmd")
        _append_run_end(logger, cycles=1)

    with pytest.raises(ValueError, match="run_end.status=complete"):
        plot_stage2_training(path, tmp_path / "must-fail")


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        pytest.param(
            lambda metadata: metadata["terminal_counts"].update(
                fake_score_updates=5, generator_updates=1, cycles=1
            ),
            "terminal_counts.*resolved_config",
            id="terminal-counts",
        ),
        pytest.param(
            lambda metadata: metadata["phase_boundaries"][0].update(generator_update=0),
            "phase_boundaries.*resolved_config",
            id="phase-boundaries",
        ),
        pytest.param(
            lambda metadata: metadata["workload"].update(score_input_frames=24),
            "workload.*resolved_config",
            id="workload",
        ),
        pytest.param(
            lambda metadata: metadata["resolved_config"]["derived"].update(
                metrics_schema="wrong/v1"
            ),
            "metrics_schema",
            id="schema",
        ),
        pytest.param(
            lambda metadata: metadata.update(config_launch_sha256="e" * 64),
            "config_launch_sha256",
            id="launch-hash",
        ),
    ],
)
def test_stage2_writer_rejects_run_metadata_that_disagrees_with_resolved_config(
    tmp_path, mutation, error_match
):
    metadata = _run_metadata(cycles=2)
    mutation(metadata)

    with pytest.raises(ValueError, match=error_match):
        Stage2MetricsLogger(
            tmp_path / "metrics.jsonl",
            experiment_id="metadata-contract",
            run_metadata=metadata,
        )


def test_plotter_defensively_rejects_resolved_terminal_drift_in_existing_jsonl(
    tmp_path,
):
    path = tmp_path / "metrics.jsonl"
    _produce_run(path, cycles=1)
    records = load_stage2_metrics(path)
    start = next(record for record in records if record["record_type"] == "run_start")
    start["resolved_config"]["derived"].update(
        total_fake_updates=10,
        total_generator_updates=2,
        total_cycles=2,
        phase_b_generator_updates=1,
        phase_b_fake_updates=5,
        phase_b_epochs=1,
    )
    start["config_launch_sha256"] = _canonical_sha256(start["resolved_config"])
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="terminal_counts.*resolved_config"):
        plot_stage2_training(path, tmp_path / "must-fail")


def test_dry_run_is_excluded_by_default_and_never_marked_complete(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _produce_run(path, cycles=1, dry_run=True)

    with pytest.raises(ValueError, match="No non-dry-run"):
        plot_stage2_training(path, tmp_path / "excluded")
    outputs = plot_stage2_training(
        path, tmp_path / "included", include_dry_run=True, rolling_window=2
    )
    html = outputs[-1].read_text(encoding="utf-8")
    assert "Status: <strong>partial</strong>" in html
    assert "smoke_complete" in html


def test_stage2_lineage_child_overrides_parent_stale_cycle_without_deleting_jsonl(
    tmp_path,
):
    path = tmp_path / "metrics.jsonl"
    _produce_run(path, run_id="parent")
    original_line_count = len(path.read_text(encoding="utf-8").splitlines())

    with Stage2MetricsLogger(
        path,
        experiment_id="stage2-test",
        run_id="child",
        parent_run_id="parent",
        resume_from_logical_substep=6,
        run_metadata=_run_metadata(cycles=2),
    ) as child:
        _append_cycle(
            child,
            cycle_index=1,
            branch="dfd",
            loss_scale=0.5,
        )
        _append_run_end(child)

    records = load_stage2_metrics(path)
    selected = stage2_records_for_latest_lineage(records)
    assert len(selected) == 12
    assert selected[6]["loss"] == pytest.approx(0.5 / 6)
    cycles = stage2_records_for_latest_lineage(records, record_type="cycle_summary")
    assert len(cycles) == 2
    assert len(path.read_text(encoding="utf-8").splitlines()) > original_line_count
    outputs = plot_stage2_training(path, tmp_path / "lineage", require_complete=True)
    assert "<strong>complete</strong>" in outputs[-1].read_text(encoding="utf-8")


def test_stage2_writer_rejects_noncycle_resume_boundary_and_child_prefix_write(
    tmp_path,
):
    path = tmp_path / "metrics.jsonl"
    _produce_run(path, run_id="parent")

    with pytest.raises(ValueError, match="complete 5F.*1G cycle boundary"):
        Stage2MetricsLogger(
            path,
            experiment_id="stage2-test",
            run_id="bad-boundary",
            parent_run_id="parent",
            resume_from_logical_substep=5,
            run_metadata=_run_metadata(cycles=2),
        )

    with Stage2MetricsLogger(
        path,
        experiment_id="stage2-test",
        run_id="child",
        parent_run_id="parent",
        resume_from_logical_substep=6,
        run_metadata=_run_metadata(cycles=2),
    ) as child:
        with pytest.raises(ValueError, match="resume boundary"):
            child.append(
                "train_step",
                _train_fields(
                    role="fake_score",
                    fake=1,
                    generator=0,
                    substep="F1",
                    loss=9.0,
                ),
            )

    sequence_path = tmp_path / "sequence.jsonl"
    with Stage2MetricsLogger(
        sequence_path,
        experiment_id="stage2-sequence",
        run_metadata=_run_metadata(cycles=1),
    ) as logger:
        logger.append(
            "train_step",
            _train_fields(
                role="fake_score",
                fake=1,
                generator=0,
                substep="F1",
                loss=1.0,
            ),
        )
        with pytest.raises(ValueError, match="next logical_substep_id"):
            logger.append(
                "train_step",
                _train_fields(
                    role="fake_score",
                    fake=3,
                    generator=0,
                    substep="F3",
                    loss=1.0,
                ),
            )


def test_stage2_reader_rejects_child_record_before_declared_resume_boundary(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _produce_run(path, run_id="parent")
    with Stage2MetricsLogger(
        path,
        experiment_id="stage2-test",
        run_id="child",
        parent_run_id="parent",
        resume_from_logical_substep=6,
        run_metadata=_run_metadata(cycles=2),
    ):
        pass

    records = load_stage2_metrics(path)
    child_start = next(
        record
        for record in records
        if record.get("record_type") == "run_start" and record.get("run_id") == "child"
    )
    rogue = {
        "schema": child_start["schema"],
        "schema_version": child_start["schema_version"],
        "record_type": "train_step",
        "experiment_id": child_start["experiment_id"],
        "run_id": "child",
        "parent_run_id": "parent",
        "resume_from_step": 6,
        "attempt_index": child_start["next_attempt_index"],
        **_train_fields(
            role="fake_score",
            fake=1,
            generator=0,
            substep="F1",
            loss=9.0,
        ),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rogue, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="before its resume boundary"):
        stage2_records_for_latest_lineage(load_stage2_metrics(path))


def test_stage2_reader_rejects_a_gap_inside_the_five_f_then_g_sequence(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _produce_run(path, cycles=1)
    records = [
        record
        for record in load_stage2_metrics(path)
        if not (
            record.get("record_type") == "train_step"
            and record.get("logical_substep_id") == 1
        )
    ]
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous F1.*F5.*G"):
        plot_stage2_training(path, tmp_path / "must-fail")


def test_stage2_lineage_drops_parent_nonfinite_suffix_after_clean_child_replay(
    tmp_path,
):
    path = tmp_path / "metrics.jsonl"
    with Stage2MetricsLogger(
        path,
        experiment_id="stage2-test",
        run_id="parent",
        run_metadata=_run_metadata(cycles=2),
    ) as parent:
        _append_cycle(parent, cycle_index=0, branch="dmd")
        parent.append(
            "nonfinite_attempt",
            _nonfinite_fields(cycle_index=1, attempt=1),
        )

    with Stage2MetricsLogger(
        path,
        experiment_id="stage2-test",
        run_id="child",
        parent_run_id="parent",
        resume_from_logical_substep=6,
        run_metadata=_run_metadata(cycles=2),
    ) as child:
        _append_cycle(child, cycle_index=1, branch="dfd")

    selected = stage2_records_for_latest_lineage(
        load_stage2_metrics(path), record_type="nonfinite_attempt"
    )
    assert [
        (record["run_id"], record["logical_substep_id"]) for record in selected
    ] == [("parent", 0)]


@pytest.mark.parametrize(
    ("role", "missing"),
    [
        pytest.param("generator", "raw_score_difference_l2", id="generator-loss"),
        pytest.param("fake_score", "target_flow_l2", id="fake-loss"),
        pytest.param(
            "generator", "real_score_tokens_per_second", id="generator-throughput"
        ),
        pytest.param(
            "fake_score", "generated_latents_per_second", id="fake-throughput"
        ),
    ],
)
def test_real_producer_fails_fast_when_core_loss_or_role_throughput_is_missing(
    tmp_path, role, missing
):
    path = tmp_path / f"{role}-{missing}.jsonl"
    with Stage2MetricsLogger(
        path,
        experiment_id="missing-core",
        run_metadata=_run_metadata(cycles=1),
    ) as logger:
        fields = (
            _train_fields(
                role="generator",
                fake=5,
                generator=1,
                substep="G",
                loss=0.25,
                branch="dmd",
            )
            if role == "generator"
            else _train_fields(
                role="fake_score",
                fake=1,
                generator=0,
                substep="F1",
                loss=1.0,
            )
        )
        del fields[missing]
        with pytest.raises(ValueError, match=missing):
            logger.append("train_step", fields)


def test_plotter_defensively_rejects_missing_core_field_in_existing_jsonl(tmp_path):
    source = tmp_path / "source.jsonl"
    _produce_run(source, cycles=1)
    records = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
    ]
    target = next(
        record
        for record in records
        if record.get("record_type") == "train_step"
        and record.get("role") == "fake_score"
    )
    del target["fake_score_tokens_per_second"]
    corrupted = tmp_path / "missing.jsonl"
    corrupted.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fake_score_tokens_per_second"):
        plot_stage2_training(corrupted, tmp_path / "must-fail")


def test_stage2_logger_rejects_bad_clock_nonfinite_json_and_accepts_zero_panels(
    tmp_path,
):
    path = tmp_path / "metrics.jsonl"
    with Stage2MetricsLogger(
        path,
        experiment_id="validation",
        run_metadata=_run_metadata(cycles=1),
    ) as logger:
        valid = _train_fields(
            role="fake_score", fake=1, generator=0, substep="F1", loss=1.0
        )
        assert logger.append("train_step", valid) == 0
        bad = dict(valid)
        bad["completed_cycles"] = 1
        with pytest.raises(ValueError, match="completed_cycles"):
            logger.append("train_step", bad)
        bad = dict(valid)
        bad["loss"] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            logger.append("train_step", bad)

        for substep_index in range(2, 6):
            logger.append(
                "train_step",
                _train_fields(
                    role="fake_score",
                    fake=substep_index,
                    generator=0,
                    substep=f"F{substep_index}",
                    loss=1.0,
                ),
            )

        # Generator diagnostics expose the raw pre-clamp denominator.  Exact
        # zero is legal and must remain loggable after the optimizer commits.
        zero_denominator = _train_fields(
            role="generator",
            fake=5,
            generator=1,
            substep="G",
            loss=0.5,
            branch="dmd",
        )
        zero_denominator.update(
            denominator_min=0.0,
            denominator_mean=0.0,
            denominator_max=0.0,
        )
        assert logger.append("train_step", zero_denominator) == 5

    # A single truncated tail remains crash-tolerant.
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"partial"')
    assert json.loads(json.dumps(load_stage2_metrics(path)))


def test_stage2_writer_enforces_authoritative_timing_closure_threshold(tmp_path):
    accepted_path = tmp_path / "accepted.jsonl"
    accepted = _train_fields(
        role="fake_score", fake=1, generator=0, substep="F1", loss=1.0
    )
    accepted["loss_build_seconds_max"] -= 0.05
    accepted["timing_closure_error_seconds"] = 0.10
    with Stage2MetricsLogger(
        accepted_path,
        experiment_id="timing-accepted",
        run_metadata=_run_metadata(cycles=1),
    ) as logger:
        logger.append("train_step", accepted)

    rejected = dict(accepted)
    rejected["loss_build_seconds_max"] -= 0.001
    rejected["timing_closure_error_seconds"] = 0.101
    with Stage2MetricsLogger(
        tmp_path / "rejected.jsonl",
        experiment_id="timing-rejected",
        run_metadata=_run_metadata(cycles=1),
    ) as logger:
        with pytest.raises(ValueError, match=r"timing closure.*max\(0.1"):
            logger.append("train_step", rejected)


def test_stage2_timing_contract_classifies_runtime_orchestration():
    assert "orchestration_seconds_max" in STAGE2_TIMING_FIELDS

    fields = _train_fields(
        role="fake_score", fake=1, generator=0, substep="F1", loss=1.0
    )
    assert fields["orchestration_seconds_max"] > 0.0
    assert sum(fields[key] for key in STAGE2_TIMING_FIELDS) + fields[
        "timing_closure_error_seconds"
    ] == pytest.approx(fields["step_seconds_max"])


def test_plotter_defensively_rejects_excessive_timing_closure_in_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"
    _produce_run(path, cycles=1)
    records = load_stage2_metrics(path)
    target = next(
        record for record in records if record.get("record_type") == "train_step"
    )
    target["loss_build_seconds_max"] -= 0.15
    target["timing_closure_error_seconds"] += 0.15
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"timing closure.*max\(0.1"):
        plot_stage2_training(path, tmp_path / "must-fail")


def test_two_nonfinite_retries_at_one_logical_substep_remain_distinct(tmp_path):
    path = tmp_path / "nonfinite.jsonl"
    with Stage2MetricsLogger(
        path,
        experiment_id="nonfinite-audit",
        run_metadata=_run_metadata(cycles=1),
    ) as logger:
        logger.append("nonfinite_attempt", _nonfinite_fields(attempt=1))
        logger.append("nonfinite_attempt", _nonfinite_fields(attempt=2))

    selected = stage2_records_for_latest_lineage(
        load_stage2_metrics(path), record_type="nonfinite_attempt"
    )
    assert [record["attempt_number_for_substep"] for record in selected] == [1, 2]
