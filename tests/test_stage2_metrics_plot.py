from __future__ import annotations

import json

import pytest

from scripts.plot_stage2_training import plot_stage2_training
from utils.stage2_metrics import (
    Stage2MetricsLogger,
    load_stage2_metrics,
    stage2_records_for_latest_lineage,
)


def _run_metadata(*, cycles=2):
    return {
        "config_contract_sha256": "a" * 64,
        "config_launch_sha256": "b" * 64,
        "terminal_counts": {
            "fake_score_updates": cycles * 5,
            "generator_updates": cycles,
            "cycles": cycles,
        },
        "phase_boundaries": [
            {"label": "Phase A end", "generator_update": 1},
            {"label": "Phase B end", "generator_update": cycles},
        ],
    }


def _timing_fields(*, role):
    values = {
        "data_seconds_max": 0.10,
        "h2d_seconds_max": 0.05,
        "rollout_seconds_max": 0.60,
        "fake_score_seconds_max": 0.30,
        "real_cond_seconds_max": 0.20 if role == "generator" else 0.0,
        "real_uncond_seconds_max": 0.20 if role == "generator" else 0.0,
        "loss_build_seconds_max": 0.05,
        "backward_seconds_max": 0.25,
        "clip_optimizer_seconds_max": 0.15,
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
        fields[key]
        for key in (
            "data_seconds_max",
            "h2d_seconds_max",
            "rollout_seconds_max",
            "fake_score_seconds_max",
            "real_cond_seconds_max",
            "real_uncond_seconds_max",
            "loss_build_seconds_max",
            "backward_seconds_max",
            "clip_optimizer_seconds_max",
            "ema_seconds_max",
        )
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
        run_metadata=_run_metadata(cycles=cycles),
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
        assert logger.append("train_step", zero_denominator) == 1

    # A single truncated tail remains crash-tolerant.
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"partial"')
    assert json.loads(json.dumps(load_stage2_metrics(path)))


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
