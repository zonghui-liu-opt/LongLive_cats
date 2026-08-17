import pytest

from scripts.plot_stage1_training import plot_training_metrics
from utils.jsonl_logger import (
    JsonlLogger,
    logical_workload,
    read_jsonl_tolerant,
    records_for_latest_lineage,
    throughput_fields,
)


def _fields(step, loss):
    return {
        "optimizer_step": step,
        "update_index": step - 1,
        "loss_total": loss,
        "block0_loss": loss + 0.1,
        "block1_loss": loss + 0.2,
        "block2_loss": loss + 0.3,
        "lr": 1e-5,
        "pre_clip_grad_norm": 1.0,
        "logical_dit_tokens_per_second": 100.0,
        "supervised_tokens_per_second": 50.0,
        "samples_per_second": 1.0,
        "step_seconds_max": 4.0,
        "gpu_memory_allocated_gib_max": 10.0,
        "gpu_memory_reserved_gib_max": 12.0,
        "straggler_ratio": 1.1,
        "scheduled_active_probability": 0.0,
        "actual_applied_rate": 0.0,
        "buffer_entries_mean": 2.0,
    }


def test_locked_logical_workload_does_not_multiply_sp():
    landscape = logical_workload(latent_height=30, latent_width=52, global_samples=4)
    portrait = logical_workload(latent_height=52, latent_width=30, global_samples=4)
    assert landscape == portrait
    assert landscape.patch_tokens_per_latent_frame == 390
    assert landscape.dit_tokens == 74_880
    assert landscape.supervised_tokens == 35_880
    assert landscape.logical_source_frames == 372
    assert throughput_fields(landscape, 2.0)["samples_per_second"] == 2.0


def test_resume_lineage_overrides_parent_stale_suffix_and_attempts_monotonic(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with JsonlLogger(
        path, experiment_id="exp", run_id="parent", fsync_every_steps=1
    ) as parent:
        assert parent.append_attempt("train_step", _fields(1, 1.0)) == 0
        assert parent.append_attempt("train_step", _fields(2, 2.0)) == 1
        # This suffix is stale after the checkpoint used for resume.
        assert parent.append_attempt("train_step", _fields(3, 30.0)) == 2
    with JsonlLogger(
        path,
        experiment_id="exp",
        run_id="child",
        parent_run_id="parent",
        resume_from_step=2,
        checkpoint_next_attempt_index=2,
        fsync_every_steps=1,
    ) as child:
        assert child.append_attempt("train_step", _fields(3, 3.0)) == 3
        assert (
            child.append_attempt(
                "nonfinite_attempt", {"optimizer_step": 4, "duration": 0.5}
            )
            == 4
        )
        assert child.append_attempt("train_step", _fields(4, 4.0)) == 5

    records = read_jsonl_tolerant(path)
    selected = records_for_latest_lineage(records)
    assert [record["optimizer_step"] for record in selected] == [1, 2, 3, 4]
    assert selected[2]["loss_total"] == 3.0


def test_truncated_tail_is_ignored_but_middle_corruption_is_not(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"record_type":"run_start","run_id":"a"}\n{"broken"')
    assert len(read_jsonl_tolerant(path)) == 1
    path.write_text('{"record_type":"run_start","run_id":"a"}\nnot-json\n{}\n')
    with pytest.raises(ValueError, match="only a truncated final line"):
        read_jsonl_tolerant(path)


def test_newline_terminated_invalid_tail_is_not_treated_as_a_truncation(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"record_type":"run_start","run_id":"a"}\nnot-json\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only a truncated final line"):
        read_jsonl_tolerant(path)


def test_resume_repairs_one_truncated_tail_before_appending_new_run(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with JsonlLogger(
        path, experiment_id="exp", run_id="parent", fsync_every_steps=1
    ) as parent:
        parent.append_attempt("train_step", _fields(1, 1.0))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"partial"')

    with JsonlLogger(
        path,
        experiment_id="exp",
        run_id="child",
        parent_run_id="parent",
        resume_from_step=1,
        checkpoint_next_attempt_index=1,
        fsync_every_steps=1,
    ) as child:
        child.append_attempt("train_step", _fields(2, 0.5))

    records = read_jsonl_tolerant(path)
    assert [record["record_type"] for record in records] == [
        "run_start",
        "train_step",
        "run_start",
        "train_step",
    ]
    assert records_for_latest_lineage(records)[-1]["loss_total"] == 0.5


def test_resume_repairs_tail_truncated_inside_utf8_character(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with JsonlLogger(
        path, experiment_id="exp", run_id="parent", fsync_every_steps=1
    ) as parent:
        parent.append_attempt("train_step", _fields(1, 1.0))
    with path.open("ab") as handle:
        handle.write(b'{"reason":"' + "失败".encode("utf-8")[:-1])

    # Reading and reopening must both tolerate the sole partial UTF-8 tail.
    assert len(read_jsonl_tolerant(path)) == 2
    with JsonlLogger(
        path,
        experiment_id="exp",
        run_id="child",
        parent_run_id="parent",
        resume_from_step=1,
        checkpoint_next_attempt_index=1,
        fsync_every_steps=1,
    ) as child:
        child.append_attempt("train_step", _fields(2, 0.25))

    records = read_jsonl_tolerant(path)
    assert records_for_latest_lineage(records)[-1]["loss_total"] == 0.25


def test_plotter_writes_all_png_and_svg_outputs(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with JsonlLogger(path, experiment_id="exp", run_id="only") as logger:
        logger.append_attempt("train_step", _fields(1, 1.0))
        logger.append_attempt("train_step", _fields(2, 0.5))
    # A broken final write from an interrupted process is tolerated.
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"partial"')
    outputs = plot_training_metrics(path, tmp_path / "plots", rolling_window=2)
    assert len(outputs) == 12
    assert all(output.is_file() and output.stat().st_size > 0 for output in outputs)
