from __future__ import annotations

import csv
import io
import json
from copy import deepcopy
from dataclasses import dataclass

import pytest

from utils.stage2_inference_timing_report import (
    STAGE2_INFERENCE_TIMING_SCHEMA,
    STAGE2_INFERENCE_TIMING_SUMMARY_SCHEMA,
    build_stage2_timing_report,
    validate_stage2_timing,
)


@dataclass(frozen=True)
class Sample:
    row_id: int
    dataset: str = "single_action"
    profile: str = "c4w16k4s1"
    height: int = 480
    width: int = 832
    seed: int = 1

    @property
    def sample_key(self):
        return f"{self.dataset}/row{self.row_id:03d}/seed{self.seed:04d}/{self.profile}"


def _timing(*, seconds=1.0, index=0, rank=0, run="a" * 32, **overrides):
    return {
        "schema": STAGE2_INFERENCE_TIMING_SCHEMA,
        "method": "cuda_synchronized_wall",
        "device": f"cuda:{rank}",
        "device_name": "NVIDIA H100 80GB HBM3",
        "dit_seconds": seconds,
        "dit_calls": 31,
        "vae_decode_seconds": 2 * seconds,
        "vae_decode_calls": 1,
        "video_postprocess_seconds": 3 * seconds,
        "video_postprocess_calls": 2,
        "total_seconds": 7 * seconds,
        "other_seconds": seconds,
        "rank": rank,
        "rank_sample_index": index,
        "cold_start": index == 0,
        "process_run_id": run,
        **overrides,
    }


def _trace(timing, *, frames=96, digest="b" * 64):
    generation = {"output_pixel_frames": frames}
    if timing is not None:
        generation["timing"] = timing
    result = {"generation": generation}
    if digest is not None:
        result["trace_sha256"] = digest
    return result


def _report(samples, traces):
    return build_stage2_timing_report(
        samples,
        traces,
        checkpoint={"directory": "/checkpoint", "manifest_sha256": "c" * 64},
        code_version={"stage2_source_sha256": "d" * 64},
    )


def test_valid_timing_is_copied_and_zero_calls_are_supported():
    timing = _timing(dit_calls=0, vae_decode_calls=0)
    checked = validate_stage2_timing(timing)
    assert checked == timing
    assert checked is not timing
    assert (
        validate_stage2_timing(
            _timing(method="cpu_wall", device="cpu", device_name="cpu")
        )["method"]
        == "cpu_wall"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"dit_seconds": float("nan")},
        {"vae_decode_seconds": float("inf")},
        {"video_postprocess_seconds": -1},
        {"other_seconds": True},
        {"dit_calls": True},
        {"rank": False},
        {"rank_sample_index": 0.0},
        {"cold_start": 1},
        {"rank_sample_index": 1, "cold_start": True},
        {"total_seconds": 99},
        {"device": "cuda"},
        {"device": "cpu"},
        {"method": "cpu_wall"},
        {"device_name": " "},
        {"process_run_id": "a" * 31},
        {"process_run_id": "g" * 32},
        {"schema": "v0"},
        {"extra": 0},
    ],
)
def test_invalid_timing_is_rejected(changes):
    with pytest.raises(ValueError, match="timing"):
        validate_stage2_timing(_timing(**changes))


def test_timing_requires_every_key_and_allows_only_small_closure_roundoff():
    timing = _timing()
    del timing["vae_decode_calls"]
    with pytest.raises(ValueError, match="keys"):
        validate_stage2_timing(timing)
    validate_stage2_timing(_timing(total_seconds=7.000001))


def test_report_groups_datasets_geometry_device_and_method_and_keeps_durations():
    first = Sample(0)
    second = Sample(1)
    double = Sample(0, dataset="two_action")
    portrait = Sample(2, height=832, width=480)
    other_gpu = Sample(3)
    cpu = Sample(4)
    profile = Sample(5, profile="baseline_c8w16k4s1")
    samples = [first, second, double, portrait, other_gpu, cpu, profile]
    traces = {
        first.sample_key: _trace(_timing(seconds=1)),
        second.sample_key: _trace(_timing(seconds=3, index=1), frames=192),
        double.sample_key: _trace(_timing(seconds=5, index=2), frames=192),
        portrait.sample_key: _trace(_timing(seconds=7, index=3)),
        other_gpu.sample_key: _trace(_timing(index=4, device_name="NVIDIA A100")),
        cpu.sample_key: _trace(
            _timing(index=5, device="cpu", device_name="cpu", method="cpu_wall")
        ),
        profile.sample_key: _trace(_timing(index=6)),
    }
    before = deepcopy(traces)
    summary, csv_bytes = _report(samples, traces)
    assert summary["schema"] == STAGE2_INFERENCE_TIMING_SUMMARY_SCHEMA
    assert summary["sample_count"] == summary["measured_sample_count"] == 7
    assert summary["missing_timing_sample_count"] == 0
    assert len(summary["groups"]) == 6
    group = next(
        group
        for group in summary["groups"]
        if group["all_samples"]["sample_count"] == 2
    )
    assert group["all_samples"]["total_output_frames"] == 288
    assert group["all_samples"]["metrics"]["dit_seconds"] == {
        "mean": 2.0,
        "median": 2.0,
        "p90": 2.8,
        "min": 1.0,
        "max": 3.0,
    }
    assert group["excluding_first_per_rank_run"]["sample_count"] == 1
    assert group["excluding_first_per_rank_run"]["metrics"]["dit_seconds"]["mean"] == 3
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"))))
    by_key = {row["sample_key"]: row for row in rows}
    assert [
        int(by_key[sample.sample_key]["output_frames"])
        for sample in (first, second, double)
    ] == [96, 192, 192]
    assert rows[0]["cold_start"] == "true"
    assert rows[1]["cold_start"] == "false"
    assert rows[0]["trace_sha256"] == "b" * 64
    assert traces == before
    json.dumps(summary, allow_nan=False)


def test_missing_timings_are_explicit_and_never_enter_statistics_or_csv():
    measured, old, absent = Sample(0), Sample(1), Sample(2)
    summary, csv_bytes = _report(
        [measured, old, absent],
        {
            measured.sample_key: _trace(_timing(), digest=None),
            old.sample_key: _trace(None),
        },
    )
    assert summary["measured_sample_count"] == 1
    assert summary["missing_timing_sample_count"] == 2
    assert summary["missing_timing_sample_keys"] == [old.sample_key, absent.sample_key]
    assert summary["source"]["trace_sha256_by_sample"] == {old.sample_key: "b" * 64}
    group = summary["groups"][0]
    assert group["all_samples"]["metrics"]["dit_seconds"]["mean"] == 1
    assert group["excluding_first_per_rank_run"]["sample_count"] == 0
    assert all(
        value is None
        for value in group["excluding_first_per_rank_run"]["metrics"].values()
    )
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode())))
    assert len(rows) == 1
    assert rows[0]["trace_sha256"] == ""
    summary, csv_bytes = _report([old], {old.sample_key: _trace(None)})
    assert summary["groups"] == []
    assert list(csv.DictReader(io.StringIO(csv_bytes.decode()))) == []


def test_resume_excludes_each_recorded_rank_run_cold_start():
    samples = [Sample(index) for index in range(6)]
    timings = [
        _timing(seconds=100, rank=0, run="a" * 32, index=0),
        _timing(seconds=1, rank=0, run="a" * 32, index=1),
        _timing(seconds=200, rank=1, run="a" * 32, index=0),
        _timing(seconds=3, rank=1, run="a" * 32, index=1),
        _timing(seconds=300, rank=0, run="b" * 32, index=0),
        _timing(seconds=5, rank=0, run="b" * 32, index=1),
    ]
    traces = {
        sample.sample_key: _trace(timing) for sample, timing in zip(samples, timings)
    }
    summary, _ = _report(list(reversed(samples)), traces)
    assert len(summary["groups"]) == 1
    group = summary["groups"][0]
    assert group["all_samples"]["sample_count"] == 6
    subset = group["excluding_first_per_rank_run"]
    assert subset["sample_count"] == 3
    assert subset["total_output_frames"] == 288
    assert subset["metrics"]["dit_seconds"]["mean"] == 3


def test_present_but_invalid_timing_is_not_treated_as_missing():
    sample = Sample(0)
    trace = _trace(None)
    trace["generation"]["timing"] = None
    with pytest.raises(ValueError, match="timing"):
        _report([sample], {sample.sample_key: trace})


def test_verified_video_frame_count_is_authoritative_and_csv_order_is_stable():
    first, second = Sample(0), Sample(1)
    first_trace = _trace(_timing(), frames=192)
    first_trace["output"] = {"frame_count": 96}
    second_trace = _trace(_timing(index=1))
    del second_trace["generation"]["output_pixel_frames"]
    second_trace["output"] = {"frame_count": 192}
    forward = {first.sample_key: first_trace, second.sample_key: second_trace}
    backward = dict(reversed(list(forward.items())))
    summary, csv_bytes = _report([first, second], forward)
    assert _report([second, first], backward) == (summary, csv_bytes)
    assert summary["groups"][0]["all_samples"]["total_output_frames"] == 288
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode())))
    assert [int(row["output_frames"]) for row in rows] == [96, 192]
