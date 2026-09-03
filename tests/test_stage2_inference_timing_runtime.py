from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.stage2_rollout_profile import resolve_stage2_rollout_profile
from tests.test_stage2_inference_artifacts import _generation, _samples
from tests.test_stage2_inference_runtime import _config, _context, _runtime_ops, _sample
from utils.stage2_inference_artifacts import _validate_generation_trace
from utils.stage2_inference_batch import (
    STAGE2_SINGLE_DATASET,
    STAGE2_TWO_ACTION_DATASET,
)
from utils.stage2_inference_runtime import run_stage2_inference


def test_runtime_persists_timing_and_resume_preserves_measurements(tmp_path: Path):
    config = _config(tmp_path, profile="c4w16k4s1")
    samples = tuple(
        _sample(
            tmp_path,
            dataset=dataset,
            row_id=index,
            seed=index + 1,
            prompts=prompts,
            profile="c4w16k4s1",
        )
        for index, (dataset, prompts) in enumerate(
            (
                (STAGE2_SINGLE_DATASET, ("single prompt",)),
                (STAGE2_TWO_ACTION_DATASET, ("action A", "action B")),
            )
        )
    )
    calls = {}
    ops = _runtime_ops(samples, calls)
    result = run_stage2_inference(config, context=_context(), ops=ops)
    root = Path(config.output_root)
    trace_paths = [root / sample.trace_relative_path for sample in samples]
    timings = [
        json.loads(path.read_text())["generation"]["timing"] for path in trace_paths
    ]
    assert [item["cold_start"] for item in timings] == [True, False]
    assert [item["rank_sample_index"] for item in timings] == [0, 1]
    assert len({item["process_run_id"] for item in timings}) == 1
    assert all(item["method"] == "cpu_wall" for item in timings)
    assert all(item["video_postprocess_calls"] == 1 for item in timings)
    assert all(item["video_postprocess_seconds"] > 0 for item in timings)
    summary_path = Path(result["timing_summary"])
    csv_path = Path(result["timing_samples"])
    summary = json.loads(summary_path.read_text())
    assert summary["measured_sample_count"] == 2
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    assert {int(row["output_frames"]) for row in rows} == {96, 192}
    before = {
        path: path.read_bytes() for path in (*trace_paths, summary_path, csv_path)
    }

    resumed = run_stage2_inference(config, context=_context(), ops=ops)
    assert resumed["local_generated"] == 0
    assert resumed["local_skipped"] == 2
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert calls["video_writes"] == 2


@pytest.mark.parametrize("dataset", [STAGE2_SINGLE_DATASET, STAGE2_TWO_ACTION_DATASET])
def test_c4_timing_trace_validates_actual_forward_counts(dataset: str):
    sample = replace(
        next(item for item in _samples() if item.dataset == dataset),
        profile="c4w16k4s1",
    )
    spec = resolve_stage2_rollout_profile(sample.profile)
    generation = _generation(sample)
    two_action = dataset == STAGE2_TWO_ACTION_DATASET
    generation["timing"] = {
        "schema": "longlive_stage2_inference_timing/v1",
        "method": "cpu_wall",
        "device": "cpu",
        "device_name": "cpu",
        "dit_seconds": 1.0,
        "dit_calls": 61 if two_action else 31,
        "vae_decode_seconds": 2.0,
        "vae_decode_calls": 2 if two_action else 1,
        "video_postprocess_seconds": 3.0,
        "video_postprocess_calls": 4 if two_action else 2,
        "total_seconds": 7.0,
        "other_seconds": 1.0,
        "rank": 0,
        "rank_sample_index": 0,
        "cold_start": True,
        "process_run_id": "a" * 32,
    }
    _validate_generation_trace(sample, generation, spec)
    generation["timing"]["dit_calls"] -= 1
    with pytest.raises(ValueError, match="timing call counts"):
        _validate_generation_trace(sample, generation, spec)
