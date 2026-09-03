"""Resume old output through production artifact validation after code updates."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pipeline.stage2_rollout_profile import resolve_stage2_rollout_profile
from tests.test_stage2_inference_artifacts import _generation
from tests.test_stage2_inference_runtime import (
    _CHECKPOINT,
    _context,
    _runtime_asset_identity,
    _runtime_ops,
    _sample,
    _sweep_config,
    _video_probe,
    _video_result,
    _video_writer,
)
from utils.stage1_io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
)
from utils.stage2_inference_artifacts import (
    STAGE2_INFERENCE_MANIFEST_NAME,
    STAGE2_REVIEW_INDEX_NAME,
    build_stage2_inference_config_identity,
    build_stage2_inference_manifest,
    build_stage2_sample_trace,
    validate_stage2_inference_artifact_set,
    validate_stage2_inference_manifest,
    validate_stage2_inference_manifest_artifacts,
    validate_stage2_sample_trace,
    write_stage2_inference_manifest,
    write_stage2_review_index,
    write_stage2_sample_trace,
)
from utils.stage2_inference_batch import (
    STAGE2_SINGLE_DATASET,
    STAGE2_TWO_ACTION_DATASET,
    Stage2InferenceSample,
)
from utils.stage2_inference_runtime import (
    Stage2InferenceRuntimeOps,
    run_stage2_inference,
)
from utils.stage2_inference_timing import stage2_timing_span
from utils.stage2_inference_timing_report import build_stage2_timing_report


def _legacy_timing(sample: Stage2InferenceSample, index: int) -> dict[str, Any]:
    two_action = sample.dataset == STAGE2_TWO_ACTION_DATASET
    return {
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
        "rank_sample_index": index,
        "cold_start": index == 0,
        "process_run_id": "a" * 32,
    }


def _historical_metadata(payload: dict[str, Any], kind: str, hash_key: str) -> None:
    if kind == "missing":
        payload.pop("code_version")
    elif kind == "malformed":
        payload["code_version"] = ["historical", {"source_hash": None}]
    body = dict(payload)
    body.pop(hash_key)
    payload[hash_key] = canonical_json_sha256(body)


def _production_artifact_ops(
    samples: tuple[Stage2InferenceSample, ...], calls: dict[str, Any]
) -> Stage2InferenceRuntimeOps:
    """Mock expensive model/video boundaries, retaining all artifact checks."""

    model_ops = _runtime_ops(samples, calls)
    by_dataset = {sample.dataset: sample for sample in samples}

    def generate(dataset: str, generator: Any, *args: Any, **kwargs: Any) -> Any:
        sample = by_dataset[dataset]
        expected_timing = _legacy_timing(sample, 0)
        for stage in ("dit", "vae_decode", "video_postprocess"):
            # Runtime contributes the final MP4 encoding postprocess span.
            calls_to_record = expected_timing[f"{stage}_calls"] - (
                1 if stage == "video_postprocess" else 0
            )
            for _ in range(calls_to_record):
                with stage2_timing_span(stage, "cpu"):
                    pass
        result = generator(*args, **kwargs)
        return replace(result, trace=_generation(sample))

    def forbidden_source_hash() -> dict[str, str]:
        pytest.fail("resuming inference must never capture or hash source code")

    return replace(
        model_ops,
        generate_single=lambda *args, **kwargs: generate(
            STAGE2_SINGLE_DATASET, model_ops.generate_single, *args, **kwargs
        ),
        generate_two=lambda *args, **kwargs: generate(
            STAGE2_TWO_ACTION_DATASET, model_ops.generate_two, *args, **kwargs
        ),
        build_sample_trace=build_stage2_sample_trace,
        validate_sample_trace=validate_stage2_sample_trace,
        write_sample_trace=write_stage2_sample_trace,
        validate_artifact_set=validate_stage2_inference_artifact_set,
        build_manifest=build_stage2_inference_manifest,
        validate_manifest=validate_stage2_inference_manifest,
        write_manifest=write_stage2_inference_manifest,
        write_review_index=write_stage2_review_index,
        capture_code_version=forbidden_source_hash,
    )


@pytest.mark.parametrize("complete", [True, False], ids=["complete", "partial"])
@pytest.mark.parametrize("legacy_code", ["historical", "missing", "malformed"])
def test_code_update_resumes_real_artifacts_and_rebuilds_derived_reports(
    tmp_path: Path, complete: bool, legacy_code: str
) -> None:
    spec = resolve_stage2_rollout_profile("c4w16k4s1")
    config = replace(
        _sweep_config(tmp_path),
        profiles=(spec.name,),
        seeds=(1,),
        profile_set_sha256=canonical_json_sha256([spec.to_dict()]),
        _rollout_specs=(spec,),
    )
    samples = tuple(
        replace(
            _sample(
                tmp_path,
                dataset=dataset,
                row_id=0,
                seed=1,
                prompts=prompts,
                profile=spec.name,
            ),
            width=64,
        )
        for dataset, prompts in (
            (STAGE2_SINGLE_DATASET, ("single prompt",)),
            (STAGE2_TWO_ACTION_DATASET, ("action A", "action B")),
        )
    )
    root = Path(config.output_root)
    root.mkdir()
    inference_config = build_stage2_inference_config_identity(
        config, runtime_assets=_runtime_asset_identity(config)
    )
    metadata = {
        dataset: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
        for dataset, path in (
            (STAGE2_SINGLE_DATASET, config.single_metadata),
            (STAGE2_TWO_ACTION_DATASET, config.two_action_metadata),
        )
    }
    traces: dict[str, dict[str, Any]] = {}
    trace_paths: dict[str, Path] = {}
    committed_samples = samples if complete else samples[:1]
    for index, sample in enumerate(committed_samples):
        video = root / sample.output_relative_path
        video.parent.mkdir(parents=True, exist_ok=True)
        _video_writer(_video_result(sample).video, video, fps=24)
        trace = build_stage2_sample_trace(
            sample=sample,
            generation_trace={
                **_generation(sample),
                "timing": _legacy_timing(sample, index),
            },
            output_root=root,
            video_path=video,
            checkpoint=_CHECKPOINT,
            inference_config=inference_config,
            code_version={"stage2_source_sha256": ("b" if index else "a") * 64},
            probe_fn=_video_probe,
        )
        _historical_metadata(trace, legacy_code, "trace_sha256")
        traces[sample.sample_key] = trace
        trace_paths[sample.sample_key] = write_stage2_sample_trace(
            root, sample=sample, trace=trace
        )

    manifest_path = root / STAGE2_INFERENCE_MANIFEST_NAME
    old_manifest_bytes = None
    if complete:
        manifest = build_stage2_inference_manifest(
            output_root=root,
            samples=samples,
            traces=traces,
            trace_paths=trace_paths,
            checkpoint=_CHECKPOINT,
            inference_config=inference_config,
            metadata=metadata,
            code_version={"stage2_source_sha256": "c" * 64},
        )
        _historical_metadata(manifest, legacy_code, "manifest_sha256")
        write_stage2_inference_manifest(root, manifest=manifest)
        write_stage2_review_index(root, manifest=manifest)
        validate_stage2_inference_manifest_artifacts(root, manifest)
        old_manifest_bytes = manifest_path.read_bytes()

    # Represent reports left by an older renderer, including an interrupted run.
    old_summary, old_csv = build_stage2_timing_report(
        samples,
        traces,
        checkpoint=_CHECKPOINT,
        code_version={"stage2_source_sha256": "d" * 64},
    )
    old_summary["statistics"]["total_scope"] = "Historical report wording"
    atomic_write_json(root / "timing_summary.json", old_summary)
    old_rows = list(csv.reader(io.StringIO(old_csv.decode("utf-8"))))
    old_rows[0].append("legacy_code_version")
    for row in old_rows[1:]:
        row.append("d" * 64)
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(old_rows)
    atomic_write_bytes(root / "timing_samples.csv", stream.getvalue().encode("utf-8"))
    atomic_write_bytes(
        root / STAGE2_REVIEW_INDEX_NAME, b"<html>Old review index</html>"
    )
    old_reports = {
        path: path.read_bytes()
        for path in (
            root / "timing_summary.json",
            root / "timing_samples.csv",
            root / STAGE2_REVIEW_INDEX_NAME,
        )
    }
    old_artifacts = {
        path: path.read_bytes()
        for sample in committed_samples
        for path in (
            root / sample.output_relative_path,
            root / sample.trace_relative_path,
        )
    }
    calls: dict[str, Any] = {}
    result = run_stage2_inference(
        config, context=_context(), ops=_production_artifact_ops(samples, calls)
    )

    assert result["status"] == "complete"
    assert result["local_generated"] == (0 if complete else 1)
    assert result["local_skipped"] == len(committed_samples)
    assert calls["video_writes"] == (0 if complete else 1)
    assert calls["single_calls"] == []
    assert len(calls["two_calls"]) == (0 if complete else 1)
    assert all(path.read_bytes() == before for path, before in old_artifacts.items())
    assert all(path.read_bytes() != before for path, before in old_reports.items())
    if complete:
        assert manifest_path.read_bytes() == old_manifest_bytes

    final_manifest = json.loads(manifest_path.read_text())
    assert (
        validate_stage2_inference_manifest_artifacts(
            root,
            final_manifest,
            expected_checkpoint=_CHECKPOINT,
            expected_resolved_config=inference_config["resolved"],
            expected_samples=samples,
        )
        == final_manifest
    )
    final_traces = {
        sample.sample_key: json.loads((root / sample.trace_relative_path).read_text())
        for sample in samples
    }
    if not complete:
        assert final_traces[samples[1].sample_key]["code_version"] == {}
    expected_summary, expected_csv = build_stage2_timing_report(
        samples, final_traces, checkpoint=_CHECKPOINT, code_version={}
    )
    assert expected_summary["measured_sample_count"] == 2
    assert (root / "timing_summary.json").read_bytes() == (
        canonical_json_bytes(expected_summary) + b"\n"
    )
    assert (root / "timing_samples.csv").read_bytes() == expected_csv


def test_single_action_c4_run_writes_measured_reports_and_resumes_without_generation(
    tmp_path: Path,
) -> None:
    spec = resolve_stage2_rollout_profile("c4w16k4s1")
    config = replace(
        _sweep_config(tmp_path),
        profiles=(spec.name,),
        seeds=(1,),
        two_action_row_ids=(),
        profile_set_sha256=canonical_json_sha256([spec.to_dict()]),
        _rollout_specs=(spec,),
    )
    sample = replace(
        _sample(
            tmp_path,
            dataset=STAGE2_SINGLE_DATASET,
            row_id=0,
            seed=1,
            prompts=("single prompt",),
            profile=spec.name,
        ),
        width=64,
    )
    samples = (sample,)
    calls: dict[str, Any] = {}

    def forbidden_two_generation(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("single-action configuration must never generate two actions")

    ops = replace(
        _production_artifact_ops(samples, calls),
        generate_two=forbidden_two_generation,
    )
    result = run_stage2_inference(config, context=_context(), ops=ops)

    assert result["status"] == "complete"
    assert result["local_generated"] == 1
    assert result["local_skipped"] == 0
    assert calls["single_calls"] == [spec.name]
    assert calls["two_calls"] == []
    assert calls["video_writes"] == 1

    root = Path(config.output_root)
    manifest = json.loads((root / STAGE2_INFERENCE_MANIFEST_NAME).read_text())
    inference_config = build_stage2_inference_config_identity(
        config, runtime_assets=_runtime_asset_identity(config)
    )
    assert (
        validate_stage2_inference_manifest_artifacts(
            root,
            manifest,
            expected_checkpoint=_CHECKPOINT,
            expected_resolved_config=inference_config["resolved"],
            expected_samples=samples,
        )
        == manifest
    )
    assert manifest["expected_sample_count"] == 1
    assert [entry["dataset"] for entry in manifest["samples"]] == [
        STAGE2_SINGLE_DATASET
    ]
    summary = json.loads((root / "timing_summary.json").read_text())
    assert summary["sample_count"] == summary["measured_sample_count"] == 1
    assert summary["missing_timing_sample_count"] == 0
    assert len(summary["groups"]) == 1
    assert summary["groups"][0]["dataset"] == STAGE2_SINGLE_DATASET
    assert summary["groups"][0]["profile"] == spec.name
    with (root / "timing_samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["sample_key"] == sample.sample_key
    assert rows[0]["dataset"] == STAGE2_SINGLE_DATASET
    assert rows[0]["profile"] == spec.name
    assert rows[0]["dit_calls"] == "31"
    assert rows[0]["vae_decode_calls"] == "1"
    assert rows[0]["video_postprocess_calls"] == "2"
    for stage in ("dit", "vae_decode", "video_postprocess"):
        assert float(rows[0][f"{stage}_seconds"]) > 0

    def forbidden_resume_generation(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("completed single-action sample must be reused on resume")

    resumed = run_stage2_inference(
        config,
        context=_context(),
        ops=replace(ops, generate_single=forbidden_resume_generation),
    )
    assert resumed["status"] == "complete"
    assert resumed["local_generated"] == 0
    assert resumed["local_skipped"] == 1
    assert calls["single_calls"] == [spec.name]
    assert calls["two_calls"] == []
    assert calls["video_writes"] == 1
    assert json.loads((root / "timing_summary.json").read_text()) == summary
