from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import pytest

from utils.config import DEFAULT_NEGATIVE_PROMPT
from utils.stage1_continuation_validation import (
    load_prepared_continuation_manifest,
    prepare_continuation_inference,
    validate_continuation_outputs,
)
from utils.stage1_io import atomic_write_json, canonical_json_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_METADATA = (
    PROJECT_ROOT
    / "testsets"
    / "metadata_8cases_two_actions_continuation_480x832_253frames.csv"
)


def _model_paths(root: Path) -> dict[str, Path]:
    architecture_root = root / "architecture"
    tokenizer_dir = root / "tokenizer"
    architecture_root.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "base_checkpoint": root / "base.pt",
        "architecture_root": architecture_root,
        "t5_checkpoint": root / "t5.pt",
        "tokenizer_dir": tokenizer_dir,
        "vae_checkpoint": root / "vae.pt",
    }
    for name, path in paths.items():
        if name not in {"architecture_root", "tokenizer_dir"}:
            path.write_bytes(f"fixture:{name}".encode())
    return paths


def _prepare_kwargs(tmp_path: Path, *, name: str = "run") -> dict[str, Any]:
    root = tmp_path / name
    return {
        "metadata_path": FORMAL_METADATA,
        "output_root": root / "prepared",
        "video_root": root / "videos",
        **_model_paths(root / "models"),
    }


@pytest.fixture
def prepared(tmp_path):
    kwargs = _prepare_kwargs(tmp_path)
    manifest = prepare_continuation_inference(**kwargs)
    manifest_path = Path(kwargs["output_root"]) / "prepared_manifest.json"
    return manifest_path, manifest


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _valid_session_trace(
    manifest: dict[str, Any],
    entry: dict[str, Any],
    record,
    output: dict[str, Any],
) -> dict[str, Any]:
    sink_size = int(output["sink_size"])
    latent_height = record.height // 16
    latent_width = record.width // 16
    frame_seq_length = latent_height * latent_width // 4
    negative_hash = manifest["sampling"]["negative_prompt_sha256"]
    segment_specs = (
        ("action_a", 0, 23, 24, 3, record.action_a_prompt, False),
        ("hold", 24, 39, 16, 2, record.hold_prompt, False),
        ("action_b", 40, 63, 24, 3, record.action_b_prompt, True),
    )
    segments = []
    prompt_hashes = []
    for name, start, end, frames, blocks, prompt, carry in segment_specs:
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompt_hashes.append(prompt_hash)
        segments.append(
            {
                "name": name,
                "start_latent": start,
                "end_latent": end,
                "latent_frames": frames,
                "blocks": blocks,
                "carry_last_latent_as_anchor": carry,
                "prompt": {
                    "sha256": prompt_hash,
                    "token_count": 128,
                    "token_limit": 512,
                    "cleaning": "whitespace",
                    "add_special_tokens": True,
                    "truncation": False,
                },
            }
        )

    segment_indices = (0, 0, 0, 1, 1, 2, 2, 2)
    blocks = []
    for block_index, segment_index in enumerate(segment_indices):
        end_frames = (block_index + 1) * 8
        blocks.append(
            {
                "block_index": block_index,
                "segment_index": segment_index,
                "global_start_latent": block_index * 8,
                "global_end_latent": end_frames - 1,
                "positive_prompt_sha256": prompt_hashes[segment_index],
                "negative_prompt_sha256": negative_hash,
                # These are raw token cursors, not latent-frame cursors.
                "global_end_tokens": frame_seq_length * end_frames,
                "local_end_tokens": frame_seq_length * min(end_frames, 24),
                "global_end_frames": end_frames,
                "local_end_frames": min(end_frames, 24),
                "cache_capacity_tokens": frame_seq_length * 24,
                "effective_attention_local_size": 24,
                "pinned_start": -1,
                "pinned_len": 0,
            }
        )

    shared_initial = _sha256(f"{record.case_group}:initial")
    shared_noise = _sha256(f"{record.case_group}:noise-plan")
    return {
        "schema": "longlive_stage1_continuation_session",
        "schema_version": 1,
        "status": "finished",
        "failure": None,
        "metadata": {
            "row_id": record.row_id,
            "row_sha256": record.row_sha256,
            "image_sha256": record.image_sha256,
            "case_group": record.case_group,
            "cat_id": record.cat_id,
            "action_order": record.action_order,
            "block_schedule": [3, 2, 3],
            "soft_reanchor": True,
        },
        "sampling": {
            "solver": "unipc",
            "sampling_steps": 50,
            "guidance_scale": 5.0,
            "seed": 1,
            "negative_prompt_sha256": negative_hash,
            "sink_size": sink_size,
        },
        "locked": {
            "batch_size": 1,
            "latent_shape": [48, latent_height, latent_width],
            "dtype": "bfloat16",
            "device": "cuda:0",
            "guidance_scale": 5.0,
            "sample_solver": "unipc",
            "sampling_steps": 50,
            "timestep_shift": 5.0,
            "num_train_timesteps": 1000,
            "negative_prompt_sha256": negative_hash,
            "block_size": 8,
            "sink_size": sink_size,
            "pipeline_sink_size": 0,
            "global_sink_size": 0,
            "multi_shot_sink": False,
            "shot_clean_recache": False,
            "multi_shot_rope_offset": 0.0,
            "quantize_kv": False,
            "independent_first_frame": True,
            "streaming_vae": False,
            "async_vae": False,
            "local_attn_size_config": -1,
            "use_relative_rope": False,
            "rope_method": "linear",
            "effective_t_scale": 1.0,
            "effective_local_attn_size": -1,
            "effective_attention_local_size": 24,
            "effective_use_relative_rope": False,
            "effective_original_seq_len": None,
            "effective_rope_temporal_offset": 0.0,
            "frame_seq_length": frame_seq_length,
        },
        "cursor_frames": 64,
        "initial_latent_sha256": shared_initial,
        "noise_identity_sha256": shared_noise,
        "negative_prompt": {
            "sha256": negative_hash,
            "token_count": 64,
            "token_limit": 512,
            "cleaning": "whitespace",
            "add_special_tokens": True,
            "truncation": False,
        },
        "segments": segments,
        "noise_slices": [
            {
                "segment_index": segment_index,
                "start_latent": start,
                "end_latent_exclusive": end,
                "sha256": _sha256(f"{record.case_group}:noise:{start}:{end}"),
            }
            for segment_index, (start, end) in enumerate(((0, 24), (24, 40), (40, 64)))
        ],
        "blocks": blocks,
        "anchors": [
            {"kind": "initial", "source": "initial_latent", "destination": 0},
            {"kind": "soft_reanchor", "source": 39, "destination": 40},
        ],
        "result": {
            "latent_shape": [1, 64, 48, latent_height, latent_width],
            "pixel_shape": [1, 253, 3, record.height, record.width],
            "decode_calls": 1,
            "decode_mode": "single_full_sequence",
            "latent_boundaries": manifest["frame_policy"]["latent_boundaries"],
            "pixel_boundaries": manifest["frame_policy"]["pixel_boundaries"],
        },
        "output": {
            "output_video": output["output_video"],
            "session_trace": output["session_trace"],
        },
    }


def _write_valid_artifacts(
    manifest_path: Path,
) -> list[tuple[Path, Path, dict[str, Any]]]:
    manifest, records = load_prepared_continuation_manifest(manifest_path)
    records_by_row = {record.row_id: record for record in records}
    written = []
    for entry in manifest["records"]:
        record = records_by_row[int(entry["row_id"])]
        for output in entry["outputs"]:
            video_path = Path(output["output_video"])
            trace_path = Path(output["session_trace"])
            video_path.write_bytes(
                f"{record.case_group}/sink{output['sink_size']}".encode()
            )
            trace = _valid_session_trace(manifest, entry, record, output)
            atomic_write_json(trace_path, trace)
            written.append((video_path, trace_path, trace))
    assert len(written) == 16
    return written


def _passing_video_gate(video_path, source_image_path, **kwargs):
    return {
        "stream": {
            "width": kwargs["expected_width"],
            "height": kwargs["expected_height"],
            "frame_count": kwargs["expected_frames"],
            "fps": float(kwargs["expected_fps"]),
        },
        "metrics": {
            "decoded_frame_count": float(kwargs["expected_frames"]),
            "first_frame_psnr_db": 30.0,
            "mean_frame_std": 20.0,
            "mean_temporal_abs_diff": 1.0,
        },
    }


def test_prepare_builds_two_locked_geometry_configs_and_canonical_matrix(prepared):
    manifest_path, manifest = prepared
    loaded, records = load_prepared_continuation_manifest(manifest_path)

    assert loaded == manifest
    assert len(records) == len(manifest["records"]) == 8
    assert manifest["frame_policy"]["num_latent_frames"] == 64
    assert manifest["frame_policy"]["expected_pixel_frames"] == 253
    assert manifest["frame_policy"]["fps"] == 24
    assert manifest["sampling"]["sampling_steps"] == 50
    assert manifest["sampling"]["guidance_scale"] == 5.0
    assert manifest["sampling"]["seed"] == 1
    assert manifest["sink_sizes"] == [0, 1]

    expected_groups = {record.case_group for record in records}
    assert {entry["case_group"] for entry in manifest["records"]} == expected_groups
    mappings = []
    video_root = Path(manifest["video_root"])
    for entry in manifest["records"]:
        assert entry["block_schedule"] == [3, 2, 3]
        assert entry["latent_frame_schedule"] == [24, 16, 24]
        assert [output["sink_size"] for output in entry["outputs"]] == [0, 1]
        for output in entry["outputs"]:
            sink_size = output["sink_size"]
            expected_dir = video_root / entry["case_group"]
            assert Path(output["output_video"]) == expected_dir / f"sink{sink_size}.mp4"
            assert Path(output["session_trace"]) == (
                expected_dir / f"sink{sink_size}.session.json"
            )
            mappings.append((entry["case_group"], sink_size))
    assert len(mappings) == len(set(mappings)) == 16

    assert [bucket["bucket_id"] for bucket in manifest["buckets"]] == [
        "landscape_480x832",
        "portrait_832x480",
    ]
    expected_shapes = {
        "landscape_480x832": [1, 64, 48, 30, 52],
        "portrait_832x480": [1, 64, 48, 52, 30],
    }
    all_rows = []
    for bucket in manifest["buckets"]:
        config = OmegaConf.load(bucket["config_path"])
        assert (
            list(config.data.image_or_video_shape)
            == expected_shapes[bucket["bucket_id"]]
        )
        assert config.num_output_frames == 64
        assert config.model_kwargs.num_frame_per_block == 8
        assert config.model_kwargs.timestep_shift == 5.0
        assert config.model_kwargs.local_attn_size == -1
        assert config.inference.sampling_steps == 50
        assert config.inference.guidance_scale == 5.0
        assert config.inference.sink_size == 0
        assert config.inference.local_attn_size == -1
        assert config.inference.negative_prompt == DEFAULT_NEGATIVE_PROMPT
        assert config.logging.seed == 1
        assert config.continuation.enabled is True
        assert Path(config.continuation.manifest_path) == manifest_path
        all_rows.extend(bucket["row_ids"])
    assert sorted(all_rows) == list(range(8))


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("sampling_steps", 49, "sampling_steps is locked to 50"),
        ("guidance_scale", 4.5, "guidance_scale is locked to 5.0"),
        ("seed", 2, "seed is locked to 1"),
        ("negative_prompt", "different", "negative prompt must remain"),
    ],
)
def test_prepare_rejects_unlocked_sampling_parameters(tmp_path, name, value, message):
    kwargs = _prepare_kwargs(tmp_path)
    kwargs[name] = value
    with pytest.raises(ValueError, match=message):
        prepare_continuation_inference(**kwargs)


def test_prepare_rejects_nonempty_preparation_or_video_directories(tmp_path):
    first = _prepare_kwargs(tmp_path, name="nonempty-prepared")
    Path(first["output_root"]).mkdir(parents=True)
    (Path(first["output_root"]) / "stale.txt").write_text("stale")
    with pytest.raises(FileExistsError, match="preparation root.*must be empty"):
        prepare_continuation_inference(**first)

    second = _prepare_kwargs(tmp_path, name="nonempty-videos")
    Path(second["video_root"]).mkdir(parents=True)
    (Path(second["video_root"]) / "stale.txt").write_text("stale")
    with pytest.raises(FileExistsError, match="video root.*must be empty"):
        prepare_continuation_inference(**second)

    third = _prepare_kwargs(tmp_path, name="same-root")
    third["video_root"] = third["output_root"]
    with pytest.raises(ValueError, match="must be distinct"):
        prepare_continuation_inference(**third)


@pytest.mark.parametrize("tamper_target", ["manifest", "config"])
def test_prepared_loader_rejects_manifest_or_config_hash_tampering(
    prepared, tamper_target
):
    manifest_path, manifest = prepared
    if tamper_target == "manifest":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["sampling"]["seed"] = 2
        # Deliberately keep the recorded hash unchanged.
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        message = "manifest SHA256 mismatch"
    else:
        config_path = Path(manifest["buckets"][0]["config_path"])
        config_path.write_text(
            config_path.read_text(encoding="utf-8") + "\n# tampered\n",
            encoding="utf-8",
        )
        message = "config is missing or changed"

    with pytest.raises(RuntimeError, match=message):
        load_prepared_continuation_manifest(manifest_path)


@pytest.mark.parametrize(
    ("field", "value_kind", "message"),
    [
        ("output_video", "escaped", "escaped its video root"),
        ("session_trace", "escaped", "escaped its video root"),
        ("output_video", "noncanonical", "path mapping is not canonical"),
        ("session_trace", "noncanonical", "path mapping is not canonical"),
    ],
)
def test_prepared_loader_rejects_resigned_unsafe_output_paths(
    prepared, tmp_path, field, value_kind, message
):
    manifest_path, _manifest = prepared
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = payload["records"][0]["outputs"][0]
    if value_kind == "escaped":
        suffix = ".mp4" if field == "output_video" else ".session.json"
        output[field] = str(tmp_path / f"escaped{suffix}")
    else:
        original = Path(output[field])
        output[field] = str(original.with_name(f"wrong-{original.name}"))
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    atomic_write_json(manifest_path, payload)

    with pytest.raises(RuntimeError, match=message):
        load_prepared_continuation_manifest(manifest_path)


def test_prepared_loader_rejects_non_mapping_output_entry(prepared):
    manifest_path, _manifest = prepared
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records"][0]["outputs"][0] = None
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    atomic_write_json(manifest_path, payload)

    with pytest.raises(RuntimeError, match="output sink mapping is invalid"):
        load_prepared_continuation_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_mp4", "continuation MP4 set mismatch"),
        ("extra_mp4", "continuation MP4 set mismatch"),
        ("missing_trace", "continuation session trace set mismatch"),
        ("extra_trace", "continuation session trace set mismatch"),
    ],
)
def test_output_validation_requires_exact_mp4_and_session_sets(
    prepared, mutation, message
):
    manifest_path, manifest = prepared
    artifacts = _write_valid_artifacts(manifest_path)
    video_root = Path(manifest["video_root"])
    if mutation == "missing_mp4":
        artifacts[0][0].unlink()
    elif mutation == "extra_mp4":
        (video_root / "unexpected.mp4").write_bytes(b"stale")
    elif mutation == "missing_trace":
        artifacts[0][1].unlink()
    elif mutation == "extra_trace":
        (video_root / "unexpected.session.json").write_text("{}")
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=message):
        validate_continuation_outputs(
            manifest_path,
            video_validator=_passing_video_gate,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("global_tokens", "field global_end_tokens mismatch"),
        ("local_tokens", "field local_end_tokens mismatch"),
        ("segments", "exactly three segments"),
        ("anchors", "anchor mapping is invalid"),
        ("latent_shape", "result shape/decode policy is invalid"),
        ("decode_mode", "result shape/decode policy is invalid"),
        ("block_schedule", "metadata block_schedule mismatch"),
        ("prompt_special_tokens", "prompt audit is invalid"),
        ("negative_prompt", "negative prompt audit is invalid"),
        ("locked_streaming", "locked streaming_vae mismatch"),
    ],
)
def test_output_validation_rejects_trace_cursor_segment_anchor_and_decode_drift(
    prepared, mutation, message
):
    manifest_path, _manifest = prepared
    artifacts = _write_valid_artifacts(manifest_path)
    trace_path = artifacts[0][1]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if mutation == "global_tokens":
        trace["blocks"][4]["global_end_tokens"] += 1
    elif mutation == "local_tokens":
        trace["blocks"][3]["local_end_tokens"] += 1
    elif mutation == "segments":
        trace["segments"].pop()
    elif mutation == "anchors":
        trace["anchors"][1]["destination"] = 41
    elif mutation == "latent_shape":
        trace["result"]["latent_shape"][1] = 63
    elif mutation == "decode_mode":
        trace["result"]["decode_calls"] = 3
        trace["result"]["decode_mode"] = "per_segment"
    elif mutation == "block_schedule":
        trace["metadata"]["block_schedule"] = [4, 1, 3]
    elif mutation == "prompt_special_tokens":
        trace["segments"][0]["prompt"]["add_special_tokens"] = False
    elif mutation == "negative_prompt":
        trace["negative_prompt"]["token_count"] = 512
    elif mutation == "locked_streaming":
        trace["locked"]["streaming_vae"] = True
    else:  # pragma: no cover
        raise AssertionError(mutation)
    atomic_write_json(trace_path, trace)

    with pytest.raises(RuntimeError, match=message):
        validate_continuation_outputs(
            manifest_path,
            video_validator=_passing_video_gate,
        )


@pytest.mark.parametrize(
    "identity_field", ["initial_latent_sha256", "noise_identity_sha256"]
)
def test_output_validation_rejects_sink_pair_identity_mismatch(
    prepared, identity_field
):
    manifest_path, _manifest = prepared
    artifacts = _write_valid_artifacts(manifest_path)
    sink1_trace_path = artifacts[1][1]
    trace = json.loads(sink1_trace_path.read_text(encoding="utf-8"))
    trace[identity_field] = _sha256(f"different:{identity_field}")
    atomic_write_json(sink1_trace_path, trace)

    with pytest.raises(
        RuntimeError,
        match="sink0/1 initial latent or noise identity differs",
    ):
        validate_continuation_outputs(
            manifest_path,
            video_validator=_passing_video_gate,
        )


def test_video_gate_uses_legacy_thresholds_and_trace_updates_are_all_or_none(
    prepared,
):
    manifest_path, _manifest = prepared
    artifacts = _write_valid_artifacts(manifest_path)
    failed_calls = []

    def fail_on_last(video_path, source_image_path, **kwargs):
        failed_calls.append((Path(video_path), Path(source_image_path), kwargs))
        if len(failed_calls) == 16:
            raise RuntimeError("synthetic final gate failure")
        return _passing_video_gate(video_path, source_image_path, **kwargs)

    with pytest.raises(RuntimeError, match="synthetic final gate failure"):
        validate_continuation_outputs(
            manifest_path,
            video_validator=fail_on_last,
        )
    assert len(failed_calls) == 16
    for _video, _source, kwargs in failed_calls:
        assert kwargs["expected_frames"] == 253
        assert kwargs["expected_fps"] == 24
        assert kwargs["minimum_first_frame_psnr_db"] == 12.0
        assert kwargs["minimum_frame_std"] == 5.0
        assert kwargs["minimum_temporal_abs_diff"] == 0.05
    # Fifteen in-memory passes must not leak a pass status to disk.
    for _video_path, trace_path, _trace in artifacts:
        persisted = json.loads(trace_path.read_text(encoding="utf-8"))
        assert "technical_validation" not in persisted

    passed_calls = []

    def record_pass(video_path, source_image_path, **kwargs):
        passed_calls.append((Path(video_path), Path(source_image_path), kwargs))
        return _passing_video_gate(video_path, source_image_path, **kwargs)

    report = validate_continuation_outputs(
        manifest_path,
        video_validator=record_pass,
    )
    assert report["status"] == "pass"
    assert report["sample_count"] == report["expected_sample_count"] == 16
    assert len(passed_calls) == 16
    for video_path, trace_path, _trace in artifacts:
        persisted = json.loads(trace_path.read_text(encoding="utf-8"))
        technical = persisted["technical_validation"]
        assert technical["status"] == "pass"
        assert len(technical["video_sha256"]) == 64
        assert technical["stream"]["frame_count"] == 253
        assert technical["metrics"]["decoded_frame_count"] == 253.0
