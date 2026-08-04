from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from pipeline.causal_diffusion_continuation import (
    tensor_identity_sha256 as real_tensor_identity_sha256,
)
from utils import stage1_continuation_inference as orchestration
from utils.stage1_continuation_validation import ContinuationMetadataRecord


class _CudaBackedByCpuTensor:
    """Tiny tensor façade that keeps test bytes on CPU but accepts CUDA `.to()`."""

    def __init__(self, value: torch.Tensor):
        self.value = value.detach().clone()

    @property
    def shape(self):
        return self.value.shape

    def clone(self):
        return type(self)(self.value)

    def to(self, *, device=None, dtype=None):
        del device
        value = self.value if dtype is None else self.value.to(dtype=dtype)
        return type(self)(value)

    def __getitem__(self, item):
        return type(self)(self.value[item])


class _ShapeOnlyTensor:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = torch.Size(shape)


class _FakeGenerator:
    def __init__(self, *, device):
        self.device = str(device)
        self.seed = None

    def manual_seed(self, seed: int):
        self.seed = int(seed)
        return self


class _FakeSession:
    def __init__(
        self,
        *,
        initial_latent: _CudaBackedByCpuTensor,
        noise_plan: _CudaBackedByCpuTensor,
        sink_size: int,
        calls: dict[str, Any],
        record: ContinuationMetadataRecord,
        fail_on_segment: str | None = None,
    ):
        self.initial_latent = initial_latent
        self.noise_plan = noise_plan
        self.sink_size = sink_size
        self.calls = calls
        self.record = record
        self.fail_on_segment = fail_on_segment
        self.segments: list[dict[str, Any]] = []
        self.cursor = 0

    def generate_segment(
        self,
        prompt: str,
        *,
        noise: _CudaBackedByCpuTensor,
        carry_last_latent_as_anchor: bool = False,
        segment_name: str | None = None,
    ):
        name = str(segment_name)
        if name == self.fail_on_segment:
            self.segments.append(
                {
                    "name": name,
                    "prompt": prompt,
                    "start": self.cursor,
                    "end": self.cursor + int(noise.shape[1]),
                    "carry": bool(carry_last_latent_as_anchor),
                    "noise": noise.clone(),
                }
            )
            raise RuntimeError(f"synthetic failure in {name}")
        start = self.cursor
        self.cursor += int(noise.shape[1])
        self.segments.append(
            {
                "name": name,
                "prompt": prompt,
                "start": start,
                "end": self.cursor,
                "carry": bool(carry_last_latent_as_anchor),
                "noise": noise.clone(),
            }
        )
        return _ShapeOnlyTensor((1, int(noise.shape[1]), 48, 2, 2))

    def finish(self):
        self.calls["finish"].append(self)
        latent_shape = (1, 64, 48, 2, 2)
        pixel_shape = (1, 253, 3, self.record.height, self.record.width)
        trace = {
            "schema": "longlive_stage1_continuation_session",
            "schema_version": 1,
            "status": "finished",
            "cursor_frames": self.cursor,
            "locked": {"sink_size": self.sink_size},
            "initial_latent_sha256": _identity(self.initial_latent),
            "noise_identity_sha256": _identity(self.noise_plan),
            "noise_slices": [],
            "negative_prompt": {},
            "segments": [],
            "blocks": [],
            "anchors": [],
            "failure": None,
        }
        return SimpleNamespace(
            latents=_ShapeOnlyTensor(latent_shape),
            video=_ShapeOnlyTensor(pixel_shape),
            trace=trace,
        )

    def trace_snapshot(self):
        return {
            "schema": "longlive_stage1_continuation_session",
            "schema_version": 1,
            "status": "active",
            "cursor_frames": self.cursor,
            "locked": {"sink_size": self.sink_size},
            "initial_latent_sha256": _identity(self.initial_latent),
            "noise_identity_sha256": _identity(self.noise_plan),
            "noise_slices": [],
            "negative_prompt": {},
            "segments": [
                {
                    "name": segment["name"],
                    "start_latent": segment["start"],
                    "end_latent": segment["end"] - 1,
                }
                for segment in self.segments
            ],
            "blocks": [],
            "anchors": [],
            "failure": None,
        }


class _FakeVae:
    def __init__(self, calls: dict[str, Any]):
        self.calls = calls
        self.model = SimpleNamespace(clear_cache=self._clear_cache)

    def _clear_cache(self):
        self.calls["clear_vae_cache"] += 1

    def encode_to_latent(self, image):
        self.calls["encode"].append(image.clone())
        # Non-constant bytes make accidental regeneration/equality mistakes visible.
        value = torch.arange(192, dtype=torch.float32).reshape(1, 1, 48, 2, 2)
        return _CudaBackedByCpuTensor(value)


class _FakePipeline:
    num_frame_per_block = 8
    frame_seq_length = 1
    sample_solver = "unipc"
    sampling_steps = 50
    guidance_scale = 5.0
    shift = 5.0

    def __init__(
        self,
        calls: dict[str, Any],
        record: ContinuationMetadataRecord,
        *,
        fail_on_segment: str | None = None,
    ):
        self.calls = calls
        self.record = record
        self.fail_on_segment = fail_on_segment
        self.vae = _FakeVae(calls)

    def begin_session(self, *, initial_latent, sink_size, noise_plan):
        session = _FakeSession(
            initial_latent=initial_latent,
            noise_plan=noise_plan,
            sink_size=int(sink_size),
            calls=self.calls,
            record=self.record,
            fail_on_segment=self.fail_on_segment,
        )
        self.calls["sessions"].append(session)
        return session


def _identity(value: _CudaBackedByCpuTensor) -> str:
    return real_tensor_identity_sha256(value.value)


def _record(tmp_path: Path, *, row_id: int = 1, case_group: str = "cat1_ab"):
    return ContinuationMetadataRecord(
        row_id=row_id,
        input_image_path=tmp_path / f"{case_group}.png",
        action_a_prompt=f"{case_group}: action A",
        hold_prompt=f"{case_group}: hold",
        action_b_prompt=f"{case_group}: action B",
        height=32,
        width=32,
        bucket="portrait",
        case_group=case_group,
        cat_id="cat1",
        action_order="A_then_B",
        action_a_blocks=3,
        hold_blocks=2,
        action_b_blocks=3,
        soft_reanchor=True,
        image_sha256="1" * 64,
        canonical_row={"case_group": case_group},
        row_sha256=(f"{row_id:x}" * 64)[:64],
    )


def _entry(record: ContinuationMetadataRecord, root: Path) -> dict[str, Any]:
    case_root = root / record.case_group
    return {
        "row_id": record.row_id,
        "case_group": record.case_group,
        "outputs": [
            {
                "sink_size": sink_size,
                "output_video": str(case_root / f"sink{sink_size}.mp4"),
                "session_trace": str(case_root / f"sink{sink_size}.session.json"),
            }
            for sink_size in (0, 1)
        ],
    }


def _manifest(
    tmp_path: Path,
    records: list[ContinuationMetadataRecord],
    *,
    target_rows: list[int] | None = None,
) -> dict[str, Any]:
    target_rows = target_rows or [records[0].row_id]
    target_groups = [
        record.case_group for record in records if record.row_id in target_rows
    ]
    return {
        "sampling": {
            "solver": "unipc",
            "sampling_steps": 50,
            "guidance_scale": 5.0,
            "seed": 1,
            "negative_prompt_sha256": "f" * 64,
        },
        "frame_policy": {
            "num_latent_frames": 64,
            "num_frame_per_block": 8,
            "temporal_compression_ratio": 4,
            "expected_pixel_frames": 253,
            "fps": 24,
            "latent_boundaries": {
                "action_a": [0, 23],
                "hold": [24, 39],
                "action_b": [40, 63],
            },
            "pixel_boundaries": {
                "action_a": [0, 92],
                "hold": [93, 156],
                "action_b": [157, 252],
            },
        },
        "buckets": [
            {
                "bucket_id": "portrait_32x32",
                "bucket": "portrait",
                "height": 32,
                "width": 32,
                "latent_height": 2,
                "latent_width": 2,
                "row_ids": target_rows,
                "case_groups": target_groups,
            },
            {
                "bucket_id": "unused_32x32",
                "bucket": "portrait",
                "height": 32,
                "width": 32,
                "latent_height": 2,
                "latent_width": 2,
                "row_ids": [record.row_id for record in records[1:]],
                "case_groups": [record.case_group for record in records[1:]],
            },
        ],
        "records": [_entry(record, tmp_path / "videos") for record in records],
    }


def _config(*, bucket_id: str = "portrait_32x32"):
    return SimpleNamespace(
        num_samples=1,
        num_output_frames=64,
        i2v=True,
        image_or_video_shape=[1, 64, 48, 2, 2],
        continuation=SimpleNamespace(
            enabled=True,
            manifest_path="prepared_manifest.json",
            bucket_id=bucket_id,
        ),
    )


def _calls() -> dict[str, Any]:
    return {
        "encode": [],
        "randn": [],
        "sessions": [],
        "finish": [],
        "save_video": [],
        "clear_vae_cache": 0,
    }


def _patch_cpu_cuda_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest: dict[str, Any],
    records: list[ContinuationMetadataRecord],
    calls: dict[str, Any],
) -> None:
    real_isfinite = torch.isfinite

    monkeypatch.setattr(
        orchestration,
        "load_prepared_continuation_manifest",
        lambda _path: (manifest, records),
    )
    monkeypatch.setattr(
        orchestration,
        "load_stage1_input_image",
        lambda record: _CudaBackedByCpuTensor(
            torch.full((1, 3, record.height, record.width), record.row_id)
        ),
    )
    monkeypatch.setattr(orchestration, "tensor_identity_sha256", _identity)
    monkeypatch.setattr(orchestration.torch, "Generator", _FakeGenerator)

    def fake_randn(shape, *, generator, device, dtype):
        assert isinstance(generator, _FakeGenerator)
        assert generator.seed == 1
        assert str(device).startswith("cuda")
        assert dtype == torch.bfloat16
        calls["randn"].append(tuple(shape))
        value = torch.arange(
            int(torch.tensor(shape).prod().item()), dtype=torch.float32
        ).reshape(tuple(shape))
        return _CudaBackedByCpuTensor(value.to(dtype=dtype))

    monkeypatch.setattr(orchestration.torch, "randn", fake_randn)
    monkeypatch.setattr(
        orchestration.torch,
        "isfinite",
        lambda value: (
            real_isfinite(value.value)
            if isinstance(value, _CudaBackedByCpuTensor)
            else real_isfinite(value)
        ),
    )
    monkeypatch.setattr(orchestration.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(orchestration.torch.distributed, "is_available", lambda: False)

    def fake_save_video(video, path, *, fps):
        calls["save_video"].append(
            {"shape": tuple(video.shape), "path": Path(path), "fps": fps}
        )
        Path(path).write_bytes(b"synthetic-mp4")

    monkeypatch.setattr(orchestration, "save_video", fake_save_video)


def test_orchestration_reuses_case_inputs_and_noise_across_both_sinks(
    tmp_path, monkeypatch
):
    target = _record(tmp_path)
    # The loader returns another prepared row, but the chosen bucket must not run it.
    unused = _record(tmp_path, row_id=2, case_group="cat1_ba")
    records = [target, unused]
    manifest = _manifest(tmp_path, records, target_rows=[target.row_id])
    calls = _calls()
    _patch_cpu_cuda_boundary(
        monkeypatch, manifest=manifest, records=records, calls=calls
    )
    pipeline = _FakePipeline(calls, target)

    report = orchestration.run_prepared_continuation_bucket(
        pipeline,
        _config(),
        device=torch.device("cuda:0"),
    )

    assert report["bucket_id"] == "portrait_32x32"
    assert report["row_count"] == 1
    assert report["sample_count"] == 2
    assert [sample["sink_size"] for sample in report["samples"]] == [0, 1]
    assert {sample["case_group"] for sample in report["samples"]} == {"cat1_ab"}

    # The image VAE and seeded full 64-frame noise plan run exactly once per case.
    assert len(calls["encode"]) == 1
    assert calls["randn"] == [(1, 64, 48, 2, 2)]
    assert len(calls["sessions"]) == 2
    assert [session.sink_size for session in calls["sessions"]] == [0, 1]
    assert _identity(calls["sessions"][0].initial_latent) == _identity(
        calls["sessions"][1].initial_latent
    )
    assert _identity(calls["sessions"][0].noise_plan) == _identity(
        calls["sessions"][1].noise_plan
    )
    assert torch.equal(
        calls["sessions"][0].noise_plan.value,
        calls["sessions"][1].noise_plan.value,
    )

    expected_segments = [
        ("action_a", target.action_a_prompt, 0, 24, False),
        ("hold", target.hold_prompt, 24, 40, False),
        ("action_b", target.action_b_prompt, 40, 64, True),
    ]
    for session in calls["sessions"]:
        assert [
            (
                segment["name"],
                segment["prompt"],
                segment["start"],
                segment["end"],
                segment["carry"],
            )
            for segment in session.segments
        ] == expected_segments
        for segment in session.segments:
            assert torch.equal(
                segment["noise"].value,
                session.noise_plan.value[:, segment["start"] : segment["end"]],
            )

    # Each sink finishes one full 64-latent continuation and saves one 253-frame video.
    assert len(calls["finish"]) == 2
    assert [session.cursor for session in calls["finish"]] == [64, 64]
    assert [item["shape"] for item in calls["save_video"]] == [
        (1, 253, 3, 32, 32),
        (1, 253, 3, 32, 32),
    ]
    assert [item["fps"] for item in calls["save_video"]] == [24, 24]

    for sink_size in (0, 1):
        video_path = tmp_path / "videos" / target.case_group / f"sink{sink_size}.mp4"
        trace_path = (
            tmp_path / "videos" / target.case_group / f"sink{sink_size}.session.json"
        )
        assert video_path.read_bytes() == b"synthetic-mp4"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        assert trace["status"] == "finished"
        assert trace["metadata"]["row_id"] == target.row_id
        assert trace["metadata"]["case_group"] == target.case_group
        assert trace["sampling"]["sink_size"] == sink_size
        assert trace["result"]["latent_shape"] == [1, 64, 48, 2, 2]
        assert trace["result"]["pixel_shape"] == [1, 253, 3, 32, 32]
        assert trace["result"]["decode_calls"] == 1
        assert trace["result"]["decode_mode"] == "single_full_sequence"
        assert trace["output"]["output_video"] == str(video_path.resolve())
        assert trace["output"]["session_trace"] == str(trace_path.resolve())


def test_orchestration_failure_writes_partial_failed_trace_without_mp4(
    tmp_path, monkeypatch
):
    record = _record(tmp_path)
    manifest = _manifest(tmp_path, [record])
    calls = _calls()
    _patch_cpu_cuda_boundary(
        monkeypatch, manifest=manifest, records=[record], calls=calls
    )
    pipeline = _FakePipeline(calls, record, fail_on_segment="hold")

    with pytest.raises(RuntimeError, match="synthetic failure in hold"):
        orchestration.run_prepared_continuation_bucket(
            pipeline,
            _config(),
            device=torch.device("cuda:0"),
        )

    output_root = tmp_path / "videos" / record.case_group
    assert not (output_root / "sink0.mp4").exists()
    assert not (output_root / "sink1.mp4").exists()
    assert not (output_root / "sink1.session.json").exists()
    trace_path = output_root / "sink0.session.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["status"] == "failed"
    assert trace["failure"] == {
        "type": "RuntimeError",
        "message": "synthetic failure in hold",
    }
    assert trace["cursor_frames"] == 24
    assert [segment["name"] for segment in trace["segments"]] == [
        "action_a",
        "hold",
    ]
    assert trace["metadata"]["case_group"] == record.case_group
    assert trace["sampling"]["sink_size"] == 0
    assert calls["save_video"] == []
    assert calls["finish"] == []


def test_orchestration_rejects_cpu_before_loading_an_image(tmp_path, monkeypatch):
    record = _record(tmp_path)
    manifest = _manifest(tmp_path, [record])
    calls = _calls()
    monkeypatch.setattr(
        orchestration,
        "load_prepared_continuation_manifest",
        lambda _path: (manifest, [record]),
    )
    monkeypatch.setattr(
        orchestration,
        "load_stage1_input_image",
        lambda _record: pytest.fail("CPU rejection must precede image loading"),
    )

    with pytest.raises(ValueError, match="requires one CUDA GPU"):
        orchestration.run_prepared_continuation_bucket(
            _FakePipeline(calls, record),
            _config(),
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config, pipeline: setattr(config, "num_samples", 2), "num_samples=1"),
        (
            lambda config, pipeline: setattr(
                config, "image_or_video_shape", [1, 63, 48, 2, 2]
            ),
            "runtime shape mismatch",
        ),
        (
            lambda config, pipeline: setattr(pipeline, "sample_solver", "dpm++"),
            "sampling contract is not locked",
        ),
        (
            lambda config, pipeline: setattr(pipeline, "frame_seq_length", 2),
            "frame_seq_length differs",
        ),
    ],
)
def test_orchestration_rejects_unlocked_runtime_before_image_io(
    tmp_path, monkeypatch, mutation, message
):
    record = _record(tmp_path)
    manifest = _manifest(tmp_path, [record])
    config = _config()
    calls = _calls()
    pipeline = _FakePipeline(calls, record)
    mutation(config, pipeline)
    monkeypatch.setattr(
        orchestration,
        "load_prepared_continuation_manifest",
        lambda _path: (manifest, [record]),
    )
    monkeypatch.setattr(
        orchestration,
        "load_stage1_input_image",
        lambda _record: pytest.fail("runtime rejection must precede image loading"),
    )

    with pytest.raises(ValueError, match=message):
        orchestration.run_prepared_continuation_bucket(
            pipeline,
            config,
            device=torch.device("cuda:0"),
        )


def test_orchestration_requires_an_exact_prepared_bucket(tmp_path, monkeypatch):
    record = _record(tmp_path)
    manifest = _manifest(tmp_path, [record])
    calls = _calls()
    monkeypatch.setattr(
        orchestration,
        "load_prepared_continuation_manifest",
        lambda _path: (manifest, [record]),
    )

    with pytest.raises(ValueError, match="no unique bucket"):
        orchestration.run_prepared_continuation_bucket(
            _FakePipeline(calls, record),
            _config(bucket_id="not-prepared"),
            device=torch.device("cuda:0"),
        )
