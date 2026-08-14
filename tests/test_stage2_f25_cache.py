import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from safetensors.torch import load_file, save_file
import torch

from utils.stage1_i2v_data import (
    Stage1I2VRecord,
    decode_stage1_video,
    load_stage1_i2v_manifest,
)
from utils.stage1_io import (
    aggregate_file_hash,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
    tree_file_hashes,
)
from utils.stage2_f25_cache import (
    _input_kind,
    _safe_source_artifact,
    prepare_stage2_f25_cache,
    stable_f25_row_ids,
)
from utils.stage2_i2v_data import (
    STAGE2_F25_SOURCE_CACHE_SCHEMA,
    STAGE2_TEXT_ENCODING_UPGRADE_KEY,
    STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
    load_source_cache_manifest,
    upgrade_legacy_source_cache_manifest_text_encoding,
)


def _video(frames: int, *, h: int = 30, w: int = 52, row_id: int = 0):
    values = torch.empty(frames, 48, h, w, dtype=torch.bfloat16)
    for frame in range(frames):
        values[frame].fill_(row_id * 32 + frame)
    return values


def _tensors(frames: int, *, row_id: int = 0):
    prompt = torch.zeros(512, 4096, dtype=torch.bfloat16)
    prompt[0, 0] = row_id + 1
    mask = torch.zeros(512, dtype=torch.bool)
    mask[:2] = True
    return {
        "video_latent": _video(frames, row_id=row_id),
        "initial_latent": torch.full(
            (1, 48, 30, 52), row_id + 0.5, dtype=torch.bfloat16
        ),
        "prompt_embeds": prompt,
        "prompt_mask": mask,
    }


def _shape_dtype(tensors):
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        for name, tensor in sorted(tensors.items())
    }


def _tree_hash(path: Path) -> str:
    return aggregate_file_hash(tree_file_hashes(path))


def _fixture(tmp_path: Path, frame_counts, *, proven_f25: bool):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "stage1"
    source.mkdir()
    videos = tmp_path / "videos"
    images = tmp_path / "images"
    videos.mkdir()
    images.mkdir()
    metadata = tmp_path / "metadata.csv"
    fields = ["video", "prompt", "input_image", "height", "width", "bucket"]
    rows = []
    for row_id in range(len(frame_counts)):
        video = videos / f"{row_id}.mp4"
        image = images / f"{row_id}.png"
        video.write_bytes(f"video-{row_id}".encode())
        Image.new("RGB", (832, 480), (row_id, 0, 0)).save(image)
        rows.append(
            {
                "video": str(video),
                "prompt": f"cat {row_id}",
                "input_image": str(image),
                "height": "480",
                "width": "832",
                "bucket": "landscape",
            }
        )
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    records = load_stage1_i2v_manifest(metadata, expected_num_samples=len(rows))

    vae = tmp_path / "vae.bin"
    t5 = tmp_path / "t5.bin"
    tokenizer = tmp_path / "tokenizer"
    vae.write_bytes(b"vae")
    t5.write_bytes(b"t5")
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    entries = []
    for record, frames in zip(records, frame_counts):
        tensors = _tensors(frames, row_id=record.row_id)
        artifact = source / f"sample_{record.row_id:06d}.safetensors"
        save_file(tensors, str(artifact))
        entries.append(
            {
                "row_id": record.row_id,
                "row_sha256": record.row_sha256,
                "height": record.height,
                "width": record.width,
                "bucket": record.bucket,
                "path": artifact.name,
                "size": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
                "tensors": _shape_dtype(tensors),
            }
        )
    preprocessing = {
        "min_source_frames": 97,
        "selected_frame_start": 0,
        "selected_frame_count": 97 if proven_f25 else 93,
        "expected_fps": 24,
        "allow_resize": False,
        "allow_padding": False,
        "dtype": "bfloat16",
        "prompt_mode": "repeat_global",
    }
    fingerprint = {
        "records": [
            {
                "row_id": record.row_id,
                "row_sha256": record.row_sha256,
                "video_sha256": sha256_file(record.video_path),
                "input_image_sha256": sha256_file(record.input_image_path),
            }
            for record in records
        ],
        "models": {
            "vae_checkpoint": {
                "files": tree_file_hashes(vae),
                "aggregate_sha256": _tree_hash(vae),
            },
            "t5_checkpoint": {
                "files": tree_file_hashes(t5),
                "aggregate_sha256": _tree_hash(t5),
            },
            "tokenizer_dir": {
                "files": tree_file_hashes(tokenizer),
                "aggregate_sha256": _tree_hash(tokenizer),
            },
        },
        "preprocessing": preprocessing,
        "cache_schema_version": 1,
    }
    fingerprint["aggregate_sha256"] = canonical_json_sha256(fingerprint)
    for record, entry in zip(records, entries):
        artifact = source / entry["path"]
        tensors = load_file(str(artifact))
        save_file(
            tensors,
            str(artifact),
            metadata={
                "schema": "longlive_stage1_i2v_cache_record",
                "schema_version": "1",
                "row_id": str(record.row_id),
                "row_sha256": record.row_sha256,
                "source_aggregate_sha256": fingerprint["aggregate_sha256"],
                "prompt_mode": "repeat_global",
            },
        )
        entry["size"] = artifact.stat().st_size
        entry["sha256"] = sha256_file(artifact)
    manifest = {
        "schema": "longlive_stage1_i2v_cache",
        "schema_version": 1,
        "num_samples": len(rows),
        "source_fingerprint": fingerprint,
        "records": entries,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    source_manifest = source / "cache_manifest.json"
    atomic_write_json(source_manifest, manifest)
    config = tmp_path / "stage2.yaml"
    config.write_text("config_schema: fixture\n", encoding="utf-8")
    return {
        "metadata": metadata,
        "records": records,
        "source": source,
        "source_manifest": source_manifest,
        "vae": vae,
        "t5": t5,
        "tokenizer": tokenizer,
        "config": config,
        "entries": entries,
    }


class _FakeVAE:
    def __init__(self, calls):
        self.calls = calls

    def encode_to_latent(self, pixels):
        self.calls.append(tuple(pixels.shape))
        row_id = int(pixels[0, 0, 0, 0, 0].item())
        return _video(25, row_id=row_id).unsqueeze(0)


def _decoder(record, **kwargs):
    assert kwargs == {
        "min_source_frames": 97,
        "selected_frame_start": 0,
        "selected_frame_count": 97,
        "expected_fps": 24.0,
        "fps_abs_tolerance": 1.0e-3,
    }
    return torch.full((1, 3, 97, 1, 1), record.row_id, dtype=torch.float32)


def _prepare(fixture, tmp_path, calls, *, output_name="f25"):
    return prepare_stage2_f25_cache(
        metadata_path=fixture["metadata"],
        source_cache_manifest_path=fixture["source_manifest"],
        output_dir=tmp_path / output_name,
        config_path=fixture["config"],
        config_contract_sha256="1" * 64,
        config_launch_sha256="2" * 64,
        expected_num_samples=len(fixture["records"]),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        vae_checkpoint_path=fixture["vae"],
        vae_factory=lambda _path, _device: _FakeVAE(calls),
        decode_video=_decoder,
    )


def test_proven_f25_is_byte_reused_without_vae_or_decode(tmp_path):
    fixture = _fixture(tmp_path, [25], proven_f25=True)
    calls = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("F25 reuse must not decode or load VAE")

    manifest_path = prepare_stage2_f25_cache(
        metadata_path=fixture["metadata"],
        source_cache_manifest_path=fixture["source_manifest"],
        output_dir=tmp_path / "f25",
        config_path=fixture["config"],
        config_contract_sha256="1" * 64,
        config_launch_sha256="2" * 64,
        expected_num_samples=1,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        vae_factory=forbidden,
        decode_video=forbidden,
    )
    manifest = json.loads(manifest_path.read_text())
    output = manifest_path.parent / manifest["records"][0]["path"]
    source = fixture["source"] / fixture["entries"][0]["path"]
    assert output.read_bytes() == source.read_bytes()
    assert manifest["records"][0]["decision"] == "reused_f25"
    assert manifest["preparation"]["summary"] == {
        "reencoded_f24": 0,
        "reused_f25": 1,
        "reverified_f25": 0,
    }
    assert calls == []


def test_mixed_f25_reuse_and_f24_reencode_once_with_prefix_parity(tmp_path, capsys):
    fixture = _fixture(tmp_path, [25, 24], proven_f25=True)
    calls = []
    manifest_path = _prepare(fixture, tmp_path, calls)
    manifest = json.loads(manifest_path.read_text())
    assert [entry["decision"] for entry in manifest["records"]] == [
        "reused_f25",
        "reencoded_f24",
    ]
    assert calls == [(1, 3, 97, 1, 1)]
    old_f24 = load_file(str(fixture["source"] / fixture["entries"][1]["path"]))
    new_f25 = load_file(str(manifest_path.parent / manifest["records"][1]["path"]))
    assert torch.equal(new_f25["video_latent"][:24], old_f24["video_latent"])
    for name in ("initial_latent", "prompt_embeds", "prompt_mask"):
        assert torch.equal(new_f25[name], old_f24[name])
    progress = capsys.readouterr().out
    assert "source_scan=2/2 row_id=1" in progress
    assert "completed=2/2 row_id=1 decision=reencoded_f24" in progress
    assert "final_verify=2/2 row_id=1" in progress


def test_unproven_f25_is_fully_reverified_then_original_bytes_reused(tmp_path):
    fixture = _fixture(tmp_path, [25], proven_f25=False)
    calls = []
    manifest_path = _prepare(fixture, tmp_path, calls)
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["records"][0]
    assert entry["decision"] == "reverified_f25"
    assert calls == [(1, 3, 97, 1, 1)]
    assert (manifest_path.parent / entry["path"]).read_bytes() == (
        fixture["source"] / fixture["entries"][0]["path"]
    ).read_bytes()


def test_f25_with_missing_old_preprocessing_policy_is_reverified(tmp_path):
    fixture = _fixture(tmp_path, [25], proven_f25=True)
    manifest = json.loads(fixture["source_manifest"].read_text())
    manifest.pop("manifest_sha256")
    fingerprint = manifest["source_fingerprint"]
    fingerprint.pop("aggregate_sha256")
    fingerprint.pop("preprocessing")
    fingerprint["aggregate_sha256"] = canonical_json_sha256(fingerprint)

    entry = manifest["records"][0]
    source_artifact = fixture["source"] / entry["path"]
    tensors = load_file(str(source_artifact))
    save_file(
        tensors,
        str(source_artifact),
        metadata={
            "schema": "longlive_stage1_i2v_cache_record",
            "schema_version": "1",
            "row_id": "0",
            "row_sha256": entry["row_sha256"],
            "source_aggregate_sha256": fingerprint["aggregate_sha256"],
            "prompt_mode": "repeat_global",
        },
    )
    entry["size"] = source_artifact.stat().st_size
    entry["sha256"] = sha256_file(source_artifact)
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(fixture["source_manifest"], manifest)

    calls = []
    output_manifest = _prepare(fixture, tmp_path, calls)
    output = json.loads(output_manifest.read_text())
    assert output["records"][0]["decision"] == "reverified_f25"
    assert calls == [(1, 3, 97, 1, 1)]
    assert (output_manifest.parent / output["records"][0]["path"]).read_bytes() == (
        source_artifact.read_bytes()
    )


def test_reverification_or_f24_prefix_mismatch_publishes_nothing(tmp_path):
    fixture = _fixture(tmp_path, [25], proven_f25=False)

    class WrongVAE:
        def encode_to_latent(self, _pixels):
            return torch.full((1, 25, 48, 30, 52), -7, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="differs bitwise"):
        prepare_stage2_f25_cache(
            metadata_path=fixture["metadata"],
            source_cache_manifest_path=fixture["source_manifest"],
            output_dir=tmp_path / "bad",
            config_path=fixture["config"],
            config_contract_sha256="1" * 64,
            config_launch_sha256="2" * 64,
            expected_num_samples=1,
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
            vae_checkpoint_path=fixture["vae"],
            vae_factory=lambda *_args: WrongVAE(),
            decode_video=_decoder,
        )
    assert not (tmp_path / "bad" / "sample_000000.safetensors").exists()
    assert not (tmp_path / "bad" / "sample_000000.complete.json").exists()


def test_f24_prefix_mismatch_does_not_publish(tmp_path):
    fixture = _fixture(tmp_path, [24], proven_f25=True)

    class WrongPrefixVAE:
        def encode_to_latent(self, _pixels):
            return torch.full((1, 25, 48, 30, 52), -3, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match=r"F25\[:24\] differs bitwise"):
        prepare_stage2_f25_cache(
            metadata_path=fixture["metadata"],
            source_cache_manifest_path=fixture["source_manifest"],
            output_dir=tmp_path / "bad-prefix",
            config_path=fixture["config"],
            config_contract_sha256="1" * 64,
            config_launch_sha256="2" * 64,
            expected_num_samples=1,
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
            vae_checkpoint_path=fixture["vae"],
            vae_factory=lambda *_args: WrongPrefixVAE(),
            decode_video=_decoder,
        )
    assert not (tmp_path / "bad-prefix" / "sample_000000.safetensors").exists()


@pytest.mark.parametrize("latent_frames", [24, 26])
def test_vae_must_return_exact_f25(tmp_path, latent_frames):
    fixture = _fixture(tmp_path, [24], proven_f25=True)

    class WrongFramesVAE:
        def encode_to_latent(self, _pixels):
            return torch.zeros(1, latent_frames, 48, 30, 52, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match=r"must return \[1,25,48,H,W\]"):
        prepare_stage2_f25_cache(
            metadata_path=fixture["metadata"],
            source_cache_manifest_path=fixture["source_manifest"],
            output_dir=tmp_path / f"bad-{latent_frames}",
            config_path=fixture["config"],
            config_contract_sha256="1" * 64,
            config_launch_sha256="2" * 64,
            expected_num_samples=1,
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
            vae_checkpoint_path=fixture["vae"],
            vae_factory=lambda *_args: WrongFramesVAE(),
            decode_video=_decoder,
        )


def test_decoded_pixel_window_must_be_exactly_97(tmp_path):
    fixture = _fixture(tmp_path, [24], proven_f25=True)
    with pytest.raises(RuntimeError, match=r"must be \[1,3,97,H,W\]"):
        prepare_stage2_f25_cache(
            metadata_path=fixture["metadata"],
            source_cache_manifest_path=fixture["source_manifest"],
            output_dir=tmp_path / "bad-pixels",
            config_path=fixture["config"],
            config_contract_sha256="1" * 64,
            config_launch_sha256="2" * 64,
            expected_num_samples=1,
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
            vae_checkpoint_path=fixture["vae"],
            vae_factory=lambda *_args: _FakeVAE([]),
            decode_video=lambda *_args, **_kwargs: torch.zeros(1, 3, 96, 1, 1),
        )


def test_resume_detects_tamper_and_rebuilds_owned_row(tmp_path):
    fixture = _fixture(tmp_path, [24], proven_f25=True)
    first_calls = []
    manifest_path = _prepare(fixture, tmp_path, first_calls)
    manifest = json.loads(manifest_path.read_text())
    artifact = manifest_path.parent / manifest["records"][0]["path"]
    artifact.write_bytes(b"tampered")
    second_calls = []
    _prepare(fixture, tmp_path, second_calls)
    assert second_calls == [(1, 3, 97, 1, 1)]
    assert load_file(str(artifact))["video_latent"].shape[0] == 25


@pytest.mark.parametrize("metadata_mode", ["missing", "wrong-source"])
def test_source_safetensors_metadata_is_mandatory_and_bound(tmp_path, metadata_mode):
    fixture = _fixture(tmp_path, [25], proven_f25=True)
    source_manifest = json.loads(fixture["source_manifest"].read_text())
    artifact = fixture["source"] / source_manifest["records"][0]["path"]
    tensors = load_file(str(artifact))
    metadata = None
    if metadata_mode == "wrong-source":
        metadata = {
            "schema": "longlive_stage1_i2v_cache_record",
            "schema_version": "1",
            "row_id": "0",
            "row_sha256": source_manifest["records"][0]["row_sha256"],
            "source_aggregate_sha256": "f" * 64,
            "prompt_mode": "repeat_global",
        }
    save_file(tensors, str(artifact), metadata=metadata)
    source_manifest.pop("manifest_sha256")
    source_manifest["records"][0]["size"] = artifact.stat().st_size
    source_manifest["records"][0]["sha256"] = sha256_file(artifact)
    source_manifest["manifest_sha256"] = canonical_json_sha256(source_manifest)
    atomic_write_json(fixture["source_manifest"], source_manifest)
    with pytest.raises(
        RuntimeError,
        match=r"(no cache provenance metadata|metadata source_aggregate_sha256 mismatch)",
    ):
        _prepare(fixture, tmp_path, [])


def test_source_model_file_list_aggregate_is_verified(tmp_path):
    fixture = _fixture(tmp_path, [25], proven_f25=True)
    manifest = json.loads(fixture["source_manifest"].read_text())
    manifest.pop("manifest_sha256")
    fingerprint = manifest["source_fingerprint"]
    fingerprint.pop("aggregate_sha256")
    fingerprint["models"]["vae_checkpoint"]["files"][0]["size"] += 1
    fingerprint["aggregate_sha256"] = canonical_json_sha256(fingerprint)
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(fixture["source_manifest"], manifest)
    with pytest.raises(RuntimeError, match="file-list aggregate hash mismatch"):
        _prepare(fixture, tmp_path, [])


def test_native_manifest_upgrades_outside_cache_and_loader_closes_chain(
    tmp_path, monkeypatch
):
    fixture = _fixture(tmp_path, [25, 24], proven_f25=True)
    base = _prepare(fixture, tmp_path, [])
    base_loaded = load_source_cache_manifest(base, expected_num_samples=2)
    assert STAGE2_TEXT_ENCODING_UPGRADE_KEY not in base_loaded
    assert "text_encoding" not in base_loaded["source_fingerprint"]
    monkeypatch.setattr(
        "utils.wan_5b_wrapper.audit_wan_text_encoding_tokenizer_contract",
        lambda _path: {
            "cleaning": "whitespace",
            "add_special_tokens": True,
            "sequence_length": 512,
            "padding_side": "right",
            "embedding_padding_value": 0.0,
            "validated_special_token_growth": True,
            "validated_right_padding_mask": True,
        },
    )
    external = tmp_path / "attested" / "cache_manifest.attested.json"
    external.parent.mkdir()
    base_value = json.loads(base.read_text())
    upgrade_legacy_source_cache_manifest_text_encoding(
        base,
        external,
        expected_source_manifest_sha256=base_value["manifest_sha256"],
        t5_checkpoint_path=fixture["t5"],
        tokenizer_dir=fixture["tokenizer"],
        operator_id="test",
        operator_attestation=STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
        expected_num_samples=2,
    )
    loaded = load_source_cache_manifest(
        external,
        expected_num_samples=2,
        require_text_encoding_upgrade=True,
    )
    assert loaded["schema"] == STAGE2_F25_SOURCE_CACHE_SCHEMA


def test_escape_path_fails_without_parent_loop(tmp_path):
    with pytest.raises(RuntimeError, match="must be relative"):
        _safe_source_artifact(tmp_path, "../escape.safetensors", row_id=0)


@pytest.mark.parametrize("relation", ["child", "parent"])
def test_output_directory_cannot_nest_with_stage1_source(tmp_path, relation):
    fixture = _fixture(tmp_path / "fixture", [25], proven_f25=True)
    output = (
        fixture["source"] / "nested-f25"
        if relation == "child"
        else fixture["source"].parent
    )
    with pytest.raises(RuntimeError, match="independent non-nested"):
        prepare_stage2_f25_cache(
            metadata_path=fixture["metadata"],
            source_cache_manifest_path=fixture["source_manifest"],
            output_dir=output,
            config_path=fixture["config"],
            config_contract_sha256="1" * 64,
            config_launch_sha256="2" * 64,
            expected_num_samples=1,
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )


def test_lightweight_600_row_decisions_and_modulo_shards_cover_exactly_once():
    row_ids = list(range(600))
    assignments = [
        stable_f25_row_ids(600, rank=rank, world_size=8) for rank in range(8)
    ]
    flattened = [row_id for shard in assignments for row_id in shard]
    assert sorted(flattened) == row_ids
    assert len(flattened) == len(set(flattened)) == 600
    decisions = [
        "reused_f25" if row_id % 3 == 0 else "reencoded_f24" for row_id in row_ids
    ]
    assert decisions.count("reused_f25") == 200
    assert decisions.count("reencoded_f24") == 400


def test_input_kind_rejects_shape_and_dtype_without_padding_or_truncation():
    manifest = {
        "schema": "longlive_stage1_i2v_cache",
        "source_fingerprint": {
            "preprocessing": {
                "min_source_frames": 97,
                "selected_frame_start": 0,
                "selected_frame_count": 97,
                "expected_fps": 24,
                "allow_resize": False,
                "allow_padding": False,
                "dtype": "bfloat16",
            }
        },
    }
    bad = _tensors(23)
    with pytest.raises(RuntimeError, match="never pads, truncates, or duplicates"):
        _input_kind(
            bad,
            source_manifest=manifest,
            expected_spatial_shape=(30, 52),
            row_id=0,
        )
    wrong_dtype = _tensors(25)
    wrong_dtype["video_latent"] = wrong_dtype["video_latent"].float()
    with pytest.raises(ValueError, match="bfloat16"):
        _input_kind(
            wrong_dtype,
            source_manifest=manifest,
            expected_spatial_shape=(30, 52),
            row_id=0,
        )


class _FakeFrame:
    def __init__(self, height, width, value):
        self.height = height
        self.width = width
        self.value = value

    def to_ndarray(self, *, format):
        assert format == "rgb24"
        return np.full((self.height, self.width, 3), self.value, dtype=np.uint8)


class _FakeContainer:
    def __init__(self, record, *, frames, fps=24.0, rotation=0.0):
        self.stream = SimpleNamespace(average_rate=fps, metadata={}, rotation=rotation)
        self.streams = SimpleNamespace(video=[self.stream])
        self.frames = [
            _FakeFrame(record.height, record.width, value=index % 256)
            for index in range(frames)
        ]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def decode(self, stream):
        assert stream is self.stream
        return iter(self.frames)


def _small_decoder_record(tmp_path):
    video = tmp_path / "tiny.mp4"
    video.write_bytes(b"video")
    return Stage1I2VRecord(
        row_id=0,
        video_path=video,
        prompt="cat",
        input_image_path=tmp_path / "unused.png",
        height=2,
        width=4,
        bucket="landscape",
        canonical_row={"video": str(video)},
        row_sha256="unused",
    )


def test_decoder_accepts_longer_source_but_selects_only_zero_through_96(
    tmp_path, monkeypatch
):
    record = _small_decoder_record(tmp_path)
    container = _FakeContainer(record, frames=98)
    monkeypatch.setitem(
        sys.modules, "av", SimpleNamespace(open=lambda *_args, **_kwargs: container)
    )
    pixels = decode_stage1_video(
        record,
        min_source_frames=97,
        selected_frame_start=0,
        selected_frame_count=97,
        expected_fps=24.0,
    )
    assert pixels.shape == (1, 3, 97, 2, 4)
    assert pixels[0, 0, -1, 0, 0].item() == pytest.approx(96 / 127.5 - 1)


@pytest.mark.parametrize(
    ("frames", "fps", "rotation", "match"),
    [
        (96, 24.0, 0.0, "only 96 decoded frames"),
        (97, 25.0, 0.0, "expected 24.0 fps"),
        (97, 24.0, 90.0, "rotation metadata"),
    ],
)
def test_decoder_rejects_short_wrong_fps_or_rotation(
    tmp_path, monkeypatch, frames, fps, rotation, match
):
    record = _small_decoder_record(tmp_path)
    container = _FakeContainer(record, frames=frames, fps=fps, rotation=rotation)
    monkeypatch.setitem(
        sys.modules, "av", SimpleNamespace(open=lambda *_args, **_kwargs: container)
    )
    with pytest.raises(ValueError, match=match):
        decode_stage1_video(
            record,
            min_source_frames=97,
            selected_frame_start=0,
            selected_frame_count=97,
            expected_fps=24.0,
        )
