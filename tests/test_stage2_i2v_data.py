import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from utils.config import DEFAULT_NEGATIVE_PROMPT
from utils.stage1_i2v_data import (
    STAGE1_CACHE_SCHEMA_VERSION,
    build_source_fingerprint,
    load_stage1_i2v_manifest,
    validate_cache_tensors as validate_stage1_cache_tensors,
)
from utils.stage1_io import (
    aggregate_file_hash,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
    tree_file_hashes,
)
from utils.stage2_i2v_data import (
    STAGE2_CACHE_TENSOR_SCHEMA_SHA256,
    STAGE2_NEGATIVE_PROMPT_SHA256,
    STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
    Stage2I2VCacheDataset,
    audit_stage2_i2v_cache,
    load_negative_conditioning,
    load_source_cache_manifest,
    load_stage2_i2v_manifest,
    save_negative_conditioning_artifact,
    stage2_i2v_cache_collate,
    upgrade_legacy_source_cache_manifest_text_encoding,
    validate_stage2_cache_tensors,
    write_negative_conditioning_manifest,
)
from utils.stage2_f25_cache import prepare_stage2_f25_cache

ACTIONS = ("head_tilt", "jump", "toy_play")
CONFIG_CONTRACT_HASH = "3" * 64
CONFIG_LAUNCH_HASH = "4" * 64
UPGRADE_CODE_VERSION = "git:" + "a" * 40


@pytest.fixture(autouse=True)
def _trusted_clean_repo_code_version(monkeypatch):
    monkeypatch.setattr(
        "utils.stage2_i2v_data._resolve_clean_repo_code_version",
        lambda: UPGRADE_CODE_VERSION,
    )
    monkeypatch.setattr(
        "utils.stage2_f25_cache._resolve_clean_repo_code_version",
        lambda: UPGRADE_CODE_VERSION,
    )


def _locked_tokenizer_runtime_audit():
    return {
        "cleaning": "whitespace",
        "add_special_tokens": True,
        "sequence_length": 512,
        "padding_side": "right",
        "embedding_padding_value": 0.0,
        "validated_special_token_growth": True,
        "validated_right_padding_mask": True,
    }


def _cache_tensors(*, h=30, w=52, frames=25, channels=48, dtype=torch.bfloat16):
    video = torch.zeros(frames, channels, h, w, dtype=dtype)
    if frames == 25 and channels == 48 and dtype == torch.bfloat16:
        for frame in range(frames):
            video[frame].fill_(frame)
    prompt = torch.zeros(512, 4096, dtype=torch.bfloat16)
    prompt[:3, 0] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    mask = torch.zeros(512, dtype=torch.bool)
    mask[:3] = True
    return {
        "video_latent": video,
        "initial_latent": torch.full((1, 48, h, w), 99.0, dtype=torch.bfloat16),
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


def _rehash_manifest(path: Path, mutate):
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("manifest_sha256")
    mutate(value)
    value["manifest_sha256"] = canonical_json_sha256(value)
    atomic_write_json(path, value)


def _rehash_source_fingerprint(value):
    fingerprint = value["source_fingerprint"]
    fingerprint.pop("aggregate_sha256", None)
    fingerprint["aggregate_sha256"] = canonical_json_sha256(fingerprint)


def _remove_upgrade_for_nonformal_audit(fixture):
    _rehash_manifest(
        fixture["source"],
        lambda manifest: manifest.pop("stage2_text_encoding_upgrade"),
    )


def _build_fixture(root: Path, *, source_actions=False):
    root.mkdir(parents=True, exist_ok=True)
    input_cache_dir = root / "stage1_cache"
    input_cache_dir.mkdir()
    cache_dir = root / "cache"
    cache_dir.mkdir()
    metadata_path = root / "metadata.csv"
    fields = ["video", "prompt", "input_image", "height", "width", "bucket"]
    rows = []
    labels = []
    for row_id in range(6):
        landscape = row_id % 2 == 0
        height, width = (480, 832) if landscape else (832, 480)
        rows.append(
            {
                "video": f"videos/{row_id}.mp4",
                "prompt": f"cat action prompt {row_id}",
                "input_image": f"images/{row_id}.png",
                "height": str(height),
                "width": str(width),
                "bucket": "landscape" if landscape else "portrait",
            }
        )
        labels.append(ACTIONS[row_id % 3])
        video_path = root / rows[-1]["video"]
        image_path = root / rows[-1]["input_image"]
        video_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"fixture-video-{row_id}".encode())
        Image.new("RGB", (width, height), (row_id, 0, 0)).save(image_path)
    with metadata_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    records = load_stage1_i2v_manifest(
        metadata_path,
        expected_num_samples=6,
        require_files=True,
        validate_images=True,
    )

    model_root = root / "models"
    tokenizer_dir = model_root / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    t5_checkpoint = model_root / "t5.bin"
    vae_checkpoint = model_root / "vae.bin"
    t5_checkpoint.write_bytes(b"fixture-t5")
    vae_checkpoint.write_bytes(b"fixture-vae")
    (tokenizer_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    t5_files = tree_file_hashes(t5_checkpoint)
    tokenizer_files = tree_file_hashes(tokenizer_dir)
    vae_files = tree_file_hashes(vae_checkpoint)
    t5_hash = aggregate_file_hash(t5_files)
    tokenizer_hash = aggregate_file_hash(tokenizer_files)
    vae_hash = aggregate_file_hash(vae_files)

    source_entries = []
    for record in records:
        h, w = (record.height // 16, record.width // 16)
        tensors = _cache_tensors(h=h, w=w)
        validate_stage2_cache_tensors(tensors)
        artifact_path = input_cache_dir / f"sample_{record.row_id:06d}.safetensors"
        save_file(tensors, str(artifact_path))
        entry = {
            "row_id": record.row_id,
            "row_sha256": record.row_sha256,
            "height": record.height,
            "width": record.width,
            "bucket": record.bucket,
            "path": artifact_path.name,
            "size": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
            "tensors": _shape_dtype(tensors),
        }
        if source_actions:
            entry["action_id"] = labels[record.row_id]
        source_entries.append(entry)
    legacy_fingerprint = {
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
            "t5_checkpoint": {
                "files": t5_files,
                "aggregate_sha256": t5_hash,
            },
            "tokenizer_dir": {
                "files": tokenizer_files,
                "aggregate_sha256": tokenizer_hash,
            },
            "vae_checkpoint": {
                "files": vae_files,
                "aggregate_sha256": vae_hash,
            },
        },
        "preprocessing": {
            "min_source_frames": 97,
            "selected_frame_start": 0,
            "selected_frame_count": 97,
            "expected_fps": 24,
            "allow_resize": False,
            "allow_padding": False,
            "dtype": "bfloat16",
            "prompt_mode": "repeat_global",
        },
        "cache_schema_version": STAGE1_CACHE_SCHEMA_VERSION,
        "aggregate_sha256": "pending",
    }
    legacy_fingerprint["aggregate_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in legacy_fingerprint.items()
            if key != "aggregate_sha256"
        }
    )
    for record, entry in zip(records, source_entries):
        artifact_path = input_cache_dir / entry["path"]
        tensors = load_file(str(artifact_path))
        save_file(
            tensors,
            str(artifact_path),
            metadata={
                "schema": "longlive_stage1_i2v_cache_record",
                "schema_version": "1",
                "row_id": str(record.row_id),
                "row_sha256": record.row_sha256,
                "source_aggregate_sha256": legacy_fingerprint["aggregate_sha256"],
                "prompt_mode": "repeat_global",
            },
        )
        entry["size"] = artifact_path.stat().st_size
        entry["sha256"] = sha256_file(artifact_path)
    legacy_manifest = {
        "schema": "longlive_stage1_i2v_cache",
        "schema_version": STAGE1_CACHE_SCHEMA_VERSION,
        "num_samples": 6,
        "source_fingerprint": legacy_fingerprint,
        "records": source_entries,
    }
    legacy_manifest["manifest_sha256"] = canonical_json_sha256(legacy_manifest)
    legacy_path = input_cache_dir / "cache_manifest.json"
    atomic_write_json(legacy_path, legacy_manifest)

    config_path = root / "stage2_fixture.yaml"
    config_path.write_text("config_schema: fixture\n", encoding="utf-8")
    base_path = prepare_stage2_f25_cache(
        metadata_path=metadata_path,
        source_cache_manifest_path=legacy_path,
        output_dir=cache_dir,
        config_path=config_path,
        config_contract_sha256=CONFIG_CONTRACT_HASH,
        config_launch_sha256=CONFIG_LAUNCH_HASH,
        expected_num_samples=6,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    assert base_path is not None
    base_manifest = json.loads(base_path.read_text(encoding="utf-8"))

    text_encoding = {
        "t5_checkpoint_aggregate_sha256": t5_hash,
        "tokenizer_aggregate_sha256": tokenizer_hash,
        "tokenizer_revision": f"local-tree-sha256:{tokenizer_hash}",
        "cleaning": "whitespace",
        "add_special_tokens": True,
        "sequence_length": 512,
        "padding_side": "right",
        "embedding_padding_value": 0.0,
    }
    source_manifest = json.loads(json.dumps(base_manifest))
    source_manifest.pop("manifest_sha256")
    source_manifest["source_fingerprint"].pop("aggregate_sha256")
    source_manifest["source_fingerprint"]["text_encoding"] = text_encoding
    _rehash_source_fingerprint(source_manifest)
    validator_path = Path(__file__).resolve().parents[1] / "utils/wan_5b_wrapper.py"
    source_manifest["stage2_text_encoding_upgrade"] = {
        "schema": "longlive_stage2_text_encoding_upgrade",
        "schema_version": 1,
        "original_source_manifest": {
            "path": str(base_path.resolve()),
            "manifest_sha256": base_manifest["manifest_sha256"],
            "source_fingerprint_sha256": base_manifest["source_fingerprint"][
                "aggregate_sha256"
            ],
        },
        "verification": {
            "code_version": UPGRADE_CODE_VERSION,
            "validator_file": "utils/wan_5b_wrapper.py",
            "validator_file_sha256": sha256_file(validator_path),
            "text_encoding_contract_sha256": canonical_json_sha256(text_encoding),
            "t5_checkpoint_aggregate_sha256": t5_hash,
            "tokenizer_aggregate_sha256": tokenizer_hash,
            "tokenizer_runtime_audit": _locked_tokenizer_runtime_audit(),
        },
        "operator_attestation": {
            "operator_id": "fixture-operator",
            "statement": STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
        },
    }
    source_manifest["manifest_sha256"] = canonical_json_sha256(source_manifest)
    source_path = cache_dir / "cache_manifest.attested.json"
    atomic_write_json(source_path, source_manifest)

    sidecar_path = root / "actions.csv"
    with sidecar_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "action_id"])
        writer.writeheader()
        for row, label in zip(rows, labels):
            writer.writerow({"video": row["video"], "action_id": label})

    negative_dir = root / "negative"
    negative_dir.mkdir()
    negative_embeds = torch.zeros(512, 4096, dtype=torch.bfloat16)
    negative_embeds[:4, 0] = 1
    negative_mask = torch.zeros(512, dtype=torch.bool)
    negative_mask[:4] = True
    negative_artifact = negative_dir / "negative_conditioning.safetensors"
    save_negative_conditioning_artifact(
        negative_artifact,
        prompt_embeds=negative_embeds,
        prompt_mask=negative_mask,
    )
    negative_manifest = negative_dir / "negative_conditioning_manifest.json"
    write_negative_conditioning_manifest(
        negative_manifest,
        artifact_path=negative_artifact,
        source_cache_manifest_path=source_path,
        expected_num_samples=6,
    )
    return {
        "cache_dir": cache_dir,
        "metadata": metadata_path,
        "source": source_path,
        "legacy_source": legacy_path,
        "base_source": base_path,
        "sidecar": sidecar_path,
        "negative": negative_manifest,
        "labels": labels,
        "t5_checkpoint": t5_checkpoint,
        "tokenizer_dir": tokenizer_dir,
        "vae_checkpoint": vae_checkpoint,
    }


def _build_official_legacy_upgrade_fixture(root: Path):
    """Build a legacy manifest using the real Stage-1 fingerprint producer."""

    fixture = _build_fixture(root)
    for row_id in range(6):
        video = root / "videos" / f"{row_id}.mp4"
        image = root / "images" / f"{row_id}.png"
        video.parent.mkdir(parents=True, exist_ok=True)
        image.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"official-video-{row_id}".encode())
        image.write_bytes(f"official-image-{row_id}".encode())

    model_root = root / "official_models"
    t5_checkpoint = model_root / "t5.pth"
    tokenizer_dir = model_root / "tokenizer"
    vae_checkpoint = model_root / "vae.pth"
    tokenizer_dir.mkdir(parents=True)
    t5_checkpoint.write_bytes(b"official-stage1-t5-tree")
    vae_checkpoint.write_bytes(b"official-stage1-vae-tree")
    (tokenizer_dir / "tokenizer.json").write_text(
        '{"fixture":"official-stage1-tokenizer-tree"}\n', encoding="utf-8"
    )
    records = load_stage1_i2v_manifest(
        fixture["metadata"],
        expected_num_samples=6,
        require_files=True,
        validate_images=False,
    )
    fingerprint = build_source_fingerprint(
        records,
        model_paths={
            "vae_checkpoint": vae_checkpoint,
            "t5_checkpoint": t5_checkpoint,
            "tokenizer_dir": tokenizer_dir,
        },
        preprocessing={"fixture": "official-stage1-producer"},
    )
    assert set(fingerprint) == {
        "records",
        "models",
        "preprocessing",
        "cache_schema_version",
        "aggregate_sha256",
    }
    assert "text_encoding" not in fingerprint

    legacy = json.loads(fixture["legacy_source"].read_text(encoding="utf-8"))
    legacy.pop("manifest_sha256")
    legacy["source_fingerprint"] = fingerprint
    legacy["manifest_sha256"] = canonical_json_sha256(legacy)
    # Match the real deployment layout: the official Stage-1 producer writes
    # cache_manifest.json inside the latent cache directory.
    atomic_write_json(fixture["source"], legacy)
    return fixture, legacy, t5_checkpoint, tokenizer_dir


def _audit(fixture, **overrides):
    kwargs = {
        "metadata_path": fixture["metadata"],
        "cache_dir": fixture["cache_dir"],
        "source_cache_manifest_path": fixture["source"],
        "negative_conditioning_manifest_path": fixture["negative"],
        "expected_action_ids": ACTIONS,
        "action_labels_path": fixture["sidecar"],
        "expected_num_samples": 6,
        "expected_samples_per_action": 2,
        "config_contract_sha256": CONFIG_CONTRACT_HASH,
        "config_launch_sha256": CONFIG_LAUNCH_HASH,
    }
    kwargs.update(overrides)
    return audit_stage2_i2v_cache(**kwargs)


def _dataset_kwargs(fixture, **overrides):
    kwargs = {
        "metadata_path": fixture["metadata"],
        "source_cache_manifest_path": fixture["source"],
        "negative_conditioning_manifest_path": fixture["negative"],
        "config_contract_sha256": CONFIG_CONTRACT_HASH,
        "config_launch_sha256": CONFIG_LAUNCH_HASH,
        "expected_num_samples": 6,
    }
    kwargs.update(overrides)
    return kwargs


def test_f25_audit_dataset_and_collate_preserve_all_new_future_frames(
    tmp_path, monkeypatch
):
    fixture = _build_fixture(tmp_path)
    manifest = _audit(fixture)

    assert manifest["num_samples"] == 6
    assert manifest["actions"]["counts"] == {action: 2 for action in ACTIONS}
    assert manifest["actions"]["source"]["kind"] == "operator_confirmed_sidecar"
    assert manifest["common_tensor_schema_sha256"] == STAGE2_CACHE_TENSOR_SCHEMA_SHA256
    assert manifest["records"][0]["real_future_shape"] == [24, 48, 30, 52]
    assert manifest["records"][0]["initial_vs_video0"]["exact_equal"] is False
    assert manifest["records"][0]["initial_vs_video0"]["max_abs_diff"] == 99.0

    loaded = load_stage2_i2v_manifest(fixture["cache_dir"], expected_num_samples=6)
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]

    def forbid_rehash(_tensor):
        raise AssertionError("getitem must not redo the full-cache tensor audit")

    dataset = Stage2I2VCacheDataset(
        fixture["cache_dir"],
        **_dataset_kwargs(fixture),
    )
    monkeypatch.setattr("utils.stage2_i2v_data.tensor_sha256", forbid_rehash)
    first = dataset[0]
    assert tuple(first["initial_latent"].shape) == (1, 48, 30, 52)
    assert tuple(first["real_future"].shape) == (24, 48, 30, 52)
    assert torch.all(first["initial_latent"] == 99)
    # Source slots 1 and 24 are the first/last new future sentinels.  Cached
    # sink slot 0 must not leak into real_future.
    assert torch.all(first["real_future"][0] == 1)
    assert torch.all(first["real_future"][-1] == 24)
    assert "video_latent" not in first
    batch = stage2_i2v_cache_collate([first, dataset[2]])
    assert batch["initial_latent"].shape == (2, 1, 48, 30, 52)
    assert batch["real_future"].shape == (2, 24, 48, 30, 52)
    assert batch["sample_id"].tolist() == [0, 2]


def test_official_stage1_f24_passes_stage1_but_is_missing_stage2_future_24():
    tensors = _cache_tensors(frames=24)
    validate_stage1_cache_tensors(tensors)
    with pytest.raises(
        ValueError,
        match=(
            "official Stage-1 F24.*only 23 new future targets.*missing Stage-2 "
            "future frame 24.*Generate a separate Stage-2 F25 cache"
        ),
    ):
        validate_stage2_cache_tensors(tensors)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda tensors: tensors.__setitem__(
                "video_latent", torch.zeros(24, 48, 30, 52, dtype=torch.bfloat16)
            ),
            "official Stage-1 F24.*only 23 new future targets.*Stage-2 F25",
        ),
        (
            lambda tensors: tensors.__setitem__(
                "video_latent", torch.zeros(23, 48, 30, 52, dtype=torch.bfloat16)
            ),
            "F25",
        ),
        (
            lambda tensors: tensors.__setitem__(
                "video_latent", torch.zeros(25, 47, 30, 52, dtype=torch.bfloat16)
            ),
            r"\[25,48,H,W\]",
        ),
        (
            lambda tensors: tensors.__setitem__(
                "video_latent", torch.zeros(25, 48, 31, 52, dtype=torch.bfloat16)
            ),
            "spatial shape",
        ),
        (
            lambda tensors: tensors.__setitem__(
                "video_latent", tensors["video_latent"].float()
            ),
            "dtype",
        ),
    ],
)
def test_cache_tensor_gate_rejects_frame_channel_spatial_and_dtype(mutate, message):
    tensors = _cache_tensors()
    mutate(tensors)
    with pytest.raises(ValueError, match=message):
        validate_stage2_cache_tensors(tensors)


def test_cache_tensor_gate_rejects_nonzero_or_nonprefix_prompt_padding():
    tensors = _cache_tensors()
    tensors["prompt_embeds"][10, 0] = 1
    with pytest.raises(ValueError, match="padding positions"):
        validate_stage2_cache_tensors(tensors)
    tensors = _cache_tensors()
    tensors["prompt_mask"][1] = False
    tensors["prompt_mask"][3] = True
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        validate_stage2_cache_tensors(tensors)


def test_full_audit_rejects_one_official_f24_record_in_f25_cache(tmp_path):
    fixture = _build_fixture(tmp_path)
    bad_path = fixture["cache_dir"] / "sample_000005.safetensors"
    tensors = _cache_tensors(h=52, w=30, frames=24)
    save_file(tensors, str(bad_path))
    # A materialized native F25 row cannot be silently replaced and re-signed;
    # its success/source chain catches the changed bytes before tensor semantics.
    with pytest.raises(
        RuntimeError, match=r"materialized artifact row 5 (size|hash) mismatch"
    ):
        _audit(fixture)


def test_audit_rejects_cache_file_hash_mismatch(tmp_path):
    fixture = _build_fixture(tmp_path)
    path = fixture["cache_dir"] / "sample_000000.safetensors"
    with path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match=r"artifact row 0 (size|hash) mismatch"):
        _audit(fixture)


def test_action_labels_are_manifest_first_and_sidecar_must_agree(tmp_path):
    fixture = _build_fixture(tmp_path, source_actions=True)
    manifest = _audit(fixture, action_labels_path=None)
    assert manifest["actions"]["source"]["kind"] == "cache_manifest"

    rows = list(csv.DictReader(fixture["sidecar"].open(encoding="utf-8")))
    rows[0]["action_id"] = ACTIONS[1]
    with fixture["sidecar"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "action_id"])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="disagrees"):
        _audit(fixture)


def test_missing_partial_unknown_and_unbalanced_actions_fail_fast(tmp_path):
    fixture = _build_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="operator-confirmed"):
        _audit(fixture, action_labels_path=None)

    rows = list(csv.DictReader(fixture["sidecar"].open(encoding="utf-8")))
    rows.pop()
    with fixture["sidecar"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "action_id"])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="missing 1"):
        _audit(fixture)

    fixture = _build_fixture(tmp_path / "second")
    rows = list(csv.DictReader(fixture["sidecar"].open(encoding="utf-8")))
    rows[0]["action_id"] = ACTIONS[1]
    with fixture["sidecar"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "action_id"])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="Action counts"):
        _audit(fixture)

    fixture = _build_fixture(tmp_path / "third")
    rows = list(csv.DictReader(fixture["sidecar"].open(encoding="utf-8")))
    rows[0]["action_id"] = "not_a_frozen_action"
    with fixture["sidecar"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "action_id"])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="unknown action_id"):
        _audit(fixture)

    fixture = _build_fixture(tmp_path / "fourth")
    _rehash_manifest(
        fixture["source"],
        lambda value: value["records"][0].__setitem__("action_id", ACTIONS[0]),
    )
    with pytest.raises(RuntimeError, match="invented an action_id|unique append-only"):
        _audit(fixture)


def test_negative_conditioning_binds_exact_text_encoder_tensor_and_padding(tmp_path):
    fixture = _build_fixture(tmp_path)
    source = load_source_cache_manifest(fixture["source"], expected_num_samples=6)
    negative = load_negative_conditioning(
        fixture["negative"], source_cache_manifest=source
    )
    assert negative["manifest"]["text"] == DEFAULT_NEGATIVE_PROMPT
    assert (
        negative["manifest"]["text_utf8_sha256"]
        == hashlib.sha256(DEFAULT_NEGATIVE_PROMPT.encode("utf-8")).hexdigest()
    )
    assert negative["manifest"]["text_utf8_sha256"] == STAGE2_NEGATIVE_PROMPT_SHA256
    assert (
        torch.count_nonzero(negative["prompt_embeds"][~negative["prompt_mask"]]).item()
        == 0
    )

    _rehash_manifest(
        fixture["negative"],
        lambda value: value.__setitem__("text", DEFAULT_NEGATIVE_PROMPT + " "),
    )
    with pytest.raises(RuntimeError, match="byte-for-byte"):
        load_negative_conditioning(fixture["negative"], source_cache_manifest=source)


def test_negative_conditioning_rejects_encoder_and_artifact_hash_mismatch(tmp_path):
    fixture = _build_fixture(tmp_path)
    source = load_source_cache_manifest(fixture["source"], expected_num_samples=6)
    _rehash_manifest(
        fixture["negative"],
        lambda value: value["encoder"].__setitem__(
            "t5_checkpoint_aggregate_sha256", "9" * 64
        ),
    )
    with pytest.raises(RuntimeError, match="text encoding differ"):
        load_negative_conditioning(fixture["negative"], source_cache_manifest=source)

    fixture = _build_fixture(tmp_path / "second")
    source = load_source_cache_manifest(fixture["source"], expected_num_samples=6)
    artifact = fixture["negative"].parent / "negative_conditioning.safetensors"
    with artifact.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match="artifact (size|hash) mismatch"):
        load_negative_conditioning(fixture["negative"], source_cache_manifest=source)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cleaning", "none"),
        ("add_special_tokens", False),
        ("padding_side", "left"),
        ("sequence_length", 256),
    ],
)
def test_positive_cache_must_declare_exact_text_encoding_contract(
    tmp_path, field, value
):
    fixture = _build_fixture(tmp_path)
    _rehash_manifest(
        fixture["source"],
        lambda manifest: (
            manifest["source_fingerprint"]["text_encoding"].__setitem__(field, value),
            _rehash_source_fingerprint(manifest),
        ),
    )
    with pytest.raises(RuntimeError, match="Positive cache text encoding contract"):
        load_source_cache_manifest(fixture["source"], expected_num_samples=6)


def test_source_fingerprint_inner_aggregate_hash_is_mandatory(tmp_path):
    fixture = _build_fixture(tmp_path)
    _rehash_manifest(
        fixture["source"],
        lambda manifest: manifest["source_fingerprint"].__setitem__(
            "aggregate_sha256", "9" * 64
        ),
    )
    with pytest.raises(RuntimeError, match="fingerprint aggregate hash mismatch"):
        load_source_cache_manifest(fixture["source"], expected_num_samples=6)


def test_negative_manifest_checks_tensor_and_padding_hashes_not_only_file_hash(
    tmp_path,
):
    fixture = _build_fixture(tmp_path)
    source = load_source_cache_manifest(fixture["source"], expected_num_samples=6)
    _rehash_manifest(
        fixture["negative"],
        lambda manifest: manifest["artifact"]["tensors"]["prompt_embeds"].__setitem__(
            "sha256", "9" * 64
        ),
    )
    with pytest.raises(RuntimeError, match="tensor shape/dtype/hash"):
        load_negative_conditioning(fixture["negative"], source_cache_manifest=source)

    fixture = _build_fixture(tmp_path / "padding")
    source = load_source_cache_manifest(fixture["source"], expected_num_samples=6)
    _rehash_manifest(
        fixture["negative"],
        lambda manifest: manifest["artifact"].__setitem__("padding_sha256", "8" * 64),
    )
    with pytest.raises(RuntimeError, match="padding hash"):
        load_negative_conditioning(fixture["negative"], source_cache_manifest=source)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"config_contract_sha256": "8" * 64}, "config_contract_sha256"),
        ({"config_launch_sha256": "7" * 64}, "config_launch_sha256"),
    ],
)
def test_dataset_rejects_stale_config_bindings(tmp_path, override, match):
    fixture = _build_fixture(tmp_path)
    _audit(fixture)
    with pytest.raises(RuntimeError, match=match):
        Stage2I2VCacheDataset(
            fixture["cache_dir"],
            **_dataset_kwargs(fixture, **override),
        )


def test_dataset_resolves_clean_code_internally_and_has_no_public_bypass(
    tmp_path, monkeypatch
):
    fixture = _build_fixture(tmp_path)
    _audit(fixture)
    monkeypatch.setattr(
        "utils.stage2_i2v_data._resolve_clean_repo_code_version",
        lambda: "git:" + "b" * 40,
    )
    with pytest.raises(RuntimeError, match="code_version"):
        Stage2I2VCacheDataset(
            fixture["cache_dir"],
            **_dataset_kwargs(fixture),
        )
    with pytest.raises(TypeError, match="code_version"):
        Stage2I2VCacheDataset(
            fixture["cache_dir"],
            **_dataset_kwargs(fixture),
            code_version=UPGRADE_CODE_VERSION,
        )
    with pytest.raises(TypeError, match="verify_on_read"):
        Stage2I2VCacheDataset(
            fixture["cache_dir"],
            **_dataset_kwargs(fixture),
            verify_on_read=False,
        )


def test_dataset_rejects_rehashed_action_swap_that_preserves_global_counts(tmp_path):
    fixture = _build_fixture(tmp_path)
    manifest = _audit(fixture)

    def swap_actions(value):
        first = value["records"][0]["action_id"]
        value["records"][0]["action_id"] = value["records"][1]["action_id"]
        value["records"][1]["action_id"] = first

    _rehash_manifest(Path(manifest["manifest_path"]), swap_actions)
    with pytest.raises(RuntimeError, match="action_id.*source/metadata binding"):
        Stage2I2VCacheDataset(
            fixture["cache_dir"],
            **_dataset_kwargs(fixture),
        )


def test_dataset_rechecks_operator_action_sidecar_hash_at_training_start(tmp_path):
    fixture = _build_fixture(tmp_path)
    _audit(fixture)
    rows = list(csv.DictReader(fixture["sidecar"].open(encoding="utf-8")))
    rows[0]["action_id"], rows[1]["action_id"] = (
        rows[1]["action_id"],
        rows[0]["action_id"],
    )
    with fixture["sidecar"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "action_id"])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(RuntimeError, match="action sidecar hash changed"):
        Stage2I2VCacheDataset(
            fixture["cache_dir"],
            **_dataset_kwargs(fixture),
        )


def test_dataset_rehashes_all_artifacts_once_and_rejects_stale_source(tmp_path):
    fixture = _build_fixture(tmp_path)
    _audit(fixture)
    artifact = fixture["cache_dir"] / "sample_000000.safetensors"
    with artifact.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))
    with pytest.raises(
        RuntimeError,
        match="(startup cache row 0|F25 materialized artifact row 0) hash mismatch",
    ):
        Stage2I2VCacheDataset(
            fixture["cache_dir"],
            **_dataset_kwargs(fixture),
        )

    fixture = _build_fixture(tmp_path / "source")
    _audit(fixture)
    _rehash_manifest(
        fixture["source"],
        lambda manifest: manifest.__setitem__("producer_revision", "changed"),
    )
    with pytest.raises(
        RuntimeError,
        match="(?i)(source cache manifest differs|F25 source manifest keys mismatch)",
    ):
        Stage2I2VCacheDataset(
            fixture["cache_dir"],
            **_dataset_kwargs(fixture),
        )


def test_dataset_getitem_rejects_same_shape_replacement_after_initialization(tmp_path):
    fixture = _build_fixture(tmp_path)
    _audit(fixture)
    dataset = Stage2I2VCacheDataset(
        fixture["cache_dir"],
        **_dataset_kwargs(fixture),
    )
    artifact = fixture["cache_dir"] / "sample_000000.safetensors"
    original_size = artifact.stat().st_size
    replacement = _cache_tensors()
    replacement["video_latent"].fill_(42)
    with safe_open(str(artifact), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    save_file(replacement, str(artifact), metadata=metadata)
    assert artifact.stat().st_size == original_size

    with pytest.raises(RuntimeError, match="cache row 0 hash mismatch"):
        dataset[0]


def test_dataset_getitem_parses_the_exact_verified_byte_snapshot(tmp_path, monkeypatch):
    import safetensors.torch as safetensors_torch

    fixture = _build_fixture(tmp_path)
    _audit(fixture)
    dataset = Stage2I2VCacheDataset(
        fixture["cache_dir"],
        **_dataset_kwargs(fixture),
    )
    artifact = fixture["cache_dir"] / "sample_000000.safetensors"
    replacement = _cache_tensors()
    replacement["video_latent"].fill_(42)
    original_load = safetensors_torch.load
    swapped = False

    def swap_path_then_parse_payload(payload):
        nonlocal swapped
        if not swapped:
            with safe_open(str(artifact), framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
            save_file(replacement, str(artifact), metadata=metadata)
            swapped = True
        return original_load(payload)

    monkeypatch.setattr(safetensors_torch, "load", swap_path_then_parse_payload)
    item = dataset[0]
    assert swapped is True
    assert torch.all(item["real_future"][0] == 1)
    assert torch.all(item["real_future"][-1] == 24)
    with pytest.raises(RuntimeError, match="cache row 0 hash mismatch"):
        dataset[0]


def test_stage2_manifest_self_hash_is_mandatory(tmp_path):
    fixture = _build_fixture(tmp_path)
    manifest = _audit(fixture)
    path = Path(manifest["manifest_path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["records"][0]["action_id"] = ACTIONS[1]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_stage2_i2v_manifest(path)


def test_legacy_text_upgrade_remains_migration_only_for_formal_negative(
    tmp_path, monkeypatch
):
    from scripts import audit_stage2_i2v_cache as audit_cli
    from utils import wan_5b_wrapper

    fixture, legacy, t5_checkpoint, tokenizer_dir = (
        _build_official_legacy_upgrade_fixture(tmp_path / "fixture")
    )
    legacy_bytes = fixture["source"].read_bytes()

    monkeypatch.setattr(
        wan_5b_wrapper,
        "audit_wan_text_encoding_tokenizer_contract",
        lambda _path: _locked_tokenizer_runtime_audit(),
    )
    monkeypatch.setattr(audit_cli, "_git_code_version", lambda: UPGRADE_CODE_VERSION)
    upgraded_source = tmp_path / "upgraded" / "stage2_source_manifest.json"
    assert upgraded_source.parent.resolve() != fixture["cache_dir"].resolve()
    audit_cli._upgrade_source_manifest(
        SimpleNamespace(
            legacy_source_cache_manifest=str(fixture["source"]),
            output_manifest=str(upgraded_source),
            expected_source_manifest_sha256=legacy["manifest_sha256"],
            t5_checkpoint=str(t5_checkpoint),
            tokenizer_dir=str(tokenizer_dir),
            operator_id="operator-test",
            operator_attestation=STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
            expected_num_samples=6,
        )
    )
    assert fixture["source"].read_bytes() == legacy_bytes
    upgraded = load_source_cache_manifest(
        upgraded_source,
        expected_num_samples=6,
        require_text_encoding_upgrade=True,
        expected_upgrade_code_version=UPGRADE_CODE_VERSION,
    )
    assert (
        upgraded["stage2_text_encoding_upgrade"]["original_source_manifest"][
            "manifest_sha256"
        ]
        == legacy["manifest_sha256"]
    )

    encoder_loaded = False

    class _ForbiddenWanTextEncoder:
        def __init__(self, **_kwargs):
            nonlocal encoder_loaded
            encoder_loaded = True
            raise AssertionError("legacy input must fail before loading T5")

    monkeypatch.setattr(wan_5b_wrapper, "WanTextEncoder", _ForbiddenWanTextEncoder)
    negative_output = tmp_path / "prepared-negative"
    with pytest.raises(RuntimeError, match="only the native attested Stage-2 F25"):
        audit_cli._prepare_negative(
            SimpleNamespace(
                source_cache_manifest=str(upgraded_source),
                t5_checkpoint=str(t5_checkpoint),
                tokenizer_dir=str(tokenizer_dir),
                output_dir=str(negative_output),
                expected_num_samples=6,
                device="cpu",
                force=False,
            )
        )
    assert encoder_loaded is False
    assert not negative_output.exists()


@pytest.mark.parametrize("changed_asset", ["t5", "tokenizer"])
def test_prepare_negative_rehashes_model_trees_after_encoding_without_publishing(
    tmp_path, monkeypatch, changed_asset
):
    from scripts import audit_stage2_i2v_cache as audit_cli
    from utils import wan_5b_wrapper

    fixture = _build_fixture(tmp_path / "fixture")
    t5_checkpoint = fixture["t5_checkpoint"]
    tokenizer_dir = fixture["tokenizer_dir"]
    monkeypatch.setattr(audit_cli, "_git_code_version", lambda: UPGRADE_CODE_VERSION)

    class _MutatingWanTextEncoder:
        def __init__(self, *, t5_checkpoint, tokenizer_dir, device):
            self.t5_checkpoint = Path(t5_checkpoint)
            self.tokenizer_dir = Path(tokenizer_dir)

        def eval(self):
            return self

        def __call__(self, prompts, *, return_mask):
            assert prompts == [DEFAULT_NEGATIVE_PROMPT]
            assert return_mask is True
            target = (
                self.t5_checkpoint
                if changed_asset == "t5"
                else self.tokenizer_dir / "tokenizer.json"
            )
            with target.open("ab") as handle:
                handle.write(b"changed-during-negative-encoding")
            embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)
            mask = torch.zeros(1, 512, dtype=torch.bool)
            mask[:, :4] = True
            return {"prompt_embeds": embeds, "prompt_mask": mask}

    monkeypatch.setattr(wan_5b_wrapper, "WanTextEncoder", _MutatingWanTextEncoder)
    output_dir = tmp_path / f"negative-{changed_asset}"
    with pytest.raises(
        RuntimeError,
        match=rf"{changed_asset.title()}.*changed during negative-conditioning",
    ):
        audit_cli._prepare_negative(
            SimpleNamespace(
                source_cache_manifest=str(fixture["source"]),
                t5_checkpoint=str(t5_checkpoint),
                tokenizer_dir=str(tokenizer_dir),
                output_dir=str(output_dir),
                expected_num_samples=6,
                device="cpu",
                force=False,
            )
        )
    assert not (output_dir / "negative_conditioning.safetensors").exists()
    assert not (output_dir / "negative_conditioning_manifest.json").exists()


@pytest.mark.parametrize("changed_asset", ["t5", "tokenizer"])
def test_native_upgrade_rechecks_assets_after_candidate_closure_scan(
    tmp_path, monkeypatch, changed_asset
):
    from utils import stage2_i2v_data, wan_5b_wrapper

    fixture = _build_fixture(tmp_path / "fixture")
    monkeypatch.setattr(
        wan_5b_wrapper,
        "audit_wan_text_encoding_tokenizer_contract",
        lambda _path: _locked_tokenizer_runtime_audit(),
    )
    real_loader = stage2_i2v_data.load_source_cache_manifest

    def load_then_mutate(*args, **kwargs):
        loaded = real_loader(*args, **kwargs)
        target = (
            fixture["t5_checkpoint"]
            if changed_asset == "t5"
            else fixture["tokenizer_dir"] / "tokenizer.json"
        )
        with target.open("ab") as handle:
            handle.write(b"changed-during-candidate-closure")
        return loaded

    monkeypatch.setattr(stage2_i2v_data, "load_source_cache_manifest", load_then_mutate)
    base = json.loads(fixture["base_source"].read_text(encoding="utf-8"))
    output = tmp_path / f"must-not-publish-{changed_asset}.json"
    with pytest.raises(
        RuntimeError,
        match=rf"{changed_asset.title()}.*changed during the upgrade audit",
    ):
        upgrade_legacy_source_cache_manifest_text_encoding(
            fixture["base_source"],
            output,
            expected_source_manifest_sha256=base["manifest_sha256"],
            t5_checkpoint_path=fixture["t5_checkpoint"],
            tokenizer_dir=fixture["tokenizer_dir"],
            operator_id="operator-test",
            operator_attestation=STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
            expected_num_samples=6,
        )
    assert not output.exists()


@pytest.mark.parametrize("changed_asset", ["t5", "tokenizer"])
def test_legacy_upgrade_rehashes_model_trees_after_runtime_probe(
    tmp_path, monkeypatch, changed_asset
):
    from utils import wan_5b_wrapper

    fixture, legacy, t5_checkpoint, tokenizer_dir = (
        _build_official_legacy_upgrade_fixture(tmp_path / "fixture")
    )
    legacy_bytes = fixture["source"].read_bytes()

    def mutate_asset_during_probe(_tokenizer_dir):
        target = (
            t5_checkpoint if changed_asset == "t5" else tokenizer_dir / "tokenizer.json"
        )
        with target.open("ab") as handle:
            handle.write(b"changed-during-runtime-probe")
        return _locked_tokenizer_runtime_audit()

    monkeypatch.setattr(
        wan_5b_wrapper,
        "audit_wan_text_encoding_tokenizer_contract",
        mutate_asset_during_probe,
    )
    output = tmp_path / f"upgraded-{changed_asset}.json"
    with pytest.raises(RuntimeError, match=rf"{changed_asset.title()}.*changed during"):
        upgrade_legacy_source_cache_manifest_text_encoding(
            fixture["source"],
            output,
            expected_source_manifest_sha256=legacy["manifest_sha256"],
            t5_checkpoint_path=t5_checkpoint,
            tokenizer_dir=tokenizer_dir,
            operator_id="operator-test",
            operator_attestation=STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
            expected_num_samples=6,
        )
    assert fixture["source"].read_bytes() == legacy_bytes
    assert not output.exists()


def test_legacy_upgrade_rechecks_clean_head_before_atomic_publish(
    tmp_path, monkeypatch
):
    from utils import wan_5b_wrapper

    fixture, legacy, t5_checkpoint, tokenizer_dir = (
        _build_official_legacy_upgrade_fixture(tmp_path / "fixture")
    )
    monkeypatch.setattr(
        wan_5b_wrapper,
        "audit_wan_text_encoding_tokenizer_contract",
        lambda _path: _locked_tokenizer_runtime_audit(),
    )
    revisions = iter([UPGRADE_CODE_VERSION, "git:" + "b" * 40])
    monkeypatch.setattr(
        "utils.stage2_i2v_data._resolve_clean_repo_code_version",
        lambda: next(revisions),
    )
    output = tmp_path / "must-not-publish.json"
    with pytest.raises(RuntimeError, match="code version changed during"):
        upgrade_legacy_source_cache_manifest_text_encoding(
            fixture["source"],
            output,
            expected_source_manifest_sha256=legacy["manifest_sha256"],
            t5_checkpoint_path=t5_checkpoint,
            tokenizer_dir=tokenizer_dir,
            operator_id="operator-test",
            operator_attestation=STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
            expected_num_samples=6,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest: manifest["records"][0].__setitem__(
                "path", "resigned-but-not-attested.safetensors"
            ),
            "unique append-only upgrade",
        ),
        (
            lambda manifest: manifest["stage2_text_encoding_upgrade"].__setitem__(
                "schema_version", 1.0
            ),
            "Unsupported Stage-2 text-encoding upgrade schema",
        ),
        (
            lambda manifest: manifest["stage2_text_encoding_upgrade"]["verification"][
                "tokenizer_runtime_audit"
            ].__setitem__("sequence_length", 512.0),
            "sequence_length differs from the locked",
        ),
    ],
)
def test_attested_upgrade_rejects_resigned_tamper_and_loose_types(
    tmp_path, mutate, message
):
    fixture = _build_fixture(tmp_path)
    _rehash_manifest(fixture["source"], mutate)
    with pytest.raises(RuntimeError, match=message):
        load_source_cache_manifest(
            fixture["source"],
            expected_num_samples=6,
            require_text_encoding_upgrade=True,
            expected_upgrade_code_version=UPGRADE_CODE_VERSION,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "schema",
            "producer_cache_fixture",
            "Unsupported source cache manifest schema",
        ),
        ("schema_version", 1.0, "Unsupported source cache manifest schema"),
        ("num_samples", 6.0, "num_samples does not match records"),
    ],
)
def test_legacy_upgrade_accepts_only_strict_official_stage1_manifest(
    tmp_path, field, value, message
):
    fixture, legacy, t5_checkpoint, tokenizer_dir = (
        _build_official_legacy_upgrade_fixture(tmp_path / "fixture")
    )
    _rehash_manifest(
        fixture["source"], lambda manifest: manifest.__setitem__(field, value)
    )
    changed = json.loads(fixture["source"].read_text(encoding="utf-8"))
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(RuntimeError, match=message):
        upgrade_legacy_source_cache_manifest_text_encoding(
            fixture["source"],
            output,
            expected_source_manifest_sha256=changed["manifest_sha256"],
            t5_checkpoint_path=t5_checkpoint,
            tokenizer_dir=tokenizer_dir,
            operator_id="operator-test",
            operator_attestation=STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
            expected_num_samples=6,
        )
    assert not output.exists()


def test_dataset_runtime_rejects_resigned_unattested_source_manifest(tmp_path):
    fixture = _build_fixture(tmp_path)
    _audit(fixture)
    legacy = json.loads(fixture["legacy_source"].read_text(encoding="utf-8"))
    legacy.pop("manifest_sha256")
    legacy["source_fingerprint"].pop("aggregate_sha256")
    model_fingerprint = legacy["source_fingerprint"]["models"]
    t5_hash = model_fingerprint["t5_checkpoint"]["aggregate_sha256"]
    tokenizer_hash = model_fingerprint["tokenizer_dir"]["aggregate_sha256"]
    legacy["source_fingerprint"]["text_encoding"] = {
        "t5_checkpoint_aggregate_sha256": t5_hash,
        "tokenizer_aggregate_sha256": tokenizer_hash,
        "tokenizer_revision": f"local-tree-sha256:{tokenizer_hash}",
        "cleaning": "whitespace",
        "add_special_tokens": True,
        "sequence_length": 512,
        "padding_side": "right",
        "embedding_padding_value": 0.0,
    }
    _rehash_source_fingerprint(legacy)
    legacy["manifest_sha256"] = canonical_json_sha256(legacy)
    unattested = tmp_path / "unattested-source-manifest.json"
    atomic_write_json(unattested, legacy)

    with pytest.raises(RuntimeError, match="requires the attested output"):
        Stage2I2VCacheDataset(
            fixture["cache_dir"],
            **_dataset_kwargs(
                fixture, source_cache_manifest_path=str(unattested.resolve())
            ),
        )


def test_legacy_upgrade_never_overwrites_original_and_requires_exact_attestation(
    tmp_path,
):
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text("{}\n", encoding="utf-8")
    kwargs = {
        "expected_source_manifest_sha256": "1" * 64,
        "t5_checkpoint_path": tmp_path / "t5",
        "tokenizer_dir": tmp_path / "tokenizer",
        "operator_id": "operator-test",
        "operator_attestation": STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
        "expected_num_samples": 6,
    }
    with pytest.raises(ValueError, match="must not overwrite"):
        upgrade_legacy_source_cache_manifest_text_encoding(
            legacy_path, legacy_path, **kwargs
        )
    with pytest.raises(RuntimeError, match="operator_attestation"):
        upgrade_legacy_source_cache_manifest_text_encoding(
            legacy_path,
            tmp_path / "upgraded.json",
            **{**kwargs, "operator_attestation": "yes"},
        )
