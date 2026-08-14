"""End-to-end guards for the native Stage-2 F25 trust chain.

These tests deliberately start from an immutable Stage-1 manifest.  A legacy
manifest, or an F25-shaped artifact merely described by that legacy schema, is
never treated as a formal Stage-2 training input.
"""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
from safetensors.torch import load_file, save_file
import torch

from scripts import prepare_stage2_i2v_f25_cache as f25_cli
from utils.stage1_i2v_data import load_stage1_i2v_manifest
from utils.stage1_io import (
    aggregate_file_hash,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
    tree_file_hashes,
)
from utils.stage2_f25_cache import (
    _input_kind,
    prepare_stage2_f25_cache,
    stable_f25_row_ids,
)
from utils.stage2_i2v_data import (
    STAGE2_F25_SOURCE_CACHE_SCHEMA,
    STAGE2_F25_SUCCESS_NAME,
    STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
    Stage2I2VCacheDataset,
    audit_stage2_i2v_cache,
    load_negative_conditioning,
    load_source_cache_manifest,
    load_stage2_i2v_manifest,
    save_negative_conditioning_artifact,
    stage2_i2v_cache_collate,
    upgrade_legacy_source_cache_manifest_text_encoding,
    write_negative_conditioning_manifest,
)

ACTIONS = ("head_tilt", "jump", "toy_play")
CONFIG_CONTRACT_SHA256 = "1" * 64
CONFIG_LAUNCH_SHA256 = "2" * 64


def _tree_hash(path: Path) -> str:
    return aggregate_file_hash(tree_file_hashes(path))


def _shape_dtype(tensors):
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        for name, tensor in sorted(tensors.items())
    }


def _cache_tensors(row_id: int):
    video = torch.empty(25, 48, 30, 52, dtype=torch.bfloat16)
    for frame in range(25):
        video[frame].fill_(frame)
    prompt = torch.zeros(512, 4096, dtype=torch.bfloat16)
    prompt[:3, 0] = torch.tensor(
        [row_id + 1, row_id + 2, row_id + 3], dtype=torch.bfloat16
    )
    mask = torch.zeros(512, dtype=torch.bool)
    mask[:3] = True
    return {
        "video_latent": video,
        "initial_latent": torch.full(
            (1, 48, 30, 52), 99 + row_id, dtype=torch.bfloat16
        ),
        "prompt_embeds": prompt,
        "prompt_mask": mask,
    }


def _resign_json(path: Path, value: dict) -> Path:
    value = json.loads(json.dumps(value))
    value.pop("manifest_sha256", None)
    value["manifest_sha256"] = canonical_json_sha256(value)
    atomic_write_json(path, value)
    return path


@pytest.fixture(autouse=True)
def _trusted_test_checkout(monkeypatch):
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


def _build_native_chain(root: Path) -> dict[str, Path]:
    stage1_cache = root / "stage1_cache"
    f25_cache = root / "f25_cache"
    videos = root / "videos"
    images = root / "images"
    for directory in (stage1_cache, videos, images):
        directory.mkdir(parents=True)

    metadata_path = root / "metadata.csv"
    metadata_rows = []
    labels = []
    for row_id in range(6):
        video = videos / f"{row_id}.mp4"
        image = images / f"{row_id}.png"
        video.write_bytes(f"source-video-{row_id}".encode())
        Image.new("RGB", (832, 480), (row_id, 0, 0)).save(image)
        metadata_rows.append(
            {
                "video": f"videos/{row_id}.mp4",
                "prompt": f"cat action {row_id}",
                "input_image": f"images/{row_id}.png",
                "height": "480",
                "width": "832",
                "bucket": "landscape",
            }
        )
        labels.append(ACTIONS[row_id % len(ACTIONS)])
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)
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
    t5_checkpoint.write_bytes(b"locked-test-t5")
    vae_checkpoint.write_bytes(b"locked-test-vae")
    (tokenizer_dir / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    source_entries = []
    for record in records:
        tensors = _cache_tensors(record.row_id)
        artifact = stage1_cache / f"sample_{record.row_id:06d}.safetensors"
        save_file(tensors, str(artifact))
        source_entries.append(
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
                "files": tree_file_hashes(vae_checkpoint),
                "aggregate_sha256": _tree_hash(vae_checkpoint),
            },
            "t5_checkpoint": {
                "files": tree_file_hashes(t5_checkpoint),
                "aggregate_sha256": _tree_hash(t5_checkpoint),
            },
            "tokenizer_dir": {
                "files": tree_file_hashes(tokenizer_dir),
                "aggregate_sha256": _tree_hash(tokenizer_dir),
            },
        },
        # This is the only legacy policy that permits byte reuse of actual F25.
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
        "cache_schema_version": 1,
    }
    fingerprint["aggregate_sha256"] = canonical_json_sha256(fingerprint)
    # Official Stage-1 artifacts bind each tensor file back to the complete
    # source fingerprint.  F25 preparation rejects shape-only fixture files.
    for record, entry in zip(records, source_entries):
        artifact = stage1_cache / entry["path"]
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
    legacy = {
        "schema": "longlive_stage1_i2v_cache",
        "schema_version": 1,
        "num_samples": 6,
        "source_fingerprint": fingerprint,
        "records": source_entries,
    }
    legacy["manifest_sha256"] = canonical_json_sha256(legacy)
    legacy_path = stage1_cache / "cache_manifest.json"
    atomic_write_json(legacy_path, legacy)

    config_path = root / "stage2.yaml"
    config_path.write_text("config_schema: integration_guard\n", encoding="utf-8")
    base_path = prepare_stage2_f25_cache(
        metadata_path=metadata_path,
        source_cache_manifest_path=legacy_path,
        output_dir=f25_cache,
        config_path=config_path,
        config_contract_sha256=CONFIG_CONTRACT_SHA256,
        config_launch_sha256=CONFIG_LAUNCH_SHA256,
        expected_num_samples=6,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    assert base_path is not None
    base = json.loads(base_path.read_text(encoding="utf-8"))
    assert base["schema"] == STAGE2_F25_SOURCE_CACHE_SCHEMA
    assert base["preparation"]["summary"] == {
        "reused_f25": 6,
        "reencoded_f24": 0,
        "reverified_f25": 0,
    }
    assert (f25_cache / STAGE2_F25_SUCCESS_NAME).is_file()

    attested_path = f25_cache / "cache_manifest.attested.json"
    upgrade_legacy_source_cache_manifest_text_encoding(
        base_path,
        attested_path,
        expected_source_manifest_sha256=base["manifest_sha256"],
        t5_checkpoint_path=t5_checkpoint,
        tokenizer_dir=tokenizer_dir,
        operator_id="integration-guard",
        operator_attestation=STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
        expected_num_samples=6,
    )

    negative_dir = root / "negative"
    negative_dir.mkdir()
    negative_artifact = negative_dir / "negative_conditioning.safetensors"
    negative_embeds = torch.zeros(512, 4096, dtype=torch.bfloat16)
    negative_embeds[:4, 0] = 1
    negative_mask = torch.zeros(512, dtype=torch.bool)
    negative_mask[:4] = True
    save_negative_conditioning_artifact(
        negative_artifact,
        prompt_embeds=negative_embeds,
        prompt_mask=negative_mask,
    )
    negative_manifest = negative_dir / "negative_conditioning_manifest.json"
    write_negative_conditioning_manifest(
        negative_manifest,
        artifact_path=negative_artifact,
        source_cache_manifest_path=attested_path,
        expected_num_samples=6,
        require_text_encoding_upgrade=True,
    )

    sidecar = root / "actions.csv"
    with sidecar.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "action_id"])
        writer.writeheader()
        for row, label in zip(metadata_rows, labels):
            writer.writerow({"video": row["video"], "action_id": label})
    return {
        "cache": f25_cache,
        "metadata": metadata_path,
        "legacy": legacy_path,
        "base": base_path,
        "attested": attested_path,
        "negative": negative_manifest,
        "negative_artifact": negative_artifact,
        "sidecar": sidecar,
        "t5": t5_checkpoint,
        "tokenizer": tokenizer_dir,
    }


def _audit_kwargs(chain: dict[str, Path], **overrides):
    kwargs = {
        "metadata_path": chain["metadata"],
        "cache_dir": chain["cache"],
        "source_cache_manifest_path": chain["attested"],
        "negative_conditioning_manifest_path": chain["negative"],
        "expected_action_ids": ACTIONS,
        "action_labels_path": chain["sidecar"],
        "expected_num_samples": 6,
        "expected_samples_per_action": 2,
        "config_contract_sha256": CONFIG_CONTRACT_SHA256,
        "config_launch_sha256": CONFIG_LAUNCH_SHA256,
        "require_text_encoding_upgrade": True,
        "require_native_f25_source": True,
    }
    kwargs.update(overrides)
    return kwargs


def _dataset_kwargs(chain: dict[str, Path], **overrides):
    kwargs = {
        "metadata_path": chain["metadata"],
        "source_cache_manifest_path": chain["attested"],
        "negative_conditioning_manifest_path": chain["negative"],
        "config_contract_sha256": CONFIG_CONTRACT_SHA256,
        "config_launch_sha256": CONFIG_LAUNCH_SHA256,
        "expected_num_samples": 6,
    }
    kwargs.update(overrides)
    return kwargs


def test_native_chain_is_the_only_formal_audit_and_dataset_input(tmp_path):
    chain = _build_native_chain(tmp_path)

    formal = audit_stage2_i2v_cache(**_audit_kwargs(chain))
    assert formal["num_samples"] == 6
    assert formal["actions"]["counts"] == {action: 2 for action in ACTIONS}
    assert (
        formal["provenance"]["source_cache_manifest_sha256"]
        == json.loads(chain["attested"].read_text(encoding="utf-8"))["manifest_sha256"]
    )
    loaded = load_stage2_i2v_manifest(chain["cache"], expected_num_samples=6)
    assert loaded["manifest_sha256"] == formal["manifest_sha256"]

    dataset = Stage2I2VCacheDataset(
        chain["cache"],
        **_dataset_kwargs(chain),
    )
    first = dataset[0]
    assert tuple(first["initial_latent"].shape) == (1, 48, 30, 52)
    assert tuple(first["real_future"].shape) == (24, 48, 30, 52)
    assert torch.all(first["real_future"][0] == 1)
    assert torch.all(first["real_future"][-1] == 24)
    assert "video_latent" not in first
    batch = stage2_i2v_cache_collate([first, dataset[3]])
    assert tuple(batch["real_future"].shape) == (2, 24, 48, 30, 52)

    # The producer's raw native base is deliberately not a formal input: the
    # append-only text attestation must happen before negative-cache creation.
    with pytest.raises(RuntimeError, match="requires the attested output"):
        audit_stage2_i2v_cache(
            **_audit_kwargs(chain, source_cache_manifest_path=chain["base"])
        )
    with pytest.raises(RuntimeError, match="requires the attested output"):
        write_negative_conditioning_manifest(
            tmp_path / "negative" / "raw-base.json",
            artifact_path=chain["negative_artifact"],
            source_cache_manifest_path=chain["base"],
            expected_num_samples=6,
            require_text_encoding_upgrade=True,
        )

    # The F25 audit contract binds the exact metadata bytes, not merely the
    # fact that another CSV parses to the same six rows.
    equivalent_metadata = tmp_path / "metadata-equivalent.csv"
    equivalent_metadata.write_text(
        chain["metadata"].read_text(encoding="utf-8"), encoding="utf-8-sig"
    )
    with pytest.raises(RuntimeError, match="data contract differs"):
        audit_stage2_i2v_cache(
            **_audit_kwargs(chain, metadata_path=equivalent_metadata)
        )

    # Even a correctly text-attested legacy Stage-1 manifest remains migration
    # input only.  It cannot produce a negative manifest, pass formal audit, or
    # satisfy the training dataset's runtime bindings.
    legacy = json.loads(chain["legacy"].read_text(encoding="utf-8"))
    legacy_attested = tmp_path / "legacy.attested.json"
    upgrade_legacy_source_cache_manifest_text_encoding(
        chain["legacy"],
        legacy_attested,
        expected_source_manifest_sha256=legacy["manifest_sha256"],
        t5_checkpoint_path=chain["t5"],
        tokenizer_dir=chain["tokenizer"],
        operator_id="legacy-rejection-guard",
        operator_attestation=STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
        expected_num_samples=6,
    )
    with pytest.raises(RuntimeError, match="native attested F25"):
        write_negative_conditioning_manifest(
            tmp_path / "negative" / "legacy.json",
            artifact_path=chain["negative_artifact"],
            source_cache_manifest_path=legacy_attested,
            expected_num_samples=6,
            require_text_encoding_upgrade=True,
        )
    with pytest.raises(RuntimeError, match="Formal Stage-2 audit.*native attested F25"):
        audit_stage2_i2v_cache(
            **_audit_kwargs(chain, source_cache_manifest_path=legacy_attested)
        )
    with pytest.raises(RuntimeError, match="training runtime.*native attested F25"):
        Stage2I2VCacheDataset(
            chain["cache"],
            **_dataset_kwargs(chain, source_cache_manifest_path=legacy_attested),
        )

    # Recomputing the outer self-hash is not authority to change an attested
    # source record.  The append-only derivation from the successful base wins.
    attested = json.loads(chain["attested"].read_text(encoding="utf-8"))
    attested["records"][0]["path"] = "../outside.safetensors"
    resigned_tamper = _resign_json(tmp_path / "resigned-tamper.json", attested)
    with pytest.raises(RuntimeError, match="unique append-only upgrade"):
        load_source_cache_manifest(
            resigned_tamper,
            expected_num_samples=6,
            require_text_encoding_upgrade=True,
        )

    # A self-consistent but stale negative manifest must remain bound to the
    # exact positive-manifest hash used when it was encoded.
    negative = json.loads(chain["negative"].read_text(encoding="utf-8"))
    negative["positive_cache_manifest_sha256"] = json.loads(
        legacy_attested.read_text(encoding="utf-8")
    )["manifest_sha256"]
    stale_negative = _resign_json(
        chain["negative"].parent / "stale-negative.json", negative
    )
    with pytest.raises(RuntimeError, match="not built from this positive cache"):
        audit_stage2_i2v_cache(
            **_audit_kwargs(chain, negative_conditioning_manifest_path=stale_negative)
        )

    # Relative-path binding is checked independently of the manifest self-hash.
    escaped = json.loads(chain["negative"].read_text(encoding="utf-8"))
    escaped["artifact"]["path"] = "../negative_conditioning.safetensors"
    escaped_negative = _resign_json(
        chain["negative"].parent / "escaped-negative.json", escaped
    )
    source = load_source_cache_manifest(
        chain["attested"],
        expected_num_samples=6,
        require_text_encoding_upgrade=True,
    )
    with pytest.raises(RuntimeError, match="must be a relative path|escapes"):
        load_negative_conditioning(
            escaped_negative,
            source_cache_manifest=source,
            load_tensors=False,
        )


@pytest.mark.parametrize("damage", ["missing_success", "corrupt_f25_artifact"])
def test_native_upgrade_rejects_incomplete_materialization_before_tokenizer_probe(
    tmp_path, monkeypatch, damage
):
    chain = _build_native_chain(tmp_path)
    base = json.loads(chain["base"].read_text(encoding="utf-8"))
    success = chain["cache"] / STAGE2_F25_SUCCESS_NAME
    artifact = chain["cache"] / base["records"][0]["path"]
    restored_path = None
    original_artifact = None
    if damage == "missing_success":
        restored_path = tmp_path / "saved-success.json"
        success.rename(restored_path)
    else:
        original_artifact = artifact.read_bytes()
        artifact.write_bytes(b"corrupt-f25-artifact")

    probe_calls = []

    def tokenizer_probe(_path):
        probe_calls.append(True)
        return {
            "cleaning": "whitespace",
            "add_special_tokens": True,
            "sequence_length": 512,
            "padding_side": "right",
            "embedding_padding_value": 0.0,
            "validated_special_token_growth": True,
            "validated_right_padding_mask": True,
        }

    monkeypatch.setattr(
        "utils.wan_5b_wrapper.audit_wan_text_encoding_tokenizer_contract",
        tokenizer_probe,
    )
    rejected_output = tmp_path / f"rejected-{damage}.json"
    try:
        with pytest.raises((FileNotFoundError, RuntimeError)):
            upgrade_legacy_source_cache_manifest_text_encoding(
                chain["base"],
                rejected_output,
                expected_source_manifest_sha256=base["manifest_sha256"],
                t5_checkpoint_path=chain["t5"],
                tokenizer_dir=chain["tokenizer"],
                operator_id="must-not-reach-probe",
                operator_attestation=STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
                expected_num_samples=6,
            )
        assert probe_calls == []
        assert not rejected_output.exists()
    finally:
        if restored_path is not None:
            restored_path.rename(success)
        if original_artifact is not None:
            artifact.write_bytes(original_artifact)


def test_f25_cli_refuses_nonisolated_startup_before_argparse_shadow(tmp_path):
    shadow_root = tmp_path / "shadow"
    shadow_root.mkdir()
    marker = tmp_path / "argparse-imported.txt"
    (shadow_root / "argparse.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        "raise RuntimeError('argparse shadow executed')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(shadow_root)
    completed = subprocess.run(
        [sys.executable, str(f25_cli.__file__), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "python -I -B scripts/prepare_stage2_i2v_f25_cache.py" in completed.stderr
    assert not marker.exists()


def test_600_row_lightweight_plan_uses_real_decision_and_sharding_helpers():
    """Cover 600-row planning without pretending to materialize 600 tensors."""

    proven_manifest = {
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
    unproven_manifest = json.loads(json.dumps(proven_manifest))
    unproven_manifest["source_fingerprint"]["preprocessing"][
        "selected_frame_count"
    ] = 93
    f25 = _cache_tensors(0)
    f24 = {**f25, "video_latent": f25["video_latent"][:24].contiguous()}
    real_decisions = (
        _input_kind(
            f25,
            source_manifest=proven_manifest,
            expected_spatial_shape=(30, 52),
            row_id=0,
        ),
        _input_kind(
            f24,
            source_manifest=proven_manifest,
            expected_spatial_shape=(30, 52),
            row_id=1,
        ),
        _input_kind(
            f25,
            source_manifest=unproven_manifest,
            expected_spatial_shape=(30, 52),
            row_id=2,
        ),
    )
    assert real_decisions == (
        "reused_f25",
        "reencoded_f24",
        "reverified_f25",
    )

    assignments = [
        stable_f25_row_ids(600, rank=rank, world_size=8) for rank in range(8)
    ]
    flattened = [row_id for shard in assignments for row_id in shard]
    assert sorted(flattened) == list(range(600))
    assert len(flattened) == len(set(flattened)) == 600
    planned = [real_decisions[row_id % 3] for row_id in range(600)]
    assert {decision: planned.count(decision) for decision in real_decisions} == {
        "reused_f25": 200,
        "reencoded_f24": 200,
        "reverified_f25": 200,
    }


def test_f25_cli_isolated_startup_does_not_depend_on_git_state(tmp_path):
    shadow_root = tmp_path / "shadow"
    shadow_root.mkdir()
    marker = tmp_path / "omegaconf-imported.txt"
    (shadow_root / "omegaconf.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        "raise RuntimeError('isolated startup imported shadow module')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["GIT_DIR"] = str(tmp_path / "decoy.git")
    environment["PYTHONPATH"] = str(shadow_root)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(f25_cli.__file__), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()


def test_f25_cli_uses_only_config_bound_metadata_and_output_paths(
    tmp_path, monkeypatch
):
    parser = f25_cli._parser()
    help_text = parser.format_help()
    assert "--metadata-path" not in help_text
    assert "--output-dir" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--config-path",
                "stage2.yaml",
                "--source-cache-manifest",
                "source.json",
                "--output-dir",
                "elsewhere",
            ]
        )

    config_path = tmp_path / "stage2.yaml"
    config_path.write_text("config_schema: guard\n", encoding="utf-8")
    bound_metadata = tmp_path / "bound" / "metadata.csv"
    bound_output = tmp_path / "bound" / "f25"
    resolved = SimpleNamespace(
        video_latent_frames=25,
        future_latent_frames=24,
        expected_world_size=1,
        metadata_path=str(bound_metadata),
        cache_dir=str(bound_output),
        expected_num_samples=6,
        contract_hash=lambda: CONFIG_CONTRACT_SHA256,
        launch_hash=lambda: CONFIG_LAUNCH_SHA256,
    )
    captured = {}

    class ProducerReached(RuntimeError):
        pass

    def capture_producer_call(**kwargs):
        captured.update(kwargs)
        raise ProducerReached

    monkeypatch.setattr(f25_cli.OmegaConf, "load", lambda _path: object())
    monkeypatch.setattr(f25_cli, "resolve_stage2_config", lambda _raw: resolved)
    monkeypatch.setattr(
        f25_cli,
        "_distributed_device",
        lambda: (0, 1, torch.device("cpu")),
    )
    monkeypatch.setattr(
        f25_cli,
        "prepare_stage2_f25_cache",
        capture_producer_call,
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    source_manifest = tmp_path / "source.json"
    args = parser.parse_args(
        [
            "--config-path",
            str(config_path),
            "--source-cache-manifest",
            str(source_manifest),
        ]
    )
    with pytest.raises(ProducerReached):
        f25_cli._run(args)
    assert captured["metadata_path"] == str(bound_metadata)
    assert captured["output_dir"] == str(bound_output)
    assert captured["config_path"] == config_path.resolve()
    assert captured["source_cache_manifest_path"] == str(source_manifest)
