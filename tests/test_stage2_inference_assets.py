from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from utils.stage1_io import (
    aggregate_file_hash,
    canonical_json_sha256,
    sha256_file,
    tree_file_hashes,
)
from utils.stage2_checkpoint import (
    STAGE2_CHECKPOINT_SCHEMA,
    STAGE2_CHECKPOINT_SCHEMA_VERSION,
    STAGE2_PROVENANCE_SCHEMA,
    STAGE2_PROVENANCE_SCHEMA_VERSION,
)
from utils.stage2_inference_assets import (
    assert_stage2_runtime_asset_identities,
    build_stage2_runtime_assets,
    validate_stage2_runtime_asset_identity,
    validate_stage2_runtime_assets,
)
from utils.stage2_inference_config import ResolvedStage2InferenceConfig


def _file_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _model_entry(path: Path) -> dict[str, Any]:
    files = tree_file_hashes(path)
    return {"files": files, "aggregate_sha256": aggregate_file_hash(files)}


def _write_json(path: Path, value: Any) -> bytes:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(data)
    return data


def _fixture(
    tmp_path: Path,
) -> tuple[ResolvedStage2InferenceConfig, dict[str, Any], dict[str, int]]:
    t5 = tmp_path / "t5.pth"
    t5.write_bytes(b"t5-production-weights")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_bytes(b'{"vocab":"fixture"}')
    (tokenizer / "special_tokens_map.json").write_bytes(b'{"eos":"</s>"}')
    vae = tmp_path / "vae.pth"
    vae.write_bytes(b"vae-production-weights")
    architecture = tmp_path / "architecture"
    architecture.mkdir()
    architecture_file = architecture / "config.json"
    architecture_file.write_bytes(b'{"dim":3072}')
    generator_base = tmp_path / "stage1_step3075.pt"
    generator_base.write_bytes(b"merged-generator-base")
    generator_manifest = tmp_path / "stage1_step3075.manifest.json"
    generator_manifest.write_bytes(b'{"fixture":true}')

    source_body = {
        "schema": "longlive_stage1_cache_manifest/v1",
        "source_fingerprint": {
            "models": {
                "t5_checkpoint": _model_entry(t5),
                "tokenizer_dir": _model_entry(tokenizer),
                "vae_checkpoint": _model_entry(vae),
            }
        },
    }
    source = {
        **source_body,
        "manifest_sha256": canonical_json_sha256(source_body),
    }
    source_path = tmp_path / "source-cache-manifest.json"
    _write_json(source_path, source)

    generator_file = {
        "name": generator_base.name,
        "path": str(generator_base.resolve()),
        "size": generator_base.stat().st_size,
        "sha256": sha256_file(generator_base),
        "identity": _file_identity(generator_base),
    }
    generator_asset = {
        "manifest_path": str(generator_manifest.resolve()),
        "checkpoint_path": str(generator_base.resolve()),
        "checkpoint_sha256": sha256_file(generator_base),
        "checkpoint_files": [generator_file],
        "checkpoint_format": "longlive_stage1_causal_ema_merged",
        "state_dict_selector": "generator",
        "source_step": 3075,
        "architecture_file": {
            "name": "config.json",
            "path": str(architecture_file.resolve()),
            "size": architecture_file.stat().st_size,
            "sha256": sha256_file(architecture_file),
            "identity": _file_identity(architecture_file),
        },
    }
    provenance = {
        "schema": STAGE2_PROVENANCE_SCHEMA,
        "schema_version": STAGE2_PROVENANCE_SCHEMA_VERSION,
        "code_version": {"stage2_source_sha256": "a" * 64},
        "assets": {
            "generator": generator_asset,
            "real_score": {"checkpoint_sha256": "b" * 64},
            "fake_score": {"checkpoint_sha256": "b" * 64},
        },
        "data": {
            "stage2_manifest_sha256": "c" * 64,
            "source_manifest_sha256": source["manifest_sha256"],
            "negative_manifest_sha256": "d" * 64,
            "negative_artifact_sha256": "e" * 64,
        },
        "lineage": {
            "parent_checkpoint": None,
            "parent_checkpoint_manifest_sha256": None,
        },
        "smoke_probe": None,
    }
    checkpoint = tmp_path / "checkpoint_stage2_g000040"
    checkpoint.mkdir()
    (checkpoint / "_SUCCESS").touch()
    provenance_bytes = _write_json(checkpoint / "provenance.json", provenance)
    manifest_body = {
        "schema": STAGE2_CHECKPOINT_SCHEMA,
        "schema_version": STAGE2_CHECKPOINT_SCHEMA_VERSION,
        "provenance_sha256": canonical_json_sha256(provenance),
        "files": [
            {
                "name": "provenance.json",
                "size": len(provenance_bytes),
                "sha256": hashlib.sha256(provenance_bytes).hexdigest(),
            }
        ],
    }
    _write_json(
        checkpoint / "checkpoint_manifest.json",
        {**manifest_body, "manifest_sha256": canonical_json_sha256(manifest_body)},
    )
    single = tmp_path / "single.csv"
    two = tmp_path / "two.csv"
    single.write_text("single\n", encoding="utf-8")
    two.write_text("two\n", encoding="utf-8")
    config = ResolvedStage2InferenceConfig(
        schema="longlive_stage2_inference/v1",
        stage2_checkpoint=str(checkpoint.resolve()),
        source_cache_manifest=str(source_path.resolve()),
        architecture_root=str(architecture.resolve()),
        t5_checkpoint=str(t5.resolve()),
        tokenizer_dir=str(tokenizer.resolve()),
        vae_checkpoint=str(vae.resolve()),
        single_metadata=str(single.resolve()),
        two_action_metadata=str(two.resolve()),
        output_root=str((tmp_path / "output").resolve()),
        profiles=("baseline_c8w16k4s1",),
        seeds=(1, 2, 3, 4),
        dtype="bfloat16",
        cfg_scale=1.0,
        fps=24,
        merge_ema_lora=True,
        batch_size_per_device=1,
    )
    calls = {"manifest": 0}

    def validate_generator_manifest(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["manifest"] += 1
        assert kwargs["expected_checkpoint_path"] == str(generator_base.resolve())
        assert kwargs["expected_step"] == 3075
        return {
            key: value
            for key, value in generator_asset.items()
            if key != "architecture_file"
        }

    return config, {"validator": validate_generator_manifest}, calls


def test_rank0_runtime_asset_attestation_authenticates_production_shape(
    tmp_path: Path,
) -> None:
    config, fixture, calls = _fixture(tmp_path)
    assets = build_stage2_runtime_assets(
        config,
        validate_generator_manifest_fn=fixture["validator"],
    )

    assert calls["manifest"] == 1
    assert validate_stage2_runtime_assets(assets) == assets
    assert (
        validate_stage2_runtime_asset_identity(assets["identity"]) == assets["identity"]
    )
    assert assets["identity"]["checkpoint"]["directory"] == (config.stage2_checkpoint)
    assert set(assets["identity"]["assets"]) == {
        "t5_checkpoint",
        "tokenizer_dir",
        "vae_checkpoint",
        "architecture_config",
        "generator_base",
    }
    assert_stage2_runtime_asset_identities(assets, include_source_manifest=True)


def test_same_path_equal_size_model_replacement_is_rejected_before_attestation(
    tmp_path: Path,
) -> None:
    config, fixture, _ = _fixture(tmp_path)
    t5 = Path(config.t5_checkpoint)
    original = t5.read_bytes()
    t5.write_bytes(b"X" * len(original))

    with pytest.raises(RuntimeError, match="hash/size differs from provenance"):
        build_stage2_runtime_assets(
            config,
            validate_generator_manifest_fn=fixture["validator"],
        )


def test_runtime_assets_enforce_real_t5_tokenizer_and_vae_path_kinds(
    tmp_path: Path,
) -> None:
    config, fixture, _ = _fixture(tmp_path)

    with pytest.raises(RuntimeError, match="tokenizer_dir must be a regular directory"):
        build_stage2_runtime_assets(
            replace(config, tokenizer_dir=config.t5_checkpoint),
            validate_generator_manifest_fn=fixture["validator"],
        )
    with pytest.raises(RuntimeError, match="t5_checkpoint must be a regular file"):
        build_stage2_runtime_assets(
            replace(config, t5_checkpoint=config.tokenizer_dir),
            validate_generator_manifest_fn=fixture["validator"],
        )
    with pytest.raises(RuntimeError, match="vae_checkpoint must be a regular file"):
        build_stage2_runtime_assets(
            replace(config, vae_checkpoint=config.tokenizer_dir),
            validate_generator_manifest_fn=fixture["validator"],
        )


def test_stat_identity_closure_rejects_mutation_after_rank0_authentication(
    tmp_path: Path,
) -> None:
    config, fixture, _ = _fixture(tmp_path)
    assets = build_stage2_runtime_assets(
        config,
        validate_generator_manifest_fn=fixture["validator"],
    )
    vae = Path(config.vae_checkpoint)
    vae.write_bytes(b"Y" * vae.stat().st_size)

    with pytest.raises(RuntimeError, match="vae_checkpoint identity changed"):
        assert_stage2_runtime_asset_identities(assets, names=("vae_checkpoint",))


def test_detailed_attestation_rejects_resigned_file_aggregate_disagreement(
    tmp_path: Path,
) -> None:
    config, fixture, _ = _fixture(tmp_path)
    assets = deepcopy(
        build_stage2_runtime_assets(
            config,
            validate_generator_manifest_fn=fixture["validator"],
        )
    )
    assets["assets"]["t5_checkpoint"]["files"][0]["sha256"] = "f" * 64
    unsigned = dict(assets)
    unsigned.pop("attestation_sha256")
    assets["attestation_sha256"] = canonical_json_sha256(unsigned)

    with pytest.raises(RuntimeError, match="aggregate SHA256 differs"):
        validate_stage2_runtime_assets(assets)

    duplicated = deepcopy(
        build_stage2_runtime_assets(
            config,
            validate_generator_manifest_fn=fixture["validator"],
        )
    )
    tokenizer_files = duplicated["assets"]["tokenizer_dir"]["files"]
    tokenizer_files[1]["relative_path"] = tokenizer_files[0]["relative_path"]
    unsigned = dict(duplicated)
    unsigned.pop("attestation_sha256")
    duplicated["attestation_sha256"] = canonical_json_sha256(unsigned)
    with pytest.raises(RuntimeError, match="sorted and unique"):
        validate_stage2_runtime_assets(duplicated)


def test_checkpoint_provenance_rejects_rehashed_source_manifest_replacement(
    tmp_path: Path,
) -> None:
    config, fixture, _ = _fixture(tmp_path)
    source = Path(config.source_cache_manifest)
    changed_body = {
        "schema": "longlive_stage1_cache_manifest/v1",
        "source_fingerprint": {"models": {}},
    }
    _write_json(
        source,
        {
            **changed_body,
            "manifest_sha256": canonical_json_sha256(changed_body),
        },
    )

    with pytest.raises(RuntimeError, match="differs from checkpoint provenance"):
        build_stage2_runtime_assets(
            config,
            validate_generator_manifest_fn=fixture["validator"],
        )
