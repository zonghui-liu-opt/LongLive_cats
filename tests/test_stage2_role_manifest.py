from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.stage1_checkpoint import write_checkpoint_manifest, write_success_marker
from utils.stage1_io import atomic_write_json, canonical_json_sha256, sha256_file
from utils.stage2_role_manifest import (
    _LOCKED_WAN_ARCHITECTURE,
    ROLE_INIT_COMPLETE_MARKER,
    build_stage2_role_init_manifest,
    build_stage2_teacher_manifest,
    require_stage2_cold_start,
    validate_stage2_generator_manifest,
    validate_stage2_teacher_manifest,
    write_stage2_role_init_artifacts,
)
from utils.stage2_roles import expected_stage2_target_names


def _architecture_root(tmp_path: Path) -> Path:
    root = tmp_path / "architecture"
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(_LOCKED_WAN_ARCHITECTURE), encoding="utf-8"
    )
    return root


def _teacher_manifest(tmp_path: Path) -> tuple[Path, Path, dict, Path]:
    checkpoint = tmp_path / "teacher.pt"
    torch.save(
        {
            "real_score": {
                "model.weight": torch.ones(2, dtype=torch.bfloat16),
            },
            "checkpoint_format": "cat_teacher",
            "checkpoint_version": 1,
        },
        checkpoint,
    )
    architecture_root = _architecture_root(tmp_path)
    path = tmp_path / "teacher.manifest.json"
    manifest = build_stage2_teacher_manifest(
        checkpoint_path=checkpoint,
        architecture_root=architecture_root,
        checkpoint_format="longlive_wrapper_pt",
        state_dict_selector="real_score",
        source_kind="direct_longlive_checkpoint",
        source_identifier="cat-domain-ti2v-teacher",
        source_sha256=sha256_file(checkpoint),
        conversion_command=("none",),
        attest_cat_domain_bidirectional_ti2v=True,
        attest_video_global_flow=True,
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint, path, manifest, architecture_root


def _generator_manifest(tmp_path: Path) -> tuple[Path, Path, dict]:
    base = tmp_path / "stage1-base.pt"
    base.write_bytes(b"stage1-base")
    training = tmp_path / "checkpoint_model_003075"
    training.mkdir()
    raw = training / "adapter_raw.safetensors"
    ema = training / "adapter_ema.safetensors"
    resolved_config = training / "resolved_config.yaml"
    raw.write_bytes(b"raw-adapter")
    ema.write_bytes(b"ema-adapter")
    resolved_config.write_text("stage1: config\n", encoding="utf-8")
    atomic_write_json(
        training / "base_reference.json", {"base_sha256": sha256_file(base)}
    )
    output = tmp_path / "merged.pt"
    output.write_bytes(b"merged-generator")

    def metadata(kind):
        return {
            "schema": "longlive_stage1_lora_checkpoint",
            "schema_version": "1",
            "kind": kind,
            "completed_step": "3075",
            "tensor_count": "360",
            "global_numel": "57016320",
            "dtype": "float32",
        }

    targets = list(expected_stage2_target_names())
    tensor_schema = []
    for target in targets:
        if target.endswith(
            ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o")
        ):
            input_features = output_features = 3072
        elif target.endswith("ffn.0"):
            input_features, output_features = 3072, 14336
        else:
            input_features, output_features = 14336, 3072
        tensor_schema.extend(
            [
                {
                    "key": f"base_model.model.{target}.lora_A.weight",
                    "shape": [32, input_features],
                    "dtype": "float32",
                },
                {
                    "key": f"base_model.model.{target}.lora_B.weight",
                    "shape": [output_features, 32],
                    "dtype": "float32",
                },
            ]
        )
    checkpoint_manifest = write_checkpoint_manifest(
        training,
        completed_step=3075,
        world_size=6,
        sequence_parallel_size=3,
        data_parallel_size=2,
        resumable=False,
    )
    write_success_marker(training, resumable=False)
    value = {
        "schema": "longlive_stage1_merge_manifest",
        "schema_version": 2,
        "checkpoint_format": "longlive_stage1_causal_ema_merged",
        "checkpoint_version": 1,
        "model_name": "Wan2.2-TI2V-5B",
        "base": {
            "path": str(base),
            "sha256": sha256_file(base),
            "size": base.stat().st_size,
        },
        "training_checkpoint": {
            "path": str(training),
            "completed_step": 3075,
            "manifest_sha256": checkpoint_manifest["manifest_sha256"],
            "resolved_config_sha256": sha256_file(resolved_config),
            "adapter": ema.name,
            "adapter_sha256": sha256_file(ema),
            "raw_adapter": {
                "name": raw.name,
                "sha256": sha256_file(raw),
                "size": raw.stat().st_size,
                "metadata": metadata("raw"),
            },
            "ema_adapter": {
                "name": ema.name,
                "sha256": sha256_file(ema),
                "size": ema.stat().st_size,
                "metadata": metadata("ema"),
            },
        },
        "adapter": {
            "type": "lora",
            "rank": 32,
            "alpha": 32,
            "dropout": 0.0,
            "bias": "none",
            "modules_to_save": [],
            "target_patterns": [
                r"^blocks\.[0-9]+\.self_attn\.(q|k|v|o)$",
                r"^blocks\.[0-9]+\.ffn\.(0|2)$",
            ],
            "target_module_names": targets,
            "target_module_count": 180,
            "target_schema_sha256": canonical_json_sha256(targets),
            "tensor_schema": tensor_schema,
            "tensor_schema_sha256": canonical_json_sha256(tensor_schema),
            "adapter_tensor_count": 360,
            "trainable_parameter_count": 57_016_320,
        },
        "adapter_a_b_tensors": 360,
        "target_modules": 180,
        "trainable_parameters": 57_016_320,
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size": output.stat().st_size,
            "dtype": "bfloat16",
            "strict_reload": True,
        },
    }
    value["manifest_sha256"] = canonical_json_sha256(value)
    path = tmp_path / "generator.manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return output, path, value


def test_teacher_manifest_binds_checkpoint_provenance_and_flow_contract(tmp_path):
    checkpoint, path, manifest, architecture_root = _teacher_manifest(tmp_path)
    audit = validate_stage2_teacher_manifest(
        path,
        expected_checkpoint_path=checkpoint,
        expected_architecture_root=architecture_root,
    )
    assert audit["checkpoint_sha256"] == sha256_file(checkpoint)
    assert audit["checkpoint_format"] == "longlive_wrapper_pt"
    assert audit["state_dict_selector"] == "real_score"
    contract = manifest["required_model_contract"]
    assert contract["backbone"] == "bidirectional_ti2v"
    assert contract["prediction_type"] == "flow"
    assert contract["timestep_scope"] == "video_global"
    assert manifest["checkpoint"]["strict_reload_required"] is True
    assert manifest["checkpoint"]["payload_contract"]["load_target"] == "role_wrapper"


def test_longlive_teacher_accepts_legacy_serialization_and_generator_selector(
    tmp_path,
):
    checkpoint = tmp_path / "legacy-teacher.pt"
    torch.save(
        {
            "generator": {"model.weight": torch.ones(2, dtype=torch.bfloat16)},
            "schema": "longlive_teacher_checkpoint",
            "schema_version": 1,
        },
        checkpoint,
        _use_new_zipfile_serialization=False,
    )
    architecture_root = _architecture_root(tmp_path)
    manifest = build_stage2_teacher_manifest(
        checkpoint_path=checkpoint,
        architecture_root=architecture_root,
        checkpoint_format="longlive_wrapper_pt",
        state_dict_selector="generator",
        source_kind="legacy_longlive_checkpoint",
        source_identifier="cat-domain-ti2v-teacher",
        source_sha256=sha256_file(checkpoint),
        conversion_command=("none",),
        attest_cat_domain_bidirectional_ti2v=True,
        attest_video_global_flow=True,
    )
    manifest_path = tmp_path / "legacy-teacher.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    audit = validate_stage2_teacher_manifest(
        manifest_path,
        expected_checkpoint_path=checkpoint,
        expected_architecture_root=architecture_root,
    )
    assert audit["state_dict_selector"] == "generator"


def test_longlive_teacher_rejects_non_bf16_state_before_manifest(tmp_path):
    checkpoint = tmp_path / "fp32-teacher.pt"
    torch.save({"real_score": {"model.weight": torch.ones(2)}}, checkpoint)
    with pytest.raises(TypeError, match="must be BF16"):
        build_stage2_teacher_manifest(
            checkpoint_path=checkpoint,
            architecture_root=_architecture_root(tmp_path),
            checkpoint_format="longlive_wrapper_pt",
            state_dict_selector="real_score",
            source_kind="direct_longlive_checkpoint",
            source_identifier="cat-domain-ti2v-teacher",
            source_sha256=sha256_file(checkpoint),
            conversion_command=("none",),
            attest_cat_domain_bidirectional_ti2v=True,
            attest_video_global_flow=True,
        )


def test_generator_manifest_requires_enriched_step3075_provenance(
    tmp_path, monkeypatch
):
    checkpoint, path, manifest = _generator_manifest(tmp_path)
    tensor_schema = manifest["adapter"]["tensor_schema"]

    def read_contract(source):
        kind = "raw" if "raw" in source.name else "ema"
        return (
            manifest["training_checkpoint"][f"{kind}_adapter"]["metadata"],
            tensor_schema,
        )

    monkeypatch.setattr(
        "utils.stage2_role_manifest._read_safetensors_contract", read_contract
    )
    audit = validate_stage2_generator_manifest(
        path,
        expected_checkpoint_path=checkpoint,
    )
    assert audit["source_step"] == 3075
    assert audit["adapter"]["target_module_count"] == 180
    assert audit["adapter"]["adapter_tensor_count"] == 360
    assert audit["adapter"]["trainable_parameter_count"] == 57_016_320

    old = copy.deepcopy(manifest)
    old["schema_version"] = 1
    old["manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in old.items() if key != "manifest_sha256"}
    )
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version=2"):
        validate_stage2_generator_manifest(path, expected_checkpoint_path=checkpoint)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["output"].__setitem__("path", "/wrong/merged.pt"),
            "output.path",
        ),
        (lambda value: value.__setitem__("unknown", True), "top-level schema"),
    ],
)
def test_generator_manifest_rejects_path_or_schema_drift(
    tmp_path, monkeypatch, mutation, match
):
    checkpoint, path, manifest = _generator_manifest(tmp_path)
    tensor_schema = manifest["adapter"]["tensor_schema"]

    def read_contract(source):
        kind = "raw" if "raw" in source.name else "ema"
        return (
            manifest["training_checkpoint"][f"{kind}_adapter"]["metadata"],
            tensor_schema,
        )

    monkeypatch.setattr(
        "utils.stage2_role_manifest._read_safetensors_contract", read_contract
    )
    bad = copy.deepcopy(manifest)
    mutation(bad)
    bad["manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in bad.items() if key != "manifest_sha256"}
    )
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        validate_stage2_generator_manifest(path, expected_checkpoint_path=checkpoint)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["checkpoint"].__setitem__("dtype", "float32"), "dtype"),
        (
            lambda value: value["required_model_contract"].__setitem__(
                "timestep_scope", "per_frame"
            ),
            "timestep_scope",
        ),
        (
            lambda value: value["provenance"].__setitem__("source_sha256", ""),
            "source_sha256",
        ),
        (
            lambda value: value["checkpoint"].__setitem__("path", "/wrong/teacher.pt"),
            "checkpoint.path",
        ),
        (lambda value: value.__setitem__("unknown", True), "top-level schema"),
    ],
)
def test_teacher_manifest_rejects_semantic_or_provenance_drift(
    tmp_path, mutation, match
):
    checkpoint, path, manifest, architecture_root = _teacher_manifest(tmp_path)
    bad = copy.deepcopy(manifest)
    mutation(bad)
    bad["manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in bad.items() if key != "manifest_sha256"}
    )
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match=match):
        validate_stage2_teacher_manifest(
            path,
            expected_checkpoint_path=checkpoint,
            expected_architecture_root=architecture_root,
        )


def test_teacher_manifest_detects_self_hash_and_checkpoint_tampering(tmp_path):
    checkpoint, path, manifest, architecture_root = _teacher_manifest(tmp_path)
    bad = copy.deepcopy(manifest)
    bad["checkpoint"]["source_files"][0]["size"] += 1
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="self hash"):
        validate_stage2_teacher_manifest(
            path,
            expected_checkpoint_path=checkpoint,
            expected_architecture_root=architecture_root,
        )
    path.write_text(json.dumps(manifest), encoding="utf-8")
    tampered = bytearray(checkpoint.read_bytes())
    tampered[-1] ^= 1
    checkpoint.write_bytes(tampered)
    with pytest.raises(RuntimeError, match="source file hashes"):
        validate_stage2_teacher_manifest(
            path,
            expected_checkpoint_path=checkpoint,
            expected_architecture_root=architecture_root,
        )


def test_native_teacher_binds_index_all_shards_and_source_bf16(tmp_path):
    from safetensors.torch import save_file

    architecture_root = _architecture_root(tmp_path)
    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"a": torch.ones(2, dtype=torch.bfloat16)}, shard_a)
    save_file({"b": torch.ones(3, dtype=torch.bfloat16)}, shard_b)
    index = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": shard_a.name,
                    "b": shard_b.name,
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = build_stage2_teacher_manifest(
        checkpoint_path=index,
        architecture_root=architecture_root,
        checkpoint_format="wan_native_transformer",
        state_dict_selector="root",
        source_kind="converted_native_wan",
        source_identifier="cat-domain-teacher",
        source_sha256="a" * 64,
        conversion_command=("convert_teacher.py",),
        attest_cat_domain_bidirectional_ti2v=True,
        attest_video_global_flow=True,
    )
    manifest_path = tmp_path / "native.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    audit = validate_stage2_teacher_manifest(
        manifest_path,
        expected_checkpoint_path=index,
        expected_architecture_root=architecture_root,
    )
    assert len(audit["checkpoint_files"]) == 3
    assert audit["native_tensor_contract"]["source_dtype"] == "bfloat16"

    save_file({"b": torch.zeros(3, dtype=torch.bfloat16)}, shard_b)
    with pytest.raises(RuntimeError, match="source file hashes"):
        validate_stage2_teacher_manifest(
            manifest_path,
            expected_checkpoint_path=index,
            expected_architecture_root=architecture_root,
        )


def test_native_teacher_rejects_non_bf16_source_before_manifest(tmp_path):
    from safetensors.torch import save_file

    checkpoint = tmp_path / "teacher.safetensors"
    save_file({"weight": torch.ones(2, dtype=torch.float32)}, checkpoint)
    with pytest.raises(TypeError, match="must be BF16"):
        build_stage2_teacher_manifest(
            checkpoint_path=checkpoint,
            architecture_root=_architecture_root(tmp_path),
            checkpoint_format="wan_native_transformer",
            state_dict_selector="root",
            source_kind="converted_native_wan",
            source_identifier="cat-domain-teacher",
            source_sha256="a" * 64,
            conversion_command=("convert_teacher.py",),
            attest_cat_domain_bidirectional_ti2v=True,
            attest_video_global_flow=True,
        )


def test_resume_is_rejected_before_any_asset_path_is_read():
    resumed = SimpleNamespace(
        initialization_mode="resume_stage2",
        resume_stage2_checkpoint="/definitely/not/read/checkpoint",
    )
    with pytest.raises(NotImplementedError, match="Step 10"):
        require_stage2_cold_start(resumed)


def _clean_side_effect_audit():
    return {
        "tripwires_enforced": [
            "module_call_impl",
            "direct_module_forward",
            "optimizer",
            "ema",
            "text_encoder",
            "vae",
            "dataloader",
        ],
        "forward_calls": 0,
        "optimizer_created": False,
        "ema_created": False,
        "text_encoder_created": False,
        "vae_created": False,
        "dataloader_created": False,
    }


def _role_init_audits():
    values = {}
    for role, rank, targets, tensors, parameters, base_sha in (
        ("generator", 32, 180, 360, 57_016_320, "d" * 64),
        ("real_score", None, 0, 0, 0, "e" * 64),
        ("fake_score", 64, 180, 360, 114_032_640, "e" * 64),
    ):
        values[role] = {
            "base_checkpoint_sha256": base_sha,
            "strict_reload_succeeded": True,
            "backbone": "causal" if role == "generator" else "bidirectional_ti2v",
            "base_dtype": "bfloat16",
            "activation_checkpointing": role == "fake_score",
            "trainable_policy": "frozen" if role == "real_score" else "adapter_only",
            "adapter_digest": None if role == "real_score" else "9" * 64,
            "target_audit": (
                None
                if role == "real_score"
                else {
                    "rank": rank,
                    "target_module_count": targets,
                    "adapter_tensor_count": tensors,
                    "trainable_parameter_count": parameters,
                    "target_module_names": expected_stage2_target_names(),
                }
            ),
            "canonical_adapter_keys": tuple(f"key-{index}" for index in range(tensors)),
            "pre_fsdp_adapter_tensor_count": tensors,
            "pre_fsdp_trainable_parameters": parameters,
            "post_fsdp": {
                "trainable_tensor_count": tensors,
                "global_trainable_parameters": parameters,
                "all_parameters_are_dtensor": True,
                "fsdp_module_count": 31,
                "root_and_30_blocks_independently_wrapped": True,
                "mesh_shape": (8,),
                "mesh_dim_names": ("shard",),
                "placements": ("shard:0",),
                "frozen_tensor_count": 900,
            },
        }
    return values


def _role_init_manifest_kwargs():
    return {
        "contract_hash": "a" * 64,
        "launch_hash": "b" * 64,
        "generator_asset": {"checkpoint_sha256": "d" * 64},
        "real_score_asset": {"checkpoint_sha256": "e" * 64},
        "role_audits": _role_init_audits(),
        "role_isolation_audit": {
            "parameter_objects_disjoint": True,
            "parameter_storage_disjoint": True,
            "roles": ("generator", "real_score", "fake_score"),
        },
        "fsdp_audits": {
            "world_size": 8,
            "sequence_parallel_size": 1,
            "data_parallel_size": 8,
            "mesh_shape": (8,),
            "mesh_dim_names": ("shard",),
            "sharding_strategy": "FULL_SHARD",
        },
        "rank_consensus_sha256": "f" * 64,
        "side_effect_audit": _clean_side_effect_audit(),
    }


def test_role_init_manifest_is_stable_self_hashed_and_not_a_training_checkpoint(
    tmp_path,
):
    kwargs = _role_init_manifest_kwargs()
    payload = build_stage2_role_init_manifest(**kwargs)
    assert payload == build_stage2_role_init_manifest(**kwargs)
    assert payload["side_effects"] == _clean_side_effect_audit()

    output = tmp_path / "role-init"
    write_stage2_role_init_artifacts(output, payload)
    assert (output / "role_init_manifest.json").is_file()
    assert (output / ROLE_INIT_COMPLETE_MARKER).is_file()
    assert not (output / "_SUCCESS").exists()
    with pytest.raises(FileExistsError):
        write_stage2_role_init_artifacts(output, payload)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["role_audits"]["generator"].__setitem__(
                "base_checkpoint_sha256", "0" * 64
            ),
            "asset/role checkpoint SHA",
        ),
        (
            lambda value: value["role_audits"]["fake_score"].__setitem__(
                "activation_checkpointing", False
            ),
            "activation-checkpoint",
        ),
        (
            lambda value: value["side_effect_audit"].__setitem__(
                "optimizer_created", True
            ),
            "side-effect",
        ),
        (
            lambda value: value["role_audits"]["generator"]["post_fsdp"].__setitem__(
                "all_parameters_are_dtensor", False
            ),
            "post-FSDP",
        ),
        (
            lambda value: value["role_isolation_audit"].__setitem__(
                "parameter_storage_disjoint", False
            ),
            "role isolation",
        ),
    ],
)
def test_role_init_manifest_rejects_unproven_audits(mutate, match):
    kwargs = _role_init_manifest_kwargs()
    mutate(kwargs)
    with pytest.raises(ValueError, match=match):
        build_stage2_role_init_manifest(**kwargs)


def test_role_init_writer_rejects_tampered_self_hash(tmp_path):
    payload = build_stage2_role_init_manifest(**_role_init_manifest_kwargs())
    payload["rank_consensus_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="self hash"):
        write_stage2_role_init_artifacts(tmp_path / "must-not-exist", payload)
    assert not (tmp_path / "must-not-exist").exists()


def test_role_init_writer_cleans_temporary_directory_on_publication_failure(
    tmp_path, monkeypatch
):
    payload = build_stage2_role_init_manifest(**_role_init_manifest_kwargs())

    def fail_write(*args, **kwargs):
        del args, kwargs
        raise OSError("injected manifest write failure")

    monkeypatch.setattr("utils.stage2_role_manifest.atomic_write_json", fail_write)
    output = tmp_path / "must-not-exist"
    with pytest.raises(OSError, match="injected"):
        write_stage2_role_init_artifacts(output, payload)
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*"))
