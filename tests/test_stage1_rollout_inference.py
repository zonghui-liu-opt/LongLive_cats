from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

from utils.config import normalize_config
from utils.stage1_io import canonical_json_sha256, sha256_file
from utils.stage1_rollout_inference import (
    resolve_stage1_rollout_global_prompt,
    resolve_stage1_rollout_inference_plan,
    resolve_stage1_rollout_sample_shape,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = ROOT / "configs" / "infer_i2v_stage1_teacher_forcing_rollout.yaml"


def _write_checkpoint(tmp_path: Path, *, step: int = 4500):
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "converted_causal_base.pt"
    base.write_bytes(b"causal-stage1-base")
    checkpoint = tmp_path / f"checkpoint_model_{step:06d}"
    checkpoint.mkdir()
    (checkpoint / "adapter_ema.safetensors").write_bytes(b"ema-adapter")
    (checkpoint / "adapter_raw.safetensors").write_bytes(b"raw-adapter")
    (checkpoint / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    (checkpoint / "base_reference.json").write_text(
        json.dumps({"base_sha256": sha256_file(base)}), encoding="utf-8"
    )
    resolved = {
        "teacher_forcing": True,
        "algorithm": {
            "i2v": True,
            "causal": True,
            "teacher_forcing": True,
            "independent_first_frame": True,
        },
        "num_frame_per_block": 8,
        "model_kwargs": {"num_frame_per_block": 8},
        "error_recycling": {"enabled": True},
        "expected_num_samples": 600,
        "batch_size": 1,
        "data_parallel_size": 2,
        "gradient_accumulation_steps": 2,
        "phases": [
            {
                "name": "phase_a_teacher_forcing",
                "epochs": 10,
                "error_recycling": {"mode": "collect_only"},
            },
            {
                "name": "phase_b_error_recycling",
                "epochs": 20,
                "error_recycling": {"mode": "collect_and_inject"},
            },
        ],
    }
    (checkpoint / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    files = []
    for name in (
        "adapter_ema.safetensors",
        "adapter_raw.safetensors",
        "base_reference.json",
        "resolved_config.yaml",
    ):
        path = checkpoint / name
        files.append(
            {"name": name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = {
        "schema": "longlive_stage1_lora_checkpoint",
        "schema_version": 1,
        "completed_step": step,
        "next_update_index": step,
        "resumable": False,
        "topology": {
            "world_size": 6,
            "sequence_parallel_size": 3,
            "data_parallel_size": 2,
        },
        "files": files,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return base, checkpoint


def _refresh_manifest(checkpoint: Path, *, omit: str | None = None) -> None:
    path = checkpoint / "checkpoint_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256", None)
    entries = []
    for entry in manifest["files"]:
        name = entry["name"]
        if name == omit:
            continue
        artifact = checkpoint / name
        entries.append(
            {
                "name": name,
                "size": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        )
    manifest["files"] = entries
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _config(base: Path, checkpoint: Path, *, expected_step: int):
    return SimpleNamespace(
        stage1_rollout={
            "enabled": True,
            "profile": {"sampling_steps": 4},
            "checkpoint": {
                "require_ema": True,
                "verify_hashes": True,
                "expected_completed_step": expected_step,
                "expected_phase_epochs": [10, 20],
            },
        },
        i2v=True,
        num_samples=1,
        model_quant=False,
        fp8_quant=False,
        kv_quant=False,
        merge_lora=False,
        use_ema=False,
        generator_ckpt=str(base),
        lora_ckpt=str(checkpoint / "adapter_ema.safetensors"),
        adapter={
            "type": "lora",
            "rank": 32,
            "alpha": 32,
            "dropout": 0.0,
            "bias": "none",
            "expected_target_modules": 180,
            "expected_trainable_parameters": 57016320,
            "expected_adapter_tensors": 360,
        },
    )


def test_global_prompt_rejects_silent_per_block_conditioning_drift():
    assert resolve_stage1_rollout_global_prompt(["cat", "cat", "cat"]) == "cat"
    with pytest.raises(ValueError, match="one global prompt"):
        resolve_stage1_rollout_global_prompt(["cat starts", "cat lands"])
    with pytest.raises(ValueError, match="non-empty global prompt"):
        resolve_stage1_rollout_global_prompt([])


def test_rollout_sample_shape_follows_each_metadata_orientation():
    configured = [1, 24, 48, 30, 52]

    assert resolve_stage1_rollout_sample_shape(
        configured,
        conditioning_image_size=(480, 832),
        spatial_compression_ratio=16,
        frame_seq_length=390,
    ) == (1, 24, 48, 30, 52)
    assert resolve_stage1_rollout_sample_shape(
        configured,
        conditioning_image_size=(832, 480),
        spatial_compression_ratio=16,
        frame_seq_length=390,
    ) == (1, 24, 48, 52, 30)


@pytest.mark.parametrize(
    ("configured", "image_size", "frame_seq_length", "message"),
    [
        ([1, 24, 32, 30, 52], (480, 832), 390, "B/T/C"),
        ([1, 24, 48, 30, 52], (512, 512), 390, "canonical geometry"),
        ([1, 24, 48, 30, 51], (480, 816), 390, "spatial patch"),
        ([1, 24, 48, 30, 52], (480, 832), 391, "frame_seq_length"),
    ],
)
def test_rollout_sample_shape_rejects_incompatible_geometry(
    configured, image_size, frame_seq_length, message
):
    with pytest.raises(ValueError, match=message):
        resolve_stage1_rollout_sample_shape(
            configured,
            conditioning_image_size=image_size,
            spatial_compression_ratio=16,
            frame_seq_length=frame_seq_length,
        )


def test_formal_checkpoint_preflight_binds_ema_base_hash_and_completed_phases(
    tmp_path,
):
    base, checkpoint = _write_checkpoint(tmp_path, step=4500)
    plan = resolve_stage1_rollout_inference_plan(
        _config(base, checkpoint, expected_step=4500)
    )

    assert plan.profile.sampling_steps == 4
    assert plan.checkpoint_provenance["runtime_mode"] == "dynamic_stage1_lora"
    assert plan.checkpoint_provenance["base"]["sha256"] == sha256_file(base)
    assert plan.checkpoint_provenance["adapter"]["variant"] == "ema"
    stage1 = plan.checkpoint_provenance["checkpoint"]
    assert stage1["completed_step"] == 4500
    assert stage1["declared_training_end_step"] == 4500
    assert stage1["phase_status"] == "all_phases_complete"
    assert stage1["all_declared_phases_complete"] is True
    assert stage1["updates_per_epoch"] == 150
    assert stage1["training_block_size"] == 8


def test_formal_omegaconf_yaml_phase_epochs_reaches_checkpoint_preflight(
    tmp_path, monkeypatch
):
    base, checkpoint = _write_checkpoint(tmp_path, step=4500)
    environment = {
        "LONG_LIVE_STAGE1_ARCHITECTURE_ROOT": str(tmp_path / "architecture"),
        "LONG_LIVE_STAGE1_T5_CHECKPOINT": str(tmp_path / "t5.pth"),
        "LONG_LIVE_STAGE1_TOKENIZER_DIR": str(tmp_path / "tokenizer"),
        "LONG_LIVE_STAGE1_VAE_CHECKPOINT": str(tmp_path / "vae.pth"),
        "LONG_LIVE_STAGE1_BASE_CHECKPOINT": str(base),
        "LONG_LIVE_STAGE1_CHECKPOINT_DIR": str(checkpoint),
        "LONG_LIVE_STAGE1_ROLLOUT_INPUT": str(tmp_path / "testset"),
        "LONG_LIVE_STAGE1_ROLLOUT_METADATA": str(tmp_path / "metadata.csv"),
        "LONG_LIVE_STAGE1_ROLLOUT_OUTPUT": str(tmp_path / "outputs"),
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    config = normalize_config(OmegaConf.load(FORMAL_CONFIG))
    phase_epochs = config.stage1_rollout.checkpoint.expected_phase_epochs
    assert OmegaConf.is_list(phase_epochs)
    assert not isinstance(phase_epochs, (list, tuple))

    plan = resolve_stage1_rollout_inference_plan(config)
    checkpoint_provenance = plan.checkpoint_provenance["checkpoint"]
    assert checkpoint_provenance["phase_status"] == "all_phases_complete"
    assert checkpoint_provenance["declared_training_end_step"] == 4500


def test_preflight_accepts_tuple_phase_epochs(tmp_path):
    base, checkpoint = _write_checkpoint(tmp_path, step=4500)
    config = _config(base, checkpoint, expected_step=4500)
    config.stage1_rollout["checkpoint"]["expected_phase_epochs"] = (10, 20)

    plan = resolve_stage1_rollout_inference_plan(config)
    assert plan.checkpoint_provenance["checkpoint"]["phase_status"] == (
        "all_phases_complete"
    )


@pytest.mark.parametrize(
    "phase_epochs",
    [
        "10,20",
        b"10,20",
        None,
        10,
        {"phase_a": 10, "phase_b": 20},
        [10],
        [10, 20, 30],
        [0, 20],
        [-1, 20],
        [True, 20],
        [10, True],
        [10, 20.0],
        ["10", 20],
    ],
)
def test_preflight_rejects_invalid_phase_epoch_contract(tmp_path, phase_epochs):
    base, checkpoint = _write_checkpoint(tmp_path, step=4500)
    config = _config(base, checkpoint, expected_step=4500)
    config.stage1_rollout["checkpoint"]["expected_phase_epochs"] = phase_epochs

    with pytest.raises(ValueError, match="must be two positive integers"):
        resolve_stage1_rollout_inference_plan(config)


def test_3750_checkpoint_is_labeled_phase_b_in_progress(tmp_path):
    base, checkpoint = _write_checkpoint(tmp_path, step=3750)
    plan = resolve_stage1_rollout_inference_plan(
        _config(base, checkpoint, expected_step=3750)
    )
    stage1 = plan.checkpoint_provenance["checkpoint"]
    assert stage1["phase_status"] == "phase_b_error_recycling_in_progress"
    assert stage1["all_declared_phases_complete"] is False


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (lambda cfg: setattr(cfg, "i2v", False), ValueError, "i2v=true"),
        (lambda cfg: setattr(cfg, "num_samples", 2), ValueError, "num_samples=1"),
        (lambda cfg: setattr(cfg, "kv_quant", True), ValueError, "kv_quant=false"),
        (lambda cfg: setattr(cfg, "merge_lora", True), ValueError, "dynamic LoRA"),
        (lambda cfg: setattr(cfg, "use_ema", True), ValueError, "use_ema"),
        (
            lambda cfg: cfg.adapter.__setitem__("rank", 128),
            ValueError,
            "adapter.rank",
        ),
        (
            lambda cfg: cfg.stage1_rollout["checkpoint"].__setitem__(
                "expected_completed_step", 3750
            ),
            RuntimeError,
            "step mismatch",
        ),
    ],
)
def test_preflight_rejects_runtime_checkpoint_and_adapter_drift(
    tmp_path, mutation, error, message
):
    base, checkpoint = _write_checkpoint(tmp_path)
    config = _config(base, checkpoint, expected_step=4500)
    mutation(config)
    with pytest.raises(error, match=message):
        resolve_stage1_rollout_inference_plan(config)


def test_preflight_rejects_raw_adapter_incomplete_bundle_and_hash_drift(tmp_path):
    base, checkpoint = _write_checkpoint(tmp_path)
    config = _config(base, checkpoint, expected_step=4500)
    config.lora_ckpt = str(checkpoint / "adapter_raw.safetensors")
    with pytest.raises(FileNotFoundError, match="adapter_ema"):
        resolve_stage1_rollout_inference_plan(config)

    config.lora_ckpt = str(checkpoint / "adapter_ema.safetensors")
    (checkpoint / "_SUCCESS").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        resolve_stage1_rollout_inference_plan(config)

    (checkpoint / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    (checkpoint / "adapter_ema.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash/size mismatch"):
        resolve_stage1_rollout_inference_plan(config)


def test_preflight_rejects_t2v_wrong_phase_mode_and_unhashed_ema(tmp_path):
    base, checkpoint = _write_checkpoint(tmp_path / "t2v")
    config = _config(base, checkpoint, expected_step=4500)
    resolved_path = checkpoint / "resolved_config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    resolved["algorithm"]["i2v"] = False
    resolved_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    _refresh_manifest(checkpoint)
    with pytest.raises(RuntimeError, match="i2v=true"):
        resolve_stage1_rollout_inference_plan(config)

    base, checkpoint = _write_checkpoint(tmp_path / "wrong_phase")
    config = _config(base, checkpoint, expected_step=4500)
    resolved_path = checkpoint / "resolved_config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    resolved["phases"][1]["error_recycling"]["mode"] = "collect_only"
    resolved_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    _refresh_manifest(checkpoint)
    with pytest.raises(RuntimeError, match="Phase A/B contract mismatch"):
        resolve_stage1_rollout_inference_plan(config)

    base, checkpoint = _write_checkpoint(tmp_path / "unhashed_ema")
    config = _config(base, checkpoint, expected_step=4500)
    _refresh_manifest(checkpoint, omit="adapter_ema.safetensors")
    with pytest.raises(RuntimeError, match="uniquely hash EMA/raw"):
        resolve_stage1_rollout_inference_plan(config)


def test_preflight_rejects_non_c8_training_block_provenance(tmp_path):
    base, checkpoint = _write_checkpoint(tmp_path)
    config = _config(base, checkpoint, expected_step=4500)
    resolved_path = checkpoint / "resolved_config.yaml"
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    resolved["num_frame_per_block"] = 4
    resolved["model_kwargs"]["num_frame_per_block"] = 4
    resolved_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    _refresh_manifest(checkpoint)
    with pytest.raises(RuntimeError, match="num_frame_per_block=8"):
        resolve_stage1_rollout_inference_plan(config)
