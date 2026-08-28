from __future__ import annotations

import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from utils.config import normalize_config
from utils.stage1_rollout_profile import Stage1RolloutProfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "infer_i2v_stage1_teacher_forcing_rollout.yaml"
LAUNCHER = ROOT / "infer_stage1_teacher_forcing_rollout.sh"


def test_example_config_resolves_formal_ema_profile(monkeypatch):
    env = {
        "LONG_LIVE_STAGE1_ARCHITECTURE_ROOT": "/assets/architecture",
        "LONG_LIVE_STAGE1_T5_CHECKPOINT": "/assets/t5.pth",
        "LONG_LIVE_STAGE1_TOKENIZER_DIR": "/assets/tokenizer",
        "LONG_LIVE_STAGE1_VAE_CHECKPOINT": "/assets/vae.pth",
        "LONG_LIVE_STAGE1_BASE_CHECKPOINT": "/assets/base.pt",
        "LONG_LIVE_STAGE1_CHECKPOINT_DIR": "/assets/checkpoint_model_004500",
        "LONG_LIVE_STAGE1_ROLLOUT_INPUT": "/inputs",
        "LONG_LIVE_STAGE1_ROLLOUT_METADATA": "/inputs/metadata_20cases_480x832.csv",
        "LONG_LIVE_STAGE1_ROLLOUT_OUTPUT": "/outputs",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    normalized = normalize_config(OmegaConf.load(CONFIG))
    assert normalized.generator_ckpt == "/assets/base.pt"
    assert normalized.lora_ckpt.endswith(
        "checkpoint_model_004500/adapter_ema.safetensors"
    )
    assert normalized.data_path == "/inputs"
    assert normalized.metadata_path == "/inputs/metadata_20cases_480x832.csv"
    assert list(normalized.image_or_video_shape) == [1, 24, 48, 30, 52]
    config = OmegaConf.to_container(normalized, resolve=True)
    profile = Stage1RolloutProfile.from_mapping(config["stage1_rollout"]["profile"])
    assert profile == Stage1RolloutProfile()
    assert "negative_prompt" not in config["stage1_rollout"]["profile"]
    assert config["checkpoints"]["generator_ckpt"] == "/assets/base.pt"
    assert config["checkpoints"]["lora_ckpt"].endswith(
        "checkpoint_model_004500/adapter_ema.safetensors"
    )
    assert config["adapter"]["rank"] == config["adapter"]["alpha"] == 32
    assert config["adapter"]["expected_target_modules"] == 180
    assert config["adapter"]["expected_adapter_tensors"] == 360
    assert config["model_kwargs"]["init_weights"] is False
    assert config["stage1_rollout"]["checkpoint"]["expected_completed_step"] == 4500
    assert config["stage1_rollout"]["checkpoint"]["expected_phase_epochs"] == [
        10,
        20,
    ]
    assert config["inference"]["streaming_vae"] is False
    assert config["inference"]["multi_shot_sink"] is False


def test_example_config_keeps_legacy_directory_fallback_without_metadata(monkeypatch):
    env = {
        "LONG_LIVE_STAGE1_ARCHITECTURE_ROOT": "/assets/architecture",
        "LONG_LIVE_STAGE1_T5_CHECKPOINT": "/assets/t5.pth",
        "LONG_LIVE_STAGE1_TOKENIZER_DIR": "/assets/tokenizer",
        "LONG_LIVE_STAGE1_VAE_CHECKPOINT": "/assets/vae.pth",
        "LONG_LIVE_STAGE1_BASE_CHECKPOINT": "/assets/base.pt",
        "LONG_LIVE_STAGE1_CHECKPOINT_DIR": "/assets/checkpoint_model_004500",
        "LONG_LIVE_STAGE1_ROLLOUT_INPUT": "/inputs",
        "LONG_LIVE_STAGE1_ROLLOUT_OUTPUT": "/outputs",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("LONG_LIVE_STAGE1_ROLLOUT_METADATA", raising=False)

    normalized = normalize_config(OmegaConf.load(CONFIG))
    assert normalized.metadata_path is None


def test_launcher_is_executable_single_gpu_and_forwards_complete_profile():
    assert os.access(LAUNCHER, os.X_OK)
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    for field in (
        "chunk_size",
        "window_size",
        "sampling_steps",
        "timestep_shift",
        "guidance_scale",
        "expected_step",
    ):
        assert f"--stage1_rollout_{field}" in text
    assert "adapter_ema.safetensors" in text
    assert "adapter_raw.safetensors" in text
    assert "_SUCCESS" in text
    assert "exactly one visible GPU" in text
    assert "LONG_LIVE_STAGE1_ROLLOUT_METADATA" in text
    assert "metadata file does not exist" in text


def test_launcher_requires_metadata_before_checkpoint_validation(tmp_path):
    input_root = tmp_path / "testset"
    input_root.mkdir()
    environment = os.environ.copy()
    environment.pop("LONG_LIVE_STAGE1_ROLLOUT_METADATA", None)
    environment.update(
        {
            "LONG_LIVE_STAGE1_ARCHITECTURE_ROOT": str(tmp_path / "architecture"),
            "LONG_LIVE_STAGE1_T5_CHECKPOINT": str(tmp_path / "t5.pth"),
            "LONG_LIVE_STAGE1_TOKENIZER_DIR": str(tmp_path / "tokenizer"),
            "LONG_LIVE_STAGE1_VAE_CHECKPOINT": str(tmp_path / "vae.pth"),
            "LONG_LIVE_STAGE1_BASE_CHECKPOINT": str(tmp_path / "base.pt"),
            "LONG_LIVE_STAGE1_CHECKPOINT_DIR": str(tmp_path / "checkpoint"),
            "LONG_LIVE_STAGE1_ROLLOUT_INPUT": str(input_root),
            "LONG_LIVE_STAGE1_ROLLOUT_OUTPUT": str(tmp_path / "outputs"),
        }
    )

    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert (
        "Missing required environment variable: LONG_LIVE_STAGE1_ROLLOUT_METADATA"
        in completed.stderr
    )
    assert "Incomplete Stage-1 checkpoint" not in completed.stderr


def test_launcher_rejects_configured_missing_metadata_before_checkpoint(tmp_path):
    input_root = tmp_path / "testset"
    input_root.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "LONG_LIVE_STAGE1_ARCHITECTURE_ROOT": str(tmp_path / "architecture"),
            "LONG_LIVE_STAGE1_T5_CHECKPOINT": str(tmp_path / "t5.pth"),
            "LONG_LIVE_STAGE1_TOKENIZER_DIR": str(tmp_path / "tokenizer"),
            "LONG_LIVE_STAGE1_VAE_CHECKPOINT": str(tmp_path / "vae.pth"),
            "LONG_LIVE_STAGE1_BASE_CHECKPOINT": str(tmp_path / "base.pt"),
            "LONG_LIVE_STAGE1_CHECKPOINT_DIR": str(tmp_path / "checkpoint"),
            "LONG_LIVE_STAGE1_ROLLOUT_INPUT": str(input_root),
            "LONG_LIVE_STAGE1_ROLLOUT_METADATA": str(
                input_root / "missing_metadata.csv"
            ),
            "LONG_LIVE_STAGE1_ROLLOUT_OUTPUT": str(tmp_path / "outputs"),
        }
    )

    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "metadata file does not exist" in completed.stderr
    assert "Incomplete Stage-1 checkpoint" not in completed.stderr


def test_inference_opt_in_keeps_fixed_negative_and_writes_trace():
    text = (ROOT / "inference.py").read_text(encoding="utf-8")
    assert "resolve_stage1_rollout_profile" in text
    assert "stage1_rollout_section.profile = raw_profile" in text
    assert "resolve_stage1_rollout_inference_plan" in text
    assert "resolve_stage1_rollout_global_prompt" in text
    assert "resolve_stage1_rollout_sample_shape" in text
    assert "run_stage1_rollout" in text
    assert "stage1_rollout_profile.negative_prompt" in text
    assert "post_load_sha256_verified" in text
    assert 'load_report"] = copy.deepcopy' in text
    assert 'load_report"] = repr' not in text
    assert "_stage1_rollout_trace.json" in text
    assert "atomic_write_json" in text
    assert "dist.is_initialized()" in text
    assert "MetadataImagePromptDataset" in text
    assert "allow_transposed_image_size=True" in text
    assert 'rollout_trace["input"]["metadata_record"]' in text
    assert '"metadata_sha256": dataset.metadata_sha256' in text
    assert '"row_sha256": metadata_record.row_sha256' in text
    assert text.index('if "LOCAL_RANK" in os.environ:') < text.index(
        "stage1_rollout_input_dataset = MetadataImagePromptDataset("
    )
    assert text.index(
        "stage1_rollout_input_dataset = MetadataImagePromptDataset("
    ) < text.index("pipeline = CausalDiffusionInferencePipeline(")
    assert text.index("for rollout_image_size in sorted(") < text.index(
        "pipeline = CausalDiffusionInferencePipeline("
    )
    assert text.index("shape = resolve_stage1_rollout_sample_shape(") < text.index(
        "sampled_noise = torch.randn("
    )
    assert "initial_latent.shape[2:]" in text
    assert '"batch_size": 1' in text
    assert text.index("if stage1_rollout_input_dataset is not None:") < text.index(
        "elif not _has_video_layout:"
    )
    dataset_selection = text[
        text.index("# Create dataset") : text.index("num_prompts = len(dataset)")
    ]
    assert "elif metadata_path:" not in dataset_selection
    assert "if stage1_rollout_input_dataset is not None" in dataset_selection
