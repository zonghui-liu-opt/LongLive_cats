from __future__ import annotations

import copy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from utils.stage2_inference_config import (
    load_stage2_inference_config,
    resolve_stage2_inference_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "infer_i2v_stage2_baseline.yaml"


def _raw(tmp_path):
    value = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    value["stage2_checkpoint"] = str(tmp_path / "checkpoint")
    value["source_cache_manifest"] = str(source_manifest)
    value["architecture_root"] = str(tmp_path / "architecture")
    value["t5_checkpoint"] = str(tmp_path / "t5.pth")
    value["tokenizer_dir"] = str(tmp_path / "tokenizer")
    value["vae_checkpoint"] = str(tmp_path / "vae.pth")
    value["output_root"] = str(tmp_path / "outputs")
    value["single_metadata"] = str(
        PROJECT_ROOT / "testsets" / "metadata_6cases_480x832.csv"
    )
    value["two_action_metadata"] = str(
        PROJECT_ROOT
        / "testsets"
        / "metadata_8cases_two_actions_continuation_480x832_253frames.csv"
    )
    return value


def test_formal_inference_config_locks_ema_cfg1_bf16_seeds_and_baseline(tmp_path):
    resolved = resolve_stage2_inference_config(_raw(tmp_path))
    assert resolved.profiles == ("baseline_c8w16k4s1",)
    assert resolved.seeds == (1, 2, 3, 4)
    assert resolved.dtype == "bfloat16"
    assert resolved.cfg_scale == 1.0
    assert resolved.fps == 24
    assert resolved.merge_ema_lora is True
    assert resolved.batch_size_per_device == 1
    assert len(resolved.contract_hash()) == 64
    assert len(resolved.launch_hash()) == 64

    moved = copy.deepcopy(_raw(tmp_path))
    moved["output_root"] = str(tmp_path / "elsewhere")
    moved = resolve_stage2_inference_config(moved)
    assert moved.contract_hash() == resolved.contract_hash()
    assert moved.launch_hash() != resolved.launch_hash()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("profiles",), ["unknown"], "unknown Stage-2 rollout profile"),
        (("profiles",), [], "non-empty unique"),
        (("seeds",), [1], "seeds must"),
        (("seeds",), [True, 2, 3, 4], "seeds must"),
        (("runtime", "dtype"), "float16", "bfloat16"),
        (("runtime", "cfg_scale"), 5.0, "CFG1"),
        (("runtime", "cfg_scale"), "1.0", "CFG1"),
        (("runtime", "fps"), 16, "24fps"),
        (("runtime", "merge_ema_lora"), False, "safe-merge"),
        (("runtime", "batch_size_per_device"), 2, "batch_size_per_device=1"),
    ],
)
def test_inference_config_rejects_semantic_drift(tmp_path, path, value, message):
    raw = _raw(tmp_path)
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        resolve_stage2_inference_config(raw)


def test_config_file_resolves_repo_metadata_and_env_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("LONG_LIVE_STAGE2_INFERENCE_CHECKPOINT", str(tmp_path / "ckpt"))
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("LONG_LIVE_STAGE2_SOURCE_MANIFEST", str(source_manifest))
    monkeypatch.setenv("LONG_LIVE_STAGE2_ARCHITECTURE_ROOT", str(tmp_path / "arch"))
    monkeypatch.setenv("LONG_LIVE_STAGE2_T5_CHECKPOINT", str(tmp_path / "t5"))
    monkeypatch.setenv("LONG_LIVE_STAGE2_TOKENIZER_DIR", str(tmp_path / "tok"))
    monkeypatch.setenv("LONG_LIVE_STAGE2_VAE_CHECKPOINT", str(tmp_path / "vae"))
    monkeypatch.setenv("LONG_LIVE_STAGE2_INFERENCE_OUTPUT", str(tmp_path / "out"))
    resolved = load_stage2_inference_config(CONFIG)
    assert resolved.single_metadata.endswith("metadata_6cases_480x832.csv")
    assert resolved.two_action_metadata.endswith(
        "metadata_8cases_two_actions_continuation_480x832_253frames.csv"
    )


def test_config_file_loader_rejects_a_symlink(tmp_path):
    link = tmp_path / "infer.yaml"
    link.symlink_to(CONFIG)
    with pytest.raises(FileNotFoundError):
        load_stage2_inference_config(link)


def test_inference_config_supports_only_named_compression_and_sink_profiles(tmp_path):
    raw = _raw(tmp_path)
    raw["profiles"] = [
        "baseline_c8w16k4s1",
        "c4w12k4s1",
        "c4w8k2s1",
        "c8w16k4s4",
        "c8w16k4s8",
    ]
    resolved = resolve_stage2_inference_config(raw)
    assert resolved.profiles == tuple(raw["profiles"])
