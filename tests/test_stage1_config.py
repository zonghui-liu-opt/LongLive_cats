from pathlib import Path

from omegaconf import OmegaConf
import pytest

from utils.config import normalize_config
from utils.distributed import validate_stage1_fsdp1_gate0
from utils.stage1_schedule import resolve_stage1_schedule


def test_release_stage1_config_has_one_locked_source_of_truth():
    root = Path(__file__).resolve().parents[1]
    config = normalize_config(OmegaConf.load(root / "configs" / "train_i2v_ar.yaml"))
    assert config.expected_world_size == 6
    assert config.sequence_parallel_size == 3
    assert config.data_parallel_size == 2
    assert config.fsdp_backend == "fsdp2"
    assert config.sharding_strategy == "hsdp"
    assert config.device_mesh_shape == [2, 3]
    assert config.device_mesh_dim_names == ["replicate", "shard"]
    assert config.cache_precompute.expected_world_size == 4
    assert config.cache_precompute.require_cuda is True
    assert config.cache_precompute.minimum_cuda_capability == [9, 0]
    assert config.image_or_video_shape == [1, 24, 48, 30, 52]
    assert config.gradient_accumulation_steps == 2
    assert "max_iters" not in config
    assert config.evaluation.interval == 0
    assert config.disable_wandb is True
    assert "rank" not in config  # adapter.rank must remain nested
    assert config.adapter.rank == 32
    schedule = resolve_stage1_schedule(config, dataloader_length=300)
    assert schedule.total_updates == 750
    assert schedule.checkpoint_interval_updates == 75


def test_release_stage1_backend_is_gate0_fail_fast():
    with pytest.raises(RuntimeError, match="Stage-1 Gate 0 failed"):
        validate_stage1_fsdp1_gate0()
