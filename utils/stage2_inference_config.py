"""Strict resolver for the isolated Stage-2 deployment inference entrypoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from pipeline.stage2_rollout_profile import (
    STAGE2_ROLLOUT_PROFILE_NAMES,
    resolve_stage2_rollout_profile,
)
from utils.stage1_io import canonical_json_sha256
from utils.stage2_inference import STAGE2_INFERENCE_SEEDS

STAGE2_INFERENCE_CONFIG_SCHEMA = "longlive_stage2_inference/v1"
_ROOT_KEYS = {
    "schema",
    "stage2_checkpoint",
    "source_cache_manifest",
    "architecture_root",
    "t5_checkpoint",
    "tokenizer_dir",
    "vae_checkpoint",
    "single_metadata",
    "two_action_metadata",
    "output_root",
    "profiles",
    "seeds",
    "runtime",
}
_RUNTIME_KEYS = {
    "dtype",
    "cfg_scale",
    "fps",
    "merge_ema_lora",
    "batch_size_per_device",
}


def _plain_mapping(value: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError:  # pragma: no cover - project dependency
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError("Stage-2 inference config must be a mapping")
    return dict(value)


def _path(value: Any, label: str, *, require_file: bool = False) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical path string")
    source = Path(value).expanduser()
    if require_file and (source.is_symlink() or not source.is_file()):
        raise FileNotFoundError(source)
    path = source.resolve()
    return os.fspath(path)


@dataclass(frozen=True)
class ResolvedStage2InferenceConfig:
    schema: str
    stage2_checkpoint: str
    source_cache_manifest: str
    architecture_root: str
    t5_checkpoint: str
    tokenizer_dir: str
    vae_checkpoint: str
    single_metadata: str
    two_action_metadata: str
    output_root: str
    profiles: tuple[str, ...]
    seeds: tuple[int, ...]
    dtype: str
    cfg_scale: float
    fps: int
    merge_ema_lora: bool
    batch_size_per_device: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def contract_hash(self) -> str:
        value = self.to_dict()
        value.pop("output_root")
        return canonical_json_sha256(value)

    def launch_hash(self) -> str:
        return canonical_json_sha256(self.to_dict())


def resolve_stage2_inference_config(value: Any) -> ResolvedStage2InferenceConfig:
    raw = _plain_mapping(value)
    if set(raw) != _ROOT_KEYS:
        raise ValueError(
            "Stage-2 inference config root schema mismatch: "
            f"missing={sorted(_ROOT_KEYS - set(raw))}, "
            f"extra={sorted(set(raw) - _ROOT_KEYS)}"
        )
    if raw["schema"] != STAGE2_INFERENCE_CONFIG_SCHEMA:
        raise ValueError("Stage-2 inference config schema mismatch")
    runtime = _plain_mapping(raw["runtime"])
    if set(runtime) != _RUNTIME_KEYS:
        raise ValueError("Stage-2 inference runtime schema mismatch")
    profiles = tuple(raw["profiles"]) if isinstance(raw["profiles"], list) else ()
    if not profiles or len(profiles) != len(set(profiles)):
        raise ValueError("Stage-2 inference profiles must be a non-empty unique list")
    for profile in profiles:
        resolve_stage2_rollout_profile(profile)
    if any(profile not in STAGE2_ROLLOUT_PROFILE_NAMES for profile in profiles):
        raise AssertionError(
            "Stage-2 rollout profile resolver accepted an unknown name"
        )
    seeds = tuple(raw["seeds"]) if isinstance(raw["seeds"], list) else ()
    if any(type(seed) is not int for seed in seeds) or seeds != STAGE2_INFERENCE_SEEDS:
        raise ValueError(
            f"formal Stage-2 inference seeds must be {STAGE2_INFERENCE_SEEDS}"
        )
    if runtime["dtype"] != "bfloat16":
        raise ValueError("Stage-2 inference dtype must be bfloat16")
    cfg_scale = runtime["cfg_scale"]
    if (
        isinstance(cfg_scale, bool)
        or not isinstance(cfg_scale, (int, float))
        or not math.isfinite(float(cfg_scale))
        or float(cfg_scale) != 1.0
    ):
        raise ValueError("Stage-2 Generator deployment requires CFG1")
    if runtime["fps"] != 24 or type(runtime["fps"]) is not int:
        raise ValueError("Stage-2 inference output must be 24fps")
    if runtime["merge_ema_lora"] is not True:
        raise ValueError("Stage-2 inference must safe-merge the validated EMA LoRA")
    if (
        runtime["batch_size_per_device"] != 1
        or type(runtime["batch_size_per_device"]) is not int
    ):
        raise ValueError(
            "formal Stage-2 inference currently locks batch_size_per_device=1"
        )

    resolved = ResolvedStage2InferenceConfig(
        schema=raw["schema"],
        stage2_checkpoint=_path(raw["stage2_checkpoint"], "stage2_checkpoint"),
        source_cache_manifest=_path(
            raw["source_cache_manifest"],
            "source_cache_manifest",
            require_file=True,
        ),
        architecture_root=_path(raw["architecture_root"], "architecture_root"),
        t5_checkpoint=_path(raw["t5_checkpoint"], "t5_checkpoint"),
        tokenizer_dir=_path(raw["tokenizer_dir"], "tokenizer_dir"),
        vae_checkpoint=_path(raw["vae_checkpoint"], "vae_checkpoint"),
        single_metadata=_path(
            raw["single_metadata"], "single_metadata", require_file=True
        ),
        two_action_metadata=_path(
            raw["two_action_metadata"],
            "two_action_metadata",
            require_file=True,
        ),
        output_root=_path(raw["output_root"], "output_root"),
        profiles=profiles,
        seeds=seeds,
        dtype=runtime["dtype"],
        cfg_scale=float(cfg_scale),
        fps=runtime["fps"],
        merge_ema_lora=runtime["merge_ema_lora"],
        batch_size_per_device=runtime["batch_size_per_device"],
    )
    # Prove the dataclass is strict-JSON serializable before model construction.
    json.dumps(resolved.to_dict(), allow_nan=False, sort_keys=True)
    return resolved


def load_stage2_inference_config(
    path: str | os.PathLike[str],
) -> ResolvedStage2InferenceConfig:
    from omegaconf import OmegaConf

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    source = candidate.resolve()
    return resolve_stage2_inference_config(OmegaConf.load(source))


__all__ = [
    "STAGE2_INFERENCE_CONFIG_SCHEMA",
    "ResolvedStage2InferenceConfig",
    "load_stage2_inference_config",
    "resolve_stage2_inference_config",
]
