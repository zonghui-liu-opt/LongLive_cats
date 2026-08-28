"""Strict inference preflight for Stage-1 teacher-forcing LoRA rollout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from utils.stage1_checkpoint import validate_checkpoint
from utils.stage1_io import sha256_file
from utils.stage1_rollout_profile import (
    Stage1RolloutProfile,
    resolve_stage1_rollout_profile,
)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a mapping") from exc


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def resolve_stage1_rollout_global_prompt(block_prompts: Sequence[Any]) -> str:
    """Resolve the one prompt actually used by a repeat-global rollout.

    ``ImagePromptDataset`` can expose multiple block prompts.  Stage-1
    self-rollout intentionally has one conditioning embedding for the whole
    episode, so accepting different block prompts would silently ignore all
    but the first and make the prompt sidecar inaccurate.
    """

    if isinstance(block_prompts, (str, bytes)) or not isinstance(
        block_prompts, Sequence
    ):
        raise TypeError("Stage-1 rollout block_prompts must be a sequence")
    prompts = list(block_prompts)
    if not prompts or any(
        not isinstance(prompt, str) or not prompt.strip() for prompt in prompts
    ):
        raise ValueError(
            "Stage-1 rollout requires at least one non-empty global prompt"
        )
    if any(prompt != prompts[0] for prompt in prompts[1:]):
        raise ValueError(
            "Stage-1 rollout supports one global prompt; all dataset block "
            "prompts must be identical"
        )
    return prompts[0]


def resolve_stage1_rollout_sample_shape(
    configured_shape: Sequence[Any],
    *,
    conditioning_image_size: Sequence[Any],
    spatial_compression_ratio: int,
    frame_seq_length: int,
) -> tuple[int, int, int, int, int]:
    """Resolve one row's latent geometry without rotating its input image."""

    if isinstance(configured_shape, (str, bytes)) or not isinstance(
        configured_shape, Sequence
    ):
        raise TypeError("Stage-1 rollout configured shape must be a sequence")
    shape = tuple(configured_shape)
    if len(shape) != 5:
        raise ValueError("Stage-1 rollout configured shape must have five values")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in shape):
        raise TypeError("Stage-1 rollout configured shape values must be integers")
    if any(value <= 0 for value in shape):
        raise ValueError(
            "Stage-1 rollout configured shape must contain five positive integers"
        )
    if shape[:3] != (1, 24, 48):
        raise ValueError(
            "Stage-1 rollout configured B/T/C must be (1, 24, 48), " f"got {shape[:3]}"
        )
    if isinstance(conditioning_image_size, (str, bytes)) or not isinstance(
        conditioning_image_size, Sequence
    ):
        raise TypeError("conditioning image size must be a sequence")
    image_size = tuple(conditioning_image_size)
    if len(image_size) != 2:
        raise ValueError("conditioning image size must have two values")
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in image_size
    ):
        raise TypeError("conditioning image size values must be integers")
    if any(value <= 0 for value in image_size):
        raise ValueError("conditioning image size must contain two positive integers")
    if isinstance(spatial_compression_ratio, bool) or not isinstance(
        spatial_compression_ratio, int
    ):
        raise TypeError("spatial_compression_ratio must be an integer")
    if spatial_compression_ratio <= 0:
        raise ValueError("spatial_compression_ratio must be a positive integer")
    if isinstance(frame_seq_length, bool) or not isinstance(frame_seq_length, int):
        raise TypeError("frame_seq_length must be an integer")
    if frame_seq_length <= 0:
        raise ValueError("frame_seq_length must be a positive integer")

    canonical_image_size = (
        shape[3] * spatial_compression_ratio,
        shape[4] * spatial_compression_ratio,
    )
    if image_size not in {
        canonical_image_size,
        canonical_image_size[::-1],
    }:
        raise ValueError(
            "conditioning image must use the configured canonical geometry "
            f"{canonical_image_size} or its transpose; got {image_size}"
        )
    if any(value % spatial_compression_ratio for value in image_size):
        raise ValueError(
            "conditioning image geometry must be divisible by the VAE spatial "
            "compression ratio"
        )
    latent_height = image_size[0] // spatial_compression_ratio
    latent_width = image_size[1] // spatial_compression_ratio
    if latent_height % 2 or latent_width % 2:
        raise ValueError(
            "conditioning latent geometry must be divisible by the Wan spatial patch"
        )
    resolved_frame_seq_length = (latent_height // 2) * (latent_width // 2)
    if resolved_frame_seq_length != frame_seq_length:
        raise ValueError(
            "conditioning geometry does not match pipeline frame_seq_length: "
            f"expected={frame_seq_length}, actual={resolved_frame_seq_length}"
        )
    return (shape[0], shape[1], shape[2], latent_height, latent_width)


def _phase_provenance(
    checkpoint_dir: Path,
    completed_step: int,
    *,
    expected_phase_epochs: tuple[int, int],
) -> dict[str, Any]:
    resolved_path = checkpoint_dir / "resolved_config.yaml"
    with resolved_path.open("r", encoding="utf-8") as handle:
        resolved = yaml.safe_load(handle)
    if not isinstance(resolved, Mapping):
        raise TypeError("Stage-1 resolved_config.yaml must contain a mapping")
    algorithm = _mapping(resolved.get("algorithm"), "resolved.algorithm")
    for name in ("i2v", "causal", "teacher_forcing", "independent_first_frame"):
        if algorithm.get(name) is not True:
            raise RuntimeError(f"Stage-1 resolved algorithm requires {name}=true")
    model_kwargs = _mapping(resolved.get("model_kwargs"), "resolved.model_kwargs")
    block_values = [
        value
        for value in (
            resolved.get("num_frame_per_block"),
            model_kwargs.get("num_frame_per_block"),
        )
        if value is not None
    ]
    if not block_values or any(
        isinstance(value, bool) or not isinstance(value, int) or value != 8
        for value in block_values
    ):
        raise RuntimeError("Stage-1 resolved config requires num_frame_per_block=8")
    recycling = _mapping(resolved.get("error_recycling"), "resolved.error_recycling")
    if recycling.get("enabled") is not True:
        raise RuntimeError(
            "Stage-1 resolved config requires error_recycling.enabled=true"
        )
    phases = resolved.get("phases")
    if not isinstance(phases, list) or len(phases) != 2:
        raise RuntimeError(
            "Stage-1 resolved config must contain exactly Phase A + Phase B"
        )
    expected_phase_contract = (
        ("phase_a_teacher_forcing", "collect_only", expected_phase_epochs[0]),
        ("phase_b_error_recycling", "collect_and_inject", expected_phase_epochs[1]),
    )

    expected_samples = int(
        resolved.get(
            "expected_num_samples",
            _mapping(resolved.get("data"), "resolved.data").get(
                "expected_num_samples", 0
            ),
        )
    )
    topology = _mapping(resolved.get("topology"), "resolved.topology")
    data_parallel = int(
        resolved.get("data_parallel_size", topology.get("data_parallel_size", 0))
    )
    batch_size = int(
        resolved.get(
            "batch_size",
            _mapping(resolved.get("data"), "resolved.data").get("batch_size", 0),
        )
    )
    accumulation = int(
        resolved.get(
            "gradient_accumulation_steps",
            _mapping(resolved.get("training"), "resolved.training").get(
                "gradient_accumulation_steps", 0
            ),
        )
    )
    denominator = data_parallel * batch_size * accumulation
    if expected_samples <= 0 or denominator <= 0 or expected_samples % denominator:
        raise RuntimeError("cannot derive exact Stage-1 updates per epoch")
    updates_per_epoch = expected_samples // denominator

    phase_entries: list[dict[str, Any]] = []
    cursor = 0
    active_phase = "all_phases_complete"
    for phase, (expected_name, expected_mode, expected_epochs) in zip(
        phases, expected_phase_contract, strict=True
    ):
        if not isinstance(phase, Mapping):
            raise TypeError("Stage-1 phase entries must be mappings")
        name = str(phase.get("name", ""))
        epochs = int(phase.get("epochs", 0))
        mode = _mapping(
            phase.get("error_recycling"), f"resolved phase {name}.error_recycling"
        ).get("mode")
        if (name, mode, epochs) != (expected_name, expected_mode, expected_epochs):
            raise RuntimeError(
                "Stage-1 Phase A/B contract mismatch: "
                f"expected={(expected_name, expected_mode, expected_epochs)}, "
                f"actual={(name, mode, epochs)}"
            )
        start = cursor
        cursor += epochs * updates_per_epoch
        phase_entries.append(
            {
                "name": name,
                "epochs": epochs,
                "start_completed_step": start,
                "end_completed_step": cursor,
            }
        )
        if start <= completed_step < cursor:
            active_phase = f"{name}_in_progress"
    if completed_step > cursor:
        active_phase = "past_declared_training_plan"

    return {
        "i2v": True,
        "causal": True,
        "teacher_forcing": True,
        "training_block_size": 8,
        "updates_per_epoch": updates_per_epoch,
        "phases": phase_entries,
        "declared_training_end_step": cursor,
        "phase_status": active_phase,
        "all_declared_phases_complete": completed_step == cursor,
        "resolved_config_sha256": sha256_file(resolved_path),
    }


@dataclass(frozen=True)
class Stage1RolloutInferencePlan:
    profile: Stage1RolloutProfile
    checkpoint_provenance: dict[str, Any]


def resolve_stage1_rollout_inference_plan(config: Any) -> Stage1RolloutInferencePlan:
    """Fail closed before constructing CUDA models or loading a LoRA adapter."""

    section = _mapping(_get(config, "stage1_rollout"), "stage1_rollout")
    if not _strict_bool(section.get("enabled", False), "stage1_rollout.enabled"):
        raise ValueError("stage1_rollout.enabled must be true")
    profile = resolve_stage1_rollout_profile(
        _mapping(section.get("profile"), "stage1_rollout.profile")
    )

    if not bool(_get(config, "i2v", False)):
        raise ValueError("Stage-1 rollout requires i2v=true")
    if int(_get(config, "num_samples", 1)) != 1:
        raise ValueError("Stage-1 rollout currently requires num_samples=1")
    for name in ("model_quant", "fp8_quant", "kv_quant"):
        if bool(_get(config, name, False)):
            raise ValueError(f"Stage-1 rollout requires {name}=false")
    if bool(_get(config, "merge_lora", False)):
        raise ValueError("Stage-1 rollout requires dynamic LoRA (merge_lora=false)")
    if bool(_get(config, "use_ema", False)):
        raise ValueError(
            "use_ema must be false; EMA selection is adapter_ema.safetensors"
        )

    adapter = _mapping(_get(config, "adapter"), "adapter")
    expected_adapter = {
        "type": "lora",
        "rank": 32,
        "alpha": 32,
        "dropout": 0.0,
        "bias": "none",
        "expected_target_modules": 180,
        "expected_trainable_parameters": 57016320,
        "expected_adapter_tensors": 360,
    }
    for name, expected in expected_adapter.items():
        if adapter.get(name) != expected:
            raise ValueError(
                f"Stage-1 rollout adapter.{name} must be {expected!r}, "
                f"got {adapter.get(name)!r}"
            )

    base_path = Path(str(_get(config, "generator_ckpt", ""))).expanduser().resolve()
    adapter_path = Path(str(_get(config, "lora_ckpt", ""))).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if adapter_path.name != "adapter_ema.safetensors" or not adapter_path.is_file():
        raise FileNotFoundError(
            "Stage-1 rollout requires checkpoint/adapter_ema.safetensors: "
            f"{adapter_path}"
        )
    checkpoint_dir = adapter_path.parent

    checkpoint = _mapping(section.get("checkpoint"), "stage1_rollout.checkpoint")
    if not _strict_bool(
        checkpoint.get("require_ema", True), "stage1_rollout.checkpoint.require_ema"
    ):
        raise ValueError("Stage-1 rollout requires the EMA adapter")
    if not _strict_bool(
        checkpoint.get("verify_hashes", True),
        "stage1_rollout.checkpoint.verify_hashes",
    ):
        raise ValueError(
            "Stage-1 rollout checkpoint hash verification cannot be disabled"
        )

    base_sha256 = sha256_file(base_path)
    manifest = validate_checkpoint(
        checkpoint_dir,
        expected_base_sha256=base_sha256,
        verify_file_hashes=True,
    )
    manifest_names = [str(entry.get("name", "")) for entry in manifest.get("files", [])]
    required_hashed_artifacts = {
        "adapter_ema.safetensors",
        "adapter_raw.safetensors",
        "base_reference.json",
        "resolved_config.yaml",
    }
    if len(manifest_names) != len(
        set(manifest_names)
    ) or not required_hashed_artifacts.issubset(manifest_names):
        raise RuntimeError(
            "Stage-1 checkpoint manifest must uniquely hash EMA/raw adapters, "
            "base_reference.json, and resolved_config.yaml"
        )
    completed_step = int(manifest["completed_step"])
    expected_step = checkpoint.get("expected_completed_step")
    if expected_step is not None:
        if isinstance(expected_step, bool) or not isinstance(expected_step, int):
            raise TypeError("expected_completed_step must be an integer")
        if completed_step != expected_step:
            raise RuntimeError(
                "Stage-1 checkpoint step mismatch: "
                f"expected={expected_step}, actual={completed_step}"
            )
    expected_phase_epochs_raw = checkpoint.get("expected_phase_epochs")
    # OmegaConf keeps YAML lists as ListConfig.  Validate its generic sequence
    # contract, then freeze it to a tuple before downstream consumption.
    if (
        isinstance(expected_phase_epochs_raw, (str, bytes))
        or not isinstance(expected_phase_epochs_raw, Sequence)
        or len(expected_phase_epochs_raw) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in expected_phase_epochs_raw
        )
    ):
        raise ValueError(
            "stage1_rollout.checkpoint.expected_phase_epochs must be two "
            "positive integers"
        )
    expected_phase_epochs = tuple(expected_phase_epochs_raw)
    phase = _phase_provenance(
        checkpoint_dir,
        completed_step,
        expected_phase_epochs=expected_phase_epochs,
    )
    if phase["teacher_forcing"] is not True:
        raise RuntimeError("Stage-1 rollout checkpoint is not teacher-forcing trained")

    return Stage1RolloutInferencePlan(
        profile=profile,
        checkpoint_provenance={
            "runtime_mode": "dynamic_stage1_lora",
            "base": {
                "path": str(base_path),
                "sha256": base_sha256,
                "size": base_path.stat().st_size,
            },
            "adapter": {
                "variant": "ema",
                "path": str(adapter_path),
                "sha256": sha256_file(adapter_path),
                "size": adapter_path.stat().st_size,
            },
            "checkpoint": {
                "path": str(checkpoint_dir),
                "completed_step": completed_step,
                "manifest_sha256": sha256_file(
                    checkpoint_dir / "checkpoint_manifest.json"
                ),
                "topology": dict(manifest.get("topology", {})),
                **phase,
            },
        },
    )


__all__ = [
    "Stage1RolloutInferencePlan",
    "resolve_stage1_rollout_global_prompt",
    "resolve_stage1_rollout_inference_plan",
    "resolve_stage1_rollout_sample_shape",
]
