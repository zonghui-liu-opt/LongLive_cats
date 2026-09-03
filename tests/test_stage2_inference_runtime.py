from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from pipeline.stage2_rollout_profile import (
    build_stage2_deployment_rollout_spec,
    resolve_stage2_rollout_profile,
)
from utils import stage2_inference_runtime
from utils.stage1_io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from utils.stage2_inference import Stage2InferenceResult
from utils.stage2_inference_artifacts import (
    STAGE2_INFERENCE_MANIFEST_NAME,
    STAGE2_REVIEW_INDEX_NAME,
    build_stage2_inference_config_identity,
)
from utils.stage2_inference_assets import STAGE2_RUNTIME_ASSET_IDENTITY_SCHEMA
from utils.stage2_inference_batch import (
    STAGE2_SINGLE_DATASET,
    STAGE2_TWO_ACTION_DATASET,
    Stage2InferenceSample,
)
from utils.stage2_inference_config import ResolvedStage2InferenceConfig
from utils.stage2_inference_runtime import (
    Stage2InferenceDistributedContext,
    Stage2InferenceRuntimeOps,
    _ensure_regular_parents,
    _prepare_output_root,
    run_stage2_inference,
)
from utils.stage2_inference_sweep_config import (
    STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA,
    ResolvedStage2InferenceSweepConfig,
)

_CHECKPOINT = {
    "directory": "/checkpoint/stage2_g000280",
    "manifest_sha256": "1" * 64,
    "completed_generator_updates": 280,
    "contract_hash": "2" * 64,
    "generator_ema_sha256": "3" * 64,
}
_CODE_VERSION = {"stage2_source_sha256": "4" * 64}
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_output_root_may_be_inside_the_project_tree() -> None:
    with TemporaryDirectory(
        prefix=".stage2-output-root-test-", dir=_PROJECT_ROOT
    ) as directory:
        root = Path(directory).resolve()
        guard = _prepare_output_root(root)
        assert guard.root == root


def test_regular_parent_creation_is_safe_for_concurrent_ranks(
    tmp_path: Path,
) -> None:
    guard = _prepare_output_root(tmp_path)
    root = guard.root
    barrier = Barrier(8)

    def create(rank: int) -> None:
        barrier.wait()
        _ensure_regular_parents(
            guard,
            root / "baseline" / "single_action" / f"sample-{rank}.mp4",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(create, range(8)))

    assert (root / "baseline" / "single_action").is_dir()
    assert not (root / "baseline").is_symlink()


def test_output_root_guard_accepts_stable_shared_storage_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = _prepare_output_root(tmp_path / "output")
    root = guard.root
    original = stage2_inference_runtime._lstat_directory_identity

    def drifted_identity(path: Path) -> tuple[int, int, int]:
        device, inode, mode = original(path)
        if path == root:
            return device + 101, inode + 103, mode
        return device, inode, mode

    monkeypatch.setattr(
        stage2_inference_runtime,
        "_lstat_directory_identity",
        drifted_identity,
    )

    _ensure_regular_parents(
        guard,
        root / "baseline" / "single_action" / "sample.mp4",
    )

    assert (root / "baseline" / "single_action").is_dir()


def test_output_root_anchor_is_shared_by_concurrent_rank_preparation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    barrier = Barrier(8)

    def prepare(_: int) -> Any:
        barrier.wait()
        return _prepare_output_root(root)

    with ThreadPoolExecutor(max_workers=8) as executor:
        guards = tuple(executor.map(prepare, range(8)))

    assert {guard.root for guard in guards} == {root.resolve()}
    assert len({guard.anchor_sha256 for guard in guards}) == 1
    resumed = _prepare_output_root(root)
    assert resumed.anchor_sha256 == guards[0].anchor_sha256


@pytest.mark.parametrize("damage", ["missing", "invalid", "symlink", "other-anchor"])
def test_output_root_guard_rejects_anchor_damage(
    tmp_path: Path,
    damage: str,
) -> None:
    root = tmp_path / "output"
    guard = _prepare_output_root(root)
    anchor = root / stage2_inference_runtime._OUTPUT_ROOT_ANCHOR_NAME
    anchor.unlink()
    if damage == "invalid":
        anchor.write_text("not an anchor\n", encoding="utf-8")
    elif damage == "symlink":
        outside = tmp_path / "outside-anchor.json"
        outside.write_text("not an anchor\n", encoding="utf-8")
        anchor.symlink_to(outside)
    elif damage == "other-anchor":
        other_root = tmp_path / "other-output"
        _prepare_output_root(other_root)
        other_anchor = other_root / stage2_inference_runtime._OUTPUT_ROOT_ANCHOR_NAME
        anchor.write_bytes(other_anchor.read_bytes())

    with pytest.raises(RuntimeError, match="output root identity changed"):
        _ensure_regular_parents(
            guard,
            root / "baseline" / "single_action" / "sample.mp4",
        )


def test_output_root_rank_identity_uses_persistent_anchor(tmp_path: Path) -> None:
    guard = _prepare_output_root(tmp_path / "output")

    identity = stage2_inference_runtime._output_root_guard_identity(guard)

    assert identity == {
        "schema": "longlive_stage2_output_root_guard/v2",
        "path": str(guard.root),
        "anchor_sha256": guard.anchor_sha256,
    }
    assert (
        stage2_inference_runtime._validate_output_root_guard_identity(identity)
        == identity
    )


@pytest.mark.parametrize("competitor", ["file", "symlink"])
def test_regular_parent_creation_rejects_file_or_symlink_competitor(
    tmp_path: Path,
    competitor: str,
) -> None:
    guard = _prepare_output_root(tmp_path)
    root = guard.root
    target = root / "shared-parent"
    barrier = Barrier(2)
    competitor_won = Event()

    def attack() -> None:
        barrier.wait()
        if competitor == "file":
            target.write_text("not a directory", encoding="utf-8")
        else:
            safe_directory = root / "safe-directory"
            safe_directory.mkdir()
            target.symlink_to(safe_directory, target_is_directory=True)
        competitor_won.set()

    def create() -> None:
        barrier.wait()
        assert competitor_won.wait(timeout=5)
        _ensure_regular_parents(guard, target / "sample.mp4")

    with ThreadPoolExecutor(max_workers=2) as executor:
        attacker = executor.submit(attack)
        creator = executor.submit(create)
        attacker.result()
        with pytest.raises(RuntimeError, match="not a regular directory"):
            creator.result()


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_output_root_guard_rejects_root_replacement_before_parent_creation(
    tmp_path: Path,
    replacement: str,
) -> None:
    root = tmp_path / "output"
    guard = _prepare_output_root(root)
    assert all(type(value) is int for value in (guard.device, guard.inode, guard.mode))
    saved_root = tmp_path / "authenticated-output"
    root.rename(saved_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    if replacement == "symlink":
        root.symlink_to(outside, target_is_directory=True)
    else:
        root.mkdir()

    with pytest.raises(RuntimeError, match="output root .*changed|regular directory"):
        _ensure_regular_parents(
            guard,
            guard.root / "baseline" / "single_action" / "sample.mp4",
        )

    assert not (outside / "baseline").exists()
    assert not (root / "baseline").exists()
    assert saved_root.is_dir()


def _sample(
    root: Path,
    *,
    dataset: str,
    row_id: int,
    seed: int,
    prompts: tuple[str, ...],
    profile: str = "c8w16k4s4",
) -> Stage2InferenceSample:
    image_path = root / "input.png"
    if not image_path.exists():
        image_path.write_bytes(b"dependency-injected-image")
    return Stage2InferenceSample(
        dataset=dataset,
        row_id=row_id,
        seed=seed,
        profile=profile,
        input_image=image_path,
        image_sha256=sha256_file(image_path),
        row_sha256=chr(ord("b") + row_id) * 64,
        height=32,
        width=48,
        bucket="landscape",
        prompts=prompts,
        case_group=f"case_{row_id}",
    )


def _config(
    tmp_path: Path,
    *,
    profile: str = "c8w16k4s4",
) -> ResolvedStage2InferenceConfig:
    single = tmp_path / "single.csv"
    double = tmp_path / "double.csv"
    single.write_text("single\n", encoding="utf-8")
    double.write_text("double\n", encoding="utf-8")
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    return ResolvedStage2InferenceConfig(
        schema="longlive_stage2_inference/v1",
        stage2_checkpoint="/checkpoint/stage2_g000280",
        source_cache_manifest=str(source_manifest),
        architecture_root="/models/wan",
        t5_checkpoint="/models/t5.pth",
        tokenizer_dir="/models/tokenizer",
        vae_checkpoint="/models/vae.pth",
        single_metadata=str(single),
        two_action_metadata=str(double),
        output_root=str(tmp_path / "output"),
        profiles=(profile,),
        seeds=(1, 2, 3, 4),
        dtype="bfloat16",
        cfg_scale=1.0,
        fps=24,
        merge_ema_lora=True,
        batch_size_per_device=1,
    )


def _sweep_config(tmp_path: Path) -> ResolvedStage2InferenceSweepConfig:
    base = _config(tmp_path, profile="baseline_c8w16k4s1")
    spec = build_stage2_deployment_rollout_spec(
        chunk_frames=6,
        local_window_frames=12,
        num_denoising_steps=3,
    )
    return ResolvedStage2InferenceSweepConfig(
        schema=STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA,
        base_config_path=str(tmp_path / "base.yaml"),
        base_config_sha256="a" * 64,
        stage2_checkpoint=base.stage2_checkpoint,
        source_cache_manifest=base.source_cache_manifest,
        architecture_root=base.architecture_root,
        t5_checkpoint=base.t5_checkpoint,
        tokenizer_dir=base.tokenizer_dir,
        vae_checkpoint=base.vae_checkpoint,
        single_metadata=base.single_metadata,
        two_action_metadata=base.two_action_metadata,
        output_root=base.output_root,
        profiles=(spec.name,),
        seeds=(7,),
        dtype=base.dtype,
        cfg_scale=base.cfg_scale,
        fps=base.fps,
        merge_ema_lora=base.merge_ema_lora,
        batch_size_per_device=base.batch_size_per_device,
        evaluation_mode="quick",
        single_row_ids=(0,),
        two_action_row_ids=(0,),
        profile_set_sha256=canonical_json_sha256([spec.to_dict()]),
        _rollout_specs=(spec,),
    )


def _runtime_asset_identity(
    config: ResolvedStage2InferenceConfig,
) -> dict[str, Any]:
    paths = {
        "t5_checkpoint": config.t5_checkpoint,
        "tokenizer_dir": config.tokenizer_dir,
        "vae_checkpoint": config.vae_checkpoint,
        "architecture_config": str(Path(config.architecture_root) / "config.json"),
        "generator_base": "/models/generator-stage1-step3075.pt",
    }
    body = {
        "schema": STAGE2_RUNTIME_ASSET_IDENTITY_SCHEMA,
        "checkpoint": {
            "directory": config.stage2_checkpoint,
            "manifest_sha256": _CHECKPOINT["manifest_sha256"],
            "generator_recorded_asset_sha256": "5" * 64,
        },
        "source_manifest": {
            "path": config.source_cache_manifest,
            "manifest_sha256": "6" * 64,
            "file_sha256": "7" * 64,
            "size": 1,
        },
        "assets": {
            name: {
                "path": path,
                "aggregate_sha256": f"{index:x}" * 64,
                "file_count": 1,
                "total_size": 1,
            }
            for index, (name, path) in enumerate(paths.items(), start=8)
        },
    }
    return {**body, "identity_sha256": canonical_json_sha256(body)}


def _runtime_assets(config: ResolvedStage2InferenceConfig) -> dict[str, Any]:
    identity = _runtime_asset_identity(config)
    return {
        "identity": identity,
        "checkpoint": identity["checkpoint"],
        "generator_asset": {},
    }


class _TextEncoder(torch.nn.Module):
    def __init__(self, calls: dict[str, Any]):
        super().__init__()
        self.calls = calls

    def forward(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        self.calls["prompt_batches"].append(tuple(prompts))
        return {
            "prompt_embeds": torch.arange(
                len(prompts) * 6,
                dtype=torch.float32,
            ).reshape(len(prompts), 2, 3)
        }


class _VAEModel:
    def __init__(self, calls: dict[str, Any]):
        self.calls = calls

    def clear_cache(self) -> None:
        self.calls["vae_cache_clears"] += 1


class _VAE:
    def __init__(self, calls: dict[str, Any]):
        self.calls = calls
        self.model = _VAEModel(calls)

    def encode_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        self.calls["vae_encodes"] += 1
        return torch.zeros(
            (1, 1, 48, image.shape[-2] // 16, image.shape[-1] // 16),
            # Match the real WanVAEWrapper boundary: it always returns FP32.
            dtype=torch.float32,
            device=image.device,
        )


@dataclass(frozen=True)
class _Pipeline:
    name: str
    global_sink_frames: int


def _video_result(sample: Stage2InferenceSample) -> Stage2InferenceResult:
    frames = 96 if sample.dataset == STAGE2_SINGLE_DATASET else 192
    return Stage2InferenceResult(
        mode=sample.dataset,
        video=torch.zeros((1, frames, 3, sample.height, sample.width)),
        episode_latents=(),
        episode_videos=(),
        trace={"fake": sample.sample_key},
    )


def _video_writer(video: torch.Tensor, path: Path, *, fps: int) -> None:
    path.write_text(
        json.dumps(
            {
                "width": int(video.shape[-1]),
                "height": int(video.shape[-2]),
                "frame_count": int(video.shape[1]),
                "fps": fps,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _video_probe(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fake_artifact_ops(calls: dict[str, Any]) -> dict[str, Any]:
    def build_trace(
        *,
        sample: Stage2InferenceSample,
        generation_trace: dict[str, Any],
        output_root: Path,
        video_path: Path,
        checkpoint: dict[str, Any],
        inference_config: dict[str, Any],
        code_version: dict[str, Any],
        probe_fn: Any,
    ) -> dict[str, Any]:
        probe = probe_fn(video_path)
        return {
            "sample_key": sample.sample_key,
            "checkpoint": dict(checkpoint),
            "inference_config": dict(inference_config),
            "code_version": dict(code_version),
            "generation": dict(generation_trace),
            "output": {
                "relative_path": sample.output_relative_path.as_posix(),
                **probe,
                "size": video_path.stat().st_size,
                "sha256": sha256_file(video_path),
            },
        }

    def validate_trace(
        trace: dict[str, Any],
        *,
        sample: Stage2InferenceSample,
        inference_config: dict[str, Any] | None = None,
        code_version: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if trace.get("sample_key") != sample.sample_key:
            raise RuntimeError("wrong fake trace sample")
        if (
            inference_config is not None
            and trace.get("inference_config") != inference_config
        ):
            raise RuntimeError("wrong fake trace inference config")
        if code_version is not None and trace.get("code_version") != code_version:
            raise RuntimeError("wrong fake trace code version")
        return dict(trace)

    def write_trace(
        output_root: Path,
        *,
        sample: Stage2InferenceSample,
        trace: dict[str, Any],
    ) -> Path:
        path = Path(output_root) / sample.trace_relative_path
        if path.exists():
            raise FileExistsError(path)
        atomic_write_json(path, trace)
        calls["trace_writes"] += 1
        return path

    def build_manifest(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "complete",
            "checkpoint": dict(kwargs["checkpoint"]),
            "inference_config": dict(kwargs["inference_config"]),
            "code_version": dict(kwargs["code_version"]),
            "samples": [sample.sample_key for sample in kwargs["samples"]],
            "traces": {
                key: value["output"]["sha256"]
                for key, value in sorted(kwargs["traces"].items())
            },
        }

    def validate_manifest(
        manifest: dict[str, Any],
        *,
        inference_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if manifest.get("status") != "complete":
            raise RuntimeError("fake manifest is incomplete")
        if (
            inference_config is not None
            and manifest.get("inference_config") != inference_config
        ):
            raise RuntimeError("wrong fake manifest inference config")
        return dict(manifest)

    def write_manifest(output_root: Path, *, manifest: dict[str, Any]) -> Path:
        path = Path(output_root) / STAGE2_INFERENCE_MANIFEST_NAME
        if path.exists():
            raise FileExistsError(path)
        atomic_write_json(path, manifest)
        calls["manifest_writes"] += 1
        return path

    def write_index(output_root: Path, *, manifest: dict[str, Any]) -> Path:
        path = Path(output_root) / STAGE2_REVIEW_INDEX_NAME
        payload = ("index:" + ",".join(manifest["samples"])).encode("utf-8")
        atomic_write_bytes(path, payload)
        calls["index_writes"] += 1
        return path

    return {
        "build_sample_trace": build_trace,
        "validate_sample_trace": validate_trace,
        "write_sample_trace": write_trace,
        "build_manifest": build_manifest,
        "validate_manifest": validate_manifest,
        "write_manifest": write_manifest,
        "write_review_index": write_index,
    }


def _runtime_ops(
    samples: tuple[Stage2InferenceSample, ...],
    calls: dict[str, Any],
    *,
    probe_fn: Any = _video_probe,
) -> Stage2InferenceRuntimeOps:
    calls.update(
        {
            "text_encoder_builds": 0,
            "generator_loads": 0,
            "vae_builds": 0,
            "vae_encodes": 0,
            "vae_cache_clears": 0,
            "image_loads": 0,
            "prompt_batches": [],
            "pipelines": [],
            "single_calls": [],
            "two_calls": [],
            "video_writes": 0,
            "trace_writes": 0,
            "manifest_writes": 0,
            "index_writes": 0,
            "runtime_asset_builds": 0,
            "runtime_asset_assertions": [],
        }
    )

    def build_runtime_assets(config: ResolvedStage2InferenceConfig) -> dict[str, Any]:
        calls["runtime_asset_builds"] += 1
        return _runtime_assets(config)

    def assert_runtime_assets(
        _assets: dict[str, Any],
        *,
        names: tuple[str, ...] | None = None,
        include_source_manifest: bool = False,
    ) -> None:
        calls["runtime_asset_assertions"].append((names, include_source_manifest))

    def build_text(*_args: Any) -> _TextEncoder:
        calls["text_encoder_builds"] += 1
        return _TextEncoder(calls)

    def load_generator(*_args: Any) -> Any:
        calls["generator_loads"] += 1
        return SimpleNamespace(
            generator=torch.nn.Identity(),
            resolved_training_config=SimpleNamespace(
                num_train_timesteps=1000,
                patch_tokens_per_frame=390,
            ),
            checkpoint_identity=lambda: dict(_CHECKPOINT),
        )

    def build_vae(*_args: Any) -> _VAE:
        calls["vae_builds"] += 1
        return _VAE(calls)

    def build_pipeline(_generator: Any, _loaded: Any, profile: str) -> _Pipeline:
        calls["pipelines"].append(profile)
        return _Pipeline(
            name=profile,
            global_sink_frames=resolve_stage2_rollout_profile(
                profile
            ).global_sink_frames,
        )

    def load_image(sample: Stage2InferenceSample) -> torch.Tensor:
        calls["image_loads"] += 1
        return torch.zeros((1, 3, 1, sample.height, sample.width))

    sample_by_prompt = {sample.prompts: sample for sample in samples}

    def generate_single(pipeline: _Pipeline, _vae: Any, **kwargs: Any) -> Any:
        calls["single_calls"].append(pipeline.name)
        assert kwargs["initial_latent"].dtype == torch.bfloat16
        prompt = next(
            prompts
            for prompts, sample in sample_by_prompt.items()
            if sample.dataset == STAGE2_SINGLE_DATASET
        )
        assert kwargs["seeds"] == (sample_by_prompt[prompt].seed,)
        return _video_result(sample_by_prompt[prompt])

    def generate_two(
        episode1: _Pipeline, episode2: _Pipeline, _vae: Any, **kwargs: Any
    ) -> Any:
        calls["two_calls"].append((episode1.name, episode2.name))
        assert kwargs["initial_latent"].dtype == torch.bfloat16
        prompt = next(
            prompts
            for prompts, sample in sample_by_prompt.items()
            if sample.dataset == STAGE2_TWO_ACTION_DATASET
        )
        assert kwargs["seeds"] == (sample_by_prompt[prompt].seed,)
        return _video_result(sample_by_prompt[prompt])

    def counted_video_writer(video: torch.Tensor, path: Path, *, fps: int) -> None:
        calls["video_writes"] += 1
        _video_writer(video, path, fps=fps)

    return Stage2InferenceRuntimeOps(
        build_samples=lambda **_kwargs: samples,
        build_runtime_assets=build_runtime_assets,
        validate_runtime_assets=lambda value: dict(value),
        assert_runtime_asset_identities=assert_runtime_assets,
        build_text_encoder=build_text,
        load_generator=load_generator,
        build_vae=build_vae,
        build_pipeline=build_pipeline,
        load_image=load_image,
        generate_single=generate_single,
        generate_two=generate_two,
        save_video=counted_video_writer,
        probe_video=probe_fn,
        capture_code_version=lambda: dict(_CODE_VERSION),
        **_fake_artifact_ops(calls),
    )


def _context(
    *,
    rank: int = 0,
    world_size: int = 1,
) -> Stage2InferenceDistributedContext:
    return Stage2InferenceDistributedContext(
        rank=rank,
        local_rank=0,
        world_size=world_size,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("profile", ["c8w16k4s4", "c8w16k4s8"])
def test_runtime_loads_once_preencodes_unique_prompts_and_uses_multisink_only_for_episode2(
    tmp_path: Path,
    profile: str,
) -> None:
    config = _config(tmp_path, profile=profile)
    samples = (
        _sample(
            tmp_path,
            dataset=STAGE2_SINGLE_DATASET,
            row_id=0,
            seed=1,
            prompts=("single prompt",),
            profile=profile,
        ),
        _sample(
            tmp_path,
            dataset=STAGE2_TWO_ACTION_DATASET,
            row_id=1,
            seed=2,
            prompts=("action A", "action B"),
            profile=profile,
        ),
    )
    calls: dict[str, Any] = {}
    ops = _runtime_ops(samples, calls)

    result = run_stage2_inference(config, context=_context(), ops=ops)

    assert result["status"] == "complete"
    assert result["local_generated"] == 2
    assert result["local_skipped"] == 0
    assert calls["text_encoder_builds"] == 1
    assert calls["generator_loads"] == 1
    assert calls["vae_builds"] == 1
    assert calls["image_loads"] == 1
    assert calls["vae_encodes"] == 1
    assert calls["prompt_batches"] == [("single prompt", "action A", "action B")]
    assert calls["pipelines"] == ["baseline_c8w16k4s1", profile]
    assert calls["single_calls"] == ["baseline_c8w16k4s1"]
    assert calls["two_calls"] == [("baseline_c8w16k4s1", profile)]
    assert calls["video_writes"] == calls["trace_writes"] == 2
    assert calls["index_writes"] == 2  # one dry render plus one atomic publish
    assert calls["manifest_writes"] == 1
    assert (Path(config.output_root) / STAGE2_INFERENCE_MANIFEST_NAME).is_file()
    assert (Path(config.output_root) / STAGE2_REVIEW_INDEX_NAME).is_file()


def test_runtime_executes_dynamic_sweep_and_forwards_exact_quick_selection(
    tmp_path: Path,
) -> None:
    config = _sweep_config(tmp_path)
    profile = config.profiles[0]
    samples = (
        _sample(
            tmp_path,
            dataset=STAGE2_SINGLE_DATASET,
            row_id=0,
            seed=7,
            prompts=("single prompt",),
            profile=profile,
        ),
        _sample(
            tmp_path,
            dataset=STAGE2_TWO_ACTION_DATASET,
            row_id=0,
            seed=7,
            prompts=("action A", "action B"),
            profile=profile,
        ),
    )
    calls: dict[str, Any] = {}
    seen_plan: dict[str, Any] = {}
    ops = _runtime_ops(samples, calls)

    def build_samples(**kwargs: Any) -> tuple[Stage2InferenceSample, ...]:
        seen_plan.update(kwargs)
        return samples

    result = run_stage2_inference(
        config,
        context=_context(),
        ops=replace(ops, build_samples=build_samples),
    )

    assert result["status"] == "complete"
    assert seen_plan == {
        "single_metadata": config.single_metadata,
        "two_action_metadata": config.two_action_metadata,
        "seeds": (7,),
        "profiles": (profile,),
        "single_row_ids": (0,),
        "two_action_row_ids": (0,),
    }
    assert calls["text_encoder_builds"] == 1
    assert calls["generator_loads"] == 1
    assert calls["vae_builds"] == 1
    assert calls["pipelines"] == [profile]
    assert calls["single_calls"] == [profile]
    assert calls["two_calls"] == [(profile, profile)]


def test_runtime_completes_when_root_identity_drifts_after_video_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    calls: dict[str, Any] = {}
    ops = _runtime_ops((sample,), calls)
    original_identity = stage2_inference_runtime._lstat_directory_identity
    root = Path(config.output_root).resolve()
    drift = {"active": False}

    def drifted_identity(path: Path) -> tuple[int, int, int]:
        device, inode, mode = original_identity(path)
        if drift["active"] and path == root:
            return device + 107, inode + 109, mode
        return device, inode, mode

    def save_then_drift(video: torch.Tensor, path: Path, *, fps: int) -> None:
        ops.save_video(video, path, fps=fps)
        drift["active"] = True

    monkeypatch.setattr(
        stage2_inference_runtime,
        "_lstat_directory_identity",
        drifted_identity,
    )

    result = run_stage2_inference(
        config,
        context=_context(),
        ops=replace(ops, save_video=save_then_drift),
    )

    assert result["status"] == "complete"
    assert result["local_generated"] == 1
    assert (root / sample.output_relative_path).is_file()
    assert (root / sample.trace_relative_path).is_file()
    assert (root / STAGE2_INFERENCE_MANIFEST_NAME).is_file()
    assert (root / STAGE2_REVIEW_INDEX_NAME).is_file()


def test_complete_video_trace_pairs_are_strictly_revalidated_and_never_overwritten(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    samples = (
        _sample(
            tmp_path,
            dataset=STAGE2_SINGLE_DATASET,
            row_id=0,
            seed=1,
            prompts=("single prompt",),
        ),
    )
    first_calls: dict[str, Any] = {}
    run_stage2_inference(
        config,
        context=_context(),
        ops=_runtime_ops(samples, first_calls),
    )
    video = Path(config.output_root) / samples[0].output_relative_path
    trace = Path(config.output_root) / samples[0].trace_relative_path
    before = (video.read_bytes(), trace.read_bytes())

    second_calls: dict[str, Any] = {}
    result = run_stage2_inference(
        config,
        context=_context(),
        ops=_runtime_ops(samples, second_calls),
    )

    assert result["local_generated"] == 0
    assert result["local_skipped"] == 1
    assert second_calls["generator_loads"] == 1
    assert second_calls["video_writes"] == 0
    assert second_calls["trace_writes"] == 0
    assert second_calls["manifest_writes"] == 0
    assert second_calls["index_writes"] == 1  # compare-only dry render
    assert (video.read_bytes(), trace.read_bytes()) == before


def test_same_path_runtime_asset_content_drift_rejects_resume_before_any_write(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    samples = (
        _sample(
            tmp_path,
            dataset=STAGE2_SINGLE_DATASET,
            row_id=0,
            seed=1,
            prompts=("single prompt",),
        ),
        _sample(
            tmp_path,
            dataset=STAGE2_TWO_ACTION_DATASET,
            row_id=1,
            seed=2,
            prompts=("action A", "action B"),
        ),
    )
    run_stage2_inference(
        config,
        context=_context(),
        ops=_runtime_ops(samples, {}),
    )

    calls: dict[str, Any] = {}
    ops = _runtime_ops(samples, calls)
    changed_assets = deepcopy(_runtime_assets(config))
    identity_body = dict(changed_assets["identity"])
    identity_body.pop("identity_sha256")
    identity_body["assets"]["t5_checkpoint"]["aggregate_sha256"] = "f" * 64
    changed_assets["identity"] = {
        **identity_body,
        "identity_sha256": canonical_json_sha256(identity_body),
    }

    with pytest.raises(RuntimeError, match="inference config"):
        run_stage2_inference(
            config,
            context=_context(),
            ops=replace(
                ops,
                build_runtime_assets=lambda _config: changed_assets,
            ),
        )

    assert calls["generator_loads"] == 1
    assert calls["vae_builds"] == 0
    assert calls["video_writes"] == 0
    assert calls["trace_writes"] == 0


def test_code_version_drift_resumes_without_rewriting_existing_samples(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    run_stage2_inference(
        config,
        context=_context(),
        ops=_runtime_ops((sample,), {}),
    )
    root = Path(config.output_root)
    trace_path = root / sample.trace_relative_path
    trace = json.loads(trace_path.read_text())
    trace["code_version"] = {"stage2_source_sha256": "f" * 64}
    atomic_write_json(trace_path, trace)
    manifest_path = root / STAGE2_INFERENCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["code_version"] = {"stage2_source_sha256": "f" * 64}
    atomic_write_json(manifest_path, manifest)
    original_trace = trace_path.read_bytes()
    original_manifest = manifest_path.read_bytes()

    calls: dict[str, Any] = {}
    ops = _runtime_ops((sample,), calls)
    result = run_stage2_inference(config, context=_context(), ops=ops)

    assert result["local_skipped"] == 1
    assert result["local_generated"] == 0
    assert calls["generator_loads"] == 1
    assert calls["video_writes"] == 0
    assert calls["trace_writes"] == 0
    assert trace_path.read_bytes() == original_trace
    assert manifest_path.read_bytes() == original_manifest


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("architecture_root", "/models/wan-v2"),
        ("t5_checkpoint", "/models/t5-v2.pth"),
        ("tokenizer_dir", "/models/tokenizer-v2"),
        ("vae_checkpoint", "/models/vae-v2.pth"),
    ],
)
def test_resume_rejects_dependency_config_drift_instead_of_skipping(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    first_calls: dict[str, Any] = {}
    run_stage2_inference(
        config,
        context=_context(),
        ops=_runtime_ops((sample,), first_calls),
    )

    changed = replace(config, **{field: replacement})
    second_calls: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="inference config"):
        run_stage2_inference(
            changed,
            context=_context(),
            ops=_runtime_ops((sample,), second_calls),
        )

    assert second_calls["generator_loads"] == 1
    assert second_calls["video_writes"] == 0
    assert second_calls["trace_writes"] == 0


def test_cross_rank_resolved_config_drift_fails_before_loading_models(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    calls: dict[str, Any] = {}
    ops = _runtime_ops((sample,), calls)
    remote_config = replace(config, t5_checkpoint="/models/t5-v2.pth")
    remote_identity = build_stage2_inference_config_identity(
        remote_config,
        runtime_assets=_runtime_asset_identity(remote_config),
    )

    def gather(_context: Any, payload: dict[str, Any]) -> tuple[Any, ...]:
        if "runtime_assets" in payload:
            return payload, {"error": None, "runtime_assets": None}
        remote = dict(payload)
        remote["inference_config"] = remote_identity
        remote["local_sample_keys"] = []
        return payload, remote

    with pytest.raises(RuntimeError, match="different inference configs"):
        run_stage2_inference(
            config,
            context=_context(world_size=2),
            ops=replace(ops, all_gather_object=gather),
        )

    assert calls["text_encoder_builds"] == 0
    assert calls["generator_loads"] == 0
    assert calls["vae_builds"] == 0


def test_checkpoint_consensus_fails_before_vae_or_sample_write(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    calls: dict[str, Any] = {}
    ops = _runtime_ops((sample,), calls)

    def gather(_context: Any, payload: dict[str, Any]) -> tuple[Any, ...]:
        if "runtime_assets" in payload:
            return payload, {"error": None, "runtime_assets": None}
        remote = dict(payload)
        if "inference_config" in payload:
            remote["local_sample_keys"] = []
        elif "checkpoint" in payload and "generated" not in payload:
            remote["checkpoint"] = {
                **dict(payload["checkpoint"]),
                "manifest_sha256": "f" * 64,
            }
        return payload, remote

    with pytest.raises(RuntimeError, match="different Generator checkpoints"):
        run_stage2_inference(
            config,
            context=_context(world_size=2),
            ops=replace(ops, all_gather_object=gather),
        )

    assert calls["generator_loads"] == 1
    assert calls["vae_builds"] == 0
    assert calls["video_writes"] == 0
    assert calls["trace_writes"] == 0


def test_remote_rank_startup_error_is_observed_before_loading_models(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    calls: dict[str, Any] = {}
    ops = _runtime_ops((sample,), calls)

    def gather(_context: Any, payload: dict[str, Any]) -> tuple[Any, ...]:
        if "runtime_assets" in payload:
            return payload, {"error": None, "runtime_assets": None}
        remote = dict(payload)
        remote["error"] = "RuntimeError: output root preparation failed"
        remote["local_sample_keys"] = []
        return payload, remote

    with pytest.raises(RuntimeError, match="startup failed on another rank"):
        run_stage2_inference(
            config,
            context=_context(world_size=2),
            ops=replace(ops, all_gather_object=gather),
        )

    assert calls["text_encoder_builds"] == 0
    assert calls["generator_loads"] == 0
    assert calls["vae_builds"] == 0


@pytest.mark.parametrize("failure", ["output_root", "build_samples", "shard"])
def test_every_local_startup_failure_enters_collective_before_raising(
    tmp_path: Path,
    failure: str,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    calls: dict[str, Any] = {}
    ops = _runtime_ops((sample,), calls)
    gathered_payloads: list[dict[str, Any]] = []

    if failure == "output_root":
        blocked_root = tmp_path / "blocked-output"
        blocked_root.write_text("not a directory", encoding="utf-8")
        config = replace(config, output_root=str(blocked_root))
    elif failure == "build_samples":

        def fail_build(**_kwargs: Any) -> tuple[Stage2InferenceSample, ...]:
            raise RuntimeError("sample build failed")

        ops = replace(ops, build_samples=fail_build)
    else:

        def fail_shard(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
            raise RuntimeError("sample shard failed")

        ops = replace(ops, shard_samples=fail_shard)

    def gather(_context: Any, payload: dict[str, Any]) -> tuple[Any, ...]:
        gathered_payloads.append(dict(payload))
        if "runtime_assets" in payload:
            return payload, {"error": None, "runtime_assets": None}
        return payload, dict(payload)

    with pytest.raises((RuntimeError, ValueError)):
        run_stage2_inference(
            config,
            context=_context(world_size=2),
            ops=replace(ops, all_gather_object=gather),
        )

    startup_payloads = [
        payload for payload in gathered_payloads if "inference_config" in payload
    ]
    assert len(startup_payloads) == 1
    assert startup_payloads[0]["error"] is not None
    assert calls["text_encoder_builds"] == 0
    assert calls["generator_loads"] == 0
    assert calls["vae_builds"] == 0


def test_valid_video_only_artifact_is_quarantined_and_regenerated(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    root = Path(config.output_root)
    video = root / sample.output_relative_path
    video.parent.mkdir(parents=True)
    _video_writer(
        torch.zeros((1, 96, 3, sample.height, sample.width)),
        video,
        fps=24,
    )
    orphan_payload = video.read_bytes()
    calls: dict[str, Any] = {}

    result = run_stage2_inference(
        config,
        context=_context(),
        ops=_runtime_ops((sample,), calls),
    )

    recovered = tuple((root / ".stage2-incomplete").rglob("*.mp4"))
    assert result["status"] == "complete"
    assert result["local_generated"] == 1
    assert calls["generator_loads"] == 1
    assert calls["video_writes"] == calls["trace_writes"] == 1
    assert len(recovered) == 1
    assert recovered[0].read_bytes() == orphan_payload
    assert video.is_file()
    assert (root / sample.trace_relative_path).is_file()
    assert (root / STAGE2_INFERENCE_MANIFEST_NAME).is_file()


def test_invalid_video_only_artifact_fails_without_mutation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    root = Path(config.output_root)
    video = root / sample.output_relative_path
    video.parent.mkdir(parents=True)
    video.write_bytes(b"invalid orphan")
    calls: dict[str, Any] = {}

    def reject_orphan(_path: Path) -> dict[str, Any]:
        raise RuntimeError("invalid orphan video")

    with pytest.raises(RuntimeError, match="invalid orphan video"):
        run_stage2_inference(
            config,
            context=_context(),
            ops=_runtime_ops((sample,), calls, probe_fn=reject_orphan),
        )

    assert calls["generator_loads"] == 1
    assert calls["vae_builds"] == 0
    assert video.read_bytes() == b"invalid orphan"
    assert not (root / ".stage2-incomplete").exists()


def test_video_only_symlink_fails_without_mutation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    root = Path(config.output_root)
    video = root / sample.output_relative_path
    video.parent.mkdir(parents=True)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    video.symlink_to(outside)

    with pytest.raises(RuntimeError, match="non-empty regular file"):
        run_stage2_inference(
            config,
            context=_context(),
            ops=_runtime_ops((sample,), {}),
        )

    assert video.is_symlink()
    assert outside.read_bytes() == b"outside"
    assert not (root / ".stage2-incomplete").exists()


def test_video_only_artifact_is_not_recovered_when_manifest_exists(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    root = Path(config.output_root)
    video = root / sample.output_relative_path
    video.parent.mkdir(parents=True)
    _video_writer(
        torch.zeros((1, 96, 3, sample.height, sample.width)),
        video,
        fps=24,
    )
    atomic_write_json(root / STAGE2_INFERENCE_MANIFEST_NAME, {"status": "complete"})

    with pytest.raises(RuntimeError, match="complete manifest"):
        run_stage2_inference(
            config,
            context=_context(),
            ops=_runtime_ops((sample,), {}),
        )

    assert video.is_file()
    assert not (root / ".stage2-incomplete").exists()


def test_quarantined_video_only_artifact_survives_another_generation_failure(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    root = Path(config.output_root)
    video = root / sample.output_relative_path
    video.parent.mkdir(parents=True)
    _video_writer(
        torch.zeros((1, 96, 3, sample.height, sample.width)),
        video,
        fps=24,
    )
    orphan_payload = video.read_bytes()
    probe_calls = {"count": 0}

    def reject_new_video_after_recovery(path: Path) -> dict[str, Any]:
        probe_calls["count"] += 1
        if probe_calls["count"] > 2:
            raise RuntimeError("new video probe failed")
        return _video_probe(path)

    with pytest.raises(RuntimeError, match="new video probe failed"):
        run_stage2_inference(
            config,
            context=_context(),
            ops=_runtime_ops(
                (sample,),
                {},
                probe_fn=reject_new_video_after_recovery,
            ),
        )

    recovered = tuple((root / ".stage2-incomplete").rglob("*.mp4"))
    assert len(recovered) == 1
    assert recovered[0].read_bytes() == orphan_payload
    assert not video.exists()
    assert not (root / sample.trace_relative_path).exists()
    assert not (root / STAGE2_INFERENCE_MANIFEST_NAME).exists()

    result = run_stage2_inference(
        config,
        context=_context(),
        ops=_runtime_ops((sample,), {}),
    )

    assert result["status"] == "complete"
    assert result["local_generated"] == 1
    assert video.is_file()
    assert (root / sample.trace_relative_path).is_file()
    assert recovered[0].read_bytes() == orphan_payload


def test_trace_only_artifact_still_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    root = Path(config.output_root)
    trace = root / sample.trace_relative_path
    trace.parent.mkdir(parents=True)
    trace.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"video\+trace pair"):
        run_stage2_inference(
            config,
            context=_context(),
            ops=_runtime_ops((sample,), {}),
        )

    assert trace.read_text(encoding="utf-8") == "{}\n"
    assert not (root / ".stage2-incomplete").exists()


def test_video_probe_failure_commits_neither_pair_nor_complete_manifest(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    calls: dict[str, Any] = {}

    def reject_probe(_path: Path) -> dict[str, Any]:
        raise RuntimeError("probe rejected")

    with pytest.raises(RuntimeError, match="probe rejected"):
        run_stage2_inference(
            config,
            context=_context(),
            ops=_runtime_ops((sample,), calls, probe_fn=reject_probe),
        )

    root = Path(config.output_root)
    assert not (root / sample.output_relative_path).exists()
    assert not (root / sample.trace_relative_path).exists()
    assert not (root / STAGE2_INFERENCE_MANIFEST_NAME).exists()


def test_inference_never_invokes_legacy_code_version_callback(tmp_path: Path) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    calls: dict[str, Any] = {}

    def reject_capture():
        raise AssertionError("source code must not be hashed during inference")

    ops = replace(
        _runtime_ops((sample,), calls),
        capture_code_version=reject_capture,
    )
    result = run_stage2_inference(config, context=_context(), ops=ops)
    assert result["status"] == "complete"
    assert calls["video_writes"] == 1
    manifest_path = Path(config.output_root) / STAGE2_INFERENCE_MANIFEST_NAME
    assert json.loads(manifest_path.read_text())["code_version"] == {}


def test_metadata_change_during_sample_planning_stops_before_loading_models(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    calls: dict[str, Any] = {}
    ops = _runtime_ops((sample,), calls)

    def mutate_metadata(**_kwargs: Any):
        Path(config.single_metadata).write_text("changed\n", encoding="utf-8")
        return (sample,)

    with pytest.raises(RuntimeError, match="metadata changed while planning"):
        run_stage2_inference(
            config,
            context=_context(),
            ops=replace(ops, build_samples=mutate_metadata),
        )

    assert calls["text_encoder_builds"] == 0
    assert calls["generator_loads"] == 0
    assert calls["vae_builds"] == 0


def test_manifest_publish_failure_never_leaves_a_complete_marker(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    sample = _sample(
        tmp_path,
        dataset=STAGE2_SINGLE_DATASET,
        row_id=0,
        seed=1,
        prompts=("single prompt",),
    )
    calls: dict[str, Any] = {}
    ops = _runtime_ops((sample,), calls)

    def reject_manifest(*_args: Any, **_kwargs: Any) -> Path:
        raise RuntimeError("manifest publish rejected")

    with pytest.raises(RuntimeError, match="manifest publish rejected"):
        run_stage2_inference(
            config,
            context=_context(),
            ops=replace(ops, write_manifest=reject_manifest),
        )

    root = Path(config.output_root)
    assert (root / STAGE2_REVIEW_INDEX_NAME).is_file()
    assert not (root / STAGE2_INFERENCE_MANIFEST_NAME).exists()
