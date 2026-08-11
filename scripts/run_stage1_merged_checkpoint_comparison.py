#!/usr/bin/env python3
"""Compare pre-merged and runtime-LoRA Stage-1 inference on four GPUs.

When an old validation result is available, its prepared carrier videos and
configs are reused.  Otherwise they are built directly from the metadata CSV
and model assets.  Both checkpoint formats always receive the same inputs,
sampling parameters, and per-row noise seeds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable
from urllib.parse import quote

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import DEFAULT_NEGATIVE_PROMPT, normalize_config  # noqa: E402
from utils.stage1_causal_validation import (  # noqa: E402
    CAUSAL_TESTSET_SCHEMA_VERSION,
    load_causal_testset_records,
    prepare_causal_testsets,
    probe_video,
    validate_causal_testset_outputs,
)
from utils.stage1_io import (  # noqa: E402
    atomic_output_path,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from utils.stage1_checkpoint import (  # noqa: E402
    checkpoint_step,
    validate_checkpoint,
)

REPORT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class InferenceJob:
    variant: str
    bucket_id: str
    gpu_id: str
    sample_indices: tuple[int, ...]
    config_path: str
    log_path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-checkpoint", required=True)
    parser.add_argument(
        "--merged-manifest",
        required=True,
        help="Companion merge manifest proving the pre-merged checkpoint provenance.",
    )
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        help="Immutable causal base used by the dynamic LoRA inference path.",
    )
    parser.add_argument(
        "--training-checkpoint",
        required=True,
        help="Stage-1 checkpoint_model_XXXXXX containing adapter_ema.safetensors.",
    )
    parser.add_argument(
        "--reference-checkpoint-dir",
        default=None,
        help=(
            "Optional existing checkpoint_model_XXXXXX inference result. "
            "When omitted, shared inputs are prepared directly from metadata."
        ),
    )
    parser.add_argument(
        "--architecture-root",
        help="Required only when --reference-checkpoint-dir is omitted.",
    )
    parser.add_argument(
        "--t5-checkpoint",
        help="Required only when --reference-checkpoint-dir is omitted.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        help="Required only when --reference-checkpoint-dir is omitted.",
    )
    parser.add_argument(
        "--vae-checkpoint",
        help="Required only when --reference-checkpoint-dir is omitted.",
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument(
        "--gpu-ids",
        default="0,1,2,3",
        help="Comma-separated physical GPU ids; four H100s are recommended.",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="A new or empty directory. Existing results are never overwritten.",
    )
    parser.add_argument("--minimum-first-frame-psnr-db", type=float, default=12.0)
    parser.add_argument("--minimum-frame-std", type=float, default=5.0)
    parser.add_argument("--minimum-temporal-abs-diff", type=float, default=0.05)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    return parser


def _parse_gpu_ids(value: str) -> tuple[str, ...]:
    gpu_ids = tuple(piece.strip() for piece in str(value).split(","))
    if not gpu_ids or any(not gpu_id for gpu_id in gpu_ids):
        raise ValueError("--gpu-ids must be a comma-separated non-empty list")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpu-ids contains duplicates: {gpu_ids}")
    return gpu_ids


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _assert_reference_manifest_integrity(manifest: dict[str, Any]) -> None:
    if int(manifest.get("schema_version", -1)) != CAUSAL_TESTSET_SCHEMA_VERSION:
        raise RuntimeError("Unsupported reference prepared-manifest schema version")
    recorded_hash = manifest.get("manifest_sha256")
    unhashed = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    actual_hash = canonical_json_sha256(unhashed)
    if recorded_hash != actual_hash:
        raise RuntimeError(
            "Reference prepared manifest does not match its recorded SHA256"
        )


def _assert_metadata_matches_reference(
    metadata_path: Path,
    reference_manifest: dict[str, Any],
) -> list[Any]:
    metadata = reference_manifest.get("metadata", {})
    actual_sha256 = sha256_file(metadata_path)
    if metadata.get("sha256") != actual_sha256:
        raise RuntimeError(
            "Requested metadata CSV differs from the reference inference metadata: "
            f"expected {metadata.get('sha256')}, got {actual_sha256}"
        )
    records = load_causal_testset_records(metadata_path)
    if int(metadata.get("record_count", -1)) != len(records):
        raise RuntimeError(
            "Reference metadata record count differs from the current CSV"
        )

    current_by_row = {record.row_id: record for record in records}
    reference_by_row: dict[int, dict[str, Any]] = {}
    for bucket in reference_manifest.get("buckets", []):
        for record in bucket.get("records", []):
            row_id = int(record["row_id"])
            if row_id in reference_by_row:
                raise RuntimeError(f"Reference row {row_id} appears more than once")
            reference_by_row[row_id] = record
    if set(reference_by_row) != set(current_by_row):
        raise RuntimeError(
            "Reference rows differ from the requested metadata rows: "
            f"reference={sorted(reference_by_row)}, current={sorted(current_by_row)}"
        )
    for row_id, record in current_by_row.items():
        reference = reference_by_row[row_id]
        expected = {
            "row_sha256": record.row_sha256,
            "image_sha256": record.image_sha256,
            "height": record.height,
            "width": record.width,
            "bucket_id": record.bucket_id,
        }
        wrong = {
            key: {"expected": value, "actual": reference.get(key)}
            for key, value in expected.items()
            if reference.get(key) != value
        }
        if wrong:
            raise RuntimeError(f"Reference metadata row {row_id} mismatch: {wrong}")
    return records


def prepare_fresh_comparison_inputs(
    *,
    metadata_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    base_checkpoint: str | os.PathLike[str],
    architecture_root: str | os.PathLike[str],
    t5_checkpoint: str | os.PathLike[str],
    tokenizer_dir: str | os.PathLike[str],
    vae_checkpoint: str | os.PathLike[str],
    sampling_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: int = 1,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    preparation_builder: Callable[..., dict[str, Any]] = prepare_causal_testsets,
) -> tuple[Path, dict[str, Any]]:
    """Build one immutable shared preparation when no old result exists."""

    output_root = Path(output_root).expanduser().resolve()
    manifest = preparation_builder(
        metadata_path=metadata_path,
        output_root=output_root,
        base_checkpoint=base_checkpoint,
        architecture_root=architecture_root,
        t5_checkpoint=t5_checkpoint,
        tokenizer_dir=tokenizer_dir,
        vae_checkpoint=vae_checkpoint,
        num_latent_frames=24,
        num_frame_per_block=8,
        temporal_compression_ratio=4,
        minimum_source_frames=97,
        fps=24,
        sampling_steps=int(sampling_steps),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        negative_prompt=str(negative_prompt),
    )
    manifest_path = output_root / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Fresh preparation did not publish its manifest: {manifest_path}"
        )
    persisted = _load_json(manifest_path)
    _assert_reference_manifest_integrity(persisted)
    if persisted != manifest:
        raise RuntimeError("Fresh prepared manifest differs from the returned manifest")
    return manifest_path, persisted


def _resolve_fresh_preparation_assets(args: argparse.Namespace) -> dict[str, Path]:
    specs = {
        "architecture_root": (getattr(args, "architecture_root", None), True),
        "t5_checkpoint": (getattr(args, "t5_checkpoint", None), False),
        "tokenizer_dir": (getattr(args, "tokenizer_dir", None), True),
        "vae_checkpoint": (getattr(args, "vae_checkpoint", None), False),
    }
    missing_args = [name for name, (value, _is_dir) in specs.items() if not value]
    if missing_args:
        flags = [f"--{name.replace('_', '-')}" for name in missing_args]
        raise ValueError(
            "Fresh preparation requires model asset arguments when no reference "
            f"directory is supplied: {flags}"
        )
    resolved: dict[str, Path] = {}
    for name, (value, is_dir) in specs.items():
        path = Path(value).expanduser().resolve()
        exists = path.is_dir() if is_dir else path.is_file()
        if not exists:
            expected = "directory" if is_dir else "file"
            raise FileNotFoundError(f"Expected {name} {expected}: {path}")
        resolved[name] = path
    return resolved


def validate_merged_checkpoint_provenance(
    *,
    merged_checkpoint: str | os.PathLike[str],
    merged_manifest_path: str | os.PathLike[str],
    base_sha256: str,
    training_manifest_sha256: str,
    training_step: int,
    adapter_sha256: str,
) -> dict[str, Any]:
    """Prove that the full checkpoint was merged from this base and EMA LoRA."""

    merged_checkpoint = Path(merged_checkpoint).expanduser().resolve()
    merged_manifest_path = Path(merged_manifest_path).expanduser().resolve()
    if not merged_checkpoint.is_file():
        raise FileNotFoundError(merged_checkpoint)
    if not merged_manifest_path.is_file():
        raise FileNotFoundError(merged_manifest_path)
    manifest = _load_json(merged_manifest_path)
    recorded_manifest_sha256 = manifest.get("manifest_sha256")
    unhashed = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    actual_manifest_sha256 = canonical_json_sha256(unhashed)
    if recorded_manifest_sha256 != actual_manifest_sha256:
        raise RuntimeError("Merged checkpoint manifest SHA256 is invalid")
    if manifest.get("schema") != "longlive_stage1_merge_manifest" or int(
        manifest.get("schema_version", -1)
    ) != 2:
        raise RuntimeError("Unsupported Stage-1 merged checkpoint manifest")

    output = manifest.get("output", {})
    training = manifest.get("training_checkpoint", {})
    expected = {
        "base.sha256": base_sha256,
        "training.completed_step": int(training_step),
        "training.manifest_sha256": training_manifest_sha256,
        "training.adapter": "adapter_ema.safetensors",
        "training.ema_adapter.sha256": adapter_sha256,
        "output.sha256": sha256_file(merged_checkpoint),
        "output.size": merged_checkpoint.stat().st_size,
        "output.dtype": "bfloat16",
        "output.strict_reload": True,
    }
    actual = {
        "base.sha256": manifest.get("base", {}).get("sha256"),
        "training.completed_step": training.get("completed_step"),
        "training.manifest_sha256": training.get("manifest_sha256"),
        "training.adapter": training.get("adapter"),
        "training.ema_adapter.sha256": training.get("ema_adapter", {}).get(
            "sha256"
        ),
        "output.sha256": output.get("sha256"),
        "output.size": output.get("size"),
        "output.dtype": output.get("dtype"),
        "output.strict_reload": output.get("strict_reload"),
    }
    wrong = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    if wrong:
        raise RuntimeError(f"Merged checkpoint provenance mismatch: {wrong}")
    return {
        "path": os.fspath(merged_checkpoint),
        "sha256": expected["output.sha256"],
        "size": expected["output.size"],
        "manifest_path": os.fspath(merged_manifest_path),
        "manifest_sha256": recorded_manifest_sha256,
        "source_training_step": int(training_step),
        "source_adapter": "adapter_ema.safetensors",
    }


def _assert_reference_config_contract(
    config_path: Path,
    *,
    bucket: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    config = normalize_config(OmegaConf.load(config_path))
    if getattr(config, "adapter", None) is not None:
        raise RuntimeError(
            f"Reference config unexpectedly enables a LoRA adapter: {config_path}"
        )
    if getattr(config, "lora_ckpt", None):
        raise RuntimeError(
            f"Reference config unexpectedly loads LoRA weights: {config_path}"
        )
    if bool(getattr(config, "use_ema", False)):
        raise RuntimeError(
            f"Reference config must load a full merged generator: {config_path}"
        )
    if int(getattr(config, "num_samples", -1)) != 1:
        raise RuntimeError(f"Reference config num_samples must be 1: {config_path}")
    if not bool(getattr(config, "save_with_index", False)):
        raise RuntimeError(
            f"Reference config must use indexed filenames: {config_path}"
        )

    sampling = manifest["sampling"]
    frame_policy = manifest["frame_policy"]
    expected = {
        "sampling_steps": int(sampling["sampling_steps"]),
        "guidance_scale": float(sampling["guidance_scale"]),
        "seed": int(sampling["seed"]),
        "negative_prompt": str(sampling["negative_prompt"]),
        "num_output_frames": int(frame_policy["num_latent_frames"]),
        "data_path": os.fspath(Path(bucket["data_root"]).resolve()),
    }
    actual = {
        "sampling_steps": int(config.sampling_steps),
        "guidance_scale": float(config.guidance_scale),
        "seed": int(config.seed),
        "negative_prompt": str(config.negative_prompt),
        "num_output_frames": int(config.num_output_frames),
        "data_path": os.fspath(Path(config.data_path).resolve()),
    }
    wrong = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value
    }
    if wrong:
        raise RuntimeError(f"Reference inference config/manifest mismatch: {wrong}")


def _clone_reference_variant(
    *,
    reference_manifest_path: str | os.PathLike[str],
    generator_checkpoint: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    variant: str,
    output_model_type: str,
    adapter_config: Any | None = None,
    lora_checkpoint: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Clone one reference-prepared dataset for a checkpoint loading variant."""

    reference_manifest_path = Path(reference_manifest_path).expanduser().resolve()
    generator_checkpoint = Path(generator_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not reference_manifest_path.is_file():
        raise FileNotFoundError(reference_manifest_path)
    if not generator_checkpoint.is_file():
        raise FileNotFoundError(generator_checkpoint)
    lora_path = (
        None
        if lora_checkpoint is None
        else Path(lora_checkpoint).expanduser().resolve()
    )
    if (adapter_config is None) != (lora_path is None):
        raise ValueError("adapter_config and lora_checkpoint must be provided together")
    if lora_path is not None and not lora_path.is_file():
        raise FileNotFoundError(lora_path)
    if variant not in {"premerged", "dynamic_lora"}:
        raise ValueError(f"Unsupported inference variant: {variant!r}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Merged inference directory must be empty: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    reference = _load_json(reference_manifest_path)
    _assert_reference_manifest_integrity(reference)
    cloned = deepcopy(reference)
    cloned.pop("manifest_sha256", None)
    cloned["reference_prepared_manifest"] = os.fspath(reference_manifest_path)
    cloned["model_paths"]["base_checkpoint"] = os.fspath(generator_checkpoint)
    cloned["inference_variant"] = {
        "name": variant,
        "generator_checkpoint": os.fspath(generator_checkpoint),
        "lora_checkpoint": None if lora_path is None else os.fspath(lora_path),
        "output_model_type": output_model_type,
    }
    base_seed = int(reference["sampling"]["seed"])

    for bucket in cloned["buckets"]:
        bucket_id = str(bucket["bucket_id"])
        source_config_path = Path(bucket["config_path"]).expanduser().resolve()
        data_root = Path(bucket["data_root"]).expanduser().resolve()
        if not source_config_path.is_file():
            raise FileNotFoundError(source_config_path)
        if not data_root.is_dir():
            raise FileNotFoundError(data_root)
        for record in bucket["records"]:
            for key in ("carrier_video", "caption_json"):
                artifact = Path(record[key]).expanduser().resolve()
                if not artifact.is_file():
                    raise FileNotFoundError(artifact)

        _assert_reference_config_contract(
            source_config_path,
            bucket=bucket,
            manifest=reference,
        )
        config = OmegaConf.load(source_config_path)
        if config.get("adapter", None) is not None:
            raise RuntimeError(
                f"Refusing to clone adapter-enabled config: {source_config_path}"
            )
        if config.get("checkpoints", None) is None:
            config.checkpoints = OmegaConf.create({})
        config.checkpoints.generator_ckpt = os.fspath(generator_checkpoint)
        config.checkpoints.pop("lora_ckpt", None)
        config.generator_ckpt = os.fspath(generator_checkpoint)
        config.pop("lora_ckpt", None)
        config.use_ema = False
        config.num_samples = 1
        config.save_with_index = True
        config.merge_lora = False

        if lora_path is None:
            config.pop("adapter", None)
        else:
            config.adapter = OmegaConf.create(
                OmegaConf.to_container(adapter_config, resolve=True)
                if OmegaConf.is_config(adapter_config)
                else deepcopy(adapter_config)
            )
            config.checkpoints.lora_ckpt = os.fspath(lora_path)
            config.lora_ckpt = os.fspath(lora_path)

        sample_seeds = [
            base_seed + int(record["row_id"]) for record in bucket["records"]
        ]

        output_dir = output_root / "videos" / bucket_id
        output_dir.mkdir(parents=True, exist_ok=True)
        config.output_folder = os.fspath(output_dir)
        if config.get("inference", None) is not None:
            config.inference.output_folder = os.fspath(output_dir)
            config.inference.use_ema = False
            config.inference.merge_lora = False
            config.inference.sample_seeds = sample_seeds
            config.inference.dataloader_num_workers = 1
            config.inference.prefetch_factor = 2
            config.inference.pin_memory = True
        else:
            config.inference = OmegaConf.create(
                {
                    "sample_seeds": sample_seeds,
                    "dataloader_num_workers": 1,
                    "prefetch_factor": 2,
                    "pin_memory": True,
                }
            )

        config_path = output_root / "configs" / f"{bucket_id}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_text = OmegaConf.to_yaml(config, resolve=False, sort_keys=False)
        atomic_write_bytes(config_path, config_text.encode("utf-8"))
        bucket["reference_config_path"] = os.fspath(source_config_path)
        bucket["config_path"] = os.fspath(config_path)
        bucket["output_dir"] = os.fspath(output_dir)
        bucket["output_model_type"] = output_model_type
        bucket["sample_seeds"] = sample_seeds

    cloned["manifest_sha256"] = canonical_json_sha256(cloned)
    atomic_write_json(output_root / "prepared_manifest.json", cloned)
    return cloned


def clone_reference_preparation(
    *,
    reference_manifest_path: str | os.PathLike[str],
    merged_checkpoint: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Clone reference inputs for a pre-merged full-generator checkpoint."""

    return _clone_reference_variant(
        reference_manifest_path=reference_manifest_path,
        generator_checkpoint=merged_checkpoint,
        output_root=output_root,
        variant="premerged",
        output_model_type="regular",
    )


def clone_reference_lora_preparation(
    *,
    reference_manifest_path: str | os.PathLike[str],
    base_checkpoint: str | os.PathLike[str],
    lora_checkpoint: str | os.PathLike[str],
    adapter_config: Any,
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Clone reference inputs for runtime ``base + LoRA`` inference."""

    return _clone_reference_variant(
        reference_manifest_path=reference_manifest_path,
        generator_checkpoint=base_checkpoint,
        lora_checkpoint=lora_checkpoint,
        adapter_config=adapter_config,
        output_root=output_root,
        variant="dynamic_lora",
        output_model_type="lora",
    )


def _allocate_bucket_workers(
    sample_counts: list[int],
    worker_budget: int,
) -> list[int]:
    if not sample_counts or any(count <= 0 for count in sample_counts):
        raise ValueError(f"Bucket sample counts must be positive: {sample_counts}")
    if worker_budget < len(sample_counts):
        raise ValueError(
            f"At least {len(sample_counts)} GPUs are required for the geometry buckets"
        )
    worker_budget = min(int(worker_budget), sum(sample_counts))
    workers = [1] * len(sample_counts)
    while sum(workers) < worker_budget:
        candidates = [
            index
            for index, count in enumerate(sample_counts)
            if workers[index] < count
        ]
        if not candidates:
            break
        selected = max(
            candidates,
            key=lambda index: (sample_counts[index] / workers[index], -index),
        )
        workers[selected] += 1
    return workers


def build_parallel_inference_jobs(
    prepared_manifest_path: str | os.PathLike[str],
    *,
    gpu_ids: tuple[str, ...],
) -> list[InferenceJob]:
    """Shard each fixed-geometry bucket over independent single-GPU workers."""

    prepared_manifest_path = Path(prepared_manifest_path).expanduser().resolve()
    manifest = _load_json(prepared_manifest_path)
    _assert_reference_manifest_integrity(manifest)
    buckets = manifest["buckets"]
    counts = [len(bucket["records"]) for bucket in buckets]
    worker_counts = _allocate_bucket_workers(counts, len(gpu_ids))
    output_root = prepared_manifest_path.parent
    variant = str(manifest["inference_variant"]["name"])
    jobs: list[InferenceJob] = []
    gpu_cursor = 0

    for bucket, sample_count, worker_count in zip(
        buckets, counts, worker_counts, strict=True
    ):
        bucket_id = str(bucket["bucket_id"])
        source_config = OmegaConf.load(bucket["config_path"])
        for worker_index in range(worker_count):
            sample_indices = tuple(range(worker_index, sample_count, worker_count))
            gpu_id = gpu_ids[gpu_cursor]
            gpu_cursor += 1
            config = OmegaConf.create(
                OmegaConf.to_container(source_config, resolve=False)
            )
            if config.get("inference", None) is None:
                config.inference = OmegaConf.create({})
            config.inference.sample_indices = list(sample_indices)
            worker_name = f"{bucket_id}_gpu{gpu_id}_worker{worker_index}"
            config_path = output_root / "configs" / "workers" / f"{worker_name}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(
                config_path,
                OmegaConf.to_yaml(config, resolve=False, sort_keys=False).encode(
                    "utf-8"
                ),
            )
            jobs.append(
                InferenceJob(
                    variant=variant,
                    bucket_id=bucket_id,
                    gpu_id=gpu_id,
                    sample_indices=sample_indices,
                    config_path=os.fspath(config_path),
                    log_path=os.fspath(
                        output_root / "logs" / f"{worker_name}.log"
                    ),
                )
            )
    if gpu_cursor != len(jobs) or len({job.gpu_id for job in jobs}) != len(jobs):
        raise RuntimeError("Parallel inference jobs did not receive unique GPUs")
    expected = {
        (str(bucket["bucket_id"]), index)
        for bucket in buckets
        for index in range(len(bucket["records"]))
    }
    actual = {
        (job.bucket_id, index) for job in jobs for index in job.sample_indices
    }
    if actual != expected or sum(len(job.sample_indices) for job in jobs) != len(
        expected
    ):
        raise RuntimeError(
            f"Parallel inference sharding is incomplete or duplicated: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return jobs


def run_parallel_inference_jobs(
    jobs: list[InferenceJob],
    *,
    command_runner: Callable[..., Any] = subprocess.run,
) -> list[dict[str, Any]]:
    """Run one independent model process per GPU and preserve per-worker logs."""

    if not jobs:
        raise ValueError("At least one inference job is required")

    def run_job(job: InferenceJob) -> dict[str, Any]:
        env = os.environ.copy()
        for name in (
            "LOCAL_RANK",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            env.pop(name, None)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": job.gpu_id,
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        command = [
            sys.executable,
            os.fspath(PROJECT_ROOT / "inference.py"),
            "--config_path",
            job.config_path,
        ]
        log_path = Path(job.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        print(
            f"[stage1-format-comparison] variant={job.variant} gpu={job.gpu_id} "
            f"bucket={job.bucket_id} indices={list(job.sample_indices)}"
        )
        with log_path.open("wb") as log_handle:
            command_runner(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return {
            **asdict(job),
            "elapsed_seconds": time.monotonic() - started,
            "status": "pass",
        }

    reports: list[dict[str, Any]] = []
    failures = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                reports.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        **asdict(job),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    if failures:
        raise RuntimeError(f"Parallel inference workers failed: {failures}")
    return sorted(reports, key=lambda item: (item["gpu_id"], item["bucket_id"]))


def write_side_by_side_video(
    left_video: str | os.PathLike[str],
    right_video: str | os.PathLike[str],
    output_video: str | os.PathLike[str],
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    video_probe: Callable[[str | os.PathLike[str]], dict[str, Any]] = probe_video,
) -> dict[str, Any]:
    """Write one MP4 with reference on the left and merged output on the right."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create side-by-side videos")
    left_video = Path(left_video).expanduser().resolve()
    right_video = Path(right_video).expanduser().resolve()
    output_video = Path(output_video).expanduser().resolve()
    left = video_probe(left_video)
    right = video_probe(right_video)
    required_equal = ("width", "height", "frame_count", "fps")
    mismatch = {
        key: {"left": left[key], "right": right[key]}
        for key in required_equal
        if left[key] != right[key]
    }
    if mismatch:
        raise RuntimeError(
            f"Cannot pair videos with different stream properties: {mismatch}"
        )
    fps = f"{float(left['fps']):.12g}"
    frame_count = int(left["frame_count"])

    with atomic_output_path(output_video, suffix=".mp4") as temporary:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            os.fspath(left_video),
            "-i",
            os.fspath(right_video),
            "-filter_complex",
            (
                f"[0:v]settb=expr=1/{fps},setpts=N,setsar=1[left];"
                f"[1:v]settb=expr=1/{fps},setpts=N,setsar=1[right];"
                f"[left][right]hstack=inputs=2:shortest=0[stacked];"
                f"[stacked]trim=start_frame=0:end_frame={frame_count},"
                f"settb=expr=1/{fps},setpts=N[paired]"
            ),
            "-map",
            "[paired]",
            "-frames:v",
            str(frame_count),
            "-fps_mode",
            "passthrough",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            os.fspath(temporary),
        ]
        command_runner(command, check=True)

    paired = video_probe(output_video)
    expected = {
        "width": 2 * int(left["width"]),
        "height": int(left["height"]),
        "frame_count": int(left["frame_count"]),
        "fps": float(left["fps"]),
    }
    wrong = {
        key: {"expected": value, "actual": paired[key]}
        for key, value in expected.items()
        if paired[key] != value
    }
    if wrong:
        raise RuntimeError(f"Side-by-side output stream mismatch: {wrong}")
    return paired


def _comparison_html(
    *,
    samples: list[dict[str, Any]],
    work_dir: Path,
) -> str:
    style = """
body{font-family:system-ui,sans-serif;margin:24px;color:#202124;background:#fafafa}
.case{margin:0 0 28px;padding:16px;background:#fff;border:1px solid #ddd;border-radius:8px}
video{display:block;max-width:100%;max-height:640px;background:#111}.meta{font-size:13px;color:#555}
"""
    lines = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        f"<title>Stage-1 merged vs dynamic LoRA</title><style>{style}</style></head><body>",
        "<h1>Stage-1 merged vs dynamic LoRA</h1>",
        "<p><strong>并排视频左侧：</strong>预 merged 完整权重；"
        "<strong>右侧：</strong>base + adapter_ema.safetensors 动态 LoRA。"
        "两路使用相同的逐 row noise seed。</p>",
    ]
    has_historical_reference = any(sample.get("reference_video") for sample in samples)
    if has_historical_reference:
        lines.append(
            "<p>原 infer_stage1 视频仅作为历史参考单独展示；其旧版顺序 RNG "
            "不参与两种权重格式的严格等价判断。</p>"
        )
    for sample in samples:
        video_path = Path(sample["comparison_video"]).resolve()
        relative = video_path.relative_to(work_dir).as_posix()
        source = quote(relative, safe="/._-")
        lines.extend(
            [
                '<section class="case">',
                f"<h2>row {int(sample['row_id'])} · "
                f"{html.escape(sample['bucket_id'])} · seed {int(sample['sample_seed'])}</h2>",
                f'<video controls loop preload="metadata" src="{html.escape(source)}"></video>',
                '<p class="meta">left = pre-merged checkpoint · right = dynamic LoRA</p>',
            ]
        )
        reference_value = sample.get("reference_video")
        if reference_value:
            reference_path = Path(reference_value).resolve()
            try:
                reference_source_path = reference_path.relative_to(
                    work_dir
                ).as_posix()
            except ValueError:
                reference_source_path = os.fspath(reference_path)
            reference_source = quote(reference_source_path, safe="/._-")
            lines.extend(
                [
                    "<details><summary>原 infer_stage1 历史参考</summary>",
                    '<video controls loop preload="metadata" '
                    f'src="{html.escape(reference_source)}"></video>',
                    "</details>",
                ]
            )
        lines.append("</section>")
    lines.append("</body></html>")
    return "".join(lines)


def run_comparison(
    args: argparse.Namespace,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    output_validator: Callable[..., dict[str, Any]] = validate_causal_testset_outputs,
    pair_writer: Callable[..., dict[str, Any]] = write_side_by_side_video,
    preparation_builder: Callable[..., dict[str, Any]] = prepare_causal_testsets,
) -> dict[str, Any]:
    merged_checkpoint = Path(args.merged_checkpoint).expanduser().resolve()
    merged_manifest_path = Path(args.merged_manifest).expanduser().resolve()
    base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    training_checkpoint = Path(args.training_checkpoint).expanduser().resolve()
    reference_dir_value = getattr(args, "reference_checkpoint_dir", None)
    reference_dir = (
        Path(reference_dir_value).expanduser().resolve()
        if reference_dir_value
        else None
    )
    metadata_path = Path(args.metadata).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if len(gpu_ids) != 4:
        raise ValueError(
            "This comparison requires exactly four GPUs: one per "
            "(checkpoint format, geometry bucket) worker"
        )
    reference_manifest_path = (
        reference_dir / "prepared" / "prepared_manifest.json"
        if reference_dir is not None
        else work_dir / "shared_prepared" / "prepared_manifest.json"
    )
    merged_inference_root = work_dir / "merged_inference"
    lora_inference_root = work_dir / "lora_inference"
    report_path = work_dir / "comparison_report.json"

    if not merged_checkpoint.is_file():
        raise FileNotFoundError(merged_checkpoint)
    if not merged_manifest_path.is_file():
        raise FileNotFoundError(merged_manifest_path)
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    if not training_checkpoint.is_dir():
        raise FileNotFoundError(training_checkpoint)
    if reference_dir is not None and not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    training_step = checkpoint_step(training_checkpoint)
    if reference_dir is not None:
        reference_step = checkpoint_step(reference_dir)
        if reference_step != training_step:
            raise RuntimeError(
                "Reference and dynamic LoRA checkpoints select different optimizer "
                f"steps: reference={reference_step}, training={training_step}"
            )
    base_sha256 = sha256_file(base_checkpoint)
    training_manifest = validate_checkpoint(
        training_checkpoint,
        require_resumable=False,
        expected_base_sha256=base_sha256,
        expected_topology=(6, 3, 2),
    )
    resolved_training_config_path = training_checkpoint / "resolved_config.yaml"
    resolved_training_config = normalize_config(
        OmegaConf.load(resolved_training_config_path)
    )
    adapter_config = getattr(resolved_training_config, "adapter", None)
    if adapter_config is None:
        raise RuntimeError(
            f"Stage-1 training config has no adapter section: {resolved_training_config_path}"
        )
    lora_checkpoint = training_checkpoint / "adapter_ema.safetensors"
    if not lora_checkpoint.is_file():
        raise FileNotFoundError(lora_checkpoint)
    adapter_sha256 = sha256_file(lora_checkpoint)
    merged_provenance = validate_merged_checkpoint_provenance(
        merged_checkpoint=merged_checkpoint,
        merged_manifest_path=merged_manifest_path,
        base_sha256=base_sha256,
        training_manifest_sha256=training_manifest["manifest_sha256"],
        training_step=training_step,
        adapter_sha256=adapter_sha256,
    )
    fresh_assets = (
        None if reference_dir is not None else _resolve_fresh_preparation_assets(args)
    )
    if reference_dir is not None and (
        work_dir == reference_dir or work_dir.is_relative_to(reference_dir)
    ):
        raise ValueError(
            "Work directory must be outside the immutable reference result"
        )
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(
            f"Comparison work directory must be empty to reject stale outputs: {work_dir}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "longlive_stage1_premerged_dynamic_lora_comparison",
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "initializing",
        "work_dir": os.fspath(work_dir),
        "metadata": os.fspath(metadata_path),
        "merged_checkpoint": merged_provenance,
        "base_checkpoint": {
            "path": os.fspath(base_checkpoint),
            "sha256": base_sha256,
        },
        "dynamic_lora": {
            "training_checkpoint": os.fspath(training_checkpoint),
            "optimizer_step": training_step,
            "adapter_checkpoint": os.fspath(lora_checkpoint),
            "adapter_sha256": adapter_sha256,
            "checkpoint_manifest_sha256": training_manifest["manifest_sha256"],
        },
        "reference_checkpoint_dir": (
            None if reference_dir is None else os.fspath(reference_dir)
        ),
        "input_preparation": {
            "mode": "reference" if reference_dir is not None else "fresh",
            "reference_checkpoint_dir": (
                None if reference_dir is None else os.fspath(reference_dir)
            ),
            "prepared_manifest": os.fspath(reference_manifest_path),
        },
        "gpu_ids": list(gpu_ids),
        "execution": "four_concurrent_single_gpu_format_bucket_workers",
        "side_by_side_order": {
            "left": "premerged_checkpoint",
            "right": "dynamic_lora",
        },
    }
    atomic_write_json(report_path, report)

    try:
        validator_kwargs = {
            "minimum_first_frame_psnr_db": args.minimum_first_frame_psnr_db,
            "minimum_frame_std": args.minimum_frame_std,
            "minimum_temporal_abs_diff": args.minimum_temporal_abs_diff,
        }
        reference_outputs: dict[str, Any] | None = None
        if reference_dir is not None:
            reference_manifest = _load_json(reference_manifest_path)
            _assert_reference_manifest_integrity(reference_manifest)
            report["status"] = "validating_reference"
            atomic_write_json(report_path, report)
            reference_outputs = output_validator(
                reference_manifest_path,
                **validator_kwargs,
            )
            reference_output_validation_path = (
                work_dir / "reference_output_validation.json"
            )
            atomic_write_json(
                reference_output_validation_path,
                reference_outputs,
            )
            report["reference_output_validation"] = os.fspath(
                reference_output_validation_path
            )
        else:
            assert fresh_assets is not None
            report["status"] = "preparing_shared_inputs"
            atomic_write_json(report_path, report)
            reference_manifest_path, reference_manifest = (
                prepare_fresh_comparison_inputs(
                    metadata_path=metadata_path,
                    output_root=work_dir / "shared_prepared",
                    base_checkpoint=base_checkpoint,
                    architecture_root=fresh_assets["architecture_root"],
                    t5_checkpoint=fresh_assets["t5_checkpoint"],
                    tokenizer_dir=fresh_assets["tokenizer_dir"],
                    vae_checkpoint=fresh_assets["vae_checkpoint"],
                    sampling_steps=getattr(args, "sampling_steps", 50),
                    guidance_scale=getattr(args, "guidance_scale", 5.0),
                    seed=getattr(args, "seed", 1),
                    negative_prompt=getattr(
                        args,
                        "negative_prompt",
                        DEFAULT_NEGATIVE_PROMPT,
                    ),
                    preparation_builder=preparation_builder,
                )
            )
            report["input_preparation"]["prepared_manifest"] = os.fspath(
                reference_manifest_path
            )

        records = _assert_metadata_matches_reference(
            metadata_path,
            reference_manifest,
        )
        report["sampling"] = reference_manifest["sampling"]
        report["frame_policy"] = reference_manifest["frame_policy"]

        report["status"] = "preparing_inference_variants"
        atomic_write_json(report_path, report)
        cloned_manifest = clone_reference_preparation(
            reference_manifest_path=reference_manifest_path,
            merged_checkpoint=merged_checkpoint,
            output_root=merged_inference_root,
        )
        lora_manifest = clone_reference_lora_preparation(
            reference_manifest_path=reference_manifest_path,
            base_checkpoint=base_checkpoint,
            lora_checkpoint=lora_checkpoint,
            adapter_config=adapter_config,
            output_root=lora_inference_root,
        )

        if len(cloned_manifest["buckets"]) != 2 or len(lora_manifest["buckets"]) != 2:
            raise RuntimeError(
                "The optimized four-GPU comparison expects exactly two geometry buckets"
            )
        merged_jobs = build_parallel_inference_jobs(
            merged_inference_root / "prepared_manifest.json",
            gpu_ids=gpu_ids[:2],
        )
        lora_jobs = build_parallel_inference_jobs(
            lora_inference_root / "prepared_manifest.json",
            gpu_ids=gpu_ids[2:],
        )
        all_jobs = merged_jobs + lora_jobs
        if len(all_jobs) != 4:
            raise RuntimeError(
                f"Expected four format/bucket workers, built {len(all_jobs)}"
            )
        report["status"] = "inferencing_premerged_and_dynamic_lora"
        report["active_variants"] = ["premerged", "dynamic_lora"]
        report["gpu_assignment"] = {
            job.gpu_id: {
                "variant": job.variant,
                "bucket_id": job.bucket_id,
                "sample_indices": list(job.sample_indices),
            }
            for job in all_jobs
        }
        atomic_write_json(report_path, report)
        worker_reports = run_parallel_inference_jobs(
            all_jobs,
            command_runner=command_runner,
        )
        report["merged_jobs"] = [
            item for item in worker_reports if item["variant"] == "premerged"
        ]
        report["lora_jobs"] = [
            item for item in worker_reports if item["variant"] == "dynamic_lora"
        ]
        report.pop("active_variants", None)

        report["status"] = "validating_merged_outputs"
        atomic_write_json(report_path, report)
        cloned_manifest_path = merged_inference_root / "prepared_manifest.json"
        merged_outputs = output_validator(cloned_manifest_path, **validator_kwargs)
        merged_output_validation_path = work_dir / "merged_output_validation.json"
        atomic_write_json(merged_output_validation_path, merged_outputs)
        report["merged_output_validation"] = os.fspath(merged_output_validation_path)

        report["status"] = "validating_lora_outputs"
        atomic_write_json(report_path, report)
        lora_manifest_path = lora_inference_root / "prepared_manifest.json"
        lora_outputs = output_validator(lora_manifest_path, **validator_kwargs)
        lora_output_validation_path = work_dir / "lora_output_validation.json"
        atomic_write_json(lora_output_validation_path, lora_outputs)
        report["lora_output_validation"] = os.fspath(lora_output_validation_path)

        reference_by_row = (
            {
                int(sample["row_id"]): sample
                for sample in reference_outputs["samples"]
            }
            if reference_outputs is not None
            else {}
        )
        merged_by_row = {
            int(sample["row_id"]): sample for sample in merged_outputs["samples"]
        }
        lora_by_row = {
            int(sample["row_id"]): sample for sample in lora_outputs["samples"]
        }
        expected_rows = {record.row_id for record in records}
        reference_rows_are_valid = reference_outputs is None or (
            set(reference_by_row) == expected_rows
        )
        if not reference_rows_are_valid or set(merged_by_row) != expected_rows or set(
            lora_by_row
        ) != expected_rows:
            raise RuntimeError(
                "Validated output row sets are incomplete: "
                f"expected={sorted(expected_rows)}, "
                f"reference={None if reference_outputs is None else sorted(reference_by_row)}, "
                f"merged={sorted(merged_by_row)}, lora={sorted(lora_by_row)}"
            )

        merged_seed_by_row = {
            int(record["row_id"]): int(seed)
            for bucket in cloned_manifest["buckets"]
            for record, seed in zip(
                bucket["records"], bucket["sample_seeds"], strict=True
            )
        }
        lora_seed_by_row = {
            int(record["row_id"]): int(seed)
            for bucket in lora_manifest["buckets"]
            for record, seed in zip(
                bucket["records"], bucket["sample_seeds"], strict=True
            )
        }
        if merged_seed_by_row != lora_seed_by_row or set(merged_seed_by_row) != expected_rows:
            raise RuntimeError(
                "Merged and dynamic LoRA sample seed mappings differ: "
                f"merged={merged_seed_by_row}, lora={lora_seed_by_row}"
            )

        report["status"] = "pairing_videos"
        atomic_write_json(report_path, report)
        comparisons_dir = work_dir / "side_by_side"
        samples = []
        for record in records:
            left = Path(merged_by_row[record.row_id]["output_video"])
            right = Path(lora_by_row[record.row_id]["output_video"])
            output = comparisons_dir / (
                f"row{record.row_id:04d}_{record.bucket_id}_"
                "premerged-left_dynamic-lora-right.mp4"
            )
            stream = pair_writer(
                left,
                right,
                output,
                command_runner=command_runner,
            )
            sample = {
                "row_id": record.row_id,
                "bucket_id": record.bucket_id,
                "prompt": record.prompt,
                "sample_seed": merged_seed_by_row[record.row_id],
                "left_premerged_video": os.fspath(left.resolve()),
                "right_dynamic_lora_video": os.fspath(right.resolve()),
                "comparison_video": os.fspath(output.resolve()),
                "comparison_sha256": sha256_file(output),
                "stream": stream,
            }
            if reference_outputs is not None:
                reference = Path(reference_by_row[record.row_id]["output_video"])
                sample["reference_video"] = os.fspath(reference.resolve())
            samples.append(sample)

        comparison_html_path = work_dir / "comparison.html"
        atomic_write_bytes(
            comparison_html_path,
            _comparison_html(samples=samples, work_dir=work_dir).encode("utf-8"),
        )
        report.update(
            {
                "status": "pass",
                "sample_count": len(samples),
                "samples": samples,
                "comparison_html": os.fspath(comparison_html_path),
            }
        )
        atomic_write_json(report_path, report)
        return report
    except Exception as exc:
        report.pop("active_variants", None)
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(report_path, report)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_comparison(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
