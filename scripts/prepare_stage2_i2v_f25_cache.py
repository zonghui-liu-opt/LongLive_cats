#!/usr/bin/env python3
"""Build native Stage-2 F25: reuse proven F25, freshly encode all other rows."""

import sys

if __name__ == "__main__" and (
    not sys.flags.isolated or not sys.flags.dont_write_bytecode
):
    raise SystemExit(
        "Refusing non-isolated Stage-2 CLI startup. Run exactly with: python -I -B "
        "scripts/prepare_stage2_i2v_f25_cache.py ..."
    )

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from utils.stage2_config import resolve_stage2_config  # noqa: E402
from utils.stage2_f25_cache import prepare_stage2_f25_cache  # noqa: E402
from utils.stage1_io import sha256_file  # noqa: E402
from utils.stage2_i2v_data import STAGE2_F25_SUCCESS_NAME  # noqa: E402


def _distributed_device() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Formal Stage-2 F25 preparation requires CUDA.")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def _run(args: argparse.Namespace) -> None:
    config_path = Path(args.config_path).expanduser().resolve()
    resolved = resolve_stage2_config(OmegaConf.load(config_path))
    if resolved.video_latent_frames != 25 or resolved.future_latent_frames != 24:
        raise RuntimeError(
            "Resolved Stage-2 config is not the locked sink+24/F25 contract."
        )
    environment_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if environment_world_size != resolved.expected_world_size:
        raise RuntimeError(
            f"F25 preparation world-size mismatch: resolved config requires "
            f"{resolved.expected_world_size}, got {environment_world_size}."
        )
    rank, world_size, device = _distributed_device()
    manifest = prepare_stage2_f25_cache(
        metadata_path=resolved.metadata_path,
        source_cache_manifest_path=args.source_cache_manifest,
        output_dir=resolved.cache_dir,
        config_path=config_path,
        config_contract_sha256=resolved.contract_hash(),
        config_launch_sha256=resolved.launch_hash(),
        expected_num_samples=resolved.expected_num_samples,
        rank=rank,
        world_size=world_size,
        device=device,
        vae_checkpoint_path=args.vae_checkpoint,
    )
    if rank == 0:
        if manifest is None:
            raise AssertionError("rank 0 did not receive the F25 base manifest path")
        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
        success_path = manifest.parent / STAGE2_F25_SUCCESS_NAME
        success = json.loads(success_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "status": "ok",
                    "base_manifest": str(manifest),
                    "base_manifest_sha256": manifest_value["manifest_sha256"],
                    "base_manifest_file_sha256": sha256_file(manifest),
                    "success_marker": str(success_path),
                    "success_marker_sha256": success["manifest_sha256"],
                    "success_marker_file_sha256": sha256_file(success_path),
                    "summary": manifest_value["preparation"]["summary"],
                    "output_dir": str(Path(resolved.cache_dir).expanduser().resolve()),
                    "next": (
                        "Run audit_stage2_i2v_cache.py upgrade-source-manifest on "
                        "this native F25 base manifest, then prepare-negative and audit."
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "4/8×H100 example: torchrun --standalone --nproc-per-node=$STAGE2_GPUS --no-python "
            "/absolute/path/to/python -I -B scripts/prepare_stage2_i2v_f25_cache.py "
            "--config-path configs/train_i2v_stage2_600cats.yaml --source-cache-manifest "
            "/absolute/path/to/stage1/cache_manifest.json --vae-checkpoint "
            "/absolute/path/to/Wan2.2_VAE.pth"
        ),
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--source-cache-manifest", required=True)
    parser.add_argument(
        "--vae-checkpoint",
        help=(
            "Required when any row needs fresh native-F25 materialization; "
            "must hash-match the Stage-1 manifest."
        ),
    )
    return parser


def main() -> None:
    _run(_parser().parse_args())


if __name__ == "__main__":
    main()
