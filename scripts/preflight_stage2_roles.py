#!/usr/bin/env python3
"""Initialize and audit Stage-2 roles on 8xH100 without any forward or training."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/train_i2v_stage2_600cats.yaml",
        help="Raw Stage-2 YAML. It is resolved before any legacy normalize_config call.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for role_init_manifest.json and ROLE_INIT_COMPLETE.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # All heavy imports are intentionally below argument parsing so --help is
    # safe on login nodes and never imports the model/trainer/T5/VAE stack.
    import torch
    import torch.distributed as dist

    from utils.stage1_io import canonical_json_sha256
    from utils.stage2_config import load_stage2_config
    from utils.stage2_fsdp2 import (
        build_stage2_fsdp2_device_mesh,
        validate_stage2_fsdp2_topology,
    )
    from utils.stage2_role_init import (
        initialize_stage2_roles,
        stage2_init_only_side_effect_guard,
    )
    from utils.stage2_role_manifest import (
        audit_stage2_init_assets,
        build_stage2_role_init_manifest,
        require_stage2_cold_start,
        write_stage2_role_init_artifacts,
    )

    resolved = load_stage2_config(args.config)
    # This check precedes output inspection, asset hashing, CUDA model creation,
    # and all other side effects. Resume remains exclusively Step 10.
    require_stage2_cold_start(resolved)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    initialized_here = False
    try:
        if not dist.is_initialized():
            if "LOCAL_RANK" not in os.environ:
                raise RuntimeError(
                    "Stage-2 role preflight must run under torchrun with 8 processes"
                )
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl", timeout=timedelta(minutes=60))
            initialized_here = True
        rank = dist.get_rank()

        def world_checked(label: str, callback: Callable[[], _T]) -> _T:
            value = None
            local_error = None
            try:
                value = callback()
            except Exception as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            statuses: list[dict[str, Any] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(
                statuses,
                {"rank": rank, "label": label, "error": local_error},
            )
            errors = [
                f"rank{item['rank']}: {item['error']}"
                for item in statuses
                if item is not None and item["error"] is not None
            ]
            if errors or any(item is None for item in statuses):
                raise RuntimeError(f"Stage-2 WORLD stage failed ({label}): {errors}")
            return value

        mesh, runtime_audit = build_stage2_fsdp2_device_mesh()
        topology_audit = validate_stage2_fsdp2_topology(
            world_size=resolved.expected_world_size,
            sequence_parallel_size=resolved.sequence_parallel_size,
            data_parallel_size=resolved.data_parallel_size,
            mesh_shape=(resolved.expected_world_size,),
            mesh_dim_names=("shard",),
        )

        asset_payload = [None]
        if rank == 0:
            try:
                asset_payload[0] = {
                    "ok": True,
                    "value": audit_stage2_init_assets(resolved),
                }
            except Exception as exc:
                asset_payload[0] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(asset_payload, src=0)
        asset_result = asset_payload[0]
        if not isinstance(asset_result, dict) or not asset_result.get("ok"):
            raise RuntimeError(
                "Stage-2 rank-0 asset manifest/hash audit failed: "
                f"{asset_result.get('error') if isinstance(asset_result, dict) else asset_result}"
            )
        assets = asset_result["value"]

        with stage2_init_only_side_effect_guard() as side_effect_audit:
            initialized = initialize_stage2_roles(
                resolved,
                assets=assets,
                mesh=mesh,
                stage_runner=world_checked,
                is_main_process=rank == 0,
            )
            local_audit = world_checked(
                "all roles: final independence audit",
                lambda: {
                    "roles": {
                        role: audit.to_dict()
                        for role, audit in initialized.role_audits.items()
                    },
                    "adapter_digests": dict(initialized.adapter_digests),
                    "storage": initialized.model.audit_independent_role_storage(),
                },
            )
        local_audit["side_effects"] = dict(side_effect_audit)
        local_hash = canonical_json_sha256(local_audit)
        all_hashes: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(all_hashes, local_hash)
        if set(all_hashes) != {local_hash}:
            raise RuntimeError(
                f"Stage-2 role init audit differs across ranks: {all_hashes}"
            )
        rank_consensus_sha256 = canonical_json_sha256(all_hashes)

        def publish_rank0():
            if rank != 0:
                return None
            manifest = build_stage2_role_init_manifest(
                contract_hash=resolved.contract_hash(),
                launch_hash=resolved.launch_hash(),
                generator_asset=assets["generator"],
                real_score_asset=assets["real_score"],
                role_audits=local_audit["roles"],
                role_isolation_audit=local_audit["storage"],
                fsdp_audits={
                    **topology_audit,
                    "runtime": runtime_audit,
                    "all_roles_independently_wrapped": True,
                },
                rank_consensus_sha256=rank_consensus_sha256,
                side_effect_audit=local_audit["side_effects"],
            )
            write_stage2_role_init_artifacts(output_dir, manifest)
            return {
                "status": "passed",
                "output_dir": str(output_dir),
                "manifest_sha256": manifest["manifest_sha256"],
                "rank_consensus_sha256": rank_consensus_sha256,
            }

        publication = world_checked(
            "rank0: atomic role-init artifact publication", publish_rank0
        )
        if rank == 0:
            print(json.dumps(publication, sort_keys=True))
        return 0
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
