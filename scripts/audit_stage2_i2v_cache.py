#!/usr/bin/env python3
"""Prepare exact negative conditioning or audit the complete Stage-2 cache."""

import sys

if __name__ == "__main__" and (
    not sys.flags.isolated or not sys.flags.dont_write_bytecode
):
    raise SystemExit(
        "Refusing non-isolated Stage-2 CLI startup. Run exactly with Python's "
        "isolated/no-bytecode flags: python -I -B "
        "scripts/audit_stage2_i2v_cache.py ..."
    )

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from utils.config import DEFAULT_NEGATIVE_PROMPT  # noqa: E402
from utils.stage1_io import aggregate_file_hash, tree_file_hashes  # noqa: E402
from utils.stage2_config import resolve_stage2_config  # noqa: E402
from utils.stage2_i2v_data import (  # noqa: E402
    STAGE2_CACHE_MANIFEST_NAME,
    STAGE2_F25_SOURCE_CACHE_SCHEMA,
    STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION,
    audit_stage2_i2v_cache,
    load_source_cache_manifest,
    save_negative_conditioning_artifact,
    upgrade_legacy_source_cache_manifest_text_encoding,
    write_negative_conditioning_manifest,
)


def _tree_hash(path: str | os.PathLike[str]) -> str:
    return aggregate_file_hash(tree_file_hashes(path))


def _positive_encoder_hashes(source: dict[str, Any]) -> tuple[str, str]:
    try:
        models = source["source_fingerprint"]["models"]
        return (
            models["t5_checkpoint"]["aggregate_sha256"],
            models["tokenizer_dir"]["aggregate_sha256"],
        )
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Source cache manifest does not contain positive T5/tokenizer hashes."
        ) from exc


def _prepare_negative(args: argparse.Namespace) -> None:
    import torch

    source = load_source_cache_manifest(
        args.source_cache_manifest,
        expected_num_samples=args.expected_num_samples,
        require_text_encoding_upgrade=True,
    )
    if source.get("schema") != STAGE2_F25_SOURCE_CACHE_SCHEMA:
        raise RuntimeError(
            "prepare-negative accepts only the native attested Stage-2 F25 source "
            "manifest; migrate F24 first."
        )
    expected_t5, expected_tokenizer = _positive_encoder_hashes(source)
    actual_t5 = _tree_hash(args.t5_checkpoint)
    actual_tokenizer = _tree_hash(args.tokenizer_dir)
    if actual_t5 != expected_t5:
        raise RuntimeError(
            "--t5-checkpoint is not the checkpoint used by the positive cache: "
            f"{actual_t5} != {expected_t5}."
        )
    if actual_tokenizer != expected_tokenizer:
        raise RuntimeError(
            "--tokenizer-dir is not the tokenizer used by the positive cache: "
            f"{actual_tokenizer} != {expected_tokenizer}."
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "negative_conditioning.safetensors"
    manifest_path = output_dir / "negative_conditioning_manifest.json"
    existing = [path for path in (artifact_path, manifest_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Negative conditioning output already exists; inspect and reuse it, or "
            f"pass --force explicitly: {[str(path) for path in existing]}"
        )

    # This is the only subcommand that loads T5.  The audit and formal trainer
    # consume only the resulting safetensors artifact and manifest.
    from utils.wan_5b_wrapper import WanTextEncoder

    device = torch.device(args.device)
    encoder = WanTextEncoder(
        t5_checkpoint=args.t5_checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        device=device,
    ).eval()
    with torch.inference_mode():
        encoded = encoder([DEFAULT_NEGATIVE_PROMPT], return_mask=True)
    final_t5 = _tree_hash(args.t5_checkpoint)
    final_tokenizer = _tree_hash(args.tokenizer_dir)
    if final_t5 != actual_t5 or final_t5 != expected_t5:
        raise RuntimeError(
            "T5 checkpoint tree changed during negative-conditioning encoding; "
            "no artifact was published."
        )
    if final_tokenizer != actual_tokenizer or final_tokenizer != expected_tokenizer:
        raise RuntimeError(
            "Tokenizer tree changed during negative-conditioning encoding; "
            "no artifact was published."
        )
    artifact = save_negative_conditioning_artifact(
        artifact_path,
        prompt_embeds=encoded["prompt_embeds"][0],
        prompt_mask=encoded["prompt_mask"][0],
    )
    write_negative_conditioning_manifest(
        manifest_path,
        artifact_path=artifact_path,
        source_cache_manifest_path=args.source_cache_manifest,
        expected_num_samples=args.expected_num_samples,
        require_text_encoding_upgrade=True,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str(manifest_path),
                "artifact": str(artifact_path),
                "artifact_sha256": artifact["sha256"],
                "prompt_valid_tokens": artifact["prompt_valid_tokens"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _upgrade_source_manifest(args: argparse.Namespace) -> None:
    output = upgrade_legacy_source_cache_manifest_text_encoding(
        args.legacy_source_cache_manifest,
        args.output_manifest,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        t5_checkpoint_path=args.t5_checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        operator_id=args.operator_id,
        operator_attestation=args.operator_attestation,
        expected_num_samples=args.expected_num_samples,
    )
    upgraded = load_source_cache_manifest(
        output,
        expected_num_samples=args.expected_num_samples,
        require_text_encoding_upgrade=True,
    )
    upgrade = upgraded["stage2_text_encoding_upgrade"]
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str(output),
                "manifest_sha256": upgraded["manifest_sha256"],
                "original_manifest_sha256": upgrade["original_source_manifest"][
                    "manifest_sha256"
                ],
                "text_encoding_contract_sha256": upgrade["verification"][
                    "text_encoding_contract_sha256"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _audit(args: argparse.Namespace) -> None:
    raw_config = OmegaConf.load(args.config_path)
    resolved = resolve_stage2_config(raw_config)
    cache_dir = Path(resolved.cache_dir).expanduser().resolve()
    metadata_path = Path(resolved.metadata_path).expanduser().resolve()
    negative_manifest = (
        Path(resolved.negative_conditioning_manifest).expanduser().resolve()
    )
    resolved_action_sidecar = (
        None
        if resolved.action_labels_path is None
        else Path(resolved.action_labels_path).expanduser().resolve()
    )
    expected_output = (cache_dir / STAGE2_CACHE_MANIFEST_NAME).resolve()
    for supplied, expected, label in (
        (args.cache_dir, cache_dir, "--cache-dir"),
        (args.metadata_path, metadata_path, "--metadata-path"),
        (
            args.negative_conditioning_manifest,
            negative_manifest,
            "--negative-conditioning-manifest",
        ),
        (args.output_manifest, expected_output, "--output-manifest"),
    ):
        if supplied is not None and (
            expected is None or Path(supplied).expanduser().resolve() != expected
        ):
            raise RuntimeError(
                f"{label} must exactly match the resolved Stage-2 config path: "
                f"{expected}. Update and resolve the config first (or set its "
                "environment variable when that field is environment-backed); do "
                "not use this CLI flag to point the signed launch at different data."
            )
    supplied_action_sidecar = (
        None
        if args.action_labels_path is None
        else Path(args.action_labels_path).expanduser().resolve()
    )
    if (
        supplied_action_sidecar is not None
        and resolved_action_sidecar is not None
        and supplied_action_sidecar != resolved_action_sidecar
    ):
        raise RuntimeError(
            "--action-labels-path must exactly match the resolved Stage-2 config "
            f"path: {resolved_action_sidecar}."
        )
    source_manifest = Path(args.source_cache_manifest).expanduser().resolve()
    action_sidecar = supplied_action_sidecar or resolved_action_sidecar
    output = expected_output
    manifest = audit_stage2_i2v_cache(
        metadata_path=metadata_path,
        cache_dir=cache_dir,
        source_cache_manifest_path=source_manifest,
        negative_conditioning_manifest_path=negative_manifest,
        expected_action_ids=args.action_id,
        action_labels_path=action_sidecar,
        output_manifest_path=output,
        expected_num_samples=resolved.expected_num_samples,
        expected_action_counts=dict(resolved.expected_action_counts),
        allowed_latent_spatial_shapes=resolved.allowed_latent_spatial_shapes,
        config_contract_sha256=resolved.contract_hash(),
        config_launch_sha256=resolved.launch_hash(),
        require_text_encoding_upgrade=True,
        require_native_f25_source=True,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": manifest["manifest_path"],
                "manifest_sha256": manifest["manifest_sha256"],
                "num_samples": manifest["num_samples"],
                "action_counts": manifest["actions"]["counts"],
                "orientation_counts": manifest["orientation_counts"],
                "total_bytes": manifest["total_bytes"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    upgrade = subparsers.add_parser(
        "upgrade-source-manifest",
        help=(
            "Create a new attested Stage-2 text-provenance manifest without "
            "rewriting the native F25 base manifest or any latent cache."
        ),
    )
    upgrade.add_argument(
        "--base-source-cache-manifest",
        "--legacy-source-cache-manifest",
        dest="legacy_source_cache_manifest",
        required=True,
        help=(
            "Use the native F25 cache_manifest.json in the formal Stage-2 flow; "
            "the legacy spelling remains an input-compatible alias."
        ),
    )
    upgrade.add_argument("--output-manifest", required=True)
    upgrade.add_argument("--expected-source-manifest-sha256", required=True)
    upgrade.add_argument("--t5-checkpoint", required=True)
    upgrade.add_argument("--tokenizer-dir", required=True)
    upgrade.add_argument("--operator-id", required=True)
    upgrade.add_argument(
        "--operator-attestation",
        required=True,
        help=("Must exactly equal: " f"{STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION}"),
    )
    upgrade.add_argument("--expected-num-samples", type=int, default=600)
    upgrade.set_defaults(func=_upgrade_source_manifest)

    negative = subparsers.add_parser(
        "prepare-negative",
        help=(
            "Encode DEFAULT_NEGATIVE_PROMPT from an attested upgraded source "
            "manifest and write its audited artifact."
        ),
    )
    negative.add_argument("--source-cache-manifest", required=True)
    negative.add_argument("--t5-checkpoint", required=True)
    negative.add_argument("--tokenizer-dir", required=True)
    negative.add_argument("--output-dir", required=True)
    negative.add_argument("--expected-num-samples", type=int, default=600)
    negative.add_argument("--device", default="cuda")
    negative.add_argument("--force", action="store_true")
    negative.set_defaults(func=_prepare_negative)

    audit = subparsers.add_parser(
        "audit",
        help="Scan all Stage-2 cache files and write the training gate manifest.",
    )
    audit.add_argument("--config-path", required=True)
    audit.add_argument(
        "--action-id",
        action="append",
        required=True,
        help="One operator-confirmed action enum; pass exactly three in rotation order.",
    )
    audit.add_argument("--metadata-path")
    audit.add_argument("--cache-dir")
    audit.add_argument("--source-cache-manifest", required=True)
    audit.add_argument("--negative-conditioning-manifest")
    audit.add_argument("--action-labels-path")
    audit.add_argument("--output-manifest")
    audit.set_defaults(func=_audit)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
