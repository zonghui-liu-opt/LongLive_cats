"""Authenticated, low-I/O runtime asset identity for Stage-2 inference.

Rank 0 performs the only cryptographic scan of the large immutable model
assets.  Every rank then brackets each real loader with cheap canonical-path
and stat-identity checks.  The compact identity returned here is embedded in
every trace and the final manifest; the detailed per-file attestation remains
an in-memory runtime object.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from utils.stage1_io import aggregate_file_hash, canonical_json_sha256
from utils.stage2_checkpoint import (
    STAGE2_CHECKPOINT_SCHEMA,
    STAGE2_CHECKPOINT_SCHEMA_VERSION,
    validate_stage2_provenance,
)
from utils.stage2_inference_config import ResolvedStage2InferenceConfig

STAGE2_RUNTIME_ASSETS_SCHEMA = "longlive_stage2_inference_runtime_assets/v1"
STAGE2_RUNTIME_ASSET_IDENTITY_SCHEMA = (
    "longlive_stage2_inference_runtime_asset_identity/v1"
)
STAGE2_INFERENCE_ASSET_API_VERSION = "longlive_stage2_inference_assets/v2"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_KEYS = {"device", "inode", "size", "mtime_ns", "ctime_ns"}
_MODEL_ASSET_NAMES = ("t5_checkpoint", "tokenizer_dir", "vae_checkpoint")
_RUNTIME_FILE_ASSET_NAMES = (
    "t5_checkpoint",
    "tokenizer_dir",
    "vae_checkpoint",
    "architecture_config",
    "generator_base",
)


def stage2_generator_asset_content_sha256(value: Mapping[str, Any]) -> str:
    """Hash persistent Generator provenance without host-local stat identity.

    ``device``/``inode`` and timestamps authenticate a file only for the live
    process that observed them.  They necessarily change when the same bytes
    are mounted or copied onto another inference node.  Content fields remain
    covered here, while the live identity stays in the runtime attestation and
    is checked immediately around the real model load.
    """

    if not isinstance(value, Mapping):
        raise TypeError("Stage-2 Generator asset must be an object")
    normalized = dict(value)
    checkpoint_files = normalized.get("checkpoint_files")
    if checkpoint_files is not None:
        if not isinstance(checkpoint_files, list):
            raise TypeError("Stage-2 Generator checkpoint_files must be a list")
        normalized_files: list[dict[str, Any]] = []
        for index, entry in enumerate(checkpoint_files):
            if not isinstance(entry, Mapping):
                raise TypeError(
                    f"Stage-2 Generator checkpoint_files[{index}] must be an object"
                )
            normalized_entry = dict(entry)
            normalized_entry.pop("identity", None)
            normalized_files.append(normalized_entry)
        normalized["checkpoint_files"] = normalized_files
    return canonical_json_sha256(normalized)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be a plain integer >= {minimum}")
    return value


def _strict_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON in {label}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _stat_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _validate_identity(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_KEYS:
        raise ValueError(f"{label} identity schema mismatch")
    return {
        key: _plain_int(value[key], f"{label}.{key}") for key in sorted(_IDENTITY_KEYS)
    }


def _canonical_regular_path(value: str | os.PathLike[str], *, label: str) -> Path:
    source = Path(value).expanduser()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"{label} is not a regular file: {source}")
    resolved = source.resolve()
    if resolved != source.absolute():
        # ``absolute`` preserves ``..`` and therefore distinguishes a canonical
        # caller path without following a symlink that was already rejected.
        raise RuntimeError(f"{label} path must be canonical: {source}")
    return resolved


def _read_regular_snapshot(
    value: str | os.PathLike[str],
    *,
    label: str,
) -> tuple[Path, bytes, dict[str, int]]:
    path = _canonical_regular_path(value, label=label)
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} descriptor is not a regular file")
        data = handle.read()
        after = os.fstat(handle.fileno())
    before_identity = _stat_identity(before)
    after_identity = _stat_identity(after)
    if before_identity != after_identity or len(data) != after_identity["size"]:
        raise RuntimeError(f"{label} changed while taking its byte snapshot")
    if path.is_symlink() or _stat_identity(path.stat()) != after_identity:
        raise RuntimeError(f"{label} path changed after taking its byte snapshot")
    return path, data, after_identity


def _hash_regular_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(f"{label} is not a canonical regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} descriptor is not a regular file")
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    before_identity = _stat_identity(before)
    after_identity = _stat_identity(after)
    if before_identity != after_identity:
        raise RuntimeError(f"{label} changed during SHA256 authentication")
    if path.is_symlink() or _stat_identity(path.stat()) != after_identity:
        raise RuntimeError(f"{label} path changed after SHA256 authentication")
    actual_sha256 = digest.hexdigest()
    if after_identity["size"] != expected_size or actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} hash/size differs from provenance: "
            f"expected=({expected_size},{expected_sha256}), "
            f"actual=({after_identity['size']},{actual_sha256})"
        )
    return {
        "path": str(path),
        "size": expected_size,
        "sha256": expected_sha256,
        "identity": after_identity,
    }


def _safe_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{label} is not a safe canonical relative path")
    return value


def _expected_model_entry(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"files", "aggregate_sha256"}:
        raise ValueError(f"{label} provenance schema mismatch")
    files = value["files"]
    if isinstance(files, (str, bytes)) or not isinstance(files, Sequence) or not files:
        raise ValueError(f"{label}.files must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    paths: list[str] = []
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping) or set(entry) != {"path", "size", "sha256"}:
            raise ValueError(f"{label}.files[{index}] schema mismatch")
        relative = _safe_relative_path(
            entry["path"], label=f"{label}.files[{index}].path"
        )
        paths.append(relative)
        normalized.append(
            {
                "path": relative,
                "size": _plain_int(
                    entry["size"], f"{label}.files[{index}].size", minimum=1
                ),
                "sha256": _sha256(entry["sha256"], f"{label}.files[{index}].sha256"),
            }
        )
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RuntimeError(f"{label} file list must be sorted and unique")
    aggregate = _sha256(value["aggregate_sha256"], f"{label}.aggregate_sha256")
    if aggregate_file_hash(normalized) != aggregate:
        raise RuntimeError(f"{label} aggregate SHA256 differs from its file list")
    return {"files": normalized, "aggregate_sha256": aggregate}


def _live_tree_files(root: Path, *, kind: str, label: str) -> list[Path]:
    if kind == "file":
        if root.is_symlink() or not root.is_file():
            raise RuntimeError(f"{label} is not a regular file: {root}")
        return [root.resolve()]
    if kind != "directory" or root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"{label} is not a regular directory: {root}")
    resolved = root.resolve()
    files: list[Path] = []
    for child in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise RuntimeError(f"{label} contains a symlink: {child}")
        if child.is_file():
            files.append(child)
        elif not child.is_dir():
            raise RuntimeError(f"{label} contains a non-regular entry: {child}")
    if not files:
        raise RuntimeError(f"{label} tree is empty")
    return files


def _authenticate_model_tree(
    root_value: str | os.PathLike[str],
    *,
    expected: Mapping[str, Any],
    label: str,
    required_kind: str,
) -> dict[str, Any]:
    source = Path(root_value).expanduser()
    kind = "file" if source.is_file() and not source.is_symlink() else "directory"
    if required_kind not in {"file", "directory"}:
        raise ValueError(f"{label} required kind is invalid")
    if kind != required_kind:
        raise RuntimeError(f"{label} must be a regular {required_kind}: {source}")
    root = source.resolve()
    live_files = _live_tree_files(root, kind=kind, label=label)
    relative_paths = [
        (path.name if kind == "file" else path.relative_to(root).as_posix())
        for path in live_files
    ]
    expected_files = list(expected["files"])
    expected_paths = [entry["path"] for entry in expected_files]
    if relative_paths != expected_paths:
        raise RuntimeError(
            f"{label} live file set differs from provenance: "
            f"expected={expected_paths}, actual={relative_paths}"
        )
    authenticated = []
    for path, entry in zip(live_files, expected_files):
        authenticated.append(
            {
                "relative_path": entry["path"],
                **_hash_regular_file(
                    path,
                    expected_size=entry["size"],
                    expected_sha256=entry["sha256"],
                    label=f"{label}:{entry['path']}",
                ),
            }
        )
    return {
        "root": str(root),
        "kind": kind,
        "aggregate_sha256": expected["aggregate_sha256"],
        "files": authenticated,
    }


def _checkpoint_provenance_snapshot(
    checkpoint: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    candidate = Path(checkpoint).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(
            f"Stage-2 checkpoint is not a regular directory: {candidate}"
        )
    directory = candidate.resolve()
    success = directory / "_SUCCESS"
    if success.is_symlink() or not success.is_file() or success.stat().st_size != 0:
        raise RuntimeError(f"Stage-2 checkpoint success marker is invalid: {success}")
    _, manifest_bytes, _ = _read_regular_snapshot(
        directory / "checkpoint_manifest.json",
        label="Stage-2 checkpoint manifest",
    )
    manifest = _strict_json_bytes(manifest_bytes, label="Stage-2 checkpoint manifest")
    claimed_manifest_sha256 = _sha256(
        manifest.get("manifest_sha256"), "checkpoint manifest_sha256"
    )
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_json_sha256(body) != claimed_manifest_sha256:
        raise RuntimeError("Stage-2 checkpoint manifest self hash mismatch")
    if (
        manifest.get("schema") != STAGE2_CHECKPOINT_SCHEMA
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != STAGE2_CHECKPOINT_SCHEMA_VERSION
    ):
        raise RuntimeError("unsupported Stage-2 checkpoint manifest schema")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise TypeError("Stage-2 checkpoint manifest files must be a list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("name") == "provenance.json"
    ]
    if len(matches) != 1 or set(matches[0]) != {"name", "size", "sha256"}:
        raise RuntimeError("Stage-2 checkpoint must bind exactly one provenance.json")
    entry = matches[0]
    _, provenance_bytes, _ = _read_regular_snapshot(
        directory / "provenance.json",
        label="Stage-2 checkpoint provenance",
    )
    if len(provenance_bytes) != _plain_int(
        entry["size"], "checkpoint provenance size"
    ) or hashlib.sha256(provenance_bytes).hexdigest() != _sha256(
        entry["sha256"], "checkpoint provenance file SHA256"
    ):
        raise RuntimeError("Stage-2 checkpoint provenance file hash/size mismatch")
    provenance = _strict_json_bytes(
        provenance_bytes, label="Stage-2 checkpoint provenance"
    )
    if canonical_json_sha256(provenance) != _sha256(
        manifest.get("provenance_sha256"), "checkpoint provenance_sha256"
    ):
        raise RuntimeError("Stage-2 checkpoint provenance self binding mismatch")
    return manifest, validate_stage2_provenance(provenance), directory


def _architecture_expected(recorded: Any) -> dict[str, Any]:
    if not isinstance(recorded, Mapping) or set(recorded) != {
        "name",
        "path",
        "size",
        "sha256",
        "identity",
    }:
        raise ValueError("Stage-2 Generator architecture_file schema mismatch")
    if recorded.get("name") != "config.json":
        raise ValueError("Stage-2 Generator architecture file must be config.json")
    _validate_identity(recorded.get("identity"), label="recorded architecture")
    return {
        "files": [
            {
                "path": "config.json",
                "size": _plain_int(
                    recorded.get("size"), "recorded architecture size", minimum=1
                ),
                "sha256": _sha256(
                    recorded.get("sha256"), "recorded architecture SHA256"
                ),
            }
        ],
        "aggregate_sha256": aggregate_file_hash(
            [
                {
                    "path": "config.json",
                    "size": recorded["size"],
                    "sha256": recorded["sha256"],
                }
            ]
        ),
    }


def _generic_generator_base(verified: Mapping[str, Any]) -> dict[str, Any]:
    files = verified.get("checkpoint_files")
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError(
            "verified Stage-2 Generator asset must contain one checkpoint file"
        )
    normalized = []
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise TypeError("verified Generator checkpoint file must be an object")
        path = Path(str(entry.get("path", ""))).expanduser().resolve()
        identity = _validate_identity(
            entry.get("identity"), label=f"Generator checkpoint_files[{index}]"
        )
        normalized.append(
            {
                "relative_path": str(entry.get("name")),
                "path": str(path),
                "size": _plain_int(
                    entry.get("size"),
                    f"Generator checkpoint_files[{index}].size",
                    minimum=1,
                ),
                "sha256": _sha256(
                    entry.get("sha256"),
                    f"Generator checkpoint_files[{index}].sha256",
                ),
                "identity": identity,
            }
        )
    aggregate_entries = [
        {
            "path": entry["relative_path"],
            "size": entry["size"],
            "sha256": entry["sha256"],
        }
        for entry in normalized
    ]
    checkpoint_sha256 = _sha256(
        verified.get("checkpoint_sha256"), "Generator checkpoint SHA256"
    )
    if normalized[0]["sha256"] != checkpoint_sha256:
        raise RuntimeError("Generator checkpoint SHA256 differs from its file entry")
    return {
        "root": str(Path(verified["checkpoint_path"]).expanduser().resolve()),
        "kind": "file",
        "aggregate_sha256": aggregate_file_hash(aggregate_entries),
        "files": normalized,
    }


def _compact_asset(value: Mapping[str, Any]) -> dict[str, Any]:
    files = value["files"]
    return {
        "path": value["root"],
        "aggregate_sha256": value["aggregate_sha256"],
        "file_count": len(files),
        "total_size": sum(int(entry["size"]) for entry in files),
    }


def build_stage2_runtime_assets(
    config: ResolvedStage2InferenceConfig,
    *,
    validate_generator_manifest_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Cryptographically authenticate every live inference asset on rank 0."""

    if not isinstance(config, ResolvedStage2InferenceConfig):
        raise TypeError("Stage-2 runtime assets require a resolved inference config")
    manifest, provenance, checkpoint_directory = _checkpoint_provenance_snapshot(
        config.stage2_checkpoint
    )
    expected_source_sha256 = _sha256(
        provenance["data"]["source_manifest_sha256"],
        "checkpoint source_manifest_sha256",
    )
    source_path, source_bytes, source_identity = _read_regular_snapshot(
        config.source_cache_manifest,
        label="Stage-2 source cache manifest",
    )
    source_manifest = _strict_json_bytes(
        source_bytes, label="Stage-2 source cache manifest"
    )
    source_claimed_sha256 = _sha256(
        source_manifest.get("manifest_sha256"), "source manifest_sha256"
    )
    source_body = {
        key: value for key, value in source_manifest.items() if key != "manifest_sha256"
    }
    if canonical_json_sha256(source_body) != source_claimed_sha256:
        raise RuntimeError("Stage-2 source cache manifest self hash mismatch")
    if source_claimed_sha256 != expected_source_sha256:
        raise RuntimeError(
            "Stage-2 inference source manifest differs from checkpoint provenance"
        )
    fingerprint = source_manifest.get("source_fingerprint")
    models = fingerprint.get("models") if isinstance(fingerprint, Mapping) else None
    if not isinstance(models, Mapping):
        raise TypeError("Stage-2 source manifest lacks source_fingerprint.models")

    live_assets: dict[str, dict[str, Any]] = {}
    config_paths = {
        "t5_checkpoint": config.t5_checkpoint,
        "tokenizer_dir": config.tokenizer_dir,
        "vae_checkpoint": config.vae_checkpoint,
    }
    required_kinds = {
        "t5_checkpoint": "file",
        "tokenizer_dir": "directory",
        "vae_checkpoint": "file",
    }
    for name in _MODEL_ASSET_NAMES:
        if name not in models:
            raise RuntimeError(f"Stage-2 source manifest lacks {name} provenance")
        expected = _expected_model_entry(models[name], label=f"source model {name}")
        live_assets[name] = _authenticate_model_tree(
            config_paths[name],
            expected=expected,
            label=f"Stage-2 {name}",
            required_kind=required_kinds[name],
        )

    assets = provenance.get("assets")
    generator_recorded = (
        assets.get("generator") if isinstance(assets, Mapping) else None
    )
    if not isinstance(generator_recorded, Mapping):
        raise TypeError("Stage-2 checkpoint provenance lacks its Generator asset")
    recorded_generator_asset = dict(generator_recorded)
    recorded_architecture = recorded_generator_asset.pop("architecture_file", None)
    if validate_generator_manifest_fn is None:
        from utils.stage2_role_manifest import validate_stage2_generator_manifest

        validate_generator_manifest_fn = validate_stage2_generator_manifest
    verified_generator = dict(
        validate_generator_manifest_fn(
            recorded_generator_asset.get("manifest_path"),
            expected_checkpoint_path=recorded_generator_asset.get("checkpoint_path"),
            expected_step=3075,
        )
    )
    if stage2_generator_asset_content_sha256(
        verified_generator
    ) != stage2_generator_asset_content_sha256(recorded_generator_asset):
        raise RuntimeError(
            "Stage-2 checkpoint Generator provenance differs in content from "
            "its live manifest"
        )
    architecture = _authenticate_model_tree(
        Path(config.architecture_root) / "config.json",
        expected=_architecture_expected(recorded_architecture),
        label="Stage-2 architecture config",
        required_kind="file",
    )
    # An architecture root may contain unrelated model files.  Only config.json
    # is consumed by the Stage-2 meta-constructor and belongs in this identity.
    if [entry["relative_path"] for entry in architecture["files"]] != ["config.json"]:
        raise AssertionError("Stage-2 architecture authentication selected extra files")
    architecture_file = architecture["files"][0]
    verified_generator["architecture_file"] = {
        "name": "config.json",
        "path": architecture_file["path"],
        "size": architecture_file["size"],
        "sha256": architecture_file["sha256"],
        "identity": architecture_file["identity"],
    }
    generator_base = _generic_generator_base(verified_generator)
    live_assets["architecture_config"] = architecture
    live_assets["generator_base"] = generator_base

    source_file = {
        "path": str(source_path),
        "size": source_identity["size"],
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "identity": source_identity,
    }
    compact_body = {
        "schema": STAGE2_RUNTIME_ASSET_IDENTITY_SCHEMA,
        "checkpoint": {
            "directory": str(checkpoint_directory),
            "manifest_sha256": manifest["manifest_sha256"],
            "generator_recorded_asset_sha256": canonical_json_sha256(
                dict(generator_recorded)
            ),
        },
        "source_manifest": {
            "path": str(source_path),
            "manifest_sha256": source_claimed_sha256,
            "file_sha256": source_file["sha256"],
            "size": source_file["size"],
        },
        "assets": {
            name: _compact_asset(live_assets[name])
            for name in _RUNTIME_FILE_ASSET_NAMES
        },
    }
    compact = {
        **compact_body,
        "identity_sha256": canonical_json_sha256(compact_body),
    }
    payload = {
        "schema": STAGE2_RUNTIME_ASSETS_SCHEMA,
        "checkpoint": compact["checkpoint"],
        "source_manifest_file": source_file,
        "assets": live_assets,
        "generator_asset": verified_generator,
        "identity": compact,
    }
    payload["attestation_sha256"] = canonical_json_sha256(payload)
    return validate_stage2_runtime_assets(payload)


def validate_stage2_runtime_asset_identity(value: Any) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "checkpoint",
        "source_manifest",
        "assets",
        "identity_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("Stage-2 runtime asset identity schema mismatch")
    if value.get("schema") != STAGE2_RUNTIME_ASSET_IDENTITY_SCHEMA:
        raise ValueError("Stage-2 runtime asset identity version mismatch")
    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "directory",
        "manifest_sha256",
        "generator_recorded_asset_sha256",
    }:
        raise ValueError("Stage-2 runtime checkpoint identity schema mismatch")
    if (
        not isinstance(checkpoint["directory"], str)
        or not Path(checkpoint["directory"]).is_absolute()
    ):
        raise ValueError("Stage-2 runtime checkpoint path must be absolute")
    _sha256(checkpoint["manifest_sha256"], "runtime checkpoint manifest SHA256")
    _sha256(
        checkpoint["generator_recorded_asset_sha256"],
        "runtime checkpoint Generator asset SHA256",
    )
    source = value.get("source_manifest")
    if not isinstance(source, Mapping) or set(source) != {
        "path",
        "manifest_sha256",
        "file_sha256",
        "size",
    }:
        raise ValueError("Stage-2 runtime source manifest identity schema mismatch")
    if not isinstance(source["path"], str) or not Path(source["path"]).is_absolute():
        raise ValueError("Stage-2 runtime source manifest path must be absolute")
    _sha256(source["manifest_sha256"], "runtime source manifest self SHA256")
    _sha256(source["file_sha256"], "runtime source manifest file SHA256")
    _plain_int(source["size"], "runtime source manifest size", minimum=1)
    assets = value.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != set(_RUNTIME_FILE_ASSET_NAMES):
        raise ValueError("Stage-2 compact runtime asset set mismatch")
    normalized_assets: dict[str, dict[str, Any]] = {}
    for name in _RUNTIME_FILE_ASSET_NAMES:
        entry = assets[name]
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "aggregate_sha256",
            "file_count",
            "total_size",
        }:
            raise ValueError(f"Stage-2 compact runtime asset {name} schema mismatch")
        if not isinstance(entry["path"], str) or not Path(entry["path"]).is_absolute():
            raise ValueError(f"Stage-2 compact runtime asset {name} path is invalid")
        normalized_assets[name] = {
            "path": entry["path"],
            "aggregate_sha256": _sha256(
                entry["aggregate_sha256"], f"runtime asset {name} aggregate SHA256"
            ),
            "file_count": _plain_int(
                entry["file_count"], f"runtime asset {name} file_count", minimum=1
            ),
            "total_size": _plain_int(
                entry["total_size"], f"runtime asset {name} total_size", minimum=1
            ),
        }
    normalized = {
        "schema": STAGE2_RUNTIME_ASSET_IDENTITY_SCHEMA,
        "checkpoint": dict(checkpoint),
        "source_manifest": dict(source),
        "assets": normalized_assets,
    }
    claimed = _sha256(value.get("identity_sha256"), "runtime asset identity SHA256")
    if claimed != canonical_json_sha256(normalized):
        raise RuntimeError("Stage-2 runtime asset identity self hash mismatch")
    return {**normalized, "identity_sha256": claimed}


def _validate_runtime_file_asset(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "root",
        "kind",
        "aggregate_sha256",
        "files",
    }:
        raise ValueError(f"{label} schema mismatch")
    root = value["root"]
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError(f"{label}.root must be absolute")
    if value["kind"] not in {"file", "directory"}:
        raise ValueError(f"{label}.kind is invalid")
    aggregate = _sha256(value["aggregate_sha256"], f"{label}.aggregate_sha256")
    files = value["files"]
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label}.files must be a non-empty list")
    normalized_files = []
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping) or set(entry) != {
            "relative_path",
            "path",
            "size",
            "sha256",
            "identity",
        }:
            raise ValueError(f"{label}.files[{index}] schema mismatch")
        relative = _safe_relative_path(
            entry["relative_path"], label=f"{label}.files[{index}].relative_path"
        )
        path = entry["path"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"{label}.files[{index}].path must be absolute")
        size = _plain_int(entry["size"], f"{label}.files[{index}].size", minimum=1)
        identity = _validate_identity(
            entry["identity"], label=f"{label}.files[{index}]"
        )
        if identity["size"] != size:
            raise RuntimeError(f"{label}.files[{index}] size/identity mismatch")
        normalized_files.append(
            {
                "relative_path": relative,
                "path": path,
                "size": size,
                "sha256": _sha256(entry["sha256"], f"{label}.files[{index}].sha256"),
                "identity": identity,
            }
        )
    relative_paths = [entry["relative_path"] for entry in normalized_files]
    if relative_paths != sorted(relative_paths) or len(relative_paths) != len(
        set(relative_paths)
    ):
        raise RuntimeError(f"{label} files must be sorted and unique")
    expected_aggregate = aggregate_file_hash(
        [
            {
                "path": entry["relative_path"],
                "size": entry["size"],
                "sha256": entry["sha256"],
            }
            for entry in normalized_files
        ]
    )
    if aggregate != expected_aggregate:
        raise RuntimeError(f"{label} aggregate SHA256 differs from its file list")
    return {
        "root": root,
        "kind": value["kind"],
        "aggregate_sha256": aggregate,
        "files": normalized_files,
    }


def validate_stage2_runtime_assets(value: Any) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "checkpoint",
        "source_manifest_file",
        "assets",
        "generator_asset",
        "identity",
        "attestation_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("Stage-2 runtime assets schema mismatch")
    if value.get("schema") != STAGE2_RUNTIME_ASSETS_SCHEMA:
        raise ValueError("Stage-2 runtime assets version mismatch")
    identity = validate_stage2_runtime_asset_identity(value.get("identity"))
    if value.get("checkpoint") != identity["checkpoint"]:
        raise RuntimeError("Stage-2 runtime checkpoint identities disagree")
    source_file = value.get("source_manifest_file")
    if not isinstance(source_file, Mapping) or set(source_file) != {
        "path",
        "size",
        "sha256",
        "identity",
    }:
        raise ValueError("Stage-2 runtime source manifest file schema mismatch")
    if source_file["path"] != identity["source_manifest"]["path"]:
        raise RuntimeError("Stage-2 runtime source manifest paths disagree")
    if source_file["sha256"] != identity["source_manifest"]["file_sha256"]:
        raise RuntimeError("Stage-2 runtime source manifest hashes disagree")
    if source_file["size"] != identity["source_manifest"]["size"]:
        raise RuntimeError("Stage-2 runtime source manifest sizes disagree")
    source_file_identity = _validate_identity(
        source_file["identity"], label="runtime source manifest"
    )
    if source_file_identity["size"] != source_file["size"]:
        raise RuntimeError("Stage-2 runtime source manifest size/identity mismatch")
    assets = value.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != set(_RUNTIME_FILE_ASSET_NAMES):
        raise ValueError("Stage-2 runtime detailed asset set mismatch")
    normalized_assets = {
        name: _validate_runtime_file_asset(
            assets[name], label=f"Stage-2 runtime asset {name}"
        )
        for name in _RUNTIME_FILE_ASSET_NAMES
    }
    for name, asset in normalized_assets.items():
        if _compact_asset(asset) != identity["assets"][name]:
            raise RuntimeError(
                f"Stage-2 runtime asset {name} compact identity mismatch"
            )
    generator_asset = value.get("generator_asset")
    if not isinstance(generator_asset, Mapping):
        raise TypeError("Stage-2 trusted Generator asset must be an object")
    canonical_json_sha256(dict(generator_asset))
    without_hash = dict(value)
    claimed = _sha256(
        without_hash.pop("attestation_sha256"), "runtime assets attestation SHA256"
    )
    if claimed != canonical_json_sha256(without_hash):
        raise RuntimeError("Stage-2 runtime assets attestation self hash mismatch")
    return {
        "schema": STAGE2_RUNTIME_ASSETS_SCHEMA,
        "checkpoint": dict(value["checkpoint"]),
        "source_manifest_file": dict(source_file),
        "assets": normalized_assets,
        "generator_asset": dict(generator_asset),
        "identity": identity,
        "attestation_sha256": claimed,
    }


def assert_stage2_runtime_asset_identities(
    value: Mapping[str, Any],
    *,
    names: Sequence[str] | None = None,
    include_source_manifest: bool = False,
) -> None:
    """Cheaply prove that rank-0-authenticated paths did not change."""

    attestation = validate_stage2_runtime_assets(value)
    selected = tuple(names) if names is not None else _RUNTIME_FILE_ASSET_NAMES
    if len(selected) != len(set(selected)) or any(
        name not in _RUNTIME_FILE_ASSET_NAMES for name in selected
    ):
        raise ValueError("invalid Stage-2 runtime asset identity selection")
    assets = attestation["assets"]
    for name in selected:
        asset = assets[name]
        root = Path(asset["root"])
        live_files = _live_tree_files(root, kind=asset["kind"], label=name)
        if [
            path.name if asset["kind"] == "file" else path.relative_to(root).as_posix()
            for path in live_files
        ] != [entry["relative_path"] for entry in asset["files"]]:
            raise RuntimeError(f"Stage-2 runtime asset {name} file set changed")
        for path, entry in zip(live_files, asset["files"]):
            if path.is_symlink() or str(path.resolve()) != entry["path"]:
                raise RuntimeError(f"Stage-2 runtime asset {name} path changed")
            if _stat_identity(path.stat()) != entry["identity"]:
                raise RuntimeError(f"Stage-2 runtime asset {name} identity changed")
    if include_source_manifest:
        source = attestation["source_manifest_file"]
        path = Path(source["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve() != path
            or _stat_identity(path.stat()) != source["identity"]
        ):
            raise RuntimeError("Stage-2 runtime source manifest identity changed")


__all__ = [
    "STAGE2_INFERENCE_ASSET_API_VERSION",
    "STAGE2_RUNTIME_ASSETS_SCHEMA",
    "STAGE2_RUNTIME_ASSET_IDENTITY_SCHEMA",
    "assert_stage2_runtime_asset_identities",
    "build_stage2_runtime_assets",
    "stage2_generator_asset_content_sha256",
    "validate_stage2_runtime_asset_identity",
    "validate_stage2_runtime_assets",
]
