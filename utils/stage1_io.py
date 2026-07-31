"""Small deterministic I/O primitives shared by Stage-1 tooling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 8 << 20) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def tree_file_hashes(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Return stable relative-path/size/hash entries for one file or tree."""
    root = Path(path).resolve()
    if root.is_file():
        return [{"path": root.name, "size": root.stat().st_size, "sha256": sha256_file(root)}]
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries: list[dict[str, Any]] = []
    for child in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise RuntimeError(f"Symlinks are not accepted in hashed model trees: {child}")
        if child.is_file():
            entries.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "size": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
    if not entries:
        raise RuntimeError(f"Hashed model tree is empty: {root}")
    return entries


def aggregate_file_hash(entries: Iterable[Mapping[str, Any]]) -> str:
    normalized = [dict(entry) for entry in entries]
    return canonical_json_sha256(normalized)


@contextmanager
def atomic_output_path(
    destination: str | os.PathLike[str], *, suffix: str = ".tmp"
) -> Iterator[Path]:
    """Yield a same-directory temporary path and atomically replace on success."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        yield temporary_path
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_bytes(destination: str | os.PathLike[str], payload: bytes) -> None:
    with atomic_output_path(destination) as temporary:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def atomic_write_json(destination: str | os.PathLike[str], value: Any) -> None:
    atomic_write_bytes(destination, canonical_json_bytes(value) + b"\n")


def atomic_torch_save(destination: str | os.PathLike[str], value: Any) -> None:
    import torch

    with atomic_output_path(destination) as temporary:
        torch.save(value, temporary)
