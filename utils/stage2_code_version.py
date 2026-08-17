"""Version Stage-2 runtime code without relying on repository metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path


def capture_stage2_source_version() -> dict[str, str]:
    """Return a deterministic digest of every Stage-2 runtime source file."""

    root = Path(__file__).resolve().parents[1]
    source_paths: set[Path] = {root / "utils" / "distributed.py"}
    for pattern in (
        "trainer/stage2_*.py",
        "model/stage2_*.py",
        "pipeline/stage2_*.py",
        "utils/stage2_*.py",
        "scripts/*stage2*.py",
    ):
        source_paths.update(path for path in root.glob(pattern) if path.is_file())

    digest = hashlib.sha256()
    for path in sorted(
        source_paths, key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {"stage2_source_sha256": digest.hexdigest()}


__all__ = ["capture_stage2_source_version"]
