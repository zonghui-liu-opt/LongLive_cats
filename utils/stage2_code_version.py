"""Version Stage-2 runtime code without relying on repository metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path

# Keep this boundary explicit: it is the inference execution closure, not a
# repository snapshot.  Stage-2-owned files use narrow name-based globs while
# shared helpers and Wan implementation directories are listed deliberately.
_STAGE2_SOURCE_FILES = (
    "utils/distributed.py",
    "utils/inference_utils.py",
    "utils/lora_utils.py",
    "utils/parameter_names.py",
    "utils/position_embedding_utils.py",
    "utils/scheduler.py",
    "utils/stage1_causal_validation.py",
    "utils/stage1_continuation_validation.py",
    "utils/stage1_i2v_data.py",
    "utils/stage1_i2v_schema.py",
    "utils/stage1_io.py",
)
_STAGE2_SOURCE_GLOBS = (
    "trainer/stage2_*.py",
    "model/stage2_*.py",
    "pipeline/stage2_*.py",
    "utils/stage2_*.py",
    "scripts/*stage2*.py",
    # Stage-2 constructs and executes these local Wan implementations directly.
    "wan_5b/configs/*.py",
    "wan_5b/modules/*.py",
    "wan_5b/utils/*.py",
    "utils/wan*.py",
)


def _stage2_source_paths(root: Path) -> tuple[Path, ...]:
    source_paths: set[Path] = set()
    for relative in _STAGE2_SOURCE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"required Stage-2 runtime source is missing: {relative}"
            )
        source_paths.add(path)

    for pattern in _STAGE2_SOURCE_GLOBS:
        matches = tuple(sorted(root.glob(pattern)))
        if not matches:
            raise RuntimeError(f"Stage-2 runtime source glob is empty: {pattern}")
        for path in matches:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    "Stage-2 runtime source must be a regular file: "
                    f"{path.relative_to(root).as_posix()}"
                )
            source_paths.add(path)
    return tuple(
        sorted(source_paths, key=lambda item: item.relative_to(root).as_posix())
    )


def capture_stage2_source_version() -> dict[str, str]:
    """Return a deterministic digest of the Stage-2 inference source closure."""

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in _stage2_source_paths(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return {"stage2_source_sha256": digest.hexdigest()}


__all__ = ["capture_stage2_source_version"]
