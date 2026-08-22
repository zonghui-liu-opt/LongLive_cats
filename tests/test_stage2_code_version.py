from __future__ import annotations

from pathlib import Path

import pytest

from utils import stage2_code_version

_MINIMAL_GLOB_SOURCES = (
    "trainer/stage2_trainer.py",
    "model/stage2_model.py",
    "pipeline/stage2_rollout.py",
    "utils/stage2_code_version.py",
    "scripts/run_stage2_inference.py",
    "wan_5b/configs/wan_ti2v_5B.py",
    "wan_5b/modules/model.py",
    "wan_5b/utils/fm_solvers_unipc.py",
    "utils/wan_5b_wrapper.py",
)


def _write_minimal_source_tree(root: Path) -> None:
    relative_paths = {
        *stage2_code_version._STAGE2_SOURCE_FILES,
        *_MINIMAL_GLOB_SOURCES,
    }
    for relative in sorted(relative_paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# source: {relative}\n", encoding="utf-8")


def test_source_closure_contains_wan_runtime_and_media_helpers() -> None:
    root = Path(stage2_code_version.__file__).resolve().parents[1]
    paths = set(stage2_code_version._stage2_source_paths(root))

    assert root / "wan_5b/utils/fm_solvers_unipc.py" in paths
    assert set((root / "wan_5b/configs").glob("*.py")) <= paths
    assert set((root / "wan_5b/modules").glob("*.py")) <= paths
    assert set(root.glob("utils/wan*.py")) <= paths
    assert root / "utils/stage1_i2v_data.py" in paths
    assert root / "utils/inference_utils.py" in paths
    assert root / "utils/stage1_causal_validation.py" in paths


@pytest.mark.parametrize(
    "relative",
    (
        "wan_5b/utils/fm_solvers_unipc.py",
        "wan_5b/configs/wan_ti2v_5B.py",
        "wan_5b/modules/model.py",
        "utils/wan_5b_wrapper.py",
        "utils/stage1_i2v_data.py",
        "utils/inference_utils.py",
    ),
)
def test_runtime_dependency_change_changes_source_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    _write_minimal_source_tree(tmp_path)
    monkeypatch.setattr(
        stage2_code_version,
        "__file__",
        str(tmp_path / "utils/stage2_code_version.py"),
    )
    before = stage2_code_version.capture_stage2_source_version()

    target = tmp_path / relative
    target.write_text(
        target.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8"
    )

    after = stage2_code_version.capture_stage2_source_version()
    assert after != before
