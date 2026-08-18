from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import scripts.apply_stage2_innernet_hotfix as hotfix_module
from scripts.apply_stage2_innernet_hotfix import (
    HotfixError,
    apply_stage2_innernet_hotfix,
    transform_stage2_dmd_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_MODEL_SOURCE = (PROJECT_ROOT / "model" / "stage2_dmd.py").read_text(
    encoding="utf-8"
)
CURRENT_RUNTIME_SOURCES = {
    relative: (PROJECT_ROOT / relative).read_text(encoding="utf-8")
    for relative in (
        "model/stage2_dmd.py",
        "trainer/stage2_distillation.py",
        "utils/distributed.py",
        "utils/lora_utils.py",
        "utils/parameter_names.py",
        "utils/stage2_checkpoint.py",
        "utils/stage2_fsdp2.py",
        "utils/stage2_metrics.py",
        "scripts/plot_stage2_training.py",
    )
}


def _replace_once(source: str, current: str, legacy: str) -> str:
    assert source.count(current) == 1
    return source.replace(current, legacy, 1)


def _legacy_model_source() -> str:
    source = CURRENT_MODEL_SOURCE
    source = _replace_once(
        source,
        "from collections.abc import Callable, Iterable, Mapping",
        "from collections.abc import Iterable, Mapping",
    )
    source = _replace_once(
        source,
        '\n    RUNTIME_API_VERSION = "longlive_stage2_dmd_runtime/v2"\n',
        "",
    )
    source = _replace_once(
        source,
        """        timing_callback: Callable[[str, Callable[[], object]], object] | None = None,
    ) -> Stage2GeneratorLossOutput:
""",
        """    ) -> Stage2GeneratorLossOutput:
""",
    )
    source = _replace_once(
        source,
        """        # These teachers are inference-only for a Generator update.  no_grad
        # also prevents an expensive x_hat->score input Jacobian from forming.
        def measured(label: str, callback: Callable[[], object]) -> object:
            return (
                callback()
                if timing_callback is None
                else timing_callback(label, callback)
            )

        with torch.no_grad():
            fake_raw_flow, _ = measured(
                "fake_score",
                lambda: self.fake_score.forward_score(
                    noisy_image_or_video=noisy_fake_score,
                    conditional_dict=conditional_dict,
                    frame_timestep=frame_timestep,
                ),
            )
            real_cond_raw_flow, _ = measured(
                "real_cond",
                lambda: self.real_score.forward_score(
                    noisy_image_or_video=real_teacher_input,
                    conditional_dict=conditional_dict,
                    frame_timestep=frame_timestep,
                ),
            )
            real_uncond_raw_flow, _ = measured(
                "real_uncond",
                lambda: self.real_score.forward_score(
                    noisy_image_or_video=real_teacher_input,
                    conditional_dict=real_unconditional_dict,
                    frame_timestep=frame_timestep,
                ),
            )
""",
        """        # These teachers are inference-only for a Generator update.  no_grad
        # also prevents an expensive x_hat->score input Jacobian from forming.
        with torch.no_grad():
            fake_raw_flow, _ = self.fake_score.forward_score(
                noisy_image_or_video=noisy_fake_score,
                conditional_dict=conditional_dict,
                frame_timestep=frame_timestep,
            )
            real_cond_raw_flow, _ = self.real_score.forward_score(
                noisy_image_or_video=real_teacher_input,
                conditional_dict=conditional_dict,
                frame_timestep=frame_timestep,
            )
            real_uncond_raw_flow, _ = self.real_score.forward_score(
                noisy_image_or_video=real_teacher_input,
                conditional_dict=real_unconditional_dict,
                frame_timestep=frame_timestep,
            )
""",
    )
    source = _replace_once(
        source,
        """        timing_callback: Callable[[str, Callable[[], object]], object] | None = None,
    ) -> Stage2FakeScoreLossOutput:
""",
        """    ) -> Stage2FakeScoreLossOutput:
""",
    )
    source = _replace_once(
        source,
        """        def callback():
            return self.fake_score.forward_score(
                noisy_image_or_video=noisy_fake_score,
                conditional_dict=conditional_dict,
                frame_timestep=noising.frame_timestep,
            )

        fake_raw_flow, _ = (
            callback()
            if timing_callback is None
            else timing_callback("fake_score", callback)
        )
""",
        """        fake_raw_flow, _ = self.fake_score.forward_score(
            noisy_image_or_video=noisy_fake_score,
            conditional_dict=conditional_dict,
            frame_timestep=noising.frame_timestep,
        )
""",
    )
    return source


def _write_fixture(
    tmp_path: Path,
    source: str,
    *,
    runtime_sources: dict[str, str] | None = None,
) -> Path:
    sources = dict(CURRENT_RUNTIME_SOURCES)
    sources["model/stage2_dmd.py"] = source
    if runtime_sources is not None:
        sources.update(runtime_sources)
    for relative, value in sources.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value, encoding="utf-8")
        target.chmod(0o640)
    return tmp_path / "model" / "stage2_dmd.py"


def _legacy_timing_sources() -> dict[str, str]:
    sources = dict(CURRENT_RUNTIME_SOURCES)
    patch_units = {
        "trainer/stage2_distillation.py": hotfix_module._TRAINER_PATCH_UNITS,
        "utils/stage2_metrics.py": hotfix_module._METRICS_PATCH_UNITS,
        "scripts/plot_stage2_training.py": hotfix_module._PLOT_PATCH_UNITS,
    }
    for relative, units in patch_units.items():
        source = sources[relative]
        for unit in reversed(units):
            assert source.count(unit.current) == 1, (relative, unit.name)
            source = source.replace(unit.current, unit.legacy, 1)
        sources[relative] = source
    return sources


def _legacy_naming_sources() -> dict[str, str]:
    sources = dict(CURRENT_RUNTIME_SOURCES)
    patch_units = {
        "trainer/stage2_distillation.py": tuple(
            unit
            for unit in hotfix_module._TRAINER_PATCH_UNITS
            if unit.name == "generator_ema_schema_names"
        ),
        "utils/distributed.py": hotfix_module._DISTRIBUTED_PATCH_UNITS,
        "utils/lora_utils.py": hotfix_module._LORA_PATCH_UNITS,
        "utils/stage2_checkpoint.py": hotfix_module._STAGE2_CHECKPOINT_PATCH_UNITS,
        "utils/stage2_fsdp2.py": hotfix_module._STAGE2_FSDP2_PATCH_UNITS,
    }
    for relative, units in patch_units.items():
        source = sources[relative]
        for unit in reversed(units):
            assert source.count(unit.current) == 1, (relative, unit.name)
            source = source.replace(unit.current, unit.legacy, 1)
        sources[relative] = source
    return sources


def test_transform_exact_legacy_source_to_current_source():
    result = transform_stage2_dmd_source(_legacy_model_source())

    assert result.changed is True
    assert result.source == CURRENT_MODEL_SOURCE


def test_apply_hotfix_backs_up_atomically_and_is_idempotent(tmp_path: Path):
    legacy = _legacy_model_source()
    target = _write_fixture(tmp_path, legacy)

    first = apply_stage2_innernet_hotfix(tmp_path, verify_runtime=False)

    assert first.status == "PATCHED"
    assert first.backup_path is not None
    assert first.backup_path.read_text(encoding="utf-8") == legacy
    assert target.read_text(encoding="utf-8") == CURRENT_MODEL_SOURCE
    assert target.stat().st_mode & 0o777 == 0o640
    backup_paths = tuple(target.parent.glob("stage2_dmd.py.pre_hotfix_*.bak"))
    assert backup_paths == (first.backup_path,)

    second = apply_stage2_innernet_hotfix(tmp_path, verify_runtime=False)

    assert second.status == "ALREADY_APPLIED"
    assert second.backup_path is None
    assert tuple(target.parent.glob("stage2_dmd.py.pre_hotfix_*.bak")) == backup_paths


def test_current_source_is_a_noop_without_backup(tmp_path: Path):
    target = _write_fixture(tmp_path, CURRENT_MODEL_SOURCE)

    result = apply_stage2_innernet_hotfix(tmp_path, verify_runtime=False)

    assert result.status == "ALREADY_APPLIED"
    assert result.backup_path is None
    assert target.read_text(encoding="utf-8") == CURRENT_MODEL_SOURCE
    assert not tuple(target.parent.glob("*.bak"))
    assert result.backup_paths == ()


def test_phase21_timing_sources_are_patched_together_and_idempotently(tmp_path: Path):
    legacy_sources = _legacy_timing_sources()
    _write_fixture(
        tmp_path,
        CURRENT_MODEL_SOURCE,
        runtime_sources=legacy_sources,
    )

    first = apply_stage2_innernet_hotfix(tmp_path, verify_runtime=False)

    assert first.status == "PATCHED"
    assert first.changed_files == (
        "trainer/stage2_distillation.py",
        "utils/stage2_metrics.py",
        "scripts/plot_stage2_training.py",
    )
    assert len(first.backup_paths) == 3
    for relative, expected in CURRENT_RUNTIME_SOURCES.items():
        assert (tmp_path / relative).read_text(encoding="utf-8") == expected

    second = apply_stage2_innernet_hotfix(tmp_path, verify_runtime=False)
    assert second.status == "ALREADY_APPLIED"
    assert second.changed_files == ()
    assert second.backup_paths == ()


def test_phase25_naming_sources_and_new_resolver_are_patched_as_one_transaction(
    tmp_path: Path,
):
    legacy_sources = _legacy_naming_sources()
    _write_fixture(
        tmp_path,
        CURRENT_MODEL_SOURCE,
        runtime_sources=legacy_sources,
    )
    (tmp_path / "utils" / "parameter_names.py").unlink()

    first = apply_stage2_innernet_hotfix(tmp_path, verify_runtime=False)

    assert first.status == "PATCHED"
    assert first.changed_files == (
        "trainer/stage2_distillation.py",
        "utils/distributed.py",
        "utils/lora_utils.py",
        "utils/parameter_names.py",
        "utils/stage2_checkpoint.py",
        "utils/stage2_fsdp2.py",
    )
    assert len(first.backup_paths) == 5
    for relative, expected in CURRENT_RUNTIME_SOURCES.items():
        assert (tmp_path / relative).read_text(encoding="utf-8") == expected

    second = apply_stage2_innernet_hotfix(tmp_path, verify_runtime=False)
    assert second.status == "ALREADY_APPLIED"
    assert second.changed_files == ()
    assert second.backup_paths == ()


def test_multifile_hotfix_rolls_back_if_one_atomic_replace_fails(
    tmp_path: Path, monkeypatch
):
    legacy_sources = _legacy_timing_sources()
    _write_fixture(
        tmp_path,
        CURRENT_MODEL_SOURCE,
        runtime_sources=legacy_sources,
    )
    real_atomic_replace = hotfix_module._atomic_replace
    calls = 0

    def fail_second_replace(target, replacement, mode):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-file replace failure")
        return real_atomic_replace(target, replacement, mode)

    monkeypatch.setattr(hotfix_module, "_atomic_replace", fail_second_replace)

    with pytest.raises(HotfixError, match="all written files rolled back"):
        apply_stage2_innernet_hotfix(tmp_path, verify_runtime=False)

    for relative, expected in legacy_sources.items():
        assert (tmp_path / relative).read_text(encoding="utf-8") == expected


def test_mixed_v2_marker_with_legacy_methods_is_repaired(tmp_path: Path):
    mixed = _legacy_model_source().replace(
        '''class Stage2DMD(nn.Module):
    """Three owned roles plus explicit, gradient-audited loss reductions."""
''',
        '''class Stage2DMD(nn.Module):
    """Three owned roles plus explicit, gradient-audited loss reductions."""

    RUNTIME_API_VERSION = "longlive_stage2_dmd_runtime/v2"
''',
        1,
    )
    target = _write_fixture(tmp_path, mixed)

    result = apply_stage2_innernet_hotfix(tmp_path, verify_runtime=False)

    assert result.status == "PATCHED"
    assert "runtime_api_version" not in result.changed_units
    assert target.read_text(encoding="utf-8") == CURRENT_MODEL_SOURCE


def test_partial_or_unknown_source_is_rejected_without_writing(tmp_path: Path):
    partial = _legacy_model_source().replace(
        """        noised_fake_score: Stage2NoisedScoreInput,
        conditional_dict: Mapping[str, torch.Tensor],
    ) -> Stage2FakeScoreLossOutput:
""",
        """        noised_fake_score: Stage2NoisedScoreInput,
        conditioning_dict: Mapping[str, torch.Tensor],
    ) -> Stage2FakeScoreLossOutput:
""",
        1,
    )
    target = _write_fixture(tmp_path, partial)

    with pytest.raises(HotfixError, match="unrecognized|partial"):
        apply_stage2_innernet_hotfix(tmp_path, verify_runtime=False)

    assert target.read_text(encoding="utf-8") == partial
    assert not tuple(target.parent.glob("*.bak"))


def test_cli_check_verifies_current_checkout_runtime_api():
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "apply_stage2_innernet_hotfix.py"),
            "--project-root",
            str(PROJECT_ROOT),
            "--check",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "STAGE2_INNERNET_HOTFIX=ALREADY_APPLIED" in completed.stdout
    assert "STAGE2_DMD_RUNTIME_API=PASS" in completed.stdout
    assert "STAGE2_PARAMETER_NAMES_API=PASS" in completed.stdout
    assert "STAGE2_LORA_LOAD_API=PASS" in completed.stdout


def test_innernet_runbooks_publish_the_same_no_git_repair_command():
    runbooks = (
        PROJECT_ROOT / "docs" / "STAGE2_H100_QUICK_DEPLOY_ZH.md",
        PROJECT_ROOT / "docs" / "STAGE2_H100_TRAINING_INFERENCE_RUNBOOK_ZH.md",
    )
    for runbook in runbooks:
        text = runbook.read_text(encoding="utf-8")
        assert "scripts/apply_stage2_innernet_hotfix.py" in text
        assert "STAGE2_DMD_RUNTIME_API=PASS" in text
        assert "STAGE2_PARAMETER_NAMES_API=PASS" in text
        assert "STAGE2_LORA_LOAD_API=PASS" in text
        assert "STAGE2_INNERNET_HOTFIX=PATCHED" in text

    trainer_source = (PROJECT_ROOT / "trainer" / "stage2_distillation.py").read_text(
        encoding="utf-8"
    )
    assert "Run scripts/apply_stage2_innernet_hotfix.py" in trainer_source
