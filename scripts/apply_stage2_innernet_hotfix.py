#!/usr/bin/env python3
"""Apply the cumulative Stage-2 DMD timing API fix without requiring Git.

The transformer recognizes exact legacy/current source fragments, prepares the
entire result in memory, compiles and audits it, creates a content-addressed
backup, and only then atomically replaces ``model/stage2_dmd.py``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HOTFIX_API_VERSION = "longlive_stage2_dmd_runtime/v2"


class HotfixError(RuntimeError):
    """Raised before writing when the target is unsafe or unrecognized."""


@dataclass(frozen=True)
class TransformResult:
    source: str
    changed: bool
    changed_units: tuple[str, ...]


@dataclass(frozen=True)
class HotfixResult:
    status: str
    target_path: Path
    backup_path: Path | None
    changed_units: tuple[str, ...]


@dataclass(frozen=True)
class _PatchUnit:
    name: str
    legacy: str
    current: str


_PATCH_UNITS = (
    _PatchUnit(
        "callable_import",
        "from collections.abc import Iterable, Mapping",
        "from collections.abc import Callable, Iterable, Mapping",
    ),
    _PatchUnit(
        "runtime_api_version",
        '''class Stage2DMD(nn.Module):
    """Three owned roles plus explicit, gradient-audited loss reductions."""

    def __init__(
''',
        '''class Stage2DMD(nn.Module):
    """Three owned roles plus explicit, gradient-audited loss reductions."""

    RUNTIME_API_VERSION = "longlive_stage2_dmd_runtime/v2"

    def __init__(
''',
    ),
    _PatchUnit(
        "generator_timing_signature",
        """    def generator_distribution_matching_loss_from_models(
        self,
        *,
        branch: str,
        generated_future: torch.Tensor,
        noised_score: Stage2NoisedScoreInput | Stage2NoisedScorePair,
        conditional_dict: Mapping[str, torch.Tensor],
        real_unconditional_dict: Mapping[str, torch.Tensor],
    ) -> Stage2GeneratorLossOutput:
""",
        """    def generator_distribution_matching_loss_from_models(
        self,
        *,
        branch: str,
        generated_future: torch.Tensor,
        noised_score: Stage2NoisedScoreInput | Stage2NoisedScorePair,
        conditional_dict: Mapping[str, torch.Tensor],
        real_unconditional_dict: Mapping[str, torch.Tensor],
        timing_callback: Callable[[str, Callable[[], object]], object] | None = None,
    ) -> Stage2GeneratorLossOutput:
""",
    ),
    _PatchUnit(
        "generator_timing_calls",
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
    ),
    _PatchUnit(
        "fake_score_timing_signature",
        """    def fake_score_flow_dsm_loss_from_model(
        self,
        *,
        generated_future: torch.Tensor,
        noised_fake_score: Stage2NoisedScoreInput,
        conditional_dict: Mapping[str, torch.Tensor],
    ) -> Stage2FakeScoreLossOutput:
""",
        """    def fake_score_flow_dsm_loss_from_model(
        self,
        *,
        generated_future: torch.Tensor,
        noised_fake_score: Stage2NoisedScoreInput,
        conditional_dict: Mapping[str, torch.Tensor],
        timing_callback: Callable[[str, Callable[[], object]], object] | None = None,
    ) -> Stage2FakeScoreLossOutput:
""",
    ),
    _PatchUnit(
        "fake_score_timing_call",
        """        fake_raw_flow, _ = self.fake_score.forward_score(
            noisy_image_or_video=noisy_fake_score,
            conditional_dict=conditional_dict,
            frame_timestep=noising.frame_timestep,
        )
""",
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
    ),
)


_EXPECTED_METHODS = {
    "fake_score_flow_dsm_loss_from_model": (
        "generated_future",
        "noised_fake_score",
        "conditional_dict",
        "timing_callback",
    ),
    "generator_distribution_matching_loss_from_models": (
        "branch",
        "generated_future",
        "noised_score",
        "conditional_dict",
        "real_unconditional_dict",
        "timing_callback",
    ),
}


def _apply_exact_unit(source: str, unit: _PatchUnit) -> tuple[str, bool]:
    legacy_count = source.count(unit.legacy)
    current_count = source.count(unit.current)
    if legacy_count == 1 and current_count == 0:
        return source.replace(unit.legacy, unit.current, 1), True
    if legacy_count == 0 and current_count == 1:
        return source, False
    raise HotfixError(
        f"unrecognized or partial Stage-2 source at {unit.name}: "
        f"legacy_matches={legacy_count}, current_matches={current_count}"
    )


def _validate_current_ast(source: str) -> None:
    try:
        tree = ast.parse(source, filename="model/stage2_dmd.py")
        compile(source, "model/stage2_dmd.py", "exec")
    except (SyntaxError, ValueError) as error:
        raise HotfixError(
            f"transformed Stage-2 source does not compile: {error}"
        ) from error

    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Stage2DMD"
    ]
    if len(classes) != 1:
        raise HotfixError(f"expected one Stage2DMD class, found {len(classes)}")
    model_class = classes[0]
    version_values = [
        statement.value.value
        for statement in model_class.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "RUNTIME_API_VERSION"
        and isinstance(statement.value, ast.Constant)
    ]
    if version_values != [HOTFIX_API_VERSION]:
        raise HotfixError(
            "Stage2DMD runtime API version is missing or unexpected: "
            f"actual={version_values!r}"
        )

    methods = {
        statement.name: statement
        for statement in model_class.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for method_name, expected_names in _EXPECTED_METHODS.items():
        method = methods.get(method_name)
        if method is None:
            raise HotfixError(f"Stage2DMD lacks {method_name}")
        positional_names = tuple(
            argument.arg for argument in (*method.args.posonlyargs, *method.args.args)
        )
        keyword_names = tuple(argument.arg for argument in method.args.kwonlyargs)
        if positional_names != ("self",) or keyword_names != expected_names:
            raise HotfixError(
                f"Stage2DMD {method_name} signature mismatch: "
                f"positional={positional_names}, keyword_only={keyword_names}"
            )
        if method.args.vararg is not None or method.args.kwarg is not None:
            raise HotfixError(f"Stage2DMD {method_name} must not use *args/**kwargs")
        defaults = method.args.kw_defaults
        if any(value is not None for value in defaults[:-1]) or not (
            isinstance(defaults[-1], ast.Constant) and defaults[-1].value is None
        ):
            raise HotfixError(f"Stage2DMD {method_name} defaults mismatch")


def transform_stage2_dmd_source(source: str) -> TransformResult:
    """Return a fully validated v2 source or reject before any disk write."""

    if not isinstance(source, str) or not source:
        raise HotfixError("model/stage2_dmd.py is empty or unreadable")
    transformed = source
    changed_units: list[str] = []
    for unit in _PATCH_UNITS:
        transformed, changed = _apply_exact_unit(transformed, unit)
        if changed:
            changed_units.append(unit.name)
    _validate_current_ast(transformed)
    return TransformResult(
        source=transformed,
        changed=bool(changed_units),
        changed_units=tuple(changed_units),
    )


def _write_backup(target: Path, original: bytes) -> Path:
    digest = hashlib.sha256(original).hexdigest()[:16]
    backup = target.with_name(f"{target.name}.pre_hotfix_{digest}.bak")
    try:
        descriptor = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if backup.read_bytes() != original:
            raise HotfixError(f"existing backup has unexpected contents: {backup}")
        return backup
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(original)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        backup.unlink(missing_ok=True)
        raise
    return backup


def _atomic_replace(target: Path, replacement: bytes, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.hotfix.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(replacement)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            directory_descriptor = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_runtime(project_root: Path, target: Path) -> str:
    probe = r"""
import inspect
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected_source = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(root))
from model.stage2_dmd import Stage2DMD
from trainer.stage2_distillation import _audit_stage2_dmd_runtime_api

actual_source = Path(inspect.getsourcefile(Stage2DMD) or "<unknown>").resolve()
if actual_source != expected_source:
    raise RuntimeError(
        f"Stage2DMD loaded from unexpected source: {actual_source} != {expected_source}"
    )
audit = _audit_stage2_dmd_runtime_api(Stage2DMD)
print(
    "STAGE2_DMD_RUNTIME_API=PASS "
    f"version={audit['api_version']} source={audit['source_file']}"
)
"""
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    with tempfile.TemporaryDirectory(prefix="stage2_hotfix_pycache_") as pycache:
        environment["PYTHONPYCACHEPREFIX"] = pycache
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-X",
                f"pycache_prefix={pycache}",
                "-c",
                probe,
                str(project_root),
                str(target),
            ],
            cwd=project_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise HotfixError(
            "isolated Stage-2 runtime verification failed"
            + (f":\n{details}" if details else "")
        )
    return completed.stdout.strip()


def apply_stage2_innernet_hotfix(
    project_root: str | Path,
    *,
    check_only: bool = False,
    verify_runtime: bool = True,
) -> HotfixResult:
    root = Path(project_root).expanduser().resolve()
    target = root / "model" / "stage2_dmd.py"
    if not target.is_file():
        raise HotfixError(f"Stage-2 model source does not exist: {target}")
    original_bytes = target.read_bytes()
    try:
        original_source = original_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HotfixError(f"Stage-2 model source is not UTF-8: {target}") from error
    transformed = transform_stage2_dmd_source(original_source)

    if transformed.changed and check_only:
        return HotfixResult(
            status="NEEDS_PATCH",
            target_path=target,
            backup_path=None,
            changed_units=transformed.changed_units,
        )

    backup: Path | None = None
    status = "ALREADY_APPLIED"
    if transformed.changed:
        backup = _write_backup(target, original_bytes)
        replacement = transformed.source.encode("utf-8")
        _atomic_replace(target, replacement, target.stat().st_mode & 0o7777)
        status = "PATCHED"

    runtime_message = _verify_runtime(root, target) if verify_runtime else ""
    if runtime_message:
        print(runtime_message)
    return HotfixResult(
        status=status,
        target_path=target,
        backup_path=backup,
        changed_units=transformed.changed_units,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely apply the cumulative no-Git Stage-2 DMD runtime hotfix."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="LongLive project root (default: parent of this script's scripts dir)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate only; exit 2 if the recognized legacy source needs patching",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = apply_stage2_innernet_hotfix(
            args.project_root,
            check_only=args.check,
            verify_runtime=True,
        )
    except (HotfixError, OSError) as error:
        print(f"STAGE2_INNERNET_HOTFIX=FAIL error={error}", file=sys.stderr)
        return 1

    units = ",".join(result.changed_units) or "none"
    backup = str(result.backup_path) if result.backup_path is not None else "none"
    print(
        f"STAGE2_INNERNET_HOTFIX={result.status} "
        f"target={result.target_path} backup={backup} units={units}"
    )
    return 2 if result.status == "NEEDS_PATCH" else 0


if __name__ == "__main__":
    raise SystemExit(main())
