#!/usr/bin/env python3
"""Apply cumulative Stage-2 runtime fixes without requiring Git.

The transformer recognizes exact legacy/current source fragments, prepares the
entire multi-file result in memory, compiles and audits it, creates
content-addressed backups, and only then atomically replaces runtime sources.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
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
    target_paths: tuple[Path, ...]
    backup_paths: tuple[Path, ...]
    changed_files: tuple[str, ...]


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


_TRAINER_PATCH_UNITS = (
    _PatchUnit(
        "timing_runtime_fields",
        """_STAGE2_DMD_RUNTIME_REPAIR = (
    "Run scripts/apply_stage2_innernet_hotfix.py from the project root, or "
    "deploy one complete stage-2 source snapshot; do not mix trainer and model files."
)
_STAGE2_DMD_RUNTIME_METHODS = {
""",
        """_STAGE2_DMD_RUNTIME_REPAIR = (
    "Run scripts/apply_stage2_innernet_hotfix.py from the project root, or "
    "deploy one complete stage-2 source snapshot; do not mix trainer and model files."
)
_STAGE2_TIMING_RUNTIME_FIELDS = (
    "data_seconds_max",
    "h2d_seconds_max",
    "rollout_seconds_max",
    "fake_score_seconds_max",
    "real_cond_seconds_max",
    "real_uncond_seconds_max",
    "loss_build_seconds_max",
    "backward_seconds_max",
    "clip_optimizer_seconds_max",
    "orchestration_seconds_max",
    "ema_seconds_max",
)
_STAGE2_DMD_RUNTIME_METHODS = {
""",
    ),
    _PatchUnit(
        "timing_runtime_audit",
        """        methods[method_name] = str(signature)
    return {
        "api_version": actual_version,
        "model_type": f"{model_type.__module__}.{model_type.__qualname__}",
        "source_file": str(Path(source_file).expanduser().resolve()),
        "methods": methods,
    }
""",
        """        methods[method_name] = str(signature)
    from utils.stage2_metrics import STAGE2_TIMING_FIELDS

    actual_timing_fields = tuple(STAGE2_TIMING_FIELDS)
    if actual_timing_fields != _STAGE2_TIMING_RUNTIME_FIELDS:
        raise RuntimeError(
            "Stage-2 timing runtime API mismatch: "
            f"expected={list(_STAGE2_TIMING_RUNTIME_FIELDS)}, "
            f"actual={list(actual_timing_fields)}. {_STAGE2_DMD_RUNTIME_REPAIR}"
        )
    return {
        "api_version": actual_version,
        "model_type": f"{model_type.__module__}.{model_type.__qualname__}",
        "source_file": str(Path(source_file).expanduser().resolve()),
        "methods": methods,
        "timing_fields": actual_timing_fields,
    }
""",
    ),
    _PatchUnit(
        "timing_summary_category",
        """            "backward",
            "clip_optimizer",
            "ema",
""",
        """            "backward",
            "clip_optimizer",
            "orchestration",
            "ema",
""",
    ),
    _PatchUnit(
        "timing_summary_field",
        """            "backward_seconds_max": selected["backward"],
            "clip_optimizer_seconds_max": selected["clip_optimizer"],
            "compute_seconds_max": compute,
""",
        """            "backward_seconds_max": selected["backward"],
            "clip_optimizer_seconds_max": selected["clip_optimizer"],
            "orchestration_seconds_max": selected["orchestration"],
            "compute_seconds_max": compute,
""",
    ),
    _PatchUnit(
        "attempt_wall_start",
        """        from utils.distributed import fsdp2_accumulation

        module = getattr(self.model, role)
""",
        """        from utils.distributed import fsdp2_accumulation

        attempt_started = time.perf_counter()
        module = getattr(self.model, role)
""",
    ),
    _PatchUnit(
        "attempt_orchestration_measurement",
        """        numerator, count = self._reduce_loss(local_numerator, local_count)
        merged = self._reduce_diagnostics(diagnostics)
        return {
""",
        """        numerator, count = self._reduce_loss(local_numerator, local_count)
        merged = self._reduce_diagnostics(diagnostics)
        torch.cuda.synchronize(self.device)
        attempt_seconds = time.perf_counter() - attempt_started
        classified_attempt_seconds = sum(phase_timings.values()) + optimizer_seconds
        attempt_orchestration_seconds = max(
            0.0, attempt_seconds - classified_attempt_seconds
        )
        return {
""",
    ),
    _PatchUnit(
        "attempt_orchestration_result",
        """            "compute_seconds": compute_seconds,
            "optimizer_seconds": optimizer_seconds,
            "phase_timings": phase_timings,
""",
        """            "compute_seconds": compute_seconds,
            "optimizer_seconds": optimizer_seconds,
            "attempt_seconds": attempt_seconds,
            "attempt_orchestration_seconds": attempt_orchestration_seconds,
            "phase_timings": phase_timings,
""",
    ),
    _PatchUnit(
        "pre_attempt_control_start",
        """            h2d_seconds = time.perf_counter() - h2d_started
            exits = self.exit_rng.draw(
""",
        """            h2d_seconds = time.perf_counter() - h2d_started
            control_started = time.perf_counter()
            exits = self.exit_rng.draw(
""",
    ),
    _PatchUnit(
        "pre_attempt_control_elapsed",
        """            branch = (
                self._draw_branch(probability) if role == "generator" else "flow_dsm"
            )
            result = self._run_update_attempt(
""",
        """            branch = (
                self._draw_branch(probability) if role == "generator" else "flow_dsm"
            )
            control_seconds = time.perf_counter() - control_started
            result = self._run_update_attempt(
""",
    ),
    _PatchUnit(
        "post_attempt_control_start",
        """            result = self._run_update_attempt(
                role=role, batches=batches, exits=exits, branch=branch
            )
            torch.cuda.synchronize(self.device)
""",
        """            result = self._run_update_attempt(
                role=role, batches=batches, exits=exits, branch=branch
            )
            post_attempt_control_started = time.perf_counter()
            torch.cuda.synchronize(self.device)
""",
    ),
    _PatchUnit(
        "ema_control_split",
        """            if role == "generator":
                next_completed_g = self.state.completed_g + 1
                ema_started = time.perf_counter()
                ema_action = self._runtime_world_checked(
""",
        """            if role == "generator":
                next_completed_g = self.state.completed_g + 1
                ema_started = time.perf_counter()
                control_seconds += ema_started - post_attempt_control_started
                ema_action = self._runtime_world_checked(
""",
    ),
    _PatchUnit(
        "post_ema_control_restart",
        """                )
                ema_seconds = time.perf_counter() - ema_started
                self._runtime_world_checked(
                    "commit generator clock",
""",
        """                )
                ema_seconds = time.perf_counter() - ema_started
                post_attempt_control_started = time.perf_counter()
                self._runtime_world_checked(
                    "commit generator clock",
""",
    ),
    _PatchUnit(
        "post_attempt_control_elapsed",
        """            self._assert_state_consensus()
            torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
""",
        """            self._assert_state_consensus()
            torch.cuda.synchronize(self.device)
            control_seconds += time.perf_counter() - post_attempt_control_started
            elapsed = time.perf_counter() - started
""",
    ),
    _PatchUnit(
        "logical_substep_orchestration_category",
        """                    ),
                    "clip_optimizer": result["optimizer_seconds"],
                    "ema": ema_seconds,
""",
        """                    ),
                    "clip_optimizer": result["optimizer_seconds"],
                    "orchestration": result.get("attempt_orchestration_seconds", 0.0)
                    + control_seconds,
                    "ema": ema_seconds,
""",
    ),
)


_METRICS_PATCH_UNITS = (
    _PatchUnit(
        "orchestration_timing_field",
        """    "backward_seconds_max",
    "clip_optimizer_seconds_max",
    "ema_seconds_max",
""",
        """    "backward_seconds_max",
    "clip_optimizer_seconds_max",
    "orchestration_seconds_max",
    "ema_seconds_max",
""",
    ),
)


_PLOT_PATCH_UNITS = (
    _PatchUnit(
        "orchestration_plot_label",
        """        "backward_seconds_max": "Backward",
        "clip_optimizer_seconds_max": "Clip + optimizer",
        "ema_seconds_max": "EMA",
""",
        """        "backward_seconds_max": "Backward",
        "clip_optimizer_seconds_max": "Clip + optimizer",
        "orchestration_seconds_max": "Runtime orchestration + audits",
        "ema_seconds_max": "EMA",
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


def _transform_runtime_source(
    relative_path: str,
    source: str,
    patch_units: tuple[_PatchUnit, ...],
) -> TransformResult:
    if not isinstance(source, str) or not source:
        raise HotfixError(f"{relative_path} is empty or unreadable")
    transformed = source
    changed_units: list[str] = []
    for unit in patch_units:
        transformed, changed = _apply_exact_unit(transformed, unit)
        if changed:
            changed_units.append(f"{relative_path}:{unit.name}")
    try:
        ast.parse(transformed, filename=relative_path)
        compile(transformed, relative_path, "exec")
    except (SyntaxError, ValueError) as error:
        raise HotfixError(
            f"transformed {relative_path} does not compile: {error}"
        ) from error
    return TransformResult(
        source=transformed,
        changed=bool(changed_units),
        changed_units=tuple(changed_units),
    )


def transform_stage2_runtime_sources(
    sources: Mapping[str, str],
) -> dict[str, TransformResult]:
    """Validate and transform every cumulative Stage-2 runtime target in memory."""

    required = {
        "model/stage2_dmd.py",
        "trainer/stage2_distillation.py",
        "utils/stage2_metrics.py",
        "scripts/plot_stage2_training.py",
    }
    missing = sorted(required - set(sources))
    unexpected = sorted(set(sources) - required)
    if missing or unexpected:
        raise HotfixError(
            f"runtime source set mismatch: missing={missing}, unexpected={unexpected}"
        )
    return {
        "model/stage2_dmd.py": transform_stage2_dmd_source(
            sources["model/stage2_dmd.py"]
        ),
        "trainer/stage2_distillation.py": _transform_runtime_source(
            "trainer/stage2_distillation.py",
            sources["trainer/stage2_distillation.py"],
            _TRAINER_PATCH_UNITS,
        ),
        "utils/stage2_metrics.py": _transform_runtime_source(
            "utils/stage2_metrics.py",
            sources["utils/stage2_metrics.py"],
            _METRICS_PATCH_UNITS,
        ),
        "scripts/plot_stage2_training.py": _transform_runtime_source(
            "scripts/plot_stage2_training.py",
            sources["scripts/plot_stage2_training.py"],
            _PLOT_PATCH_UNITS,
        ),
    }


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
    relative_paths = (
        "model/stage2_dmd.py",
        "trainer/stage2_distillation.py",
        "utils/stage2_metrics.py",
        "scripts/plot_stage2_training.py",
    )
    targets = {relative: root / relative for relative in relative_paths}
    originals: dict[str, bytes] = {}
    sources: dict[str, str] = {}
    modes: dict[str, int] = {}
    for relative, target in targets.items():
        if not target.is_file():
            raise HotfixError(f"Stage-2 runtime source does not exist: {target}")
        original = target.read_bytes()
        try:
            source = original.decode("utf-8")
        except UnicodeDecodeError as error:
            raise HotfixError(
                f"Stage-2 runtime source is not UTF-8: {target}"
            ) from error
        originals[relative] = original
        sources[relative] = source
        modes[relative] = target.stat().st_mode & 0o7777

    transformed = transform_stage2_runtime_sources(sources)
    changed_files = tuple(
        relative for relative in relative_paths if transformed[relative].changed
    )
    changed_units = tuple(
        unit
        for relative in relative_paths
        for unit in transformed[relative].changed_units
    )
    primary_target = targets["model/stage2_dmd.py"]

    if changed_files and check_only:
        return HotfixResult(
            status="NEEDS_PATCH",
            target_path=primary_target,
            backup_path=None,
            changed_units=changed_units,
            target_paths=tuple(targets.values()),
            backup_paths=(),
            changed_files=changed_files,
        )

    backup_by_file: dict[str, Path] = {}
    status = "ALREADY_APPLIED"
    written: list[str] = []
    if changed_files:
        for relative in changed_files:
            backup_by_file[relative] = _write_backup(
                targets[relative], originals[relative]
            )
        try:
            for relative in changed_files:
                _atomic_replace(
                    targets[relative],
                    transformed[relative].source.encode("utf-8"),
                    modes[relative],
                )
                written.append(relative)
            runtime_message = (
                _verify_runtime(root, primary_target) if verify_runtime else ""
            )
        except Exception as error:
            rollback_failures = []
            for relative in reversed(written):
                try:
                    _atomic_replace(
                        targets[relative], originals[relative], modes[relative]
                    )
                except OSError as rollback_error:
                    rollback_failures.append(f"{relative}: {rollback_error}")
            details = (
                f"; rollback_failures={rollback_failures}"
                if rollback_failures
                else "; all written files rolled back"
            )
            raise HotfixError(
                f"cumulative Stage-2 hotfix failed: {error}{details}"
            ) from error
        status = "PATCHED"
    else:
        runtime_message = (
            _verify_runtime(root, primary_target) if verify_runtime else ""
        )
    if runtime_message:
        print(runtime_message)
    return HotfixResult(
        status=status,
        target_path=primary_target,
        backup_path=backup_by_file.get("model/stage2_dmd.py"),
        changed_units=changed_units,
        target_paths=tuple(targets.values()),
        backup_paths=tuple(backup_by_file.values()),
        changed_files=changed_files,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely apply cumulative no-Git Stage-2 runtime hotfixes."
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
    backups = ",".join(str(path) for path in result.backup_paths) or "none"
    files = ",".join(result.changed_files) or "none"
    print(
        f"STAGE2_INNERNET_HOTFIX={result.status} "
        f"targets={len(result.target_paths)} changed_files={files} "
        f"backups={backups} units={units}"
    )
    return 2 if result.status == "NEEDS_PATCH" else 0


if __name__ == "__main__":
    raise SystemExit(main())
