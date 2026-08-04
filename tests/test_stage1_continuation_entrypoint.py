from __future__ import annotations

import os
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT_ROOT / "infer_stage1_two_actions_continuation_10s.sh"


def _shell_fixture(tmp_path: Path):
    project = tmp_path / "project"
    metadata = (
        project
        / "testsets"
        / "metadata_8cases_two_actions_continuation_480x832_253frames.csv"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text("fixture\n", encoding="utf-8")
    train = tmp_path / "training"
    (train / "checkpoint_model_003750").mkdir(parents=True)
    log = tmp_path / "invocation.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$CUDA_VISIBLE_DEVICES" "$@" > "$FAKE_PYTHON_LOG"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "LONG_LIVE_STAGE1_PROJECT_ROOT": str(project),
        "LONG_LIVE_STAGE1_TRAIN_DIR": str(train),
        "LONG_LIVE_STAGE1_ARCHITECTURE_ROOT": str(tmp_path / "architecture"),
        "LONG_LIVE_STAGE1_T5_CHECKPOINT": str(tmp_path / "t5.pt"),
        "LONG_LIVE_STAGE1_TOKENIZER_DIR": str(tmp_path / "tokenizer"),
        "LONG_LIVE_STAGE1_VAE_CHECKPOINT": str(tmp_path / "vae.pt"),
        "LONG_LIVE_STAGE1_BASE_CHECKPOINT": str(tmp_path / "base.pt"),
        "LONG_LIVE_STAGE1_BASE_MANIFEST": str(tmp_path / "base.json"),
        "LONG_LIVE_STAGE1_PYTHON": str(fake_python),
        "FAKE_PYTHON_LOG": str(log),
    }
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return project, train, log, env


def test_continuation_shell_syntax_and_fixed_invocation(tmp_path):
    subprocess.run(["bash", "-n", ENTRYPOINT], check=True)
    project, train, log, env = _shell_fixture(tmp_path)
    work_dir = tmp_path / "new-work"

    subprocess.run(
        [ENTRYPOINT, work_dir],
        check=True,
        env=env,
        cwd=PROJECT_ROOT,
    )

    invocation = log.read_text(encoding="utf-8").splitlines()
    assert invocation == [
        "0",
        "scripts/run_stage1_continuation_validation.py",
        "--training-checkpoint",
        str(train / "checkpoint_model_003750"),
        "--metadata",
        str(
            project
            / "testsets"
            / "metadata_8cases_two_actions_continuation_480x832_253frames.csv"
        ),
        "--work-dir",
        str(work_dir),
    ]
    source = ENTRYPOINT.read_text(encoding="utf-8")
    assert "LONG_LIVE_STAGE1_CONTINUATION_WORK_DIR" in source
    for forbidden in (
        "--sampling-steps",
        "--guidance-scale",
        "--seed",
        "--keep-merged",
        "--sink-size",
    ):
        assert forbidden not in source


def test_continuation_shell_rejects_nonempty_workdir_and_multiple_gpus(tmp_path):
    _project, _train, log, env = _shell_fixture(tmp_path)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("user data", encoding="utf-8")
    rejected = subprocess.run(
        [ENTRYPOINT, nonempty],
        check=False,
        env=env,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "must be empty" in rejected.stderr
    assert (nonempty / "keep.txt").read_text(encoding="utf-8") == "user data"
    assert not log.exists()

    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    rejected_gpu = subprocess.run(
        [ENTRYPOINT, tmp_path / "gpu-work"],
        check=False,
        env=env,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert rejected_gpu.returncode == 2
    assert "exactly one visible GPU" in rejected_gpu.stderr
    assert not log.exists()
