from __future__ import annotations

from pathlib import Path
import subprocess

from omegaconf import OmegaConf

from utils.stage2_config import resolve_stage2_config

PROJECT_ROOT = Path(__file__).parents[1]
RUNBOOK = PROJECT_ROOT / "docs" / "STAGE2_H100_QUICK_DEPLOY_ZH.md"
CONFIG = PROJECT_ROOT / "configs" / "train_i2v_stage2_600cats.yaml"
PREPARE = PROJECT_ROOT / "prepare_stage2.sh"
PRECOMPUTE = PROJECT_ROOT / "precompute_stage2_i2v_cache_h100_8gpu.sh"
CONTRACT_HASH = "dae3f4075f073351f27126d86a61be38d3c370fd5399a381788a0f51d959a5ea"


def test_stage2_runbook_exposes_only_the_five_pretrain_checks():
    text = RUNBOOK.read_text(encoding="utf-8")

    required = (
        "ACTION_SIDECAR_600",
        "ATTEST_STAGE2_TEACHER=1",
        "bash prepare_stage2.sh",
        "CHECK_1_TEACHER_PASS",
        "CHECK_2_GENERATOR_PASS",
        "CHECK_3_CONFIG_PASS",
        "CHECK_4_DATA_PASS",
        "CHECK_5_ROLE_INIT_PASS",
        "STAGE2_PRETRAIN_PASS",
    )
    assert all(item in text for item in required)
    assert text.count("```") % 2 == 0
    assert "--stage2-smoke" not in text
    assert "STAGE2_FORMAL_TRAINING_PASS" not in text
    assert "checkpoint_stage2_" not in text
    assert "configs/train_i2v_stage2.yaml" not in text
    assert len(text.splitlines()) < 60


def test_stage2_runtime_asset_environment_reaches_the_resolver(monkeypatch):
    paths = {
        "LONG_LIVE_STAGE2_ARCHITECTURE_ROOT": "/external/architecture",
        "LONG_LIVE_STAGE2_GENERATOR_BASE": "/external/generator.pt",
        "LONG_LIVE_STAGE2_GENERATOR_MANIFEST": "/external/generator.manifest.json",
        "LONG_LIVE_STAGE2_REAL_SCORE_BASE": "/external/teacher",
        "LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST": "/external/teacher.manifest.json",
        "LONG_LIVE_STAGE2_METADATA_PATH": "/external/metadata.csv",
        "LONG_LIVE_STAGE2_SOURCE_MANIFEST": "/external/f25.attested.json",
        "LONG_LIVE_STAGE2_ACTION_LABELS_PATH": "/external/actions.csv",
        "LONG_LIVE_STAGE2_CACHE_DIR": "/external/cache",
        "LONG_LIVE_STAGE2_NEGATIVE_MANIFEST": "/external/negative.json",
    }
    for name, value in paths.items():
        monkeypatch.setenv(name, value)

    resolved = resolve_stage2_config(OmegaConf.load(CONFIG))

    assert resolved.architecture_root == paths["LONG_LIVE_STAGE2_ARCHITECTURE_ROOT"]
    assert (
        resolved.init_generator_checkpoint == paths["LONG_LIVE_STAGE2_GENERATOR_BASE"]
    )
    assert (
        resolved.init_generator_manifest == paths["LONG_LIVE_STAGE2_GENERATOR_MANIFEST"]
    )
    assert (
        resolved.init_real_score_checkpoint == paths["LONG_LIVE_STAGE2_REAL_SCORE_BASE"]
    )
    assert (
        resolved.init_real_score_manifest
        == paths["LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST"]
    )
    assert resolved.source_cache_manifest == paths["LONG_LIVE_STAGE2_SOURCE_MANIFEST"]
    assert (
        resolved.negative_conditioning_manifest
        == paths["LONG_LIVE_STAGE2_NEGATIVE_MANIFEST"]
    )
    assert resolved.generator_stage1_step == 3750
    assert resolved.contract_hash() == CONTRACT_HASH


def test_prepare_stage2_runs_exactly_the_five_pretrain_gates_without_training():
    text = PREPARE.read_text(encoding="utf-8")

    for marker in (
        "CHECK_1_TEACHER_PASS",
        "CHECK_2_GENERATOR_PASS",
        "CHECK_3_CONFIG_PASS",
        "CHECK_4_DATA_PASS",
        "CHECK_5_ROLE_INIT_PASS",
        "STAGE2_PRETRAIN_PASS",
    ):
        assert text.count(marker) == 1

    for command in (
        "scripts/create_stage2_teacher_manifest.py",
        "scripts/merge_lora_generator.py",
        "scripts/prepare_stage2_i2v_f25_cache.py",
        "upgrade-source-manifest",
        "prepare-negative",
        "scripts/preflight_stage2_roles.py",
    ):
        assert command in text

    assert "expected_step=3750" in text
    assert "wan_native_transformer" in text
    assert "negative_conditioning_manifest.json" in text
    assert "init_only_audit_not_training_checkpoint" in text
    assert "--stage2-smoke" not in text
    assert "train.py" not in text
    assert "--no-auto-resume" not in text
    assert "stage1_step3075" not in text
    assert "git " not in text
    assert "STAGE2_PRETRAIN_INNER" not in text

    completed = subprocess.run(
        ["bash", "-n", str(PREPARE)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_stage2_audit_commands_pass_the_operator_action_sidecar_explicitly():
    prepare_text = PREPARE.read_text(encoding="utf-8")
    precompute_text = PRECOMPUTE.read_text(encoding="utf-8")

    sidecar_argument = '--action-labels-path "$ACTION_SIDECAR_600"'
    assert prepare_text.count(sidecar_argument) == 1
    assert precompute_text.count(sidecar_argument) == 2


def test_prepare_stage2_call_chain_has_no_git_clean_or_code_version_gate():
    paths = (
        PREPARE,
        PROJECT_ROOT / "scripts" / "prepare_stage2_i2v_f25_cache.py",
        PROJECT_ROOT / "scripts" / "audit_stage2_i2v_cache.py",
        PROJECT_ROOT / "scripts" / "preflight_stage2_roles.py",
        PROJECT_ROOT / "precompute_stage2_i2v_cache_h100_8gpu.sh",
        PROJECT_ROOT / "trainer" / "stage2_distillation.py",
        PROJECT_ROOT / "utils" / "stage2_f25_cache.py",
        PROJECT_ROOT / "utils" / "stage2_i2v_data.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in (
        "git rev-parse",
        "git status",
        "git clone",
        "_resolve_clean_repo_code_version",
        "_git_code_version",
        "expected_upgrade_code_version",
        "expected-git-commit",
        "_repo_identity",
        "clean Git identity",
        "worktree_clean",
        "ignored_files_absent",
        "producer file changed after preparation",
        "validator file changed after upgrade",
        "_PRODUCER_RELATIVE_PATH",
    ):
        assert forbidden not in text
