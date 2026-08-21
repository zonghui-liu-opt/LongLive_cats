from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

from utils.stage2_config import resolve_stage2_config

PROJECT_ROOT = Path(__file__).parents[1]
TASK = PROJECT_ROOT / "TASK-stage2-self-forcing-dmd-dfd.md"
RUNBOOK = PROJECT_ROOT / "docs" / "STAGE2_H100_QUICK_DEPLOY_ZH.md"
FULL_RUNBOOK = PROJECT_ROOT / "docs" / "STAGE2_H100_TRAINING_INFERENCE_RUNBOOK_ZH.md"
CONFIG = PROJECT_ROOT / "configs" / "train_i2v_stage2_600cats.yaml"
LONG_CONFIG = PROJECT_ROOT / "configs" / "train_i2v_stage2_600cats_micro1_acc8.yaml"
PREPARE = PROJECT_ROOT / "prepare_stage2.sh"
PRECOMPUTE = PROJECT_ROOT / "precompute_stage2_i2v_cache_h100_8gpu.sh"
CONTRACT_HASH = "a7365f2ec45f74c3918ec05725b5d19b488fa4447a409cc6b5db4ccb114dd6c6"


def test_stage2_runbook_exposes_only_the_six_pretrain_checks():
    text = RUNBOOK.read_text(encoding="utf-8")

    required = (
        "ACTION_SIDECAR_600",
        "ATTEST_STAGE2_TEACHER=1",
        "STAGE2_WORK_ROOT",
        "STAGE2_H100_TRAINING_INFERENCE_RUNBOOK_ZH.md",
        "run_stage2_h100.sh",
        "bash prepare_stage2.sh",
        "CHECK_1_TEACHER_PASS",
        "CHECK_2_GENERATOR_PASS",
        "CHECK_3_CONFIG_PASS",
        "CHECK_4_DATA_PASS",
        "CHECK_5_ROLE_INIT_PASS",
        "CHECK_6_FSDP2_ACCUMULATION_PASS",
        "STAGE2_PRETRAIN_PASS",
    )
    assert all(item in text for item in required)
    assert text.count("```") % 2 == 0
    assert "--stage2-smoke" not in text
    assert "STAGE2_FORMAL_TRAINING_PASS" not in text
    assert "checkpoint_stage2_" not in text
    assert "configs/train_i2v_stage2.yaml" not in text
    assert len(text.splitlines()) < 60
    assert "docs/STAGE2_H100_TRAINING_INFERENCE_RUNBOOK_ZH.md" in TASK.read_text(
        encoding="utf-8"
    )


def test_full_stage2_runbook_covers_exact_release_lifecycle():
    text = FULL_RUNBOOK.read_text(encoding="utf-8")
    required = (
        "head_tilt_and_wink=198",
        "jump=202",
        "play_with_a_cat_wand=200",
        "checkpoint_model_003075",
        "CHECK_6_FSDP2_ACCUMULATION_PASS",
        "--stage2-smoke C0",
        "--stage2-smoke C1",
        "--stage2-smoke C2",
        "train_i2v_stage2_600cats_micro1_acc8.yaml",
        "8卡×micro1×acc8`或`4卡×micro1×acc16",
        '--nproc-per-node="$STAGE2_GPUS"',
        "smoke_longrun",
        "Phase A=360 epoch",
        "Generator/Fake-score",
        "LR=`1e-5/2e-6`",
        "checkpoint_stage2_g004000",
        "checkpoint_stage2_g003600",
        "formal_matched_b0",
        'phase_b_mode = "dmd_only"',
        "phase_b_dfd_probability_max = 0.0",
        "immutable ancestry anchor",
        "metrics_lineage.jsonl",
        "--require-complete",
        "bash run_stage2_h100.sh infer all",
        "bash run_stage2_h100.sh infer 40 80 120 160 200 240 400 800 1200 1600 2400 3200 3600 4000",
        "STAGE2_GUIDE_INFER=PASS checkpoints=N",
        ".stage2-incomplete/",
        "Stage-2 源码 SHA-256",
        "abs(error_seconds) <= max(0.1, 0.05 * step_seconds_max)",
        "SIGKILL",
        "run_stage2_h100.sh help",
    )
    assert all(item in text for item in required)
    assert text.count("```") % 2 == 0
    assert "200/200/200" not in text
    assert "checkpoint_model_003750" not in text
    assert "git status" not in text
    assert "G60/G70" not in text


def test_documented_b0_transform_is_a_resolvable_same_contract_resume(tmp_path):
    baseline_config = OmegaConf.load(CONFIG)
    baseline = resolve_stage2_config(baseline_config)
    anchor = tmp_path / "checkpoint_stage2_g000240"
    anchor.mkdir()

    b0_config = OmegaConf.load(CONFIG)
    b0_config.checkpoints.init_from_stage1 = None
    b0_config.checkpoints.resume_stage2 = str(anchor.resolve(strict=True))
    b0_config.training.phase_b_mode = "dmd_only"
    b0_config.training.phase_b_dfd_probability_max = 0.0
    b0 = resolve_stage2_config(b0_config)

    assert b0.initialization_mode == "resume_stage2"
    assert b0.resume_stage2_checkpoint == str(anchor.resolve())
    assert b0.phase_b_mode == "dmd_only"
    assert b0.phase_b_dfd_probability_max == 0.0
    assert b0.contract_hash() == baseline.contract_hash() == CONTRACT_HASH


def test_documented_longrun_b0_transform_uses_g3600_same_contract_parent(tmp_path):
    b1_config = OmegaConf.load(LONG_CONFIG)
    b1 = resolve_stage2_config(b1_config)
    anchor = tmp_path / "checkpoint_stage2_g003600"
    anchor.mkdir()

    b0_config = OmegaConf.load(LONG_CONFIG)
    b0_config.checkpoints.init_from_stage1 = None
    b0_config.checkpoints.resume_stage2 = str(anchor.resolve(strict=True))
    b0_config.training.phase_b_mode = "dmd_only"
    b0_config.training.phase_b_dfd_probability_max = 0.0
    b0 = resolve_stage2_config(b0_config)

    assert b0.phase_a_generator_updates == 3_600
    assert b0.total_generator_updates == 4_000
    assert b0.resume_stage2_checkpoint == str(anchor.resolve())
    assert b0.contract_hash() == b1.contract_hash()
    assert b0.launch_hash() != b1.launch_hash()


def test_stage2_release_work_root_defaults_outside_the_checkout():
    text = PREPARE.read_text(encoding="utf-8")
    assert "/stage2_runs/LongLive-2.0_stage2_new" in text
    assert "LongLive-2.0/checkpoints/stage2_new" not in text


def test_stage2_isolated_release_entrypoints_bootstrap_the_checkout():
    commands = (
        [sys.executable, "-I", "-B", "train.py", "--help"],
        [
            sys.executable,
            "-I",
            "-B",
            "tests/stage2_fsdp2_accumulation_gate.py",
            "--help",
        ],
    )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


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
    assert resolved.generator_stage1_step == 3075
    assert resolved.contract_hash() == CONTRACT_HASH


def test_prepare_stage2_runs_exactly_the_six_pretrain_gates_without_training():
    text = PREPARE.read_text(encoding="utf-8")

    for marker in (
        "CHECK_1_TEACHER_PASS",
        "CHECK_2_GENERATOR_PASS",
        "CHECK_3_CONFIG_PASS",
        "CHECK_4_DATA_PASS",
        "CHECK_5_ROLE_INIT_PASS",
        "CHECK_6_FSDP2_ACCUMULATION_PASS",
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
        "tests/stage2_fsdp2_accumulation_gate.py",
        "--require-h100",
    ):
        assert command in text

    assert "expected_step=3075" in text
    assert "ATTEST_STAGE2_TEACHER" in text
    assert "wan_native_transformer" in text
    assert "negative_conditioning_manifest.json" in text
    assert "init_only_audit_not_training_checkpoint" in text
    assert "--stage2-smoke" not in text
    assert "train.py" not in text
    assert "--no-auto-resume" not in text
    assert "stage1_step3750" not in text
    assert "git " not in text
    assert "STAGE2_PRETRAIN_INNER" not in text
    assert 'cd -- "$SOURCE_REPO"' in text
    assert "rm -rf" not in text

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


def test_stage2_shells_recompute_config_hashes_without_shell_temporaries():
    for path in (PREPARE, PRECOMPUTE):
        text = path.read_text(encoding="utf-8")
        assert "$CONTRACT_HASH" not in text
        assert "$LAUNCH_HASH" not in text
        assert "config_fields=" not in text
        assert "resolve_stage2_config(OmegaConf.load" in text


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
