from __future__ import annotations

from pathlib import Path
import subprocess

from omegaconf import OmegaConf

from utils.stage2_config import resolve_stage2_config

PROJECT_ROOT = Path(__file__).parents[1]
RUNBOOK = PROJECT_ROOT / "docs" / "STAGE2_H100_QUICK_DEPLOY_ZH.md"
CONFIG = PROJECT_ROOT / "configs" / "train_i2v_stage2_600cats.yaml"
PREPARE = PROJECT_ROOT / "prepare_stage2.sh"
CONTRACT_HASH = "aa4d7be1e05c846df14cee5417a298afe668429f41faa671f3021754a5616c00"


def test_stage2_runbook_covers_manifest_through_formal_training():
    text = RUNBOOK.read_text(encoding="utf-8")

    required = (
        "REAL_SCORE_MANIFEST_VERIFY_PASS",
        "GENERATOR_STEP3750_MANIFEST_PASS",
        "CONFIG_BINDING_PASS",
        "FORMAL_CACHE_AUDIT_PASS",
        "ROLE_INIT_ONLY_PASS",
        "--stage2-smoke C0",
        "--stage2-smoke C1",
        "--stage2-smoke C2",
        "STAGE2_C0_C1_C2_SMOKE_PASS",
        "--no-auto-resume",
        "checkpoint_stage2_g000280",
        "--require-complete",
        "STAGE2_FORMAL_TRAINING_PASS",
        "LONG_LIVE_STAGE2_SOURCE_MANIFEST",
        "LONG_LIVE_STAGE2_NEGATIVE_MANIFEST",
        "LONG_LIVE_STAGE2_GENERATOR_BASE",
        "LONG_LIVE_STAGE2_REAL_SCORE_BASE",
        "configs/train_i2v_stage2_600cats.yaml",
    )
    assert all(item in text for item in required)
    assert text.count("```") % 2 == 0
    assert "CHECKPOINT_A_H100_PREP_PASS" not in text
    assert "不要启动 Stage-2 训练" not in text
    assert "configs/train_i2v_stage2.yaml" not in text
    assert "329 passed" not in text


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


def test_prepare_stage2_is_teacher_only_and_emits_a_verified_handoff():
    text = PREPARE.read_text(encoding="utf-8")

    assert "REAL_SCORE_TEACHER_MANIFEST_PASS" in text
    assert "provenance.source_sha256 is not the merge_manifest.json" in text
    assert "docs/STAGE2_H100_QUICK_DEPLOY_ZH.md" in text
    assert "STAGE1_CKPT=" not in text
    assert "G_MERGED=" not in text
    assert "stage1_step3075" not in text

    completed = subprocess.run(
        ["bash", "-n", str(PREPARE)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
