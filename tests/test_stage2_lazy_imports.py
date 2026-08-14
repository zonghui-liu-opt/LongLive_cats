from pathlib import Path
import subprocess
import sys


def test_stage2_prepare_imports_do_not_require_initialized_stage1_module():
    root = Path(__file__).resolve().parents[1]
    program = """
import sys
import types

partial_stage1 = types.ModuleType("utils.stage1_i2v_data")
sys.modules["utils.stage1_i2v_data"] = partial_stage1

from scripts import (
    audit_stage2_i2v_cache,
    prepare_stage2_i2v_f25_cache,
    validate_stage2_i2v_cache_inputs,
)
from utils import stage2_f25_cache, stage2_i2v_data

assert stage2_i2v_data.STAGE1_CACHE_SCHEMA_VERSION == 1
assert stage2_f25_cache.F25_BASE_MANIFEST_NAME == "cache_manifest.json"
assert prepare_stage2_i2v_f25_cache.__name__.endswith("prepare_stage2_i2v_f25_cache")
assert audit_stage2_i2v_cache.__name__.endswith("audit_stage2_i2v_cache")
assert validate_stage2_i2v_cache_inputs.__name__.endswith(
    "validate_stage2_i2v_cache_inputs"
)
print("stage2-prepare-lazy-stage1-import-ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stage2-prepare-lazy-stage1-import-ok"
