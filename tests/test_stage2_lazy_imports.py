from pathlib import Path
import subprocess
import sys


def test_stage2_i2v_data_import_does_not_require_initialized_stage1_module():
    root = Path(__file__).resolve().parents[1]
    program = """
import sys
import types

partial_stage1 = types.ModuleType("utils.stage1_i2v_data")
sys.modules["utils.stage1_i2v_data"] = partial_stage1

from utils import stage2_i2v_data

assert stage2_i2v_data.STAGE1_CACHE_SCHEMA_VERSION == 1
print("stage2-lazy-stage1-import-ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stage2-lazy-stage1-import-ok"
