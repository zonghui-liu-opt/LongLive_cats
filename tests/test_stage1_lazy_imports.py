from pathlib import Path
import subprocess
import sys


def test_diffusion_trainer_import_does_not_eager_load_legacy_dmd_pipeline():
    root = Path(__file__).resolve().parents[1]
    program = """
import sys
from trainer import DiffusionTrainer
assert DiffusionTrainer.__name__ == 'Trainer'
assert 'trainer.distillation' not in sys.modules
assert 'model.dmd' not in sys.modules
assert 'pipeline.causal_diffusion_inference' not in sys.modules
assert 'pipeline.self_forcing_training' not in sys.modules
print('stage1-lazy-import-ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stage1-lazy-import-ok"
