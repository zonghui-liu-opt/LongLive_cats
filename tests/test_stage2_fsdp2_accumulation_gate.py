from pathlib import Path
import os
import socket
import subprocess
import sys

import pytest
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "tests" / "stage2_fsdp2_accumulation_gate.py"


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="real CPU FSDP2 parity gate requires torch.distributed gloo",
)
def test_real_fsdp2_sync_and_no_sync_match_global64_reference():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT), environment.get("PYTHONPATH")) if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--master-addr=127.0.0.1",
            f"--master-port={_free_local_port()}",
            str(GATE),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STAGE2_FSDP2_ACCUMULATION_GATE=PASS mode=CPU-world2" in result.stdout
