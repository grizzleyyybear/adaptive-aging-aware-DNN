import pytest
import os
import sys
import subprocess
from pathlib import Path

# scripts/run_full_pipeline.py uses Hydra which is incompatible with Python 3.14
# (argparse.LazyCompletionHelp changed). Use run_eval.py smoke mode instead.
@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="Hydra LazyCompletionHelp incompatible with Python 3.14 argparse"
)
def test_full_pipeline_smoke_hydra():
    cwd = Path(__file__).parent.parent.absolute()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    result = subprocess.run(
        [sys.executable, "scripts/run_full_pipeline.py"],
        cwd=cwd, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}"
    assert (cwd / "checkpoints/trajectory_best.pt").exists()
    assert (cwd / "checkpoints/rl_policy_final.pt").exists()


def test_full_pipeline_smoke():
    """End-to-end smoke test using run_eval.py (smoke mode, 200 samples)."""
    import tempfile
    cwd = Path(__file__).parent.parent.absolute()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    with tempfile.TemporaryDirectory() as tmpdir:
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, "run_eval.py", "--smoke", "--ckpt-dir", tmpdir],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, (
            f"run_eval.py --smoke failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        # Mandatory checkpoint artifacts
        assert (Path(tmpdir) / "predictor_best.pt").exists(), "predictor checkpoint missing"
