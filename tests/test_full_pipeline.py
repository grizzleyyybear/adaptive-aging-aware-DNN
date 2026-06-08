import pytest
import os
import sys
import subprocess
from pathlib import Path

def test_full_pipeline_smoke_plain_config(tmp_path):
    cwd = Path(__file__).parent.parent.absolute()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_full_pipeline.py",
            f"output_dir={tmp_path / 'outputs'}",
            f"paper_dir={tmp_path / 'paper'}",
            f"dataset.root={tmp_path / 'data'}",
            f"checkpoint_dir={tmp_path / 'checkpoints'}",
            "mirror_checkpoints=false",
            "dataset.size=4",
            "training.epochs=1",
            "training.patience=1",
            "model.hidden_dim=32",
            "model.transformer_layers=1",
            "nsga2.pop_size=4",
            "nsga2.population_size=4",
            "nsga2.n_gen=1",
            "ppo.total_timesteps=8",
            "ppo.n_steps=4",
            "ppo.batch_size=4",
            "ppo.n_epochs=1",
            "ablation.test_size=4",
        ],
        cwd=cwd, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}"
    assert (tmp_path / "checkpoints/trajectory_best.pt").exists()
    assert (tmp_path / "outputs/models/rl_policy_final.pt").exists()


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
            [
                sys.executable,
                "run_eval.py",
                "--smoke",
                "--ckpt-dir",
                tmpdir,
                "--results-path",
                str(Path(tmpdir) / "eval_results.json"),
            ],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, (
            f"run_eval.py --smoke failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        # Mandatory checkpoint artifacts
        assert (Path(tmpdir) / "predictor_best.pt").exists(), "predictor checkpoint missing"
