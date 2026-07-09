import os
import subprocess
import sys
from pathlib import Path


def test_determinism_gate_stub_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    fixtures = tmp_path / "extractor_gate_fixtures.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "extractor_determinism_gate.py"),
            "--stub",
            "--make-fixtures",
            "--fixtures",
            str(fixtures),
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert fixtures.exists()
