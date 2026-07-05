"""--n-docs is threaded (no hardcoded 16 left on the doc-loading paths)."""
import re
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ["scripts/train_ranker.py", "scripts/reward_gate.py", "scripts/build_probes.py",
           "scripts/spikes/roundtrip_support_scan.py"]


def test_no_hardcoded_doc_count():
    for s in SCRIPTS:
        src = (ROOT / s).read_text()
        assert not re.search(r"load_task_docs\([^)]*,\s*16\s*\)", src), s
        assert "--n-docs" in src, s


def test_help_exposes_n_docs():
    for s in ["scripts/train_ranker.py"]:
        out = subprocess.run([sys.executable, str(ROOT / s), "--help"],
                             capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": f"{ROOT}/src:{ROOT}/scripts"})
        assert "--n-docs" in out.stdout
