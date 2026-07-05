"""--n-docs / --env / --arms are threaded (no hardcoded doc-count or artifact/env
paths left on the touched load paths)."""
import re
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ["scripts/train_ranker.py", "scripts/reward_gate.py", "scripts/build_probes.py",
           "scripts/spikes/roundtrip_support_scan.py"]


def _help(script):
    return subprocess.run([sys.executable, str(ROOT / script), "--help"],
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": f"{ROOT}/src:{ROOT}/scripts"}).stdout


def test_no_hardcoded_doc_count():
    for s in SCRIPTS:
        src = (ROOT / s).read_text()
        assert not re.search(r"load_task_docs\([^)]*,\s*16\s*\)", src), s
        assert "--n-docs" in src, s


def test_no_hardcoded_env_or_arms_path():
    # the env/arms artifact paths may appear ONLY as argparse defaults, never baked into a
    # load call — any other occurrence is a hardcoded path the pilot chain can't redirect.
    for s in SCRIPTS:
        for line in (ROOT / s).read_text().splitlines():
            if '"data/ranker_env.json"' in line or '"data/task_arms_tau0.02.json"' in line:
                assert "default=" in line, f"{s}: {line.strip()}"


def test_help_exposes_flags():
    for s in ["scripts/train_ranker.py"]:
        assert "--n-docs" in _help(s)
    for s in SCRIPTS:
        h = _help(s)
        assert "--env" in h, s
        assert "--arms" in h, s
