import json
import os
import subprocess
import sys


def test_cloak_helpers_do_not_import_retired_inferdpt_pipeline():
    env = {**os.environ, "PYTHONPATH": "src"}
    code = """
import json
import sys
import cloak.concurrent
import cloak.llm
print(json.dumps({
    name: name in sys.modules
    for name in [
        "inferdpt.pipeline",
        "inferdpt.rantext",
        "inferdpt.embeddings",
        "inferdpt.extraction",
    ]
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    assert json.loads(proc.stdout) == {
        "inferdpt.pipeline": False,
        "inferdpt.rantext": False,
        "inferdpt.embeddings": False,
        "inferdpt.extraction": False,
    }


def test_llm_cache_path_uses_cloak_env_name(monkeypatch, tmp_path):
    from cloak.llm import _cache_path

    monkeypatch.delenv("CLOAK_LLM_CACHE", raising=False)
    monkeypatch.setenv("CLOAK_LLM_CACHE", str(tmp_path))

    path = _cache_path(
        "model",
        [{"role": "user", "content": "hello"}],
        {"temperature": 0.0},
        "http://example.test/v1",
    )

    assert path is not None
    assert path.startswith(str(tmp_path))
    assert path.endswith(".json")
