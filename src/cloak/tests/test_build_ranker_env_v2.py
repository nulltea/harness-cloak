from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from cloak.train.qa_builder import freeze_v2_environment_from_legacy_arms


ROOT = Path(__file__).resolve().parents[3]


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        return importlib.import_module("build_ranker_env")
    finally:
        sys.path.pop(0)


def _forbidden_keys(value):
    forbidden = {"tau", "tau_walk", "walk_risk", "p6", "k_floors", "bc_action", "exhausted"}
    if isinstance(value, dict):
        found = forbidden & set(value)
        for child in value.values():
            found.update(_forbidden_keys(child))
        return found
    if isinstance(value, list):
        found = set()
        for child in value:
            found.update(_forbidden_keys(child))
        return found
    return set()


def test_default_ranker_builder_emits_policy_free_embedded_v2_input(tmp_path, monkeypatch):
    mod = _module()
    source = "Aspirin helps."
    legacy_env = {"corpora": {"clinical": {"aci/D1": {"spans": [{
        "surface": "Aspirin", "type": "drug", "start": 0, "end": 7,
        "bc_action": 0,
        "actions": [
            {"fill": "an analgesic", "mode": "level", "aset": 100,
             "walk_risk": 0.01, "p6": 0.7},
            {"fill": None, "mode": "placeholder"},
        ],
    }]}}}}
    legacy_arms = {"clinical": {"aci/D1": {"tau_walk": [source, [{
        "surface": "Aspirin", "type": "drug", "start": 0, "end": 7,
        "lattice": ["an analgesic"], "action": "placeholder",
        "replacement": "<DRUG_1>", "risk": 0.9, "exhausted": True,
    }]]}}}
    frozen = freeze_v2_environment_from_legacy_arms(
        legacy_env, legacy_arms, source_documents={"aci/D1": source},
    )
    arms_path = tmp_path / "arms.json"
    out_path = tmp_path / "ranker-v2.json"
    arms_path.write_text(json.dumps({
        "_meta": {"v2_frozen_environment": {
            "environment_hash": frozen["environment_hash"],
        }},
        "clinical": {"aci/D1": {
            "v2_frozen_input": frozen["documents"]["aci/D1"],
        }},
    }))
    monkeypatch.setattr(sys, "argv", [
        "build_ranker_env.py", "--corpora", "clinical", "--arms", str(arms_path),
        "--out", str(out_path),
    ])

    mod.main()

    output = json.loads(out_path.read_text())
    assert output["artifact_version"] == "ranker-v2-environment-v1"
    assert output["compatibility_adapter"] == "legacy-arms-policy-free-v1"
    assert _forbidden_keys(output) == set()
    assert output["corpora"]["clinical"]["aci/D1"]["trainable"] is True
