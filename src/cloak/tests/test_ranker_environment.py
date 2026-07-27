import hashlib
import importlib
import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import get_type_hints

import pytest


ROOT = Path(__file__).resolve().parents[3]
REAL_ENVIRONMENT = ROOT / "results/ranker_v2/environment/ranker-env.json"


def _module():
    try:
        return importlib.import_module("cloak.ranker.environment")
    except ModuleNotFoundError:
        pytest.fail("cloak.ranker.environment is not implemented")


def _hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _fixture(tmp_path, monkeypatch):
    mod = _module()
    text = "Alpha met Beta and alpha ID."
    occurrences = [
        {"occurrence_id": "o-alpha-1", "decision_id": "decision-alpha",
         "start": 0, "end": 5, "surface": "Alpha", "controlled": True},
        {"occurrence_id": "o-beta", "decision_id": "decision-beta",
         "start": 10, "end": 14, "surface": "Beta", "controlled": True},
        {"occurrence_id": "o-alpha-2", "decision_id": "decision-alpha",
         "start": 19, "end": 24, "surface": "alpha", "controlled": True},
        {"occurrence_id": "o-fixed", "decision_id": "decision-fixed",
         "start": 25, "end": 27, "surface": "ID", "controlled": True},
    ]

    def action(action_id, mode, fill, authored_level_index=None):
        row = {"action_id": action_id, "mode": mode, "fill": fill, "legal": True}
        if authored_level_index is not None:
            row["authored_level_index"] = authored_level_index
        return row

    alpha = {
        "decision_id": "decision-alpha", "profile_id": "profile-alpha",
        "runtime_type": "TYPE", "canonical_key": "alpha",
        "occurrence_ids": ["o-alpha-1", "o-alpha-2"], "ranker_selectable": True,
        "actions": [
            action("alpha-level", "level", "shared fill", 0),
            action("alpha-keep", "keep", "alpha"),
            action("alpha-placeholder", "placeholder", None),
        ],
    }
    beta = {
        "decision_id": "decision-beta", "profile_id": None,
        "runtime_type": "TYPE", "canonical_key": "beta",
        "occurrence_ids": ["o-beta"], "ranker_selectable": True,
        "actions": [
            action("beta-level", "level", "shared fill", 0),
            action("beta-keep", "keep", "beta"),
            action("beta-placeholder", "placeholder", None),
        ],
    }
    fixed = {
        "decision_id": "decision-fixed", "profile_id": None,
        "runtime_type": "FIXED", "canonical_key": "id",
        "occurrence_ids": ["o-fixed"], "ranker_selectable": False,
        "actions": [action("fixed-placeholder", "placeholder", None)],
    }
    document = {
        "corpus": "fixture",
        "occurrences": occurrences,
        # Intentionally not occurrence-ordered: the loader owns policy walk order.
        "decisions": [beta, alpha, fixed],
        "detector_diagnostics": {},
    }
    document["environment_document_hash"] = _hash(document)
    frozen = {
        "artifact_version": "occurrence-decisions-v2",
        "documents": {"fixture/doc": document},
    }
    frozen["environment_hash"] = _hash(frozen)
    artifact = {
        "artifact_version": "ranker-v2-environment-v2",
        "compatibility_adapter": "frozen-arms-count-provenance-v1",
        "frozen_environment": frozen,
        "corpora": {"fixture": {"fixture/doc": {
            "decisions": json.loads(json.dumps(document["decisions"])),
            "occurrences": json.loads(json.dumps(occurrences)),
            "policy_decision_ids": ["decision-beta", "decision-alpha"],
            "trainable": True,
        }}},
    }
    path = tmp_path / "ranker-env.json"
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(
        mod, "load_task_docs", lambda corpus: [{"id": "fixture/doc", "text": text}],
    )
    return mod, path, artifact


def _rewrite(path, artifact, *, sync_inventory=True):
    document = artifact["frozen_environment"]["documents"]["fixture/doc"]
    document.pop("environment_document_hash", None)
    document["environment_document_hash"] = _hash(document)
    frozen = artifact["frozen_environment"]
    frozen.pop("environment_hash", None)
    frozen["environment_hash"] = _hash(frozen)
    if sync_inventory:
        inventory = artifact["corpora"]["fixture"]["fixture/doc"]
        inventory["decisions"] = json.loads(json.dumps(document["decisions"]))
        inventory["occurrences"] = json.loads(json.dumps(document["occurrences"]))
    path.write_text(json.dumps(artifact))


def test_public_environment_dataclasses_are_exact_and_frozen(tmp_path, monkeypatch):
    mod, path, _ = _fixture(tmp_path, monkeypatch)
    assert [field.name for field in fields(mod.RankerAction)] == [
        "action_id", "mode", "fill", "authored_level_index", "runtime_type",
    ]
    assert [field.name for field in fields(mod.RankerDecision)] == [
        "decision_id", "profile_id", "runtime_type", "canonical_key",
        "occurrence_ids", "actions",
    ]
    assert get_type_hints(mod.RankerDecision)["profile_id"] == str | None
    assert [field.name for field in fields(mod.RankerDocument)] == [
        "doc_id", "corpus", "text", "occurrences", "policy_decisions",
        "fixed_decisions",
    ]

    document = mod.load_ranker_environment(path)["fixture/doc"]
    with pytest.raises(FrozenInstanceError):
        document.policy_decisions[0].canonical_key = "changed"
    with pytest.raises(TypeError):
        document.occurrences[0]["start"] = 99


def test_loader_sorts_policy_decisions_and_preserves_nullable_profile(tmp_path, monkeypatch):
    mod, path, _ = _fixture(tmp_path, monkeypatch)
    document = mod.load_ranker_environment(path)["fixture/doc"]

    assert [row.decision_id for row in document.policy_decisions] == [
        "decision-alpha", "decision-beta",
    ]
    assert document.policy_decisions[1].profile_id is None
    assert [row.decision_id for row in document.fixed_decisions] == ["decision-fixed"]
    assert document.policy_decisions[0].occurrence_ids == ("o-alpha-1", "o-alpha-2")
    assert document.policy_decisions[0].actions[0].runtime_type == "TYPE"


@pytest.mark.parametrize("duplicate_kind", ["decision", "action"])
def test_loader_rejects_duplicate_stable_ids(tmp_path, monkeypatch, duplicate_kind):
    mod, path, artifact = _fixture(tmp_path, monkeypatch)
    decisions = artifact["frozen_environment"]["documents"]["fixture/doc"]["decisions"]
    if duplicate_kind == "decision":
        decisions.append(json.loads(json.dumps(decisions[0])))
    else:
        decisions[0]["actions"][1]["action_id"] = decisions[0]["actions"][0]["action_id"]
    _rewrite(path, artifact)

    with pytest.raises(ValueError, match=f"duplicate {duplicate_kind}"):
        mod.load_ranker_environment(path)


def test_loader_rejects_missing_mapped_occurrence(tmp_path, monkeypatch):
    mod, path, artifact = _fixture(tmp_path, monkeypatch)
    decision = artifact["frozen_environment"]["documents"]["fixture/doc"]["decisions"][1]
    decision["occurrence_ids"].append("o-missing")
    _rewrite(path, artifact)

    with pytest.raises(ValueError, match="missing mapped occurrence"):
        mod.load_ranker_environment(path)


def test_loader_rejects_fixed_decision_in_declared_policy_set(tmp_path, monkeypatch):
    mod, path, artifact = _fixture(tmp_path, monkeypatch)
    artifact["corpora"]["fixture"]["fixture/doc"]["policy_decision_ids"].append(
        "decision-fixed"
    )
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="policy decision ids"):
        mod.load_ranker_environment(path)


def test_loader_rejects_environment_and_document_hash_mismatches(tmp_path, monkeypatch):
    mod, path, artifact = _fixture(tmp_path, monkeypatch)
    artifact["frozen_environment"]["environment_hash"] = "sha256:wrong"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="environment_hash"):
        mod.load_ranker_environment(path)

    mod, path, artifact = _fixture(tmp_path, monkeypatch)
    artifact["frozen_environment"]["documents"]["fixture/doc"][
        "environment_document_hash"
    ] = "sha256:wrong"
    frozen = artifact["frozen_environment"]
    frozen["environment_hash"] = _hash({k: v for k, v in frozen.items() if k != "environment_hash"})
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="environment_document_hash"):
        mod.load_ranker_environment(path)


def test_loader_rejects_inventory_and_local_text_mismatches(tmp_path, monkeypatch):
    mod, path, artifact = _fixture(tmp_path, monkeypatch)
    artifact["corpora"]["fixture"]["fixture/doc"]["occurrences"] = []
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="inventory mismatch"):
        mod.load_ranker_environment(path)

    mod, path, _ = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mod, "load_task_docs", lambda corpus: [{"id": "fixture/doc", "text": "wrong"}],
    )
    with pytest.raises(ValueError, match="source text mismatch"):
        mod.load_ranker_environment(path)


def test_loader_rejects_legacy_environment_version(tmp_path, monkeypatch):
    mod, path, artifact = _fixture(tmp_path, monkeypatch)
    artifact["artifact_version"] = "ranker-v2-environment-v1"
    path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="ranker-v2-environment-v2"):
        mod.load_ranker_environment(path)


def test_real_environment_loads_all_documents_without_content_output():
    mod = _module()
    documents = mod.load_ranker_environment(REAL_ENVIRONMENT)

    assert len(documents) == 67
    assert sum(len(row.policy_decisions) for row in documents.values()) == 705
    assert sum(len(row.fixed_decisions) for row in documents.values()) == 208
    assert sum(
        decision.profile_id is None
        for row in documents.values()
        for decision in row.policy_decisions
    ) == 5
    assert all(
        list(row.policy_decisions) == sorted(
            row.policy_decisions,
            key=lambda decision: (
                min(
                    occurrence["start"]
                    for occurrence in row.occurrences
                    if occurrence["occurrence_id"] in decision.occurrence_ids
                ),
                decision.decision_id,
            ),
        )
        for row in documents.values()
    )
