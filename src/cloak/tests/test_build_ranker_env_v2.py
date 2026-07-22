from __future__ import annotations

import importlib
import json
import math
import sys
from copy import deepcopy
from collections.abc import Iterator
from pathlib import Path

from cloak.train.qa_builder import (
    compare_frozen_environment_semantics,
    freeze_v2_environment_from_legacy_arms,
    migrate_frozen_environment_count_provenance,
)


ROOT = Path(__file__).resolve().parents[3]
CURRENT_ENVIRONMENT = ROOT / "results/ranker_v2/environment/ranker-env.json"
V16_ENVIRONMENT = ROOT / "results/qa_v2_aci_full/ranker-env.json"
ACCEPTED_COUNT_GROUNDING_STATUSES = {
    "certifying",
    "model-proposed",
    "proposal-universe",
}


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


def _current_environment() -> dict:
    return json.loads(CURRENT_ENVIRONMENT.read_text())


def _frozen_documents(environment: dict) -> dict[str, dict]:
    return environment["frozen_environment"]["documents"]


def _iter_decisions(environment: dict) -> Iterator[tuple[str, dict]]:
    for doc_id, document in _frozen_documents(environment).items():
        for decision in document["decisions"]:
            yield doc_id, decision


def _summarize(errors: list[str], *, limit: int = 12) -> str:
    sample = "\n".join(errors[:limit])
    suffix = f"\n... {len(errors) - limit} more" if len(errors) > limit else ""
    return f"{len(errors)} contract violations:\n{sample}{suffix}"


def _level_action_contract_errors(doc_id: str, decision: dict, action: dict) -> list[str]:
    decision_id = decision["decision_id"]
    prefix = f"{doc_id}:{decision_id}:{action.get('fill')!r}"
    errors = []
    if not isinstance(action.get("action_id"), str) or not action["action_id"].startswith("sha256:"):
        errors.append(f"{prefix} missing stable action_id")
    if not isinstance(action.get("fill"), str) or not action["fill"].strip():
        errors.append(f"{prefix} missing level fill")
    index = action.get("authored_level_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        errors.append(f"{prefix} missing authored_level_index")
    if "count" not in action or "count_grounding" not in action:
        errors.append(f"{prefix} missing explicit count/count_grounding fields")
        return errors
    count = action["count"]
    grounding = action["count_grounding"]
    if count is None or grounding is None:
        if count is not None or grounding is not None:
            errors.append(f"{prefix} count and grounding must both be null or both complete")
        return errors
    if (
        not isinstance(count, (int, float))
        or isinstance(count, bool)
        or not math.isfinite(float(count))
        or float(count) < 1.0
    ):
        errors.append(f"{prefix} missing finite count >= 1")
    if not isinstance(grounding, dict):
        errors.append(f"{prefix} missing count_grounding")
        return errors
    if grounding.get("status") not in ACCEPTED_COUNT_GROUNDING_STATUSES:
        errors.append(f"{prefix} unaccepted count grounding status {grounding.get('status')!r}")
    if not grounding.get("source_family"):
        errors.append(f"{prefix} missing count grounding source_family")
    status = grounding.get("status")
    if status == "certifying" and not grounding.get("member_set_ref"):
        errors.append(f"{prefix} missing certifying member_set_ref")
    if status == "model-proposed" and not (
        grounding.get("selector") and grounding.get("count_evidence")
    ):
        errors.append(f"{prefix} missing model-proposed selector/count_evidence")
    if status == "proposal-universe" and not (
        grounding.get("member_set_ref") or grounding.get("generated_universe_ref")
    ):
        errors.append(f"{prefix} missing proposal-universe reference")
    return errors


def test_current_environment_ranker_decisions_carry_profile_and_grounded_level_counts():
    environment = _current_environment()
    errors = []
    for doc_id, decision in _iter_decisions(environment):
        if not decision.get("ranker_selectable", True):
            continue
        if "profile_id" not in decision:
            errors.append(f"{doc_id}:{decision['decision_id']} missing explicit profile_id")
        for action in decision["actions"]:
            if action.get("legal", True) and action.get("mode") == "level":
                errors.extend(_level_action_contract_errors(doc_id, decision, action))

    assert not errors, _summarize(errors)


def test_current_environment_keep_and_placeholder_actions_have_no_lattice_count():
    environment = _current_environment()
    errors = []
    lattice_count_fields = {"count", "count_grounding", "authored_level_index"}
    for doc_id, decision in _iter_decisions(environment):
        for action in decision["actions"]:
            if action.get("mode") not in {"keep", "placeholder"}:
                continue
            unexpected = sorted(lattice_count_fields & action.keys())
            if unexpected:
                errors.append(
                    f"{doc_id}:{decision['decision_id']}:{action['mode']} has {unexpected}"
                )

    assert not errors, _summarize(errors)


def test_current_environment_policy_occurrences_map_to_complete_selectable_decisions():
    environment = _current_environment()
    frozen_documents = _frozen_documents(environment)
    errors = []
    for corpus, corpus_documents in environment["corpora"].items():
        for doc_id, policy_document in corpus_documents.items():
            frozen = frozen_documents[doc_id]
            decisions = {row["decision_id"]: row for row in frozen["decisions"]}
            occurrences = {row["occurrence_id"]: row for row in frozen["occurrences"]}
            policy_ids = set(policy_document["policy_decision_ids"])
            selectable_ids = {
                decision_id
                for decision_id, decision in decisions.items()
                if decision.get("ranker_selectable", True)
            }
            if policy_ids != selectable_ids:
                errors.append(
                    f"{corpus}:{doc_id} policy ids differ from selectable decisions: "
                    f"missing={sorted(selectable_ids - policy_ids)} "
                    f"extra={sorted(policy_ids - selectable_ids)}"
                )
            for decision_id in policy_ids:
                decision = decisions.get(decision_id)
                if decision is None:
                    continue
                if "profile_id" not in decision:
                    errors.append(f"{corpus}:{doc_id}:{decision_id} missing explicit profile_id")
                for occurrence_id in decision["occurrence_ids"]:
                    occurrence = occurrences.get(occurrence_id)
                    if occurrence is None:
                        errors.append(f"{corpus}:{doc_id}:{decision_id} missing {occurrence_id}")
                    elif occurrence.get("decision_id") != decision_id:
                        errors.append(
                            f"{corpus}:{doc_id}:{occurrence_id} maps to "
                            f"{occurrence.get('decision_id')}, expected {decision_id}"
                        )
                for action in decision["actions"]:
                    if action.get("legal", True) and action.get("mode") == "level":
                        errors.extend(_level_action_contract_errors(doc_id, decision, action))

    assert not errors, _summarize(errors)


def test_current_environment_forced_decisions_are_not_ranker_selectable():
    environment = _current_environment()
    errors = []
    for doc_id, decision in _iter_decisions(environment):
        forced = any(action.get("forced_placeholder") for action in decision["actions"])
        if forced and decision.get("ranker_selectable") is not False:
            errors.append(f"{doc_id}:{decision['decision_id']} forced decision is selectable")

    assert not errors, _summarize(errors)


def test_current_environment_authored_level_counts_are_non_decreasing():
    environment = _current_environment()
    errors = []
    for doc_id, decision in _iter_decisions(environment):
        if not decision.get("ranker_selectable", True):
            continue
        levels = [
            action for action in decision["actions"]
            if action.get("legal", True) and action.get("mode") == "level"
        ]
        # Legacy rows have neither contract field. Their serialized order and `aset` values are
        # used only to make the RED failure diagnose the pre-existing nonmonotone profiles; the
        # complete-field test above independently prevents this compatibility path from passing.
        authored = sorted(
            enumerate(levels),
            key=lambda pair: pair[1].get("authored_level_index", pair[0]),
        )
        indices = [action.get("authored_level_index") for _, action in authored]
        if all(isinstance(index, int) and not isinstance(index, bool) for index in indices):
            if len(indices) != len(set(indices)):
                errors.append(
                    f"{doc_id}:{decision['decision_id']} duplicate authored indices {indices}"
                )
        non_null = [
            (position, action) for position, action in authored
            if isinstance(action.get("count"), (int, float))
            and not isinstance(action.get("count"), bool)
        ]
        for (left_position, left), (right_position, right) in zip(non_null, non_null[1:]):
            left_count = float(left["count"])
            right_count = float(right["count"])
            if right_count < left_count:
                errors.append(
                    f"{doc_id}:{decision['decision_id']} authored count decreases "
                    f"{left.get('fill')!r}={left_count:g} -> "
                    f"{right.get('fill')!r}={right_count:g} "
                    f"(serialized positions {left_position}->{right_position})"
                )

    assert not errors, _summarize(errors)


def _migration_fixture() -> tuple[dict, dict]:
    grounding = {
        "status": "certifying",
        "source_family": "fixture-universe",
        "member_set_ref": "fixture:members",
        "selector": "fixture.parent",
    }
    document = {
        "corpus": "aci",
        "occurrences": [{
            "occurrence_id": "sha256:occurrence",
            "start": 0,
            "end": 11,
            "surface": "condition a",
            "aliases": [],
            "runtime_type": "health-condition",
            "polarity": "unknown",
            "detector_provenance": {"source": "frozen-fixture", "score": 0.9},
            "overlap_disposition": "accepted",
            "decision_id": "sha256:decision",
            "controlled": True,
            "profile_match": {"outcome": "exact", "entry": "condition a"},
        }],
        "decisions": [{
            "decision_id": "sha256:decision",
            "runtime_type": "health-condition",
            "canonical_key": "condition a",
            "occurrence_ids": ["sha256:occurrence"],
            "controlled": True,
            "ranker_selectable": True,
            "actions": [
                {"action_id": "sha256:level-a", "mode": "level", "fill": "specific class",
                 "legal": True, "aset": 999.0, "coarseness_rank": 999.0,
                 "entails": ["specific class"]},
                {"action_id": "sha256:level-b", "mode": "level", "fill": "broad class",
                 "legal": True, "aset": 2.0, "coarseness_rank": 2.0,
                 "entails": ["broad class"]},
                {"action_id": "sha256:keep", "mode": "keep", "fill": "condition a",
                 "keep": True, "source_identity": True, "legal": True,
                 "entails": ["condition a"]},
                {"action_id": "sha256:placeholder", "mode": "placeholder", "fill": None,
                 "legal": True, "placeholder_type": "health-condition", "entails": []},
            ],
            "action_menu_hash": "sha256:old-menu",
            "protected_aliases": [],
            "semantic_chain": [],
        }],
        "environment_document_hash": "sha256:old-document",
    }
    frozen = {
        "artifact_version": "occurrence-decisions-v1",
        "documents": {"aci/D1": document},
        "environment_hash": "sha256:old-environment",
    }
    profiles = {
        "schema_version": 1,
        "profiles": {"health-condition": {
            "condition a": {
                "levels": ["specific class", "broad class"],
                "level_counts": {"specific class": 7.0},
                "level_grounding": {"specific class": grounding},
            },
            # A cross-profile maximum must never leak into condition a's action count.
            "condition b": {
                "levels": ["specific class"],
                "level_counts": {"specific class": 5000.0},
                "level_grounding": {"specific class": grounding},
            },
        }},
    }
    return frozen, profiles


def test_count_migration_uses_only_matched_profile_row_and_preserves_frozen_detection():
    frozen, profiles = _migration_fixture()
    frozen_before = deepcopy(frozen)
    profiles_before = deepcopy(profiles)

    migrated = migrate_frozen_environment_count_provenance(frozen, profiles)

    decision = migrated["documents"]["aci/D1"]["decisions"][0]
    actions = decision["actions"]
    assert decision["profile_id"] == "health-condition:condition a"
    assert actions[0]["authored_level_index"] == 0
    assert actions[0]["count"] == 7.0
    assert actions[0]["count_grounding"] == profiles["profiles"]["health-condition"][
        "condition a"
    ]["level_grounding"]["specific class"]
    assert actions[0]["aset"] == 999.0
    assert actions[1]["authored_level_index"] == 1
    assert actions[1]["count"] is None
    assert actions[1]["count_grounding"] is None
    assert all("count" not in action for action in actions if action["mode"] != "level")
    assert migrated["documents"]["aci/D1"]["occurrences"] == frozen_before[
        "documents"
    ]["aci/D1"]["occurrences"]
    assert [action["action_id"] for action in actions] == [
        action["action_id"] for action in frozen_before["documents"]["aci/D1"]["decisions"][0][
            "actions"
        ]
    ]
    assert profiles == profiles_before
    assert frozen == frozen_before


def test_count_migration_hash_covers_count_and_grounding_fields():
    frozen, profiles = _migration_fixture()
    migrated = migrate_frozen_environment_count_provenance(frozen, profiles)
    changed_profiles = deepcopy(profiles)
    changed_profiles["profiles"]["health-condition"]["condition a"]["level_counts"][
        "specific class"
    ] = 8.0

    changed = migrate_frozen_environment_count_provenance(frozen, changed_profiles)

    assert changed["environment_hash"] != migrated["environment_hash"]
    assert changed["documents"]["aci/D1"]["environment_document_hash"] != migrated[
        "documents"
    ]["aci/D1"]["environment_document_hash"]


def test_migrated_environment_is_semantically_compatible_with_v16():
    comparison = compare_frozen_environment_semantics(
        json.loads(V16_ENVIRONMENT.read_text())["frozen_environment"],
        _current_environment()["frozen_environment"],
    )

    assert comparison["verdict"] == "count-only compatible", _summarize(
        comparison["differences"]
    )


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
