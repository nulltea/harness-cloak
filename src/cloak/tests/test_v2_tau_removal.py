from __future__ import annotations

import copy

from cloak.detect import Span
from cloak.substitute import substitute
from cloak.train.qa_audit import build_environment_audit
from cloak.train.qa_builder import (
    build_joint_representative_anchor,
    freeze_v2_environment_from_legacy_arms,
    render_frozen_action_vector,
)


def _legacy_inputs():
    source = "Aspirin treats arthritis. Arthritis remains active."
    first = source.index("arthritis")
    second = source.index("Arthritis")
    environment = {
        "tau": 0.02,
        "k_floors": {"drug": 100},
        "corpora": {"clinical": {"aci/D1": {"spans": [
            {
                "surface": "Aspirin", "type": "drug", "start": 0, "end": 7,
                "bc_action": 0,
                "actions": [
                    {"fill": "an analgesic", "mode": "level", "aset": 100,
                     "walk_risk": 0.01, "p6": 0.7},
                    {"fill": None, "mode": "placeholder", "walk_risk": 0.0,
                     "p6": 0.0},
                ],
            },
            {
                "surface": "arthritis", "type": "health-condition",
                "start": first, "end": first + len("arthritis"), "bc_action": 0,
                "actions": [
                    {"fill": "a joint disease", "mode": "level", "aset": 200,
                     "walk_risk": 0.01, "p6": 0.8},
                    {"fill": None, "mode": "placeholder", "walk_risk": 0.0,
                     "p6": 0.0},
                ],
            },
        ]}}},
    }
    rows = [
        {
            "surface": "Aspirin", "type": "drug", "start": 0, "end": 7,
            "score": 0.95, "action": "generalize", "replacement": "an analgesic",
            "risk": 0.01, "lattice": ["an analgesic"],
            "match": {"kind": "exact", "entry": "aspirin"},
            "profile_match": {"outcome": "exact", "reason": "exact_entry"},
        },
        {
            "surface": "arthritis", "type": "health-condition",
            "start": first, "end": first + len("arthritis"), "score": 0.94,
            "action": "generalize", "replacement": "a joint disease", "risk": 0.01,
            "lattice": ["a joint disease"],
            "match": {"kind": "exact", "entry": "arthritis"},
            "profile_match": {"outcome": "exact", "reason": "exact_entry"},
        },
        {
            "surface": "Arthritis", "type": "health-condition",
            "start": second, "end": second + len("Arthritis"), "score": 0.93,
            "action": "generalize", "replacement": "a joint disease", "risk": 0.01,
            "lattice": ["a joint disease"],
            "match": {"kind": "exact", "entry": "arthritis"},
            "profile_match": {"outcome": "exact", "reason": "exact_entry"},
        },
    ]
    arms = {
        "_meta": {"tau": 0.02, "detector": {"model": "test-detector"}},
        "clinical": {"aci/D1": {
            "tau_walk": [source, rows],
            "detector_diagnostics": {"accepted": [
                {"text": row["surface"], "type": row["type"],
                 "start": row["start"], "end": row["end"]}
                for row in rows
            ]},
        }},
    }
    return source, environment, arms


def _mutate_legacy_policy(environment, arms):
    changed_environment = copy.deepcopy(environment)
    changed_environment["tau"] = 0.9
    changed_environment["k_floors"] = {"drug": 999999}
    for span in changed_environment["corpora"]["clinical"]["aci/D1"]["spans"]:
        span["bc_action"] = 1
        for action in span["actions"]:
            action["walk_risk"] = 0.99
            action["p6"] = -123.0
    changed_arms = copy.deepcopy(arms)
    changed_arms["_meta"]["tau"] = 0.9
    for row in changed_arms["clinical"]["aci/D1"]["tau_walk"][1]:
        row.update({
            "action": "placeholder", "replacement": "<LEGACY_99>",
            "risk": 0.99, "exhausted": True,
        })
    return changed_environment, changed_arms


def _forbidden_keys(value):
    forbidden = {"tau", "tau_walk", "walk_risk", "p6", "k_floors", "bc_action", "exhausted"}
    found = set()
    if isinstance(value, dict):
        found.update(forbidden & set(value))
        for child in value.values():
            found.update(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys(child))
    return found


def test_v2_freeze_render_and_audit_are_invariant_to_legacy_tau_policy():
    source, environment, arms = _legacy_inputs()
    changed_environment, changed_arms = _mutate_legacy_policy(environment, arms)

    first = freeze_v2_environment_from_legacy_arms(
        environment, arms, source_documents={"aci/D1": source}
    )
    second = freeze_v2_environment_from_legacy_arms(
        changed_environment, changed_arms, source_documents={"aci/D1": source}
    )

    assert first == second
    assert _forbidden_keys(first) == set()
    document = first["documents"]["aci/D1"]
    assert all(decision["decision_id"].startswith("sha256:") for decision in document["decisions"])
    assert all(action["action_id"].startswith("sha256:")
               for decision in document["decisions"] for action in decision["actions"])
    assert all(any(action["mode"] == "placeholder" and action["placeholder_type"]
                   == decision["runtime_type"] for action in decision["actions"])
               for decision in document["decisions"])

    vector = {
        decision["decision_id"]: next(
            action["action_id"] for action in decision["actions"]
            if action["mode"] == "level"
        )
        for decision in document["decisions"]
    }
    expected = "An analgesic treats a joint disease. A joint disease remains active."
    assert render_frozen_action_vector(source, document, vector)[0] == expected
    assert build_environment_audit(first) == build_environment_audit(second)
    assert "privacy_policy_exhausted_profiled_span" not in {
        event["code"] for event in build_environment_audit(first)["events"]
    }


def test_legacy_substitute_still_honors_tau(monkeypatch):
    span = Span(0, 7, "Aspirin", "drug", 0.9, "test")
    monkeypatch.setattr("cloak.substitute.match_spans_batch", lambda items, **_kwargs: {})
    monkeypatch.setattr("cloak.substitute.lookup_entry", lambda *args: object())
    monkeypatch.setattr(
        "cloak.substitute.lattice_for",
        lambda *args, **kwargs: ["an analgesic", "a medication", "<DRUG_1>"],
    )
    monkeypatch.setattr(
        "cloak.substitute.walk_risk",
        lambda _sentence, _surface, fill, _type: {
            "an analgesic": 0.2,
            "a medication": 0.05,
        }[fill],
    )

    strict_doc, _ = substitute("Aspirin helps.", [span], tau=0.1)
    permissive_doc, _ = substitute("Aspirin helps.", [span], tau=0.3)

    assert strict_doc == "A medication helps."
    assert permissive_doc == "An analgesic helps."


def test_direct_identifiers_are_forced_placeholders_in_every_v2_render():
    source = "Alice sent CODE-7 aspirin."
    environment = {"corpora": {"clinical": {"d1": {"spans": [
        {"surface": "Alice", "type": "PERSON", "start": 0, "end": 5,
         "actions": [{"fill": None, "mode": "placeholder", "forced_placeholder": True}]},
        {"surface": "CODE-7", "type": "CODE", "start": 11, "end": 17,
         "actions": [{"fill": None, "mode": "placeholder", "forced_placeholder": True}]},
    ]}}}}
    arms = {"clinical": {"d1": {"v2_occurrences": [
        {"surface": "Alice", "type": "PERSON", "start": 0, "end": 5,
         "lattice": [], "forced_placeholder": True, "uncontrolled": False},
        {"surface": "CODE-7", "type": "CODE", "start": 11, "end": 17,
         "lattice": [], "forced_placeholder": True, "uncontrolled": False},
    ]}}}
    frozen = freeze_v2_environment_from_legacy_arms(
        environment, arms, source_documents={"d1": source},
    )
    document = frozen["documents"]["d1"]
    assert all(not decision["ranker_selectable"] for decision in document["decisions"])
    assert all([action["mode"] for action in decision["actions"]] == ["placeholder"]
               for decision in document["decisions"])
    vector = {
        decision["decision_id"]: decision["actions"][0]["action_id"]
        for decision in document["decisions"]
    }
    rendered, _ = render_frozen_action_vector(source, document, vector)
    assert rendered == "<PERSON_1> sent <CODE_1> aspirin."


def test_mixed_forced_identifier_and_linked_drug_anchor_renders_both_safely():
    source = "Alice takes aspirin."
    environment = {"corpora": {"clinical": {"d1": {"spans": [
        {"surface": "Alice", "type": "PERSON", "start": 0, "end": 5,
         "actions": [{"fill": None, "mode": "placeholder", "forced_placeholder": True}]},
        {"surface": "aspirin", "type": "drug", "start": 12, "end": 19,
         "actions": [{"fill": "medication", "mode": "level", "aset": 100},
                     {"fill": None, "mode": "placeholder"}]},
    ]}}}}
    arms = {"clinical": {"d1": {"v2_occurrences": [
        {"surface": "Alice", "type": "PERSON", "start": 0, "end": 5,
         "lattice": [], "forced_placeholder": True, "uncontrolled": False},
        {"surface": "aspirin", "type": "drug", "start": 12, "end": 19,
         "lattice": ["medication"], "uncontrolled": False},
    ]}}}
    frozen = freeze_v2_environment_from_legacy_arms(
        environment, arms, source_documents={"d1": source},
    )
    document = frozen["documents"]["d1"]
    drug = next(row for row in document["decisions"] if row["runtime_type"] == "drug")
    anchor = build_joint_representative_anchor(
        {"decision_requirements": {drug["decision_id"]: "medication"}},
        document["decisions"],
    )
    rendered, _ = render_frozen_action_vector(source, document, anchor["action_vector"])
    assert rendered == "<PERSON_1> takes medication."
