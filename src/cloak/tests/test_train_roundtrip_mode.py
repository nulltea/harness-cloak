"""Round-trip trainer mode — offline: roundtrip_batch monkeypatched with a deterministic
fake that rewards keeping level fills (so RLOO has a real gradient direction)."""
import json
import math
import sys
from pathlib import Path

import pytest
import torch

import cloak.extract as extract
import cloak.train.roundtrip as rt
from cloak.train.qa_builder import (
    AciTaskAdapter,
    _stable_hash,
    build_utility_artifact,
    builder_pin,
    context_reader_pin,
    package_utility_artifact,
    read_context_batch,
    relation_teacher_pin,
    utility_scorer_pin,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import train_ranker as tr  # noqa: E402


_COST_BUDGETS = {
    "base": {
        "remote_round_trips_per_rollout": 1,
        "context_reader_batches_per_rollout": 1,
    },
    "counterfactual": {
        "remote_round_trips_per_selected_pair": 1,
        "context_reader_batches_per_selected_pair": 1,
    },
}


def _sealed_utility_artifact(assertion=None):
    assertion = dict(assertion or {
        "assertion_id": "a1", "doc_id": "d0", "family": "delivered",
        "scope": "global", "occurrence_ids": [], "weight": 1.0,
        "scoring_contract": {"kind": "contains", "value": "documented fact"},
    })
    assertion.setdefault("family", "delivered")
    assertion.setdefault("group_id", "default")
    assertion.setdefault("weight", 1.0)
    if assertion["family"] == "delivered":
        assertion.setdefault(
            "scoring_contract", {"kind": "contains", "value": "documented fact"},
        )
    family = assertion["family"]
    family_budgets = {"context": 0.5, "delivered": 0.5}
    missing_family = "context" if family == "delivered" else "delivered"
    assertion["weight"] = family_budgets[family]
    threshold_manifest = {
        "family_budgets": family_budgets,
        "min_context_assertions": 0,
        "reader_threshold": 1.0,
        "reader_stability_repetitions": 1,
        "reader_option_permutations": 1,
        "reader_stability_threshold": 1.0,
        "cost_budgets": _COST_BUDGETS,
    }
    artifact = {
        "artifact_version": "utility-assertions-v1",
        "environment_hash": "env-v1",
        "task_pin": AciTaskAdapter.task_pin,
        "builder_pin": builder_pin(),
        "teacher_pin": relation_teacher_pin(False),
        "reader_pin": context_reader_pin(),
        "scorer_pin": utility_scorer_pin(),
        "gate_manifest_hash": _stable_hash(threshold_manifest),
        "threshold_manifest_pin": {
            "schema": "qa-threshold-manifest-v1",
            "sha256": _stable_hash(threshold_manifest),
        },
        "threshold_manifest": threshold_manifest,
        "cost_budgets": _COST_BUDGETS,
        "family_budgets": family_budgets,
        "documents": {"d0": {"utility_weight_denominator": 1.0,
                                "measurement_state": "partial",
                                "present_family_budgets": [family],
                                "missing_family_budgets": [missing_family],
                                "weight_groups": {
                                    family: {assertion["group_id"]: {
                                        "assertion_ids": ["a1"],
                                        "weight": family_budgets[family],
                                    }},
                                },
                                "assertion_ids": ["a1"],
                                "controlled_decision_ids": ["dec1"],
                                "occurrence_to_decision": {"o1": "dec1"},
                                "uncovered_decision_ids": (
                                    [] if assertion.get("scope") == "linked" else ["dec1"]
                                )}},
        "assertions": {"a1": assertion},
    }
    artifact["artifact_hash"] = _stable_hash(artifact)
    return artifact


def _accepted_context_assertion(status="accepted"):
    vector = {"dec1": "general"}
    return {
        "assertion_id": "a1", "doc_id": "d0", "family": "context",
        "scope": "linked", "occurrence_ids": ["o1"], "weight": 1.0,
        "status": status,
        "expected_action_support": {
            "joint_anchor_action_vector": vector,
            "joint_anchor_hash": _stable_hash(vector),
            "property_level": {"dec1": "endocrine"},
        },
        "evidence": {"validation": {
            "verdict": "accepted",
            "scores": {"original": 1.0, "representative": 1.0, "placeholder": 0.0},
            "stability": {"threshold": 1.0, "passing_fraction": 1.0},
        }},
    }


def _seal_artifact(artifact):
    if "threshold_manifest" in artifact:
        manifest_hash = _stable_hash(artifact["threshold_manifest"])
        artifact["gate_manifest_hash"] = manifest_hash
        artifact["threshold_manifest_pin"] = {
            "schema": "qa-threshold-manifest-v1",
            "sha256": manifest_hash,
        }
    artifact["artifact_hash"] = _stable_hash({
        key: value for key, value in artifact.items() if key != "artifact_hash"
    })
    return artifact


def _frozen_anchor_environment():
    return {
        "environment_hash": "env-v1",
        "documents": {"d0": {
            "occurrences": [{"occurrence_id": "o1", "decision_id": "dec1"}],
            "decisions": [
                {
                    "decision_id": "dec1", "controlled": True,
                    "runtime_type": "QUANTITY", "canonical_key": "metformin",
                    "actions": [
                        {"action_id": "keep-1", "mode": "keep", "legal": True,
                         "entails": ["exact"]},
                        {"action_id": "fine-1", "mode": "level", "legal": True,
                         "entails": ["endocrine"]},
                        {"action_id": "general-1", "mode": "level", "legal": True,
                         "entails": ["endocrine"]},
                        {"action_id": "placeholder-1", "mode": "placeholder", "legal": True,
                         "entails": []},
                    ],
                },
                {
                    "decision_id": "dec2", "controlled": True,
                    "runtime_type": "DRUG", "canonical_key": "synthroid",
                    "actions": [
                        {"action_id": "keep-2", "mode": "keep", "legal": True,
                         "entails": ["exact"]},
                        {"action_id": "general-2", "mode": "level", "legal": True,
                         "entails": ["other"]},
                        {"action_id": "placeholder-2", "mode": "placeholder", "legal": True,
                         "entails": []},
                    ],
                },
            ],
        }},
    }


def _verified_context_artifact():
    vector = {"dec1": "general-1", "dec2": "keep-2"}
    scores = {"original": 1.0, "representative": 1.0, "placeholder": 0.0}
    context = {
        "assertion_id": "a-context", "doc_id": "d0", "family": "context",
        "scope": "linked", "occurrence_ids": ["o1"], "group_id": "condition:one",
        "weight": 0.6, "status": "accepted",
        "expected_action_support": {
            "joint_anchor_action_vector": vector,
            "joint_anchor_hash": _stable_hash(vector),
            "property_level": {"dec1": "endocrine"},
        },
        "evidence": {"validation": {
            "verdict": "accepted", "scores": scores,
            "stability": {
                "repetitions": 1, "option_permutations": 1, "threshold": 1.0,
                "passing_fraction": 1.0,
                "trials": [{"repetition": 0, "permutation_index": 0,
                            "scores": scores, "passed": True}],
            },
        }},
    }
    delivered = {
        "assertion_id": "a-delivered", "doc_id": "d0", "family": "delivered",
        "scope": "global", "occurrence_ids": [], "group_id": "schema:one", "weight": 0.4,
        "scoring_contract": {"kind": "contains", "value": "delivered fact"},
    }
    threshold_manifest = {
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "min_context_assertions": 0,
        "reader_threshold": 1.0,
        "reader_stability_repetitions": 1,
        "reader_option_permutations": 1,
        "reader_stability_threshold": 1.0,
        "cost_budgets": _COST_BUDGETS,
    }
    return _seal_artifact({
        "artifact_version": "utility-assertions-v1", "environment_hash": "env-v1",
        "task_pin": AciTaskAdapter.task_pin, "builder_pin": builder_pin(),
        "teacher_pin": relation_teacher_pin(False),
        "reader_pin": context_reader_pin(),
        "scorer_pin": utility_scorer_pin(),
        "gate_manifest_hash": _stable_hash(threshold_manifest),
        "threshold_manifest_pin": {
            "schema": "qa-threshold-manifest-v1",
            "sha256": _stable_hash(threshold_manifest),
        },
        "threshold_manifest": threshold_manifest,
        "cost_budgets": _COST_BUDGETS,
        "family_budgets": {"context": 0.6, "delivered": 0.4},
        "documents": {"d0": {
            "utility_weight_denominator": 1.0,
            "measurement_state": "measured",
            "present_family_budgets": ["context", "delivered"],
            "missing_family_budgets": [],
            "weight_groups": {
                "context": {"condition:one": {
                    "assertion_ids": ["a-context"], "weight": 0.6,
                }},
                "delivered": {"schema:one": {
                    "assertion_ids": ["a-delivered"], "weight": 0.4,
                }},
            },
            "assertion_ids": ["a-context", "a-delivered"],
            "controlled_decision_ids": ["dec1", "dec2"],
            "occurrence_to_decision": {"o1": "dec1"},
            "decision_keys": [
                {"decision_id": "dec1", "runtime_type": "QUANTITY",
                 "canonical_key": "metformin"},
                {"decision_id": "dec2", "runtime_type": "DRUG",
                 "canonical_key": "synthroid"},
            ],
            "uncovered_decision_ids": ["dec2"],
        }},
        "assertions": {"a-context": context, "a-delivered": delivered},
    })


def _doc():
    actions = [{"mode": "level", "fill": "a biguanide", "aset": 500.0, "p6": 0.8,
                "walk_risk": 0.0},
               {"mode": "placeholder", "fill": "<QUANTITY_1>", "p6": 0.0, "walk_risk": 0.0}]
    span = {"surface": "metformin", "type": "QUANTITY", "start": 0, "actions": actions}
    raw = [dict(span)]
    spans, feats = tr.derive_spans(raw, {"QUANTITY": 100.0}, "clinical", "cpu")
    return {"id": "d0", "corpus": "clinical", "text": "metformin daily",
            "R_walk": [{"surface": "metformin", "type": "QUANTITY", "action": "generalize",
                        "replacement": "a biguanide", "start": 0, "end": 9,
                        "lattice": ["a biguanide"]}],
            "raw_spans": raw, "spans": spans, "feats": feats,
            "probes_train": [{"surface": "metformin", "question": "What drug?"}]}


def test_inert_floors_make_all_actions_legal_but_bc_skips_keep_original():
    actions = [
        {"mode": "level", "fill": "a narrow clinical drug class", "aset": 17.0, "p6": 0.6,
         "walk_risk": 0.0},
        {"mode": "level", "fill": "a broad medication", "aset": 250.0, "p6": 0.4,
         "walk_risk": 0.0},
        {"mode": "level", "fill": "metformin", "keep": True, "aset": 1.0, "p6": 1.0,
         "walk_risk": 1.0},
        {"mode": "placeholder", "fill": "<DRUG_1>", "p6": 0.0, "walk_risk": 0.0},
    ]
    raw = [{"surface": "metformin", "type": "drug", "start": 0, "actions": actions}]

    spans, _feats = tr.derive_spans(raw, {"drug": 1.0, "OTHER": 1.0}, "clinical", "cpu")
    span = spans[0]

    assert span["legal"] == [0, 1, 2, 3]
    assert span["bc_action"] == 0
    assert not span["actions"][span["bc_action"]].get("keep")
    assert tr.floor_walk_choice(spans)["metformin"]["fill"] == "a narrow clinical drug class"


def test_artifact_cf_frac_is_rejected_before_training_initialization(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_ranker.py",
        "--reward", "roundtrip",
        "--utility-artifact", "utility.json",
        "--expected-utility-manifest-hash", "sha256:expected",
        "--cf-frac", "0.1",
    ])
    monkeypatch.setattr(
        tr.torch,
        "manual_seed",
        lambda seed: pytest.fail("training initialization must not run"),
    )

    with pytest.raises(SystemExit, match="--cf-frac.*--utility-artifact"):
        tr.main()


def test_utility_artifact_requires_roundtrip_before_training_initialization(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_ranker.py",
        "--reward", "surrogate",
        "--utility-artifact", "utility.json",
    ])
    monkeypatch.setattr(
        tr.torch,
        "manual_seed",
        lambda seed: pytest.fail("training initialization must not run"),
    )
    monkeypatch.setattr(
        tr,
        "roundtrip_batch",
        lambda *args, **kwargs: pytest.fail("roundtrip must not run"),
    )

    with pytest.raises(SystemExit, match="--utility-artifact.*--reward roundtrip"):
        tr.main()


def test_utility_artifact_requires_expected_manifest_hash_before_training_initialization(
    monkeypatch,
):
    monkeypatch.setattr(sys, "argv", [
        "train_ranker.py",
        "--reward", "roundtrip",
        "--utility-artifact", "utility.json",
    ])
    monkeypatch.setattr(
        tr.torch,
        "manual_seed",
        lambda seed: pytest.fail("training initialization must not run"),
    )

    with pytest.raises(SystemExit, match="expected.*manifest hash"):
        tr.main()


def _exit_doc():
    actions = [{"mode": "level", "fill": "a biguanide", "aset": 100.0, "p6": 0.8,
                "walk_risk": 0.0},
               {"mode": "level", "fill": "a medication", "aset": 200.0, "p6": 0.7,
                "walk_risk": 0.0},
               {"mode": "placeholder", "fill": "<QUANTITY_1>", "p6": 0.0,
                "walk_risk": 0.0}]
    span = {"surface": "metformin", "type": "QUANTITY", "start": 0, "actions": actions}
    raw = [dict(span)]
    spans, feats = tr.derive_spans(raw, {"QUANTITY": 100.0}, "clinical", "cpu")
    return {"id": "d0", "corpus": "clinical", "text": "metformin daily",
            "R_walk": [{"surface": "metformin", "type": "QUANTITY", "action": "generalize",
                        "replacement": "a biguanide", "start": 0, "end": 9,
                        "lattice": ["a biguanide", "a medication"]}],
            "raw_spans": raw, "spans": spans, "feats": feats,
            "probes_train": [{"surface": "metformin", "question": "What drug?"}]}


def _artifact_exit_doc():
    doc = _exit_doc()
    artifact = _sealed_utility_artifact()
    artifact["documents"]["d0"]["decision_keys"] = [{
        "decision_id": "dec1", "runtime_type": "QUANTITY", "canonical_key": "metformin",
    }]
    doc["utility_artifact"] = _seal_artifact(artifact)
    return doc


def fake_roundtrip(jobs, workers=1, reader_refresh=False):
    # reward 1.0 iff the level fill survived into doc_p, else 0.0
    return [{"out_p": "", "out_final": j["doc_p"], "f1s": [float("biguanide" in j["doc_p"])],
             "recall": float("biguanide" in j["doc_p"])} for j in jobs]


def _with_carrier(doc):
    doc = dict(doc)
    doc["ladder"] = [{"surface": "metformin", "rung": 0, "q": "What drug?",
                      "rungs": ["metformin", "a biguanide"]}]
    doc["decisions"] = [
        {"q": "Which route?", "options": ["endocrinology", "primary care"],
         "gold": "primary care", "span_ids": ["metformin"]},
    ]
    doc["out_hi"] = "ASSESSMENT: metformin - medication - active"
    doc["schema"] = True
    return doc


def _candidate_rollout(doc, span_rows, feats, policy, decision_ids=None):
    choice = {"metformin": span_rows[0]["actions"][1]}
    doc_p, R = tr.assemble(doc["text"], doc["R_walk"], span_rows, choice)
    return choice, [], 0.0, doc_p, R, []


def _rollout_sequence(action_indices):
    actions = iter(action_indices)

    def rollout(doc, span_rows, feats, policy, decision_ids=None):
        action_idx = next(actions)
        choice = {"metformin": span_rows[0]["actions"][action_idx]}
        doc_p, R = tr.assemble(doc["text"], doc["R_walk"], span_rows, choice)
        return choice, [], 0.0, doc_p, R, []

    return rollout


def _assert_serial_verify_calls(calls):
    assert len(calls) == 3
    assert calls[0]["workers"] == 9
    assert calls[0]["reader_refresh"] is False
    assert len(calls[0]["doc_ps"]) == 2
    assert all(call["workers"] == 1 for call in calls[1:])
    assert all(call["reader_refresh"] is True for call in calls[1:])
    assert sorted(call["doc_ps"][0] for call in calls[1:]) == [
        "A biguanide daily", "A medication daily"]


def test_exit_round_keeps_candidate_after_serial_reverification(monkeypatch):
    doc = _exit_doc()
    calls = []

    def fake_roundtrip_exit(jobs, workers=1, reader_refresh=False):
        calls.append({"doc_ps": [j["doc_p"] for j in jobs], "workers": workers,
                      "reader_refresh": reader_refresh})
        if not reader_refresh:
            return [{"recall": 0.2}, {"recall": 0.9}]
        return [{"recall": 0.8 if "medication" in jobs[0]["doc_p"].lower() else 0.4}]

    monkeypatch.setattr(tr, "sample_rollout", _candidate_rollout)
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_exit)

    winners, stats = tr.exit_round([doc], tr.RankerPolicy(), G=1, rt_workers=9, seed=0)

    assert winners == [(0, {"metformin": 1})]
    assert stats["n_candidates"] == 1
    assert stats["n_winners"] == 1
    assert stats["n_verify_dropped"] == 0
    _assert_serial_verify_calls(calls)


def test_exit_round_drops_candidate_when_clean_baseline_catches_up(monkeypatch):
    doc = _exit_doc()
    calls = []

    def fake_roundtrip_exit(jobs, workers=1, reader_refresh=False):
        calls.append({"doc_ps": [j["doc_p"] for j in jobs], "workers": workers,
                      "reader_refresh": reader_refresh})
        if not reader_refresh:
            return [{"recall": 0.2}, {"recall": 0.9}]
        return [{"recall": 0.3 if "medication" in jobs[0]["doc_p"].lower() else 0.4}]

    monkeypatch.setattr(tr, "sample_rollout", _candidate_rollout)
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_exit)

    winners, stats = tr.exit_round([doc], tr.RankerPolicy(), G=1, rt_workers=9, seed=0)

    assert winners == []
    assert stats["n_candidates"] == 1
    assert stats["n_winners"] == 0
    assert stats["n_verify_dropped"] == 1
    _assert_serial_verify_calls(calls)


def test_exit_round_uses_verification_as_tiebreak_and_keeps_one_winner(monkeypatch):
    doc = _exit_doc()
    calls = []

    def fake_roundtrip_exit(jobs, workers=1, reader_refresh=False):
        calls.append({"doc_ps": [j["doc_p"] for j in jobs], "workers": workers,
                      "reader_refresh": reader_refresh})
        if not reader_refresh:
            return [{"recall": 0.2}, {"recall": 0.9}, {"recall": 0.9}]
        doc_p = jobs[0]["doc_p"].lower()
        if "biguanide" in doc_p:
            return [{"recall": 0.4}]
        if "medication" in doc_p:
            return [{"recall": 0.8}]
        return [{"recall": 0.3}]

    monkeypatch.setattr(tr, "sample_rollout", _rollout_sequence([2, 1]))
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_exit)

    winners, stats = tr.exit_round([doc], tr.RankerPolicy(), G=2, rt_workers=9, seed=0)

    assert winners == [(0, {"metformin": 1})]
    assert stats["n_candidates"] == 1
    assert stats["n_winners"] == 1
    assert stats["n_verify_dropped"] == 0
    assert stats["n_candidates"] == stats["n_winners"] + stats["n_verify_dropped"]
    assert len(calls) == 4
    assert calls[0]["workers"] == 9
    assert calls[0]["reader_refresh"] is False
    assert len(calls[0]["doc_ps"]) == 3
    assert all(call["workers"] == 1 for call in calls[1:])
    assert all(call["reader_refresh"] is True for call in calls[1:])
    clean_doc_ps = [call["doc_ps"][0].lower() for call in calls[1:]]
    assert clean_doc_ps == ["<quantity_1> daily", "a biguanide daily",
                            "a medication daily"]
    assert sum("biguanide" in doc_p for doc_p in clean_doc_ps) == 1


def test_exit_round_artifact_cache_coalesces_duplicate_floor_and_sample_jobs(monkeypatch, tmp_path):
    doc = _artifact_exit_doc()
    calls = []

    def fake_roundtrip_exit(jobs, workers=1, reader_refresh=False):
        calls.append({"jobs": list(jobs), "workers": workers, "reader_refresh": reader_refresh})
        return [{"recall": 0.2, "out_p": "remote", "out_final": job["doc_p"],
                 "component_scores": {"a1": 0.4}} for job in jobs]

    cache = tr.UtilityRewardCache(tmp_path / "utility-cache.json")
    monkeypatch.setattr(tr, "sample_rollout", _rollout_sequence([0, 0]))
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_exit)

    winners, stats = tr.exit_round(
        [doc], tr.RankerPolicy(), G=2, rt_workers=9, seed=0, utility_reward_cache=cache,
    )

    assert winners == []
    assert stats["qa_cache_hits"] == 2
    assert stats["qa_cache_misses"] == 1
    assert len(calls) == 1
    assert calls[0]["workers"] == 9
    assert calls[0]["reader_refresh"] is False
    assert len(calls[0]["jobs"]) == 1


def test_exit_round_artifact_cache_reuses_persisted_full_results(monkeypatch, tmp_path):
    doc = _artifact_exit_doc()
    cache_path = tmp_path / "utility-cache.json"
    calls = []

    def fake_roundtrip_exit(jobs, workers=1, reader_refresh=False):
        calls.append({"jobs": list(jobs), "workers": workers, "reader_refresh": reader_refresh})
        results = []
        for job in jobs:
            candidate = "medication" in job["doc_p"].lower()
            results.append({
                "recall": 0.45 if candidate else 0.1,
                "out_p": "remote-candidate" if candidate else "remote-floor",
                "out_final": job["doc_p"],
                "component_scores": {"a1": 0.9 if candidate else 0.2},
            })
        if reader_refresh:
            return [{"recall": 0.8 if "medication" in job["doc_p"].lower() else 0.4}
                    for job in jobs]
        return results

    monkeypatch.setattr(tr, "sample_rollout", _candidate_rollout)
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_exit)

    first_cache = tr.UtilityRewardCache(cache_path)
    first_winners, first_stats = tr.exit_round(
        [doc], tr.RankerPolicy(), G=1, rt_workers=9, seed=0,
        utility_reward_cache=first_cache,
    )

    assert first_winners == [(0, {"metformin": 1})]
    assert first_stats["qa_cache_hits"] == 0
    assert first_stats["qa_cache_misses"] == 2
    assert len(calls) == 3
    payload = [json.loads(line) for line in cache_path.read_text().splitlines()]
    assert {record["entry"]["result"]["out_p"] for record in payload} == {
        "remote-floor", "remote-candidate",
    }

    calls.clear()
    second_cache = tr.UtilityRewardCache(cache_path)
    second_winners, second_stats = tr.exit_round(
        [doc], tr.RankerPolicy(), G=1, rt_workers=9, seed=0,
        utility_reward_cache=second_cache,
    )

    assert second_winners == first_winners
    assert second_stats["qa_cache_hits"] == 2
    assert second_stats["qa_cache_misses"] == 0
    assert len(calls) == 2
    assert all(call["workers"] == 1 and call["reader_refresh"] for call in calls)


def test_exit_round_legacy_documents_bypass_utility_reward_cache(monkeypatch, tmp_path):
    doc = _exit_doc()
    calls = []

    def fake_roundtrip_exit(jobs, workers=1, reader_refresh=False):
        calls.append({"jobs": list(jobs), "workers": workers, "reader_refresh": reader_refresh})
        if reader_refresh:
            return [{"recall": 0.8 if "medication" in job["doc_p"].lower() else 0.4}
                    for job in jobs]
        return [{"recall": 0.9 if "medication" in job["doc_p"].lower() else 0.2}
                for job in jobs]

    cache = tr.UtilityRewardCache(tmp_path / "utility-cache.json")
    monkeypatch.setattr(tr, "sample_rollout", _candidate_rollout)
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_exit)

    winners, stats = tr.exit_round(
        [doc], tr.RankerPolicy(), G=1, rt_workers=9, seed=0, utility_reward_cache=cache,
    )

    assert winners == [(0, {"metformin": 1})]
    assert stats["qa_cache_hits"] == 0
    assert stats["qa_cache_misses"] == 0
    assert cache.entries == {}
    _assert_serial_verify_calls([
        {"doc_ps": [job["doc_p"] for job in call["jobs"]], **{
            key: call[key] for key in ("workers", "reader_refresh")
        }} for call in calls
    ])


def test_sample_rollout_shapes():
    doc = _doc()
    policy = tr.RankerPolicy()
    choice, logps, ph_rate, doc_p, R, legals = tr.sample_rollout(doc, doc["spans"],
                                                                 doc["feats"], policy)
    assert set(choice) == {"metformin"} and len(logps) == 1
    assert isinstance(doc_p, str) and isinstance(R, list)
    assert len(legals) == 1 and legals[0] == doc["spans"][0]["legal"]


def test_rloo_advantage_no_std():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    adv = tr.rloo_advantage(r)
    # b_g = mean of others: adv_0 = 1 - 0 = 1.0; adv_j = 0 - 1/3
    assert torch.allclose(adv, torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3]))


def test_roundtrip_epoch_moves_policy(monkeypatch):
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    doc = _doc()
    torch.manual_seed(0)
    policy = tr.RankerPolicy()
    before = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"]).detach().clone()
    stats = tr.train_roundtrip([doc], policy, G=4, epochs=3, lr=0.05,
                               entropy_coef=0.01, kl_coef=0.0, ref=None,
                               rt_workers=1, seed=0)
    after = policy.log_probs(doc["feats"][0], doc["spans"][0]["legal"]).detach()
    assert not torch.allclose(before, after)          # first-smoke movement canary
    assert after[0] > before[0]                       # level action (rewarded) went UP
    assert "ties_skipped" in stats[-1]


def test_train_roundtrip_threads_optional_carrier_fields(monkeypatch):
    doc = _with_carrier(_doc())
    captured = []

    def fake_roundtrip_carrier(jobs, workers=1):
        captured.extend(jobs)
        return [{"recall": float(i % 2), "out_p": "", "out_final": "", "f1s": []}
                for i, _ in enumerate(jobs)]

    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_carrier)

    tr.train_roundtrip([doc], tr.RankerPolicy(), G=2, epochs=1, lr=0.01,
                       entropy_coef=0.0, kl_coef=0.0, ref=None,
                       rt_workers=1, seed=0)

    assert len(captured) == 2
    assert all(j["ladder"] == doc["ladder"] for j in captured)
    assert all(j["decisions"] == doc["decisions"] for j in captured)
    assert all(j["out_hi"] == doc["out_hi"] for j in captured)
    assert all(j["schema"] is True for j in captured)


def test_counterfactual_terms_threads_optional_carrier_fields(monkeypatch):
    doc = _with_carrier(_doc())
    choice = {"metformin": doc["spans"][0]["actions"][0]}
    logps = [torch.tensor(0.0, requires_grad=True)]
    captured = []

    def fake_roundtrip_carrier(jobs, workers=1):
        captured.extend(jobs)
        return [{"recall": 0.0, "out_p": "", "out_final": "", "f1s": []} for _ in jobs]

    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_carrier)

    term, n_cf = tr.counterfactual_terms(doc, tr.RankerPolicy(), choice, logps, 1.0,
                                         frac=1.0, rng=__import__("random").Random(0),
                                         rt_workers=1)

    assert n_cf == 1
    assert isinstance(term, torch.Tensor)
    assert captured[0]["ladder"] == doc["ladder"]
    assert captured[0]["decisions"] == doc["decisions"]
    assert captured[0]["out_hi"] == doc["out_hi"]
    assert captured[0]["schema"] is True


def test_counterfactual_terms_excludes_span_free_decisions(monkeypatch):
    doc = _with_carrier(_doc())
    doc["decisions"] = [
        {"q": "Which route?", "options": ["endocrinology", "primary care"],
         "gold": "primary care", "span_ids": ["metformin"]},
        {"q": "Which billing path?", "options": ["routine", "complex"],
         "gold": "routine", "span_ids": []},
    ]
    choice = {"metformin": doc["spans"][0]["actions"][0]}
    logps = [torch.tensor(-0.7, requires_grad=True)]
    captured = []

    def fake_roundtrip_carrier(jobs, workers=1):
        captured.extend(jobs)
        out = []
        for job in jobs:
            has_span_free = any(not d.get("span_ids") for d in job.get("decisions", []))
            span_placeholdered = "<QUANTITY_1>" in job["doc_p"]
            reward = 0.0 if has_span_free and span_placeholdered else 1.0
            out.append({"recall": reward, "out_p": "", "out_final": "", "f1s": []})
        return out

    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip_carrier)

    term, n_cf = tr.counterfactual_terms(doc, tr.RankerPolicy(), choice, logps, base_r=999.0,
                                         frac=1.0, rng=__import__("random").Random(0),
                                         rt_workers=1)

    assert n_cf == 1
    assert term.item() == pytest.approx(0.0)
    assert len(captured) == 2
    assert all(job["decisions"] == [doc["decisions"][0]] for job in captured)


def test_utility_artifact_keeps_measured_documents_without_legacy_probe_threshold():
    docs = [_doc()]
    docs[0]["probes_train"] = []
    artifact = {
        "documents": {
            "d0": {
                "measurement_state": "partial",
                "assertion_ids": ["a1"],
                "utility_weight_denominator": 1.0,
            }
        },
        "assertions": {"a1": {"assertion_id": "a1", "doc_id": "d0",
                                "scope": "global", "occurrence_ids": []}},
    }

    attached = tr.attach_utility_artifact(docs, artifact)

    assert [doc["id"] for doc in attached] == ["d0"]
    assert attached[0]["utility_artifact"] is artifact
    assert "utility_decision_ids" not in attached[0]
    job = tr._roundtrip_job(attached[0], "generalized", [])
    assert job["doc_id"] == "d0"
    assert job["utility_artifact"] is artifact


@pytest.mark.parametrize(
    "environment_documents",
    [
        {},
        {"other": {}},
        {"d0": {}, "extra": {}},
    ],
)
def test_utility_artifact_gate_requires_exact_document_coverage(environment_documents):
    artifact = _sealed_utility_artifact()

    with pytest.raises(SystemExit, match="document coverage"):
        tr.enforce_utility_artifact_gate(
            artifact,
            {"environment_hash": "env-v1", "documents": environment_documents},
        )


def test_attach_utility_artifact_rejects_missing_document_state():
    with pytest.raises(ValueError, match="missing document state"):
        tr.attach_utility_artifact([_doc()], {"documents": {}, "assertions": {}})


@pytest.mark.parametrize("measurement_state", ["unsupported", "build_failed"])
def test_attach_utility_artifact_rejects_unusable_document_state(measurement_state):
    artifact = {
        "documents": {"d0": {
            "measurement_state": measurement_state,
            "assertion_ids": ["a1"],
        }},
        "assertions": {"a1": {"assertion_id": "a1"}},
    }

    with pytest.raises(ValueError, match="measurement_state"):
        tr.attach_utility_artifact([_doc()], artifact)


def _cache_binding(ids=("a1",), weights=None):
    weights = weights or {assertion_id: 1.0 for assertion_id in ids}
    return {
        "artifact_hash": "artifact-v1", "assertion_ids": list(ids), "weights": weights,
        "utility_weight_denominator": 1.0,
    }


def test_utility_reward_cache_reuses_full_result_and_reloads(tmp_path):
    cache_path = tmp_path / "utility-reward-cache.json"
    inputs = {
        "doc_id": "d0",
        "action_vector": {"dec1": "general-1"},
        "doc_p": "generalized document",
        "artifact_hash": "artifact-v1",
        "scorer_pin": {"reader": "reader-v1", "remote": "remote-v1"},
        "utility_binding": _cache_binding(("a1", "a2"), {"a1": 0.5, "a2": 0.5}),
    }
    result = {
        "out_p": "remote output",
        "out_final": "delivered output",
        "recall": 0.75,
        "component_scores": {"a1": 1.0, "a2": 0.5},
        "f1s": [],
    }
    cache = tr.UtilityRewardCache(cache_path)

    assert cache.lookup(**inputs) is None
    cache.store(**inputs, result=result)
    stored = cache.lookup(**inputs)
    assert stored == {
        **result,
        "result_version": "utility-roundtrip-result-v1",
        "status": "complete",
    }
    assert cache.hits == 1
    assert cache.misses == 1

    reloaded = tr.UtilityRewardCache(cache_path)
    assert reloaded.lookup(**inputs) == stored


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("out_p", "tampered remote output"),
        ("out_final", "tampered delivered output"),
        ("component_scores", {"a1": 0.0, "a2": 0.5}),
        ("recall", 0.25),
        ("status", "partial"),
        ("result_version", "other-version"),
    ],
)
def test_utility_reward_cache_fails_closed_on_result_tampering(tmp_path, field, tampered):
    cache_path = tmp_path / "utility-reward-cache.json"
    inputs = {
        "doc_id": "d0",
        "action_vector": {"dec1": "general-1"},
        "doc_p": "generalized document",
        "artifact_hash": "artifact-v1",
        "scorer_pin": {"reader": "reader-v1", "remote": "remote-v1"},
        "utility_binding": _cache_binding(("a1", "a2"), {"a1": 0.5, "a2": 0.5}),
    }
    cache = tr.UtilityRewardCache(cache_path)
    cache.store(**inputs, result={
        "out_p": "remote output",
        "out_final": "delivered output",
        "recall": 0.75,
        "component_scores": {"a1": 1.0, "a2": 0.5},
        "f1s": [],
    })
    payload = json.loads(cache_path.read_text())
    entry = payload["entry"]
    entry["result"][field] = tampered
    cache_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="invalid utility reward cache"):
        tr.UtilityRewardCache(cache_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("recall", math.nan),
        ("recall", -0.01),
        ("recall", 1.01),
        ("component_scores", {"a1": math.inf}),
        ("component_scores", {"a1": -0.01}),
        ("component_scores", {"a1": 1.01}),
    ],
)
def test_utility_reward_cache_rejects_invalid_scores(tmp_path, field, value):
    result = {
        "out_p": "remote output",
        "out_final": "delivered output",
        "recall": 0.75,
        "component_scores": {"a1": 1.0},
        "f1s": [],
    }
    result[field] = value
    cache = tr.UtilityRewardCache(tmp_path / "utility-reward-cache.json")

    with pytest.raises(ValueError, match="complete round-trip result"):
        cache.store(
            doc_id="d0",
            action_vector={"dec1": "general-1"},
            doc_p="generalized document",
            artifact_hash="artifact-v1",
            scorer_pin={"reader": "reader-v1"},
            utility_binding=_cache_binding(),
            result=result,
        )


def test_cached_utility_roundtrips_persists_once_per_dispatched_batch(tmp_path, monkeypatch):
    cache = tr.UtilityRewardCache(tmp_path / "utility-reward-cache.json")
    persist_calls = []
    original_persist = cache._persist

    def counted_persist(entries):
        persist_calls.append(True)
        original_persist(entries)

    def fake_roundtrip(jobs, workers=1):
        return [{
            "out_p": f"remote {index}",
            "out_final": f"delivered {index}",
            "recall": 0.5,
            "component_scores": {"a1": 0.5},
            "f1s": [],
        } for index, _job in enumerate(jobs)]

    monkeypatch.setattr(cache, "_persist", counted_persist)
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    inputs = [{
        "doc_id": f"d{index}",
        "action_vector": {"dec1": f"action-{index}"},
        "doc_p": f"document {index}",
        "artifact_hash": "artifact-v1",
        "scorer_pin": {"reader": "reader-v1"},
        "utility_binding": _cache_binding(),
    } for index in range(2)]

    results = tr.cached_utility_roundtrips(
        [{"doc_p": row["doc_p"]} for row in inputs], inputs, cache, workers=2
    )

    assert len(results) == 2
    assert persist_calls == [True]
    assert cache.hits == 0
    assert cache.misses == 2


def _complete_cache_result(label):
    return {
        "out_p": f"remote {label}",
        "out_final": f"delivered {label}",
        "recall": 0.5,
        "component_scores": {"a1": 0.5},
        "f1s": [],
    }


def _cache_inputs(label):
    return {
        "doc_id": f"doc-{label}",
        "action_vector": {"dec1": f"action-{label}"},
        "doc_p": f"document {label}",
        "artifact_hash": "artifact-v1",
        "scorer_pin": {"reader": "reader-v1"},
        "utility_binding": _cache_binding(),
    }


def test_utility_reward_cache_appends_one_jsonl_record_per_new_identity(tmp_path):
    cache_path = tmp_path / "utility-reward-cache.jsonl"
    cache = tr.UtilityRewardCache(cache_path)

    cache.store(**_cache_inputs("first"), result=_complete_cache_result("first"))
    first_record = cache_path.read_text()
    cache.store(**_cache_inputs("second"), result=_complete_cache_result("second"))
    lines = cache_path.read_text().splitlines()

    assert len(first_record.splitlines()) == 1
    assert len(lines) == 2
    assert all(json.loads(line)["version"] == tr.UtilityRewardCache._VERSION for line in lines)
    assert len(cache.entries) == 2


def test_utility_reward_cache_second_write_excludes_prior_history(tmp_path, monkeypatch):
    persisted_batches = []

    for name, history_size in (("small", 1), ("large", 25)):
        cache = tr.UtilityRewardCache(tmp_path / f"{name}.jsonl")
        cache.store_many([
            (_cache_inputs(f"history-{index:02d}"), _complete_cache_result(f"history-{index:02d}"))
            for index in range(history_size)
        ])
        original_persist = cache._persist

        def measured_persist(entries, *, _original=original_persist):
            persisted_batches.append(tuple(entries))
            return _original(entries)

        monkeypatch.setattr(cache, "_persist", measured_persist)
        cache.store(**_cache_inputs("new"), result=_complete_cache_result("new"))

    assert len(persisted_batches) == 2
    assert persisted_batches[0] == persisted_batches[1]
    assert len(persisted_batches[0]) == 1


def test_utility_reward_cache_rejects_truncated_jsonl_record(tmp_path):
    cache_path = tmp_path / "utility-reward-cache.jsonl"
    cache = tr.UtilityRewardCache(cache_path)
    cache.store(**_cache_inputs("first"), result=_complete_cache_result("first"))
    cache_path.write_text(cache_path.read_text().removesuffix("\n"))

    with pytest.raises(ValueError, match="truncated"):
        tr.UtilityRewardCache(cache_path)


def test_utility_reward_cache_rejects_conflicting_duplicate_identity(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    inputs = _cache_inputs("same")
    first = tr.UtilityRewardCache(first_path)
    second = tr.UtilityRewardCache(second_path)
    first.store(**inputs, result=_complete_cache_result("first"))
    second.store(**inputs, result=_complete_cache_result("second"))
    first_path.write_text(first_path.read_text() + second_path.read_text())

    with pytest.raises(ValueError, match="conflicting duplicate"):
        tr.UtilityRewardCache(first_path)


def test_utility_reward_cache_rejects_changed_reward_identity(tmp_path):
    cache = tr.UtilityRewardCache(tmp_path / "utility-reward-cache.json")
    inputs = {
        "doc_id": "d0",
        "action_vector": {"dec1": "general-1"},
        "doc_p": "generalized document",
        "artifact_hash": "artifact-v1",
        "scorer_pin": {"reader": "reader-v1", "remote": "remote-v1"},
        "utility_binding": _cache_binding(),
    }
    cache.store(**inputs, result={
        "out_p": "remote output", "out_final": "delivered output",
        "recall": 1.0, "component_scores": {"a1": 1.0}, "f1s": [],
    })

    variants = [
        {"doc_id": "d1"},
        {"action_vector": {"dec1": "placeholder-1"}},
        {"doc_p": "other document"},
        {"artifact_hash": "artifact-v2"},
        {"scorer_pin": {"reader": "reader-v2", "remote": "remote-v1"}},
    ]
    for variant in variants:
        assert cache.lookup(**(inputs | variant)) is None

    entry = next(iter(cache.entries.values()))
    entry["storage_identity"] = tr.utility_rollout_cache_identity(
        **inputs,
        out_final="other delivered output",
    )
    with pytest.raises(ValueError, match="storage identity"):
        cache.lookup(**inputs)


def test_utility_reward_cache_invalidates_task_prompt_or_invert_provenance(
    tmp_path, monkeypatch,
):
    doc = _artifact_exit_doc()
    cache = tr.UtilityRewardCache(tmp_path / "utility-reward-cache.jsonl")
    inputs = {
        "doc_id": doc["id"],
        "action_vector": {"dec1": "action-v1"},
        "doc_p": "generalized document",
        "artifact_hash": doc["utility_artifact"]["artifact_hash"],
        "scorer_pin": tr._utility_scorer_pin(doc),
        "utility_binding": tr.utility_cache_binding(doc["utility_artifact"], doc["id"]),
    }
    cache.store(**inputs, result={**_complete_cache_result("baseline"), "recall": 0.25})

    with monkeypatch.context() as context:
        context.setitem(rt.TASK_TEMPLATE, "clinical", "changed prompt\n{doc}")
        changed_prompt_pin = tr._utility_scorer_pin(doc)
    with monkeypatch.context() as context:
        context.setattr(
            extract,
            "INVERT_EXTRACTOR_VERSION",
            "invert-rule-cascade-test-change",
            raising=False,
        )
        changed_invert_pin = tr._utility_scorer_pin(doc)
    with monkeypatch.context() as context:
        context.setattr(
            extract,
            "_module_source_hash",
            lambda module: f"sha256:changed:{module.__name__}",
            raising=False,
        )
        changed_helper_pin = tr._utility_scorer_pin(doc)
    with monkeypatch.context() as context:
        context.setattr(
            extract,
            "SEMANTIC_MODEL_REVISION",
            "1111111111111111111111111111111111111111",
            raising=False,
        )
        changed_model_revision_pin = tr._utility_scorer_pin(doc)
    original_distribution_version = extract._distribution_version
    with monkeypatch.context() as context:
        context.setattr(
            extract,
            "_distribution_version",
            lambda name: "changed-torch" if name == "torch"
            else original_distribution_version(name),
            raising=False,
        )
        changed_dependency_pin = tr._utility_scorer_pin(doc)

    assert cache.lookup(**(inputs | {"scorer_pin": changed_prompt_pin})) is None
    assert cache.lookup(**(inputs | {"scorer_pin": changed_invert_pin})) is None
    assert changed_helper_pin != inputs["scorer_pin"]
    assert cache.lookup(**(inputs | {"scorer_pin": changed_helper_pin})) is None
    assert changed_model_revision_pin != inputs["scorer_pin"]
    assert cache.lookup(**(inputs | {"scorer_pin": changed_model_revision_pin})) is None
    assert changed_dependency_pin != inputs["scorer_pin"]
    assert cache.lookup(**(inputs | {"scorer_pin": changed_dependency_pin})) is None


def test_utility_reward_cache_rejects_previous_roundtrip_reward_pin(tmp_path):
    doc = _artifact_exit_doc()
    cache = tr.UtilityRewardCache(tmp_path / "utility-reward-cache.jsonl")
    current_pin = tr._utility_scorer_pin(doc)
    inputs = {
        "doc_id": doc["id"],
        "action_vector": {"dec1": "action-v1"},
        "doc_p": "generalized document",
        "artifact_hash": doc["utility_artifact"]["artifact_hash"],
        "scorer_pin": current_pin,
        "utility_binding": tr.utility_cache_binding(doc["utility_artifact"], doc["id"]),
    }
    cache.store(**inputs, result={**_complete_cache_result("current"), "recall": 0.25})
    previous_pin = {
        **current_pin,
        "reward_pin_version": "qa-builder-v2-roundtrip-reward-v3",
    }

    assert current_pin["reward_pin_version"] == "qa-builder-v2-roundtrip-reward-v4"
    assert cache.lookup(**(inputs | {"scorer_pin": previous_pin})) is None


def test_greedy_artifact_readout_reuses_persisted_reward_cache(tmp_path, monkeypatch):
    doc = _artifact_exit_doc()
    calls = []

    def deterministic_greedy(doc, span_rows, feats, policy, greedy=False, decision_ids=None):
        assert greedy is True
        choice = {"metformin": span_rows[0]["actions"][0]}
        doc_p, replacements = tr.assemble(doc["text"], doc["R_walk"], span_rows, choice)
        return choice, {}, 0.0, doc_p, replacements, []

    def fake_roundtrip(jobs, workers=1):
        calls.append(len(jobs))
        return [{
            "out_p": "remote output",
            "out_final": "delivered output",
            "recall": 0.5,
            "component_scores": {"a1": 1.0},
            "f1s": [],
        } for _ in jobs]

    monkeypatch.setattr(tr, "sample_rollout", deterministic_greedy)
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    cache_path = tmp_path / "utility-reward-cache.jsonl"

    first_results, _first_phs, first_accounting = tr.greedy_roundtrip_readout(
        [doc], tr.RankerPolicy(), rt_workers=2,
        utility_reward_cache=tr.UtilityRewardCache(cache_path),
    )
    second_results, _second_phs, second_accounting = tr.greedy_roundtrip_readout(
        [doc], tr.RankerPolicy(), rt_workers=2,
        utility_reward_cache=tr.UtilityRewardCache(cache_path),
    )

    assert first_results == second_results
    assert calls == [1]
    assert first_accounting == {"qa_cache_hits": 0, "qa_cache_misses": 1}
    assert second_accounting == {"qa_cache_hits": 1, "qa_cache_misses": 0}


def test_train_roundtrip_reuses_identical_artifact_rollout_cache(tmp_path, monkeypatch):
    doc = _doc()
    artifact = {
        "artifact_hash": "artifact-v1",
        "reader_pin": "reader-v1",
        "scorer_pin": utility_scorer_pin(),
        "documents": {"d0": {
            "utility_weight_denominator": 1.0,
            "controlled_decision_ids": ["dec1"],
            "occurrence_to_decision": {"o1": "dec1"},
            "decision_keys": [{
                "decision_id": "dec1", "runtime_type": "QUANTITY",
                "canonical_key": "metformin",
            }],
        }},
        "assertions": {"a1": {
            "assertion_id": "a1", "doc_id": "d0", "scope": "linked",
            "occurrence_ids": ["o1"], "weight": 1.0,
        }},
    }
    doc["utility_artifact"] = artifact
    batch_sizes = []

    def identical_rollout(doc, span_rows, feats, policy, greedy=False, decision_ids=None):
        action_index = 0
        choice = {"metformin": span_rows[0]["actions"][action_index]}
        log_prob = policy.log_probs(feats[0], [0, 1])[action_index]
        doc_p, replacements = tr.assemble(doc["text"], doc["R_walk"], span_rows, choice)
        return choice, {decision_ids[0]: log_prob}, 0.0, doc_p, replacements, [[0, 1]]

    def fake_roundtrip(jobs, workers=1):
        batch_sizes.append(len(jobs))
        return [{
            "out_p": "remote output", "out_final": "delivered output", "f1s": [],
            "recall": 1.0, "component_scores": {"a1": 1.0},
        } for _ in jobs]

    monkeypatch.setattr(tr, "sample_rollout", identical_rollout)
    monkeypatch.setattr(tr, "roundtrip_batch", fake_roundtrip)
    cache = tr.UtilityRewardCache(tmp_path / "utility-reward-cache.json")

    tr.train_roundtrip(
        [doc], tr.RankerPolicy(), G=2, epochs=1, lr=0.01,
        entropy_coef=0.0, kl_coef=0.0, ref=None, rt_workers=1, seed=0,
        utility_reward_cache=cache,
    )

    assert batch_sizes == [1]
    assert cache.hits == 1
    assert cache.misses == 1


def test_utility_artifact_gate_checks_environment_and_denominator():
    artifact = _sealed_utility_artifact()

    tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})

    with pytest.raises(SystemExit, match="environment_hash"):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "other"})


def test_utility_artifact_gate_recomputes_manifest_hash_and_checks_expected_identity():
    artifact = _sealed_utility_artifact()

    with pytest.raises(SystemExit, match="expected manifest hash"):
        tr.enforce_utility_artifact_gate(
            artifact,
            {"environment_hash": "env-v1"},
            expected_manifest_hash="sha256:expected",
        )

    artifact["gate_manifest_hash"] = "sha256:forged"
    _seal_artifact(artifact)
    artifact["gate_manifest_hash"] = "sha256:forged"
    artifact["artifact_hash"] = _stable_hash({
        key: value for key, value in artifact.items() if key != "artifact_hash"
    })
    with pytest.raises(SystemExit, match="gate_manifest_hash"):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})


def test_training_gate_rejects_under_floor_representative_action_from_frozen_environment():
    frozen = tr.freeze_ranker_environment({
        "k_floors": {"QUANTITY": 100.0, "OTHER": 100.0},
        "corpora": {"clinical": {"d0": {"spans": [{
            "surface": "metformin",
            "type": "QUANTITY",
            "start": 0,
            "end": 9,
            "actions": [
                {"fill": "an endocrine treatment", "mode": "level", "aset": 10.0},
                {"fill": "metformin", "mode": "level", "keep": True, "aset": 1.0},
                {"fill": None, "mode": "placeholder"},
            ],
        }]}}},
    })
    live_document = frozen["documents"]["d0"]
    decision = live_document["decisions"][0]
    occurrence = live_document["occurrences"][0]
    general = next(action for action in decision["actions"] if action["fill"])
    vector = {decision["decision_id"]: general["action_id"]}
    scores = {"original": 1.0, "representative": 1.0, "placeholder": 0.0}
    artifact = package_utility_artifact(
        frozen,
        {"d0": [{
            "assertion_id": "a-context",
            "family": "context",
            "scope": "linked",
            "subtype": "semantic_property",
            "occurrence_ids": [occurrence["occurrence_id"]],
            "group_id": "treatment:category",
            "weight": 0.5,
            "status": "accepted",
            "expected_action_support": {
                "joint_anchor_action_vector": vector,
                "joint_anchor_hash": _stable_hash(vector),
                "property_level": {
                    decision["decision_id"]: "an endocrine treatment",
                },
            },
            "evidence": {"validation": {
                "verdict": "accepted",
                "scores": scores,
                "stability": {
                    "repetitions": 1,
                    "option_permutations": 1,
                    "threshold": 1.0,
                    "passing_fraction": 1.0,
                    "trials": [{
                        "repetition": 0,
                        "permutation_index": 0,
                        "scores": scores,
                        "passed": True,
                    }],
                },
            }},
        }]},
        family_budgets={"context": 0.5, "delivered": 0.5},
        pins={
            "task_pin": AciTaskAdapter.task_pin,
            "builder_pin": builder_pin(),
            "teacher_pin": relation_teacher_pin(False),
            "reader_pin": context_reader_pin(),
            "scorer_pin": utility_scorer_pin(),
            "gate_manifest_hash": "temporary",
            "threshold_manifest_pin": {
                "schema": "qa-threshold-manifest-v1",
                "sha256": "temporary",
            },
        },
    )
    artifact["threshold_manifest"] = {
        "family_budgets": {"context": 0.5, "delivered": 0.5},
        "min_context_assertions": 0,
        "reader_threshold": 1.0,
        "reader_stability_repetitions": 1,
        "reader_option_permutations": 1,
        "reader_stability_threshold": 1.0,
        "cost_budgets": _COST_BUDGETS,
    }
    artifact["cost_budgets"] = _COST_BUDGETS
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="unknown frozen action"):
        tr.enforce_utility_artifact_gate(artifact, frozen)


def test_utility_artifact_gate_rejects_subset_environment_hash_mismatch():
    artifact = _sealed_utility_artifact()
    artifact["environment_hash"] = "subset-env"
    artifact["documents"]["d0"]["environment_document_hash"] = "doc-v1"
    artifact["artifact_hash"] = _stable_hash({
        key: value for key, value in artifact.items() if key != "artifact_hash"
    })
    full_environment = {
        "environment_hash": "full-env",
        "documents": {
            "d0": {"environment_document_hash": "doc-v1"},
            "d1": {"environment_document_hash": "doc-v2"},
        },
    }

    with pytest.raises(SystemExit, match="environment_hash"):
        tr.enforce_utility_artifact_gate(artifact, full_environment)


@pytest.mark.parametrize("measurement_state", ["unsupported", "build_failed"])
def test_utility_artifact_gate_rejects_unmeasured_documents(measurement_state):
    artifact = _sealed_utility_artifact()
    artifact["documents"]["d0"]["measurement_state"] = measurement_state
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="measurement_state"):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})


def test_utility_artifact_gate_rejects_unknown_measurement_state():
    artifact = _sealed_utility_artifact()
    artifact["documents"]["d0"]["measurement_state"] = "mystery"
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="measurement_state"):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})


def test_utility_artifact_gate_requires_frozen_cost_budgets():
    artifact = _sealed_utility_artifact()
    artifact["threshold_manifest"].pop("cost_budgets", None)
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="cost budgets"):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})


@pytest.mark.parametrize("family_budgets", [
    {"context": 1.0},
    {"context": 0.5, "delivered": 0.5, "unknown": 0.1},
    {"context": 0.0, "delivered": 1.0},
    {"context": float("nan"), "delivered": 1.0},
    {"context": float("inf"), "delivered": 1.0},
])
@pytest.mark.parametrize("entrypoint", [
    tr.enforce_utility_artifact_gate,
    tr.qa_utility_preflight_report,
])
def test_artifact_gate_and_preflight_require_exact_positive_family_budgets(
    family_budgets, entrypoint,
):
    artifact = _verified_context_artifact()
    artifact["family_budgets"] = family_budgets
    artifact["threshold_manifest"]["family_budgets"] = family_budgets
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="family budgets"):
        entrypoint(artifact, _frozen_anchor_environment())


def test_utility_artifact_gate_requires_weight_groups():
    artifact = _sealed_utility_artifact()
    artifact["documents"]["d0"].pop("weight_groups", None)
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="group weight metadata"):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})


def test_utility_artifact_gate_recomputes_uncovered_decisions():
    artifact = _verified_context_artifact()
    artifact["documents"]["d0"]["uncovered_decision_ids"] = []
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="uncovered decisions"):
        tr.enforce_utility_artifact_gate(artifact, _frozen_anchor_environment())


def test_utility_artifact_gate_uses_authoritative_controlled_decision_order():
    artifact = _verified_context_artifact()
    state = artifact["documents"]["d0"]
    state["controlled_decision_ids"] = ["dec2", "dec1"]
    state["uncovered_decision_ids"] = ["dec2"]
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="controlled decision IDs"):
        tr.enforce_utility_artifact_gate(artifact, _frozen_anchor_environment())


def test_utility_artifact_gate_rejects_live_reader_pin_mismatch():
    artifact = _sealed_utility_artifact()
    artifact["reader_pin"] = {"model": "wrong-reader"}
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="live reader pin"):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})


def test_utility_artifact_gate_rejects_injected_builder_dependencies():
    class InjectedTeacher:
        pass

    injected_reader_artifact = build_utility_artifact(
        {"environment_hash": "env-v1", "documents": {}},
        AciTaskAdapter({}),
        {},
        threshold_manifest={
            "family_budgets": {"context": 0.5, "delivered": 0.5},
            "cost_budgets": _COST_BUDGETS,
        },
        pins={},
        reader=lambda questions, context: [],
        render_action_vector=lambda doc_id, vector: "unused",
    )
    with pytest.raises(SystemExit, match="live reader pin"):
        tr.enforce_utility_artifact_gate(
            injected_reader_artifact, {"environment_hash": "env-v1"}
        )

    injected_teacher_artifact = build_utility_artifact(
        {"environment_hash": "env-v1", "documents": {}},
        AciTaskAdapter({}),
        {},
        threshold_manifest={
            "family_budgets": {"context": 0.5, "delivered": 0.5},
            "cost_budgets": _COST_BUDGETS,
        },
        pins={},
        reader=read_context_batch,
        render_action_vector=lambda doc_id, vector: "unused",
        relation_teacher=InjectedTeacher(),
    )
    with pytest.raises(SystemExit, match="teacher_pin"):
        tr.enforce_utility_artifact_gate(
            injected_teacher_artifact, {"environment_hash": "env-v1"}
        )


def test_utility_artifact_gate_rejects_forged_cross_scope_semantic_duplicate():
    artifact = _sealed_utility_artifact({
        "assertion_id": "a1", "doc_id": "d0", "family": "delivered",
        "scope": "linked", "occurrence_ids": ["o1"], "group_id": "content:dec1",
        "scoring_contract": {"kind": "contains", "value": "62-year-old"},
    })
    artifact["assertions"]["a-global"] = {
        "assertion_id": "a-global", "doc_id": "d0", "family": "delivered",
        "scope": "global", "occurrence_ids": [], "group_id": "demographic:age",
        "status": "accepted", "weight": 0.5,
        "scoring_contract": {"kind": "contains", "value": "62-year-old"},
    }
    artifact["documents"]["d0"]["assertion_ids"].append("a-global")
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="duplicate semantic utility fact"):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})


@pytest.mark.parametrize(("contract", "message"), [
    (None, "missing scoring_contract"),
    ({}, "empty scoring_contract"),
    ({"kind": "equals", "value": "documented fact"}, "unsupported scoring_contract"),
    ({"kind": "contains", "value": ""}, "non-empty string value"),
    ({"kind": "contains", "value": "  "}, "non-empty string value"),
])
def test_utility_artifact_gate_rejects_invalid_delivered_scoring_contract(
    contract, message,
):
    artifact = _sealed_utility_artifact()
    if contract is None:
        artifact["assertions"]["a1"].pop("scoring_contract")
    else:
        artifact["assertions"]["a1"]["scoring_contract"] = contract
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match=message):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})


@pytest.mark.parametrize(("pin_name", "previous_pin", "message"), [
    (
        "builder_pin",
        {"builder": "qa-builder-v2", "version": "assertion-compiler-v3"},
        "live builder",
    ),
    (
        "scorer_pin",
        {**utility_scorer_pin(), "pin_version": "qa-utility-scorer-v2"},
        "live scorer",
    ),
])
def test_utility_artifact_gate_rejects_previous_semantic_pins(
    pin_name, previous_pin, message
):
    artifact = _sealed_utility_artifact()
    artifact[pin_name] = previous_pin
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match=message):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})


@pytest.mark.parametrize(("field", "live_value", "message"), [
    ("source_hash", "sha256:changed-source", "source hash"),
    (
        "authoritative_reference_hash",
        "sha256:changed-reference",
        "authoritative reference hash",
    ),
])
def test_utility_artifact_gate_rejects_source_or_reference_mismatch(
    field, live_value, message
):
    artifact = _sealed_utility_artifact()
    artifact["documents"]["d0"].update({
        "source_hash": "sha256:source",
        "authoritative_reference_hash": "sha256:reference",
    })
    _seal_artifact(artifact)
    live_document = {
        "source_hash": "sha256:source",
        "authoritative_reference_hash": "sha256:reference",
    }
    live_document[field] = live_value

    with pytest.raises(SystemExit, match=message):
        tr.enforce_utility_artifact_gate(
            artifact,
            {"environment_hash": "env-v1", "documents": {"d0": live_document}},
        )


def test_frozen_training_environment_binds_loaded_source_and_gold_reference():
    environment = {"corpora": {"clinical": {"d0": {"spans": []}}}}
    arms = {"clinical": {"d0": {"tau_walk": ["source", []]}}}
    docs = [{
        "id": "d0",
        "corpus": "clinical",
        "text": "source document",
        "gold_ref": "gold reference",
    }]

    frozen = tr.frozen_training_environment(environment, arms, docs)

    assert frozen["documents"]["d0"]["source_hash"] == _stable_hash("source document")
    assert frozen["documents"]["d0"]["authoritative_reference_hash"] == _stable_hash(
        "gold reference"
    )


def test_qa_preflight_recomputes_counts_and_enforces_call_budgets():
    artifact = _verified_context_artifact()
    state = artifact["documents"]["d0"]
    state["accepted_assertion_count"] = 999
    state["context_reader_batches_per_rollout"] = 0
    _seal_artifact(artifact)

    report = tr.qa_utility_preflight_report(artifact, _frozen_anchor_environment())

    assert report["documents"]["d0"]["accepted_assertion_count"] == 2
    assert report["documents"]["d0"]["context_reader_batches_per_rollout"] == 1

    artifact["threshold_manifest"]["cost_budgets"]["base"][
        "context_reader_batches_per_rollout"
    ] = 0
    _seal_artifact(artifact)
    with pytest.raises(SystemExit, match="cost budget"):
        tr.qa_utility_preflight_report(artifact, _frozen_anchor_environment())


def test_utility_artifact_gate_rejects_document_without_accepted_assertions():
    artifact = _sealed_utility_artifact()
    artifact["documents"]["d0"]["assertion_ids"] = []
    artifact["assertions"] = {}
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="no accepted assertions"):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})


def test_utility_rollout_cache_identity_covers_complete_reward_inputs():
    baseline = tr.utility_rollout_cache_identity(
        doc_id="d0",
        action_vector={"dec1": "general-1"},
        doc_p="generalized document",
        out_final="delivered output",
        artifact_hash="artifact-v1",
        scorer_pin={"reader": "reader-v1", "extractor": "invert-v1"},
    )
    variants = [
        {"doc_id": "d1"},
        {"action_vector": {"dec1": "placeholder-1"}},
        {"doc_p": "different generalized document"},
        {"out_final": "different delivered output"},
        {"artifact_hash": "artifact-v2"},
        {"scorer_pin": {"reader": "reader-v2", "extractor": "invert-v1"}},
    ]
    inputs = {
        "doc_id": "d0",
        "action_vector": {"dec1": "general-1"},
        "doc_p": "generalized document",
        "out_final": "delivered output",
        "artifact_hash": "artifact-v1",
        "scorer_pin": {"reader": "reader-v1", "extractor": "invert-v1"},
    }

    assert baseline.startswith("sha256:")
    for variant in variants:
        assert tr.utility_rollout_cache_identity(**(inputs | variant)) != baseline


def test_train_roundtrip_uses_structured_credit_when_scalar_utility_is_tied(monkeypatch):
    doc = _doc()
    artifact = {
        "documents": {"d0": {
            "utility_weight_denominator": 1.0,
            "controlled_decision_ids": ["dec1"],
            "occurrence_to_decision": {"o1": "dec1"},
            "decision_keys": [{
                "decision_id": "dec1", "runtime_type": "QUANTITY",
                "canonical_key": "metformin",
            }],
        }},
        "assertions": {"a1": {
            "assertion_id": "a1", "doc_id": "d0", "scope": "linked",
            "occurrence_ids": ["o1"], "weight": 1.0,
        }},
    }
    doc["utility_artifact"] = artifact
    actions = iter([0, 1])

    def sample(doc, span_rows, feats, policy, greedy=False, decision_ids=None):
        assert decision_ids == ["dec1"]
        action_index = next(actions)
        choice = {"metformin": span_rows[0]["actions"][action_index]}
        log_prob = policy.log_probs(feats[0], [0, 1])[action_index]
        doc_p, replacements = tr.assemble(doc["text"], doc["R_walk"], span_rows, choice)
        return (choice, {decision_ids[0]: log_prob}, float(action_index == 1), doc_p,
                replacements, [[0, 1]])

    def tied_scalar_roundtrip(jobs, workers=1):
        return [{
            "recall": 0.5,
            "component_scores": {"a1": float("biguanide" in job["doc_p"].lower())},
            "out_p": "", "out_final": "", "f1s": [],
        } for job in jobs]

    monkeypatch.setattr(tr, "sample_rollout", sample)
    monkeypatch.setattr(tr, "roundtrip_batch", tied_scalar_roundtrip)

    policy = tr.RankerPolicy()
    before = policy.log_probs(doc["feats"][0], [0, 1]).detach().clone()
    rows = tr.train_roundtrip(
        [doc], policy, G=2, epochs=1, lr=0.01,
        entropy_coef=0.0, kl_coef=0.0, ref=None,
        rt_workers=1, seed=0,
    )
    after = policy.log_probs(doc["feats"][0], [0, 1]).detach()

    assert rows[0]["ties_skipped"] == 0
    assert not torch.allclose(before, after)


def test_structured_utility_loss_replaces_tested_pair_at_group_weight():
    log_probs = [
        {"dec1": torch.tensor(2.0, requires_grad=True)},
        {"dec1": torch.tensor(3.0, requires_grad=True)},
    ]
    provisional = {(0, "dec1"): 4.0, (1, "dec1"): -5.0}
    counterfactual = {("d0", 0, "dec1"): torch.tensor(7.0, requires_grad=True)}

    loss = tr.structured_utility_loss(
        "d0", log_probs, provisional, counterfactual_losses=counterfactual
    )

    assert loss.item() == pytest.approx(11.0)


def test_utility_decision_ids_ignore_artifact_decision_order():
    doc = {"id": "d0", "utility_artifact": {"documents": {"d0": {
        "controlled_decision_ids": ["dec2", "dec1"],
        "decision_keys": [
            {"decision_id": "dec1", "runtime_type": "QUANTITY",
             "canonical_key": "metformin"},
            {"decision_id": "dec2", "runtime_type": "DRUG",
             "canonical_key": "synthroid"},
        ],
    }}}}
    spans = [
        {"type": "QUANTITY", "surface": "metformin"},
        {"type": "DRUG", "surface": "Synthroid"},
    ]

    assert tr.utility_decision_ids(doc, spans) == ["dec1", "dec2"]


def test_structured_utility_loss_attaches_gradients_by_decision_id():
    first_a = torch.tensor(2.0, requires_grad=True)
    first_b = torch.tensor(3.0, requires_grad=True)
    second_a = torch.tensor(5.0, requires_grad=True)
    second_b = torch.tensor(7.0, requires_grad=True)

    loss = tr.structured_utility_loss(
        "d0",
        [{"dec2": first_b, "dec1": first_a}, {"dec2": second_b, "dec1": second_a}],
        {(0, "dec1"): 1.0, (0, "dec2"): 0.0,
         (1, "dec1"): -1.0, (1, "dec2"): 0.0},
    )
    loss.backward()

    assert first_a.grad.item() == pytest.approx(-0.5)
    assert second_a.grad.item() == pytest.approx(0.5)
    assert first_b.grad.item() == pytest.approx(0.0)
    assert second_b.grad.item() == pytest.approx(0.0)


def test_utility_artifact_gate_rejects_uncontrolled_occurrence_link():
    artifact = _verified_context_artifact()
    artifact["assertions"]["a-context"]["occurrence_ids"] = ["o-uncontrolled"]
    _seal_artifact(artifact)
    environment = _frozen_anchor_environment()
    environment["documents"]["d0"]["occurrences"].append({
        "occurrence_id": "o-uncontrolled", "decision_id": None, "controlled": False,
    })

    with pytest.raises(SystemExit, match="uncontrolled occurrence"):
        tr.enforce_utility_artifact_gate(artifact, environment)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda state: state.update({"controlled_decision_ids": ["dec1"]}),
         "controlled decision IDs"),
        (lambda state: state.update({"occurrence_to_decision": {"o1": "dec2"}}),
         "occurrence-to-decision"),
    ],
)
def test_utility_artifact_gate_requires_exact_frozen_decision_identity(mutate, message):
    artifact = _verified_context_artifact()
    mutate(artifact["documents"]["d0"])
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match=message):
        tr.enforce_utility_artifact_gate(artifact, _frozen_anchor_environment())


def test_utility_artifact_gate_rejects_swapped_frozen_decision_keys():
    artifact = _verified_context_artifact()
    keys = artifact["documents"]["d0"]["decision_keys"]
    keys[0]["decision_id"], keys[1]["decision_id"] = (
        keys[1]["decision_id"], keys[0]["decision_id"]
    )
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="decision key bindings"):
        tr.enforce_utility_artifact_gate(artifact, _frozen_anchor_environment())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda keys: keys.pop(),
        lambda keys: keys[1].update({
            "runtime_type": keys[0]["runtime_type"],
            "canonical_key": keys[0]["canonical_key"],
        }),
    ],
)
def test_utility_artifact_gate_rejects_missing_or_duplicate_frozen_decision_keys(mutate):
    artifact = _verified_context_artifact()
    mutate(artifact["documents"]["d0"]["decision_keys"])
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="decision keys"):
        tr.enforce_utility_artifact_gate(artifact, _frozen_anchor_environment())


@pytest.mark.parametrize(
    ("assertion", "message"),
    [
        ({"assertion_id": "a1", "doc_id": "other", "scope": "global",
          "occurrence_ids": []}, "belongs to document"),
        ({"assertion_id": "a1", "doc_id": "d0", "scope": "global",
          "occurrence_ids": ["o1"]}, "global assertion"),
        ({"assertion_id": "a1", "doc_id": "d0", "scope": "linked",
          "occurrence_ids": []}, "linked assertion"),
    ],
)
def test_utility_artifact_gate_rejects_scope_and_document_link_mismatches(assertion, message):
    artifact = _sealed_utility_artifact(assertion)

    with pytest.raises(SystemExit, match=message):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})


def test_utility_artifact_gate_requires_hash_and_required_pins():
    artifact = _sealed_utility_artifact()
    artifact.pop("reader_pin")
    artifact["artifact_hash"] = _stable_hash({
        key: value for key, value in artifact.items() if key != "artifact_hash"
    })

    with pytest.raises(SystemExit, match="reader_pin"):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})


def test_utility_artifact_gate_rejects_hand_edited_hash_and_context_contract():
    artifact = _sealed_utility_artifact()
    artifact["reader_pin"] = "hand-edited"

    with pytest.raises(SystemExit, match="artifact_hash"):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})

    context_assertion = {
        "assertion_id": "a1", "doc_id": "d0", "family": "context",
        "scope": "linked", "occurrence_ids": ["o1"], "weight": 1.0,
        "status": "accepted",
    }
    artifact = _sealed_utility_artifact(context_assertion)
    environment = {
        "environment_hash": "env-v1",
        "documents": {"d0": {
            "occurrences": [{"occurrence_id": "o1", "decision_id": "dec1"}],
            "decisions": [{"decision_id": "dec1", "controlled": True}],
        }},
    }

    with pytest.raises(SystemExit, match="expected_action_support"):
        tr.enforce_utility_artifact_gate(artifact, environment)


def test_utility_artifact_gate_rejects_link_to_dangling_environment_decision():
    artifact = _sealed_utility_artifact({
        "assertion_id": "a1", "doc_id": "d0", "family": "delivered",
        "scope": "linked", "occurrence_ids": ["o1"], "weight": 1.0,
    })
    environment = {
        "environment_hash": "env-v1",
        "documents": {"d0": {
            "occurrences": [{"occurrence_id": "o1", "decision_id": "missing"}],
            "decisions": [{"decision_id": "dec1", "controlled": True}],
        }},
    }

    with pytest.raises(SystemExit, match="invalid controlled occurrence"):
        tr.enforce_utility_artifact_gate(artifact, environment)


def test_utility_artifact_gate_requires_accepted_context_rows():
    artifact = _sealed_utility_artifact(_accepted_context_assertion(status="rejected"))
    environment = {
        "environment_hash": "env-v1",
        "documents": {"d0": {
            "occurrences": [{"occurrence_id": "o1", "decision_id": "dec1"}],
            "decisions": [{"decision_id": "dec1", "controlled": True}],
        }},
    }

    with pytest.raises(SystemExit, match="not accepted"):
        tr.enforce_utility_artifact_gate(artifact, environment)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact["assertions"]["a-context"]["expected_action_support"]
         ["joint_anchor_action_vector"].pop("dec2"), "does not cover controlled decisions"),
        (lambda artifact: artifact["assertions"]["a-context"]["expected_action_support"]
         ["joint_anchor_action_vector"].update({"dec1": "unknown-action"}),
         "unknown frozen action"),
        (lambda artifact: artifact["assertions"]["a-context"]["expected_action_support"]
         ["joint_anchor_action_vector"].update({"dec1": "placeholder-1"}),
         "requires a non-placeholder generalization"),
        (lambda artifact: artifact["assertions"]["a-context"]["expected_action_support"]
         ["joint_anchor_action_vector"].update({"dec2": "general-2"}),
         "must keep unrelated decision"),
    ],
)
def test_utility_artifact_gate_recomputes_forged_joint_anchor_invariants(mutate, message):
    artifact = _verified_context_artifact()
    mutate(artifact)
    vector = artifact["assertions"]["a-context"]["expected_action_support"][
        "joint_anchor_action_vector"
    ]
    artifact["assertions"]["a-context"]["expected_action_support"][
        "joint_anchor_hash"
    ] = _stable_hash(vector)
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match=message):
        tr.enforce_utility_artifact_gate(artifact, _frozen_anchor_environment())


def test_utility_artifact_gate_recomputes_forged_validation_evidence():
    artifact = _verified_context_artifact()
    validation = artifact["assertions"]["a-context"]["evidence"]["validation"]
    validation["stability"]["trials"][0]["scores"]["representative"] = 0.0
    validation["stability"]["trials"][0]["passed"] = True
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="recomputed validation"):
        tr.enforce_utility_artifact_gate(artifact, _frozen_anchor_environment())


def test_utility_artifact_gate_rejects_finer_entailing_joint_anchor():
    artifact = _verified_context_artifact()
    support = artifact["assertions"]["a-context"]["expected_action_support"]
    support["joint_anchor_action_vector"]["dec1"] = "fine-1"
    support["joint_anchor_hash"] = _stable_hash(support["joint_anchor_action_vector"])
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="coarsest entailing action"):
        tr.enforce_utility_artifact_gate(artifact, _frozen_anchor_environment())


def test_utility_artifact_gate_rejects_context_link_to_uncontrolled_occurrence():
    artifact = _verified_context_artifact()
    vector = artifact["assertions"]["a-context"]["expected_action_support"][
        "joint_anchor_action_vector"
    ]
    vector.pop("dec1")
    artifact["assertions"]["a-context"]["expected_action_support"][
        "joint_anchor_hash"
    ] = _stable_hash(vector)
    _seal_artifact(artifact)
    environment = _frozen_anchor_environment()
    environment["documents"]["d0"]["decisions"][0]["controlled"] = False
    environment["documents"]["d0"]["occurrences"][0]["controlled"] = False
    artifact["documents"]["d0"]["controlled_decision_ids"] = ["dec2"]
    artifact["documents"]["d0"]["occurrence_to_decision"] = {}
    artifact["documents"]["d0"]["decision_keys"] = [
        key for key in artifact["documents"]["d0"]["decision_keys"]
        if key["decision_id"] == "dec2"
    ]
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match="uncontrolled occurrence"):
        tr.enforce_utility_artifact_gate(artifact, environment)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact["documents"]["d0"].update(
            {"utility_weight_denominator": 0.9}), "invalid denominator"),
        (lambda artifact: artifact["documents"]["d0"].update(
            {"present_family_budgets": ["context"],
             "missing_family_budgets": ["context", "delivered"]}), "invalid family state"),
        (lambda artifact: artifact["assertions"]["a-context"].update({"weight": 0.5}),
         "weights do not match family budget"),
        (lambda artifact: artifact["documents"]["d0"]["weight_groups"]["context"]
         ["condition:one"].update({"weight": 0.5}), "invalid group weight"),
    ],
)
def test_utility_artifact_gate_rejects_tampered_fixed_weights(mutate, message):
    artifact = _verified_context_artifact()
    mutate(artifact)
    _seal_artifact(artifact)

    with pytest.raises(SystemExit, match=message):
        tr.enforce_utility_artifact_gate(artifact, _frozen_anchor_environment())


def test_utility_reward_cache_binds_assertion_weights_denominator_and_recall(tmp_path):
    binding = {
        "artifact_hash": "artifact-v1",
        "assertion_ids": ["a1", "a2"],
        "weights": {"a1": 0.6, "a2": 0.4},
        "utility_weight_denominator": 1.0,
    }
    inputs = {
        "doc_id": "d0", "action_vector": {"dec1": "general-1"},
        "doc_p": "generalized", "artifact_hash": "artifact-v1",
        "scorer_pin": {"reader": "reader-v1"}, "utility_binding": binding,
    }
    cache = tr.UtilityRewardCache(tmp_path / "cache.jsonl")
    cache.store(**inputs, result={
        "out_p": "remote", "out_final": "final", "f1s": [],
        "component_scores": {"a1": 1.0, "a2": 0.5}, "recall": 0.8,
    })

    assert cache.lookup(**inputs)["recall"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="component assertion set"):
        cache.store(**inputs, result={
            "out_p": "remote", "out_final": "final", "f1s": [],
            "component_scores": {"a1": 1.0}, "recall": 0.6,
        })
    with pytest.raises(ValueError, match="recall"):
        cache.store(**inputs, result={
            "out_p": "remote", "out_final": "final", "f1s": [],
            "component_scores": {"a1": 1.0, "a2": 0.5}, "recall": 0.7,
        })


def test_qa_builder_source_digest_invalidates_live_gate_and_cache_identity(monkeypatch):
    artifact = _sealed_utility_artifact()
    baseline_pin = utility_scorer_pin()
    baseline_identity = tr.utility_rollout_cache_identity(
        doc_id="d0", action_vector={"dec1": "a1"}, doc_p="doc", artifact_hash="artifact-v1",
        scorer_pin=baseline_pin,
    )

    import cloak.train.qa_builder as qa_builder
    monkeypatch.setattr(qa_builder, "_qa_builder_source_digest", lambda: "sha256:changed")

    with pytest.raises(SystemExit, match="live builder"):
        tr.enforce_utility_artifact_gate(artifact, {"environment_hash": "env-v1"})
    assert tr.utility_rollout_cache_identity(
        doc_id="d0", action_vector={"dec1": "a1"}, doc_p="doc", artifact_hash="artifact-v1",
        scorer_pin=utility_scorer_pin(),
    ) != baseline_identity


def test_preflight_reports_frozen_wall_time_budget_and_build_measurement():
    artifact = _sealed_utility_artifact()
    artifact["threshold_manifest"]["wall_time_budgets"] = {
        "artifact_build_seconds_per_document": 1.0,
        "base_seconds_per_rollout": 2.0,
        "counterfactual_seconds_per_selected_pair": 2.0,
    }
    artifact["wall_time_budgets"] = artifact["threshold_manifest"]["wall_time_budgets"]
    artifact["documents"]["d0"]["artifact_build_seconds"] = 0.25
    _seal_artifact(artifact)

    report = tr.qa_utility_preflight_report(artifact, {"environment_hash": "env-v1"})

    assert report["wall_time"] == {
        "budgets": artifact["wall_time_budgets"],
        "artifact_build_seconds_per_document": {"d0": 0.25},
    }
