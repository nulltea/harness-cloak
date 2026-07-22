"""Complete ranker-v2 utility results and append-only cache contracts."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import MappingProxyType

import pytest

import cloak.train.roundtrip as roundtrip
from cloak.train.qa_builder import DEFAULT_CONTEXT_READER_PIN
from cloak.train.ranker_environment import RankerAction, RankerDecision, RankerDocument


def _document() -> RankerDocument:
    decision = RankerDecision(
        decision_id="decision-a",
        profile_id="profile-a",
        runtime_type="PERSON",
        canonical_key="alice",
        occurrence_ids=("occurrence-a",),
        actions=(
            RankerAction(
                action_id="action-level",
                mode="level",
                fill="person",
                authored_level_index=0,
                runtime_type="PERSON",
            ),
            RankerAction(
                action_id="action-keep",
                mode="keep",
                fill="Alice",
                authored_level_index=None,
                runtime_type="PERSON",
            ),
        ),
    )
    return RankerDocument(
        doc_id="clinical/fixture",
        corpus="clinical",
        text="Alice completed the task.",
        occurrences=(MappingProxyType({
            "occurrence_id": "occurrence-a",
            "decision_id": "decision-a",
            "start": 0,
            "end": 5,
            "surface": "Alice",
            "controlled": True,
        }),),
        policy_decisions=(decision,),
        fixed_decisions=(),
    )


def _artifact(
    *, reader_revision="reader-r1", context_weight=0.5,
    environment_hash="sha256:environment-a",
):
    from cloak.train.utility_cache import stable_hash

    reader_pin = {**DEFAULT_CONTEXT_READER_PIN, "revision": reader_revision}
    artifact = {
        "artifact_version": "utility-assertions-v2",
        "environment_hash": environment_hash,
        "reader_pin": reader_pin,
        "documents": {
            "clinical/fixture": {
                "assertion_ids": ["context-a", "delivered-a"],
                "utility_weight_denominator": 1.0,
                "decisions": [],
            },
        },
        "assertions": {
            "context-a": {
                "assertion_id": "context-a",
                "doc_id": "clinical/fixture",
                "family": "context",
                "question": "Who completed the task?",
                "accepted_values": ["person"],
                "weight": context_weight,
            },
            "delivered-a": {
                "assertion_id": "delivered-a",
                "doc_id": "clinical/fixture",
                "family": "delivered",
                "scoring_contract": {"kind": "contains", "value": "completed"},
                "weight": 1.0 - context_weight,
            },
        },
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact


def _request(
    artifact=None, *, environment_hash="sha256:environment-a",
    action_id="action-level",
):
    from cloak.train.utility_cache import UtilityRequest

    return UtilityRequest(
        document=_document(),
        action_vector=MappingProxyType({"decision-a": action_id}),
        utility_artifact=_artifact() if artifact is None else artifact,
        environment_hash=environment_hash,
    )


class _Remote:
    def __init__(self, reply="person completed the task."):
        self.reply = reply
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return self.reply


def _reader_for(pin, *, fail=False, calls=None):
    def reader(questions, context, clauses=None, refresh=False):
        if calls is not None:
            calls.append((tuple(questions), context, tuple(clauses or ()), refresh))
        if fail:
            raise RuntimeError("reader failed")
        return ["person"]

    reader.pin = pin
    return reader


def _wire(monkeypatch, artifact, *, remote=None, reader_fail=False, reader_calls=None):
    remote = remote or _Remote()
    reader = _reader_for(
        artifact["reader_pin"], fail=reader_fail, calls=reader_calls,
    )
    monkeypatch.setattr(roundtrip, "_remote", lambda: remote)
    monkeypatch.setattr(roundtrip, "invert", lambda out_p, replacements: (out_p, {}))
    monkeypatch.setattr(roundtrip, "read_context_batch", reader)
    monkeypatch.setattr(roundtrip, "read_context_set_batch", reader)
    return remote


def test_public_result_contract_is_exact_and_frozen():
    from cloak.train.utility_cache import UtilityResult

    assert [field.name for field in fields(UtilityResult)] == [
        "doc_id", "action_vector", "doc_p", "out_p", "out_final",
        "component_scores", "utility", "result_hash",
    ]
    result = UtilityResult(
        doc_id="d", action_vector={}, doc_p="p", out_p="o", out_final="f",
        component_scores={"a": 1.0}, utility=1.0, result_hash="sha256:x",
    )
    with pytest.raises(FrozenInstanceError):
        result.utility = 0.0
    with pytest.raises(TypeError):
        result.component_scores["a"] = 0.0


def test_batch_deduplicates_then_hits_append_only_cache(monkeypatch, tmp_path):
    from cloak.train.utility_cache import UtilityCache

    artifact = _artifact()
    remote = _wire(monkeypatch, artifact)
    cache = UtilityCache(tmp_path / "utility.jsonl")
    request = _request(artifact)

    first = roundtrip.score_roundtrip_batch(
        [request, request], cache=cache, remote_workers=2, reader_workers=2,
    )
    first_metrics = json.loads(json.dumps(cache.last_batch_metrics))
    second = roundtrip.score_roundtrip_batch(
        [request], cache=cache, remote_workers=1, reader_workers=1,
    )

    assert remote.calls == 1
    assert first[0] == first[1] == second[0]
    assert set(first[0].component_scores) == {"context-a", "delivered-a"}
    assert all(0.0 <= score <= 1.0 for score in first[0].component_scores.values())
    assert first[0].utility == pytest.approx(sum(
        artifact["assertions"][assertion_id]["weight"] * score
        for assertion_id, score in first[0].component_scores.items()
    ))
    assert cache.last_batch_metrics["cache_hits"] == 1
    assert first_metrics["rollouts"] == 2
    assert first_metrics["context_assertions"] == 1
    assert first_metrics["reader_work_items"] == 1
    assert first_metrics["transport_calls"] == 2
    assert first_metrics["cache_hits"] == 0
    assert first_metrics["peak_concurrency"] == {"generation": 1, "reader": 1}
    assert set(first_metrics["stage_latency_seconds"]) == {
        "render", "cache_lookup", "generation", "extraction",
        "deterministic_scoring", "reader", "finalize", "cache_append",
    }
    assert all(value >= 0.0 for value in first_metrics["stage_latency_seconds"].values())
    lines = (tmp_path / "utility.jsonl").read_text().splitlines()
    assert len(lines) == 1
    identity = json.loads(lines[0])["identity"]
    assert identity["ordered_action_vector"] == [["decision-a", "action-level"]]
    assert identity["doc_p_hash"].startswith("sha256:")
    assert "doc_p" not in identity
    assert identity["environment_hash"] == "sha256:environment-a"
    assert identity["utility_artifact_hash"] == artifact["artifact_hash"]
    assert identity["reader_pin"] == artifact["reader_pin"]
    assert identity["extractor_pin"]
    assert identity["task_prompt_pin"]
    assert identity["remote_model_pin"]
    assert identity["runtime_scorer_version"]
    assert identity["execution_contract_version"]


@pytest.mark.parametrize("changed", ["reader", "weight", "environment", "refresh"])
def test_identity_changes_are_cache_misses(monkeypatch, tmp_path, changed):
    from cloak.train.utility_cache import UtilityCache

    base_artifact = _artifact()
    remote = _wire(monkeypatch, base_artifact)
    cache = UtilityCache(tmp_path / "utility.jsonl")
    roundtrip.score_roundtrip_batch(
        [_request(base_artifact)], cache=cache, remote_workers=1, reader_workers=1,
    )

    artifact = base_artifact
    environment_hash = "sha256:environment-a"
    refresh = False
    if changed == "reader":
        artifact = _artifact(reader_revision="reader-r2")
        _wire(monkeypatch, artifact, remote=remote)
    elif changed == "weight":
        artifact = _artifact(context_weight=0.25)
    elif changed == "environment":
        environment_hash = "sha256:environment-b"
        artifact = _artifact(environment_hash=environment_hash)
    else:
        refresh = True
    roundtrip.score_roundtrip_batch(
        [_request(artifact, environment_hash=environment_hash)],
        cache=cache, remote_workers=1, reader_workers=1, reader_refresh=refresh,
    )

    assert remote.calls == 2
    assert len((tmp_path / "utility.jsonl").read_text().splitlines()) == 2


def test_tampered_artifact_or_environment_binding_is_rejected(monkeypatch, tmp_path):
    from cloak.train.utility_cache import UtilityCache

    artifact = _artifact()
    _wire(monkeypatch, artifact)
    tampered = json.loads(json.dumps(artifact))
    tampered["assertions"]["context-a"]["weight"] = 0.25
    with pytest.raises(ValueError, match="artifact_hash"):
        roundtrip.score_roundtrip_batch(
            [_request(tampered)], cache=UtilityCache(tmp_path / "tampered.jsonl"),
            remote_workers=1, reader_workers=1,
        )
    with pytest.raises(ValueError, match="environment_hash"):
        roundtrip.score_roundtrip_batch(
            [_request(artifact, environment_hash="sha256:other")],
            cache=UtilityCache(tmp_path / "mismatch.jsonl"),
            remote_workers=1, reader_workers=1,
        )


def test_cache_rejects_incomplete_invalid_or_denominator_drifting_vectors(
    monkeypatch, tmp_path,
):
    from cloak.train.utility_cache import UtilityCache, make_result

    artifact = _artifact()
    _wire(monkeypatch, artifact)
    path = tmp_path / "utility.jsonl"
    cache = UtilityCache(path)
    valid = roundtrip.score_roundtrip_batch(
        [_request(artifact)], cache=cache, remote_workers=1, reader_workers=1,
    )[0]
    identity = json.loads(path.read_text())["identity"]

    def changed(scores, utility):
        return make_result(
            doc_id=valid.doc_id, action_vector=valid.action_vector,
            doc_p=valid.doc_p, out_p=valid.out_p, out_final=valid.out_final,
            component_scores=scores, utility=utility,
        )

    with pytest.raises(ValueError, match="component assertion set"):
        cache.store_many([(identity, changed({"context-a": 1.0}, 0.5))])
    with pytest.raises(ValueError, match="invalid component score"):
        cache.store_many([(
            identity, changed({"context-a": 2.0, "delivered-a": 1.0}, 1.0),
        )])
    with pytest.raises(ValueError, match="fixed-denominator parity"):
        cache.store_many([(
            identity, changed({"context-a": 1.0, "delivered-a": 1.0}, 0.5),
        )])


def test_multi_decision_action_order_survives_canonical_json_reload(monkeypatch, tmp_path):
    from cloak.train.utility_cache import UtilityCache, make_result

    artifact = _artifact()
    _wire(monkeypatch, artifact)
    seed_path = tmp_path / "seed.jsonl"
    valid = roundtrip.score_roundtrip_batch(
        [_request(artifact)], cache=UtilityCache(seed_path),
        remote_workers=1, reader_workers=1,
    )[0]
    identity = json.loads(seed_path.read_text())["identity"]
    identity["ordered_action_vector"] = [["z-decision", "z-action"],
                                         ["a-decision", "a-action"]]
    ordered_result = make_result(
        doc_id=valid.doc_id,
        action_vector={"z-decision": "z-action", "a-decision": "a-action"},
        doc_p=valid.doc_p,
        out_p=valid.out_p,
        out_final=valid.out_final,
        component_scores=valid.component_scores,
        utility=valid.utility,
    )
    path = tmp_path / "ordered.jsonl"
    UtilityCache(path).store_many([(identity, ordered_result)])

    reloaded = UtilityCache(path).lookup(identity)
    assert list(reloaded.action_vector) == ["z-decision", "a-decision"]


def test_incomplete_or_self_inconsistent_pins_are_rejected(monkeypatch, tmp_path):
    from cloak.train.utility_cache import UtilityCache

    artifact = _artifact()
    _wire(monkeypatch, artifact)
    path = tmp_path / "utility.jsonl"
    roundtrip.score_roundtrip_batch(
        [_request(artifact)], cache=UtilityCache(path),
        remote_workers=1, reader_workers=1,
    )
    identity = json.loads(path.read_text())["identity"]
    identity["remote_model_pin"].pop("model")
    with pytest.raises(ValueError, match="remote_model_pin"):
        UtilityCache.request_identity(identity)
    identity = json.loads(path.read_text())["identity"]
    identity["extractor_pin"]["sha256"] = "sha256:tampered"
    with pytest.raises(ValueError, match="extractor_pin"):
        UtilityCache.request_identity(identity)


def test_stale_cache_writers_reload_under_lock_and_reject_conflicts(
    monkeypatch, tmp_path,
):
    from cloak.train.utility_cache import UtilityCache, make_result

    artifact = _artifact()
    _wire(monkeypatch, artifact)
    seed_path = tmp_path / "seed.jsonl"
    valid = roundtrip.score_roundtrip_batch(
        [_request(artifact)], cache=UtilityCache(seed_path),
        remote_workers=1, reader_workers=1,
    )[0]
    identity = json.loads(seed_path.read_text())["identity"]
    path = tmp_path / "shared.jsonl"
    first, identical, conflicting = (UtilityCache(path) for _ in range(3))

    first.store_many([(identity, valid)])
    identical.store_many([(identity, valid)])
    assert len(path.read_text().splitlines()) == 1
    changed = make_result(
        doc_id=valid.doc_id, action_vector=valid.action_vector, doc_p=valid.doc_p,
        out_p="different", out_final=valid.out_final,
        component_scores=valid.component_scores, utility=valid.utility,
    )
    with pytest.raises(ValueError, match="conflicting duplicate utility cache identity"):
        conflicting.store_many([(identity, changed)])
    assert len(path.read_text().splitlines()) == 1


def test_multiple_misses_keep_stage_barriers_and_failure_is_batch_atomic(
    monkeypatch, tmp_path,
):
    from cloak.train.utility_cache import UtilityCache

    artifact = _artifact()
    events = []

    class Remote:
        def generate(self, prompt):
            events.append("generate")
            return "person completed the task."

    reader = _reader_for(artifact["reader_pin"])
    monkeypatch.setattr(roundtrip, "_remote", lambda: Remote())
    monkeypatch.setattr(
        roundtrip, "invert",
        lambda out_p, replacements: (events.append("extract") or out_p, {}),
    )

    def tracked_reader(*args, **kwargs):
        events.append("reader")
        return reader(*args, **kwargs)

    tracked_reader.pin = artifact["reader_pin"]
    monkeypatch.setattr(roundtrip, "read_context_batch", tracked_reader)
    monkeypatch.setattr(roundtrip, "read_context_set_batch", tracked_reader)
    requests = [_request(artifact), _request(artifact, action_id="action-keep")]
    roundtrip.score_roundtrip_batch(
        requests, cache=UtilityCache(tmp_path / "stages.jsonl"),
        remote_workers=1, reader_workers=1,
    )
    assert events == ["generate", "generate", "extract", "extract", "reader", "reader"]

    calls = 0

    class FailingRemote:
        def generate(self, prompt):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("second generation failed")
            return "first completed"

    monkeypatch.setattr(roundtrip, "_remote", lambda: FailingRemote())
    failed_path = tmp_path / "failed-batch.jsonl"
    failed_cache = UtilityCache(failed_path)
    with pytest.raises(RuntimeError, match="second generation failed"):
        roundtrip.score_roundtrip_batch(
            requests, cache=failed_cache, remote_workers=1, reader_workers=1,
        )
    assert not failed_path.exists()
    assert failed_cache.last_batch_metrics["transport_calls"] == 2
    assert failed_cache.last_batch_metrics["stage_latency_seconds"]["generation"] > 0.0


@pytest.mark.parametrize("stage", ["deterministic_scoring", "cache_append"])
def test_nontransport_failure_stages_record_latency(monkeypatch, tmp_path, stage):
    from cloak.train.utility_cache import UtilityCache

    artifact = _artifact()
    _wire(monkeypatch, artifact)
    cache = UtilityCache(tmp_path / f"{stage}.jsonl")
    if stage == "deterministic_scoring":
        monkeypatch.setattr(
            roundtrip, "prepare_utility_scoring",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("score failed")),
        )
    else:
        monkeypatch.setattr(
            cache, "store_many",
            lambda items: (_ for _ in ()).throw(RuntimeError("append failed")),
        )

    with pytest.raises(RuntimeError):
        roundtrip.score_roundtrip_batch(
            [_request(artifact)], cache=cache, remote_workers=1, reader_workers=1,
        )

    assert cache.last_batch_metrics["stage_latency_seconds"][stage] > 0.0
    assert not cache.path.exists()


@pytest.mark.parametrize("stage", ["generation", "extraction", "reader"])
def test_failed_stage_appends_nothing(monkeypatch, tmp_path, stage):
    from cloak.train.utility_cache import UtilityCache

    artifact = _artifact()
    cache_path = tmp_path / "utility.jsonl"
    remote = _wire(monkeypatch, artifact, reader_fail=stage == "reader")
    if stage == "generation":
        remote.generate = lambda prompt: (_ for _ in ()).throw(RuntimeError("generation failed"))
    if stage == "extraction":
        monkeypatch.setattr(
            roundtrip, "invert",
            lambda out_p, replacements: (_ for _ in ()).throw(RuntimeError("extract failed")),
        )

    with pytest.raises(RuntimeError):
        roundtrip.score_roundtrip_batch(
            [_request(artifact)], cache=UtilityCache(cache_path),
            remote_workers=1, reader_workers=1,
        )
    assert not cache_path.exists() or cache_path.read_bytes() == b""


def test_cache_rejects_truncated_or_malformed_jsonl(tmp_path):
    from cloak.train.utility_cache import UtilityCache

    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text('{"version":1}')
    with pytest.raises(ValueError, match="truncated"):
        UtilityCache(truncated)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_bytes(b"not-json\n")
    with pytest.raises(ValueError, match="invalid utility cache"):
        UtilityCache(malformed)


def test_cache_rejects_conflicting_duplicate_identity(monkeypatch, tmp_path):
    from cloak.train.utility_cache import UtilityCache, result_hash, stable_hash

    path = tmp_path / "conflict.jsonl"
    artifact = _artifact()
    _wire(monkeypatch, artifact)
    roundtrip.score_roundtrip_batch(
        [_request(artifact)], cache=UtilityCache(path),
        remote_workers=1, reader_workers=1,
    )
    original = json.loads(path.read_text().strip())
    conflicting = json.loads(json.dumps(original))
    conflicting["result"]["out_p"] = "different but individually valid output"
    conflicting["result"]["result_hash"] = result_hash(conflicting["result"])
    conflicting["result_hash"] = conflicting["result"]["result_hash"]
    conflicting["storage_identity"] = stable_hash({
        "request_identity": conflicting["request_identity"],
        "result_hash": conflicting["result_hash"],
    })
    with path.open("a") as handle:
        handle.write(json.dumps(conflicting) + "\n")

    with pytest.raises(ValueError, match="conflicting duplicate request identity"):
        UtilityCache(path)
