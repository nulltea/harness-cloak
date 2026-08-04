"""Cache-fixture and report-contract tests for the epsilon-zero lexicographic gate."""
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "spikes"))
import epsilon_zero_lexicographic_gate as gate  # noqa: E402

from cloak.reward.utility_cache import (  # noqa: E402
    UtilityCache,
    make_result,
    stable_hash,
    utility_binding,
)
from cloak.tests.test_interactive_ranker import _count_reward, _document  # noqa: E402

ENVIRONMENT_HASH = "sha256:fixture-environment"


def _doc(doc_id: str):
    return replace(_document(), doc_id=doc_id)


def _artifact(doc_ids, *, weights=(1.0, 1.0)):
    """Utility artifact over the fixture document shape, one assertion pair per doc.

    `<doc>-linked` carries the policy mass (it declares a dependency on `alpha`);
    `<doc>-monitor` is residual mass, reported per component but outside the
    weighted key — the same split the production artifact uses.
    """
    monitor_weight, linked_weight = weights
    assertions = {}
    documents = {}
    for doc_id in doc_ids:
        linked, monitor = f"{doc_id}-linked", f"{doc_id}-monitor"
        documents[doc_id] = {
            "assertion_ids": [monitor, linked],
            "policy_decision_ids": ["alpha", "beta"],
            "utility_weight_denominator": monitor_weight + linked_weight,
        }
        assertions[monitor] = {
            "assertion_id": monitor, "doc_id": doc_id, "family": "delivered",
            "status": "accepted", "weight": monitor_weight,
            "credit_routing": "residual", "policy_dependency_decision_ids": [],
        }
        assertions[linked] = {
            "assertion_id": linked, "doc_id": doc_id, "family": "delivered",
            "status": "accepted", "weight": linked_weight,
            "credit_routing": "linked", "policy_dependency_decision_ids": ["alpha"],
        }
    artifact = {
        "artifact_version": "utility-assertions-v2",
        "environment_hash": ENVIRONMENT_HASH,
        "reader_pin": {
            "model": "fixture-reader", "endpoint": "fixture-endpoint",
            "prompt_version": "fixture-prompt-v1",
            "response_schema": {"type": "string"}, "revision": "fixture-revision",
        },
        "documents": documents,
        "assertions": assertions,
    }
    artifact["artifact_hash"] = stable_hash(artifact)
    return artifact


def _extractor_pin():
    pin = {
        "kind": "fixture", "version": "1", "pin_version": "1", "semantic": False,
        "semantic_model": "none", "modules": {}, "packages": {},
    }
    pin["sha256"] = stable_hash(pin)
    return pin


def _identity(
    doc_id, vector_key, artifact, *,
    environment_hash=ENVIRONMENT_HASH, reader_refresh=False, doc_p=None,
):
    return {
        "doc_id": doc_id,
        "ordered_action_vector": [[decision, action] for decision, action in vector_key],
        "doc_p_hash": stable_hash(doc_p if doc_p is not None else f"{doc_id}:{vector_key}"),
        "environment_hash": environment_hash,
        "utility_artifact_hash": artifact["artifact_hash"],
        "utility_binding": utility_binding(artifact, doc_id),
        "task_prompt_pin": {
            "version": "v1", "corpus": "fixture", "template_hash": f"sha256:{'0' * 64}",
        },
        "remote_model_pin": {
            "model": "fixture-model", "endpoint": "fixture-endpoint",
            "temperature": 0.0, "max_tokens": 32, "thinking": False,
            "single_flight": True,
        },
        "extractor_pin": _extractor_pin(),
        "reader_pin": artifact["reader_pin"],
        "runtime_scorer_version": "fixture-scorer-v2",
        "execution_contract_version": "fixture-contract-v2",
        "reader_refresh": bool(reader_refresh),
    }


def _row(doc_id, vector_key, artifact, *, linked, monitor=1.0, **identity_kwargs):
    """One cache record: identity plus a result whose utility satisfies cache parity."""
    binding = utility_binding(artifact, doc_id)
    component_scores = {f"{doc_id}-monitor": monitor, f"{doc_id}-linked": linked}
    utility = sum(
        float(weight) * component_scores[assertion_id]
        for assertion_id, weight in binding["weights"].items()
    ) / float(binding["utility_weight_denominator"])
    doc_p = identity_kwargs.pop("doc_p", None)
    identity = _identity(doc_id, vector_key, artifact, doc_p=doc_p, **identity_kwargs)
    result = make_result(
        doc_id=doc_id,
        action_vector=dict(vector_key),
        doc_p=doc_p if doc_p is not None else f"{doc_id}:{vector_key}",
        out_p="remote",
        out_final="final",
        component_scores=component_scores,
        utility=utility,
    )
    return identity, result


def _cache(tmp_path, rows, name="utility-results.jsonl"):
    cache = UtilityCache(tmp_path / name)
    cache.store_many(rows)
    return UtilityCache(tmp_path / name)


# The fixture document's two levels share one fill, so `beta-level` is masked once
# `alpha-level` claims it. The frozen five anchors therefore deduplicate to three
# legal vectors, all of which have distinct frozen count scores.
BC_VECTOR = (("alpha", "alpha-level"), ("beta", "beta-placeholder"))
KEEP_VECTOR = (("alpha", "alpha-keep"), ("beta", "beta-keep"))
PLACEHOLDER_VECTOR = (("alpha", "alpha-placeholder"), ("beta", "beta-placeholder"))
SLATE = (BC_VECTOR, KEEP_VECTOR, PLACEHOLDER_VECTOR)
ILLEGAL_VECTOR = (("alpha", "alpha-level"), ("beta", "beta-level"))
EXTRA_VECTOR = (("alpha", "alpha-keep"), ("beta", "beta-placeholder"))


def _corpus(tmp_path, rows, doc_ids=("valid",), artifact=None, targets=None):
    artifact = artifact if artifact is not None else _artifact(doc_ids)
    targets = targets if targets is not None else _count_reward()
    documents = tuple(_doc(doc_id) for doc_id in doc_ids)
    return gate.load_candidate_corpus(
        documents,
        _cache(tmp_path, rows),
        artifact,
        targets,
        environment_hash=ENVIRONMENT_HASH,
        utility_artifact_hash=artifact["artifact_hash"],
    )


def test_candidate_corpus_joins_the_standardized_slate_and_audits_every_exclusion(
    tmp_path,
):
    artifact = _artifact(("valid",))
    rows = [
        _row("valid", BC_VECTOR, artifact, linked=1.0),
        _row("valid", KEEP_VECTOR, artifact, linked=1.0),
        _row("valid", PLACEHOLDER_VECTOR, artifact, linked=0.5),
        _row("valid", EXTRA_VECTOR, artifact, linked=1.0),
        # excluded: refreshed reader, incomplete vector, wrong pin, illegal vector
        _row("valid", BC_VECTOR, artifact, linked=0.0, reader_refresh=True),
        _row("valid", (("alpha", "alpha-keep"),), artifact, linked=1.0),
        _row("valid", PLACEHOLDER_VECTOR, artifact, linked=1.0,
             environment_hash="sha256:other-environment"),
        _row("valid", ILLEGAL_VECTOR, artifact, linked=1.0),
    ]
    corpus, audit = _corpus(tmp_path, rows, artifact=artifact)

    document = corpus["valid"]
    assert [row.vector_key for row in document.expected_slate] == sorted(SLATE)
    assert {row.vector_key for row in document.gate_candidates} == set(SLATE)
    assert [row.vector_key for row in document.expanded_cache_candidates] == [EXTRA_VECTOR]
    assert document.candidate_support_status == "support-complete"
    assert audit["reader_refresh_excluded"] == 1
    assert audit["incomplete_vector_excluded"] == 1
    assert audit["pin_mismatch_excluded"] == 1
    assert audit["illegal_vector_excluded"] == 1
    assert audit["validated_candidates"] == 4


def test_candidate_corpus_marks_a_missing_anchor_unsupported_despite_extra_vectors(
    tmp_path,
):
    artifact = _artifact(("valid",))
    rows = [
        _row("valid", BC_VECTOR, artifact, linked=1.0),
        _row("valid", KEEP_VECTOR, artifact, linked=1.0),
        _row("valid", EXTRA_VECTOR, artifact, linked=1.0),
    ]
    corpus, _audit = _corpus(tmp_path, rows, artifact=artifact)

    document = corpus["valid"]
    assert document.candidate_support_status == "unsupported-missing-slate"
    assert [row.vector_key for row in document.missing_expected_vectors] == [
        PLACEHOLDER_VECTOR
    ]
    # an adaptively cached extra vector cannot repair the missing standardized anchor
    assert EXTRA_VECTOR in {row.vector_key for row in document.expanded_cache_candidates}
    assert EXTRA_VECTOR not in {row.vector_key for row in document.gate_candidates}


def test_candidate_corpus_marks_an_excluded_row_missing_but_keeps_its_reason(tmp_path):
    artifact = _artifact(("valid",))
    rows = [
        _row("valid", BC_VECTOR, artifact, linked=1.0),
        _row("valid", KEEP_VECTOR, artifact, linked=1.0),
        _row("valid", PLACEHOLDER_VECTOR, artifact, linked=1.0, reader_refresh=True),
    ]
    corpus, audit = _corpus(tmp_path, rows, artifact=artifact)

    assert corpus["valid"].candidate_support_status == "unsupported-missing-slate"
    assert audit["reader_refresh_excluded"] == 1


def test_candidate_corpus_without_count_contrast_is_unsupported(tmp_path):
    artifact = _artifact(("valid",))
    uniform = _count_reward(alpha_level=1.0, beta_level=1.0)
    for row in uniform.target_rows():
        object.__setattr__(row, "profile_score", 1.0)
    rows = [_row("valid", vector, artifact, linked=1.0) for vector in SLATE]
    corpus, _audit = _corpus(tmp_path, rows, artifact=artifact, targets=uniform)

    document = corpus["valid"]
    assert not document.missing_expected_vectors
    assert len({row.privacy_key for row in document.gate_candidates}) == 1
    assert document.candidate_support_status == "unsupported-no-count-contrast"


def test_candidate_corpus_preserves_deduplicated_anchor_sources(tmp_path):
    artifact = _artifact(("valid",))
    rows = [_row("valid", vector, artifact, linked=1.0) for vector in SLATE]
    corpus, _audit = _corpus(tmp_path, rows, artifact=artifact)

    sources = {row.vector_key: row.sources for row in corpus["valid"].expected_slate}
    assert sources[BC_VECTOR] == (
        "behavior_cloning", "midpoint_level_walk", "minimum_count_non_keep_walk",
    )
    assert sources[KEEP_VECTOR] == ("keep_walk",)
    assert sources[PLACEHOLDER_VECTOR] == ("all_placeholder_walk",)


def test_candidate_corpus_rejects_conflicting_base_results(tmp_path):
    artifact = _artifact(("valid",))
    rows = [
        _row("valid", BC_VECTOR, artifact, linked=1.0, doc_p="rendering-a"),
        _row("valid", BC_VECTOR, artifact, linked=0.0, doc_p="rendering-b"),
    ]
    with pytest.raises(ValueError, match="conflicting base cache results"):
        _corpus(tmp_path, rows, artifact=artifact)


def test_candidate_corpus_privacy_key_is_the_document_mean_over_active_decisions(
    tmp_path,
):
    artifact = _artifact(("valid",))
    rows = [_row("valid", vector, artifact, linked=1.0) for vector in SLATE]
    corpus, _audit = _corpus(tmp_path, rows, artifact=artifact)

    keys = {row.vector_key: row.privacy_key for row in corpus["valid"].gate_candidates}
    # `_count_reward` scores: alpha level 0.2, beta level 0.9, keep 0.0, placeholder 1.0
    assert keys[KEEP_VECTOR] == Decimal("0")
    assert keys[PLACEHOLDER_VECTOR] == Decimal("1")
    assert keys[BC_VECTOR] == (Decimal("0.2") + Decimal("1")) / Decimal(2)
