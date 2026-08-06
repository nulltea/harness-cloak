"""Cache-fixture and report-contract tests for the epsilon-zero lexicographic gate."""
import json
import random
import statistics as st
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
# adaptively cached, legal, and MORE private than the BC anchor (0.95 vs 0.6)
PRIVATE_EXTRA_VECTOR = (("alpha", "alpha-placeholder"), ("beta", "beta-level"))


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


def _count_state():
    """Minimal count-target provenance for the fixture actions."""
    rows = {}
    for decision in _document().policy_decisions:
        for action in decision.actions:
            rows[action.action_id] = {
                "mode": action.mode,
                "profile_id": str(decision.profile_id),
                "grounding_status": "grounded" if action.mode == "level" else None,
                "source_family": "fixture" if action.mode == "level" else None,
            }
    return {"artifact_hash": "sha256:fixture-count-targets", "action_targets": rows}


def _gate_report(tmp_path, plans, *, targets=None, campaign=()):
    """Run the whole synthetic gate. `plans` maps doc_id -> [(vector, linked score)]."""
    artifact = _artifact(tuple(plans))
    rows = [
        _row(doc_id, vector, artifact, linked=linked)
        for doc_id, entries in plans.items()
        for vector, linked in entries
    ]
    documents = tuple(_doc(doc_id) for doc_id in plans)
    monkeypatched = gate.CAMPAIGN_DOCUMENTS
    gate.CAMPAIGN_DOCUMENTS = tuple(campaign)
    try:
        report = gate.run_gate(
            documents,
            _cache(tmp_path, rows),
            artifact,
            targets if targets is not None else _count_reward(),
            _count_state(),
            environment_hash=ENVIRONMENT_HASH,
            utility_artifact_hash=artifact["artifact_hash"],
        )
    finally:
        gate.CAMPAIGN_DOCUMENTS = monkeypatched
    return report, {row["doc_id"]: row for row in report["documents"]}


def test_document_records_score_gain_only_inside_the_exact_utility_optimum(tmp_path):
    plans = {
        # 1: complete support, one optimum, no tie
        "one-optimum": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 0.5), (PLACEHOLDER_VECTOR, 0.25)],
        # 3: two exact optima with different privacy
        "two-optima": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 0.5), (PLACEHOLDER_VECTOR, 1.0)],
        # 4: the more-private vector is worse by 1e-12 and must be excluded
        "near-miss": [
            (BC_VECTOR, 0.5), (KEEP_VECTOR, 0.25), (PLACEHOLDER_VECTOR, 0.5 - 1e-12),
        ],
        # 5: three exact optima; BC-nearest and privacy-max differ
        "three-optima": [(vector, 1.0) for vector in SLATE],
        # 6: one missing standardized vector despite an extra cached vector
        "missing-anchor": [
            (BC_VECTOR, 1.0), (KEEP_VECTOR, 1.0), (EXTRA_VECTOR, 1.0),
        ],
        # 7: a one-vector document
        "one-vector": [(BC_VECTOR, 1.0)],
    }
    _report, records = _gate_report(tmp_path, plans)

    positive = {
        doc_id for doc_id, row in records.items()
        if row["free_count_gain"] not in (None, 0.0) and row["free_count_gain"] > 0.0
    }
    assert positive == {"two-optima", "three-optima"}

    assert records["one-optimum"]["exact_optimal_set_size"] == 1
    assert records["one-optimum"]["opportunity_status"] == "supported-no-opportunity"

    two = records["two-optima"]
    assert two["exact_optimal_set_size"] == 2
    assert two["utility_only"]["vector"] == [list(pair) for pair in BC_VECTOR]
    assert two["epsilon_zero_lexicographic"]["vector"] == [
        list(pair) for pair in PLACEHOLDER_VECTOR
    ]
    assert two["free_count_gain"] == pytest.approx(0.4)
    assert two["selection_hamming_distance"] == 1
    assert two["selector_changes_baseline"] is True
    assert two["opportunity_status"] == "supported-opportunity"
    assert set(two["selected_count_provenance"]) == {"alpha-placeholder", "beta-placeholder"}

    near = records["near-miss"]
    assert near["exact_optimal_set_size"] == 1
    assert near["epsilon_zero_lexicographic"]["vector"] == [list(p) for p in BC_VECTOR]
    assert near["free_count_gain"] == 0.0

    three = records["three-optima"]
    assert three["exact_optimal_set_size"] == 3
    assert three["exact_optimal_count_min"] == 0.0
    assert three["exact_optimal_count_max"] == 1.0
    assert three["free_count_gain"] == pytest.approx(0.4)

    for doc_id in ("missing-anchor", "one-vector"):
        row = records[doc_id]
        assert row["candidate_support_status"] == "unsupported-missing-slate"
        assert row["free_count_gain"] is None
        assert row["opportunity_status"] is None
        assert row["epsilon_zero_lexicographic"] is None


def test_two_exact_optima_with_equal_privacy_have_no_opportunity(tmp_path):
    # alpha_level = 1.0 makes the BC and all-placeholder vectors count-identical.
    targets = _count_reward(alpha_level=1.0)
    plans = {"equal-privacy": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 0.5), (PLACEHOLDER_VECTOR, 1.0)]}
    _report, records = _gate_report(tmp_path, plans, targets=targets)

    row = records["equal-privacy"]
    assert row["exact_optimal_set_size"] == 2
    assert row["exact_optimal_count_spread"] == 0.0
    assert row["free_count_gain"] == 0.0
    assert row["selector_changes_baseline"] is False
    assert row["opportunity_status"] == "supported-no-opportunity"


def test_verdict_adopts_composition_only_when_every_primary_document_is_supported(
    tmp_path,
):
    supported = {
        f"doc-{index}": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 0.5), (PLACEHOLDER_VECTOR, 1.0)]
        for index in range(3)
    }
    report, _records = _gate_report(tmp_path, supported)
    assert report["verdict"] == "adopt-exact-lexicographic-composition"
    assert report["adjudication_checks"]["all_primary_documents_support_complete"] is True
    assert report["summary"]["primary"]["bootstrap"]["lower_bound_95_one_sided"] > 0.0

    # one primary document short of one standardized anchor closes the corpus verdict,
    # even though the other documents show positive gain
    starved = {**supported, "doc-thin": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 1.0)]}
    report, _records = _gate_report(tmp_path / "starved", starved)
    assert report["verdict"] == "insufficient-candidate-breadth"
    assert report["adjudication_checks"]["any_primary_document_has_positive_gain"] is True
    assert report["summary"]["primary"]["unsupported_document_ids_by_reason"] == {
        "unsupported-missing-slate": ["doc-thin"],
    }


def test_no_observed_exact_opportunity_when_every_supported_gain_is_zero(tmp_path):
    plans = {
        f"doc-{index}": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 0.5), (PLACEHOLDER_VECTOR, 0.25)]
        for index in range(3)
    }
    report, _records = _gate_report(tmp_path, plans)
    assert report["verdict"] == "no-observed-exact-opportunity"
    assert report["summary"]["primary"]["support_complete_fraction"] == 1.0
    assert report["summary"]["primary"]["mean_free_count_gain"] == 0.0


def test_campaign_documents_are_reported_but_never_adjudicate(tmp_path):
    plans = {
        "campaign-doc": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 0.5), (PLACEHOLDER_VECTOR, 1.0)],
        "primary-doc": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 0.5), (PLACEHOLDER_VECTOR, 0.25)],
    }
    report, records = _gate_report(tmp_path, plans, campaign=("campaign-doc",))
    assert records["campaign-doc"]["population"] == "campaign"
    assert records["campaign-doc"]["free_count_gain"] == pytest.approx(0.4)
    assert report["summary"]["campaign"]["positive_gain_document_count"] == 1
    assert report["adjudication_checks"]["primary_document_count"] == 1
    # the campaign document's gain cannot rescue the primary population
    assert report["verdict"] == "no-observed-exact-opportunity"


def test_bootstrap_ignores_unsupported_documents_and_reports_its_settings(tmp_path):
    plans = {
        f"doc-{index}": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 0.5), (PLACEHOLDER_VECTOR, 1.0)]
        for index in range(3)
    }
    baseline, _records = _gate_report(tmp_path, plans)
    with_unsupported, _records = _gate_report(
        tmp_path / "with-unsupported", {**plans, "doc-thin": [(BC_VECTOR, 1.0)]},
    )
    assert (
        with_unsupported["summary"]["primary"]["bootstrap"]
        == baseline["summary"]["primary"]["bootstrap"]
    )
    assert baseline["summary"]["primary"]["bootstrap"]["seed"] == 20260804
    assert baseline["summary"]["primary"]["bootstrap"]["resamples"] == 10_000
    assert baseline["summary"]["primary"]["bootstrap"]["document_count"] == 3
    assert with_unsupported["summary"]["primary"]["support_complete_fraction"] == 0.75


def test_lower_mean_gain_bound_is_deterministic_and_bounded_by_the_mean():
    gains = [0.0, 0.1, 0.4, 0.4, 0.0, 0.2]
    first = gate.lower_mean_gain_bound(gains)
    assert first == gate.lower_mean_gain_bound(list(reversed(gains)) and gains)
    assert 0.0 < first < st.fmean(gains)
    assert gate.lower_mean_gain_bound([0.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="bootstrap"):
        gate.lower_mean_gain_bound([])


def test_expanded_cache_vectors_cannot_change_the_verdict_or_the_selection(tmp_path):
    plans = {
        f"doc-{index}": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 0.5), (PLACEHOLDER_VECTOR, 0.25)]
        for index in range(3)
    }
    baseline, records = _gate_report(tmp_path, plans)
    # this extra vector is adaptively cached, ties the exact optimum, and is MORE private
    expanded = {
        doc_id: [*entries, (PRIVATE_EXTRA_VECTOR, 1.0)] for doc_id, entries in plans.items()
    }
    widened, widened_records = _gate_report(tmp_path / "expanded", expanded)

    assert widened["verdict"] == baseline["verdict"] == "no-observed-exact-opportunity"
    for doc_id in plans:
        assert (
            widened_records[doc_id]["epsilon_zero_lexicographic"]["vector_hash"]
            == records[doc_id]["epsilon_zero_lexicographic"]["vector_hash"]
        )
        assert widened_records[doc_id]["free_count_gain"] == 0.0
        # the diagnostic block does see it, and says so
        diagnostic = widened_records[doc_id]["expanded_cache_diagnostic"]
        assert diagnostic["adjudicating"] is False
        assert diagnostic["candidate_vector_count"] == 4
        assert diagnostic["free_count_gain"] > 0.0


def test_gate_report_is_byte_identical_under_shuffled_cache_order(tmp_path):
    plans = {
        f"doc-{index}": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 0.5), (PLACEHOLDER_VECTOR, 1.0)]
        for index in range(3)
    }
    artifact = _artifact(tuple(plans))
    rows = [
        _row(doc_id, vector, artifact, linked=linked)
        for doc_id, entries in plans.items()
        for vector, linked in entries
    ]
    documents = tuple(_doc(doc_id) for doc_id in plans)

    def report(order, name):
        return gate.run_gate(
            documents,
            _cache(tmp_path, order, name=name),
            artifact,
            _count_reward(),
            _count_state(),
            environment_hash=ENVIRONMENT_HASH,
            utility_artifact_hash=artifact["artifact_hash"],
        )

    generator = random.Random(20260804)
    shuffled = list(rows)
    generator.shuffle(shuffled)
    first = json.dumps(report(rows, "a.jsonl"), indent=1, sort_keys=True)
    second = json.dumps(report(shuffled, "b.jsonl"), indent=1, sort_keys=True)
    assert first == second
    assert "timestamp" not in first and "wall_time" not in first


def _comparator_report(tmp_path, plans, vectors_by_label, *, metadata=None):
    artifact = _artifact(tuple(plans))
    rows = [
        _row(doc_id, vector, artifact, linked=linked)
        for doc_id, entries in plans.items()
        for vector, linked in entries
    ]
    return gate.run_gate(
        tuple(_doc(doc_id) for doc_id in plans),
        _cache(tmp_path, rows),
        artifact,
        _count_reward(),
        _count_state(),
        environment_hash=ENVIRONMENT_HASH,
        utility_artifact_hash=artifact["artifact_hash"],
        comparator_vectors=vectors_by_label,
        comparator_metadata=metadata or {},
    )


def test_additive_comparator_reports_both_failure_modes_and_cache_misses(tmp_path):
    plans = {
        "doc-a": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 0.5), (PLACEHOLDER_VECTOR, 1.0)],
    }
    vectors = {
        "detached": {
            "doc-a": {
                # inside the exact optimum (utility 1.0) but not the privacy maximum
                "lambda-zero": BC_VECTOR,
                # never cached under this document
                "lambda-max": PRIVATE_EXTRA_VECTOR,
            },
        },
        "coupled": {
            "doc-a": {
                "lambda-zero": KEEP_VECTOR,          # below the exact optimum
                "lambda-max": PLACEHOLDER_VECTOR,    # the lexicographic choice itself
            },
        },
    }
    report = _comparator_report(
        tmp_path, plans, vectors, metadata={"detached": {"sha256": "sha256:fixture"}},
    )

    detached = report["additive_comparators"]["detached"]
    assert detached["adjudicating"] is False
    assert detached["checkpoint"] == {"sha256": "sha256:fixture"}
    zero = detached["documents"]["doc-a"]["profiles"]["lambda-zero"]
    assert zero["status"] == "cache-hit"
    assert zero["inside_exact_optimal_set"] is True
    assert zero["chooses_count_max_inside_exact_set"] is False
    assert zero["utility_gap_to_exact_optimum"] == 0.0
    assert zero["count_gap_to_lexicographic"] == pytest.approx(0.4)
    assert detached["documents"]["doc-a"]["profiles"]["lambda-max"]["status"] == "cache-miss"
    assert detached["cache_hit_count"] == 1 and detached["cache_miss_count"] == 1
    assert detached["fraction_below_exact_optimum"] == 0.0
    assert detached["fraction_utility_feasible_missing_count_max"] == 1.0

    coupled = report["additive_comparators"]["coupled"]
    below = coupled["documents"]["doc-a"]["profiles"]["lambda-zero"]
    assert below["inside_exact_optimal_set"] is False
    assert below["utility_gap_to_exact_optimum"] == pytest.approx(0.5)
    assert coupled["fraction_below_exact_optimum"] == 0.5
    assert coupled["fraction_utility_feasible_missing_count_max"] == 0.0


def test_additive_comparator_output_never_changes_the_verdict_or_the_selection(tmp_path):
    plans = {
        f"doc-{index}": [(BC_VECTOR, 1.0), (KEEP_VECTOR, 0.5), (PLACEHOLDER_VECTOR, 1.0)]
        for index in range(3)
    }
    optimistic = {
        "arm": {
            doc_id: {"lambda-zero": PLACEHOLDER_VECTOR, "lambda-max": PLACEHOLDER_VECTOR}
            for doc_id in plans
        },
    }
    pessimistic = {
        "arm": {
            doc_id: {"lambda-zero": KEEP_VECTOR, "lambda-max": PRIVATE_EXTRA_VECTOR}
            for doc_id in plans
        },
    }
    first = _comparator_report(tmp_path / "optimistic", plans, optimistic)
    second = _comparator_report(tmp_path / "pessimistic", plans, pessimistic)

    assert first["additive_comparators"] != second["additive_comparators"]
    for key in ("verdict", "adjudication_checks", "documents", "summary"):
        assert first[key] == second[key]


def test_comparator_separates_below_equal_and_above_the_slate_optimum(tmp_path):
    """A comparator vector from the expanded cache can EXCEED the slate optimum;
    counting that as "not optimal" would report a utility loss that never happened."""
    plans = {
        "doc-a": [
            (BC_VECTOR, 0.5), (KEEP_VECTOR, 0.25), (PLACEHOLDER_VECTOR, 0.5),
            (PRIVATE_EXTRA_VECTOR, 1.0),   # adaptively cached, HIGHER utility
        ],
    }
    vectors = {
        "arm": {
            "doc-a": {
                "lambda-zero": KEEP_VECTOR,            # below
                "lambda-max": PRIVATE_EXTRA_VECTOR,    # above
            },
        },
    }
    report = _comparator_report(tmp_path, plans, vectors)
    block = report["additive_comparators"]["arm"]
    profiles = block["documents"]["doc-a"]["profiles"]

    assert profiles["lambda-zero"]["utility_relation_to_exact_optimum"] == "below"
    assert profiles["lambda-max"]["utility_relation_to_exact_optimum"] == "above"
    assert profiles["lambda-max"]["inside_exact_optimal_set"] is False
    assert profiles["lambda-max"]["utility_gap_to_exact_optimum"] < 0.0
    assert block["utility_relation_counts"] == {"below": 1, "equal": 0, "above": 1}
    assert block["fraction_below_exact_optimum"] == 0.5
    assert block["fraction_above_exact_optimum"] == 0.5
    # no utility-feasible vector, so the tie-ownership rate is undefined, not zero
    assert block["fraction_utility_feasible_missing_count_max"] is None
    assert block["documents_with_utility_loss"] == ["doc-a"]
    assert block["documents_missing_count_max_inside_exact_set"] == []


def test_expected_input_hash_mismatch_is_invalid_before_any_evaluation(tmp_path):
    plans = {"doc-a": [(vector, 1.0) for vector in SLATE]}
    artifact = _artifact(tuple(plans))
    rows = [_row("doc-a", vector, artifact, linked=1.0) for vector in SLATE]
    report = gate.run_gate(
        (_doc("doc-a"),),
        _cache(tmp_path, rows),
        artifact,
        _count_reward(),
        _count_state(),
        environment_hash=ENVIRONMENT_HASH,
        utility_artifact_hash=artifact["artifact_hash"],
        input_hashes={"utility_cache": "aa" * 32},
        expected_input_hashes={"utility_cache": "bb" * 32},
    )
    assert report["verdict"] == "invalid"
    assert report["documents"] == [] and report["summary"] == {}
    assert report["adjudication_checks"]["invalid_reasons"] == [
        f"utility_cache: expected {'bb' * 32} got {'aa' * 32}"
    ]


def test_greedy_comparator_vectors_evaluates_only_lambda_zero_and_lambda_max():
    from cloak.ranker.environment import LambdaProfile
    from cloak.tests.test_interactive_ranker import SequencePolicy

    document = _doc("doc-a")
    profiles = (
        LambdaProfile("lambda-zero", 0.0),
        LambdaProfile("lambda-mid", 0.5),
        LambdaProfile("lambda-max", 1.5),
    )
    policy = SequencePolicy([dict(BC_VECTOR), dict(KEEP_VECTOR)])
    vectors = gate.greedy_comparator_vectors(policy, (document,), profiles)

    assert set(vectors["doc-a"]) == {"lambda-zero", "lambda-max"}
    assert vectors["doc-a"]["lambda-zero"] == BC_VECTOR
    assert vectors["doc-a"]["lambda-max"] == KEEP_VECTOR


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
