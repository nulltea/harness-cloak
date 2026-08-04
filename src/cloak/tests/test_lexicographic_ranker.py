"""Exact epsilon-zero lexicographic selection: equality, determinism, dominance."""
import random
from decimal import Decimal

import pytest

from cloak.ranker.lexicographic import (
    LexicographicCandidate,
    exact_document_utility_key,
    select_epsilon_zero,
    select_utility_only,
)
from cloak.tests.test_interactive_ranker import _utility_artifact


def candidate(action_id: str, utility: str, privacy: str):
    return LexicographicCandidate(
        doc_id="doc",
        vector_key=(("d", action_id),),
        utility_key=Decimal(utility),
        utility=float(utility),
        privacy_key=Decimal(privacy),
        privacy_score=float(privacy),
        result_hash=f"sha256:{action_id}",
    )


def pair_candidate(first: str, second: str, utility: str, privacy: str):
    return LexicographicCandidate(
        doc_id="doc",
        vector_key=(("d1", first), ("d2", second)),
        utility_key=Decimal(utility),
        utility=float(utility),
        privacy_key=Decimal(privacy),
        privacy_score=float(privacy),
        result_hash=f"sha256:{first}-{second}",
    )


def test_epsilon_zero_excludes_numerically_close_lower_utility():
    exact = candidate("specific", "1.000000000000", "0.1")
    lower = candidate("general", "0.999999999999", "1.0")
    selected = select_epsilon_zero((exact, lower))
    assert selected.selected.vector_key == exact.vector_key
    assert selected.feasible_count == 1


def test_epsilon_zero_maximizes_privacy_only_inside_exact_argmax():
    tied_low = candidate("specific", "1.0", "0.2")
    tied_high = candidate("general", "1.0", "0.8")
    assert select_epsilon_zero((tied_low, tied_high)).selected == tied_high


def test_selection_reports_the_feasible_privacy_interval():
    rows = (
        candidate("a", "1.0", "0.2"),
        candidate("b", "1.0", "0.8"),
        candidate("c", "0.5", "1.0"),
    )
    selection = select_epsilon_zero(rows)
    assert selection.feasible_count == 2
    assert selection.optimal_utility_key == Decimal("1.0")
    assert selection.feasible_privacy_min == Decimal("0.2")
    assert selection.feasible_privacy_max == Decimal("0.8")


def test_utility_only_is_privacy_blind_and_picks_the_bc_nearest_vector():
    bc = (("d1", "keep"), ("d2", "keep"))
    near = pair_candidate("keep", "level", "1.0", "0.1")
    far = pair_candidate("level", "level", "1.0", "0.9")
    selection = select_utility_only((far, near), bc)
    assert selection.selected == near
    assert select_epsilon_zero((far, near)).selected == far


def test_utility_only_never_leaves_the_exact_optimum_for_a_closer_vector():
    bc = (("d1", "keep"), ("d2", "keep"))
    optimal = pair_candidate("level", "level", "1.0", "0.1")
    closer_but_worse = pair_candidate("keep", "keep", "0.999999999999", "0.9")
    selection = select_utility_only((closer_but_worse, optimal), bc)
    assert selection.selected == optimal
    assert selection.feasible_count == 1


def test_equal_utility_and_privacy_resolve_by_stable_vector_key():
    first = candidate("aaa", "1.0", "0.5")
    second = candidate("bbb", "1.0", "0.5")
    assert select_epsilon_zero((second, first)).selected == first
    bc = (("d", "zzz"),)
    assert select_utility_only((second, first), bc).selected == first


def test_selection_is_invariant_under_seeded_input_permutations():
    rows = [
        pair_candidate("keep", "level", "1.0", "0.4"),
        pair_candidate("level", "keep", "1.0", "0.4"),
        pair_candidate("level", "level", "1.0", "0.7"),
        pair_candidate("keep", "keep", "0.9", "0.9"),
    ]
    bc = (("d1", "keep"), ("d2", "keep"))
    expected_lex = select_epsilon_zero(rows).selected
    expected_utility = select_utility_only(rows, bc).selected
    generator = random.Random(20260804)
    for _ in range(100):
        shuffled = list(rows)
        generator.shuffle(shuffled)
        assert select_epsilon_zero(shuffled).selected == expected_lex
        assert select_utility_only(shuffled, bc).selected == expected_utility


def test_empty_candidate_collection_is_rejected():
    with pytest.raises(ValueError, match="candidate"):
        select_epsilon_zero(())
    with pytest.raises(ValueError, match="candidate"):
        select_utility_only((), (("d", "keep"),))


def test_candidates_from_two_documents_are_rejected():
    other = LexicographicCandidate(
        doc_id="other",
        vector_key=(("d", "keep"),),
        utility_key=Decimal("1.0"),
        utility=1.0,
        privacy_key=Decimal("0.5"),
        privacy_score=0.5,
        result_hash="sha256:other",
    )
    with pytest.raises(ValueError, match="one document"):
        select_epsilon_zero((candidate("keep", "1.0", "0.5"), other))


def test_candidates_over_different_decision_sets_are_rejected():
    with pytest.raises(ValueError, match="decision"):
        select_epsilon_zero((
            candidate("keep", "1.0", "0.5"),
            pair_candidate("keep", "keep", "1.0", "0.5"),
        ))


@pytest.mark.parametrize("vector_key", [
    ((("d", "keep"), ("d", "level"))),
    ((("", "keep"),)),
    ((("d", ""),)),
    (()),
])
def test_malformed_vector_keys_are_rejected(vector_key):
    with pytest.raises(ValueError):
        LexicographicCandidate(
            doc_id="doc",
            vector_key=vector_key,
            utility_key=Decimal("1.0"),
            utility=1.0,
            privacy_key=Decimal("0.5"),
            privacy_score=0.5,
            result_hash="sha256:row",
        )


@pytest.mark.parametrize("utility,privacy", [
    (float("nan"), 0.5), (float("inf"), 0.5), (1.0, float("nan")),
])
def test_non_finite_float_summaries_are_rejected(utility, privacy):
    with pytest.raises(ValueError, match="finite"):
        LexicographicCandidate(
            doc_id="doc",
            vector_key=(("d", "keep"),),
            utility_key=Decimal("1.0"),
            utility=utility,
            privacy_key=Decimal("0.5"),
            privacy_score=privacy,
            result_hash="sha256:row",
        )


def test_bc_vector_with_a_different_decision_ordering_is_rejected():
    rows = (pair_candidate("keep", "level", "1.0", "0.4"),)
    with pytest.raises(ValueError, match="decision"):
        select_utility_only(rows, (("d2", "keep"), ("d1", "keep")))
    with pytest.raises(ValueError, match="decision"):
        select_utility_only(rows, (("d1", "keep"),))


def test_exact_utility_key_separates_differences_below_one_e_minus_nine():
    artifact = _utility_artifact()
    base = exact_document_utility_key(
        {"a-delivered": 1.0, "a-linked": 0.5}, artifact, "fixture/doc",
    )
    nudged = exact_document_utility_key(
        {"a-delivered": 1.0, "a-linked": 0.5 + 1e-13}, artifact, "fixture/doc",
    )
    assert base == Decimal("0.5")
    assert base != nudged
    assert abs(float(nudged - base)) < 1e-9


def test_exact_utility_key_ignores_non_policy_mass_and_fails_on_missing_policy_mass():
    artifact = _utility_artifact()
    # a-delivered carries residual (monitoring) mass: outside the weighted key.
    assert exact_document_utility_key(
        {"a-delivered": 0.0, "a-linked": 0.25}, artifact, "fixture/doc",
    ) == exact_document_utility_key(
        {"a-delivered": 1.0, "a-linked": 0.25}, artifact, "fixture/doc",
    )
    with pytest.raises(ValueError, match="a-linked"):
        exact_document_utility_key(
            {"a-delivered": 1.0}, artifact, "fixture/doc",
        )


def test_exact_utility_key_over_the_binding_denominator_is_the_document_utility():
    """The key is a numerator; dividing by the (constant) policy mass recovers the
    same scalar `document_utility` produces, so exact-set membership and the
    objective's own units cannot disagree."""
    from cloak.reward.utility_cache import utility_binding
    from cloak.reward.utility_credit import document_utility

    artifact = _utility_artifact()
    scores = {"a-delivered": 1.0, "a-linked": 0.75}
    binding = utility_binding(artifact, "fixture/doc")
    key = exact_document_utility_key(scores, artifact, "fixture/doc")
    assert float(key) / float(binding["utility_weight_denominator"]) == pytest.approx(
        document_utility(scores, artifact, "fixture/doc"), abs=1e-12,
    )
