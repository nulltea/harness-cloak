"""Epsilon-zero lexicographic gate — does the exact-utility tie set hold free count gain?

Cache-only, CPU, no reward calls. Plan: docs/plans/2026-08-04-epsilon-zero-lexicographic-gate.md
Pre-registration: research-wiki/experiments/2026-08-04-epsilon-zero-lexicographic-gate.md

The gate executes the lexicographic operator directly over already-measured complete
document vectors, so no tower logit, `alpha`, gain head, structural dependency label,
or 0.044 threshold is an input. Candidate support is the frozen five-anchor slate from
`build_anchor_trajectories` — never whichever vectors prior policies happened to cache,
because adaptive coverage is not a population and a cache miss is missing evidence
rather than zero opportunity.

Run:
  PYTHONPATH=src:scripts .venv/bin/python -u scripts/spikes/epsilon_zero_lexicographic_gate.py \
    --environment results/ranker_v2/environment/ranker-env.json \
    --utility-artifact results/ranker_v2/qa/aci-full.utility \
    --profile-count-targets results/ranker_v2/reward/profile-count-targets.json \
    --utility-cache results/ranker_v2/cache/utility-results.jsonl \
    --output results/ranker_v2/architecture/epsilon-zero-lexicographic-gate.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from cloak.ranker.environment import RankerDocument  # noqa: E402
from cloak.ranker.lexicographic import (  # noqa: E402
    LexicographicCandidate,
    VectorKey,
    exact_document_utility_key,
    select_epsilon_zero,
    select_utility_only,
)

# The documents that drove the controller campaign: reported separately, never
# adjudicating, and never used to tune a rule then presented as held-out.
CAMPAIGN_DOCUMENTS = ("aci/D2N005", "aci/D2N027", "aci/D2N031", "aci/D2N063")
BOOTSTRAP_SEED = 20260804
BOOTSTRAP_SAMPLES = 10_000
_PARITY_TOLERANCE = 1e-12


class GateVerdict(str, Enum):
    INVALID = "invalid"
    INSUFFICIENT_CANDIDATE_BREADTH = "insufficient-candidate-breadth"
    NO_OBSERVED_EXACT_OPPORTUNITY = "no-observed-exact-opportunity"
    INSUFFICIENT_PRIMARY_SUPPORT = "insufficient-primary-support"
    ADOPT_EXACT_LEXICOGRAPHIC_COMPOSITION = "adopt-exact-lexicographic-composition"


SupportStatus = Literal[
    "unsupported-missing-slate",
    "unsupported-no-count-contrast",
    "support-complete",
]


@dataclass(frozen=True)
class ExpectedSlateVector:
    vector_key: VectorKey
    sources: tuple[str, ...]
    privacy_key: Decimal


@dataclass(frozen=True)
class CandidateCorpusDocument:
    doc_id: str
    expected_slate: tuple[ExpectedSlateVector, ...]
    gate_candidates: tuple[LexicographicCandidate, ...]
    expanded_cache_candidates: tuple[LexicographicCandidate, ...]
    missing_expected_vectors: tuple[ExpectedSlateVector, ...]
    candidate_support_status: SupportStatus


# ── candidate corpus ────────────────────────────────────────────────────────────


def _vector_key(ordered_action_vector: Any) -> VectorKey:
    return tuple(
        (str(pair[0]), str(pair[1])) for pair in ordered_action_vector
    )


def _document_privacy_scores(document: RankerDocument, targets) -> dict[str, Decimal]:
    """Frozen per-action count score for every action of every policy decision.

    Same expression as the production diagnostic, memoised per document: the score
    is a property of the action, so recomputing it per cached vector would only cost
    time.
    """
    scores: dict[str, Decimal] = {}
    for decision in document.policy_decisions:
        for action in decision.actions:
            scores[action.action_id] = Decimal(str(float(
                targets.action_scores(decision.decision_id, (action.action_id,))[0]
            )))
    return scores


def _privacy_key(
    document: RankerDocument, vector_key: VectorKey, scores: dict[str, Decimal],
) -> Decimal:
    """Document count score: the mean over the document's active policy decisions.

    `ProfileCountTargets.selected_document_score` is deliberately not used — it
    expects the artifact-wide decision set rather than one document's vector.
    """
    selected = dict(vector_key)
    keys = [scores[selected[decision.decision_id]] for decision in document.policy_decisions]
    return sum(keys, Decimal(0)) / Decimal(len(keys))


def _legality_state(document: RankerDocument):
    from cloak.ranker.environment import _fixed_fill_claims, _occurrence_maps

    occurrence_by_id, _first_offsets = _occurrence_maps(document)
    return tuple(_fixed_fill_claims(document, occurrence_by_id))


def _is_legal_vector(
    document: RankerDocument, vector_key: VectorKey, reserved: tuple[str, ...],
) -> bool:
    """Replay the sequential claimed-fill / injectivity rules without any policy."""
    from cloak.ranker.environment import _action_by_id, _fill_key, legal_action_ids

    decision_by_id = {
        decision.decision_id: decision for decision in document.policy_decisions
    }
    claimed: dict[str, str] = {}
    for decision_id, action_id in vector_key:
        decision = decision_by_id.get(decision_id)
        if decision is None:
            return False
        if action_id not in legal_action_ids(decision, claimed, reserved):
            return False
        try:
            action = _action_by_id(decision, action_id)
        except ValueError:
            return False
        if action.mode == "level":
            assert action.fill is not None
            claimed.setdefault(_fill_key(action.fill), decision.decision_id)
    return True


def build_expected_slate(
    document: RankerDocument, targets, scores: dict[str, Decimal],
    reserved: tuple[str, ...],
) -> tuple[ExpectedSlateVector, ...]:
    """The frozen five-anchor standardized slate, before any outcome is read.

    Deliberately blind to utility outcomes, gains, tower logits, controller
    behaviour, and prior cache frequency: the anchors may consult frozen count
    targets only (the minimum-count anchor is defined by them).
    """
    from cloak.ranker.lambda_menu import build_anchor_trajectories

    slate: list[ExpectedSlateVector] = []
    for trajectory in build_anchor_trajectories(document, targets):
        vector_key = _vector_key(trajectory.ordered_action_vector)
        expected = tuple(
            decision.decision_id for decision in document.policy_decisions
        )
        if tuple(decision_id for decision_id, _ in vector_key) != expected:
            raise ValueError(f"anchor vector is incomplete for {document.doc_id}")
        if not _is_legal_vector(document, vector_key, reserved):
            raise ValueError(f"anchor vector is illegal for {document.doc_id}")
        slate.append(ExpectedSlateVector(
            vector_key=vector_key,
            sources=tuple(trajectory.sources),
            privacy_key=_privacy_key(document, vector_key, scores),
        ))
    return tuple(sorted(slate, key=lambda row: row.vector_key))


def load_candidate_corpus(
    documents: tuple[RankerDocument, ...],
    cache,
    utility_artifact: dict,
    targets,
    *,
    environment_hash: str,
    utility_artifact_hash: str,
) -> tuple[dict[str, CandidateCorpusDocument], dict[str, int]]:
    """Join the standardized slate to fully validated, legal, pinned cache rows."""
    from cloak.reward.utility_credit import document_utility

    by_id = {document.doc_id: document for document in documents}
    scores = {
        doc_id: _document_privacy_scores(document, targets)
        for doc_id, document in by_id.items()
    }
    reserved = {doc_id: _legality_state(document) for doc_id, document in by_id.items()}
    audit = {
        "unknown_document_excluded": 0,
        "reader_refresh_excluded": 0,
        "pin_mismatch_excluded": 0,
        "incomplete_vector_excluded": 0,
        "illegal_vector_excluded": 0,
        "utility_parity_excluded": 0,
        "validated_candidates": 0,
    }
    validated: dict[str, dict[VectorKey, LexicographicCandidate]] = {
        doc_id: {} for doc_id in by_id
    }
    for identity, result in cache.entries.values():
        doc_id = str(identity["doc_id"])
        document = by_id.get(doc_id)
        if document is None:
            audit["unknown_document_excluded"] += 1
            continue
        if bool(identity["reader_refresh"]):
            audit["reader_refresh_excluded"] += 1
            continue
        if (
            str(identity["environment_hash"]) != environment_hash
            or str(identity["utility_artifact_hash"]) != utility_artifact_hash
        ):
            audit["pin_mismatch_excluded"] += 1
            continue
        vector_key = _vector_key(identity["ordered_action_vector"])
        expected = tuple(
            decision.decision_id for decision in document.policy_decisions
        )
        if tuple(decision_id for decision_id, _ in vector_key) != expected:
            audit["incomplete_vector_excluded"] += 1
            continue
        if not _is_legal_vector(document, vector_key, reserved[doc_id]):
            audit["illegal_vector_excluded"] += 1
            continue
        recomputed = document_utility(result.component_scores, utility_artifact, doc_id)
        if abs(recomputed - float(result.utility)) > _PARITY_TOLERANCE:
            audit["utility_parity_excluded"] += 1
            continue
        candidate = LexicographicCandidate(
            doc_id=doc_id,
            vector_key=vector_key,
            utility_key=exact_document_utility_key(
                result.component_scores, utility_artifact, doc_id,
            ),
            utility=recomputed,
            privacy_key=_privacy_key(document, vector_key, scores[doc_id]),
            privacy_score=float(_privacy_key(document, vector_key, scores[doc_id])),
            result_hash=result.result_hash,
        )
        previous = validated[doc_id].get(vector_key)
        if previous is not None and (
            previous.result_hash != candidate.result_hash
            or previous.utility_key != candidate.utility_key
        ):
            raise ValueError(
                f"conflicting base cache results for {doc_id} {vector_key}: "
                f"{previous.result_hash} vs {candidate.result_hash}"
            )
        validated[doc_id][vector_key] = candidate
        audit["validated_candidates"] += 1

    corpus: dict[str, CandidateCorpusDocument] = {}
    for doc_id, document in by_id.items():
        slate = build_expected_slate(document, targets, scores[doc_id], reserved[doc_id])
        cached = validated[doc_id]
        gate = tuple(
            cached[row.vector_key] for row in slate if row.vector_key in cached
        )
        missing = tuple(row for row in slate if row.vector_key not in cached)
        expanded_keys = sorted(set(cached) - {row.vector_key for row in slate})
        if missing:
            status: SupportStatus = "unsupported-missing-slate"
        elif len({row.privacy_key for row in gate}) < 2:
            status = "unsupported-no-count-contrast"
        else:
            status = "support-complete"
        corpus[doc_id] = CandidateCorpusDocument(
            doc_id=doc_id,
            expected_slate=slate,
            gate_candidates=gate,
            expanded_cache_candidates=tuple(cached[key] for key in expanded_keys),
            missing_expected_vectors=missing,
            candidate_support_status=status,
        )
    return corpus, audit


# ── inputs ──────────────────────────────────────────────────────────────────────


def _bc_vector_key(corpus_document: CandidateCorpusDocument) -> VectorKey:
    matches = [
        row.vector_key for row in corpus_document.expected_slate
        if "behavior_cloning" in row.sources
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one BC anchor for {corpus_document.doc_id}")
    return matches[0]


# ── per-document adjudication ───────────────────────────────────────────────────


def _vector_hash(vector_key: VectorKey) -> str:
    from cloak.reward.utility_cache import stable_hash

    return stable_hash([[decision_id, action_id] for decision_id, action_id in vector_key])


def _selector_record(selection, *, vector_hash: bool = True) -> dict[str, Any]:
    candidate = selection.selected
    record = {
        "vector": [[decision_id, action_id] for decision_id, action_id in candidate.vector_key],
        "utility_key": str(candidate.utility_key),
        "utility": candidate.utility,
        "privacy_key": str(candidate.privacy_key),
        "privacy_score": candidate.privacy_score,
        "result_hash": candidate.result_hash,
    }
    if vector_hash:
        record["vector_hash"] = _vector_hash(candidate.vector_key)
    return record


def _count_provenance(
    vector_key: VectorKey, count_state: dict,
) -> dict[str, dict[str, Any]]:
    rows = count_state.get("action_targets", {})
    provenance = {}
    for _decision_id, action_id in vector_key:
        row = rows.get(action_id)
        if not isinstance(row, dict):
            raise ValueError(f"count state lacks provenance for action {action_id}")
        provenance[action_id] = {
            key: row.get(key)
            for key in ("mode", "profile_id", "grounding_status", "source_family")
        }
    return provenance


def _diagnostic_gains(
    candidate_document: CandidateCorpusDocument, bc_vector_key: VectorKey,
) -> dict[str, Any]:
    """Expanded-cache block: adaptively sampled vectors, explicitly non-adjudicating."""
    rows = (*candidate_document.gate_candidates, *candidate_document.expanded_cache_candidates)
    if not rows:
        return {"adjudicating": False, "candidate_vector_count": 0, "free_count_gain": None}
    lexicographic = select_epsilon_zero(rows)
    utility_only = select_utility_only(rows, bc_vector_key)
    return {
        "adjudicating": False,
        "candidate_vector_count": len(rows),
        "exact_optimal_set_size": lexicographic.feasible_count,
        "free_count_gain": float(
            lexicographic.selected.privacy_key - utility_only.selected.privacy_key
        ),
        "epsilon_zero_lexicographic": _selector_record(lexicographic),
        "utility_only": _selector_record(utility_only),
    }


def evaluate_document(
    document: RankerDocument,
    candidate_document: CandidateCorpusDocument,
    bc_vector_key: VectorKey,
    *,
    population: Literal["primary", "campaign"],
    count_state: dict | None = None,
) -> dict[str, Any]:
    """One support-aware gate record. Unsupported documents get `null`, never zero."""
    coverage = {
        "doc_id": candidate_document.doc_id,
        "population": population,
        "policy_decision_count": len(document.policy_decisions),
        "candidate_vector_count": len(candidate_document.gate_candidates),
        "expected_slate_size": len(candidate_document.expected_slate),
        "cached_slate_size": len(candidate_document.gate_candidates),
        "missing_expected_vectors": [
            {
                "vector": [[d, a] for d, a in row.vector_key],
                "sources": list(row.sources),
                "vector_hash": _vector_hash(row.vector_key),
            }
            for row in candidate_document.missing_expected_vectors
        ],
        "missing_anchor_sources": sorted({
            source
            for row in candidate_document.missing_expected_vectors
            for source in row.sources
        }),
        "candidate_support_status": candidate_document.candidate_support_status,
        "bc_vector_hash": _vector_hash(bc_vector_key),
        "expanded_cache_diagnostic": _diagnostic_gains(candidate_document, bc_vector_key),
    }
    if candidate_document.candidate_support_status != "support-complete":
        return {
            **coverage,
            "opportunity_status": None,
            "exact_optimal_set_size": None,
            "exact_optimal_privacy_min": None,
            "exact_optimal_privacy_baseline": None,
            "exact_optimal_privacy_max": None,
            "exact_optimal_privacy_spread": None,
            "free_count_gain": None,
            "selector_changes_baseline": None,
            "epsilon_zero_lexicographic": None,
            "utility_only": None,
            "selection_hamming_distance": None,
            "selected_count_provenance": None,
        }

    candidates = candidate_document.gate_candidates
    lexicographic = select_epsilon_zero(candidates)
    utility_only = select_utility_only(candidates, bc_vector_key)
    maximum = max(row.utility_key for row in candidates)
    gain = lexicographic.selected.privacy_key - utility_only.selected.privacy_key
    # Impossible under the definition; an assertion failure here is a defect, not a
    # finding (plan §5 failure matrix, row 1).
    assert lexicographic.selected.utility_key == maximum
    assert utility_only.selected.utility_key == maximum
    assert lexicographic.selected.privacy_key >= utility_only.selected.privacy_key
    assert gain >= 0
    return {
        **coverage,
        "opportunity_status": (
            "supported-opportunity" if gain > 0 else "supported-no-opportunity"
        ),
        "exact_optimal_set_size": lexicographic.feasible_count,
        "exact_optimal_privacy_min": float(lexicographic.feasible_privacy_min),
        "exact_optimal_privacy_baseline": float(utility_only.selected.privacy_key),
        "exact_optimal_privacy_max": float(lexicographic.feasible_privacy_max),
        "exact_optimal_privacy_spread": float(
            lexicographic.feasible_privacy_max - lexicographic.feasible_privacy_min
        ),
        "free_count_gain": float(gain),
        "selector_changes_baseline": (
            lexicographic.selected.vector_key != utility_only.selected.vector_key
        ),
        "epsilon_zero_lexicographic": _selector_record(lexicographic),
        "utility_only": _selector_record(utility_only),
        "selection_hamming_distance": sum(
            1 for (_, chosen), (_, baseline) in zip(
                lexicographic.selected.vector_key,
                utility_only.selected.vector_key,
                strict=True,
            )
            if chosen != baseline
        ),
        "selected_count_provenance": (
            _count_provenance(lexicographic.selected.vector_key, count_state)
            if count_state is not None else None
        ),
    }


def lower_mean_gain_bound(
    gains: list[float], *, seed: int = BOOTSTRAP_SEED, samples: int = BOOTSTRAP_SAMPLES,
) -> float:
    """One-sided 95% document-bootstrap lower bound (5th percentile of resample means)."""
    values = [float(value) for value in gains]
    if not values:
        raise ValueError("bootstrap requires at least one support-complete document")
    generator = random.Random(seed)
    size = len(values)
    means = sorted(
        st.fmean(generator.choices(values, k=size)) for _ in range(samples)
    )
    return means[int(0.05 * samples)]


def _distribution(values: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in sorted(values):
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _population_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in records if row["candidate_support_status"] == "support-complete"]
    gains = [float(row["free_count_gain"]) for row in supported]
    unsupported: dict[str, list[str]] = {}
    for row in records:
        status = row["candidate_support_status"]
        if status != "support-complete":
            unsupported.setdefault(status, []).append(row["doc_id"])
    summary = {
        "document_count": len(records),
        "support_complete_count": len(supported),
        "support_complete_fraction": (len(supported) / len(records)) if records else None,
        "unsupported_document_ids_by_reason": {
            status: sorted(ids) for status, ids in sorted(unsupported.items())
        },
        "positive_gain_document_count": sum(1 for value in gains if value > 0.0),
        "positive_gain_fraction": (
            sum(1 for value in gains if value > 0.0) / len(gains) if gains else None
        ),
        "mean_free_count_gain": st.fmean(gains) if gains else None,
        "median_free_count_gain": st.median(gains) if gains else None,
        "changed_vector_document_count": sum(
            1 for row in supported if row["selector_changes_baseline"]
        ),
        "exact_optimal_set_size_distribution": _distribution(
            [int(row["exact_optimal_set_size"]) for row in supported]
        ),
        "standardized_slate_coverage_distribution": _distribution(
            [int(row["cached_slate_size"]) for row in records]
        ),
        "expanded_cache_coverage_distribution": _distribution([
            int(row["expanded_cache_diagnostic"]["candidate_vector_count"])
            for row in records
        ]),
        "bootstrap": None,
    }
    if gains:
        summary["bootstrap"] = {
            "lower_bound_95_one_sided": lower_mean_gain_bound(gains),
            "observed_mean": st.fmean(gains),
            "document_count": len(gains),
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_SAMPLES,
        }
    return summary


def adjudicate(
    records: list[dict[str, Any]], *, invalid_reasons: list[str],
) -> tuple[GateVerdict, dict[str, Any]]:
    """Section 3.5, in order. Only primary documents can decide the verdict."""
    primary = [row for row in records if row["population"] == "primary"]
    supported = [
        row for row in primary if row["candidate_support_status"] == "support-complete"
    ]
    gains = [float(row["free_count_gain"]) for row in supported]
    lower_bound = lower_mean_gain_bound(gains) if gains else None
    checks = {
        "invalid_reasons": sorted(invalid_reasons),
        "primary_document_count": len(primary),
        "all_primary_documents_support_complete": bool(primary) and len(supported) == len(primary),
        "any_primary_document_has_positive_gain": any(value > 0.0 for value in gains),
        "any_primary_document_changes_vector": any(
            row["selector_changes_baseline"] for row in supported
        ),
        "primary_bootstrap_lower_bound": lower_bound,
        "primary_bootstrap_lower_bound_positive": bool(
            lower_bound is not None and lower_bound > 0.0
        ),
    }
    if invalid_reasons or not primary:
        return GateVerdict.INVALID, checks
    if not checks["all_primary_documents_support_complete"]:
        return GateVerdict.INSUFFICIENT_CANDIDATE_BREADTH, checks
    if not checks["any_primary_document_has_positive_gain"]:
        return GateVerdict.NO_OBSERVED_EXACT_OPPORTUNITY, checks
    if not checks["primary_bootstrap_lower_bound_positive"]:
        return GateVerdict.INSUFFICIENT_PRIMARY_SUPPORT, checks
    if not checks["any_primary_document_changes_vector"]:
        return GateVerdict.INSUFFICIENT_PRIMARY_SUPPORT, checks
    return GateVerdict.ADOPT_EXACT_LEXICOGRAPHIC_COMPOSITION, checks


def run_gate(
    documents: tuple[RankerDocument, ...],
    cache,
    utility_artifact: dict,
    targets,
    count_state: dict,
    *,
    environment_hash: str,
    utility_artifact_hash: str,
    input_hashes: dict[str, str] | None = None,
    comparators: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The complete cache-only gate report. Canonical: no timestamps, no wall time."""
    corpus, audit = load_candidate_corpus(
        documents, cache, utility_artifact, targets,
        environment_hash=environment_hash,
        utility_artifact_hash=utility_artifact_hash,
    )
    records: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    for document in sorted(documents, key=lambda row: row.doc_id):
        corpus_document = corpus[document.doc_id]
        population = "campaign" if document.doc_id in CAMPAIGN_DOCUMENTS else "primary"
        records.append(evaluate_document(
            document,
            corpus_document,
            _bc_vector_key(corpus_document),
            population=population,
            count_state=count_state,
        ))
    verdict, checks = adjudicate(records, invalid_reasons=invalid_reasons)
    primary = [row for row in records if row["population"] == "primary"]
    campaign = [row for row in records if row["population"] == "campaign"]
    return {
        "epsilon": "0",
        "remote_tasks": 0,
        "reader_work_items": 0,
        "inputs": {
            "environment_hash": environment_hash,
            "utility_artifact_hash": utility_artifact_hash,
            "count_target_artifact_hash": count_state.get("artifact_hash"),
            "file_sha256": dict(sorted((input_hashes or {}).items())),
        },
        "campaign_document_ids": list(CAMPAIGN_DOCUMENTS),
        "rejection_audit": audit,
        "documents": records,
        "summary": {
            "primary": _population_summary(primary),
            "campaign": _population_summary(campaign),
            "all_documents": _population_summary(records),
        },
        "additive_comparators": comparators or {},
        "adjudication_checks": checks,
        "verdict": verdict.value,
    }


def load_scoped_documents(
    environment_path: Path, utility_artifact: dict, count_state: dict,
) -> tuple[tuple[RankerDocument, ...], str]:
    """Production training scope: in-scope, count-covered menus; nonzero-signal docs."""
    from cloak.ranker.environment import load_ranker_environment
    from cloak.ranker.privacy import DirectCountPrivacyProvider
    from train_interactive_ranker import (
        _demote_out_of_scope_decisions,
        _drop_zero_signal_documents,
    )

    environment_artifact = json.loads(Path(environment_path).read_text())
    environment_hash = environment_artifact.get("frozen_environment", {}).get(
        "environment_hash"
    )
    if not isinstance(environment_hash, str) or not environment_hash:
        raise ValueError("ranker environment lacks environment_hash")
    if utility_artifact.get("environment_hash") != environment_hash:
        raise ValueError("utility artifact environment_hash differs from the environment")
    documents = tuple(load_ranker_environment(Path(environment_path)).values())
    documents, _demoted = _demote_out_of_scope_decisions(
        documents, DirectCountPrivacyProvider(count_state)
    )
    documents, _dropped = _drop_zero_signal_documents(documents, utility_artifact)
    documents = tuple(
        document for document in documents if document.policy_decisions
    )
    return documents, environment_hash
