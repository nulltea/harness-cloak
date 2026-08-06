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
        "rows_for_non_retained_documents": 0,
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
            audit["rows_for_non_retained_documents"] += 1
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
        "profile_count_key": str(candidate.privacy_key),
        "profile_count_score": candidate.privacy_score,
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
            "exact_optimal_count_min": None,
            "exact_optimal_count_baseline": None,
            "exact_optimal_count_max": None,
            "exact_optimal_count_spread": None,
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
        "exact_optimal_count_min": float(lexicographic.feasible_privacy_min),
        "exact_optimal_count_baseline": float(utility_only.selected.privacy_key),
        "exact_optimal_count_max": float(lexicographic.feasible_privacy_max),
        "exact_optimal_count_spread": float(
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
    expected_input_hashes: dict[str, str] | None = None,
    comparator_vectors: dict[str, dict[str, dict[str, VectorKey]]] | None = None,
    comparator_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The complete cache-only gate report. Canonical: no timestamps, no wall time."""
    hash_mismatches = sorted(
        f"{name}: expected {expected} got {(input_hashes or {}).get(name)}"
        for name, expected in (expected_input_hashes or {}).items()
        if (input_hashes or {}).get(name) != expected
    )
    if hash_mismatches:
        # Fail BEFORE any evaluation: a pin difference makes every downstream number
        # a measurement of a different experiment (plan §3.5 rule 1).
        return {
            "epsilon": "0",
            "remote_tasks": 0,
            "reader_work_items": 0,
            "inputs": {
                "environment_hash": environment_hash,
                "utility_artifact_hash": utility_artifact_hash,
                "count_target_artifact_hash": count_state.get("artifact_hash"),
                "file_sha256": dict(sorted((input_hashes or {}).items())),
                "expected_file_sha256": dict(sorted((expected_input_hashes or {}).items())),
            },
            "campaign_document_ids": list(CAMPAIGN_DOCUMENTS),
            "rejection_audit": {},
            "documents": [],
            "summary": {},
            "additive_comparators": {},
            "adjudication_checks": {"invalid_reasons": hash_mismatches},
            "verdict": GateVerdict.INVALID.value,
        }
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
    # Built AFTER the verdict inputs and never read by `adjudicate`: a cache miss is a
    # property of which vectors an archived controller happened to explore.
    comparators = build_comparator_report(
        corpus, records, comparator_vectors or {}, comparator_metadata or {},
    )
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
            "expected_file_sha256": dict(sorted((expected_input_hashes or {}).items())),
        },
        "campaign_document_ids": list(CAMPAIGN_DOCUMENTS),
        "rejection_audit": audit,
        "documents": records,
        "summary": {
            "primary": _population_summary(primary),
            "campaign": _population_summary(campaign),
            "all_documents": _population_summary(records),
        },
        "additive_comparators": comparators,
        "adjudication_checks": checks,
        "verdict": verdict.value,
    }


# ── archived additive-controller comparators (report-only) ──────────────────────


def load_comparator_policy(
    checkpoint_path: Path,
    documents: tuple[RankerDocument, ...],
    profiles: tuple,
    *,
    representation_manifest: Path,
    profile_count_targets: Path,
    environment_hash: str,
) -> tuple[Any, dict[str, Any]]:
    """Rebuild one archived policy on CPU over the FULL retained document set.

    Runtime-type indices are derived from the documents handed to the factory, so a
    narrower set would shift them and silently mis-embed every decision.
    """
    import hashlib
    from types import SimpleNamespace

    import torch

    from train_interactive_ranker import (
        _apply_controller_options,
        _semantic_training_policy,
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("policy_state_dict") or payload.get("state_dict")
    if state_dict is None:
        raise ValueError(f"checkpoint lacks a policy state dict: {checkpoint_path}")
    pinned = payload.get("artifact_pins", {}).get("environment_hash")
    if pinned is not None and str(pinned) != environment_hash:
        raise ValueError(f"checkpoint environment pin differs: {checkpoint_path}")
    config = dict(payload.get("training_config", {}))
    policy = _semantic_training_policy(
        SimpleNamespace(
            representation_manifest=str(representation_manifest),
            profile_count_targets=str(profile_count_targets),
            privacy_checkpoint=None,
            device="cpu",
        ),
        documents,
        profiles,
    )
    # Reconstruct EVERY archived controller option through the trainer's own helper,
    # not a hand-picked subset: the softcap changes the forward pass, so replaying
    # without it produces vectors that are not the archived policy's. The transform
    # tag is retagged by exactly the options that change forward semantics, so
    # comparing it against the checkpoint's own semantic contract turns any future
    # omission into a hard failure instead of a silently different policy.
    _apply_controller_options(policy, SimpleNamespace(**config))
    contract = dict(payload.get("semantic_contract", {}))
    expected_transform = contract.get("controller_transform")
    rebuilt_transform = getattr(policy, "controller_transform", None)
    if expected_transform is not None and rebuilt_transform != expected_transform:
        raise ValueError(
            f"replayed controller transform differs for {checkpoint_path}: "
            f"{rebuilt_transform!r} != {expected_transform!r}"
        )
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.float()
    metadata = {
        "path": str(checkpoint_path),
        "sha256": "sha256:" + hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_version": payload.get("checkpoint_version"),
        "policy_architecture": payload.get("policy_architecture"),
        "architecture_pin": payload.get("architecture_pin"),
        "code_revision": payload.get("code_revision"),
        "epoch": payload.get("epoch"),
        "environment_hash": pinned,
        "controller_gain": config.get("controller_gain"),
        "count_to_gain": config.get("count_to_gain"),
        "utility_logit_softcap": config.get("utility_logit_softcap"),
        "controller_gap_scaling": config.get("controller_gap_scaling"),
        "controller_transform": rebuilt_transform,
        "controller_transform_pin": expected_transform,
    }
    return policy, metadata


def greedy_comparator_vectors(
    policy: Any, documents: tuple[RankerDocument, ...], profiles: tuple,
) -> dict[str, dict[str, VectorKey]]:
    """Greedy lambda-zero and lambda-max vectors, through the production menu paths."""
    from cloak.ranker.interactive import sample_trajectory

    evaluated = (profiles[0], profiles[-1])
    vectors: dict[str, dict[str, VectorKey]] = {}
    for document in documents:
        for profile in evaluated:
            trajectory = sample_trajectory(
                policy, document, profile, greedy=True, generator=None,
            )
            vectors.setdefault(document.doc_id, {})[profile.name] = tuple(
                (step.decision_id, step.selected_action_id) for step in trajectory.steps
            )
    return vectors


def score_comparator_vector(
    corpus_document: CandidateCorpusDocument,
    record: dict[str, Any],
    vector_key: VectorKey,
) -> dict[str, Any]:
    """One report-only row. Never feeds the verdict: cache misses differ by checkpoint."""
    row: dict[str, Any] = {
        "vector": [[decision_id, action_id] for decision_id, action_id in vector_key],
        "vector_hash": _vector_hash(vector_key),
    }
    if corpus_document.candidate_support_status != "support-complete":
        return {**row, "status": "unsupported-document"}
    validated = {
        candidate.vector_key: candidate
        for candidate in (
            *corpus_document.gate_candidates,
            *corpus_document.expanded_cache_candidates,
        )
    }
    candidate = validated.get(vector_key)
    if candidate is None:
        return {**row, "status": "cache-miss"}
    optimum = max(item.utility_key for item in corpus_document.gate_candidates)
    # A comparator vector may come from the expanded cache and therefore EXCEED the
    # standardized-slate optimum. "not inside the exact optimal set" is not the same
    # claim as "loses utility", so the relation is reported three ways and the
    # aggregates count strictly-below separately from above.
    if candidate.utility_key == optimum:
        relation = "equal"
    elif candidate.utility_key < optimum:
        relation = "below"
    else:
        relation = "above"
    count_max = max(
        item.privacy_key for item in corpus_document.gate_candidates
        if item.utility_key == optimum
    )
    return {
        **row,
        "status": "cache-hit",
        "in_standardized_slate": vector_key in {
            item.vector_key for item in corpus_document.gate_candidates
        },
        "utility": candidate.utility,
        "utility_relation_to_exact_optimum": relation,
        "utility_gap_to_exact_optimum": float(record["epsilon_zero_lexicographic"]["utility"])
        - candidate.utility,
        "profile_count_score": candidate.privacy_score,
        "count_gap_to_lexicographic": float(
            record["epsilon_zero_lexicographic"]["profile_count_key"]
        ) - candidate.privacy_score,
        "inside_exact_optimal_set": relation == "equal",
        "chooses_count_max_inside_exact_set": bool(
            relation == "equal" and candidate.privacy_key == count_max
        ),
    }


def build_comparator_report(
    corpus: dict[str, CandidateCorpusDocument],
    records: list[dict[str, Any]],
    vectors_by_label: dict[str, dict[str, dict[str, VectorKey]]],
    metadata_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_id = {row["doc_id"]: row for row in records}
    report: dict[str, Any] = {}
    for label in sorted(vectors_by_label):
        rows: dict[str, dict[str, Any]] = {}
        for doc_id in sorted(vectors_by_label[label]):
            record = by_id.get(doc_id)
            if record is None:
                continue
            rows[doc_id] = {
                "population": record["population"],
                "profiles": {
                    profile_name: score_comparator_vector(
                        corpus[doc_id], record, vector_key,
                    )
                    for profile_name, vector_key in sorted(
                        vectors_by_label[label][doc_id].items()
                    )
                },
            }
        hits = [
            row
            for document in rows.values()
            for row in document["profiles"].values()
            if row["status"] == "cache-hit"
        ]
        feasible = [row for row in hits if row["inside_exact_optimal_set"]]
        below = [
            row for row in hits
            if row["utility_relation_to_exact_optimum"] == "below"
        ]
        missing_max = [
            row for row in feasible if not row["chooses_count_max_inside_exact_set"]
        ]
        # Vector-level fractions are what §3.4 asks for, but the document is the
        # statistical unit everywhere else, so both are reported and neither is
        # allowed to stand alone.
        documents_with_loss = sorted({
            doc_id for doc_id, document in rows.items()
            if any(
                row.get("utility_relation_to_exact_optimum") == "below"
                for row in document["profiles"].values()
            )
        })
        documents_missing_max = sorted({
            doc_id for doc_id, document in rows.items()
            if any(
                row["status"] == "cache-hit"
                and row["inside_exact_optimal_set"]
                and not row["chooses_count_max_inside_exact_set"]
                for row in document["profiles"].values()
            )
        })
        report[label] = {
            "adjudicating": False,
            "checkpoint": metadata_by_label.get(label, {}),
            "documents": rows,
            "cache_hit_count": len(hits),
            "cache_miss_count": sum(
                1
                for document in rows.values()
                for row in document["profiles"].values()
                if row["status"] == "cache-miss"
            ),
            "utility_relation_counts": {
                relation: sum(
                    1 for row in hits
                    if row["utility_relation_to_exact_optimum"] == relation
                )
                for relation in ("below", "equal", "above")
            },
            "fraction_below_exact_optimum": (
                len(below) / len(hits) if hits else None
            ),
            "fraction_above_exact_optimum": (
                sum(
                    1 for row in hits
                    if row["utility_relation_to_exact_optimum"] == "above"
                ) / len(hits) if hits else None
            ),
            "fraction_utility_feasible_missing_count_max": (
                len(missing_max) / len(feasible) if feasible else None
            ),
            "documents_with_utility_loss": documents_with_loss,
            "documents_missing_count_max_inside_exact_set": documents_missing_max,
        }
    return report


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


def _file_hashes(paths: dict[str, Path]) -> dict[str, str]:
    import hashlib

    return {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in sorted(paths.items())
    }


def _lambda_profiles(lambda_menu_path: Path) -> tuple:
    from cloak.ranker.environment import LambdaProfile

    menu = json.loads(Path(lambda_menu_path).read_text())
    return tuple(
        LambdaProfile(name, float(value))
        for name, value in zip(menu["profile_names"], menu["values"], strict=True)
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--utility-artifact", required=True)
    parser.add_argument("--profile-count-targets", required=True)
    parser.add_argument("--utility-cache", required=True)
    parser.add_argument(
        "--checkpoint", action="append", default=[], metavar="LABEL=PATH",
        help="archived additive controller, report-only; repeatable",
    )
    parser.add_argument(
        "--representation-manifest",
        default="results/ranker_v2/architecture/representation-full/manifest.json",
    )
    parser.add_argument(
        "--lambda-menu", default="results/ranker_v2/preflight/lambda-menu.json",
    )
    parser.add_argument(
        "--expect-sha256", action="append", default=[], metavar="NAME=HEX",
        help="preregistered input hash; a mismatch is INVALID before any evaluation",
    )
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--preflight-primary-docs", type=int, default=None)
    parser.add_argument("--output")
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args(argv)


def _evaluation_population(
    documents: tuple[RankerDocument, ...],
    corpus: dict[str, CandidateCorpusDocument],
    args: argparse.Namespace,
) -> tuple[RankerDocument, ...]:
    """Preflight scoping: chosen from support status ONLY, before any gain is read."""
    if not args.doc_id and args.preflight_primary_docs is None:
        return documents
    selected = list(dict.fromkeys(args.doc_id))
    if args.preflight_primary_docs is not None:
        eligible = [
            doc_id for doc_id in sorted(corpus)
            if doc_id not in CAMPAIGN_DOCUMENTS
            and corpus[doc_id].candidate_support_status == "support-complete"
            and doc_id not in selected
        ]
        if len(eligible) < args.preflight_primary_docs:
            raise SystemExit(
                f"{GateVerdict.INSUFFICIENT_CANDIDATE_BREADTH.value}: only "
                f"{len(eligible)} support-complete non-campaign documents exist"
            )
        selected.extend(eligible[: args.preflight_primary_docs])
    chosen = set(selected)
    unknown = sorted(chosen - set(corpus))
    if unknown:
        raise SystemExit(f"unknown document ids: {unknown}")
    return tuple(document for document in documents if document.doc_id in chosen)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from cloak.ranker.profile_count import ProfileCountTargets
    from cloak.reward.utility_cache import UtilityCache

    utility_artifact = json.loads(Path(args.utility_artifact).read_text())
    count_state = json.loads(Path(args.profile_count_targets).read_text())
    targets = ProfileCountTargets.from_artifact(count_state)
    documents, environment_hash = load_scoped_documents(
        Path(args.environment), utility_artifact, count_state,
    )
    utility_artifact_hash = str(utility_artifact["artifact_hash"])
    cache = UtilityCache(Path(args.utility_cache))
    print(f"retained documents {len(documents)}  cache entries {len(cache.entries)}")

    corpus, _audit = load_candidate_corpus(
        documents, cache, utility_artifact, targets,
        environment_hash=environment_hash,
        utility_artifact_hash=utility_artifact_hash,
    )
    unsupported = [
        (doc_id, corpus[doc_id].candidate_support_status, sorted({
            source
            for row in corpus[doc_id].missing_expected_vectors
            for source in row.sources
        }))
        for doc_id in sorted(corpus)
        if doc_id not in CAMPAIGN_DOCUMENTS
        and corpus[doc_id].candidate_support_status != "support-complete"
    ]
    print(f"unsupported primary documents {len(unsupported)}")
    for row in unsupported[:5]:
        print(f"  {row[0]}  {row[1]}  missing anchors: {row[2]}")

    evaluated = _evaluation_population(documents, corpus, args)
    if len(evaluated) != len(documents):
        print(f"evaluation population restricted to {len(evaluated)} documents: "
              f"{sorted(document.doc_id for document in evaluated)}")

    comparator_vectors: dict[str, dict[str, dict[str, VectorKey]]] = {}
    comparator_metadata: dict[str, dict[str, Any]] = {}
    if args.checkpoint:
        profiles = _lambda_profiles(Path(args.lambda_menu))
        for entry in args.checkpoint:
            label, _, path = entry.partition("=")
            if not label or not path:
                raise SystemExit(f"--checkpoint expects LABEL=PATH, got {entry!r}")
            policy, metadata = load_comparator_policy(
                Path(path), documents, profiles,
                representation_manifest=Path(args.representation_manifest),
                profile_count_targets=Path(args.profile_count_targets),
                environment_hash=environment_hash,
            )
            comparator_vectors[label] = greedy_comparator_vectors(
                policy, evaluated, profiles,
            )
            comparator_metadata[label] = metadata
            print(f"replayed comparator {label} over {len(evaluated)} documents")

    hashed = {
        "environment": Path(args.environment),
        "utility_artifact": Path(args.utility_artifact),
        "profile_count_targets": Path(args.profile_count_targets),
        "utility_cache": Path(args.utility_cache),
    }
    if args.checkpoint:
        # Comparator replay depends on these two as much as on the frozen artifacts.
        hashed["representation_manifest"] = Path(args.representation_manifest)
        hashed["lambda_menu"] = Path(args.lambda_menu)
    expected: dict[str, str] = {}
    for entry in args.expect_sha256:
        name, _, digest = entry.partition("=")
        if not name or not digest:
            raise SystemExit(f"--expect-sha256 expects NAME=HEX, got {entry!r}")
        if name not in hashed:
            raise SystemExit(
                f"--expect-sha256 names an unknown input {name!r}; known: {sorted(hashed)}"
            )
        expected[name] = digest.strip().lower()

    report = run_gate(
        evaluated, cache, utility_artifact, targets, count_state,
        environment_hash=environment_hash,
        utility_artifact_hash=utility_artifact_hash,
        input_hashes=_file_hashes(hashed),
        expected_input_hashes=expected,
        comparator_vectors=comparator_vectors,
        comparator_metadata=comparator_metadata,
    )
    _print_summary(report)
    payload = json.dumps(report, indent=1, sort_keys=True) + "\n"
    if args.stdout or not args.output:
        print(payload)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload)
        print(f"wrote {destination}")
    return 0


def _print_summary(report: dict[str, Any]) -> None:
    print(f"\n=== verdict: {report['verdict']} ===")
    if report["verdict"] == GateVerdict.INVALID.value:
        for reason in report["adjudication_checks"]["invalid_reasons"]:
            print(f"  INVALID: {reason}")
        return
    print(f"  rejection audit: {report['rejection_audit']}")
    for name in ("primary", "campaign"):
        summary = report["summary"][name]
        bootstrap = summary["bootstrap"]
        print(f"  {name}: {summary['document_count']} documents, "
              f"support-complete {summary['support_complete_count']}, "
              f"positive gain {summary['positive_gain_document_count']}, "
              f"mean G {summary['mean_free_count_gain']}, "
              f"median G {summary['median_free_count_gain']}")
        if bootstrap is not None:
            print(f"    bootstrap 95% one-sided lower bound "
                  f"{bootstrap['lower_bound_95_one_sided']:.6f} "
                  f"over {bootstrap['document_count']} documents")
        print(f"    exact-optimal-set sizes {summary['exact_optimal_set_size_distribution']}")
        print(f"    standardized-slate coverage {summary['standardized_slate_coverage_distribution']}")
    for label, block in report["additive_comparators"].items():
        print(f"  comparator {label}: hits {block['cache_hit_count']}, "
              f"misses {block['cache_miss_count']}, "
              f"utility relation {block['utility_relation_counts']}, "
              f"below optimum {block['fraction_below_exact_optimum']}, "
              f"feasible-but-not-max-count "
              f"{block['fraction_utility_feasible_missing_count_max']}")
        print(f"    transform {block['checkpoint'].get('controller_transform')} "
              f"(pin {block['checkpoint'].get('controller_transform_pin')}), "
              f"softcap {block['checkpoint'].get('utility_logit_softcap')}")
        print(f"    documents with utility loss "
              f"{len(block['documents_with_utility_loss'])}, "
              f"missing count max {len(block['documents_missing_count_max_inside_exact_set'])}")


if __name__ == "__main__":
    raise SystemExit(main())
