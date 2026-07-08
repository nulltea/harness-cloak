"""Safety gates for proposed lattice levels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from cloak.anonymity import K_FLOORS
from cloak.lattice import is_type_name_phrase
from cloak.lattice_producer.vocabulary import CanonicalVocabulary
from cloak.runtime_types import FORCED_PLACEHOLDER_TYPES, PLACEHOLDER_RE

# empirically calibrated against real anchor labels, not the naive "near duplicate" intuition of
# ~0.8: two genuine paraphrases of the same 2-3 word concept (e.g. "pharmaceutical product" vs
# the anchored "pharmaceutical compound") typically score 0.3-0.4 token-Jaccard, since they only
# share one word out of a small union. 0.8 would never fire on real multi-word labels.
_VOCABULARY_NEAR_DUPLICATE_THRESHOLD = 0.3


@dataclass
class GateResult:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


_GENERIC_PROFESSION_LEVELS = {
    "worker",
    "professional worker",
    "technical worker",
    "production worker",
    "education worker",
    "arts and media worker",
    "business and financial occupation",
    "architecture and engineering occupation",
    "science occupation",
    "construction worker",
    "installation and repair worker",
    "transportation and material moving worker",
}


def _is_model_proposed(candidate: dict[str, Any]) -> bool:
    grounding = candidate.get("level_grounding") or {}
    return candidate.get("source_family") == "model-proposed" or grounding.get("status") == "model-proposed"


def _aliases_for(item: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    values = [*item.get("aliases", []), *candidate.get("aliases", [])]
    return [str(value).strip() for value in values if str(value).strip()]


def _has_model_evidence(candidate: dict[str, Any]) -> bool:
    grounding = candidate.get("level_grounding") or {}
    count_evidence = str(candidate.get("count_evidence") or grounding.get("count_evidence") or "").strip()
    rationale = str(candidate.get("rationale") or candidate.get("level_rationale") or "").strip()
    selector = str(candidate.get("selector") or grounding.get("selector") or "").strip()
    return bool(count_evidence and rationale and selector)


def _is_generic_profession_level(level: str) -> bool:
    text = _norm(level)
    if text in _GENERIC_PROFESSION_LEVELS:
        return True
    tokens = text.split()
    return len(tokens) <= 3 and (text.endswith(" worker") or text.endswith(" occupation") or text.endswith(" professional"))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


# The exact vocabulary that caught 12 confirmed model non-answers in the reviewed
# drug-health-procedure run (mcnuggates, camila, abcdes, ...): when the model has no real
# referent for a surface, it emits templated filler that satisfies "aliases are non-empty"
# (_aliases_for) without containing any actual identifying content.
_GENERIC_FILLER_TOKENS = {
    "clinical", "agent", "substance", "reference", "entry", "record", "formulation",
    "preparation", "product", "variant", "drug", "medication", "therapeutic", "pharmaceutical",
    "designated", "indexed", "target", "compound", "chemical", "medicinal", "molecule",
    "material", "biological", "code", "fragment",
}


def _is_generic_filler_alias_set(aliases: list[str]) -> bool:
    if len(aliases) <= 1:
        return False  # a single alias can't be checked against "all aliases are filler"
    for alias in aliases:
        tokens = _tokens(alias)
        if not tokens or not tokens <= _GENERIC_FILLER_TOKENS:
            return False
    return True


def _vocabulary_tokens(vocabulary: CanonicalVocabulary | None) -> set[str]:
    if vocabulary is None:
        return set()
    tokens: set[str] = set()
    for label in vocabulary.all_labels():
        tokens |= _tokens(label)
    return tokens


def gate_candidates(
    item: dict[str, Any], candidates: list[dict[str, Any]], *, proposed_out: str | None = None
) -> GateResult:
    runtime_type = item.get("runtime_type")
    surface = str(item.get("surface") or item.get("canonical_value") or "")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    model_candidates = [candidate for candidate in candidates if _is_model_proposed(candidate)]
    model_positions = {id(candidate): idx for idx, candidate in enumerate(model_candidates)}
    model_counts = [float(candidate.get("level_count", 1.0)) for candidate in model_candidates]
    flat_model_counts = len(model_counts) > 1 and len(set(model_counts)) == 1
    generic_only_model_chain = bool(model_candidates) and all(
        _is_generic_profession_level(str(candidate.get("level", ""))) for candidate in model_candidates
    )
    # proposed_out (the run's own in-progress output) grows this vocabulary with every level
    # already accepted THIS run, not just the ~40-label static anchor set -- without it, this
    # gate can only catch a duplicate of a hand-curated anchor, never a duplicate of a paraphrase
    # the model itself invented for an earlier item in the same run (the actual majority case).
    vocabulary = CanonicalVocabulary(str(runtime_type), proposed_out=proposed_out) if runtime_type else None
    domain_tokens = _vocabulary_tokens(vocabulary)
    # chain-wide, not per-level: a legitimate entry's broadest ("ceiling") level will almost
    # always overlap the seeded vocabulary even if a narrower level doesn't, so only flag when
    # NONE of the model's proposed levels for this item touch the domain at all -- exactly the
    # real "baseball"/["sport","game","human activity"] filed under health-condition case, and
    # "pizza burgers" filed under drug, both of which shared zero tokens with any seeded label.
    # Runtime types with no seeded vocabulary (domain_tokens empty) skip this check entirely --
    # an empty allowlist can't validate anything.
    no_domain_overlap_chain = (
        bool(model_candidates)
        and bool(domain_tokens)
        and not any(_tokens(str(candidate.get("level", ""))) & domain_tokens for candidate in model_candidates)
    )
    # backstop for short/ambiguous surfaces even when the model over-claims confidence: real
    # examples from the reviewed run (bun -> bunion instead of blood urea nitrogen, bph ->
    # hypertension instead of benign prostatic hyperplasia, cad -> cadmium derivative instead of
    # coronary artery disease) all had the model confidently resolve a short clinical
    # abbreviation to a single, sometimes wrong, referent. If we're in the model-proposed branch
    # at all, deterministic_lookup already found no real source for this surface (otherwise the
    # graph would never have called the model), so "no exact match in deterministic sources" is
    # already implied; only the vocabulary and length checks need to run here.
    surface_key = surface.replace(" ", "")
    short_ambiguous_surface = bool(surface_key) and len(surface_key) <= 4 and not (vocabulary and vocabulary.has_exact(surface))
    for candidate in candidates:
        level = str(candidate.get("level", "")).strip()
        record = {**candidate, "item_id": item.get("item_id"), "runtime_type": runtime_type}
        reason = None
        if runtime_type in FORCED_PLACEHOLDER_TYPES or runtime_type == "DEM":
            reason = "ineligible_runtime_type"
        elif PLACEHOLDER_RE.search(level):
            reason = "placeholder_terminal"
        elif surface and _norm(surface) in _norm(level):
            reason = "self_leak"
        elif re.search(r"\b\d{3,}\b", level):
            reason = "distinctive_number"
        elif is_type_name_phrase(level):
            reason = "type_name_phrase"
        if reason:
            rejected.append({**record, "reason": reason})
            continue
        grounding_status = (candidate.get("level_grounding") or {}).get("status")
        floor = float(K_FLOORS.get(str(runtime_type), 100.0))
        if _is_model_proposed(candidate):
            if candidate.get("surface_confidence") in {"low", "ambiguous"} or short_ambiguous_surface:
                diagnostics.append({**record, "reason": "low_confidence_surface"})
                continue
            if not _aliases_for(item, candidate):
                diagnostics.append({**record, "reason": "missing_aliases"})
                continue
            if _is_generic_filler_alias_set(_aliases_for(item, candidate)):
                diagnostics.append({**record, "reason": "generic_filler_aliases"})
                continue
            if no_domain_overlap_chain:
                diagnostics.append({**record, "reason": "no_domain_overlap"})
                continue
            if grounding_status != "model-proposed":
                diagnostics.append({**record, "reason": "missing_model_count"})
                continue
            if not _has_model_evidence(candidate):
                diagnostics.append({**record, "reason": "missing_model_evidence"})
                continue
            if flat_model_counts and (not generic_only_model_chain or model_positions.get(id(candidate), 0) == 0):
                diagnostics.append({**record, "reason": "flat_model_counts"})
                continue
            if generic_only_model_chain:
                diagnostics.append({**record, "reason": "weak_semantic_relevance"})
                continue
            if (
                vocabulary is not None
                and not vocabulary.has_exact(level)
                and not candidate.get("reused_canonical_label")
            ):
                near_duplicates = vocabulary.nearest(level, k=3, min_overlap=_VOCABULARY_NEAR_DUPLICATE_THRESHOLD)
                if near_duplicates:
                    diagnostics.append(
                        {**record, "reason": "unreused_near_duplicate_label", "near_duplicates": near_duplicates}
                    )
                    continue
        if grounding_status == "fail-closed":
            diagnostics.append({**record, "reason": (candidate.get("level_grounding") or {}).get("reason", "fail_closed")})
            continue
        if grounding_status != "proposal-universe" and float(candidate.get("level_count", 1.0)) < floor:
            diagnostics.append({**record, "reason": "below_floor"})
            continue
        accepted.append(record)
    return GateResult(accepted=accepted, rejected=rejected, diagnostics=diagnostics)
