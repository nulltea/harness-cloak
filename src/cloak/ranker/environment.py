"""Immutable loader for the published ranker-v2 environment, plus its renderer.

The renderer (`assemble_action_vector`, `legal_action_ids`) turns an action vector over
the frozen decisions into `doc_p` and the R schema `invert()` consumes. It sits here
rather than with the policy so `cloak.reward` can render rollouts without importing any
of `cloak.ranker`'s policy/training modules.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cloak.corpora import load_task_docs
from cloak.runtime_types import placeholder_token, placeholder_type_token


@dataclass(frozen=True)
class LambdaProfile:
    name: str
    value: float


ENVIRONMENT_VERSION = "ranker-v2-environment-v2"
FROZEN_ENVIRONMENT_VERSION = "occurrence-decisions-v2"


@dataclass(frozen=True)
class RankerAction:
    action_id: str
    mode: str
    fill: str | None
    authored_level_index: int | None
    runtime_type: str


@dataclass(frozen=True)
class RankerDecision:
    decision_id: str
    profile_id: str | None
    runtime_type: str
    canonical_key: str
    occurrence_ids: tuple[str, ...]
    actions: tuple[RankerAction, ...]


@dataclass(frozen=True)
class RankerDocument:
    doc_id: str
    corpus: str
    text: str
    occurrences: tuple[Mapping, ...]
    policy_decisions: tuple[RankerDecision, ...]
    fixed_decisions: tuple[RankerDecision, ...]


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _without(mapping: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in mapping.items() if name != key}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _load_source_documents(corpus: str) -> dict[str, str]:
    sources: dict[str, str] = {}
    for row in load_task_docs(corpus):
        source_id = _require_string(row.get("id"), f"{corpus} source id")
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"source text missing for {source_id}")
        if source_id in sources:
            raise ValueError(f"duplicate source document id: {source_id}")
        sources[source_id] = text
    return sources


def _validate_occurrences(
    doc_id: str,
    raw_occurrences: Any,
    text: str,
    global_occurrence_ids: set[str],
) -> tuple[tuple[Mapping, ...], dict[str, Mapping[str, Any]]]:
    if not isinstance(raw_occurrences, list):
        raise ValueError(f"occurrences must be a list for {doc_id}")
    occurrences: list[Mapping] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_occurrences:
        occurrence = _require_mapping(raw, f"occurrence in {doc_id}")
        occurrence_id = _require_string(
            occurrence.get("occurrence_id"), f"occurrence id in {doc_id}"
        )
        if occurrence_id in global_occurrence_ids:
            raise ValueError(f"duplicate occurrence id: {occurrence_id}")
        global_occurrence_ids.add(occurrence_id)
        by_id[occurrence_id] = occurrence

        start = occurrence.get("start")
        end = occurrence.get("end")
        surface = occurrence.get("surface")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
            or end > len(text)
            or not isinstance(surface, str)
            or text[start:end] != surface
        ):
            raise ValueError(
                f"source text mismatch for occurrence {occurrence_id} in {doc_id}"
            )
        occurrences.append(_freeze(occurrence))
    return tuple(occurrences), by_id


def _load_decision(
    doc_id: str,
    raw: Any,
    occurrence_by_id: Mapping[str, Mapping[str, Any]],
    global_decision_ids: set[str],
    global_action_ids: set[str],
) -> tuple[RankerDecision, bool]:
    decision = _require_mapping(raw, f"decision in {doc_id}")
    decision_id = _require_string(
        decision.get("decision_id"), f"decision id in {doc_id}"
    )
    if decision_id in global_decision_ids:
        raise ValueError(f"duplicate decision id: {decision_id}")
    global_decision_ids.add(decision_id)

    profile_id = decision.get("profile_id")
    if profile_id is not None and not isinstance(profile_id, str):
        raise ValueError(f"invalid profile_id for {decision_id}")
    runtime_type = _require_string(
        decision.get("runtime_type"), f"runtime_type for {decision_id}"
    )
    canonical_key = _require_string(
        decision.get("canonical_key"), f"canonical_key for {decision_id}"
    )

    raw_occurrence_ids = decision.get("occurrence_ids")
    if not isinstance(raw_occurrence_ids, list) or not raw_occurrence_ids:
        raise ValueError(f"missing mapped occurrence for {decision_id}")
    occurrence_ids: list[str] = []
    for occurrence_id in raw_occurrence_ids:
        if not isinstance(occurrence_id, str) or occurrence_id not in occurrence_by_id:
            raise ValueError(f"missing mapped occurrence for {decision_id}")
        if occurrence_id in occurrence_ids:
            raise ValueError(f"duplicate mapped occurrence for {decision_id}")
        occurrence = occurrence_by_id[occurrence_id]
        if occurrence.get("decision_id") != decision_id:
            raise ValueError(f"occurrence mapping mismatch for {decision_id}")
        occurrence_ids.append(occurrence_id)

    raw_actions = decision.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError(f"decision {decision_id} has no actions")
    actions: list[RankerAction] = []
    for raw_action in raw_actions:
        action = _require_mapping(raw_action, f"action for {decision_id}")
        action_id = _require_string(
            action.get("action_id"), f"action id for {decision_id}"
        )
        if action_id in global_action_ids:
            raise ValueError(f"duplicate action id: {action_id}")
        global_action_ids.add(action_id)
        if action.get("legal") is not True:
            raise ValueError(f"illegal action present in menu for {decision_id}")
        mode = _require_string(action.get("mode"), f"mode for {action_id}")
        if mode not in {"level", "keep", "placeholder"}:
            raise ValueError(f"unsupported action mode for {action_id}: {mode}")
        fill = action.get("fill")
        if fill is not None and not isinstance(fill, str):
            raise ValueError(f"invalid fill for {action_id}")
        if mode != "placeholder" and not fill:
            raise ValueError(f"missing fill for {action_id}")
        authored_level_index = action.get("authored_level_index")
        if authored_level_index is not None and (
            not isinstance(authored_level_index, int)
            or isinstance(authored_level_index, bool)
            or authored_level_index < 0
        ):
            raise ValueError(f"invalid authored_level_index for {action_id}")
        actions.append(
            RankerAction(
                action_id=action_id,
                mode=mode,
                fill=fill,
                authored_level_index=authored_level_index,
                runtime_type=runtime_type,
            )
        )

    ranker_selectable = decision.get("ranker_selectable")
    if not isinstance(ranker_selectable, bool):
        raise ValueError(f"missing ranker_selectable for {decision_id}")
    if not ranker_selectable and len(actions) != 1:
        raise ValueError(f"fixed decision {decision_id} must have exactly one action")
    return (
        RankerDecision(
            decision_id=decision_id,
            profile_id=profile_id,
            runtime_type=runtime_type,
            canonical_key=canonical_key,
            occurrence_ids=tuple(occurrence_ids),
            actions=tuple(actions),
        ),
        ranker_selectable,
    )


def load_ranker_environment(path: Path) -> dict[str, RankerDocument]:
    """Load, validate, and freeze a ranker-v2 environment artifact."""

    with Path(path).open(encoding="utf-8") as handle:
        artifact = _require_mapping(json.load(handle), "ranker environment")
    if artifact.get("artifact_version") != ENVIRONMENT_VERSION:
        raise ValueError(f"expected {ENVIRONMENT_VERSION}")

    frozen = _require_mapping(artifact.get("frozen_environment"), "frozen_environment")
    if frozen.get("artifact_version") != FROZEN_ENVIRONMENT_VERSION:
        raise ValueError(f"expected {FROZEN_ENVIRONMENT_VERSION}")
    if frozen.get("environment_hash") != _stable_hash(_without(frozen, "environment_hash")):
        raise ValueError("environment_hash mismatch")

    raw_documents = _require_mapping(frozen.get("documents"), "frozen documents")
    corpora = _require_mapping(artifact.get("corpora"), "corpora inventory")
    source_cache: dict[str, dict[str, str]] = {}
    result: dict[str, RankerDocument] = {}
    global_occurrence_ids: set[str] = set()
    global_decision_ids: set[str] = set()
    global_action_ids: set[str] = set()

    for doc_id, raw_document in raw_documents.items():
        doc_id = _require_string(doc_id, "document id")
        document = _require_mapping(raw_document, f"document {doc_id}")
        if document.get("environment_document_hash") != _stable_hash(
            _without(document, "environment_document_hash")
        ):
            raise ValueError(f"environment_document_hash mismatch for {doc_id}")
        corpus = _require_string(document.get("corpus"), f"corpus for {doc_id}")

        corpus_inventory = _require_mapping(
            corpora.get(corpus), f"corpora inventory for {corpus}"
        )
        inventory = _require_mapping(
            corpus_inventory.get(doc_id), f"corpora inventory for {doc_id}"
        )
        if (
            inventory.get("decisions") != document.get("decisions")
            or inventory.get("occurrences") != document.get("occurrences")
        ):
            raise ValueError(f"inventory mismatch for {doc_id}")

        if corpus not in source_cache:
            source_cache[corpus] = _load_source_documents(corpus)
        try:
            text = source_cache[corpus][doc_id]
        except KeyError as exc:
            raise ValueError(f"source text missing for {doc_id}") from exc

        occurrences, occurrence_by_id = _validate_occurrences(
            doc_id, document.get("occurrences"), text, global_occurrence_ids
        )
        raw_decisions = document.get("decisions")
        if not isinstance(raw_decisions, list):
            raise ValueError(f"decisions must be a list for {doc_id}")
        policy_decisions: list[RankerDecision] = []
        fixed_decisions: list[RankerDecision] = []
        decision_occurrence_ids: set[str] = set()
        for raw_decision in raw_decisions:
            loaded, selectable = _load_decision(
                doc_id,
                raw_decision,
                occurrence_by_id,
                global_decision_ids,
                global_action_ids,
            )
            decision_occurrence_ids.update(loaded.occurrence_ids)
            (policy_decisions if selectable else fixed_decisions).append(loaded)

        controlled_occurrence_ids = {
            occurrence_id
            for occurrence_id, occurrence in occurrence_by_id.items()
            if occurrence.get("controlled") is True
        }
        if controlled_occurrence_ids != decision_occurrence_ids:
            raise ValueError(f"occurrence mapping mismatch for {doc_id}")

        declared_policy_ids = inventory.get("policy_decision_ids")
        derived_policy_ids = {decision.decision_id for decision in policy_decisions}
        if (
            not isinstance(declared_policy_ids, list)
            or len(declared_policy_ids) != len(set(declared_policy_ids))
            or set(declared_policy_ids) != derived_policy_ids
        ):
            raise ValueError(f"policy decision ids mismatch for {doc_id}")

        def first_occurrence(decision: RankerDecision) -> tuple[int, str]:
            return (
                min(
                    occurrence_by_id[occurrence_id]["start"]
                    for occurrence_id in decision.occurrence_ids
                ),
                decision.decision_id,
            )

        policy_decisions.sort(key=first_occurrence)
        fixed_decisions.sort(key=first_occurrence)
        result[doc_id] = RankerDocument(
            doc_id=doc_id,
            corpus=corpus,
            text=text,
            occurrences=occurrences,
            policy_decisions=tuple(policy_decisions),
            fixed_decisions=tuple(fixed_decisions),
        )
    return result


# ── Rendering: an action vector -> doc_p + the R schema invert() consumes ──
# Lives with the environment, not the policy: cloak.reward renders rollouts and must
# not depend on cloak.ranker's policy/training modules.


def _fill_key(fill: str) -> str:
    return fill.lower()


def _action_by_id(decision: RankerDecision, action_id: str) -> RankerAction:
    matches = [action for action in decision.actions if action.action_id == action_id]
    if len(matches) != 1:
        raise ValueError(
            f"unknown or duplicate action {action_id!r} for {decision.decision_id}"
        )
    return matches[0]


def _occurrence_maps(
    document: RankerDocument,
) -> tuple[dict[str, Mapping], dict[str, tuple[int, str]]]:
    by_id: dict[str, Mapping] = {}
    for occurrence in document.occurrences:
        occurrence_id = occurrence.get("occurrence_id")
        if not isinstance(occurrence_id, str) or occurrence_id in by_id:
            raise ValueError(f"duplicate or invalid occurrence id in {document.doc_id}")
        start = occurrence.get("start")
        end = occurrence.get("end")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
            or end > len(document.text)
            or document.text[start:end] != occurrence.get("surface")
        ):
            raise ValueError(f"source occurrence mismatch for {occurrence_id}")
        by_id[occurrence_id] = occurrence

    first_offsets: dict[str, tuple[int, str]] = {}
    all_decisions = (*document.policy_decisions, *document.fixed_decisions)
    decision_ids: set[str] = set()
    action_ids: set[str] = set()
    mapped_ids: set[str] = set()
    for decision in all_decisions:
        if decision.decision_id in decision_ids:
            raise ValueError(f"duplicate decision id: {decision.decision_id}")
        decision_ids.add(decision.decision_id)
        if not decision.occurrence_ids:
            raise ValueError(f"missing mapped occurrence for {decision.decision_id}")
        starts = []
        for occurrence_id in decision.occurrence_ids:
            occurrence = by_id.get(occurrence_id)
            if occurrence is None:
                raise ValueError(f"missing mapped occurrence for {decision.decision_id}")
            if occurrence.get("decision_id") != decision.decision_id:
                raise ValueError(f"occurrence mapping mismatch for {decision.decision_id}")
            if occurrence_id in mapped_ids:
                raise ValueError(f"occurrence mapped more than once: {occurrence_id}")
            mapped_ids.add(occurrence_id)
            starts.append(int(occurrence["start"]))
        first_offsets[decision.decision_id] = (
            min(starts), decision.decision_id,
        )
        if not decision.actions:
            raise ValueError(f"decision {decision.decision_id} has no actions")
        for action in decision.actions:
            if action.action_id in action_ids:
                raise ValueError(f"duplicate action id: {action.action_id}")
            action_ids.add(action.action_id)
            if action.mode not in {"level", "keep", "placeholder"}:
                raise ValueError(f"unsupported action mode: {action.mode}")
            if action.mode == "level" and not action.fill:
                raise ValueError(f"level action {action.action_id} has no fill")

    policy_ids = {decision.decision_id for decision in document.policy_decisions}
    fixed_ids = {decision.decision_id for decision in document.fixed_decisions}
    if policy_ids & fixed_ids:
        raise ValueError("fixed decision present in policy set")
    for fixed in document.fixed_decisions:
        if len(fixed.actions) != 1:
            raise ValueError(f"fixed decision {fixed.decision_id} must have one action")

    for occurrence_id, occurrence in by_id.items():
        decision_id = occurrence.get("decision_id")
        if occurrence.get("controlled") is True and occurrence_id not in mapped_ids:
            raise ValueError(f"missing mapped occurrence for {decision_id}")
        if decision_id is not None and decision_id not in decision_ids:
            raise ValueError(f"unknown occurrence decision: {decision_id}")

    expected_policy_order = tuple(
        sorted(document.policy_decisions, key=lambda row: first_offsets[row.decision_id])
    )
    if document.policy_decisions != expected_policy_order:
        raise ValueError(f"unordered policy decisions in {document.doc_id}")
    return by_id, first_offsets


def legal_action_ids(
    decision: RankerDecision,
    claimed_fills: Mapping[str, str],
    reserved_fixed_fills: Collection[str],
) -> tuple[str, ...]:
    """Return the stable-ID menu after dynamic injectivity masking."""

    normalized_claims = {
        _fill_key(str(fill)): owner for fill, owner in claimed_fills.items()
    }
    reserved = {_fill_key(str(fill)) for fill in reserved_fixed_fills}
    legal: list[str] = []
    for action in decision.actions:
        if action.mode in {"keep", "placeholder"}:
            legal.append(action.action_id)
            continue
        if action.mode != "level" or not action.fill:
            raise ValueError(f"invalid action semantics for {action.action_id}")
        fill_key = _fill_key(action.fill)
        owner = normalized_claims.get(fill_key)
        if fill_key not in reserved and (owner is None or owner == decision.decision_id):
            legal.append(action.action_id)
    return tuple(legal)


def _fixed_fill_claims(
    document: RankerDocument, occurrence_by_id: Mapping[str, Mapping]
) -> dict[str, str]:
    claims: dict[str, str] = {}
    for decision in document.fixed_decisions:
        action = decision.actions[0]
        if action.mode == "placeholder":
            continue
        if action.mode == "level":
            fills = (action.fill,)
        else:
            fills = tuple(
                str(occurrence_by_id[occurrence_id]["surface"])
                for occurrence_id in decision.occurrence_ids
            )
        for fill in fills:
            if not fill:
                raise ValueError(f"fixed decision {decision.decision_id} has no exact fill")
            key = _fill_key(fill)
            owner = claims.setdefault(key, decision.decision_id)
            if owner != decision.decision_id:
                raise ValueError(f"fixed fill collision: {fill!r}")
    return claims


def _case_adjust(fill: str, text: str, start: int) -> str:
    previous = text[:start].rstrip()
    sentence_start = not previous or previous[-1] in ".!?\n"
    return (fill[0].upper() if sentence_start else fill[0].lower()) + fill[1:]


def assemble_action_vector(
    document: RankerDocument,
    action_vector: Mapping[str, str],
) -> tuple[str, list[dict]]:
    """Render one complete policy vector and all automatic fixed decisions."""

    occurrence_by_id, first_offsets = _occurrence_maps(document)
    expected_ids = {decision.decision_id for decision in document.policy_decisions}
    supplied_ids = set(action_vector)
    missing = sorted(expected_ids - supplied_ids)
    if missing:
        raise ValueError(f"action-vector omissions: {missing}")
    extra = sorted(supplied_ids - expected_ids)
    if extra:
        raise ValueError(f"unknown policy decision ids: {extra}")

    selected: dict[str, RankerAction] = {}
    for decision in document.policy_decisions:
        selected[decision.decision_id] = _action_by_id(
            decision, str(action_vector[decision.decision_id])
        )
    for decision in document.fixed_decisions:
        selected[decision.decision_id] = decision.actions[0]

    fixed_claims = _fixed_fill_claims(document, occurrence_by_id)
    claims = dict(fixed_claims)
    for decision in document.policy_decisions:
        action = selected[decision.decision_id]
        if action.mode != "level":
            continue
        assert action.fill is not None
        key = _fill_key(action.fill)
        owner = claims.setdefault(key, decision.decision_id)
        if owner != decision.decision_id:
            raise ValueError(
                f"action-vector fill collision for {decision.decision_id}: {action.fill!r}"
            )

    placeholder_by_decision: dict[str, str] = {}
    counters: dict[str, int] = {}
    all_decisions = sorted(
        (*document.policy_decisions, *document.fixed_decisions),
        key=lambda row: first_offsets[row.decision_id],
    )
    for decision in all_decisions:
        if selected[decision.decision_id].mode != "placeholder":
            continue
        token_type = placeholder_type_token(decision.runtime_type)
        counters[token_type] = counters.get(token_type, 0) + 1
        placeholder_by_decision[decision.decision_id] = placeholder_token(
            decision.runtime_type, counters[token_type]
        )

    runtime_types = {
        decision.decision_id: decision.runtime_type
        for decision in (*document.policy_decisions, *document.fixed_decisions)
    }
    replacements: list[dict] = []
    for occurrence in document.occurrences:
        decision_id = occurrence.get("decision_id")
        if decision_id is None:
            continue
        action = selected.get(str(decision_id))
        if action is None:
            raise ValueError(f"unresolved controlled occurrence decision {decision_id}")
        start = int(occurrence["start"])
        end = int(occurrence["end"])
        original = document.text[start:end]
        if action.mode == "keep":
            replacement = original
        elif action.mode == "placeholder":
            replacement = placeholder_by_decision[str(decision_id)]
        else:
            assert action.fill is not None
            replacement = _case_adjust(action.fill, document.text, start)
        replacements.append({
            "occurrence_id": occurrence["occurrence_id"],
            "decision_id": str(decision_id),
            "start": start,
            "end": end,
            "surface": original,
            "replacement": replacement,
            "action_id": action.action_id,
            "mode": action.mode,
            # invert()'s R schema: keep/placeholder pass through, level = generalize.
            "action": "generalize" if action.mode == "level" else action.mode,
            "type": runtime_types[str(decision_id)],
        })

    # Distinct surfaces sharing one fill inside a decision are ALIASES of one
    # canonical entity (cross-decision fill collisions raise above), so restoring
    # an echoed fill to any of its surfaces is entity-correct: extraction
    # restores every echo to the group's first surface. The legacy retain-generalization
    # rule treated this as unrecoverable ambiguity and blocked ~6% of restorations
    # (pre-RL audit R2 follow-up, 2026-07-27).
    ordered = sorted(replacements, key=lambda row: (row["start"], row["end"]))
    for left, right in zip(ordered, ordered[1:]):
        if right["start"] < left["end"]:
            raise ValueError("overlapping occurrences cannot be assembled")
    rendered = document.text
    for row in reversed(ordered):
        rendered = rendered[:row["start"]] + row["replacement"] + rendered[row["end"]:]
    return rendered, ordered
