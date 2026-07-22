"""Immutable loader for the published ranker-v2 environment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cloak.corpora import load_task_docs


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
