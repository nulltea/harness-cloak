"""QA-builder v2 artifact weighting, support anchors, validation, and scoring."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from cloak.train.reward import QA_BASE_URL, QA_MODEL, canon, fact_score

CONTEXT_READER_PIN_VERSION = "qa-context-reader-v1"
CONTEXT_READER_PROMPT_VERSION = "qa-context-batch-prompt-v1"
CONTEXT_READER_RESPONSE_SCHEMA_VERSION = "qa-context-answers-v1"
CONTEXT_READER_MAX_TOKENS = 512
CONTEXT_READER_PROMPT = (
    "Answer every question using the document. Preserve semantic category and "
    "function distinctions stated or entailed by the document. If an answer is not "
    "supported, use NONE. Return only a JSON object with an answers array in the "
    "same order as the questions.\n\n"
    "DOCUMENT:\n{context}\n\nQUESTIONS:\n{questions_json}"
)
CONTEXT_READER_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["answers"],
    "properties": {
        "answers": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}
CONTEXT_READER_RESPONSE_FORMAT = {"type": "json_object"}
CONTEXT_READER_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
BUILDER_PIN = {"builder": "qa-builder-v2", "version": "assertion-compiler-v5"}
UTILITY_SCORER_PIN_VERSION = "qa-utility-scorer-v4"
THRESHOLD_MANIFEST_SCHEMA = "qa-threshold-manifest-v1"

RELATION_TEACHER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
RELATION_TEACHER_BASE_URL = "https://openrouter.ai/api/v1"
RELATION_TEACHER_PROMPT_VERSION = "qa-relation-teacher-prompt-v2"
RELATION_TEACHER_RESPONSE_SCHEMA_VERSION = "qa-relation-proposals-v1"
RELATION_ONTOLOGY = (
    "treated_with",
    "monitored_by",
    "contraindicated_because_of",
    "causes_or_explains",
    "referred_to",
    "has_status",
    "has_category",
)
RELATION_TEACHER_OUTPUT_SCHEMA = {
    "relations": [{
        "relation": f"one of: {', '.join(RELATION_ONTOLOGY)}",
        "argument_occurrence_ids": ["existing occurrence IDs"],
        "support_properties": {
            "occurrence ID": "one exact legal_support_properties value"
        },
        "answer_occurrence_id": "one argument occurrence ID",
        "answer_property": "that occurrence's exact selected support property",
        "question": "natural question not containing a protected surface or its answer",
        "evidence_quote": "one exact source substring directly connecting all arguments",
    }]
}
RELATION_TEACHER_PROMPT_TEMPLATE = (
    "Extract only explicit, task-relevant clinical relations from the source. "
    "Use the closed relation vocabulary and existing occurrence IDs. Select support "
    "properties verbatim from the inventory. Do not use medical knowledge absent from "
    "the source. Abstain with an empty relations list when evidence is insufficient. "
    "Return only the requested JSON object.\n\n"
    "DOCUMENT ID:\n{doc_id}\n\n"
    "OCCURRENCE INVENTORY:\n{inventory_json}\n\n"
    "OUTPUT SCHEMA:\n{schema_json}\n\n"
    "AUTHORITATIVE REFERENCE EVIDENCE:\n{authoritative_reference}\n\n"
    "SOURCE DOCUMENT:\n{document}"
)

_RUNTIME_TYPE_CLASSES = {
    "health-condition": "condition",
    "condition": "condition",
    "diagnosis": "condition",
    "disease": "condition",
    "symptom": "symptom",
    "drug": "treatment",
    "medication": "treatment",
    "treatment": "treatment",
    "procedure": "procedure",
    "lab": "monitoring",
    "test": "monitoring",
    "monitoring": "monitoring",
    "provider": "provider",
    "specialty": "provider",
    "status": "status",
    "category": "category",
}
_RELATION_ARGUMENT_CLASSES = {
    "treated_with": (("condition",), ("treatment", "procedure")),
    "monitored_by": (("condition",), ("monitoring", "procedure", "provider")),
    "contraindicated_because_of": (("treatment", "procedure"), ("condition",)),
    "causes_or_explains": (("condition",), ("condition", "symptom")),
    "referred_to": (("condition",), ("provider", "procedure")),
    "has_status": (("condition", "symptom", "treatment", "procedure"), ("status",)),
    "has_category": (("condition", "symptom", "treatment", "procedure"), ("category",)),
}
ACI_RELATION_CONTRACT = {
    "treated_with": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["treated_with"],
        "cues": ("treated with",),
    },
    "monitored_by": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["monitored_by"],
        "cues": ("monitored by",),
    },
    "contraindicated_because_of": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["contraindicated_because_of"],
        "cues": ("contraindicated because of",),
    },
    "causes_or_explains": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["causes_or_explains"],
        "cues": ("causes", "explains"),
    },
    "referred_to": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["referred_to"],
        "cues": ("referred to",),
    },
    "has_status": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["has_status"],
        "cues": ("has status", "status is"),
    },
    "has_category": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["has_category"],
        "cues": ("has category", "category is"),
    },
}
_NEGATION_PATTERN = re.compile(r"\b(?:no|not|without|denies|denied)\b")
_LEAKAGE_GENERIC_TOKENS = {
    "answer", "category", "clinic", "condition", "disease", "doctor", "documented",
    "hospital", "medication", "monitoring", "option", "procedure", "provider", "status",
    "symptom", "treatment", "type", "used", "which", "what", "where", "when",
}


def context_reader_pin() -> dict:
    """Return the complete live identity of the artifact context reader."""
    return {
        "pin_version": CONTEXT_READER_PIN_VERSION,
        "model": QA_MODEL,
        "base_url": QA_BASE_URL,
        "prompt": {
            "version": CONTEXT_READER_PROMPT_VERSION,
            "sha256": _stable_hash(CONTEXT_READER_PROMPT),
        },
        "response_schema": {
            "version": CONTEXT_READER_RESPONSE_SCHEMA_VERSION,
            "schema": json.loads(json.dumps(CONTEXT_READER_RESPONSE_SCHEMA)),
        },
        "decoding": {
            "temperature": 0.0,
            "max_tokens": CONTEXT_READER_MAX_TOKENS,
            "enable_thinking": False,
            "response_format": dict(CONTEXT_READER_RESPONSE_FORMAT),
        },
    }


def builder_pin() -> dict:
    return {**BUILDER_PIN, "implementation_sha256": _qa_builder_source_digest()}


def utility_scorer_pin() -> dict:
    return {
        "pin_version": UTILITY_SCORER_PIN_VERSION,
        "scorer": "qa-builder-v2-score-utility",
        "reader": context_reader_pin(),
        "delivered": {"kind": "fact-score", "version": "fact-score-v1"},
        "builder_implementation_sha256": _qa_builder_source_digest(),
    }


def relation_teacher_pin(enabled: bool) -> dict:
    if not enabled:
        return {"enabled": False, "pin_version": "qa-relation-teacher-disabled-v1"}
    return {
        "enabled": True,
        "provider": "openrouter",
        "base_url": RELATION_TEACHER_BASE_URL,
        "model": RELATION_TEACHER_MODEL,
        "prompt": {
            "version": RELATION_TEACHER_PROMPT_VERSION,
            "sha256": _stable_hash(RELATION_TEACHER_PROMPT_TEMPLATE),
        },
        "response_schema": {
            "version": RELATION_TEACHER_RESPONSE_SCHEMA_VERSION,
            "schema": json.loads(json.dumps(RELATION_TEACHER_OUTPUT_SCHEMA)),
        },
        "decoding": {
            "temperature": 0.0,
            "max_tokens": 4096,
            "reasoning_excluded": True,
            "response_format": {"type": "json_object"},
        },
    }


def _injected_dependency_pin(kind: str, dependency) -> dict:
    target = dependency if inspect.isfunction(dependency) else type(dependency)
    module = str(getattr(target, "__module__", "unknown"))
    qualname = str(getattr(target, "__qualname__", getattr(target, "__name__", "unknown")))
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        source = f"{module}.{qualname}"
    return {
        "production": False,
        "pin_version": f"qa-{kind}-injected-v1",
        "implementation": {
            "module": module,
            "qualname": qualname,
            "sha256": _stable_hash(source),
        },
    }


def reader_dependency_pin(reader) -> dict:
    if reader is read_context_batch:
        return context_reader_pin()
    return _injected_dependency_pin("context-reader", reader)


def teacher_dependency_pin(teacher) -> dict:
    if teacher is None:
        return relation_teacher_pin(False)
    if type(teacher) is OpenRouterRelationTeacher:
        return relation_teacher_pin(True)
    return {
        "enabled": True,
        **_injected_dependency_pin("relation-teacher", teacher),
    }


_COST_BUDGET_FIELDS = {
    "base": (
        "remote_round_trips_per_rollout",
        "context_reader_batches_per_rollout",
    ),
    "counterfactual": (
        "remote_round_trips_per_selected_pair",
        "context_reader_batches_per_selected_pair",
    ),
}

_WALL_TIME_BUDGET_FIELDS = (
    "artifact_build_seconds_per_document",
    "base_seconds_per_rollout",
    "counterfactual_seconds_per_selected_pair",
)

_FAMILY_BUDGET_NAMES = ("context", "delivered")


def normalize_family_budgets(value: Mapping) -> dict[str, float]:
    """Validate the exact positive QA utility-family allocation."""
    if not isinstance(value, Mapping) or set(value) != set(_FAMILY_BUDGET_NAMES):
        raise ValueError("family budgets require exactly context and delivered")
    normalized = {}
    for family in _FAMILY_BUDGET_NAMES:
        budget = value[family]
        if isinstance(budget, bool) or not isinstance(budget, (int, float)):
            raise ValueError(f"family budgets: {family} must be a positive finite number")
        budget = float(budget)
        if not math.isfinite(budget) or budget <= 0.0:
            raise ValueError(f"family budgets: {family} must be a positive finite number")
        normalized[family] = budget
    return normalized


def normalize_cost_budgets(value: Mapping) -> dict:
    """Validate and freeze the manifest-owned QA call budgets."""
    if not isinstance(value, Mapping) or set(value) != set(_COST_BUDGET_FIELDS):
        raise ValueError("cost budgets require base and counterfactual sections")
    normalized = {}
    for section, fields in _COST_BUDGET_FIELDS.items():
        row = value.get(section)
        if not isinstance(row, Mapping) or set(row) != set(fields):
            raise ValueError(f"cost budgets have invalid {section} fields")
        normalized[section] = {}
        for field in fields:
            amount = row[field]
            if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                raise ValueError(f"cost budget {section}.{field} must be a nonnegative integer")
            normalized[section][field] = amount
    return normalized


def normalize_wall_time_budgets(value: Mapping | None) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or set(value) != set(_WALL_TIME_BUDGET_FIELDS):
        raise ValueError("wall time budgets require all build/base/counterfactual fields")
    normalized = {}
    for field in _WALL_TIME_BUDGET_FIELDS:
        amount = value[field]
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValueError(f"wall time budget {field} must be a nonnegative finite number")
        amount = float(amount)
        if not math.isfinite(amount) or amount < 0.0:
            raise ValueError(f"wall time budget {field} must be a nonnegative finite number")
        normalized[field] = amount
    return normalized


def normalize_threshold_manifest(value: Mapping) -> dict:
    """Return the canonical threshold manifest embedded and hashed by the artifact."""
    if not isinstance(value, Mapping):
        raise ValueError("threshold manifest must be an object")
    normalized = dict(value)
    normalized["family_budgets"] = normalize_family_budgets(value.get("family_budgets"))
    normalized["cost_budgets"] = normalize_cost_budgets(value.get("cost_budgets"))
    if "wall_time_budgets" in value:
        normalized["wall_time_budgets"] = normalize_wall_time_budgets(value["wall_time_budgets"])
    for field in ("reader_stability_repetitions", "reader_option_permutations"):
        count = value.get(field, 1)
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"{field} must be an integer")
        normalized[field] = count
    min_context = value.get("min_context_assertions", 0)
    if isinstance(min_context, bool) or not isinstance(min_context, int) or min_context < 0:
        raise ValueError("min_context_assertions must be a nonnegative integer")
    try:
        normalized["min_context_assertions"] = min_context
        normalized["reader_threshold"] = float(value.get("reader_threshold", 1.0))
        normalized["reader_stability_threshold"] = float(
            value.get("reader_stability_threshold", 1.0)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("threshold manifest has invalid reader thresholds") from error
    if not 0.0 <= normalized["reader_threshold"] <= 1.0:
        raise ValueError("reader_threshold must be in [0, 1]")
    if (
        normalized["reader_stability_repetitions"] < 1
        or normalized["reader_option_permutations"] < 1
        or not 0.0 < normalized["reader_stability_threshold"] <= 1.0
    ):
        raise ValueError("reader stability settings are invalid")
    return json.loads(json.dumps(normalized, sort_keys=True, allow_nan=False))


def effective_count_floors(
    ranker_environment: Mapping,
    floors: Mapping[str, float] | None = None,
) -> dict[str, float]:
    effective = {
        str(runtime_type): float(value)
        for runtime_type, value in dict(ranker_environment.get("k_floors") or {}).items()
    }
    if floors is not None:
        effective.update({
            str(runtime_type): float(value)
            for runtime_type, value in floors.items()
        })
    effective.setdefault("OTHER", 100.0)
    if any(not math.isfinite(value) or value < 0.0 for value in effective.values()):
        raise ValueError("count floors must be finite nonnegative numbers")
    return dict(sorted(effective.items()))


def floor_for_runtime_type(runtime_type: str, floors: Mapping[str, float]) -> float:
    return float(floors.get(runtime_type, floors["OTHER"]))


def action_is_floor_legal(action: Mapping, floor: float) -> bool:
    mode = "keep" if action.get("keep") else action.get("mode")
    return mode in {"keep", "placeholder"} or float(action.get("aset", 0.0)) >= floor


class OpenRouterRelationTeacher:
    """Optional cached JSON relation proposer pinned to Nemotron on OpenRouter."""

    def __init__(self):
        from cloak.llm import LLMClient

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for relation escalation")
        if not os.getenv("CLOAK_LLM_CACHE"):
            raise ValueError("CLOAK_LLM_CACHE is required for relation escalation")
        self._client = LLMClient(
            RELATION_TEACHER_MODEL,
            base_url=RELATION_TEACHER_BASE_URL,
            api_key=api_key,
            temperature=0.0,
            max_tokens=4096,
            response_format={"type": "json_object"},
            extra_body={"reasoning": {"exclude": True}},
            single_flight=True,
        )

    def propose(self, prompt: str) -> list[dict]:
        payload = json.loads(self._client.generate(prompt))
        relations = payload.get("relations") if isinstance(payload, dict) else None
        if not isinstance(relations, list):
            raise ValueError("relation teacher reply must contain a relations list")
        return [dict(row) for row in relations if isinstance(row, dict)]


class BatchedContextReader:
    """Pinned one-request reader for all context assertions on one document."""

    def __init__(self, client=None):
        if client is None:
            from cloak.llm import LLMClient

            if not os.getenv("CLOAK_LLM_CACHE"):
                raise ValueError("CLOAK_LLM_CACHE is required for context scoring")
            client = LLMClient(
                QA_MODEL,
                base_url=QA_BASE_URL,
                api_key="x",
                temperature=0.0,
                max_tokens=CONTEXT_READER_MAX_TOKENS,
                response_format=CONTEXT_READER_RESPONSE_FORMAT,
                extra_body=CONTEXT_READER_EXTRA_BODY,
            )
        self._client = client

    def __call__(
        self,
        questions: list[str],
        context: str,
        *,
        refresh: bool = False,
    ) -> list[str]:
        return _read_batch(self._client, questions, context, refresh=refresh)


def _read_batch(client, questions: list[str], context: str, *, refresh: bool) -> list[str]:
    """Issue the single pinned context-reader request for one document."""
    if not questions:
        return []
    prompt = CONTEXT_READER_PROMPT.format(
        context=context,
        questions_json=json.dumps(questions, indent=2),
    )
    payload = json.loads(client.generate(prompt, refresh=refresh))
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, list) or len(answers) != len(questions):
        raise ValueError("context reader returned the wrong number of answers")
    if any(not isinstance(answer, str) for answer in answers):
        raise ValueError("context reader answers must be strings")
    return ["" if answer.strip().upper() == "NONE" else answer.strip()
            for answer in answers]


def read_context_batch(
    questions: list[str],
    context: str,
    *,
    refresh: bool = False,
) -> list[str]:
    global _batched_context_reader
    if _batched_context_reader is None:
        _batched_context_reader = BatchedContextReader()
    return _batched_context_reader(questions, context, refresh=refresh)


_batched_context_reader = None


def _masked_local_context(document: str, target: Mapping, protected_values: Sequence[str]) -> str:
    """Return a bounded sentence/window with no usable protected locator."""
    start, end = target.get("start"), target.get("end")
    surface = str(target.get("surface", ""))
    if not isinstance(start, int) or not isinstance(end, int) or document[start:end] != surface:
        match = re.search(re.escape(surface), document, re.IGNORECASE)
        if match is None:
            return ""
        start, end = match.span()
    left = max(document.rfind(".", 0, start) + 1, start - 180)
    right_stop = document.find(".", end)
    right = min(len(document), right_stop + 1 if right_stop >= 0 else end + 180)
    rows = [(start, end, "[BLANK]")]
    for value in protected_values:
        if value and canon(value) != canon(surface):
            rows.extend((left + match.start(), left + match.end(), "[SENSITIVE]")
                        for match in re.finditer(re.escape(value), document[left:right], re.IGNORECASE))
    local = document[left:right]
    adjusted = []
    for row_start, row_end, replacement in rows:
        if row_start >= left and row_end <= right:
            adjusted.append((row_start - left, row_end - left, replacement))
    for row_start, row_end, replacement in sorted(adjusted, reverse=True):
        local = local[:row_start] + replacement + local[row_end:]
    return " ".join(local.split())


def _is_informative_semantic_property(value: str) -> bool:
    return canon(value) not in {"", "something", "something else", "thing", "item", "entity"}


class AciTaskAdapter:
    """Authoritative deterministic ACI delivered facts and relation compilation."""

    task_pin = {"adapter": "aci", "version": "aci-utility-v1"}
    relation_contract = ACI_RELATION_CONTRACT

    def __init__(self, references: Mapping[str, str]):
        self._references = dict(references)

    def authoritative_reference(self, doc_id: str) -> str:
        return self._references[doc_id]

    def deterministic_candidates(
        self,
        doc_id: str,
        document: str,
        environment_document: Mapping,
    ) -> list[dict]:
        reference = self._references[doc_id]
        candidates = []
        occurrences_by_decision: dict[str, list[Mapping]] = defaultdict(list)
        for occurrence in environment_document.get("occurrences", []):
            decision_id = occurrence.get("decision_id")
            if decision_id is None or not occurrence.get("controlled", True):
                continue
            occurrences_by_decision[str(decision_id)].append(occurrence)
        decisions = {
            str(row["decision_id"]): row for row in environment_document.get("decisions", [])
        }
        protected_values = [
            value
            for occurrence in environment_document.get("occurrences", [])
            for value in [occurrence.get("surface", ""), *(occurrence.get("aliases") or [])]
        ]
        for decision_id, occurrences in occurrences_by_decision.items():
            surface = str(occurrences[0].get("surface", ""))
            if not surface:
                continue
            authority_text = reference if _contains(reference, surface) else document
            if not _contains(authority_text, surface):
                continue
            candidates.append({
                "family": "delivered",
                "scope": "linked",
                "subtype": "content",
                "occurrence_ids": [str(row["occurrence_id"]) for row in occurrences],
                "group_id": f"content:{decision_id}",
                "scoring_contract": {"kind": "contains", "value": surface},
                "evidence": {
                    "authority": "human_reference" if authority_text == reference else "doc_orig",
                    "quote": surface,
                },
            })
            decision = decisions.get(decision_id)
            if decision is None:
                continue
            levels = [
                action for action in decision.get("actions", [])
                if action.get("legal", True)
                and action.get("mode") == "level"
                and not action.get("keep")
                and action.get("fill")
            ]
            if not levels:
                continue
            property_level = str(levels[-1]["fill"])
            masked = _masked_local_context(document, occurrences[0], protected_values)
            if not masked or not _is_informative_semantic_property(property_level):
                continue
            candidates.append({
                "family": "context",
                "scope": "linked",
                "subtype": "semantic_property",
                "occurrence_ids": [str(row["occurrence_id"]) for row in occurrences],
                "group_id": f"semantic:{decision_id}:{canon(property_level)}",
                "question": (
                    f"Masked local context: {masked}\n"
                    "What broad semantic category is retained for [BLANK]?"
                ),
                "accepted_values": [property_level],
                "decision_requirements": {decision_id: property_level},
                "evidence": {"authority": "frozen_lattice", "masked_context": masked},
            })

        demographic = re.search(
            r"\b(\d{1,3}-year-old)\s+(male|female)\b", document, re.IGNORECASE
        )
        if demographic:
            for name, value in zip(("age", "sex"), demographic.groups()):
                authority_text = reference if _contains(reference, value) else document
                if _contains(authority_text, value):
                    candidates.append({
                        "family": "delivered",
                        "scope": "global",
                        "subtype": "field",
                        "occurrence_ids": [],
                        "group_id": f"demographic:{name}",
                        "scoring_contract": {"kind": "contains", "value": value},
                        "evidence": {
                            "authority": "human_reference" if authority_text == reference else "doc_orig",
                            "quote": value,
                        },
                    })
        return candidates

    def compile_relations(
        self,
        doc_id: str,
        document: str,
        environment_document: Mapping,
        proposals: Sequence[Mapping],
    ) -> tuple[list[dict], list[dict]]:
        return compile_relational_assertions(
            doc_id, document, environment_document, proposals,
            relation_contract=self.relation_contract,
        )


def _stable_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _qa_builder_source_digest() -> str:
    """Conservatively bind pins to the complete live QA-builder implementation."""
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _contains(text: str, value: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(canon(value))}(?!\w)", canon(text)))


def _canonical_leakage_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", canon(value)))


def _meaningful_leakage_tokens(value: str) -> set[str]:
    return {
        token for token in _canonical_leakage_phrase(value).split()
        if len(token) >= 4 and token not in _LEAKAGE_GENERIC_TOKENS
    }


def _text_leaks_values(texts: Sequence[str], values: Sequence[str]) -> bool:
    canonical_texts = [_canonical_leakage_phrase(str(text)) for text in texts]
    text_tokens = {
        token
        for text in canonical_texts
        for token in text.split()
        if len(token) >= 4
    }
    for value in values:
        phrase = _canonical_leakage_phrase(str(value))
        meaningful = _meaningful_leakage_tokens(str(value))
        if not phrase:
            continue
        if (
            len(phrase.replace(" ", "")) > 1
            and any(f" {phrase} " in f" {text} " for text in canonical_texts)
        ):
            return True
        if meaningful & text_tokens:
            return True
    return False


def _relation_argument_types_are_legal(
    relation: str,
    occurrence_ids: Sequence[str],
    occurrences: Mapping[str, Mapping],
    relation_contract: Mapping[str, Mapping],
) -> bool:
    allowed = relation_contract[relation]["argument_classes"]
    if len(occurrence_ids) != len(allowed):
        return False
    actual = [
        _RUNTIME_TYPE_CLASSES.get(canon(str(occurrences[occurrence_id].get("runtime_type", ""))))
        for occurrence_id in occurrence_ids
    ]
    return all(type_class in permitted
               for type_class, permitted in zip(actual, allowed))


def _relation_evidence_connects_arguments(
    relation: str,
    quote: str,
    occurrence_ids: Sequence[str],
    occurrences: Mapping[str, Mapping],
    relation_contract: Mapping[str, Mapping],
) -> bool:
    canonical_quote = canon(quote)
    positions = []
    for occurrence_id in occurrence_ids:
        surface = canon(str(occurrences[occurrence_id].get("surface", "")))
        match = re.search(rf"(?<!\w){re.escape(surface)}(?!\w)", canonical_quote)
        if match is None:
            return False
        positions.append((match.start(), match.end()))
    if positions != sorted(positions):
        return False
    connector_text = canonical_quote[positions[0][1]:positions[-1][0]]
    return any(cue in connector_text for cue in relation_contract[relation]["cues"])


def _proposal_polarity_matches_frozen_occurrences(
    proposal: Mapping,
    occurrence_ids: Sequence[str],
    occurrences: Mapping[str, Mapping],
) -> bool:
    declared = proposal.get("argument_polarities")
    if declared is None:
        return True
    declared_by_occurrence = {
        str(occurrence_id): canon(str(polarity))
        for occurrence_id, polarity in dict(declared).items()
    }
    frozen_by_occurrence = {
        occurrence_id: canon(str(occurrences[occurrence_id].get("polarity", "unknown")))
        for occurrence_id in occurrence_ids
    }
    return declared_by_occurrence == frozen_by_occurrence


def _source_contains_relation_contradiction(
    relation: str,
    document: str,
    occurrence_ids: Sequence[str],
    occurrences: Mapping[str, Mapping],
    relation_contract: Mapping[str, Mapping],
) -> bool:
    for sentence in re.split(r"(?<=[.!?])\s+", document):
        if not _NEGATION_PATTERN.search(canon(sentence)):
            continue
        if _relation_evidence_connects_arguments(
            relation, sentence, occurrence_ids, occurrences, relation_contract
        ):
            return True
    return False


def relation_teacher_prompt(
    doc_id: str,
    document: str,
    environment_document: Mapping,
    *,
    authoritative_reference: str | None = None,
) -> str:
    """Build the single bounded teacher prompt from frozen IDs and legal properties."""
    inventory = []
    decisions = {
        str(row["decision_id"]): row
        for row in environment_document.get("decisions", [])
    }
    for occurrence in environment_document.get("occurrences", []):
        decision = decisions.get(str(occurrence.get("decision_id")), {})
        properties = [
            str(action.get("fill"))
            for action in decision.get("actions", [])
            if action.get("legal", True)
            and action.get("mode") == "level"
            and not action.get("keep")
            and action.get("fill")
        ]
        inventory.append({
            "occurrence_id": occurrence.get("occurrence_id"),
            "surface": occurrence.get("surface"),
            "runtime_type": occurrence.get("runtime_type"),
            "legal_support_properties": properties,
        })
    return RELATION_TEACHER_PROMPT_TEMPLATE.format(
        doc_id=doc_id,
        inventory_json=json.dumps(inventory, indent=2),
        schema_json=json.dumps(RELATION_TEACHER_OUTPUT_SCHEMA, indent=2),
        authoritative_reference=authoritative_reference or "NONE PROVIDED",
        document=document,
    )


def compile_relational_assertions(
    doc_id: str,
    document: str,
    environment_document: Mapping,
    proposals: Sequence[Mapping],
    *,
    relation_contract: Mapping[str, Mapping] = ACI_RELATION_CONTRACT,
) -> tuple[list[dict], list[dict]]:
    """Compile bounded teacher proposals into frozen, evidence-checked assertions."""
    occurrences = {
        str(row["occurrence_id"]): row
        for row in environment_document.get("occurrences", [])
    }
    decisions = {
        str(row["decision_id"]): row
        for row in environment_document.get("decisions", [])
    }
    legal_properties: dict[str, set[str]] = {}
    for occurrence_id, occurrence in occurrences.items():
        decision = decisions.get(str(occurrence.get("decision_id")), {})
        legal_properties[occurrence_id] = {
            canon(str(action["fill"]))
            for action in decision.get("actions", [])
            if action.get("legal", True)
            and action.get("mode") == "level"
            and not action.get("keep")
            and action.get("fill")
        }

    accepted, rejected = [], []
    for index, proposal_value in enumerate(proposals):
        proposal = dict(proposal_value)

        def reject(reason: str) -> None:
            rejected.append({"proposal_index": index, "reason": reason})

        relation = str(proposal.get("relation", ""))
        if relation not in relation_contract:
            reject("invalid_relation")
            continue
        occurrence_ids = [str(value) for value in proposal.get(
            "argument_occurrence_ids", []
        )]
        if len(occurrence_ids) < 2 or len(set(occurrence_ids)) != len(occurrence_ids):
            reject("invalid_arguments")
            continue
        if any(occurrence_id not in occurrences for occurrence_id in occurrence_ids):
            reject("invalid_arguments")
            continue
        if not _relation_argument_types_are_legal(
            relation, occurrence_ids, occurrences, relation_contract
        ):
            reject("invalid_argument_types")
            continue
        quote = str(proposal.get("evidence_quote", ""))
        if not quote or quote not in document or any(
            not _contains(quote, str(occurrences[occurrence_id].get("surface", "")))
            for occurrence_id in occurrence_ids
        ):
            reject("invalid_evidence")
            continue
        if not _relation_evidence_connects_arguments(
            relation, quote, occurrence_ids, occurrences, relation_contract
        ):
            reject("invalid_evidence")
            continue
        if not _proposal_polarity_matches_frozen_occurrences(
            proposal, occurrence_ids, occurrences
        ):
            reject("invalid_polarity")
            continue
        if _source_contains_relation_contradiction(
            relation, document, occurrence_ids, occurrences, relation_contract
        ):
            reject("source_contradiction")
            continue
        support = {
            str(key): canon(str(value))
            for key, value in dict(proposal.get("support_properties") or {}).items()
        }
        if set(support) != set(occurrence_ids) or any(
            support[occurrence_id] not in legal_properties[occurrence_id]
            for occurrence_id in occurrence_ids
        ):
            reject("invalid_property")
            continue
        answer_occurrence_id = str(proposal.get("answer_occurrence_id", ""))
        answer_property = canon(str(proposal.get("answer_property", "")))
        if (
            answer_occurrence_id not in occurrence_ids
            or answer_property != support.get(answer_occurrence_id)
        ):
            reject("invalid_property")
            continue
        question = str(proposal.get("question", "")).strip()
        if not question.endswith("?"):
            reject("invalid_question")
            continue
        options_value = proposal.get("options") or []
        if (
            not isinstance(options_value, Sequence)
            or isinstance(options_value, (str, bytes))
            or any(not isinstance(option, str) for option in options_value)
        ):
            reject("invalid_question")
            continue
        options = [str(option).strip() for option in options_value]
        lint_texts = [question, *options]
        answer_aliases = []
        answer_decision = decisions.get(str(occurrences[answer_occurrence_id].get("decision_id")), {})
        for action in answer_decision.get("actions", []):
            if canon(str(action.get("fill", ""))) == answer_property:
                answer_aliases.extend(str(alias) for alias in action.get("aliases") or [])
        if _text_leaks_values(lint_texts, [answer_property, *answer_aliases]):
            reject("answer_leakage")
            continue
        protected_values = []
        for occurrence in occurrences.values():
            protected_values.append(str(occurrence.get("surface", "")))
            protected_values.extend(str(alias) for alias in occurrence.get("aliases") or [])
        if _text_leaks_values(lint_texts, protected_values):
            reject("protected_locator")
            continue
        decision_requirements = {
            str(occurrences[occurrence_id]["decision_id"]): support[occurrence_id]
            for occurrence_id in occurrence_ids
        }
        accepted.append({
            "family": "context",
            "scope": "linked",
            "subtype": "contextual_relation",
            "relation": relation,
            "occurrence_ids": occurrence_ids,
            "group_id": f"relation:{relation}:{':'.join(occurrence_ids)}",
            "question": question,
            **({"options": options} if options else {}),
            "accepted_values": [answer_property],
            "decision_requirements": decision_requirements,
            "evidence": {"source_quotes": [quote]},
        })
    return accepted, rejected


def assign_static_weights(
    assertions: Sequence[Mapping],
    family_budgets: Mapping[str, float],
) -> tuple[list[dict], dict]:
    """Assign family -> group -> assertion weights with a fixed family denominator."""
    groups: dict[str, dict[str, list[Mapping]]] = defaultdict(lambda: defaultdict(list))
    for assertion in assertions:
        family = str(assertion["family"])
        if family not in family_budgets:
            raise ValueError(f"unknown assertion family: {family}")
        groups[family][str(assertion["group_id"])].append(assertion)

    weights: dict[str, float] = {}
    for family, family_groups in groups.items():
        group_budget = float(family_budgets[family]) / len(family_groups)
        for rows in family_groups.values():
            assertion_weight = group_budget / len(rows)
            for row in rows:
                weights[str(row["assertion_id"])] = assertion_weight

    weighted = [{**dict(row), "weight": weights[str(row["assertion_id"])]}
                for row in assertions]
    present = [family for family in family_budgets if family in groups]
    missing = [family for family in family_budgets if family not in groups]
    state = {
        "utility_weight_denominator": sum(float(v) for v in family_budgets.values()),
        "present_family_budgets": present,
        "missing_family_budgets": missing,
    }
    return weighted, state


def utility_assertion_semantic_key(assertion: Mapping) -> str:
    family = str(assertion.get("family", ""))
    if family == "delivered":
        contract = assertion.get("scoring_contract") or {}
        payload = {
            "family": family,
            "kind": str(contract.get("kind", "")),
            "value": canon(str(contract.get("value", ""))),
        }
    elif family == "context":
        payload = {
            "family": family,
            "question": canon(str(assertion.get("question", ""))),
            "accepted_values": sorted(
                canon(str(value)) for value in assertion.get("accepted_values") or []
            ),
        }
    else:
        payload = {"family": family, "assertion": dict(assertion)}
    return _stable_hash(payload)


def _prefer_linked_cross_scope_facts(candidates: Sequence[Mapping]) -> list[Mapping]:
    scopes_by_fact: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        scopes_by_fact[utility_assertion_semantic_key(candidate)].add(
            str(candidate.get("scope", ""))
        )
    return [
        candidate for candidate in candidates
        if not (
            candidate.get("scope") == "global"
            and "linked" in scopes_by_fact[utility_assertion_semantic_key(candidate)]
        )
    ]


def _weight_group_state(assertions: Sequence[Mapping], family_budgets: Mapping[str, float]) -> dict:
    """Persist the derived family/group allocation used to assign assertion weights."""
    groups: dict[str, dict[str, list[Mapping]]] = defaultdict(lambda: defaultdict(list))
    for assertion in assertions:
        groups[str(assertion["family"])][str(assertion["group_id"])].append(assertion)
    return {
        family: {
            group_id: {
                "assertion_ids": [str(row["assertion_id"]) for row in rows],
                "weight": float(family_budgets[family]) / len(family_groups),
            }
            for group_id, rows in family_groups.items()
        }
        for family, family_groups in groups.items()
    }


def package_utility_artifact(
    frozen_environment: Mapping,
    candidates_by_document: Mapping[str, Sequence[Mapping]],
    *,
    family_budgets: Mapping[str, float],
    pins: Mapping,
    document_failures: Mapping[str, Mapping] | None = None,
    document_build_seconds: Mapping[str, float] | None = None,
) -> dict:
    """Compile validated candidates into one deterministic, link-checked utility artifact."""
    family_budgets = normalize_family_budgets(family_budgets)
    assertions: dict[str, dict] = {}
    documents: dict[str, dict] = {}
    environment_documents = frozen_environment.get("documents", {})
    for doc_id, environment_document in environment_documents.items():
        occurrences = {
            str(row["occurrence_id"]): row
            for row in environment_document.get("occurrences", [])
        }
        if len(occurrences) != len(environment_document.get("occurrences", [])):
            raise ValueError(f"duplicate occurrence ids for {doc_id}")
        decision_rows = environment_document.get("decisions", [])
        decision_ids = {str(row["decision_id"]) for row in decision_rows}
        if len(decision_ids) != len(decision_rows):
            raise ValueError(f"duplicate decision ids for {doc_id}")
        controlled = []
        for row in decision_rows:
            if not row.get("controlled", True):
                continue
            decision_id = row.get("decision_id")
            if decision_id is None:
                raise ValueError(f"controlled decision lacks an id for {doc_id}")
            controlled.append(str(decision_id))
        controlled_set = set(controlled)
        if len(controlled_set) != len(controlled):
            raise ValueError(f"duplicate controlled decision ids for {doc_id}")
        controlled_occurrences = {
            occurrence_id: row
            for occurrence_id, row in occurrences.items()
            if row.get("controlled", row.get("decision_id") is not None)
        }
        missing_decision_ids = sorted(
            occurrence_id for occurrence_id, row in controlled_occurrences.items()
            if row.get("decision_id") is None
        )
        if missing_decision_ids:
            raise ValueError(
                f"controlled occurrences lack decision ids for {doc_id}: {missing_decision_ids}"
            )
        dangling_decisions = sorted({
            str(row["decision_id"])
            for row in controlled_occurrences.values()
            if str(row["decision_id"]) not in controlled_set
        })
        if dangling_decisions:
            raise ValueError(
                f"unknown decision links for {doc_id}: {dangling_decisions}"
            )
        compiled = []
        compiled_ids = set()
        linked_decisions = set()
        for candidate in _prefer_linked_cross_scope_facts(
            candidates_by_document.get(doc_id, [])
        ):
            row = {**dict(candidate), "doc_id": doc_id, "status": "accepted"}
            scope = row.get("scope")
            occurrence_ids = [str(value) for value in row.get("occurrence_ids", [])]
            if scope == "global" and occurrence_ids:
                raise ValueError("global assertion must not link occurrences")
            if scope == "linked" and not occurrence_ids:
                raise ValueError("linked assertion must link occurrences")
            if scope not in {"linked", "global"}:
                raise ValueError(f"invalid assertion scope: {scope}")
            missing = sorted(set(occurrence_ids) - set(occurrences))
            if missing:
                raise ValueError(f"unknown occurrence links for {doc_id}: {missing}")
            uncontrolled = sorted(
                occurrence_id for occurrence_id in occurrence_ids
                if not occurrences[occurrence_id].get(
                    "controlled", occurrences[occurrence_id].get("decision_id") is not None
                )
            )
            if uncontrolled:
                raise ValueError(
                    f"uncontrolled occurrence links for {doc_id}: {uncontrolled}"
                )
            for occurrence_id in occurrence_ids:
                decision_id = occurrences[occurrence_id].get("decision_id")
                if decision_id is not None:
                    linked_decisions.add(str(decision_id))
            identity_payload = {
                key: value for key, value in row.items()
                if key not in {"assertion_id", "weight", "status"}
            }
            assertion_id = str(row.get("assertion_id") or _stable_hash(identity_payload))
            if assertion_id in assertions or assertion_id in compiled_ids:
                raise ValueError(f"duplicate assertion id: {assertion_id}")
            row["assertion_id"] = assertion_id
            row["occurrence_ids"] = occurrence_ids
            compiled.append(row)
            compiled_ids.add(assertion_id)

        weighted, weight_state = assign_static_weights(compiled, family_budgets)
        for row in weighted:
            assertions[row["assertion_id"]] = row
        missing_families = weight_state["missing_family_budgets"]
        build_failure = dict((document_failures or {}).get(doc_id) or {})
        documents[doc_id] = {
            "environment_document_hash": environment_document.get(
                "environment_document_hash", _stable_hash(environment_document)
            ),
            **({"artifact_build_seconds": float(document_build_seconds[doc_id])}
               if document_build_seconds is not None and doc_id in document_build_seconds else {}),
            **{
                field: environment_document[field]
                for field in ("source_hash", "authoritative_reference_hash")
                if field in environment_document
            },
            "measurement_state": (
                "build_failed" if build_failure else
                "unsupported" if not weighted else "partial" if missing_families else "measured"
            ),
            **({"build_failure": build_failure} if build_failure else {}),
            **weight_state,
            "weight_groups": _weight_group_state(weighted, family_budgets),
            "assertion_ids": [row["assertion_id"] for row in weighted],
            "controlled_decision_ids": controlled,
            "occurrence_to_decision": {
                occurrence_id: str(row["decision_id"])
                for occurrence_id, row in controlled_occurrences.items()
            },
            "decision_keys": [{
                "decision_id": str(row["decision_id"]),
                "runtime_type": row.get("runtime_type"),
                "canonical_key": row.get("canonical_key"),
            } for row in decision_rows if row.get("controlled", True)],
            "uncovered_decision_ids": [
                decision_id for decision_id in controlled if decision_id not in linked_decisions
            ],
        }

    artifact = {
        "artifact_version": "utility-assertions-v1",
        "environment_hash": frozen_environment.get("environment_hash"),
        **dict(pins),
        "family_budgets": dict(family_budgets),
        "documents": documents,
        "assertions": assertions,
        "rejections": {"summary_by_reason": {}},
    }
    artifact["artifact_hash"] = _stable_hash(artifact)
    return artifact


def freeze_ranker_environment(
    ranker_environment: Mapping,
    *,
    occurrences_by_document: Mapping[str, Sequence[Mapping]] | None = None,
    floors: Mapping[str, float] | None = None,
    source_documents: Mapping[str, str] | None = None,
    authoritative_references: Mapping[str, str] | None = None,
) -> dict:
    """Migrate embedded ranker spans to stable occurrence/decision identities, without detection."""
    effective_floors = effective_count_floors(ranker_environment, floors)
    documents: dict[str, dict] = {}
    for corpus, per_document in ranker_environment.get("corpora", {}).items():
        for doc_id, document in per_document.items():
            decisions_by_key: dict[tuple[str, str], dict] = {}
            for span in document.get("spans", []):
                runtime_type = str(span.get("type", ""))
                floor = floor_for_runtime_type(runtime_type, effective_floors)
                surface = str(span.get("surface", ""))
                decision_key = (runtime_type, canon(surface))
                decision_id = _stable_hash({
                    "doc_id": doc_id,
                    "runtime_type": runtime_type,
                    "canonical_surface": decision_key[1],
                })
                normalized_actions = []
                hierarchy = []
                for action in span.get("actions", []):
                    mode = (
                        "keep" if action.get("keep") else
                        "placeholder" if action.get("mode") == "placeholder" else
                        "level"
                    )
                    fill = action.get("fill")
                    normalized_actions.append((dict(action), mode, fill))
                    if mode == "level" and fill:
                        hierarchy.append(canon(str(fill)))
                hierarchy_positions = {
                    canon(str(fill)): index
                    for index, (_action, mode, fill) in enumerate(
                        row for row in normalized_actions if row[1] == "level" and row[2]
                    )
                }
                actions = []
                action_ids = set()
                for action, mode, fill in normalized_actions:
                    action_semantics = {
                        **dict(action),
                        "mode": mode,
                        "fill": fill,
                    }
                    action_semantics.pop("action_id", None)
                    action_semantics.pop("entails", None)
                    action_id = _stable_hash({
                        "decision_id": decision_id,
                        "action": action_semantics,
                    })
                    if action_id in action_ids:
                        raise ValueError(f"duplicate action semantics for decision {decision_id}")
                    action_ids.add(action_id)
                    if mode == "placeholder" or not fill:
                        entails = []
                    elif mode == "keep":
                        entails = list(dict.fromkeys([canon(surface), *hierarchy]))
                    else:
                        entails = hierarchy[hierarchy_positions[canon(str(fill))]:]
                    actions.append({
                        **dict(action),
                        "action_id": action_id,
                        "mode": mode,
                        "legal": action_is_floor_legal(action_semantics, floor),
                        "entails": entails,
                    })
                action_menu_hash = _stable_hash(actions)
                previous = decisions_by_key.get(decision_key)
                if previous and previous["action_menu_hash"] != action_menu_hash:
                    raise ValueError(
                        f"inconsistent action menus for repeated decision {decision_id}"
                    )
                if previous is None:
                    decisions_by_key[decision_key] = {
                        "decision_id": decision_id,
                        "runtime_type": runtime_type,
                        "canonical_key": decision_key[1],
                        "occurrence_ids": [],
                        "controlled": True,
                        "actions": actions,
                        "action_menu_hash": action_menu_hash,
                    }
            occurrence_source = (
                occurrences_by_document[doc_id]
                if occurrences_by_document is not None and doc_id in occurrences_by_document
                else document.get("spans", [])
            )
            occurrences = []
            for row in occurrence_source:
                runtime_type = str(row.get("type", row.get("runtime_type", "")))
                surface = str(row.get("surface", ""))
                decision = decisions_by_key.get((runtime_type, canon(surface)))
                occurrence_id = _stable_hash({
                    "doc_id": doc_id,
                    "runtime_type": runtime_type,
                    "surface": surface,
                    "start": row.get("start"),
                    "end": row.get("end"),
                })
                occurrence = {
                    "occurrence_id": occurrence_id,
                    "start": row.get("start"),
                    "end": row.get("end"),
                    "surface": surface,
                    "runtime_type": runtime_type,
                    "polarity": row.get("polarity", "unknown"),
                    "detector_provenance": row.get("detector_provenance", {
                        "source": "frozen_arms_migration",
                        "score": row.get("score"),
                    }),
                    "overlap_disposition": row.get("overlap_disposition", "accepted"),
                    "decision_id": decision["decision_id"] if decision is not None else None,
                    "controlled": decision is not None,
                }
                if row.get("aliases") is not None:
                    occurrence["aliases"] = [str(alias) for alias in row.get("aliases") or []]
                occurrences.append(occurrence)
                if decision is not None:
                    decision["occurrence_ids"].append(occurrence_id)
            frozen_document = {
                "corpus": corpus,
                "occurrences": occurrences,
                "decisions": list(decisions_by_key.values()),
            }
            if source_documents is not None:
                if doc_id not in source_documents:
                    raise ValueError(f"source document identity is missing for {doc_id}")
                frozen_document["source_hash"] = _stable_hash(source_documents[doc_id])
            if authoritative_references is not None:
                if doc_id not in authoritative_references:
                    raise ValueError(
                        f"authoritative reference identity is missing for {doc_id}"
                    )
                frozen_document["authoritative_reference_hash"] = _stable_hash(
                    authoritative_references[doc_id]
                )
            frozen_document["environment_document_hash"] = _stable_hash(frozen_document)
            documents[doc_id] = frozen_document
    frozen = {
        "artifact_version": "occurrence-decisions-v1",
        "effective_floors": effective_floors,
        "documents": documents,
    }
    frozen["environment_hash"] = _stable_hash(frozen)
    return frozen


def frozen_occurrences_from_arms(arms: Mapping) -> dict[str, list[dict]]:
    """Read controlled occurrence rows from an already-frozen arms artifact."""
    return {
        doc_id: [dict(row) for row in document["tau_walk"][1] if row.get("lattice")]
        for corpus, documents in arms.items()
        if corpus != "_meta"
        for doc_id, document in documents.items()
    }


def _infrastructure_failure(stage: str, error: Exception) -> dict:
    if isinstance(error, (AssertionError, AttributeError, KeyError, NameError, TypeError)):
        raise error
    return {"stage": stage, "reason": type(error).__name__}


def build_utility_artifact(
    frozen_environment: Mapping,
    task_adapter,
    source_documents: Mapping[str, str],
    *,
    threshold_manifest: Mapping,
    pins: Mapping,
    reader: Callable[[list[str], str], Sequence[str]],
    render_action_vector: Callable[[str, Mapping[str, str]], str],
    relation_teacher: OpenRouterRelationTeacher | None = None,
) -> dict:
    """Build and validate one artifact through deterministic candidates and optional escalation."""
    frozen_threshold_manifest = normalize_threshold_manifest(threshold_manifest)
    family_budgets = frozen_threshold_manifest["family_budgets"]
    cost_budgets = frozen_threshold_manifest["cost_budgets"]
    min_context = frozen_threshold_manifest["min_context_assertions"]
    reader_threshold = frozen_threshold_manifest["reader_threshold"]
    stability_repetitions = frozen_threshold_manifest["reader_stability_repetitions"]
    option_permutations = frozen_threshold_manifest["reader_option_permutations"]
    stability_threshold = frozen_threshold_manifest["reader_stability_threshold"]
    candidates_by_document: dict[str, list[dict]] = {}
    rejection_counts: dict[str, int] = defaultdict(int)
    document_failures: dict[str, dict] = {}
    document_build_seconds: dict[str, float] = {}

    for doc_id, environment_document in frozen_environment.get("documents", {}).items():
        started_at = time.perf_counter()
        source = source_documents[doc_id]
        if (
            environment_document.get("source_hash") is not None
            and environment_document["source_hash"] != _stable_hash(source)
        ):
            raise ValueError(f"source document hash does not match frozen identity for {doc_id}")
        authoritative_reference = (
            task_adapter.authoritative_reference(doc_id)
            if hasattr(task_adapter, "authoritative_reference")
            else None
        )
        if (
            environment_document.get("authoritative_reference_hash") is not None
            and (
                authoritative_reference is None
                or environment_document["authoritative_reference_hash"]
                != _stable_hash(authoritative_reference)
            )
        ):
            raise ValueError(
                f"authoritative reference hash does not match frozen identity for {doc_id}"
            )
        candidates = [
            dict(row) for row in task_adapter.deterministic_candidates(
                doc_id, source, environment_document
            )
        ]
        if min_context == 0 and reader is read_context_batch:
            candidates = [row for row in candidates if row.get("family") != "context"]
        context_count = sum(row.get("family") == "context" for row in candidates)
        if context_count < min_context and relation_teacher is not None:
            try:
                proposals = relation_teacher.propose(
                    relation_teacher_prompt(
                        doc_id,
                        source,
                        environment_document,
                        authoritative_reference=authoritative_reference,
                    )
                )
                if not proposals:
                    rejection_counts["not_generated"] += 1
                else:
                    relation_candidates, relation_rejections = task_adapter.compile_relations(
                        doc_id, source, environment_document, proposals
                    )
                    candidates.extend(dict(row) for row in relation_candidates)
                    for rejection in relation_rejections:
                        reason = str(rejection["reason"])
                        stable_reason = "leakage" if reason in {
                            "answer_leakage", "protected_locator"
                        } else "invalid"
                        rejection_counts[stable_reason] += 1
            except Exception as error:
                document_failures[doc_id] = _infrastructure_failure("relation_teacher", error)
                rejection_counts["generation_failed"] += 1

        decisions = environment_document.get("decisions", [])
        placeholder_vector = {}
        for decision in decisions:
            placeholder = next(
                (action for action in decision.get("actions", [])
                 if action.get("legal", True) and action.get("mode") == "placeholder"),
                None,
            )
            if placeholder is None:
                raise ValueError(f"decision {decision['decision_id']} has no legal placeholder")
            placeholder_vector[str(decision["decision_id"])] = str(placeholder["action_id"])
        placeholder_context = render_action_vector(doc_id, placeholder_vector)

        accepted = []
        for candidate in candidates:
            if doc_id in document_failures and candidate.get("family") == "context":
                continue
            if candidate.get("family") != "context":
                accepted.append(candidate)
                continue
            try:
                anchor = build_joint_representative_anchor(candidate, decisions)
            except ValueError:
                rejection_counts["unsupported"] += 1
                continue
            representative_context = render_action_vector(doc_id, anchor["action_vector"])
            try:
                validated, validation_evidence = validate_context_assertions(
                    [candidate],
                    original_context=source,
                    representative_context=representative_context,
                    placeholder_context=placeholder_context,
                    reader=reader,
                    threshold=reader_threshold,
                    stability_repetitions=stability_repetitions,
                    option_permutations=option_permutations,
                    stability_threshold=stability_threshold,
                )
            except Exception as error:
                document_failures[doc_id] = _infrastructure_failure("context_reader", error)
                rejection_counts["infrastructure_failed"] += 1
                break
            evidence_row = next(iter(validation_evidence.values()))
            if not validated:
                if evidence_row["verdict"] == "unstable":
                    reason = "unstable"
                else:
                    reason = (
                        "floor_answerable"
                        if evidence_row["scores"]["placeholder"] >= reader_threshold
                        else "unsupported"
                    )
                rejection_counts[reason] += 1
                continue
            row = dict(candidate)
            row["expected_action_support"] = {
                "joint_anchor_action_vector": anchor["action_vector"],
                "joint_anchor_hash": anchor["action_vector_hash"],
                "property_level": dict(candidate.get("decision_requirements") or {}),
            }
            row["evidence"] = {
                **dict(row.get("evidence") or {}),
                "validation": evidence_row,
            }
            accepted.append(row)
        elapsed = time.perf_counter() - started_at
        document_build_seconds[doc_id] = elapsed
        wall_budgets = frozen_threshold_manifest.get("wall_time_budgets", {})
        if (
            wall_budgets
            and elapsed > wall_budgets["artifact_build_seconds_per_document"]
        ):
            document_failures[doc_id] = {
                "stage": "artifact_build_wall_time",
                "reason": "budget_exceeded",
                "elapsed_seconds": elapsed,
            }
        if doc_id in document_failures:
            document_failures[doc_id].setdefault("elapsed_seconds", elapsed)
        candidates_by_document[doc_id] = accepted

    manifest_hash = _stable_hash(frozen_threshold_manifest)
    task_pin = getattr(task_adapter, "task_pin", None)
    if task_pin is None:
        task_pin = {
            "adapter": f"{type(task_adapter).__module__}.{type(task_adapter).__qualname__}",
            "version": "unversioned",
        }
    artifact = package_utility_artifact(
        frozen_environment,
        candidates_by_document,
        family_budgets=family_budgets,
        pins={
            **dict(pins),
            "source_hashes": {
                doc_id: document["source_hash"]
                for doc_id, document in frozen_environment.get("documents", {}).items()
                if "source_hash" in document
            },
            "reference_hashes": {
                doc_id: document["authoritative_reference_hash"]
                for doc_id, document in frozen_environment.get("documents", {}).items()
                if "authoritative_reference_hash" in document
            },
            "task_pin": json.loads(json.dumps(task_pin, sort_keys=True)),
            "builder_pin": builder_pin(),
            "teacher_pin": teacher_dependency_pin(relation_teacher),
            "reader_pin": reader_dependency_pin(reader),
            "scorer_pin": utility_scorer_pin(),
            "gate_manifest_hash": manifest_hash,
            "threshold_manifest_pin": {
                "schema": THRESHOLD_MANIFEST_SCHEMA,
                "sha256": manifest_hash,
            },
            "threshold_manifest": frozen_threshold_manifest,
        },
        document_failures=document_failures,
        document_build_seconds=document_build_seconds,
    )
    artifact["cost_budgets"] = cost_budgets
    if "wall_time_budgets" in frozen_threshold_manifest:
        artifact["wall_time_budgets"] = frozen_threshold_manifest["wall_time_budgets"]
    artifact["rejections"] = {"summary_by_reason": dict(rejection_counts)}
    artifact["artifact_hash"] = _stable_hash({
        key: value for key, value in artifact.items() if key != "artifact_hash"
    })
    return artifact


def coarsest_entailing_legal_action(
    decision: Mapping,
    property_level: str,
) -> Mapping | None:
    """Return the last legal non-KEEP action entailing a required property."""
    expected_property = canon(str(property_level))
    candidates = [
        action for action in decision.get("actions", [])
        if action.get("legal", True)
        and action.get("mode") not in {"keep", "placeholder"}
        and expected_property in {canon(str(value)) for value in action.get("entails", [])}
    ]
    return candidates[-1] if candidates else None


def build_joint_representative_anchor(
    assertion: Mapping,
    decisions: Sequence[Mapping],
) -> dict:
    """Choose one joint vector: linked coarsest entailing levels, unrelated KEEP."""
    requirements = dict(assertion.get("decision_requirements") or {})
    action_vector: dict[str, str] = {}
    seen = set()
    for decision in decisions:
        decision_id = str(decision["decision_id"])
        seen.add(decision_id)
        actions = [action for action in decision.get("actions", []) if action.get("legal", True)]
        if decision_id in requirements:
            property_level = requirements[decision_id]
            action = coarsest_entailing_legal_action(decision, property_level)
            if action is None:
                raise ValueError(
                    f"no legal generalization for decision {decision_id} entails {property_level}"
                )
            action_vector[decision_id] = str(action["action_id"])
        else:
            keep = next((action for action in actions if action.get("mode") == "keep"), None)
            if keep is None:
                raise ValueError(f"unrelated decision {decision_id} has no legal KEEP action")
            action_vector[decision_id] = str(keep["action_id"])

    missing = sorted(set(requirements) - seen)
    if missing:
        raise ValueError(f"unknown linked decisions: {missing}")
    return {
        "action_vector": action_vector,
        "action_vector_hash": _stable_hash(action_vector),
    }


def _answer_score(answer: str, accepted_values: Sequence[str]) -> float:
    if not accepted_values:
        return 0.0
    return max(fact_score(answer, value) for value in accepted_values)


def _permuted_reader_question(assertion: Mapping, permutation_index: int) -> str:
    question = str(assertion["question"])
    options = [str(option) for option in assertion.get("options") or []]
    if not options:
        return question
    semantic_identity = {
        "family": str(assertion.get("family", "")),
        "scope": str(assertion.get("scope", "")),
        "subtype": str(assertion.get("subtype", "")),
        "relation": str(assertion.get("relation", "")),
        "occurrence_ids": [str(value) for value in assertion.get("occurrence_ids") or []],
        "group_id": str(assertion.get("group_id", "")),
        "question": question,
        "options": sorted(options),
        "accepted_values": sorted(
            str(value) for value in assertion.get("accepted_values") or []
        ),
        "decision_requirements": {
            str(key): str(value)
            for key, value in dict(assertion.get("decision_requirements") or {}).items()
        },
    }
    ordered = sorted(
        options,
        key=lambda option: _stable_hash({
            "assertion_semantics": semantic_identity,
            "option": option,
        }),
    )
    shift = permutation_index % len(ordered)
    permutation = ordered[shift:] + ordered[:shift]
    return f"{question}\nOptions: {' | '.join(permutation)}"


def _call_context_reader(reader, questions, context, *, refresh: bool):
    if not refresh:
        return reader(questions, context)
    return reader(questions, context, refresh=True)


def validate_context_assertions(
    assertions: Sequence[Mapping],
    *,
    original_context: str,
    representative_context: str,
    placeholder_context: str,
    reader: Callable[[list[str], str], Sequence[str]],
    threshold: float = 1.0,
    stability_repetitions: int = 1,
    option_permutations: int = 1,
    stability_threshold: float = 1.0,
) -> tuple[list[dict], dict[str, dict]]:
    """Apply frozen repeated original/generalization/placeholder reader checks."""
    rows = [row for row in assertions if row.get("family") == "context"]
    if stability_repetitions < 1 or option_permutations < 1:
        raise ValueError("reader stability repetitions and option permutations must be positive")
    if not 0.0 < stability_threshold <= 1.0:
        raise ValueError("reader stability threshold must be in (0, 1]")

    trials_by_assertion: dict[str, list[dict]] = defaultdict(list)
    for repetition in range(stability_repetitions):
        for permutation_index in range(option_permutations):
            questions = [
                _permuted_reader_question(row, permutation_index) for row in rows
            ]
            refresh = repetition > 0
            original_answers = list(_call_context_reader(
                reader, questions, original_context, refresh=refresh
            ))
            representative_answers = list(_call_context_reader(
                reader, questions, representative_context, refresh=refresh
            ))
            placeholder_answers = list(_call_context_reader(
                reader, questions, placeholder_context, refresh=refresh
            ))
            if not all(len(answers) == len(rows) for answers in (
                original_answers, representative_answers, placeholder_answers
            )):
                raise ValueError("reader returned the wrong number of answers")
            for row, original, representative, placeholder in zip(
                rows, original_answers, representative_answers, placeholder_answers
            ):
                values = list(row.get("accepted_values") or [])
                scores = {
                    "original": _answer_score(original, values),
                    "representative": _answer_score(representative, values),
                    "placeholder": _answer_score(placeholder, values),
                }
                assertion_id = str(row.get("assertion_id") or _stable_hash(dict(row)))
                trials_by_assertion[assertion_id].append({
                    "repetition": repetition,
                    "permutation_index": permutation_index,
                    "scores": scores,
                    "passed": (
                        scores["original"] >= threshold
                        and scores["representative"] >= threshold
                        and scores["placeholder"] < threshold
                    ),
                })

    accepted, evidence = [], {}
    for row in rows:
        assertion_id = str(row.get("assertion_id") or _stable_hash(dict(row)))
        trials = trials_by_assertion[assertion_id]
        passing_fraction = sum(trial["passed"] for trial in trials) / len(trials)
        if passing_fraction >= stability_threshold:
            verdict = "accepted"
        elif any(trial["passed"] for trial in trials):
            verdict = "unstable"
        else:
            verdict = "unsupported"
        evidence[assertion_id] = {
            "scores": trials[0]["scores"],
            "verdict": verdict,
            "stability": {
                "repetitions": stability_repetitions,
                "option_permutations": option_permutations,
                "threshold": stability_threshold,
                "passing_fraction": passing_fraction,
                "trials": trials,
            },
        }
        if verdict == "accepted":
            accepted.append(dict(row))
    return accepted, evidence


def score_utility(
    artifact: Mapping,
    doc_id: str,
    *,
    doc_p: str,
    out_final: str,
    reader: Callable[[list[str], str], Sequence[str]],
    reader_refresh: bool = False,
) -> dict:
    """Score one document with one context-reader batch and deterministic delivered checks."""
    assertions = [
        row for row in artifact.get("assertions", {}).values()
        if row.get("doc_id") == doc_id and row.get("status", "accepted") == "accepted"
    ]
    assertions.sort(key=lambda row: str(row["assertion_id"]))
    context_rows = [row for row in assertions if row.get("family") == "context"]
    context_answers = list(reader(
        [_permuted_reader_question(row, 0) for row in context_rows],
        doc_p,
        **({"refresh": True} if reader_refresh else {}),
    )) if context_rows else []
    if len(context_answers) != len(context_rows):
        raise ValueError("reader returned the wrong number of answers")

    scores: dict[str, float] = {}
    for row, answer in zip(context_rows, context_answers):
        scores[str(row["assertion_id"])] = _answer_score(
            answer, list(row.get("accepted_values") or [])
        )
    for row in assertions:
        if row.get("family") != "delivered":
            continue
        contract = row.get("scoring_contract") or {}
        if contract.get("kind") != "contains":
            raise ValueError(f"unsupported delivered scoring contract: {contract}")
        expected = str(contract.get("value", ""))
        scores[str(row["assertion_id"])] = fact_score(out_final, expected)

    return {"component_scores": scores}
