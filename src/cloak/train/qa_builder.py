"""QA-builder v2 artifact weighting, support anchors, validation, and scoring."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from math import isfinite
from numbers import Real

from cloak.train.reward import QA_BASE_URL, QA_MODEL, canon, fact_score

RELATION_TEACHER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
RELATION_TEACHER_BASE_URL = "https://openrouter.ai/api/v1"
RELATION_TEACHER_PROMPT_VERSION = "qa-relation-teacher-v1"
RELATION_TEACHER_RESPONSE_SCHEMA = {"type": "relations-array", "version": 1}
RELATION_TEACHER_REVISION = "qa-relation-teacher-r1"
CONTEXT_READER_PROMPT_VERSION = "qa-context-reader-v1"
CONTEXT_READER_RESPONSE_SCHEMA = {"type": "answers-array", "version": 1}
CONTEXT_READER_REVISION = "qa-reader-r1"
DEFAULT_CONTEXT_READER_PIN = {
    "model": QA_MODEL,
    "endpoint": QA_BASE_URL,
    "prompt_version": CONTEXT_READER_PROMPT_VERSION,
    "response_schema": CONTEXT_READER_RESPONSE_SCHEMA,
    "revision": CONTEXT_READER_REVISION,
}
READER_PIN_FIELDS = frozenset({
    "model", "endpoint", "prompt_version", "response_schema", "revision",
})
RELATION_ONTOLOGY = (
    "treated_with",
    "monitored_by",
    "contraindicated_because_of",
    "causes_or_explains",
    "referred_to",
    "has_status",
    "has_category",
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
        "connector_patterns": (
            r"\s+(?:(?:is|was|are|were|has been|had been|is being|was being)\s+)?"
            r"treated\s+with\s+",
        ),
    },
    "monitored_by": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["monitored_by"],
        "cues": ("monitored by",),
        "connector_patterns": (
            r"\s+(?:(?:is|was|are|were|has been|had been|is being|was being)\s+)?"
            r"monitored\s+by\s+",
        ),
    },
    "contraindicated_because_of": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["contraindicated_because_of"],
        "cues": ("contraindicated because of",),
        "connector_patterns": (
            r"\s+(?:(?:is|was|are|were|has been|had been|is being|was being)\s+)?"
            r"contraindicated\s+because\s+of\s+",
        ),
    },
    "causes_or_explains": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["causes_or_explains"],
        "cues": ("causes", "explains"),
        "connector_patterns": (r"\s+(?:causes|explains)\s+",),
    },
    "referred_to": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["referred_to"],
        "cues": ("referred to",),
        "connector_patterns": (
            r"\s+(?:(?:is|was|are|were|has been|had been|is being|was being)\s+)?"
            r"referred\s+to\s+",
        ),
    },
    "has_status": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["has_status"],
        "cues": ("has status", "status is"),
        "connector_patterns": (r"\s+(?:has\s+status|status\s+is)\s+",),
    },
    "has_category": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["has_category"],
        "cues": ("has category", "category is"),
        "connector_patterns": (r"\s+(?:has\s+category|category\s+is)\s+",),
    },
}
_NEGATION_PATTERN = re.compile(r"\b(?:no|not|without|denies|denied)\b")
_CLAUSE_DELIMITER_PATTERN = re.compile(r"[\n.!?;]")
ACI_REQUIRED_SECTIONS = (
    "HISTORY OF PRESENT ILLNESS",
    "ASSESSMENT",
    "PLAN",
)
_ACI_HEADING_PATTERN = re.compile(
    r"^\s*(?P<heading>[A-Z][A-Z /&-]*[A-Z])\s*(?::\s*(?P<content>.*))?$"
)
_ACI_ROW_DELIMITER = re.compile(r"\s+(?:—|–|-)\s+")


class OpenRouterRelationTeacher:
    """Optional cached JSON relation proposer pinned to Nemotron on OpenRouter."""

    @property
    def pin(self) -> dict:
        return {
            "provider": "openrouter",
            "model": RELATION_TEACHER_MODEL,
            "base_url": RELATION_TEACHER_BASE_URL,
            "prompt_version": RELATION_TEACHER_PROMPT_VERSION,
            "response_schema": deepcopy(RELATION_TEACHER_RESPONSE_SCHEMA),
            "revision": RELATION_TEACHER_REVISION,
        }

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

    @property
    def pin(self) -> dict:
        return deepcopy(DEFAULT_CONTEXT_READER_PIN)

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
                max_tokens=512,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        self._client = client

    def __call__(self, questions: list[str], context: str) -> list[str]:
        if not questions:
            return []
        prompt = (
            "Answer every question using the document. Preserve semantic category and "
            "function distinctions stated or entailed by the document. If an answer is not "
            "supported, use NONE. Return only a JSON object with an answers array in the "
            "same order as the questions.\n\n"
            f"DOCUMENT:\n{context}\n\n"
            f"QUESTIONS:\n{json.dumps(questions, indent=2)}"
        )
        payload = json.loads(self._client.generate(prompt))
        answers = payload.get("answers") if isinstance(payload, dict) else None
        if not isinstance(answers, list) or len(answers) != len(questions):
            raise ValueError("context reader returned the wrong number of answers")
        return ["" if str(answer).strip().upper() == "NONE" else str(answer).strip()
                for answer in answers]


_batched_context_reader = None


def read_context_batch(questions: list[str], context: str) -> list[str]:
    global _batched_context_reader
    if _batched_context_reader is None:
        _batched_context_reader = BatchedContextReader()
    return _batched_context_reader(questions, context)


read_context_batch.pin = deepcopy(DEFAULT_CONTEXT_READER_PIN)


class AciTaskAdapter:
    """Authoritative deterministic ACI delivered facts and relation compilation."""

    task_pin = "aci-utility-v1"
    relation_contract = ACI_RELATION_CONTRACT
    controlled_runtime_types = frozenset({
        "health-condition", "drug", "medical-procedure", "LOC",
    })
    semantic_type_contract = {
        "drug": {
            "placeholder_labels": frozenset({"drug", "medication", "treatment"}),
            "role_patterns": (
                ("taking", r"\b(?:take|takes|taking)\s+(?:(?:a|an|the)\s+)?{surface}"),
                (
                    "prescribed",
                    r"\b(?:prescribe|prescribes|prescribed)\s+"
                    r"(?:(?:a|an|the)\s+)?{surface}",
                ),
                (
                    "prescribed",
                    r"{surface}\s+(?:(?:is|was|are|were)\s+)?prescribed\b",
                ),
                (
                    "treated with",
                    r"\b(?:is|was|are|were)?\s*treated\s+with\s+{surface}",
                ),
            ),
            "question": (
                'What specific treatment category is the patient taking or prescribed in '
                'this context: "{locator}"?'
            ),
        },
        "health condition": {
            "placeholder_labels": frozenset({
                "health condition", "condition", "disease", "illness",
                "medical condition",
            }),
            "role_patterns": (
                ("history", r"\bhistory\s+of\s+{surface}"),
                ("diagnosis", r"\bdiagnosis\s+of\s+{surface}"),
                ("diagnosed", r"\bdiagnosed\s+with\s+{surface}"),
                ("presentation", r"\b(?:presents|presented)\s+with\s+{surface}"),
                ("complaint", r"\bcomplaints?\s+of\s+{surface}"),
            ),
            "question": (
                'What specific condition category is documented in the diagnosis, history, '
                'presentation, or complaint in this context: "{locator}"?'
            ),
        },
        "medical procedure": {
            "placeholder_labels": frozenset({"medical procedure", "procedure", "test"}),
            "role_patterns": (
                ("ordered", r"\border(?:s|ed|ing)?\s+{surface}"),
                ("test", r"\b(?:test|panel)\s+(?:for\s+)?{surface}"),
            ),
            "question": (
                'What specific procedure or test category is ordered in this context: '
                '"{locator}"?'
            ),
        },
        "loc": {
            "placeholder_labels": frozenset({"loc", "location", "place"}),
            "role_patterns": (
                ("located", r"\blocated\s+in\s+{surface}"),
                ("address", r"\baddress(?:\s+is|\s+at)?\s+{surface}"),
                ("travel", r"\btravel(?:s|ed|ing)?\s+to\s+{surface}"),
            ),
            "question": (
                'What specific location category is referenced by the location, address, or '
                'travel role in this context: "{locator}"?'
            ),
        },
    }

    def __init__(self, references: Mapping[str, str]):
        self._references = dict(references)

    def validate_environment(self, frozen_environment: Mapping) -> None:
        unsupported = sorted({
            str(decision.get("runtime_type", ""))
            for document in frozen_environment.get("documents", {}).values()
            for decision in document.get("decisions", [])
            if decision.get("runtime_type") not in self.controlled_runtime_types
        })
        if unsupported:
            raise ValueError(
                "legacy or unsupported ACI decision types in frozen environment: "
                + ", ".join(unsupported)
            )

    def deterministic_candidates(
        self,
        doc_id: str,
        document: str,
        environment_document: Mapping,
    ) -> list[dict]:
        return [
            *self.semantic_property_candidates(doc_id, document, environment_document),
            *self.delivered_candidates(
                doc_id, document, self._references[doc_id], environment_document
            ),
        ]

    def semantic_property_candidates(
        self,
        doc_id: str,
        document: str,
        environment_document: Mapping,
    ) -> list[dict]:
        """Compile safe category probes from legal frozen action entailments."""
        occurrences_by_decision: dict[str, list[Mapping]] = defaultdict(list)
        protected_terms = []
        for occurrence in environment_document.get("occurrences", []):
            protected_terms.extend(_occurrence_protected_terms(occurrence))
            decision_id = occurrence.get("decision_id")
            if decision_id is None or not occurrence.get("controlled", True):
                continue
            occurrences_by_decision[str(decision_id)].append(occurrence)

        records = []
        for decision in environment_document.get("decisions", []):
            if not decision.get("controlled", True):
                continue
            decision_id = str(decision["decision_id"])
            occurrences = occurrences_by_decision.get(decision_id, [])
            occurrence_ids = [str(row["occurrence_id"]) for row in occurrences]
            runtime_type = str(decision.get("runtime_type", ""))
            type_contract = self.semantic_type_contract.get(_semantic_label(runtime_type))
            actions = [
                action for action in decision.get("actions", [])
                if action.get("legal", True)
                and action.get("mode") not in {"keep", "placeholder"}
            ]
            properties = []
            supporting_actions: dict[str, list[str]] = defaultdict(list)
            for action in actions:
                for value in action.get("entails") or []:
                    property_level = canon(str(value))
                    if not property_level:
                        continue
                    if property_level not in supporting_actions:
                        properties.append(property_level)
                    supporting_actions[property_level].append(str(action["action_id"]))
            if not properties:
                records.append(_rejection_record(
                    reason="not_generated",
                    detail_reason="no_legal_support_properties",
                    attempt={
                        "doc_id": doc_id,
                        "decision_id": decision_id,
                        "runtime_type": runtime_type,
                        "occurrence_ids": occurrence_ids,
                        "definition_version": "aci-semantic-property-v1",
                    },
                    evidence={
                        "source": "deterministic_template",
                        "decision_id": decision_id,
                        "runtime_type": runtime_type,
                        "occurrence_ids": occurrence_ids,
                        "legal_action_hashes": [
                            _stable_hash({
                                "action_id": action.get("action_id"),
                                "mode": action.get("mode"),
                                "entails": list(action.get("entails") or []),
                            })
                            for action in decision.get("actions", [])
                            if action.get("legal", True)
                        ],
                    },
                ))
                continue

            if type_contract is None:
                for property_level in properties:
                    attempt = {
                        "doc_id": doc_id,
                        "decision_id": decision_id,
                        "runtime_type": runtime_type,
                        "occurrence_ids": occurrence_ids,
                        "property_hash": _stable_hash(property_level),
                        "definition_version": "aci-semantic-property-v1",
                    }
                    records.append(_rejection_record(
                        reason="not_generated",
                        detail_reason="unsupported_runtime_type",
                        attempt=attempt,
                        evidence={
                            "source": "deterministic_template",
                            "decision_id": decision_id,
                            "runtime_type": runtime_type,
                            "occurrence_ids": occurrence_ids,
                            "property_hash": _stable_hash(property_level),
                            "supporting_action_hashes": [
                                _stable_hash(action_id)
                                for action_id in supporting_actions[property_level]
                            ],
                        },
                    ))
                continue

            locator, role_cue, locator_rejection = _task_role_context_locator(
                document,
                occurrences,
                protected_terms=[*protected_terms, *properties],
                role_patterns=type_contract["role_patterns"],
            )
            for property_level in properties:
                attempt = {
                    "doc_id": doc_id,
                    "decision_id": decision_id,
                    "runtime_type": runtime_type,
                    "occurrence_ids": occurrence_ids,
                    "property_hash": _stable_hash(property_level),
                    "definition_version": "aci-semantic-property-v1",
                }
                if (
                    _semantic_property_label(property_level)
                    in type_contract["placeholder_labels"]
                ):
                    records.append(_rejection_record(
                        reason="not_generated",
                        detail_reason="placeholder_type_only",
                        attempt=attempt,
                        evidence={
                            "source": "deterministic_template",
                            "decision_id": decision_id,
                            "runtime_type": runtime_type,
                            "occurrence_ids": occurrence_ids,
                            "property_hash": _stable_hash(property_level),
                            "supporting_action_hashes": [
                                _stable_hash(action_id)
                                for action_id in supporting_actions[property_level]
                            ],
                        },
                    ))
                    continue
                if locator is None:
                    records.append(_rejection_record(
                        reason="not_generated",
                        detail_reason=locator_rejection,
                        attempt=attempt,
                        evidence={
                            "source": "deterministic_template",
                            "decision_id": decision_id,
                            "runtime_type": runtime_type,
                            "occurrence_ids": occurrence_ids,
                            "property_hash": _stable_hash(property_level),
                            "supporting_action_hashes": [
                                _stable_hash(action_id)
                                for action_id in supporting_actions[property_level]
                            ],
                        },
                    ))
                    continue
                question = str(type_contract["question"]).format(locator=locator)
                if (
                    _question_leaks_answer(question, property_level, runtime_type)
                    or _question_leaks_protected_term(question, protected_terms)
                ):
                    records.append(_rejection_record(
                        reason="not_generated",
                        detail_reason="unsafe_template_leakage",
                        attempt=attempt,
                        evidence={
                            "source": "deterministic_template",
                            "decision_id": decision_id,
                            "runtime_type": runtime_type,
                            "occurrence_ids": occurrence_ids,
                            "property_hash": _stable_hash(property_level),
                            "supporting_action_hashes": [
                                _stable_hash(action_id)
                                for action_id in supporting_actions[property_level]
                            ],
                            "locator_hash": _stable_hash(locator),
                        },
                    ))
                    continue
                records.append({
                    "family": "context",
                    "scope": "linked",
                    "subtype": "semantic_property",
                    "occurrence_ids": occurrence_ids,
                    "group_id": (
                        f"semantic-property:{decision_id}:"
                        f"{_stable_hash(property_level)}"
                    ),
                    "question": question,
                    "accepted_values": [property_level],
                    "decision_requirements": {decision_id: property_level},
                    "evidence": {
                        "authority": "frozen_action_entails",
                        "template": "aci-context-category-v1",
                        "runtime_type": runtime_type,
                        "role_cue": role_cue,
                        "locator_hash": _stable_hash(locator),
                        "supporting_action_ids": supporting_actions[property_level],
                    },
                })
        return records

    def delivered_candidates(
        self,
        doc_id: str,
        document: str,
        reference: str,
        environment_document: Mapping,
    ) -> list[dict]:
        """Compile reference-backed ACI delivered assertions without clinical inference."""
        candidates = []
        parsed_reference = _parse_aci_note(reference)
        parsed_source = _parse_aci_note(document)
        occurrences_by_decision: dict[str, list[Mapping]] = defaultdict(list)
        occurrences_by_surface: dict[str, list[Mapping]] = defaultdict(list)
        occurrences_by_id = {}
        for occurrence in environment_document.get("occurrences", []):
            decision_id = occurrence.get("decision_id")
            if decision_id is None or not occurrence.get("controlled", True):
                continue
            occurrence_id = str(occurrence["occurrence_id"])
            occurrences_by_id[occurrence_id] = occurrence
            occurrences_by_decision[str(decision_id)].append(occurrence)
            occurrences_by_surface[canon(str(occurrence.get("surface", "")))].append(
                occurrence
            )
        for decision_id, occurrences in occurrences_by_decision.items():
            surface = str(occurrences[0].get("surface", ""))
            if not surface:
                continue
            occurrence_ids = [str(row["occurrence_id"]) for row in occurrences]
            if _contains(reference, surface):
                reference_match = re.search(
                    rf"(?<!\w){re.escape(surface)}(?!\w)",
                    reference,
                    flags=re.IGNORECASE,
                )
                candidates.append({
                    "family": "delivered",
                    "scope": "linked",
                    "subtype": "content",
                    "occurrence_ids": occurrence_ids,
                    "group_id": f"content:{decision_id}",
                    "scoring_contract": {"kind": "contains", "value": surface},
                    "evidence": {
                        "authority": "human_reference",
                        "provenance": "delivered_reference",
                        "reference_hash": _stable_hash(reference),
                        "reference_span": (
                            [reference_match.start(), reference_match.end()]
                            if reference_match is not None else None
                        ),
                    },
                })
                continue
            exact_spans = [
                {
                    "start": span[0],
                    "end": span[1],
                    "surface_hash": _stable_hash(str(occurrence.get("surface", ""))),
                }
                for occurrence in occurrences
                if (span := _exact_occurrence_span(document, occurrence)) is not None
            ]
            if exact_spans:
                candidates.append(_rejection_record(
                    reason="not_generated",
                    detail_reason="not_authoritative_for_delivery",
                    attempt={
                        "doc_id": doc_id,
                        "coverage_kind": "controlled_source_fact",
                        "decision_id": decision_id,
                        "occurrence_ids": occurrence_ids,
                        "source_hash": _stable_hash(document),
                        "definition_version": "aci-delivered-authority-v1",
                    },
                    evidence={
                        "source": "delivered_authority",
                        "coverage_kind": "controlled_source_fact",
                        "delivery_authority": "not_authoritative_for_delivery",
                        "decision_id": decision_id,
                        "occurrence_ids": occurrence_ids,
                        "source_spans": exact_spans,
                    },
                ))

        for name in ("age", "sex"):
            if name in parsed_reference["demographic"]:
                value = parsed_reference["demographic"][name]
                evidence = {
                    "authority": "human_reference",
                    "provenance": "delivered_reference",
                    "reference_hash": _stable_hash(reference),
                    "reference_span": parsed_reference["demographic_evidence"][name],
                }
            elif name in parsed_source["demographic"]:
                value = parsed_source["demographic"][name]
                evidence = {
                    "authority": "source_document_task_schema_fallback",
                    "provenance": "exact_doc_orig",
                    "source_span": parsed_source["demographic_evidence"][name],
                    "source_hash": _stable_hash(document),
                    "value_hash": _stable_hash(value),
                    "task_schema_required": True,
                    "reference_missing": True,
                }
            else:
                continue
            candidates.append({
                "family": "delivered",
                "scope": "global",
                "subtype": "field",
                "occurrence_ids": [],
                "group_id": f"field:demographic:{name}",
                "scoring_contract": {
                    "kind": "field_value",
                    "section": "DEMOGRAPHIC",
                    "field": name,
                    "value": value,
                },
                "evidence": evidence,
            })

        duplicated_conditions = _duplicated_aci_conditions(
            parsed_reference["assessment_rows"]
        )
        for row in parsed_reference["assessment_rows"]:
            if canon(row["condition"]) in duplicated_conditions:
                continue
            for field in ("category", "status"):
                candidates.append({
                    "family": "delivered",
                    "scope": "global",
                    "subtype": "field",
                    "occurrence_ids": [],
                    "group_id": f"field:assessment:{canon(row['condition'])}:{field}",
                    "scoring_contract": {
                        "kind": "field_value",
                        "section": "ASSESSMENT",
                        "row": row["condition"],
                        "field": field,
                        "value": row[field],
                    },
                    "evidence": {"authority": "human_reference"},
                })

        required_sections = list(ACI_REQUIRED_SECTIONS)
        if (
            all(parsed_reference["sections"].get(section) for section in required_sections)
            and parsed_reference["assessment_shape"]["kind"] in {"rows", "none"}
            and parsed_reference["plan_shape"]["kind"] in {"rows", "none"}
        ):
            candidates.append({
                "family": "delivered",
                "scope": "global",
                "subtype": "structure",
                "occurrence_ids": [],
                "group_id": "structure:required_sections",
                "scoring_contract": {
                    "kind": "required_sections",
                    "sections": required_sections,
                    "parseability": {
                        "assessment": parsed_reference["assessment_shape"],
                        "plan": parsed_reference["plan_shape"],
                    },
                },
                "evidence": {"authority": "human_reference"},
            })

        for row in parsed_reference["plan_rows"]:
            if canon(row["condition"]) in duplicated_conditions:
                continue
            values = (row["condition"], row["treatment"], row["test"])
            relation_occurrences = [
                occurrences_by_surface.get(canon(value), []) for value in values
            ]
            occurrence_ids = [
                str(rows[0]["occurrence_id"])
                for rows in relation_occurrences if len(rows) == 1
            ]
            linked = len(occurrence_ids) == len(values)
            candidates.append({
                "family": "delivered",
                "scope": "linked" if linked else "global",
                "subtype": "exact_relation",
                "occurrence_ids": occurrence_ids if linked else [],
                "group_id": "relation:" + ":".join(canon(value) for value in values),
                "scoring_contract": {
                    "kind": "exact_relation",
                    "section": "PLAN",
                    "condition": row["condition"],
                    "treatment": row["treatment"],
                    "test": row["test"],
                },
                "evidence": {"authority": "human_reference"},
            })
        for source_relation in _deterministic_source_relations(
            document, environment_document, self.relation_contract
        ):
            occurrence_ids = source_relation["occurrence_ids"]
            if _aci_reference_authorizes_relation(
                source_relation["relation"],
                occurrence_ids,
                occurrences_by_id,
                parsed_reference,
            ):
                continue
            candidates.append(_rejection_record(
                reason="not_generated",
                detail_reason="not_authoritative_for_delivery",
                attempt={
                    "doc_id": doc_id,
                    "coverage_kind": "controlled_source_relation",
                    "relation": source_relation["relation"],
                    "occurrence_ids": occurrence_ids,
                    "source_span": source_relation["source_span"],
                    "quote_hash": source_relation["quote_hash"],
                    "definition_version": "aci-delivered-authority-v1",
                },
                evidence={
                    "source": "delivered_authority",
                    "coverage_kind": "controlled_source_relation",
                    "delivery_authority": "not_authoritative_for_delivery",
                    "relation": source_relation["relation"],
                    "occurrence_ids": occurrence_ids,
                    "source_span": source_relation["source_span"],
                    "quote_hash": source_relation["quote_hash"],
                },
            ))
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


def _validated_reader_pin(value, *, label: str = "reader_pin") -> dict:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a structured non-empty mapping")
    missing = sorted(READER_PIN_FIELDS - set(value))
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")
    empty = sorted(
        field for field in READER_PIN_FIELDS
        if value.get(field) is None or value.get(field) == "" or value.get(field) == {}
    )
    if empty:
        raise ValueError(f"{label} has empty required fields: {empty}")
    return deepcopy(dict(value))


def _validated_build_reader_pin(reader, pins: Mapping) -> dict:
    provided = _validated_reader_pin(pins.get("reader_pin"))
    injected = getattr(reader, "pin", None)
    if not isinstance(injected, Mapping) or not injected:
        raise ValueError("injected reader requires an explicit structured pin")
    actual = _validated_reader_pin(injected, label="injected reader pin")
    if provided != actual:
        raise ValueError("reader_pin must match the injected reader pin")
    return provided


def _rejection_record(
    *,
    reason: str,
    detail_reason: str,
    attempt: Mapping,
    evidence: Mapping,
) -> dict:
    attempt_value = dict(attempt)
    doc_id = str(attempt_value.get("doc_id", ""))
    if not doc_id:
        raise ValueError("rejection attempt requires doc_id")
    attempt_hash = _stable_hash(attempt_value)
    identity = {
        "reason": reason,
        "detail_reason": detail_reason,
        "attempt_hash": attempt_hash,
    }
    return {
        "status": "rejected",
        "reason": reason,
        "detail_reason": detail_reason,
        "rejection_id": _stable_hash(identity),
        "attempt_hash": attempt_hash,
        "doc_id": doc_id,
        "evidence": dict(evidence),
    }


def _stable_rejection_reason(detail_reason: str) -> str:
    if detail_reason in {"answer_leakage", "protected_locator"}:
        return "leakage"
    if detail_reason in {
        "not_generated",
        "generation_failed",
        "leakage",
        "unsupported",
        "floor_answerable",
        "unstable",
        "infrastructure_failed",
    }:
        return detail_reason
    return "invalid"


def _semantic_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", canon(value)))


def _semantic_property_label(value: str) -> str:
    tokens = _semantic_label(value).split()
    while tokens and tokens[0] in {"a", "an", "the"}:
        tokens.pop(0)
    return " ".join(tokens)


_LEAKAGE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "being", "by", "did", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "with",
})


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", canon(value))
        if len(token) > 2 and token not in _LEAKAGE_STOPWORDS
    }


def _normalized_aliases(value) -> list[str]:
    if value is None:
        return []
    values = [value] if isinstance(value, str) else list(value)
    aliases = []
    for alias in values:
        exact_alias = str(alias)
        if exact_alias.strip() and exact_alias not in aliases:
            aliases.append(exact_alias)
    return aliases


def _occurrence_protected_terms(occurrence: Mapping) -> list[str]:
    terms = [str(occurrence.get("surface", "")).strip()]
    terms.extend(alias.strip() for alias in _normalized_aliases(occurrence.get("aliases")))
    return [term for term in terms if term]


def _placeholder_meaning_tokens(runtime_type: str) -> set[str]:
    contract = AciTaskAdapter.semantic_type_contract.get(_semantic_label(runtime_type), {})
    return {
        token
        for label in contract.get("placeholder_labels", ())
        for token in _meaningful_tokens(str(label))
    }


def _question_leaks_answer(question: str, answer: str, runtime_type: str) -> bool:
    answer_tokens = _meaningful_tokens(answer) - _placeholder_meaning_tokens(runtime_type)
    return bool(answer_tokens & _meaningful_tokens(question))


def _question_leaks_protected_term(question: str, protected_terms: Sequence[str]) -> bool:
    if any(_contains(question, term) for term in protected_terms if term):
        return True
    question_tokens = _meaningful_tokens(question)
    return any(_meaningful_tokens(term) & question_tokens for term in protected_terms)


def _sentence_at(document: str, position: int) -> str:
    left = max(
        document.rfind(".", 0, position),
        document.rfind("!", 0, position),
        document.rfind("?", 0, position),
        document.rfind("\n", 0, position),
    ) + 1
    boundaries = [
        value for value in (
            document.find(".", position),
            document.find("!", position),
            document.find("?", position),
            document.find("\n", position),
        ) if value >= 0
    ]
    right = min(boundaries) + 1 if boundaries else len(document)
    return document[left:right].strip()


def _task_role_context_locator(
    document: str,
    occurrences: Sequence[Mapping],
    *,
    protected_terms: Sequence[str],
    role_patterns: Sequence[tuple[str, str]],
) -> tuple[str | None, str | None, str]:
    targets = []
    for occurrence in occurrences:
        surface = str(occurrence.get("surface", "")).strip()
        start = occurrence.get("start")
        if surface and isinstance(start, int) and 0 <= start < len(document):
            targets.append((start, surface))
            continue
        if surface:
            targets.extend(
                (match.start(), surface)
                for match in re.finditer(
                    rf"(?<!\w){re.escape(surface)}(?!\w)",
                    document,
                    flags=re.IGNORECASE,
                )
            )
    if not targets:
        return None, None, "no_safe_contextual_locator"
    locator = None
    matched_cue = None
    has_surviving_context = False
    protected_tokens = {
        token for term in protected_terms for token in _meaningful_tokens(term)
    }
    for position, surface in sorted(set(targets)):
        sentence = _sentence_at(document, position)
        sentence_tokens = _meaningful_tokens(sentence)
        if not sentence_tokens - protected_tokens:
            continue
        has_surviving_context = True
        escaped_surface = rf"(?<!\w){re.escape(surface)}(?!\w)"
        matched_cue = next((
            cue for cue, pattern in role_patterns
            if re.search(
                pattern.format(surface=escaped_surface),
                sentence,
                flags=re.IGNORECASE,
            )
        ), None)
        if matched_cue is not None:
            locator = sentence
            break
    if locator is None:
        return (
            None,
            None,
            "no_task_role_cue" if has_surviving_context else "no_safe_contextual_locator",
        )
    target_terms = {
        str(occurrence.get("surface", "")).strip()
        for occurrence in occurrences
        if str(occurrence.get("surface", "")).strip()
    }

    def redact(terms: Sequence[str], marker: str) -> None:
        nonlocal locator
        for term in sorted(
            set(terms), key=lambda value: (-len(value), canon(value), value)
        ):
            if not term:
                continue
            locator = re.sub(
                rf"(?<!\w){re.escape(term)}(?!\w)",
                marker,
                locator,
                flags=re.IGNORECASE,
            )

    redact(list(target_terms), "[target item]")
    redact([
        term for term in protected_terms
        if canon(term) not in {canon(target) for target in target_terms}
    ], "[protected item]")
    locator = re.sub(r"\s+", " ", locator).strip()
    return locator, matched_cue, ""


def _contains(text: str, value: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(canon(value))}(?!\w)", canon(text)))


def _exact_occurrence_span(document: str, occurrence: Mapping) -> list[int] | None:
    start = occurrence.get("start")
    end = occurrence.get("end")
    surface = str(occurrence.get("surface", ""))
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not 0 <= start < end <= len(document)
        or document[start:end] != surface
    ):
        return None
    return [start, end]


def _parse_aci_note(text: str) -> dict:
    """Parse only the fixed ACI headings and unambiguous assessment/plan triples."""
    sections: dict[str, list[str]] = {}
    current_section = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        heading = _ACI_HEADING_PATTERN.match(line)
        heading_name = heading.group("heading").strip() if heading else ""
        if heading_name in ACI_REQUIRED_SECTIONS:
            current_section = heading_name
            sections[current_section] = []
            content = heading.group("content")
            if content and content.strip():
                sections[current_section].append(content.strip())
        elif line and current_section is not None:
            sections[current_section].append(line)

    demographic_match = re.search(
        r"\b(\d{1,3}-year-old)\s+(male|female)\b", text, re.IGNORECASE
    )
    demographic = (
        {"age": demographic_match.group(1), "sex": demographic_match.group(2)}
        if demographic_match else {}
    )
    demographic_evidence = (
        {
            "age": [demographic_match.start(1), demographic_match.end(1)],
            "sex": [demographic_match.start(2), demographic_match.end(2)],
        }
        if demographic_match else {}
    )
    assessment_lines = sections.get("ASSESSMENT", [])
    plan_lines = sections.get("PLAN", [])
    assessment_rows = _parse_aci_rows(
        assessment_lines, ("condition", "category", "status")
    )
    plan_rows = _parse_aci_rows(
        plan_lines, ("condition", "treatment", "test")
    )
    return {
        "sections": sections,
        "demographic": demographic,
        "demographic_evidence": demographic_evidence,
        "assessment_rows": assessment_rows,
        "assessment_shape": _aci_section_shape(assessment_lines, assessment_rows),
        "plan_rows": plan_rows,
        "plan_shape": _aci_section_shape(plan_lines, plan_rows),
    }


def _parse_aci_rows(lines: Sequence[str], fields: Sequence[str]) -> list[dict]:
    rows = []
    for line in lines:
        values = [value.strip() for value in _ACI_ROW_DELIMITER.split(line) if value.strip()]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values)))
    return rows


def _aci_section_shape(lines: Sequence[str], rows: Sequence[Mapping]) -> dict:
    if len(lines) == 1 and canon(lines[0]) == "none":
        return {"kind": "none", "count": 0}
    if lines and len(rows) == len(lines):
        return {"kind": "rows", "count": len(rows)}
    return {"kind": "invalid", "count": len(rows)}


def _duplicated_aci_conditions(rows: Sequence[Mapping]) -> set[str]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[canon(str(row["condition"]))] += 1
    return {condition for condition, count in counts.items() if count > 1}


def _validated_structural_cap(
    assertions: Sequence[Mapping],
    structural_cap: float | None,
) -> float | None:
    has_structure = any(
        row.get("family") == "delivered" and row.get("subtype") == "structure"
        for row in assertions
    )
    if structural_cap is None:
        if has_structure:
            raise ValueError("structural_cap is required for delivered structure assertions")
        return None
    if (
        isinstance(structural_cap, bool)
        or not isinstance(structural_cap, Real)
        or not 0.0 < float(structural_cap) < 1.0
    ):
        raise ValueError("structural_cap must be a numeric value between zero and one")
    return float(structural_cap)


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
    return any(
        _relation_clause_connects_arguments(
            relation, clause, occurrence_ids, occurrences, relation_contract
        )
        for clause in re.split(r"[\n.!?;]+", quote)
        if clause.strip()
    )


def _relation_clause_connects_arguments(
    relation: str,
    clause: str,
    occurrence_ids: Sequence[str],
    occurrences: Mapping[str, Mapping],
    relation_contract: Mapping[str, Mapping],
) -> bool:
    canonical_clause = canon(clause)
    positions = []
    for occurrence_id in occurrence_ids:
        surface = canon(str(occurrences[occurrence_id].get("surface", "")))
        match = re.search(rf"(?<!\w){re.escape(surface)}(?!\w)", canonical_clause)
        if match is None:
            return False
        positions.append((match.start(), match.end()))
    if positions != sorted(positions):
        return False
    connector_text = canonical_clause[positions[0][1]:positions[-1][0]]
    connector_patterns = relation_contract[relation].get("connector_patterns", ())
    return any(
        re.fullmatch(pattern, connector_text) is not None
        for pattern in connector_patterns
    )


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
        affirmative_form = _NEGATION_PATTERN.sub("", sentence)
        if _relation_evidence_connects_arguments(
            relation, affirmative_form, occurrence_ids, occurrences, relation_contract
        ):
            return True
    return False


def _deterministic_source_relations(
    document: str,
    environment_document: Mapping,
    relation_contract: Mapping[str, Mapping],
) -> list[dict]:
    occurrences = {
        str(row["occurrence_id"]): row
        for row in environment_document.get("occurrences", [])
        if row.get("controlled", row.get("decision_id") is not None)
        and _exact_occurrence_span(document, row) is not None
    }
    relations = []
    seen = set()
    for clause_match in re.finditer(r"[^\n.!?;]+(?:[.!?;]|$)", document):
        clause = clause_match.group(0)
        clause_span = (clause_match.start(), clause_match.end())
        if _NEGATION_PATTERN.search(canon(clause)):
            continue
        contained = [
            occurrence_id for occurrence_id, occurrence in occurrences.items()
            if clause_span[0] <= occurrence["start"] < occurrence["end"] <= clause_span[1]
        ]
        for relation in relation_contract:
            for first_id in contained:
                for second_id in contained:
                    occurrence_ids = [first_id, second_id]
                    if first_id == second_id or not _relation_argument_types_are_legal(
                        relation, occurrence_ids, occurrences, relation_contract
                    ):
                        continue
                    if not _relation_clause_connects_arguments(
                        relation, clause, occurrence_ids, occurrences, relation_contract
                    ):
                        continue
                    identity = (relation, first_id, second_id, *clause_span)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    relations.append({
                        "relation": relation,
                        "occurrence_ids": occurrence_ids,
                        "source_span": [clause_span[0], clause_span[1]],
                        "quote_hash": _stable_hash(clause),
                    })
    return relations


def _aci_reference_authorizes_relation(
    relation: str,
    occurrence_ids: Sequence[str],
    occurrences: Mapping[str, Mapping],
    parsed_reference: Mapping,
) -> bool:
    if len(occurrence_ids) != 2:
        return False
    first = canon(str(occurrences[occurrence_ids[0]].get("surface", "")))
    second = canon(str(occurrences[occurrence_ids[1]].get("surface", "")))
    if relation == "treated_with":
        return any(
            canon(str(row["condition"])) == first
            and canon(str(row["treatment"])) == second
            for row in parsed_reference.get("plan_rows", [])
        )
    if relation == "monitored_by":
        return any(
            canon(str(row["condition"])) == first
            and canon(str(row["test"])) == second
            for row in parsed_reference.get("plan_rows", [])
        )
    return False


def _exact_substring_starts(document: str, quote: str) -> list[int]:
    starts = []
    cursor = 0
    while quote:
        start = document.find(quote, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + 1
    return starts


def _resolve_relation_evidence_span(
    document: str,
    quote: str,
    proposed_start,
) -> tuple[tuple[int, int] | None, str | None]:
    starts = _exact_substring_starts(document, quote)
    if not quote or not starts:
        return None, "invalid_evidence"
    if proposed_start is None:
        if len(starts) != 1:
            return None, "ambiguous_evidence_start"
        start = starts[0]
    else:
        if isinstance(proposed_start, bool) or not isinstance(proposed_start, int):
            return None, "invalid_evidence_start"
        if proposed_start not in starts:
            return None, "invalid_evidence_start"
        start = proposed_start
    return (start, start + len(quote)), None


def _evidence_span_contains_occurrences(
    document: str,
    evidence_span: tuple[int, int],
    occurrence_ids: Sequence[str],
    occurrences: Mapping[str, Mapping],
) -> bool:
    evidence_start, evidence_end = evidence_span
    for occurrence_id in occurrence_ids:
        occurrence = occurrences[occurrence_id]
        start = occurrence.get("start")
        end = occurrence.get("end")
        surface = str(occurrence.get("surface", ""))
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(document)
            or document[start:end] != surface
            or not evidence_start <= start < end <= evidence_end
        ):
            return False
    return True


def _relation_evidence_connects_selected_occurrences(
    relation: str,
    quote: str,
    evidence_span: tuple[int, int],
    occurrence_ids: Sequence[str],
    occurrences: Mapping[str, Mapping],
    relation_contract: Mapping[str, Mapping],
) -> bool:
    evidence_start, _ = evidence_span
    local_spans = []
    clause_spans = []
    for occurrence_id in occurrence_ids:
        occurrence = occurrences[occurrence_id]
        local_start = int(occurrence["start"]) - evidence_start
        local_end = int(occurrence["end"]) - evidence_start
        surface = str(occurrence.get("surface", ""))
        if not 0 <= local_start < local_end <= len(quote):
            return False
        if quote[local_start:local_end] != surface:
            return False
        if _CLAUSE_DELIMITER_PATTERN.search(quote, local_start, local_end):
            return False
        previous_delimiters = list(
            _CLAUSE_DELIMITER_PATTERN.finditer(quote, 0, local_start)
        )
        next_delimiter = _CLAUSE_DELIMITER_PATTERN.search(quote, local_end)
        clause_spans.append((
            previous_delimiters[-1].end() if previous_delimiters else 0,
            next_delimiter.start() if next_delimiter else len(quote),
        ))
        local_spans.append((local_start, local_end))
    if local_spans != sorted(local_spans):
        return False
    if any(first[1] > second[0] for first, second in zip(local_spans, local_spans[1:])):
        return False
    if len(set(clause_spans)) != 1:
        return False
    connector_text = canon(quote[local_spans[0][1]:local_spans[-1][0]])
    return any(
        re.fullmatch(pattern, connector_text) is not None
        for pattern in relation_contract[relation].get("connector_patterns", ())
    )


def relation_teacher_prompt(
    doc_id: str,
    document: str,
    environment_document: Mapping,
) -> str:
    """Build the single bounded teacher prompt from frozen IDs and legal properties."""
    inventory = []
    decisions = {
        str(row["decision_id"]): row
        for row in environment_document.get("decisions", [])
    }
    for occurrence in environment_document.get("occurrences", []):
        decision = decisions.get(str(occurrence.get("decision_id")), {})
        properties = list(dict.fromkeys(
            canon(str(property_level))
            for action in decision.get("actions", [])
            if action.get("legal", True)
            and action.get("mode") not in {"keep", "placeholder"}
            for property_level in (action.get("entails") or [])
            if canon(str(property_level))
        ))
        inventory.append({
            "occurrence_id": occurrence.get("occurrence_id"),
            "start": occurrence.get("start"),
            "end": occurrence.get("end"),
            "surface": occurrence.get("surface"),
            "aliases": _normalized_aliases(occurrence.get("aliases")),
            "runtime_type": occurrence.get("runtime_type"),
            "legal_support_properties": properties,
        })
    schema = {
        "relations": [{
            "subtype": "contextual_relation",
            "relation": f"one of: {', '.join(RELATION_ONTOLOGY)}",
            "argument_occurrence_ids": ["existing occurrence IDs"],
            "support_properties": {
                "occurrence ID": "one exact legal_support_properties value"
            },
            "answer_occurrence_id": "one argument occurrence ID",
            "answer_property": "that occurrence's exact selected support property",
            "question": "natural question not containing a protected surface or its answer",
            "evidence_quote": "one exact source substring directly connecting all arguments",
            "evidence_start": (
                "exact zero-based quote start; required when evidence_quote occurs more "
                "than once, optional when unique"
            ),
        }]
    }
    return (
        "Extract only explicit, task-relevant clinical relations from the source. "
        "Use the closed relation vocabulary and existing occurrence IDs. Select support "
        "properties verbatim from the inventory. Do not use medical knowledge absent from "
        "the source. Abstain with an empty relations list when evidence is insufficient. "
        "Return only the requested JSON object.\n\n"
        f"DOCUMENT ID:\n{doc_id}\n\n"
        f"OCCURRENCE INVENTORY:\n{json.dumps(inventory, indent=2)}\n\n"
        f"OUTPUT SCHEMA:\n{json.dumps(schema, indent=2)}\n\n"
        f"SOURCE DOCUMENT:\n{document}"
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
            canon(str(property_level))
            for action in decision.get("actions", [])
            if action.get("legal", True)
            and action.get("mode") not in {"keep", "placeholder"}
            for property_level in (action.get("entails") or [])
            if canon(str(property_level))
        }

    accepted, rejected = [], []
    for index, proposal_value in enumerate(proposals):
        proposal = dict(proposal_value)
        proposal_hash = _stable_hash(proposal)

        def reject(reason: str) -> None:
            quote = str(proposal.get("evidence_quote", ""))
            quote_span, _ = _resolve_relation_evidence_span(
                document, quote, proposal.get("evidence_start")
            )
            evidence = {
                "source": "relation_teacher",
                "proposal_index": index,
                "argument_occurrence_ids": [
                    (
                        str(value) if str(value) in occurrences
                        else _stable_hash(str(value))
                    )
                    for value in proposal.get("argument_occurrence_ids", [])
                ],
                "evidence_quote_hash": _stable_hash(quote),
                "evidence_span": list(quote_span) if quote_span is not None else None,
            }
            record = _rejection_record(
                reason=_stable_rejection_reason(reason),
                detail_reason=reason,
                attempt={
                    "doc_id": doc_id,
                    "proposal_index": index,
                    "proposal_hash": proposal_hash,
                    "definition_version": "contextual-relation-v1",
                },
                evidence=evidence,
            )
            record["proposal_index"] = index
            record["proposal_hash"] = proposal_hash
            rejected.append(record)

        proposed_subtype = proposal.get("subtype")
        if proposed_subtype not in {None, "contextual_relation"}:
            reject("invalid_subtype")
            continue
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
        evidence_span, evidence_error = _resolve_relation_evidence_span(
            document, quote, proposal.get("evidence_start")
        )
        if evidence_error is not None:
            reject(evidence_error)
            continue
        if not _evidence_span_contains_occurrences(
            document, evidence_span, occurrence_ids, occurrences
        ):
            reject("invalid_evidence_occurrence")
            continue
        if not _relation_evidence_connects_selected_occurrences(
            relation,
            quote,
            evidence_span,
            occurrence_ids,
            occurrences,
            relation_contract,
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
        answer_runtime_type = str(
            occurrences[answer_occurrence_id].get("runtime_type", "")
        )
        if _question_leaks_answer(question, answer_property, answer_runtime_type):
            reject("answer_leakage")
            continue
        protected_terms = [
            term
            for occurrence in occurrences.values()
            for term in _occurrence_protected_terms(occurrence)
        ]
        if _question_leaks_protected_term(question, protected_terms):
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
            "accepted_values": [answer_property],
            "decision_requirements": decision_requirements,
            "evidence": {
                "authority": "source_document",
                "proposal_hash": proposal_hash,
                "source_span": {
                    "start": evidence_span[0],
                    "end": evidence_span[1],
                    "quote_hash": _stable_hash(quote),
                },
                "argument_spans": {
                    occurrence_id: [
                        int(occurrences[occurrence_id]["start"]),
                        int(occurrences[occurrence_id]["end"]),
                    ]
                    for occurrence_id in occurrence_ids
                },
            },
        })
    return accepted, rejected


def assign_static_weights(
    assertions: Sequence[Mapping],
    family_budgets: Mapping[str, float],
    *,
    structural_cap: float | None = None,
) -> tuple[list[dict], dict]:
    """Assign family -> group -> assertion weights with a fixed family denominator."""
    structural_cap = _validated_structural_cap(assertions, structural_cap)
    groups: dict[str, dict[str, list[Mapping]]] = defaultdict(lambda: defaultdict(list))
    for assertion in assertions:
        family = str(assertion["family"])
        if family not in family_budgets:
            raise ValueError(f"unknown assertion family: {family}")
        groups[family][str(assertion["group_id"])].append(assertion)

    weights: dict[str, float] = {}
    for family, family_groups in groups.items():
        family_budget = float(family_budgets[family])
        structural_groups = {
            group_id for group_id, rows in family_groups.items()
            if family == "delivered" and any(row.get("subtype") == "structure" for row in rows)
        }
        semantic_groups = set(family_groups) - structural_groups
        if structural_groups and structural_cap is not None:
            structural_budget = min(
                family_budget * float(structural_cap),
                family_budget * len(structural_groups) / len(family_groups),
            )
            group_budgets = {
                group_id: structural_budget / len(structural_groups)
                for group_id in structural_groups
            }
            if semantic_groups:
                group_budgets.update({
                    group_id: (family_budget - structural_budget) / len(semantic_groups)
                    for group_id in semantic_groups
                })
        else:
            group_budgets = {
                group_id: family_budget / len(family_groups)
                for group_id in family_groups
            }
        for group_id, rows in family_groups.items():
            assertion_weight = group_budgets[group_id] / len(rows)
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


def package_utility_artifact(
    frozen_environment: Mapping,
    candidates_by_document: Mapping[str, Sequence[Mapping]],
    *,
    family_budgets: Mapping[str, float],
    structural_cap: float | None = None,
    pins: Mapping,
) -> dict:
    """Compile validated candidates into one deterministic, link-checked utility artifact."""
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
        decision_ids = {
            str(row["decision_id"])
            for row in environment_document.get("decisions", [])
        }
        if len(decision_ids) != len(environment_document.get("decisions", [])):
            raise ValueError(f"duplicate decision ids for {doc_id}")
        dangling_decisions = sorted({
            str(row["decision_id"])
            for row in occurrences.values()
            if row.get("controlled", row.get("decision_id") is not None)
            and row.get("decision_id") not in decision_ids
        })
        if dangling_decisions:
            raise ValueError(
                f"unknown decision links for {doc_id}: {dangling_decisions}"
            )
        controlled = [
            str(row["decision_id"])
            for row in environment_document.get("decisions", [])
            if row.get("controlled", True)
        ]
        compiled = []
        compiled_ids = set()
        linked_decisions = set()
        for candidate in candidates_by_document.get(doc_id, []):
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

        weighted, weight_state = assign_static_weights(
            compiled, family_budgets, structural_cap=structural_cap
        )
        for row in weighted:
            assertions[row["assertion_id"]] = row
        missing_families = weight_state["missing_family_budgets"]
        documents[doc_id] = {
            "environment_document_hash": environment_document.get(
                "environment_document_hash", _stable_hash(environment_document)
            ),
            "measurement_state": (
                "unsupported" if not weighted else "partial" if missing_families else "measured"
            ),
            **weight_state,
            "assertion_ids": [row["assertion_id"] for row in weighted],
            "controlled_decision_ids": controlled,
            "occurrence_to_decision": {
                occurrence_id: str(row["decision_id"])
                for occurrence_id, row in occurrences.items()
            },
            "decision_keys": [{
                "decision_id": str(row["decision_id"]),
                "runtime_type": row.get("runtime_type"),
                "canonical_key": row.get("canonical_key"),
            } for row in environment_document.get("decisions", [])],
            "occurrences": deepcopy(environment_document.get("occurrences", [])),
            "decisions": deepcopy(environment_document.get("decisions", [])),
            "uncovered_decision_ids": [
                decision_id for decision_id in controlled if decision_id not in linked_decisions
            ],
        }

    all_candidates = [
        row for candidates in candidates_by_document.values() for row in candidates
    ]
    structural_cap = _validated_structural_cap(all_candidates, structural_cap)
    artifact = {
        "artifact_version": "utility-assertions-v1",
        "environment_hash": frozen_environment.get("environment_hash"),
        **dict(pins),
        "family_budgets": dict(family_budgets),
        "structural_cap": structural_cap,
        "documents": documents,
        "assertions": assertions,
        "rejections": {"summary_by_reason": {}},
    }
    artifact["artifact_hash"] = _stable_hash(artifact)
    return artifact


def artifact_views(artifact: Mapping) -> tuple[dict, dict]:
    """Project inspectable assertion and QA-pair views from one normative artifact."""
    view_header = {
        "source_artifact_version": artifact.get("artifact_version"),
        "source_artifact_hash": artifact.get("artifact_hash"),
        "family_budgets": deepcopy(artifact.get("family_budgets", {})),
    }
    rejection_state = deepcopy(artifact.get("rejections", {
        "summary_by_reason": {}, "records": [],
    }))
    rejection_records = sorted(
        rejection_state.get("records", []),
        key=lambda row: str(row.get("rejection_id", "")),
    )
    assertions_by_document: dict[str, list[dict]] = defaultdict(list)
    for assertion in artifact.get("assertions", {}).values():
        assertions_by_document[str(assertion["doc_id"])].append(deepcopy(assertion))

    assertions_view = {
        "artifact_version": "utility-assertions-view-v1",
        **deepcopy(view_header),
        "documents": {},
        "rejections": rejection_state,
    }
    qa_pairs_view = {
        "artifact_version": "utility-qa-pairs-view-v1",
        **deepcopy(view_header),
        "documents": {},
        "rejections": deepcopy(rejection_state),
    }
    for doc_id, document in sorted(artifact.get("documents", {}).items()):
        occurrence_inventory = {
            str(row["occurrence_id"]): deepcopy(row)
            for row in document.get("occurrences", [])
        }
        decision_inventory = {
            str(row["decision_id"]): deepcopy(row)
            for row in document.get("decisions", [])
        }
        rows = sorted(
            assertions_by_document.get(str(doc_id), []),
            key=lambda row: str(row["assertion_id"]),
        )
        groups = {
            "structure": [],
            "field_content": [],
            "exact_relation": [],
            "contextual": [],
        }
        for row in rows:
            subtype = row.get("subtype")
            if subtype == "structure":
                groups["structure"].append(row)
            elif subtype in {"field", "content"}:
                groups["field_content"].append(row)
            elif subtype == "exact_relation":
                groups["exact_relation"].append(row)
            elif row.get("family") == "context":
                groups["contextual"].append(row)

        assertion_document = {
            key: deepcopy(value)
            for key, value in document.items()
            if key not in {"decisions", "occurrences"}
        }
        assertion_document["occurrences"] = deepcopy(occurrence_inventory)
        assertion_document["decisions"] = deepcopy(decision_inventory)
        assertion_document["assertion_groups"] = groups
        assertions_view["documents"][doc_id] = assertion_document

        occurrence_to_decision = {
            str(key): str(value)
            for key, value in document.get("occurrence_to_decision", {}).items()
        }
        document_rejections = [
            deepcopy(record) for record in rejection_records
            if str(record.get("doc_id", "")) == str(doc_id)
        ]
        decisions = {}
        rejection_assignments: set[str] = set()
        for decision in decision_inventory.values():
            decision_id = str(decision["decision_id"])
            linked_pairs = [
                deepcopy(row) for row in rows
                if row.get("family") == "context"
                and decision_id in {
                    *map(str, (row.get("decision_requirements") or {}).keys()),
                    *(occurrence_to_decision.get(str(occurrence_id), "")
                      for occurrence_id in row.get("occurrence_ids", [])),
                }
            ]
            linked_rejections = []
            for record in document_rejections:
                evidence = record.get("evidence") or {}
                explicit_decisions = {
                    str(value) for value in (
                        record.get("decision_id"), evidence.get("decision_id")
                    ) if value is not None
                }
                linked_occurrences = [
                    *record.get("occurrence_ids", []),
                    *evidence.get("occurrence_ids", []),
                    *evidence.get("argument_occurrence_ids", []),
                ]
                linked_decisions = {
                    occurrence_to_decision.get(str(occurrence_id))
                    for occurrence_id in linked_occurrences
                }
                if decision_id in explicit_decisions | linked_decisions:
                    linked_rejections.append(deepcopy(record))
                    rejection_assignments.add(str(record.get("rejection_id", "")))
            decisions[decision_id] = {
                **deepcopy(decision),
                "legal_actions": [
                    deepcopy(action) for action in decision.get("actions", [])
                    if action.get("legal", True)
                ],
                "qa_pairs": linked_pairs,
                "rejections": linked_rejections,
            }
        qa_pairs_view["documents"][doc_id] = {
            "measurement_state": document.get("measurement_state"),
            "present_family_budgets": deepcopy(
                document.get("present_family_budgets", [])
            ),
            "missing_family_budgets": deepcopy(
                document.get("missing_family_budgets", [])
            ),
            "occurrences": occurrence_inventory,
            "decisions": decisions,
            "unassigned_rejections": [
                record for record in document_rejections
                if str(record.get("rejection_id", "")) not in rejection_assignments
            ],
        }
    return assertions_view, qa_pairs_view


def freeze_ranker_environment(
    ranker_environment: Mapping,
    *,
    occurrences_by_document: Mapping[str, Sequence[Mapping]] | None = None,
) -> dict:
    """Migrate embedded ranker spans to stable occurrence/decision identities, without detection."""
    documents: dict[str, dict] = {}
    for corpus, per_document in ranker_environment.get("corpora", {}).items():
        for doc_id, document in per_document.items():
            decisions_by_key: dict[tuple[str, str], dict] = {}
            for span in document.get("spans", []):
                runtime_type = str(span.get("type", ""))
                surface = str(span.get("surface", ""))
                decision_key = (runtime_type, canon(surface))
                decision_id = _stable_hash({
                    "doc_id": doc_id,
                    "runtime_type": runtime_type,
                    "canonical_surface": decision_key[1],
                })
                actions = []
                action_ids = set()
                for action in span.get("actions", []):
                    declared_mode = str(action.get("mode", "level"))
                    if declared_mode not in {"level", "keep", "placeholder"}:
                        raise ValueError(
                            f"invalid action mode for decision {decision_id}: {declared_mode}"
                        )
                    fill = action.get("fill")
                    if declared_mode == "placeholder" and action.get("keep"):
                        raise ValueError(
                            f"placeholder action cannot be KEEP for decision {decision_id}"
                        )
                    source_fill = bool(fill) and canon(str(fill)) == decision_key[1]
                    is_keep = bool(action.get("keep")) or declared_mode == "keep" or (
                        declared_mode == "level" and source_fill
                    )
                    if action.get("source_identity") and not is_keep:
                        raise ValueError(
                            f"source_identity is reserved for KEEP actions on {decision_id}"
                        )
                    if is_keep and not source_fill:
                        raise ValueError(
                            f"KEEP action must preserve the source identity for {decision_id}"
                        )
                    mode = (
                        "keep" if is_keep else
                        "placeholder" if declared_mode == "placeholder" else
                        "level"
                    )
                    action_semantics = {
                        **dict(action),
                        "mode": mode,
                        "fill": decision_key[1] if is_keep else fill,
                        "legal": bool(action.get("legal", True)),
                    }
                    action_semantics.pop("action_id", None)
                    if mode == "keep":
                        action_semantics.update({
                            "keep": True,
                            "source_identity": True,
                            "entails": [decision_key[1]],
                        })
                        action_semantics.pop("coarseness_rank", None)
                    elif mode == "placeholder":
                        action_semantics["entails"] = []
                        action_semantics.pop("coarseness_rank", None)
                    else:
                        rank = action.get("coarseness_rank", action.get("aset"))
                        if action_semantics["legal"] and (
                            isinstance(rank, bool)
                            or not isinstance(rank, Real)
                            or not isfinite(float(rank))
                        ):
                            raise ValueError(
                                "legal level action requires numeric coarseness_rank "
                                f"or aset for decision {decision_id}"
                            )
                        if isinstance(rank, Real) and not isinstance(rank, bool):
                            action_semantics["coarseness_rank"] = rank
                        action_semantics["entails"] = (
                            [canon(str(fill))] if fill else []
                        )
                    action_id = _stable_hash({
                        "decision_id": decision_id,
                        "action": action_semantics,
                    })
                    if action_id in action_ids:
                        raise ValueError(f"duplicate action semantics for decision {decision_id}")
                    action_ids.add(action_id)
                    actions.append({
                        **action_semantics,
                        "action_id": action_id,
                    })
                if not any(action["mode"] == "keep" for action in actions):
                    keep_semantics = {
                        "fill": decision_key[1],
                        "mode": "keep",
                        "keep": True,
                        "source_identity": True,
                        "legal": True,
                        "entails": [decision_key[1]],
                    }
                    keep_action = {
                        **keep_semantics,
                        "action_id": _stable_hash({
                            "decision_id": decision_id,
                            "action": keep_semantics,
                        }),
                    }
                    placeholder_index = next(
                        (index for index, action in enumerate(actions)
                         if action["mode"] == "placeholder"),
                        len(actions),
                    )
                    actions.insert(placeholder_index, keep_action)
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
                occurrences.append({
                    "occurrence_id": occurrence_id,
                    "start": row.get("start"),
                    "end": row.get("end"),
                    "surface": surface,
                    "aliases": _normalized_aliases(row.get("aliases")),
                    "runtime_type": runtime_type,
                    "polarity": row.get("polarity", "unknown"),
                    "detector_provenance": row.get("detector_provenance", {
                        "source": "frozen_arms_migration",
                        "score": row.get("score"),
                    }),
                    "overlap_disposition": row.get("overlap_disposition", "accepted"),
                    "decision_id": decision["decision_id"] if decision is not None else None,
                    "controlled": decision is not None,
                })
                if decision is not None:
                    decision["occurrence_ids"].append(occurrence_id)
            occurrences_by_id = {
                str(row["occurrence_id"]): row for row in occurrences
            }
            for decision in decisions_by_key.values():
                protected_aliases = []
                for occurrence_id in decision["occurrence_ids"]:
                    for alias in occurrences_by_id[occurrence_id]["aliases"]:
                        if alias not in protected_aliases:
                            protected_aliases.append(alias)
                decision["protected_aliases"] = protected_aliases
            frozen_document = {
                "corpus": corpus,
                "occurrences": occurrences,
                "decisions": list(decisions_by_key.values()),
            }
            frozen_document["environment_document_hash"] = _stable_hash(frozen_document)
            documents[doc_id] = frozen_document
    frozen = {"artifact_version": "occurrence-decisions-v1", "documents": documents}
    frozen["environment_hash"] = _stable_hash(frozen)
    return frozen


def frozen_occurrences_from_arms(
    arms: Mapping,
    *,
    detector_provenance: Mapping | None = None,
) -> dict[str, list[dict]]:
    """Read controlled occurrence rows from an already-frozen arms artifact."""
    return {
        doc_id: [
            {
                **dict(row),
                **({
                    "detector_provenance": {
                        **dict(detector_provenance),
                        "score": row.get("score"),
                    }
                } if detector_provenance is not None else {}),
            }
            for row in document["tau_walk"][1] if row.get("lattice")
        ]
        for corpus, documents in arms.items()
        if corpus != "_meta"
        for doc_id, document in documents.items()
    }


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
    reader_pin = _validated_build_reader_pin(reader, pins)
    validate_environment = getattr(task_adapter, "validate_environment", None)
    if validate_environment is not None:
        validate_environment(frozen_environment)
    family_budgets = threshold_manifest["family_budgets"]
    min_contextual_relations = int(threshold_manifest.get(
        "min_contextual_relation_assertions",
        threshold_manifest.get("min_context_assertions", 0),
    ))
    if min_contextual_relations < 0:
        raise ValueError("min_contextual_relation_assertions must be non-negative")
    reader_threshold = float(threshold_manifest.get("reader_threshold", 1.0))
    stability_repetitions = int(threshold_manifest.get("reader_stability_repetitions", 1))
    option_permutations = int(threshold_manifest.get("reader_option_permutations", 1))
    stability_threshold = float(threshold_manifest.get("reader_stability_threshold", 1.0))
    candidates_by_document: dict[str, list[dict]] = {}
    rejection_records: list[dict] = []

    def preserve_rejection(record_value: Mapping, *, doc_id: str) -> None:
        record = dict(record_value)
        detail_reason = str(record.get("detail_reason") or record.get("reason") or "invalid")
        stable_reason = _stable_rejection_reason(str(record.get("reason") or detail_reason))
        attributed_doc_id = str(record.get("doc_id") or doc_id)
        if attributed_doc_id != doc_id:
            raise ValueError(
                f"rejection document mismatch: {attributed_doc_id} != {doc_id}"
            )
        record["status"] = "rejected"
        record["reason"] = stable_reason
        record["detail_reason"] = detail_reason
        record["doc_id"] = doc_id
        if not record.get("attempt_hash"):
            record["attempt_hash"] = _stable_hash({
                "doc_id": doc_id,
                "reason": stable_reason,
                "detail_reason": detail_reason,
                "rejection_id": record.get("rejection_id"),
                "evidence": dict(record.get("evidence") or {}),
                "definition_version": "qa-builder-attempt-v1",
            })
        if not record.get("rejection_id"):
            record["rejection_id"] = _stable_hash({
                "doc_id": doc_id,
                "reason": stable_reason,
                "detail_reason": detail_reason,
                "attempt_hash": record["attempt_hash"],
                "definition_version": "qa-builder-rejection-v1",
            })
        rejection_records.append(record)

    def reject_context_candidate(
        *,
        doc_id: str,
        candidate: Mapping,
        reason: str,
        detail_reason: str,
        anchor: Mapping | None = None,
        validation: Mapping | None = None,
        error: Exception | None = None,
        extra_evidence: Mapping | None = None,
    ) -> None:
        candidate_hash = _stable_hash(candidate)
        attempt = {
            "doc_id": doc_id,
            "candidate_hash": candidate_hash,
            "reason": reason,
            "definition_version": "context-validation-v1",
        }
        evidence = {
            "source": "context_validation",
            "candidate_hash": candidate_hash,
            "subtype": candidate.get("subtype"),
            "occurrence_ids": list(candidate.get("occurrence_ids") or []),
        }
        if anchor is not None:
            evidence.update({
                "joint_anchor_action_vector": anchor["action_vector"],
                "joint_anchor_hash": anchor["action_vector_hash"],
            })
        if validation is not None:
            evidence["validation"] = dict(validation)
        if error is not None:
            error_type = type(error).__name__
            attempt["error_type"] = error_type
            evidence["error_type"] = error_type
        if extra_evidence is not None:
            evidence.update(dict(extra_evidence))
        preserve_rejection(_rejection_record(
            reason=reason,
            detail_reason=detail_reason,
            attempt=attempt,
            evidence=evidence,
        ), doc_id=doc_id)

    for doc_id, environment_document in frozen_environment.get("documents", {}).items():
        source = source_documents[doc_id]
        decisions = environment_document.get("decisions", [])
        occurrences = {
            str(row["occurrence_id"]): row
            for row in environment_document.get("occurrences", [])
        }
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

        def validate_candidate_rows(rows: Sequence[Mapping]) -> list[dict]:
            accepted_rows = []
            for candidate in rows:
                if candidate.get("family") != "context":
                    accepted_rows.append(dict(candidate))
                    continue
                occurrence_ids = [
                    str(value) for value in candidate.get("occurrence_ids") or []
                ]
                linked_occurrences = [occurrences.get(value) for value in occurrence_ids]
                requirements = {
                    str(key): value
                    for key, value in dict(
                        candidate.get("decision_requirements") or {}
                    ).items()
                }
                linked_decision_ids = {
                    str(occurrence.get("decision_id"))
                    for occurrence in linked_occurrences
                    if occurrence is not None and occurrence.get("decision_id") is not None
                }
                if (
                    not occurrence_ids
                    or any(occurrence is None for occurrence in linked_occurrences)
                    or not linked_decision_ids
                    or not linked_decision_ids.issubset(requirements)
                ):
                    reject_context_candidate(
                        doc_id=doc_id,
                        candidate=candidate,
                        reason="unsupported",
                        detail_reason="linked_occurrence_not_transformed",
                    )
                    continue
                try:
                    anchor = build_joint_representative_anchor(candidate, decisions)
                except ValueError:
                    reject_context_candidate(
                        doc_id=doc_id,
                        candidate=candidate,
                        reason="unsupported",
                        detail_reason="no_joint_representative_anchor",
                    )
                    continue
                representative_context = render_action_vector(
                    doc_id, anchor["action_vector"]
                )
                protected_terms = list(dict.fromkeys(
                    term
                    for occurrence in linked_occurrences
                    for term in _occurrence_protected_terms(occurrence)
                ))
                surviving_terms = [
                    term for term in protected_terms
                    if _contains(representative_context, term)
                ]
                if surviving_terms:
                    reject_context_candidate(
                        doc_id=doc_id,
                        candidate=candidate,
                        reason="leakage",
                        detail_reason="representative_protected_identity_survived",
                        anchor=anchor,
                        extra_evidence={
                            "protected_term_hashes": [
                                _stable_hash(term) for term in surviving_terms
                            ],
                        },
                    )
                    continue
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
                    reject_context_candidate(
                        doc_id=doc_id,
                        candidate=candidate,
                        reason="infrastructure_failed",
                        detail_reason="context_reader_failed",
                        anchor=anchor,
                        error=error,
                    )
                    continue
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
                    reject_context_candidate(
                        doc_id=doc_id,
                        candidate=candidate,
                        reason=reason,
                        detail_reason=(
                            "reader_unstable" if reason == "unstable"
                            else "placeholder_answerable" if reason == "floor_answerable"
                            else "three_point_gate_failed"
                        ),
                        anchor=anchor,
                        validation=evidence_row,
                    )
                    continue
                row = dict(candidate)
                row["expected_action_support"] = {
                    "joint_anchor_action_vector": anchor["action_vector"],
                    "joint_anchor_hash": anchor["action_vector_hash"],
                    "property_level": requirements,
                }
                row["evidence"] = {
                    **dict(row.get("evidence") or {}),
                    "validation": evidence_row,
                }
                accepted_rows.append(row)
            return accepted_rows

        deterministic_records = [
            dict(row) for row in task_adapter.deterministic_candidates(
                doc_id, source, environment_document
            )
        ]
        deterministic_candidates = []
        for record in deterministic_records:
            if record.get("status") == "rejected":
                preserve_rejection(record, doc_id=doc_id)
            else:
                deterministic_candidates.append(record)
        accepted = validate_candidate_rows(deterministic_candidates)
        contextual_relation_count = sum(
            row.get("family") == "context"
            and row.get("subtype") == "contextual_relation"
            for row in accepted
        )
        if contextual_relation_count < min_contextual_relations:
            if relation_teacher is None:
                preserve_rejection(_rejection_record(
                    reason="not_generated",
                    detail_reason="contextual_relation_threshold_unmet_no_teacher",
                    attempt={
                        "doc_id": doc_id,
                        "accepted_contextual_relation_count": contextual_relation_count,
                        "required_contextual_relation_count": min_contextual_relations,
                        "definition_version": "contextual-relation-v1",
                    },
                    evidence={
                        "source": "relation_escalation",
                        "accepted_contextual_relation_count": contextual_relation_count,
                        "required_contextual_relation_count": min_contextual_relations,
                    },
                ), doc_id=doc_id)
            else:
                prompt = relation_teacher_prompt(doc_id, source, environment_document)
                prompt_hash = _stable_hash(prompt)
                try:
                    proposals = relation_teacher.propose(prompt)
                    if not proposals:
                        preserve_rejection(_rejection_record(
                            reason="not_generated",
                            detail_reason="teacher_abstained",
                            attempt={
                                "doc_id": doc_id,
                                "prompt_hash": _stable_hash(prompt),
                                "definition_version": "contextual-relation-v1",
                            },
                            evidence={
                                "source": "relation_teacher",
                                "prompt_hash": _stable_hash(prompt),
                                "proposal_count": 0,
                                "required_contextual_relation_count": (
                                    min_contextual_relations
                                ),
                            },
                        ), doc_id=doc_id)
                    else:
                        relation_candidates, relation_rejections = (
                            task_adapter.compile_relations(
                                doc_id, source, environment_document, proposals
                            )
                        )
                        relation_candidates = [dict(row) for row in relation_candidates]
                        for relation_candidate in relation_candidates:
                            relation_candidate["evidence"] = {
                                **dict(relation_candidate.get("evidence") or {}),
                                "prompt_hash": prompt_hash,
                            }
                        for rejection in relation_rejections:
                            preserve_rejection(rejection, doc_id=doc_id)
                        accepted_relations = validate_candidate_rows(
                            [dict(row) for row in relation_candidates]
                        )
                        accepted.extend(accepted_relations)
                        contextual_relation_count = sum(
                            row.get("family") == "context"
                            and row.get("subtype") == "contextual_relation"
                            for row in accepted
                        )
                        if (
                            contextual_relation_count < min_contextual_relations
                            and (
                                accepted_relations
                                or not relation_candidates and not relation_rejections
                            )
                        ):
                            preserve_rejection(_rejection_record(
                                reason="not_generated",
                                detail_reason=(
                                    "contextual_relation_threshold_unmet_after_teacher"
                                ),
                                attempt={
                                    "doc_id": doc_id,
                                    "prompt_hash": _stable_hash(prompt),
                                    "accepted_contextual_relation_count": (
                                        contextual_relation_count
                                    ),
                                    "required_contextual_relation_count": (
                                        min_contextual_relations
                                    ),
                                    "definition_version": "contextual-relation-v1",
                                },
                                evidence={
                                    "source": "relation_teacher",
                                    "prompt_hash": _stable_hash(prompt),
                                    "proposal_count": len(proposals),
                                    "accepted_contextual_relation_count": (
                                        contextual_relation_count
                                    ),
                                    "required_contextual_relation_count": (
                                        min_contextual_relations
                                    ),
                                },
                            ), doc_id=doc_id)
                except Exception as error:
                    preserve_rejection(_rejection_record(
                        reason="generation_failed",
                        detail_reason="teacher_generation_failed",
                        attempt={
                            "doc_id": doc_id,
                            "prompt_hash": _stable_hash(prompt),
                            "error_type": type(error).__name__,
                            "definition_version": "contextual-relation-v1",
                        },
                        evidence={
                            "source": "relation_teacher",
                            "prompt_hash": _stable_hash(prompt),
                            "error_type": type(error).__name__,
                        },
                    ), doc_id=doc_id)
        candidates_by_document[doc_id] = accepted

    artifact = package_utility_artifact(
        frozen_environment,
        candidates_by_document,
        family_budgets=family_budgets,
        structural_cap=threshold_manifest.get("structural_cap"),
        pins={
            **dict(pins),
            "reader_pin": reader_pin,
            "threshold_manifest": dict(threshold_manifest),
        },
    )
    summary_by_reason: dict[str, int] = defaultdict(int)
    for record in rejection_records:
        summary_by_reason[str(record["reason"])] += 1
    artifact["rejections"] = {
        "summary_by_reason": dict(summary_by_reason),
        "records": rejection_records,
    }
    artifact["artifact_hash"] = _stable_hash({
        key: value for key, value in artifact.items() if key != "artifact_hash"
    })
    return artifact


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
            candidates = [
                action for action in actions
                if action.get("mode") not in {"keep", "placeholder"}
                and not action.get("source_identity", False)
                and property_level in (action.get("entails") or [])
            ]
            if not candidates:
                raise ValueError(
                    f"no legal generalization for decision {decision_id} entails {property_level}"
                )
            for action in candidates:
                rank = action.get("coarseness_rank")
                if (
                    isinstance(rank, bool)
                    or not isinstance(rank, Real)
                    or not isfinite(float(rank))
                ):
                    raise ValueError(
                        f"legal level action for decision {decision_id} lacks numeric "
                        "coarseness_rank"
                    )
            maximum_rank = max(float(action["coarseness_rank"]) for action in candidates)
            coarsest = [
                action for action in candidates
                if float(action["coarseness_rank"]) == maximum_rank
            ]
            if len(coarsest) != 1:
                raise ValueError(
                    f"multiple legal generalizations for decision {decision_id} share "
                    f"maximum coarseness_rank {maximum_rank}"
                )
            selected = coarsest[0]
            action_vector[decision_id] = str(selected["action_id"])
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
    ordered = sorted(
        options,
        key=lambda option: _stable_hash({
            "assertion_id": str(assertion.get("assertion_id", "")),
            "option": option,
        }),
    )
    shift = permutation_index % len(ordered)
    permutation = ordered[shift:] + ordered[:shift]
    return f"{question}\nOptions: {' | '.join(permutation)}"


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
            original_answers = list(reader(questions, original_context))
            representative_answers = list(reader(questions, representative_context))
            placeholder_answers = list(reader(questions, placeholder_context))
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


def _score_delivered_contract(contract: Mapping, out_final: str, parsed_output: Mapping) -> float:
    kind = contract.get("kind")
    if kind == "contains":
        return fact_score(out_final, str(contract.get("value", "")))
    if kind == "required_sections":
        sections = [str(section) for section in contract.get("sections") or []]
        if (
            not sections
            or not set(ACI_REQUIRED_SECTIONS).issubset(sections)
            or not all(parsed_output["sections"].get(section) for section in sections)
        ):
            return 0.0
        parseability = contract.get("parseability") or {
            "assessment": {"kind": "rows", "count": 1},
            "plan": {"kind": "rows", "count": 1},
        }
        if not isinstance(parseability, Mapping):
            return 0.0
        for section in ("assessment", "plan"):
            expected = parseability.get(section)
            observed = parsed_output[f"{section}_shape"]
            if (
                not isinstance(expected, Mapping)
                or expected.get("kind") not in {"rows", "none"}
                or isinstance(expected.get("count"), bool)
                or not isinstance(expected.get("count"), int)
                or expected["count"] < 0
                or expected["kind"] == "none" and expected["count"] != 0
                or expected["kind"] != observed["kind"]
                or expected["count"] != observed["count"]
            ):
                return 0.0
        return 1.0
    if kind == "field_value":
        section = str(contract.get("section", ""))
        field = str(contract.get("field", ""))
        expected = canon(str(contract.get("value", "")))
        if section == "DEMOGRAPHIC":
            observed = parsed_output["demographic"].get(field)
            return float(observed is not None and canon(observed) == expected)
        if section == "ASSESSMENT":
            row_key = canon(str(contract.get("row", "")))
            matches = [
                row for row in parsed_output["assessment_rows"]
                if canon(row["condition"]) == row_key
            ]
            return float(
                len(matches) == 1
                and field in {"category", "status"}
                and canon(matches[0][field]) == expected
            )
        return 0.0
    if kind == "exact_relation":
        expected = {
            field: canon(str(contract.get(field, "")))
            for field in ("condition", "treatment", "test")
        }
        if not all(expected.values()) or contract.get("section") != "PLAN":
            return 0.0
        return float(any(
            all(canon(row[field]) == value for field, value in expected.items())
            for row in parsed_output["plan_rows"]
        ))
    raise ValueError(f"unsupported delivered scoring contract: {contract}")


def score_utility(
    artifact: Mapping,
    doc_id: str,
    *,
    doc_p: str,
    out_final: str,
    reader: Callable[[list[str], str], Sequence[str]],
) -> dict:
    """Score one document with one context-reader batch and deterministic delivered checks."""
    _validated_build_reader_pin(reader, artifact)
    assertions = [
        row for row in artifact.get("assertions", {}).values()
        if row.get("doc_id") == doc_id and row.get("status", "accepted") == "accepted"
    ]
    assertions.sort(key=lambda row: str(row["assertion_id"]))
    context_rows = [row for row in assertions if row.get("family") == "context"]
    context_answers = list(reader(
        [str(row["question"]) for row in context_rows], doc_p
    )) if context_rows else []
    if len(context_answers) != len(context_rows):
        raise ValueError("reader returned the wrong number of answers")

    scores: dict[str, float] = {}
    for row, answer in zip(context_rows, context_answers):
        scores[str(row["assertion_id"])] = _answer_score(
            answer, list(row.get("accepted_values") or [])
        )
    parsed_output = _parse_aci_note(out_final)
    for row in assertions:
        if row.get("family") != "delivered":
            continue
        contract = row.get("scoring_contract") or {}
        scores[str(row["assertion_id"])] = _score_delivered_contract(
            contract, out_final, parsed_output
        )

    numerator = sum(float(row["weight"]) * scores[str(row["assertion_id"])]
                    for row in assertions)
    denominator = float(
        artifact["documents"][doc_id]["utility_weight_denominator"]
    )
    utility = numerator / denominator if denominator else 0.0
    return {"component_scores": scores, "utility": utility}
