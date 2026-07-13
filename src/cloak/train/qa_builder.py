"""QA-builder v2 artifact weighting, support anchors, validation, and scoring."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from numbers import Real

from cloak.train.reward import QA_BASE_URL, QA_MODEL, canon, fact_score

RELATION_TEACHER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
RELATION_TEACHER_BASE_URL = "https://openrouter.ai/api/v1"
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


class AciTaskAdapter:
    """Authoritative deterministic ACI delivered facts and relation compilation."""

    task_pin = "aci-utility-v1"
    relation_contract = ACI_RELATION_CONTRACT

    def __init__(self, references: Mapping[str, str]):
        self._references = dict(references)

    def deterministic_candidates(
        self,
        doc_id: str,
        document: str,
        environment_document: Mapping,
    ) -> list[dict]:
        return self.delivered_candidates(
            doc_id, document, self._references[doc_id], environment_document
        )

    def delivered_candidates(
        self,
        doc_id: str,
        document: str,
        reference: str,
        environment_document: Mapping,
    ) -> list[dict]:
        """Compile reference-backed ACI delivered assertions without clinical inference."""
        del doc_id, document
        candidates = []
        parsed_reference = _parse_aci_note(reference)
        occurrences_by_decision: dict[str, list[Mapping]] = defaultdict(list)
        occurrences_by_surface: dict[str, list[Mapping]] = defaultdict(list)
        for occurrence in environment_document.get("occurrences", []):
            decision_id = occurrence.get("decision_id")
            if decision_id is None or not occurrence.get("controlled", True):
                continue
            occurrences_by_decision[str(decision_id)].append(occurrence)
            occurrences_by_surface[canon(str(occurrence.get("surface", "")))].append(
                occurrence
            )
        for decision_id, occurrences in occurrences_by_decision.items():
            surface = str(occurrences[0].get("surface", ""))
            if not surface or not _contains(reference, surface):
                continue
            candidates.append({
                "family": "delivered",
                "scope": "linked",
                "subtype": "content",
                "occurrence_ids": [str(row["occurrence_id"]) for row in occurrences],
                "group_id": f"content:{decision_id}",
                "scoring_contract": {"kind": "contains", "value": surface},
                "evidence": {"authority": "human_reference"},
            })

        for name, value in parsed_reference["demographic"].items():
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
                "evidence": {"authority": "human_reference"},
            })

        for row in parsed_reference["assessment_rows"]:
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
            and parsed_reference["assessment_rows"]
            and parsed_reference["plan_rows"]
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
                        "assessment_rows": len(parsed_reference["assessment_rows"]),
                        "plan_rows": len(parsed_reference["plan_rows"]),
                    },
                },
                "evidence": {"authority": "human_reference"},
            })

        for row in parsed_reference["plan_rows"]:
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


def _contains(text: str, value: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(canon(value))}(?!\w)", canon(text)))


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
    return {
        "sections": sections,
        "demographic": demographic,
        "assessment_rows": _parse_aci_rows(
            sections.get("ASSESSMENT", []), ("condition", "category", "status")
        ),
        "plan_rows": _parse_aci_rows(
            sections.get("PLAN", []), ("condition", "treatment", "test")
        ),
    }


def _parse_aci_rows(lines: Sequence[str], fields: Sequence[str]) -> list[dict]:
    rows = []
    for line in lines:
        values = [value.strip() for value in _ACI_ROW_DELIMITER.split(line) if value.strip()]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values)))
    return rows


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
            if action.get("mode") == "level" and not action.get("keep") and action.get("fill")
        ]
        inventory.append({
            "occurrence_id": occurrence.get("occurrence_id"),
            "surface": occurrence.get("surface"),
            "runtime_type": occurrence.get("runtime_type"),
            "legal_support_properties": properties,
        })
    schema = {
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
            canon(str(action["fill"]))
            for action in decision.get("actions", [])
            if action.get("mode") == "level"
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
        if _contains(question, answer_property):
            reject("answer_leakage")
            continue
        if any(_contains(question, str(occurrences[value].get("surface", "")))
               for value in occurrence_ids):
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
            "evidence": {"source_quotes": [quote]},
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
                    mode = (
                        "keep" if action.get("keep") else
                        "placeholder" if action.get("mode") == "placeholder" else
                        "level"
                    )
                    fill = action.get("fill")
                    action_semantics = {
                        **dict(action),
                        "mode": mode,
                        "fill": fill,
                    }
                    action_semantics.pop("action_id", None)
                    action_id = _stable_hash({
                        "decision_id": decision_id,
                        "action": action_semantics,
                    })
                    if action_id in action_ids:
                        raise ValueError(f"duplicate action semantics for decision {decision_id}")
                    action_ids.add(action_id)
                    actions.append({
                        **dict(action),
                        "action_id": action_id,
                        "mode": mode,
                        "legal": True,
                        "entails": [canon(str(fill))] if fill and mode != "placeholder" else [],
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
                occurrences.append({
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
                })
                if decision is not None:
                    decision["occurrence_ids"].append(occurrence_id)
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


def frozen_occurrences_from_arms(arms: Mapping) -> dict[str, list[dict]]:
    """Read controlled occurrence rows from an already-frozen arms artifact."""
    return {
        doc_id: [dict(row) for row in document["tau_walk"][1] if row.get("lattice")]
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
    family_budgets = threshold_manifest["family_budgets"]
    min_context = int(threshold_manifest.get("min_context_assertions", 0))
    reader_threshold = float(threshold_manifest.get("reader_threshold", 1.0))
    stability_repetitions = int(threshold_manifest.get("reader_stability_repetitions", 1))
    option_permutations = int(threshold_manifest.get("reader_option_permutations", 1))
    stability_threshold = float(threshold_manifest.get("reader_stability_threshold", 1.0))
    candidates_by_document: dict[str, list[dict]] = {}
    rejection_counts: dict[str, int] = defaultdict(int)

    for doc_id, environment_document in frozen_environment.get("documents", {}).items():
        source = source_documents[doc_id]
        candidates = [
            dict(row) for row in task_adapter.deterministic_candidates(
                doc_id, source, environment_document
            )
        ]
        context_count = sum(row.get("family") == "context" for row in candidates)
        if context_count < min_context and relation_teacher is not None:
            try:
                proposals = relation_teacher.propose(
                    relation_teacher_prompt(doc_id, source, environment_document)
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
            except Exception:
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
            except Exception:
                rejection_counts["infrastructure_failed"] += 1
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
        candidates_by_document[doc_id] = accepted

    artifact = package_utility_artifact(
        frozen_environment,
        candidates_by_document,
        family_budgets=family_budgets,
        structural_cap=threshold_manifest.get("structural_cap"),
        pins={**dict(pins), "threshold_manifest": dict(threshold_manifest)},
    )
    artifact["rejections"] = {"summary_by_reason": dict(rejection_counts)}
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
                and property_level in (action.get("entails") or [])
            ]
            if not candidates:
                raise ValueError(
                    f"no legal generalization for decision {decision_id} entails {property_level}"
                )
            action_vector[decision_id] = str(candidates[-1]["action_id"])
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
            "assessment_rows": 1,
            "plan_rows": 1,
        }
        if not isinstance(parseability, Mapping):
            return 0.0
        expected_counts = {
            name: parseability.get(name)
            for name in ("assessment_rows", "plan_rows")
        }
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in expected_counts.values()
        ):
            return 0.0
        return float(
            len(parsed_output["assessment_rows"])
            == expected_counts["assessment_rows"]
            and len(parsed_output["plan_rows"])
            == expected_counts["plan_rows"]
        )
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
