"""QA-builder v2 artifact weighting, support anchors, validation, and scoring."""
from __future__ import annotations

import bisect
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
RELATION_TEACHER_PROMPT_VERSION = "qa-relation-teacher-v12"
RELATION_TEACHER_RESPONSE_SCHEMA = {"type": "relation-qa-batch", "version": 7}
RELATION_TEACHER_REVISION = "qa-relation-teacher-r24"
RELATION_TEACHER_MAX_RELATIONS = 12
# Nemotron's OpenRouter route has mandatory reasoning.  Token caps repeatedly
# broke the teacher: completion caps returned empty replies, and the r16
# smoke's 1,024-token reasoning cap truncated the source scan mid-document,
# leaving three further explicit relations unproposed.  The v5 contract sets
# no completion or reasoning cap; only the trace exclusion remains.
RELATION_TEACHER_GENERATION_CONFIG = {
    "reasoning": {"exclude": True},
}
_RELATION_RECORD_ITEM = {
    "type": "object",
    "additionalProperties": False,
    "required": ["relation", "arguments", "question", "accepted_answers",
                 "scoring_contract"],
    "properties": {
        "relation": {"enum": [
            "prescribed_with", "treated_with", "monitored_by",
            "contraindicated_because_of", "causes_or_explains",
            "referred_to", "has_status", "has_category",
        ]},
        "arguments": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["role", "kind", "span_label",
                             "support_property", "literal"],
                "properties": {
                    "role": {"enum": ["subject", "object"]},
                    "kind": {"enum": ["linked", "context"]},
                    "span_label": {"type": ["string", "null"]},
                    "support_property": {"type": ["string", "null"]},
                    "literal": {"type": ["string", "null"]},
                },
            },
        },
        "question": {"type": "string"},
        "accepted_answers": {
            "type": "array", "minItems": 1,
            "items": {"type": "string"},
        },
        "scoring_contract": {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "match"],
            "properties": {
                "kind": {"const": "semantic_qa"},
                "match": {"const": "fact_score"},
            },
        },
    },
}
# Sectioned response (A/B spike 2026-07-14): a single mixed relation list let
# the span-pair sub-task crowd out span<->literal proposals (0-3 literal per
# draw); requiring the two lists separately yielded 4-6 literal proposals in
# 3/3 draws. span_relations pairs two S-labels; context_relations pairs
# exactly one S-label with one uncontrolled literal.
RELATION_TEACHER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "relation_qa_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["span_relations", "context_relations", "candidate_accounting"],
            "properties": {
                "span_relations": {
                    "type": "array",
                    "maxItems": RELATION_TEACHER_MAX_RELATIONS,
                    "items": deepcopy(_RELATION_RECORD_ITEM),
                },
                "context_relations": {
                    "type": "array",
                    "maxItems": RELATION_TEACHER_MAX_RELATIONS,
                    "items": deepcopy(_RELATION_RECORD_ITEM),
                },
                "candidate_accounting": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["candidate_label", "state", "reason"],
                        "properties": {
                            "candidate_label": {"type": "string"},
                            # duplicate_mention: a repeated mention of a value whose
                            # fact was already emitted at another label; without it
                            # the v5 smoke fabricated one relation per duplicate.
                            "state": {
                                "enum": ["emitted", "duplicate_mention",
                                         "exhausted_no_relation", "unsupported"]
                            },
                            # minLength: the same smoke returned "" reasons for
                            # emitted rows, invalidating the entire ledger.
                            "reason": {"type": "string", "minLength": 1, "maxLength": 160},
                        },
                    },
                },
            },
        },
    },
}
CONTEXT_READER_PROMPT_VERSION = "qa-context-reader-v3"
CONTEXT_READER_RESPONSE_SCHEMA = {"type": "single-span", "version": 2}
CONTEXT_READER_REVISION = "qa-reader-r4"
# The reader is given only the transcript turns covering an assertion's
# arguments/evidence, plus this many neighbor turns each side, instead of the
# whole document. Removes distractor spans (e.g. an unrelated drug elsewhere in
# the note) that a small reader would otherwise return.
CONTEXT_READER_TURN_WINDOW = 0
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
    "prescribed_with",
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
    "medical-procedure": "procedure",
    "lab": "monitoring",
    "test": "monitoring",
    "monitoring": "monitoring",
    "provider": "provider",
    "specialty": "provider",
    "status": "status",
    "category": "category",
}
_RELATION_ARGUMENT_CLASSES = {
    # Relation names are directional: condition/diagnosis first.  Medication is
    # intentionally separate from a procedure performed to treat a condition.
    "prescribed_with": (("condition",), ("treatment",)),
    "treated_with": (("condition",), ("procedure",)),
    "monitored_by": (("condition",), ("monitoring", "procedure", "provider")),
    "contraindicated_because_of": (("treatment", "procedure"), ("condition",)),
    "causes_or_explains": (("condition",), ("condition", "symptom")),
    "referred_to": (("condition",), ("provider", "procedure")),
    "has_status": (("condition", "symptom", "treatment", "procedure"), ("status",)),
    "has_category": (("condition", "symptom", "treatment", "procedure"), ("category",)),
}
# Human-facing answer-type words for the reader hint. Keyed by argument class.
_CLASS_ANSWER_HINT = {
    "condition": "medical condition",
    "symptom": "symptom",
    "treatment": "medication",
    "procedure": "procedure",
    "monitoring": "test or procedure",
    "provider": "provider or specialist",
    "status": "status",
    "category": "category",
}


def _relation_answer_type_hint(relation: str, answer_role: str) -> str | None:
    """The reader answer-type word for a relation's answered argument, derived
    from the directional argument-class contract (deterministic, no inference).
    A union of classes joins with ' or '; unknown relations -> no hint."""
    classes = _RELATION_ARGUMENT_CLASSES.get(relation)
    if not classes:
        return None
    role_classes = classes[0] if answer_role == "subject" else classes[1]
    hints = [
        _CLASS_ANSWER_HINT[cls] for cls in role_classes if cls in _CLASS_ANSWER_HINT
    ]
    if not hints:
        return None
    # de-duplicate while preserving order
    seen = list(dict.fromkeys(hints))
    return " or ".join(seen)
ACI_RELATION_CONTRACT = {
    "prescribed_with": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["prescribed_with"],
        "definition": "a drug prescribed or used for a condition or diagnosis",
        "cues": ("prescribe", "prescribed", "initiate", "continue", "uses", "taking", "takes", "treated with"),
        "connector_patterns": (
            r"\s+(?:(?:is|was)\s+)?(?:prescribed|used)\s+(?:with|for)\s+",
            r"\s+(?:(?:is|was)\s+)?treated\s+with\s+",
        ),
    },
    "treated_with": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["treated_with"],
        "definition": "a medical procedure used for a condition or diagnosis",
        "cues": ("treated with",),
        "connector_patterns": (
            r"\s+(?:(?:is|was|are|were|has been|had been|is being|was being)\s+)?"
            r"treated\s+with\s+",
        ),
        # Indication form, procedure textually first: "had the kidney
        # transplant a few years ago for some polycystic kidneys" (verbatim in
        # the D2N002 reference). Class gating already restricts this to a
        # procedure-class object and condition-class subject.
        "reversed_connector_patterns": (
            r"\s+(?:[\w',]+\s+){0,5}?for\s+(?:some\s+|your\s+|the\s+|a\s+|an\s+|his\s+|her\s+)?",
        ),
    },
    "monitored_by": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["monitored_by"],
        "cues": ("monitored by", "monitor", "check", "order"),
        "connector_patterns": (
            r"\s+(?:(?:is|was|are|were|has been|had been|is being|was being)\s+)?"
            r"monitored\s+by\s+",
        ),
    },
    "contraindicated_because_of": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["contraindicated_because_of"],
        # ACI dialogue states contraindication as "you ca n't take X because
        # of Y" (transcript-tokenized negation), not with the clinical word.
        "cues": ("contraindicated", "ca n't take", "can't take", "cannot take"),
        "connector_patterns": (
            r"\s+(?:(?:is|was|are|were|has been|had been|is being|was being)\s+)?"
            r"contraindicated\s+because\s+of\s+",
        ),
    },
    "causes_or_explains": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["causes_or_explains"],
        # Clinical attribution is stated as "exacerbation of"/"due to"/
        # "secondary to" far more often than with the verb "causes".
        "cues": ("causes", "explains", "due to", "secondary to", "caused by",
                 "explained by", "exacerbation of"),
        "connector_patterns": (r"\s+(?:causes|explains)\s+",),
    },
    "referred_to": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["referred_to"],
        "cues": ("referred to", "refer"),
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
# Closed procedure-form lexicon: detector-typed conditions whose surface names
# a performed procedure ("kidney transplant" in past medical history) may also
# fill procedure slots; ordinary condition surfaces may not.
_PROCEDURE_FORM_PATTERN = re.compile(
    r"\b(?:transplant\w*|surgery|surgeries|\w+ectomy|\w+plasty|bypass|graft\w*"
    r"|replacement|repair)\b",
    re.IGNORECASE,
)


def _argument_relation_classes(runtime_type: str, surface: str) -> set[str]:
    base = _RUNTIME_TYPE_CLASSES.get(canon(str(runtime_type)))
    classes = {base} if base else set()
    if base == "condition" and _PROCEDURE_FORM_PATTERN.search(str(surface)):
        classes.add("procedure")
    return classes


_NEGATION_PATTERN = re.compile(r"\b(?:no|not|without|denies|denied)\b")
# A conditional/hypothetical statement is not an asserted fact. The broadened
# problem-block/plan-section anchors would otherwise ground the conditional PT
# referral ("if symptoms continue ... possibly refer to physical therapy"),
# which the authoritative reference itself states only conditionally.
# The `if` arm requires a real conditional subject so the spoken disfluency
# "if ... and prescribe some ultram" is not mistaken for a condition, and
# "as needed"/"prn" are excluded (they modify dosing, not existence).
_HEDGE_PATTERN = re.compile(
    r"\b(?:possibly|possible|maybe|perhaps|might|consider|talk\s+about|discuss"
    r"|if\s+(?:your|you|we|he|she|it|they|the|his|her|symptoms|there|needed))\b",
    re.IGNORECASE,
)


def _relation_window_is_hedged(
    document: str, arguments: Sequence[Mapping], occurrences: Mapping[str, Mapping],
) -> bool:
    """True if the source region spanning the arguments is conditional/hedged."""
    bounds = []
    for argument in arguments:
        if argument["kind"] == "linked":
            occurrence = occurrences[argument["occurrence_id"]]
            bounds.append((int(occurrence["start"]), int(occurrence["end"])))
        elif isinstance(argument.get("start"), int) and isinstance(argument.get("end"), int):
            bounds.append((int(argument["start"]), int(argument["end"])))
    if not bounds:
        return False
    lo = min(start for start, _ in bounds)
    hi = max(end for _, end in bounds)
    return _HEDGE_PATTERN.search(document[lo:hi]) is not None
_CLAUSE_DELIMITER_PATTERN = re.compile(r"[\n.!?;]")
_PLAN_SECTION_HEADING_PATTERN = re.compile(
    r"(?m)^(?P<title>[A-Za-z][A-Za-z0-9 /-]{0,96})\.\s*$"
)
_RELATION_WINDOW_CUE_PATTERN = re.compile(
    r"\b(?:prescrib\w*|continue|taking|takes|treated|refer\w*|order\w*|"
    r"monitor\w*|contraindicat\w*|causes?|explains?|status|category)\b",
    re.IGNORECASE,
)
_ACI_SPEAKER_TURN_PATTERN = re.compile(r"\[(?:doctor|patient)\]", re.IGNORECASE)
ACI_REQUIRED_SECTIONS = (
    "HISTORY OF PRESENT ILLNESS",
    "ASSESSMENT",
    "PLAN",
)
ACI_COMBINED_ASSESSMENT_PLAN = "ASSESSMENT AND PLAN"
ACI_TASK_HEADINGS = frozenset({
    "CHIEF COMPLAINT",
    "HPI",
    "HISTORY OF PRESENT ILLNESS",
    "REVIEW OF SYSTEMS",
    "PHYSICAL EXAMINATION",
    "RESULTS",
    "ASSESSMENT",
    "PLAN",
    ACI_COMBINED_ASSESSMENT_PLAN,
})
_ACI_CAPTURED_HEADING_SECTIONS = {
    "HPI": ("HISTORY OF PRESENT ILLNESS",),
    "HISTORY OF PRESENT ILLNESS": ("HISTORY OF PRESENT ILLNESS",),
    "ASSESSMENT": ("ASSESSMENT",),
    "PLAN": ("PLAN",),
    ACI_COMBINED_ASSESSMENT_PLAN: ("ASSESSMENT", "PLAN"),
}
_ACI_HEADING_PATTERN = re.compile(
    r"^\s*(?P<heading>[A-Z][A-Z /&-]*[A-Z])\s*(?::\s*(?P<content>.*))?$"
)
_ACI_ROW_DELIMITER = re.compile(r"\s+(?:—|–|-)\s+")
_ACI_CONDITION_ENTRY_PATTERN = re.compile(r"^(?P<condition>[^:]+?)\.\s*$")
_ACI_LABELED_FIELD_PATTERN = re.compile(
    r"^(?:[•*\-]\s*)?(?P<label>Medical Reasoning|Additional Testing|Medical Treatment)"
    r"\s*:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
_ACI_LABELED_PLAN_FIELDS = {
    "additional testing": "test",
    "medical treatment": "treatment",
}
_ACI_CONDITION_PREAMBLE_WORDS = frozenset({
    "a", "an", "the", "patient", "this", "he", "she", "they", "we", "i",
})
_CONTEXT_CANDIDATE_PATTERNS = (
    ("test", re.compile(
        r"\b(?:order|ordered|check|checking|monitor|monitoring|monitored\s+by|repeat|recheck)\s+"
        r"(?:some\s+)?(?P<literal>(?:[A-Za-z][A-Za-z-]*\s+){0,4}"
        r"(?:labs?|panel|test))\b", re.IGNORECASE)),
    ("procedure", re.compile(
        r"\b(?:refer|referred)\s+(?:\w+\s+){0,2}to\s+"
        r"(?P<literal>(?:[A-Za-z][A-Za-z-]*\s+){0,3}"
        r"(?:therapy|surgery|procedure))\b", re.IGNORECASE)),
    ("status", re.compile(
        r"\bstatus\s+(?:is|:)?\s*(?P<literal>[A-Za-z][A-Za-z -]{0,48})"
        r"(?=[.;\n]|$)", re.IGNORECASE)),
    ("category", re.compile(
        r"\bcategory\s+(?:is|:)?\s*(?P<literal>[A-Za-z][A-Za-z -]{0,48})"
        r"(?=[.;\n]|$)", re.IGNORECASE)),
)


class RelationTeacherProposals(list):
    """List-compatible teacher proposals plus mandatory v2 coverage accounting."""

    def __init__(self, relations: Sequence[Mapping], candidate_accounting: Sequence[Mapping]):
        super().__init__(dict(row) for row in relations if isinstance(row, Mapping))
        self.candidate_accounting = [
            dict(row) for row in candidate_accounting if isinstance(row, Mapping)
        ]


class RelationTeacherResponseError(ValueError):
    """A non-sensitive, artifact-safe failure while decoding a teacher reply."""

    def __init__(
        self,
        code: str,
        *,
        raw_length: int,
        completion_state: str | None = None,
        parser_message: str | None = None,
    ):
        self.code = code
        self.raw_length = raw_length
        self.completion_state = completion_state
        self.parser_message = parser_message
        message = f"{code} (raw_length={raw_length})"
        if parser_message:
            message = f"{message}: {parser_message}"
        super().__init__(message)


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
            "response_format": deepcopy(RELATION_TEACHER_RESPONSE_FORMAT),
            "generation_config": deepcopy(RELATION_TEACHER_GENERATION_CONFIG),
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
            response_format=deepcopy(RELATION_TEACHER_RESPONSE_FORMAT),
            extra_body={
                "reasoning": deepcopy(RELATION_TEACHER_GENERATION_CONFIG["reasoning"])
            },
            single_flight=True,
        )

    def propose(
        self, prompt: str, *, response_format: Mapping | None = None,
    ) -> RelationTeacherProposals:
        raw = self._client.generate(
            prompt,
            **({"response_format": dict(response_format)} if response_format else {}),
        )
        if not raw or not raw.strip():
            # `LLMClient` deliberately does not cache an OpenRouter HTTP-200 reply
            # with no choices/content.  Do not turn that provider condition into a
            # misleading JSONDecodeError, and never persist potentially sensitive raw
            # model text in the artifact.
            state = getattr(self._client, "last_completion_state", {})
            outcome = state.get("outcome") if isinstance(state, Mapping) else None
            code = {
                "no_choices": "teacher_no_choices",
                "empty_content": "teacher_empty_content",
            }.get(outcome, "teacher_empty_response")
            raise RelationTeacherResponseError(
                code,
                raw_length=len(raw or ""),
                completion_state=outcome,
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RelationTeacherResponseError(
                "teacher_invalid_json",
                raw_length=len(raw),
                parser_message=str(error),
            ) from error
        relations = None
        if isinstance(payload, dict):
            if isinstance(payload.get("span_relations"), list) and isinstance(
                payload.get("context_relations"), list
            ):
                # Sectioned v8 reply; span pairs first so the shared cap
                # cannot be consumed by literals alone.
                relations = list(payload["span_relations"]) + list(payload["context_relations"])
            elif isinstance(payload.get("relations"), list):
                relations = payload["relations"]
        if not isinstance(relations, list):
            raise RelationTeacherResponseError(
                "teacher_invalid_schema", raw_length=len(raw),
                parser_message="relation teacher reply must contain span_relations"
                               " and context_relations lists",
            )
        accounting = payload.get("candidate_accounting") if isinstance(payload, dict) else None
        if not isinstance(accounting, list):
            raise RelationTeacherResponseError(
                "teacher_invalid_schema", raw_length=len(raw),
                parser_message="relation teacher reply must contain candidate_accounting",
            )
        return RelationTeacherProposals(relations, accounting)


class BatchedContextReader:
    """Pinned reader for context assertions — one model request per question."""

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
                max_tokens=128,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        self._client = client

    def _read_one(self, question: str, context: str) -> str:
        prompt = (
            "Read the DOCUMENT and complete the REQUEST. Copy the answer span "
            "exactly from the DOCUMENT — do not rephrase, summarize, or add words. "
            "If the DOCUMENT does not answer it, reply with exactly NONE.\n\n"
            f"DOCUMENT:\n{context}\n\n"
            f"REQUEST: {question}"
        )
        raw = self._client.generate(prompt).strip()
        if raw.startswith("```") and raw.endswith("```"):
            raw = raw.strip("`").strip()
        raw = raw.strip().strip('"').strip("'").strip()
        return "" if raw.upper() == "NONE" else raw

    def __call__(self, questions: list[str], context: str) -> list[str]:
        return [self._read_one(question, context) for question in questions]


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
                    r"(?:(?:a|an|the|some)\s+)?{surface}",
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
                (
                    "past medical history",
                    r"\bpast\s+medical\s+history\b[^.!?\n]{{0,160}}?{surface}",
                ),
                ("history", r"\bhistory\s+of\s+{surface}"),
                ("diagnosis", r"\bdiagnosis\s+of\s+{surface}"),
                ("diagnosed", r"\bdiagnosed\s+with\s+{surface}"),
                ("presentation", r"\b(?:presents|presented)\s+with\s+{surface}"),
                ("complaint", r"\bcomplaints?\s+of\s+{surface}"),
                ("problem", r"\bproblem\b[^.!?\n]{{0,80}}?{surface}"),
                ("in terms of", r"\bin\s+terms\s+of\s+(?:your\s+)?{surface}"),
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
                if not row.get(field):
                    continue
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
            and parsed_reference["assessment_shape"]["kind"] in {"rows", "labeled_rows", "none"}
            and parsed_reference["plan_shape"]["kind"] in {"rows", "labeled_rows", "none"}
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
            if any(not isinstance(value, str) or not value.strip() for value in values):
                continue
            if "evidence" in row:
                occurrence_ids = _exact_labeled_field_occurrence_ids(
                    document,
                    values,
                    occurrences_by_surface,
                    occurrences_by_decision,
                )
                if occurrence_ids is None:
                    continue
                if not occurrence_ids:
                    continue
                scope = "linked"
            else:
                relation_occurrences = [
                    occurrences_by_surface.get(canon(value), []) for value in values
                ]
                occurrence_ids = [
                    str(rows[0]["occurrence_id"])
                    for rows in relation_occurrences if len(rows) == 1
                ]
                scope = "linked" if len(occurrence_ids) == len(values) else "global"
            candidates.append({
                "family": "delivered",
                "scope": scope,
                "subtype": "exact_relation",
                "occurrence_ids": occurrence_ids if scope == "linked" else [],
                "group_id": "relation:" + ":".join(canon(value) for value in values),
                "scoring_contract": {
                    "kind": "exact_relation",
                    "section": "PLAN",
                    "condition": row["condition"],
                    "treatment": row["treatment"],
                    "test": row["test"],
                },
                "evidence": {
                    "authority": "human_reference",
                    **({"reference_spans": dict(row["evidence"])} if "evidence" in row else {}),
                },
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


def relation_context_candidates(document: str) -> list[dict]:
    """Extract finite typed source context without creating detector/ranker decisions."""
    candidates, seen = [], set()
    for runtime_type, pattern in _CONTEXT_CANDIDATE_PATTERNS:
        for match in pattern.finditer(document):
            start, end = match.span("literal")
            literal = document[start:end]
            identity = (runtime_type, start, end, literal)
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append({
                "context_candidate_id": "context:" + _stable_hash({
                    "runtime_type": runtime_type, "start": start, "end": end,
                    "literal": literal,
                }),
                "kind": "context_literal",
                "runtime_type": runtime_type,
                "literal": literal,
                "start": start,
                "end": end,
                "provenance": "aci_explicit_context_cue",
            })
    return sorted(candidates, key=lambda row: (
        int(row["start"]), str(row["context_candidate_id"])
    ))


def relation_teacher_span_inventory(environment_document: Mapping) -> list[dict]:
    """Return prompt-safe, source-ordered controlled spans and private mappings.

    This is intentionally a representability prefilter, not a relation-pair
    prefilter.  The teacher receives only the resulting short labels.
    """
    decisions = {
        str(row["decision_id"]): row
        for row in environment_document.get("decisions", [])
        if row.get("decision_id") is not None
    }
    rows = []
    for occurrence in environment_document.get("occurrences", []):
        if not occurrence.get("controlled", True):
            continue
        start, end = occurrence.get("start"), occurrence.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            continue
        decision = decisions.get(str(occurrence.get("decision_id")), {})
        properties = list(dict.fromkeys(
            canon(str(value))
            for action in decision.get("actions", [])
            if action.get("legal", True) and action.get("mode") not in {"keep", "placeholder"}
            for value in action.get("entails") or []
            if canon(str(value))
        ))
        relation_class = _RUNTIME_TYPE_CLASSES.get(canon(str(occurrence.get("runtime_type", ""))))
        if not properties or relation_class is None:
            continue
        if not any(relation_class in classes for contract in _RELATION_ARGUMENT_CLASSES.values()
                   for classes in contract):
            continue
        rows.append({
            "occurrence_id": str(occurrence["occurrence_id"]),
            "surface": str(occurrence.get("surface", "")),
            "start": start,
            "end": end,
            "runtime_type": str(occurrence.get("runtime_type", "")),
            "relation_class": relation_class,
            "properties": properties,
        })
    rows.sort(key=lambda row: (row["start"], row["end"], row["occurrence_id"]))
    for index, row in enumerate(rows, start=1):
        row["span_label"] = f"S{index}"
    return rows


def _source_clause_spans(document: str) -> list[tuple[int, int]]:
    """Return source clauses without treating speaker turns as relation bridges."""
    markers = list(_ACI_SPEAKER_TURN_PATTERN.finditer(document))
    if len(markers) >= 2:
        return [
            (marker.start(), markers[index + 1].start() if index + 1 < len(markers) else len(document))
            for index, marker in enumerate(markers)
            if document[marker.start():markers[index + 1].start() if index + 1 < len(markers) else len(document)].strip()
        ]
    spans, left = [], 0
    for delimiter in _CLAUSE_DELIMITER_PATTERN.finditer(document):
        right = delimiter.end()
        if document[left:right].strip():
            spans.append((left, right))
        left = right
    if document[left:].strip():
        spans.append((left, len(document)))
    return spans


def _clinical_plan_sections(document: str) -> list[tuple[int, int]]:
    """Return bounded assessment-plan sections headed by a clinical concept.

    A heading alone does not make a relation.  This only recognizes the common
    note structure ``Condition.`` followed by plan bullets; the relation cue
    and argument checks still run over the returned source span.
    """
    headings = list(_PLAN_SECTION_HEADING_PATTERN.finditer(document))
    sections = []
    for index, heading in enumerate(headings):
        start = heading.start()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(document)
        body = document[heading.end():end]
        if re.search(r"(?m)^\s*[-•]\s+(?:Medical Reasoning|Additional Testing|Medical Treatment|Patient Education)", body):
            sections.append((start, end))
    return sections


def _shared_plan_section(
    document: str, spans: Sequence[tuple[int, int]],
) -> tuple[int, int] | None:
    """Find the one explicit plan section containing every source argument."""
    candidates = [
        section for section in _clinical_plan_sections(document)
        if all(section[0] <= start < end <= section[1] for start, end in spans)
    ]
    return candidates[0] if len(candidates) == 1 else None


# One spoken assessment/plan discussion of a single problem. The doctor opens
# the assessment, then discusses each problem in turn; a relation may connect
# spans across the patient acknowledgments inside one such block, but never
# across a problem switch. Spike-confirmed on D2N002
# (monitored_by(arthritis -> autoimmune panel)).
_PROBLEM_BLOCK_BOUNDARY = re.compile(
    r"assessment and (?:my |the )?plan"
    r"|for (?:your|the|his|her)\s+(?:\w+\s+)?problem"
    r"|for (?:your|the|his|her)\s+(?:second|third|fourth|next|last|final)\b",
    re.IGNORECASE,
)


def _problem_blocks(document: str) -> list[tuple[int, int]]:
    cuts = sorted({m.start() for m in _PROBLEM_BLOCK_BOUNDARY.finditer(document)})
    if not cuts:
        return []
    edges = [*cuts, len(document)]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _shared_problem_block(
    document: str, spans: Sequence[tuple[int, int]],
) -> tuple[int, int] | None:
    """The one problem-discussion block containing every source argument."""
    candidates = [
        block for block in _problem_blocks(document)
        if all(block[0] <= start < end <= block[1] for start, end in spans)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _prompt_relation_class(relation_class: str) -> str:
    return {"treatment": "drug", "monitoring": "test"}.get(relation_class, relation_class)


def _prompt_display_classes(row: Mapping) -> str:
    """Teacher-facing class label; procedure-form conditions show both roles."""
    classes = _argument_relation_classes(str(row["runtime_type"]), str(row["surface"]))
    ordered = [cls for cls in ("condition", "symptom", "treatment", "procedure",
                               "monitoring", "provider", "status", "category")
               if cls in classes] or [str(row["relation_class"])]
    return "/".join(_prompt_relation_class(cls) for cls in ordered)


def relation_evidence_windows(document: str, environment_document: Mapping) -> list[dict]:
    """Finite one/two-clause source spans that can ground a typed relation.

    These windows do not assert a relation.  They only prevent a teacher from
    pairing a source quote with a duplicate occurrence elsewhere in the document.
    """
    speaker_markers = list(_ACI_SPEAKER_TURN_PATTERN.finditer(document))
    if len(speaker_markers) >= 2:
        clause_spans = [
            (marker.start(), speaker_markers[index + 1].start()
             if index + 1 < len(speaker_markers) else len(document))
            for index, marker in enumerate(speaker_markers)
        ]
        window_widths = (1,)
    else:
        clause_spans, start = [], 0
        for delimiter in _CLAUSE_DELIMITER_PATTERN.finditer(document):
            end = delimiter.end()
            if document[start:end].strip():
                clause_spans.append((start, end))
            start = end
        if document[start:].strip():
            clause_spans.append((start, len(document)))
        window_widths = (1, 2)
    decisions = {
        str(row["decision_id"]): row
        for row in environment_document.get("decisions", [])
        if row.get("decision_id") is not None
    }
    arguments = [
        {
            "candidate_id": str(row["occurrence_id"]),
            "runtime_type": str(row.get("runtime_type", "")),
            "start": row.get("start"), "end": row.get("end"),
            "kind": "linked",
            "linkable": bool({
                canon(str(property_level))
                for action in decisions.get(str(row.get("decision_id")), {}).get("actions", [])
                if action.get("legal", True)
                and action.get("mode") not in {"keep", "placeholder"}
                for property_level in (action.get("entails") or [])
                if canon(str(property_level))
            }),
        }
        for row in environment_document.get("occurrences", [])
        if row.get("occurrence_id") is not None
    ] + [
        {
            "candidate_id": str(row["context_candidate_id"]),
            "runtime_type": str(row.get("runtime_type", "")),
            "start": row.get("start"), "end": row.get("end"),
            "kind": "context",
            "linkable": True,
        }
        for row in relation_context_candidates(document)
    ]
    windows = []
    for index, (left, _) in enumerate(clause_spans):
        for width in window_widths:
            if index + width > len(clause_spans):
                continue
            right = clause_spans[index + width - 1][1]
            quote = document[left:right]
            if not _RELATION_WINDOW_CUE_PATTERN.search(quote):
                continue
            contained = [
                row for row in arguments
                if isinstance(row["start"], int) and isinstance(row["end"], int)
                and left <= row["start"] < row["end"] <= right
            ]
            if not any(
                _RUNTIME_TYPE_CLASSES.get(canon(first["runtime_type"])) in subject_types
                and _RUNTIME_TYPE_CLASSES.get(canon(second["runtime_type"])) in object_types
                for first in contained for second in contained if first is not second
                for subject_types, object_types in _RELATION_ARGUMENT_CLASSES.values()
            ):
                continue
            eligible_pairs = []
            for relation, (subject_types, object_types) in _RELATION_ARGUMENT_CLASSES.items():
                for subject in contained:
                    for obj in contained:
                        if subject is obj or not subject["linkable"] or not obj["linkable"]:
                            continue
                        if (
                            _RUNTIME_TYPE_CLASSES.get(canon(subject["runtime_type"])) in subject_types
                            and _RUNTIME_TYPE_CLASSES.get(canon(obj["runtime_type"])) in object_types
                            and _window_pair_has_relation_shape(
                                relation, subject, obj, clause_spans, document
                            )
                        ):
                            eligible_pairs.append({
                                "relation": relation,
                                "subject_candidate_id": subject["candidate_id"],
                                "object_candidate_id": obj["candidate_id"],
                            })
            windows.append({
                "evidence_window_id": "window:" + _stable_hash({"start": left, "end": right}),
                "start": left,
                "end": right,
                "quote": quote,
                "candidate_ids": [row["candidate_id"] for row in contained],
                "eligible_pairs": eligible_pairs,
            })
    return windows


def _window_pair_has_relation_shape(
    relation: str,
    subject: Mapping,
    obj: Mapping,
    clause_spans: Sequence[tuple[int, int]],
    document: str,
) -> bool:
    """Keep only local source pairings with a relation-specific cue.

    The pair is still a candidate, not a claimed fact: the teacher's semantics
    and compiler evidence checks remain mandatory.
    """
    def clause_index(row: Mapping) -> int | None:
        return next((index for index, (left, right) in enumerate(clause_spans)
                     if left <= row["start"] < row["end"] <= right), None)

    subject_clause, object_clause = clause_index(subject), clause_index(obj)
    if subject_clause is None or object_clause is None:
        return False
    cue_patterns = {
        "prescribed_with": r"\b(?:prescrib\w*|continue\s+.{0,24}\bon\b|treated\s+with)\b",
        "treated_with": r"\btreated\s+with\b",
        "monitored_by": r"\b(?:monitor\w*|check\w*|order\w*)\b",
        "contraindicated_because_of": r"\bcontraindicat\w*\b",
        "causes_or_explains": r"\b(?:causes?|explains?)\b",
        "referred_to": r"\brefer\w*\b",
        "has_status": r"\bstatus\b",
        "has_category": r"\bcategory\b",
    }[relation]
    if subject_clause == object_clause:
        if abs(int(obj["start"]) - int(subject["end"])) > 320:
            return False
        local_left = min(int(subject["start"]), int(obj["start"]))
        local_right = max(int(subject["end"]), int(obj["end"]))
        return re.search(cue_patterns, document[local_left:local_right], re.IGNORECASE) is not None
    if subject_clause + 1 != object_clause:
        return False
    object_text = document[clause_spans[object_clause][0]:clause_spans[object_clause][1]]
    return re.search(cue_patterns, object_text, re.IGNORECASE) is not None


def relation_teacher_response_format(
    environment_document: Mapping, document: str,
) -> dict:
    """Bind strict wire fields to displayed labels, never internal IDs or answers."""
    response_format = deepcopy(RELATION_TEACHER_RESPONSE_FORMAT)
    schema = response_format["json_schema"]["schema"]
    inventory = relation_teacher_span_inventory(environment_document)
    labels = [row["span_label"] for row in inventory]
    support_properties = sorted({property_level for row in inventory for property_level in row["properties"]})
    def argument_branch(role: str, kind: str) -> dict:
        fields = (
            {
                "span_label": {"enum": labels},
                "support_property": {"enum": support_properties},
                "literal": {"const": None},
            }
            if kind == "linked" else
            {
                "span_label": {"const": None},
                "support_property": {"const": None},
                "literal": {"type": "string", "minLength": 1},
            }
        )
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["role", "kind", "span_label", "support_property", "literal"],
            "properties": {"role": {"const": role}, "kind": {"const": kind}, **fields},
        }

    def pair_shapes(kind_pairs: tuple[tuple[str, str], ...]) -> dict:
        return {
            "anyOf": [
                {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "prefixItems": [
                        argument_branch("subject", subject_kind),
                        argument_branch("object", object_kind),
                    ],
                    "items": False,
                }
                for subject_kind, object_kind in kind_pairs
            ]
        }

    # Zero-linked pairs stay unrepresentable (the compiler always rejects
    # them), and each section binds its own argument structure: span pairs
    # are label-only, context pairs carry exactly one uncontrolled literal.
    schema["properties"]["span_relations"]["items"]["properties"]["arguments"] = (
        pair_shapes((("linked", "linked"),))
    )
    schema["properties"]["context_relations"]["items"]["properties"]["arguments"] = (
        pair_shapes((("linked", "context"), ("context", "linked")))
    )
    ledger = schema["properties"]["candidate_accounting"]
    ledger["minItems"] = len(labels)
    ledger["maxItems"] = len(labels)
    ledger_item_properties = ledger["items"]["properties"]
    ledger["prefixItems"] = [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["candidate_label", "state", "reason"],
            "properties": {
                "candidate_label": {"const": label},
                "state": deepcopy(ledger_item_properties["state"]),
                "reason": deepcopy(ledger_item_properties["reason"]),
            },
        }
        for label in labels
    ]
    ledger["items"] = False
    return response_format


def _sanitized_ledger_reason(reason: str, protected_terms: Sequence[str]) -> str:
    """Redact protected surfaces/aliases the teacher repeated despite the prompt."""
    sanitized = reason
    for term in sorted({term for term in protected_terms if term}, key=len, reverse=True):
        sanitized = re.sub(
            rf"(?<!\w){re.escape(term)}(?!\w)", "[protected]", sanitized,
            flags=re.IGNORECASE,
        )
    return sanitized.strip() or "[protected]"


def _validated_candidate_accounting(
    accounting: Sequence[Mapping], environment_document: Mapping, document: str,
) -> list[dict]:
    del document
    expected = {str(row["span_label"]) for row in relation_teacher_span_inventory(environment_document)}
    records = [dict(row) for row in accounting if isinstance(row, Mapping)]
    by_id = {str(row.get("candidate_label", row.get("candidate_id", ""))): row for row in records}
    if len(by_id) != len(records) or set(by_id) != expected:
        raise ValueError("candidate_accounting must contain exactly one record per supplied candidate")
    allowed_states = {"emitted", "duplicate_mention", "exhausted_no_relation", "unsupported"}
    if any(
        str(row.get("state", "")) not in allowed_states
        or not str(row.get("reason", "")).strip()
        for row in records
    ):
        raise ValueError("candidate_accounting has invalid state or empty reason")
    protected_terms = [
        term
        for occurrence in environment_document.get("occurrences", [])
        if occurrence.get("controlled", True)
        for term in _occurrence_protected_terms(occurrence)
    ]
    for row in records:
        row["reason"] = _sanitized_ledger_reason(str(row["reason"]), protected_terms)
    return [by_id[label] for label in sorted(expected, key=lambda value: int(value[1:]))]


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
    if detail_reason in {"answer_leakage", "protected_locator", "protected_context_literal"}:
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


def _question_leaks_answer(
    question: str, answer: str, runtime_type: str | Sequence[str],
) -> bool:
    types = [runtime_type] if isinstance(runtime_type, str) else list(runtime_type)
    exempt: set[str] = set()
    for type_value in types:
        exempt |= _placeholder_meaning_tokens(type_value)
    answer_tokens = _meaningful_tokens(answer) - exempt
    return bool(answer_tokens & _meaningful_tokens(question))


def _question_leaks_protected_term(
    question: str,
    protected_terms: Sequence[str],
    allowed_tokens_by_term: Mapping[str, frozenset[str]] | None = None,
) -> bool:
    """Full-term containment always leaks; token overlap leaks unless the
    token belongs to the decision's declared legal generalization levels,
    which the spec directs questions and answers to use verbatim."""
    if any(_contains(question, term) for term in protected_terms if term):
        return True
    question_tokens = _meaningful_tokens(question)
    return any(
        (_meaningful_tokens(term) - (allowed_tokens_by_term or {}).get(term, frozenset()))
        & question_tokens
        for term in protected_terms
    )


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
    """Parse compact ACI rows and real combined assessment/plan bullet blocks."""
    sections: dict[str, list[str]] = {}
    section_spans: dict[str, list[list[int]]] = {}
    current_sections: tuple[str, ...] = ()
    combined_assessment_plan = False
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        raw = raw_line.rstrip("\r\n")
        line = raw.strip()
        heading = _ACI_HEADING_PATTERN.match(line)
        heading_name = heading.group("heading").strip() if heading else ""
        if heading is not None:
            current_sections = ()
            if heading_name not in ACI_TASK_HEADINGS:
                offset += len(raw_line)
                continue
            captured_sections = _ACI_CAPTURED_HEADING_SECTIONS.get(heading_name)
            if captured_sections is None:
                offset += len(raw_line)
                continue
            current_sections = captured_sections
            for section in current_sections:
                sections[section] = []
                section_spans[section] = []
            content = heading.group("content")
            if content and content.strip():
                value = content.strip()
                start = offset + raw.find(value)
                for section in current_sections:
                    sections[section].append(value)
                    section_spans[section].append([start, start + len(value)])
            combined_assessment_plan = heading_name == ACI_COMBINED_ASSESSMENT_PLAN
        elif line and current_sections:
            start = offset + len(raw) - len(raw.lstrip())
            for section in current_sections:
                sections[section].append(line)
                section_spans[section].append([start, start + len(line)])
        offset += len(raw_line)

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
    if combined_assessment_plan:
        labeled_rows = _parse_aci_combined_rows(
            assessment_lines, section_spans.get("ASSESSMENT", [])
        )
        assessment_rows = [{"condition": row["condition"]} for row in labeled_rows]
        plan_rows = labeled_rows
        assessment_shape = _aci_labeled_section_shape(assessment_rows)
        plan_shape = _aci_labeled_section_shape(plan_rows)
    else:
        assessment_rows = _parse_aci_rows(
            assessment_lines, ("condition", "category", "status")
        )
        plan_rows = _parse_aci_rows(
            plan_lines, ("condition", "treatment", "test")
        )
        assessment_shape = _aci_section_shape(assessment_lines, assessment_rows)
        plan_shape = _aci_section_shape(plan_lines, plan_rows)
    return {
        "sections": sections,
        "demographic": demographic,
        "demographic_evidence": demographic_evidence,
        "assessment_rows": assessment_rows,
        "assessment_shape": assessment_shape,
        "plan_rows": plan_rows,
        "plan_shape": plan_shape,
    }


def _parse_aci_rows(lines: Sequence[str], fields: Sequence[str]) -> list[dict]:
    rows = []
    for line in lines:
        values = [value.strip() for value in _ACI_ROW_DELIMITER.split(line) if value.strip()]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values)))
    return rows


def _parse_aci_combined_rows(
    lines: Sequence[str], spans: Sequence[Sequence[int]],
) -> list[dict]:
    rows = []
    condition_entries = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := _aci_condition_entry_match(line)) is not None
    ]
    for entry_index, (line_index, condition_match) in enumerate(condition_entries):
        next_line_index = (
            condition_entries[entry_index + 1][0]
            if entry_index + 1 < len(condition_entries) else len(lines)
        )
        condition = condition_match.group("condition")
        span = spans[line_index]
        condition_start = span[0] + condition_match.start("condition")
        row = {
            "condition": condition,
            "treatment": None,
            "test": None,
            "evidence": {
                "condition": [condition_start, condition_start + len(condition)],
            },
        }
        ambiguous_fields = set()
        recognized_label = False
        for field_index in range(line_index + 1, next_line_index):
            field_match = _ACI_LABELED_FIELD_PATTERN.match(lines[field_index])
            if field_match is None:
                continue
            recognized_label = True
            field = _ACI_LABELED_PLAN_FIELDS.get(canon(field_match.group("label")))
            if field is None or field in ambiguous_fields:
                continue
            if row[field] is not None:
                row[field] = None
                row["evidence"].pop(field, None)
                ambiguous_fields.add(field)
                continue
            value = field_match.group("value")
            field_span = spans[field_index]
            value_start = field_span[0] + field_match.start("value")
            row[field] = value
            row["evidence"][field] = [value_start, value_start + len(value)]
        if recognized_label:
            rows.append(row)
    return rows


def _aci_condition_entry_match(line: str):
    match = _ACI_CONDITION_ENTRY_PATTERN.match(line)
    if match is None:
        return None
    condition = match.group("condition").strip()
    tokens = re.findall(r"[A-Za-z0-9]+", condition)
    if (
        not condition
        or "," in condition
        or len(condition) > 80
        or not 1 <= len(tokens) <= 8
        or canon(tokens[0]) in _ACI_CONDITION_PREAMBLE_WORDS
    ):
        return None
    return match


def _aci_section_shape(lines: Sequence[str], rows: Sequence[Mapping]) -> dict:
    if len(lines) == 1 and canon(lines[0]) == "none":
        return {"kind": "none", "count": 0}
    if lines and len(rows) == len(lines):
        return {"kind": "rows", "count": len(rows)}
    return {"kind": "invalid", "count": len(rows)}


def _aci_labeled_section_shape(rows: Sequence[Mapping]) -> dict:
    if rows:
        return {"kind": "labeled_rows", "count": len(rows)}
    return {"kind": "invalid", "count": 0}


def _exact_labeled_field_occurrence_ids(
    document: str,
    values: Sequence[str],
    occurrences_by_surface: Mapping[str, Sequence[Mapping]],
    occurrences_by_decision: Mapping[str, Sequence[Mapping]],
) -> list[str] | None:
    occurrence_ids = []
    for value in values:
        matching_decisions = set()
        for surface, occurrences in occurrences_by_surface.items():
            if _contains(value, surface):
                decision_ids = {str(row.get("decision_id", "")) for row in occurrences}
                if len(decision_ids) != 1 or "" in decision_ids:
                    return None
                matching_decisions.update(decision_ids)
        if len(matching_decisions) > 1:
            return None
        if not matching_decisions:
            continue
        decision_id = matching_decisions.pop()
        decision_occurrences = occurrences_by_decision.get(decision_id, ())
        if not decision_occurrences or any(
            _exact_occurrence_span(document, occurrence) is None
            for occurrence in decision_occurrences
        ):
            return None
        occurrence_ids.extend(
            str(occurrence["occurrence_id"]) for occurrence in decision_occurrences
        )
    return list(dict.fromkeys(occurrence_ids))


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
    """Build the compact human-facing v4 relation-teacher prompt."""
    del doc_id
    inventory = relation_teacher_span_inventory(environment_document)
    spans = "\n".join(
        f"[{row['span_label']}: {row['surface']} | {_prompt_display_classes(row)} | levels: {'; '.join(row['properties'])}]"
        for row in inventory
    ) or "(No eligible controlled spans.)"
    cards = []
    for index, (start, end) in enumerate(_source_clause_spans(document), start=1):
        labels = [row["span_label"] for row in inventory if start <= row["start"] < row["end"] <= end]
        if labels:
            cards.append(f"E{index}: {document[start:end].strip()}\nLabels: {', '.join(labels)}")
    relation_inventory = """prescribed_with: condition or diagnosis -> drug; explicit prescription/use only, never a procedure.
treated_with: condition or diagnosis -> medical procedure; never a drug.
monitored_by: condition or diagnosis -> monitoring test, procedure, or provider; require explicit monitoring/evaluation/follow-up, not proximity.
contraindicated_because_of: drug or procedure -> condition or diagnosis; require explicit contraindication.
causes_or_explains: condition or diagnosis -> condition or symptom; require explicit causation/explanation.
referred_to: condition or diagnosis -> provider or procedure; require explicit referral.
has_status: clinical concept -> status; require an explicit status statement.
has_category: clinical concept -> category; require an explicit classification statement."""
    return f"""TASK
Find as many explicit, source-grounded, non-duplicate relations as the cap ({RELATION_TEACHER_MAX_RELATIONS}) permits. Prefer diversity only when supported. Abstain rather than inventing a fact.

HOW TO INSPECT THE SOURCE
Read the full source. Evidence cards are navigation aids only, not pair gates. Use S-labels for linked controlled arguments. A repeated value has several S-labels: always use the S-label whose mention is inside the sentence that states the relation; the compiler grounds the relation at that exact mention. For an uncontrolled argument, quote its exact source text as a context literal.
A relation may connect spans from different turns of the SAME problem discussion, the block where the doctor assesses and plans one problem, including short patient acknowledgments between the doctor's sentences (for example a condition named when the problem is introduced and a test ordered for it a sentence later). Never link spans from a different problem discussion or from unrelated small talk, and never assert a conditional or hypothetical statement ("if symptoms continue", "possibly", "we can consider") as a relation.

PRIVACY-SAFE QA
Author the question, accepted answers, and scoring contract. Do not repeat a displayed controlled source span or alias in a question or accepted answer; use its listed generalization level. For a linked argument, accepted answers come from its listed levels, never its source text. When a label lists several levels, use the most specific one that still conveys the relation, not the broadest (a level so generic it fits almost any concept measures nothing). An exact uncontrolled context literal may be an answer only when it is the measured fact.

RELATION INVENTORY
{relation_inventory}

WORKED EXAMPLES (illustrative patterns using unrelated conditions; do not copy these entities — read this document's own source and spans)
Drug, never treated_with: source "for the [S1: migraine | condition | levels: neurological disorder] ... prescribe [S2: sumatriptan | drug | levels: triptan]" => prescribed_with(S1, S2).
Safe question: "Which medication category was prescribed for the neurological disorder?" Accepted answer: "triptan". Refer to linked spans by their levels, never the source words.
Procedure: "cataract treated with phacoemulsification" => treated_with (a procedure, not a drug).
Monitoring: source "to follow the [S3: diabetes | condition | levels: metabolic disorder], order hemoglobin A1c" => monitored_by(S3, context literal "hemoglobin A1c").
Safe question: "What testing follows the metabolic disorder?" Accepted answer: "hemoglobin A1c" (an uncontrolled literal answer is the measured fact). A test mentioned elsewhere is not monitoring.
Contraindication: source "you ca n't use beta-blockers because of your [S4: asthma | condition | levels: reactive airway disease]" => contraindicated_because_of(context literal "beta-blockers", S4). Use the condition S-label at the sentence that states the contraindication, not an earlier history-list mention of the same condition.
Safe question: "What history rules out the use of that drug class?" Accepted answer: "reactive airway disease" (the condition's level, never the drug and never the source words).

DETECTED SPANS
{spans}

EVIDENCE CARDS
{chr(10).join(cards) or '(No span-local cards; inspect the source directly.)'}

RESPONSE
Return two relation lists. Each relation record contains: relation; a subject argument then an object argument; a question; accepted answers; the fixed scoring contract.
span_relations: relations whose subject and object are both displayed spans. Each argument is kind linked, with span_label set to its S-label and support_property set to exactly one of that label's listed levels, copied verbatim.
context_relations: relations pairing exactly one linked S-label argument with one uncontrolled argument of kind context, whose literal is exact source text that is not any displayed span.
Never quote a displayed span as a context literal. Emit each distinct fact once, in the list its argument kinds require, at the S-label inside the sentence that states the relation. Do not repeat the same fact for other S-labels of the same value.
Example span_relations record (illustrative, unrelated entities): relation prescribed_with; subject linked S1 with one listed S1 level as support_property; object linked S2 with one listed S2 level; question "Which medication category was prescribed for the neurological disorder?"; accepted answer "triptan".
Example context_relations record (illustrative, unrelated entities): relation prescribed_with; subject linked S3 with one listed S3 level; object context literal "azithromycin" quoted from the relation sentence; accepted answer "azithromycin".
Return exactly one candidate_accounting row per S-label covering both lists, with a short reason for every row. emitted means a relation record in either list uses the label; duplicate_mention means another S-label of the same value already carries the fact (name that label in the reason); exhausted_no_relation means no explicit supported relation; unsupported means insufficient source role/connection. Reasons must reference labels and levels only and never repeat displayed span text. Return only the structured response.

SOURCE DOCUMENT
{document}"""


def _teacher_relation_arguments(
    proposal: Mapping,
    occurrences: Mapping[str, Mapping],
    context_candidates: Mapping[str, Mapping],
    span_labels: Mapping[str, str] | None = None,
) -> tuple[list[dict] | None, str | None]:
    """Normalize v4 labels/literals while retaining cached v1-v3 compatibility."""
    raw = proposal.get("arguments")
    if raw is None:
        occurrence_ids = [str(value) for value in proposal.get("argument_occurrence_ids", [])]
        support = {str(key): canon(str(value)) for key, value in
                   dict(proposal.get("support_properties") or {}).items()}
        raw = [{"role": role, "kind": "linked", "occurrence_id": occurrence_id,
                "support_property": support.get(occurrence_id, "")}
               for role, occurrence_id in zip(("subject", "object"), occurrence_ids)]
    if not isinstance(raw, list) or len(raw) != 2:
        return None, "invalid_arguments"
    arguments = []
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            return None, "invalid_arguments"
        argument = dict(value)
        if argument.get("role") != ("subject", "object")[index]:
            return None, "invalid_argument_roles"
        kind = str(argument.get("kind", ""))
        if kind == "linked":
            occurrence_id = str((span_labels or {}).get(
                str(argument.get("span_label", "")), argument.get("occurrence_id", "")
            ))
            if occurrence_id not in occurrences:
                return None, "invalid_arguments"
            argument = {
                "role": argument["role"], "kind": kind,
                "occurrence_id": occurrence_id,
                "surface": str(occurrences[occurrence_id].get("surface", "")),
                "runtime_type": str(occurrences[occurrence_id].get("runtime_type", "")),
                "support_property": canon(str(argument.get("support_property", ""))),
            }
        elif kind == "context":
            literal = str(argument.get("literal", "")).strip()
            if literal:
                argument = {
                    "role": argument["role"], "kind": kind, "literal": literal,
                    "runtime_type": "", "start": None, "end": None,
                }
            else:
                candidate_id = str(argument.get("context_candidate_id", ""))
                candidate = context_candidates.get(candidate_id)
                if candidate is None:
                    return None, "unknown_context_candidate"
                argument = {
                    "role": argument["role"], "kind": kind,
                    "context_candidate_id": candidate_id,
                    "literal": str(candidate["literal"]),
                    "runtime_type": str(candidate["runtime_type"]),
                    "start": int(candidate["start"]), "end": int(candidate["end"]),
                }
        else:
            return None, "invalid_argument_kind"
        arguments.append(argument)
    return arguments, None


def _normalize_teacher_text(value: str) -> str:
    """Teacher answers/questions are inconsistently snake-cased
    ("solid_organ_transplant"); the lexical fact scorer tokenizes an underscore
    run as one token instead of its words, so map underscores to spaces and
    collapse whitespace before storing and scoring."""
    return re.sub(r"\s+", " ", str(value).replace("_", " ")).strip()


def _substitute_linked_surfaces(
    text: str, arguments: Sequence[Mapping], occurrences: Mapping[str, Mapping],
) -> str:
    """Replace a linked argument's protected surface/alias with the teacher's
    own selected support_property. Three consecutive live smokes wrote the
    surface into questions/answers despite prompt guidance; this mechanical
    substitution uses only teacher-chosen content and the leakage gates rerun
    on the result."""
    substituted = text
    for argument in arguments:
        if argument.get("kind") != "linked":
            continue
        level = str(argument.get("support_property", ""))
        if not level:
            continue
        for term in _occurrence_protected_terms(occurrences[argument["occurrence_id"]]):
            substituted = re.sub(
                rf"(?<!\w){re.escape(term)}(?!\w)", level, substituted,
                flags=re.IGNORECASE,
            )
    return substituted


def _relation_arguments_are_legal(
    relation: str, arguments: Sequence[Mapping], relation_contract: Mapping[str, Mapping],
) -> bool:
    allowed = relation_contract[relation]["argument_classes"]
    return len(arguments) == len(allowed) and all(
        _argument_relation_classes(
            str(argument.get("runtime_type", "")),
            str(argument.get("surface", argument.get("literal", ""))),
        ) & set(permitted)
        for argument, permitted in zip(arguments, allowed)
    )


def _argument_is_grounded(
    argument: Mapping, document: str, evidence_span: tuple[int, int],
    occurrences: Mapping[str, Mapping],
) -> bool:
    start, end = evidence_span
    if argument["kind"] == "linked":
        occurrence = occurrences[argument["occurrence_id"]]
        left, right, surface = occurrence.get("start"), occurrence.get("end"), argument["surface"]
    else:
        left, right, surface = argument["start"], argument["end"], argument["literal"]
    return (isinstance(left, int) and isinstance(right, int) and 0 <= left < right <= len(document)
            and document[left:right] == surface and start <= left < right <= end)


def _derived_relation_anchor(
    document: str,
    arguments: list[dict],
    occurrences: Mapping[str, Mapping],
    context_candidates: Mapping[str, Mapping],
    relation: str,
    relation_contract: Mapping[str, Mapping],
) -> tuple[str, tuple[int, int] | None, str | None, str | None]:
    """Resolve a v4 literal after deriving a source-local anchor from linked spans."""
    clauses = _source_clause_spans(document)

    def clause_index(start: int, end: int) -> int | None:
        return next((index for index, (left, right) in enumerate(clauses)
                     if left <= start < end <= right), None)

    linked = [argument for argument in arguments if argument["kind"] == "linked"]
    if not linked:
        return "", None, "missing_linked_argument", None
    linked_spans = [
        (int(occurrences[argument["occurrence_id"]]["start"]),
         int(occurrences[argument["occurrence_id"]]["end"]))
        for argument in linked
    ]
    linked_clause_indices = [clause_index(start, end) for start, end in linked_spans]
    if any(index is None for index in linked_clause_indices):
        return "", None, "invalid_evidence_occurrence", None
    linked_plan_section = _shared_plan_section(document, linked_spans)
    context = next((argument for argument in arguments if argument["kind"] == "context"), None)
    if context is not None and context.get("start") is None:
        literal = str(context["literal"])
        matches = _exact_substring_starts(document, literal)
        candidates = []
        for start in matches:
            end = start + len(literal)
            literal_clause = clause_index(start, end)
            if literal_clause is None:
                continue
            all_indices = [*linked_clause_indices, literal_clause]
            if max(all_indices) - min(all_indices) <= 1:
                candidates.append((start, end))
        if not candidates and linked_plan_section is not None:
            candidates = [
                (start, start + len(literal)) for start in matches
                if linked_plan_section[0] <= start < start + len(literal) <= linked_plan_section[1]
            ]
        if len(candidates) != 1:
            return "", None, "ambiguous_context_literal" if candidates else "unknown_context_literal", None
        start, end = candidates[0]
        # A literal that lands on a protected controlled span is an S-label
        # smuggled past the linked contract; the teacher must reference it by
        # its label (leakage-scope rule).
        if any(
            occurrence.get("controlled", True)
            and isinstance(occurrence.get("start"), int)
            and isinstance(occurrence.get("end"), int)
            and start < int(occurrence["end"]) and int(occurrence["start"]) < end
            for occurrence in occurrences.values()
        ):
            return "", None, "protected_context_literal", None
        typed = [row for row in context_candidates.values()
                 if row.get("literal") == literal and int(row.get("start", -1)) == start
                 and int(row.get("end", -1)) == end]
        if typed:
            if len(typed) != 1:
                return "", None, "untyped_context_literal", None
            context.update({
                "runtime_type": str(typed[0]["runtime_type"]),
                "context_candidate_id": str(typed[0]["context_candidate_id"]),
            })
        else:
            # Closed lexical rules only cover test/procedure/provider/status/
            # category; per the literal contract the relation slot's permitted
            # class types the rest (observed: drug literals such as
            # "synthroid").  Grounding, cue, leakage, and reader gates remain
            # the semantic filters.
            slot_index = arguments.index(context)
            slot_classes = relation_contract[relation]["argument_classes"][slot_index]
            context["runtime_type"] = slot_classes[0]
        context.update({"start": start, "end": end})
    all_indices = [
        clause_index(
            int(occurrences[argument["occurrence_id"]]["start"])
            if argument["kind"] == "linked" else int(argument["start"]),
            int(occurrences[argument["occurrence_id"]]["end"])
            if argument["kind"] == "linked" else int(argument["end"]),
        )
        for argument in arguments
    ]
    if not any(index is None for index in all_indices) and max(all_indices) - min(all_indices) <= 1:
        left, right = clauses[min(all_indices)][0], clauses[max(all_indices)][1]
        return document[left:right], (left, right), None, "clause"
    argument_spans = [
        (int(occurrences[argument["occurrence_id"]]["start"]),
         int(occurrences[argument["occurrence_id"]]["end"]))
        if argument["kind"] == "linked" else (int(argument["start"]), int(argument["end"]))
        for argument in arguments
    ]
    plan_section = _shared_plan_section(document, argument_spans)
    if plan_section is not None:
        return document[plan_section[0]:plan_section[1]], plan_section, None, "plan_section"
    # Spoken transcript: the arguments may sit in different turns of one
    # problem discussion (a patient acknowledgment between them). Ground within
    # that block; the hedge guard and cue check in compilation still apply.
    problem_block = _shared_problem_block(document, argument_spans)
    if problem_block is not None:
        return document[problem_block[0]:problem_block[1]], problem_block, None, "problem_block"
    return "", None, "invalid_evidence", None


def _relation_quote_has_direct_support(
    relation: str,
    quote: str,
    arguments: Sequence[Mapping],
    relation_contract: Mapping[str, Mapping],
    *,
    allow_adjacent_clauses: bool = False,
    allow_plan_section: bool = False,
) -> bool:
    """Require both exact arguments in one clause and an adapter-declared relation cue.

    Context literals may occur before the condition (for example, 'takes X for Y'),
    so source word order does not redefine the directional relation signature.
    """
    # A dot inside a run ("if ... and prescribe") is a transcript hesitation
    # marker, not a sentence boundary; only a lone period delimits a clause.
    clause_ranges = [
        (match.start(), match.end())
        for match in re.finditer(r"(?:[^\n.!?;]|\.(?=\.)|(?<=\.)\.)+", canon(quote))
        if match.group().strip()
    ]
    if len(clause_ranges) > 1 and not allow_adjacent_clauses:
        return False
    if len(clause_ranges) > 2 and not allow_adjacent_clauses:
        return False
    normalized = canon(quote)
    positions = []
    for argument in arguments:
        match = re.search(rf"(?<!\w){re.escape(canon(str(argument.get('surface', argument.get('literal', '')))))}(?!\w)", normalized)
        if match is None:
            return False
        positions.append((match.start(), match.end()))
    if not positions:
        return False
    cues = relation_contract[relation].get("cues", ())
    if len(clause_ranges) > 2:
        if allow_plan_section:
            local_left = min(position[0] for position in positions)
            local_right = max(position[1] for position in positions)
        else:
            # A multi-sentence speaker turn: the arguments must share one clause
            # (or two adjacent ones) and support is searched within those clause
            # bounds — explicit cues/connectors can precede the subject
            # ("you ca n't take X because of Y"; "<procedure> ... for <condition>"),
            # and between-args-only search missed them.
            argument_clauses = [
                next((index for index, (left, right) in enumerate(clause_ranges)
                      if left <= position[0] < right), None)
                for position in positions
            ]
            if (
                None in argument_clauses
                or abs(argument_clauses[0] - argument_clauses[-1]) > 1
            ):
                return False
            local_left = min(clause_ranges[index][0] for index in argument_clauses)
            local_right = max(clause_ranges[index][1] for index in argument_clauses)
        # An exact directional connector between the two arguments is direct
        # support on its own (forward "<subject> connector <object>", or the
        # reversed indication form "<object> ... for <subject>").
        if positions[0] < positions[1]:
            connector = normalized[positions[0][1]:positions[1][0]]
            connector_patterns = relation_contract[relation].get("connector_patterns", ())
        else:
            connector = normalized[positions[1][1]:positions[0][0]]
            connector_patterns = relation_contract[relation].get("reversed_connector_patterns", ())
        if any(re.fullmatch(pattern, connector) for pattern in connector_patterns):
            return True
        return any(re.search(re.escape(cue), normalized[local_left:local_right])
                   for cue in cues)
    if positions[0] < positions[1]:
        connector = normalized[positions[0][1]:positions[1][0]]
        if any(re.fullmatch(pattern, connector) for pattern in
               relation_contract[relation].get("connector_patterns", ())):
            return True
        if allow_adjacent_clauses:
            argument_clauses = [
                next((index for index, (left, right) in enumerate(clause_ranges)
                      if left <= position[0] < right), None)
                for position in positions
            ]
            if (
                None not in argument_clauses
                and abs(argument_clauses[0] - argument_clauses[1]) <= 1
            ):
                # Search the clause(s) holding the arguments, not only the
                # text between them: explicit cues can precede the subject
                # ("you ca n't take X because of Y"), and the reversed-order
                # single-clause branch below already searches the full clause.
                local_left = min(clause_ranges[index][0] for index in argument_clauses)
                local_right = max(clause_ranges[index][1] for index in argument_clauses)
                return any(
                    re.search(re.escape(cue), normalized[local_left:local_right])
                    for cue in relation_contract[relation].get("cues", ())
                )
        return False
    # Reversed textual order: an explicit reversed connector between object
    # and subject ("kidney transplant ... for some polycystic kidneys") is
    # direct support on its own.
    reversed_connector = normalized[positions[1][1]:positions[0][0]]
    if any(re.fullmatch(pattern, reversed_connector) for pattern in
           relation_contract[relation].get("reversed_connector_patterns", ())):
        return True
    cues = relation_contract[relation].get("cues", ())
    if len(clause_ranges) == 1:
        return any(re.search(re.escape(cue), normalized) for cue in cues)
    # Reversed textual order ("for your knee pain ... exacerbation of your
    # arthritis"): same rule as the forward branches — arguments share one
    # clause or two adjacent ones, and the cue must sit inside those clauses.
    argument_clauses = [
        next((index for index, (left, right) in enumerate(clause_ranges)
              if left <= position[0] < right), None)
        for position in positions
    ]
    if None in argument_clauses or abs(argument_clauses[0] - argument_clauses[1]) > 1:
        return False
    local_left = min(clause_ranges[index][0] for index in argument_clauses)
    local_right = max(clause_ranges[index][1] for index in argument_clauses)
    return any(re.search(re.escape(cue), normalized[local_left:local_right]) for cue in cues)


def compile_relational_assertions(
    doc_id: str,
    document: str,
    environment_document: Mapping,
    proposals: Sequence[Mapping],
    *,
    relation_contract: Mapping[str, Mapping] = ACI_RELATION_CONTRACT,
    context_candidates: Sequence[Mapping] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Compile bounded teacher proposals into frozen, evidence-checked assertions."""
    occurrences = {
        str(row["occurrence_id"]): row
        for row in environment_document.get("occurrences", [])
    }
    # Token sets of every detected entity surface. A context-literal argument
    # whose surface embeds one of these is a detected span that gets substituted
    # at render (e.g. brand "synthroid" -> <DRUG_1>), so the literal cannot
    # survive the generalized/placeholder docs and the relation is doomed.
    substitutable_token_sets = [
        tokens for occurrence in occurrences.values()
        if (tokens := _meaningful_tokens(str(occurrence.get("surface", ""))))
    ]
    decisions = {
        str(row["decision_id"]): row
        for row in environment_document.get("decisions", [])
    }
    context_by_id = {
        str(row["context_candidate_id"]): row
        for row in (relation_context_candidates(document)
                    if context_candidates is None else context_candidates)
    }
    span_labels = {
        str(row["span_label"]): str(row["occurrence_id"])
        for row in relation_teacher_span_inventory(environment_document)
    }
    legacy_windows = {
        str(row["evidence_window_id"]): row
        for row in relation_evidence_windows(document, environment_document)
    }

    def proposal_evidence(proposal: Mapping) -> tuple[str, tuple[int, int] | None, str | None]:
        window = legacy_windows.get(str(proposal.get("evidence_window_id", "")))
        if window is not None:
            return str(window["quote"]), (int(window["start"]), int(window["end"])), None
        quote = str(proposal.get("evidence_quote", ""))
        span, error = _resolve_relation_evidence_span(
            document, quote, proposal.get("evidence_start")
        )
        return quote, span, error
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
    fact_groups = set()
    for index, proposal_value in enumerate(proposals):
        proposal = dict(proposal_value)
        proposal_hash = _stable_hash(proposal)

        def reject(reason: str) -> None:
            quote, quote_span, _ = proposal_evidence(proposal)
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

        if index >= RELATION_TEACHER_MAX_RELATIONS:
            reject("relation_cap_exceeded")
            continue

        proposed_subtype = proposal.get("subtype")
        if proposed_subtype not in {None, "contextual_relation"}:
            reject("invalid_subtype")
            continue
        relation = str(proposal.get("relation", ""))
        # Cached v1 proposals conflated drug and procedure under treated_with.
        # Migrate only that old wire shape to the directional v2 relation; a v2
        # proposal labelled treated_with with a drug still fails its type contract.
        if proposal.get("arguments") is None and relation == "treated_with":
            legacy_ids = [str(value) for value in proposal.get("argument_occurrence_ids", [])]
            if len(legacy_ids) == 2 and legacy_ids[1] in occurrences and (
                _RUNTIME_TYPE_CLASSES.get(canon(str(occurrences[legacy_ids[1]].get("runtime_type", ""))))
                == "treatment"
            ):
                relation = "prescribed_with"
        if relation not in relation_contract:
            reject("invalid_relation")
            continue
        arguments, argument_error = _teacher_relation_arguments(
            proposal, occurrences, context_by_id, span_labels
        )
        if argument_error is not None:
            reject(argument_error)
            continue
        occurrence_ids = [argument["occurrence_id"] for argument in arguments
                          if argument["kind"] == "linked"]
        if len(set(occurrence_ids)) != len(occurrence_ids):
            reject("invalid_arguments")
            continue
        uses_v4_arguments = any("span_label" in argument or "literal" in argument
                                for argument in proposal.get("arguments") or [])
        anchor_kind = None
        if uses_v4_arguments:
            quote, evidence_span, evidence_error, anchor_kind = _derived_relation_anchor(
                document, arguments, occurrences, context_by_id,
                relation, relation_contract,
            )
        else:
            quote, evidence_span, evidence_error = proposal_evidence(proposal)
        if evidence_error is not None:
            reject(evidence_error)
            continue
        if not _relation_arguments_are_legal(relation, arguments, relation_contract):
            reject("invalid_argument_types")
            continue
        # A context literal that coincides with a detected entity will be
        # substituted in the generalized/placeholder renders, so a literal
        # answer target can never match there. Reject up front instead of
        # emitting a relation that is guaranteed to fail the three-point gate;
        # once the entity is lattice-resolvable it comes back as a linked arg.
        if any(
            argument.get("kind") == "context"
            and argument.get("literal")
            and any(
                occurrence_tokens <= _meaningful_tokens(str(argument["literal"]))
                for occurrence_tokens in substitutable_token_sets
            )
            for argument in arguments
        ):
            reject("literal_will_be_substituted")
            continue
        if not all(_argument_is_grounded(argument, document, evidence_span, occurrences)
                   for argument in arguments):
            reject("invalid_evidence_occurrence")
            continue
        if not _relation_quote_has_direct_support(
            relation,
            quote,
            arguments,
            relation_contract,
            allow_adjacent_clauses=(uses_v4_arguments or proposal.get("evidence_window_id") is not None),
            allow_plan_section=anchor_kind in {"plan_section", "problem_block"},
        ):
            reject("invalid_evidence")
            continue
        # A conditional/hypothetical statement is not an asserted fact at any
        # anchor scope ("possibly refer to physical therapy" is hedged even in
        # one clause); the tightened pattern clears the "if ..." disfluency.
        if uses_v4_arguments and _relation_window_is_hedged(
            document, arguments, occurrences
        ):
            reject("hedged_relation")
            continue
        if not _proposal_polarity_matches_frozen_occurrences(
            proposal, occurrence_ids, occurrences
        ):
            reject("invalid_polarity")
            continue
        if occurrence_ids and _source_contains_relation_contradiction(
            relation, document, occurrence_ids, occurrences, relation_contract
        ):
            reject("source_contradiction")
            continue
        support = {argument["occurrence_id"]: argument["support_property"]
                   for argument in arguments if argument["kind"] == "linked"}
        if any(not support[occurrence_id] or
               support[occurrence_id] not in legal_properties[occurrence_id]
               for occurrence_id in occurrence_ids):
            reject("invalid_property")
            continue
        question = _normalize_teacher_text(str(proposal.get("question", "")))
        if not question.endswith("?"):
            reject("invalid_question")
            continue
        accepted_values = [_normalize_teacher_text(str(value))
                           for value in proposal.get("accepted_answers", [])
                           if _normalize_teacher_text(str(value))]
        # v1 migration: preserve old cached proposals, but new requests must author answers.
        if not accepted_values:
            answer_occurrence_id = str(proposal.get("answer_occurrence_id", ""))
            answer_property = canon(str(proposal.get("answer_property", "")))
            if answer_occurrence_id not in occurrence_ids or answer_property != support.get(answer_occurrence_id):
                reject("invalid_property")
                continue
            accepted_values = [answer_property]
        scoring_contract = dict(proposal.get("scoring_contract") or {
            "kind": "semantic_qa", "match": "fact_score"
        })
        if scoring_contract.get("kind") != "semantic_qa" or not accepted_values:
            reject("invalid_scoring_contract")
            continue
        sanitized_qa = False
        if uses_v4_arguments:
            sanitized_question = _substitute_linked_surfaces(question, arguments, occurrences)
            sanitized_values = [
                _substitute_linked_surfaces(answer, arguments, occurrences)
                for answer in accepted_values
            ]
            sanitized_qa = (sanitized_question, sanitized_values) != (question, accepted_values)
            question, accepted_values = sanitized_question, sanitized_values
        # Placeholder-label tokens of the linked argument types ("medication",
        # "condition") are information-free for level-based QA; only the v4
        # contract exempts them so cached legacy replies keep their outcomes.
        answer_exempt_types = (
            [argument["runtime_type"] for argument in arguments
             if argument["kind"] == "linked"]
            if uses_v4_arguments else ""
        )
        if any(_question_leaks_answer(question, answer, answer_exempt_types)
               for answer in accepted_values):
            reject("answer_leakage")
            continue
        protected_terms = []
        allowed_level_tokens: dict[str, frozenset[str]] = {}
        for occurrence_id_value, occurrence in occurrences.items():
            if not (occurrence.get("controlled", True)
                    and decisions.get(str(occurrence.get("decision_id")), {}).get("controlled", True)):
                continue
            level_tokens = frozenset(
                token
                for property_level in legal_properties.get(occurrence_id_value, ())
                for token in _meaningful_tokens(property_level)
            )
            for term in _occurrence_protected_terms(occurrence):
                protected_terms.append(term)
                allowed_level_tokens[term] = allowed_level_tokens.get(term, frozenset()) | level_tokens
        if _question_leaks_protected_term(question, protected_terms, allowed_level_tokens):
            reject("protected_locator")
            continue
        if proposal.get("arguments") is not None and any(
            _question_leaks_protected_term(answer, protected_terms, allowed_level_tokens)
            for answer in accepted_values
        ):
            reject("protected_answer")
            continue
        decision_requirements = {
            str(occurrences[occurrence_id]["decision_id"]): support[occurrence_id]
            for occurrence_id in occurrence_ids
        }
        fact_group = (relation, tuple(
            (argument["kind"], argument.get("occurrence_id", canon(argument.get("literal", ""))))
            for argument in arguments
        ))
        if fact_group in fact_groups:
            reject("duplicate_fact_group")
            continue
        fact_groups.add(fact_group)
        # The answered argument (default: the object). A linked answer is scored
        # by lattice entailment against its decision's frozen chain; a literal
        # answer keeps lexical matching against the exact grounded span.
        answer_role = str(proposal.get("answer_role", "object"))
        answer_argument = arguments[0] if answer_role == "subject" else arguments[1]
        if answer_argument["kind"] == "linked":
            answer_target = {
                "kind": "linked_decision",
                "decision_id": str(occurrences[answer_argument["occurrence_id"]]["decision_id"]),
                "required_property": answer_argument["support_property"],
            }
        else:
            answer_target = {"kind": "literal", "expected_values": accepted_values}
        argument_spans = {
            occurrence_id: [
                int(occurrences[occurrence_id]["start"]),
                int(occurrences[occurrence_id]["end"]),
            ]
            for occurrence_id in occurrence_ids
        }
        source_span = {
            "start": evidence_span[0],
            "end": evidence_span[1],
            "quote_hash": _stable_hash(quote),
        }
        evidence = {
            "authority": "source_document",
            "proposal_hash": proposal_hash,
            "sanitized_qa": sanitized_qa,
            "source_span": source_span,
            "argument_spans": argument_spans,
            # Turn indices the answer depends on, resolved against the source now
            # so gate and runtime excerpt the same turns without the source text.
            "reader_turns": _source_turns_for_ranges(
                document,
                [(source_span["start"], source_span["end"])]
                + [(span[0], span[1]) for span in argument_spans.values()],
            ),
            "arguments": arguments,
        }
        accepted.append({
            "family": "context",
            "scope": "linked",
            "subtype": "contextual_relation",
            "relation": relation,
            "occurrence_ids": occurrence_ids,
            "group_id": "relation:" + relation + ":" + _stable_hash([
                {key: argument[key] for key in ("kind", "occurrence_id", "literal") if key in argument}
                for argument in arguments
            ]),
            "question": question,
            "accepted_values": accepted_values,
            "answer_target": answer_target,
            "answer_type": _relation_answer_type_hint(relation, answer_role),
            "scoring_contract": scoring_contract,
            "decision_requirements": decision_requirements,
            "evidence": evidence,
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


def _frozen_semantic_chain(decision: Mapping, source_aliases: Sequence[str]) -> list[dict]:
    """Freeze explicit entailment closure for one decision's lattice.

    Ordered specific -> coarse: KEEP entails every level; a level entails itself
    and every coarser level; placeholder entails nothing. This is a declaration
    by the accepted lattice profile, not proven natural-language entailment; the
    linked-answer scorer reads it instead of re-deriving from mutable profiles,
    counts, or lexical similarity (docs/handoffs/2026-07-14-qa-reader-lattice-scoring.md).
    """
    levels = sorted(
        (action for action in decision.get("actions", []) if action.get("mode") == "level"),
        key=lambda action: float(action.get("coarseness_rank", 0.0)),
    )
    ordered_properties = [
        canon(str(action["entails"][0]))
        for action in levels
        if action.get("entails")
    ]
    canonical_key = str(decision.get("canonical_key", ""))
    keep_aliases = list(dict.fromkeys(
        [canonical_key, *[str(alias) for alias in source_aliases]]
    ))
    chain = [{
        "node": "keep",
        "answer_aliases": [alias for alias in keep_aliases if alias],
        "entailed_properties": list(ordered_properties),
    }]
    for index, prop in enumerate(ordered_properties):
        chain.append({
            "node": prop,
            "answer_aliases": [prop],
            "entailed_properties": ordered_properties[index:],
        })
    chain.append({"node": "placeholder", "answer_aliases": [], "entailed_properties": []})
    return chain


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
                decision["semantic_chain"] = _frozen_semantic_chain(decision, protected_aliases)
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
                        **dict(detector_provenance or {}),
                        **dict(row.get("detector_provenance") or {}),
                        "score": row.get("score"),
                    }
                } if (
                    detector_provenance is not None or row.get("detector_provenance")
                ) else {}),
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
    candidate_accounting_by_document: dict[str, list[dict]] = {}
    relation_generation_by_document: dict[str, list[dict]] = {}
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
        chain_by_decision = {
            str(decision["decision_id"]): decision.get("semantic_chain", [])
            for decision in decisions
        }
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
                        chain_by_decision=chain_by_decision,
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
        if relation_teacher is not None and relation_teacher_span_inventory(environment_document):
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
                    if isinstance(relation_teacher, OpenRouterRelationTeacher):
                        proposals = relation_teacher.propose(
                            prompt,
                            response_format=relation_teacher_response_format(
                                environment_document, source,
                            ),
                        )
                    else:
                        proposals = relation_teacher.propose(prompt)
                    if isinstance(proposals, RelationTeacherProposals):
                        try:
                            candidate_accounting_by_document[doc_id] = (
                                _validated_candidate_accounting(
                                    proposals.candidate_accounting,
                                    environment_document,
                                    source,
                                )
                            )
                            emitted_labels = {
                                str(argument.get("span_label"))
                                for proposal in proposals
                                for argument in proposal.get("arguments") or []
                                if isinstance(argument, Mapping)
                                and argument.get("kind") == "linked"
                            }
                            ledger_emitted = {
                                str(row.get("candidate_label"))
                                for row in candidate_accounting_by_document[doc_id]
                                if row.get("state") == "emitted"
                            }
                            if emitted_labels != ledger_emitted:
                                candidate_accounting_by_document[doc_id].append({
                                    "state": "ledger_inconsistent",
                                    "reason": "emitted_labels_do_not_match_proposals",
                                })
                        except ValueError as error:
                            candidate_accounting_by_document[doc_id] = [{
                                "state": "ledger_inconsistent",
                                "reason": "invalid_teacher_candidate_accounting",
                                "detail": str(error),
                            }]
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
                        relation_attempts = [{
                            "proposal_index": proposal_index,
                            "relation": proposal.get("relation"),
                            "arguments": proposal.get("arguments"),
                            "question": proposal.get("question"),
                            "accepted_answers": proposal.get("accepted_answers"),
                            "scoring_contract": proposal.get("scoring_contract"),
                            "status": "rejected",
                            "reason": "uncompiled",
                        } for proposal_index, proposal in enumerate(proposals)]
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
                        attempts_by_proposal_hash = {
                            _stable_hash(proposal): attempt
                            for proposal, attempt in zip(proposals, relation_attempts)
                        }
                        attempts_by_candidate_hash = {
                            _stable_hash(candidate): attempts_by_proposal_hash.get(
                                candidate.get("evidence", {}).get("proposal_hash")
                            )
                            for candidate in relation_candidates
                        }
                        for rejection in relation_rejections:
                            preserve_rejection(rejection, doc_id=doc_id)
                            proposal_index = rejection.get("proposal_index")
                            if isinstance(proposal_index, int) and proposal_index < len(relation_attempts):
                                relation_attempts[proposal_index].update({
                                    "status": "rejected",
                                    "reason": rejection.get("detail_reason", rejection.get("reason")),
                                })
                        rejection_count_before_validation = len(rejection_records)
                        accepted_relations = validate_candidate_rows(
                            [dict(row) for row in relation_candidates]
                        )
                        for row in accepted_relations:
                            attempt = attempts_by_proposal_hash.get(
                                row.get("evidence", {}).get("proposal_hash")
                            )
                            if attempt is not None:
                                attempt.update({"status": "kept", "reason": "accepted"})
                        for rejection in rejection_records[rejection_count_before_validation:]:
                            attempt = attempts_by_candidate_hash.get(
                                rejection.get("evidence", {}).get("candidate_hash")
                            )
                            if attempt is not None:
                                attempt.update({
                                    "status": "rejected",
                                    "reason": rejection.get("detail_reason", rejection.get("reason")),
                                })
                        relation_generation_by_document[doc_id] = relation_attempts
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
                    error_code = (
                        error.code
                        if isinstance(error, RelationTeacherResponseError)
                        else "teacher_generation_failed"
                    )
                    preserve_rejection(_rejection_record(
                        reason="generation_failed",
                        detail_reason=error_code,
                        attempt={
                            "doc_id": doc_id,
                            "prompt_hash": _stable_hash(prompt),
                            "error_type": type(error).__name__,
                            "error_code": error_code,
                            "raw_length": (
                                error.raw_length
                                if isinstance(error, RelationTeacherResponseError)
                                else None
                            ),
                            "completion_state": (
                                error.completion_state
                                if isinstance(error, RelationTeacherResponseError)
                                else None
                            ),
                            "definition_version": "contextual-relation-v1",
                        },
                        evidence={
                            "source": "relation_teacher",
                            "prompt_hash": _stable_hash(prompt),
                            "error_type": type(error).__name__,
                            "error_code": error_code,
                            "raw_length": (
                                error.raw_length
                                if isinstance(error, RelationTeacherResponseError)
                                else None
                            ),
                            "completion_state": (
                                error.completion_state
                                if isinstance(error, RelationTeacherResponseError)
                                else None
                            ),
                        },
                    ), doc_id=doc_id)
        elif relation_teacher is not None:
            preserve_rejection(_rejection_record(
                reason="not_generated",
                detail_reason="no_eligible_relation_spans",
                attempt={"doc_id": doc_id, "definition_version": "contextual-relation-v4"},
                evidence={"source": "relation_teacher", "eligible_span_count": 0},
            ), doc_id=doc_id)
        elif contextual_relation_count < min_contextual_relations:
            preserve_rejection(_rejection_record(
                reason="not_generated",
                detail_reason="contextual_relation_threshold_unmet_no_teacher",
                attempt={"doc_id": doc_id, "definition_version": "contextual-relation-v4"},
                evidence={"source": "relation_teacher", "teacher_configured": False},
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
    artifact["relation_candidate_accounting"] = candidate_accounting_by_document
    artifact["relation_generation"] = relation_generation_by_document
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


def _resolve_semantic_node(chain: Sequence[Mapping], answer: str) -> dict | None:
    """Resolve a free-form reader answer to exactly one node in one decision's
    frozen semantic chain, by its answer_aliases. Decision-scoped and lexical
    (containment on meaningful tokens); ambiguous or unresolved -> None. A
    protected source value resolves to KEEP locally without ever entering the
    artifact's public answer golds."""
    answer_tokens = _meaningful_tokens(answer)
    if not answer_tokens:
        return None
    matches = [
        node for node in chain
        if any(
            (alias_tokens := _meaningful_tokens(str(alias)))
            and alias_tokens <= answer_tokens
            for alias in node.get("answer_aliases") or []
        )
    ]
    # A coarser level's words are often a subset of a finer level's answer
    # (e.g. "analgesic" <= "opioid analgesic"), so several nodes can match. The
    # chain is linear specific->coarse, so matches[0] is the finest match; finer
    # entails coarser, making it the correct and strictest resolution.
    # ponytail: assumes a linear chain (no sibling levels); revisit for a DAG.
    return dict(matches[0]) if matches else None


def _linked_answer_score(answer: str, chain: Sequence[Mapping], required_property: str) -> float:
    """Binary credit iff the resolved node entails the required property. KEEP
    and any supported generalization pass; coarser meanings and placeholder
    fail. No token-F1 partial credit for a semantic band."""
    node = _resolve_semantic_node(chain, answer)
    if node is None:
        return 0.0
    return 1.0 if canon(str(required_property)) in {
        canon(str(prop)) for prop in node.get("entailed_properties") or []
    } else 0.0


def _source_turns_for_ranges(
    source: str, char_ranges: Sequence[tuple[int, int]],
) -> list[int]:
    """The newline-delimited turn indices of `source` that cover `char_ranges`
    (offsets into `source`). Computed once at build time so the runtime scorer
    can excerpt without ever seeing the source text."""
    if not char_ranges:
        return []
    starts, position = [], 0
    for line in source.splitlines(keepends=True):
        starts.append(position)
        position += len(line)
    def line_of(offset: int) -> int:
        return max(0, bisect.bisect_right(starts, offset) - 1)
    turns: set[int] = set()
    for start, end in char_ranges:
        for index in range(line_of(start), line_of(max(start, end - 1)) + 1):
            turns.add(index)
    return sorted(turns)


def _turn_excerpt(context: str, core_turns: Sequence[int], *, window: int) -> str:
    """Slice `context` to the given turn indices plus `window` neighbor turns
    each side. Empty `core_turns`, or indices past the end of `context`, fall
    back to the full `context` so a diverged render is never mis-sliced."""
    if not core_turns:
        return context
    lines = context.splitlines()
    if max(core_turns) >= len(lines):
        return context
    low = max(0, min(core_turns) - window)
    high = min(len(lines) - 1, max(core_turns) + window)
    return "\n".join(lines[low:high + 1])


def _context_answer_score(
    row: Mapping, answer: str, chain_by_decision: Mapping[str, Sequence[Mapping]],
) -> float:
    """Route a context answer to the linked (lattice-entailment) or literal
    (lexical) scorer per the assertion's answer_target; fall back to legacy
    accepted_values lexical scoring when no target is present."""
    target = row.get("answer_target") or {}
    if target.get("kind") == "linked_decision":
        chain = chain_by_decision.get(str(target.get("decision_id")))
        if not chain:
            return 0.0
        return _linked_answer_score(answer, chain, str(target.get("required_property", "")))
    if target.get("kind") == "literal":
        return _answer_score(answer, list(target.get("expected_values") or []))
    return _answer_score(answer, list(row.get("accepted_values") or []))


def _permuted_reader_question(assertion: Mapping, permutation_index: int) -> str:
    question = str(assertion["question"])
    answer_type = assertion.get("answer_type")
    if answer_type:
        # The typed extraction directive rides in the reader-facing question
        # only; the stored artifact question stays clean.
        question = f"Extract the shortest {answer_type} span that answers: {question}"
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
    chain_by_decision: Mapping[str, Sequence[Mapping]] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    """Apply frozen repeated original/generalization/placeholder reader checks."""
    chain_by_decision = chain_by_decision or {}
    rows = [row for row in assertions if row.get("family") == "context"]
    if stability_repetitions < 1 or option_permutations < 1:
        raise ValueError("reader stability repetitions and option permutations must be positive")
    if not 0.0 < stability_threshold <= 1.0:
        raise ValueError("reader stability threshold must be in (0, 1]")

    trials_by_assertion: dict[str, list[dict]] = defaultdict(list)
    for repetition in range(stability_repetitions):
        for permutation_index in range(option_permutations):
            # Each assertion reads only the transcript turns covering its own
            # arguments/evidence (same turn indices across all three renders),
            # not the whole note.
            original_answers, representative_answers, placeholder_answers = [], [], []
            for row in rows:
                question = _permuted_reader_question(row, permutation_index)
                turns = (row.get("evidence") or {}).get("reader_turns") or []
                original_answers += reader([question], _turn_excerpt(
                    original_context, turns, window=CONTEXT_READER_TURN_WINDOW))
                representative_answers += reader([question], _turn_excerpt(
                    representative_context, turns, window=CONTEXT_READER_TURN_WINDOW))
                placeholder_answers += reader([question], _turn_excerpt(
                    placeholder_context, turns, window=CONTEXT_READER_TURN_WINDOW))
            if not all(len(answers) == len(rows) for answers in (
                original_answers, representative_answers, placeholder_answers
            )):
                raise ValueError("reader returned the wrong number of answers")
            for row, original, representative, placeholder in zip(
                rows, original_answers, representative_answers, placeholder_answers
            ):
                scores = {
                    "original": _context_answer_score(row, original, chain_by_decision),
                    "representative": _context_answer_score(row, representative, chain_by_decision),
                    "placeholder": _context_answer_score(row, placeholder, chain_by_decision),
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
                or expected.get("kind") not in {"rows", "labeled_rows", "none"}
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
                and isinstance(matches[0].get(field), str)
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
            all(
                isinstance(row.get(field), str) and canon(row[field]) == value
                for field, value in expected.items()
            )
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
    # Match the gate exactly: typed-directive question + per-assertion turn
    # excerpt of doc_p (same stored turn indices), so an assertion accepted on
    # its excerpt at build time is scored on the same excerpt at runtime.
    context_answers = []
    for row in context_rows:
        question = _permuted_reader_question(row, 0)
        turns = (row.get("evidence") or {}).get("reader_turns") or []
        context_answers += reader(
            [question], _turn_excerpt(doc_p, turns, window=CONTEXT_READER_TURN_WINDOW)
        )
    if len(context_answers) != len(context_rows):
        raise ValueError("reader returned the wrong number of answers")

    chain_by_decision = {
        str(decision["decision_id"]): decision.get("semantic_chain", [])
        for decision in artifact["documents"][doc_id].get("decisions", [])
    }
    scores: dict[str, float] = {}
    for row, answer in zip(context_rows, context_answers):
        scores[str(row["assertion_id"])] = _context_answer_score(
            row, answer, chain_by_decision
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
