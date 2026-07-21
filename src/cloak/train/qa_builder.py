"""QA-builder v2 artifact weighting, support anchors, validation, and scoring."""
from __future__ import annotations

import bisect
import hashlib
import inspect
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from copy import deepcopy
from math import ceil, isfinite
from numbers import Real

from cloak.runtime_types import placeholder_token, placeholder_type_token
from cloak.train.reward import QA_BASE_URL, QA_MODEL, canon, fact_score
from cloak.train.qa_audit import build_qa_audit

RELATION_TEACHER_MODEL = "openai/gpt-oss-120b"
RELATION_TEACHER_BASE_URL = "https://openrouter.ai/api/v1"
# Pin a reliable OpenRouter provider: the free Nemotron route was intermittently
# empty/rate-limited/errored; deepinfra/turbo serves gpt-oss-120b with stable,
# valid structured output. allow_fallbacks stays off so routing never drifts.
RELATION_TEACHER_PROVIDER = "deepinfra/turbo"
RELATION_TEACHER_PROMPT_VERSION = "qa-relation-teacher-v21"
RELATION_TEACHER_RESPONSE_SCHEMA = {"type": "relation-qa-batch", "version": 9}
RELATION_TEACHER_REVISION = "qa-relation-teacher-r32"
RELATION_TEACHER_MAX_RELATIONS = 12
# The gleaning+repair pass batches its targets: one teacher call per <=N targets, so a note with
# many opportunities (D2N002 hit 54) is not crammed into a single unfocusable prompt.
RELATION_REPAIR_MAX_TARGETS_PER_CALL = 20
# Retry an empty (HTTP-200, no choices/content) teacher reply before failing -- transient.
_TEACHER_EMPTY_RETRIES = 3
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
            "prescribed_with", "procedure_for", "tests_for",
            "contraindicated_because_of", "causes_or_explains",
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
        "answer_role": {"enum": ["subject", "object"]},
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
    "procedure_for",
    "tests_for",
    "contraindicated_because_of",
    "causes_or_explains",
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
    # Therapeutic procedure or referral for a condition (incl. a past treatment).
    # Object is procedure-class only: the detector emits no provider/specialty type,
    # so a referral grounds via its procedure target (e.g. physical therapy).
    "procedure_for": (("condition",), ("procedure",)),
    # Diagnostic/monitoring test that discovers or monitors a condition.
    "tests_for": (("condition",), ("monitoring", "procedure")),
    "contraindicated_because_of": (("treatment", "procedure"), ("condition",)),
    "causes_or_explains": (("condition",), ("condition", "symptom")),
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
    "procedure_for": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["procedure_for"],
        "definition": "a therapeutic procedure or referral for a condition, including a past "
                      "treatment the patient already had (e.g. a prior transplant/surgery)",
        "cues": ("treated with", "treat", "treated", "referred to", "refer",
                 "had", "underwent", "received", "status post", "history of"),
        "connector_patterns": (
            # forward verbs: condition <treated with|referred to> procedure
            r"\s+(?:(?:is|was|are|were|has been|had been|is being|was being)\s+)?"
            r"(?:treated\s+with|referred\s+to)\s+",
        ),
        # Past-treatment indication, procedure textually first: "had the kidney
        # transplant a few years ago for some polycystic kidneys".
        "reversed_connector_patterns": (
            r"\s+(?:[\w',]+\s+){0,5}?for\s+(?:some\s+|your\s+|the\s+|a\s+|an\s+|his\s+|her\s+)?",
        ),
    },
    "tests_for": {
        "argument_classes": _RELATION_ARGUMENT_CLASSES["tests_for"],
        "definition": "a diagnostic or monitoring test/study that discovers or monitors a "
                      "condition (a test ordered to follow it, or one whose result showed it)",
        "cues": ("order", "ordered", "check", "monitor", "monitored by", "follow",
                 "evaluate", "assess", "panel", "show", "shows", "showed", "reveal",
                 "reveals", "revealed", "demonstrat", "notice", "found", "came back",
                 "consistent with", "notable for", "significant for", "positive for"),
        "connector_patterns": (
            # forward monitoring: condition <monitored by|order|check|follow> test
            r"\s+(?:(?:is|was|are|were|has been|had been|is being|was being)\s+)?monitored\s+by\s+",
            r"\s+(?:[\w',-]+\s+){0,6}?(?:order(?:ed|s)?|check(?:ed|s)?|monitor(?:ed|s)?|"
            r"follow(?:ed|s)?|evaluat\w+|assess\w+)\s+(?:[\w',-]+\s+){0,3}?",
        ),
        # Reversed discovery, test textually first: "the ct shows a stone",
        # "x-ray ... i do notice dorsal displacement", "biopsy came back as dcis".
        "reversed_connector_patterns": (
            r"\s+(?:[\w',-]+\s+){0,8}?(?:show(?:s|ed|n)?|reveal(?:s|ed)?|demonstrat\w+|"
            r"notic\w+|found|came\s+back\w*|positive\s+for)\s+(?:[\w',-]+\s+){0,4}?",
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
}
# Relations whose lexical cue is NECESSARY BUT NOT SUFFICIENT: their argument classes span every
# condition x condition permutation, and the block-level cue match (allow_plan_section) fires for any
# such pair sharing a causal word, so cue-ok is unreliable. When an escalator (MedGemma judge) is
# configured, the miner routes even cue-OK pairs of these relations through it for confirmation --
# so the escalator acts as a precision FILTER here, not only additive recovery. With no escalator,
# they fall back to cue-only (the no-regression invariant is unchanged in that mode).
_JUDGE_GATED_RELATIONS = frozenset({"causes_or_explains"})
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


# A conditional/planned relation is valid only when its QUESTION carries the
# conditionality (so it is not asserted as already done).
_CONDITIONAL_QUESTION_PATTERN = re.compile(
    r"\b(?:may|might|could|would|possibl\w*|potential\w*|planned|plan\s+to|"
    r"consider\w*|recommend\w*|in\s+the\s+future|future|if\s+\w+)\b",
    re.IGNORECASE,
)


def _question_is_conditional(question: str) -> bool:
    """True if the question is phrased as a conditional/future plan, not an assertion."""
    return _CONDITIONAL_QUESTION_PATTERN.search(question or "") is not None


def _relation_window_is_hedged(
    document: str, arguments: Sequence[Mapping], occurrences: Mapping[str, Mapping],
    relation: str | None = None,
) -> bool:
    """True if the source region spanning the arguments is conditional/hedged.

    tests_for is exempt: a diagnostic test's window is pervasively finding-hedged
    ("a possible kidney stone", "there might be a question of a stone", "likely ...")
    because uncertainty about the RESULT is the point -- yet the test was still performed,
    so the relation is not conditional. Positional finding-hedge detection proved to be
    whack-a-mole (finding hedges appear anywhere in the window), so the gate is skipped
    for tests_for. It stays meaningful for prescribed_with / procedure_for, where
    "might refer" / "might prescribe" genuinely mark a planned-vs-done action.
    """
    if relation == "tests_for":
        return False
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
    """Optional cached JSON relation proposer with an explicit OpenRouter route."""

    @property
    def pin(self) -> dict:
        pin = {
            "provider": "openrouter",
            "model": self._model,
            "routed_provider": self._routed_provider,
            "base_url": self._base_url,
            "prompt_version": self._prompt_version,
            "response_schema": deepcopy(RELATION_TEACHER_RESPONSE_SCHEMA),
            "response_format": deepcopy(RELATION_TEACHER_RESPONSE_FORMAT),
            "generation_config": deepcopy(self._generation_config),
            "revision": self._revision,
        }
        if self._allow_fallbacks or self._routed_provider is None:
            pin["allow_fallbacks"] = self._allow_fallbacks
        return pin

    def __init__(
        self,
        *,
        model: str = RELATION_TEACHER_MODEL,
        routed_provider: str | None = RELATION_TEACHER_PROVIDER,
        allow_fallbacks: bool = False,
        base_url: str = RELATION_TEACHER_BASE_URL,
        generation_config: Mapping | None = None,
        prompt_version: str = RELATION_TEACHER_PROMPT_VERSION,
        revision: str = RELATION_TEACHER_REVISION,
        include_reasoning: bool = False,
    ):
        from cloak.llm import LLMClient

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for relation escalation")
        if not os.getenv("CLOAK_LLM_CACHE"):
            raise ValueError("CLOAK_LLM_CACHE is required for relation escalation")
        self._model = str(model)
        self._routed_provider = routed_provider
        self._allow_fallbacks = bool(allow_fallbacks)
        self._base_url = str(base_url)
        self._generation_config = deepcopy(
            RELATION_TEACHER_GENERATION_CONFIG if generation_config is None else generation_config
        )
        # Diagnostic only: have OpenRouter return the reasoning trace (for prompt A/B tweaking). The
        # `exclude` flag changes only whether the trace is echoed, not the generated content -- but it
        # is part of extra_body/the cache key, so enable it ONLY on a teacher whose prompt already
        # differs (e.g. the gleaning pass), never the primary, to avoid busting the primary cache.
        if include_reasoning:
            self._generation_config = {**self._generation_config, "reasoning": {"exclude": False}}
        self._prompt_version = str(prompt_version)
        self._revision = str(revision)
        provider_config = {"allow_fallbacks": self._allow_fallbacks}
        if self._routed_provider:
            provider_config["order"] = [self._routed_provider]
        self._client = LLMClient(
            self._model,
            base_url=self._base_url,
            api_key=api_key,
            temperature=0.0,
            response_format=deepcopy(RELATION_TEACHER_RESPONSE_FORMAT),
            extra_body={
                "reasoning": deepcopy(self._generation_config["reasoning"]),
                "provider": provider_config,
            },
            single_flight=True,
        )

    def propose(
        self, prompt: str, *, response_format: Mapping | None = None,
    ) -> RelationTeacherProposals:
        fmt = {"response_format": dict(response_format)} if response_format else {}
        # An OpenRouter HTTP-200 with no choices/content is a TRANSIENT provider condition
        # (observed intermittently on both gpt-oss and the free routes). Retry it like a
        # throttle before giving up -- refresh=True forces a fresh call so a non-cached empty
        # is never reused. The SDK already retries 429/5xx underneath; this covers empty-200.
        raw = ""
        for attempt in range(_TEACHER_EMPTY_RETRIES + 1):
            raw = self._client.generate(prompt, **({"refresh": True} if attempt else {}), **fmt)
            if raw and raw.strip():
                break
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

    def _read_set_one(self, question: str, context: str) -> str:
        # Set-valued read (validated wrapper: scripts/spikes/set_valued_gate_probe.py): the raw
        # JSON-array reply is returned verbatim; _context_answer_score parses and scores recall.
        prompt = (
            "Read the DOCUMENT and complete the REQUEST. Copy each answer verbatim as a short "
            "phrase from the DOCUMENT; include nothing not in the DOCUMENT. Respond with ONLY a "
            'JSON array of strings, e.g. ["x","y"]. If there are none, respond [].\n\n'
            f"DOCUMENT:\n{context}\n\n"
            f"REQUEST: {question}"
        )
        return self._client.generate(prompt).strip()

    def read_set(self, questions: list[str], context: str) -> list[str]:
        return [self._read_set_one(question, context) for question in questions]

    def __call__(self, questions: list[str], context: str) -> list[str]:
        return [self._read_one(question, context) for question in questions]


_batched_context_reader = None


def read_context_batch(questions: list[str], context: str) -> list[str]:
    global _batched_context_reader
    if _batched_context_reader is None:
        _batched_context_reader = BatchedContextReader()
    return _batched_context_reader(questions, context)


read_context_batch.pin = deepcopy(DEFAULT_CONTEXT_READER_PIN)


def read_context_set_batch(questions: list[str], context: str) -> list[str]:
    """Set-valued reads (same pinned reader model/endpoint, JSON-array response contract)."""
    global _batched_context_reader
    if _batched_context_reader is None:
        _batched_context_reader = BatchedContextReader()
    return _batched_context_reader.read_set(questions, context)


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
        def forced_render_only(decision: Mapping) -> bool:
            actions = decision.get("actions")
            return (
                decision.get("ranker_selectable") is False
                and isinstance(actions, list)
                and bool(actions)
                and all(
                    isinstance(action, Mapping)
                    and action.get("mode") == "placeholder"
                    and action.get("forced_placeholder") is True
                    for action in actions
                )
            )

        unsupported = sorted({
            str(decision.get("runtime_type", ""))
            for document in frozen_environment.get("documents", {}).values()
            for decision in document.get("decisions", [])
            if (
                decision.get("runtime_type") not in self.controlled_runtime_types
                and not forced_render_only(decision)
            )
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
        *,
        relation_opportunities: Sequence[Mapping] | None = None,
        context_judge: "Callable[..., bool] | None" = None,
    ) -> list[dict]:
        return [
            *self.semantic_property_candidates(
                doc_id, document, environment_document,
                relation_opportunities=relation_opportunities,
                context_judge=context_judge,
            ),
            *self.delivered_candidates(
                doc_id, document, self._references[doc_id], environment_document
            ),
        ]

    def semantic_property_candidates(
        self,
        doc_id: str,
        document: str,
        environment_document: Mapping,
        *,
        relation_opportunities: Sequence[Mapping] | None = None,
        context_judge: "Callable[..., bool] | None" = None,
    ) -> list[dict]:
        """Compile safe category probes from legal frozen action entailments.

        On a role-cue regex miss, the informative-context escalation (admit-only) applies:
        an entity already carried by a mined relation opportunity passes for free (its role
        in context is established), else `context_judge` decides on the redacted locator
        sentence. Both defaults off -> byte-identical to the pure regex lexicon."""
        if SEMANTIC_PROPERTY_PROBES_DISABLED:
            return []
        occurrences_by_decision: dict[str, list[Mapping]] = defaultdict(list)
        protected_terms = []
        for occurrence in environment_document.get("occurrences", []):
            protected_terms.extend(_occurrence_protected_terms(occurrence))
            decision_id = occurrence.get("decision_id")
            if decision_id is None or not occurrence.get("controlled", True):
                continue
            occurrences_by_decision[str(decision_id)].append(occurrence)
        relation_decision_ids: set[str] = set()
        if relation_opportunities:
            decision_by_occurrence = {
                str(row["occurrence_id"]): str(row["decision_id"])
                for row in environment_document.get("occurrences", [])
                if row.get("occurrence_id") is not None
                and row.get("decision_id") is not None
            }
            relation_decision_ids = {
                decision_by_occurrence[str(argument.get("occurrence_id"))]
                for opportunity in relation_opportunities
                for argument in opportunity.get("arguments") or []
                if str(argument.get("occurrence_id")) in decision_by_occurrence
            }

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

            semantic_label = _semantic_label(runtime_type)
            locator, role_cue, locator_rejection = _task_role_context_locator(
                document,
                occurrences,
                protected_terms=[*protected_terms, *properties],
                role_patterns=type_contract["role_patterns"],
                in_accepted_relation=decision_id in relation_decision_ids,
                context_judge=context_judge,
                judge_category={"loc": "location"}.get(semantic_label, semantic_label),
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
        *,
        reverse_framing_only: bool = False,
    ) -> tuple[list[dict], list[dict]]:
        return compile_relational_assertions(
            doc_id, document, environment_document, proposals,
            relation_contract=self.relation_contract,
            reverse_framing_only=reverse_framing_only,
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
        start, end = occurrence.get("start"), occurrence.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            continue
        relation_class = _RUNTIME_TYPE_CLASSES.get(canon(str(occurrence.get("runtime_type", ""))))
        if relation_class is None:
            continue
        if not any(relation_class in classes for contract in _RELATION_ARGUMENT_CLASSES.values()
                   for classes in contract):
            continue
        if not occurrence.get("controlled", True):
            continue
        decision = decisions.get(str(occurrence.get("decision_id")), {})
        properties = list(dict.fromkeys(
            canon(str(value))
            for action in decision.get("actions", [])
            if action.get("legal", True) and action.get("mode") not in {"keep", "placeholder"}
            for value in action.get("entails") or []
            if canon(str(value))
        ))
        if not properties:
            continue
        rows.append({
            "occurrence_id": str(occurrence["occurrence_id"]),
            "decision_id": str(occurrence.get("decision_id")),
            "surface": str(occurrence.get("surface", "")),
            "start": start,
            "end": end,
            "runtime_type": str(occurrence.get("runtime_type", "")),
            "relation_class": relation_class,
            "properties": properties,
        })
    rows.sort(key=lambda row: (row["start"], row["end"], row["occurrence_id"]))
    # One S-label per DECISION, not per occurrence: a term mentioned N times (e.g. "knee"
    # ×8) must not flood the inventory N times -- it distracts the teacher's draw and makes
    # it unstable to detector recall. Keep the earliest occurrence as the representative;
    # compile-time grounding remap resolves to the best sibling occurrence of the decision.
    deduped, seen = [], set()
    for row in rows:
        key = row["decision_id"] or f"occ:{row['occurrence_id']}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    for index, row in enumerate(deduped, start=1):
        row["span_label"] = f"S{index}"
    return deduped


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

# A clinical problem discussion routinely states the condition in the exam/HPI and its
# treatment/test in that problem's plan, with assessment, rationale, acknowledgment, and
# order clauses in between (measured: muscle-strain->meloxicam spans 7 clauses across the
# exam->plan transition). Twelve is the pre-registered soft diagnostic boundary: distant
# pairs remain reader-gated, while a PROBLEM SWITCH (below) still hard-stops the bridge.
_CROSS_CLAUSE_ANCHOR_MAX_DISTANCE = 12

# A PROBLEM SWITCH (moving to a different problem) blocks a cross-clause bridge; the
# exam->plan transition of the SAME problem ("assessment and plan", "for your first
# problem") does not -- clinical notes routinely state the condition in the exam/HPI and
# its treatment/test in that problem's plan.
_PROBLEM_SWITCH_BOUNDARY = re.compile(
    r"for (?:your|the|his|her)\s+(?:second|third|fourth|fifth|sixth|next|last|final)\b",
    re.IGNORECASE,
)


def _argument_clause_ranges(
    clauses: Sequence[tuple[int, int]], indices: Sequence[int | None],
) -> list[tuple[int, int]] | None:
    """Return the distinct source clauses containing relation arguments."""
    if any(index is None or index < 0 or index >= len(clauses) for index in indices):
        return None
    return [clauses[index] for index in sorted(set(indices)) if index is not None]


def _soft_cross_clause_cap_diagnostic(
    document: str, spans: Sequence[tuple[int, int]],
) -> dict | None:
    """Record, but never reject, a relation whose source arguments exceed the
    cross-clause locality operating point. The reader gate remains authoritative."""
    clauses = _source_clause_spans(document)
    if not clauses or not spans:
        return None
    indices = [
        next((index for index, (left, right) in enumerate(clauses)
              if left <= start < end <= right), None)
        for start, end in spans
    ]
    if any(index is None for index in indices):
        return None
    distance = max(indices) - min(indices)
    if distance <= _CROSS_CLAUSE_ANCHOR_MAX_DISTANCE:
        return None
    return {
        "kind": "soft_cross_clause_cap_exceeded",
        "clause_distance": distance,
        "soft_cap": _CROSS_CLAUSE_ANCHOR_MAX_DISTANCE,
    }


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
        "procedure_for": r"\b(?:treated\s+with|treat\w*|refer\w*|had|underwent|received|"
                         r"status\s+post|history\s+of)\b",
        "tests_for": r"\b(?:monitor\w*|check\w*|order\w*|follow\w*|evaluat\w*|assess\w*|"
                     r"panel|show\w*|reveal\w*|demonstrat\w*|notic\w*|found|came\s+back\w*|"
                     r"consistent\s+with|positive\s+for)\b",
        "contraindicated_because_of": r"\bcontraindicat\w*\b",
        "causes_or_explains": r"\b(?:causes?|explains?)\b",
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


CLINICAL_RELATIONS = (
    "prescribed_with", "procedure_for", "tests_for",
    "contraindicated_because_of", "causes_or_explains",
)


def relation_teacher_response_format(
    environment_document: Mapping, document: str, *,
    allowed_labels: "Sequence[str] | None" = None,
) -> dict:
    """Bind strict wire fields to displayed labels, never internal IDs or answers.

    `allowed_labels` scopes the schema to a SUBSET of the inventory (the repair pass shows only the
    target regions' labels): the span_label enum and the candidate_accounting ledger then cover exactly
    those labels, so the teacher can neither emit a relation over a label it was not shown nor be forced
    to account for one. Default (None) = full inventory (the primary teacher, which sees the whole note)."""
    response_format = deepcopy(RELATION_TEACHER_RESPONSE_FORMAT)
    schema = response_format["json_schema"]["schema"]
    inventory = relation_teacher_span_inventory(environment_document)
    if allowed_labels is not None:
        allowed = set(allowed_labels)
        inventory = [row for row in inventory if row["span_label"] in allowed]
    relations = list(CLINICAL_RELATIONS)
    labels = [row["span_label"] for row in inventory]
    support_properties = sorted({property_level for row in inventory for property_level in row["properties"]})
    for section in ("span_relations", "context_relations"):
        schema["properties"][section]["items"]["properties"]["relation"]["enum"] = relations
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


def _singularize(token: str) -> str:
    """Cheap English plural fold so answer/alias token matching survives inflection
    ("kidney stones" alias vs "kidney stone" answer). Applied to BOTH sides, so it need
    not be linguistically perfect — only consistent."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _stemmed_tokens(value: str) -> set[str]:
    return {_singularize(token) for token in _meaningful_tokens(value)}


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


def _answer_leak_tokens(
    question: str,
    answer: str,
    runtime_type: str | Sequence[str],
    *,
    extra_exempt_tokens: Sequence[str] = (),
) -> set[str]:
    """The discriminative answer tokens that also appear in the question (the actual leak)."""
    types = [runtime_type] if isinstance(runtime_type, str) else list(runtime_type)
    exempt = set(extra_exempt_tokens)
    for type_value in types:
        exempt |= _placeholder_meaning_tokens(type_value)
    answer_tokens = _meaningful_tokens(answer) - exempt
    return answer_tokens & _meaningful_tokens(question)


def _question_leaks_answer(
    question: str,
    answer: str,
    runtime_type: str | Sequence[str],
    *,
    extra_exempt_tokens: Sequence[str] = (),
) -> bool:
    return bool(_answer_leak_tokens(
        question, answer, runtime_type, extra_exempt_tokens=extra_exempt_tokens))


def _ordered_decision_levels(decision: Mapping) -> list[str]:
    """Legal non-placeholder generalization levels of a decision, most-specific
    to coarsest (same flattening/order as the teacher span inventory)."""
    return list(dict.fromkeys(
        str(value)
        for action in decision.get("actions", [])
        if action.get("legal", True) and action.get("mode") not in {"keep", "placeholder"}
        for value in action.get("entails") or []
        if str(value).strip()
    ))


def _repair_leaked_relation(
    question: str,
    accepted_values: Sequence[str],
    arguments: Sequence[Mapping],
    answer_role: str,
    answer_exempt_types: Sequence[str] | str,
    decisions: Mapping[str, Mapping],
    occurrences: Mapping[str, Mapping],
) -> tuple[str, list[str], list[dict], dict] | None:
    """Clear a lexical answer-leakage WITHOUT any generic-word whitelist and WITHOUT lowering the
    answer floor. `answer_leakage` fires when the question shares a meaningful token with the
    accepted answer. The answer is taken CANONICALLY from the selected answer argument (its
    support_property if linked, else its literal) -- never from teacher prose.

    Repair, composed per candidate locator level (current, then coarser legal levels):
      * recolor the LOCATOR (non-answer argument) reference in the question to that level, then
      * strip every answer-overlapping token that lies OUTSIDE the (recolored) locator span,
        preserving the locator span verbatim -- no medication/procedure/diagnostic exemptions,
      * normalize whitespace/articles/punctuation.
    Accept the first candidate whose result no longer leaks and is still a well-formed question.
    The answer level is never coarsened (the measured floor is preserved), so floor_lowered is
    always False. Returns (question, accepted_values, arguments, repair) or None (caller rejects
    honestly, then the three-point gate is the backstop for vague/non-context-dependent wording)."""
    object_index = 0 if answer_role == "subject" else 1
    args = [dict(argument) for argument in arguments]
    answer_arg = args[object_index]
    other_arg = args[1 - object_index]

    # canonical answer strings: the answer argument's own level (linked) or literal (context)
    if answer_arg.get("kind") == "linked" and answer_arg.get("support_property"):
        answer_values = [str(answer_arg["support_property"])]
    elif answer_arg.get("literal"):
        answer_values = [str(answer_arg["literal"])]
    else:
        answer_values = list(accepted_values)
    answer_tokens: set[str] = set()
    for value in answer_values:
        answer_tokens |= _meaningful_tokens(value)
    if not answer_tokens:
        return None

    def leaks(text: str) -> bool:  # strict: no generic-word exemption
        return bool(answer_tokens & _meaningful_tokens(text))

    def strip_outside(text: str, protect: tuple[int, int] | None) -> str:
        # remove every answer-token occurrence not inside the protected [start, end) span
        out, cursor = [], 0
        for match in re.finditer(r"\w+", text):
            token = match.group(0)
            keep = True
            if _meaningful_tokens(token) & answer_tokens:
                if not (protect and match.start() < protect[1] and protect[0] < match.end()):
                    keep = False
            if keep:
                out.append(text[cursor:match.end()])
            else:
                out.append(text[cursor:match.start()])  # drop the token, keep separators
            cursor = match.end()
        out.append(text[cursor:])
        joined = "".join(out)
        joined = re.sub(r"\b([Aa]n?|[Tt]he)\b(?=\s*(?:\?|$|\s\b(?:was|were|is|are|for)\b))", "",
                        joined)  # dangling article left by a stripped noun
        joined = re.sub(r"\s+", " ", joined)
        joined = re.sub(r"\s+([?.,])", r"\1", joined).strip()
        return joined

    def levels(argument: Mapping) -> list[str]:
        if argument.get("kind") != "linked":
            return []
        occurrence = occurrences.get(argument.get("occurrence_id")) or {}
        return _ordered_decision_levels(decisions.get(str(occurrence.get("decision_id"))) or {})

    current = _normalize_teacher_text(str(other_arg.get("support_property") or ""))
    candidate_locators = [current] + [
        level for level in levels(other_arg) if canon(level) != canon(current)
    ]
    for locator in candidate_locators:
        recolored = question
        if current and locator != current:
            recolored = re.compile(re.escape(current), re.IGNORECASE).sub(locator, recolored)
        span = re.search(re.escape(locator), recolored, re.IGNORECASE) if locator else None
        protect = (span.start(), span.end()) if span else None
        repaired = strip_outside(recolored, protect)
        # well-formed: non-empty, still an interrogative, still references the locator
        well_formed = (
            repaired
            and repaired.endswith("?")
            and len(_meaningful_tokens(repaired)) >= 3
            and (not locator or locator.lower() in repaired.lower())
        )
        if repaired != question and well_formed and not leaks(repaired):
            if locator != current:
                other_arg["support_property"] = locator
            repair = {
                "kind": "strict_answer_token_strip",
                "argument_role": other_arg.get("role"),
                "locator_from": current, "locator_to": locator,
                "floor_lowered": False,
            }
            return repaired, list(accepted_values), args, repair
    return None


def _question_leaks_protected_term(
    question: str,
    protected_terms: Sequence[str],
    allowed_tokens_by_term: Mapping[str, frozenset[str]] | None = None,
) -> bool:
    """Token overlap leaks only on a term's DISCRIMINATIVE tokens -- those not in its authorized
    publishable set (`allowed_tokens_by_term`: its declared legal generalization levels + placeholder
    labels, minus any level identical to the raw surface, filtered by the caller), which the spec
    directs questions and answers to use verbatim. Full-term containment leaks too, EXCEPT a single
    fully-authorized token: a detected drug SURFACE "medication" (a placeholder word) or a condition
    surface "diabetes" whose token rides its own published coarser level "diabetes mellitus" is not an
    identity locator. A short alias ("Li","AF" -> empty meaningful set) and any multi-token surface are
    never exempt, so authorized tokens cannot recompose into a sensitive identity. The caller must drop
    a level equal to the surface from `allowed` so a raw brand ("Synthroid") whose only authorization
    would be a surface-echoing level stays discriminative and is rejected."""
    allowed = allowed_tokens_by_term or {}
    question_tokens = _meaningful_tokens(question)
    for term in protected_terms:
        if not term:
            continue
        term_tokens = _meaningful_tokens(term)
        discriminative = term_tokens - allowed.get(term, frozenset())
        single_authorized = len(term_tokens) == 1 and not discriminative
        if _contains(question, term) and not single_authorized:
            return True
        if discriminative & question_tokens:
            return True
    return False


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


def _redacted_locator(
    sentence: str,
    target_terms: set[str],
    protected_terms: Sequence[str],
) -> str:
    def redact(text: str, terms: Sequence[str], marker: str) -> str:
        for term in sorted(
            set(terms), key=lambda value: (-len(value), canon(value), value)
        ):
            if not term:
                continue
            text = re.sub(
                rf"(?<!\w){re.escape(term)}(?!\w)",
                marker,
                text,
                flags=re.IGNORECASE,
            )
        return text

    redacted = redact(sentence, list(target_terms), "[target item]")
    redacted = redact(redacted, [
        term for term in protected_terms
        if canon(term) not in {canon(target) for target in target_terms}
    ], "[protected item]")
    return re.sub(r"\s+", " ", redacted).strip()


def _task_role_context_locator(
    document: str,
    occurrences: Sequence[Mapping],
    *,
    protected_terms: Sequence[str],
    role_patterns: Sequence[tuple[str, str]],
    in_accepted_relation: bool = False,
    context_judge: "Callable[..., bool] | None" = None,
    judge_category: str | None = None,
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
    surviving_sentences: list[str] = []
    protected_tokens = {
        token for term in protected_terms for token in _meaningful_tokens(term)
    }
    for position, surface in sorted(set(targets)):
        sentence = _sentence_at(document, position)
        sentence_tokens = _meaningful_tokens(sentence)
        if not sentence_tokens - protected_tokens:
            continue
        if sentence not in surviving_sentences:
            surviving_sentences.append(sentence)
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
    target_terms = {
        str(occurrence.get("surface", "")).strip()
        for occurrence in occurrences
        if str(occurrence.get("surface", "")).strip()
    }
    if locator is None and surviving_sentences:
        # Cue-miss escalation (admit-only; the reader three-point gate remains the real
        # acceptance check, so the cue-matched set can never regress):
        if in_accepted_relation:
            # relation evidence already established this entity's clinical role in context
            locator, matched_cue = surviving_sentences[0], "relation_evidence"
        elif context_judge is not None:
            # ponytail: judge at most 3 candidate sentences per decision; raise if recall needs it
            for sentence in surviving_sentences[:3]:
                if context_judge(
                    locator=_redacted_locator(sentence, target_terms, protected_terms),
                    category=judge_category or "clinical item",
                ):
                    locator, matched_cue = sentence, "semantic_judge"
                    break
            else:
                return None, None, "uninformative_context_judged"
    if locator is None:
        return (
            None,
            None,
            "no_task_role_cue" if surviving_sentences else "no_safe_contextual_locator",
        )
    return _redacted_locator(locator, target_terms, protected_terms), matched_cue, ""


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
    if relation == "procedure_for":
        return any(
            canon(str(row["condition"])) == first
            and canon(str(row["treatment"])) == second
            for row in parsed_reference.get("plan_rows", [])
        )
    if relation == "tests_for":
        return any(
            canon(str(row["condition"])) == first
            and canon(str(row["test"])) == second
            for row in parsed_reference.get("plan_rows", [])
        )
    return False


# ASCII hyphen plus the unicode hyphen/dash/minus variants LLM teachers emit (e.g. gpt-oss
# writes "x‑ray" U+2011 for source ASCII "x-ray"). Equated in literal AND evidence-quote
# grounding. Dash folding is 1-char->1-char, so it preserves string length and offsets.
_LITERAL_DASH_CHARS = "-‐‑‒–—−"
_DASH_FOLD = str.maketrans({ch: "-" for ch in _LITERAL_DASH_CHARS})


def _exact_substring_starts(document: str, quote: str) -> list[int]:
    """Start offsets where `quote` occurs in `document`, equating unicode dash variants with ASCII
    '-' (teachers emit U+2011 etc. for source '-'). Folding is length-preserving, so the returned
    starts and `len(quote)` still index the ORIGINAL document -- callers relying on
    `start + len(quote)` (evidence spans) stay correct. Otherwise byte-exact (case + whitespace)."""
    document = document.translate(_DASH_FOLD)
    quote = quote.translate(_DASH_FOLD)
    starts = []
    cursor = 0
    while quote:
        start = document.find(quote, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + 1
    return starts


def _source_literal_spans(document: str, literal: str) -> list[tuple[int, int]]:
    """Source `(start, end)` spans matching `literal` under case-fold + whitespace-collapse +
    dash-variant equivalence at token boundaries. Offsets index the ORIGINAL document, so the
    matched text is exactly ``document[start:end]`` -- a context literal resolved through this and
    then written back as its source substring still satisfies the exact-source grounding contract
    (`_argument_is_grounded`). Only orthographic casing, inter-token whitespace, and unicode dash
    variants are equated; a different token sequence, synonym, abbreviation, or reordering never
    matches.

    ponytail: `re.IGNORECASE` (not full Unicode case-fold) and ASCII `\\w` boundaries suit the
    ASCII clinical corpus; move to str.casefold + an explicit offset map only if non-ASCII
    literals appear."""
    tokens = str(literal).split()
    if not tokens:
        return []

    def _escape_token(token: str) -> str:
        # per-char escape so any dash variant in the literal matches any dash variant in the
        # source; every other char keeps exact (case-insensitive) escaping.
        return "".join(
            "[" + _LITERAL_DASH_CHARS + "]" if ch in _LITERAL_DASH_CHARS else re.escape(ch)
            for ch in token
        )
    escaped = [_escape_token(token) for token in tokens]
    # Plural fold on the FINAL token only, matched-extent-inclusive and symmetric: strip one trailing
    # "s" to a stem, then allow an optional "s". So a singular literal ("x-ray", "aldosterone level")
    # matches a plural source ("x-rays", "levels") and vice versa, and the matched span still indexes
    # the real source substring (grounding stays exact). ponytail: single trailing "s" only -- covers
    # the observed x-ray/level/lab cases; "es"/"ies" plurals are rare here and left out.
    last = tokens[-1]
    stem = last[:-1] if len(last) > 1 and last[-1].lower() == "s" else last
    escaped[-1] = _escape_token(stem) + "s?"
    core = r"\s+".join(escaped)
    # (?<!\w)/(?!\w) are token-boundary guards that, unlike \b, still match when the literal's
    # first/last char is non-word ("(MRI)", "C++") -- \b there fails to find an exact source
    # occurrence, a grounding regression from the prior exact-substring lookup.
    pattern = r"(?<!\w)" + core + r"(?!\w)"
    return [(match.start(), match.end())
            for match in re.finditer(pattern, document, re.IGNORECASE)]


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


CLINICAL_RELATION_INVENTORY = """prescribed_with: condition or diagnosis -> drug; explicit prescription/use only, never a procedure.
procedure_for: condition or diagnosis -> medical procedure; a therapeutic procedure or referral for the condition, including a past treatment the patient already had ("had the <procedure> for <condition>", a prior transplant/surgery in the history). Never a drug.
tests_for: condition or diagnosis -> diagnostic or monitoring test/study; a test that discovers or monitors the condition — ordered to follow it, or whose result showed it. A test ordered or resulted while the doctor is assessing/planning one problem is linked to that problem's condition, even if the order sentence does not repeat the condition name; reject only a test from a different problem block or unrelated small talk. Use this (not procedure_for) for any lab, panel, imaging, or exam.
contraindicated_because_of: drug or procedure -> condition or diagnosis; require explicit contraindication.
causes_or_explains: condition or diagnosis -> condition or symptom; require explicit causation/explanation."""

CLINICAL_WORKED_EXAMPLES = """Drug (prescribed_with, never a procedure): source "for the [S1: migraine | condition | levels: neurological disorder] ... prescribe [S2: sumatriptan | drug | levels: triptan]" => prescribed_with(S1, S2).
Safe question: "Which medication was prescribed for the neurological disorder?" Accepted answer: "triptan". Refer to linked spans by their levels, never the source words; ask for the answered argument with a generic answer-type word (never "category"/"type"/"class"/"kind"). WRONG (leaks the answer): "Which triptan was prescribed …" — never put the answer's level in the question; ask "Which medication …" instead.
Treatment procedure (procedure_for): "cataract treated with phacoemulsification" => procedure_for (a procedure, not a drug). Past treatment also counts: source "had the [S3: appendectomy | procedure | levels: abdominal surgery] years ago for [S4: appendicitis | condition | levels: abdominal inflammation]" => procedure_for(S4, S3). Safe question: "What procedure was performed for the abdominal inflammation?" Accepted answer: "abdominal surgery".
Test (tests_for), monitoring OR discovery: source "to follow the [S5: diabetes | condition | levels: metabolic disorder], order hemoglobin A1c" => tests_for(S5, context literal "hemoglobin A1c"). QA orientation is literal -> linked span (ask for the condition, not the test). Safe question: "For what medical condition was hemoglobin A1c ordered?" Accepted answer: "metabolic disorder". A discovered finding counts too: "the ct shows a [S6: kidney stone | condition | levels: urinary tract stone]" => tests_for(S6, context literal "ct"). Use tests_for for any lab/panel/imaging/exam; a test merely mentioned elsewhere is not linked.
Contraindication: source "you ca n't use beta-blockers because of your [S7: asthma | condition | levels: reactive airway disease]" => contraindicated_because_of(context literal "beta-blockers", S7). Use the condition S-label at the sentence that states the contraindication, not an earlier history-list mention of the same condition. The drug argument may be a drug-class phrase quoted as a context literal (e.g. "anti-inflammatory medications", "certain medications"), not only a specific drug name — emit the contraindication whenever the source says a drug or drug class cannot be used because of the condition.
Safe question: "What medical condition makes beta-blockers unsuitable?" Set answer_role to object. Accepted answer: "reactive airway disease" (the condition's level, never the drug and never the source words).
Conditional plan (procedure_for / tests_for; allowed with a CONDITIONAL question): source "if symptoms persist , possibly refer to [S8: cardiac rehabilitation | procedure | levels: rehabilitation program]" while planning the [S9: heart failure | condition | levels: cardiac disorder] => procedure_for(S9, S8). Phrase it conditionally: "What procedure MAY the patient be referred to for the cardiac disorder?" Accepted answer: "rehabilitation program". A definite question ("What procedure was performed?") for a conditional plan is rejected."""


def _relation_evidence_cards(
    document: str, environment_document: Mapping, shown: Sequence[Mapping],
) -> list[dict]:
    """Per-clause evidence cards: each source clause holding >=1 shown S-label, with the labels
    tagged in it. A value mentioned in several clauses carries its ONE S-label in every clause its
    occurrences fall in. Shared by the primary teacher prompt and the gleaning repair prompt."""
    label_by_decision = {str(row["decision_id"]): row["span_label"] for row in shown}
    occ_spans = sorted(
        (int(occ["start"]), int(occ["end"]), label_by_decision[str(occ.get("decision_id"))])
        for occ in environment_document.get("occurrences", [])
        if isinstance(occ.get("start"), int) and isinstance(occ.get("end"), int)
        and str(occ.get("decision_id")) in label_by_decision
    )
    cards = []
    for index, (start, end) in enumerate(_source_clause_spans(document), start=1):
        labels = list(dict.fromkeys(
            label for (s, e, label) in occ_spans if start <= s < e <= end))
        if labels:
            cards.append({"index": index, "start": start, "end": end, "labels": labels,
                          "text": document[start:end].strip()})
    return cards


def relation_teacher_prompt(
    doc_id: str,
    document: str,
    environment_document: Mapping,
) -> str:
    """Build the compact human-facing relation-teacher prompt for the clinical
    relations over the controlled condition/drug/procedure spans."""
    del doc_id
    inventory = relation_teacher_span_inventory(environment_document)
    shown = inventory
    relation_inventory = CLINICAL_RELATION_INVENTORY
    worked_examples = CLINICAL_WORKED_EXAMPLES
    def _span_line(row: Mapping) -> str:
        return (f"[{row['span_label']}: {row['surface']} | {_prompt_display_classes(row)} "
                f"| levels: {'; '.join(row['properties'])}]")
    spans = "\n".join(_span_line(row) for row in shown) or "(No eligible controlled spans.)"
    # Evidence cards annotate each source clause with the S-labels appearing in it. The
    # inventory carries ONE label per decision, but a value may be mentioned in several
    # clauses; tag that label in EVERY clause any of its occurrences fall in, so a relation
    # stated at a non-earliest mention (e.g. arthritis named in the plan sentence, not just
    # the history list) still co-locates with its partner in the card.
    cards = [
        f"E{card['index']}: {card['text']}\nLabels: {', '.join(card['labels'])}"
        for card in _relation_evidence_cards(document, environment_document, shown)
    ]
    return f"""TASK
Find as many explicit, source-grounded, non-duplicate relations as the cap ({RELATION_TEACHER_MAX_RELATIONS}) permits. Prefer diversity only when supported. Abstain rather than inventing a fact.

HOW TO INSPECT THE SOURCE
Read the full source. Evidence cards are navigation aids only, not pair gates. Use S-labels for linked controlled arguments. A value mentioned several times has ONE S-label; the evidence cards show every clause it appears in, so use that single label for the relation and the compiler grounds it at the mention inside the sentence that states the relation. For an uncontrolled argument, quote its exact source text as a context literal.
A relation may connect spans from different turns of the SAME problem discussion, the block where the doctor assesses and plans one problem, including short patient acknowledgments between the doctor's sentences (for example a condition named when the problem is introduced and a test ordered for it a sentence later). Never link spans from a different problem discussion or from unrelated small talk. A conditional or planned statement ("possibly referral to physical therapy", "if symptoms continue we will order X") IS a valid relation, but you MUST phrase its question conditionally (may / might / would / if …) so you never assert it as already done — a conditional plan paired with a definite question is rejected.
"Explicit" means the source states the relationship through this problem discussion; the subject and object need NOT appear in the same sentence. A drug or test named while the doctor is assessing/planning one problem is linked to that problem's condition even if its own sentence does not repeat the condition name.

PRIVACY-SAFE QA
Author the question, accepted answers, and scoring contract. Do not repeat a displayed controlled source span or alias in a question or accepted answer; use its listed generalization level. For a linked argument, accepted answers come from its listed levels, never its source text. When a label lists several levels, use the most specific one that still conveys the relation, not the broadest (a level so generic it fits almost any concept measures nothing).
CRITICAL — the QUESTION must never contain the accepted answer's words or level. Ask with a GENERIC answer-type word only for the argument being asked ("Which medication …", "Which procedure …", "What test …", "What condition …") — never "category", "type", "class", or "kind", since the original render may hold a named entity rather than an explicit category — then put the specific level in the accepted answer. Writing the answer's level into the question ("Which triptan was prescribed …" when the answer is "triptan") leaks it, and the relation is rejected. For two linked spans, ask for the object span as usual. For exactly one linked span plus one uncontrolled context literal, the QA orientation is ALWAYS literal -> linked span: use the literal verbatim as the locator in the question, make the linked span's listed level the accepted answer, and set answer_role to that linked argument's subject/object role. This changes QA orientation only, never the directional relation fact. Never use PERSON as the locator.

RELATION INVENTORY
{relation_inventory}

WORKED EXAMPLES (illustrative patterns using unrelated conditions; do not copy these entities — read this document's own source and spans)
{worked_examples}

DETECTED SPANS
{spans}

EVIDENCE CARDS
{chr(10).join(cards) or '(No span-local cards; inspect the source directly.)'}

RESPONSE
Return two relation lists. Each relation record contains: relation; a subject argument then an object argument; a question; answer_role (subject or object); accepted answers; the fixed scoring contract.
span_relations: relations whose subject and object are both displayed spans. Each argument is kind linked, with span_label set to its S-label and support_property set to exactly one of that label's listed levels, copied verbatim. Subject and object MUST be two DIFFERENT S-labels — never the same label twice.
context_relations: relations pairing exactly one linked S-label argument with one uncontrolled argument of kind context, whose literal is exact source text that is not any displayed span. Put the two arguments in the relation's DIRECTIONAL order (subject then object), which is NOT always the linked one first: for tests_for the linked condition is the subject and the test literal is the object, but for contraindicated_because_of the drug/drug-class literal is the SUBJECT and the linked condition is the OBJECT. Match the direction in the relation inventory, not the order the spans appear in text. Then set answer_role to the linked argument's role: the literal is the question locator and the linked span's support_property is the accepted answer.
Never quote a displayed span as a context literal. Emit each distinct fact EXACTLY ONCE, in the single list its argument kinds require — never emit both a span-pair and a context version of the same fact, and never emit the same fact under a different S-label of the same value. A fact whose second argument is an uncontrolled literal (a test name, a drug class) belongs ONLY in context_relations; do not force it into span_relations by reusing the linked label as both arguments.
Example span_relations record (illustrative, unrelated entities): relation prescribed_with; subject linked S1 with one listed S1 level as support_property; object linked S2 with one listed S2 level; question "Which medication was prescribed for the neurological disorder?"; accepted answer "triptan".
Example context_relations record (illustrative, unrelated entities): relation prescribed_with; subject linked S3 with one listed S3 level as support_property; object context literal "azithromycin" quoted from the relation sentence; answer_role subject; question "For what medical condition was azithromycin prescribed?"; accepted answer the listed S3 condition level.
Return exactly one candidate_accounting row per S-label covering both lists, with a short reason for every row. emitted means a relation record in either list uses the label; duplicate_mention means another S-label of the same value already carries the fact (name that label in the reason); exhausted_no_relation means no explicit supported relation; unsupported means insufficient source role/connection. Reasons must reference labels and levels only and never repeat displayed span text. Return only the structured response.
FINAL CHECK before you return: reconcile the accounting against the two relation lists so they agree exactly. Every S-label you mark emitted MUST appear in an actual relation record in span_relations or context_relations, and every S-label used by a relation record MUST be marked emitted. Never mark a label emitted without including its relation record, and include every supported relation you identified — a fact named in the accounting but absent from the lists, or a relation you found but omitted, is an error to fix before returning.

SOURCE DOCUMENT
{document}"""


def relation_repair_prompt(
    doc_id: str,
    document: str,
    environment_document: Mapping,
    targets: Sequence[Mapping],
    *,
    shown_labels_out: "set[str] | None" = None,
) -> str:
    """Second-pass gleaning+repair prompt: same privacy/response contract as the primary, but the
    DETECTED SPANS and EVIDENCE CARDS are restricted to the clauses relevant to `targets`; a FIX
    GUIDE states each distinct per-reason hint once, and the REPAIR TARGETS lines reference it by
    tag. Kept relations and 100%-legitimate rejections are not shown. The primary prompt is
    untouched (cache-safe)."""
    del doc_id
    inventory = relation_teacher_span_inventory(environment_document)
    label_by_decision = {str(row["decision_id"]): row["span_label"] for row in inventory}
    occ_by_id = {
        str(row["occurrence_id"]): row
        for row in environment_document.get("occurrences", [])
        if row.get("occurrence_id") is not None
    }

    def _arg_label(argument: Mapping) -> str | None:
        if argument.get("kind") == "linked":
            occurrence = occ_by_id.get(str(argument.get("occurrence_id"))) or {}
            return label_by_decision.get(str(occurrence.get("decision_id")))
        return None

    def _arg_span(argument: Mapping) -> tuple[int, int] | None:
        if argument.get("kind") == "linked":
            occurrence = occ_by_id.get(str(argument.get("occurrence_id"))) or {}
            if isinstance(occurrence.get("start"), int):
                return int(occurrence["start"]), int(occurrence["end"])
        elif isinstance(argument.get("start"), int):
            return int(argument["start"]), int(argument["end"])
        return None

    occurrence_spans_by_decision: dict[str, list[tuple[int, int]]] = {}
    for occurrence in occ_by_id.values():
        start, end = occurrence.get("start"), occurrence.get("end")
        if isinstance(start, int) and isinstance(end, int):
            occurrence_spans_by_decision.setdefault(
                str(occurrence.get("decision_id")), []).append((int(start), int(end)))

    clauses = _source_clause_spans(document)

    def _target_clause_indices(target: Mapping) -> list[int]:
        # ALL occurrence clauses of each argument (every mention of a linked arg's DECISION, every
        # occurrence of a context literal) -- NOT a single representative occurrence, and NOT the
        # evidence-span envelope. The relation's supporting evidence often sits at a DIFFERENT mention
        # than the grounded one (HPI "taking ibuprofen for the pain" vs a plan-section question), so
        # the region must carry every mention; the irrelevant middle is elided (mirrors the judge premise).
        ranges: list[tuple[int, int]] = []
        for argument in target.get("arguments") or []:
            if argument.get("kind") == "linked":
                occurrence = occ_by_id.get(str(argument.get("occurrence_id"))) or {}
                spans = occurrence_spans_by_decision.get(str(occurrence.get("decision_id")))
                if spans:
                    ranges.extend(spans)
                else:
                    span = _arg_span(argument)
                    if span:
                        ranges.append(span)
            else:
                if argument.get("literal"):
                    ranges.extend(_source_literal_spans(document, str(argument["literal"])))
                span = _arg_span(argument)
                if span:
                    ranges.append(span)
        return [
            index for index, (lo, hi) in enumerate(clauses)
            if document[lo:hi].strip()
            and any(not (hi <= r0 or r1 <= lo) for r0, r1 in ranges)
        ]

    target_clause_indices = [_target_clause_indices(target) for target in targets]
    rendered_clause_spans = [
        clauses[index] for index in sorted({i for indices in target_clause_indices for i in indices})
    ]

    def _label_in_rendered_region(row: Mapping) -> bool:
        # DETECTED SPANS lists a label ONLY if one of its occurrences falls in a RENDERED clause, so
        # the teacher never sees a label whose source text was elided from the region (#4). This
        # replaces the prior card-based filter, where a kept card leaked its co-occurring labels even
        # when their clauses were not shown.
        for start, end in occurrence_spans_by_decision.get(str(row["decision_id"]), []):
            if any(not (hi <= start or end <= lo) for lo, hi in rendered_clause_spans):
                return True
        return False

    shown = [row for row in inventory if _label_in_rendered_region(row)]
    # Expose the shown labels so the repair RESPONSE SCHEMA can be scoped to them (the teacher must
    # not be allowed to emit, or be required to account for, a label absent from its rendered region).
    if shown_labels_out is not None:
        shown_labels_out.update(row["span_label"] for row in shown)

    def _span_line(row: Mapping) -> str:
        return (f"[{row['span_label']}: {row['surface']} | {_prompt_display_classes(row)} "
                f"| levels: {'; '.join(row['properties'])}]")

    spans = "\n".join(_span_line(row) for row in shown) or "(No eligible controlled spans.)"

    def _arg_desc(argument: Mapping | None) -> str:
        if argument is None:
            return "?"
        if argument.get("kind") == "linked":
            label = _arg_label(argument)
            occurrence = occ_by_id.get(str(argument.get("occurrence_id"))) or {}
            surface = str(occurrence.get("surface") or argument.get("surface") or "").strip()
            # label + surface (e.g. "S12 (blood sugar)") so the teacher can match the label to its
            # mention in the cited clauses; the surface is already shown in DETECTED SPANS/SOURCE CLAUSES.
            if label and surface:
                return f"{label} ({surface})"
            return label or "?"
        return f'context literal "{argument.get("literal", "")}"'

    # GROUP BY SOURCE REGION: cluster targets that share any source clause (a problem discussion),
    # then show each region's source text ONCE followed by its targets. A clause shared by K targets
    # in a region is inlined once, not K times; and there is no doc-wide clause index to cross-map --
    # the S-label surfaces (e.g. "S13 (vitamin d deficiency)") locate each argument within the region.
    # `clauses` / `_target_clause_indices` / `target_clause_indices` are computed above (also feed the
    # rendered-region-aligned DETECTED SPANS).
    parent = list(range(len(targets)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    clause_owner: dict[int, int] = {}
    for ti, indices in enumerate(target_clause_indices):
        for ci in indices:
            if ci in clause_owner:
                parent[_find(ti)] = _find(clause_owner[ci])
            else:
                clause_owner[ci] = ti
    clusters: dict[int, list[int]] = {}
    for ti in range(len(targets)):
        clusters.setdefault(_find(ti), []).append(ti)

    def _cluster_first_clause(members: list[int]) -> int:
        return min((ci for ti in members for ci in target_clause_indices[ti]), default=10**9)

    # FIX GUIDE: each distinct fix hint is written ONCE, keyed by the target's rejection reason
    # (fixable) or kind (missed/ambiguous); a target line carries only its key. Batched targets
    # share a handful of hints, so repeating the multi-sentence hint per target was pure prompt
    # weight with no information.
    def _fix_key(target: Mapping) -> str:
        return str(target.get("reason") or str(target.get("kind", "")).upper())

    fix_guide_entries: dict[str, str] = {}
    for target in targets:
        fix_guide_entries.setdefault(_fix_key(target), str(target.get("hint", "")))
    fix_guide = "\n".join(
        f"- {key}: {hint}" for key, hint in fix_guide_entries.items()) or "(No targets.)"

    def _target_line(number: int, target: Mapping) -> str:
        arguments = target.get("arguments") or []
        subject = next((a for a in arguments if a.get("role") == "subject"), None)
        obj = next((a for a in arguments if a.get("role") == "object"), None)
        tag = str(target.get("kind", "")).upper()
        key = _fix_key(target)
        fix = f" · fix: {key}" if key != tag else ""
        return (
            f"  [{number}] {tag}{fix} · {target.get('relation', '?')} · "
            f"{_arg_desc(subject)} → {_arg_desc(obj)}")

    blocks: list[str] = []
    number = 0
    for members in sorted(clusters.values(), key=_cluster_first_clause):
        region_clauses = sorted({ci for ti in members for ci in target_clause_indices[ti]})
        source = " […] ".join(
            document[clauses[ci][0]:clauses[ci][1]].strip() for ci in region_clauses
        ) or "(no clause located; inspect the source directly)"
        lines = []
        for ti in sorted(members, key=lambda t: (target_clause_indices[t] or [10**9], t)):
            number += 1
            lines.append(_target_line(number, targets[ti]))
        blocks.append(f'SOURCE: "{source}"\n' + "\n".join(lines))
    repair_targets = "\n\n".join(blocks) or "(No targets.)"

    relation_inventory = CLINICAL_RELATION_INVENTORY
    worked_examples = CLINICAL_WORKED_EXAMPLES
    return f"""TASK
REPAIR + GLEAN pass over one clinical note. A first pass already ran. Below are ONLY the relations to fix or recover — kept relations and legitimately-rejected ones are omitted. Address EACH numbered target: fix the noted problem (rewrite/re-anchor/disambiguate) or emit the missed relation if the source supports it; abstain on any target you cannot support from the source. Do NOT re-emit relations that are not listed. Same output schema and privacy rules as the first pass.

HOW TO INSPECT THE SOURCE
REPAIR TARGETS are grouped by source region: each SOURCE line is the source text for the targets listed under it — judge those targets only from that region. Arguments are named by S-label with their surface in parentheses (e.g. "S13 (vitamin d deficiency)") so you can locate each one in the SOURCE text; DETECTED SPANS lists the level choices for those labels. Use S-labels for linked controlled arguments (one label per value); quote an uncontrolled argument's exact source text as a context literal. A relation may connect spans across turns of the SAME problem discussion (short patient acknowledgments between the doctor's sentences are fine), never across different problems or small talk. A conditional/planned statement IS a valid relation but its question MUST be phrased conditionally (may / might / would / if …).

PRIVACY-SAFE QA
Author the question, accepted answers, and scoring contract. Do not repeat a displayed controlled source span or alias in a question or accepted answer; use its listed generalization level. For a linked argument, accepted answers come from its listed levels, never its source text. When a label lists several levels, use the most specific one that still conveys the relation, not the broadest.
CRITICAL — the QUESTION must never contain the accepted answer's words or level. Ask with a GENERIC answer-type word only for the argument being asked ("Which medication …", "Which procedure …", "What test …", "What condition …") — never "category", "type", "class", or "kind" — then put the specific level in the accepted answer. For two linked spans, ask for the object span. For exactly one linked span plus one uncontrolled context literal, use the literal verbatim as the question locator, make the linked span's listed level the accepted answer, and set answer_role to the linked argument's subject/object role. This changes QA orientation only, never the directional relation fact. Never use PERSON as the locator.

RELATION INVENTORY
{relation_inventory}

WORKED EXAMPLES (illustrative patterns using unrelated conditions; do not copy these entities)
{worked_examples}

DETECTED SPANS (level choices for the labels named in the targets)
{spans}

FIX GUIDE (one entry per tag; apply it to EVERY target labeled with that tag)
{fix_guide}

REPAIR TARGETS (grouped by source region; address only these — each is tagged ambiguous / fixable / missed, and names its FIX GUIDE entry)
{repair_targets}

RESPONSE
Return two relation lists with the same records as the first pass: relation; a subject then an object argument; a question; answer_role; accepted answers; the fixed scoring contract.
span_relations: subject and object both displayed spans, each kind linked with span_label + one listed level as support_property; the two labels MUST differ.
context_relations: exactly one linked S-label argument and one uncontrolled context argument whose literal is exact source text (not any displayed span), in the relation's DIRECTIONAL order per the inventory. Set answer_role to the linked argument's role: the literal is the question locator and the linked support_property is the accepted answer.
Emit each distinct fact EXACTLY ONCE. Return exactly one candidate_accounting row per shown S-label with a short reason (emitted / duplicate_mention / exhausted_no_relation / unsupported) referencing labels and levels only. Return only the structured response."""


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
) -> tuple[
    str, tuple[int, int] | None, list[tuple[int, int]] | None, str | None, str | None,
]:
    """Resolve a v4 literal after deriving a source-local anchor from linked spans."""
    clauses = _source_clause_spans(document)

    def clause_index(start: int, end: int) -> int | None:
        return next((index for index, (left, right) in enumerate(clauses)
                     if left <= start < end <= right), None)

    linked = [argument for argument in arguments if argument["kind"] == "linked"]
    if not linked:
        return "", None, None, "missing_linked_argument", None
    try:
        linked_spans = [
            (int(occurrences[argument["occurrence_id"]]["start"]),
             int(occurrences[argument["occurrence_id"]]["end"]))
            for argument in linked
        ]
    except (KeyError, TypeError, ValueError):
        return "", None, None, "invalid_evidence_occurrence", None
    linked_clause_indices = [clause_index(start, end) for start, end in linked_spans]
    if any(index is None for index in linked_clause_indices):
        return "", None, None, "invalid_evidence_occurrence", None
    linked_plan_section = _shared_plan_section(document, linked_spans)
    linked_problem_block = _shared_problem_block(document, linked_spans)
    context = next((argument for argument in arguments if argument["kind"] == "context"), None)
    stitched_context_clause_ranges: list[tuple[int, int]] | None = None
    if context is not None:
        literal = str(context["literal"])
        # Source spans under case/whitespace equivalence (not exact substring): the teacher
        # routinely re-cases a literal ("MRI" for source "mri", "hemoglobin A1c" for "a1c").
        # Each span indexes the original document, so grounding stays exact once the literal
        # is written back as its source substring below.
        if context.get("start") is None:
            matches = _source_literal_spans(document, literal)
        else:
            try:
                resolved_start = int(context["start"])
                resolved_end = int(context["end"])
            except (KeyError, TypeError, ValueError):
                return "", None, None, "invalid_evidence_occurrence", None
            if not (0 <= resolved_start < resolved_end <= len(document)
                    and document[resolved_start:resolved_end] == literal):
                return "", None, None, "invalid_evidence_occurrence", None
            matches = [(resolved_start, resolved_end)]

        def _linked_gap(start: int, end: int) -> int:
            return min(max(0, left - end, start - right) for left, right in linked_spans)

        # A context literal grounds when it co-locates with the linked argument by
        # adjacent clause OR within the one section the arguments share. That section is
        # detected in either note format: the structured written-note plan
        # (_shared_plan_section) or the spoken-dialogue problem block
        # (_shared_problem_block) -- symmetric with the evidence-window logic below (P3).
        # When several mentions of the literal qualify, pick the one nearest the linked
        # argument: fewest clauses away, then fewest chars (P2, deterministic tie-break).
        scored = []
        for start, end in matches:
            literal_clause = clause_index(start, end)
            if literal_clause is None:
                continue
            clause_dist = min(abs(literal_clause - index) for index in linked_clause_indices)
            in_clause_window = clause_dist <= 1
            in_plan = (linked_plan_section is not None
                       and linked_plan_section[0] <= start < end <= linked_plan_section[1])
            in_block = (linked_problem_block is not None
                        and linked_problem_block[0] <= start < end <= linked_problem_block[1])
            if in_clause_window or in_plan or in_block:
                scored.append((clause_dist, _linked_gap(start, end), start, end))
        if scored:
            scored.sort()
            start, end = scored[0][2], scored[0][3]
            # Co-located via the shared plan/problem section but NOT an adjacent
            # clause: anchor surgically to [linked clause, literal clause] instead
            # of the whole section. Handing the reader the entire multi-problem
            # section buries which condition the literal pairs with (it answers
            # empty), so mirror the span->span stitch and elide the middle. The
            # adjacent case (<=1 clause) keeps its single-region quote below.
            literal_clause = clause_index(start, end)
            if literal_clause is not None:
                nearest_linked = min(
                    linked_clause_indices, key=lambda index: abs(index - literal_clause)
                )
                if abs(literal_clause - nearest_linked) > 1:
                    stitched_context_clause_ranges = [
                        clauses[index] for index in sorted({nearest_linked, literal_clause})
                    ]
        else:
            # No explicit plan/problem marker is common in encounter summaries. Mirror the
            # span->span bridge for one uncontrolled literal. A repeated linked value is
            # grounded at its nearest same-decision occurrence rather than rejected for
            # having a closer sibling: action identity remains decision-level, while the
            # source evidence uses the occurrence that actually co-locates with the literal.
            stitched = []
            for start, end in matches:
                literal_clause = clause_index(start, end)
                if literal_clause is None:
                    continue
                for linked_index, linked_argument in enumerate(linked):
                    decision_id = occurrences[linked_argument["occurrence_id"]].get("decision_id")
                    siblings = sorted(
                        (
                            (str(occurrence_id), occurrence)
                            for occurrence_id, occurrence in occurrences.items()
                            if decision_id is not None and occurrence.get("decision_id") == decision_id
                            and isinstance(occurrence.get("start"), int)
                            and isinstance(occurrence.get("end"), int)
                        ),
                        key=lambda row: (int(row[1]["start"]), int(row[1]["end"]), row[0]),
                    )
                    for occurrence_id, occurrence in siblings:
                        linked_start, linked_end = int(occurrence["start"]), int(occurrence["end"])
                        linked_clause = clause_index(linked_start, linked_end)
                        if linked_clause is None:
                            continue
                        clause_dist = abs(literal_clause - linked_clause)
                        first_end, last_start = (
                            (linked_end, start) if linked_end <= start else (end, linked_start)
                        )
                        if _PROBLEM_SWITCH_BOUNDARY.search(document, first_end, last_start) is not None:
                            continue
                        char_gap = max(0, linked_start - end, start - linked_end)
                        stitched.append((
                            clause_dist, char_gap, start, end, linked_index, occurrence_id,
                            linked_clause, literal_clause,
                        ))
            if not stitched:
                return "", None, None, "unknown_context_literal", None
            stitched.sort()
            _, _, start, end, linked_index, occurrence_id, linked_clause, literal_clause = stitched[0]
            linked_argument = linked[linked_index]
            occurrence = occurrences[occurrence_id]
            linked_argument.update({
                "occurrence_id": occurrence_id,
                "surface": str(occurrence.get("surface", "")),
                "runtime_type": str(occurrence.get("runtime_type", "")),
            })
            stitched_context_clause_ranges = [
                clauses[index] for index in sorted({linked_clause, literal_clause})
            ]
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
            return "", None, None, "protected_context_literal", None
        # Ground against the exact source substring, which may differ from the teacher
        # spelling only by case/whitespace; identify the typed candidate by the resolved
        # source span rather than the teacher's spelling.
        context["literal"] = document[start:end]
        literal = context["literal"]
        typed = [row for row in context_candidates.values()
                 if int(row.get("start", -1)) == start and int(row.get("end", -1)) == end]
        if typed:
            if len(typed) != 1:
                return "", None, None, "untyped_context_literal", None
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
        if stitched_context_clause_ranges is not None:
            quote = "\n".join(
                document[left:right].strip() for left, right in stitched_context_clause_ranges
            )
            if not quote:
                return "", None, None, "invalid_evidence", None
            envelope = (
                stitched_context_clause_ranges[0][0],
                stitched_context_clause_ranges[-1][1],
            )
            return quote, envelope, stitched_context_clause_ranges, None, "stitched_clauses"
    try:
        all_indices = [
            clause_index(
                int(occurrences[argument["occurrence_id"]]["start"])
                if argument["kind"] == "linked" else int(argument["start"]),
                int(occurrences[argument["occurrence_id"]]["end"])
                if argument["kind"] == "linked" else int(argument["end"]),
            )
            for argument in arguments
        ]
    except (KeyError, TypeError, ValueError):
        return "", None, None, "invalid_evidence_occurrence", None
    if not any(index is None for index in all_indices) and max(all_indices) - min(all_indices) <= 1:
        left, right = clauses[min(all_indices)][0], clauses[max(all_indices)][1]
        # A speaker-turn "clause" (dialogue, >=2 turn markers) can hold several period
        # sub-clauses; the cue must be searched across the whole turn, not per sub-clause.
        # _source_clause_spans (here) and the support-check's period regex otherwise
        # disagree on clause boundaries, failing valid same-turn relations (Case 3).
        kind = "speaker_turn" if len(_ACI_SPEAKER_TURN_PATTERN.findall(document)) >= 2 else "clause"
        return document[left:right], (left, right), None, None, kind
    argument_spans = [
        (int(occurrences[argument["occurrence_id"]]["start"]),
         int(occurrences[argument["occurrence_id"]]["end"]))
        if argument["kind"] == "linked" else (int(argument["start"]), int(argument["end"]))
        for argument in arguments
    ]
    plan_section = _shared_plan_section(document, argument_spans)
    if plan_section is not None:
        reader_ranges = _argument_clause_ranges(clauses, all_indices)
        # Keep the full plan envelope for source integrity and support checks, but
        # do not hand unrelated plan rows to the reader when the arguments are
        # separated within it. The persisted ranges are source clauses, so the
        # runtime still excerpts the corresponding stable turn indices after render.
        if reader_ranges is not None and len(reader_ranges) > 1:
            return (
                document[plan_section[0]:plan_section[1]], plan_section,
                reader_ranges, None, "plan_section",
            )
        return document[plan_section[0]:plan_section[1]], plan_section, None, None, "plan_section"
    # Spoken transcript: the arguments may sit in different turns of one
    # problem discussion (a patient acknowledgment between them). Ground within
    # that block; the hedge guard and cue check in compilation still apply.
    problem_block = _shared_problem_block(document, argument_spans)
    if problem_block is not None:
        reader_ranges = _argument_clause_ranges(clauses, all_indices)
        if reader_ranges is not None and len(reader_ranges) > 1:
            return (
                document[problem_block[0]:problem_block[1]], problem_block,
                reader_ranges, None, "problem_block",
            )
        return document[problem_block[0]:problem_block[1]], problem_block, None, None, "problem_block"
    # No explicit problem marker is common in short encounter summaries. Permit
    # a span->span bridge subject to no assessment/problem switch between the
    # source spans. Distance beyond the global soft cap is recorded downstream,
    # rather than silently discarding a reader-verifiable relation.
    if len(linked) == len(arguments):
        first_index, last_index = min(linked_clause_indices), max(linked_clause_indices)
        first_span, last_span = min(linked_spans), max(linked_spans)
        boundary = _PROBLEM_SWITCH_BOUNDARY.search(document, first_span[1], last_span[0])
        # Do not bridge past a same-decision mention. The sibling remapper must
        # get the chance to select that closer occurrence, rather than treating
        # an earlier history mention as evidence for a later plan clause.
        def is_intervening_sibling(index: int, argument: Mapping, occurrence: Mapping) -> bool:
            start, end = occurrence.get("start"), occurrence.get("end")
            return (
                isinstance(start, int) and isinstance(end, int)
                and occurrence.get("decision_id") == occurrences[argument["occurrence_id"]].get("decision_id")
                and (start, end) != linked_spans[index]
                and first_span[1] <= start < end <= last_span[0]
            )

        intervening_sibling = any(
            is_intervening_sibling(index, argument, occurrence)
            for index, argument in enumerate(linked)
            for occurrence in occurrences.values()
        )
        if boundary is None and not intervening_sibling:
            clause_ranges = _argument_clause_ranges(clauses, linked_clause_indices)
            if clause_ranges is None:
                return "", None, None, "invalid_evidence", None
            quote = "\n".join(document[left:right].strip() for left, right in clause_ranges)
            if not quote:
                return "", None, None, "invalid_evidence", None
            # The contiguous envelope remains the stable integrity locator.
            # Consumers that render reader context must use clause_ranges so
            # elided middle clauses are never sent as evidence.
            envelope = (clause_ranges[0][0], clause_ranges[-1][1])
            return quote, envelope, clause_ranges, None, "stitched_clauses"
    return "", None, None, "invalid_evidence", None


# Natural-language hypotheses for the NLI support fallback. Filled from the arguments'
# roles (subject/object per _RELATION_ARGUMENT_CLASSES). Used ONLY as a union fallback to
# the fixed cue/connector lexicon, so a valid relation phrased with an out-of-list
# connector ("we have you on X for Y") is not rejected as invalid_evidence.
RELATION_SUPPORT_HYPOTHESIS = {
    "prescribed_with": "{object} is a medication prescribed for {subject}.",
    "procedure_for": "{object} is a procedure or treatment for {subject}.",
    "tests_for": "{object} is a test or scan used to investigate {subject}.",
    "contraindicated_because_of": "{subject} must be avoided because of {object}.",
    "causes_or_explains": "{subject} is caused or explained by {object}.",
}


def _relation_support_hypothesis(relation: str, arguments: Sequence[Mapping]) -> str | None:
    """Fill the relation's NLI hypothesis from the arguments' subject/object surfaces."""
    template = RELATION_SUPPORT_HYPOTHESIS.get(relation)
    if template is None:
        return None
    by_role = {}
    for argument in arguments:
        text = str(argument.get("surface") or argument.get("literal") or "").strip()
        if argument.get("role") and text:
            by_role[str(argument["role"])] = text
    if "subject" not in by_role or "object" not in by_role:
        return None
    return template.format(**by_role)


def _relation_quote_has_semantic_support(
    relation: str,
    quote: str,
    arguments: Sequence[Mapping],
    relation_contract: Mapping[str, Mapping],
    *,
    allow_adjacent_clauses: bool = False,
    allow_plan_section: bool = False,
) -> bool:
    """NLI fallback to the lexical cue check: both arguments present within the allowed
    clause span, and the quote entails the relation hypothesis. Reuses the same
    structural proximity gate so entailment is never judged across unrelated distant text."""
    normalized = canon(quote)
    clause_ranges = [
        (match.start(), match.end())
        for match in re.finditer(r"(?:[^\n.!?;]|\.(?=\.)|(?<=\.)\.)+", normalized)
        if match.group().strip()
    ]
    if not clause_ranges:
        return False
    argument_clauses = []
    for argument in arguments:
        text = canon(str(argument.get("surface", argument.get("literal", ""))))
        match = re.search(rf"(?<!\w){re.escape(text)}(?!\w)", normalized)
        if match is None:
            return False
        clause = next((index for index, (left, right) in enumerate(clause_ranges)
                       if left <= match.start() < right), None)
        if clause is None:
            return False
        argument_clauses.append(clause)
    clause_span = max(argument_clauses) - min(argument_clauses)
    if not (clause_span == 0
            or (clause_span == 1 and (allow_adjacent_clauses or allow_plan_section))
            or allow_plan_section):
        return False
    hypothesis = _relation_support_hypothesis(relation, arguments)
    if hypothesis is None:
        return False
    from cloak.lattice import nli_entails
    return nli_entails(normalized, hypothesis)


# Relation cue gates in the COMPILER (fixed lexical-cue lexicon + NLI rescue) are DISABLED:
# for a teacher-proposed relation the three-point reader gate is the semantic acceptance
# check, and the maintained cue lexicon is not sustainable on informal clinical speech
# (recurring false-negatives; same rationale as the literal-probe cue drop, extended to
# span->span). Structural safeguards -- anchor locality, exact grounding, no problem-switch
# crossing -- all remain. NOT applied to the opportunity miner: there the cue is the only
# precision filter on a combinatorial pair enumeration (disabling it inflated span_span
# opportunities ~10x), so the miner keeps it. Cue code retained but dormant pending removal:
# see docs/issues/qa-builder-dept.md.
RELATION_CUE_GATES_DISABLED = True

# Deterministic semantic_property category probes are DISABLED (2026-07-18): across D2N001-007
# they yielded 1 kept assertion against 114 no_task_role_cue rejections, and the QA focus is
# teacher relations. Generation code (incl. the informative-context judge escalation) retained
# dormant; see docs/issues/qa-builder-dept.md.
SEMANTIC_PROPERTY_PROBES_DISABLED = True


def _relation_quote_has_direct_support(
    relation: str,
    quote: str,
    arguments: Sequence[Mapping],
    relation_contract: Mapping[str, Mapping],
    *,
    allow_adjacent_clauses: bool = False,
    allow_plan_section: bool = False,
    require_lexical_cue: bool = False,
) -> bool:
    """Direct support = a fixed cue/connector match OR (fallback) NLI entailment of the
    relation hypothesis. The lexical check stays authoritative (DeBERTa-mnli can lack
    clinical world knowledge); NLI only RESCUES valid relations whose connector is not in
    the fixed lexicon ("we have you on X for Y") -- it never overrides a lexical accept."""
    if _relation_quote_has_lexical_cue_support(
        relation, quote, arguments, relation_contract,
        allow_adjacent_clauses=allow_adjacent_clauses, allow_plan_section=allow_plan_section,
    ):
        return True
    if require_lexical_cue:
        return False
    # NLI only rescues the lenient modes, where the lexical path already searches for a
    # cue *inside the argument clause(s)* -- so NLI just widens that cue vocabulary. In
    # strict single-clause mode the lexical path instead requires a clean directional
    # connector (no intervening content); that precision guard against conjunction
    # ambiguity ("Hypothyroidism and diabetes is treated with X") must stand, so NLI
    # does not fire there.
    if not (allow_adjacent_clauses or allow_plan_section):
        return False
    return _relation_quote_has_semantic_support(
        relation, quote, arguments, relation_contract,
        allow_adjacent_clauses=allow_adjacent_clauses, allow_plan_section=allow_plan_section,
    )


def _relation_quote_has_lexical_cue_support(
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


def _remap_to_groundable_siblings(
    document, arguments, occurrences, context_by_id, relation, relation_contract,
):
    """A repeated controlled value has several occurrences (S-labels); the teacher
    sometimes names one (e.g. a history-list mention) that does not sit in the
    relation sentence, so grounding fails. Every occurrence of the same value shares
    one decision, so swap each linked argument to the same-decision occurrence that
    actually grounds with the other argument. Support/levels/answer_target are keyed
    by decision and are unchanged; only the grounding mention moves."""
    from itertools import product

    def grounds(args) -> bool:
        quote, span, _, err, akind = _derived_relation_anchor(
            document, args, occurrences, context_by_id, relation, relation_contract)
        if err is not None:
            return False
        if not all(_argument_is_grounded(a, document, span, occurrences) for a in args):
            return False
        # A literal->linked probe is judged by the later three-point reader
        # gate, not a lexical/NLI relation parser. Exact literal grounding and
        # the anchor's locality constraints still prevent document-wide pairs.
        if RELATION_CUE_GATES_DISABLED:
            return True
        if sum(argument.get("kind") == "linked" for argument in args) == 1:
            return True
        return _relation_quote_has_direct_support(
            relation, quote, args, relation_contract, allow_adjacent_clauses=True,
            allow_plan_section=akind in {"plan_section", "problem_block", "speaker_turn"})

    if grounds(arguments):
        return arguments
    linked_idx = [i for i, a in enumerate(arguments) if a.get("kind") == "linked"]
    if not linked_idx:
        return arguments
    sibling_lists = []
    for i in linked_idx:
        decision_id = (occurrences.get(arguments[i]["occurrence_id"]) or {}).get("decision_id")
        siblings = [oid for oid, occ in occurrences.items()
                    if decision_id is not None and occ.get("decision_id") == decision_id]
        sibling_lists.append(siblings or [arguments[i]["occurrence_id"]])
    for combo in product(*sibling_lists):
        if len(set(combo)) != len(combo):
            continue
        trial = [dict(a) for a in arguments]
        for j, i in enumerate(linked_idx):
            trial[i]["occurrence_id"] = combo[j]
            trial[i]["surface"] = str(occurrences[combo[j]].get("surface", ""))
        if grounds(trial):
            return trial
    return arguments


_RELATION_ESCALATION_SCOPES = ("span_span", "span_literal")


def _relation_scope(arguments: Sequence[Mapping]) -> str:
    linked_count = sum(argument.get("kind") == "linked" for argument in arguments)
    if linked_count == 2:
        return "span_span"
    if linked_count == 1:
        return "span_literal"
    raise ValueError("relation requires one or two linked arguments")


def _relation_fact_key(
    relation: str, arguments: Sequence[Mapping], occurrences: Mapping[str, Mapping],
) -> tuple:
    """Decision-level directional identity for one relation fact.

    Evidence occurrences may differ between repeated mentions, but the ranker acts
    once per decision.  A context literal remains a literal identity because it has
    no ranker decision and may legitimately be an exact answer.
    """
    identities = []
    for argument in arguments:
        if argument.get("kind") == "linked":
            occurrence = occurrences.get(str(argument.get("occurrence_id")))
            if occurrence is None or occurrence.get("decision_id") is None:
                raise ValueError("linked relation argument has no decision")
            identity = ("linked_decision", str(occurrence["decision_id"]))
        elif argument.get("kind") == "context":
            literal = canon(str(argument.get("literal", "")))
            if not literal:
                raise ValueError("context relation argument has no literal")
            identity = ("context_literal", literal)
        else:
            raise ValueError("unknown relation argument kind")
        identities.append(identity)
    return (str(relation), *identities)


def _compiled_relation_fact_key(candidate: Mapping, occurrences: Mapping[str, Mapping]) -> tuple:
    arguments = list(dict(candidate.get("evidence") or {}).get("arguments") or [])
    if len(arguments) != 2:
        raise ValueError("compiled relation has no two-argument evidence")
    return _relation_fact_key(str(candidate.get("relation", "")), arguments, occurrences)


def _pair_fact_keys(rows: Sequence[Mapping], occurrences: Mapping[str, Mapping]) -> set[tuple]:
    """Pair-level fact keys covered by relation rows: a two-argument row contributes its own
    key; a compound row (one subject + N objects: set-valued, compound-locator, multi-literal)
    is decomposed into one subject x object key per object. Rows whose arguments cannot key
    (unknown literal/decision) contribute nothing."""
    keys: set[tuple] = set()
    for row in rows:
        arguments = list(dict(row.get("evidence") or {}).get("arguments") or [])
        subjects = [a for a in arguments if a.get("role") == "subject"]
        objects = [a for a in arguments if a.get("role") == "object"]
        if len(subjects) != 1 or not objects:
            continue
        for obj in objects:
            try:
                keys.add(_relation_fact_key(
                    str(row.get("relation", "")), [subjects[0], obj], occurrences))
            except ValueError:
                continue
    return keys


def _remap_to_lexically_groundable_siblings(
    document: str,
    arguments: Sequence[Mapping],
    occurrences: Mapping[str, Mapping],
    context_by_id: Mapping[str, Mapping],
    relation: str,
    relation_contract: Mapping[str, Mapping],
) -> list[dict]:
    """Cheap sibling remap for the escalation ledger; never calls NLI."""
    from itertools import product

    def grounds(trial: Sequence[Mapping]) -> bool:
        quote, span, _, error, anchor_kind = _derived_relation_anchor(
            document, list(trial), occurrences, context_by_id, relation, relation_contract,
        )
        return (
            error is None
            and span is not None
            and all(_argument_is_grounded(argument, document, span, occurrences)
                    for argument in trial)
            and _relation_quote_has_lexical_cue_support(
                relation, quote, trial, relation_contract,
                allow_adjacent_clauses=True,
                allow_plan_section=anchor_kind in {"plan_section", "problem_block", "speaker_turn"},
            )
        )

    initial = [dict(argument) for argument in arguments]
    if grounds(initial):
        return initial
    linked_indices = [index for index, argument in enumerate(initial)
                      if argument.get("kind") == "linked"]
    sibling_lists = []
    for index in linked_indices:
        decision_id = (occurrences.get(initial[index]["occurrence_id"]) or {}).get("decision_id")
        siblings = [occurrence_id for occurrence_id, occurrence in occurrences.items()
                    if decision_id is not None and occurrence.get("decision_id") == decision_id]
        sibling_lists.append(siblings or [initial[index]["occurrence_id"]])
    for sibling_ids in product(*sibling_lists):
        if len(set(sibling_ids)) != len(sibling_ids):
            continue
        trial = [dict(argument) for argument in initial]
        for index, occurrence_id in zip(linked_indices, sibling_ids):
            trial[index]["occurrence_id"] = occurrence_id
            trial[index]["surface"] = str(occurrences[occurrence_id].get("surface", ""))
        if grounds(trial):
            return trial
    return initial


def _opportunity_record(relation, key, anchor_kind, arguments, span, *, recovered):
    return {
        "relation": relation,
        "scope": _relation_scope(arguments),
        "fact_key": key,
        "anchor_kind": anchor_kind,
        "recovered_by_escalation": recovered,
        # carried for the gleaning "missed" evidence card (unused by escalation counts)
        "arguments": deepcopy(arguments),
        "evidence_span": list(span),
    }


# Deterministic POSITIVE-STRUCTURE junk pre-filter for escalation eligibility. A pending pair is
# skipped from the judge ONLY when the surface structure PROVES it non-assertive: coordinated list
# siblings (only list punctuation + and/or between them, e.g. a PMH enumeration "CHF, depression and
# hypertension"), or an argument under an explicit negation cue ("no evidence of X"). These never
# carry an asserted relation, so skipping costs no recall (validated: 0/36 reader-passed relations
# flagged). Absence-of-signal rules (no cue / large distance) are deliberately NOT used here -- they
# drop real relations (verified). "with"/"as" are NOT coordinators ("CHF with diastolic dysfunction"
# is a qualifier, not a flat list). See docs/issues/qa-builder-dept.md.
_ESCALATION_COORDINATION = re.compile(
    r"^[\s,;:/&()\-\.]*(?:\b(?:and|or)\b[\s,;:/&()\-\.]*)*$", re.I)
_ESCALATION_NEGATION = re.compile(
    r"\b(denies|denied|no evidence of|negative for|ruled out|no history of|no known|"
    r"non-?contributory|without any)\b", re.I)
# causes_or_explains-ONLY co-occurrence connector: two clinical findings joined by coordination
# ("edema and some erythema") or attributive comorbidity ("heart failure with diastolic
# dysfunction") state adjacency, never causation. The between-args gap (intervening entities
# blanked) is REJECTED when it consists solely of these tokens -- so any real linking verb in the
# gap ("X, causing Y", "X from Y") leaves a non-matching token and survives to the judge. Scoped to
# this relation because "with" is a legitimate prescribed_with cue ("treated with"). See the
# co-occurrence-vs-causation analysis in results/qa_v2_stage_ab/finer_level_failures_for_fable.md.
_COOCCURRENCE_CONNECTOR = re.compile(
    r"^[\s,;:/&()\-\.]*(?:\b(?:and|or|with|without|w|associated|along|some|a|an|the)\b"
    r"[\s,;:/&()\-\.]*)+$", re.I)
# causes_or_explains proximity cap: a causal assertion in a clinical note is LOCAL -- the two
# findings sit in the same or an adjacent clause ("X is due to Y"; "X. that's causing the Y").
# `_derived_relation_anchor` tags that <=1-clause-apart case "clause" (or "speaker_turn" in a
# transcript -- a MISNOMER for the local tier, not a whole-turn window). The wider anchors
# (plan_section / problem_block / the stitched_clauses span-bridge) are only reached when the
# arguments are >1 clause apart; for a causal claim those recover whole-note-apart pairs that are
# co-occurrence, not causation (audit: the ~4900-char escalation miss). Adjacent-clause causality
# is caught by the local branch BEFORE the wide tiers, so this never drops a genuinely local cause.
# ponytail: a real cause whose direction is only clarified by a DISTANT excerpt (a rare
# false-negative the local reader window can't confirm) is a known, deferred side case -- widen the
# reader's evidence turns for causes_or_explains if it ever proves material.
_CAUSAL_LOCAL_ANCHOR_KINDS = frozenset({"clause", "speaker_turn"})


def _escalation_prefilter_reason(
    document: str, arguments: Sequence[Mapping],
    occurrences: Mapping[str, Mapping], contexts: Mapping[str, Mapping],
    relation: str | None = None,
) -> str | None:
    """Junk reason if this pair is provably non-assertive by surface structure, else None."""
    def span_of(argument: Mapping) -> tuple[int, int] | None:
        if argument.get("kind") == "linked":
            occ = occurrences.get(str(argument.get("occurrence_id"))) or {}
            start, end = occ.get("start"), occ.get("end")
        else:
            start, end = argument.get("start"), argument.get("end")
        return (int(start), int(end)) if isinstance(start, int) and isinstance(end, int) else None

    spans = [span_of(argument) for argument in arguments]
    if len(spans) != 2 or any(span is None for span in spans):
        return None
    (a0, a1), (b0, b1) = spans
    lo, hi = (a1, b0) if a1 <= b0 else (b1, a0)
    lo, hi = min(lo, hi), max(lo, hi)
    if lo <= hi:
        gap = list(document[lo:hi])
        others = [(int(o["start"]), int(o["end"])) for o in occurrences.values()
                  if isinstance(o.get("start"), int) and isinstance(o.get("end"), int)]
        others += [(int(r["start"]), int(r["end"])) for r in contexts.values()]
        for x0, x1 in others:                      # blank intervening list neighbours
            if lo <= x0 and x1 <= hi:
                for i in range(max(0, x0 - lo), min(len(gap), x1 - lo)):
                    gap[i] = " "
        gap_text = "".join(gap)
        # causes_or_explains: a coordination/comorbidity connector between the two findings is
        # co-occurrence, never an asserted cause -- the model won't reliably reject it (its judge
        # rule already forbids co-occurrence, yet accepts it), so reject deterministically here.
        if relation == "causes_or_explains" and _COOCCURRENCE_CONNECTOR.match(gap_text):
            return "cooccurrence_connector"
        if _ESCALATION_COORDINATION.match(gap_text):
            # Only SAME-TYPE siblings are a coordinated enumeration (two conditions in a PMH list).
            # A cross-type "and" (condition AND drug) is not a list -- it can join a condition to its
            # treatment (e.g. "hypothyroidism and synthroid"), a real relation -- so never skip it.
            classes = [
                _argument_relation_classes(
                    str(argument.get("runtime_type", "")),
                    str(argument.get("surface") or argument.get("literal") or ""))
                for argument in arguments
            ]
            if classes[0] & classes[1]:
                return "coordination_sibling"
    for start, _ in spans:                          # explicit negation in the argument's own clause
        pre = document[max(0, start - 55):start]
        matches = list(_ESCALATION_NEGATION.finditer(pre))
        if matches and not re.search(r"[.?!]", pre[matches[-1].end():]):
            return "negation_scope"
    return None


def _all_occurrence_judge_premise(
    document: str, arguments: Sequence[Mapping],
    occurrences: Mapping[str, Mapping], clauses: Sequence[tuple[int, int]],
) -> str:
    r"""Recall-oriented judge premise: source clauses of ALL occurrences of both arguments, deduped,
    source-ordered, middle elided ("clause2\nclause13\nclause14" -- NEVER the contiguous span 2..14).
    Gives the accept-biased judge every place each entity is discussed so a document-level relation
    is visible, without needing to pick the "right" occurrence. Escalation-verdict ONLY: never
    persisted on an assertion and never reaches the reader (which uses the compiler anchor's clause
    ranges via `_derived_relation_anchor`). See docs/issues/qa-builder-dept.md."""
    def clause_index(start: int, end: int) -> int | None:
        return next((i for i, (left, right) in enumerate(clauses)
                     if left <= start < end <= right), None)

    by_decision: dict[str, list[tuple[int, int]]] = {}
    for occ in occurrences.values():
        start, end = occ.get("start"), occ.get("end")
        if isinstance(start, int) and isinstance(end, int):
            by_decision.setdefault(str(occ.get("decision_id")), []).append((int(start), int(end)))

    indices: set[int] = set()
    for argument in arguments:
        if argument.get("kind") == "linked":
            occ = occurrences.get(str(argument.get("occurrence_id"))) or {}
            spans = by_decision.get(str(occ.get("decision_id")), [])
        elif argument.get("literal"):
            spans = _source_literal_spans(document, str(argument["literal"]))
            if not spans and isinstance(argument.get("start"), int):
                spans = [(int(argument["start"]), int(argument["end"]))]
        else:
            spans = []
        for start, end in spans:
            clause = clause_index(start, end)
            if clause is not None:
                indices.add(clause)
    return "\n".join(document[clauses[i][0]:clauses[i][1]].strip() for i in sorted(indices))


# LLM-prefilter set-call, one per relation, keyed on a controlled CONDITION anchor. The gazetteer
# (relation_context_candidates) can only emit test/procedure/status/category literals, so drug,
# symptom, and condition literal objects are structurally unreachable -- the prefilter recovers them.
# `{2}` is the runtime_type stamped on returned phrases so they type into the relation's object class
# (test->monitoring, drug->treatment, symptom->symptom, procedure->procedure). The condition may be
# the object (contraindicated_because_of) or subject; the combinatorial pairing in
# relation_support_opportunities assigns the role, so anchoring on the condition suffices.
# (kind, phrase, runtime_type, constraint). `constraint` is an extra form/precision instruction for
# relations whose objects the small reader tends to answer loosely; "" for the entity-typed relations
# (test/drug/procedure objects are naturally named noun phrases).
_RELATION_PREFILTER_SETCALL: dict[str, tuple[str, str, str, str]] = {
    "prescribed_with": (
        "medication or drug", "was prescribed, started, continued, or given to treat", "drug", ""),
    "procedure_for": (
        "medical procedure, therapy, surgery, or referral",
        "was performed, planned, referred, or previously done to treat", "procedure", ""),
    "tests_for": (
        "diagnostic test, lab, panel, imaging study, or exam",
        "was ordered, performed, or resulted to work up, monitor, or evaluate", "test", ""),
    "contraindicated_because_of": (
        "medication, drug, drug class, or procedure",
        "must be avoided or is contraindicated because of", "drug", ""),
    # causes_or_explains objects are open-ended findings, so the reader returns imperatives/advice and
    # clauses ("push fluids", "take it easy", "wbc is not elevated"). Pin the FORM hard: a named
    # noun-phrase finding only. Final attempt -- drop the relation from the prefilter if this misses.
    "causes_or_explains": (
        "symptom, sign, or abnormal test or exam finding",
        "is caused or explained by", "symptom",
        "Each answer MUST be a short NOUN PHRASE naming one specific symptom, sign, or abnormal "
        "finding (for example: chest pain, jaundice, elevated white cell count, lower-extremity "
        "swelling). Do NOT return: advice or instructions, actions or verb phrases, full sentences "
        "or clauses, negated or normal findings, or the patient's own words about their life, diet, "
        "or behavior."),
}
_RELATION_PREFILTER_FRAME = (
    "Clinical note:\n\"\"\"\n{ctx}\n\"\"\"\n\n"
    "List EVERY distinct {kind} that the note says {phrase} the patient's {anchor}.\n"
    "{constraint}"
    "Copy each answer verbatim as a short phrase from the note; include nothing not in the note.\n"
    "Respond with ONLY a JSON array of strings, e.g. [\"x\",\"y\"]. If there are none, respond []."
)


def _parse_llm_json_array(raw: str) -> list[str]:
    """Extract a JSON string array from a (possibly ```json-fenced) reply; [] on any malformity."""
    match = re.search(r"\[.*\]", raw or "", re.DOTALL)
    if match is None:
        return []
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(value).strip() for value in parsed if str(value).strip()]


def llm_prefilter_context_candidates(
    document: str,
    environment_document: Mapping,
    propose: "Callable[[str], str]",
) -> list[dict]:
    """High-recall LLM proposer of context-literal relation candidates (opt-in, augment-only).

    For each controlled condition, run one enumeration set-call per relation on the ORIGINAL document
    (build-time, never published), locate each returned phrase verbatim, and emit it as a typed
    context-literal candidate. Feeds relation_support_opportunities' `extra_context_candidates`, where
    the SAME grounding + lexical-cue + judge pipeline decides acceptance -- the prefilter only widens
    recall, it never bypasses a gate. Phrases that are not verbatim-locatable are dropped (cannot ground).
    """
    conditions: dict[str, str] = {}
    for occurrence in environment_document.get("occurrences", []):
        if (occurrence.get("controlled", True)
                and canon(str(occurrence.get("runtime_type", ""))) == "health-condition"
                and occurrence.get("surface")):
            conditions.setdefault(str(occurrence.get("decision_id")), str(occurrence["surface"]))
    candidates: dict[tuple, dict] = {}
    for anchor in conditions.values():
        for kind, phrase, runtime_type, constraint in _RELATION_PREFILTER_SETCALL.values():
            prompt = _RELATION_PREFILTER_FRAME.format(
                ctx=document, kind=kind, phrase=phrase, anchor=anchor,
                constraint=(constraint + "\n") if constraint else "")
            for item in _parse_llm_json_array(propose(prompt)):
                match = re.search(re.escape(item), document, re.IGNORECASE)
                if match is None:
                    continue
                start, end = match.span()
                literal = document[start:end]
                key = (runtime_type, start, end, literal)
                candidates.setdefault(key, {
                    "context_candidate_id": "context:" + _stable_hash({
                        "runtime_type": runtime_type, "start": start, "end": end, "literal": literal,
                    }),
                    "kind": "context_literal",
                    "runtime_type": runtime_type,
                    "literal": literal,
                    "start": start,
                    "end": end,
                    "provenance": "llm_prefilter",
                })
    return sorted(candidates.values(),
                  key=lambda row: (int(row["start"]), str(row["context_candidate_id"])))


def relation_support_opportunities(
    document: str,
    environment_document: Mapping,
    *,
    relation_contract: Mapping[str, Mapping] = ACI_RELATION_CONTRACT,
    escalator: "Callable[..., bool] | None" = None,
    extra_context_candidates: "Sequence[Mapping] | None" = None,
) -> list[dict]:
    """Enumerate conservative, source-supported relation opportunities.

    This is deliberately a structural lower-bound signal for teacher escalation,
    never an assertion generator.  It shares compiler argument typing, anchors,
    sibling remapping, and lexical evidence checks, but avoids the NLI fallback so
    target calculation remains local, deterministic, and cheap.
    """
    occurrences = {
        str(row["occurrence_id"]): row
        for row in environment_document.get("occurrences", [])
        if row.get("occurrence_id") is not None
    }
    # Gazetteer context literals, optionally AUGMENTED with an LLM-prefilter's typed literals. Union
    # (never replace): the augmented set is a superset of the gazetteer set, so a currently-accepted
    # pair can never be dropped (validated on 5 docs -- scripts/spikes/relation_prefilter_regression.py).
    # Dedup by context_candidate_id, so a prefilter literal identical to a gazetteer one collapses.
    context_rows = list(relation_context_candidates(document))
    if extra_context_candidates:
        context_rows += list(extra_context_candidates)
    contexts = {
        str(row["context_candidate_id"]): row
        for row in context_rows
    }
    linked = [
        {
            "kind": "linked",
            "occurrence_id": str(row["occurrence_id"]),
            "surface": str(row["surface"]),
            "runtime_type": str(row["runtime_type"]),
            "support_property": str(row["properties"][0]),
        }
        for row in relation_teacher_span_inventory(environment_document)
        if row.get("properties")
    ]
    context = [
        {
            "kind": "context",
            "literal": str(row["literal"]),
            "runtime_type": str(row["runtime_type"]),
            "start": int(row["start"]),
            "end": int(row["end"]),
            "context_candidate_id": str(row["context_candidate_id"]),
        }
        for row in contexts.values()
    ]
    opportunities: dict[tuple, dict] = {}
    pending: dict[tuple, tuple] = {}  # cue-misses awaiting the escalator (batched after the loop)
    for relation in relation_contract:
        for subject_template in [*linked, *context]:
            for object_template in [*linked, *context]:
                if subject_template is object_template:
                    continue
                arguments = [
                    {"role": "subject", **subject_template},
                    {"role": "object", **object_template},
                ]
                if not any(argument["kind"] == "linked" for argument in arguments):
                    continue
                if not _relation_arguments_are_legal(relation, arguments, relation_contract):
                    continue
                if (
                    subject_template["kind"] == object_template["kind"] == "linked"
                    and occurrences[subject_template["occurrence_id"]].get("decision_id")
                    == occurrences[object_template["occurrence_id"]].get("decision_id")
                ):
                    continue
                arguments = _remap_to_lexically_groundable_siblings(
                    document, arguments, occurrences, contexts, relation, relation_contract,
                )
                quote, span, _, error, anchor_kind = _derived_relation_anchor(
                    document, arguments, occurrences, contexts, relation, relation_contract,
                )
                if error is not None or span is None:
                    continue
                if not all(_argument_is_grounded(argument, document, span, occurrences)
                           for argument in arguments):
                    continue
                # Proximity cap for causes_or_explains: only a local (<=1-clause-apart) anchor can
                # carry a causal assertion; a wide anchor means the pair is whole-note-apart
                # co-occurrence, not causation. Applied before cue/escalation so neither path can
                # recover it. See _CAUSAL_LOCAL_ANCHOR_KINDS.
                if (relation == "causes_or_explains"
                        and anchor_kind not in _CAUSAL_LOCAL_ANCHOR_KINDS):
                    continue
                cue_ok = _relation_quote_has_lexical_cue_support(
                    relation, quote, arguments, relation_contract,
                    allow_adjacent_clauses=True,
                    allow_plan_section=anchor_kind in {"plan_section", "problem_block", "speaker_turn"},
                )
                key = _relation_fact_key(relation, arguments, occurrences)
                # For _JUDGE_GATED_RELATIONS the lexical cue is necessary but NOT sufficient: its
                # block-level cue match accepts every condition x condition pair sharing a causal
                # word, so a cue-ok pair still requires judge confirmation (validated: 14 -> 1 on
                # D2N002). With no escalator this falls back to cue-only, preserving that mode.
                judge_gated = escalator is not None and relation in _JUDGE_GATED_RELATIONS
                if cue_ok and not judge_gated:
                    opportunities.setdefault(key, _opportunity_record(
                        relation, key, anchor_kind, arguments, span, recovered=False))
                elif escalator is not None and key not in pending:
                    # cue-miss (any relation) or cue-ok judge-gated relation: the escalator decides,
                    # unless a deterministic positive-structure rule proves the pair non-assertive
                    # (coordinated list siblings / explicitly-negated argument) -- those are skipped
                    # so the accept-biased judge never turns them into junk seeds.
                    if _escalation_prefilter_reason(
                        document, arguments, occurrences, contexts, relation) is None:
                        pending[key] = (relation, quote, deepcopy(arguments), anchor_kind, list(span))

    # NO-REGRESSION INVARIANT: escalator=None leaves `pending` unconsulted, so the result is exactly
    # the cue gate. For non-judge-gated relations the escalator only sees cue-misses and may only
    # ACCEPT, so their returned set is a superset of the cue-only set. EXCEPTION: _JUDGE_GATED_RELATIONS
    # also route their cue-OK pairs through the escalator (cue necessary, judge sufficient), so for
    # those the escalator is a precision filter and the set is NOT a cue-only superset. See
    # docs/issues/qa-builder-dept.md.
    if escalator is not None and pending:
        items = [(key, payload) for key, payload in pending.items() if key not in opportunities]
        clauses = _source_clause_spans(document)
        calls = [
            # Judge premise = ALL occurrence-clauses of both arguments (recall-oriented), falling
            # back to the anchor quote if no clause resolves. This changes only what the judge reads;
            # the anchor `span`/clause_ranges (reader + evidence) stay from `_derived_relation_anchor`.
            {"relation": r,
             "quote": _all_occurrence_judge_premise(document, a, occurrences, clauses) or q,
             "arguments": a, "anchor_kind": ak, "document": document}
            for _, (r, q, a, ak, _sp) in items
        ]
        judge_batch = getattr(escalator, "judge_batch", None)
        verdicts = (judge_batch(calls) if callable(judge_batch)
                    else [bool(escalator(**call)) for call in calls])
        for (key, (relation, _q, arguments, anchor_kind, span)), accept in zip(items, verdicts):
            if accept:
                opportunities.setdefault(key, _opportunity_record(
                    relation, key, anchor_kind, arguments, span, recovered=True))
    return [opportunities[key] for key in sorted(opportunities)]


# Gleaning+repair rejection taxonomy (approved 2026-07-16, docs/plans/qa-relation-gleaning-repair.md).
# A rejected relation is a repair target iff its reason is FIXABLE by re-authoring the proposal;
# every other reason is 100% legitimate (genuine absence/invalidity), data-owned, reader-owned, or
# infra/malformed and is EXCLUDED. `three_point_gate_failed` is NOT here: it is fixable only when the
# relation is ambiguous (answer_competing), which the ambiguous-target rule already catches; its
# lattice_level_suspect / representative_unreadable variants are data/reader-owned (excluded).
_GLEANING_FIX_HINTS: dict[str, str] = {
    "invalid_evidence":
        "Subject and object are stated in the same problem discussion but not the same clause; anchor "
        "the relation at the mention where they co-locate (the entity may be named again nearby).",
    "invalid_evidence_occurrence":
        "Anchor the relation at the specific mention where the subject and object co-locate.",
    "protected_locator":
        "Rephrase the question to reference the other argument only by its listed generalization "
        "level, never a raw source surface.",
    "protected_answer":
        "Give the accepted answer as one of the answered argument's listed generalization levels, "
        "not a raw source surface.",
    "answer_leakage":
        "Rewrite the question so it shares no meaningful word with the accepted answer.",
    "hedged_relation":
        "The source states this conditionally/as a plan; phrase the question conditionally "
        "(may / might / would / if) so it is not asserted as already done.",
    "literal_will_be_substituted":
        "That literal is a detected controlled span; reference it by its S-label as a linked argument.",
    "placeholder_answerable":
        "The answer is recoverable even at the placeholder floor; choose a more specific, "
        "discriminative answer level.",
    "floor_answerable":
        "The answer is recoverable even at the placeholder floor; choose a more specific, "
        "discriminative answer level.",
    "invalid_question":
        "Re-author a well-formed question that asks for the answered argument with a generic slot word.",
    "invalid_property":
        "The accepted answer must be copied VERBATIM from the answered span's listed levels (shown "
        "with its S-label) -- the previous attempt used a raw term that is not one of them. Never use "
        "the source word or your own phrasing; if no listed level accurately fits, abstain.",
}
_GLEANING_FIXABLE_REASONS = frozenset(_GLEANING_FIX_HINTS)
_GLEANING_AMBIGUOUS_HINT = (
    "More than one candidate answer of this type appears in the subject's problem discussion, so a "
    "free-form reader cannot uniquely answer. Re-author the question so exactly one answer is correct "
    "(add a distinguishing detail from the source), or abstain if it cannot be made unique."
)
_GLEANING_MISSED_HINT = (
    "This subject/object pair is source-supported (see its evidence card) but no relation was emitted "
    "for it. Emit the relation if the source states it, else leave it."
)
# A literal->linked relation whose literal could not be confirmed for the paired condition: the
# literal is likely attached to the wrong problem. Its real evidence clauses are shown so the teacher
# can re-pair it. (Reason not in _GLEANING_FIX_HINTS: it is detected structurally below, not by reason.)
_MISPAIRED_LITERAL_REASONS = frozenset({"unknown_context_literal", "three_point_gate_failed"})
_GLEANING_MISPAIRED_HINT = (
    "This test/procedure literal was paired with a condition the source does not state it is for. Find "
    "where the literal appears in the source and re-pair it with THAT problem's condition (see its "
    "evidence cards), or abstain if no condition is actually stated for it."
)
_GLEANING_TARGET_PRIORITY = {"ambiguous": 3, "fixable": 2, "missed": 1}

# A linked argument's `surface` is a controlled entity's raw source text; rejection records are
# shareable diagnostics, so it is stripped before an argument identity is stamped onto one
# (occurrence_id / support_property / runtime_type carry the fact key). A context argument's
# `literal` is UNCONTROLLED source text (kept verbatim in doc_p by definition), so it is NOT a
# protected leak and is retained -- the gleaning repair pass needs it to name a mispaired literal.
_ARGUMENT_RAW_SOURCE_FIELDS = frozenset({"surface"})


def _rejection_safe_arguments(arguments: Sequence[Mapping]) -> list[dict]:
    """Identity fields of `arguments` minus any raw-source echo (surface/literal)."""
    return [
        {key: value for key, value in dict(argument).items()
         if key not in _ARGUMENT_RAW_SOURCE_FIELDS}
        for argument in arguments
    ]


def _reader_outcome_route(rejection: Mapping, reader_threshold: float = 1.0) -> str | None:
    """Route a reader-gated rejection by its stored three-point scores.

    * "lattice_suspect" -- the orig/rep verdicts DISAGREE (readable raw but not at the required
      level, or vice versa), or only the placeholder render reads: a generalization-level /
      chain-alias data defect, not something a teacher can re-author. Any authorship.
    * "no_relation" -- a DETERMINISTIC-stage relation the reader confirmed NOWHERE (orig, rep,
      placeholder all below threshold) after the stage exhausted its direction/level/compound/set
      ladder: reader-verified co-occurrence junk from the recall-oriented miner. Teacher-authored
      rejections are NOT routed here -- their questions carried real judgment, so their existing
      repair paths stand.
    * None -- no exclusion (no reader scores, or a pattern the repair taxonomy handles).
    """
    if str(rejection.get("detail_reason")) == "finer_level_unreadable":
        # hard finer-level check: the fact reads at its supported level but a finer band level
        # does not -- lattice-owned data, unfixable by re-authoring; never a repair target
        return "lattice_suspect"
    evidence = rejection.get("evidence") or {}
    scores = (evidence.get("validation") or {}).get("scores") or {}
    if not scores:
        return None
    original = float(scores.get("original", 0.0)) >= reader_threshold
    representative = float(scores.get("representative", 0.0)) >= reader_threshold
    placeholder = float(scores.get("placeholder", 0.0)) >= reader_threshold
    if original != representative or (not original and not representative and placeholder):
        return "lattice_suspect"
    if (not original and not representative and not placeholder
            and str(evidence.get("teacher_id")) == "deterministic"):
        return "no_relation"
    return None


def _gleaning_targets(
    document: str,
    kept_relations: Sequence[Mapping],
    rejections: Sequence[Mapping],
    opportunities: Sequence[Mapping],
    occurrences: Mapping[str, Mapping],
    reader_threshold: float = 1.0,
) -> list[dict]:
    """Compute the gleaning+repair target set from the primary pass (diagnostic/planning only).

    Targets, deduplicated by decision-level fact key with priority ambiguous > fixable > missed:
      * ambiguous  -- a REJECTED relation with a co-valid same-type answer in scope (a KEPT relation
                      already answered uniquely at the reader gate, and its multi-answer coverage is
                      handled deterministically by reverse-framing -- so it is never re-authored);
      * fixable    -- a rejection whose reason is in the FIXABLE taxonomy;
      * missed     -- a source-supported opportunity the teacher never proposed.
    `kept_relations`/`rejections` must carry `evidence.arguments` (compiled form). Conservative:
    only certainly-legitimate reasons are dropped, so a fixable relation is never silently excluded.
    """
    def _args(row: Mapping) -> list[dict]:
        return list((row.get("evidence") or {}).get("arguments") or [])

    def _fact_key(row: Mapping) -> tuple | None:
        args = _args(row)
        if len(args) != 2:
            return None
        try:
            return _relation_fact_key(str(row.get("relation", "")), args, occurrences)
        except ValueError:
            return None

    def _subject_and_answer(row: Mapping) -> tuple[Mapping | None, Mapping | None, Mapping]:
        args = _args(row)
        if len(args) != 2:
            return None, None, {}
        role = str(row.get("answer_role") or "object")
        answer = args[0] if role == "subject" else args[1]
        subject = args[1] if role == "subject" else args[0]
        return subject, answer, (row.get("answer_target") or {})

    # PROPOSED = every pair fact some row attempted (kept or rejected), with compound rows
    # (set-valued / compound-locator / multi-literal, >2 args) decomposed into their pair keys --
    # else a compound attempt's pairs would resurrect as "missed" targets.
    proposed_keys = _pair_fact_keys([*kept_relations, *rejections], occurrences)
    # Reader-outcome routing: a rejection whose three-point scores say "lattice data defect" or
    # "reader-verified no-relation" is EXCLUDED from repair targeting (see _reader_outcome_route)
    # -- but stays in proposed_keys above, so the excluded fact can never resurrect as "missed".
    repairable_rejections = [
        row for row in rejections if _reader_outcome_route(row, reader_threshold) is None
    ]
    # Pair-level fact keys COVERED by a kept row. A fact can be attempted several times per build
    # (forward teacher proposal, deterministic reverse flip, compound row): a rejection for one
    # attempt must not re-target a fact another attempt already kept.
    kept_pair_keys = _pair_fact_keys(kept_relations, occurrences)
    targets: dict[tuple, dict] = {}

    def _fallback_key(row: Mapping) -> tuple:
        # Conservative: a fixable/ambiguous reject whose arguments never compiled (e.g.
        # invalid_evidence with no grounded args) must still be a target -- never a
        # false-negative -- so key it by its own rejection identity when the fact key is None.
        return ("reject", str(row.get("rejection_id") or row.get("attempt_hash")
                             or row.get("proposal_hash") or _stable_hash(dict(row))))

    def _is_self_pair(arguments: Sequence[Mapping]) -> bool:
        # A degenerate target whose two linked arguments resolve to the SAME decision (e.g. a
        # rejected teacher proposal pairing "knees" with "knees"): never a real relation, so it
        # must not be handed back to the repair teacher.
        linked = [a for a in arguments if a.get("kind") == "linked"]
        if len(linked) != 2:
            return False
        decisions = {
            (occurrences.get(str(a.get("occurrence_id"))) or {}).get("decision_id")
            for a in linked
        }
        return len(decisions) == 1 and None not in decisions

    def _add(key: tuple | None, kind: str, **extra) -> None:
        if key is None:
            return
        if key in kept_pair_keys:
            return  # the fact is already covered by a kept row; re-authoring it wastes a call
        if _is_self_pair(extra.get("arguments") or []):
            return
        existing = targets.get(key)
        if existing and _GLEANING_TARGET_PRIORITY[existing["kind"]] >= _GLEANING_TARGET_PRIORITY[kind]:
            return
        targets[key] = {"fact_key": key, "kind": kind, **extra}

    # ambiguous: a REJECTED relation with a co-valid same-type answer in the subject's block. Kept
    # relations are excluded -- they already answered uniquely, so re-authoring them wastes a teacher
    # call and re-emits the same fact; their multi-answer siblings are recovered by reverse-framing.
    for row in repairable_rejections:
        competing = list((row.get("evidence") or {}).get("answer_competing") or [])
        if not competing:
            subject, answer, target = _subject_and_answer(row)
            if subject is not None and answer is not None:
                competing = _answer_competing_surfaces(
                    document, subject, answer, target, occurrences)
        if competing:
            _add(_fact_key(row) or _fallback_key(row), "ambiguous", relation=row.get("relation"),
                 hint=_GLEANING_AMBIGUOUS_HINT, competing=competing, arguments=_args(row))

    # fixable: a rejection whose reason is in the FIXABLE taxonomy
    for row in repairable_rejections:
        reason = str(row.get("detail_reason") or row.get("reason") or "")
        if reason == "protected_locator" and str(
                (row.get("evidence") or {}).get("leak_source") or "") == "context_literal":
            continue  # the leak sits inside an unrecolorable context literal: no author -- the
            # repair teacher included -- can phrase around it, so it is dead weight, not fixable
        if reason in _GLEANING_FIXABLE_REASONS:
            _add(_fact_key(row) or _fallback_key(row), "fixable", relation=row.get("relation"),
                 reason=reason, hint=_GLEANING_FIX_HINTS[reason], arguments=_args(row))

    # hedge modality: the hedge guard is a non-blocking diagnostic (not a reject) -- but a
    # hedge-flagged relation the READER then could not confirm is routed back to repair to
    # re-phrase the question conditionally. Restores the pre-demotion repair path, now gated on
    # actually failing the reader (relations the reader confirms are kept, not needlessly redrawn).
    for row in repairable_rejections:
        reason = str(row.get("detail_reason") or row.get("reason") or "")
        modality = (row.get("evidence") or {}).get("modality_diagnostics") or []
        if reason == "three_point_gate_failed" and "hedged_source_definite_question" in modality:
            _add(_fact_key(row) or _fallback_key(row), "fixable", relation=row.get("relation"),
                 reason="hedged_relation", hint=_GLEANING_FIX_HINTS["hedged_relation"],
                 arguments=_args(row))

    # mispaired context literal: a literal->linked relation the grounding/reader could not confirm for
    # the paired condition. Structural detection (one linked + one context arg), so the failed relation
    # goes back to the teacher with the literal's real evidence to re-pair -- not silently re-paired here.
    for row in repairable_rejections:
        reason = str(row.get("detail_reason") or row.get("reason") or "")
        if reason not in _MISPAIRED_LITERAL_REASONS:
            continue
        args = _args(row)
        linked = [a for a in args if a.get("kind") == "linked"]
        context = [a for a in args if a.get("kind") == "context" and a.get("literal")]
        if len(linked) == 1 and len(context) == 1:
            _add(_fact_key(row) or _fallback_key(row), "fixable", relation=row.get("relation"),
                 reason="mispaired_context_literal", hint=_GLEANING_MISPAIRED_HINT, arguments=args)

    # missed: a source-supported opportunity the teacher never proposed
    for opportunity in opportunities:
        if opportunity.get("fact_key") not in proposed_keys:
            _add(opportunity.get("fact_key"), "missed", relation=opportunity.get("relation"),
                 hint=_GLEANING_MISSED_HINT, arguments=list(opportunity.get("arguments") or []),
                 evidence_span=opportunity.get("evidence_span"))

    # Order by source locality so problem-block-adjacent targets cluster together (a note's targets
    # for one problem sit in one contiguous source range). Downstream this is sliced into teacher
    # batches, so co-locating a problem's targets lets the teacher's own "emit each fact once"
    # dedupe the cross-target near-duplicates (e.g. "thyroid panel" / "thyroid labs", or the same
    # drug attributed to a symptom and its condition) that arbitrary ordering scattered across calls.
    def _target_source_pos(target: Mapping) -> int:
        positions: list[int] = []
        for argument in target.get("arguments") or []:
            if argument.get("kind") == "linked":
                occurrence = occurrences.get(str(argument.get("occurrence_id"))) or {}
                if isinstance(occurrence.get("start"), int):
                    positions.append(int(occurrence["start"]))
            elif isinstance(argument.get("start"), int):
                positions.append(int(argument["start"]))
        evidence_span = target.get("evidence_span")
        if evidence_span and len(evidence_span) == 2:
            positions.append(int(evidence_span[0]))
        return min(positions) if positions else len(document)

    return [
        targets[key]
        for key in sorted(targets, key=lambda k: (_target_source_pos(targets[k]), repr(k)))
    ]


def _validate_relation_escalation_policy(policy: Mapping | None) -> dict | None:
    """Validate a fully manifest-pinned escalation policy; absent means disabled."""
    if policy is None:
        return None
    if not isinstance(policy, Mapping):
        raise ValueError("relation_escalation_policy must be a mapping")
    required = {"version", "min_opportunities", "coverage_fraction", "scope_caps"}
    if set(policy) != required or not str(policy.get("version", "")).strip():
        raise ValueError("relation_escalation_policy must pin version and all target fields")
    normalized = {"version": str(policy["version"])}
    for field in ("min_opportunities", "coverage_fraction", "scope_caps"):
        values = policy[field]
        if not isinstance(values, Mapping) or set(values) != set(_RELATION_ESCALATION_SCOPES):
            raise ValueError(f"relation_escalation_policy.{field} must cover every relation scope")
        normalized[field] = dict(values)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in normalized["min_opportunities"].values()
    ):
        raise ValueError("relation escalation min_opportunities must be non-negative integers")
    if any(
        not isinstance(value, Real) or not 0.0 < float(value) <= 1.0
        for value in normalized["coverage_fraction"].values()
    ):
        raise ValueError("relation escalation coverage_fraction must be in (0, 1]")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in normalized["scope_caps"].values()
    ) or sum(normalized["scope_caps"].values()) > RELATION_TEACHER_MAX_RELATIONS:
        raise ValueError("relation escalation scope_caps must be non-negative and fit the relation cap")
    return normalized


def relation_escalation_targets(
    opportunity_counts: Mapping[str, int], policy: Mapping,
) -> dict[str, int]:
    """Manifest-pinned target count by relation scope."""
    validated = _validate_relation_escalation_policy(policy)
    if validated is None:
        raise ValueError("relation escalation policy is required")
    return {
        scope: (
            0 if int(opportunity_counts.get(scope, 0)) < validated["min_opportunities"][scope]
            else min(
                validated["scope_caps"][scope],
                ceil(validated["coverage_fraction"][scope] * int(opportunity_counts.get(scope, 0))),
            )
        )
        for scope in _RELATION_ESCALATION_SCOPES
    }


def needs_relation_escalation(
    kept_counts: Mapping[str, int], targets: Mapping[str, int],
) -> bool:
    return any(
        int(kept_counts.get(scope, 0)) < int(targets.get(scope, 0))
        for scope in _RELATION_ESCALATION_SCOPES
    )


def merge_kept_relation_rows(
    primary_rows: Sequence[Mapping],
    secondary_rows: Sequence[Mapping],
    occurrences: Mapping[str, Mapping],
) -> tuple[list[dict], dict[str, int]]:
    """Merge post-gate rows by fact identity, retaining the primary formulation."""
    primary_by_key: dict[tuple, dict] = {}
    for row in primary_rows:
        primary_by_key.setdefault(_compiled_relation_fact_key(row, occurrences), dict(row))
    secondary_by_key: dict[tuple, dict] = {}
    for row in secondary_rows:
        secondary_by_key.setdefault(_compiled_relation_fact_key(row, occurrences), dict(row))
    merged = list(primary_by_key.values())
    merged.extend(row for key, row in secondary_by_key.items() if key not in primary_by_key)
    disposition = {
        "primary_only": sum(key not in secondary_by_key for key in primary_by_key),
        "primary_preferred": sum(key in secondary_by_key for key in primary_by_key),
        "secondary_only": sum(key not in primary_by_key for key in secondary_by_key),
    }
    return merged, disposition


def _promote_context_literals_on_detected_entities(
    arguments: list[dict], proposal: Mapping,
    occurrences: Mapping[str, Mapping], decisions: Mapping[str, Mapping],
) -> tuple[list[dict], Mapping]:
    """Fix B: a context literal that names a DETECTED (controlled) entity is an S-label the
    teacher passed as free text. Promote it to a linked argument on that decision -- a
    detected entity is substituted in doc_p (so its generalized level is a hideable
    answer), whereas a bare literal survives verbatim and can never pass the floor gate.
    Only promotes when the decision carries a legal generalization level; the co-locating
    occurrence is then chosen by the normal sibling remap. Universal, deterministic."""
    for argument in arguments:
        if argument.get("kind") != "context" or not argument.get("literal"):
            continue
        literal_tokens = _meaningful_tokens(str(argument["literal"]))
        if not literal_tokens:
            continue
        promoted = None
        for occurrence_id, occurrence in occurrences.items():
            if not occurrence.get("controlled", True):
                continue
            surface_tokens = _meaningful_tokens(str(occurrence.get("surface", "")))
            if not surface_tokens or not surface_tokens <= literal_tokens:
                continue
            levels = _ordered_decision_levels(decisions.get(str(occurrence.get("decision_id")), {}))
            if levels:
                promoted = (occurrence_id, occurrence, levels[0])
                break
        if promoted is None:
            continue
        occurrence_id, occurrence, level = promoted
        argument["kind"] = "linked"
        argument["occurrence_id"] = occurrence_id
        argument["surface"] = str(occurrence.get("surface", ""))
        argument["runtime_type"] = str(occurrence.get("runtime_type", ""))
        argument["support_property"] = level
        argument.pop("literal", None)
        if str(argument.get("role")) == str(proposal.get("answer_role", "object")):
            proposal = {**proposal, "accepted_answers": [level]}
    return arguments, proposal


def _answer_competing_surfaces(
    document: str,
    subject_argument: Mapping,
    answer_argument: Mapping,
    answer_target: Mapping,
    occurrences: Mapping[str, Mapping],
) -> list[str]:
    """MONITOR (diagnostic only, never rejects): surfaces of OTHER controlled entities that share
    the answered argument's runtime_type and sit in the SAME clinical problem block(s) as the
    subject condition, but belong to a different decision than the answer. A non-empty result means
    the probe "which <type> for <subject>?" has more than one co-valid answer where the reader looks,
    so a free-form reader may name a different (also-valid) one -- the non-deterministic utility
    signal to surface. Scope is the subject DECISION's problem block(s) (not the narrow relation
    anchor): the reader reads the whole problem, so a competitor anywhere in it can be surfaced.
    Deterministic: derived only from frozen detections + source problem-block segmentation."""
    if answer_argument.get("kind") == "linked":
        answer_type = canon(str(occurrences.get(answer_argument.get("occurrence_id"), {})
                                 .get("runtime_type", "")))
        answer_decision = str(answer_target.get("decision_id") or "")
    else:
        answer_type = canon(str(answer_argument.get("runtime_type", "")))
        answer_decision = ""  # a context literal has no decision to exclude by
    if not answer_type or subject_argument.get("kind") != "linked":
        return []
    subject_decision = str(
        occurrences.get(subject_argument.get("occurrence_id"), {}).get("decision_id") or "")
    if not subject_decision:
        return []
    blocks = _problem_blocks(document)

    def block_of(start: int, end: int) -> tuple[int, int] | None:
        return next(((lo, hi) for lo, hi in blocks if lo <= start and end <= hi), None)

    scope_ranges = {
        block_of(int(o["start"]), int(o["end"]))
        for o in occurrences.values()
        if str(o.get("decision_id") or "") == subject_decision
        and isinstance(o.get("start"), int) and isinstance(o.get("end"), int)
    }
    scope_ranges.discard(None)
    if not scope_ranges:  # no problem-block structure -> whole document
        scope_ranges = {(0, len(document))}
    competing: dict[str, str] = {}
    for occurrence in occurrences.values():
        if not occurrence.get("controlled", True):
            continue
        if canon(str(occurrence.get("runtime_type", ""))) != answer_type:
            continue
        start, end = occurrence.get("start"), occurrence.get("end")
        if not (isinstance(start, int) and isinstance(end, int)):
            continue
        if not any(lo <= start and end <= hi for lo, hi in scope_ranges):
            continue
        decision = str(occurrence.get("decision_id") or "")
        if decision and decision != answer_decision:
            competing[decision] = str(occurrence.get("surface", ""))
    return sorted(competing.values())


# Deterministic reverse-orientation question templates (condition-subject relations only). The
# locator is the object's generalized level; the answer is the subject condition's level, asked with
# a generic answer-type word ("medical condition") so it never leaks. Mirrors the context-literal
# reverse questions the teacher already authors.
_REVERSE_FRAME_TEMPLATES = {
    "prescribed_with": "For what medical condition was the {locator} prescribed?",
    "tests_for": "For what medical condition was the {locator} ordered?",
    "procedure_for": "For what medical condition was the {locator} performed?",
}


def _reverse_framed_proposals(
    proposals: Sequence[Mapping],
    span_labels: Mapping[str, str],
    occurrences: Mapping[str, Mapping],
) -> list[dict]:
    """Additive reverse-orientation QAs that recover ambiguity losses without touching the teacher.

    A forward relation "Which <type> for the <subject>?" with >=2 same-type objects for one subject
    is unanswerable. When an object is tied to EXACTLY ONE subject in the doc (reverse-unique), emit
    a reverse QA: the object is the question locator, the subject is the answer (answer_role=subject)
    -- the same directional fact, uniquely answerable. Roles are PRESERVED (subject stays the
    condition, object stays the treatment), so the argument-type contract still holds and
    answer-type/leakage/anchor logic stay correct. Never replaces a forward proposal; the reader gate
    keeps whichever is answerable. Deterministic: no teacher call, no model judgment of uniqueness."""
    def linked_decision(arg: Mapping) -> str | None:
        occ_id = arg.get("occurrence_id") or span_labels.get(str(arg.get("span_label")))
        occ = occurrences.get(str(occ_id))
        return str(occ["decision_id"]) if occ and occ.get("decision_id") is not None else None

    def obj_identity(arg: Mapping) -> tuple | None:
        if arg.get("kind") == "context":
            lit = canon(str(arg.get("literal", "")))
            return ("lit", lit) if lit else None
        dec = linked_decision(arg)
        return ("dec", dec) if dec else None

    forwards: list = []
    fwd_objs: dict[tuple, set] = {}
    for p in proposals:
        rel = str(p.get("relation", ""))
        if rel not in _REVERSE_FRAME_TEMPLATES or str(p.get("answer_role", "object")) == "subject":
            continue
        args = p.get("arguments") or []
        subj = next((a for a in args if a.get("role") == "subject"), None)
        obj = next((a for a in args if a.get("role") == "object"), None)
        if subj is None or obj is None or subj.get("kind") != "linked":
            continue  # subject must be a linked condition to be a valid answer
        subj_dec, obj_id = linked_decision(subj), obj_identity(obj)
        if subj_dec is None or obj_id is None:
            continue
        forwards.append((p, rel, subj_dec, obj_id, subj, obj))
        fwd_objs.setdefault((rel, subj_dec), set()).add(obj_id)

    variants, seen = [], set()
    for p, rel, subj_dec, obj_id, subj, obj in forwards:
        if len(fwd_objs[(rel, subj_dec)]) < 2:
            continue  # only ambiguous groups (>=2 objects for the subject); flip EVERY object and
            # let the reader keep the ones a source-supported single subject makes answerable.
        if obj_id == ("dec", subj_dec):
            continue  # self-pair: the object resolves to the SAME decision as the subject
        key = (rel, subj_dec, obj_id)
        if key in seen:
            continue
        seen.add(key)
        subj_level = subj.get("support_property")
        obj_level = obj.get("support_property") or obj.get("literal")
        if not subj_level or not obj_level:
            continue
        variant = dict(p)
        variant["arguments"] = [dict(subj), dict(obj)]  # order [subject, object] preserved
        variant["answer_role"] = "subject"
        variant["accepted_answers"] = [str(subj_level)]
        variant["question"] = _REVERSE_FRAME_TEMPLATES[rel].format(locator=str(obj_level))
        variant["scoring_contract"] = {"kind": "semantic_qa", "match": "fact_score"}
        variant["_reverse_framed"] = True
        variants.append(variant)
    return variants


# Deterministic literal->span (reverse) templates: the literal object(s) are the question LOCATOR,
# the controlled condition is the ANSWER (answer_role=subject, answer-type word first so the extractive
# reader names the diagnosis, not the tests). A >=2-literal compound locator pins a single condition
# where one generic literal ("labs") cannot. Validated on 5 docs (scripts/spikes/reverse_flip_probe.py).
_LITERAL_REVERSE_TEMPLATES = {
    "tests_for": "What single medical condition or diagnosis were {locators} ordered to evaluate or "
                 "treat? Name only the condition.",
    "prescribed_with": "What single medical condition or diagnosis were {locators} prescribed to "
                       "treat? Name only the condition.",
    "procedure_for": "What single medical condition or diagnosis were {locators} performed or planned "
                     "to treat? Name only the condition.",
    "contraindicated_because_of": "What single medical condition makes {locators} contraindicated or "
                                  "unsuitable? Name only the condition.",
    "causes_or_explains": "What single medical condition causes or explains {locators}? Name only the "
                          "condition.",
}


_LOCATOR_LEAD_STRIP = re.compile(
    r"^(?:order(?:ed|ing)?|check(?:ed|ing)?|obtain(?:ed|ing)?|get|perform(?:ed|ing)?|plan(?:ned)?|"
    r"start(?:ed|ing)?|refer(?:ral)?(?:\s+to)?|the|a|an|some|his|her|your|our|my)\s+", re.I)


def _clean_locator(text: str) -> str:
    """Strip leading order/refer verbs and articles so a locator reads as the entity itself
    ('order a thyroid panel' -> 'thyroid panel'). Iterated to peel a verb+article stack."""
    previous = None
    text = text.strip()
    while text and text != previous:
        previous = text
        text = _LOCATOR_LEAD_STRIP.sub("", text).strip()
    return text or previous or ""


def _display_locators(literals: Sequence[str]) -> list[str]:
    """Clean + drop near-duplicate supersets (keep 'physical therapy' over 'referral to physical
    therapy'), so the compound locator is concise and non-redundant."""
    cleaned = list(dict.fromkeys(filter(None, (_clean_locator(literal) for literal in literals))))
    cleaned.sort(key=len)  # prefer the shortest form of a near-duplicate
    display: list[str] = []
    for candidate in cleaned:
        tokens = set(canon(candidate).split())
        if tokens and any(set(canon(kept).split()) <= tokens for kept in display):
            continue  # a shorter kept locator already covers this one
        display.append(candidate)
    return display


def _join_locators(items: Sequence[str]) -> str:
    unique = list(dict.fromkeys(items))
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return ", ".join(unique[:-1]) + ", and " + unique[-1]


def _literal_reverse_groups(
    opportunities: Sequence[Mapping],
    occurrences: Mapping[str, Mapping],
    *,
    judge_recovered_only: bool = True,
) -> dict[tuple, dict]:
    """(relation, condition decision_id) -> literal group eligible for a literal->span reverse QA.
    `judge_recovered_only=True` is the original seed (judge-recovered span_literal pairs only);
    False widens to every accepted opportunity (the deterministic stage's seed). Each literal
    entry carries its pair's fact_key so callers can drop already-kept facts."""
    groups: dict[tuple, dict] = {}
    for opportunity in opportunities:
        if judge_recovered_only and not opportunity.get("recovered_by_escalation"):
            continue
        if opportunity.get("scope") != "span_literal":
            continue
        relation = str(opportunity.get("relation", ""))
        if relation not in _LITERAL_REVERSE_TEMPLATES:
            continue
        arguments = opportunity.get("arguments") or []
        condition = next(
            (arg for arg in arguments if arg.get("kind") == "linked" and arg.get("occurrence_id")), None)
        literal = next(
            (arg for arg in arguments if arg.get("kind") == "context" and arg.get("literal")), None)
        if condition is None or literal is None:
            continue
        occurrence = occurrences.get(str(condition.get("occurrence_id")))
        if (occurrence is None or occurrence.get("decision_id") is None
                or not occurrence.get("controlled", True)):
            continue
        try:
            start, end = int(literal["start"]), int(literal["end"])
        except (KeyError, TypeError, ValueError):
            continue
        group = groups.setdefault(
            (relation, str(occurrence["decision_id"])),
            {"occurrence_id": str(condition["occurrence_id"]), "literals": {}})
        group["literals"].setdefault(canon(str(literal["literal"])), {
            "literal": str(literal["literal"]), "start": start, "end": end,
            "fact_key": opportunity.get("fact_key"),
        })
    return groups


def _literal_reverse_row(
    document: str,
    relation: str,
    decision_id: str,
    group: Mapping,
    occurrences: Mapping[str, Mapping],
    answer_level: str,
) -> dict:
    """One literal->span reverse context-assertion row: the group's literal(s) are the question
    locator, the controlled condition answers at `answer_level`."""
    occurrence_id = group["occurrence_id"]
    occurrence = occurrences[occurrence_id]
    literals = list(group["literals"].values())
    display = _display_locators([entry["literal"] for entry in literals])
    question = _LITERAL_REVERSE_TEMPLATES[relation].format(
        locators=_join_locators(display or [entry["literal"] for entry in literals]))
    condition_span = (int(occurrence["start"]), int(occurrence["end"]))
    ranges = [condition_span] + [(entry["start"], entry["end"]) for entry in literals]
    arguments = [
        {"role": "subject", "kind": "linked", "occurrence_id": occurrence_id,
         "runtime_type": occurrence.get("runtime_type"), "support_property": answer_level},
        *[{"role": "object", "kind": "context", "literal": entry["literal"],
           "start": entry["start"], "end": entry["end"]}
          for entry in literals],
    ]
    lo, hi = min(span[0] for span in ranges), max(span[1] for span in ranges)
    return {
        "family": "context", "scope": "linked", "subtype": "contextual_relation",
        "relation": relation,
        "occurrence_ids": [occurrence_id],
        "group_id": "literal_reverse:" + relation + ":" + _stable_hash(
            [decision_id, sorted(group["literals"])]),
        "question": question,
        "accepted_values": [answer_level],
        "answer_target": {"kind": "linked_decision", "decision_id": decision_id,
                          "required_property": answer_level},
        "answer_type": _relation_answer_type_hint(relation, "subject"),
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
        "decision_requirements": {decision_id: answer_level},
        "evidence": {
            "authority": "source_document",
            "arguments": arguments,
            "argument_spans": {occurrence_id: [condition_span[0], condition_span[1]]},
            "reader_turns": _source_turns_for_ranges(document, ranges),
            "source_span": {"start": lo, "end": hi, "quote_hash": _stable_hash(document[lo:hi])},
            "teacher_id": "deterministic", "run_id": "literal_reverse",
        },
    }


def _literal_reverse_assertions(
    document: str,
    opportunities: Sequence[Mapping],
    occurrences: Mapping[str, Mapping],
    decisions_by_id: Mapping[str, Mapping],
) -> list[dict]:
    """Deterministic literal->span reverse context assertions from JUDGE-ACCEPTED span_literal
    opportunities: for each (relation, controlled condition) group, the literal object(s) become the
    question locator and the condition is the answer (answer_role=subject). Compound (>=2 literals)
    disambiguates a single condition where one generic literal cannot; a single specific literal still
    yields a QA. The condition is controlled, so the placeholder render hides it and the three-point
    gate holds by construction. No teacher call. Gated downstream by validate_candidate_rows."""
    rows = []
    for (relation, decision_id), group in _literal_reverse_groups(
            opportunities, occurrences).items():
        decision = decisions_by_id.get(decision_id)
        levels = _ordered_decision_levels(decision) if decision else []
        if not levels:
            continue  # no legal generalization -> nothing to answer with
        rows.append(_literal_reverse_row(
            document, relation, decision_id, group, occurrences, levels[0]))
    return rows


# Deterministic relation stage (opt-in, between the primary teacher pass and gleaning): mined
# opportunity pairs become template QAs without a teacher call. Forward = subject locator in the
# question, object answers; reverse = object locator, subject answers. The answered argument must be
# a controlled span (hidden by the placeholder render), so template choice is what makes the
# three-point gate satisfiable by construction.
_FORWARD_RELATION_TEMPLATES = {
    "prescribed_with": "Which medication was prescribed or used to treat the {locator}?",
    "tests_for": "Which test or investigation was ordered to evaluate the {locator}?",
    "procedure_for": "Which procedure or therapy was performed or planned for the {locator}?",
    "contraindicated_because_of": "Which medical condition makes the {locator} contraindicated "
                                  "or unsuitable?",
    "causes_or_explains": "Which symptom, finding, or condition does the {locator} cause or explain?",
}

# The stage's reverse templates: the ambiguity-repair set plus the two relations it never needed.
# _REVERSE_FRAME_TEMPLATES itself is untouched -- its membership gates the post-gleaning
# reverse-framing pass, which must stay byte-identical when the stage is off.
_DETERMINISTIC_REVERSE_TEMPLATES = {
    **_REVERSE_FRAME_TEMPLATES,
    "contraindicated_because_of": "Which medication or treatment must be avoided because of "
                                  "the {locator}?",
    "causes_or_explains": "Which underlying medical condition causes or explains the {locator}?",
}

# Bare type-word levels are useless answers (and usually echo the question's answer-type word):
# the coarsest->finest answer-level search skips them instead of spending reader calls.
_DETERMINISTIC_LEVEL_SKIP = frozenset(canon(level) for level in (
    # conditions
    "medical condition", "health condition", "condition", "medical problem", "diagnosis",
    "disease", "disease of anatomical entity", "clinical finding", "finding", "clinical symptom", "symptom",
    # procedures
    "medical procedure", "surgical procedure", "diagnostic procedure", "therapeutic procedure",
    "procedure", "therapy", "treatment",
    # diagnostics
    "diagnostic test", "test", "investigation", "medical device",
    # drugs
    "therapeutic agent", "medication", "drug", "medicine",
    "dietary supplement", "pharmaceutical compound", "chemical substance", "prescription medication",
))


def _deterministic_answer_levels(decision: Mapping) -> list[str]:
    """Answer-level trial order for the deterministic stage: coarsest -> finest, so the FIRST
    three-point-gate pass is the coarsest supported level (same semantics as the teacher's
    supported-level prior). Degenerate type-word levels are skipped; a decision whose every
    level is degenerate still gets one trial at its finest level."""
    levels = _ordered_decision_levels(decision)  # most-specific -> coarsest
    trials = [level for level in reversed(levels)
              if canon(level) not in _DETERMINISTIC_LEVEL_SKIP]
    return trials or levels[:1]


def _deterministic_relation_plans(
    opportunities: Sequence[Mapping],
    occurrences: Mapping[str, Mapping],
    decisions: Mapping[str, Mapping],
) -> list[dict]:
    """Direction plans for the deterministic stage's span<->span pairs (literal pairs are the
    literal-reverse path's job). Grouped by (relation, subject decision): a single span object
    generates FORWARD (object answers) with a REVERSE fallback; >=2 span objects each generate
    REVERSE only (subject answers) -- the forward question is ambiguous by construction. Both
    arguments must be controlled decisions with legal levels (all span-inventory args are)."""
    def controlled_levels(argument: Mapping) -> bool:
        occurrence = occurrences.get(str(argument.get("occurrence_id")))
        if occurrence is None or occurrence.get("decision_id") is None:
            return False
        if not occurrence.get("controlled", True):
            return False
        decision = decisions.get(str(occurrence["decision_id"]))
        return bool(decision and _ordered_decision_levels(decision))

    groups: dict[tuple, dict[str, dict]] = {}
    for opportunity in opportunities:
        if opportunity.get("scope") != "span_span":
            continue
        relation = str(opportunity.get("relation", ""))
        if relation not in _FORWARD_RELATION_TEMPLATES:
            continue
        arguments = opportunity.get("arguments") or []
        subject = next((a for a in arguments if a.get("role") == "subject"), None)
        obj = next((a for a in arguments if a.get("role") == "object"), None)
        if subject is None or obj is None:
            continue
        if not (controlled_levels(subject) and controlled_levels(obj)):
            continue
        subject_decision = str(occurrences[str(subject["occurrence_id"])]["decision_id"])
        object_decision = str(occurrences[str(obj["occurrence_id"])]["decision_id"])
        groups.setdefault((relation, subject_decision), {}).setdefault(
            object_decision, {"subject": dict(subject), "object": dict(obj)})

    plans: list[dict] = []
    for (relation, _subject_decision), by_object in sorted(groups.items()):
        directions = ["forward", "reverse"] if len(by_object) == 1 else ["reverse"]
        group_fact_keys: list[tuple] = []
        for _object_decision, pair in sorted(by_object.items()):
            try:
                fact_key = _relation_fact_key(
                    relation, [pair["subject"], pair["object"]], occurrences)
            except ValueError:
                continue
            group_fact_keys.append(fact_key)
            plans.append({
                "relation": relation,
                "fact_key": fact_key,
                "subject": pair["subject"],
                "object": pair["object"],
                "directions": list(directions),
            })
        if len(by_object) >= 2 and group_fact_keys:
            # Compound fallback for the ambiguous group, AFTER its per-object flips in plan
            # order: all object levels in one locator pin a single subject where one object
            # (tied to several subjects) cannot. Executed only while some pair is still unkept.
            pairs = [pair for _dec, pair in sorted(by_object.items())]
            plans.append({
                "relation": relation,
                "compound": True,
                "subject": pairs[0]["subject"],
                "objects": [pair["object"] for pair in pairs],
                "fact_keys": group_fact_keys,
            })
    return plans


def _foreign_protected_terms(
    occurrences: Mapping[str, Mapping], own_decision_id: str,
) -> list[str]:
    """Protected surfaces/aliases of every controlled decision EXCEPT the locator's own."""
    return [
        term
        for occurrence in occurrences.values()
        if occurrence.get("controlled", True)
        and occurrence.get("decision_id") is not None
        and str(occurrence["decision_id"]) != str(own_decision_id)
        for term in _occurrence_protected_terms(occurrence)
    ]


def _first_noncolliding_level(levels: Sequence[str], foreign_terms: Sequence[str]) -> str:
    """First level that neither equals nor contains a FOREIGN protected surface. A level that
    echoes another decision's raw surface reads as a locator for THAT span (protected_locator),
    even though it is a legal generalization of its own -- so escalate past it to the next
    coarser level. Falls back to the finest level when every level collides; compile's leak
    check remains the backstop."""
    for level in levels:
        if not any(_contains(str(level), str(term)) for term in foreign_terms):
            return str(level)
    return str(levels[0])


def _deterministic_stage_proposal(
    plan: Mapping,
    direction: str,
    answer_level: str,
    occurrences: Mapping[str, Mapping],
    decisions: Mapping[str, Mapping],
    span_label_by_decision: Mapping[str, str],
) -> dict | None:
    """A teacher-style relation proposal for one (plan, direction, answer level) trial. The
    locator argument is fixed at its finest legal level (most readable); the answered argument
    carries the trial level. Compiled by the normal compile_relations path, so every teacher
    guard (anchor, cue, leakage repair, protected locator) applies unchanged."""
    subject, obj = dict(plan["subject"]), dict(plan["object"])
    answer_argument, locator_argument = (obj, subject) if direction == "forward" else (subject, obj)
    templates = (_FORWARD_RELATION_TEMPLATES if direction == "forward"
                 else _DETERMINISTIC_REVERSE_TEMPLATES)
    locator_decision_id = str(
        (occurrences.get(str(locator_argument.get("occurrence_id"))) or {}).get("decision_id"))
    locator_levels = _ordered_decision_levels(decisions.get(locator_decision_id) or {})
    if not locator_levels:
        return None
    locator_argument["support_property"] = _first_noncolliding_level(
        locator_levels, _foreign_protected_terms(occurrences, locator_decision_id))
    answer_argument["support_property"] = str(answer_level)
    for argument in (subject, obj):
        decision_id = (occurrences.get(str(argument.get("occurrence_id"))) or {}).get("decision_id")
        label = span_label_by_decision.get(str(decision_id))
        if label:  # v4 anchor path needs a span_label on each linked argument
            argument.setdefault("span_label", label)
    return {
        "relation": str(plan["relation"]),
        "arguments": [subject, obj],
        "question": templates[str(plan["relation"])].format(
            locator=locator_argument["support_property"]),
        "answer_role": "object" if direction == "forward" else "subject",
        "accepted_answers": [str(answer_level)],
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
    }


# Set-valued forward templates (validated: scripts/spikes/set_valued_gate_probe.py): the
# exhaustive object-direction question for an ambiguous group. The subject is referenced by its
# LEVEL, the answer is the FULL controlled object set (JSON array via the set reader); literals
# are pre-excluded from scoring -- recall counts controlled members only.
_SET_FORWARD_TEMPLATES = {
    "prescribed_with": "List EVERY distinct medication that the document says was prescribed or "
                       "used to treat the patient's {locator}.",
    "tests_for": "List EVERY distinct test, lab, or imaging study that the document says was "
                 "ordered or performed to evaluate the patient's {locator}.",
    "procedure_for": "List EVERY distinct procedure or therapy that the document says was "
                     "performed or planned to treat the patient's {locator}.",
    "contraindicated_because_of": "List EVERY distinct medical condition that the document says "
                                  "makes the patient's {locator} contraindicated or unsuitable.",
    "causes_or_explains": "List EVERY distinct symptom, finding, or condition that the document "
                          "says is caused or explained by the patient's {locator}.",
}


def _set_forward_row(
    document: str,
    relation: str,
    subject_argument: Mapping,
    object_arguments: Sequence[Mapping],
    occurrences: Mapping[str, Mapping],
    decisions: Mapping[str, Mapping],
) -> dict | None:
    """One set-valued forward row for an ambiguous group: subject locator at its finest level,
    answer = every controlled object at its finest level (answer_target linked_decision_set,
    scored as one-to-one member recall). No answer-level search: the probe validated finest-level
    members, and a per-member level product would explode the trial space."""
    subject_occurrence = occurrences.get(str(subject_argument.get("occurrence_id")))
    if subject_occurrence is None or subject_occurrence.get("decision_id") is None:
        return None
    subject_decision_id = str(subject_occurrence["decision_id"])
    subject_levels = _ordered_decision_levels(decisions.get(subject_decision_id) or {})
    if not subject_levels:
        return None
    subject_level = _first_noncolliding_level(
        subject_levels, _foreign_protected_terms(occurrences, subject_decision_id))
    members: list[dict] = []
    compiled_objects: list[dict] = []
    requirements: dict[str, str] = {subject_decision_id: subject_level}
    for object_argument in object_arguments:
        occurrence = occurrences.get(str(object_argument.get("occurrence_id")))
        if occurrence is None or occurrence.get("decision_id") is None:
            return None
        decision_id = str(occurrence["decision_id"])
        levels = _ordered_decision_levels(decisions.get(decision_id) or {})
        if not levels:
            return None
        member_level = str(levels[0])
        requirements[decision_id] = member_level
        members.append({"decision_id": decision_id, "required_property": member_level})
        compiled_objects.append({
            "role": "object", "kind": "linked",
            "occurrence_id": str(occurrence["occurrence_id"]),
            "runtime_type": occurrence.get("runtime_type"),
            "support_property": member_level,
        })
    if len(requirements) != len(compiled_objects) + 1:
        return None  # an object shares the subject's (or a sibling's) decision -> degenerate
    question = _SET_FORWARD_TEMPLATES[relation].format(locator=subject_level)
    occurrence_ids = [str(subject_occurrence["occurrence_id"])] + [
        argument["occurrence_id"] for argument in compiled_objects]
    ranges = [
        (int(occurrences[occurrence_id]["start"]), int(occurrences[occurrence_id]["end"]))
        for occurrence_id in occurrence_ids
    ]
    lo, hi = min(span[0] for span in ranges), max(span[1] for span in ranges)
    return {
        "family": "context", "scope": "linked", "subtype": "contextual_relation",
        "relation": relation,
        "occurrence_ids": occurrence_ids,
        "group_id": "set_forward:" + relation + ":" + _stable_hash(
            [subject_decision_id, sorted(requirements)]),
        "question": question,
        "accepted_values": [member["required_property"] for member in members],
        "answer_target": {"kind": "linked_decision_set", "members": members},
        "answer_type": _relation_answer_type_hint(relation, "object"),
        "scoring_contract": {"kind": "semantic_qa", "match": "set_recall"},
        "decision_requirements": requirements,
        "evidence": {
            "authority": "source_document",
            "arguments": [
                {"role": "subject", "kind": "linked",
                 "occurrence_id": occurrence_ids[0],
                 "runtime_type": subject_occurrence.get("runtime_type"),
                 "support_property": subject_level},
                *compiled_objects,
            ],
            "argument_spans": {
                occurrence_id: [span[0], span[1]]
                for occurrence_id, span in zip(occurrence_ids, ranges)
            },
            "reader_turns": _source_turns_for_ranges(document, ranges),
            "source_span": {"start": lo, "end": hi, "quote_hash": _stable_hash(document[lo:hi])},
            "teacher_id": "deterministic", "run_id": "deterministic_stage",
        },
    }


# Compound span-locator reverse templates: ALL of an ambiguous group's object levels in one
# question. A single object's flip fails exactly when that object ties to several subjects; the
# conjunction pins one subject the way the compound literal locator pins one condition. "all"/
# "single" + the name-only tail steer the extractive reader to one answer.
_COMPOUND_REVERSE_TEMPLATES = {
    "prescribed_with": "For what single medical condition were {locators} all prescribed? "
                       "Name only the condition.",
    "tests_for": "What single medical condition were {locators} all ordered to evaluate? "
                 "Name only the condition.",
    "procedure_for": "What single medical condition were {locators} all performed or planned to "
                     "treat? Name only the condition.",
    "contraindicated_because_of": "Which single medication or treatment must be avoided because "
                                  "of {locators}? Name only that medication or treatment.",
    "causes_or_explains": "What single underlying medical condition causes or explains "
                          "{locators}? Name only the condition.",
}


def _compound_span_reverse_row(
    document: str,
    relation: str,
    subject_argument: Mapping,
    object_arguments: Sequence[Mapping],
    occurrences: Mapping[str, Mapping],
    decisions: Mapping[str, Mapping],
    answer_level: str,
) -> dict | None:
    """One compound span-locator reverse row: every object at its finest level forms the question
    locator, the subject answers at `answer_level`. All arguments are controlled linked spans, so
    the placeholder render hides subject AND locators -- the gate's floor holds by construction."""
    subject_occurrence = occurrences.get(str(subject_argument.get("occurrence_id")))
    if subject_occurrence is None or subject_occurrence.get("decision_id") is None:
        return None
    subject_decision_id = str(subject_occurrence["decision_id"])
    locators: list[str] = []
    compiled_objects: list[dict] = []
    requirements: dict[str, str] = {subject_decision_id: str(answer_level)}
    for object_argument in object_arguments:
        occurrence = occurrences.get(str(object_argument.get("occurrence_id")))
        if occurrence is None or occurrence.get("decision_id") is None:
            return None
        decision = decisions.get(str(occurrence["decision_id"]))
        levels = _ordered_decision_levels(decision or {})
        if not levels:
            return None
        locator_level = _first_noncolliding_level(
            levels, _foreign_protected_terms(occurrences, str(occurrence["decision_id"])))
        locators.append(locator_level)
        requirements[str(occurrence["decision_id"])] = locator_level
        compiled_objects.append({
            "role": "object", "kind": "linked",
            "occurrence_id": str(occurrence["occurrence_id"]),
            "runtime_type": occurrence.get("runtime_type"),
            "support_property": locator_level,
        })
    if len(requirements) != len(compiled_objects) + 1:
        return None  # an object shares the subject's (or a sibling's) decision -> degenerate
    display = _display_locators(locators)
    question = _COMPOUND_REVERSE_TEMPLATES[relation].format(
        locators=_join_locators(display or locators))
    occurrence_ids = [str(subject_occurrence["occurrence_id"])] + [
        argument["occurrence_id"] for argument in compiled_objects]
    ranges = [
        (int(occurrences[occurrence_id]["start"]), int(occurrences[occurrence_id]["end"]))
        for occurrence_id in occurrence_ids
    ]
    lo, hi = min(span[0] for span in ranges), max(span[1] for span in ranges)
    return {
        "family": "context", "scope": "linked", "subtype": "contextual_relation",
        "relation": relation,
        "occurrence_ids": occurrence_ids,
        "group_id": "compound_span_reverse:" + relation + ":" + _stable_hash(
            [subject_decision_id, sorted(requirements)]),
        "question": question,
        "accepted_values": [str(answer_level)],
        "answer_target": {"kind": "linked_decision", "decision_id": subject_decision_id,
                          "required_property": str(answer_level)},
        "answer_type": _relation_answer_type_hint(relation, "subject"),
        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
        "decision_requirements": requirements,
        "evidence": {
            "authority": "source_document",
            "arguments": [
                {"role": "subject", "kind": "linked",
                 "occurrence_id": occurrence_ids[0],
                 "runtime_type": subject_occurrence.get("runtime_type"),
                 "support_property": str(answer_level)},
                *compiled_objects,
            ],
            "argument_spans": {
                occurrence_id: [span[0], span[1]]
                for occurrence_id, span in zip(occurrence_ids, ranges)
            },
            "reader_turns": _source_turns_for_ranges(document, ranges),
            "source_span": {"start": lo, "end": hi, "quote_hash": _stable_hash(document[lo:hi])},
            "teacher_id": "deterministic", "run_id": "deterministic_stage",
        },
    }


def compile_relational_assertions(
    doc_id: str,
    document: str,
    environment_document: Mapping,
    proposals: Sequence[Mapping],
    *,
    relation_contract: Mapping[str, Mapping] = ACI_RELATION_CONTRACT,
    context_candidates: Sequence[Mapping] | None = None,
    reverse_framing_only: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Compile bounded teacher proposals into frozen, evidence-checked assertions.

    `reverse_framing_only=True` compiles ONLY the deterministic reverse-orientation ambiguity
    variants derived from `proposals` (the doc-global ambiguity pass passes primary+gleaning
    combined here, so {objects} is complete); the forward proposals themselves are not re-compiled."""
    occurrences = {
        str(row["occurrence_id"]): row
        for row in environment_document.get("occurrences", [])
    }
    # Token sets of surfaces the RANKER REWRITES in a scored render -- i.e. controlled spans
    # carrying a decision (e.g. brand "synthroid" -> <DRUG_1>). A context-literal argument whose
    # surface embeds one of these can't survive the generalized/placeholder docs, so the relation
    # is doomed. Detector-only evidence (controlled False / no decision, e.g. "physical therapy"
    # when it has no lattice entry) is NEVER rewritten -- the literal survives verbatim and is a
    # safe answer -- so it must NOT count here (the guard was over-firing on it).
    substitutable_token_sets = [
        tokens for occurrence in occurrences.values()
        if occurrence.get("controlled", True)
        and occurrence.get("decision_id") is not None
        and (tokens := _meaningful_tokens(str(occurrence.get("surface", ""))))
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
    # Normal mode compiles the forward proposals as given. reverse_framing_only compiles ONLY the
    # reverse-orientation variants derived from `proposals` (deterministic, roles preserved) -- run
    # as a separate doc-global pass so it can't perturb the forward passes' fact-group dedup.
    active_proposals = (_reverse_framed_proposals(proposals, span_labels, occurrences)
                        if reverse_framing_only else list(proposals))
    for index, proposal_value in enumerate(active_proposals):
        proposal = dict(proposal_value)
        proposal_hash = _stable_hash(proposal)

        def reject(reason: str, detail: Mapping | None = None) -> None:
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
            if competing_answers:  # ambiguity monitor, recorded even for pre-gate rejects
                evidence["answer_competing"] = competing_answers
            if grounded_arguments:  # compiled arg identity, so gleaning can key/repair the reject
                evidence["arguments"] = _rejection_safe_arguments(grounded_arguments)
            if detail:  # reason-specific diagnostics (e.g. the leaking token + answer level)
                evidence.update(detail)
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
            if grounded_relation:
                record["relation"] = grounded_relation
            rejected.append(record)

        # Ambiguity monitor accumulator (diagnostic only): set once the relation is grounded,
        # so it is recorded whether the relation is later kept or rejected at any gate.
        competing_answers: list[str] = []
        # Compiled relation/arguments accumulators: filled as soon as they resolve, so a later
        # reject() carries the identity gleaning needs (fact key + repair-prompt arguments).
        grounded_relation: str | None = None
        grounded_arguments: list[dict] = []
        # Hedge/modality observation (diagnostic only, non-blocking -- see docs/issues/
        # qa-builder-dept.md): set at the hedge site below, attached to the accepted relation's
        # evidence so compute_review_flags can surface it without blocking the reader gate.
        modality_diagnostic: str | None = None

        # No hard relation cap: the teacher's response schema already bounds each section to
        # RELATION_TEACHER_MAX_RELATIONS, and a genuinely relation-dense note should not have its
        # extra valid relations dropped. Over-count is surfaced (not rejected) as a soft review
        # flag in compute_review_flags.
        proposed_subtype = proposal.get("subtype")
        if proposed_subtype not in {None, "contextual_relation"}:
            reject("invalid_subtype")
            continue
        relation = str(proposal.get("relation", ""))
        # Cached v1 proposals conflated drug and procedure under treated_with.
        # Migrate only that old wire shape to the directional v2 relation; a v2
        # proposal labelled treated_with with a drug still fails its type contract.
        if proposal.get("arguments") is None and relation in {"treated_with", "procedure_for"}:
            legacy_ids = [str(value) for value in proposal.get("argument_occurrence_ids", [])]
            if len(legacy_ids) == 2 and legacy_ids[1] in occurrences and (
                _RUNTIME_TYPE_CLASSES.get(canon(str(occurrences[legacy_ids[1]].get("runtime_type", ""))))
                == "treatment"
            ):
                relation = "prescribed_with"
        if relation not in relation_contract:
            reject("invalid_relation")
            continue
        grounded_relation = relation
        arguments, argument_error = _teacher_relation_arguments(
            proposal, occurrences, context_by_id, span_labels
        )
        if argument_error is not None:
            reject(argument_error)
            continue
        arguments, proposal = _promote_context_literals_on_detected_entities(
            arguments, proposal, occurrences, decisions)
        grounded_arguments = arguments
        occurrence_ids = [argument["occurrence_id"] for argument in arguments
                          if argument["kind"] == "linked"]
        if len(set(occurrence_ids)) != len(occurrence_ids):
            reject("invalid_arguments")
            continue
        uses_v4_arguments = any("span_label" in argument or "literal" in argument
                                for argument in proposal.get("arguments") or [])
        if uses_v4_arguments:
            # A repeated value's mislabeled occurrence (history-list S-label instead
            # of the relation sentence) still grounds via a same-decision sibling.
            arguments = _remap_to_groundable_siblings(
                document, arguments, occurrences, context_by_id, relation, relation_contract)
            grounded_arguments = arguments
            occurrence_ids = [argument["occurrence_id"] for argument in arguments
                              if argument["kind"] == "linked"]
        anchor_kind = None
        anchor_clause_ranges = None
        if uses_v4_arguments:
            quote, evidence_span, anchor_clause_ranges, evidence_error, anchor_kind = _derived_relation_anchor(
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
        # Literal->linked probes are task-native reader tests. Their semantic
        # acceptance is the three-point reader gate; lexical and NLI cue gates
        # are brittle across informal and new-domain text. Exact literal
        # grounding plus the anchor's locality constraints remain mandatory.
        is_literal_probe = (
            sum(argument.get("kind") == "linked" for argument in arguments) == 1
            and sum(argument.get("kind") == "context" for argument in arguments) == 1
        )
        if (not RELATION_CUE_GATES_DISABLED
                and not is_literal_probe
                and not _relation_quote_has_direct_support(
                    relation,
                    quote,
                    arguments,
                    relation_contract,
                    allow_adjacent_clauses=(uses_v4_arguments or proposal.get("evidence_window_id") is not None),
                    allow_plan_section=anchor_kind in {"plan_section", "problem_block", "speaker_turn"},
                )):
            reject("invalid_evidence")
            continue
        # Ambiguity monitor (diagnostic only): the relation is now grounded, so record any co-valid
        # same-type answer in the subject's problem block(s). Set here -- before the leak/gate checks
        # -- so it is preserved whether the relation is later kept or rejected at ANY gate. Never
        # affects accept/reject or reward.
        _answer_role_mon = str(proposal.get("answer_role", "object"))
        _answer_arg_mon = arguments[0] if _answer_role_mon == "subject" else arguments[1]
        _subject_arg_mon = arguments[1] if _answer_role_mon == "subject" else arguments[0]
        _answer_scope_target = (
            {"decision_id": str(occurrences[_answer_arg_mon["occurrence_id"]]["decision_id"])}
            if _answer_arg_mon.get("kind") == "linked"
            and _answer_arg_mon.get("occurrence_id") in occurrences
            else {}
        )
        competing_answers = _answer_competing_surfaces(
            document, _subject_arg_mon, _answer_arg_mon, _answer_scope_target, occurrences)
        # A conditional/planned statement ("possibly refer to physical therapy", "if symptoms
        # persist, order X") paired with a DEFINITE question ("which test was ordered?") is a
        # modality mismatch. DEMOTED 2026-07-19 from a blocker to a non-blocking diagnostic: the
        # regex hedge/question detectors are too blunt to reject on (whack-a-mole false pos/neg,
        # tests_for blanket-exempt; see docs/issues/qa-builder-dept.md). Record the observation
        # and let the relation reach the reader gate; a strict semantic modality judge is the
        # planned replacement. No longer emits `hedged_relation`.
        if (uses_v4_arguments
                and _relation_window_is_hedged(document, arguments, occurrences, relation)
                and not _question_is_conditional(str(proposal.get("question", "")))):
            modality_diagnostic = "hedged_source_definite_question"
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
                   for argument in arguments
                   if argument["kind"] == "linked"}
        if any(not support.get(occurrence_id)
               or support.get(occurrence_id) not in legal_properties.get(occurrence_id, set())
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
        answer_role = str(proposal.get("answer_role", "object"))
        # A context literal is uncontrolled and can never be a valid gated answer (it survives the
        # placeholder render -> placeholder_answerable). For a relation pairing one linked span with
        # one context literal, the answer is ALWAYS the linked span, so force answer_role to the
        # linked argument's role regardless of the teacher's (frequently omitted) answer_role. The
        # prompt steers the literal-locator question; this enforces the matching answer direction.
        _linked_args = [argument for argument in arguments if argument.get("kind") == "linked"]
        _context_args = [argument for argument in arguments if argument.get("kind") == "context"]
        if len(_linked_args) == 1 and len(_context_args) == 1:
            answer_role = str(_linked_args[0].get("role") or "object")
        answer_argument = arguments[0] if answer_role == "subject" else arguments[1]
        # Placeholder-label tokens for the answered linked argument (for
        # example, "medication") are generic question syntax, not answer
        # leakage.
        answer_exempt_types = (
            str(answer_argument.get("runtime_type", ""))
            if answer_argument.get("kind") == "linked" else ""
        )
        # A linked answer is scored only against its selected support_property.
        # Derive the stored accepted value from that same authoritative value
        # rather than letting teacher prose affect privacy gates or artifact
        # semantics. Teacher answers remain required by the wire contract and
        # authoritative only for literal/legacy answer targets.
        if answer_argument.get("kind") == "linked":
            accepted_values = [str(answer_argument["support_property"])]
        answer_type_hint = _relation_answer_type_hint(relation, answer_role)
        leakage_repair = None
        # Generic answer-type words are derived from the answered runtime type's
        # placeholder contract (for example, "medication" in a question whose
        # canonical linked answer is "thyroid medication"). Every remaining
        # overlap is discriminative and still triggers the strict repair.
        leak_exempt = _meaningful_tokens(answer_type_hint or "")
        # Reverse-orientation question (answer=subject): the locator (the object named in the
        # question, e.g. "thyroid panel") is the GIVEN premise, not the answer -- its tokens must
        # not count as answer-leakage, else the repair strips them ("thyroid panel" -> "panel").
        if answer_role == "subject":
            locator_argument = arguments[1]
            locator_text = str(locator_argument.get("support_property")
                               or locator_argument.get("literal") or "")
            leak_exempt = leak_exempt | _meaningful_tokens(locator_text)
        leaking_tokens: set[str] = set()
        for answer in accepted_values:
            leaking_tokens |= _answer_leak_tokens(
                question, answer, answer_exempt_types, extra_exempt_tokens=leak_exempt)
        if leaking_tokens:
            repaired = _repair_leaked_relation(
                question, accepted_values, arguments, answer_role,
                answer_exempt_types, decisions, occurrences,
            ) if uses_v4_arguments else None
            if repaired is None:
                reject("answer_leakage", detail={
                    "leaking_tokens": sorted(leaking_tokens),
                    "answer_levels": [str(value) for value in accepted_values],
                })
                continue
            # Recolored to a legal level that clears the lexical collision; the
            # linked answer target and support are rebuilt from the new levels.
            question, accepted_values, arguments, leakage_repair = repaired
            support = {argument["occurrence_id"]: argument["support_property"]
                       for argument in arguments if argument["kind"] == "linked"}
            answer_argument = arguments[0] if answer_role == "subject" else arguments[1]
            if answer_argument.get("kind") == "linked":
                accepted_values = [str(answer_argument["support_property"])]
        protected_terms = []
        allowed_level_tokens: dict[str, frozenset[str]] = {}
        for occurrence_id_value, occurrence in occurrences.items():
            if not (occurrence.get("controlled", True)
                    and decisions.get(str(occurrence.get("decision_id")), {}).get("controlled", True)):
                continue
            # Generic type words ("medication(s)", "condition") are information-free
            # placeholder labels, not a locator for a specific protected span: a
            # question that says "certain medications" does not reveal the controlled
            # drug "immunosuppressive medications" (only the discriminative token
            # "immunosuppressive" would). Exempt them, singular and plural, alongside
            # the legal levels.
            type_words = _placeholder_meaning_tokens(str(occurrence.get("runtime_type", "")))
            type_words = type_words | {f"{word}s" for word in type_words}
            levels = list(legal_properties.get(occurrence_id_value, ()))
            for term in _occurrence_protected_terms(occurrence):
                # A level identical to the raw surface is not a real generalization (a broken
                # lattice could entail the surface itself); drop it so a raw brand's ONLY
                # authorization can't be a surface-echoing level. A genuine coarser level that
                # merely shares a token (surface "diabetes" in level "diabetes mellitus") stays.
                term_level_tokens = frozenset(
                    token
                    for property_level in levels
                    if canon(property_level) != canon(term)
                    for token in _meaningful_tokens(property_level)
                ) | type_words
                protected_terms.append(term)
                allowed_level_tokens[term] = allowed_level_tokens.get(term, frozenset()) | term_level_tokens
        # A relation publishes its own arguments' generalization levels, so a token
        # from any argument level (e.g. "kidney" in the subject's "cystic kidney
        # disease") is legitimately in the question even when it collides with a
        # different argument's raw surface ("kidney transplant"). Pool the levels of
        # the relation's argument occurrences and exempt them for the question check
        # only — never the answer check, which must stay strict per answered argument.
        argument_level_tokens = frozenset(
            token
            for occurrence_id in occurrence_ids
            for property_level in legal_properties.get(occurrence_id, ())
            for token in _meaningful_tokens(property_level)
        )
        question_allowed_tokens = {
            term: tokens | argument_level_tokens
            for term, tokens in allowed_level_tokens.items()
        }
        if _question_leaks_protected_term(question, protected_terms, question_allowed_tokens):
            # Attribute the leak: if exempting the CONTEXT LITERAL arguments' tokens clears it,
            # the collision comes solely from an unrecolorable literal (e.g. a lab-function
            # phrase overlapping a protected organ surface) -- no author, teacher included, can
            # rephrase around it, so gleaning drops these as dead weight.
            literal_tokens = frozenset(
                token
                for argument in arguments
                if argument.get("kind") == "context"
                for token in _meaningful_tokens(str(argument.get("literal") or ""))
            )
            literal_only_leak = bool(literal_tokens) and not _question_leaks_protected_term(
                question, protected_terms,
                {term: tokens | literal_tokens
                 for term, tokens in question_allowed_tokens.items()},
            )
            reject("protected_locator",
                   detail={"leak_source": "context_literal"} if literal_only_leak else None)
            continue
        # Only literal/legacy answer golds are teacher-authored text and need this raw-surface
        # leak check. Linked answer golds above are local lattice properties and never enter the
        # remote question/context; checking them against unrelated split decisions can reject a
        # valid property solely because another occurrence has a bad lattice resolution.
        if (proposal.get("arguments") is not None
                and answer_argument.get("kind") != "linked"
                and any(
            _question_leaks_protected_term(answer, protected_terms, allowed_level_tokens)
            for answer in accepted_values
        )):
            reject("protected_answer")
            continue
        decision_requirements = {
            str(occurrences[occurrence_id]["decision_id"]): support[occurrence_id]
            for occurrence_id in occurrence_ids
        }
        fact_group = _relation_fact_key(relation, arguments, occurrences)
        if fact_group in fact_groups:
            reject("duplicate_fact_group")
            continue
        fact_groups.add(fact_group)
        # The answered argument (default: the object). A linked answer is scored
        # by lattice entailment against its decision's frozen chain; a literal
        # answer keeps lexical matching against the exact grounded span.
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
        source_argument_ranges = [
            (
                int(occurrences[argument["occurrence_id"]]["start"]),
                int(occurrences[argument["occurrence_id"]]["end"]),
            )
            if argument["kind"] == "linked" else (
                int(argument["start"]), int(argument["end"]),
            )
            for argument in arguments
        ]
        soft_cap_diagnostic = _soft_cross_clause_cap_diagnostic(
            document, source_argument_ranges,
        )
        reader_clauses = (
            _source_reader_clause_refs(document, source_argument_ranges)
            if anchor_kind == "speaker_turn" else []
        )
        source_span = {
            "start": evidence_span[0],
            "end": evidence_span[1],
            "quote_hash": _stable_hash(document[evidence_span[0]:evidence_span[1]]),
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
                list(anchor_clause_ranges or [(source_span["start"], source_span["end"])])
                + [(span[0], span[1]) for span in argument_spans.values()],
            ),
            "arguments": arguments,
        }
        if anchor_clause_ranges is not None:
            evidence["source_clause_ranges"] = [list(span) for span in anchor_clause_ranges]
        if reader_clauses:
            evidence["reader_clauses"] = reader_clauses
        if soft_cap_diagnostic is not None:
            evidence["anchor_diagnostics"] = [soft_cap_diagnostic]
        if modality_diagnostic is not None:
            evidence["modality_diagnostics"] = [modality_diagnostic]
        if leakage_repair is not None:
            evidence["leakage_repair"] = leakage_repair
        if competing_answers:  # ambiguity monitor (computed above, pre-gate); diagnostic only
            evidence["answer_competing"] = competing_answers
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
    # Specific -> coarse order is the decision's AUTHORED level ladder (the profile's `levels`
    # list order, preserved into `actions` at freeze time). The profile loader validates that
    # ladder is monotone in the profile's OWN level_counts (lattice_profiles.py), so authored
    # order == profile-count order and is the semantic hierarchy.
    #
    # We deliberately do NOT re-sort by `coarseness_rank`: that is a GLOBAL per-fill anonymity-set
    # size (aset_count -> lookup_count, keyed on the level string across ALL profiles, aggregated
    # by max), not a per-profile generality rank. It is miscalibrated across profiles -- e.g.
    # global "heart disease"=400 > "thoracic disease"=390 -- so sorting by it inverts organ vs
    # region levels (heart ranked coarser than thoracic), which drops the organ->region entailment
    # edge and inverts the RL reward gradient (a policy generalizing CHF to the more-specific
    # "heart disease" would score 0 while the coarser "thoracic disease" scores 1). aset /
    # coarseness_rank is UNCHANGED for its real jobs (ranker policy feature, BC target selection,
    # hiding / representative-anchor selection); only the entailment ORDER uses the profile ladder.
    levels = [action for action in decision.get("actions", []) if action.get("mode") == "level"]
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


def _entity_key(surface: str, runtime_type: str) -> str:
    """Canonical identity for a detected surface, shared by decision keys and occurrence
    matching so plural/alias variants ("migraines"->"migraine", brand->generic) resolve to
    one decision. Reuses the lattice profile resolver (single source of the plural fold);
    falls back to canon() for surfaces with no profile row."""
    from cloak.lattice_profiles import lookup_entry
    entry = lookup_entry(surface, runtime_type)
    return canon(entry[0]) if entry else canon(surface)


def _append_text_anchored_occurrences(
    document_text: str, doc_id: str,
    decisions_by_key: Mapping[tuple[str, str], dict], occurrences: list[dict],
) -> None:
    """Register exact-text occurrences of already-decided entities that the detector
    dropped locally -- e.g. a repeat mention scored below the per-type admission gate
    ('kidney stones' at 0.44 vs the 0.5 health-condition gate) even though the same
    surface was admitted >=gate elsewhere in the document. Grounding can then anchor a
    relation argument to the dropped mention.

    Threshold-free: only surfaces that already back a *controlled decision in this
    document* are propagated, so no new entity type or value is introduced -- the value
    is confirmed sensitive; we only recover its other verbatim positions. Positions
    already covered by an occurrence are skipped, so detected spans are never duplicated.
    One compiled regex + one finditer pass over the document -> O(len(document)); the
    per-match overlap test is O(#occurrences), tiny for a single note.
    """
    from cloak.lattice_profiles import singularize
    if not document_text:
        return
    base_to_decision: dict[str, dict] = {}
    decisions_by_id = {d["decision_id"]: d for d in decisions_by_key.values()}
    for occ in occurrences:
        decision_id = occ.get("decision_id")
        if not occ.get("controlled") or decision_id is None:
            continue
        base = singularize(str(occ.get("surface", "")))
        if len(base) >= 3:  # ponytail: skip 1-2 char bases ("ms","mi") -- match everything
            base_to_decision.setdefault(base, decisions_by_id[decision_id])
    if not base_to_decision:
        return
    # longest base first so "kidney stone" wins over "stone"; trailing s? folds plurals,
    # hyphen/word guards keep "ct" out of "contract" and "stone" out of "stoned".
    bases = sorted(base_to_decision, key=len, reverse=True)
    pattern = re.compile(
        r"(?<![\w-])(" + "|".join(re.escape(b) for b in bases) + r")s?(?![\w-])",
        re.IGNORECASE,
    )
    covered = [
        (int(o["start"]), int(o["end"])) for o in occurrences
        if isinstance(o.get("start"), int) and isinstance(o.get("end"), int)
    ]
    for match in pattern.finditer(document_text):
        start, end = match.start(), match.end()
        if any(start < ce and cs < end for cs, ce in covered):
            continue
        decision = base_to_decision.get(singularize(match.group(1)))
        if decision is None:
            continue
        surface = document_text[start:end]
        occurrence_id = _stable_hash({
            "doc_id": doc_id, "runtime_type": decision["runtime_type"],
            "surface": surface, "start": start, "end": end,
        })
        occurrences.append({
            "occurrence_id": occurrence_id,
            "start": start, "end": end, "surface": surface, "aliases": [],
            "runtime_type": decision["runtime_type"], "polarity": "unknown",
            "detector_provenance": {
                "source": "text_anchored", "anchored_to": decision["decision_id"]},
            "overlap_disposition": "accepted",
            "decision_id": decision["decision_id"], "controlled": True,
        })
        decision["occurrence_ids"].append(occurrence_id)
        covered.append((start, end))


def freeze_ranker_environment(
    ranker_environment: Mapping,
    *,
    occurrences_by_document: Mapping[str, Sequence[Mapping]] | None = None,
    source_documents: Mapping[str, str] | None = None,
) -> dict:
    """Migrate embedded ranker spans to stable occurrence/decision identities, without detection.

    When ``source_documents`` is provided, exact-text repeats of an already-decided entity
    that the detector dropped locally are recovered as occurrences (see
    ``_append_text_anchored_occurrences``). Both callers pass the same source text so the
    ``environment_hash`` stays identical between build and train.
    """
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
                forced_placeholder = any(
                    action.get("mode") == "placeholder" and action.get("forced_placeholder")
                    for action in actions
                )
                if not forced_placeholder and not any(action["mode"] == "keep" for action in actions):
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
                        "ranker_selectable": not forced_placeholder,
                        "actions": actions,
                        "action_menu_hash": action_menu_hash,
                    }
            # KEEP action semantics intentionally remain keyed by each source surface
            # above. Once those menus are complete, aliases that resolve to the same
            # lattice entry can share the first decision -- but only when their
            # non-KEEP choices are indistinguishable. Redirecting the surface keys
            # preserves the KEEP construction invariant while making every occurrence
            # receive the shared decision_id below.
            def non_keep_menu(decision: Mapping) -> tuple:
                entries = []
                for action in decision["actions"]:
                    if action["mode"] == "keep":
                        continue
                    if action["mode"] == "level":
                        entries.append((
                            "level", action.get("fill"),
                            float(action["coarseness_rank"]),
                        ))
                    else:
                        entries.append(("placeholder", action.get("fill")))
                return tuple(sorted(entries))

            keys_by_entity: dict[tuple[str, str], list[tuple[str, str]]] = {}
            for decision_key in decisions_by_key:
                runtime_type, source_key = decision_key
                keys_by_entity.setdefault(
                    (runtime_type, _entity_key(source_key, runtime_type)), []
                ).append(decision_key)
            for decision_keys in keys_by_entity.values():
                primary = decisions_by_key[decision_keys[0]]
                primary_menu = non_keep_menu(primary)
                if all(
                    non_keep_menu(decisions_by_key[decision_key]) == primary_menu
                    for decision_key in decision_keys[1:]
                ):
                    for decision_key in decision_keys[1:]:
                        decisions_by_key[decision_key] = primary
            occurrence_source = (
                occurrences_by_document[doc_id]
                if occurrences_by_document is not None and doc_id in occurrences_by_document
                else document.get("spans", [])
            )
            # Secondary index by profile-canonical identity so a plural/alias occurrence
            # ("migraines", a brand name) still links to its decision even though the
            # decision key stays the source surface (which KEEP semantics rely on).
            decisions_by_entity: dict[tuple[str, str], dict] = {}
            for (rtype, ckey), decision in decisions_by_key.items():
                decisions_by_entity.setdefault((rtype, _entity_key(ckey, rtype)), decision)
            occurrences = []
            for row in occurrence_source:
                runtime_type = str(row.get("type", row.get("runtime_type", "")))
                surface = str(row.get("surface", ""))
                decision = (
                    decisions_by_key.get((runtime_type, canon(surface)))
                    or decisions_by_entity.get((runtime_type, _entity_key(surface, runtime_type)))
                )
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
                    **({"match": dict(row["match"])}
                       if isinstance(row.get("match"), Mapping) else {}),
                    **({"profile_match": dict(row["profile_match"])}
                       if isinstance(row.get("profile_match"), Mapping) else {}),
                })
                if decision is not None:
                    decision["occurrence_ids"].append(occurrence_id)
            if source_documents is not None and doc_id in source_documents:
                _append_text_anchored_occurrences(
                    source_documents[doc_id], doc_id, decisions_by_key, occurrences)
            occurrences_by_id = {
                str(row["occurrence_id"]): row for row in occurrences
            }
            decisions_by_id = {
                decision["decision_id"]: decision
                for decision in decisions_by_key.values()
            }
            for decision in decisions_by_id.values():
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
                "decisions": list(decisions_by_id.values()),
            }
            frozen_document["environment_document_hash"] = _stable_hash(frozen_document)
            documents[doc_id] = frozen_document
    frozen = {"artifact_version": "occurrence-decisions-v1", "documents": documents}
    frozen["environment_hash"] = _stable_hash(frozen)
    return frozen


_V2_ACTION_FIELDS = frozenset({
    "fill", "mode", "keep", "source_identity", "aset", "coarseness_rank",
    "level_grounding", "legal", "entails", "forced_placeholder",
})
_V2_OCCURRENCE_FIELDS = frozenset({
    "surface", "type", "runtime_type", "start", "end", "score", "aliases",
    "polarity", "detector_provenance", "overlap_disposition", "match",
    "profile_match", "forced_placeholder",
})
_V2_DETECTOR_ROW_FIELDS = frozenset({
    "text", "surface", "type", "runtime_type", "proposed_runtime_type", "start",
    "end", "score", "source", "raw_label", "recognizer", "status", "reason",
    "min_health_condition_score", "detector_provenance", "overlap_disposition",
})


def _policy_free_action(action: Mapping, runtime_type: str) -> dict:
    clean = {key: action[key] for key in _V2_ACTION_FIELDS if key in action}
    if str(clean.get("mode", "level")) == "placeholder":
        clean["fill"] = None
        clean["placeholder_type"] = runtime_type
    return clean


def _policy_free_ranker_environment(ranker_environment: Mapping) -> dict:
    """Whitelist V2 decision facts from a legacy ranker environment.

    This is deliberately a compatibility adapter, not a policy migration: tau,
    floors, behavior-clone labels, cached risk, and proximity never cross it.
    """
    corpora = {}
    for corpus, documents in (ranker_environment.get("corpora") or {}).items():
        corpora[corpus] = {}
        for doc_id, document in documents.items():
            spans = []
            for span in document.get("spans", []):
                runtime_type = str(span.get("type", span.get("runtime_type", "")))
                spans.append({
                    key: span[key]
                    for key in ("surface", "type", "start", "end", "sent")
                    if key in span
                } | {
                    "type": runtime_type,
                    "actions": [
                        _policy_free_action(action, runtime_type)
                        for action in span.get("actions", [])
                    ],
                })
            corpora[corpus][doc_id] = {"spans": spans}
    return {"corpora": corpora}


def legacy_arms_ranker_environment(arms: Mapping) -> dict:
    """Expose historical action tables through the policy-free V2 whitelist."""
    corpora = {}
    for corpus, documents in arms.items():
        if corpus == "_meta" or not isinstance(documents, Mapping):
            continue
        corpora[corpus] = {}
        for doc_id, document in documents.items():
            table = document.get("v2_action_table", document.get("action_table", {}))
            rows = list(table.values()) if isinstance(table, Mapping) else []
            corpora[corpus][doc_id] = {"spans": rows}
    return _policy_free_ranker_environment({"corpora": corpora})


def _legacy_arms_occurrences(
    arms: Mapping,
    *,
    detector_provenance: Mapping | None = None,
) -> dict[str, list[dict]]:
    """Read only policy-independent rows from a historical arms artifact."""
    result = {}
    for corpus, documents in arms.items():
        if corpus == "_meta" or not isinstance(documents, Mapping):
            continue
        for doc_id, document in documents.items():
            rows = document.get("v2_occurrences")
            if not isinstance(rows, list):
                walk = document.get("tau_walk")
                rows = walk[1] if isinstance(walk, (list, tuple)) and len(walk) > 1 else []
            clean_rows = []
            for row in rows if isinstance(rows, list) else []:
                clean = {key: row[key] for key in _V2_OCCURRENCE_FIELDS if key in row}
                if detector_provenance is not None or row.get("detector_provenance"):
                    clean["detector_provenance"] = {
                        **dict(detector_provenance or {}),
                        **dict(row.get("detector_provenance") or {}),
                        "score": row.get("score"),
                    }
                clean_rows.append(clean)
            result[str(doc_id)] = clean_rows
    return result


def _policy_free_detector_diagnostics(value: object) -> dict:
    if not isinstance(value, Mapping):
        return {}
    clean = {}
    for section in ("accepted", "rejected", "post_detection_rejections"):
        rows = value.get(section)
        if isinstance(rows, list):
            clean[section] = [
                {key: row[key] for key in _V2_DETECTOR_ROW_FIELDS if key in row}
                for row in rows if isinstance(row, Mapping)
            ]
    return clean


def freeze_v2_environment_from_legacy_arms(
    ranker_environment: Mapping,
    arms: Mapping,
    *,
    detector_provenance: Mapping | None = None,
    source_documents: Mapping[str, str] | None = None,
) -> dict:
    """Compatibility boundary from historical arms/env files to the V2 contract.

    The returned representation contains canonical occurrence/decision/action IDs,
    legal lattice actions, typed placeholders, and render offsets. It cannot depend
    on legacy policy outcomes because the adapter uses an explicit field whitelist.
    """
    frozen = freeze_ranker_environment(
        _policy_free_ranker_environment(ranker_environment),
        occurrences_by_document=_legacy_arms_occurrences(
            arms, detector_provenance=detector_provenance,
        ),
        source_documents=source_documents,
    )
    for corpus, documents in arms.items():
        if corpus == "_meta" or not isinstance(documents, Mapping):
            continue
        for doc_id, legacy_document in documents.items():
            document = frozen["documents"].get(str(doc_id))
            if document is None or not isinstance(legacy_document, Mapping):
                continue
            diagnostics = _policy_free_detector_diagnostics(
                legacy_document.get("detector_diagnostics")
            )
            if diagnostics:
                document["detector_diagnostics"] = diagnostics
                document["environment_document_hash"] = _stable_hash(document)
    frozen["environment_hash"] = _stable_hash({
        key: value for key, value in frozen.items() if key != "environment_hash"
    })
    return frozen


def render_frozen_action_vector(
    source: str,
    frozen_document: Mapping,
    action_vector: Mapping[str, str],
) -> tuple[str, list[dict]]:
    """Render one V2 action vector using only frozen offsets and action semantics."""
    decisions = {
        str(decision["decision_id"]): decision
        for decision in frozen_document.get("decisions", [])
    }
    chosen = {}
    placeholder_by_decision = {}
    counters: dict[str, int] = {}
    used_fills = {}
    for decision in frozen_document.get("decisions", []):
        decision_id = str(decision["decision_id"])
        selected_id = str(action_vector[decision_id])
        selected = next(
            (action for action in decision["actions"]
             if str(action["action_id"]) == selected_id),
            None,
        )
        if selected is None or not selected.get("legal", True):
            raise ValueError(f"illegal or unknown action for decision {decision_id}")
        if selected["mode"] == "placeholder":
            runtime_type = str(selected.get("placeholder_type") or decision["runtime_type"])
            token_type = placeholder_type_token(runtime_type)
            counters[token_type] = counters.get(token_type, 0) + 1
            placeholder_by_decision[decision_id] = placeholder_token(
                runtime_type, counters[token_type]
            )
        elif selected["mode"] == "level":
            fill_key = canon(str(selected.get("fill", "")))
            owner = used_fills.setdefault(fill_key, decision_id)
            if owner != decision_id:
                raise ValueError(f"non-injective V2 action vector fill {selected.get('fill')!r}")
        chosen[decision_id] = selected

    replacements = []
    for occurrence in frozen_document.get("occurrences", []):
        decision_id = occurrence.get("decision_id")
        if decision_id is None:
            continue
        decision_id = str(decision_id)
        if decision_id not in decisions or decision_id not in chosen:
            raise ValueError(f"unresolved controlled occurrence decision {decision_id}")
        selected = chosen[decision_id]
        original = source[int(occurrence["start"]):int(occurrence["end"])]
        if selected["mode"] == "keep":
            replacement = original
        elif selected["mode"] == "placeholder":
            replacement = placeholder_by_decision[decision_id]
        else:
            replacement = str(selected["fill"])
            if original[:1].isupper() and replacement:
                replacement = replacement[:1].upper() + replacement[1:]
        replacements.append({
            "occurrence_id": occurrence["occurrence_id"],
            "decision_id": decision_id,
            "start": int(occurrence["start"]),
            "end": int(occurrence["end"]),
            "surface": original,
            "replacement": replacement,
            "action_id": selected["action_id"],
            "mode": selected["mode"],
        })
    ordered = sorted(replacements, key=lambda row: (row["start"], row["end"]))
    for left, right in zip(ordered, ordered[1:]):
        if right["start"] < left["end"]:
            raise ValueError("overlapping frozen V2 occurrences cannot be rendered")
    rendered = source
    for row in reversed(ordered):
        rendered = rendered[:row["start"]] + row["replacement"] + rendered[row["end"]:]
    return rendered, ordered


def frozen_occurrences_from_arms(
    arms: Mapping,
    *,
    detector_provenance: Mapping | None = None,
) -> dict[str, list[dict]]:
    """Legacy compatibility reader; V2 callers use the frozen environment directly."""
    return _legacy_arms_occurrences(arms, detector_provenance=detector_provenance)


# Detected runtime types that are supposed to carry a generalization lattice
# (identity types like PERSON/LOC are placeholder-by-rule and excluded).
_REVIEW_LADDER_TYPES = frozenset({"drug", "health-condition", "medical-procedure"})


def _review_flag(code: str, stage: str, fix_class: str, severity: str, detail: dict) -> dict:
    return {"code": code, "stage": stage, "fix_class": fix_class,
            "severity": severity, "detail": detail}


def compute_review_flags(artifact: Mapping) -> dict[str, list[dict]]:
    """Per-document diagnostics classifying *why* a document may be worth
    re-processing after a fix. Pure function over the built artifact; each flag
    carries a `fix_class` (data_lattice / teacher_redraw / reader /
    ontology_review) so a change can re-select exactly the affected documents.
    Diagnostic only — never enters the artifact hash or any measurement."""
    flags: dict[str, list[dict]] = defaultdict(list)

    # A) a detected, lattice-eligible span with no legal generalization level
    #    (e.g. an unaliased drug brand -> placeholder-only, the synthroid case).
    for doc_id, document in (artifact.get("documents") or {}).items():
        decisions = {str(d.get("decision_id")): d for d in document.get("decisions") or []}
        seen: set[tuple[str, str]] = set()
        for occurrence in document.get("occurrences") or []:
            runtime_type = canon(str(occurrence.get("runtime_type", "")))
            if runtime_type not in _REVIEW_LADDER_TYPES:
                continue
            key = (runtime_type, canon(str(occurrence.get("surface", ""))))
            if key in seen:
                continue
            seen.add(key)
            decision = decisions.get(str(occurrence.get("decision_id")))
            if not (_ordered_decision_levels(decision) if decision else []):
                flags[doc_id].append(_review_flag(
                    "missing_generalization", "freeze", "data_lattice", "warn",
                    {"runtime_type": occurrence.get("runtime_type"),
                     "surface": occurrence.get("surface"),
                     "resolved_decision": bool(decision)}))

    # C) soft relation-count cap: the hard K-cap reject was removed, so a relation-dense note
    #    keeps all its valid relations. Log (info, no fix_class action) when a document's kept
    #    relation count exceeds the soft cap, so unusually dense notes are still visible.
    for doc_id, records in (artifact.get("relation_generation") or {}).items():
        kept = sum(1 for record in records if record.get("status") == "kept")
        if kept > RELATION_TEACHER_MAX_RELATIONS:
            flags[str(doc_id)].append(_review_flag(
                "relation_count_over_soft_cap", "compile", "ontology_review", "info",
                {"kept_relations": kept, "soft_cap": RELATION_TEACHER_MAX_RELATIONS}))

    # False-positive guard for the lattice_level_suspect probe: a lattice node used by a KEPT relation
    # in the same doc is provably readable, so a gate failure elsewhere is context-specific, not bad
    # data. Map each doc's kept-assertion decisions back to their canonical surfaces.
    _surface_by_decision = {
        str(doc_id): {
            str(d.get("decision_id")): canon(str(d.get("canonical_key") or ""))
            for d in (document.get("decisions") or [])
        }
        for doc_id, document in (artifact.get("documents") or {}).items()
    }
    kept_lattice_surfaces: dict[str, set[str]] = defaultdict(set)
    for assertion in (artifact.get("assertions") or {}).values():
        _dec_surface = _surface_by_decision.get(str(assertion.get("doc_id")), {})
        for decision_id in (assertion.get("decision_requirements") or {}):
            surface = _dec_surface.get(str(decision_id))
            if surface:
                kept_lattice_surfaces[str(assertion.get("doc_id"))].add(surface)

    # D) classify signals the build already emits
    for record in (artifact.get("rejections") or {}).get("records") or []:
        doc_id = str(record.get("doc_id"))
        reason = record.get("detail_reason") or record.get("reason")
        evidence = record.get("evidence") or {}
        if reason == "literal_will_be_substituted":
            flags[doc_id].append(_review_flag(
                "literal_will_be_substituted", "compile", "data_lattice", "warn", {}))
        elif reason == "answer_leakage":
            flags[doc_id].append(_review_flag(
                "unrepaired_answer_leakage", "compile", "ontology_review", "warn",
                {"leaking_tokens": evidence.get("leaking_tokens"),
                 "answer_levels": evidence.get("answer_levels")}))
        elif reason == "placeholder_answerable":
            flags[doc_id].append(_review_flag(
                "placeholder_answerable", "gate", "ontology_review", "info", {}))
        elif reason == "three_point_gate_failed":
            scores = (evidence.get("validation") or {}).get("scores") or {}
            probe = evidence.get("lattice_probe")
            if isinstance(probe, Mapping) and probe.get("readable_coarser_level"):
                # a COARSER legal level in the same chain reads -> the chosen fine level, not the
                # relation, could be bad lattice data. But this probe is FALSE-POSITIVE-PRONE: the
                # failure is usually ANSWER AMBIGUITY (subject has >=2 same-type answers -> no unique
                # QA, not a bad level), or the flagged node is used by a KEPT relation in-doc (provably
                # readable). Suppress both; only a node that fails AND is unambiguous AND is never kept
                # in-doc is a genuine data suspect worth a human fixing lattice_profiles.json.
                if evidence.get("answer_competing") or (
                        canon(str(probe.get("surface", "")))
                        in kept_lattice_surfaces.get(doc_id, set())):
                    pass
                else:
                    flags[doc_id].append(_review_flag(
                        "lattice_level_suspect", "gate", "data_lattice", "warn",
                        {"surface": probe.get("surface"),
                         "runtime_type": probe.get("runtime_type"),
                         "unreadable_level": probe.get("unreadable_level"),
                         "readable_coarser_level": probe.get("readable_coarser_level"),
                         "chain": probe.get("chain"),
                         "scores": scores}))
            elif (scores.get("original", 0.0) >= 1.0
                    and scores.get("representative", 0.0) < 1.0
                    and scores.get("placeholder", 1.0) < 1.0):
                # source answerable, generalized form not -> the chosen generalization level is
                # not reader-recoverable. Treated as a lattice-profile signal (level too coarse /
                # mislabeled for the surface). NOTE: a minority of these are genuine reader limits
                # rather than data; the `scores` detail lets a human triage.
                flags[doc_id].append(_review_flag(
                    "representative_unreadable", "gate", "data_lattice", "warn",
                    {"scores": scores}))
            elif (scores.get("original", 1.0) < 1.0
                    and scores.get("representative", 0.0) >= 1.0):
                # answer recoverable ONLY when the level is literally rendered (generalized
                # render passes, source render does not) -> the accepted level is not
                # reader-recoverable from the source surface. Strong lattice-profile signal:
                # missing/insufficient answer_aliases or a level too abstract for the surface.
                flags[doc_id].append(_review_flag(
                    "answer_only_readable_when_generalized", "gate", "data_lattice", "warn",
                    {"scores": scores, "subtype": evidence.get("subtype"),
                     "occurrence_ids": evidence.get("occurrence_ids")}))

    # D) a repair that had to lower the answer floor is worth a human look
    for row in (artifact.get("assertions") or {}).values():
        repair = (row.get("evidence") or {}).get("leakage_repair")
        if isinstance(repair, Mapping) and repair.get("floor_lowered"):
            flags[str(row.get("doc_id"))].append(_review_flag(
                "floor_lowered_repair", "compile", "ontology_review", "info",
                {"from_level": repair.get("from_level"), "to_level": repair.get("to_level")}))
        # Hedge/modality observation (non-blocking): the source is conditional/planned but the
        # question is definite -- surfaced for review, the relation was NOT blocked from the reader.
        modality = (row.get("evidence") or {}).get("modality_diagnostics")
        if modality:
            flags[str(row.get("doc_id"))].append(_review_flag(
                "hedged_source_definite_question", "compile", "ontology_review", "info",
                {"diagnostics": list(modality)}))

    # ambiguity MONITOR: a relation whose answered type has >1 co-valid answer in scope. The
    # free-form reader can name a different (also-valid) one -> non-deterministic utility signal.
    # Diagnostic only (info); reported for kept relations and gate-rejected ones alike.
    for row in (artifact.get("assertions") or {}).values():
        competing = (row.get("evidence") or {}).get("answer_competing")
        if competing:
            flags[str(row.get("doc_id"))].append(_review_flag(
                "relation_answer_ambiguous", "compile", "ambiguity", "info",
                {"relation": row.get("relation"), "competing_answers": competing}))
    for record in (artifact.get("rejections") or {}).get("records") or []:
        competing = (record.get("evidence") or {}).get("answer_competing")
        if competing:
            flags[str(record.get("doc_id"))].append(_review_flag(
                "relation_answer_ambiguous", "gate", "ambiguity", "info",
                {"detail_reason": record.get("detail_reason"), "competing_answers": competing}))

    # teacher self-report inconsistency -> the draw is unreliable, re-run it
    for doc_id, rows in (artifact.get("relation_candidate_accounting") or {}).items():
        if any(isinstance(r, Mapping) and r.get("state") == "ledger_inconsistent" for r in rows):
            flags[doc_id].append(_review_flag(
                "ledger_inconsistent", "compile", "teacher_redraw", "warn", {}))

    # teacher-repairable but unresolved: a relation whose only outcomes are FIXABLE-reason
    # rejections (no kept sibling for the same fact) that the repair pass did not rescue -- either
    # the secondary teacher abstained, or it re-proposed and the fix still rejects. These are the
    # relations a better teacher draw could recover; deterministic rewriting is deliberately NOT
    # attempted (surface recolor already ran in _substitute_linked_surfaces; the residue is
    # un-substitutable source text). fix_class teacher_redraw -> re-selectable for a repair re-run.
    def _fact_key(row: Mapping) -> tuple:
        parts = []
        for argument in row.get("arguments") or []:
            parts.append((
                argument.get("role"),
                argument.get("span_label") or argument.get("literal")
                or argument.get("support_property"),
            ))
        return (row.get("relation"), tuple(parts))

    gleaning = artifact.get("relation_gleaning") or {}
    for doc_id, rows in (artifact.get("relation_generation") or {}).items():
        kept_keys = {_fact_key(r) for r in rows if r.get("status") == "kept"}
        by_key: dict[tuple, list[Mapping]] = defaultdict(list)
        for r in rows:
            if r.get("status") == "rejected" and r.get("reason") in _GLEANING_FIXABLE_REASONS:
                by_key[_fact_key(r)].append(r)
        abstained = str((gleaning.get(doc_id) or {}).get("secondary_status")) == "abstained"
        for key, rejected in by_key.items():
            if key in kept_keys:
                continue  # recovered elsewhere -> not an unresolved loss
            run_ids = {r.get("run_id") for r in rejected}
            if "gleaning" in run_ids:
                disposition = "unresolved_after_repair"  # C2 re-proposed, still rejects
            elif abstained:
                disposition = "repair_abstained"          # C2 declined to re-author it
            else:
                disposition = "repair_not_returned"        # C2 ran but did not target it
            latest = rejected[-1]
            try:
                scope = _relation_scope(latest.get("arguments") or [])
            except ValueError:
                scope = None
            flags[doc_id].append(_review_flag(
                "teacher_repairable_unresolved", "gate", "teacher_redraw", "warn",
                {"relation": latest.get("relation"),
                 "reason": latest.get("reason"),
                 "hint": _GLEANING_FIX_HINTS.get(str(latest.get("reason"))),
                 "disposition": disposition,
                 "scope": scope}))

    return {doc_id: doc_flags for doc_id, doc_flags in flags.items()}


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
    secondary_relation_teacher: OpenRouterRelationTeacher | None = None,
    environment_audit: Mapping | None = None,
    relation_support_escalator: "Callable[..., bool] | None" = None,
    informative_context_judge: "Callable[..., bool] | None" = None,
    context_prefilter: "Callable[[str, Mapping], Sequence[Mapping]] | None" = None,
    deterministic_relation_stage: bool = False,
    set_reader: "Callable[[list[str], str], Sequence[str]] | None" = None,
    finer_level_check: "str | bool | None" = None,
) -> dict:
    """Build and validate one artifact through deterministic candidates and optional escalation.

    `relation_support_escalator` (opt-in) recovers opportunity-miner cue-misses without regressing
    the cue-matched set — see relation_support_opportunities' no-regression invariant.
    `informative_context_judge` (opt-in) analogously recovers semantic_property role-cue regex
    misses: free via relation-opportunity membership, else one judge call on the locator sentence.
    `deterministic_relation_stage` (opt-in) turns mined opportunity pairs into template relation
    QAs between the primary teacher pass and gleaning (no teacher call), so gleaning only
    re-authors facts the free deterministic generation could not keep."""
    # Finer-level reward-band check mode: "hard" rejects a QA whose finer band has an unreadable
    # level (routed lattice_suspect, never repair-targeted); "soft" only records/emits it.
    finer_level_mode = (
        "hard" if finer_level_check is True
        else str(finer_level_check) if finer_level_check else None
    )
    if finer_level_mode not in {None, "hard", "soft"}:
        raise ValueError(f"finer_level_check must be 'hard' or 'soft', got {finer_level_check!r}")
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
    escalation_policy = (
        _validate_relation_escalation_policy(
            threshold_manifest.get("relation_escalation_policy")
        ) if secondary_relation_teacher is not None else None
    )
    escalation_enabled = escalation_policy is not None and relation_teacher is not None
    candidates_by_document: dict[str, list[dict]] = {}
    candidate_accounting_by_document: dict[str, list[dict]] = {}
    relation_generation_by_document: dict[str, list[dict]] = {}
    relation_teacher_runs_by_document: dict[str, list[dict]] = {}
    relation_escalation_by_document: dict[str, dict] = {}
    relation_opportunities_by_document: dict[str, list[dict]] = {}
    relation_coverage_by_document: dict[str, dict] = {}
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
        candidate_evidence = candidate.get("evidence") or {}
        # Authorship carries through to the rejection (teacher run vs deterministic pass), so
        # downstream target routing and failure attribution never have to guess it back.
        for authorship_key in ("teacher_id", "run_id"):
            if candidate_evidence.get(authorship_key):
                evidence[authorship_key] = candidate_evidence[authorship_key]
        competing_answers = candidate_evidence.get("answer_competing")
        if competing_answers:  # carry the ambiguity monitor through to the rejection record
            evidence["answer_competing"] = competing_answers
        candidate_arguments = candidate_evidence.get("arguments")
        if candidate_arguments:  # compiled arg identity, so gleaning can key/repair the reject
            evidence["arguments"] = _rejection_safe_arguments(candidate_arguments)
        anchor_diagnostics = candidate_evidence.get("anchor_diagnostics")
        if anchor_diagnostics:
            # Diagnostic-only: a wide cross-clause relation may still be reader
            # validated, but both kept and rejected attempts must remain auditable.
            evidence["anchor_diagnostics"] = list(anchor_diagnostics)
        modality_diagnostics = candidate_evidence.get("modality_diagnostics")
        if modality_diagnostics:
            # Carry the hedge/modality diagnostic onto the rejection so a reader-FAILED
            # hedge-flagged relation can be routed back to repair (see _gleaning_targets).
            evidence["modality_diagnostics"] = list(modality_diagnostics)
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
        candidate_record = _rejection_record(
            reason=reason,
            detail_reason=detail_reason,
            attempt=attempt,
            evidence=evidence,
        )
        if candidate.get("relation"):
            candidate_record["relation"] = candidate.get("relation")
        preserve_rejection(candidate_record, doc_id=doc_id)

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
        # Always retain the cheap structural ledger, even when teacher escalation
        # is disabled. Otherwise a primary-teacher miss is invisible in the QA
        # report and cannot be selectively re-run later.
        # Opt-in LLM prefilter (augment-only) widens the context-literal candidate set before the same
        # cue+judge pipeline runs; None => byte-identical to the gazetteer-only path.
        extra_context = (
            context_prefilter(source, environment_document) if context_prefilter is not None else None
        )
        opportunities = relation_support_opportunities(
            source, environment_document, escalator=relation_support_escalator,
            extra_context_candidates=extra_context,
        )
        relation_opportunities_by_document[doc_id] = opportunities
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
                protected_terms = list(dict.fromkeys(
                    term
                    for occurrence in linked_occurrences
                    for term in _occurrence_protected_terms(occurrence)
                ))
                # Co-referent leak guard (Case 1): a DIFFERENT controlled decision whose
                # surface contains a protected term as a whole word ("acid reflux" holds
                # "reflux") is generalized in the deployed doc_p, but the isolation anchor
                # KEEPs it -- spuriously surviving the identity. Hide those decisions here
                # so both the leak check and the reader gate match deployment.
                representative_vector = dict(anchor["action_vector"])
                for decision in decisions:
                    decision_id = str(decision["decision_id"])
                    surface = str(decision.get("canonical_key") or "")
                    if surface and any(
                        canon(surface) != canon(term) and _contains(surface, term)
                        for term in protected_terms
                    ):
                        hiding = _hiding_action_id(
                            decision,
                            excluded_level_fill_keys=_selected_level_fill_keys(
                                decisions,
                                representative_vector,
                                exclude_decision_id=decision_id,
                            ),
                        )
                        if hiding is not None:
                            representative_vector[decision_id] = hiding
                representative_context = render_action_vector(doc_id, representative_vector)
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
                        set_reader=set_reader,
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
                    detail_reason = (
                        "reader_unstable" if reason == "unstable"
                        else "placeholder_answerable" if reason == "floor_answerable"
                        else "three_point_gate_failed"
                    )
                    extra_evidence = None
                    scores = evidence_row.get("scores") or {}
                    if (detail_reason == "three_point_gate_failed"
                            and scores.get("original", 0.0) >= reader_threshold
                            and scores.get("representative", 0.0) < reader_threshold):
                        # relation supported, generalized level not -> is the CHOSEN
                        # level the problem (a coarser one in the chain reads)? if so it's
                        # very likely a lattice_profiles.json data issue -- surface it.
                        suspect = _diagnose_coarser_readable(
                            candidate, decisions, protected_terms,
                            doc_id=doc_id, render_action_vector=render_action_vector,
                            reader=reader, chain_by_decision=chain_by_decision,
                            reader_threshold=reader_threshold,
                        )
                        if suspect is not None:
                            extra_evidence = {"lattice_probe": suspect}
                    reject_context_candidate(
                        doc_id=doc_id,
                        candidate=candidate,
                        reason=reason,
                        detail_reason=detail_reason,
                        anchor=anchor,
                        validation=evidence_row,
                        extra_evidence=extra_evidence,
                    )
                    continue
                row = dict(candidate)
                row["expected_action_support"] = {
                    "joint_anchor_action_vector": anchor["action_vector"],
                    "joint_anchor_hash": anchor["action_vector_hash"],
                    "property_level": requirements,
                }
                # Opt-in reward-band certification: per-finer-level readability of the ANSWER
                # decision. soft: record on the kept row and emit to the worklist. hard: an
                # unreadable finer level REJECTS the QA -- and the rejection routes as
                # lattice_suspect (data-owned), so it can never become a repair/glean target.
                if finer_level_mode:
                    finer_scores = _finer_level_readability(
                        candidate, decisions, protected_terms,
                        doc_id=doc_id, render_action_vector=render_action_vector,
                        reader=reader, chain_by_decision=chain_by_decision,
                    )
                    if finer_scores is not None:
                        evidence_row = {**evidence_row, "finer_levels": finer_scores}
                        if finer_level_mode == "hard" and any(
                                entry["score"] < reader_threshold
                                for entry in finer_scores.values()):
                            reject_context_candidate(
                                doc_id=doc_id,
                                candidate=candidate,
                                reason="unsupported",
                                detail_reason="finer_level_unreadable",
                                anchor=anchor,
                                validation=evidence_row,
                                extra_evidence={
                                    # what the worklist emitter needs, since a hard-mode
                                    # failure never becomes an assertion row
                                    "finer_levels": finer_scores,
                                    "question": candidate.get("question"),
                                    "answer_target": dict(
                                        candidate.get("answer_target") or {}),
                                },
                            )
                            continue
                row["evidence"] = {
                    **dict(row.get("evidence") or {}),
                    "validation": evidence_row,
                }
                accepted_rows.append(row)
            return accepted_rows

        # kwargs only when the judge is on: one flag governs the whole escalation (including the
        # free relation-reuse tier), and adapters without the kwargs keep working when it's off.
        deterministic_kwargs = (
            {"relation_opportunities": opportunities,
             "context_judge": informative_context_judge}
            if informative_context_judge is not None else {}
        )
        deterministic_records = [
            dict(row) for row in task_adapter.deterministic_candidates(
                doc_id, source, environment_document, **deterministic_kwargs
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
        pre_teacher_accepted = list(accepted)
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
                if escalation_enabled:
                    opportunity_counts = {
                        scope: sum(row["scope"] == scope for row in opportunities)
                        for scope in _RELATION_ESCALATION_SCOPES
                    }
                    escalation_targets = relation_escalation_targets(
                        opportunity_counts, escalation_policy,
                    )
                    relation_escalation_by_document[doc_id] = {
                        "policy_version": escalation_policy["version"],
                        "opportunity_counts": opportunity_counts,
                        "opportunity_fact_key_hashes": [
                            _stable_hash(row["fact_key"]) for row in opportunities
                        ],
                        "targets": escalation_targets,
                        "primary_kept_counts": {},
                        "triggered": False,
                        "secondary_status": "not_needed",
                    }
                primary_accepted_relations: list[dict] = []
                primary_proposal_count = 0
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
                    primary_proposal_count = len(proposals)
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
                    if escalation_enabled:
                        relation_teacher_runs_by_document.setdefault(doc_id, []).append({
                            "teacher_id": "gpt_oss",
                            "run_id": "primary",
                            "prompt_hash": prompt_hash,
                            "teacher_pin_hash": _stable_hash(getattr(relation_teacher, "pin", {})),
                            "proposal_count": len(proposals),
                            "candidate_accounting": deepcopy(
                                candidate_accounting_by_document.get(doc_id, [])
                            ),
                            "status": "proposed" if proposals else "abstained",
                        })
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
                        if escalation_enabled:
                            for attempt in relation_attempts:
                                attempt.update({"teacher_id": "gpt_oss", "run_id": "primary"})
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
                                **(
                                    {"teacher_id": "gpt_oss", "run_id": "primary"}
                                    if escalation_enabled else {}
                                ),
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
                        primary_accepted_relations = accepted_relations
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
                        if escalation_enabled:
                            relation_teacher_runs_by_document[doc_id][-1]["status"] = (
                                "kept" if accepted_relations else "rejected"
                            )
                        relation_generation_by_document[doc_id] = relation_attempts
                        accepted.extend(accepted_relations)
                        contextual_relation_count = sum(
                            row.get("family") == "context"
                            and row.get("subtype") == "contextual_relation"
                            for row in accepted
                        )
                        if (
                            contextual_relation_count < min_contextual_relations
                            and not escalation_enabled
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
                    if escalation_enabled:
                        relation_teacher_runs_by_document.setdefault(doc_id, []).append({
                            "teacher_id": "gpt_oss",
                            "run_id": "primary",
                            "prompt_hash": prompt_hash,
                            "teacher_pin_hash": _stable_hash(getattr(relation_teacher, "pin", {})),
                            "proposal_count": primary_proposal_count,
                            "candidate_accounting": [],
                            "status": "failed",
                            "error_code": error_code,
                        })
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
                # Deterministic relation stage (opt-in): mined opportunity pairs become template
                # QAs BETWEEN the primary pass and gleaning, so gleaning only re-authors facts the
                # free deterministic generation could not keep. Span pairs ride the normal compile
                # path (forward with reverse fallback for singletons, reverse-only for ambiguous
                # groups); literal pairs ride the literal-reverse builder with the seed widened
                # from judge-recovered to every accepted opportunity. Answer levels are searched
                # coarsest->finest, so the first gate pass is the coarsest supported level (the
                # teacher-prior semantics, without a teacher).
                stage_kept_relations: list[dict] = []
                if deterministic_relation_stage:
                    stage_decisions_by_id = {
                        str(decision["decision_id"]): decision for decision in decisions
                    }
                    occurrence_decision = {
                        occurrence_id: str(occurrence["decision_id"])
                        for occurrence_id, occurrence in occurrences.items()
                        if occurrence.get("decision_id") is not None
                    }
                    span_label_by_decision: dict[str, str] = {}
                    for inventory_row in relation_teacher_span_inventory(environment_document):
                        inventory_decision = occurrence_decision.get(
                            str(inventory_row.get("occurrence_id")))
                        if inventory_decision is not None:
                            span_label_by_decision.setdefault(
                                inventory_decision, str(inventory_row.get("span_label")))
                    stage_kept_fact_keys = set()
                    for row in accepted:
                        try:
                            stage_kept_fact_keys.add(
                                _compiled_relation_fact_key(row, occurrences))
                        except ValueError:
                            pass

                    def _stage_trial(row: dict, *, final: bool) -> list[dict]:
                        # Intermediate level/direction trials are speculative: their expected
                        # failures must not flood the rejection channel (or shift gleaning-target
                        # kinds), so only a plan's FINAL trial may leave rejection records.
                        # EXCEPT finer-level-unreadable rejections (hard finer-level check): the
                        # level search legitimately moves on to a finer supported level, but the
                        # unreadable level is lattice-worklist signal that must survive -- it is
                        # routed out of repair targeting regardless.
                        checkpoint = len(rejection_records)
                        kept_rows = validate_candidate_rows([row])
                        if not kept_rows and not final:
                            preserved = [
                                record for record in rejection_records[checkpoint:]
                                if record.get("detail_reason") == "finer_level_unreadable"
                            ]
                            del rejection_records[checkpoint:]
                            rejection_records.extend(preserved)
                        return kept_rows

                    def _stage_keep(row: dict) -> None:
                        accepted.append(row)
                        stage_kept_relations.append(row)
                        try:
                            stage_kept_fact_keys.add(
                                _compiled_relation_fact_key(row, occurrences))
                        except ValueError:
                            pass
                        relation_generation_by_document.setdefault(doc_id, []).append({
                            "relation": row.get("relation"),
                            "arguments": list(
                                dict(row.get("evidence") or {}).get("arguments") or []),
                            "question": row.get("question"),
                            "accepted_answers": row.get("accepted_values"),
                            "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
                            "status": "kept", "reason": "accepted",
                            "teacher_id": "deterministic", "run_id": "deterministic_stage",
                        })

                    stage_plans = _deterministic_relation_plans(
                        opportunities, occurrences, stage_decisions_by_id)
                    for plan in stage_plans:
                        if plan.get("compound"):
                            # Ambiguous-group fallbacks, run after the group's per-object flips
                            # (plan order) and only while some pair is still unkept. Strategy 1:
                            # set-valued forward (validated probe; single trial, finest-level
                            # members). Strategy 2: compound span-locator reverse, answer-level
                            # searched coarsest->finest.
                            if all(key in stage_kept_fact_keys for key in plan["fact_keys"]):
                                continue
                            set_row = _set_forward_row(
                                source, plan["relation"], plan["subject"], plan["objects"],
                                occurrences, stage_decisions_by_id)
                            if set_row is not None:
                                kept_rows = _stage_trial(set_row, final=False)
                                if kept_rows:
                                    _stage_keep(kept_rows[0])
                                    stage_kept_fact_keys.update(plan["fact_keys"])
                                    continue
                            answer_decision = stage_decisions_by_id.get(occurrence_decision.get(
                                str(plan["subject"].get("occurrence_id")), ""))
                            stage_levels = _deterministic_answer_levels(answer_decision or {})
                            for level_index, level in enumerate(stage_levels):
                                compound_row = _compound_span_reverse_row(
                                    source, plan["relation"], plan["subject"], plan["objects"],
                                    occurrences, stage_decisions_by_id, level)
                                if compound_row is None:
                                    break
                                kept_rows = _stage_trial(
                                    compound_row, final=level_index == len(stage_levels) - 1)
                                if kept_rows:
                                    _stage_keep(kept_rows[0])
                                    stage_kept_fact_keys.update(plan["fact_keys"])
                                    break
                            continue
                        if plan["fact_key"] in stage_kept_fact_keys:
                            continue
                        trials = []
                        for direction in plan["directions"]:
                            answer_argument = (plan["object"] if direction == "forward"
                                               else plan["subject"])
                            answer_decision = stage_decisions_by_id.get(occurrence_decision.get(
                                str(answer_argument.get("occurrence_id")), ""))
                            trials.extend(
                                (direction, level)
                                for level in _deterministic_answer_levels(answer_decision or {})
                            )
                        for trial_index, (direction, level) in enumerate(trials):
                            final = trial_index == len(trials) - 1
                            proposal = _deterministic_stage_proposal(
                                plan, direction, level, occurrences, stage_decisions_by_id,
                                span_label_by_decision)
                            if proposal is None:
                                continue
                            trial_rows, trial_rejections = task_adapter.compile_relations(
                                doc_id, source, environment_document, [proposal])
                            if not trial_rows:
                                if final:
                                    for rejection in trial_rejections:
                                        rejection = dict(rejection)
                                        rejection["evidence"] = {
                                            **dict(rejection.get("evidence") or {}),
                                            "teacher_id": "deterministic",
                                            "run_id": "deterministic_stage",
                                        }
                                        preserve_rejection(rejection, doc_id=doc_id)
                                continue
                            trial_row = dict(trial_rows[0])
                            trial_row["evidence"] = {
                                **dict(trial_row.get("evidence") or {}),
                                "teacher_id": "deterministic",
                                "run_id": "deterministic_stage",
                            }
                            kept_rows = _stage_trial(trial_row, final=final)
                            if kept_rows:
                                _stage_keep(kept_rows[0])
                                break

                    stage_literal_groups = _literal_reverse_groups(
                        opportunities, occurrences, judge_recovered_only=False)
                    for (stage_relation, stage_decision_id), group in sorted(
                            stage_literal_groups.items()):
                        decision = stage_decisions_by_id.get(stage_decision_id)
                        # drop literals whose (relation, condition, literal) fact is already kept
                        remaining = {
                            key: entry for key, entry in group["literals"].items()
                            if entry.get("fact_key") not in stage_kept_fact_keys
                        }
                        if not remaining or decision is None:
                            continue
                        trial_group = {**group, "literals": remaining}
                        stage_levels = _deterministic_answer_levels(decision)
                        for level_index, level in enumerate(stage_levels):
                            literal_row = _literal_reverse_row(
                                source, stage_relation, stage_decision_id, trial_group,
                                occurrences, level)
                            literal_row["evidence"] = {
                                **literal_row["evidence"], "run_id": "deterministic_stage",
                            }
                            kept_rows = _stage_trial(
                                literal_row, final=level_index == len(stage_levels) - 1)
                            if kept_rows:
                                _stage_keep(kept_rows[0])
                                break
                    if escalation_enabled:
                        relation_escalation_by_document[doc_id]["deterministic_stage"] = {
                            "plan_count": len(stage_plans),
                            "literal_group_count": len(stage_literal_groups),
                            "kept_count": len(stage_kept_relations),
                        }
                if escalation_enabled:
                    escalation = relation_escalation_by_document[doc_id]
                    primary_accepted_relations, _ = merge_kept_relation_rows(
                        primary_accepted_relations, [], occurrences,
                    )
                    primary_counts = {
                        scope: sum(
                            _relation_scope(
                                list(dict(row.get("evidence") or {}).get("arguments") or [])
                            ) == scope
                            for row in primary_accepted_relations
                        )
                        for scope in _RELATION_ESCALATION_SCOPES
                    }
                    escalation["primary_kept_counts"] = primary_counts
                    gleaning_targets = _gleaning_targets(
                        source,
                        # Stage keeps count as covered: a fact the deterministic stage already
                        # kept must not be re-authored by the paid gleaning teacher.
                        [*primary_accepted_relations, *stage_kept_relations],
                        [r for r in rejection_records if r.get("doc_id") == doc_id],
                        opportunities,
                        occurrences,
                        reader_threshold=reader_threshold,
                    )
                    escalation["gleaning"] = {
                        "target_count": len(gleaning_targets),
                        "target_kinds": {
                            kind: sum(t["kind"] == kind for t in gleaning_targets)
                            for kind in ("ambiguous", "fixable", "missed")
                        },
                        "targets": [
                            {
                                "kind": t["kind"],
                                "relation": t.get("relation"),
                                "reason": t.get("reason"),
                                "hint": t.get("hint"),
                                "fact_key_hash": _stable_hash(t["fact_key"]),
                            }
                            for t in gleaning_targets
                        ],
                        "returned_count": 0,
                        "repair_prompt_hash": None,
                    }
                    if gleaning_targets:
                        escalation["triggered"] = True
                        # Batch the targets: one teacher CALL per <=N targets so a note with many
                        # opportunities is not crammed into one unfocusable prompt. All batches'
                        # proposals are compiled/validated together below (single pass).
                        target_batches = [
                            gleaning_targets[i:i + RELATION_REPAIR_MAX_TARGETS_PER_CALL]
                            for i in range(0, len(gleaning_targets),
                                           RELATION_REPAIR_MAX_TARGETS_PER_CALL)
                        ]
                        repair_prompt_hashes: list[str] = []
                        secondary_accounting: list[dict] = []
                        secondary_attempts: list[dict] = []
                        secondary_accepted: list[dict] = []
                        secondary_proposals: list = []
                        secondary_proposal_count = 0
                        repair_prompt_hash = None
                        phase_run_records: list[dict] = []
                        try:
                            for batch in target_batches:
                                batch_shown_labels: set[str] = set()
                                repair_prompt = relation_repair_prompt(
                                    doc_id, source, environment_document, batch,
                                    shown_labels_out=batch_shown_labels,
                                )
                                batch_hash = _stable_hash(repair_prompt)
                                repair_prompt_hashes.append(batch_hash)
                                if isinstance(secondary_relation_teacher, OpenRouterRelationTeacher):
                                    # Scope the response schema to the batch's SHOWN labels: the teacher
                                    # cannot emit, or be forced to account for, a label absent from its
                                    # rendered region (the repair prompt only shows the target regions).
                                    batch_proposals = secondary_relation_teacher.propose(
                                        repair_prompt,
                                        response_format=relation_teacher_response_format(
                                            environment_document, source,
                                            allowed_labels=batch_shown_labels,
                                        ),
                                    )
                                else:
                                    batch_proposals = secondary_relation_teacher.propose(repair_prompt)
                                # Deterministic backstop for a non-strict provider: drop any secondary
                                # proposal whose linked argument references a label outside the batch's
                                # shown set (the scoped enum should already prevent this).
                                if isinstance(batch_proposals, RelationTeacherProposals):
                                    kept = [
                                        proposal for proposal in batch_proposals
                                        if all(
                                            str(argument.get("span_label")) in batch_shown_labels
                                            for argument in (proposal.get("arguments") or [])
                                            if argument.get("kind") == "linked"
                                            and argument.get("span_label") is not None
                                        )
                                    ]
                                    batch_proposals = RelationTeacherProposals(
                                        kept, batch_proposals.candidate_accounting)
                                batch_accounting: list[dict] = []
                                if isinstance(batch_proposals, RelationTeacherProposals):
                                    try:
                                        batch_accounting = _validated_candidate_accounting(
                                            batch_proposals.candidate_accounting,
                                            environment_document, source,
                                        )
                                    except ValueError as error:
                                        batch_accounting = [{
                                            "state": "ledger_inconsistent",
                                            "reason": "invalid_teacher_candidate_accounting",
                                            "detail": str(error),
                                        }]
                                secondary_accounting.extend(batch_accounting)
                                run_record = {
                                    "teacher_id": "gpt_oss",
                                    "run_id": "gleaning",
                                    "prompt_hash": batch_hash,
                                    "teacher_pin_hash": _stable_hash(
                                        getattr(secondary_relation_teacher, "pin", {})
                                    ),
                                    "proposal_count": len(batch_proposals),
                                    "candidate_accounting": deepcopy(batch_accounting),
                                    "status": "proposed" if batch_proposals else "abstained",
                                }
                                relation_teacher_runs_by_document.setdefault(doc_id, []).append(run_record)
                                phase_run_records.append(run_record)
                                secondary_proposals.extend(batch_proposals)
                            secondary_proposal_count = len(secondary_proposals)
                            # Phase-level hash over the batch prompts groups this phase's
                            # candidates/rejections; the per-call hashes live in the run records.
                            repair_prompt_hash = _stable_hash(repair_prompt_hashes)
                            escalation["gleaning"]["repair_prompt_hash"] = repair_prompt_hash
                            escalation["gleaning"]["repair_prompt_hashes"] = repair_prompt_hashes
                            escalation["gleaning"]["batch_count"] = len(target_batches)
                            if secondary_proposals:
                                secondary_attempts = [{
                                    "proposal_index": proposal_index,
                                    "relation": proposal.get("relation"),
                                    "arguments": proposal.get("arguments"),
                                    "question": proposal.get("question"),
                                    "accepted_answers": proposal.get("accepted_answers"),
                                    "scoring_contract": proposal.get("scoring_contract"),
                                    "status": "rejected",
                                    "reason": "uncompiled",
                                    "teacher_id": "gpt_oss",
                                    "run_id": "gleaning",
                                } for proposal_index, proposal in enumerate(secondary_proposals)]
                                secondary_candidates, secondary_rejections = task_adapter.compile_relations(
                                    doc_id, source, environment_document, secondary_proposals,
                                )
                                secondary_candidates = [dict(row) for row in secondary_candidates]
                                for candidate in secondary_candidates:
                                    candidate["evidence"] = {
                                        **dict(candidate.get("evidence") or {}),
                                        "prompt_hash": repair_prompt_hash,
                                        "teacher_id": "gpt_oss",
                                        "run_id": "gleaning",
                                    }
                                attempts_by_proposal_hash = {
                                    _stable_hash(proposal): attempt
                                    for proposal, attempt in zip(secondary_proposals, secondary_attempts)
                                }
                                attempts_by_candidate_hash = {
                                    _stable_hash(candidate): attempts_by_proposal_hash.get(
                                        candidate.get("evidence", {}).get("proposal_hash")
                                    )
                                    for candidate in secondary_candidates
                                }
                                for rejection in secondary_rejections:
                                    rejection = dict(rejection)
                                    rejection["evidence"] = {
                                        **dict(rejection.get("evidence") or {}),
                                        "teacher_id": "gpt_oss",
                                        "run_id": "gleaning",
                                    }
                                    preserve_rejection(rejection, doc_id=doc_id)
                                    proposal_index = rejection.get("proposal_index")
                                    if isinstance(proposal_index, int) and proposal_index < len(secondary_attempts):
                                        secondary_attempts[proposal_index].update({
                                            "status": "rejected",
                                            "reason": rejection.get("detail_reason", rejection.get("reason")),
                                        })
                                rejection_count_before_validation = len(rejection_records)
                                secondary_accepted = validate_candidate_rows(secondary_candidates)
                                for row in secondary_accepted:
                                    attempt = attempts_by_proposal_hash.get(
                                        row.get("evidence", {}).get("proposal_hash")
                                    )
                                    if attempt is not None:
                                        attempt.update({"status": "kept", "reason": "accepted"})
                                for rejection in rejection_records[rejection_count_before_validation:]:
                                    rejection["evidence"] = {
                                        **dict(rejection.get("evidence") or {}),
                                        "teacher_id": "gpt_oss",
                                        "run_id": "gleaning",
                                    }
                                    attempt = attempts_by_candidate_hash.get(
                                        rejection.get("evidence", {}).get("candidate_hash")
                                    )
                                    if attempt is not None:
                                        attempt.update({
                                            "status": "rejected",
                                            "reason": rejection.get("detail_reason", rejection.get("reason")),
                                        })
                                phase_rollup = "kept" if secondary_accepted else "rejected"
                                for run_record in phase_run_records:
                                    if run_record["status"] == "proposed":
                                        run_record["status"] = phase_rollup
                            relation_generation_by_document.setdefault(doc_id, []).extend(secondary_attempts)
                            escalation["secondary_status"] = (
                                "kept" if secondary_accepted else "abstained"
                            )
                        except Exception as error:
                            error_code = (
                                error.code if isinstance(error, RelationTeacherResponseError)
                                else "teacher_generation_failed"
                            )
                            # a batch may have failed mid-phase; anchor the failed run to whatever
                            # batch prompts were built (phase hash) or None if none were.
                            repair_prompt_hash = repair_prompt_hash or (
                                _stable_hash(repair_prompt_hashes) if repair_prompt_hashes else None
                            )
                            escalation["gleaning"]["repair_prompt_hash"] = repair_prompt_hash
                            escalation["gleaning"]["repair_prompt_hashes"] = repair_prompt_hashes
                            escalation["gleaning"]["batch_count"] = len(target_batches)
                            relation_teacher_runs_by_document.setdefault(doc_id, []).append({
                                "teacher_id": "gpt_oss",
                                "run_id": "gleaning",
                                "prompt_hash": repair_prompt_hash,
                                "teacher_pin_hash": _stable_hash(
                                    getattr(secondary_relation_teacher, "pin", {})
                                ),
                                "proposal_count": secondary_proposal_count,
                                "candidate_accounting": deepcopy(secondary_accounting),
                                "status": "failed",
                                "error_code": error_code,
                            })
                            escalation["secondary_status"] = error_code
                            preserve_rejection(_rejection_record(
                                reason="generation_failed",
                                detail_reason=error_code,
                                attempt={
                                    "doc_id": doc_id,
                                    "prompt_hash": repair_prompt_hash,
                                    "teacher_id": "gpt_oss",
                                    "run_id": "gleaning",
                                    "definition_version": "contextual-relation-v1",
                                },
                                evidence={
                                    "source": "relation_teacher",
                                    "prompt_hash": repair_prompt_hash,
                                    "teacher_id": "gpt_oss",
                                    "run_id": "gleaning",
                                    "error_type": type(error).__name__,
                                    "error_code": error_code,
                                },
                            ), doc_id=doc_id)
                        # A GLEANING row duplicating a pair fact the stage already covers is
                        # dropped before the merge (the stage version is gate-kept and free; the
                        # teacher was told not to re-emit unlisted facts). Only the secondary
                        # side is filtered -- a primary row overlapping a stage COMPOUND row's
                        # pair set is legitimate and must stay primary-preferred.
                        stage_pair_keys = _pair_fact_keys(stage_kept_relations, occurrences)
                        if stage_pair_keys:
                            deduped_secondary = []
                            for row in secondary_accepted:
                                try:
                                    duplicate = _compiled_relation_fact_key(
                                        row, occurrences) in stage_pair_keys
                                except ValueError:
                                    duplicate = False
                                if not duplicate:
                                    deduped_secondary.append(row)
                            secondary_accepted = deduped_secondary
                        merged_relations, disposition = merge_kept_relation_rows(
                            primary_accepted_relations, secondary_accepted, occurrences,
                        )
                        escalation["gleaning"]["returned_count"] = len(secondary_accepted)
                        # Deterministic-stage keeps REJOIN the rebuilt kept set: they are neither
                        # primary nor secondary rows, and compound stage rows (>2 args) cannot
                        # ride merge_kept_relation_rows.
                        accepted = pre_teacher_accepted + stage_kept_relations + merged_relations
                        contextual_relation_count = sum(
                            row.get("family") == "context"
                            and row.get("subtype") == "contextual_relation"
                            for row in accepted
                        )
                        escalation["secondary_kept_counts"] = {
                            scope: sum(
                                _relation_scope(
                                    list(dict(row.get("evidence") or {}).get("arguments") or [])
                                ) == scope
                                for row in secondary_accepted
                            )
                            for scope in _RELATION_ESCALATION_SCOPES
                        }
                        escalation["merge_disposition"] = disposition
                        escalation["merged_kept_counts"] = {
                            scope: sum(
                                _relation_scope(
                                    list(dict(row.get("evidence") or {}).get("arguments") or [])
                                ) == scope
                                for row in merged_relations
                            )
                            for scope in _RELATION_ESCALATION_SCOPES
                        }
                    else:
                        escalation["secondary_kept_counts"] = {
                            scope: 0 for scope in _RELATION_ESCALATION_SCOPES
                        }
                        escalation["merge_disposition"] = {
                            "primary_only": len(primary_accepted_relations),
                            "primary_preferred": 0,
                            "secondary_only": 0,
                        }
                        escalation["merged_kept_counts"] = dict(primary_counts)
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
        # Source 1: doc-global reverse-orientation ambiguity recovery. Over ALL forward attempts for
        # the doc (primary + gleaning combined), flip every object in an ambiguous (relation, subject)
        # group and gate the reverse QAs in an ISOLATED pass -- additive (cannot perturb the forward
        # fact-group dedup) and doc-global (sees the full {objects} set, unlike the per-pass path).
        # Only flip forwards that actually GROUNDED (kept, or reader-gate reasons = reached the
        # reader). A forward rejected at grounding/type would only regenerate the same failure in
        # reverse, cluttering the rejection channel with junk.
        _GROUNDED_REASONS = {"three_point_gate_failed", "unstable", "placeholder_answerable",
                             "answer_leakage", "duplicate_fact_group"}
        doc_attempts = [att for att in (relation_generation_by_document.get(doc_id) or [])
                        if att.get("status") == "kept" or att.get("reason") in _GROUNDED_REASONS]
        _adapter_supports_reverse = hasattr(task_adapter, "compile_relations") and (
            "reverse_framing_only" in inspect.signature(task_adapter.compile_relations).parameters)
        if relation_teacher is not None and doc_attempts and _adapter_supports_reverse:
            # Source 2: also seed {objects} with JUDGE-ACCEPTED opportunities (recovered_by_escalation)
            # -- source-supported objects the teacher never proposed forward. Treated as forward
            # (answer=object) so the flip generates object->subject. Opportunity args carry only
            # occurrence_id; stamp each linked arg with its decision's inventory span_label so the
            # variant takes the v4 anchor path (like teacher proposals) instead of the legacy
            # evidence path (which needs an evidence_quote opportunities lack -> invalid_evidence).
            _inv = relation_teacher_span_inventory(environment_document)
            _occ2dec = {str(o.get("occurrence_id")): o.get("decision_id")
                        for o in environment_document.get("occurrences", [])}
            _dec2label: dict = {}
            for _row in _inv:
                _dec = _occ2dec.get(str(_row.get("occurrence_id")))
                if _dec is not None:
                    _dec2label.setdefault(_dec, _row.get("span_label"))

            def _stamp_span_label(arg):
                if arg.get("kind") != "linked" or arg.get("span_label"):
                    return arg
                label = _dec2label.get(_occ2dec.get(str(arg.get("occurrence_id"))))
                return {**arg, "span_label": label} if label else arg

            def _typed_opp(o):
                # keep only opportunities whose arguments satisfy the relation's arg-type contract
                # (subject=condition, object=treatment/test); drops condition->condition, finding-
                # or reagent-object opportunities that would else generate invalid_argument_types.
                args = o.get("arguments") or []
                subj = next((a for a in args if a.get("role") == "subject"), None)
                obj = next((a for a in args if a.get("role") == "object"), None)
                if subj is None or obj is None:
                    return None
                if not _relation_arguments_are_legal(o["relation"], [subj, obj], ACI_RELATION_CONTRACT):
                    return None
                return {"relation": o["relation"],
                        "arguments": [_stamp_span_label(subj), _stamp_span_label(obj)]}

            judge_opps = [
                opp for o in opportunities
                if o.get("recovered_by_escalation") and o.get("relation") in _REVERSE_FRAME_TEMPLATES
                and (opp := _typed_opp(o)) is not None
            ]
            reverse_inputs = [*doc_attempts, *judge_opps]
            rev_candidates, rev_compile_rej = task_adapter.compile_relations(
                doc_id, source, environment_document, reverse_inputs, reverse_framing_only=True,
            )
            for rejection in rev_compile_rej:
                rejection = dict(rejection)
                rejection["evidence"] = {**dict(rejection.get("evidence") or {}),
                                         "teacher_id": "deterministic", "run_id": "reverse_framing"}
                preserve_rejection(rejection, doc_id=doc_id)
            if rev_candidates:
                kept_keys = set()
                for row in accepted:
                    try:
                        kept_keys.add(_compiled_relation_fact_key(row, occurrences))
                    except ValueError:
                        pass
                # Skip a reverse whose fact is already kept forward BEFORE gating -- it is redundant,
                # and gating it only wastes reader calls and pollutes the rejection channel.
                rev_rows = []
                for r in rev_candidates:
                    try:
                        if _compiled_relation_fact_key(r, occurrences) in kept_keys:
                            continue
                    except ValueError:
                        pass
                    row = dict(r)
                    row["evidence"] = {**dict(row.get("evidence") or {}),
                                       "teacher_id": "deterministic", "run_id": "reverse_framing"}
                    rev_rows.append(row)
                for r in validate_candidate_rows(rev_rows):
                    accepted.append(r)
                    relation_generation_by_document.setdefault(doc_id, []).append({
                        "relation": r.get("relation"),
                        "arguments": list(dict(r.get("evidence") or {}).get("arguments") or []),
                        "question": r.get("question"),
                        "accepted_answers": r.get("accepted_values"),
                        "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
                        "status": "kept", "reason": "accepted",
                        "teacher_id": "deterministic", "run_id": "reverse_framing",
                    })
        # Deterministic literal->span reverse assertions: judge-accepted span_literal opportunities
        # (recall widened by the LLM prefilter when enabled) become condition-answer QAs whose locator
        # is the literal object(s). Additive, gated by the same reader; the condition answer is
        # controlled so the placeholder render hides it. No teacher call. When the deterministic
        # stage ran, it already emitted these from the WIDER all-accepted seed (with answer-level
        # search), so the narrow pass is skipped to avoid duplicating kept rows.
        literal_reverse_rows = (
            [] if deterministic_relation_stage else _literal_reverse_assertions(
                source, opportunities, occurrences,
                {str(decision["decision_id"]): decision for decision in decisions},
            )
        )
        if literal_reverse_rows:
            for r in validate_candidate_rows(literal_reverse_rows):
                accepted.append(r)
                relation_generation_by_document.setdefault(doc_id, []).append({
                    "relation": r.get("relation"),
                    "arguments": list(dict(r.get("evidence") or {}).get("arguments") or []),
                    "question": r.get("question"),
                    "accepted_answers": r.get("accepted_values"),
                    "scoring_contract": {"kind": "semantic_qa", "match": "fact_score"},
                    "status": "kept", "reason": "accepted",
                    "teacher_id": "deterministic", "run_id": "literal_reverse",
                })
        doc_rejections = [row for row in rejection_records if row.get("doc_id") == doc_id]
        coverage_targets = _gleaning_targets(
            source,
            [row for row in accepted
             if row.get("family") == "context" and row.get("subtype") == "contextual_relation"],
            doc_rejections,
            opportunities,
            occurrences,
            reader_threshold=reader_threshold,
        )
        # Rejections the reader-outcome router excluded from repair, surfaced for their real
        # owners: lattice_suspect -> a lattice_profiles data fix; no_relation -> reader-verified
        # miner co-occurrence junk (kept for miner-precision analysis, never re-authored).
        routed_out: dict[str, list[dict]] = {}
        for row in doc_rejections:
            route = _reader_outcome_route(row, reader_threshold)
            if route is None:
                continue
            evidence = row.get("evidence") or {}
            scores = (evidence.get("validation") or {}).get("scores") or {}
            routed_out.setdefault(route, []).append({
                "relation": row.get("relation"),
                "detail_reason": row.get("detail_reason"),
                "run_id": evidence.get("run_id"),
                "scores": {key: scores.get(key) for key in
                           ("original", "representative", "placeholder")},
                "rejection_id": row.get("rejection_id"),
            })
        relation_coverage_by_document[doc_id] = {
            "opportunity_count": len(opportunities),
            "unresolved_targets": [
                {
                    "kind": row["kind"],
                    "relation": row.get("relation"),
                    "reason": row.get("reason"),
                    "hint": row.get("hint"),
                    "fact_key_hash": _stable_hash(row["fact_key"]),
                    "evidence_span": row.get("evidence_span"),
                }
                for row in coverage_targets
            ],
            **({"reader_routed_out": routed_out} if routed_out else {}),
        }
        candidates_by_document[doc_id] = accepted

    artifact_pins = {
        **dict(pins),
        "reader_pin": reader_pin,
        "threshold_manifest": dict(threshold_manifest),
    }
    if escalation_enabled:
        primary_pin = getattr(relation_teacher, "pin", None)
        secondary_pin = getattr(secondary_relation_teacher, "pin", None)
        if not all(isinstance(value, Mapping) and value for value in (primary_pin, secondary_pin)):
            raise ValueError("enabled relation escalation requires explicit primary and secondary pins")
        artifact_pins.update({
            "relation_teacher_pins": {
                "primary": dict(primary_pin),
                "secondary": dict(secondary_pin),
            },
            "relation_escalation_policy": dict(escalation_policy),
        })
    artifact = package_utility_artifact(
        frozen_environment,
        candidates_by_document,
        family_budgets=family_budgets,
        structural_cap=threshold_manifest.get("structural_cap"),
        pins=artifact_pins,
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
    artifact["relation_support_opportunities"] = relation_opportunities_by_document
    artifact["relation_coverage"] = relation_coverage_by_document
    if escalation_enabled:
        artifact["relation_teacher_runs"] = relation_teacher_runs_by_document
        artifact["relation_escalation"] = relation_escalation_by_document
        artifact["relation_gleaning"] = {
            doc_id: {
                "triggered": record.get("triggered", False),
                "secondary_status": record.get("secondary_status"),
                "merge_disposition": record.get("merge_disposition"),
                **(record.get("gleaning") or {}),
            }
            for doc_id, record in relation_escalation_by_document.items()
        }
    # Diagnostic classification of the finished artifact. Deterministic from its
    # contents, so it is included in the hashed payload (keeps the downstream
    # gate's hash recompute consistent) without adding entropy.
    artifact["review_flags"] = compute_review_flags(artifact)
    artifact["qa_audit"] = build_qa_audit(
        artifact, environment_audit=environment_audit,
    )
    artifact["artifact_hash"] = _stable_hash({
        key: value for key, value in artifact.items() if key != "artifact_hash"
    })
    return artifact


def _level_fill_key(action: Mapping) -> str | None:
    if action.get("mode") != "level":
        return None
    fill = action.get("fill")
    if not isinstance(fill, str) or not fill.strip():
        return None
    return canon(fill)


def _selected_level_fill_keys(
    decisions: Sequence[Mapping],
    action_vector: Mapping[str, str],
    *,
    exclude_decision_id: str | None = None,
) -> set[str]:
    keys: set[str] = set()
    for decision in decisions:
        decision_id = str(decision["decision_id"])
        if decision_id == exclude_decision_id:
            continue
        selected_id = action_vector.get(decision_id)
        if selected_id is None:
            continue
        selected = next(
            (
                action for action in decision.get("actions", [])
                if str(action.get("action_id")) == str(selected_id)
            ),
            None,
        )
        if selected is None:
            continue
        fill_key = _level_fill_key(selected)
        if fill_key is not None:
            keys.add(fill_key)
    return keys


def _hiding_action_id(
    decision: Mapping,
    *,
    excluded_level_fill_keys: Collection[str] = (),
) -> str | None:
    """Action that hides a decision's surface for a leak-check render: the coarsest legal
    generalization level, else a placeholder. None if neither exists (the decision stays
    KEEP). Used to generalize a co-referent decision so its surface stops leaking a
    related relation's protected identity."""
    actions = [action for action in decision.get("actions", []) if action.get("legal", True)]
    levels = [
        action for action in actions
        if action.get("mode") not in {"keep", "placeholder"}
        and not action.get("source_identity")
        and isinstance(action.get("coarseness_rank"), Real)
        and not isinstance(action.get("coarseness_rank"), bool)
    ]
    excluded = set(excluded_level_fill_keys)
    available = [
        action for action in levels
        if _level_fill_key(action) not in excluded
    ]
    if available:
        return str(max(available, key=lambda action: float(action["coarseness_rank"]))["action_id"])
    placeholder = next((action for action in actions if action.get("mode") == "placeholder"), None)
    return str(placeholder["action_id"]) if placeholder else None


def build_joint_representative_anchor(
    assertion: Mapping,
    decisions: Sequence[Mapping],
) -> dict:
    """Choose one joint vector: linked coarsest entailing levels, unrelated KEEP."""
    requirements = dict(assertion.get("decision_requirements") or {})
    action_vector: dict[str, str] = {}
    used_level_fill_keys: set[str] = set()
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
                and _level_fill_key(action) not in used_level_fill_keys
            ]
            if not candidates:
                raise ValueError(
                    f"no legal generalization for decision {decision_id} entails "
                    f"{property_level} without a duplicate level fill"
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
            fill_key = _level_fill_key(selected)
            if fill_key is not None:
                used_level_fill_keys.add(fill_key)
        else:
            if decision.get("ranker_selectable") is False:
                forced = next(
                    (action for action in actions
                     if action.get("mode") == "placeholder"
                     and action.get("forced_placeholder")),
                    None,
                )
                if forced is None:
                    raise ValueError(
                        f"non-selectable decision {decision_id} has no forced placeholder"
                    )
                action_vector[decision_id] = str(forced["action_id"])
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


def _diagnose_coarser_readable(
    candidate: Mapping,
    decisions: Sequence[Mapping],
    protected_terms: Sequence[str],
    *,
    doc_id: str,
    render_action_vector: Callable[[str, Mapping[str, str]], str],
    reader: Callable[[list[str], str], Sequence[str]],
    chain_by_decision: Mapping[str, Sequence[Mapping]],
    reader_threshold: float,
) -> dict | None:
    """Lattice-data diagnostic. A relation failed the 3-point gate while genuinely supported
    (original readable, representative not). Coarsen the LOCATOR argument (the linked level
    named in the question) one legal level at a time and re-read the representative render.
    If a coarser level in the same chain reads, the chosen fine level -- not the relation -- is
    the problem: usually a bad/mis-grounded entry in lattice_profiles.json. Read-only: returns
    the offending surface + unreadable level + first readable coarser level + chain, or None.
    Does NOT change the verdict; it only fires the `lattice_level_suspect` review flag."""
    question = str(candidate.get("question") or "")
    requirements = dict(candidate.get("decision_requirements") or {})
    if not question or not requirements:
        return None
    decisions_by_id = {str(d["decision_id"]): d for d in decisions}
    for decision_id, level in requirements.items():
        # the LOCATOR is the linked argument whose level is spelled in the question
        if not level or str(level).lower() not in question.lower():
            continue
        decision = decisions_by_id.get(str(decision_id))
        if decision is None:
            continue
        levels = _ordered_decision_levels(decision)  # finest -> coarsest
        if level not in levels:
            continue
        for coarser in levels[levels.index(level) + 1:]:
            probe = dict(candidate)
            probe["decision_requirements"] = {**requirements, decision_id: coarser}
            probe["question"] = re.sub(re.escape(str(level)), coarser, question, flags=re.IGNORECASE)
            try:
                probe_anchor = build_joint_representative_anchor(probe, decisions)
            except ValueError:
                continue
            probe_vector = dict(probe_anchor["action_vector"])
            for other in decisions:  # mirror the deployed co-referent leak guard
                other_id = str(other["decision_id"])
                surface = str(other.get("canonical_key") or "")
                if surface and any(
                    canon(surface) != canon(term) and _contains(surface, term)
                    for term in protected_terms
                ):
                    hiding = _hiding_action_id(
                        other,
                        excluded_level_fill_keys=_selected_level_fill_keys(
                            decisions,
                            probe_vector,
                            exclude_decision_id=other_id,
                        ),
                    )
                    if hiding is not None:
                        probe_vector[other_id] = hiding
            probe_context = render_action_vector(doc_id, probe_vector)
            answer = reader(
                [_permuted_reader_question(probe, 0)],
                _reader_excerpt(probe_context, candidate.get("evidence") or {}),
            )[0]
            if _context_answer_score(probe, answer, chain_by_decision) >= reader_threshold:
                return {
                    "surface": str(decision.get("canonical_key") or ""),
                    "runtime_type": str(decision.get("runtime_type") or ""),
                    "unreadable_level": str(level),
                    "readable_coarser_level": str(coarser),
                    "chain": list(levels),
                }
    return None


def _finer_level_readability(
    candidate: Mapping,
    decisions: Sequence[Mapping],
    protected_terms: Sequence[str],
    *,
    doc_id: str,
    render_action_vector: Callable[[str, Mapping[str, str]], str],
    reader: Callable[[list[str], str], Sequence[str]],
    chain_by_decision: Mapping[str, Sequence[Mapping]],
) -> dict[str, dict] | None:
    """Reward-band certification for a KEPT relation QA (opt-in; see the spec's level-pinning
    gaps). The three-point gate certifies readability only at the SUPPORTED answer level; at RL
    time the policy may rank the answer decision FINER, and scoring then relies on the reader
    echoing the finer rendered text and its chain aliases resolving. This check makes that
    assumption measured: re-render the representative with the ANSWER decision at each finer
    level (question unchanged -- the answer level never appears in it) and score the read against
    the row's frozen required_property, exactly the RL-time semantics. Returns {finer_level:
    {"score", "render"}} where render is "ok" (probe rendered and was read) or "no_joint_arm"
    (no arm renders this level jointly with the row's other pinned levels -- a level-fill
    collision, a RENDER limitation rather than a bad level). Never changes the verdict by
    itself -- an unreadable finer level is lattice-owned data, not grounds to reject a QA that
    is valid at its supported level."""
    target = candidate.get("answer_target") or {}
    if target.get("kind") != "linked_decision":
        return None  # set members are pinned at their finest level: the band below is empty
    decision_id = str(target.get("decision_id"))
    supported = str(target.get("required_property") or "")
    decisions_by_id = {str(d["decision_id"]): d for d in decisions}
    decision = decisions_by_id.get(decision_id)
    if decision is None or not supported:
        return None
    levels = _ordered_decision_levels(decision)  # finest -> coarsest
    if supported not in levels:
        return None
    finer_levels = levels[:levels.index(supported)]
    if not finer_levels:
        return None
    requirements = dict(candidate.get("decision_requirements") or {})
    scores: dict[str, dict] = {}
    for finer in finer_levels:
        probe = dict(candidate)
        probe["decision_requirements"] = {**requirements, decision_id: finer}
        try:
            probe_anchor = build_joint_representative_anchor(probe, decisions)
        except ValueError:
            # no arm renders this level jointly with the row's other pinned levels: a level-fill
            # collision (render limitation), not evidence the level itself is bad
            scores[str(finer)] = {"score": 0.0, "render": "no_joint_arm"}
            continue
        probe_vector = dict(probe_anchor["action_vector"])
        for other in decisions:  # mirror the deployed co-referent leak guard
            other_id = str(other["decision_id"])
            surface = str(other.get("canonical_key") or "")
            if surface and any(
                canon(surface) != canon(term) and _contains(surface, term)
                for term in protected_terms
            ):
                hiding = _hiding_action_id(
                    other,
                    excluded_level_fill_keys=_selected_level_fill_keys(
                        decisions,
                        probe_vector,
                        exclude_decision_id=other_id,
                    ),
                )
                if hiding is not None:
                    probe_vector[other_id] = hiding
        probe_context = render_action_vector(doc_id, probe_vector)
        answer = reader(
            [_permuted_reader_question(candidate, 0)],
            _reader_excerpt(probe_context, candidate.get("evidence") or {}),
        )[0]
        # scored against the ORIGINAL row (frozen required_property): the finer node must
        # resolve via its aliases and entail the supported level, as it must at RL time
        scores[str(finer)] = {
            "score": _context_answer_score(candidate, answer, chain_by_decision),
            "render": "ok",
        }
    return scores


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
    answer_tokens = _stemmed_tokens(answer)
    if not answer_tokens:
        return None
    matches = [
        node for node in chain
        if any(
            (alias_tokens := _stemmed_tokens(str(alias)))
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


_TURN_EXCERPT_ELISION = "[...]"


def _line_clause_spans(line: str) -> list[tuple[int, int]]:
    """Return delimiter-terminated clause spans relative to one rendered turn."""
    spans, left = [], 0
    for delimiter in _CLAUSE_DELIMITER_PATTERN.finditer(line):
        right = delimiter.end()
        if line[left:right].strip():
            spans.append((left, right))
        left = right
    if line[left:].strip():
        spans.append((left, len(line)))
    return spans


def _source_reader_clause_refs(
    source: str, char_ranges: Sequence[tuple[int, int]],
) -> list[dict[str, int]]:
    """Pin source argument clauses by turn-local ordinal for a long speaker turn.

    Absolute source offsets cannot be reused after a generalized render changes span
    lengths. The ordinal is safe only when the rendered turn has the same delimiter
    shape; `_turn_excerpt` otherwise falls back to that full turn.
    """
    if not char_ranges:
        return []
    lines = source.splitlines(keepends=True)
    line_starts, position = [], 0
    for line in lines:
        line_starts.append(position)
        position += len(line)
    refs: set[tuple[int, int, int]] = set()
    for start, end in char_ranges:
        if not (0 <= start < end <= len(source)):
            return []
        for turn, line_start in enumerate(line_starts):
            line_end = line_start + len(lines[turn])
            if end <= line_start:
                break
            if start >= line_end:
                continue
            spans = _line_clause_spans(lines[turn])
            for clause, (left, right) in enumerate(spans):
                absolute_left, absolute_right = line_start + left, line_start + right
                if absolute_left < end and start < absolute_right:
                    refs.add((turn, clause, len(spans)))
    return [
        {"turn": turn, "clause": clause, "turn_clause_count": count}
        for turn, clause, count in sorted(refs)
    ]


def _turn_excerpt(
    context: str,
    core_turns: Sequence[int],
    *,
    window: int,
    core_clauses: Sequence[Mapping[str, object]] = (),
) -> str:
    """Surgically stitch `context` to the given turn indices (each plus `window`
    neighbor turns), eliding non-adjacent gaps rather than spanning first-to-last.

    Adjacent/overlapping turn windows merge into one region (so a co-located
    relation reads exactly as before); a relation whose turns sit far apart reads
    only its relevant regions joined by an elision marker -- the middle text is
    not handed to the reader. Optional turn-local clause pins narrow a long
    speaker turn further; a changed delimiter shape falls back to that full turn.
    Empty `core_turns`, or indices past the end of `context`, fall back to the
    full `context` so a diverged render is never mis-sliced."""
    if not core_turns:
        return context
    lines = context.splitlines()
    if max(core_turns) >= len(lines):
        return context
    pinned_clauses: dict[int, set[int]] = defaultdict(set)
    pinned_counts: dict[int, int] = {}
    for ref in core_clauses:
        try:
            turn = int(ref["turn"])
            clause = int(ref["clause"])
            count = int(ref["turn_clause_count"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= turn < len(lines) and 0 <= clause < count):
            continue
        if turn in pinned_counts and pinned_counts[turn] != count:
            pinned_clauses.pop(turn, None)
            pinned_counts.pop(turn, None)
            continue
        pinned_counts[turn] = count
        pinned_clauses[turn].add(clause)

    def excerpt_line(turn: int) -> str:
        clauses = pinned_clauses.get(turn)
        if not clauses:
            return lines[turn]
        spans = _line_clause_spans(lines[turn])
        if len(spans) != pinned_counts.get(turn) or max(clauses) >= len(spans):
            return lines[turn]
        parts: list[str] = []
        previous = None
        for clause in sorted(clauses):
            if previous is not None and clause > previous + 1:
                parts.append(_TURN_EXCERPT_ELISION)
            left, right = spans[clause]
            parts.append(lines[turn][left:right].strip())
            previous = clause
        return "\n".join(part for part in parts if part)

    windows = sorted(
        (max(0, turn - window), min(len(lines) - 1, turn + window))
        for turn in set(core_turns)
    )
    merged: list[list[int]] = []
    for low, high in windows:
        if merged and low <= merged[-1][1] + 1:  # touching/overlapping -> one region, no gap
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    parts: list[str] = []
    for index, (low, high) in enumerate(merged):
        if index:
            parts.append(_TURN_EXCERPT_ELISION)  # a real gap was skipped between regions
        parts.extend(excerpt_line(turn) for turn in range(low, high + 1))
    return "\n".join(parts)


def _gate_debug(exc, orig_ans, rep_ans, ph_ans, scores) -> None:
    """Dev-only dump of what the reader saw + answered per render, for gate debugging. OFF unless
    $CLOAK_GATE_DEBUG_DIR is set; filtered to reverse-orientation questions to keep output tiny."""
    out_dir = os.getenv("CLOAK_GATE_DEBUG_DIR")
    if not out_dir or not exc:
        return
    question, ox, rx, px = exc
    if not question.startswith("For what medical condition was the"):
        return
    os.makedirs(out_dir, exist_ok=True)
    key = _stable_hash(question + ox).split(":")[-1][:16]
    with open(os.path.join(out_dir, key + ".json"), "w") as f:
        json.dump({"question": question, "scores": scores,
                   "original": {"excerpt": ox, "answer": list(orig_ans)},
                   "representative": {"excerpt": rx, "answer": list(rep_ans)},
                   "placeholder": {"excerpt": px, "answer": list(ph_ans)}},
                  f, ensure_ascii=False, indent=1)


def _reader_excerpt(context: str, evidence: Mapping) -> str:
    """Render an assertion's pinned reader evidence, preserving old artifacts."""
    turns = evidence.get("reader_turns") or []
    clauses = evidence.get("reader_clauses") or []
    if clauses:
        return _turn_excerpt(
            context,
            turns,
            window=CONTEXT_READER_TURN_WINDOW,
            core_clauses=clauses,
        )
    return _turn_excerpt(context, turns, window=CONTEXT_READER_TURN_WINDOW)


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
    if target.get("kind") == "linked_decision_set":
        # Set-valued QA (validated: scripts/spikes/set_valued_gate_probe.py): the reader answers
        # with a JSON array; the score is one-to-one per-member recall (each prediction matches at
        # most one member via the linked scorer). At the production threshold 1.0 the three-point
        # gate then requires EVERY member readable on orig and rep, and at least one member hidden
        # on placeholder. Extra predictions (e.g. uncontrolled literals) are ignored: literals are
        # pre-excluded from privacy scoring by construction.
        predictions = _parse_llm_json_array(answer)
        members = list(target.get("members") or [])
        if not members:
            return 0.0
        used = [False] * len(predictions)
        hits = 0
        for member in members:
            chain = chain_by_decision.get(str(member.get("decision_id")))
            if not chain:
                continue
            for index, prediction in enumerate(predictions):
                if used[index]:
                    continue
                if _linked_answer_score(
                        prediction, chain, str(member.get("required_property", ""))) >= 1.0:
                    used[index] = True
                    hits += 1
                    break
        return hits / len(members)
    if target.get("kind") == "literal":
        return _answer_score(answer, list(target.get("expected_values") or []))
    return _answer_score(answer, list(row.get("accepted_values") or []))


def _permuted_reader_question(assertion: Mapping, permutation_index: int) -> str:
    question = str(assertion["question"])
    # NOTE: the relation answer_type prefix ("Extract the shortest <type> span ...") is
    # deliberately NOT applied -- it reintroduced a hardcoded type cue that could re-leak the
    # answer's type word after the stored question was repaired. The reader's base prompt already
    # asks for the shortest exact answer span.
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
    set_reader: Callable[[list[str], str], Sequence[str]] | None = None,
) -> tuple[list[dict], dict[str, dict]]:
    """Apply frozen repeated original/generalization/placeholder reader checks.

    `set_reader` answers linked_decision_set rows (JSON-array response contract); without it a
    set row falls back to the span reader, whose reply cannot parse as an array -> rejected."""
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
            debug_excerpts: dict[int, tuple] = {}
            for row in rows:
                question = _permuted_reader_question(row, permutation_index)
                reader_evidence = row.get("evidence") or {}
                ox = _reader_excerpt(original_context, reader_evidence)
                rx = _reader_excerpt(representative_context, reader_evidence)
                px = _reader_excerpt(placeholder_context, reader_evidence)
                debug_excerpts[id(row)] = (question, ox, rx, px)
                row_reader = (
                    set_reader
                    if set_reader is not None
                    and (row.get("answer_target") or {}).get("kind") == "linked_decision_set"
                    else reader
                )
                original_answers += row_reader([question], ox)
                representative_answers += row_reader([question], rx)
                placeholder_answers += row_reader([question], px)
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
                _gate_debug(debug_excerpts.get(id(row)), original, representative, placeholder, scores)
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
    set_reader: Callable[[list[str], str], Sequence[str]] | None = None,
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
        reader_evidence = row.get("evidence") or {}
        row_reader = (
            set_reader
            if set_reader is not None
            and (row.get("answer_target") or {}).get("kind") == "linked_decision_set"
            else reader
        )
        context_answers += row_reader(
            [question], _reader_excerpt(doc_p, reader_evidence)
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
