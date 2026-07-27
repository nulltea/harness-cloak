"""Relation-teacher transport, prompting, and teacher-reply grounding.

Layered below ``qa_builder``, so it also owns the relation-class tables and the
source-span primitives the prompt builders and grounding checks need."""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy

from cloak.train.qa_scoring import CLAUSE_DELIMITER_PATTERN, canon, occurrence_protected_terms

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


RUNTIME_TYPE_CLASSES = {
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


RELATION_ARGUMENT_CLASSES = {
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


# Closed procedure-form lexicon: detector-typed conditions whose surface names
# a performed procedure ("kidney transplant" in past medical history) may also
# fill procedure slots; ordinary condition surfaces may not.
_PROCEDURE_FORM_PATTERN = re.compile(
    r"\b(?:transplant\w*|surgery|surgeries|\w+ectomy|\w+plasty|bypass|graft\w*"
    r"|replacement|repair)\b",
    re.IGNORECASE,
)


def argument_relation_classes(runtime_type: str, surface: str) -> set[str]:
    base = RUNTIME_TYPE_CLASSES.get(canon(str(runtime_type)))
    classes = {base} if base else set()
    if base == "condition" and _PROCEDURE_FORM_PATTERN.search(str(surface)):
        classes.add("procedure")
    return classes


ACI_SPEAKER_TURN_PATTERN = re.compile(r"\[(?:doctor|patient)\]", re.IGNORECASE)


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
        relation_class = RUNTIME_TYPE_CLASSES.get(canon(str(occurrence.get("runtime_type", ""))))
        if relation_class is None:
            continue
        if not any(relation_class in classes for contract in RELATION_ARGUMENT_CLASSES.values()
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


def source_clause_spans(document: str) -> list[tuple[int, int]]:
    """Return source clauses without treating speaker turns as relation bridges."""
    markers = list(ACI_SPEAKER_TURN_PATTERN.finditer(document))
    if len(markers) >= 2:
        return [
            (marker.start(), markers[index + 1].start() if index + 1 < len(markers) else len(document))
            for index, marker in enumerate(markers)
            if document[marker.start():markers[index + 1].start() if index + 1 < len(markers) else len(document)].strip()
        ]
    spans, left = [], 0
    for delimiter in CLAUSE_DELIMITER_PATTERN.finditer(document):
        right = delimiter.end()
        if document[left:right].strip():
            spans.append((left, right))
        left = right
    if document[left:].strip():
        spans.append((left, len(document)))
    return spans


def _prompt_relation_class(relation_class: str) -> str:
    return {"treatment": "drug", "monitoring": "test"}.get(relation_class, relation_class)


def _prompt_display_classes(row: Mapping) -> str:
    """Teacher-facing class label; procedure-form conditions show both roles."""
    classes = argument_relation_classes(str(row["runtime_type"]), str(row["surface"]))
    ordered = [cls for cls in ("condition", "symptom", "treatment", "procedure",
                               "monitoring", "provider", "status", "category")
               if cls in classes] or [str(row["relation_class"])]
    return "/".join(_prompt_relation_class(cls) for cls in ordered)


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


# ASCII hyphen plus the unicode hyphen/dash/minus variants LLM teachers emit (e.g. gpt-oss
# writes "x‑ray" U+2011 for source ASCII "x-ray"). Equated in literal AND evidence-quote
# grounding. Dash folding is 1-char->1-char, so it preserves string length and offsets.
_LITERAL_DASH_CHARS = "-‐‑‒–—−"


_DASH_FOLD = str.maketrans({ch: "-" for ch in _LITERAL_DASH_CHARS})


def exact_substring_starts(document: str, quote: str) -> list[int]:
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


def source_literal_spans(document: str, literal: str) -> list[tuple[int, int]]:
    """Source `(start, end)` spans matching `literal` under case-fold + whitespace-collapse +
    dash-variant equivalence at token boundaries. Offsets index the ORIGINAL document, so the
    matched text is exactly ``document[start:end]`` -- a context literal resolved through this and
    then written back as its source substring still satisfies the exact-source grounding contract
    (`argument_is_grounded`). Only orthographic casing, inter-token whitespace, and unicode dash
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
    for index, (start, end) in enumerate(source_clause_spans(document), start=1):
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

    clauses = source_clause_spans(document)

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
                    ranges.extend(source_literal_spans(document, str(argument["literal"])))
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


def teacher_relation_arguments(
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


def normalize_teacher_text(value: str) -> str:
    """Teacher answers/questions are inconsistently snake-cased
    ("solid_organ_transplant"); the lexical fact scorer tokenizes an underscore
    run as one token instead of its words, so map underscores to spaces and
    collapse whitespace before storing and scoring."""
    return re.sub(r"\s+", " ", str(value).replace("_", " ")).strip()


def substitute_linked_surfaces(
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
        for term in occurrence_protected_terms(occurrences[argument["occurrence_id"]]):
            substituted = re.sub(
                rf"(?<!\w){re.escape(term)}(?!\w)", level, substituted,
                flags=re.IGNORECASE,
            )
    return substituted


def relation_arguments_are_legal(
    relation: str, arguments: Sequence[Mapping], relation_contract: Mapping[str, Mapping],
) -> bool:
    allowed = relation_contract[relation]["argument_classes"]
    return len(arguments) == len(allowed) and all(
        argument_relation_classes(
            str(argument.get("runtime_type", "")),
            str(argument.get("surface", argument.get("literal", ""))),
        ) & set(permitted)
        for argument, permitted in zip(arguments, allowed)
    )


def argument_is_grounded(
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
