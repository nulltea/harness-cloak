"""QA-builder v2 relation compilation, support anchors, weighting, and packaging."""
from __future__ import annotations

import bisect
import inspect
import re
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from copy import deepcopy
from math import ceil, isfinite
from numbers import Real

from cloak.qa.audit import build_qa_audit
from cloak.qa.scoring import (
    ACI_REQUIRED_SECTIONS,
    CLAUSE_DELIMITER_PATTERN,
    canon,
    context_answer_score,
    gate_debug,
    line_clause_spans,
    meaningful_tokens,
    occurrence_protected_terms,
    parse_aci_note,
    parse_llm_json_array,
    permuted_reader_question,
    reader_excerpt,
    stable_hash,
    validated_build_reader_pin,
)
from cloak.qa.teacher import (
    ACI_SPEAKER_TURN_PATTERN,
    OpenRouterRelationTeacher,
    RELATION_ARGUMENT_CLASSES,
    RELATION_REPAIR_MAX_TARGETS_PER_CALL,
    RELATION_TEACHER_MAX_RELATIONS,
    RUNTIME_TYPE_CLASSES,
    RelationTeacherProposals,
    RelationTeacherResponseError,
    argument_is_grounded,
    argument_relation_classes,
    exact_substring_starts,
    normalize_teacher_text,
    relation_arguments_are_legal,
    relation_repair_prompt,
    relation_teacher_prompt,
    relation_teacher_response_format,
    relation_teacher_span_inventory,
    source_clause_spans,
    source_literal_spans,
    substitute_linked_surfaces,
    teacher_relation_arguments,
)

RELATION_ONTOLOGY = (
    "prescribed_with",
    "procedure_for",
    "tests_for",
    "contraindicated_because_of",
    "causes_or_explains",
)


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
    classes = RELATION_ARGUMENT_CLASSES.get(relation)
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
        "argument_classes": RELATION_ARGUMENT_CLASSES["prescribed_with"],
        "definition": "a drug prescribed or used for a condition or diagnosis",
        "cues": ("prescribe", "prescribed", "initiate", "continue", "uses", "taking", "takes", "treated with"),
        "connector_patterns": (
            r"\s+(?:(?:is|was)\s+)?(?:prescribed|used)\s+(?:with|for)\s+",
            r"\s+(?:(?:is|was)\s+)?treated\s+with\s+",
        ),
    },
    "procedure_for": {
        "argument_classes": RELATION_ARGUMENT_CLASSES["procedure_for"],
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
        "argument_classes": RELATION_ARGUMENT_CLASSES["tests_for"],
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
        "argument_classes": RELATION_ARGUMENT_CLASSES["contraindicated_because_of"],
        # ACI dialogue states contraindication as "you ca n't take X because
        # of Y" (transcript-tokenized negation), not with the clinical word.
        "cues": ("contraindicated", "ca n't take", "can't take", "cannot take"),
        "connector_patterns": (
            r"\s+(?:(?:is|was|are|were|has been|had been|is being|was being)\s+)?"
            r"contraindicated\s+because\s+of\s+",
        ),
    },
    "causes_or_explains": {
        "argument_classes": RELATION_ARGUMENT_CLASSES["causes_or_explains"],
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


_PLAN_SECTION_HEADING_PATTERN = re.compile(
    r"(?m)^(?P<title>[A-Za-z][A-Za-z0-9 /-]{0,96})\.\s*$"
)


_RELATION_WINDOW_CUE_PATTERN = re.compile(
    r"\b(?:prescrib\w*|continue|taking|takes|treated|refer\w*|order\w*|"
    r"monitor\w*|contraindicat\w*|causes?|explains?|status|category)\b",
    re.IGNORECASE,
)


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


# Relation-constrained reader clause (qa-context-reader-v4). One prompt line restates the QA as
# the relation the ANSWER must satisfy, naming the LOCATOR argument(s) — every NON-answer
# argument — by the exact string the representative render substitutes. This is the only prompt
# lever that stopped the reader grabbing lexically-resonant distractor spans; the 56-row
# regression audit showed no clean regression
# (docs/handoffs/2026-07-21-relation-constrained-reader-prompt.md).
# Keyed by (relation, answer_role), NEVER by "the subject is the condition": for
# contraindicated_because_of the subject is the treatment — the mirror image of the other four
# relations — so the forward answer is a condition and the reverse answer is a treatment.
_RELATION_READER_CLAUSES = {
    ("prescribed_with", "object"):
        "is a medication prescribed or used to treat {loc}",
    ("prescribed_with", "subject"):
        "is the medical condition that {loc} was prescribed or used to treat",
    ("procedure_for", "object"):
        "is a procedure or treatment performed for {loc}",
    ("procedure_for", "subject"):
        "is the medical condition that {loc} was performed or planned for",
    ("tests_for", "object"):
        "is a test or exam ordered to evaluate or monitor {loc}",
    ("tests_for", "subject"):
        "is the medical condition that {loc} was ordered to evaluate or monitor",
    # contraindicated_because_of clauses force POLARITY, not just type+locality: the corpus's
    # dominant (treatment, condition) co-occurrence asserts the OPPOSITE relation (the drug
    # treats the condition), and an unconstrained extractive reader happily returns the
    # prescribed drug (the 5-doc inverted-contraindication audit). The explicit NOT-prescribed
    # contrast + per-clause NONE reminder push the reader to decline treatment sentences.
    ("contraindicated_because_of", "object"):
        "is a medical condition stated to make {loc} unsuitable, avoided, or stopped — NOT a "
        "condition that {loc} treats; if the DOCUMENT states no such contraindication, reply NONE",
    ("contraindicated_because_of", "subject"):
        "is a medication or procedure the DOCUMENT says is avoided, stopped, or must not be used "
        "because of {loc} — NOT one prescribed or given to treat {loc}; if the DOCUMENT states no "
        "such contraindication, reply NONE",
    ("causes_or_explains", "object"):
        "is a condition, symptom, or finding caused or explained by {loc}",
    ("causes_or_explains", "subject"):
        "is the underlying condition that causes or explains {loc}",
}


# >= 2 locators (compound span-locator reverse, multi-literal reverse): the answer is always the
# subject, so only subject rows exist. Plural verb + "all" pins uniqueness exactly as the
# compound question text does.
_RELATION_READER_CLAUSES_COMPOUND = {
    ("prescribed_with", "subject"):
        "is the single medical condition that {loc} were all prescribed or used to treat",
    ("procedure_for", "subject"):
        "is the single medical condition that {loc} were all performed or planned for",
    ("tests_for", "subject"):
        "is the single medical condition that {loc} were all ordered to evaluate or monitor",
    ("contraindicated_because_of", "subject"):
        "is a medication or procedure the DOCUMENT says is avoided, stopped, or must not be used "
        "because of all of {loc} — NOT one prescribed or given to treat them; if the DOCUMENT "
        "states no such contraindication, reply NONE",
    ("causes_or_explains", "subject"):
        "is the single underlying condition that causes or explains all of {loc}",
}


def _rendered_level_fill(decision: Mapping, level: str) -> str:
    """The exact string the render substitutes for `decision` at `level` (the level action's
    fill) — not the level/support_property text, which can differ from what appears in the
    rendered document."""
    for action in decision.get("actions", []) or []:
        if (
            action.get("mode") == "level"
            and action.get("entails")
            and canon(str(action["entails"][0])) == canon(str(level))
        ):
            return str(action.get("fill") or action["entails"][0])
    return str(level)


def _typeset_locator_phrase(
    argument: Mapping,
    occurrences: Mapping[str, Mapping],
    decisions_by_id: Mapping[str, Mapping],
    requirements: Mapping[str, str],
) -> str | None:
    """A locator argument as a clause-ready noun phrase. Linked span: "the <rendered fill>"
    (leading article stripped before prefixing). Context literal: the verbatim text in double
    quotes, article-free — quoting keeps a junk literal (e.g. a whole verb phrase) a
    syntactically inert token instead of destroying the clause grammar."""
    if argument.get("kind") == "context":
        literal = str(argument.get("literal") or "").strip()
        return f'"{literal}"' if literal else None
    occurrence = occurrences.get(str(argument.get("occurrence_id")))
    if occurrence is None or occurrence.get("decision_id") is None:
        return None
    decision_id = str(occurrence["decision_id"])
    decision = decisions_by_id.get(decision_id)
    level = requirements.get(decision_id)
    if decision is None or not level:
        return None
    fill = _rendered_level_fill(decision, str(level)).strip()
    if not fill:
        return None
    return "the " + re.sub(r"^(?:a|an|the)\s+", "", fill, flags=re.IGNORECASE)


def _join_locator_phrases(phrases: Sequence[str]) -> str:
    if len(phrases) <= 2:
        return " and ".join(phrases)
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def _relation_reader_clause(
    candidate: Mapping,
    occurrences: Mapping[str, Mapping],
    decisions: Sequence[Mapping],
) -> str | None:
    """Frozen per-assertion relation-constraint clause for the v4 context reader.

    The locators are the complement of the answer argument set (orientation-aware by
    construction: forward names the subject, reverse the object(s), set-valued forward the
    subject), so the answer argument can never be named. Locator strings are pinned-level
    rendered fills — the same strings the representative render substitutes — frozen here once
    and reused verbatim at every gate read, both lattice probes, and runtime scoring, so gate
    and runtime certify the same instrument. Returns None (no constraint line) when the row is
    not a relation QA, the answer is uncontrolled, or a locator cannot be typeset — graceful
    degradation, never an exception."""
    relation = str(candidate.get("relation") or "")
    target = candidate.get("answer_target") or {}
    arguments = list((candidate.get("evidence") or {}).get("arguments") or [])
    requirements = {
        str(key): str(value)
        for key, value in dict(candidate.get("decision_requirements") or {}).items()
    }
    if target.get("kind") == "linked_decision":
        answer_decision_ids = {str(target.get("decision_id"))}
    elif target.get("kind") == "linked_decision_set":
        answer_decision_ids = {
            str(member.get("decision_id")) for member in target.get("members") or []
        }
    else:
        return None  # legacy literal answer target: uncontrolled answer, no relation constraint

    def argument_decision(argument: Mapping) -> str | None:
        occurrence = occurrences.get(str(argument.get("occurrence_id")))
        if occurrence is None or occurrence.get("decision_id") is None:
            return None
        return str(occurrence["decision_id"])

    answer_arguments = [
        argument for argument in arguments
        if argument.get("kind") == "linked"
        and argument_decision(argument) in answer_decision_ids
    ]
    answer_argument_ids = {id(argument) for argument in answer_arguments}
    locator_arguments = [
        argument for argument in arguments if id(argument) not in answer_argument_ids
    ]
    if not answer_arguments or not locator_arguments:
        return None
    answer_roles = {str(argument.get("role") or "") for argument in answer_arguments}
    if len(answer_roles) != 1:
        return None
    answer_role = next(iter(answer_roles))
    if any(str(argument.get("role") or "") == answer_role for argument in locator_arguments):
        return None  # a locator sharing the answer's role cannot be named unambiguously
    decisions_by_id = {str(decision["decision_id"]): decision for decision in decisions}
    phrases = [
        _typeset_locator_phrase(argument, occurrences, decisions_by_id, requirements)
        for argument in locator_arguments
    ]
    if any(phrase is None for phrase in phrases):
        return None
    table = (
        _RELATION_READER_CLAUSES_COMPOUND if len(phrases) > 1 else _RELATION_READER_CLAUSES
    )
    template = table.get((relation, answer_role))
    if template is None:
        return None
    return template.format(loc=_join_locator_phrases(phrases))


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
            protected_terms.extend(occurrence_protected_terms(occurrence))
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
                            stable_hash({
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
                        "property_hash": stable_hash(property_level),
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
                            "property_hash": stable_hash(property_level),
                            "supporting_action_hashes": [
                                stable_hash(action_id)
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
                    "property_hash": stable_hash(property_level),
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
                            "property_hash": stable_hash(property_level),
                            "supporting_action_hashes": [
                                stable_hash(action_id)
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
                            "property_hash": stable_hash(property_level),
                            "supporting_action_hashes": [
                                stable_hash(action_id)
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
                            "property_hash": stable_hash(property_level),
                            "supporting_action_hashes": [
                                stable_hash(action_id)
                                for action_id in supporting_actions[property_level]
                            ],
                            "locator_hash": stable_hash(locator),
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
                        f"{stable_hash(property_level)}"
                    ),
                    "question": question,
                    "accepted_values": [property_level],
                    "decision_requirements": {decision_id: property_level},
                    "evidence": {
                        "authority": "frozen_action_entails",
                        "template": "aci-context-category-v1",
                        "runtime_type": runtime_type,
                        "role_cue": role_cue,
                        "locator_hash": stable_hash(locator),
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
        parsed_reference = parse_aci_note(reference)
        parsed_source = parse_aci_note(document)
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
                        "reference_hash": stable_hash(reference),
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
                    "surface_hash": stable_hash(str(occurrence.get("surface", ""))),
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
                        "source_hash": stable_hash(document),
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
                    "reference_hash": stable_hash(reference),
                    "reference_span": parsed_reference["demographic_evidence"][name],
                }
            elif name in parsed_source["demographic"]:
                value = parsed_source["demographic"][name]
                evidence = {
                    "authority": "source_document_task_schema_fallback",
                    "provenance": "exact_doc_orig",
                    "source_span": parsed_source["demographic_evidence"][name],
                    "source_hash": stable_hash(document),
                    "value_hash": stable_hash(value),
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
                "context_candidate_id": "context:" + stable_hash({
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
    clauses = source_clause_spans(document)
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


def relation_evidence_windows(document: str, environment_document: Mapping) -> list[dict]:
    """Finite one/two-clause source spans that can ground a typed relation.

    These windows do not assert a relation.  They only prevent a teacher from
    pairing a source quote with a duplicate occurrence elsewhere in the document.
    """
    speaker_markers = list(ACI_SPEAKER_TURN_PATTERN.finditer(document))
    if len(speaker_markers) >= 2:
        clause_spans = [
            (marker.start(), speaker_markers[index + 1].start()
             if index + 1 < len(speaker_markers) else len(document))
            for index, marker in enumerate(speaker_markers)
        ]
        window_widths = (1,)
    else:
        clause_spans, start = [], 0
        for delimiter in CLAUSE_DELIMITER_PATTERN.finditer(document):
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
                RUNTIME_TYPE_CLASSES.get(canon(first["runtime_type"])) in subject_types
                and RUNTIME_TYPE_CLASSES.get(canon(second["runtime_type"])) in object_types
                for first in contained for second in contained if first is not second
                for subject_types, object_types in RELATION_ARGUMENT_CLASSES.values()
            ):
                continue
            eligible_pairs = []
            for relation, (subject_types, object_types) in RELATION_ARGUMENT_CLASSES.items():
                for subject in contained:
                    for obj in contained:
                        if subject is obj or not subject["linkable"] or not obj["linkable"]:
                            continue
                        if (
                            RUNTIME_TYPE_CLASSES.get(canon(subject["runtime_type"])) in subject_types
                            and RUNTIME_TYPE_CLASSES.get(canon(obj["runtime_type"])) in object_types
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
                "evidence_window_id": "window:" + stable_hash({"start": left, "end": right}),
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
        for term in occurrence_protected_terms(occurrence)
    ]
    for row in records:
        row["reason"] = _sanitized_ledger_reason(str(row["reason"]), protected_terms)
    return [by_id[label] for label in sorted(expected, key=lambda value: int(value[1:]))]


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
    attempt_hash = stable_hash(attempt_value)
    identity = {
        "reason": reason,
        "detail_reason": detail_reason,
        "attempt_hash": attempt_hash,
    }
    return {
        "status": "rejected",
        "reason": reason,
        "detail_reason": detail_reason,
        "rejection_id": stable_hash(identity),
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


def _placeholder_meaning_tokens(runtime_type: str) -> set[str]:
    contract = AciTaskAdapter.semantic_type_contract.get(_semantic_label(runtime_type), {})
    return {
        token
        for label in contract.get("placeholder_labels", ())
        for token in meaningful_tokens(str(label))
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
    answer_tokens = meaningful_tokens(answer) - exempt
    return answer_tokens & meaningful_tokens(question)


def _question_leaks_answer(
    question: str,
    answer: str,
    runtime_type: str | Sequence[str],
    *,
    extra_exempt_tokens: Sequence[str] = (),
) -> bool:
    return bool(_answer_leak_tokens(
        question, answer, runtime_type, extra_exempt_tokens=extra_exempt_tokens))


def ordered_decision_levels(decision: Mapping) -> list[str]:
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
        answer_tokens |= meaningful_tokens(value)
    if not answer_tokens:
        return None

    def leaks(text: str) -> bool:  # strict: no generic-word exemption
        return bool(answer_tokens & meaningful_tokens(text))

    def strip_outside(text: str, protect: tuple[int, int] | None) -> str:
        # remove every answer-token occurrence not inside the protected [start, end) span
        out, cursor = [], 0
        for match in re.finditer(r"\w+", text):
            token = match.group(0)
            keep = True
            if meaningful_tokens(token) & answer_tokens:
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
        return ordered_decision_levels(decisions.get(str(occurrence.get("decision_id"))) or {})

    current = normalize_teacher_text(str(other_arg.get("support_property") or ""))
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
            and len(meaningful_tokens(repaired)) >= 3
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
    question_tokens = meaningful_tokens(question)
    for term in protected_terms:
        if not term:
            continue
        term_tokens = meaningful_tokens(term)
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
        token for term in protected_terms for token in meaningful_tokens(term)
    }
    for position, surface in sorted(set(targets)):
        sentence = _sentence_at(document, position)
        sentence_tokens = meaningful_tokens(sentence)
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
        RUNTIME_TYPE_CLASSES.get(canon(str(occurrences[occurrence_id].get("runtime_type", ""))))
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
                        "quote_hash": stable_hash(clause),
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


def _resolve_relation_evidence_span(
    document: str,
    quote: str,
    proposed_start,
) -> tuple[tuple[int, int] | None, str | None]:
    starts = exact_substring_starts(document, quote)
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
        if CLAUSE_DELIMITER_PATTERN.search(quote, local_start, local_end):
            return False
        previous_delimiters = list(
            CLAUSE_DELIMITER_PATTERN.finditer(quote, 0, local_start)
        )
        next_delimiter = CLAUSE_DELIMITER_PATTERN.search(quote, local_end)
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
    clauses = source_clause_spans(document)

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
            matches = source_literal_spans(document, literal)
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
        # source_clause_spans (here) and the support-check's period regex otherwise
        # disagree on clause boundaries, failing valid same-turn relations (Case 3).
        kind = "speaker_turn" if len(ACI_SPEAKER_TURN_PATTERN.findall(document)) >= 2 else "clause"
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
# roles (subject/object per RELATION_ARGUMENT_CLASSES). Used ONLY as a union fallback to
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
    from cloak.lattice.core import nli_entails
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
        if not all(argument_is_grounded(a, document, span, occurrences) for a in args):
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


def relation_scope(arguments: Sequence[Mapping]) -> str:
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


def _argument_identity_atoms(
    argument: Mapping,
    occurrences: Mapping[str, Mapping],
    decisions_by_id: Mapping[str, Mapping],
) -> frozenset:
    """Normalized identity atoms for cross-relation pair matching: decision id plus canonical
    surface for a linked argument, canonical text for a literal — so a literal and a linked span
    of the same value unify (the inverted-contraindication audit needed surface+literal
    normalization to cover every case decision-id matching alone missed)."""
    atoms: set = set()
    if argument.get("kind") == "linked":
        occurrence = occurrences.get(str(argument.get("occurrence_id"))) or {}
        decision_id = occurrence.get("decision_id")
        if decision_id is not None:
            atoms.add(("decision", str(decision_id)))
            decision = decisions_by_id.get(str(decision_id)) or {}
            for value in (occurrence.get("surface"), decision.get("canonical_key")):
                normalized = canon(str(value)) if value else ""
                if normalized:
                    atoms.add(("value", normalized))
    elif argument.get("kind") == "context":
        literal = canon(str(argument.get("literal", "")))
        if literal:
            atoms.add(("value", literal))
    return frozenset(atoms)


_TREATING_RELATIONS = frozenset({"prescribed_with", "procedure_for"})


def _treating_conflict_filter(
    accepted: Sequence[Mapping],
    occurrences: Mapping[str, Mapping],
    decisions: Sequence[Mapping],
    *,
    doc_id: str,
) -> tuple[list[dict], list[dict]]:
    """Kept-assertion cross-gate for contraindicated_because_of.

    A kept contraindication row whose (treatment, condition) pair is also covered by a KEPT
    prescribed_with / procedure_for row asserts both "given for" and "avoided because of" about
    the same pair; the treating assertion is the corpus-grounded one (measured on the 5-doc v4
    audit: every inverted contraindication the accept-biased escalation admitted had its treating
    twin kept). Pair identity is unordered and value-normalized (_argument_identity_atoms), so the
    role flip between the relations (contraindication subject = the treatment) and literal-vs-span
    argument mismatches both match. Matches KEPT rows only, never the opportunity ledger — the
    ledger contains both directions by construction. Returns (kept, rejection_records)."""
    decisions_by_id = {str(decision["decision_id"]): decision for decision in decisions}

    def row_pairs(row: Mapping) -> list[tuple[frozenset, frozenset]]:
        arguments = list(dict(row.get("evidence") or {}).get("arguments") or [])
        subjects = [a for a in arguments if a.get("role") == "subject"]
        objects = [a for a in arguments if a.get("role") == "object"]
        if len(subjects) != 1 or not objects:
            return []
        subject_atoms = _argument_identity_atoms(subjects[0], occurrences, decisions_by_id)
        return [
            (subject_atoms, _argument_identity_atoms(obj, occurrences, decisions_by_id))
            for obj in objects
        ]

    treating_pairs = [
        pair
        for row in accepted
        if row.get("subtype") == "contextual_relation"
        and str(row.get("relation")) in _TREATING_RELATIONS
        for pair in row_pairs(row)
        if pair[0] and pair[1]
    ]

    def conflicts(pair: tuple[frozenset, frozenset]) -> bool:
        a, b = pair
        if not a or not b:
            return False
        return any(
            (a & c and b & d) or (a & d and b & c) for c, d in treating_pairs
        )

    kept: list[dict] = []
    rejections: list[dict] = []
    for row in accepted:
        if not (
            row.get("subtype") == "contextual_relation"
            and str(row.get("relation")) == "contraindicated_because_of"
            and any(conflicts(pair) for pair in row_pairs(row))
        ):
            kept.append(dict(row))
            continue
        row_evidence = dict(row.get("evidence") or {})
        evidence = {
            "source": "treating_conflict_gate",
            "arguments": _rejection_safe_arguments(list(row_evidence.get("arguments") or [])),
        }
        for authorship_key in ("teacher_id", "run_id"):
            if row_evidence.get(authorship_key):
                evidence[authorship_key] = row_evidence[authorship_key]
        record = _rejection_record(
            reason="invalid",
            detail_reason="treating_relation_conflict",
            attempt={
                "doc_id": doc_id,
                "source": "treating_conflict_gate",
                "subtype": row.get("subtype"),
                "occurrence_ids": list(row.get("occurrence_ids") or []),
            },
            evidence=evidence,
        )
        record["relation"] = row.get("relation")
        rejections.append(record)
    return kept, rejections


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
            and all(argument_is_grounded(argument, document, span, occurrences)
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
        "scope": relation_scope(arguments),
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
                argument_relation_classes(
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
            spans = source_literal_spans(document, str(argument["literal"]))
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
            for item in parse_llm_json_array(propose(prompt)):
                match = re.search(re.escape(item), document, re.IGNORECASE)
                if match is None:
                    continue
                start, end = match.span()
                literal = document[start:end]
                key = (runtime_type, start, end, literal)
                candidates.setdefault(key, {
                    "context_candidate_id": "context:" + stable_hash({
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
                if not relation_arguments_are_legal(relation, arguments, relation_contract):
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
                if not all(argument_is_grounded(argument, document, span, occurrences)
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
        clauses = source_clause_spans(document)
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
GLEANING_FIX_HINTS: dict[str, str] = {
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


GLEANING_FIXABLE_REASONS = frozenset(GLEANING_FIX_HINTS)


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
# can re-pair it. (Reason not in GLEANING_FIX_HINTS: it is detected structurally below, not by reason.)
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
    * "floor_answerable" -- readable EVERYWHERE including the all-placeholder floor (orig, rep,
      placeholder all at/above threshold): the floor cannot discriminate this fact, and
      re-authoring the question does not change what the placeholder render reveals -- never a
      repair/glean target. (Compile-time placeholder-answerable rejections carry no reader
      scores and keep their fixable path: a mispaired literal IS re-authorable.)
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
    if original and representative and placeholder:
        return "floor_answerable"
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
                             or row.get("proposal_hash") or stable_hash(dict(row))))

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
        if reason in GLEANING_FIXABLE_REASONS:
            _add(_fact_key(row) or _fallback_key(row), "fixable", relation=row.get("relation"),
                 reason=reason, hint=GLEANING_FIX_HINTS[reason], arguments=_args(row))

    # hedge modality: the hedge guard is a non-blocking diagnostic (not a reject) -- but a
    # hedge-flagged relation the READER then could not confirm is routed back to repair to
    # re-phrase the question conditionally. Restores the pre-demotion repair path, now gated on
    # actually failing the reader (relations the reader confirms are kept, not needlessly redrawn).
    for row in repairable_rejections:
        reason = str(row.get("detail_reason") or row.get("reason") or "")
        modality = (row.get("evidence") or {}).get("modality_diagnostics") or []
        if reason == "three_point_gate_failed" and "hedged_source_definite_question" in modality:
            _add(_fact_key(row) or _fallback_key(row), "fixable", relation=row.get("relation"),
                 reason="hedged_relation", hint=GLEANING_FIX_HINTS["hedged_relation"],
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
        literal_tokens = meaningful_tokens(str(argument["literal"]))
        if not literal_tokens:
            continue
        promoted = None
        for occurrence_id, occurrence in occurrences.items():
            if not occurrence.get("controlled", True):
                continue
            surface_tokens = meaningful_tokens(str(occurrence.get("surface", "")))
            if not surface_tokens or not surface_tokens <= literal_tokens:
                continue
            levels = ordered_decision_levels(decisions.get(str(occurrence.get("decision_id")), {}))
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
        "group_id": "literal_reverse:" + relation + ":" + stable_hash(
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
            "source_span": {"start": lo, "end": hi, "quote_hash": stable_hash(document[lo:hi])},
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
        levels = ordered_decision_levels(decision) if decision else []
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
    levels = ordered_decision_levels(decision)  # most-specific -> coarsest
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
        return bool(decision and ordered_decision_levels(decision))

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
        for term in occurrence_protected_terms(occurrence)
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
    locator_levels = ordered_decision_levels(decisions.get(locator_decision_id) or {})
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
    subject_levels = ordered_decision_levels(decisions.get(subject_decision_id) or {})
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
        levels = ordered_decision_levels(decisions.get(decision_id) or {})
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
        "group_id": "set_forward:" + relation + ":" + stable_hash(
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
            "source_span": {"start": lo, "end": hi, "quote_hash": stable_hash(document[lo:hi])},
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
        levels = ordered_decision_levels(decision or {})
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
        "group_id": "compound_span_reverse:" + relation + ":" + stable_hash(
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
            "source_span": {"start": lo, "end": hi, "quote_hash": stable_hash(document[lo:hi])},
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
        and (tokens := meaningful_tokens(str(occurrence.get("surface", ""))))
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
        proposal_hash = stable_hash(proposal)

        def reject(reason: str, detail: Mapping | None = None) -> None:
            quote, quote_span, _ = proposal_evidence(proposal)
            evidence = {
                "source": "relation_teacher",
                "proposal_index": index,
                "argument_occurrence_ids": [
                    (
                        str(value) if str(value) in occurrences
                        else stable_hash(str(value))
                    )
                    for value in proposal.get("argument_occurrence_ids", [])
                ],
                "evidence_quote_hash": stable_hash(quote),
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
                RUNTIME_TYPE_CLASSES.get(canon(str(occurrences[legacy_ids[1]].get("runtime_type", ""))))
                == "treatment"
            ):
                relation = "prescribed_with"
        if relation not in relation_contract:
            reject("invalid_relation")
            continue
        grounded_relation = relation
        arguments, argument_error = teacher_relation_arguments(
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
        if not relation_arguments_are_legal(relation, arguments, relation_contract):
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
                occurrence_tokens <= meaningful_tokens(str(argument["literal"]))
                for occurrence_tokens in substitutable_token_sets
            )
            for argument in arguments
        ):
            reject("literal_will_be_substituted")
            continue
        if not all(argument_is_grounded(argument, document, evidence_span, occurrences)
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
        question = normalize_teacher_text(str(proposal.get("question", "")))
        if not question.endswith("?"):
            reject("invalid_question")
            continue
        accepted_values = [normalize_teacher_text(str(value))
                           for value in proposal.get("accepted_answers", [])
                           if normalize_teacher_text(str(value))]
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
            sanitized_question = substitute_linked_surfaces(question, arguments, occurrences)
            sanitized_values = [
                substitute_linked_surfaces(answer, arguments, occurrences)
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
        leak_exempt = meaningful_tokens(answer_type_hint or "")
        # Reverse-orientation question (answer=subject): the locator (the object named in the
        # question, e.g. "thyroid panel") is the GIVEN premise, not the answer -- its tokens must
        # not count as answer-leakage, else the repair strips them ("thyroid panel" -> "panel").
        if answer_role == "subject":
            locator_argument = arguments[1]
            locator_text = str(locator_argument.get("support_property")
                               or locator_argument.get("literal") or "")
            leak_exempt = leak_exempt | meaningful_tokens(locator_text)
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
            for term in occurrence_protected_terms(occurrence):
                # A level identical to the raw surface is not a real generalization (a broken
                # lattice could entail the surface itself); drop it so a raw brand's ONLY
                # authorization can't be a surface-echoing level. A genuine coarser level that
                # merely shares a token (surface "diabetes" in level "diabetes mellitus") stays.
                term_level_tokens = frozenset(
                    token
                    for property_level in levels
                    if canon(property_level) != canon(term)
                    for token in meaningful_tokens(property_level)
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
            for token in meaningful_tokens(property_level)
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
                for token in meaningful_tokens(str(argument.get("literal") or ""))
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
            "quote_hash": stable_hash(document[evidence_span[0]:evidence_span[1]]),
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
            "group_id": "relation:" + relation + ":" + stable_hash([
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


def policy_routing(
    assertion: Mapping,
    occurrence_to_decision: Mapping[str, str | None],
    policy_decision_ids: Collection[str],
) -> tuple[list[str], str]:
    """Derive policy credit from occurrence links, excluding fixed decisions."""
    occurrence_ids = [str(value) for value in assertion.get("occurrence_ids", [])]
    missing = sorted(set(occurrence_ids) - set(occurrence_to_decision))
    if missing:
        raise ValueError(f"unknown occurrence links: {missing}")
    policy_ids = {str(value) for value in policy_decision_ids}
    dependencies = sorted({
        decision_id
        for occurrence_id in occurrence_ids
        if (decision_id := occurrence_to_decision[occurrence_id]) is not None
        and decision_id in policy_ids
    })
    return dependencies, "linked" if dependencies else "residual"


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
        policy_decision_ids = [
            str(row["decision_id"])
            for row in environment_document.get("decisions", [])
            if row.get("ranker_selectable") is True
        ]
        fixed_decision_ids = [
            str(row["decision_id"])
            for row in environment_document.get("decisions", [])
            if row.get("ranker_selectable") is not True
        ]
        occurrence_to_decision = {
            occurrence_id: (
                None if row.get("decision_id") is None else str(row["decision_id"])
            )
            for occurrence_id, row in occurrences.items()
        }
        policy_dependencies: set[str] = set()
        compiled = []
        compiled_ids = set()
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
            dependencies, credit_routing = policy_routing(
                {"occurrence_ids": occurrence_ids},
                occurrence_to_decision,
                policy_decision_ids,
            )
            row["policy_dependency_decision_ids"] = dependencies
            row["credit_routing"] = credit_routing
            policy_dependencies.update(dependencies)
            identity_payload = {
                key: value for key, value in row.items()
                if key not in {
                    "assertion_id", "weight", "status",
                    "policy_dependency_decision_ids", "credit_routing",
                }
            }
            assertion_id = str(row.get("assertion_id") or stable_hash(identity_payload))
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
                "environment_document_hash", stable_hash(environment_document)
            ),
            "measurement_state": (
                "unsupported" if not weighted else "partial" if missing_families else "measured"
            ),
            **weight_state,
            "assertion_ids": [row["assertion_id"] for row in weighted],
            "policy_decision_ids": policy_decision_ids,
            "fixed_decision_ids": fixed_decision_ids,
            "occurrence_to_decision": occurrence_to_decision,
            "decision_keys": [{
                "decision_id": str(row["decision_id"]),
                "runtime_type": row.get("runtime_type"),
                "canonical_key": row.get("canonical_key"),
            } for row in environment_document.get("decisions", [])],
            "occurrences": deepcopy(environment_document.get("occurrences", [])),
            "decisions": deepcopy(environment_document.get("decisions", [])),
            "uncovered_policy_decision_ids": [
                decision_id for decision_id in policy_decision_ids
                if decision_id not in policy_dependencies
            ],
        }

    all_candidates = [
        row for candidates in candidates_by_document.values() for row in candidates
    ]
    structural_cap = _validated_structural_cap(all_candidates, structural_cap)
    artifact = {
        "artifact_version": "utility-assertions-v2",
        "environment_hash": frozen_environment.get("environment_hash"),
        **dict(pins),
        "family_budgets": dict(family_budgets),
        "structural_cap": structural_cap,
        "documents": documents,
        "assertions": assertions,
        "rejections": {"summary_by_reason": {}},
    }
    artifact["artifact_hash"] = stable_hash(artifact)
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


def build_utility_artifact(
    frozen_environment: Mapping,
    task_adapter,
    source_documents: Mapping[str, str],
    *,
    threshold_manifest: Mapping,
    pins: Mapping,
    reader: Callable[[list[str], str, list[str | None]], Sequence[str]],
    render_action_vector: Callable[[str, Mapping[str, str]], str],
    relation_teacher: OpenRouterRelationTeacher | None = None,
    secondary_relation_teacher: OpenRouterRelationTeacher | None = None,
    environment_audit: Mapping | None = None,
    relation_support_escalator: "Callable[..., bool] | None" = None,
    informative_context_judge: "Callable[..., bool] | None" = None,
    context_prefilter: "Callable[[str, Mapping], Sequence[Mapping]] | None" = None,
    deterministic_relation_stage: bool = False,
    set_reader: "Callable[[list[str], str, list[str | None]], Sequence[str]] | None" = None,
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
    reader_pin = validated_build_reader_pin(reader, pins)
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
            record["attempt_hash"] = stable_hash({
                "doc_id": doc_id,
                "reason": stable_reason,
                "detail_reason": detail_reason,
                "rejection_id": record.get("rejection_id"),
                "evidence": dict(record.get("evidence") or {}),
                "definition_version": "qa-builder-attempt-v1",
            })
        if not record.get("rejection_id"):
            record["rejection_id"] = stable_hash({
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
        candidate_hash = stable_hash(candidate)
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
                # Freeze the v4 relation-constraint clause on the row before any read: the
                # artifact carries it, so gate, lattice probes, and runtime score the same
                # instrument. None (non-relation row, untypesettable locator) => no clause key
                # and the reader omits the constraint line.
                relation_clause = _relation_reader_clause(candidate, occurrences, decisions)
                if relation_clause is not None:
                    candidate = {**candidate, "reader_clause": relation_clause}
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
                    for term in occurrence_protected_terms(occurrence)
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
                                stable_hash(term) for term in surviving_terms
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
                            reader_threshold=reader_threshold, occurrences=occurrences,
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
                prompt_hash = stable_hash(prompt)
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
                            stable_hash(row["fact_key"]) for row in opportunities
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
                            "teacher_pin_hash": stable_hash(getattr(relation_teacher, "pin", {})),
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
                                "prompt_hash": stable_hash(prompt),
                                "definition_version": "contextual-relation-v1",
                            },
                            evidence={
                                "source": "relation_teacher",
                                "prompt_hash": stable_hash(prompt),
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
                            stable_hash(proposal): attempt
                            for proposal, attempt in zip(proposals, relation_attempts)
                        }
                        attempts_by_candidate_hash = {
                            stable_hash(candidate): attempts_by_proposal_hash.get(
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
                                    "prompt_hash": stable_hash(prompt),
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
                                    "prompt_hash": stable_hash(prompt),
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
                            "teacher_pin_hash": stable_hash(getattr(relation_teacher, "pin", {})),
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
                            "prompt_hash": stable_hash(prompt),
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
                            "prompt_hash": stable_hash(prompt),
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
                            relation_scope(
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
                                "fact_key_hash": stable_hash(t["fact_key"]),
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
                                batch_hash = stable_hash(repair_prompt)
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
                                    "teacher_pin_hash": stable_hash(
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
                            repair_prompt_hash = stable_hash(repair_prompt_hashes)
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
                                    stable_hash(proposal): attempt
                                    for proposal, attempt in zip(secondary_proposals, secondary_attempts)
                                }
                                attempts_by_candidate_hash = {
                                    stable_hash(candidate): attempts_by_proposal_hash.get(
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
                                stable_hash(repair_prompt_hashes) if repair_prompt_hashes else None
                            )
                            escalation["gleaning"]["repair_prompt_hash"] = repair_prompt_hash
                            escalation["gleaning"]["repair_prompt_hashes"] = repair_prompt_hashes
                            escalation["gleaning"]["batch_count"] = len(target_batches)
                            relation_teacher_runs_by_document.setdefault(doc_id, []).append({
                                "teacher_id": "gpt_oss",
                                "run_id": "gleaning",
                                "prompt_hash": repair_prompt_hash,
                                "teacher_pin_hash": stable_hash(
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
                                relation_scope(
                                    list(dict(row.get("evidence") or {}).get("arguments") or [])
                                ) == scope
                                for row in secondary_accepted
                            )
                            for scope in _RELATION_ESCALATION_SCOPES
                        }
                        escalation["merge_disposition"] = disposition
                        escalation["merged_kept_counts"] = {
                            scope: sum(
                                relation_scope(
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
                if not relation_arguments_are_legal(o["relation"], [subj, obj], ACI_RELATION_CONTRACT):
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
        # Kept-assertion cross-gate: runs after EVERY producer (primary, stage, gleaning merge,
        # reverse framing, literal reverse) so any producer's treating keep can veto an inverted
        # contraindication, and its rejections land before coverage accounting. The new
        # detail_reason is outside the fix-hint taxonomy, so a vetoed pair is never
        # repair-targeted, and — having been attempted — can never resurrect as "missed".
        accepted, treating_conflicts = _treating_conflict_filter(
            accepted, occurrences, decisions, doc_id=doc_id,
        )
        for record in treating_conflicts:
            preserve_rejection(record, doc_id=doc_id)
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
                    "fact_key_hash": stable_hash(row["fact_key"]),
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
    # Deferred: qa/review reads this module's gleaning/scope vocabulary, so the
    # import cannot sit at module scope.
    from cloak.qa.review import compute_review_flags
    artifact["review_flags"] = compute_review_flags(artifact)
    artifact["qa_audit"] = build_qa_audit(
        artifact, environment_audit=environment_audit,
    )
    artifact["artifact_hash"] = stable_hash({
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
        "action_vector_hash": stable_hash(action_vector),
    }


def _diagnose_coarser_readable(
    candidate: Mapping,
    decisions: Sequence[Mapping],
    protected_terms: Sequence[str],
    *,
    doc_id: str,
    render_action_vector: Callable[[str, Mapping[str, str]], str],
    reader: Callable[..., Sequence[str]],
    chain_by_decision: Mapping[str, Sequence[Mapping]],
    reader_threshold: float,
    occurrences: Mapping[str, Mapping] | None = None,
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
        levels = ordered_decision_levels(decision)  # finest -> coarsest
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
            # this probe COARSENS the LOCATOR level, so the frozen clause (pinned-level fill)
            # no longer matches the probe render — recompute it from the probe's requirements
            probe_clause = (
                _relation_reader_clause(probe, occurrences, decisions)
                if occurrences is not None else candidate.get("reader_clause")
            )
            answer = reader(
                [permuted_reader_question(probe, 0)],
                reader_excerpt(probe_context, candidate.get("evidence") or {}),
                [probe_clause],
            )[0]
            if context_answer_score(probe, answer, chain_by_decision) >= reader_threshold:
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
    reader: Callable[..., Sequence[str]],
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
    levels = ordered_decision_levels(decision)  # finest -> coarsest
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
        # the frozen clause stays valid: only the ANSWER level varies here and the clause names
        # locators only, exactly as at RL time
        answer = reader(
            [permuted_reader_question(candidate, 0)],
            reader_excerpt(probe_context, candidate.get("evidence") or {}),
            [candidate.get("reader_clause")],
        )[0]
        # scored against the ORIGINAL row (frozen required_property): the finer node must
        # resolve via its aliases and entail the supported level, as it must at RL time
        scores[str(finer)] = {
            "score": context_answer_score(candidate, answer, chain_by_decision),
            "render": "ok",
        }
    return scores


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
            spans = line_clause_spans(lines[turn])
            for clause, (left, right) in enumerate(spans):
                absolute_left, absolute_right = line_start + left, line_start + right
                if absolute_left < end and start < absolute_right:
                    refs.add((turn, clause, len(spans)))
    return [
        {"turn": turn, "clause": clause, "turn_clause_count": count}
        for turn, clause, count in sorted(refs)
    ]


def validate_context_assertions(
    assertions: Sequence[Mapping],
    *,
    original_context: str,
    representative_context: str,
    placeholder_context: str,
    reader: Callable[[list[str], str, list[str | None]], Sequence[str]],
    threshold: float = 1.0,
    stability_repetitions: int = 1,
    option_permutations: int = 1,
    stability_threshold: float = 1.0,
    chain_by_decision: Mapping[str, Sequence[Mapping]] | None = None,
    set_reader: Callable[[list[str], str, list[str | None]], Sequence[str]] | None = None,
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
                question = permuted_reader_question(row, permutation_index)
                # Frozen relation-constraint clause (v4 reader): the same string on all three
                # renders and at runtime, so every read certifies the same instrument.
                clauses = [row.get("reader_clause")]
                reader_evidence = row.get("evidence") or {}
                ox = reader_excerpt(original_context, reader_evidence)
                rx = reader_excerpt(representative_context, reader_evidence)
                px = reader_excerpt(placeholder_context, reader_evidence)
                debug_excerpts[id(row)] = (question, ox, rx, px)
                row_reader = (
                    set_reader
                    if set_reader is not None
                    and (row.get("answer_target") or {}).get("kind") == "linked_decision_set"
                    else reader
                )
                original_answers += row_reader([question], ox, clauses)
                representative_answers += row_reader([question], rx, clauses)
                placeholder_answers += row_reader([question], px, clauses)
            if not all(len(answers) == len(rows) for answers in (
                original_answers, representative_answers, placeholder_answers
            )):
                raise ValueError("reader returned the wrong number of answers")
            for row, original, representative, placeholder in zip(
                rows, original_answers, representative_answers, placeholder_answers
            ):
                scores = {
                    "original": context_answer_score(row, original, chain_by_decision),
                    "representative": context_answer_score(row, representative, chain_by_decision),
                    "placeholder": context_answer_score(row, placeholder, chain_by_decision),
                }
                gate_debug(debug_excerpts.get(id(row)), original, representative, placeholder, scores)
                assertion_id = str(row.get("assertion_id") or stable_hash(dict(row)))
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
        assertion_id = str(row.get("assertion_id") or stable_hash(dict(row)))
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
