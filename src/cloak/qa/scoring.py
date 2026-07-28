"""Runtime utility scoring: the pinned context reader, answer scoring, and the
delivered-contract / three-point reader primitives the training loop executes.

Bottom layer of the QA stack -- imports no other ``qa_*`` module (pinned by
``cloak/tests/test_architecture.py``) so the reader pin and
``UTILITY_SCORER_VERSION`` cannot drift with build-time compiler changes."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from math import isfinite
from numbers import Real

# v2 (2026-07-28): utility aggregates over policy-role assertions only; probes with
# no policy-decision dependency (DEMOGRAPHIC fields, placeholder round-trips,
# structural checks) become a reported monitoring aggregate, not reward mass.
UTILITY_SCORER_VERSION = "qa-utility-runtime-v2"


# Contract kinds that demand verbatim reproduction of the gold reference (measured
# 0-pass on real remote notes, 2026-07-28): unwinnable by any action, so they carry
# no policy signal even when dependency-linked. Revisit at the next artifact rebuild.
_GOLD_EXACTNESS_CONTRACT_KINDS = frozenset({"exact_relation"})


def assertion_reward_role(assertion, document_entry) -> str:
    """"policy" when the assertion depends on a policy decision, else "monitoring"."""
    contract = assertion.get("scoring_contract") or {}
    if contract.get("kind") in _GOLD_EXACTNESS_CONTRACT_KINDS:
        return "monitoring"
    dependencies = set(assertion.get("policy_dependency_decision_ids") or ())
    policy_ids = set(document_entry.get("policy_decision_ids") or ())
    return "policy" if dependencies & policy_ids else "monitoring"


QA_MODEL = "medgemma-4b-it"  # THE context reader -- the SINGLE definition. Every reader path


                            # (DEFAULT_CONTEXT_READER_PIN, BatchedContextReader) imports this;
                            # do not hardcode a reader model elsewhere or add a per-call reader
                            # override. Served generative whole-doc reader on llama-swap :8060,
                            # temp0/greedy, non-thinking, batched via pmap (workers match -np 6).
                            # Reader sweep (2026-07-19): medgemma > gemma 4 E4B on QA yield
                            # (kept 47 vs 44, three_point_gate_failed 29 vs 31) at the same
                            # local 4B cost; a 120B gpt-oss reader rejected MORE, so
                            # gate-failure is not reader-capability-bound (see memory).
QA_BASE_URL = "http://localhost:8060/v1"


def canon(t: str) -> str:
    """Canonicalize for restatement matching: spoken-vs-written variants seen in the
    corpora ("doctor kumar"/"Dr. Kumar", "40 milligrams"/"40 mg"). ponytail: spelled-out
    numbers/dates ("July thirty first") stay unmatched; add a number normalizer if probe
    supply still short."""
    t = re.sub(r"\bdr\.?(?=\s)", "doctor", t.lower())
    return re.sub(r"\bmilligrams?\b", "mg", t)


def fact_score(pred: str, gold: str) -> float:
    """Fact-recall scorer v2 (re-pin 2026-07-06): canon-normalize -> NUMBER GATE (every gold
    numeric token must appear in the answer, else 0 — kills unit false-positives like
    "10 mg" vs "40 milligrams") -> containment (gold tokens subset of answer -> 1.0, so a
    verbose-but-correct "AT&T Corporation" for "AT&T" is a hit) -> acronym (CHF == the initials
    of "Congestive Heart Failure") -> token-F1 fallback. Residual under-scores: non-initial
    abbreviations (HTN) and pure synonyms (renal == kidney)."""
    p = re.findall(r"\w+", canon(pred)); g = re.findall(r"\w+", canon(gold))
    if not g:
        return float(not p)
    if not p:
        return 0.0
    gnum = [t for t in g if t.isdigit()]
    pnum = {t for t in p if t.isdigit()}
    if gnum and not all(n in pnum for n in gnum):
        return 0.0
    ps, gs = set(p), set(g)
    if gs <= ps:
        return 1.0
    def _acro(short, long_):
        return len(short) == 1 and len(short[0]) >= 2 and short[0] == "".join(w[0] for w in long_)
    if _acro(p, g) or _acro(g, p):
        return 1.0
    common = sum(min(p.count(t), g.count(t)) for t in gs)
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


CONTEXT_READER_PROMPT_VERSION = "qa-context-reader-v4"


# version 3: contiguity + verbosity-cap scoring — _resolve_semantic_node requires an alias to
# appear as a CONTIGUOUS stemmed subsequence of the reply (was: token-SET subset, which credited
# scattered tokens), with at most _ANSWER_EXTRA_TOKEN_CAP meaningful tokens of reply verbosity
# (which credited whole-sentence echoes). The schema version covers the full
# answer-interpretation contract, not just the wire format.
CONTEXT_READER_RESPONSE_SCHEMA = {"type": "single-span", "version": 3}


# Set from the 2026-07-21 scorer A/B and FROZEN before any re-gate; never retuned per result
# (empirical-honesty rule). Measured extra-token distribution: legitimate replies (hedge
# prefixes, dose/qualifier tails, compound spans) at 1-6 extra; two verbose two-drug compound
# sentences at 9 (deliberately left out — recovering them needs cap>=9, the near-echo regime);
# whole-sentence echoes at 26. The 6->26 gap places the cap.
_ANSWER_EXTRA_TOKEN_CAP = 6


CONTEXT_READER_REVISION = "qa-reader-r5"


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


CLAUSE_DELIMITER_PATTERN = re.compile(r"[\n.!?;]")


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

    def _read_one(
        self, question: str, context: str, clause: str | None = None, *, refresh: bool = False,
    ) -> str:
        # qa-context-reader-v4: the optional relation-constraint line restates the QA as the
        # relation the answer must satisfy (see _relation_reader_clause). Without a clause the
        # template is the same minus that line (legacy rows, non-relation probes).
        lines = [
            "Read the DOCUMENT and answer the QUESTION. Copy the answer span exactly from the "
            "DOCUMENT — do not rephrase, summarize, or add words.",
            "If the DOCUMENT does not answer it, reply with exactly NONE.",
        ]
        if clause:
            lines.append(f"Your ANSWER must satisfy the relation: ANSWER {clause}.")
        prompt = "\n".join(lines) + f"\n\nDOCUMENT:\n{context}\n\nQUESTION: {question}"
        raw = (
            self._client.generate(prompt, refresh=True)
            if refresh else self._client.generate(prompt)
        ).strip()
        if raw.startswith("```") and raw.endswith("```"):
            raw = raw.strip("`").strip()
        raw = raw.strip().strip('"').strip("'").strip()
        return "" if raw.upper() == "NONE" else raw

    def _read_set_one(
        self, question: str, context: str, clause: str | None = None, *, refresh: bool = False,
    ) -> str:
        # Set-valued read (validated wrapper: scripts/spikes/set_valued_gate_probe.py): the raw
        # JSON-array reply is returned verbatim; context_answer_score parses and scores recall.
        lines = [
            "Read the DOCUMENT and answer the QUESTION. Copy each answer verbatim as a short "
            "phrase from the DOCUMENT; include nothing not in the DOCUMENT. Respond with ONLY a "
            'JSON array of strings, e.g. ["x","y"]. If there are none, respond [].',
        ]
        if clause:
            lines.append(f"Every answer must satisfy the relation: ANSWER {clause}.")
        prompt = "\n".join(lines) + f"\n\nDOCUMENT:\n{context}\n\nQUESTION: {question}"
        return (
            self._client.generate(prompt, refresh=True)
            if refresh else self._client.generate(prompt)
        ).strip()

    def read_set(
        self, questions: list[str], context: str, clauses: Sequence[str | None] | None = None,
        *, refresh: bool = False,
    ) -> list[str]:
        clauses = clauses if clauses is not None else [None] * len(questions)
        return [
            self._read_set_one(question, context, clause, refresh=refresh)
            for question, clause in zip(questions, clauses, strict=True)
        ]

    def __call__(
        self, questions: list[str], context: str, clauses: Sequence[str | None] | None = None,
        *, refresh: bool = False,
    ) -> list[str]:
        clauses = clauses if clauses is not None else [None] * len(questions)
        return [
            self._read_one(question, context, clause, refresh=refresh)
            for question, clause in zip(questions, clauses, strict=True)
        ]


_batched_context_reader = None


def read_context_batch(
    questions: list[str], context: str, clauses: Sequence[str | None] | None = None,
    *, refresh: bool = False,
) -> list[str]:
    global _batched_context_reader
    if _batched_context_reader is None:
        _batched_context_reader = BatchedContextReader()
    return _batched_context_reader(questions, context, clauses, refresh=refresh)


read_context_batch.pin = deepcopy(DEFAULT_CONTEXT_READER_PIN)


def read_context_set_batch(
    questions: list[str], context: str, clauses: Sequence[str | None] | None = None,
    *, refresh: bool = False,
) -> list[str]:
    """Set-valued reads (same pinned reader model/endpoint, JSON-array response contract)."""
    global _batched_context_reader
    if _batched_context_reader is None:
        _batched_context_reader = BatchedContextReader()
    return _batched_context_reader.read_set(
        questions, context, clauses, refresh=refresh,
    )


read_context_set_batch.pin = deepcopy(DEFAULT_CONTEXT_READER_PIN)


def stable_hash(value) -> str:
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


def validated_build_reader_pin(reader, pins: Mapping) -> dict:
    provided = _validated_reader_pin(pins.get("reader_pin"))
    injected = getattr(reader, "pin", None)
    if not isinstance(injected, Mapping) or not injected:
        raise ValueError("injected reader requires an explicit structured pin")
    actual = _validated_reader_pin(injected, label="injected reader pin")
    if provided != actual:
        raise ValueError("reader_pin must match the injected reader pin")
    return provided


_LEAKAGE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "being", "by", "did", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "with",
})


def meaningful_tokens(value: str) -> set[str]:
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
    return {_singularize(token) for token in meaningful_tokens(value)}


def _stemmed_token_sequence(value: str) -> tuple[str, ...]:
    """Ordered variant of _stemmed_tokens (same canon/stopword/inflection folding, order kept):
    the exact-span scorer compares SEQUENCES, so a reply must be the alias, not merely contain
    its words somewhere."""
    return tuple(
        _singularize(token) for token in re.findall(r"[a-z0-9]+", canon(value))
        if len(token) > 2 and token not in _LEAKAGE_STOPWORDS
    )


def normalized_aliases(value) -> list[str]:
    if value is None:
        return []
    values = [value] if isinstance(value, str) else list(value)
    aliases = []
    for alias in values:
        exact_alias = str(alias)
        if exact_alias.strip() and exact_alias not in aliases:
            aliases.append(exact_alias)
    return aliases


def occurrence_protected_terms(occurrence: Mapping) -> list[str]:
    terms = [str(occurrence.get("surface", "")).strip()]
    terms.extend(alias.strip() for alias in normalized_aliases(occurrence.get("aliases")))
    return [term for term in terms if term]


def parse_aci_note(text: str) -> dict:
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


def parse_llm_json_array(raw: str) -> list[str]:
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


def _answer_score(answer: str, accepted_values: Sequence[str]) -> float:
    if not accepted_values:
        return 0.0
    return max(fact_score(answer, value) for value in accepted_values)


def _resolve_semantic_node(chain: Sequence[Mapping], answer: str) -> dict | None:
    """Resolve a free-form reader answer to exactly one node in one decision's
    frozen semantic chain, by its answer_aliases. Decision-scoped and lexical;
    ambiguous or unresolved -> None. A protected source value resolves to KEEP
    locally without ever entering the artifact's public answer golds.

    Contiguity + verbosity cap (reader schema v3): an alias must appear as a
    CONTIGUOUS stemmed subsequence of the reply, and the reply may carry at most
    _ANSWER_EXTRA_TOKEN_CAP meaningful tokens beyond the alias. Keeps legitimate
    reader verbosity resolvable — hedge prefixes, dose/qualifier tails, compound
    spans (measured: 32/37 strict-equality flips were such false negatives) —
    while rejecting the containment rule's unearned passes: whole-sentence
    echoes (over the cap) and tokens scattered across phrases (non-contiguous)."""
    answer_sequence = _stemmed_token_sequence(answer)
    if not answer_sequence:
        return None

    def resolves(alias: str) -> bool:
        alias_sequence = _stemmed_token_sequence(alias)
        span = len(alias_sequence)
        if not span or len(answer_sequence) - span > _ANSWER_EXTRA_TOKEN_CAP:
            return False
        return any(
            answer_sequence[start:start + span] == alias_sequence
            for start in range(len(answer_sequence) - span + 1)
        )

    matches = [
        node for node in chain
        if any(resolves(str(alias)) for alias in node.get("answer_aliases") or [])
    ]
    # Exact equality can still match several nodes when aliases repeat across
    # the chain; matches[0] is the finest (chain is linear specific->coarse).
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


_TURN_EXCERPT_ELISION = "[...]"


def line_clause_spans(line: str) -> list[tuple[int, int]]:
    """Return delimiter-terminated clause spans relative to one rendered turn."""
    spans, left = [], 0
    for delimiter in CLAUSE_DELIMITER_PATTERN.finditer(line):
        right = delimiter.end()
        if line[left:right].strip():
            spans.append((left, right))
        left = right
    if line[left:].strip():
        spans.append((left, len(line)))
    return spans


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
        spans = line_clause_spans(lines[turn])
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


def gate_debug(exc, orig_ans, rep_ans, ph_ans, scores) -> None:
    """Dev-only dump of what the reader saw + answered per render, for gate debugging. OFF unless
    $CLOAK_GATE_DEBUG_DIR is set; filtered to reverse-orientation questions to keep output tiny."""
    out_dir = os.getenv("CLOAK_GATE_DEBUG_DIR")
    if not out_dir or not exc:
        return
    question, ox, rx, px = exc
    if not question.startswith("For what medical condition was the"):
        return
    os.makedirs(out_dir, exist_ok=True)
    key = stable_hash(question + ox).split(":")[-1][:16]
    with open(os.path.join(out_dir, key + ".json"), "w") as f:
        json.dump({"question": question, "scores": scores,
                   "original": {"excerpt": ox, "answer": list(orig_ans)},
                   "representative": {"excerpt": rx, "answer": list(rep_ans)},
                   "placeholder": {"excerpt": px, "answer": list(ph_ans)}},
                  f, ensure_ascii=False, indent=1)


def reader_excerpt(context: str, evidence: Mapping) -> str:
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


def context_answer_score(
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
        predictions = parse_llm_json_array(answer)
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


def permuted_reader_question(assertion: Mapping, permutation_index: int) -> str:
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
        key=lambda option: stable_hash({
            "assertion_id": str(assertion.get("assertion_id", "")),
            "option": option,
        }),
    )
    shift = permutation_index % len(ordered)
    permutation = ordered[shift:] + ordered[:shift]
    return f"{question}\nOptions: {' | '.join(permutation)}"


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


def prepare_utility_scoring(
    artifact: Mapping, doc_id: str, *, doc_p: str, out_final: str,
) -> dict:
    """Prepare deterministic scores and exact per-assertion reader work."""
    assertions = [
        row for row in artifact.get("assertions", {}).values()
        if row.get("doc_id") == doc_id and row.get("status", "accepted") == "accepted"
    ]
    assertions.sort(key=lambda row: str(row["assertion_id"]))
    context_rows = [row for row in assertions if row.get("family") == "context"]
    context_work = []
    for row in context_rows:
        context_work.append({
            "assertion_id": str(row["assertion_id"]),
            "question": permuted_reader_question(row, 0),
            "context": reader_excerpt(doc_p, row.get("evidence") or {}),
            "reader_clause": row.get("reader_clause"),
            "set_valued": (
                (row.get("answer_target") or {}).get("kind")
                == "linked_decision_set"
            ),
        })
    chain_by_decision = {
        str(decision["decision_id"]): decision.get("semantic_chain", [])
        for decision in artifact["documents"][doc_id].get("decisions", [])
    }
    scores: dict[str, float] = {}
    parsed_output = parse_aci_note(out_final)
    for row in assertions:
        if row.get("family") != "delivered":
            continue
        contract = row.get("scoring_contract") or {}
        scores[str(row["assertion_id"])] = _score_delivered_contract(
            contract, out_final, parsed_output
        )
    return {
        "assertions": assertions,
        "context_rows": context_rows,
        "context_work": context_work,
        "chain_by_decision": chain_by_decision,
        "deterministic_scores": scores,
        "utility_weight_denominator": float(
            artifact["documents"][doc_id]["utility_weight_denominator"]
        ),
        "reward_roles": {
            str(row["assertion_id"]): assertion_reward_role(
                row, artifact["documents"][doc_id]
            )
            for row in assertions
        },
    }


def finalize_utility_scoring(
    prepared: Mapping, context_answers: Sequence[str],
) -> dict:
    """Reconstruct and validate one complete frozen component vector."""
    context_rows = prepared["context_rows"]
    if len(context_answers) != len(context_rows):
        raise ValueError("reader returned the wrong number of answers")
    scores = dict(prepared["deterministic_scores"])
    for row, answer in zip(context_rows, context_answers, strict=True):
        scores[str(row["assertion_id"])] = context_answer_score(
            row, answer, prepared["chain_by_decision"],
        )
    assertions = prepared["assertions"]
    expected_ids = {str(row["assertion_id"]) for row in assertions}
    if set(scores) != expected_ids:
        raise ValueError("utility scoring did not produce exact assertion coverage")
    if any(
        not isinstance(score, Real) or not isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
        for score in scores.values()
    ):
        raise ValueError("utility scoring produced an invalid component score")

    roles = prepared["reward_roles"]

    def _aggregate(role: str) -> tuple[float, float]:
        rows = [row for row in assertions if roles[str(row["assertion_id"])] == role]
        weight = sum(float(row["weight"]) for row in rows)
        numerator = sum(
            float(row["weight"]) * scores[str(row["assertion_id"])] for row in rows
        )
        return (numerator / weight if weight else 0.0), weight

    utility, policy_weight = _aggregate("policy")
    if policy_weight <= 0.0:
        raise ValueError("utility scoring requires positive policy reward mass")
    monitoring_utility, monitoring_weight = _aggregate("monitoring")
    return {
        "component_scores": scores,
        "utility": utility,
        "monitoring": {
            "utility": monitoring_utility,
            "weight": monitoring_weight,
        },
    }


def _call_runtime_reader(
    reader: Callable, question: str, context: str, clause: str | None, refresh: bool,
) -> Sequence[str]:
    if not refresh:
        return reader([question], context, [clause])
    parameters = inspect.signature(reader).parameters.values()
    supports_refresh = any(
        parameter.name == "refresh" or parameter.kind == parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if not supports_refresh:
        raise ValueError("reader_refresh requires a reader that accepts refresh")
    return reader([question], context, [clause], refresh=True)


def validate_utility_reader_pin(reader: Callable, artifact: Mapping) -> None:
    """Require the runtime reader to match the utility artifact's frozen pin."""
    validated_build_reader_pin(reader, artifact)


def score_utility(
    artifact: Mapping,
    doc_id: str,
    *,
    doc_p: str,
    out_final: str,
    reader: Callable[[list[str], str, list[str | None]], Sequence[str]],
    set_reader: Callable[[list[str], str, list[str | None]], Sequence[str]] | None = None,
    reader_refresh: bool = False,
) -> dict:
    """Score one document with one reader request per context assertion."""
    validated_build_reader_pin(reader, artifact)
    prepared = prepare_utility_scoring(
        artifact, doc_id, doc_p=doc_p, out_final=out_final,
    )
    if any(item["set_valued"] for item in prepared["context_work"]):
        if set_reader is None:
            set_reader = reader
        validated_build_reader_pin(set_reader, artifact)
    context_answers = []
    for item in prepared["context_work"]:
        row_reader = set_reader if item["set_valued"] else reader
        answers = _call_runtime_reader(
            row_reader,
            item["question"],
            item["context"],
            item["reader_clause"],
            reader_refresh,
        )
        if len(answers) != 1:
            raise ValueError("reader returned the wrong number of answers")
        context_answers.append(answers[0])
    return finalize_utility_scoring(prepared, context_answers)
