"""Frozen extractor: pinned tiered recovery for doc_p/out_p round trips.

The skeleton keeps the existing rule cascade as tier 0 and records the frozen
pin table in the extractor version. Later tiers slot between tier 0 and
finalization without changing the public `extract()` entrypoint.

The doc_p->out_p alignment prior is intentionally exposed as pure helpers for
the later scoring stage. It supplies only a score bonus: candidates outside the
returned window must still be generated and scored globally, never filtered out.
"""
import hashlib
import json
import re

import numpy as np

from cloak.extract import _finalize, _rule_prepass
from cloak.reconstruct import _value_compatible
from cloak.runtime_types import PLACEHOLDER_RE


_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?](?=\s|$)|\n")
_WORD_RE = re.compile(r"\b[^\W_]+(?:['-][^\W_]+)*\b", re.UNICODE)
_CAPITALIZED_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:['-][A-Za-z0-9]+)*\b")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}
_FILL_SCORE_WEIGHT = 0.6
_SURFACE_SCORE_WEIGHT = 0.4


TYPE_HYPOTHESES = {
    "PERSON": "This text mentions a person name or alias.",
    "CODE": "This text mentions a reference number, contact code, or account-like identifier.",
    "ORG": "This text mentions an organization, company, court, or institution.",
    "LOC": "This text mentions a location, address, city, or country.",
    "DATETIME": "This text mentions a date, time, or duration.",
    "QUANTITY": "This text mentions an amount, money, percentage, or count.",
    "MISC": "This text mentions an identifying residual attribute or event.",
    "nationality": "This text mentions a nationality or citizenship.",
    "ethnicity": "This text mentions an ethnicity, race, or ancestry group.",
    "religion": "This text mentions a religion, belief, denomination, or branch.",
    "profession": "This text mentions a profession, occupation, or job title.",
    "age": "This text mentions an age expression.",
    "gender": "This text mentions a gender value.",
    "marital-status": "This text mentions a marital status value.",
    "health-condition": "This text mentions a disease, diagnosis, or health condition.",
    "sexual-orientation": "This text mentions a sexual orientation value.",
    "family-role": "This text mentions a family role or relationship.",
    "demographic-other": "This text mentions a residual demographic attribute.",
}


EXTRACTOR_PINS = {
    "models": {
        "encoder": "BAAI/bge-small-en-v1.5",
        "nli": "cross-encoder/nli-deberta-v3-small",
        "mlm": "roberta-base",
    },
    "thresholds": {
        "SIM_MIN": 0.55,
        "ASSIGN_MARGIN": 0.05,
        "PRIOR_WEIGHT": 0.15,
        "NLI_ENTAIL": 0.80,
        "TYPE_ENTAIL": 0.70,
        "PLL_MIN_DELTA": -6.0,
        "EPS_MARGIN": 0.02,
        "CHUNK_MAX_WORDS": 6,
    },
    "type_hypotheses": TYPE_HYPOTHESES,
    "ladder_semver": "0.1.0",
}


def _canonical_json(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def extractor_version() -> str:
    digest = hashlib.sha256(_canonical_json(EXTRACTOR_PINS).encode("utf-8")).hexdigest()
    return "fx-" + digest[:12]


def _trim_nonempty_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return deterministic sentence character spans without loading a model."""
    spans: list[tuple[int, int]] = []
    n = len(text)
    start = 0
    while start < n and text[start].isspace():
        start += 1

    for match in _SENTENCE_BOUNDARY_RE.finditer(text, start):
        end = match.start() if match.group(0) == "\n" else match.end()
        span = _trim_nonempty_span(text, start, end)
        if span is not None:
            spans.append(span)
        start = match.end()
        while start < n and text[start].isspace():
            start += 1

    span = _trim_nonempty_span(text, start, n)
    if span is not None:
        spans.append(span)
    return spans


def _as_normalized_matrix(vecs) -> np.ndarray:
    matrix = np.asarray(vecs, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("sentence vectors must be a 2D array")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def align_sentences(doc_vecs, out_vecs) -> list[int]:
    """Map each doc sentence to an out sentence using deterministic DTW.

    The DTW recurrence uses cosine distance and allows diagonal, doc-advance,
    and out-advance moves. Equal-cost predecessors prefer the diagonal move;
    remaining ties are resolved by proximity to the normalized index diagonal.
    """
    doc = _as_normalized_matrix(doc_vecs)
    out = _as_normalized_matrix(out_vecs)
    n_doc, n_out = doc.shape[0], out.shape[0]
    if n_doc == 0:
        return []
    if n_out == 0:
        return [-1] * n_doc

    cost = 1.0 - np.clip(doc @ out.T, -1.0, 1.0)
    dp = np.full((n_doc, n_out), np.inf, dtype=np.float64)
    prev = np.full((n_doc, n_out, 2), -1, dtype=np.int64)
    dp[0, 0] = cost[0, 0]

    for i in range(n_doc):
        for j in range(n_out):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0 and j > 0:
                candidates.append((dp[i - 1, j - 1], 0, i - 1, j - 1))
            if i > 0:
                candidates.append((dp[i - 1, j], 1, i - 1, j))
            if j > 0:
                candidates.append((dp[i, j - 1], 2, i, j - 1))
            _, _, pi, pj = min(
                candidates,
                key=lambda item: (
                    item[0],
                    item[1],
                    abs(_normalized_index(item[2], n_doc) - _normalized_index(item[3], n_out)),
                ),
            )
            dp[i, j] = dp[pi, pj] + cost[i, j]
            prev[i, j] = (pi, pj)

    path: list[tuple[int, int]] = []
    i, j = n_doc - 1, n_out - 1
    while i >= 0 and j >= 0:
        path.append((i, j))
        pi, pj = prev[i, j]
        if pi < 0 or pj < 0:
            break
        i, j = int(pi), int(pj)
    path.reverse()

    alignment: list[int] = []
    for doc_idx in range(n_doc):
        choices = [out_idx for path_doc_idx, out_idx in path if path_doc_idx == doc_idx]
        if not choices:
            alignment.append(-1)
            continue
        alignment.append(
            min(
                choices,
                key=lambda out_idx: (
                    cost[doc_idx, out_idx],
                    abs(_normalized_index(doc_idx, n_doc) - _normalized_index(out_idx, n_out)),
                    out_idx,
                ),
            )
        )
    return alignment


def _normalized_index(idx: int, count: int) -> float:
    if count <= 1:
        return 0.0
    return idx / (count - 1)


def position_bonus(
    fill_span,
    doc_sent_spans: list[tuple[int, int]] | None,
    out_sent_spans: list[tuple[int, int]] | None,
    alignment: list[int],
) -> tuple[int, int] | None:
    """Return the aligned out_p window for a fill span, or None when unavailable.

    The returned window is the aligned sentence plus one neighboring sentence on
    each side. Callers may add `PRIOR_WEIGHT` inside this window, but must never
    use the window to discard candidates.
    """
    if fill_span is None or doc_sent_spans is None or out_sent_spans is None:
        return None
    if len(fill_span) != 2 or not doc_sent_spans or not out_sent_spans:
        return None

    fill_start, fill_end = int(fill_span[0]), int(fill_span[1])
    if fill_start >= fill_end:
        return None

    doc_idx = _containing_sentence(fill_start, fill_end, doc_sent_spans)
    if doc_idx is None or doc_idx >= len(alignment):
        return None
    out_idx = alignment[doc_idx]
    if out_idx < 0 or out_idx >= len(out_sent_spans):
        return None

    window_start_idx = max(0, out_idx - 1)
    window_end_idx = min(len(out_sent_spans) - 1, out_idx + 1)
    return out_sent_spans[window_start_idx][0], out_sent_spans[window_end_idx][1]


def _containing_sentence(
    fill_start: int,
    fill_end: int,
    sent_spans: list[tuple[int, int]],
) -> int | None:
    for idx, (sent_start, sent_end) in enumerate(sent_spans):
        if sent_start <= fill_start and fill_end <= sent_end:
            return idx

    overlaps = [
        (min(fill_end, sent_end) - max(fill_start, sent_start), idx)
        for idx, (sent_start, sent_end) in enumerate(sent_spans)
    ]
    overlap, idx = max(overlaps, key=lambda item: (item[0], -item[1]))
    return idx if overlap > 0 else None


def candidate_chunks(out_p: str) -> list[tuple[int, int, str]]:
    """Return deterministic word n-gram candidate spans over all of `out_p`."""
    max_words = int(EXTRACTOR_PINS["thresholds"]["CHUNK_MAX_WORDS"])
    tokens = [
        (match.start(), match.end(), match.group(0))
        for match in _WORD_RE.finditer(out_p)
    ]
    chunks: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    for start_idx in range(len(tokens)):
        for n_words in range(1, max_words + 1):
            end_idx = start_idx + n_words
            if end_idx > len(tokens):
                break
            words = [token for _, _, token in tokens[start_idx:end_idx]]
            if _only_stopwords_or_punctuation(words):
                continue
            start = tokens[start_idx][0]
            end = tokens[end_idx - 1][1]
            if (start, end) in seen:
                continue
            seen.add((start, end))
            chunks.append((start, end, out_p[start:end]))
    return chunks


def _only_stopwords_or_punctuation(words: list[str]) -> bool:
    for word in words:
        normalized = re.sub(r"[^\w]+", "", word.lower(), flags=re.UNICODE)
        if normalized and normalized not in _STOPWORDS:
            return False
    return True


def score_pairs(
    residue: list[dict],
    chunks: list[tuple[int, int, str]],
    encoder,
    windows: list[tuple[int, int] | None],
) -> list[tuple[float, int, int]]:
    """Score every residue entry against every candidate chunk in one encode batch."""
    if not residue or not chunks:
        return []

    fills = [_entry_fill(entry) for entry in residue]
    surfaces = [str(entry.get("surface", "")) for entry in residue]
    chunk_texts = [chunk_text for _, _, chunk_text in chunks]
    vecs = _as_normalized_matrix(encoder.encode(fills + surfaces + chunk_texts))

    n_entries = len(residue)
    fill_vecs = vecs[:n_entries]
    surface_vecs = vecs[n_entries:2 * n_entries]
    chunk_vecs = vecs[2 * n_entries:]

    fill_scores = np.clip(fill_vecs @ chunk_vecs.T, -1.0, 1.0)
    surface_scores = np.clip(surface_vecs @ chunk_vecs.T, -1.0, 1.0)
    scores: list[tuple[float, int, int]] = []
    for entry_idx in range(n_entries):
        window = windows[entry_idx] if entry_idx < len(windows) else None
        for chunk_idx, chunk in enumerate(chunks):
            score = (
                _FILL_SCORE_WEIGHT * fill_scores[entry_idx, chunk_idx]
                + _SURFACE_SCORE_WEIGHT * surface_scores[entry_idx, chunk_idx]
                + _prior_bonus(chunk, window)
            )
            scores.append((float(score), entry_idx, chunk_idx))
    return scores


def _entry_fill(entry: dict) -> str:
    return str(entry.get("replacement", entry.get("fill", "")))


def _prior_bonus(
    chunk: tuple[int, int, str],
    window: tuple[int, int] | None,
) -> float:
    if window is None:
        return 0.0
    window_start, window_end = int(window[0]), int(window[1])
    if window_start >= window_end:
        return 0.0

    prior_weight = float(EXTRACTOR_PINS["thresholds"]["PRIOR_WEIGHT"])
    chunk_start, chunk_end, _ = chunk
    chunk_center = (chunk_start + chunk_end) / 2.0
    if window_start <= chunk_center <= window_end:
        return prior_weight

    distance = window_start - chunk_center if chunk_center < window_start else chunk_center - window_end
    decay_width = max(1.0, float(window_end - window_start))
    return prior_weight * max(0.0, 1.0 - (distance / decay_width))


class Assignment(dict):
    """Entry-to-chunk assignment plus fail-closed reasons for unassigned entries."""

    def __init__(self, *args, abstained: dict[int, str] | None = None):
        super().__init__(*args)
        self.abstained = abstained or {}


def assign(
    scores: list[tuple[float, int, int]],
    n_entries: int,
    chunks: list[tuple[int, int, str]],
) -> Assignment:
    sim_min = float(EXTRACTOR_PINS["thresholds"]["SIM_MIN"])
    margin = float(EXTRACTOR_PINS["thresholds"]["ASSIGN_MARGIN"])
    assignments: dict[int, int] = {}
    assigned_scores: dict[int, float] = {}
    chunk_owner: dict[int, int] = {}
    taken_spans: list[tuple[int, int]] = []

    for score, entry_idx, chunk_idx in sorted(scores, key=lambda item: _assign_sort_key(item, chunks)):
        if score < sim_min:
            continue
        if entry_idx in assignments:
            continue
        if chunk_idx < 0 or chunk_idx >= len(chunks):
            continue
        chunk_start, chunk_end, _ = chunks[chunk_idx]
        if any(_spans_overlap((chunk_start, chunk_end), span) for span in taken_spans):
            continue
        assignments[entry_idx] = chunk_idx
        assigned_scores[entry_idx] = float(score)
        chunk_owner[chunk_idx] = entry_idx
        taken_spans.append((chunk_start, chunk_end))

    ambiguous: set[int] = set()
    for entry_idx, chunk_idx in assignments.items():
        taken_score = assigned_scores[entry_idx]
        alternatives = [
            score
            for score, score_entry_idx, score_chunk_idx in scores
            if score_entry_idx == entry_idx
            and score_chunk_idx != chunk_idx
            and chunk_owner.get(score_chunk_idx) not in (None, entry_idx)
        ]
        if alternatives and max(alternatives) >= taken_score - margin:
            ambiguous.add(entry_idx)

    for entry_idx in ambiguous:
        del assignments[entry_idx]

    abstained = {entry_idx: "ambiguous" for entry_idx in sorted(ambiguous)}
    for entry_idx in range(n_entries):
        if entry_idx not in assignments and entry_idx not in abstained:
            abstained[entry_idx] = "no-candidate"

    return Assignment(assignments, abstained=abstained)


def _assign_sort_key(
    item: tuple[float, int, int],
    chunks: list[tuple[int, int, str]],
) -> tuple[float, int, int, int]:
    score, entry_idx, chunk_idx = item
    if 0 <= chunk_idx < len(chunks):
        chunk_start, chunk_end, _ = chunks[chunk_idx]
    else:
        chunk_start, chunk_end = 0, 0
    return -float(score), chunk_start, chunk_end, entry_idx


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def verify(entry: dict, chunk_text: str, sentence: str, nli) -> tuple[bool, str]:
    """Verify an assigned chunk before splice; fail closed on the first failed gate."""
    fill = _entry_fill(entry)
    normalized_fill = fill.strip()
    if not normalized_fill or PLACEHOLDER_RE.fullmatch(normalized_fill):
        return False, "bad-fill"

    if _value_compatible(fill, chunk_text) is False:
        return False, "added-digit"

    ok, reason = _require_entailment(
        nli(_fill_sentence_template(fill), _chunk_sentence(chunk_text, sentence)),
        float(EXTRACTOR_PINS["thresholds"]["NLI_ENTAIL"]),
        margin_reason="margin-correspondence",
        reject_reason="correspondence",
    )
    if not ok:
        return False, reason

    runtime_type = str(entry.get("type", "MISC"))
    hypothesis = EXTRACTOR_PINS["type_hypotheses"].get(runtime_type)
    if hypothesis is not None:
        ok, reason = _require_entailment(
            nli(_chunk_sentence(chunk_text, sentence), hypothesis),
            float(EXTRACTOR_PINS["thresholds"]["TYPE_ENTAIL"]),
            margin_reason="margin-type",
            reject_reason="type",
        )
        if not ok:
            return False, reason

    if _has_added_proper_noun(chunk_text, fill, str(entry.get("surface", "")), sentence):
        return False, "added-proper-noun"

    return True, "ok"


def _fill_sentence_template(fill: str) -> str:
    return f"The text mentions {fill.strip()}."


def _chunk_sentence(chunk_text: str, sentence: str) -> str:
    sentence = str(sentence).strip()
    return sentence if sentence else str(chunk_text).strip()


def _require_entailment(
    nli_result: tuple[str, float],
    threshold: float,
    *,
    margin_reason: str,
    reject_reason: str,
) -> tuple[bool, str]:
    label, prob = nli_result
    prob = float(prob)
    eps = float(EXTRACTOR_PINS["thresholds"]["EPS_MARGIN"])
    if abs(prob - threshold) <= eps + np.finfo(float).eps * 8:
        return False, margin_reason
    if str(label).lower() != "entailment" or prob < threshold:
        return False, reject_reason
    return True, "ok"


def _has_added_proper_noun(chunk_text: str, fill: str, surface: str, sentence: str) -> bool:
    fill_tokens = _normalized_token_set(fill)
    surface_tokens = _normalized_token_set(surface)
    sentence_text = str(sentence)
    sentence_initial = _sentence_initial_token_start(sentence_text)
    chunk_offset = sentence_text.find(chunk_text) if sentence_text else -1

    for match in _CAPITALIZED_TOKEN_RE.finditer(str(chunk_text)):
        token = match.group(0).lower()
        absolute_start = chunk_offset + match.start() if chunk_offset >= 0 else match.start()
        if sentence_initial is not None and absolute_start == sentence_initial:
            continue
        if token not in fill_tokens and token not in surface_tokens:
            return True
    return False


def _normalized_token_set(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(str(text))}


def _sentence_initial_token_start(sentence: str) -> int | None:
    match = _WORD_RE.search(str(sentence))
    return match.start() if match is not None else None


def _run_tier0(out_p: str, R: list[dict]) -> tuple[str, dict, list[dict]]:
    return _rule_prepass(out_p, R, semantic=True)


def _abstain_entries(residue: list[dict], *, reason: str) -> list[dict]:
    return [
        {
            "surface": entry["surface"],
            "type": entry.get("type", "MISC"),
            "outcome": "abstained",
            "reason": reason,
        }
        for entry in residue
    ]


def _record_unresolved(stats: dict, residue: list[dict], *, reason: str) -> dict:
    stats["gen_absent"] = stats.get("gen_absent", 0) + len(residue)
    stats["entries"] = _abstain_entries(residue, reason=reason)
    stats["extractor_version"] = extractor_version()
    return stats


def extract(
    doc_p: str | None,
    R: list[dict],
    out_p: str,
    *,
    models: dict | None = None,
) -> tuple[str, dict]:
    """Recover original surfaces from `out_p`; fail closed on unresolved tier-0 residue."""
    del doc_p  # Stage 1 consumes this when alignment-prior support lands.
    prepass_text, stats, residue = _run_tier0(out_p, R)
    reason = "no-models" if models is None else "stage-not-implemented"
    stats = _record_unresolved(stats, residue, reason=reason)
    return _finalize(prepass_text, stats)
