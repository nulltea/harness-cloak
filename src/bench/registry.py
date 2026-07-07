from __future__ import annotations

from itertools import chain, zip_longest
from pathlib import Path
import random
import re

from bench.schema import BenchmarkItem, PrivacyTarget, SensitiveSpan
from cloak.corpora import CORPORA, FILES, load_task_docs, refs_of

SUITES: dict[str, list[str]] = {
    "primary_utility": ["clinical", "lexsum", "wikibio"],
    "clinical_smoke": ["primock57"],
    "email_controls": ["aeslc", "enron"],
    "detector_coverage": ["tab", "pii-bench", "synthetic-financial-pii"],
    "privacy_stress": ["rat-bench"],
}

DOMAIN: dict[str, str] = {
    "aci": "clinical",
    "mts": "clinical",
    "clinical": "clinical",
    "primock57": "clinical",
    "lexsum": "legal",
    "wikibio": "biography",
    "aeslc": "email",
    "enron": "email",
    "tab": "legal",
    "pii-bench": "detector",
    "synthetic-financial-pii": "finance",
    "rat-bench": "privacy",
}

TASK: dict[str, str] = {
    "clinical": "visit_note_generation",
    "aci": "visit_note_generation",
    "mts": "visit_note_generation",
    "primock57": "visit_note_generation",
    "lexsum": "case_summary",
    "wikibio": "biography_summary",
    "aeslc": "email_subject",
    "enron": "email_reply",
}

OPTIONAL_EXTERNAL: dict[str, list[Path]] = {
    "primock57": [CORPORA / "external/primock57/primock57-main.zip"],
    "rat-bench": [CORPORA / "external/rat-bench/benchmark/english/level_1.jsonl"],
    "synthetic-financial-pii": [CORPORA / "external/synthetic-financial-pii/Testing_Set.xlsx"],
    "pii-bench": [CORPORA / "external/pii-bench/data/test.jsonl"],
    "tab": [Path("data/TAB")],
}


def _corpus_available(corpus: str) -> bool:
    if corpus in FILES:
        return all((CORPORA / rel).exists() for rel in FILES[corpus])
    if corpus in OPTIONAL_EXTERNAL:
        return all(path.exists() for path in OPTIONAL_EXTERNAL[corpus])
    return False


def available_suites() -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for suite, corpora in SUITES.items():
        available = [c for c in corpora if _corpus_available(c)]
        missing = [c for c in corpora if c not in available]
        out[suite] = {"available": available, "missing": missing}
    return out


def load_items(suite: str, limit: int | None = None, seed: int = 0) -> list[BenchmarkItem]:
    if suite not in SUITES:
        raise KeyError(f"unknown benchmark suite: {suite}")
    corpora = [c for c in SUITES[suite] if c in FILES and _corpus_available(c)]
    if not corpora:
        return []

    by_corpus = [_rows_for_corpus(c) for c in corpora]
    rows = [row for row in chain.from_iterable(zip_longest(*by_corpus)) if row is not None]
    if seed:
        rng = random.Random(seed)
        # Keep the round-robin domain mix, but make the tail deterministic per seed.
        head, tail = rows[:len(corpora)], rows[len(corpora):]
        rng.shuffle(tail)
        rows = head + tail
    if limit is not None:
        rows = rows[:limit]
    return [_to_item(corpus, idx, row) for idx, (corpus, row) in enumerate(rows)]


def load_detector_items(dataset: str, limit: int | None = None) -> list[BenchmarkItem]:
    if dataset in {"pii-bench", "synthetic-financial-pii", "rat-bench", "tab"}:
        return []
    return load_items(dataset, limit=limit)


def gold_spans_from_text(doc_orig: str, refs: list[str]) -> list[SensitiveSpan]:
    ref_blob = _canon(" ".join(refs))
    candidates: list[tuple[str, int, int, str, str]] = []
    patterns = [
        (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", "PERSON", "DIRECT"),
        (r"\b\d{1,3}(?:-year-old|\s+years?\s+old)\b", "age", "QUASI"),
        (r"\b[A-Z]{2,}-?\d{2,}\b|\b\d{2,}/\d{2,}/\d{2,4}\b", "CODE", "DIRECT"),
        (r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}\b", "DATETIME", "QUASI"),
    ]
    for pattern, typ, ident in patterns:
        for match in re.finditer(pattern, doc_orig):
            candidates.append((match.group(), match.start(), match.end(), typ, ident))

    spans: list[SensitiveSpan] = []
    seen: set[tuple[int, int]] = set()
    for idx, (surface, start, end, typ, ident) in enumerate(sorted(candidates, key=lambda x: x[1])):
        if (start, end) in seen:
            continue
        seen.add((start, end))
        task_relevance = "gold_restated" if _restated(surface, ref_blob) else "unlabeled"
        evidence = [r for r in refs if _restated(surface, _canon(r))][:2]
        spans.append(
            SensitiveSpan(
                span_id=f"s{idx}",
                surface=surface,
                start=start,
                end=end,
                type=typ,
                identifier_class=ident,
                subject_id="document_subject",
                task_relevance=task_relevance,
                reference_evidence=evidence,
            )
        )
    return spans


def _rows_for_corpus(corpus: str) -> list[tuple[str, dict]]:
    return [(corpus, row) for row in load_task_docs(corpus)]


def _to_item(corpus: str, index: int, row: dict) -> BenchmarkItem:
    refs = refs_of(row)
    row_id = row.get("id") or row.get("doc_id") or f"{index:06d}"
    return BenchmarkItem(
        item_id=f"{corpus}/{row_id}",
        domain=DOMAIN.get(corpus, corpus),
        task=TASK.get(corpus, corpus),
        corpus=corpus,
        doc_orig=row["text"],
        task_prompt_template=corpus,
        reference_outputs=refs,
        gold_sensitive_spans=gold_spans_from_text(row["text"], refs),
        privacy_targets=[
            PrivacyTarget(
                target_id="document_subject",
                known_to_attacker="document_context_only",
                secret_attributes=["direct_identifiers", "quasi_identifiers"],
            )
        ],
    )


def _canon(text: str) -> str:
    out = text.lower()
    out = out.replace("-year-old", " years old")
    out = re.sub(r"[^a-z0-9]+", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def _restated(surface: str, canon_ref: str) -> bool:
    surface_key = _canon(surface)
    return bool(surface_key and surface_key in canon_ref)
