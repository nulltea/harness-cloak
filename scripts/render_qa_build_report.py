"""Render one QA-builder artifact as a readable Markdown inspection report.

The artifact intentionally contains hashes instead of source text and does not contain a remote
generation. Pair it with the arms artifact/corpus and an explicit recorded out_p file.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from cloak.corpora import load_task_docs
from cloak.train.qa_builder import relation_teacher_span_inventory


def _text_section(title: str, text: str) -> str:
    return f"## {title}\n\n```text\n{text.rstrip()}\n```\n"


def _json(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _source_row(corpus: str, doc_id: str) -> Mapping:
    for row in load_task_docs(corpus):
        if str(row.get("id")) == doc_id:
            return row
    raise ValueError(f"document {doc_id!r} is not present in corpus {corpus!r}")


def _detected_surfaces(document: Mapping) -> str:
    grouped: dict[str, list[str]] = defaultdict(list)
    for occurrence in document.get("occurrences", []):
        grouped[str(occurrence.get("runtime_type", "unknown"))].append(
            str(occurrence.get("surface", ""))
        )
    lines = ["## Detected surfaces by runtime type", ""]
    if not grouped:
        return "\n".join(lines + ["- None"]) + "\n"
    for runtime_type in sorted(grouped):
        counts = Counter(grouped[runtime_type])
        lines.extend([f"### {runtime_type} ({sum(counts.values())})", ""])
        lines.extend(
            f"- `{surface}` × {count}" for surface, count in sorted(counts.items())
        )
        lines.append("")
    return "\n".join(lines)


def _structural_schemas(assertions: list[Mapping]) -> str:
    rows = [row for row in assertions if row.get("subtype") == "structure"]
    lines = ["## Generated deterministic artifacts", "", "### Structural schemas (deterministic)", ""]
    if not rows:
        lines.append("- None")
    for row in rows:
        lines.extend(["```json", _json(row.get("scoring_contract", {})), "```"])
    return "\n".join(lines) + "\n"


def _source_region(source_text: str, span) -> list[str]:
    """Render the doc_orig region a relation was grounded in, when resolved."""
    if (
        not isinstance(span, (list, tuple)) or len(span) != 2
        or not all(isinstance(value, int) for value in span)
        or not 0 <= span[0] < span[1] <= len(source_text)
    ):
        return ["- Source region (doc_orig): not resolved"]
    return [
        f"- Source region (doc_orig, {span[0]}..{span[1]}):", "", "```text",
        source_text[span[0]:span[1]].strip(), "```",
    ]


def _relations(
    artifact: Mapping, doc_id: str, assertions: list[Mapping], source_text: str,
) -> str:
    rejected = [
        row for row in artifact.get("rejections", {}).get("records", [])
        if row.get("doc_id") == doc_id
        and row.get("evidence", {}).get("source") == "relation_teacher"
    ]
    rejected_spans = {
        row.get("proposal_index"): row.get("evidence", {}).get("evidence_span")
        for row in rejected
    }
    kept_rows = [row for row in assertions if row.get("subtype") == "contextual_relation"]

    # A kept relation's stored assertion carries the FINAL question/answers (after
    # any leakage recolor); the teacher attempt carries the pre-recolor question.
    # Match on (relation, answers) — stable across a subject-side recolor — so the
    # report shows what the artifact actually holds, not the raw teacher question.
    def _literals(arguments):
        return tuple(sorted(
            str(a.get("literal")) for a in arguments or []
            if isinstance(a, Mapping) and a.get("literal")))
    def _final_key(relation, answers, arguments):
        # relation + subject/object literals + answers distinguishes near-duplicate
        # relations (e.g. two contraindications with the same answer but different
        # drug-class literals) while surviving a subject-level recolor.
        return (relation, _literals(arguments),
                tuple(sorted(str(v) for v in answers or [])))
    kept_final = {}
    for row in kept_rows:
        kept_final.setdefault(
            _final_key(row.get("relation"), row.get("accepted_values"),
                       row.get("evidence", {}).get("arguments")), row)

    def kept_span(question) -> list | None:
        for row in kept_rows:
            if row.get("question") == question:
                span = row.get("evidence", {}).get("source_span", {})
                return [span.get("start"), span.get("end")]
        return None

    label_spans = {
        row["span_label"]: (row["start"], row["end"])
        for row in relation_teacher_span_inventory(artifact.get("documents", {}).get(doc_id, {}))
    }

    def argument_region(arguments) -> list | None:
        """Sentence-expanded region around the linked S-labels, when no exact
        evidence span survived (e.g. reader-gate rejections after compilation)."""
        spans = [
            label_spans[argument["span_label"]]
            for argument in arguments or []
            if isinstance(argument, Mapping) and argument.get("span_label") in label_spans
        ]
        if not spans:
            return None
        left = min(start for start, _ in spans)
        right = max(end for _, end in spans)
        left = max(source_text.rfind("\n", 0, left), 0)
        next_break = source_text.find("\n", right)
        right = next_break if next_break >= 0 else len(source_text)
        return [left, right]

    attempts = artifact.get("relation_generation", {}).get(doc_id, [])
    if attempts:
        lines = ["## Generated relations and semantic QA", ""]
        for attempt in attempts:
            final = (kept_final.get(_final_key(attempt.get("relation"),
                                               attempt.get("accepted_answers"),
                                               attempt.get("arguments")))
                     if attempt.get("status") == "kept" else None)
            question = final.get("question") if final else attempt.get("question")
            answers = final.get("accepted_values") if final else attempt.get("accepted_answers", [])
            repair = (final or {}).get("evidence", {}).get("leakage_repair") if final else None
            span = (
                rejected_spans.get(attempt.get("proposal_index"))
                if attempt.get("status") == "rejected"
                else kept_span(question)
            ) or argument_region(attempt.get("arguments"))
            recolor = (f"  (recolored: {repair.get('from_level')} -> {repair.get('to_level')})"
                       if repair else "")
            lines.extend([
                f"### {attempt.get('relation') or 'invalid relation'}",
                "",
                f"- Status: {attempt.get('status', 'unknown')}",
                f"- Reason: {attempt.get('reason', 'unknown')}",
                f"- Question: {question}{recolor}",
                f"- Accepted answers: {_json(answers)}",
                *_source_region(source_text, span),
                "- Scoring contract:", "", "```json",
                _json(attempt.get("scoring_contract", {})), "```",
                "- Arguments:", "", "```json", _json(attempt.get("arguments", [])), "```", "",
            ])
        return "\n".join(lines)
    lines = ["## Generated relations and semantic QA", ""]
    if not kept_rows:
        lines.append("- No relation was kept.")
    for row in kept_rows:
        span = row.get("evidence", {}).get("source_span", {})
        lines.extend([
            f"### {row.get('relation')}", "",
            "- Status: kept",
            "- Reason: accepted",
            f"- Question: {row.get('question')}",
            f"- Accepted answers: {_json(row.get('accepted_values', []))}",
            *_source_region(source_text, [span.get("start"), span.get("end")]),
            "- Scoring contract:", "", "```json", _json(row.get("scoring_contract", {})), "```",
            "- Arguments:", "", "```json", _json(row.get("evidence", {}).get("arguments", [])), "```", "",
        ])
    if rejected:
        lines.extend(["### Rejected relation attempts", ""])
        for row in rejected:
            lines.append(f"- Status: rejected; reason: `{row.get('detail_reason')}`")
            lines.extend(_source_region(source_text, row.get("evidence", {}).get("evidence_span")))
    return "\n".join(lines)


def _candidate_accounting(artifact: Mapping, doc_id: str) -> str:
    rows = artifact.get("relation_candidate_accounting", {}).get(doc_id, [])
    lines = ["### Candidate accounting", ""]
    if not rows:
        lines.append("- Not present (legacy/no teacher relation build).")
    else:
        lines.extend(
            f"- `{row.get('candidate_label', row.get('candidate_id'))}` — "
            f"`{row.get('state')}`: {row.get('reason')}"
            for row in rows
        )
    return "\n".join(lines) + "\n"


def render_report(
    artifact_path: str | Path,
    arms_path: str | Path,
    *,
    corpus: str,
    doc_id: str,
    out_p_path: str | Path | None = None,
    out_hi_path: str | Path | None = None,
    out_lo_p_path: str | Path | None = None,
) -> str:
    """Return a Markdown inspection report; performs no model calls."""
    artifact = json.loads(Path(artifact_path).read_text())
    arms = json.loads(Path(arms_path).read_text())
    document = artifact.get("documents", {}).get(doc_id)
    if document is None:
        raise ValueError(f"document {doc_id!r} is not present in the QA artifact")
    out_lo_p = (
        Path(out_lo_p_path).read_text() if out_lo_p_path is not None else "Not supplied."
    )
    source = _source_row(corpus, doc_id)
    out_hi = (
        Path(out_hi_path).read_text()
        if out_hi_path is not None
        else source.get("gold_ref", source.get("gold", ""))
    )
    out_hi_label = "remote high-privacy output" if out_hi_path is not None else "reference output"
    out_p = Path(out_p_path).read_text() if out_p_path is not None else "Not supplied."
    assertions = [
        row for row in artifact.get("assertions", {}).values()
        if row.get("doc_id") == doc_id
    ]
    return "\n".join([
        f"# QA build report — {doc_id}", "",
        _text_section("doc_orig", str(source.get("text", ""))),
        _text_section(f"out_hi ({out_hi_label})", str(out_hi)),
        _text_section("out_lo_p (remote output on the all_floor doc_p)", str(out_lo_p)),
        _text_section("out_p (remote model output)", out_p),
        _detected_surfaces(document),
        _structural_schemas(assertions),
        _relations(artifact, doc_id, assertions, str(source.get("text", ""))),
        _candidate_accounting(artifact, doc_id),
    ])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="QA utility artifact JSON")
    parser.add_argument("--arms", required=True, help="arms artifact with tau_walk doc_p")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--out-p", help="optional recorded remote output for the shown doc_p")
    parser.add_argument("--out-hi", help="optional recorded high-privacy remote output")
    parser.add_argument(
        "--out-lo-p",
        help="optional recorded remote output for the all_floor (all-placeholder) doc_p",
    )
    parser.add_argument("--out", required=True, help="Markdown report output path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = render_report(
        args.artifact, args.arms, corpus=args.corpus, doc_id=args.doc_id,
        out_p_path=args.out_p, out_hi_path=args.out_hi, out_lo_p_path=args.out_lo_p,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
