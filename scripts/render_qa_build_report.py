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


def _relations(artifact: Mapping, doc_id: str, assertions: list[Mapping]) -> str:
    attempts = artifact.get("relation_generation", {}).get(doc_id, [])
    if attempts:
        lines = ["## Generated relations and semantic QA", ""]
        for attempt in attempts:
            lines.extend([
                f"### {attempt.get('relation') or 'invalid relation'}",
                "",
                f"- Status: {attempt.get('status', 'unknown')}",
                f"- Reason: {attempt.get('reason', 'unknown')}",
                f"- Question: {attempt.get('question')}",
                f"- Accepted answers: {_json(attempt.get('accepted_answers', []))}",
                "- Scoring contract:", "", "```json",
                _json(attempt.get("scoring_contract", {})), "```",
                "- Arguments:", "", "```json", _json(attempt.get("arguments", [])), "```", "",
            ])
        return "\n".join(lines)
    rows = [row for row in assertions if row.get("subtype") == "contextual_relation"]
    rejected = [
        row for row in artifact.get("rejections", {}).get("records", [])
        if row.get("doc_id") == doc_id
        and row.get("evidence", {}).get("source") == "relation_teacher"
    ]
    lines = ["## Generated relations and semantic QA", ""]
    if not rows:
        lines.append("- No relation was kept.")
    for row in rows:
        lines.extend([
            f"### {row.get('relation')}", "",
            "- Status: kept",
            "- Reason: accepted",
            f"- Question: {row.get('question')}",
            f"- Accepted answers: {_json(row.get('accepted_values', []))}",
            "- Scoring contract:", "", "```json", _json(row.get("scoring_contract", {})), "```",
            "- Arguments:", "", "```json", _json(row.get("evidence", {}).get("arguments", [])), "```", "",
        ])
    if rejected:
        lines.extend(["### Rejected relation attempts", ""])
        lines.extend(
            f"- Status: rejected; reason: `{row.get('detail_reason')}`"
            for row in rejected
        )
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
) -> str:
    """Return a Markdown inspection report; performs no model calls."""
    artifact = json.loads(Path(artifact_path).read_text())
    arms = json.loads(Path(arms_path).read_text())
    document = artifact.get("documents", {}).get(doc_id)
    if document is None:
        raise ValueError(f"document {doc_id!r} is not present in the QA artifact")
    try:
        doc_p = arms[corpus][doc_id]["tau_walk"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"tau_walk doc_p is missing for {corpus}/{doc_id}") from error
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
        _text_section("doc_p (tau_walk)", str(doc_p)),
        _text_section("out_p (remote model output)", out_p),
        _detected_surfaces(document),
        _structural_schemas(assertions),
        _relations(artifact, doc_id, assertions),
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
    parser.add_argument("--out", required=True, help="Markdown report output path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = render_report(
        args.artifact, args.arms, corpus=args.corpus, doc_id=args.doc_id,
        out_p_path=args.out_p, out_hi_path=args.out_hi,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
