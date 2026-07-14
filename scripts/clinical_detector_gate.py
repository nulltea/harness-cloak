"""Gate attributable QA-v2 clinical detector outcomes on a local corpus slice."""

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

from build_arms_artifact import (
    QA_V2_CLINICAL_LABEL_SCHEMA,
    QA_V2_CLINICAL_MODEL,
    QA_V2_CLINICAL_THRESHOLD,
    QA_V2_CONTROLLED_TYPES,
)
from cloak.corpora import load_task_docs
from cloak.detect import Detector, QA_V2_CLINICAL_LABELS
from cloak.substitute import prepare_spans_for_substitution


THRESHOLDS = {
    "split_contraction_person": 0,
    "clinical_code_without_identifier_shape": 0,
    "explicit_name_not_person": 0,
    "frozen_demographic_other": 0,
}
EXPLICIT_NAME_LABELS = frozenset({"name", "first name", "last name"})
NATIVE_CLINICAL_CODE_LABELS = frozenset({"medical code", "healthcare number"})
FAMILY_NAMES = (
    "accepted",
    "rejected",
    "overlap_loser",
    "normalizations",
    "post_detection_rejected",
)
DETECTOR_PIN = {
    "config": "qa-v2-clinical",
    "model": QA_V2_CLINICAL_MODEL,
    "threshold": QA_V2_CLINICAL_THRESHOLD,
    "profile": "clinical",
    "label_schema": QA_V2_CLINICAL_LABEL_SCHEMA,
    "label_map": dict(QA_V2_CLINICAL_LABELS),
    "controlled_runtime_types": sorted(QA_V2_CONTROLLED_TYPES),
    "presidio": True,
}


def _overlaps(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start < other_end and other_start < end


def _source_hash(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def evaluate_results(rows: list[dict]) -> dict:
    """Evaluate already-attributed document rows without loading detector models."""
    counts = {name: 0 for name in THRESHOLDS}
    families = {name: [] for name in FAMILY_NAMES}
    documents = []

    for row in rows:
        doc_id = row["doc_id"]
        text = row["text"]
        detection = row["detection"]
        prepared_spans = row["prepared_spans"]
        post_rejections = row["post_detection_rejections"]
        documents.append({
            "doc_id": doc_id,
            "source_sha256": _source_hash(text),
        })

        for candidate in detection.candidates:
            status = candidate.get("status")
            family = status if status in {"accepted", "rejected", "overlap_loser"} else None
            if family is not None:
                families[family].append({"doc_id": doc_id, **dict(candidate)})
        for event in detection.normalizations:
            families["normalizations"].append({"doc_id": doc_id, **asdict(event)})
        for rejection in post_rejections:
            families["post_detection_rejected"].append({
                "doc_id": doc_id,
                **dict(rejection),
            })

        for span in detection.spans:
            if span.type == "PERSON" and any(
                _overlaps(span.start, span.end, event.start, event.end)
                for event in detection.normalizations
            ):
                counts["split_contraction_person"] += 1
            if (
                span.raw_label in NATIVE_CLINICAL_CODE_LABELS
                and span.type == "CODE"
                and not re.search(r"[0-9]", span.text)
            ):
                counts["clinical_code_without_identifier_shape"] += 1

        counts["explicit_name_not_person"] += sum(
            span.raw_label in EXPLICIT_NAME_LABELS and span.type != "PERSON"
            for span in prepared_spans
        )
        counts["frozen_demographic_other"] += sum(
            span.type == "demographic-other" for span in prepared_spans
        )

    return {
        "gate_pass": all(counts[name] <= threshold for name, threshold in THRESHOLDS.items()),
        "counts": counts,
        "thresholds": dict(THRESHOLDS),
        "families": families,
        "family_counts": {name: len(values) for name, values in families.items()},
        "documents": documents,
        "document_count": len(documents),
        "detector": dict(DETECTOR_PIN),
    }


def evaluate_documents(rows: list[dict], detector) -> dict:
    """Detect a selected corpus slice in one batch, then apply the QA-v2 runtime contract."""
    detections = detector.detect_many_with_diagnostics([row["text"] for row in rows])
    if len(detections) != len(rows):
        raise ValueError(
            f"detector returned {len(detections)} results for {len(rows)} documents"
        )
    evaluated_rows = []
    for row, detection in zip(rows, detections, strict=True):
        prepared_spans, post_rejections = prepare_spans_for_substitution(
            row["text"], detection.spans, reject_demographic_other=True
        )
        evaluated_rows.append({
            "doc_id": row["id"],
            "text": row["text"],
            "detection": detection,
            "prepared_spans": prepared_spans,
            "post_detection_rejections": post_rejections,
        })
    return evaluate_results(evaluated_rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=("aci", "clinical", "mts"), default="aci")
    parser.add_argument("--doc-id", help="optional exact source document id")
    parser.add_argument("--limit", type=int, help="maximum selected documents")
    parser.add_argument("--out", required=True, help="JSON report path")
    return parser.parse_args(argv)


def _selected_documents(args) -> list[dict]:
    rows = load_task_docs(args.corpus)
    if args.doc_id is not None:
        rows = [row for row in rows if row["id"] == args.doc_id]
        if not rows:
            raise ValueError(f"document not found in {args.corpus}: {args.doc_id}")
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError(f"no documents selected from corpus {args.corpus}")
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = _selected_documents(args)
    started = time.perf_counter()
    detector = Detector(
        gliner_model=QA_V2_CLINICAL_MODEL,
        threshold=QA_V2_CLINICAL_THRESHOLD,
        profile="clinical",
        label2type=QA_V2_CLINICAL_LABELS,
    )
    report = evaluate_documents(rows, detector)
    report["corpus"] = args.corpus
    report["wall_s"] = round(time.perf_counter() - started, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "gate_pass": report["gate_pass"],
        "counts": report["counts"],
        "family_counts": report["family_counts"],
        "document_count": report["document_count"],
        "wall_s": report["wall_s"],
        "out": str(out),
    }, indent=2))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
