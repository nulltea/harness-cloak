"""Build QA-builder v2 assertions from frozen ranker inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from cloak.corpora import load_task_docs
from cloak.train.qa_builder import (
    AciTaskAdapter,
    OpenRouterRelationTeacher,
    artifact_views,
    build_utility_artifact,
    freeze_v2_environment_from_legacy_arms,
    read_context_batch,
    render_frozen_action_vector,
)
from cloak.train.qa_audit import build_environment_audit, write_audit_sidecars


_READER_PIN_FIELDS = frozenset({
    "model", "endpoint", "prompt_version", "response_schema", "revision",
})


def _hash(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_manifest_reader_pin(manifest: Mapping) -> dict:
    reader_pin = manifest.get("reader_pin")
    if not isinstance(reader_pin, Mapping) or not reader_pin:
        raise ValueError("threshold manifest reader_pin must be a structured non-empty mapping")
    missing = sorted(_READER_PIN_FIELDS - set(reader_pin))
    if missing:
        raise ValueError(f"threshold manifest reader_pin is missing fields: {missing}")
    if any(
        reader_pin.get(field) is None
        or reader_pin.get(field) == ""
        or reader_pin.get(field) == {}
        for field in _READER_PIN_FIELDS
    ):
        raise ValueError("threshold manifest reader_pin fields must be non-empty")
    return dict(reader_pin)


def _selected_environment(environment: dict, corpus: str, doc_ids: list[str]) -> dict:
    available = environment.get("corpora", {}).get(corpus)
    if available is None:
        raise SystemExit(f"corpus {corpus!r} is absent from the ranker environment")
    missing = [doc_id for doc_id in doc_ids if doc_id not in available]
    if missing:
        raise SystemExit(f"documents absent from ranker environment: {missing}")
    return {
        **environment,
        "corpora": {corpus: {doc_id: available[doc_id] for doc_id in doc_ids}},
    }


def _source_rows(corpus: str, doc_ids: list[str]) -> dict[str, dict]:
    wanted = set(doc_ids)
    rows = {row["id"]: row for row in load_task_docs(corpus) if row["id"] in wanted}
    missing = [doc_id for doc_id in doc_ids if doc_id not in rows]
    if missing:
        raise SystemExit(f"documents absent from corpus {corpus!r}: {missing}")
    return rows


def _action_renderer(
    frozen_environment: dict,
    source_documents: dict[str, str],
):
    def render(doc_id: str, action_vector: dict[str, str]) -> str:
        frozen_document = frozen_environment["documents"][doc_id]
        return render_frozen_action_vector(
            source_documents[doc_id], frozen_document, action_vector,
        )[0]

    return render


def build_from_files(
    args, *, relation_teacher=None, secondary_relation_teacher=None,
    reader=read_context_batch,
) -> dict:
    manifest = json.loads(Path(args.threshold_manifest).read_text())
    family_budgets = manifest.get("family_budgets")
    if not isinstance(family_budgets, dict) or set(family_budgets) != {"context", "delivered"}:
        raise SystemExit("threshold manifest requires context and delivered family_budgets")
    _require_manifest_reader_pin(manifest)

    environment = json.loads(Path(args.env).read_text())
    arms = json.loads(Path(args.arms).read_text())
    arms_meta = dict(arms.pop("_meta", {}) or {})
    environment_audit = dict(arms_meta.get("environment_audit", {}) or {})
    detector_pin = dict(arms_meta.get("detector", {}) or {})
    rows = _source_rows(args.corpus, args.doc_id)
    if any(not doc_id.startswith("aci/") for doc_id in rows):
        raise SystemExit("the implemented task adapter currently supports ACI documents only")
    source_documents = {doc_id: row["text"] for doc_id, row in rows.items()}
    persisted_frozen = environment.get("frozen_environment")
    if isinstance(persisted_frozen, Mapping):
        documents = persisted_frozen.get("documents")
        if not isinstance(documents, Mapping):
            raise SystemExit("ranker-v2 environment has no frozen documents")
        missing = [doc_id for doc_id in args.doc_id if doc_id not in documents]
        if missing:
            raise SystemExit(f"documents absent from frozen ranker-v2 environment: {missing}")
        frozen_environment = {
            "artifact_version": persisted_frozen.get("artifact_version"),
            "environment_hash": persisted_frozen.get("environment_hash"),
            "documents": {doc_id: documents[doc_id] for doc_id in args.doc_id},
        }
    else:
        environment = _selected_environment(environment, args.corpus, args.doc_id)
        frozen_environment = freeze_v2_environment_from_legacy_arms(
            environment, arms,
            detector_provenance=detector_pin or None,
            source_documents=source_documents,
        )
    environment_audit = build_environment_audit(
        frozen_environment, source_documents=source_documents,
    )
    references = {doc_id: row["gold_ref"] for doc_id, row in rows.items()}
    gleaning_requested = bool(
        getattr(args, "relation_teacher_gleaning", False)
        or os.getenv("CLOAK_RELATION_TEACHER_GLEANING") == "1"
    )
    if relation_teacher is None and (args.relation_teacher or gleaning_requested):
        relation_teacher = OpenRouterRelationTeacher()
    escalation_configured = bool(
        gleaning_requested and manifest.get("relation_escalation_policy") is not None
    )
    # The gleaning+repair pass reuses the primary GPT-OSS teacher config; the second
    # call differs only by prompt (relation_repair_prompt), so it is a distinct cache key.
    if secondary_relation_teacher is None and escalation_configured:
        secondary_relation_teacher = OpenRouterRelationTeacher()
    teacher_pin = None
    if relation_teacher is not None:
        raw_teacher_pin = getattr(relation_teacher, "pin", None)
        if not isinstance(raw_teacher_pin, Mapping) or not raw_teacher_pin:
            raise ValueError("relation teacher requires an explicit non-empty pin")
        teacher_pin = dict(raw_teacher_pin)
    secondary_teacher_pin = None
    if secondary_relation_teacher is not None:
        raw_secondary_pin = getattr(secondary_relation_teacher, "pin", None)
        if not isinstance(raw_secondary_pin, Mapping) or not raw_secondary_pin:
            raise ValueError("secondary relation teacher requires an explicit non-empty pin")
        secondary_teacher_pin = dict(raw_secondary_pin)

    pins = {
        key: value for key, value in manifest.items()
        if key not in {
            "family_budgets",
            "min_context_assertions",
            "min_contextual_relation_assertions",
            "reader_threshold",
        }
    }
    pins.update({
        "gate_manifest_hash": _hash(manifest),
        "task_pin": AciTaskAdapter.task_pin,
        "builder_pin": "qa-builder-v2-assertion-compiler-v11",
        "detector_pin": detector_pin or None,
        "teacher_pin": teacher_pin,
        "environment_audit_hash": environment_audit.get("audit_hash") or None,
        "source_hashes": {doc_id: _hash(text) for doc_id, text in source_documents.items()},
        "reference_hashes": {doc_id: _hash(text) for doc_id, text in references.items()},
    })
    if escalation_configured and secondary_teacher_pin is not None:
        pins.update({
            "relation_teacher_pins": {
                "primary": teacher_pin,
                "secondary": secondary_teacher_pin,
            },
            "relation_escalation_policy": dict(manifest["relation_escalation_policy"]),
        })
    return build_utility_artifact(
        frozen_environment,
        AciTaskAdapter(references),
        source_documents,
        threshold_manifest=manifest,
        pins=pins,
        reader=reader,
        render_action_vector=_action_renderer(
            frozen_environment, source_documents
        ),
        relation_teacher=relation_teacher,
        secondary_relation_teacher=(
            secondary_relation_teacher if escalation_configured else None
        ),
        environment_audit=environment_audit or None,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--corpus", default="clinical")
    parser.add_argument("--doc-id", action="append", required=True)
    parser.add_argument("--threshold-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--relation-teacher",
        action="store_true",
        help="enable the primary cached GPT-OSS relation teacher",
    )
    parser.add_argument(
        "--relation-teacher-gleaning",
        action="store_true",
        help=(
            "with a manifest relation_escalation_policy, conditionally run one "
            "GPT-OSS gleaning+repair pass after the primary GPT-OSS build "
            "(targets ambiguous / fixable-rejected / missed relations)"
        ),
    )
    return parser.parse_args(argv)


def write_artifacts(artifact: dict, output: Path) -> tuple[Path, Path]:
    assertions_view, qa_pairs_view = artifact_views(artifact)
    assertions_output = output.with_name(f"{output.stem}.assertions.json")
    qa_pairs_output = output.with_name(f"{output.stem}.qa-pairs.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=1))
    assertions_output.write_text(json.dumps(assertions_view, indent=1))
    qa_pairs_output.write_text(json.dumps(qa_pairs_view, indent=1))
    qa_audit = artifact.get("qa_audit")
    if isinstance(qa_audit, Mapping):
        write_audit_sidecars(
            qa_audit, output.with_name(f"{output.stem}.qa-audit"),
        )
    return assertions_output, qa_pairs_output


def main(
    argv=None, *, relation_teacher=None, secondary_relation_teacher=None,
    reader=read_context_batch,
):
    args = parse_args(argv)
    artifact = build_from_files(
        args,
        relation_teacher=relation_teacher,
        secondary_relation_teacher=secondary_relation_teacher,
        reader=reader,
    )
    output = Path(args.out)
    assertions_output, qa_pairs_output = write_artifacts(artifact, output)
    qa_audit_base = output.with_name(f"{output.stem}.qa-audit")
    qa_audit_paths = [
        qa_audit_base.with_name(qa_audit_base.name + ".json"),
        qa_audit_base.with_name(qa_audit_base.name + ".jsonl"),
        qa_audit_base.with_name(qa_audit_base.name + ".md"),
    ]
    print(
        f"wrote {output}: docs={len(artifact['documents'])} "
        f"assertions={len(artifact['assertions'])} "
        f"rejections={sum(artifact['rejections']['summary_by_reason'].values())}; "
        f"views={assertions_output},{qa_pairs_output}; "
        f"qa_audit={','.join(map(str, qa_audit_paths))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
