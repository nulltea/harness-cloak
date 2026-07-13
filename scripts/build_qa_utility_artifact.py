"""Build QA-builder v2 assertions from frozen ranker inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cloak.corpora import load_task_docs
from cloak.train.qa_builder import (
    AciTaskAdapter,
    OpenRouterRelationTeacher,
    build_utility_artifact,
    context_reader_pin,
    frozen_occurrences_from_arms,
    freeze_ranker_environment,
    normalize_cost_budgets,
    read_context_batch,
)
from cloak.train.reward import canon
from train_ranker import assemble, qa_utility_preflight_report


def _hash(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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
    environment: dict,
    frozen_environment: dict,
    arms: dict,
    corpus: str,
    source_documents: dict[str, str],
):
    raw_documents = environment["corpora"][corpus]

    def render(doc_id: str, action_vector: dict[str, str]) -> str:
        raw_document = raw_documents[doc_id]
        frozen_document = frozen_environment["documents"][doc_id]
        decisions_by_key = {
            (str(row["runtime_type"]), str(row["canonical_key"])): row
            for row in frozen_document["decisions"]
        }
        choice = {}
        for span in raw_document["spans"]:
            key = (str(span.get("type", "")), canon(str(span.get("surface", ""))))
            decision = decisions_by_key[key]
            selected_id = action_vector[str(decision["decision_id"])]
            selected = next(
                action for action in decision["actions"]
                if str(action["action_id"]) == selected_id
            )
            choice[str(span["surface"]).lower()] = {
                "mode": "level" if selected["mode"] in {"level", "keep"} else "placeholder",
                "fill": selected.get("fill"),
            }
        return assemble(
            source_documents[doc_id],
            arms[corpus][doc_id]["tau_walk"][1],
            raw_document["spans"],
            choice,
        )[0]

    return render


def build_from_files(args, *, relation_teacher=None, reader=read_context_batch) -> dict:
    environment = json.loads(Path(args.env).read_text())
    environment = _selected_environment(environment, args.corpus, args.doc_id)
    arms = json.loads(Path(args.arms).read_text())
    arms.pop("_meta", None)
    all_occurrence_records = frozen_occurrences_from_arms(arms)
    occurrence_records = {
        doc_id: all_occurrence_records[doc_id] for doc_id in args.doc_id
    }
    frozen_environment = freeze_ranker_environment(
        environment, occurrences_by_document=occurrence_records
    )
    manifest = json.loads(Path(args.threshold_manifest).read_text())
    family_budgets = manifest.get("family_budgets")
    if not isinstance(family_budgets, dict) or set(family_budgets) != {"context", "delivered"}:
        raise SystemExit("threshold manifest requires context and delivered family_budgets")
    try:
        manifest["cost_budgets"] = normalize_cost_budgets(manifest.get("cost_budgets"))
    except ValueError as error:
        raise SystemExit(f"threshold manifest requires frozen cost budgets: {error}") from None

    rows = _source_rows(args.corpus, args.doc_id)
    if any(not doc_id.startswith("aci/") for doc_id in rows):
        raise SystemExit("the implemented task adapter currently supports ACI documents only")
    source_documents = {doc_id: row["text"] for doc_id, row in rows.items()}
    references = {doc_id: row["gold_ref"] for doc_id, row in rows.items()}
    if relation_teacher is None and args.relation_teacher:
        relation_teacher = OpenRouterRelationTeacher()

    pins = {
        key: value for key, value in manifest.items()
        if key not in {"family_budgets", "min_context_assertions", "reader_threshold"}
    }
    pins.update({
        "gate_manifest_hash": _hash(manifest),
        "task_pin": AciTaskAdapter.task_pin,
        "builder_pin": "qa-builder-v2-assertion-compiler-v1",
        "reader_pin": context_reader_pin(),
        "teacher_pin": ({
            "provider": "openrouter",
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
        } if args.relation_teacher else None),
        "source_hashes": {doc_id: _hash(text) for doc_id, text in source_documents.items()},
        "reference_hashes": {doc_id: _hash(text) for doc_id, text in references.items()},
    })
    return build_utility_artifact(
        frozen_environment,
        AciTaskAdapter(references),
        source_documents,
        threshold_manifest=manifest,
        pins=pins,
        reader=reader,
        render_action_vector=_action_renderer(
            environment, frozen_environment, arms, args.corpus, source_documents
        ),
        relation_teacher=relation_teacher,
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
        help="allow one cached Nemotron/OpenRouter proposal call per under-supported document",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    artifact = build_from_files(args)
    report = qa_utility_preflight_report(
        artifact,
        {"environment_hash": artifact["environment_hash"]},
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=1))
    print(
        f"wrote {output}: docs={len(artifact['documents'])} "
        f"assertions={len(artifact['assertions'])} "
        f"rejections={sum(artifact['rejections']['summary_by_reason'].values())}",
        flush=True,
    )
    print(
        "qa preflight: " + json.dumps(report, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


if __name__ == "__main__":
    main()
