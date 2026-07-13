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
    builder_pin,
    frozen_occurrences_from_arms,
    freeze_ranker_environment,
    normalize_threshold_manifest,
    read_context_batch,
    reader_dependency_pin,
    relation_teacher_pin,
    teacher_dependency_pin,
    utility_scorer_pin,
)
from cloak.train.reward import canon
from train_ranker import assemble, qa_utility_preflight_report


def _hash(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class ArtifactBuildCache:
    """Fail-closed standard-library cache for complete utility artifacts."""

    _VERSION = 1

    def __init__(self, directory):
        self.directory = Path(directory)

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key.removeprefix('sha256:')}.json"

    def load(self, key: str):
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text())
            if (
                not isinstance(record, dict)
                or record.get("version") != self._VERSION
                or record.get("key") != key
                or not isinstance(record.get("artifact"), dict)
            ):
                raise ValueError("schema or identity mismatch")
            artifact = record["artifact"]
            if not artifact.get("artifact_hash") or any(
                state.get("measurement_state") == "build_failed"
                for state in artifact.get("documents", {}).values()
            ):
                raise ValueError("incomplete artifact")
            return artifact
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise SystemExit(f"artifact build cache {path} is invalid: {error}") from None

    def store(self, key: str, artifact: dict) -> None:
        if not artifact.get("artifact_hash") or any(
            state.get("measurement_state") == "build_failed"
            for state in artifact.get("documents", {}).values()
        ):
            raise SystemExit("artifact build cache only accepts complete artifacts")
        existing = self.load(key)
        if existing is not None:
            if existing != artifact:
                raise SystemExit("artifact build cache has a conflicting entry")
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path_for(key).write_text(json.dumps(
            {"version": self._VERSION, "key": key, "artifact": artifact},
            sort_keys=True, separators=(",", ":"),
        ))


def require_build_cache_for_teacher(args) -> None:
    if args.relation_teacher and not args.build_cache:
        raise SystemExit("relation teacher escalation requires --build-cache")


def _build_cache_key(
    frozen_environment: dict,
    source_documents: dict[str, str],
    references: dict[str, str],
    manifest: dict,
    *,
    reader,
    teacher_pin: dict,
) -> str:
    return _hash({
        "environment_hash": frozen_environment["environment_hash"],
        "source_hashes": {doc_id: _hash(text) for doc_id, text in source_documents.items()},
        "reference_hashes": {doc_id: _hash(text) for doc_id, text in references.items()},
        "adapter": AciTaskAdapter.task_pin,
        "builder_pin": builder_pin(),
        "teacher_pin": teacher_pin,
        "reader_pin": reader_dependency_pin(reader),
        "scorer_pin": utility_scorer_pin(),
        "threshold_manifest": manifest,
    })


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


def _parse_floors(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    try:
        return {
            runtime_type: float(floor)
            for runtime_type, floor in (entry.split("=", 1) for entry in value.split(","))
        }
    except ValueError:
        raise SystemExit("--floors must be comma-separated TYPE=COUNT entries") from None


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


def build_from_files(
    args,
    *,
    relation_teacher=None,
    reader=read_context_batch,
    return_frozen_environment=False,
) -> dict | tuple[dict, dict]:
    require_build_cache_for_teacher(args)
    environment = json.loads(Path(args.env).read_text())
    environment = _selected_environment(environment, args.corpus, args.doc_id)
    arms = json.loads(Path(args.arms).read_text())
    arms.pop("_meta", None)
    all_occurrence_records = frozen_occurrences_from_arms(arms)
    occurrence_records = {
        doc_id: all_occurrence_records[doc_id] for doc_id in args.doc_id
    }
    try:
        manifest = normalize_threshold_manifest(
            json.loads(Path(args.threshold_manifest).read_text())
        )
    except ValueError as error:
        raise SystemExit(f"threshold manifest is invalid: {error}") from None
    rows = _source_rows(args.corpus, args.doc_id)
    if any(not doc_id.startswith("aci/") for doc_id in rows):
        raise SystemExit("the implemented task adapter currently supports ACI documents only")
    source_documents = {doc_id: row["text"] for doc_id, row in rows.items()}
    references = {doc_id: row["gold_ref"] for doc_id, row in rows.items()}
    frozen_environment = freeze_ranker_environment(
        environment,
        occurrences_by_document=occurrence_records,
        floors=_parse_floors(args.floors),
        source_documents=source_documents,
        authoritative_references=references,
    )
    cache = ArtifactBuildCache(args.build_cache) if args.build_cache else None
    cache_key = _build_cache_key(
        frozen_environment, source_documents, references, manifest,
        reader=reader,
        teacher_pin=(teacher_dependency_pin(relation_teacher)
                     if relation_teacher is not None else relation_teacher_pin(args.relation_teacher)),
    )
    if cache is not None:
        cached = cache.load(cache_key)
        if cached is not None:
            return (cached, frozen_environment) if return_frozen_environment else cached
    if relation_teacher is None and args.relation_teacher:
        relation_teacher = OpenRouterRelationTeacher()

    artifact = build_utility_artifact(
        frozen_environment,
        AciTaskAdapter(references),
        source_documents,
        threshold_manifest=manifest,
        pins={},
        reader=reader,
        render_action_vector=_action_renderer(
            environment, frozen_environment, arms, args.corpus, source_documents
        ),
        relation_teacher=relation_teacher,
    )
    if cache is not None:
        cache.store(cache_key, artifact)
    if return_frozen_environment:
        return artifact, frozen_environment
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--corpus", default="clinical")
    parser.add_argument("--doc-id", action="append", required=True)
    parser.add_argument("--threshold-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--floors",
        default=None,
        help=(
            "override per-type count floors as comma-separated TYPE=COUNT entries; "
            "the effective floors are frozen into environment identity"
        ),
    )
    parser.add_argument(
        "--build-cache",
        default=None,
        help="directory for complete content-addressed utility-artifact builds",
    )
    parser.add_argument(
        "--relation-teacher",
        action="store_true",
        help="allow one cached Nemotron/OpenRouter proposal call per under-supported document",
    )
    return parser.parse_args(argv)


def main(argv=None, *, relation_teacher=None, reader=read_context_batch):
    args = parse_args(argv)
    artifact, frozen_environment = build_from_files(
        args,
        relation_teacher=relation_teacher,
        reader=reader,
        return_frozen_environment=True,
    )
    report = qa_utility_preflight_report(artifact, frozen_environment)
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
