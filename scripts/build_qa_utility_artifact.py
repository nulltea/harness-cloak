"""Build QA-builder v2 assertions from frozen ranker inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from cloak.corpora import load_task_docs
from cloak.train.qa_builder import (
    AciTaskAdapter,
    OpenRouterRelationTeacher,
    artifact_views,
    build_utility_artifact,
    frozen_occurrences_from_arms,
    freeze_ranker_environment,
    read_context_batch,
)
from cloak.train.reward import canon
from train_ranker import assemble


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
        keep_markers = {}
        walk_rows = arms[corpus][doc_id]["tau_walk"][1]
        generalized_fill_surface: dict[str, str] = {}
        for span in raw_document["spans"]:
            key = (str(span.get("type", "")), canon(str(span.get("surface", ""))))
            decision = decisions_by_key[key]
            selected_id = action_vector[str(decision["decision_id"])]
            selected = next(
                action for action in decision["actions"]
                if str(action["action_id"]) == selected_id
            )
            fill = selected.get("fill")
            if selected["mode"] == "keep":
                marker = (
                    "__QA_KEEP_"
                    + str(selected["action_id"]).removeprefix("sha256:")[:16]
                    + "__"
                )
                if marker in source_documents[doc_id]:
                    raise ValueError(f"KEEP render marker collision for {selected_id}")
                keep_markers.setdefault(marker, [
                    str(row["surface"])
                    for row in sorted(walk_rows, key=lambda row: int(row["start"]))
                    if row.get("lattice")
                    and str(row.get("type", "")) == str(decision["runtime_type"])
                    and canon(str(row.get("surface", ""))) == str(decision["canonical_key"])
                ])
                fill = marker
            else:
                # A generalized/placeholder identity: record one raw-span surface so its
                # OTHER occurrences (gate-dropped / text-anchored repeats) get the same
                # fill applied below. KEEP decisions are excluded -- their markers are
                # count-matched to walk rows, and KEEP preserves identity anyway.
                generalized_fill_surface.setdefault(
                    str(decision["decision_id"]), str(span["surface"]))
            choice[str(span["surface"]).lower()] = {
                "mode": "level" if selected["mode"] in {"level", "keep"} else "placeholder",
                "fill": fill,
            }
        # P1: generalize EVERY occurrence of a generalized decision, not just the
        # detector-admitted walk rows. A repeat mention the detector dropped locally
        # (below the per-type gate) or that text-anchoring recovered would otherwise
        # survive verbatim in doc_p -- a real identity leak. Universal: any repeated
        # identity, any doc. Positions already in the walk are left to the walk.
        walk_boxes = [(int(row["start"]), int(row["end"])) for row in walk_rows]
        augmented_walk = list(walk_rows)
        for occurrence in frozen_document.get("occurrences", []):
            decision_id = occurrence.get("decision_id")
            fill_surface = (generalized_fill_surface.get(str(decision_id))
                            if decision_id is not None else None)
            if fill_surface is None:
                continue
            start, end = int(occurrence["start"]), int(occurrence["end"])
            if any(start < box_end and box_start < end for box_start, box_end in walk_boxes):
                continue  # already covered by a detector walk row
            augmented_walk.append({
                "surface": fill_surface, "type": str(occurrence.get("runtime_type", "")),
                "start": start, "end": end, "lattice": True,
                "replacement": str(occurrence.get("surface", "")), "action": "generalize",
            })
        rendered = assemble(
            source_documents[doc_id],
            augmented_walk,
            raw_document["spans"],
            choice,
        )[0]
        for marker, surfaces in keep_markers.items():
            for surface in surfaces:
                if marker not in rendered:
                    raise ValueError(f"missing KEEP render marker {marker}")
                rendered = rendered.replace(marker, surface, 1)
            if marker in rendered:
                raise ValueError(f"unresolved KEEP render marker {marker}")
        return rendered

    return render


def build_from_files(args, *, relation_teacher=None, reader=read_context_batch) -> dict:
    manifest = json.loads(Path(args.threshold_manifest).read_text())
    family_budgets = manifest.get("family_budgets")
    if not isinstance(family_budgets, dict) or set(family_budgets) != {"context", "delivered"}:
        raise SystemExit("threshold manifest requires context and delivered family_budgets")
    _require_manifest_reader_pin(manifest)

    environment = json.loads(Path(args.env).read_text())
    environment = _selected_environment(environment, args.corpus, args.doc_id)
    arms = json.loads(Path(args.arms).read_text())
    arms_meta = dict(arms.pop("_meta", {}) or {})
    detector_pin = dict(arms_meta.get("detector", {}) or {})
    all_occurrence_records = frozen_occurrences_from_arms(
        arms, detector_provenance=detector_pin or None
    )
    occurrence_records = {
        doc_id: all_occurrence_records[doc_id] for doc_id in args.doc_id
    }
    rows = _source_rows(args.corpus, args.doc_id)
    if any(not doc_id.startswith("aci/") for doc_id in rows):
        raise SystemExit("the implemented task adapter currently supports ACI documents only")
    source_documents = {doc_id: row["text"] for doc_id, row in rows.items()}
    frozen_environment = freeze_ranker_environment(
        environment,
        occurrences_by_document=occurrence_records,
        source_documents=source_documents,
    )
    references = {doc_id: row["gold_ref"] for doc_id, row in rows.items()}
    if relation_teacher is None and args.relation_teacher:
        relation_teacher = OpenRouterRelationTeacher()
    teacher_pin = None
    if relation_teacher is not None:
        raw_teacher_pin = getattr(relation_teacher, "pin", None)
        if not isinstance(raw_teacher_pin, Mapping) or not raw_teacher_pin:
            raise ValueError("relation teacher requires an explicit non-empty pin")
        teacher_pin = dict(raw_teacher_pin)

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


def write_artifacts(artifact: dict, output: Path) -> tuple[Path, Path]:
    assertions_view, qa_pairs_view = artifact_views(artifact)
    assertions_output = output.with_name(f"{output.stem}.assertions.json")
    qa_pairs_output = output.with_name(f"{output.stem}.qa-pairs.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=1))
    assertions_output.write_text(json.dumps(assertions_view, indent=1))
    qa_pairs_output.write_text(json.dumps(qa_pairs_view, indent=1))
    return assertions_output, qa_pairs_output


def main(argv=None, *, relation_teacher=None, reader=read_context_batch):
    args = parse_args(argv)
    artifact = build_from_files(
        args, relation_teacher=relation_teacher, reader=reader
    )
    output = Path(args.out)
    assertions_output, qa_pairs_output = write_artifacts(artifact, output)
    print(
        f"wrote {output}: docs={len(artifact['documents'])} "
        f"assertions={len(artifact['assertions'])} "
        f"rejections={sum(artifact['rejections']['summary_by_reason'].values())}; "
        f"views={assertions_output},{qa_pairs_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
