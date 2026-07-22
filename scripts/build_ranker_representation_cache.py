"""Build the frozen Ranker-v2 document and semantic-relation tensor store."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from cloak.train.ranker_environment import load_ranker_environment
from cloak.train.ranker_representation import (
    build_representation_store,
    load_pinned_encoder,
)


def _environment_hash(path: Path) -> str:
    artifact = json.loads(path.read_text())
    if artifact.get("artifact_version") != "ranker-v2-environment-v2":
        raise ValueError("expected ranker-v2-environment-v2")
    frozen = artifact.get("frozen_environment")
    if not isinstance(frozen, Mapping):
        raise ValueError("environment is missing frozen_environment")
    environment_hash = frozen.get("environment_hash")
    if not isinstance(environment_hash, str) or not environment_hash.startswith("sha256:"):
        raise ValueError("environment is missing environment_hash")
    return environment_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--doc-id", action="append", default=[],
        help="build only this document; repeat for a multi-document slice",
    )
    parser.add_argument(
        "--cache-only-model", action="store_true",
        help="require the pinned model snapshot to already exist locally",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    environment_path = Path(args.environment)
    environment_hash = _environment_hash(environment_path)
    documents = load_ranker_environment(environment_path)
    if args.doc_id:
        unknown = sorted(set(args.doc_id) - set(documents))
        if unknown:
            raise ValueError(f"unknown --doc-id values: {unknown}")
        selected = {doc_id: documents[doc_id] for doc_id in args.doc_id}
    else:
        selected = documents
    encoder = load_pinned_encoder(
        cache_only_model=args.cache_only_model,
        device=args.device,
    )
    manifest_path = build_representation_store(
        selected,
        environment_hash=environment_hash,
        out_dir=Path(args.out_dir),
        encoder=encoder,
    )
    manifest = json.loads(manifest_path.read_text())
    unique_relation_files = {
        row["tensor_file"] for row in manifest["relations"].values()
    }
    print(
        f"representation store published: documents={len(manifest['documents'])} "
        f"relations={len(manifest['relations'])} "
        f"unique_relations={len(unique_relation_files)} -> {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
