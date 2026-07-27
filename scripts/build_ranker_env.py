"""Build policy-free Ranker-v2 environments.

Emits the frozen occurrence/decision environment plus its per-corpus policy-decision
index. Consumes the arms artifact (detection is process-nondeterministic; spec §3.3-5) —
spans, NLI-gated lattices, and the embedded action tables all come from the artifact; this
script never recomputes them. Placeholder actions carry mode only; concrete <TYPE_n> tokens
are assigned at assemble() time.

Output: data/ranker_env.json
Run: PYTHONPATH=src:scripts .venv/bin/python -u scripts/build_ranker_env.py
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from build_arms_artifact import ARTIFACT, CORPORA

from cloak.corpora import load_task_docs
from cloak.train.qa_freeze import (
    freeze_v2_environment_from_legacy_arms,
    legacy_arms_ranker_environment,
)

OUT = Path("data/ranker_env.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int,
                    help="embedded documents to emit in total")
    ap.add_argument("--corpora", default=",".join(CORPORA),
                    help="comma-separated registered corpus names (e.g. include lexsum)")
    ap.add_argument("--arms", default=str(ARTIFACT),
                    help="input arms artifact path (default: the frozen historical artifact)")
    ap.add_argument("--out", default=str(OUT),
                    help="output env path (default: the frozen env — override for the pilot env)")
    args = ap.parse_args()
    out = Path(args.out)

    t0 = time.time()
    raw_art = json.loads(Path(args.arms).read_text())
    arms_meta = dict(raw_art.pop("_meta", {}) or {})
    art = raw_art
    requested = args.corpora.split(",")
    embedded_documents = {}
    for corpus in requested:
        for doc_id, entry in art.get(corpus, {}).items():
            if args.n_docs is not None and len(embedded_documents) >= args.n_docs:
                break
            if isinstance(entry, dict) and isinstance(entry.get("v2_frozen_input"), dict):
                embedded_documents[doc_id] = entry["v2_frozen_input"]
        if args.n_docs is not None and len(embedded_documents) >= args.n_docs:
            break
    if embedded_documents:
        frozen_version = (
            (arms_meta.get("v2_frozen_environment") or {}).get("artifact_version")
            or "occurrence-decisions-v1"
        )
        frozen = {
            "artifact_version": frozen_version,
            "documents": embedded_documents,
        }
        payload = json.dumps(
            frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8")
        frozen["environment_hash"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    else:
        source_documents = {}
        for corpus in requested:
            source_documents.update({
                row["id"]: row["text"]
                for row in load_task_docs(corpus, args.n_docs or 16)
            })
        frozen = freeze_v2_environment_from_legacy_arms(
            legacy_arms_ranker_environment(art), art,
            source_documents=source_documents,
        )
    selected_docs = {
        doc_id: document
        for doc_id, document in frozen["documents"].items()
        if any(doc_id in art.get(corpus, {}) for corpus in requested)
    }
    frozen = {
        "artifact_version": frozen["artifact_version"],
        "documents": selected_docs,
        "environment_hash": frozen["environment_hash"],
    }
    # Recompute via the compatibility freezer after limiting corpora only when
    # building all requested docs; per-doc hashes remain canonical either way.
    env = {
        "artifact_version": (
            "ranker-v2-environment-v2"
            if frozen["artifact_version"] == "occurrence-decisions-v2"
            else "ranker-v2-environment-v1"
        ),
        "compatibility_adapter": (
            "frozen-arms-count-provenance-v1"
            if frozen["artifact_version"] == "occurrence-decisions-v2"
            else "legacy-arms-policy-free-v1"
        ),
        "frozen_environment": frozen,
        "corpora": {
            corpus: {
                doc_id: {
                    "decisions": document["decisions"],
                    "occurrences": document["occurrences"],
                    "policy_decision_ids": [
                        decision["decision_id"] for decision in document["decisions"]
                        if decision.get("ranker_selectable", True)
                    ],
                    "trainable": any(
                        decision.get("ranker_selectable", True)
                        for decision in document["decisions"]
                    ),
                }
                for doc_id, document in selected_docs.items()
                if doc_id in art.get(corpus, {})
            }
            for corpus in requested
        },
    }
    # The V2 training artifact must be self-describing but never carry legacy
    # policy parameters or behavior-clone/probe provenance.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=1))
    print(f"wall {time.time()-t0:.0f}s -> {out} (ranker-v2, docs={len(selected_docs)})")


if __name__ == "__main__":
    main()
