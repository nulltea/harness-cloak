#!/usr/bin/env python3
"""Refactor hash-stability gate (docs/plans/2026-07-27-codebase-cleanup-refactor.md).

Asserts the refactor changed nothing observable in artifact/cache identity space:
1. the frozen environment hash recomputes byte-identically;
2. the profile-count target artifact hash recomputes byte-identically;
3. every stored utility-cache identity is still produced by the live code path
   (render -> template pin -> extractor source-hash pin -> reader/scorer pins)
   and still hits the cache.

Run after every refactor commit:
  PYTHONPATH=src:scripts .venv/bin/python scripts/refactor_hash_gate.py
"""
import hashlib
import json
import sys
from pathlib import Path

from cloak.ranker.profile_count import ProfileCountTargets, _canonical_hash
from cloak.ranker.environment import load_ranker_environment
from cloak.ranker.privacy import DirectCountPrivacyProvider
from cloak.reward.roundtrip import _cache_identity
from cloak.reward.utility_cache import UtilityCache, UtilityRequest
from cloak.ranker.environment import assemble_action_vector
from train_interactive_ranker import _demote_out_of_scope_decisions

ENVIRONMENT = Path("results/ranker_v2/environment/ranker-env.json")
TARGETS = Path("results/ranker_v2/reward/profile-count-targets.json")
UTILITY_ARTIFACT = Path("results/ranker_v2/qa/aci-full.utility")
UTILITY_CACHE = Path("results/ranker_v2/cache/utility-results.jsonl")


def main() -> None:
    failures = []

    env = json.loads(ENVIRONMENT.read_text())
    frozen = env["frozen_environment"]
    payload = json.dumps(
        {key: frozen[key] for key in ("artifact_version", "documents")},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    recomputed = "sha256:" + hashlib.sha256(payload).hexdigest()
    if recomputed != frozen["environment_hash"]:
        failures.append(f"environment hash drift: {recomputed}")

    targets = json.loads(TARGETS.read_text())
    unhashed = {k: v for k, v in targets.items() if k != "artifact_hash"}
    if _canonical_hash(unhashed) != targets["artifact_hash"]:
        failures.append("profile-count target artifact hash drift")
    ProfileCountTargets.from_artifact(targets)  # revalidates internal bindings

    utility_artifact = json.loads(UTILITY_ARTIFACT.read_text())
    documents = tuple(load_ranker_environment(ENVIRONMENT).values())
    documents, _ = _demote_out_of_scope_decisions(
        documents, DirectCountPrivacyProvider(targets)
    )
    documents_by_id = {document.doc_id: document for document in documents}
    cache = UtilityCache(UTILITY_CACHE)
    checked = 0
    skipped = 0
    for line in UTILITY_CACHE.read_text().splitlines():
        row = json.loads(line)
        result = row["result"]
        document = documents_by_id[result["doc_id"]]
        current_ids = {d.decision_id for d in document.policy_decisions}
        if set(result["action_vector"]) != current_ids:
            skipped += 1  # row predates a policy-scope change; not re-derivable today
            continue
        request = UtilityRequest(
            document=document,
            action_vector=result["action_vector"],
            utility_artifact=utility_artifact,
            environment_hash=frozen["environment_hash"],
        )
        doc_p, _ = assemble_action_vector(document, result["action_vector"])
        # Winner re-verification rows are stored with reader_refresh=True by design.
        identity = _cache_identity(
            request, doc_p,
            reader_refresh=bool(row["identity"].get("reader_refresh")),
        )
        if cache.request_identity(identity) != row["request_identity"]:
            failures.append(
                f"cache identity drift for {result['doc_id']} "
                f"(vector {list(result['action_vector'])[0][:16]}...)"
            )
        elif cache.lookup(identity) is None:
            failures.append(f"cache lookup miss for {result['doc_id']}")
        checked += 1

    if failures:
        for failure in failures:
            print(f"HASH GATE FAIL: {failure}")
        sys.exit(1)
    if checked == 0:
        print("HASH GATE FAIL: no cache identity was re-derivable — gate is vacuous")
        sys.exit(1)
    print(
        f"HASH GATE PASS: environment + targets + {checked} cache identities stable"
        f" ({skipped} pre-scope rows skipped)"
    )


if __name__ == "__main__":
    main()
