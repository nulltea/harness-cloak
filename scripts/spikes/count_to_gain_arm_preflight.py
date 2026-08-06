"""Preflight for the count->gain coupling arm (offline, no GPU, no remote calls).

**Bootstrap statistic parity.** Seed records must carry the same statistics an
online probe records, so epoch-0 labels are qualified under the same measurement
contract as every later round. Reports the label delta the fix causes, which is
the only way the fix touches training.

The preregistered gradient-routing validity check (count reaches `gain_head`
iff `count_to_gain == "coupled"`) lives in the test suite, where the semantic
policy fixture already exists:
`pytest src/cloak/tests/test_semantic_ranker.py -k count_to_gain`

Run: .venv/bin/python -u scripts/spikes/count_to_gain_arm_preflight.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

CACHE = Path("results/ranker_v2/cache/utility-results.jsonl")
ARTIFACT = Path("results/ranker_v2/qa/aci-full.utility")
ENVIRONMENT = Path("results/ranker_v2/environment/ranker-env.json")
TARGETS = Path("results/ranker_v2/reward/profile-count-targets.json")
# The run's exact documents, passed to the trainer as four --doc-id flags. The
# preflight must measure the SAME set: a figure over the whole 63-document
# environment does not describe this experiment.
RUN_DOCUMENTS = (
    "aci/D2N005", "aci/D2N027", "aci/D2N031", "aci/D2N063",
)


def _environment():
    from cloak.ranker.environment import load_ranker_environment
    from cloak.ranker.privacy import DirectCountPrivacyProvider
    from train_interactive_ranker import _demote_out_of_scope_decisions

    documents = tuple(load_ranker_environment(ENVIRONMENT).values())
    documents, _ = _demote_out_of_scope_decisions(
        documents,
        DirectCountPrivacyProvider(json.loads(TARGETS.read_text())),
    )
    selected = {
        document.doc_id: document for document in documents
        if document.doc_id in RUN_DOCUMENTS
    }
    missing = sorted(set(RUN_DOCUMENTS) - set(selected))
    assert not missing, f"run documents absent from the environment: {missing}"
    return selected


def check_bootstrap_parity(documents, artifact) -> dict:
    """Seed ledger must carry every online statistic; report the label delta."""
    from cloak.ranker.interactive import (
        bootstrap_tie_evidence_from_cache,
        compute_tie_labels,
    )
    from cloak.ranker.profile_count import ProfileCountTargets
    from cloak.reward.utility_cache import stable_hash

    ledger = bootstrap_tie_evidence_from_cache(CACHE, documents, artifact)
    records = [record for values in ledger.values() for record in values]
    missing = [
        key for key in ("delta_u", "delta_u_attributed", "delta_u_linked",
                        "movement_l1", "context_hash", "round")
        if any(key not in record for record in records)
    ]
    assert not missing, f"bootstrap records missing {missing}"
    assert all(
        record["movement_l1"] is not None for record in records
    ), "movement must be computed for every seed record, never None"

    targets = ProfileCountTargets.from_artifact(json.loads(TARGETS.read_text()))
    revised = compute_tie_labels(ledger, documents, targets)
    # The pre-fix contract: document-level delta only, which is what the seed
    # carried before. Same ledger, statistics stripped.
    legacy_ledger = {
        key: [
            {"delta_u": r["delta_u"], "context_hash": r["context_hash"],
             "round": r["round"]}
            for r in values
        ]
        for key, values in ledger.items()
    }
    legacy = compute_tie_labels(legacy_ledger, documents, targets)

    def pairs(labels):
        return {(doc, dec, a, b) for (doc, dec), ps in labels.items() for a, b in ps}

    revised_pairs, legacy_pairs = pairs(revised), pairs(legacy)
    return {
        "run_documents": list(RUN_DOCUMENTS),
        "document_set_hash": stable_hash(sorted(RUN_DOCUMENTS)),
        "seed_pairs": len(ledger),
        "seed_records": len(records),
        "labels_legacy_contract": len(legacy_pairs),
        "labels_revised_contract": len(revised_pairs),
        "decisions_legacy": len(legacy),
        "decisions_revised": len(revised),
        "lost_by_revision": sorted(legacy_pairs - revised_pairs),
        "gained_by_revision": sorted(revised_pairs - legacy_pairs),
    }


def main() -> None:
    documents = _environment()
    artifact = json.loads(ARTIFACT.read_text())

    print("\n=== bootstrap statistic parity ===")
    parity = check_bootstrap_parity(documents, artifact)
    print(f"  documents                  {' '.join(parity['run_documents'])}")
    print(f"  document_set_hash          {parity['document_set_hash']}")
    for key in ("seed_pairs", "seed_records", "decisions_legacy",
                "decisions_revised", "labels_legacy_contract",
                "labels_revised_contract"):
        print(f"  {key:26s} {parity[key]}")
    print(f"  lost by revision   {len(parity['lost_by_revision'])}")
    print(f"  gained by revision {len(parity['gained_by_revision'])}")
    print("  PASS — every seed record carries the online statistics")

    # The routing check runs in the test suite; record its outcome here so the
    # record's "executed, PASS" claim has a dated artifact behind it rather than
    # a transcript assertion.
    routing = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "src/cloak/tests/test_semantic_ranker.py"
         "::test_count_to_gain_coupling_moves_exactly_one_gradient_edge"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": "src:scripts"},
    )
    print(f"\n=== gradient-routing validity (test suite) ===\n  "
          f"{'PASS' if routing.returncode == 0 else 'FAIL'}")

    out = Path("results/ranker_v2/architecture/count_to_gain")
    out.mkdir(parents=True, exist_ok=True)
    (out / "arm-preflight.json").write_text(json.dumps({
        "executed": dt.date.today().isoformat(),
        "gradient_routing_test": {
            "test": "src/cloak/tests/test_semantic_ranker.py"
                    "::test_count_to_gain_coupling_moves_exactly_one_gradient_edge",
            "status": "pass" if routing.returncode == 0 else "fail",
        },
        "bootstrap_parity": parity,
    }, indent=1))


if __name__ == "__main__":
    main()
