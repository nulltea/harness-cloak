"""Validate and quantify the counterfactual ΔU measurement revision (offline).

Two measurement changes, both evaluated against the existing utility cache with
no remote or reader calls. Record:
research-wiki/experiments/2026-08-03-counterfactual-measurement-revision.md

targeted   — for a single-decision flip, only context assertions whose reader
             excerpt CHANGES can move (the reader runs at temperature 0, so a
             byte-identical excerpt yields a byte-identical answer). BLOCKING
             gate: every skipped assertion must already carry an identical
             cached score. Reports the reader calls the criterion would avoid.
normalized — per-decision attribution divided by the decision's own linked
             weight W_L instead of the document denominator W, so a delta reads
             as "fraction of this decision's obligations broken" in [-1, 1].
             Reports the restratification.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

READER_FLOOR = 0.044
EXACT_ATOL = 1e-9
OUT = Path("results/ranker_v2/architecture/equivalence_critic")


def _environment():
    from cloak.ranker.environment import load_ranker_environment
    from cloak.ranker.privacy import DirectCountPrivacyProvider
    from train_interactive_ranker import _demote_out_of_scope_decisions

    documents = tuple(load_ranker_environment(
        Path("results/ranker_v2/environment/ranker-env.json")
    ).values())
    documents, _ = _demote_out_of_scope_decisions(
        documents,
        DirectCountPrivacyProvider(json.loads(
            Path("results/ranker_v2/reward/profile-count-targets.json").read_text()
        )),
    )
    return {document.doc_id: document for document in documents}


def _cached_vectors(documents):
    """{doc_id: {action-vector tuple: component_scores}} for full-coverage rows."""
    vectors: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for line in Path(
        "results/ranker_v2/cache/utility-results.jsonl"
    ).read_text().splitlines():
        try:
            row = json.loads(line)["result"]
        except (ValueError, KeyError):
            continue
        document = documents.get(row.get("doc_id"))
        if document is None:
            continue
        ids = [d.decision_id for d in document.policy_decisions]
        vector = row.get("action_vector", {})
        scores = row.get("component_scores")
        if not ids or set(vector) != set(ids) or not isinstance(scores, dict):
            continue
        vectors[document.doc_id][tuple(vector[i] for i in ids)] = scores
    return vectors


def main() -> None:
    from cloak.qa.scoring import reader_excerpt
    from cloak.ranker.environment import assemble_action_vector
    from cloak.reward.utility_credit import _partitions

    artifact = json.loads(Path("results/ranker_v2/qa/aci-full.utility").read_text())
    assertions = artifact["assertions"]
    documents = _environment()
    vectors = _cached_vectors(documents)

    checked = violations = skipped = rescored = 0
    worst = 0.0
    per_document_saving = []
    strata_document: dict[str, int] = defaultdict(int)
    strata_linked: dict[str, int] = defaultdict(int)
    magnitudes_document, magnitudes_linked = [], []

    for doc_id, cached in sorted(vectors.items()):
        document = documents[doc_id]
        ids = [d.decision_id for d in document.policy_decisions]
        partitions = _partitions(artifact, doc_id)
        weights, denominator = partitions.weights, partitions.denominator
        context_rows = [
            row for row in assertions.values()
            if row.get("family") == "context" and row.get("doc_id") == doc_id
            and str(row["assertion_id"]) in weights
        ]
        # Assemble once per distinct action vector, not once per pair.
        excerpts: dict[tuple, dict[str, str]] = {}
        for key in cached:
            doc_p, _ = assemble_action_vector(
                document, dict(zip(ids, key, strict=True))
            )
            excerpts[key] = {
                str(row["assertion_id"]): reader_excerpt(
                    doc_p, row.get("evidence") or {}
                )
                for row in context_rows
            }

        keys = list(cached)
        doc_skipped = doc_total = 0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                differing = [p for p in range(len(ids)) if a[p] != b[p]]
                if len(differing) != 1:
                    continue
                decision_id = ids[differing[0]]
                linked = partitions.linked_by_decision.get(decision_id, frozenset())
                scores_a, scores_b = cached[a], cached[b]

                # --- targeted: gate + saving over this document's context rows
                for assertion_id, excerpt_a in excerpts[a].items():
                    if excerpt_a != excerpts[b][assertion_id]:
                        rescored += 1
                        doc_total += 1
                        continue
                    doc_skipped += 1
                    doc_total += 1
                    skipped += 1
                    if assertion_id in scores_a and assertion_id in scores_b:
                        checked += 1
                        gap = abs(scores_a[assertion_id] - scores_b[assertion_id])
                        if gap > EXACT_ATOL:
                            violations += 1
                            worst = max(worst, gap)

                # --- normalized: restratify the same evidence under W_L
                numerator = sum(
                    weights[x] * (scores_a[x] - scores_b[x])
                    for x in linked if x in scores_a and x in scores_b
                )
                delta_document = numerator / denominator
                linked_mass = sum(weights[x] for x in linked)
                if not linked:
                    strata_document["derivable"] += 1
                    strata_linked["derivable"] += 1
                    continue
                delta_linked = numerator / linked_mass
                magnitudes_document.append(abs(delta_document))
                magnitudes_linked.append(abs(delta_linked))
                for bucket, value in (
                    (strata_document, delta_document), (strata_linked, delta_linked),
                ):
                    if abs(value) <= EXACT_ATOL:
                        bucket["exact"] += 1
                    elif abs(value) <= READER_FLOOR:
                        bucket["sub-floor"] += 1
                    else:
                        bucket["live"] += 1
        if doc_total:
            per_document_saving.append(doc_skipped / doc_total)

    print("=== targeted context re-scoring — BLOCKING exactness gate ===")
    print(f"skipped assertion-instances with both scores cached: {checked}")
    print(f"disagreements (a byte-identical excerpt scoring differently): {violations}")
    print(f"worst disagreement: {worst:.2e}")
    verdict = "PASS — skipping is exact" if violations == 0 else "FAIL — criterion unsound"
    print(f"gate: {verdict}")

    total = skipped + rescored
    print("\n=== targeted context re-scoring — saving ===")
    print(f"context assertion-instances over all pairs: {total}")
    print(f"  would be re-scored: {rescored} ({100*rescored/total:.0f}%)")
    print(f"  provably skippable: {skipped} ({100*skipped/total:.0f}%)")
    print(
        f"per-document skip fraction: median {st.median(per_document_saving):.0%}  "
        f"min {min(per_document_saving):.0%}  max {max(per_document_saving):.0%}"
    )

    def q(values, fraction):
        values = sorted(values)
        return values[int(fraction * (len(values) - 1))]

    print("\n=== linked-mass normalization — restratification ===")
    order = ("derivable", "exact", "sub-floor", "live")
    print(f"{'stratum':10s} {'/ W (current)':>15s} {'/ W_L (proposed)':>18s}")
    for name in order:
        print(f"{name:10s} {strata_document[name]:15d} {strata_linked[name]:18d}")
    print(
        f"\n|delta| p50/p75/p90 — / W  : {q(magnitudes_document,.5):.4f} / "
        f"{q(magnitudes_document,.75):.4f} / {q(magnitudes_document,.9):.4f}"
    )
    print(
        f"|delta| p50/p75/p90 — / W_L: {q(magnitudes_linked,.5):.4f} / "
        f"{q(magnitudes_linked,.75):.4f} / {q(magnitudes_linked,.9):.4f}"
    )
    print(
        "NOTE: the 0.044 floor was measured at document granularity; under W_L it "
        "must be re-measured before any gate reuses it."
    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "measurement-revision-report.json").write_text(json.dumps({
        "targeted": {
            "gate_checked": checked, "gate_violations": violations,
            "worst_disagreement": worst,
            "instances_total": total, "instances_rescored": rescored,
            "instances_skippable": skipped,
            "per_document_skip_median": st.median(per_document_saving),
        },
        "normalized": {
            "strata_document": dict(strata_document),
            "strata_linked": dict(strata_linked),
            "magnitudes_document_p50_p75_p90": [
                q(magnitudes_document, f) for f in (.5, .75, .9)
            ],
            "magnitudes_linked_p50_p75_p90": [
                q(magnitudes_linked, f) for f in (.5, .75, .9)
            ],
        },
    }, indent=1))


if __name__ == "__main__":
    main()
