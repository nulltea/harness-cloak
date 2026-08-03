"""Validate the credit-attribution revision against the utility cache (offline).

Preregistration: research-wiki/experiments/2026-08-03-credit-attribution-validation.md
Legs A-C here; leg D (held-out predictor comparison) is a separate run.
No remote calls, no reader calls, no GPU.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

FLOOR = 0.044
OUT = Path("results/ranker_v2/architecture/credit_attribution")
# Leg B anchors. NOTE on provenance: the first draft of this file gated on
# counts measured against the BLANKET restriction, i.e. before step 2 added
# excerpt-changed linkage. Those numbers were mis-derived and are not reused.
# What replaces them is an IDENTITY relating the two designs, which cannot be
# satisfied by tuning: step 2 must convert pairs out of the silenced set exactly
# when it rescues them from a false zero.
SILENCED_UNDER_DECLARED_ONLY = 3047   # measured pre-step-2, held fixed
DOCUMENT_ROUTE_RETAINED = 1873        # route-based, invariant to steps 1-4


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


def main() -> None:
    from cloak.reward.utility_credit import (
        _partitions,
        assertion_weights,
        attributed_delta_utility,
        document_denominator,
        document_utility,
        excerpt_changed_assertions,
        subset_advantages,
    )
    from cloak.ranker.environment import assemble_action_vector

    artifact = json.loads(Path("results/ranker_v2/qa/aci-full.utility").read_text())
    documents = _environment()

    cache: dict[str, dict[tuple, dict]] = defaultdict(dict)
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
        if not ids or set(row.get("action_vector", {})) != set(ids):
            continue
        cache[document.doc_id][tuple(row["action_vector"][i] for i in ids)] = row

    counts = defaultdict(int)
    additivity_violations = 0
    old_mag, new_mag, movement = [], [], []
    strata_old, strata_new = defaultdict(int), defaultdict(int)

    for doc_id, cached in sorted(cache.items()):
        document = documents[doc_id]
        ids = [d.decision_id for d in document.policy_decisions]
        partitions = _partitions(artifact, doc_id)
        weights = assertion_weights(artifact, doc_id)
        denominator = document_denominator(artifact, doc_id)
        vectors = [row["component_scores"] for row in cached.values()]
        keys = list(cached)

        # Leg A2 — additivity of leave-one-out advantages over disjoint sets
        if len(vectors) >= 2:
            everything = frozenset(weights)
            probe = frozenset(sorted(everything)[: max(1, len(everything) // 2)])
            left = subset_advantages(vectors, artifact, doc_id, probe)
            right = subset_advantages(vectors, artifact, doc_id, everything - probe)
            whole = subset_advantages(vectors, artifact, doc_id, everything)
            if any(
                abs((a + b) - c) > 1e-9
                for a, b, c in zip(left, right, whole, strict=True)
            ):
                additivity_violations += 1

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                differing = [p for p in range(len(ids)) if a[p] != b[p]]
                if len(differing) != 1:
                    continue
                decision_id = ids[differing[0]]
                route = (
                    "linked" if decision_id in partitions.linked_by_decision
                    else "document"
                )
                scores_a = cached[a]["component_scores"]
                scores_b = cached[b]["component_scores"]

                old = (
                    document_utility(scores_a, artifact, doc_id)
                    - document_utility(scores_b, artifact, doc_id)
                )
                changed = excerpt_changed_assertions(
                    artifact, doc_id, cached[a]["doc_p"], cached[b]["doc_p"],
                )
                new, attributed = attributed_delta_utility(
                    scores_a, scores_b, artifact, doc_id, decision_id, changed,
                )
                declared_only, _ = attributed_delta_utility(
                    scores_a, scores_b, artifact, doc_id, decision_id,
                )
                old_mag.append(abs(old))
                new_mag.append(abs(new))
                move = sum(
                    abs(scores_a[k] - scores_b[k]) * weights[k]
                    for k in attributed
                    if k in scores_a and k in scores_b
                ) / denominator
                movement.append(move)

                counts["pairs"] += 1
                counts[f"route_{route}"] += 1
                if route == "document" and abs(old) > 1e-9:
                    counts["document_route_retained"] += 1
                if route == "linked" and abs(new) <= 1e-9 and abs(old) > 1e-9:
                    counts["silenced_spurious"] += 1
                if abs(new) > 1e-9 and old * new < 0:
                    counts["sign_corrected"] += 1
                if abs(declared_only) <= 1e-9 and abs(new) > 1e-9:
                    counts["false_zeros_rescued"] += 1
                    if abs(old) > 1e-9:
                        counts["rescued_from_silence"] += 1
                # Cancellation: unioning excerpt-changed assertions can drive a
                # real declared effect to exactly zero when movements oppose.
                # Found on 2 pairs (declared +0.104 / +0.107, union 0.0000).
                # The signed delta is the net effect and zero is the honest
                # answer for a utility objective; the MONOTONE movement statistic
                # used for tie qualification sums |delta_s| and cannot cancel, so
                # this cannot manufacture a false tie.
                if (
                    route == "linked" and abs(declared_only) > 1e-9
                    and abs(new) <= 1e-9
                ):
                    counts["cancelled_to_zero"] += 1
                    if abs(old) > 1e-9:
                        counts["cancelled_into_silence"] += 1
                for bucket, value in ((strata_old, abs(old)), (strata_new, move)):
                    bucket[
                        "exact" if value <= 1e-9
                        else "sub-floor" if value <= FLOOR else "live"
                    ] += 1

    print("=== Leg A2 — additivity of subset advantages (BLOCKING) ===")
    print(f"documents with a violation: {additivity_violations}")
    print(f"gate: {'PASS' if additivity_violations == 0 else 'FAIL'}")

    print("\n=== Leg B — gradient-delta accounting ===")
    # The identity: every pair step 2 rescues from a false zero AND that had a
    # nonzero document delta is exactly a pair step 1 alone would have silenced.
    reconstructed = (
        counts["silenced_spurious"]
        + counts["rescued_from_silence"]
        - counts["cancelled_into_silence"]
    )
    identity_ok = reconstructed == SILENCED_UNDER_DECLARED_ONLY
    route_ok = counts["document_route_retained"] == DOCUMENT_ROUTE_RETAINED
    ok = identity_ok and route_ok
    print(f"  silenced now                {counts['silenced_spurious']:5d}")
    print(f"  + rescued from silence      {counts['rescued_from_silence']:5d}")
    print(f"  - cancelled into silence    {counts['cancelled_into_silence']:5d}  "
          f"(union opposed a declared effect; {counts['cancelled_to_zero']} total)")
    print(f"  = silenced under step 1 only{reconstructed:5d}  "
          f"(must equal {SILENCED_UNDER_DECLARED_ONLY}) {'ok' if identity_ok else 'FAIL'}")
    print(f"  document route retained     {counts['document_route_retained']:5d}  "
          f"(must equal {DOCUMENT_ROUTE_RETAINED}) {'ok' if route_ok else 'FAIL'}")
    print(f"  false zeros rescued         {counts['false_zeros_rescued']:5d}  (reported)")
    print(f"  sign corrected              {counts['sign_corrected']:5d}  (reported)")
    print(f"  pairs {counts['pairs']} | linked route {counts['route_linked']} "
          f"| document route {counts['route_document']}")
    print(f"  mean |delta|: {st.mean(old_mag):.4f} -> {st.mean(new_mag):.4f} "
          f"(x{st.mean(new_mag)/st.mean(old_mag):.2f})")
    print(f"gate: {'PASS' if ok else 'FAIL — implementation differs from analysis'}")

    def q(values, fraction):
        values = sorted(values)
        return values[int(fraction * (len(values) - 1))]

    print("\n=== Leg C — restratification under monotone movement ===")
    print(f"{'stratum':10s} {'document |delta| (old)':>24s} {'movement (new)':>18s}")
    for name in ("exact", "sub-floor", "live"):
        print(f"{name:10s} {strata_old[name]:24d} {strata_new[name]:18d}")
    print(f"movement p50/p75/p90/p99: {q(movement,.5):.4f} / {q(movement,.75):.4f}"
          f" / {q(movement,.9):.4f} / {q(movement,.99):.4f}")
    print("NOTE: 0.044 is a DOCUMENT-unit floor and is not dimensionally valid for")
    print("      movement; strata above are indicative only until it is re-measured.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation-report.json").write_text(json.dumps({
        "additivity_violations": additivity_violations,
        "silenced_under_declared_only": SILENCED_UNDER_DECLARED_ONLY,
        "document_route_retained_anchor": DOCUMENT_ROUTE_RETAINED,
        "counts": dict(counts),
        "mean_abs_delta_old": st.mean(old_mag),
        "mean_abs_delta_new": st.mean(new_mag),
        "strata_document_old": dict(strata_old),
        "strata_movement_new": dict(strata_new),
        "movement_percentiles": {
            "p50": q(movement, .5), "p75": q(movement, .75),
            "p90": q(movement, .9), "p99": q(movement, .99),
        },
    }, indent=1))


if __name__ == "__main__":
    main()
