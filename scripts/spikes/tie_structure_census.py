"""Tie-structure census of the full ranker corpus (adoption precondition).

Offline, CPU-only. For every document in the frozen environment (after scope
demotion): decision count, menu structure, assertion linkage; plus, where the
utility cache has coverage, measured per-decision delta-U spans from cached
single-decision vector pairs (exact ties, reward-flat decisions, spans above
the 0.044 reader floor). Answers how prevalent the D2N005 class
(tie-dominated, low-D) is at production scale, and defines the strata
production gates should report over. Companion: tie-ownership fork entries in
docs/specs/RL/interactive-ranker-v2-decision-log.md.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")

from cloak.ranker.environment import load_ranker_environment
from cloak.ranker.privacy import DirectCountPrivacyProvider
from cloak.ranker.profile_count import ProfileCountTargets
from train_interactive_ranker import _demote_out_of_scope_decisions

READER_FLOOR = 0.044
OUT = Path("results/ranker_v2/census")


def main() -> None:
    targets_payload = json.loads(
        Path("results/ranker_v2/reward/profile-count-targets.json").read_text()
    )
    documents = tuple(load_ranker_environment(
        Path("results/ranker_v2/environment/ranker-env.json")
    ).values())
    documents, _ = _demote_out_of_scope_decisions(
        documents, DirectCountPrivacyProvider(targets_payload)
    )
    artifact = json.loads(Path("results/ranker_v2/qa/aci-full.utility").read_text())

    # cached full-coverage vectors per doc for measured spans
    vectors: dict[str, dict[tuple, float]] = {}
    ids_by_doc = {
        d.doc_id: [dec.decision_id for dec in d.policy_decisions]
        for d in documents
    }
    for line in Path(
        "results/ranker_v2/cache/utility-results.jsonl"
    ).read_text().splitlines():
        try:
            row = json.loads(line)["result"]
        except (ValueError, KeyError):
            continue
        ids = ids_by_doc.get(row.get("doc_id"))
        if ids is None or set(row.get("action_vector", {})) != set(ids):
            continue
        vectors.setdefault(row["doc_id"], {})[
            tuple(row["action_vector"][i] for i in ids)
        ] = float(row["utility"])

    census = []
    for document in documents:
        doc_id = document.doc_id
        ids = ids_by_doc[doc_id]
        decision_count = len(ids)
        if decision_count == 0:
            continue
        menu_sizes = [len(dec.actions) for dec in document.policy_decisions]
        level_counts = [
            sum(1 for a in dec.actions if a.mode == "level")
            for dec in document.policy_decisions
        ]
        doc_meta = artifact["documents"].get(doc_id, {})
        linked = defaultdict(set)
        for assertion_id in doc_meta.get("assertion_ids", ()):
            row = artifact["assertions"][assertion_id]
            if row.get("credit_routing") == "linked":
                for dep in row.get("policy_dependency_decision_ids", ()):
                    linked[dep].add(assertion_id)
        linked_counts = [len(linked.get(i, ())) for i in ids]

        # measured spans from cached single-decision pairs
        cached = vectors.get(doc_id, {})
        per_decision_spans: dict[str, list[float]] = {i: [] for i in ids}
        if len(cached) >= 2:
            index: dict[tuple, list[tuple]] = {}
            for key in cached:
                for position in range(decision_count):
                    rest = key[:position] + ("*",) + key[position + 1:]
                    index.setdefault((position, rest), []).append(key)
            for (position, _rest), keys in index.items():
                if len(keys) < 2:
                    continue
                us = [cached[k] for k in keys]
                per_decision_spans[ids[position]].append(max(us) - min(us))
        measured = {
            i: spans for i, spans in per_decision_spans.items() if spans
        }
        flat_decisions = sum(
            1 for spans in measured.values() if max(spans) <= 1e-9
        )
        subnoise_decisions = sum(
            1 for spans in measured.values()
            if max(spans) <= READER_FLOOR and max(spans) > 1e-9
        )
        stratum = (
            "tiny" if decision_count <= 6
            else "mid" if decision_count <= 15 else "large"
        )
        census.append({
            "doc_id": doc_id,
            "corpus": document.corpus,
            "decision_count": decision_count,
            "stratum": stratum,
            "menu_size_min_max": [min(menu_sizes), max(menu_sizes)],
            "level_count_min_max": [min(level_counts), max(level_counts)],
            "decisions_with_zero_linked_assertions": sum(
                1 for c in linked_counts if c == 0
            ),
            "linked_assertions_median": sorted(linked_counts)[
                decision_count // 2
            ],
            "cache_vectors": len(cached),
            "decisions_with_measured_spans": len(measured),
            "measured_flat_decisions": flat_decisions,
            "measured_subnoise_decisions": subnoise_decisions,
            "measured_max_span": max(
                (max(s) for s in measured.values()), default=None,
            ),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tie-structure-census.json").write_text(
        json.dumps({"reader_floor": READER_FLOOR, "documents": census}, indent=1)
    )

    strata = defaultdict(list)
    for row in census:
        strata[row["stratum"]].append(row)
    print(f"documents with policy decisions: {len(census)}")
    for name in ("tiny", "mid", "large"):
        rows = strata.get(name, [])
        with_cache = [r for r in rows if r["decisions_with_measured_spans"] > 0]
        zero_linked = sum(r["decisions_with_zero_linked_assertions"] for r in rows)
        total_dec = sum(r["decision_count"] for r in rows)
        print(
            f"{name:5s}: {len(rows):3d} docs | decisions {total_dec:4d} | "
            f"zero-linked decisions {zero_linked:3d} | docs with measured "
            f"cache coverage {len(with_cache)}"
        )
        for r in with_cache:
            print(
                f"    {r['doc_id']}: D={r['decision_count']} vectors="
                f"{r['cache_vectors']} measured={r['decisions_with_measured_spans']} "
                f"flat={r['measured_flat_decisions']} "
                f"subnoise={r['measured_subnoise_decisions']}"
            )


if __name__ == "__main__":
    main()
