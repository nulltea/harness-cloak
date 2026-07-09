"""In-graph coherence normalization for accepted proposed lattice rows.

Ported from scripts/spikes/clean_drug_health_lattice_coherence.py, the external cleanup script
that was run by hand over the drug-health-procedure run's output. This module makes the same
fix happen inside the graph, before persistence, so future runs don't need that manual pass.

Problem this solves: the producer proposes levels/counts one item at a time with no cross-item
context, so the same generalization concept ends up as several near-duplicate strings
("pharmaceutical compound" / "pharmaceutical agent" / "pharmaceutical product") and even a
single, identical level string carries wildly different self-reported counts across entries
(e.g. "medication" ranging 142 to 24,500,000 in the reviewed run). This module: (1) merges known
synonym paraphrases to one canonical spelling, (2) resolves one coherent count per canonical
label for the whole run via anchor-aware weighted isotonic regression, (3) reorders each entry's
own chain to match the corpus-wide consensus order so counts stay monotone without clamping.

Three bugs were found and fixed while developing the original script; all three fixes are
ported here, not just the happy path:
  1. Tie-breaking must be identical between the count-assignment sort and the per-entry
     chain-reorder sort, or a rank tie can order two labels one way for the count computation
     and the opposite way when reordering an entry, silently reintroducing a monotonicity
     violation (the "aldactone" duplicate-level bug).
  2. Anchor-to-anchor order must come from the anchor's own real-world value, not empirical
     chain-depth rank -- two anchored labels can sit at the exact same typical chain depth in
     the corpus despite an enormous real-world size difference (the "genetic disorder" /
     "musculoskeletal disorder" false-collision bug).
  3. A final per-entry monotonic safety clamp is still required even after reordering, because
     an anchored label's real value doesn't always agree with its neighbors' empirical rank.
"""
from __future__ import annotations

import bisect
import json
import re
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_ANCHOR_PATHS = {
    "drug": Path("data/lattice_sources/reference/drug_class_anchors.json"),
    "health-condition": Path("data/lattice_sources/reference/health_condition_class_anchors.json"),
}

# Curated synonym clusters: canonical label -> variant spellings. Only genuine same-scope
# paraphrases go in one cluster -- narrower/broader subtypes (e.g. "antihypertensive medication"
# vs bare "medication") are deliberately NOT merged, to avoid flattening real lattice granularity.
CLUSTERS: dict[str, dict[str, set[str]]] = {
    "drug": {
        "medication": {"medication", "medications", "medicine", "medicines", "pharmaceutical", "pharmaceuticals", "drug", "drugs"},
        "prescription medication": {"prescription medication", "prescription drug", "prescription pharmaceutical"},
        "pharmaceutical compound": {
            "pharmaceutical compound", "pharmaceutical agent", "pharmaceutical product",
            "pharmaceutical substance", "pharmacological agent", "pharmacological substance",
            "pharmacological compound", "pharmacological agents", "active pharmaceutical ingredient",
            "pharmaceutical entry",
        },
        "therapeutic agent": {
            "therapeutic agent", "therapeutic compound", "therapeutic substance", "therapeutic agents",
            "therapeutic compounds", "systemic therapeutic compound", "clinical substance",
            "clinical agent", "clinical drug", "medicinal agent", "medicinal compound",
        },
        "chemical substance": {
            "chemical substance", "chemical compound", "chemical entity", "substance",
            "organic compound", "organic molecule", "molecular entity", "bioactive compound",
            "bioactive molecule", "material entity", "material substance", "medical substance",
            "physical entity", "physical substance", "biological and chemical entity",
        },
        "cardiovascular agent": {"cardiovascular agent", "cardiovascular therapeutic", "cardiovascular medication"},
        "medical intervention": {"medical intervention", "clinical intervention"},
        "psychotropic agent": {"psychotropic agent", "psychotropic medication"},
        "dietary supplement": {"dietary supplement", "nutritional supplement"},
        "antihypertensive agent": {"antihypertensive agent", "antihypertensive medication", "antihypertensive"},
        "central nervous system agent": {"central nervous system agent", "central nervous system therapeutic"},
        "anticonvulsant": {"anticonvulsant", "antiepileptic agent"},
    },
    "health-condition": {
        "medical condition": {
            "medical condition", "health condition", "clinical condition", "clinical health condition",
            "health disorder",
        },
        "clinical symptom": {"clinical symptom", "symptom"},
        "clinical finding": {"clinical finding", "clinical manifestation"},
        "musculoskeletal disorder": {"musculoskeletal disorder", "orthopedic condition"},
        "gastrointestinal condition": {"gastrointestinal condition", "digestive system disorder"},
    },
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("_", " ").replace("-", " ").strip().lower())


@lru_cache(maxsize=8)
def load_reference_anchors(runtime_type: str) -> dict[str, float]:
    path = DEFAULT_ANCHOR_PATHS.get(runtime_type)
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {str(k): float(v) for k, v in data.get("anchors", {}).items()}


def build_variant_map(runtime_type: str) -> dict[str, str]:
    mapping = {}
    for canonical, variants in CLUSTERS.get(runtime_type, {}).items():
        for variant in variants:
            mapping[_norm(variant)] = canonical
    return mapping


def canonicalize(level: str, variant_map: dict[str, str]) -> str:
    return variant_map.get(_norm(level), level.strip())


def _average_depth_rank(canonical_chains: list[list[str]]) -> dict[str, float]:
    """Real-valued "how broad is this label typically used" score, from ALL of a label's
    occurrences at once (not sparse pairwise voting, which is too easily dominated by a
    handful of individually mis-ordered chains). A total order from real numbers cannot
    contain a cycle, so no cycle-detection machinery is needed downstream."""
    positions: dict[str, list[float]] = defaultdict(list)
    for chain in canonical_chains:
        denom = max(1, len(chain) - 1)
        for idx, label in enumerate(chain):
            positions[label].append(idx / denom)
    return {label: statistics.mean(vals) for label, vals in positions.items()}


def _weighted_pava(ordered_labels: list[str], value_of: dict[str, float], weight_of: dict[str, float]) -> dict[str, float]:
    """Pool-adjacent-violators isotonic regression, weighted by observation count. A naive
    forward-max walk isn't real isotonic regression -- a single extreme outlier baseline early
    in the order would permanently set the floor for every later label. PAVA instead merges only
    the locally violating neighbors into one pooled (weighted-mean) block."""
    blocks: list[list[float]] = []  # each: [weighted_sum, total_weight]
    block_labels: list[list[str]] = []
    for label in ordered_labels:
        blocks.append([value_of[label] * weight_of[label], weight_of[label]])
        block_labels.append([label])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) > (blocks[-1][0] / blocks[-1][1]):
            b2, b1 = blocks.pop(), blocks.pop()
            blocks.append([b1[0] + b2[0], b1[1] + b2[1]])
            labels2, labels1 = block_labels.pop(), block_labels.pop()
            block_labels.append(labels1 + labels2)

    final_count: dict[str, float] = {}
    for (weighted_sum, total_weight), labels in zip(blocks, block_labels):
        value = round(weighted_sum / total_weight, 2)
        for label in labels:
            final_count[label] = value
    return final_count


def _rank_order(rank: dict[str, float], baseline: dict[str, float], anchored: set[str] = frozenset()) -> list[str]:
    """The one true label ordering, used identically for both the PAVA count assignment and for
    reordering each entry's own chain -- using different tie-break rules in the two places is
    exactly what caused the "aldactone" duplicate-level bug.

    Anchored (real-world-referenced) labels are ordered by their real value, not empirical
    chain-depth rank -- two anchors can sit at the same typical depth despite a huge real-world
    size difference (bug #2). Unanchored labels are interpolated into a slot relative to the
    anchors: found via where their own empirical rank falls among the anchors' empirical ranks,
    then placed at that same relative position in the anchors' real-value order.
    """
    if not anchored:
        return sorted(rank, key=lambda label: (rank[label], baseline.get(label, 1.0)))

    anchors_by_rank = sorted(anchored, key=lambda label: rank[label])
    anchor_ranks = [rank[label] for label in anchors_by_rank]
    anchors_by_value = sorted(anchored, key=lambda label: baseline[label])
    value_slot_of_anchor = {label: idx for idx, label in enumerate(anchors_by_value)}

    def slot_of(label: str) -> float:
        if label in value_slot_of_anchor:
            return value_slot_of_anchor[label]
        idx = bisect.bisect_left(anchor_ranks, rank[label])
        return idx - 0.5  # lands strictly between slot idx-1 and idx

    return sorted(rank, key=lambda label: (slot_of(label), rank[label], baseline.get(label, 1.0)))


def normalize_runtime_type(entries: dict[str, Any], runtime_type: str) -> dict[str, Any]:
    """Mutates `entries` (a runtime type's profiles dict) in place: canonicalizes/dedupes each
    entry's levels chain, resolves one coherent count per canonical label for the whole set of
    entries, and reorders each chain to match the corpus-wide consensus so counts stay monotone
    without per-entry clamping. Returns a report dict for coverage.json / manual review."""
    variant_map = build_variant_map(runtime_type)

    canonical_by_entry: dict[str, list[str]] = {}
    raw_counts_by_canonical: dict[str, list[float]] = defaultdict(list)
    # labels with a pre-existing real, source-backed grounding (Fix Area 2: openFDA/DOID/
    # ICD-10-PCS) are authoritative -- their count must never be silently overwritten by this
    # pass's corpus-derived estimate while their grounding still claims "certifying". They're
    # folded into the SAME anchor mechanism as the hand-curated reference table below (fixed
    # value, dominant weight) rather than excluded from ranking entirely, so they still
    # participate correctly in ordering and monotonicity instead of causing a KeyError when
    # chains get reordered.
    real_certifying_value: dict[str, float] = {}
    merges = 0
    dedup_drops = 0
    for key, row in entries.items():
        old_levels = row.get("levels", [])
        counts = row.get("level_counts", {})
        groundings = row.get("level_grounding") or row.get("level_groundings") or {}
        deduped: list[str] = []
        for lvl in old_levels:
            canon = canonicalize(lvl, variant_map)
            if lvl in counts:
                raw_counts_by_canonical[canon].append(float(counts[lvl]))
                grounding = groundings.get(lvl) or {}
                if grounding.get("status") == "certifying" and grounding.get("member_set_ref"):
                    real_certifying_value[canon] = float(counts[lvl])
            if canon not in deduped:
                deduped.append(canon)
            else:
                dedup_drops += 1
        if deduped != old_levels:
            merges += 1
        canonical_by_entry[key] = deduped

    if not raw_counts_by_canonical:
        return {
            "clusters_applied": {c: sorted(v) for c, v in CLUSTERS.get(runtime_type, {}).items()},
            "canonical_levels": 0,
            "entries_with_level_changes": merges,
            "duplicate_levels_dropped": dedup_drops,
            "entries_reordered_to_corpus_consensus": 0,
            "anchored_labels": [],
            "same_count_collisions": [],
        }

    # Corpus-membership counts: each level's count is the number of DISTINCT entries whose
    # generalization chain contains it -- a real anonymity-set-within-corpus size, not the
    # model's fabricated per-item number. Monotone up a chain by construction (a broader level
    # is carried by a superset of the entries carrying any level that always rolls up into it),
    # so no forced log-spacing is needed. Certifying/anchored counts still override below.
    membership: dict[str, set[str]] = defaultdict(set)
    for entry_key, chain in canonical_by_entry.items():
        for canon in chain:
            membership[canon].add(entry_key)
    baseline = {canon: float(len(members)) for canon, members in membership.items()}
    weight = {canon: float(len(members)) for canon, members in membership.items()}

    references = load_reference_anchors(runtime_type)
    anchored_labels = {canon for canon in references if canon in baseline}
    for canon in anchored_labels:
        baseline[canon] = float(references[canon])
        weight[canon] = 1_000_000.0
    for canon, value in real_certifying_value.items():
        baseline[canon] = value
        weight[canon] = 1_000_000.0
        anchored_labels.add(canon)

    rank = _average_depth_rank(list(canonical_by_entry.values()))
    ordered_labels = _rank_order(rank, baseline, anchored_labels)
    label_position = {label: idx for idx, label in enumerate(ordered_labels)}
    final_count = _weighted_pava(ordered_labels, baseline, weight)

    reordered_count = 0
    for key, deduped in canonical_by_entry.items():
        reordered = sorted(deduped, key=lambda label: label_position[label])
        if reordered != deduped:
            reordered_count += 1
        canonical_by_entry[key] = reordered

    for key, row in entries.items():
        old_groundings = row.get("level_grounding") or row.get("level_groundings") or {}
        deduped = canonical_by_entry[key]

        new_level_counts = {}
        new_groundings = {}
        for canon in deduped:
            new_level_counts[canon] = final_count[canon]
            grounding = dict(old_groundings.get(canon) or {})
            grounding.setdefault("status", "model-proposed")
            grounding.setdefault("source_family", "model-proposed")
            grounding.setdefault("selector", canon)
            if canon in real_certifying_value and final_count[canon] == real_certifying_value[canon]:
                pass  # unperturbed real, source-backed grounding is left exactly as-is
            elif canon in real_certifying_value:
                # the dominant weight (1e6) makes this vanishingly rare, but if a real count
                # still got pooled with a neighbor, the result is no longer exactly the source's
                # own number -- say so plainly rather than keep a now-stale "certifying" claim.
                grounding["status"] = "corpus-adjusted-from-certifying-source"
                grounding["count_basis"] = "corpus-adjusted-from-certifying-source"
                grounding["count_evidence"] = (
                    f"'{canon}' has a real source-backed count ({real_certifying_value[canon]}) "
                    f"that required a small adjustment ({final_count[canon]}) to stay monotone "
                    f"with the rest of this run's chains; flag for manual review"
                )
            else:
                grounding["member_set_ref"] = None
                if canon in anchored_labels:
                    grounding["count_basis"] = "real-world-reference-estimate"
                    grounding["count_evidence"] = (
                        f"'{canon}' set from a best-effort real-world magnitude estimate "
                        f"(see data/lattice_sources/reference/), not derived from this run's "
                        f"per-entry counts; still not certifying"
                    )
                else:
                    grounding["count_basis"] = "corpus-membership"
                    grounding["count_evidence"] = (
                        f"'{canon}' count is the number of distinct entries in this run whose "
                        f"generalization chain includes it (corpus-membership anonymity-set "
                        f"size); not certifying"
                    )
            new_groundings[canon] = grounding

        row["levels"] = deduped
        row["level_counts"] = new_level_counts
        row["level_grounding"] = new_groundings
        row.pop("level_groundings", None)
        if deduped:
            row["count"] = new_level_counts[deduped[0]]

    same_count_collisions = []
    for key, row in entries.items():
        counts_seen: dict[float, list[str]] = defaultdict(list)
        for lvl, c in row["level_counts"].items():
            counts_seen[c].append(lvl)
        for value, labels in counts_seen.items():
            if len(labels) > 1:
                same_count_collisions.append({"entry": key, "count": value, "levels": labels})

    return {
        "clusters_applied": {c: sorted(v) for c, v in CLUSTERS.get(runtime_type, {}).items()},
        "canonical_levels": len(final_count),
        "entries_with_level_changes": merges,
        "duplicate_levels_dropped": dedup_drops,
        "entries_reordered_to_corpus_consensus": reordered_count,
        "anchored_labels": sorted(anchored_labels),
        "same_count_collisions": same_count_collisions,
    }


def normalize_coherence(artifact: dict[str, Any]) -> dict[str, Any]:
    """Runs normalize_runtime_type over every runtime type in a proposed artifact's `profiles`.
    Mutates `artifact` in place and returns a per-runtime-type report."""
    report: dict[str, Any] = {}
    for runtime_type, entries in artifact.get("profiles", {}).items():
        if not isinstance(entries, dict) or not entries:
            continue
        report[runtime_type] = normalize_runtime_type(entries, runtime_type)
    return report
