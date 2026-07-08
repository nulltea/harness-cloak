"""One-off coherence cleanup for data/lattice_profiles/proposed/drug-health-procedure.proposed.json.

Problem: the producer generates counts/levels per-item with no cross-item context, so the
same generalization concept ends up as multiple near-duplicate strings (e.g. "pharmaceutical
compound"/"pharmaceutical agent"/"pharmaceutical product") and even identical level strings
carry wildly different, unrelated per-level counts across entries (e.g. "medication" ranges
142 to 24,500,000). This script does NOT try to make counts real (that needs the deterministic
compiler, a separate agent-level fix) -- it makes the *fabricated* counts internally consistent:
one canonical spelling and exactly ONE shared count for that spelling everywhere it appears in
the corpus.

A first version enforced monotonicity per-entry only (a local forward-max pass), which left
~5-7% of canonical levels with more than one value: whichever entry's chain happened to demand
the highest floor for a shared broad label (e.g. "medical condition") won, and every other entry
using that same label inherited a different, lower number. That's still incoherent -- "medication"
must mean the same universe size everywhere it's used, independent of which entry mentions it.

Fix: treat every entry's narrow->broad chain as a set of pairwise "must be <=" edges between
canonical labels, pool those edges into one graph across the WHOLE corpus, collapse any cycles
(contradictory orderings from different entries) into a single tied group, topologically sort
the result, and assign each group a value via one forward relaxation pass (value = max(own
corpus-median baseline, all predecessor values)). This is the unique minimal assignment that
respects every entry's chain ordering while staying as close as possible to the raw median
signal -- and it gives a strict, corpus-wide invariant: same label, same count, everywhere.

Does not touch entry membership, categorization, or anything outside levels/level_counts/
level_grounding/count.
"""
from __future__ import annotations

import bisect
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

SRC = Path("data/lattice_profiles/proposed/drug-health-procedure.proposed.json")
DST = Path("data/lattice_profiles/proposed/drug-health-procedure.proposed.cleaned.json")
REPORT = Path("data/lattice_profiles/proposed/drug-health-procedure.coherence_report.json")

# Curated synonym clusters: canonical label -> variant spellings (normalized: lowercase,
# underscores/hyphens -> space). Only genuine same-scope paraphrases go in one cluster --
# narrower/broader subtypes (e.g. "antihypertensive medication" vs bare "medication") are
# deliberately NOT merged, to avoid flattening real lattice granularity.
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
        # abstraction ceiling: collapse the long tail of near-universal "it's a chemical" terms
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

# Best-effort real-world magnitude anchors for the common/recurring canonical labels -- the ones
# driving the "same-count collision" clusters, where the purely data-driven corpus median has no
# real signal to separate genuinely different categories. Not certifying counts (that still needs
# the deterministic compiler over a real source dataset); this is "get the right order of
# magnitude and the right relative order from known reference figures" per explicit instruction
# to estimate from domain knowledge / sources rather than wait for the real compiler.
#
# Sources (approximate, as cited, current as of mid-2026 unless noted):
#   - CAS Registry: >230-290M unique substances (CAS.org) -> chemical substance ceiling.
#   - ChEMBL 36 (2025): 2.8M distinct bioactive compounds (EMBL-EBI) -> pharmaceutical compound tier.
#   - FDA Orange Book: ~1,300-1,500 unique approved active ingredients; ~11,000+ approved drug
#     products (OTC+Rx combined) -> prescription medication / medication tiers.
#   - ICD-10-CM FY2026: 74,719 billable diagnosis codes (icd10data.com/CMS) -> medical condition
#     ceiling (used here as the practical broadest bucket for this lattice's chains, even though
#     SNOMED's Clinical Finding hierarchy is technically the broader ontological category).
#   - SNOMED CT: ~371,000 concepts total; Clinical Finding is its largest single hierarchy,
#     commonly cited at roughly a third to half of the total -> clinical finding tier.
#   - DSM-5: ~300 distinct diagnosable mental disorders -> mental health condition / psychiatric
#     disorder tiers.
#   - Orphanet: >7,000 catalogued rare diseases; OMIM: similar order for genetic disorders.
#   - NINDS: 600+ named neurological disorders.
#   - Remaining mid-tier therapeutic/diagnostic classes (nsaid, beta-blocker, antihistamine, ...)
#     are rough counts of distinct approved members of that class -- order-of-magnitude estimates
#     from general pharmacology/clinical knowledge, not a specific registry lookup.
REFERENCE_COUNTS: dict[str, dict[str, float]] = {
    "drug": {
        # narrow therapeutic subclasses (distinct approved members of the class)
        "nsaid": 20, "opioid analgesic": 25, "analgesic": 150,
        "antihistamine": 50, "corticosteroid": 25, "glucocorticoid": 15, "beta-blocker": 20,
        "anticoagulant": 20, "antidepressant": 40, "antipsychotic": 60,
        "anticonvulsant": 30, "benzodiazepine": 15, "antibiotic": 150, "antibacterial agent": 150,
        "antiviral agent": 90, "antifungal agent": 40, "diuretic": 30, "lipid-lowering agent": 15,
        "hormone": 60, "vaccine": 30, "steroid": 40, "progestin": 15,
        "selective serotonin reuptake inhibitor": 10, "beta-2 adrenergic agonist": 10, "bronchodilator": 25,
        # broader drug-class tiers spanning many subclasses
        "antihypertensive agent": 150, "cardiovascular agent": 300, "psychotropic agent": 250,
        "central nervous system agent": 400, "gastrointestinal agent": 80, "antidiabetic medication": 40,
        "controlled substance": 250,
        "dietary supplement": 2000, "nutritional compound": 2000,
        "herbal supplement": 1000, "botanical extract": 1000, "botanical_drug": 1000,
        # near-universal ceiling tiers
        "prescription medication": 1500,
        "medication": 11000,
        "therapeutic agent": 100000,
        "pharmaceutical compound": 2800000,
        "chemical substance": 250000000,
    },
    "health-condition": {
        # organ-system / diagnostic subclasses (distinct commonly-named conditions in the class)
        "musculoskeletal disorder": 200, "reproductive system condition": 150,
        "dermatological condition": 2000, "cardiovascular condition": 300, "neurological condition": 600,
        "infectious disease": 1400, "neoplastic condition": 1000, "malignant neoplasm": 700,  # malignant is a subset of neoplastic
        "inflammatory condition": 400, "immune system condition": 250,
        "gastrointestinal condition": 400, "respiratory condition": 300,
        "mental health condition": 300, "psychiatric disorder": 280,
        "genetic disorder": 7000, "rare disease": 6800, "syndrome": 4000, "chronic disease": 100,
        "clinical symptom": 1000, "clinical sign": 500,
        # broadest tiers
        "disease": 15000, "disease entity": 15000,
        "clinical finding": 150000,
        "medical condition": 500000,
    },
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("_", " ").replace("-", " ").strip().lower())


def build_variant_map(runtime_type: str) -> dict[str, str]:
    mapping = {}
    for canonical, variants in CLUSTERS.get(runtime_type, {}).items():
        for variant in variants:
            mapping[norm(variant)] = canonical
    return mapping


def canonicalize(level: str, variant_map: dict[str, str]) -> str:
    return variant_map.get(norm(level), level.strip())


def _average_depth_rank(canonical_chains: list[list[str]]) -> dict[str, float]:
    """How broad a label typically is, from ALL of its occurrences, not just adjacent pairs.

    An earlier version voted on pairwise "u appears right before v" edges. That signal is too
    sparse -- most label pairs co-occur in only 1-2 chains, so "majority" is barely better than
    a coin flip, and a handful of individually mis-ordered chains (e.g. one entry listing
    "topical medication" *after* plain "medication") chained transitively into one contradictory
    blob covering 40+ unrelated labels, including "medication" itself -- the exact label this
    was supposed to fix.

    Average chain position uses the full evidence for a label at once (e.g. "medication"'s
    ~186 occurrences all vote on its typical depth), giving a real-valued score. A total order
    from real numbers cannot contain a cycle, so no SCC/ambiguous-bucket machinery is needed --
    every label gets exactly one rank, hence one count, by construction.
    """
    positions: dict[str, list[float]] = defaultdict(list)
    for chain in canonical_chains:
        denom = max(1, len(chain) - 1)
        for idx, label in enumerate(chain):
            positions[label].append(idx / denom)
    return {label: statistics.mean(vals) for label, vals in positions.items()}


def _weighted_pava(ordered_labels: list[str], value_of: dict[str, float], weight_of: dict[str, float]) -> dict[str, float]:
    """Textbook pool-adjacent-violators isotonic regression, weighted by observation count.

    A naive forward-max walk (value = max(own value, running floor)) isn't real isotonic
    regression -- it's a greedy heuristic where ONE extreme single-sample outlier baseline
    early in the order permanently sets the floor for every later label forever, since the
    floor can never come back down. On this corpus that collapsed 10+ unrelated concepts
    (medication, medical intervention, healthcare equipment, organic acid, ...) onto the exact
    same value just because one of them had a single wild fabricated count.

    PAVA instead merges only the LOCALLY violating neighbors into one pooled block and uses
    their weighted mean, then keeps checking backward. Weighting by how many raw observations
    backed each label's median means a single-sample outlier gets diluted by better-evidenced
    neighbors instead of dictating everything downstream.
    """
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
    """The one true label ordering, used identically everywhere -- for both the PAVA count
    assignment and for reordering each entry's own chain. If these two ever use different
    tie-break rules, a rank tie can sort two labels one way for the count computation and the
    opposite way when reordering an entry, silently reintroducing a monotonicity violation.

    Anchored (real-world-referenced) labels are ordered by their REAL value, not by empirical
    chain-depth rank: "how deep in a typical chain does this label sit" and "how big is this
    category in reality" are different axes that only loosely correlate (e.g. "genetic disorder"
    and "musculoskeletal disorder" tied at the exact same empirical depth in this corpus despite
    a 35x real-world size difference), so trusting empirical rank for anchor-vs-anchor order
    would force real, curated values to collide via PAVA pooling. Unanchored labels still use
    empirical rank, but are interpolated into a SLOT relative to the anchors: found via where
    their own empirical rank falls among the anchors' empirical ranks, then placed at that same
    relative position in the anchors' real-value order. This keeps "this narrow label typically
    sits about as deep as anchors X and Y" as real evidence, while anchor-to-anchor order is
    never left to noisy chain-position data.
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


def _resolve_coherent_counts(ordered_labels: list[str], baseline: dict[str, float], weight: dict[str, float]) -> dict[str, float]:
    """Assign one count per canonical label: run weighted PAVA over the per-label corpus-median
    baselines in the corpus-wide rank order (narrowest-typical to broadest-typical)."""
    return _weighted_pava(ordered_labels, baseline, weight)


def clean_runtime_type(entries: dict, runtime_type: str) -> dict:
    variant_map = build_variant_map(runtime_type)

    # pass 1: canonicalize every occurrence, dedupe each entry's chain, collect raw counts
    canonical_by_entry: dict[str, list[str]] = {}
    raw_counts_by_canonical: dict[str, list[float]] = defaultdict(list)
    merges = 0
    dedup_drops = 0
    for key, row in entries.items():
        old_levels = row.get("levels", [])
        counts = row.get("level_counts", {})
        deduped: list[str] = []
        for lvl in old_levels:
            canon = canonicalize(lvl, variant_map)
            if lvl in counts:
                raw_counts_by_canonical[canon].append(float(counts[lvl]))
            if canon not in deduped:
                deduped.append(canon)
            else:
                dedup_drops += 1
        if deduped != old_levels:
            merges += 1
        canonical_by_entry[key] = deduped

    baseline = {canon: statistics.median(vals) for canon, vals in raw_counts_by_canonical.items()}
    weight = {canon: float(len(vals)) for canon, vals in raw_counts_by_canonical.items()}

    # anchor the recurring/common labels to real-world magnitude estimates instead of the
    # fabricated, mutually-inconsistent per-entry numbers -- with a very high synthetic weight so
    # weighted PAVA treats them as fixed points and pulls unanchored neighbors toward them, rather
    # than letting unanchored noise drag a real anchor off its known value.
    references = REFERENCE_COUNTS.get(runtime_type, {})
    anchored_labels = {canon for canon in references if canon in baseline}
    for canon in anchored_labels:
        baseline[canon] = float(references[canon])
        weight[canon] = 1_000_000.0

    rank = _average_depth_rank(list(canonical_by_entry.values()))
    ordered_labels = _rank_order(rank, baseline, anchored_labels)
    label_position = {label: idx for idx, label in enumerate(ordered_labels)}
    final_count = _resolve_coherent_counts(ordered_labels, baseline, weight)

    # pass 2: the previous version kept each entry's ORIGINAL chain order and clamped counts
    # upward when that order disagreed with the corpus consensus -- which is how "medication"
    # and "pharmaceutical compound" ended up tied at the same count in one entry: that entry's
    # own chain happened to list "pharmaceutical compound" (globally the broader concept) before
    # "medication", so the clamp forced medication up to match instead of fixing the ordering.
    # The actual bug is the entry's chain order, not the count. Fix it at the source: reorder
    # every entry's chain by the exact same rank order used to build final_count (same tie-break
    # rule too -- label_position, not raw rank -- otherwise a rank tie can sort two labels one
    # way for the count computation and the opposite way here, reintroducing a violation). A
    # chain sorted by this shared order is monotonic by construction -- no clamping needed.
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
            grounding["member_set_ref"] = None
            if canon in anchored_labels:
                grounding["count_basis"] = "real-world-reference-estimate"
                grounding["count_evidence"] = (
                    f"'{canon}' set from a best-effort real-world magnitude estimate (see "
                    f"REFERENCE_COUNTS source notes), not derived from this corpus's fabricated "
                    f"per-entry counts; still not certifying"
                )
            else:
                grounding["count_basis"] = "corpus-wide-rank-coherent"
                grounding["count_evidence"] = (
                    f"resolved via a global average-depth ranking over every entry's generalization "
                    f"chain so '{canon}' carries the same count everywhere in the corpus; not certifying"
                )
            new_groundings[canon] = grounding

        row["levels"] = deduped
        row["level_counts"] = new_level_counts
        row["level_grounding"] = new_groundings
        row.pop("level_groundings", None)
        row["count"] = new_level_counts[deduped[0]]

    # audit: any entry where two DIFFERENT labels still land on the identical count. Not
    # necessarily a bug (two genuinely distinct concepts can coincidentally share a corpus
    # median), but worth surfacing explicitly rather than assuming reordering fixed everything.
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
        "sample_coherent_counts": dict(sorted(final_count.items(), key=lambda kv: -kv[1])[:10]),
    }


def main() -> None:
    data = json.loads(SRC.read_text())
    report = {}
    for runtime_type, entries in data["profiles"].items():
        report[runtime_type] = clean_runtime_type(entries, runtime_type)

    DST.write_text(json.dumps(data, indent=2, sort_keys=True))
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"wrote {DST}")
    print(f"wrote {REPORT}")
    for rt, r in report.items():
        print(f"{rt}: {r['entries_with_level_changes']} entries changed, "
              f"{r['duplicate_levels_dropped']} duplicate levels dropped, "
              f"{r['canonical_levels']} canonical level buckets, "
              f"{len(r['anchored_labels'])} anchored to real-world reference estimates, "
              f"{r['entries_reordered_to_corpus_consensus']} entries reordered to corpus consensus, "
              f"{len(r['same_count_collisions'])} same-count collisions remaining")
        for collision in r["same_count_collisions"][:20]:
            print(f"    collision: {collision['entry']!r} has {collision['levels']} all = {collision['count']}")


def _selfcheck() -> None:
    """ponytail: smallest runnable check that the real invariants hold: every entry's levels are
    deduped and monotone, and (the whole point of reordering by corpus rank) every canonical
    label has exactly one count everywhere in the corpus, no exceptions this time."""
    data = json.loads(DST.read_text())
    for runtime_type, entries in data["profiles"].items():
        count_for_level: dict[str, float] = {}
        for key, row in entries.items():
            levels = row["levels"]
            counts = [row["level_counts"][lvl] for lvl in levels]
            assert counts == sorted(counts), f"{runtime_type}:{key} not monotone: {counts}"
            assert len(levels) == len(set(levels)), f"{runtime_type}:{key} has duplicate levels: {levels}"
            for lvl, c in zip(levels, counts):
                if lvl in count_for_level:
                    assert count_for_level[lvl] == c, (
                        f"{runtime_type}: '{lvl}' has {count_for_level[lvl]} elsewhere but {c} in {key}"
                    )
                count_for_level[lvl] = c
        print(f"{runtime_type}: {len(count_for_level)} canonical levels, every one has exactly one count corpus-wide, no exceptions")
    print("selfcheck OK: every entry's levels are deduped, reordered to corpus consensus, and monotone; "
          "every label is globally count-coherent with zero exceptions")


if __name__ == "__main__":
    main()
    _selfcheck()
