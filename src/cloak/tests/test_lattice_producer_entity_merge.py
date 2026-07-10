import json

from cloak.lattice_producer.entity_merge import (
    apply_entity_merge,
    block_pairs,
    doid_surface_index,
    merge_runtime_type,
    row_ontology_id,
)

OBO_FIXTURE = """format-version: 1.2

[Term]
id: DOID:0000001
name: blorbitis
synonym: "blorb inflammation" EXACT []
is_a: DOID:0000009 ! organ disease

[Term]
id: DOID:0000003
name: glimmerosis
synonym: "glimmer syndrome" EXACT []
is_a: DOID:0000009 ! organ disease

[Term]
id: DOID:0000004
name: shared name thing
synonym: "ambiguous surface" EXACT []

[Term]
id: DOID:0000005
name: other shared thing
synonym: "ambiguous surface" EXACT []

[Term]
id: DOID:0000009
name: organ disease
"""


def _row(levels, aliases=(), count=10.0):
    return {"aliases": list(aliases), "levels": list(levels),
            "source_ids": ["t:1"], "count": count}


def _obo(tmp_path):
    p = tmp_path / "mini.obo"
    p.write_text(OBO_FIXTURE)
    return str(p)


def test_surface_index_skips_ambiguous_and_maps_synonyms(tmp_path):
    idx = doid_surface_index(_obo(tmp_path))
    assert idx["blorbitis"] == "DOID:0000001"
    assert idx["blorb inflammation"] == "DOID:0000001"
    assert "ambiguous surface" not in idx          # claimed by two ids -> excluded


def test_row_ontology_id_requires_unanimous_surfaces(tmp_path):
    idx = doid_surface_index(_obo(tmp_path))
    assert row_ontology_id("blorbitis", _row(["organ disease"]), idx) == "DOID:0000001"
    # aliases spanning two ids -> conflicting row, no id
    mixed = _row(["organ disease"], aliases=["glimmer syndrome"])
    assert row_ontology_id("blorbitis", mixed, idx) is None
    assert row_ontology_id("unknownitis", _row(["organ disease"]), idx) is None


def test_block_pairs_identical_levels_and_embedding_neighbors(tmp_path):
    entries = {
        "a": _row(["x", "y"]), "b": _row(["x", "y"]), "c": _row(["z"]),
    }
    assert block_pairs(entries) == {("a", "b")}
    # embedding blocking adds near neighbors even with different levels
    import numpy as np
    vecs = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.99, 0.14]}
    embed = lambda texts: np.array([vecs[t.split(" ; ")[0]] for t in texts])
    got = block_pairs(entries, embed_fn=embed)
    assert ("a", "b") in got and (("a", "c") in got or ("b", "c") in got)


def test_ontology_linked_rows_merge_with_alias_union(tmp_path):
    obo = _obo(tmp_path)
    entries = {
        "blorbitis": _row(["organ disease"], aliases=["blorby"], count=50.0),
        "blorb inflammation": _row(["organ disease"], aliases=["the blorbs"], count=20.0),
        "glimmerosis": _row(["organ disease"], count=5.0),   # sibling: same levels, other id
    }
    idx = doid_surface_index(obo)
    merged, report = merge_runtime_type(
        entries, oracle_index=idx,
        preferred_name={"DOID:0000001": "blorbitis"},
        ontology_synonyms={"DOID:0000001": ["blorb inflammation"]})
    assert "blorbitis" in merged and "blorb inflammation" not in merged
    assert "glimmerosis" in merged                       # sibling never merged
    row = merged["blorbitis"]
    assert set(row["aliases"]) >= {"blorby", "the blorbs", "blorb inflammation"}
    assert row["count"] == 50.0
    assert len(report["merged"]) == 1


def test_ontology_linked_but_levels_differ_goes_to_review(tmp_path):
    obo = _obo(tmp_path)
    entries = {
        "blorbitis": _row(["organ disease"]),
        "blorb inflammation": _row(["tissue disease"]),
    }
    merged, report = merge_runtime_type(entries, oracle_index=doid_surface_index(obo))
    assert set(merged) == set(entries)
    assert len(report["review"]) == 1


def test_gate_merges_unlinked_identical_levels_above_threshold(tmp_path):
    entries = {
        "flurbitis": _row(["organ disease"], count=30.0),
        "flurb disease": _row(["organ disease"], count=10.0),
        "glimmerosis": _row(["organ disease"], count=5.0),
    }
    def gate(sa, sb):
        return 0.99 if {"flurbitis"} & set(sa + sb) and {"flurb disease"} & set(sa + sb) else 0.1
    merged, report = merge_runtime_type(entries, gate_fn=gate, gate_threshold=0.95)
    assert "flurbitis" in merged and "flurb disease" not in merged
    assert "glimmerosis" in merged
    # without a gate the same pair is review-only
    merged2, report2 = merge_runtime_type(entries)
    assert set(merged2) == set(entries)
    assert any(r["why"] == "unlinked identical levels" for r in report2["review"])


def test_gate_never_bridges_distinct_ontology_ids(tmp_path):
    # blorbitis -> DOID:0000001, glimmerosis -> DOID:0000003, flurbitis unlinked; identical
    # levels and a gate that passes EVERY pair. The unlinked bridge row may join one side,
    # but the two ontology-distinct rows must never end in one component.
    obo = _obo(tmp_path)
    entries = {
        "blorbitis": _row(["organ disease"], count=30.0),
        "glimmerosis": _row(["organ disease"], count=20.0),
        "flurbitis": _row(["organ disease"], count=10.0),
    }
    merged, report = merge_runtime_type(
        entries, oracle_index=doid_surface_index(obo),
        preferred_name={"DOID:0000001": "blorbitis", "DOID:0000003": "glimmerosis"},
        gate_fn=lambda sa, sb: 0.99, gate_threshold=0.9)
    assert "blorbitis" in merged and "glimmerosis" in merged
    assert any(r["why"] == "would bridge distinct ontology ids" for r in report["review"])


def test_merged_row_level_counts_take_per_level_max():
    entries = {
        "flurbitis": {**_row(["organ disease"]), "level_counts": {"organ disease": 100.0}},
        "flurb disease": {**_row(["organ disease"]), "level_counts": {"organ disease": 400.0}},
    }
    merged, _ = merge_runtime_type(entries, gate_fn=lambda a, b: 1.0, gate_threshold=0.9)
    (row,) = merged.values()
    assert row["level_counts"] == {"organ disease": 400.0}


def test_apply_entity_merge_reports_duplicate_surface_claims(tmp_path):
    artifact = {"schema_version": 1, "created": "2026-07-10", "sources": {}, "profiles": {
        "LOC": {"springtown": _row(["city"], aliases=["the springs"]),
                "springtown two": _row(["city"], aliases=["the springs"])},
    }}
    report = apply_entity_merge(artifact)   # no oracle, no gate -> nothing merges
    assert set(artifact["profiles"]["LOC"]) == {"springtown", "springtown two"}
    assert report["duplicate_surface_claims"]["LOC"] == [
        {"surface": "the springs", "rows": ["springtown", "springtown two"]}]
