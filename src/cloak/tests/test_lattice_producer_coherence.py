from cloak.lattice_producer.coherence import normalize_coherence, normalize_runtime_type


def _row(levels, counts, groundings=None, count=None):
    return {
        "aliases": [],
        "levels": levels,
        "source_ids": [],
        "count": count if count is not None else counts[levels[0]],
        "level_counts": dict(counts),
        "level_grounding": groundings or {level: {"status": "model-proposed", "source_family": "model-proposed"} for level in levels},
    }


def test_merges_known_synonym_clusters_and_dedupes() -> None:
    entries = {
        "bupropion": _row(
            ["medication", "pharmaceutical agent", "chemical substance"],
            {"medication": 100.0, "pharmaceutical agent": 500.0, "chemical substance": 900.0},
        ),
        "sertraline": _row(
            ["medication", "pharmaceutical compound"],
            {"medication": 50.0, "pharmaceutical compound": 400.0},
        ),
    }

    normalize_runtime_type(entries, "drug")

    # "pharmaceutical agent" and "pharmaceutical compound" are the same cluster -> one spelling
    assert entries["bupropion"]["levels"] == ["medication", "pharmaceutical compound", "chemical substance"]
    assert entries["sertraline"]["levels"] == ["medication", "pharmaceutical compound"]


def test_same_canonical_label_gets_exactly_one_count_everywhere() -> None:
    entries = {
        "a": _row(["medication", "chemical substance"], {"medication": 142.0, "chemical substance": 900.0}),
        "b": _row(["medication", "chemical substance"], {"medication": 24500000.0, "chemical substance": 5000.0}),
        "c": _row(["medication"], {"medication": 900.0}),
    }

    normalize_runtime_type(entries, "drug")

    medication_values = {entries[k]["level_counts"]["medication"] for k in entries}
    assert len(medication_values) == 1, f"expected one coherent value, got {medication_values}"
    for row in entries.values():
        counts = [row["level_counts"][lvl] for lvl in row["levels"]]
        assert counts == sorted(counts)


def test_anchors_prevent_false_collision_between_unrelated_common_labels() -> None:
    # regression: "genetic disorder" and "musculoskeletal disorder" sat at the exact same
    # empirical chain depth in the real reviewed run despite a ~35x real-world size difference;
    # without anchor-vs-value ordering they'd get pooled into one shared, wrong count.
    entries = {
        "aagenaes syndrome": _row(
            ["syndrome", "genetic disorder"],
            {"syndrome": 10.0, "genetic disorder": 12.0},
        ),
        "sprained ankle": _row(
            ["joint injury", "musculoskeletal disorder"],
            {"joint injury": 8.0, "musculoskeletal disorder": 9.0},
        ),
    }

    normalize_runtime_type(entries, "health-condition")

    genetic = entries["aagenaes syndrome"]["level_counts"]["genetic disorder"]
    musculoskeletal = entries["sprained ankle"]["level_counts"]["musculoskeletal disorder"]
    assert genetic != musculoskeletal
    assert genetic > musculoskeletal  # real-world: genetic disorder (~7000) >> musculoskeletal (~200)


def test_reorders_chain_to_corpus_consensus_instead_of_clamping() -> None:
    # regression ("aldactone" bug): an entry whose own chain lists the broader concept before
    # the narrower one must get reordered (and deduped if that produces a repeat), not clamped
    # into a duplicate-valued mess.
    entries = {
        "typical": _row(
            ["medication", "pharmaceutical compound", "chemical substance"],
            {"medication": 200.0, "pharmaceutical compound": 5000.0, "chemical substance": 90000.0},
        ),
        "backwards": _row(
            ["pharmaceutical compound", "medication", "chemical substance"],
            {"pharmaceutical compound": 300.0, "medication": 150.0, "chemical substance": 80000.0},
        ),
    }

    normalize_runtime_type(entries, "drug")

    for row in entries.values():
        assert len(row["levels"]) == len(set(row["levels"]))
        counts = [row["level_counts"][lvl] for lvl in row["levels"]]
        assert counts == sorted(counts)
    assert entries["backwards"]["levels"].index("medication") < entries["backwards"]["levels"].index("pharmaceutical compound")


def test_real_certifying_count_is_preserved_not_silently_overwritten() -> None:
    entries = {
        "bupropion": _row(
            ["aminoketone", "medication", "chemical substance"],
            {"aminoketone": 9.0, "medication": 100.0, "chemical substance": 900.0},
            groundings={
                "aminoketone": {
                    "status": "certifying",
                    "source_family": "openfda-pharm-class",
                    "member_set_ref": "openfda-ndc:pharm_class:Aminoketone [EPC]",
                },
                "medication": {"status": "model-proposed", "source_family": "model-proposed"},
                "chemical substance": {"status": "model-proposed", "source_family": "model-proposed"},
            },
        ),
        "other": _row(["medication", "chemical substance"], {"medication": 500.0, "chemical substance": 700000.0}),
    }

    normalize_runtime_type(entries, "drug")

    row = entries["bupropion"]
    assert row["level_counts"]["aminoketone"] == 9.0
    assert row["level_grounding"]["aminoketone"]["status"] == "certifying"
    assert row["level_grounding"]["aminoketone"]["member_set_ref"] == "openfda-ndc:pharm_class:Aminoketone [EPC]"
    # the real count must still stay below the run's coherent "medication" value
    assert row["level_counts"]["aminoketone"] <= row["level_counts"]["medication"]


def test_empty_entries_return_zeroed_report_without_crashing() -> None:
    report = normalize_runtime_type({}, "drug")
    assert report["canonical_levels"] == 0
    assert report["same_count_collisions"] == []


def test_normalize_coherence_covers_every_runtime_type_in_artifact() -> None:
    artifact = {
        "profiles": {
            "drug": {"aspirin": _row(["medication"], {"medication": 100.0})},
            "health-condition": {"asthma": _row(["respiratory condition"], {"respiratory condition": 50.0})},
            "profession": {},
        }
    }

    report = normalize_coherence(artifact)

    assert set(report) == {"drug", "health-condition"}
    # "medication" is a real-world anchor (11,000) -- it overrides the fabricated 100.0, which
    # is exactly the point of anchoring, not a bug in this assertion's original expectation.
    assert artifact["profiles"]["drug"]["aspirin"]["level_counts"]["medication"] == 11000.0
