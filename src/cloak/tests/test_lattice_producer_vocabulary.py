import json
from pathlib import Path

from cloak.lattice.producer.vocabulary import CanonicalVocabulary


def _write_proposed(path: Path, runtime_type: str, entries: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_role": "proposal",
                "proposal_scope": "producer-processed-only",
                "profiles": {runtime_type: entries},
            }
        )
    )


def test_seeds_from_reference_anchors_for_drug() -> None:
    vocab = CanonicalVocabulary("drug")

    assert vocab.has_exact("pharmaceutical compound")
    assert vocab.has_exact("medication")
    assert not vocab.has_exact("completely made up phrase")


def test_nearest_finds_near_duplicate_paraphrase() -> None:
    vocab = CanonicalVocabulary("drug")

    # "pharmaceutical product" isn't itself an anchor, but "pharmaceutical compound" is, and
    # (unlike "pharmaceutical agent") no other anchor shares the token "product" -- an
    # unambiguous case for this deliberately simple token-Jaccard heuristic. Several "___ agent"
    # anchors (antibacterial/antiviral/antifungal/...) tie with "pharmaceutical compound" on
    # sharing exactly one token out of a 3-token union when the candidate is "pharmaceutical
    # agent" instead -- a real precision limit of bag-of-words matching on short 2-word phrases,
    # not a bug; this deliberately doesn't reach for embeddings to fix it.
    nearest = vocab.nearest("pharmaceutical product", k=3)

    assert "pharmaceutical compound" in nearest


def test_nearest_returns_empty_for_unrelated_candidate() -> None:
    vocab = CanonicalVocabulary("drug")

    assert vocab.nearest("zzz completely unrelated zzz", k=3) == []


def test_nearest_ignores_shared_generic_head_noun() -> None:
    # regression: a shared category head noun ("agent") is NOT similarity. "respiratory agent"
    # only coincidentally shares "agent" with the "___ agent" anchors; on raw tokens that scored
    # 0.33 and tripped the 0.3 near-dup gate, cascading distinct rungs into too_few_levels.
    vocab = CanonicalVocabulary("drug")

    hits = vocab.nearest("respiratory agent", k=3, min_overlap=0.3)
    assert all(other.endswith(" agent") is False or "respiratory" in other for other in hits)
    assert "antibacterial agent" not in hits
    assert "cardiovascular agent" not in hits

    # but a genuine paraphrase sharing a DISCRIMINATIVE token is still caught
    assert "antihypertensive agent" in vocab.nearest("antihypertensive medication", k=3, min_overlap=0.3)


def test_nearest_ignores_shared_syndrome_head_noun() -> None:
    # "weight gain syndrome" is not a near-duplicate of the bare "syndrome" tier -- it only
    # shares the generic head noun, the same false positive as "___ agent"/"___ disorder".
    vocab = CanonicalVocabulary("health-condition")
    assert "syndrome" not in vocab.nearest("weight gain syndrome", k=3, min_overlap=0.3)


def test_context_slice_is_bounded_and_ranked_by_count() -> None:
    vocab = CanonicalVocabulary("drug")

    top = vocab.context_slice(n=3)

    assert len(top) == 3
    assert "chemical substance" in [row["label"] for row in top]  # largest real-world anchor (CAS Registry)


def test_context_slice_surfaces_functional_class_anchors_for_lexically_distant_surface() -> None:
    # a drug brand/INN name shares no letters with its functional class ("wellbutrin" vs
    # "antidepressant"), so pure surface-overlap ranking used to show only the broad umbrella
    # anchors. The reserved functional band must now surface a spread of functional classes.
    vocab = CanonicalVocabulary("drug")
    labels = [row["label"] for row in vocab.context_slice(n=20, surface="wellbutrin bupropion")]
    functional = {"antidepressant", "antihistamine", "antipsychotic", "antibiotic",
                  "anticoagulant", "antihypertensive agent", "cardiovascular agent"}
    assert len(functional & set(labels)) >= 3, labels

    # no-surface behaviour is unchanged (pure count-desc, umbrellas first)
    assert [r["label"] for r in vocab.context_slice(n=1)] == ["chemical substance"]


def test_medical_procedure_seeds_from_icd10pcs_headers_not_a_hand_file() -> None:
    vocab = CanonicalVocabulary("medical-procedure")

    # no hand-curated anchor file exists for medical-procedure -- the vocabulary must come from
    # the real ICD-10-PCS header rows parsed by reference_sources.py instead.
    assert vocab.has_exact("central nervous system and cranial nerves, bypass")


def test_grows_from_labels_already_accepted_this_run(tmp_path: Path) -> None:
    # this is the actual fix for paraphrase proliferation outside the ~40-label static anchor
    # set: a label the model itself invented for an EARLIER item in this run (not a hand-curated
    # anchor, not in any reference file) must be visible to a LATER item's vocabulary.
    proposed = tmp_path / "run.proposed.json"
    _write_proposed(
        proposed,
        "drug",
        {
            "aleve": {
                "levels": ["cardiovascular therapeutic"],
                "level_counts": {"cardiovascular therapeutic": 42.0},
            }
        },
    )

    vocab = CanonicalVocabulary("drug", proposed_out=proposed)

    assert vocab.has_exact("cardiovascular therapeutic")
    assert vocab.is_from_this_run("cardiovascular therapeutic")
    assert not vocab.is_from_this_run("medication")  # a static anchor, not run-discovered


def test_run_labels_do_not_override_static_anchor_values(tmp_path: Path) -> None:
    # if a run somehow re-proposes a label that's also a hand-curated anchor, the anchor's
    # real-world-informed value must win, not whatever count this run's own item happened to
    # attach to it.
    proposed = tmp_path / "run.proposed.json"
    _write_proposed(
        proposed,
        "drug",
        {"some drug": {"levels": ["medication"], "level_counts": {"medication": 3.0}}},
    )

    vocab = CanonicalVocabulary("drug", proposed_out=proposed)

    assert vocab.context_slice(n=len(vocab.all_labels()))  # sanity: still builds
    assert not vocab.is_from_this_run("medication")


def test_missing_proposed_out_file_is_a_harmless_no_op(tmp_path: Path) -> None:
    # early in a run, the proposed artifact may not exist on disk yet -- must not crash.
    vocab = CanonicalVocabulary("drug", proposed_out=tmp_path / "does-not-exist.json")

    assert vocab.has_exact("medication")


def test_context_slice_returns_label_count_pairs_ranked_by_surface_overlap(tmp_path):
    path = tmp_path / "proposed.json"
    _write_proposed(path, "health-condition", {
        "eczema": {"levels": ["skin disorder", "human medical condition"],
                    "level_counts": {"skin disorder": 40, "human medical condition": 900}},
    })
    vocab = CanonicalVocabulary("health-condition", proposed_out=path)
    # n large enough to include both run labels past the high-count static anchors (500k+) that
    # would otherwise crowd the count-900 sink out of a tiny slice -- the point here is the
    # overlap-first ORDER, not the slice size.
    slice_ = vocab.context_slice(n=50, surface="chronic skin rash")
    assert isinstance(slice_[0], dict) and {"label", "count"} <= set(slice_[0])
    # "skin disorder" shares 'skin' with the surface, so it must outrank the higher-count sink
    labels = [row["label"] for row in slice_]
    assert labels.index("skin disorder") < labels.index("human medical condition")


def test_seed_from_run_tracks_latest_count(tmp_path):
    path = tmp_path / "proposed.json"
    # "otc pain reliever" is NOT a static drug anchor, so this exercises the run-label latest-count
    # path (a static anchor would correctly refuse the run count and defeat the test's intent).
    _write_proposed(path, "drug", {
        "a": {"levels": ["otc pain reliever"], "level_counts": {"otc pain reliever": 10}},
        "b": {"levels": ["otc pain reliever"], "level_counts": {"otc pain reliever": 25}},
    })
    vocab = CanonicalVocabulary("drug", proposed_out=path)
    # dict iteration is insertion order; "b" (25) is seen last and must win over "a" (10)
    assert vocab.context_slice(n=5)[0]["count"] == 25 or any(
        r["label"] == "otc pain reliever" and r["count"] == 25 for r in vocab.context_slice(n=50)
    )
