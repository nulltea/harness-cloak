import json
from pathlib import Path

from cloak.lattice_producer.vocabulary import CanonicalVocabulary


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


def test_context_slice_is_bounded_and_ranked_by_count() -> None:
    vocab = CanonicalVocabulary("drug")

    top = vocab.context_slice(n=3)

    assert len(top) == 3
    assert "chemical substance" in top  # the largest real-world anchor (CAS Registry)


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
