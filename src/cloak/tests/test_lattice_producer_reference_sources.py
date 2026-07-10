import json
import zipfile
from pathlib import Path

from cloak.lattice_producer.reference_sources import (
    load_doid_index,
    load_icd10pcs_index,
    load_openfda_pharm_class_index,
    lookup_doid_reference,
    lookup_icd10pcs_reference,
    lookup_openfda_reference,
    reference_candidates_for,
)


def _write_ndc_zip(path: Path, records: list[dict]) -> None:
    payload = json.dumps({"meta": {}, "results": records})
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("drug-ndc-0001-of-0001.json", payload)


def test_openfda_indexes_single_ingredient_only(tmp_path: Path) -> None:
    zip_path = tmp_path / "ndc.zip"
    _write_ndc_zip(
        zip_path,
        [
            {
                "generic_name": "bupropion hydrochloride",
                "active_ingredients": [{"name": "BUPROPION HYDROCHLORIDE", "strength": "150 mg"}],
                "pharm_class": ["Aminoketone [EPC]", "Dopamine Uptake Inhibitors [MoA]"],
            },
            {
                # combo product: generic_name names one ingredient, but pharm_class describes
                # the whole combination -- must not be indexed as acetaminophen's own class.
                "generic_name": "acetaminophen",
                "active_ingredients": [
                    {"name": "ACETAMINOPHEN", "strength": "325 mg"},
                    {"name": "DIPHENHYDRAMINE HCL", "strength": "25 mg"},
                ],
                "pharm_class": ["Antihistamine [EPC]"],
            },
            {
                # a molecule whose real EPC tag is literally its own name -- must not surface
                # as a "generalization" of itself.
                "generic_name": "progesterone",
                "active_ingredients": [{"name": "PROGESTERONE", "strength": "100 mg"}],
                "pharm_class": ["Progesterone [EPC]"],
            },
        ],
    )

    index = load_openfda_pharm_class_index(str(zip_path))
    assert "bupropion" in index
    assert index["bupropion"].tag == "Aminoketone [EPC]"
    assert "acetaminophen" not in index  # combo product excluded
    assert "progesterone" in index  # indexed, but excluded at lookup time (self-leak)

    assert lookup_openfda_reference({"surface": "bupropion", "aliases": []}, raw_zip_path=str(zip_path)) == [
        {
            "level": "aminoketone",
            "source_family": "openfda-pharm-class",
            "selector": "openfda_ndc.pharm_class == 'Aminoketone [EPC]'",
            "member_set": frozenset({"bupropion hydrochloride"}),
            "member_set_ref": "openfda-ndc:pharm_class:Aminoketone [EPC]",
        }
    ]
    assert lookup_openfda_reference({"surface": "acetaminophen", "aliases": []}, raw_zip_path=str(zip_path)) is None
    assert lookup_openfda_reference({"surface": "progesterone", "aliases": []}, raw_zip_path=str(zip_path)) is None


def _write_obo(path: Path) -> None:
    path.write_text(
        "format-version: 1.2\n\n"
        "[Term]\n"
        "id: DOID:4\n"
        "name: disease\n\n"
        "[Term]\n"
        "id: DOID:0050117\n"
        "name: disease by infectious agent\n"
        "is_a: DOID:4 ! disease\n\n"
        "[Term]\n"
        "id: DOID:104\n"
        "name: bacterial infectious disease\n"
        "is_a: DOID:0050117 ! disease by infectious agent\n\n"
        "[Term]\n"
        "id: DOID:0050339\n"
        "name: commensal bacterial infectious disease\n"
        "is_a: DOID:104 ! bacterial infectious disease\n\n"
        "[Term]\n"
        "id: DOID:11263\n"
        "name: chlamydia\n"
        "is_a: DOID:0050339 ! commensal bacterial infectious disease\n\n"
        "[Term]\n"
        "id: DOID:99999\n"
        "name: other commensal disease\n"
        "is_a: DOID:0050339 ! commensal bacterial infectious disease\n\n"
    )


def test_doid_walks_is_a_chain_and_treats_root_as_ceiling(tmp_path: Path) -> None:
    obo_path = tmp_path / "doid.obo"
    _write_obo(obo_path)

    nodes = load_doid_index(str(obo_path))
    assert nodes["DOID:11263"].name == "chlamydia"

    chain = lookup_doid_reference({"surface": "chlamydia", "aliases": []}, obo_path=str(obo_path))

    assert [row["level"] for row in chain] == [
        "commensal bacterial infectious disease",
        "bacterial infectious disease",
        "disease by infectious agent",
    ]
    # descendant counts are real graph computations: DOID:0050339 has two children
    # (chlamydia, other commensal disease) plus itself.
    assert len(chain[0]["member_set"]) == 3
    assert chain[0]["member_set_ref"] == "doid:is_a_descendants:DOID:0050339"
    # "disease" (DOID:4), the ontology root, must never appear as a chain rung.
    assert "disease" not in [row["level"] for row in chain]
    assert all(row["source_family"] == "doid-is-a" for row in chain)


def test_doid_returns_none_when_no_match(tmp_path: Path) -> None:
    obo_path = tmp_path / "doid.obo"
    _write_obo(obo_path)

    assert lookup_doid_reference({"surface": "completely unrelated condition", "aliases": []}, obo_path=str(obo_path)) is None


OBO_FIXTURE = """format-version: 1.2

[Term]
id: DOID:0000001
name: blorbitis
synonym: "blorb inflammation" EXACT []
synonym: "blorby feeling" RELATED []
is_a: DOID:0000009 ! organ disease

[Term]
id: DOID:0000002
name: old blorbitis
is_obsolete: true

[Term]
id: DOID:0000009
name: organ disease
"""


def test_doid_index_parses_exact_synonyms_and_obsolete(tmp_path):
    obo = tmp_path / "mini.obo"
    obo.write_text(OBO_FIXTURE)
    from cloak.lattice_producer.reference_sources import load_doid_index
    nodes = load_doid_index(str(obo))
    assert nodes["DOID:0000001"].exact_synonyms == ["blorb inflammation"]
    assert nodes["DOID:0000001"].obsolete is False
    assert nodes["DOID:0000002"].obsolete is True
    assert nodes["DOID:0000009"].exact_synonyms == []


def _icd10pcs_line(code: str, flag: str, short_desc: str, long_desc: str) -> str:
    return f"00001 {code:<7} {flag} {short_desc:<61}{long_desc}"


def _write_icd10pcs_zip(path: Path) -> None:
    lines = [
        _icd10pcs_line("001", "0", "CNS and Cranial Nerves, Bypass", "Central Nervous System and Cranial Nerves, Bypass"),
        _icd10pcs_line("0016070", "1", "Bypass Cereb Vent to Nasophar, Open", "Bypass Cerebral Ventricle to Nasopharynx, Open Approach"),
        _icd10pcs_line("0016071", "1", "Bypass Cereb Vent to Mastoid, Open", "Bypass Cerebral Ventricle to Mastoid Sinus, Open Approach"),
        _icd10pcs_line("0DB", "0", "Gastrointestinal System, Excision", "Gastrointestinal System, Excision"),
        _icd10pcs_line("0DBJ0ZZ", "1", "Excision Appendix, Open Approach", "Excision of Appendix, Open Approach"),
    ]
    text = "\r\n".join(lines) + "\r\n"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("icd10pcs_order_2026.txt", text)
        zf.writestr("order_addenda_2026.txt", "")


def test_icd10pcs_prefix_index_and_source_id_lookup(tmp_path: Path) -> None:
    zip_path = tmp_path / "icd10pcs.zip"
    _write_icd10pcs_zip(zip_path)

    index = load_icd10pcs_index(str(zip_path))
    assert index["001"][0] == "Central Nervous System and Cranial Nerves, Bypass"
    assert index["001"][1] == frozenset({"0016070", "0016071"})
    # no fabricated 1-character Section tier -- this fixture has no header row for "0"
    assert "0" not in index

    chain = lookup_icd10pcs_reference({"surface": "xray", "source_ids": ["icd10pcs:0016070"]}, zip_path=str(zip_path))
    assert chain == [
        {
            "level": "central nervous system and cranial nerves, bypass",
            "source_family": "icd10pcs-prefix",
            "selector": "icd10pcs.prefix(001)",
            "member_set": frozenset({"0016070", "0016071"}),
            "member_set_ref": "icd10pcs:prefix:001",
        }
    ]


def test_icd10pcs_falls_back_to_exact_description_match(tmp_path: Path) -> None:
    zip_path = tmp_path / "icd10pcs.zip"
    _write_icd10pcs_zip(zip_path)

    chain = lookup_icd10pcs_reference({"surface": "Excision of Appendix, Open Approach", "source_ids": []}, zip_path=str(zip_path))
    assert chain[0]["member_set_ref"] == "icd10pcs:prefix:0DB"

    # no exact match, no source_id -- must not guess.
    assert lookup_icd10pcs_reference({"surface": "some unrelated procedure", "source_ids": []}, zip_path=str(zip_path)) is None


def test_reference_candidates_for_returns_none_for_unregistered_runtime_type() -> None:
    # no loader registered for "profession" -- must not touch any file, and must return None
    # (never an empty list) so callers can cleanly fall through to the next local source.
    assert reference_candidates_for({"runtime_type": "profession", "surface": "teacher"}) is None
