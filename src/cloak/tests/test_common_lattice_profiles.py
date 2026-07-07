import importlib.util
from pathlib import Path


def _load_common_builder():
    path = Path("scripts/spikes/build_common_lattice_profiles.py")
    spec = importlib.util.spec_from_file_location("build_common_lattice_profiles", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(levels):
    return {"aliases": [], "levels": levels, "source_ids": ["test:row"], "count": 1000.0}


def _row_with_aliases(levels, aliases):
    row = _row(levels)
    row["aliases"] = aliases
    return row


def _row_with_sources(levels, source_ids):
    row = _row(levels)
    row["source_ids"] = source_ids
    return row


def test_common_health_selection_collapses_obscure_allergy_and_cancer_subtypes():
    builder = _load_common_builder()
    entries = {
        "allergy": _row(["immune system condition"]),
        "timothy grass allergy": _row(["immune system condition"]),
        "asthma": _row(["respiratory condition"]),
        "cancer": _row(["neoplastic condition"]),
        "lung cancer": _row(["neoplastic condition"]),
        "left lung cancer": _row(["neoplastic condition"]),
        "rare pediatric syndrome": _row(["syndrome"]),
    }

    selected = builder.select_common_entries("health-condition", entries, limit=250)

    assert "allergy" in selected
    assert "timothy grass allergy" not in selected
    assert "cancer" in selected
    assert "lung cancer" not in selected
    assert "left lung cancer" not in selected
    assert "rare pediatric syndrome" not in selected


def test_common_profile_limit_defaults_to_250_per_category():
    builder = _load_common_builder()
    source = {
        "schema_version": 1,
        "profiles": {
            "profession": {
                f"worker {i}": _row(["professional worker"])
                for i in range(300)
            }
        },
    }

    artifact, report = builder.build_common_artifact(source, limit=250)

    assert len(artifact["profiles"]["profession"]) == 250
    assert report["limit_per_category"] == 250


def test_common_drug_selection_does_not_fill_from_openfda_tail():
    builder = _load_common_builder()
    entries = {
        "amphetamine": _row(["medication"]),
        "amphetamine aspartate": _row(["medication"]),
        "atorvastatin": _row(["medication"]),
        "metformin": _row(["medication"]),
        "lisinopril": _row(["medication"]),
        "pancrelipase": _row(["medication"]),
        "pancrelipase amylase": _row(["medication"]),
        "dove": _row(["medication"]),
        "water": _row(["medication"]),
        "aconitum napellus": _row(["medication"]),
    }

    selected = builder.select_common_entries("drug", entries, limit=250)

    assert set(selected) == {"amphetamine", "atorvastatin", "metformin", "lisinopril", "pancrelipase"}


def test_common_drug_aliases_are_pruned_to_curated_same_drug_aliases():
    builder = _load_common_builder()
    source = {
        "schema_version": 1,
        "profiles": {
            "drug": {
                "acetaminophen": _row_with_aliases(
                    ["medication"],
                    [
                        "apap",
                        "paracetamol",
                        "acetaminophen extra strength",
                        "acetaminophen tablet extended release",
                        "childrens tylenol",
                        "tylenol extra strength",
                        "acetaminophen and ibuprofen",
                        "aspirin",
                        "tylenol pm",
                        "tylenol cold and flu",
                        "aconitum napellus whole",
                    ],
                )
            }
        },
    }

    artifact, _report = builder.build_common_artifact(source, limit=250)

    assert artifact["profiles"]["drug"]["acetaminophen"]["aliases"] == [
        "acetaminophen extra strength",
        "acetaminophen tablet extended release",
        "apap",
        "childrens tylenol",
        "paracetamol",
        "tylenol",
        "tylenol extra strength",
    ]


def test_common_drug_source_ids_are_capped_to_representative_sample():
    builder = _load_common_builder()
    source = {
        "schema_version": 1,
        "profiles": {
            "drug": {
                "acetaminophen": _row_with_sources(
                    ["medication"],
                    [f"openfda-ndc:{i:04d}" for i in range(10, 0, -1)],
                )
            }
        },
    }

    artifact, _report = builder.build_common_artifact(source, limit=250)

    assert artifact["profiles"]["drug"]["acetaminophen"]["source_ids"] == [
        "openfda-ndc:0001",
        "openfda-ndc:0002",
        "openfda-ndc:0003",
        "openfda-ndc:0004",
        "openfda-ndc:0005",
    ]


def test_common_medical_procedures_keep_representatives_not_duplicate_variants():
    builder = _load_common_builder()
    entries = {
        "excision of appendix open approach": _row(["medical and surgical procedure", "medical procedure"]),
        "excision of appendix percutaneous approach": _row(["medical and surgical procedure", "medical procedure"]),
        "computerized tomography (ct scan) of abdomen": _row(["imaging procedure", "medical procedure"]),
        "computerized tomography (ct scan) of abdomen and pelvis": _row(["imaging procedure", "medical procedure"]),
        "magnetic resonance imaging (mri) of brain": _row(["imaging procedure", "medical procedure"]),
        "magnetic resonance imaging (mri) of abdomen": _row(["imaging procedure", "medical procedure"]),
        "plain radiography of chest": _row(["imaging procedure", "medical procedure"]),
        "plain radiography of abdomen": _row(["imaging procedure", "medical procedure"]),
        "inspection of bladder open approach": _row(["medical and surgical procedure", "medical procedure"]),
        "inspection of bladder via natural or artificial opening": _row(["medical and surgical procedure", "medical procedure"]),
        "extraction of products of conception retained open approach": _row(["obstetric procedure", "medical procedure"]),
        "extraction of products of conception low open approach": _row(["obstetric procedure", "medical procedure"]),
        "abortion of products of conception open approach": _row(["obstetric procedure", "medical procedure"]),
        "abortion of products of conception percutaneous approach": _row(["obstetric procedure", "medical procedure"]),
        "drainage of abdomen skin external approach": _row(["medical and surgical procedure", "medical procedure"]),
        "drainage of abdomen skin external approach diagnostic": _row(["medical and surgical procedure", "medical procedure"]),
        "occlusion of abdominal aorta open approach": _row(["medical and surgical procedure", "medical procedure"]),
        "occlusion of abdominal aorta percutaneous approach": _row(["medical and surgical procedure", "medical procedure"]),
        "extirpation of matter from bladder open approach": _row(["medical and surgical procedure", "medical procedure"]),
        "extirpation of matter from bladder percutaneous approach": _row(["medical and surgical procedure", "medical procedure"]),
        "acoustic reflex decay assessment": _row(["rehabilitation procedure", "medical procedure"]),
        "brief tone stimuli assessment": _row(["rehabilitation procedure", "medical procedure"]),
    }

    selected = builder.select_common_entries("medical-procedure", entries, limit=250)

    assert "appendectomy" in selected
    assert selected["appendectomy"]["aliases"] == ["excision of appendix open approach"]
    assert "excision of appendix open approach" not in selected
    assert "excision of appendix percutaneous approach" not in selected
    assert "ct scan" in selected
    assert selected["ct scan"]["aliases"] == ["computerized tomography (ct scan) of abdomen"]
    assert "computerized tomography (ct scan) of abdomen" not in selected
    assert "computerized tomography (ct scan) of abdomen and pelvis" not in selected
    assert "mri" in selected
    assert selected["mri"]["aliases"] == ["magnetic resonance imaging (mri) of brain"]
    assert "magnetic resonance imaging (mri) of brain" not in selected
    assert "magnetic resonance imaging (mri) of abdomen" not in selected
    assert "chest x-ray" in selected
    assert selected["chest x-ray"]["aliases"] == ["plain radiography of chest"]
    assert "plain radiography of chest" not in selected
    assert "plain radiography of abdomen" not in selected
    assert "cystoscopy" in selected
    assert selected["cystoscopy"]["aliases"] == ["inspection of bladder via natural or artificial opening"]
    assert "inspection of bladder via natural or artificial opening" not in selected
    assert "inspection of bladder open approach" not in selected
    assert "cesarean section" in selected
    assert selected["cesarean section"]["aliases"] == ["extraction of products of conception low open approach"]
    assert "extraction of products of conception low open approach" not in selected
    assert "extraction of products of conception retained open approach" not in selected
    assert "abortion" in selected
    assert selected["abortion"]["aliases"] == ["abortion of products of conception open approach"]
    assert "abortion of products of conception open approach" not in selected
    assert "drainage" in selected
    assert selected["drainage"]["aliases"] == ["drainage of abdomen skin external approach"]
    assert "drainage of abdomen skin external approach" not in selected
    assert "occlusion" in selected
    assert selected["occlusion"]["aliases"] == ["occlusion of abdominal aorta open approach"]
    assert "occlusion of abdominal aorta open approach" not in selected
    assert "extirpation" in selected
    assert selected["extirpation"]["aliases"] == ["extirpation of matter from bladder open approach"]
    assert "extirpation of matter from bladder open approach" not in selected
    assert "acoustic reflex decay assessment" not in selected
    assert "brief tone stimuli assessment" not in selected
