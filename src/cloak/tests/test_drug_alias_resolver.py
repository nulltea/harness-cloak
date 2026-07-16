"""Strict safe-alias resolver: a detected drug brand with no profile entry is aliased onto its
existing generic entry (via openFDA NDC), never inventing entries/levels or resolving ambiguity."""
import json

from cloak import lattice_profiles as lp


_FAKE_NDC = {
    "advil": frozenset({"ibuprofen"}),
    "ibuprofen": frozenset({"ibuprofen"}),
    "protonix": frozenset({"pantoprazole"}),          # generic absent from the test profile
    "tylenol": frozenset({"acetaminophen", "lidocaine"}),  # ambiguous brand -> must decline
}


def _profile(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "profiles": {
            "drug": {
                "ibuprofen": {"aliases": [], "levels": ["nsaid", "medication"]},
                # 'acetaminophen' present but tylenol is ambiguous, so must NOT alias
                "acetaminophen": {"aliases": [], "levels": ["analgesic", "medication"]},
            }
        },
    }))
    return path


def test_resolves_unambiguous_brand_to_existing_generic(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "_ndc_brand_ingredient_index", lambda _p: _FAKE_NDC)
    path = _profile(tmp_path)
    fixed = lp.resolve_missing_drug_aliases(
        ["advil", "protonix", "tylenol"], profile_path=path, ndc_path="x")
    assert fixed == {"advil": "ibuprofen"}                 # only the safe one
    assert lp.lookup_levels("advil", "drug", path) == ["nsaid", "medication"]
    # protonix: generic pantoprazole absent -> untouched; tylenol: ambiguous -> untouched
    assert lp.lookup_entry("protonix", "drug", path) is None
    assert lp.lookup_entry("tylenol", "drug", path) is None
    artifact = json.loads(path.read_text())
    assert "advil" in artifact["profiles"]["drug"]["ibuprofen"]["aliases"]
    assert artifact["profiles"]["drug"]["acetaminophen"]["aliases"] == []


def test_salt_strip_matches_profile_base(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "_ndc_brand_ingredient_index",
                        lambda _p: {"norvasc": frozenset({"amlodipine"})})
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"schema_version": 1, "profiles": {"drug": {
        "amlodipine besylate": {"aliases": [], "levels": ["calcium channel blocker"]}}}}))
    fixed = lp.resolve_missing_drug_aliases(["norvasc"], profile_path=path, ndc_path="x")
    assert fixed == {"norvasc": "amlodipine besylate"}     # salt-stripped base matches


def test_drug_base_ingredient_strips_salts():
    assert lp._drug_base_ingredient("Pantoprazole Sodium") == "pantoprazole"
    assert lp._drug_base_ingredient("atorvastatin trihydrate") == "atorvastatin"
    assert lp._drug_base_ingredient("acetaminophen") == "acetaminophen"
