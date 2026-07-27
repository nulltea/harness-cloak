"""lattice_for proposal semantics + WordNet demotion for fine/domain types."""
import cloak.lattice.core as cl
from cloak.lattice.profile_match import MatchResult


def _no_wordnet(monkeypatch):
    calls = []
    monkeypatch.setattr(cl, "wordnet_chain", lambda *a, **k: calls.append(a) or None)
    return calls


def test_proposal_levels_used_without_regate(monkeypatch):
    gate_calls = []
    monkeypatch.setattr(cl, "nli_gate", lambda *a, **k: gate_calls.append(a) or [])
    m = MatchResult(["endocrine condition"], "semantic", False, 0.83, "diabetes", nli=0.9)
    got = cl.lattice_for("diabetic", "health-condition", "He is diabetic.", proposal=m)
    assert got == ["endocrine condition", "<HEALTH_CONDITION_1>"]
    assert gate_calls == []          # pre-certified: no re-gate


def test_proposal_none_means_abstained_skips_profile(monkeypatch):
    wn = _no_wordnet(monkeypatch)
    called = []
    import cloak.lattice.profile_match as pm
    monkeypatch.setattr(pm, "match_profile_entry", lambda *a, **k: called.append(a))
    got = cl.lattice_for("unknowniac", "health-condition", "ctx unknowniac.", proposal=None)
    assert got == ["<HEALTH_CONDITION_1>"]  # curated/teacher missed -> placeholder terminal
    assert called == [] and wn == []        # no per-span retry, wordnet diagnostic-only


def test_no_prepass_calls_matcher(monkeypatch):
    import cloak.lattice.profile_match as pm
    m = MatchResult(["media worker"], "semantic", False, 0.8, "journalist", nli=0.7)
    monkeypatch.setattr(pm, "match_profile_entry", lambda *a, **k: m)
    got = cl.lattice_for("journalists", "profession", "They are journalists.")
    assert got == ["media worker", "<PROFESSION_1>"]


def test_fine_curated_fallback_survives(monkeypatch):
    import cloak.lattice.profile_match as pm
    monkeypatch.setattr(pm, "match_profile_entry", lambda *a, **k: None)
    _no_wordnet(monkeypatch)
    got = cl.lattice_for("cardiologist", "profession", "")
    assert got[0] == "medical specialist"   # curated map still first fallback


def test_wordnet_still_feeds_coarse_types(monkeypatch):
    import cloak.lattice.profile_match as pm
    monkeypatch.setattr(pm, "match_profile_entry", lambda *a, **k: None)
    monkeypatch.setattr(cl, "wordnet_chain", lambda *a, **k: ["an institution"])
    got = cl.lattice_for("Some Org", "ORG", "")
    assert "an institution" in got


def test_domain_type_no_wordnet(monkeypatch):
    import cloak.lattice.profile_match as pm
    monkeypatch.setattr(pm, "match_profile_entry", lambda *a, **k: None)
    wn = _no_wordnet(monkeypatch)
    got = cl.lattice_for("colonoscopy", "medical-procedure", "")
    assert got == ["<MEDICAL_PROCEDURE_1>"] and wn == []
