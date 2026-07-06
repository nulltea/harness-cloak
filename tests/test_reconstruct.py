from cloak.reconstruct import linearize_restore_map, splice_at_quote
from cloak.reconstruct import edit_guard

def _res(*pairs): return [{"surface": s, "replacement": r} for s, r in pairs]

def test_linearize_restore_map():
    residue = [{"surface": "arthritis", "replacement": "a disease", "type": "DEM"},
               {"surface": "CABG", "replacement": "a procedure", "type": "MISC"}]
    out = linearize_restore_map(residue)
    assert out == ("DEM: \"a disease\" => \"arthritis\"\n"
                   "MISC: \"a procedure\" => \"CABG\"")

def test_splice_at_quote_case_and_space_insensitive():
    text = "Patient has Early 1980s onset."
    new, ok = splice_at_quote(text, "early  1980S", "January 13th 1982")
    assert ok and new == "Patient has January 13th 1982 onset."

def test_splice_at_quote_absent_returns_unchanged():
    new, ok = splice_at_quote("no match here", "CABG surgery", "coronary bypass")
    assert not ok and new == "no match here"

def test_guard_accepts_in_place_restore():
    prepass = "Patient has a disease and takes a drug."
    cand = "Patient has arthritis and takes a drug."   # replace 'a disease' -> 'arthritis'
    assert edit_guard(prepass, cand, _res(("arthritis", "a disease")), max_edits=3)

def test_guard_rejects_hallucinated_insert():
    prepass = "Patient has a disease."
    cand = "Patient has arthritis in Boston."           # inserted 'arthritis in Boston' not tight
    assert not edit_guard(prepass, cand, _res(("arthritis", "a disease")), max_edits=3)

def test_guard_rejects_insert_elsewhere_with_deletion():
    prepass = "Filed in the early 1980s."
    cand = "January 13th 1982 note. Filed recently."    # pure insert up front + quote deleted
    assert not edit_guard(prepass, cand,
                          _res(("January 13th 1982", "the early 1980s")), max_edits=3)

def test_guard_rejects_wrong_location_replace():
    prepass = "The org filed. Patient has a disease."
    cand = "arthritis filed. Patient has a disease."    # surface put where 'The org' was, not at 'a disease'
    assert not edit_guard(prepass, cand, _res(("arthritis", "a disease")), max_edits=3)

def test_guard_rejects_too_many_edits():
    prepass = "a disease and a drug."
    cand = "arthritis and lasix."                        # 2 valid in-place edits > max_edits=1
    assert not edit_guard(prepass, cand,
                          _res(("arthritis", "a disease"), ("lasix", "a drug")), max_edits=1)

def test_build_target_splices_located_mentions_only():
    from cloak.reconstruct import build_target
    text = "The org filed in Early 1980s."
    # judge verdicts: one located (REWORDED), one abstain (ABSENT quote=None)
    located = [{"surface": "January 13th 1982", "quote": "Early 1980s"},
               {"surface": "Hamilton County Court", "quote": None}]
    tgt, n = build_target(text, located)
    assert tgt == "The org filed in January 13th 1982." and n == 1

def test_value_compatible_fail_closed():
    # security gate: NEVER returns True; False on a digit-run absent from the fill; None otherwise
    from cloak.reconstruct import _value_compatible
    assert _value_compatible("some time ago", "three years ago") is None    # no digits -> defer
    assert _value_compatible("the early 1980s", "late 1990s") is False       # 1990 absent -> reject
    assert _value_compatible("the early 1980s", "Early 1980s") is None        # subset-compatible -> defer

from cloak.reconstruct import reconstruct

class _StubModel:
    def __init__(self, reply): self.reply = reply
    def __call__(self, prompt): return self.reply

# The default cascade (semantic-window matcher, extract.invert) is strong enough to resolve a
# toy "a disease"->"arthritis" itself, leaving no residue — so we stub _rule_prepass at the
# reconstruct boundary to hand it a controlled residue. That exercises reconstruct()'s real
# orchestration (cascade -> model -> edit_guard -> do-no-harm fallback) independent of the
# cascade's matching power; end-to-end cascade integration is covered by the Task 6 eval.
def _stub_prepass(prepass, residue):
    from cloak.extract import _base_stats
    return lambda out_p, R, semantic=True: (prepass, _base_stats(), list(residue))

def test_reconstruct_accepts_guarded_edit(monkeypatch):
    import cloak.reconstruct as rc
    R = [{"action": "generalize", "surface": "arthritis", "replacement": "a disease", "type": "DEM"}]
    monkeypatch.setattr(rc, "_rule_prepass", _stub_prepass("Patient has a disease.", R))
    monkeypatch.setattr(rc, "run_model", lambda m, p: m(p))
    # model restores in-place; only 'arthritis' (an allowed surface) is novel -> guard accepts
    text, stats = reconstruct("Patient has a disease.", R, model=_StubModel("Patient has arthritis."))
    assert "arthritis" in text and stats["gen_reconstructor"] == 1

def test_reconstruct_rejects_hallucinated_edit_falls_back(monkeypatch):
    import cloak.reconstruct as rc
    R = [{"action": "generalize", "surface": "arthritis", "replacement": "a disease", "type": "DEM"}]
    monkeypatch.setattr(rc, "_rule_prepass", _stub_prepass("Patient has a disease.", R))
    monkeypatch.setattr(rc, "run_model", lambda m, p: m(p))
    # model hallucinates 'in Boston' -> edit_guard rejects -> cascade output kept (still 'a disease')
    text, stats = reconstruct("Patient has a disease.", R, model=_StubModel("Patient has arthritis in Boston."))
    assert text == "Patient has a disease." and "Boston" not in text and stats.get("gen_recon_rejected") == 1
