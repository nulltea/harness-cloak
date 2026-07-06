from cloak.reconstruct import linearize_restore_map, splice_at_quote

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
