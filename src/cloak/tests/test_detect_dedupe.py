"""_dedupe: same-type widest, cross-type score-first with a pattern-hit floor."""
from cloak.detect import Span, _dedupe


def _s(start, end, typ, score, source="gliner"):
    return Span(start, end, "x" * (end - start), typ, score, source)


def test_same_type_keeps_widest():
    wide, narrow = _s(0, 20, "PERSON", 0.4), _s(5, 10, "PERSON", 0.95)
    assert _dedupe([narrow, wide]) == [wide]


def test_cross_type_higher_score_wins():
    misc = _s(0, 30, "MISC", 0.31)
    code = _s(10, 21, "CODE", 0.99)
    assert _dedupe([misc, code]) == [code]


def test_presidio_pattern_gets_score_floor():
    # pattern recognizers report fixed low scores (0.4-0.6); they must not lose
    # a cross-type conflict to a mid-confidence gliner span
    misc = _s(0, 30, "MISC", 0.7)
    ssn = _s(10, 21, "CODE", 0.6, source="presidio-pattern")
    assert _dedupe([misc, ssn]) == [ssn]


def test_presidio_spacy_gets_no_floor():
    misc = _s(0, 30, "MISC", 0.7)
    spacy_loc = _s(10, 21, "LOC", 0.6, source="presidio")
    assert _dedupe([misc, spacy_loc]) == [misc]


def test_non_overlapping_all_kept_start_sorted():
    a, b = _s(10, 20, "PERSON", 0.9), _s(0, 5, "CODE", 0.5)
    assert _dedupe([a, b]) == [b, a]
