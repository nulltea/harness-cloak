"""_chunks: boundary placement and word-count cap."""
from cloak.detect import _chunks


def _reassemble(text, chunks):
    assert all(text[off:off + len(c)] == c for off, c in chunks)


def test_no_midword_cut():
    # 1195 filler chars then a name: the old code hard-cut inside "Sarah"
    text = "x" * 1195 + " Sarah Johnson was seen"
    chunks = list(_chunks(text))
    _reassemble(text, chunks)
    assert len(chunks) == 2
    # every chunk boundary lands on whitespace: no chunk ends or starts mid-word
    assert chunks[0][1].endswith(" ") or chunks[1][1].startswith(" ") or not chunks[0][1][-1].isalnum() or not chunks[1][1][0].isalnum()
    assert "Sarah Johnson" in chunks[1][1]


def test_sentence_cut_still_preferred():
    text = "y" * 900 + " Seen at the clinic. " + "z" * 400
    chunks = list(_chunks(text))
    _reassemble(text, chunks)
    assert chunks[0][1].rstrip().endswith("clinic.")


def test_unbroken_text_still_terminates():
    # a single 5000-char token cannot be word-preserved; hard cut is acceptable
    text = "x" * 5000
    chunks = list(_chunks(text))
    _reassemble(text, chunks)
    assert sum(len(c) for _, c in chunks) == 5000


def test_max_words_caps_chunks():
    # spaced-out OCR-style text: 800 single-char words in one 1200-char window
    text = " ".join("a" for _ in range(800))
    chunks = list(_chunks(text, max_words=100))
    _reassemble(text, chunks)
    assert all(len(c.split()) <= 100 for _, c in chunks)
    # fallback cuts overlap, so assert full coverage (no character unscanned), not concatenation
    covered = {i for off, c in chunks for i in range(off, off + len(c))}
    assert {i for i, ch in enumerate(text) if ch != " "} <= covered


def test_max_words_none_is_noop():
    text = "one two three. " * 200
    assert list(_chunks(text)) == list(_chunks(text, max_words=None))


def test_fallback_cut_overlaps_so_multiword_entity_survives():
    # cut falls between the name's words; the overlap window must re-present the whole name
    # as clean tokens (prefix ends on a word boundary — a glued token would defeat detection)
    prefix = "word " * 238  # 1190 chars, no sentence breaks
    text = prefix + "Sarah Johnson was seen in clinic today and follow-up was arranged."
    chunks = list(_chunks(text))
    _reassemble(text, chunks)
    assert any(" Sarah Johnson " in " " + c + " " for _, c in chunks)


def test_sentence_cuts_stay_contiguous():
    # overlap applies only to fallback cuts: sentence-broken prose chunks stay back-to-back
    text = "the patient was seen in clinic. " * 60
    chunks = list(_chunks(text))
    _reassemble(text, chunks)
    assert all(chunks[i + 1][0] == chunks[i][0] + len(chunks[i][1]) for i in range(len(chunks) - 1))
