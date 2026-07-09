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
    assert "".join(c for _, c in chunks).replace(" ", "") == "a" * 800


def test_max_words_none_is_noop():
    text = "one two three. " * 200
    assert list(_chunks(text)) == list(_chunks(text, max_words=None))
