"""_encoder_max_words: derive the per-model word cap from gliner config."""
from types import SimpleNamespace

from cloak.detect import _encoder_max_words


def test_reads_max_len_with_margin():
    g = SimpleNamespace(config=SimpleNamespace(max_len=768))
    assert _encoder_max_words(g) == 691  # int(768 * 0.9)


def test_missing_max_len_returns_none():
    g = SimpleNamespace(config=SimpleNamespace())
    assert _encoder_max_words(g) is None
