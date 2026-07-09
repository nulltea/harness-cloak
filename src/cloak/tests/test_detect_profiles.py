"""DetectorProfile: stop-word and recognizer selection per corpus."""
import pytest

from cloak.detect import PROFILES, _stop_words


def test_reddit_profile_is_status_quo():
    p = PROFILES["reddit"]
    assert p.slang_stop_words and p.custom_recognizers
    sw = _stop_words(p)
    assert "rn" in sw and "ngl" in sw and "she" in sw


def test_clinical_profile_keeps_rn_spans():
    p = PROFILES["clinical"]
    assert not p.slang_stop_words and not p.custom_recognizers
    sw = _stop_words(p)
    assert "rn" not in sw and "ngl" not in sw and "she" in sw


def test_legal_profile_no_slang_keeps_recognizers():
    p = PROFILES["legal"]
    assert not p.slang_stop_words and p.custom_recognizers
    assert "rn" not in _stop_words(p)


def test_unknown_profile_rejected():
    with pytest.raises(KeyError):
        PROFILES["nosuch"]
