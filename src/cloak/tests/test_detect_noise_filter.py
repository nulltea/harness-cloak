"""Model-free tests for the clinical negative noise filter."""

import pytest

from cloak.detect import Detector, PROFILES, is_noise_span, _stop_words


@pytest.mark.parametrize(
    ("surface", "runtime_type"),
    [
        ("mmr", "drug"),
        ("cad", "health-condition"),
        ("cva", "health-condition"),
        ("proton pump inhibitor", "drug"),
        ("ace inhibitor", "drug"),
    ],
)
def test_noise_filter_keep_allowlist_wins(surface, runtime_type):
    assert is_noise_span(surface, runtime_type) is False


@pytest.mark.parametrize(
    ("surface", "runtime_type", "expected"),
    [
        ("cbc", "medical-procedure", True),
        ("appendectomy", "medical-procedure", False),
        ("mri", "drug", True),
        ("metformin", "drug", False),
        ("ace wrap", "drug", True),
        ("arm", "health-condition", True),
        ("asthma", "health-condition", False),
        ("power of attorney", "drug", True),
        ("coronary artery disease", "health-condition", False),
        ("parkinsons disease", "health-condition", False),
        ("chronic obstructive pulmonary disease", "health-condition", False),
        ("increase", "medical-procedure", False),
        ("lipase", "medical-procedure", True),
        ("amylase", "drug", True),
        ("stent", "drug", True),
        ("pacemaker", "drug", True),
    ],
)
def test_noise_filter_category_examples(surface, runtime_type, expected):
    assert is_noise_span(surface, runtime_type) is expected


def test_noise_filter_fails_open_for_unmatched_real_surface():
    assert is_noise_span("prednisone", "drug") is False


def test_noise_filter_profile_gating():
    assert PROFILES["clinical"].negative_filter is True
    assert PROFILES["reddit"].negative_filter is False


def test_reddit_profile_leaves_noise_category_spans_in():
    det = _fake_detector("reddit")
    spans = det.detect("cbc")
    assert [(s.text, s.type) for s in spans] == [("cbc", "drug")]


def test_clinical_profile_drops_noise_category_spans():
    det = _fake_detector("clinical")
    assert det.detect("cbc") == []


def _fake_detector(profile: str) -> Detector:
    det = Detector.__new__(Detector)
    det.threshold = 0.3
    det.batch_size = 16
    det.fine_dem = False
    det.label2type = {"drug": "drug"}
    det.labels = ["drug"]
    det.max_words = None
    det.gliner = _FakeGliner()
    det.presidio = _FakePresidio()
    det.profile = PROFILES[profile]
    det.stop_words = _stop_words(det.profile)
    return det


class _FakeGliner:
    def batch_predict_entities(self, texts, labels, threshold, batch_size):
        return [[{"start": 0, "end": 3, "text": text[:3], "label": "drug", "score": 0.99}]
                for text in texts]


class _FakePresidio:
    def analyze(self, text, language):
        return []
