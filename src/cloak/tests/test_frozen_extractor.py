import copy
import hashlib

import numpy as np

import cloak.frozen_extractor as fx
from cloak.extract import invert


class ToyEncoder:
    def __init__(self, dims=32):
        self.dims = dims

    def encode(self, texts):
        rows = []
        for text in texts:
            vec = np.zeros(self.dims, dtype=np.float64)
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                vec[int.from_bytes(digest[:4], "big") % self.dims] += 1.0
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm else vec)
        return np.vstack(rows)


def _sentences(text, spans):
    return [text[start:end] for start, end in spans]


def test_extractor_pins_include_frozen_model_ids_and_thresholds():
    assert fx.EXTRACTOR_PINS == {
        "models": {
            "encoder": "BAAI/bge-small-en-v1.5",
            "nli": "cross-encoder/nli-deberta-v3-small",
            "mlm": "roberta-base",
        },
        "thresholds": {
            "SIM_MIN": 0.55,
            "ASSIGN_MARGIN": 0.05,
            "PRIOR_WEIGHT": 0.15,
            "NLI_ENTAIL": 0.80,
            "TYPE_ENTAIL": 0.70,
            "PLL_MIN_DELTA": -6.0,
            "EPS_MARGIN": 0.02,
            "CHUNK_MAX_WORDS": 6,
        },
        "type_hypotheses": {},
        "ladder_semver": "0.1.0",
    }


def test_extract_deterministic_only_matches_invert_for_placeholder():
    R = [
        {"action": "placeholder", "surface": "Ada Lovelace", "replacement": "<PERSON_1>",
         "type": "PERSON"}
    ]
    out_p = "<PERSON_1> filed the appeal."

    expected_text, expected_stats = invert(out_p, R)
    text, stats = fx.extract(None, R, out_p)

    assert text == expected_text
    for key, value in expected_stats.items():
        assert stats[key] == value
    assert stats["entries"] == []
    assert stats["extractor_version"] == fx.extractor_version()


def test_extract_deterministic_only_matches_invert_for_exact_fill():
    R = [
        {"action": "generalize", "surface": "Hamilton County Court",
         "replacement": "a county court", "type": "ORG"}
    ]
    out_p = "The order came from a county court."

    expected_text, expected_stats = invert(out_p, R)
    text, stats = fx.extract("The order came from a county court.", R, out_p)

    assert text == expected_text
    for key, value in expected_stats.items():
        assert stats[key] == value
    assert stats["entries"] == []
    assert stats["extractor_version"] == fx.extractor_version()


def test_extract_records_residue_entries_as_no_model_abstains(monkeypatch):
    residue = [
        {"surface": "Boston", "replacement": "a city in Massachusetts", "type": "LOC"},
        {"surface": "January 13th 1982", "replacement": "the early 1980s",
         "type": "DATETIME"},
    ]

    def fake_rule_prepass(out_p, R, *, semantic):
        assert semantic is True
        return out_p, {"gen_absent": 0, "ph_residue": 0}, list(residue)

    monkeypatch.setattr(fx, "_rule_prepass", fake_rule_prepass)

    text, stats = fx.extract(None, [], "Nothing restorable here.", models=None)

    assert text == "Nothing restorable here."
    assert stats["gen_absent"] == 2
    assert stats["entries"] == [
        {"surface": "Boston", "type": "LOC", "outcome": "abstained",
         "reason": "no-models"},
        {"surface": "January 13th 1982", "type": "DATETIME", "outcome": "abstained",
         "reason": "no-models"},
    ]


def test_extractor_version_is_stable_and_changes_when_pin_changes(monkeypatch):
    first = fx.extractor_version()
    second = fx.extractor_version()
    changed = copy.deepcopy(fx.EXTRACTOR_PINS)
    changed["thresholds"]["SIM_MIN"] = 0.56

    monkeypatch.setattr(fx, "EXTRACTOR_PINS", changed)

    assert first == second
    assert first.startswith("fx-")
    assert len(first) == 15
    assert fx.extractor_version() != first


def test_sentence_spans_split_on_punctuation_and_newline_without_whitespace():
    text = " Alpha filed.  Beta appealed?\nGamma closed! "

    spans = fx.sentence_spans(text)

    assert spans == [(1, 13), (15, 29), (30, 43)]
    assert _sentences(text, spans) == [
        "Alpha filed.",
        "Beta appealed?",
        "Gamma closed!",
    ]


def test_align_sentences_identity_maps_diagonal():
    doc_p = "Alpha filed the appeal. Beta argued venue. Gamma closed the case."
    out_p = "Alpha filed the appeal. Beta argued venue. Gamma closed the case."
    encoder = ToyEncoder()

    doc_spans = fx.sentence_spans(doc_p)
    out_spans = fx.sentence_spans(out_p)
    alignment = fx.align_sentences(
        encoder.encode(_sentences(doc_p, doc_spans)),
        encoder.encode(_sentences(out_p, out_spans)),
    )

    assert alignment == [0, 1, 2]


def test_align_sentences_reworded_sentences_align_diagonally():
    doc_p = "Alpha filed the appeal. Beta argued venue. Gamma closed the case."
    out_p = (
        "The appeal was filed by Alpha. Venue was argued by Beta. "
        "The case was closed by Gamma."
    )
    encoder = ToyEncoder()

    doc_spans = fx.sentence_spans(doc_p)
    out_spans = fx.sentence_spans(out_p)
    alignment = fx.align_sentences(
        encoder.encode(_sentences(doc_p, doc_spans)),
        encoder.encode(_sentences(out_p, out_spans)),
    )

    assert alignment == [0, 1, 2]


def test_align_sentences_reordered_pair_uses_best_monotonic_anchors():
    doc_p = "Alpha filed. Beta argued. Gamma closed."
    out_p = "Beta argued. Alpha filed. Gamma closed."
    encoder = ToyEncoder()

    doc_spans = fx.sentence_spans(doc_p)
    out_spans = fx.sentence_spans(out_p)
    alignment = fx.align_sentences(
        encoder.encode(_sentences(doc_p, doc_spans)),
        encoder.encode(_sentences(out_p, out_spans)),
    )

    assert alignment == [1, 2, 2]


def test_align_sentences_deleted_sentence_maps_around_gap():
    doc_p = "Alpha filed the appeal. Beta argued venue. Gamma closed the case."
    out_p = "Alpha filed the appeal. Gamma closed the case."
    encoder = ToyEncoder()

    doc_spans = fx.sentence_spans(doc_p)
    out_spans = fx.sentence_spans(out_p)
    alignment = fx.align_sentences(
        encoder.encode(_sentences(doc_p, doc_spans)),
        encoder.encode(_sentences(out_p, out_spans)),
    )

    assert alignment == [0, 0, 1]


def test_position_bonus_without_fill_span_returns_none():
    doc_spans = fx.sentence_spans("Alpha filed. Beta argued.")
    out_spans = fx.sentence_spans("Alpha filed. Beta argued.")

    assert fx.position_bonus(None, doc_spans, out_spans, [0, 1]) is None


def test_position_bonus_window_covers_aligned_sentence_plus_neighbors():
    doc_p = "Alpha filed the appeal. Beta argued venue. Gamma closed the case."
    out_p = "Opening context. Alpha filed the appeal. Beta argued venue. Gamma closed."
    doc_spans = fx.sentence_spans(doc_p)
    out_spans = fx.sentence_spans(out_p)
    fill_span = [doc_p.index("Beta"), doc_p.index("venue") + len("venue")]
    alignment = [1, 2, 3]

    window = fx.position_bonus(fill_span, doc_spans, out_spans, alignment)

    assert window == (out_spans[1][0], out_spans[3][1])
    assert out_p[window[0]:window[1]] == (
        "Alpha filed the appeal. Beta argued venue. Gamma closed."
    )
