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


class CountingToyEncoder(ToyEncoder):
    def __init__(self, dims=32):
        super().__init__(dims=dims)
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return super().encode(texts)


def _sentences(text, spans):
    return [text[start:end] for start, end in spans]


class ScriptedNLI:
    def __init__(self, *returns):
        self.returns = list(returns)
        self.calls = []

    def __call__(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        if not self.returns:
            raise AssertionError("scripted NLI exhausted")
        return self.returns.pop(0)


class ScriptedMLM:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = []

    def pll(self, sentence):
        self.calls.append(sentence)
        if not self.scores:
            raise AssertionError("scripted MLM exhausted")
        return self.scores.pop(0)


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
        "type_hypotheses": fx.TYPE_HYPOTHESES,
        "ladder_semver": "0.1.0",
    }


def test_type_hypotheses_cover_coarse_and_fine_runtime_types():
    assert set(fx.TYPE_HYPOTHESES) == {
        "PERSON",
        "CODE",
        "ORG",
        "LOC",
        "DATETIME",
        "QUANTITY",
        "MISC",
        "nationality",
        "ethnicity",
        "religion",
        "profession",
        "age",
        "gender",
        "marital-status",
        "health-condition",
        "sexual-orientation",
        "family-role",
        "demographic-other",
    }
    assert (
        fx.TYPE_HYPOTHESES["health-condition"]
        == "This text mentions a disease, diagnosis, or health condition."
    )


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


def test_candidate_chunks_word_ngrams_keep_offsets_and_skip_stopword_only_chunks():
    out_p = "In the city, Ada filed."

    chunks = fx.candidate_chunks(out_p)

    assert (7, 11, "city") in chunks
    assert (13, 16, "Ada") in chunks
    assert (17, 22, "filed") in chunks
    assert (3, 11, "the city") in chunks
    assert (0, 2, "In") not in chunks
    assert (0, 6, "In the") not in chunks
    assert len({(start, end) for start, end, _ in chunks}) == len(chunks)


def test_score_pairs_batches_once_and_prior_disambiguates_repeated_generic_fills():
    out_p = "a person arrived. a person left."
    first = (out_p.index("a person"), out_p.index("a person") + len("a person"))
    second_start = out_p.rindex("a person")
    second = (second_start, second_start + len("a person"))
    chunks = fx.candidate_chunks(out_p)
    residue = [
        {"surface": "Alpha One", "replacement": "a person", "type": "PERSON"},
        {"surface": "Beta Two", "replacement": "a person", "type": "PERSON"},
    ]
    encoder = CountingToyEncoder()

    scores = fx.score_pairs(residue, chunks, encoder, [first, second])
    assignment = fx.assign(scores, len(residue), chunks)

    assert len(encoder.calls) == 1
    assert encoder.calls[0][:2] == ["a person", "a person"]
    assert encoder.calls[0][2:4] == ["Alpha One", "Beta Two"]
    assert chunks[assignment[0]][:2] == first
    assert chunks[assignment[1]][:2] == second
    assert assignment.abstained == {}


def test_assign_excludes_overlapping_chunk_claims():
    chunks = [
        (0, 5, "Alpha"),
        (0, 11, "Alpha Beta"),
        (12, 16, "Beta"),
    ]
    scores = [
        (0.90, 0, 1),
        (0.80, 1, 0),
        (0.70, 1, 2),
    ]

    assignment = fx.assign(scores, 2, chunks)

    assert dict(assignment) == {0: 1, 1: 2}
    assert assignment.abstained == {}


def test_assign_abstains_sub_sim_min_pairs_as_no_candidate():
    chunks = [(0, 8, "a city")]
    scores = [(fx.EXTRACTOR_PINS["thresholds"]["SIM_MIN"] - 0.01, 0, 0)]

    assignment = fx.assign(scores, 2, chunks)

    assert dict(assignment) == {}
    assert assignment.abstained == {0: "no-candidate", 1: "no-candidate"}


def test_assign_demotes_ambiguous_entry_when_taken_chunk_has_close_claim_elsewhere():
    chunks = [
        (0, 8, "a city"),
        (20, 28, "a city"),
    ]
    scores = [
        (0.80, 0, 0),
        (0.80, 1, 0),
        (0.76, 1, 1),
    ]

    assignment = fx.assign(scores, 2, chunks)

    assert dict(assignment) == {0: 0}
    assert assignment.abstained == {1: "ambiguous"}


def test_verify_rejects_empty_and_placeholder_fills_before_nli():
    nli = ScriptedNLI(("entailment", 0.99))

    assert fx.verify({"replacement": "  ", "surface": "Ada", "type": "PERSON"},
                     "Ada", "Ada arrived.", nli) == (False, "bad-fill")
    assert fx.verify({"replacement": "<PERSON_1>", "surface": "Ada", "type": "PERSON"},
                     "Ada", "Ada arrived.", nli) == (False, "bad-fill")
    assert nli.calls == []


def test_verify_rejects_added_digit_before_nli():
    nli = ScriptedNLI(("entailment", 0.99))

    result = fx.verify(
        {"replacement": "some time ago", "surface": "three years ago", "type": "DATETIME"},
        "three years ago",
        "The event happened three years ago.",
        nli,
    )

    assert result == (False, "added-digit")
    assert nli.calls == []


def test_verify_rejects_failed_correspondence_entailment():
    nli = ScriptedNLI(("neutral", 0.95))

    result = fx.verify(
        {"replacement": "a county court", "surface": "Hamilton County Court", "type": "ORG"},
        "a private company",
        "The order came from a private company.",
        nli,
    )

    assert result == (False, "correspondence")
    assert len(nli.calls) == 1


def test_verify_abstains_correspondence_within_margin_on_either_side():
    threshold = fx.EXTRACTOR_PINS["thresholds"]["NLI_ENTAIL"]
    eps = fx.EXTRACTOR_PINS["thresholds"]["EPS_MARGIN"]

    below = fx.verify(
        {"replacement": "a city", "surface": "Boston", "type": "LOC"},
        "a city",
        "The hearing was in a city.",
        ScriptedNLI(("entailment", threshold - eps)),
    )
    above = fx.verify(
        {"replacement": "a city", "surface": "Boston", "type": "LOC"},
        "a city",
        "The hearing was in a city.",
        ScriptedNLI(("entailment", threshold + eps)),
    )

    assert below == (False, "margin-correspondence")
    assert above == (False, "margin-correspondence")


def test_verify_rejects_failed_type_entailment():
    nli = ScriptedNLI(("entailment", 0.99), ("neutral", 0.95))

    result = fx.verify(
        {"replacement": "a city", "surface": "Boston", "type": "LOC"},
        "a person",
        "The hearing concerned a person.",
        nli,
    )

    assert result == (False, "type")
    assert nli.calls[1][1] == fx.TYPE_HYPOTHESES["LOC"]


def test_verify_abstains_type_within_margin_on_either_side():
    threshold = fx.EXTRACTOR_PINS["thresholds"]["TYPE_ENTAIL"]
    eps = fx.EXTRACTOR_PINS["thresholds"]["EPS_MARGIN"]

    below = fx.verify(
        {"replacement": "a city", "surface": "Boston", "type": "LOC"},
        "a city",
        "The hearing was in a city.",
        ScriptedNLI(("entailment", 0.99), ("entailment", threshold - eps)),
    )
    above = fx.verify(
        {"replacement": "a city", "surface": "Boston", "type": "LOC"},
        "a city",
        "The hearing was in a city.",
        ScriptedNLI(("entailment", 0.99), ("entailment", threshold + eps)),
    )

    assert below == (False, "margin-type")
    assert above == (False, "margin-type")


def test_verify_skips_type_gate_for_unknown_runtime_type():
    nli = ScriptedNLI(("entailment", 0.99))

    result = fx.verify(
        {"replacement": "an attribute", "surface": "classified", "type": "legacy-type"},
        "an attribute",
        "The record contains an attribute.",
        nli,
    )

    assert result == (True, "ok")
    assert len(nli.calls) == 1


def test_verify_rejects_added_proper_noun_absent_from_fill_and_surface():
    nli = ScriptedNLI(("entailment", 0.99), ("entailment", 0.99))

    result = fx.verify(
        {"replacement": "a city", "surface": "Boston", "type": "LOC"},
        "a city in Albany",
        "The hearing was in a city in Albany.",
        nli,
    )

    assert result == (False, "added-proper-noun")


def test_verify_allows_sentence_initial_capitalized_token():
    nli = ScriptedNLI(("entailment", 0.99), ("entailment", 0.99))

    result = fx.verify(
        {"replacement": "the early 1980s", "surface": "January 13th 1982",
         "type": "DATETIME"},
        "Early 1980s",
        "Early 1980s was the relevant period.",
        nli,
    )

    assert result == (True, "ok")


def test_splice_replaces_exact_chunk_span():
    out_p = "The order came from a county court near downtown."
    chunk = (out_p.index("a county court"), out_p.index("near") - 1)

    result = fx.splice(out_p, chunk, "Hamilton County Court")

    assert result == "The order came from Hamilton County Court near downtown."


def test_splice_fixes_indefinite_article_at_boundary():
    out_p = "The witness saw a engineer outside."
    chunk = (out_p.index("engineer"), out_p.index("engineer") + len("engineer"))

    result = fx.splice(out_p, chunk, "architect")

    assert result == "The witness saw an architect outside."


def test_apply_splices_reverts_when_mlm_pll_delta_craters_and_records_fluency():
    out_p = "The hearing was held in a city."
    chunk = (out_p.index("a city"), out_p.index("a city") + len("a city"), "a city")
    residue = [{"surface": "Boston", "replacement": "a city", "type": "LOC"}]
    mlm = ScriptedMLM([-1.0, -8.0])

    text, entries = fx.apply_splices(out_p, residue, [chunk], {0: 0}, mlm)

    assert text == out_p
    assert entries == [
        {"surface": "Boston", "type": "LOC", "outcome": "abstained",
         "reason": "fluency"}
    ]
    assert mlm.calls == [
        "The hearing was held in a city.",
        "The hearing was held in Boston.",
    ]


def test_apply_splices_applies_multiple_splices_right_to_left_without_shift_errors():
    out_p = "The city hired the lawyer after the hearing."
    city_start = out_p.index("city")
    lawyer_start = out_p.index("lawyer")
    chunks = [
        (city_start, city_start + len("city"), "city"),
        (lawyer_start, lawyer_start + len("lawyer"), "lawyer"),
    ]
    residue = [
        {"surface": "New Orleans", "replacement": "city", "type": "LOC"},
        {"surface": "Ada Lovelace", "replacement": "lawyer", "type": "PERSON"},
    ]
    mlm = ScriptedMLM([-1.0, -1.0, -1.0, -1.0])

    text, entries = fx.apply_splices(out_p, residue, chunks, {0: 0, 1: 1}, mlm)

    assert text == "The New Orleans hired the Ada Lovelace after the hearing."
    assert entries == [
        {"surface": "New Orleans", "type": "LOC", "outcome": "spliced", "reason": "ok"},
        {"surface": "Ada Lovelace", "type": "PERSON", "outcome": "spliced",
         "reason": "ok"},
    ]


def test_apply_splices_defers_article_fix_until_pending_left_span_is_safe():
    out_p = "The witness saw a fruit outside."
    article_start = out_p.index("a fruit")
    fruit_start = out_p.index("fruit")
    chunks = [
        (article_start, article_start + len("a"), "a"),
        (fruit_start, fruit_start + len("fruit"), "fruit"),
    ]
    residue = [
        {"surface": "the", "replacement": "a", "type": "MISC"},
        {"surface": "orange", "replacement": "fruit", "type": "MISC"},
    ]
    mlm = ScriptedMLM([-1.0, -1.0, -1.0, -1.0])

    text, entries = fx.apply_splices(out_p, residue, chunks, {0: 0, 1: 1}, mlm)

    assert text == "The witness saw the orange outside."
    assert entries == [
        {"surface": "the", "type": "MISC", "outcome": "spliced", "reason": "ok"},
        {"surface": "orange", "type": "MISC", "outcome": "spliced",
         "reason": "ok"},
    ]
