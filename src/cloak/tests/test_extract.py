import sys
from types import SimpleNamespace

import cloak.reward.extract as ex


def test_placeholder_bare_bracket_stripped_is_restored():
    # Remote sometimes echoes placeholders with the angle brackets stripped.
    out, stats = ex.invert(
        "acute exacerbation of HEALTH_CONDITION_2",
        [{"action": "placeholder", "surface": "arthritis", "replacement": "<HEALTH_CONDITION_2>"}],
    )

    assert out == "acute exacerbation of arthritis"
    assert stats["ph_swapped"] == 1
    assert stats["ph_residue"] == 0


def test_semantic_model_loads_pinned_hf_revision(monkeypatch):
    seen = {}

    class FakeSentenceTransformer:
        def __init__(self, model_id, **kwargs):
            seen["model_id"] = model_id
            seen["kwargs"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    ex._semantic_model.cache_clear()
    try:
        ex._semantic_model()
    finally:
        ex._semantic_model.cache_clear()

    assert seen == {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "kwargs": {"revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"},
    }


def test_placeholder_bracketed_form_still_restored():
    out, stats = ex.invert(
        "acute exacerbation of <HEALTH_CONDITION_2>",
        [{"action": "placeholder", "surface": "arthritis", "replacement": "<HEALTH_CONDITION_2>"}],
    )

    assert out == "acute exacerbation of arthritis"
    assert stats["ph_swapped"] == 1
    assert stats["ph_residue"] == 0


def test_placeholder_bare_match_only_from_R_not_generic_pattern():
    # CT_SCAN_1 shapes like a token but is not in R -> untouched; bare CONDITION word must
    # not match either. Only tokens derived from R's replacements may match.
    out, stats = ex.invert(
        "Referral to CT_SCAN_1 and the CONDITION noted with HEALTH_CONDITION_2.",
        [{"action": "placeholder", "surface": "arthritis", "replacement": "<HEALTH_CONDITION_2>"}],
    )

    assert out == "Referral to CT_SCAN_1 and the CONDITION noted with arthritis."
    assert stats["ph_swapped"] == 1


def test_placeholder_inserted_surface_not_reprocessed():
    # Single-pass guarantee: one replacement's SURFACE contains another's bare token;
    # the inserted surface must NOT be rewritten by the second replacement.
    out, stats = ex.invert(
        "<DRUG_1> then check <DRUG_2>",
        [{"action": "placeholder", "surface": "take DRUG_2 daily", "replacement": "<DRUG_1>"},
         {"action": "placeholder", "surface": "aspirin", "replacement": "<DRUG_2>"}],
    )

    assert out == "take DRUG_2 daily then check aspirin"
    assert stats["ph_swapped"] == 2
    assert stats["ph_residue"] == 0


def test_semantic_window_inverts_recoverable_paraphrase(monkeypatch):
    def fake_scores(fill, snippets):
        return [0.9 if s == "that Massachusetts city" else 0.2 for s in snippets]

    monkeypatch.setattr(ex, "_semantic_scores", fake_scores)
    out, stats = ex.invert(
        "She now lives in that Massachusetts city.",
        [{"action": "generalize", "surface": "Boston", "type": "LOC",
          "replacement": "a city in Massachusetts"}],
    )

    assert out == "She now lives in Boston."
    assert stats["gen_semantic"] == 1
    assert stats["gen_absent"] == 0


def test_semantic_window_abstains_on_close_runner_up(monkeypatch):
    monkeypatch.setattr(ex, "_semantic_scores", lambda fill, snippets: [0.9] * len(snippets))

    out, stats = ex.invert(
        "She mentioned that Massachusetts city and another Massachusetts city.",
        [{"action": "generalize", "surface": "Boston", "type": "LOC",
          "replacement": "a city in Massachusetts"}],
    )

    assert out == "She mentioned that Massachusetts city and another Massachusetts city."
    assert stats["gen_semantic"] == 0
    assert stats["gen_absent"] == 1


def test_semantic_window_abstains_on_type_sanity_failure(monkeypatch):
    monkeypatch.setattr(ex, "_semantic_scores", lambda fill, snippets: [0.95] * len(snippets))

    out, stats = ex.invert(
        "The early filing was completed.",
        [{"action": "generalize", "surface": "January 2019", "type": "DATETIME",
          "replacement": "early 2019"}],
    )

    assert out == "The early filing was completed."
    assert stats["gen_semantic"] == 0
    assert stats["gen_absent"] == 1