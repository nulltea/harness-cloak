"""coref_chains: containment/first-token aliasing; no surname-only merges."""
from cloak.detection.detect import Span, coref_chains


def _spans(text, *surfaces):
    out, pos = [], 0
    for surf in surfaces:
        i = text.index(surf, pos)
        out.append(Span(i, i + len(surf), surf, "PERSON", 0.9, "gliner"))
        pos = i + len(surf)
    return out


def test_shared_surname_does_not_merge():
    text = "Anna Smith met Peter Smith"
    spans = coref_chains(text, _spans(text, "Anna Smith", "Peter Smith"))
    assert spans[0].chain != spans[1].chain


def test_containment_merges():
    text = "Anna Smith arrived. Anna spoke."
    spans = coref_chains(text, _spans(text, "Anna Smith", "Anna"))
    assert spans[0].chain == spans[1].chain


def test_first_token_match_merges():
    text = "Anna Smith arrived. Anna S. spoke."
    spans = coref_chains(text, _spans(text, "Anna Smith", "Anna S."))
    assert spans[0].chain == spans[1].chain


def test_bare_trailing_token_joins_most_recent():
    text = "Anna Smith met Peter Smith. Later Smith left."
    spans = coref_chains(text, _spans(text, "Anna Smith", "Peter Smith", "Smith"))
    assert spans[2].chain == spans[1].chain  # most recent Smith chain (Peter's)
    assert spans[0].chain != spans[1].chain


def test_bare_member_is_not_a_containment_bridge():
    # regression: a stored bare "Smith" member must not transitively merge two full names
    text = "Anna Smith testified. Smith spoke. Peter Smith arrived."
    spans = coref_chains(text, _spans(text, "Anna Smith", "Smith", "Peter Smith"))
    assert spans[0].chain == spans[1].chain      # bare mention joins most recent (Anna's)
    assert spans[2].chain != spans[0].chain      # Peter stays a distinct identity


def test_leading_bare_token_does_not_bridge():
    text = "Smith left early. Anna Smith stayed. Peter Smith arrived."
    spans = coref_chains(text, _spans(text, "Smith", "Anna Smith", "Peter Smith"))
    assert spans[1].chain != spans[2].chain      # no merge through the bare seed


def test_different_types_never_merge():
    text = "Smith worked at Smith Hospital"
    s = _spans(text, "Smith")
    i = text.index("Smith Hospital")
    s.append(Span(i, i + len("Smith Hospital"), "Smith Hospital", "ORG", 0.9, "gliner"))
    spans = coref_chains(text, s)
    assert spans[0].chain != spans[1].chain
