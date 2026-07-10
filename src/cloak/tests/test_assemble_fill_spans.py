"""fill_spans offset bookkeeping in assemble() (extractor migration, task 1).

Each R entry gains fill_spans: [[start, end], ...] — one span per applied occurrence
of that entry's (surface, replacement) pair, in the FINAL doc_p (after _cleanup).
Build invariant: doc_p[start:end] == entry["replacement"] exactly (case-adjusted rep
is what was spliced in).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import train_ranker as tr


def _level_span(surface, typ, start, fill):
    return {"surface": surface, "type": typ, "start": start,
            "actions": [{"mode": "level", "fill": fill, "p6": 0.0, "walk_risk": 0.0}]}


def _check_invariant(doc_p, R):
    for e in R:
        assert "fill_spans" in e, f"missing fill_spans on {e['replacement']!r}"
        for s0, s1 in e["fill_spans"]:
            assert doc_p[s0:s1] == e["replacement"], (
                f"span {(s0, s1)} -> {doc_p[s0:s1]!r} != rep {e['replacement']!r}")


def test_single_replacement():
    text = "Patient has diabetes now"
    R_walk = [{"surface": "diabetes", "type": "health-condition", "action": "generalize",
               "replacement": "a condition", "start": 12, "end": 20,
               "lattice": ["a condition"]}]
    spans = [_level_span("diabetes", "health-condition", 12, "a condition")]
    choice = {"diabetes": spans[0]["actions"][0]}
    doc_p, R = tr.assemble(text, R_walk, spans, choice)
    assert doc_p == "Patient has a condition now"
    (e,) = R
    assert e["fill_spans"] == [[12, 23]]
    _check_invariant(doc_p, R)


def test_multiple_replacements_different_deltas():
    # two surfaces, one grows the string, one shrinks it
    text = "Bob saw Alexandria today"
    R_walk = [
        {"surface": "Bob", "type": "PERSON", "action": "generalize",
         "replacement": "a person named someone", "start": 0, "end": 3,
         "lattice": ["a person named someone"]},
        {"surface": "Alexandria", "type": "GPE", "action": "generalize",
         "replacement": "a city", "start": 8, "end": 18, "lattice": ["a city"]},
    ]
    spans = [_level_span("bob", "PERSON", 0, "a person named someone"),
             _level_span("alexandria", "GPE", 8, "a city")]
    choice = {"bob": spans[0]["actions"][0], "alexandria": spans[1]["actions"][0]}
    doc_p, R = tr.assemble(text, R_walk, spans, choice)
    assert doc_p == "A person named someone saw a city today"
    _check_invariant(doc_p, R)
    by_rep = {e["replacement"]: e["fill_spans"] for e in R}
    assert by_rep["A person named someone"] == [[0, 22]]
    assert by_rep["a city"] == [[27, 33]]


def test_repeated_surface_two_spans():
    text = "cat and cat"
    R_walk = [
        {"surface": "cat", "type": "ANIMAL", "action": "generalize",
         "replacement": "an animal", "start": 0, "end": 3, "lattice": ["an animal"]},
        {"surface": "cat", "type": "ANIMAL", "action": "generalize",
         "replacement": "an animal", "start": 8, "end": 11, "lattice": ["an animal"]},
    ]
    spans = [_level_span("cat", "ANIMAL", 0, "an animal")]
    choice = {"cat": spans[0]["actions"][0]}
    doc_p, R = tr.assemble(text, R_walk, spans, choice)
    # fill is case-adjusted once at the decision span (start 0 -> capitalized), reused for
    # both occurrences; two spans land on the single R entry
    assert doc_p == "An animal and An animal"
    (e,) = R
    assert e["fill_spans"] == [[0, 9], [14, 23]]
    _check_invariant(doc_p, R)


def test_cleanup_shift_left():
    # "the dog" -> "the an animal"; _cleanup drops the duplicate "the ", span shifts left
    text = "the dog barks"
    R_walk = [{"surface": "dog", "type": "ANIMAL", "action": "generalize",
               "replacement": "an animal", "start": 4, "end": 7,
               "lattice": ["an animal"]}]
    spans = [_level_span("dog", "ANIMAL", 4, "an animal")]
    choice = {"dog": spans[0]["actions"][0]}
    doc_p, R = tr.assemble(text, R_walk, spans, choice)
    assert doc_p == "an animal barks"
    (e,) = R
    assert e["fill_spans"] == [[0, 9]]
    _check_invariant(doc_p, R)


def test_mixed_typing_each_entry_own_spans():
    # same surface: occurrence 1 generalized (lattice), occurrence 2 keeps its chain token
    text = "participant A met participant B"
    R_walk = [
        {"surface": "participant", "type": "PERSON", "action": "placeholder",
         "replacement": "<PERSON_1>", "start": 18, "end": 29},  # no lattice -> keeps token
        {"surface": "participant", "type": "PERSON", "action": "generalize",
         "replacement": "a person", "start": 0, "end": 11, "lattice": ["a person"]},
    ]
    spans = [_level_span("participant", "PERSON", 0, "a person")]
    choice = {"participant": spans[0]["actions"][0]}
    doc_p, R = tr.assemble(text, R_walk, spans, choice)
    _check_invariant(doc_p, R)
    by_rep = {e["replacement"]: e["fill_spans"] for e in R}
    assert set(by_rep) == {"A person", "<PERSON_1>"}
    assert len(by_rep["A person"]) == 1
    assert len(by_rep["<PERSON_1>"]) == 1
    # each entry's spans belong only to its own replacement (checked by invariant above)
