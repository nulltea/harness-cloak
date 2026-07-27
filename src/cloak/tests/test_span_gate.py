import json

import pytest

from cloak.detection import span_gate
from cloak.lattice.profile_match import span_key


@pytest.fixture()
def profile(tmp_path):
    artifact = {"schema_version": 1, "created": "2026-07-10", "sources": {}, "profiles": {
        "health-condition": {"blorbitis": {"aliases": ["blorb inflammation"],
                                           "levels": ["organ disease"],
                                           "source_ids": ["t:1"], "count": 10.0}},
        "medical-procedure": {"flurbectomy": {"aliases": [], "levels": ["surgical procedure"],
                                              "source_ids": ["t:2"], "count": 5.0}},
    }}
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps(artifact))
    return p


def test_gate_link_keep_and_retype(profile):
    got = span_gate.gate_spans(
        [("Blorbitis", "health-condition"), ("flurbectomy", "health-condition")],
        "production", profiles_path=profile)
    d1 = got[span_key("Blorbitis", "health-condition")]
    assert (d1.action, d1.layer, d1.entry) == ("keep", "link", "blorbitis")
    d2 = got[span_key("flurbectomy", "health-condition")]
    assert (d2.action, d2.new_type) == ("retype", "medical-procedure")


def test_denylist_layer_still_fires(profile, monkeypatch):
    monkeypatch.setattr(span_gate, "is_noise_span",
                        lambda s, t: s == "known junk", raising=False)
    got = span_gate.gate_spans([("known junk", "health-condition")], "miner",
                               profiles_path=profile)
    assert got[span_key("known junk", "health-condition")].layer == "denylist"


def test_residue_fails_open_by_default_no_nli(profile):
    # a span that misses the links and the deny-list stays keep/open, and no NLI is invoked
    def boom(jobs):
        raise AssertionError("nli_batch_fn must not be called when nli_verify=False")

    got = span_gate.gate_spans(
        [("brickthing", "health-condition", "the patient had brickthing")],
        "production", profiles_path=profile, nli_batch_fn=boom)
    d = got[span_key("brickthing", "health-condition")]
    assert (d.action, d.layer) == ("keep", "open")


def test_nli_layer_drops_unentailed_keeps_entailed(profile):
    seen = []

    def fake_nli(jobs):
        seen.extend(jobs)
        # blorbthing not entailed -> [] ; wobbling entailed -> approved phrase
        return [[] if surface == "blorbthing" else [("a medication", 0.95)]
                for surface, _ctx, _cands in jobs]

    got = span_gate.gate_spans(
        [("blorbthing", "health-condition", "diagnosed with blorbthing"),
         ("wobbling", "drug", "took wobbling twice daily")],
        "production", profiles_path=profile, nli_verify=True, nli_batch_fn=fake_nli)

    dropped = got[span_key("blorbthing", "health-condition")]
    assert (dropped.action, dropped.layer) == ("drop", "nli")
    kept = got[span_key("wobbling", "drug")]
    assert (kept.action, kept.layer) == ("keep", "open")
    # context and the type's substitution phrase reach the NLI call
    by_surface = {j[0]: j for j in seen}
    assert by_surface["blorbthing"][1] == "diagnosed with blorbthing"
    assert by_surface["blorbthing"][2] == ["a medical condition"]
    assert by_surface["wobbling"][2] == ["a medication"]


def test_nli_span_without_context_stays_open(profile):
    # nli_verify on, but no context -> cannot check -> fail open, no job emitted
    def boom(jobs):
        raise AssertionError("no context spans must not reach the NLI call")

    got = span_gate.gate_spans([("brickthing", "health-condition")], "production",
                               profiles_path=profile, nli_verify=True, nli_batch_fn=boom)
    d = got[span_key("brickthing", "health-condition")]
    assert (d.action, d.layer) == ("keep", "open")
