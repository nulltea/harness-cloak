import json

import numpy as np
import pytest

from cloak import span_gate
from cloak.profile_match import span_key


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


def _vec(x):  # 2-d unit vectors for fake embeddings
    v = np.asarray(x, dtype=np.float32)
    return v / np.linalg.norm(v)


FAKE_SPACE = {  # surface -> direction; positives near [1,0], junk near [0,1]
    "blorbitis": [1.0, 0.0], "blorb inflammation": [1.0, 0.05],
    "weird fragment": [0.05, 1.0], "brickish thing": [0.1, 1.0],
    "ambiguous middle": [0.7, 0.7],
}


def fake_embed(texts):
    return np.stack([_vec(FAKE_SPACE.get(t, [0.5, 0.5])) for t in texts])


def test_seed_split_is_deterministic_and_disjoint():
    surfaces = [f"surface {i}" for i in range(50)]
    a1, e1 = span_gate.anchor_seed_split(surfaces)
    a2, e2 = span_gate.anchor_seed_split(list(reversed(surfaces)))
    assert set(a1) == set(a2) and set(e1) == set(e2)
    assert set(a1).isdisjoint(e1) and set(a1) | set(e1) == set(surfaces)


def test_gate_link_keep_and_retype(profile, tmp_path):
    got = span_gate.gate_spans(
        [("Blorbitis", "health-condition"), ("flurbectomy", "health-condition")],
        "production", profiles_path=profile,
        negatives_path=tmp_path / "missing.npz",   # layer 3 disabled -> still links
        calibration_path=tmp_path / "missing.json")
    d1 = got[span_key("Blorbitis", "health-condition")]
    assert (d1.action, d1.layer, d1.entry) == ("keep", "link", "blorbitis")
    d2 = got[span_key("flurbectomy", "health-condition")]
    assert (d2.action, d2.new_type) == ("retype", "medical-procedure")


def test_gate_margin_drops_junk_keeps_ambiguous(profile, tmp_path):
    neg = tmp_path / "negatives.npz"
    span_gate.build_negative_index(out_path=neg, embed_fn=fake_embed,
                                   surfaces=["weird fragment"])
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"schema_version": 1, "model_id": "fake",
        "points": {"production": {"floor": 0.6, "margin": 0.2}}}))
    # embindex for the profile with the fake embedder
    from cloak.profile_match import build_embindex
    build_embindex(profile, embed_fn=fake_embed, model_id="fake")
    got = span_gate.gate_spans(
        [("brickish thing", "health-condition"), ("ambiguous middle", "health-condition")],
        "production", profiles_path=profile, negatives_path=neg, calibration_path=calib,
        embed_fn=fake_embed)
    assert got[span_key("brickish thing", "health-condition")].action == "drop"
    assert got[span_key("ambiguous middle", "health-condition")].action == "keep"  # fail-open


def test_gate_fails_open_without_artifacts(profile, tmp_path):
    got = span_gate.gate_spans([("brickish thing", "health-condition")], "production",
                               profiles_path=profile,
                               negatives_path=tmp_path / "none.npz",
                               calibration_path=tmp_path / "none.json")
    d = got[span_key("brickish thing", "health-condition")]
    assert (d.action, d.layer) == ("keep", "open")


def test_denylist_layer_still_fires(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(span_gate, "is_noise_span",
                        lambda s, t: s == "known junk", raising=False)
    got = span_gate.gate_spans([("known junk", "health-condition")], "miner",
                               profiles_path=profile,
                               negatives_path=tmp_path / "none.npz",
                               calibration_path=tmp_path / "none.json")
    assert got[span_key("known junk", "health-condition")].layer == "denylist"
