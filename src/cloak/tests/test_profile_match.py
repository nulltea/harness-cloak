import json

import numpy as np

from cloak import profile_match as pm

# Deterministic stub embeddings (dim=3). Unknown strings map to a far vector.
VEC = {
    "diabetes": [1.0, 0.0, 0.0],
    "diabetes mellitus": [0.9, 0.1, 0.0],
    "hypothyroidism": [0.0, 1.0, 0.0],
    "asthma": [0.0, 0.0, 1.0],
    "diabetic": [0.95, 0.05, 0.0],       # alias-adjacent to diabetes
    "endocrine disorder": [1.0, 0.99, 0.0],  # near diabetes, then hypothyroidism
}
FAR = [-1.0, -1.0, -1.0]


def stub_embed(texts):
    return np.array([VEC.get(t, FAR) for t in texts], dtype=np.float32)


def _artifact():
    return {
        "schema_version": 1,
        "created": "2026-07-09",
        "sources": {},
        "profiles": {
            "health-condition": {
                "diabetes": {
                    "aliases": ["diabetes mellitus"],
                    "levels": ["endocrine condition", "chronic condition"],
                    "count": 1000.0,
                },
                "hypothyroidism": {
                    "aliases": [],
                    "levels": ["thyroid condition", "endocrine condition"],
                    "count": 500.0,
                },
                "asthma": {
                    "aliases": [],
                    "levels": ["respiratory condition"],
                    "count": 300.0,
                },
            },
        },
    }


def _build(tmp_path, name):
    pm.load_embindex.cache_clear()
    pp = tmp_path / f"{name}.json"
    pp.write_text(json.dumps(_artifact()))
    ip = pm.build_embindex(pp, embed_fn=stub_embed)
    return pp, ip


def _approve_all(entity, context, levels):
    return list(levels)


def _refuse_all(entity, context, levels):
    return []


def test_exact_hit_is_deterministic_and_skips_embed_and_nli(tmp_path):
    pp, ip = _build(tmp_path, "exact")
    calls = {"embed": 0, "nli": 0}

    def spy_embed(texts):
        calls["embed"] += 1
        return stub_embed(texts)

    def spy_nli(entity, context, levels):
        calls["nli"] += 1
        return list(levels)

    m = pm.match_profile_entry(
        "Diabetes", "health-condition", "Patient has Diabetes.",
        profiles_path=pp, index_path=ip, embed_fn=spy_embed, nli_fn=spy_nli,
    )
    assert m is not None
    assert m.kind == "exact"
    assert m.deterministic is True
    assert m.similarity == 1.0
    assert m.entry is None
    assert m.levels == ["endocrine condition", "chronic condition"]
    assert calls == {"embed": 0, "nli": 0}


def test_semantic_hit_via_variant(tmp_path):
    pp, ip = _build(tmp_path, "semantic")
    m = pm.match_profile_entry(
        "diabetic", "health-condition", "She is diabetic.",
        profiles_path=pp, index_path=ip, embed_fn=stub_embed, nli_fn=_approve_all,
    )
    assert m is not None
    assert m.kind == "semantic"
    assert m.deterministic is False
    assert m.entry == "diabetes"
    assert m.levels == ["endocrine condition", "chronic condition"]
    assert m.similarity >= pm.SIM_FLOOR


def test_all_candidates_refused_returns_none(tmp_path):
    pp, ip = _build(tmp_path, "refused")
    m = pm.match_profile_entry(
        "diabetic", "health-condition", "She is diabetic.",
        profiles_path=pp, index_path=ip, embed_fn=stub_embed, nli_fn=_refuse_all,
    )
    assert m is None


def test_below_sim_floor_returns_none(tmp_path):
    pp, ip = _build(tmp_path, "floor")
    m = pm.match_profile_entry(
        "quarterly earnings report", "health-condition", "The report is due.",
        profiles_path=pp, index_path=ip, embed_fn=stub_embed, nli_fn=_approve_all,
    )
    assert m is None


def test_empty_context_returns_none_without_nli(tmp_path):
    pp, ip = _build(tmp_path, "nocontext")
    called = {"nli": False}

    def spy_nli(entity, context, levels):
        called["nli"] = True
        return list(levels)

    m = pm.match_profile_entry(
        "diabetic", "health-condition", "",
        profiles_path=pp, index_path=ip, embed_fn=stub_embed, nli_fn=spy_nli,
    )
    assert m is None
    assert called["nli"] is False


def test_missing_index_degrades_to_exact_only(tmp_path):
    pm.load_embindex.cache_clear()
    pp = tmp_path / "noindex.json"
    pp.write_text(json.dumps(_artifact()))
    absent = tmp_path / "noindex.embindex.npz"

    # exact still works
    exact = pm.match_profile_entry(
        "diabetes", "health-condition", "Patient has diabetes.",
        profiles_path=pp, index_path=absent, embed_fn=stub_embed, nli_fn=_approve_all,
    )
    assert exact is not None and exact.kind == "exact"

    # semantic falls closed
    sem = pm.match_profile_entry(
        "diabetic", "health-condition", "She is diabetic.",
        profiles_path=pp, index_path=absent, embed_fn=stub_embed, nli_fn=_approve_all,
    )
    assert sem is None


def test_embed_failure_degrades_to_exact_only(tmp_path):
    pp, ip = _build(tmp_path, "embedfail")

    def raising_embed(texts):
        raise RuntimeError("model failed to load")

    # semantic path swallows the embed failure and falls closed
    sem = pm.match_profile_entry(
        "diabetic", "health-condition", "She is diabetic.",
        profiles_path=pp, index_path=ip, embed_fn=raising_embed, nli_fn=_approve_all,
    )
    assert sem is None

    # exact path never touches the embedder, so it still resolves
    exact = pm.match_profile_entry(
        "diabetes", "health-condition", "Patient has diabetes.",
        profiles_path=pp, index_path=ip, embed_fn=raising_embed, nli_fn=_approve_all,
    )
    assert exact is not None and exact.kind == "exact"


def test_profile_hash_mismatch_returns_none(tmp_path):
    pp, ip = _build(tmp_path, "stale")
    # rewrite the profiles file after the index was built -> hash mismatch
    art = _artifact()
    art["profiles"]["health-condition"]["asthma"]["count"] = 999.0
    pp.write_text(json.dumps(art))
    pm.load_embindex.cache_clear()

    m = pm.match_profile_entry(
        "diabetic", "health-condition", "She is diabetic.",
        profiles_path=pp, index_path=ip, embed_fn=stub_embed, nli_fn=_approve_all,
    )
    assert m is None


def test_second_candidate_approved_when_first_refused(tmp_path):
    pp, ip = _build(tmp_path, "second")

    def nli_thyroid_only(entity, context, levels):
        return list(levels) if "thyroid condition" in levels else []

    m = pm.match_profile_entry(
        "endocrine disorder", "health-condition", "She has an endocrine disorder.",
        profiles_path=pp, index_path=ip, embed_fn=stub_embed, nli_fn=nli_thyroid_only,
    )
    assert m is not None
    assert m.entry == "hypothyroidism"
    assert m.levels == ["thyroid condition", "endocrine condition"]


def test_build_embindex_output(tmp_path):
    pp = tmp_path / "out.json"
    pp.write_text(json.dumps(_artifact()))
    ip = pm.build_embindex(pp, embed_fn=stub_embed)

    data = np.load(ip, allow_pickle=False)
    vectors = data["vectors"]
    meta = json.loads(data["meta"].item())

    # every row is L2-normalized
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    assert meta["schema_version"] == 1
    assert meta["dim"] == vectors.shape[1]
    assert meta["model_id"] == pm.DEFAULT_MODEL_ID
    assert len(meta["profile_hash"]) == 64
    assert len(meta["rows"]) == vectors.shape[0] == 4  # diabetes, mellitus, hypothyroidism, asthma

    # rows aligned with vectors: spot-check the canonical diabetes row
    di = next(i for i, r in enumerate(meta["rows"]) if r["source_text"] == "diabetes")
    assert meta["rows"][di]["runtime_type"] == "health-condition"
    assert meta["rows"][di]["canonical"] == "diabetes"
    assert np.allclose(vectors[di], [1.0, 0.0, 0.0], atol=1e-5)
