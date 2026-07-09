import json

import numpy as np
import pytest

from cloak import profile_match as pm
from cloak.profile_match import PROFILE_BACKED_TYPES, match_spans_batch, span_key


@pytest.fixture(autouse=True)
def _clear_module_state():
    # _PROPOSAL_CACHE / _WARNED_INDEX_PATHS are module-global; don't leak across tests.
    pm._PROPOSAL_CACHE.clear()
    pm._WARNED_INDEX_PATHS.clear()
    pm.load_embindex.cache_clear()
    yield

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


# --- batch matcher ---

def test_batch_mixed_exact_semantic_abstain(tmp_path):
    profiles, index = _build(tmp_path, "batch_mixed")
    embed_calls = []

    def embed(texts):
        embed_calls.append(list(texts))
        return stub_embed(texts)

    def nli_batch(jobs):
        return [[(lv, 0.9) for lv in cands] if "gib" not in e else []
                for e, ctx, cands in jobs]

    items = [
        ("diabetes", "health-condition", "He has diabetes."),        # exact
        ("diabetic", "health-condition", "He is diabetic."),         # semantic -> diabetes
        ("gibberish zz", "health-condition", "Has gibberish zz."),   # abstain: below floor
        ("diabetes", "health-condition", "dup key, second context"), # dedup: one key
    ]
    got = match_spans_batch(items, profiles_path=profiles, index_path=index,
                            embed_fn=embed, nli_batch_fn=nli_batch)
    assert set(got) == {span_key("diabetes", "health-condition"),
                        span_key("diabetic", "health-condition"),
                        span_key("gibberish zz", "health-condition")}
    assert got[span_key("diabetes", "health-condition")].kind == "exact"
    sem = got[span_key("diabetic", "health-condition")]
    assert sem.kind == "semantic" and sem.entry == "diabetes" and sem.nli == 0.9
    assert got[span_key("gibberish zz", "health-condition")] is None
    assert len(embed_calls) == 1 and sorted(embed_calls[0]) == ["diabetic", "gibberish zz"]


def test_batch_wave2_second_candidate(tmp_path):
    profiles, index = _build(tmp_path, "batch_wave2")
    waves = []

    def nli_batch(jobs):
        waves.append([e for e, _, _ in jobs])
        # refuse everything in the first wave, approve the first level in the second
        return [[] if len(waves) == 1 else [(cands[0], 0.8)] for e, ctx, cands in jobs]

    got = match_spans_batch([("endocrine disorder", "health-condition", "She has an endocrine disorder.")],
                            profiles_path=profiles, index_path=index,
                            embed_fn=stub_embed, nli_batch_fn=nli_batch)
    m = got[span_key("endocrine disorder", "health-condition")]
    assert m is not None and len(waves) == 2   # second-best entry won in wave 2
    assert m.entry == "hypothyroidism"


def test_proposal_cache_skips_embedding(tmp_path):
    profiles, index = _build(tmp_path, "batch_cache")
    embed_calls = []

    def embed(texts):
        embed_calls.append(list(texts))
        return stub_embed(texts)

    kw = dict(profiles_path=profiles, index_path=index, embed_fn=embed,
              nli_batch_fn=lambda jobs: [[] for _ in jobs])
    match_spans_batch([("diabetic", "health-condition", "c1.")], **kw)
    match_spans_batch([("diabetic", "health-condition", "c2 diabetic.")], **kw)
    assert len(embed_calls) == 1   # second doc reused cached candidates


def test_batch_no_context_and_missing_index(tmp_path):
    profiles, index = _build(tmp_path, "batch_nocontext")
    got = match_spans_batch([("diabetic", "health-condition", "")],
                            profiles_path=profiles, index_path=index,
                            embed_fn=stub_embed, nli_batch_fn=lambda j: [])
    assert got[span_key("diabetic", "health-condition")] is None
    got = match_spans_batch([("diabetic", "health-condition", "ctx diabetic.")],
                            profiles_path=profiles, index_path=tmp_path / "absent.npz",
                            embed_fn=stub_embed, nli_batch_fn=lambda j: [])
    assert got[span_key("diabetic", "health-condition")] is None  # exact-only degradation


def test_degradation_warns_once(tmp_path, caplog):
    profiles, _ = _build(tmp_path, "batch_warn")
    missing = tmp_path / "gone.npz"
    import logging
    with caplog.at_level(logging.WARNING, logger="cloak.profile_match"):
        for _ in range(3):
            match_spans_batch([("x", "health-condition", "ctx x.")],
                              profiles_path=profiles, index_path=missing,
                              embed_fn=stub_embed, nli_batch_fn=lambda j: [])
    assert sum("exact-only" in r.message for r in caplog.records) == 1


def test_cap_clear_reembeds_batch_without_keyerror(tmp_path, monkeypatch):
    profiles, index = _build(tmp_path, "batch_capclear")
    embed_calls = []

    def embed(texts):
        embed_calls.append(list(texts))
        return stub_embed(texts)

    kw = dict(profiles_path=profiles, index_path=index, embed_fn=embed,
              nli_batch_fn=lambda jobs: [[] for _ in jobs])
    # cache "diabetic"
    match_spans_batch([("diabetic", "health-condition", "c1.")], **kw)
    assert len(embed_calls) == 1
    # force a cap-clear on the next call: cache size (1) now exceeds cap
    monkeypatch.setattr(pm, "_PROPOSAL_CACHE_MAX", 0)
    got = match_spans_batch(
        [("diabetic", "health-condition", "c2 diabetic."),      # previously cached -> wiped
         ("asthma-ish", "health-condition", "c2 asthma-ish.")], # new surface
        **kw)
    # no KeyError; both keys resolved/abstained, both re-embedded in one call
    assert set(got) == {span_key("diabetic", "health-condition"),
                        span_key("asthma-ish", "health-condition")}
    assert len(embed_calls) == 2
    assert sorted(embed_calls[1]) == ["asthma-ish", "diabetic"]


def test_profile_backed_types_contents():
    assert "health-condition" in PROFILE_BACKED_TYPES and "drug" in PROFILE_BACKED_TYPES
    assert "LOC" in PROFILE_BACKED_TYPES and "ORG" in PROFILE_BACKED_TYPES
    for t in ("PERSON", "CODE", "gender", "marital-status", "sexual-orientation",
              "DATETIME", "QUANTITY", "age", "demographic-other"):
        assert t not in PROFILE_BACKED_TYPES
