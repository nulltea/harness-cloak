import json

import numpy as np
import pytest

from cloak.lattice import profile_match as pm
from cloak.lattice.profile_match import PROFILE_BACKED_TYPES, match_spans_batch, span_key


@pytest.fixture(autouse=True)
def _clear_module_state():
    # _PROPOSAL_CACHE / _WARNED_INDEX_PATHS are module-global; don't leak across tests.
    pm._PROPOSAL_CACHE.clear()
    pm._WARNED_INDEX_PATHS.clear()
    pm.load_embindex.cache_clear()
    pm._profile_certification_stats.cache_clear()
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
                    "level_counts": {"endocrine condition": 1000.0, "chronic condition": 100.0},
                    "count": 1000.0,
                },
                "hypothyroidism": {
                    "aliases": [],
                    "levels": ["thyroid condition", "endocrine condition"],
                    "level_counts": {"thyroid condition": 100.0, "endocrine condition": 1000.0},
                    "count": 500.0,
                },
                "asthma": {
                    "aliases": [],
                    "levels": ["respiratory condition", "medical condition"],
                    "level_counts": {"respiratory condition": 100.0, "medical condition": 5000.0},
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
    assert m.entry == "diabetes"
    assert m.levels == ["endocrine condition", "chronic condition"]
    assert calls == {"embed": 0, "nli": 0}


def test_exact_hit_carries_canonical_entry(tmp_path):
    artifact = {
        "schema_version": 1, "created": "2026-07-10", "sources": {},
        "profiles": {"health-condition": {
            "blorbitis": {"aliases": ["blorb inflammation"],
                          "levels": ["organ disease"], "source_ids": ["t:1"], "count": 10.0},
        }},
    }
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps(artifact))
    from cloak.lattice.profile_match import match_spans_batch, span_key
    got = match_spans_batch([("blorb inflammation", "health-condition", "ctx sentence")],
                            profiles_path=p)
    m = got[span_key("blorb inflammation", "health-condition")]
    assert m is not None and m.kind == "exact"
    assert m.entry == "blorbitis"


def test_semantic_hit_via_variant(tmp_path):
    pp, ip = _build(tmp_path, "semantic")
    m = pm.match_profile_entry(
        "diabetic", "health-condition", "She is diabetic.",
        profiles_path=pp, index_path=ip, embed_fn=stub_embed, nli_fn=_approve_all,
        entry_certify_batch_fn=_entry_certify_accept,
        entry_reverse_entailment_batch_fn=_entry_reverse_not_entailed,
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


def test_batch_trace_records_exact_semantic_and_abstention(tmp_path):
    profiles, index = _build(tmp_path, "trace")
    trace = {}
    got = match_spans_batch(
        [
            ("diabetes", "health-condition", "Patient has diabetes."),
            ("diabetic", "health-condition", "She is diabetic."),
            ("unprofiled finding", "health-condition", ""),
        ],
        profiles_path=profiles,
        index_path=index,
        embed_fn=stub_embed,
        nli_batch_fn=lambda jobs: [[(level, 0.95) for level in levels]
                                   for _, _, levels in jobs],
        entry_certify_batch_fn=_entry_certify_accept,
        entry_reverse_entailment_batch_fn=_entry_reverse_not_entailed,
        trace_out=trace,
    )

    exact_key = span_key("diabetes", "health-condition")
    semantic_key = span_key("diabetic", "health-condition")
    abstained_key = span_key("unprofiled finding", "health-condition")
    assert got[exact_key] is not None and got[semantic_key] is not None
    assert got[abstained_key] is None
    assert trace[exact_key] == {
        "runtime_type": "health-condition",
        "surface_key": "diabetes",
        "outcome": "exact",
        "reason": "exact_entry",
        "candidate_attempts": [],
        "entry": "diabetes",
        "levels": ["endocrine condition", "chronic condition"],
    }
    assert trace[semantic_key]["outcome"] == "semantic"
    assert trace[semantic_key]["reason"] == "semantic_certified"
    assert trace[semantic_key]["candidate_attempts"][-1]["status"] == "accepted"
    assert trace[abstained_key]["reason"] == "no_context"


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
        entry_certify_batch_fn=_entry_certify_accept,
        entry_reverse_entailment_batch_fn=_entry_reverse_not_entailed,
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
                            embed_fn=embed, nli_batch_fn=nli_batch,
                            entry_certify_batch_fn=_entry_certify_accept,
                            entry_reverse_entailment_batch_fn=_entry_reverse_not_entailed)
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
                            embed_fn=stub_embed, nli_batch_fn=nli_batch,
                            entry_certify_batch_fn=_entry_certify_accept,
                            entry_reverse_entailment_batch_fn=_entry_reverse_not_entailed)
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
    with caplog.at_level(logging.WARNING, logger="cloak.lattice.profile_match"):
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


def test_nli_failure_degrades_fail_closed(tmp_path, caplog):
    profiles, index = _build(tmp_path, "batch_nlifail")
    import logging

    def boom(jobs):
        raise RuntimeError("nli model unavailable")

    with caplog.at_level(logging.WARNING, logger="cloak.lattice.profile_match"):
        for _ in range(2):
            got = match_spans_batch(
                [("diabetes", "health-condition", "He has diabetes."),      # exact
                 ("diabetic", "health-condition", "He is diabetic.")],       # semantic miss
                profiles_path=profiles, index_path=index,
                embed_fn=stub_embed, nli_batch_fn=boom)
    assert got[span_key("diabetes", "health-condition")].kind == "exact"   # unaffected
    assert got[span_key("diabetic", "health-condition")] is None           # abstain, no raise
    assert sum("exact-only" in r.message for r in caplog.records) == 1     # warn once


def _entry_certify_accept(jobs):
    return [(0.95, 0.10) for _ in jobs]


def _entry_reverse_not_entailed(jobs):
    return [0.10 for _ in jobs]


def test_semantic_root_only_approval_abstains(tmp_path):
    artifact = {
        "schema_version": 1, "created": "test", "sources": {}, "profiles": {
            "health-condition": {
                "alpha": {"aliases": [], "levels": ["medical condition"],
                          "level_counts": {"medical condition": 1000.0}, "count": 10.0},
                "beta": {"aliases": [], "levels": ["medical condition"],
                         "level_counts": {"medical condition": 1000.0}, "count": 10.0},
            },
        },
    }
    profiles = tmp_path / "root-only.json"
    profiles.write_text(json.dumps(artifact))
    index = pm.build_embindex(
        profiles,
        embed_fn=lambda texts: np.array([[1.0, 0.0] for _ in texts], dtype=np.float32),
    )

    got = match_spans_batch(
        [("generic finding", "health-condition", "The generic finding was documented.")],
        profiles_path=profiles,
        index_path=index,
        embed_fn=lambda texts: np.array([[1.0, 0.0] for _ in texts], dtype=np.float32),
        nli_batch_fn=lambda jobs: [[(level, 0.95) for level in levels]
                                   for _, _, levels in jobs],
        entry_certify_batch_fn=_entry_certify_accept,
    )

    assert got[span_key("generic finding", "health-condition")] is None


def test_root_only_approval_abstains_when_root_is_not_universal(tmp_path):
    artifact = {
        "schema_version": 1, "created": "test", "sources": {}, "profiles": {
            "health-condition": {
                "knees": {"aliases": [], "levels": ["medical condition"],
                          "level_counts": {"medical condition": 1000.0}, "count": 10.0},
                "diabetes": {"aliases": [], "levels": ["endocrine condition", "medical condition"],
                             "level_counts": {"endocrine condition": 20.0, "medical condition": 1000.0},
                             "count": 20.0},
                "asthma": {"aliases": [], "levels": ["respiratory condition"],
                           "level_counts": {"respiratory condition": 20.0}, "count": 20.0},
            },
        },
    }
    profiles = tmp_path / "non-universal-root.json"
    profiles.write_text(json.dumps(artifact))
    index = pm.build_embindex(
        profiles,
        embed_fn=lambda texts: np.array([[1.0, 0.0] if text == "knees" else [0.0, 1.0]
                                         for text in texts], dtype=np.float32),
    )

    got = match_spans_batch(
        [("right knee", "health-condition", "The right knee was examined.")],
        profiles_path=profiles,
        index_path=index,
        embed_fn=lambda texts: np.array([[1.0, 0.0]], dtype=np.float32),
        nli_batch_fn=lambda jobs: [[(level, 0.95) for level in levels]
                                   for _, _, levels in jobs],
        entry_certify_batch_fn=_entry_certify_accept,
    )

    assert got[span_key("right knee", "health-condition")] is None


def test_anonymity_floor_is_not_discriminative_in_a_multi_root_type(tmp_path):
    artifact = {
        "schema_version": 1, "created": "test", "sources": {}, "profiles": {
            "health-condition": {
                "knees": {"aliases": [], "levels": ["medical condition"],
                          "level_counts": {"medical condition": 500000.0}, "count": 10.0},
                "entry-a": {"aliases": [], "levels": ["disease of anatomical entity"],
                            "level_counts": {"disease of anatomical entity": 1000.0}, "count": 10.0},
                "entry-b": {"aliases": [], "levels": ["disease of anatomical entity"],
                            "level_counts": {"disease of anatomical entity": 1000.0}, "count": 10.0},
                "entry-c": {"aliases": [], "levels": ["disease of anatomical entity"],
                            "level_counts": {"disease of anatomical entity": 1000.0}, "count": 10.0},
                "reflux": {"aliases": [],
                           "levels": ["gastrointestinal condition", "medical condition"],
                           "level_counts": {"gastrointestinal condition": 400.0,
                                            "medical condition": 500000.0}, "count": 10.0},
            },
        },
    }
    profiles = tmp_path / "multi-root-anonymity.json"
    profiles.write_text(json.dumps(artifact))
    index = pm.build_embindex(
        profiles,
        embed_fn=lambda texts: np.array(
            [[1.0, 0.0] if text == "knees" else
             [0.0, 1.0] if text == "reflux" else
             [0.0, -1.0] for text in texts],
            dtype=np.float32,
        ),
    )

    def embed(texts):
        return np.array(
            [[1.0, 0.0] if text == "right knee" else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )

    floor = match_spans_batch(
        [("right knee", "health-condition", "The right knee was examined.")],
        profiles_path=profiles, index_path=index, embed_fn=embed,
        nli_batch_fn=lambda jobs: [[(level, 0.95) for level in levels]
                                   for _, _, levels in jobs],
        entry_certify_batch_fn=_entry_certify_accept,
        entry_reverse_entailment_batch_fn=_entry_reverse_not_entailed,
    )
    specific = match_spans_batch(
        [("acid reflux", "health-condition", "The acid reflux was discussed.")],
        profiles_path=profiles, index_path=index,
        embed_fn=lambda texts: np.array([[0.0, 1.0]], dtype=np.float32),
        nli_batch_fn=lambda jobs: [[("gastrointestinal condition", 0.95)] for _ in jobs],
        entry_certify_batch_fn=_entry_certify_accept,
        entry_reverse_entailment_batch_fn=_entry_reverse_not_entailed,
    )

    assert floor[span_key("right knee", "health-condition")] is None
    assert specific[span_key("acid reflux", "health-condition")].entry == "reflux"


@pytest.mark.parametrize(
    ("surface", "canonical"),
    [("imaging", "radiography"), ("palpation", "bimanual pelvic exam")],
)
def test_broader_surface_cannot_certify_narrower_profile_entry(tmp_path, surface, canonical):
    artifact = {
        "schema_version": 1, "created": "test", "sources": {}, "profiles": {
            "medical-procedure": {
                canonical: {"aliases": [], "levels": ["diagnostic procedure", "medical procedure"],
                            "level_counts": {"diagnostic procedure": 50.0, "medical procedure": 1000.0},
                            "count": 50.0},
                "other procedure": {"aliases": [], "levels": ["therapeutic procedure", "medical procedure"],
                                    "level_counts": {"therapeutic procedure": 50.0, "medical procedure": 1000.0},
                                    "count": 50.0},
            },
        },
    }
    profiles = tmp_path / f"{surface}.json"
    profiles.write_text(json.dumps(artifact))
    index = pm.build_embindex(
        profiles,
        embed_fn=lambda texts: np.array([[1.0, 0.0] if text == canonical else [0.0, 1.0]
                                         for text in texts], dtype=np.float32),
    )
    got = match_spans_batch(
        [(surface, "medical-procedure", f"The {surface} was ordered.")],
        profiles_path=profiles,
        index_path=index,
        embed_fn=lambda texts: np.array([[1.0, 0.0]], dtype=np.float32),
        nli_batch_fn=lambda jobs: [[(level, 0.95) for level in levels]
                                   for _, _, levels in jobs],
        entry_certify_batch_fn=lambda jobs: [(0.10, 0.10) for _ in jobs],
    )

    assert got[span_key(surface, "medical-procedure")] is None


def test_true_semantic_variant_passes_entry_and_specificity_certification(tmp_path):
    profiles, index = _build(tmp_path, "entry-certify")
    got = match_spans_batch(
        [("diabetic", "health-condition", "She is diabetic.")],
        profiles_path=profiles,
        index_path=index,
        embed_fn=stub_embed,
        nli_batch_fn=lambda jobs: [[(level, 0.95) for level in levels]
                                   for _, _, levels in jobs],
        entry_certify_batch_fn=_entry_certify_accept,
        entry_reverse_entailment_batch_fn=_entry_reverse_not_entailed,
    )

    match = got[span_key("diabetic", "health-condition")]
    assert match is not None and match.entry == "diabetes"
    assert match.levels == ["endocrine condition", "chronic condition"]


def test_strong_entry_membership_survives_near_root_baseline(tmp_path):
    profiles, index = _build(tmp_path, "near-root-membership")

    got = match_spans_batch(
        [("diabetic", "health-condition", "She is diabetic.")],
        profiles_path=profiles,
        index_path=index,
        embed_fn=stub_embed,
        nli_batch_fn=lambda jobs: [[(level, 0.95) for level in levels]
                                   for _, _, levels in jobs],
        entry_certify_batch_fn=lambda jobs: [(0.974, 0.891) for _ in jobs],
        entry_reverse_entailment_batch_fn=_entry_reverse_not_entailed,
    )

    match = got[span_key("diabetic", "health-condition")]
    assert match is not None and match.entry == "diabetes"


def test_semantic_match_is_vetoed_when_canonical_entails_surface(tmp_path):
    profiles, index = _build(tmp_path, "reverse-veto")
    got = match_spans_batch(
        [("palpation", "health-condition", "The palpation was documented.")],
        profiles_path=profiles,
        index_path=index,
        embed_fn=lambda texts: np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        nli_batch_fn=lambda jobs: [[(level, 0.95) for level in levels]
                                   for _, _, levels in jobs],
        entry_certify_batch_fn=_entry_certify_accept,
        entry_reverse_entailment_batch_fn=lambda jobs: [0.95 for _ in jobs],
    )

    assert got[span_key("palpation", "health-condition")] is None


def test_semantic_matching_abstains_on_missing_statistics_or_malformed_entry_scores(tmp_path):
    artifact = _artifact()
    artifact["profiles"]["health-condition"]["diabetes"].pop("level_counts")
    profiles = tmp_path / "missing-stats.json"
    profiles.write_text(json.dumps(artifact))
    index = pm.build_embindex(profiles, embed_fn=stub_embed)
    common = dict(
        profiles_path=profiles,
        index_path=index,
        embed_fn=stub_embed,
        nli_batch_fn=lambda jobs: [[(level, 0.95) for level in levels]
                                   for _, _, levels in jobs],
    )
    missing_stats = match_spans_batch(
        [("diabetic", "health-condition", "She is diabetic.")],
        entry_certify_batch_fn=_entry_certify_accept,
        **common,
    )
    valid_profiles, valid_index = _build(tmp_path, "malformed-scores")
    malformed_scores = match_spans_batch(
        [("diabetic", "health-condition", "She is diabetic.")],
        entry_certify_batch_fn=lambda jobs: ["malformed" for _ in jobs],
        profiles_path=valid_profiles,
        index_path=valid_index,
        embed_fn=stub_embed,
        nli_batch_fn=common["nli_batch_fn"],
    )

    assert missing_stats[span_key("diabetic", "health-condition")] is None
    assert malformed_scores[span_key("diabetic", "health-condition")] is None


def test_explicit_profile_match_abstention_bypasses_lattice_fallback(monkeypatch):
    from cloak.lattice import core as lattice
    monkeypatch.setattr(
        lattice, "_fine_curated_chain",
        lambda *args: pytest.fail("explicit semantic abstention must not reach curated fallback"),
    )

    levels = lattice.lattice_for(
        "unprofiled condition", "health-condition", "The unprofiled condition is noted.",
        proposal=None,
    )

    assert len(levels) == 1 and levels[0].startswith("<HEALTH_CONDITION_")
