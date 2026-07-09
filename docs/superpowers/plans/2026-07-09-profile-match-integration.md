# Retrieve-Then-Verify Matcher: Substitutor Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the validated retrieve-then-verify profile matcher (`src/cloak/profile_match.py`) into the substitutor as a batched document-level pre-pass, with R match diagnostics, WordNet demoted to diagnostic-only for fine/domain types, and a gitignored embedding-index build step.

**Architecture:** `substitute()` collects all profile-backed spans, resolves them in one `match_spans_batch()` call (one embedding forward for uncached surfaces, wave-batched NLI certification, in-process proposal cache), then hands each span's `MatchResult | None` into `lattice_for(..., proposal=...)`. `lattice_for` treats a provided proposal as pre-certified (skips its own NLI re-gate), treats `None` as "pre-pass abstained" (skips profile+semantic, falls to curated/teacher/placeholder), and — when no pre-pass ran (other callers) — calls `match_profile_entry` itself per-span. Semantic hits record provenance in `R` entries.

**Tech Stack:** numpy (GEMV retrieval), sentence-transformers (bge-small, lazy), existing DeBERTa NLI pipeline in `cloak.lattice`, pytest.

## Global Constraints

- Constants stay exactly: `TOP_K = 5`, `SIM_FLOOR = 0.70`, `NLI_THRESH = 0.6`, `DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"`, `SCHEMA_VERSION = 1` (`src/cloak/profile_match.py:17-21`). No threshold tuning anywhere.
- Propose vs certify: a `MatchResult` handed to `lattice_for` is already certified — exact hits by construction, semantic hits by NLI in the span's sentence during the pre-pass. `lattice_for` must NOT re-gate proposal levels, and must still gate its own fallback sources exactly as today.
- Fail-closed: missing/stale index, embed-model failure, no context, all candidates refused — every path yields abstain (`None`), never an exception, never a rule-based substitute for matching.
- Levels verbatim from the matched entry (downstream `lookup_count` must resolve).
- WordNet becomes diagnostic-only for fine DEM leaves and domain types: `wordnet_chain` must no longer feed `got` for `nationality, ethnicity, profession, health-condition, religion, family-role, drug, medical-procedure, organization-medical-facility`. It stays for `LOC`, `ORG`, `MISC`, and unknown types.
- Unit tests are model-free: no model downloads, no GPU, no torch/transformers import at collection time. Real-model runs happen only in Task 5's smoke (CPU-forced).
- No remote calls; no new dependencies.
- The default index artifact is gitignored (`*.embindex.npz`); runtime degrades to exact-only with a once-per-path `logging` warning when it is absent or stale. Never build the index at runtime.
- Python for all commands: `/home/timo/repos/agent-cloak/.venv/bin/python`; pytest needs `PYTHONPATH=src`. Work from `/home/timo/repos/agent-cloak/.claude/worktrees/profile-match-mvp`.
- Pre-existing failures (do NOT fix, do NOT be blocked by): `test_bench_registry.py`, `test_bench_runner.py`, `test_lattice_producer_vocabulary.py`, `test_lattice_profile_builders.py`, `test_run_roundtrip_benchmark_cli.py` — 11 failures exist at base `563042c`.

---

### Task 1: Batched NLI gate with scores (`nli_gate_batch`)

**Files:**
- Modify: `src/cloak/lattice.py:225-248` (refactor `nli_gate`)
- Test: `src/cloak/tests/test_nli_gate_batch.py` (create)

**Interfaces:**
- Consumes: module global `_nli` pipeline in `cloak.lattice` (callable: `_nli(pairs, top_k=None, truncation=True)` → per-pair list of `{"label","score"}` dicts).
- Produces: `nli_gate_batch(jobs: list[tuple[str, str, list[str]]], thresh: float = 0.6) -> list[list[tuple[str, float]]]` — per job, the approved `(candidate, entailment_score)` pairs, order preserved within a job. `nli_gate(entity, context, candidates, thresh)` keeps its exact current signature and returns `list[str]` (behavior unchanged).

- [ ] **Step 1: Write the failing tests**

```python
"""nli_gate_batch: one pipeline call for many (entity, context, candidates) jobs."""
import cloak.lattice as cl


class FakeNLI:
    """Mimics the transformers pipeline: returns entailment score per pair."""

    def __init__(self, scores):
        self.scores = list(scores)  # consumed in call order
        self.calls = 0
        self.pair_counts = []

    def __call__(self, pairs, top_k=None, truncation=True):
        self.calls += 1
        self.pair_counts.append(len(pairs))
        out = []
        for _ in pairs:
            s = self.scores.pop(0)
            out.append([{"label": "entailment", "score": s},
                        {"label": "neutral", "score": 1 - s}])
        return out


def test_batch_matches_single_and_returns_scores(monkeypatch):
    jobs = [
        ("Oslo", "She lives in Oslo.", ["a city in Norway", "a bank"]),
        ("diabetes", "He has diabetes.", ["a chronic condition"]),
    ]
    monkeypatch.setattr(cl, "_nli", FakeNLI([0.9, 0.2, 0.8]))
    got = cl.nli_gate_batch(jobs, thresh=0.6)
    assert got[0] == [("a city in Norway", 0.9)]  # 0.2 below thresh
    assert got[1] == [("a chronic condition", 0.8)]

    # single-job wrapper: same filtering, plain list, one underlying call
    monkeypatch.setattr(cl, "_nli", FakeNLI([0.9, 0.2]))
    assert cl.nli_gate("Oslo", "She lives in Oslo.",
                       ["a city in Norway", "a bank"]) == ["a city in Norway"]


def test_batch_is_one_pipeline_call(monkeypatch):
    fake = FakeNLI([0.9, 0.9, 0.9])
    monkeypatch.setattr(cl, "_nli", fake)
    cl.nli_gate_batch([("a", "x a y.", ["p"]), ("b", "x b y.", ["q", "r"])])
    assert fake.calls == 1 and fake.pair_counts == [3]


def test_batch_preserves_per_job_fail_closed(monkeypatch):
    fake = FakeNLI([0.9])
    monkeypatch.setattr(cl, "_nli", fake)
    jobs = [
        ("missing", "entity not in this context.", ["c1"]),   # no sentence hit -> []
        ("self", "the self sentence.", ["self reference"]),   # self-ref filtered -> []
        ("ok", "an ok sentence.", ["fine phrase"]),
    ]
    got = cl.nli_gate_batch(jobs)
    assert got[0] == [] and got[1] == [] and got[2] == [("fine phrase", 0.9)]
    assert fake.pair_counts == [1]  # only the viable pair hit the pipeline


def test_empty_jobs_no_pipeline_call(monkeypatch):
    fake = FakeNLI([])
    monkeypatch.setattr(cl, "_nli", fake)
    assert cl.nli_gate_batch([]) == []
    assert cl.nli_gate_batch([("e", "no hit here.", [])]) == [[]]
    assert fake.calls == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python -m pytest src/cloak/tests/test_nli_gate_batch.py -q`
Expected: FAIL — `AttributeError: module 'cloak.lattice' has no attribute 'nli_gate_batch'`.

- [ ] **Step 3: Refactor `nli_gate` into prep + batch + wrapper**

Replace the current `nli_gate` body in `src/cloak/lattice.py` (keep `_nli`/`NLI_MODEL` globals and the lazy pipeline init) with:

```python
def _nli_prep(entity: str, context: str, candidates: list[str]):
    """Per-job viability: self-ref filter, sentence location, degenerate-dup check.
    Returns (viable_candidates, hypotheses); ([], []) fails closed."""
    candidates = [c for c in candidates if entity.lower() not in c.lower()]  # self-reference = leak
    pat = re.compile(re.escape(entity), re.IGNORECASE)
    sent = next((s for s in re.split(r"(?<=[.!?])\s+", context) if pat.search(s)), context)
    if not pat.search(sent):  # can't form the hypothesis -> fail closed (escalate/floor)
        return [], []
    hyps = [pat.sub(c, sent, count=1) for c in candidates]
    # degenerate substitution ("A city city picnics") => vacuous entailment; reject
    keep = [(c, h, sent) for c, h in zip(candidates, hyps)
            if not re.search(r"\b(\w{3,}) \1\b", h, re.IGNORECASE)]
    return [(c, sent) for c, h, sent in keep], [h for _, h, _ in keep]


def nli_gate_batch(jobs: list[tuple[str, str, list[str]]],
                   thresh: float = 0.6) -> list[list[tuple[str, float]]]:
    """One pipeline call for many (entity, context, candidates) jobs.
    Per job: approved (candidate, entailment_score) pairs, input order preserved."""
    global _nli
    prepped = [_nli_prep(e, ctx, cands) for e, ctx, cands in jobs]
    pairs, owners = [], []  # owners[i] = (job_idx, candidate)
    for j, (viable, hyps) in enumerate(prepped):
        for (cand, sent), hyp in zip(viable, hyps):
            pairs.append({"text": sent, "text_pair": hyp})
            owners.append((j, cand))
    if not pairs:
        return [[] for _ in jobs]
    if _nli is None:
        import torch
        from transformers import pipeline
        _nli = pipeline("text-classification", model=NLI_MODEL,
                        device=0 if torch.cuda.is_available() else -1)
    outs = _nli(pairs, top_k=None, truncation=True)
    approved: list[list[tuple[str, float]]] = [[] for _ in jobs]
    for (j, cand), scores in zip(owners, outs):
        ent = next(d["score"] for d in scores if d["label"] == "entailment")
        if ent >= thresh:
            approved[j].append((cand, ent))
    return approved


def nli_gate(entity: str, context: str, candidates: list[str], thresh: float = 0.6) -> list[str]:
    """Keep candidates where 'context' entails 'context with entity -> candidate'."""
    return [c for c, _ in nli_gate_batch([(entity, context, candidates)], thresh=thresh)[0]]
```

Note `_nli_prep` returns `(cand, sent)` tuples so the batch builder can reuse the located sentence per pair; keep the tuple shapes exactly as shown.

- [ ] **Step 4: Run new tests + existing NLI consumers**

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python -m pytest src/cloak/tests/test_nli_gate_batch.py src/cloak/tests/test_profile_match.py -q`
Expected: all PASS (profile_match's default certifier path still works through the wrapper).

- [ ] **Step 5: Commit**

```bash
git add src/cloak/lattice.py src/cloak/tests/test_nli_gate_batch.py
git commit -m "feat(lattice): batched NLI gate with entailment scores; nli_gate now a single-job wrapper"
```

---

### Task 2: Batch matcher (`match_spans_batch`) + proposal cache + degradation logging

**Files:**
- Modify: `src/cloak/profile_match.py`
- Test: `src/cloak/tests/test_profile_match.py` (extend)

**Interfaces:**
- Consumes: `cloak.lattice.nli_gate_batch(jobs, thresh)` from Task 1; existing `_Index`, `load_embindex`, `_l2norm`, `_st_model`, `lp.lookup_levels`, `lp._norm`.
- Produces (later tasks rely on these exact names):
  - `MatchResult` gains field `nli: float | None = None` (top approved level's entailment score; `None` for exact hits and custom `nli_fn`).
  - `span_key(span_text: str, runtime_type: str) -> tuple[str, str]` — `(runtime_type, lp._norm(span_text))`.
  - `PROFILE_BACKED_TYPES: frozenset[str]` — runtime types eligible for profile matching.
  - `match_spans_batch(items, *, profiles_path=None, index_path=None, embed_fn=None, nli_batch_fn=None) -> dict[tuple[str, str], MatchResult | None]` where `items` is an iterable of `(span_text, runtime_type, context)`; returns an entry for EVERY submitted key (dedup by `span_key`, first context wins); `nli_batch_fn(jobs) -> list[list[tuple[str, float]]]`.
  - `match_profile_entry` keeps its exact current signature and single-job `nli_fn` contract (`nli_fn(entity, context, levels) -> list[str]`), now implemented as a thin wrapper over the batch path.

- [ ] **Step 1: Write the failing tests** (append to `src/cloak/tests/test_profile_match.py`; reuse the file's existing tmp-profile + stub-embedder helpers — read them first and follow their conventions, distinct file names per test for the lru caches)

```python
# --- batch matcher ---
from cloak.profile_match import PROFILE_BACKED_TYPES, match_spans_batch, span_key


def test_batch_mixed_exact_semantic_abstain(tmp_path):
    # profiles: health-condition entries "diabetes" (alias "dm") and "asthma"
    profiles, index = make_profiles_and_index(tmp_path)   # existing-style helper
    embed_calls = []

    def embed(texts):
        embed_calls.append(list(texts))
        return stub_vectors(texts)                        # deterministic stub

    def nli_batch(jobs):
        # approve every level 0.9 for diabetes-family jobs, refuse gibberish
        return [[(lv, 0.9) for lv in cands] if "gib" not in e else []
                for e, ctx, cands in jobs]

    items = [
        ("diabetes", "health-condition", "He has diabetes."),        # exact
        ("diabetic", "health-condition", "He is diabetic."),         # semantic (stub-near "dm")
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
    # batching: exactly one embed call, containing only the two non-exact surfaces
    assert len(embed_calls) == 1 and sorted(embed_calls[0]) == ["diabetic", "gibberish zz"]


def test_batch_wave2_second_candidate(tmp_path):
    # two entries above floor; candidate 1 refused, candidate 2 approved in wave 2
    profiles, index = make_two_entry_profiles(tmp_path)
    waves = []

    def nli_batch(jobs):
        waves.append([e for e, _, _ in jobs])
        # refuse everything the first wave, approve the second
        return [[] if len(waves) == 1 else [(cands[0], 0.8)] for e, ctx, cands in jobs]

    got = match_spans_batch([("variant", "health-condition", "ctx variant.")],
                            profiles_path=profiles, index_path=index,
                            embed_fn=stub_vectors_two_entries, nli_batch_fn=nli_batch)
    m = got[span_key("variant", "health-condition")]
    assert m is not None and len(waves) == 2   # second-best entry won in wave 2
    assert m.entry == SECOND_BEST_CANONICAL


def test_proposal_cache_skips_embedding(tmp_path):
    profiles, index = make_profiles_and_index(tmp_path)
    embed_calls = []
    def embed(texts):
        embed_calls.append(list(texts)); return stub_vectors(texts)
    kw = dict(profiles_path=profiles, index_path=index, embed_fn=embed,
              nli_batch_fn=lambda jobs: [[] for _ in jobs])
    match_spans_batch([("diabetic", "health-condition", "c1.")], **kw)
    match_spans_batch([("diabetic", "health-condition", "c2 diabetic.")], **kw)
    assert len(embed_calls) == 1   # second doc reused cached candidates


def test_batch_no_context_and_missing_index(tmp_path):
    profiles, index = make_profiles_and_index(tmp_path)
    got = match_spans_batch([("diabetic", "health-condition", "")],
                            profiles_path=profiles, index_path=index,
                            embed_fn=stub_vectors, nli_batch_fn=lambda j: [])
    assert got[span_key("diabetic", "health-condition")] is None
    got = match_spans_batch([("diabetic", "health-condition", "ctx diabetic.")],
                            profiles_path=profiles, index_path=tmp_path / "absent.npz",
                            embed_fn=stub_vectors, nli_batch_fn=lambda j: [])
    assert got[span_key("diabetic", "health-condition")] is None  # exact-only degradation


def test_degradation_warns_once(tmp_path, caplog):
    profiles, _ = make_profiles_and_index(tmp_path)
    missing = tmp_path / "gone.npz"
    import logging
    with caplog.at_level(logging.WARNING, logger="cloak.profile_match"):
        for _ in range(3):
            match_spans_batch([("x", "health-condition", "ctx x.")],
                              profiles_path=profiles, index_path=missing,
                              embed_fn=stub_vectors, nli_batch_fn=lambda j: [])
    assert sum("exact-only" in r.message for r in caplog.records) == 1


def test_profile_backed_types_contents():
    assert "health-condition" in PROFILE_BACKED_TYPES and "drug" in PROFILE_BACKED_TYPES
    assert "LOC" in PROFILE_BACKED_TYPES and "ORG" in PROFILE_BACKED_TYPES
    for t in ("PERSON", "CODE", "gender", "marital-status", "sexual-orientation",
              "DATETIME", "QUANTITY", "age", "demographic-other"):
        assert t not in PROFILE_BACKED_TYPES
```

The helper names above (`make_profiles_and_index`, `stub_vectors`, …) are illustrative — implement them in the test file following its existing fixture style; the assertions are the requirements.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python -m pytest src/cloak/tests/test_profile_match.py -q`
Expected: FAIL — `ImportError: cannot import name 'match_spans_batch'`.

- [ ] **Step 3: Implement in `src/cloak/profile_match.py`**

Add near the top (after the constants):

```python
import logging

from cloak.runtime_types import (DIRECT_TYPES, DOMAIN_RUNTIME_TYPES, FINE_DEM_TYPES,
                                 PLACEHOLDER_ONLY_TYPES, COARSE_RUNTIME_TYPES)

log = logging.getLogger(__name__)

# rule-based / placeholder-only / direct types never consult profiles
PROFILE_BACKED_TYPES = frozenset(
    set(COARSE_RUNTIME_TYPES) - {"DATETIME", "QUANTITY"}
    | (set(FINE_DEM_TYPES) - set(PLACEHOLDER_ONLY_TYPES) - {"age", "demographic-other"})
    | set(DOMAIN_RUNTIME_TYPES))

# proposal cache: (index_path, runtime_type, norm_surface) -> [(canonical, sim), ...]
# retrieval only — certification is context-dependent and never cached
_PROPOSAL_CACHE: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
_PROPOSAL_CACHE_MAX = 100_000
_WARNED_INDEX_PATHS: set[str] = set()
```

Add `nli: float | None = None` as the last `MatchResult` field.

Extract retrieval from `match_profile_entry` step 3 into:

```python
def _retrieve(index: _Index, runtime_type: str, q: np.ndarray) -> list[tuple[str, float]]:
    idxs = index.type_rows(runtime_type)
    if not idxs:
        return []
    sims = index.vectors[idxs] @ q
    kept = [(idxs[p], float(sims[p])) for p in np.argsort(-sims) if sims[p] >= SIM_FLOOR][:TOP_K]
    best: dict[str, float] = {}
    for row_i, sim in kept:
        canonical = index.rows[row_i]["canonical"]
        if sim > best.get(canonical, -1.0):
            best[canonical] = sim
    return sorted(best.items(), key=lambda kv: -kv[1])


def span_key(span_text: str, runtime_type: str) -> tuple[str, str]:
    return (runtime_type, lp._norm(span_text))


def _warn_exact_only(index_path: str, reason: str) -> None:
    if index_path not in _WARNED_INDEX_PATHS:
        _WARNED_INDEX_PATHS.add(index_path)
        log.warning("profile_match: %s (%s) — degrading to exact-only matching", reason, index_path)
```

Implement the batch matcher:

```python
def match_spans_batch(items, *, profiles_path=None, index_path=None, embed_fn=None,
                      nli_batch_fn=None) -> dict[tuple[str, str], "MatchResult | None"]:
    """Document-level pre-pass: one embed batch for uncached misses, wave-batched NLI.
    Returns an entry for every submitted span_key; None = abstain (fail closed)."""
    profiles_path = Path(profiles_path or lp.DEFAULT_PROFILE_PATH)
    index_path = Path(index_path) if index_path else _index_path_for(profiles_path)

    todo: dict[tuple[str, str], tuple[str, str]] = {}   # key -> (span_text, context); first wins
    for span_text, runtime_type, context in items:
        todo.setdefault(span_key(span_text, runtime_type), (span_text, context))

    out: dict[tuple[str, str], MatchResult | None] = {}
    misses: list[tuple[tuple[str, str], str, str]] = []  # (key, span_text, context)
    for key, (span_text, context) in todo.items():
        levels = lp.lookup_levels(span_text, key[0], profiles_path)
        if levels:
            out[key] = MatchResult(levels, "exact", True, 1.0, None)
        else:
            out[key] = None
            if context:
                misses.append((key, span_text, context))
    if not misses:
        return out

    index = load_embindex(str(index_path), str(profiles_path))
    if index is None:
        _warn_exact_only(str(index_path), "index missing or stale")
        return out
    misses = [(k, s, c) for k, s, c in misses if k[0] in index.types]
    if not misses:
        return out

    # one embed batch for surfaces not in the proposal cache
    uncached = [(k, s) for k, s, _ in misses
                if (str(index_path), k[0], k[1]) not in _PROPOSAL_CACHE]
    if uncached:
        try:
            if embed_fn is None:
                model = _st_model(index.model_id)
                embed_fn = lambda t: model.encode(t, normalize_embeddings=True)
            vecs = _l2norm(embed_fn([s for _, s in uncached]))
        except Exception:
            _warn_exact_only(str(index_path), "embedding model failed")
            return out
        if len(_PROPOSAL_CACHE) > _PROPOSAL_CACHE_MAX:
            _PROPOSAL_CACHE.clear()
        for (k, _), q in zip(uncached, vecs):
            _PROPOSAL_CACHE[(str(index_path), k[0], k[1])] = _retrieve(index, k[0], q)

    if nli_batch_fn is None:
        from cloak.lattice import nli_gate_batch
        nli_batch_fn = lambda jobs: nli_gate_batch(jobs, thresh=NLI_THRESH)

    # wave-batched best-first certification: wave w tries every unresolved key's w-th candidate
    unresolved = [(k, s, c, _PROPOSAL_CACHE[(str(index_path), k[0], k[1])])
                  for k, s, c in misses]
    for wave in range(TOP_K):
        jobs, owners = [], []
        for k, s, c, cands in unresolved:
            if wave < len(cands):
                canonical, sim = cands[wave]
                jobs.append((s, c, lp.lookup_levels(canonical, k[0], profiles_path)))
                owners.append((k, canonical, sim))
        if not jobs:
            break
        results = nli_batch_fn(jobs)
        resolved = set()
        for (k, canonical, sim), approved in zip(owners, results):
            if approved:
                out[k] = MatchResult([c for c, _ in approved], "semantic", False, sim,
                                     canonical, nli=max(sc for _, sc in approved))
                resolved.add(k)
        unresolved = [u for u in unresolved if u[0] not in resolved]
        if not unresolved:
            break
    return out
```

Rewrite `match_profile_entry` as a thin wrapper preserving its exact signature and single-job `nli_fn` contract:

```python
def match_profile_entry(span_text, runtime_type, context, *, profiles_path=None,
                        index_path=None, embed_fn=None, nli_fn=None) -> MatchResult | None:
    nli_batch_fn = None
    if nli_fn is not None:  # adapt list-returning single-job fn; scores unavailable -> None
        nli_batch_fn = lambda jobs: [[(c, None) for c in nli_fn(e, ctx, cands)]
                                     for e, ctx, cands in jobs]
    got = match_spans_batch([(span_text, runtime_type, context)],
                            profiles_path=profiles_path, index_path=index_path,
                            embed_fn=embed_fn, nli_batch_fn=nli_batch_fn)
    return got[span_key(span_text, runtime_type)]
```

Watch the `nli=max(...)` line when scores are `None` (custom `nli_fn`): guard with `nli=None if any(sc is None for _, sc in approved) else max(...)`. Delete the now-duplicated retrieval/certify body from the old `match_profile_entry`. All 11 existing tests must pass unchanged — they define the wrapper's compatibility bar. If an existing test asserts embed batching per single call, adjust nothing in the test; the wrapper must satisfy it.

- [ ] **Step 4: Run full matcher suite**

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python -m pytest src/cloak/tests/test_profile_match.py src/cloak/tests/test_nli_gate_batch.py -q`
Expected: all PASS, no warnings noise.

- [ ] **Step 5: Commit**

```bash
git add src/cloak/profile_match.py src/cloak/tests/test_profile_match.py
git commit -m "feat(profile-match): batched document pre-pass with proposal cache and wave NLI certification"
```

---

### Task 3: `lattice_for` proposal wiring + WordNet demotion

**Files:**
- Modify: `src/cloak/lattice.py:433-487` (`lattice_for`)
- Test: `src/cloak/tests/test_lattice_for_proposals.py` (create)

**Interfaces:**
- Consumes: `MatchResult`, `match_profile_entry` from `cloak.profile_match` (lazy import inside `lattice_for`).
- Produces: `lattice_for(span_text, span_type, context="", proposal=NO_PREPASS)` and module-level sentinel `NO_PREPASS = object()`. Semantics: `NO_PREPASS` → `lattice_for` calls `match_profile_entry` itself; `None` → pre-pass abstained, skip profile+semantic, fallbacks only; a `MatchResult` → use its levels, already certified. Task 4 imports `NO_PREPASS` from `cloak.lattice`.

- [ ] **Step 1: Write the failing tests**

```python
"""lattice_for proposal semantics + WordNet demotion for fine/domain types."""
import cloak.lattice as cl
from cloak.profile_match import MatchResult


def _no_wordnet(monkeypatch):
    calls = []
    monkeypatch.setattr(cl, "wordnet_chain", lambda *a, **k: calls.append(a) or None)
    return calls


def test_proposal_levels_used_without_regate(monkeypatch):
    gate_calls = []
    monkeypatch.setattr(cl, "nli_gate", lambda *a, **k: gate_calls.append(a) or [])
    m = MatchResult(["endocrine condition"], "semantic", False, 0.83, "diabetes", nli=0.9)
    got = cl.lattice_for("diabetic", "health-condition", "He is diabetic.", proposal=m)
    assert got == ["endocrine condition", "<HEALTH_CONDITION_1>"]
    assert gate_calls == []          # pre-certified: no re-gate


def test_proposal_none_means_abstained_skips_profile(monkeypatch):
    wn = _no_wordnet(monkeypatch)
    called = []
    import cloak.profile_match as pm
    monkeypatch.setattr(pm, "match_profile_entry", lambda *a, **k: called.append(a))
    got = cl.lattice_for("unknowniac", "health-condition", "ctx unknowniac.", proposal=None)
    assert got == ["<HEALTH_CONDITION_1>"]  # curated/teacher missed -> placeholder terminal
    assert called == [] and wn == []        # no per-span retry, wordnet diagnostic-only


def test_no_prepass_calls_matcher(monkeypatch):
    import cloak.profile_match as pm
    m = MatchResult(["media worker"], "semantic", False, 0.8, "journalist", nli=0.7)
    monkeypatch.setattr(pm, "match_profile_entry", lambda *a, **k: m)
    got = cl.lattice_for("journalists", "profession", "They are journalists.")
    assert got == ["media worker", "<PROFESSION_1>"]


def test_fine_curated_fallback_survives(monkeypatch):
    import cloak.profile_match as pm
    monkeypatch.setattr(pm, "match_profile_entry", lambda *a, **k: None)
    _no_wordnet(monkeypatch)
    got = cl.lattice_for("cardiologist", "profession", "")
    assert got[0] == "medical specialist"   # curated map still first fallback


def test_wordnet_still_feeds_coarse_types(monkeypatch):
    import cloak.profile_match as pm
    monkeypatch.setattr(pm, "match_profile_entry", lambda *a, **k: None)
    monkeypatch.setattr(cl, "wordnet_chain", lambda *a, **k: ["an institution"])
    got = cl.lattice_for("Some Org", "ORG", "")
    assert "an institution" in got


def test_domain_type_no_wordnet(monkeypatch):
    import cloak.profile_match as pm
    monkeypatch.setattr(pm, "match_profile_entry", lambda *a, **k: None)
    wn = _no_wordnet(monkeypatch)
    got = cl.lattice_for("colonoscopy", "medical-procedure", "")
    assert got == ["<MEDICAL_PROCEDURE_1>"] and wn == []
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python -m pytest src/cloak/tests/test_lattice_for_proposals.py -q`
Expected: FAIL — `TypeError: lattice_for() got an unexpected keyword argument 'proposal'`.

- [ ] **Step 3: Rewrite `lattice_for`**

Add `NO_PREPASS = object()` above `lattice_for`, replace the function with:

```python
NO_PREPASS = object()  # caller ran no pre-pass; lattice_for matches per-span itself

_FINE_LATTICE_TYPES = {"nationality", "ethnicity", "profession", "health-condition",
                       "religion", "family-role"}


def lattice_for(span_text: str, span_type: str, context: str = "",
                proposal=NO_PREPASS) -> list[str]:
    """Zero-cost sources only; teacher entities must be pre-cached via teacher_lattices.

    Profile-backed types resolve through the retrieve-then-verify matcher
    (docs/specs/substitutor-profile-match-retrieve-verify.md): `proposal` carries the
    caller's pre-pass result (MatchResult = certified hit, None = abstained, NO_PREPASS =
    no pre-pass ran -> match per-span here). Proposal levels are certified upstream (exact
    or NLI in the span's sentence) and are not re-gated. Fallback sources still pass the
    NLI gate as before. WordNet is diagnostic-only for fine/domain types: it never feeds
    their lattices (spec: no last-word fallback for legality).
    """
    deterministic = False
    if span_type in PLACEHOLDER_ONLY_TYPES or span_type in {"PERSON", "CODE"}:
        return [placeholder_token(span_type, 1)]
    if span_type == "DATETIME":
        got = bucket_date(span_text)
        deterministic = True
    elif span_type == "QUANTITY":
        got = bucket_quantity(span_text)
        deterministic = True
    elif span_type == "age":
        got = bucket_date(span_text)
        deterministic = True
    elif span_type == "demographic-other":
        got = []
    else:
        m = proposal
        if m is NO_PREPASS:
            from cloak.profile_match import match_profile_entry
            m = match_profile_entry(span_text, span_type, context)
        if m is not None:
            got, deterministic = m.levels, True  # certified upstream; do not re-gate
        elif span_type == "LOC":
            got = geonames_chain(span_text) or wordnet_chain(span_text)
        elif span_type in _FINE_LATTICE_TYPES or span_type in DOMAIN_RUNTIME_TYPES:
            got = _fine_curated_chain(span_text, span_type)
            deterministic = got is not None
            if not got and CACHE.exists():
                got = json.loads(CACHE.read_text()).get(
                    _cache_key(span_text, span_type), {}).get("lattice")
        else:  # ORG / MISC / DEM / unknown — WordNet lattices remain
            got = wordnet_chain(span_text)
            if not got and CACHE.exists():
                cache = json.loads(CACHE.read_text())
                got = (cache.get(_cache_key(span_text, span_type), {}).get("lattice") or
                       cache.get(span_text.lower(), {}).get("lattice"))
    if got and context and not deterministic:
        got = nli_gate(span_text, context, got)
    if span_type in {"LOC", "ORG", "MISC", "DEM", "DATETIME", "QUANTITY"}:
        return _filtered_levels(span_text, got) or [TYPE_LABEL.get(span_type, "something")]
    return _with_placeholder(span_text, span_type, got)
```

Add `DOMAIN_RUNTIME_TYPES` to the existing `from cloak.runtime_types import ...` line at the top of `lattice.py`. Note `_fine_curated_chain` returns `None` for domain types (no curated map) — that is correct, they fall to teacher cache then placeholder.

- [ ] **Step 4: Run new tests + all existing lattice/profile suites**

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python -m pytest src/cloak/tests/test_lattice_for_proposals.py src/cloak/tests/test_profile_match.py src/cloak/tests/test_nli_gate_batch.py src/cloak/tests/test_lattice_profiles.py -q`

Also grep for other `lattice_for(` callers and run their test files:
`grep -rn "lattice_for(" src scripts --include="*.py" | grep -v tests | grep -v def`
Expected callers: `src/cloak/substitute.py` (Task 4), training/scripts code — positional args still work (the new param is keyword-with-default). All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cloak/lattice.py src/cloak/tests/test_lattice_for_proposals.py
git commit -m "feat(lattice): proposal-aware lattice_for; WordNet demoted to diagnostic-only for fine/domain types"
```

---

### Task 4: Substitutor pre-pass + R match diagnostics

**Files:**
- Modify: `src/cloak/substitute.py:64-145` (`substitute`)
- Test: `src/cloak/tests/test_substitute_prepass.py` (create)

**Interfaces:**
- Consumes: `match_spans_batch`, `span_key`, `PROFILE_BACKED_TYPES` from `cloak.profile_match`; `NO_PREPASS` from `cloak.lattice`.
- Produces: `R` entries gain an optional `"match"` dict — semantic hits: `{"kind": "semantic", "entry": <canonical>, "similarity": <rounded 3>, "nli": <rounded 3 | None>}`; exact hits: `{"kind": "exact"}`; abstain/rule types: key absent. `doc_p` behavior otherwise unchanged.

- [ ] **Step 1: Write the failing tests**

```python
"""substitute() batched matcher pre-pass + R match provenance."""
import cloak.substitute as sub
from cloak.detect import Span
from cloak.profile_match import MatchResult, span_key


def _spans(text, *triples):
    out = []
    for surface, typ in triples:
        i = text.index(surface)
        out.append(Span(start=i, end=i + len(surface), text=surface, type=typ))
    return out


def test_prepass_feeds_lattice_and_records_match(monkeypatch):
    text = "He is diabetic and takes aspirin, then insulin."
    spans = _spans(text, ("diabetic", "health-condition"), ("aspirin", "drug"),
                   ("insulin", "drug"))
    monkeypatch.setattr(sub, "coref_chains", lambda t, s: s)
    monkeypatch.setattr(sub, "walk_risk", lambda *a, **k: 0.0)
    submitted = []

    def fake_batch(items, **kw):
        submitted.extend(items)
        return {
            span_key("diabetic", "health-condition"):
                MatchResult(["endocrine condition"], "semantic", False, 0.84,
                            "diabetes", nli=0.91),
            span_key("aspirin", "drug"):
                MatchResult(["analgesic drug"], "exact", True, 1.0, None),
            span_key("insulin", "drug"): None,   # abstained -> teacher-cache/placeholder path
        }
    monkeypatch.setattr(sub, "match_spans_batch", fake_batch)
    doc_p, R = sub.substitute(text, spans, tau=2.0)   # tau>1: accept first level

    assert {(s, t) for s, t, _ in submitted} == {("diabetic", "health-condition"),
                                                 ("aspirin", "drug"), ("insulin", "drug")}
    by_surface = {r["surface"]: r for r in R}
    assert by_surface["diabetic"]["match"] == {"kind": "semantic", "entry": "diabetes",
                                               "similarity": 0.84, "nli": 0.91}
    assert by_surface["diabetic"]["replacement"] == "endocrine condition"
    assert by_surface["aspirin"]["match"] == {"kind": "exact"}
    assert "match" not in by_surface["insulin"]   # abstain: placeholder, no provenance
    assert by_surface["insulin"]["replacement"].startswith("<DRUG_")


def test_rule_and_direct_types_not_submitted(monkeypatch):
    text = "Sarah paid 120,000 dollars in 2019."
    spans = _spans(text, ("Sarah", "PERSON"), ("120,000 dollars", "QUANTITY"),
                   ("2019", "DATETIME"))
    monkeypatch.setattr(sub, "coref_chains", lambda t, s: s)
    monkeypatch.setattr(sub, "walk_risk", lambda *a, **k: 0.0)
    called = []
    monkeypatch.setattr(sub, "match_spans_batch",
                        lambda items, **kw: called.extend(items) or {})
    sub.substitute(text, spans, tau=2.0)
    assert called == []   # no profile-backed spans -> pre-pass not consulted


def test_repeat_surface_copies_match(monkeypatch):
    text = "diabetic today; still diabetic tomorrow."
    spans = _spans(text, ("diabetic", "health-condition"))
    j = text.rindex("diabetic")
    spans.append(type(spans[0])(start=j, end=j + len("diabetic"),
                                text="diabetic", type="health-condition"))
    monkeypatch.setattr(sub, "coref_chains", lambda t, s: s)
    monkeypatch.setattr(sub, "walk_risk", lambda *a, **k: 0.0)
    m = MatchResult(["endocrine condition"], "semantic", False, 0.84, "diabetes", nli=0.9)
    monkeypatch.setattr(sub, "match_spans_batch",
                        lambda items, **kw: {span_key("diabetic", "health-condition"): m})
    _, R = sub.substitute(text, spans, tau=2.0)
    assert all(r["match"]["entry"] == "diabetes" for r in R)   # repeat reuses match too
```

Note: `Span` construction must match `cloak.detect.Span`'s actual fields (read it first; add `chain`/`score` defaults as required). If `coref_chains` requires chain ids, set them in the monkeypatched lambda.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python -m pytest src/cloak/tests/test_substitute_prepass.py -q`
Expected: FAIL — `AttributeError: module 'cloak.substitute' has no attribute 'match_spans_batch'`.

- [ ] **Step 3: Implement in `src/cloak/substitute.py`**

Add imports:

```python
from cloak.lattice import TYPE_LABEL, lattice_for, NO_PREPASS
from cloak.profile_match import PROFILE_BACKED_TYPES, match_spans_batch, span_key
```

Inside `substitute()`, after the `coref_chains` line and before the loop:

```python
    # batched matcher pre-pass: one embed batch + wave-batched NLI for the whole doc
    # (docs/specs/substitutor-profile-match-retrieve-verify.md, Efficiency)
    items = [(s.text, s.type, _sentence_around(text, s.start, s.end))
             for s in spans if s.type in PROFILE_BACKED_TYPES]
    proposals = match_spans_batch(items) if items else {}
```

In the else-branch (non-direct, non-repeat), replace the `lattice_for` call and record provenance:

```python
            sent = _sentence_around(text, s.start, s.end)
            k = span_key(s.text, s.type)
            prop = proposals[k] if k in proposals else NO_PREPASS
            lattice = lattice_for(s.text, s.type, sent, proposal=prop)
            m = proposals.get(k)
            if m is not None:
                entry["match"] = ({"kind": "exact"} if m.kind == "exact" else
                                  {"kind": "semantic", "entry": m.entry,
                                   "similarity": round(m.similarity, 3),
                                   "nli": round(m.nli, 3) if m.nli is not None else None})
```

Extend the repeat-reuse copy to carry `match`:

```python
            by_surface[skey] = {k2: entry[k2] for k2 in
                                ("action", "replacement", "risk", "lattice", "match")
                                if k2 in entry}
```

- [ ] **Step 4: Run new tests + substitutor self-check**

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python -m pytest src/cloak/tests/test_substitute_prepass.py -q`
Expected: PASS. The module self-check (`python -m` run of substitute.py) needs live models/geonames — do NOT run it here; Task 5's smoke covers the live path.

- [ ] **Step 5: Run adjacent suites for regressions**

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python -m pytest src/cloak/tests -q -k "substitute or lattice or profile_match or nli"`
Expected: PASS (ignore the 5 pre-existing failing files listed in Global Constraints if `-k` picks any up).

- [ ] **Step 6: Commit**

```bash
git add src/cloak/substitute.py src/cloak/tests/test_substitute_prepass.py
git commit -m "feat(substitutor): batched matcher pre-pass with R match provenance"
```

---

### Task 5: Index build policy, spec updates, real-model smoke

**Files:**
- Modify: `.gitignore`, `docs/specs/substitutor-profile-match-retrieve-verify.md`
- Create: `scripts/spikes/smoke_substitute_integration.py`
- Run: `scripts/build_profile_embindex.py` (exists) against the default artifact

**Interfaces:**
- Consumes: everything from Tasks 1-4; `scripts/build_profile_embindex.py` CLI from the MVP.
- Produces: local (gitignored) `data/lattice_profiles/lattice_profiles.embindex.npz`; updated spec; committed smoke evidence.

- [ ] **Step 1: Gitignore the index artifacts**

Append to `.gitignore`:

```
data/lattice_profiles/*.embindex.npz
data/lattice_profiles/**/*.embindex.npz
```

- [ ] **Step 2: Build the default index (CPU, seconds)**

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python scripts/build_profile_embindex.py data/lattice_profiles/lattice_profiles.json`
Expected: prints the written path + row count; `git status` shows NO new tracked file.

- [ ] **Step 3: Update the spec**

In `docs/specs/substitutor-profile-match-retrieve-verify.md`:
1. Delete the dead `SIM_TRUSTED = 0.995` line from the constants block (defined but never used, flagged in review).
2. Replace the "Integration point" code block with the implemented contract: `substitute()` pre-pass via `match_spans_batch` → `lattice_for(..., proposal=...)` with the three-state semantics (`MatchResult` / `None` / `NO_PREPASS`), and note WordNet demotion is now implemented.
3. Replace Open Question 3 (index growth / ANN) with a short **Efficiency** section: per-type brute-force GEMV; one embed batch per document for uncached surfaces; wave-batched best-first NLI certification (`nli_gate_batch`); in-process proposal cache keyed `(index_path, type, norm surface)` — retrieval only, certification never cached; alias promotion from `R` match logs stays a producer-side follow-up.
4. Update the R-diagnostics example block to the implemented shape (`{"kind","entry","similarity","nli"}` / `{"kind":"exact"}`).
5. Set frontmatter `updated: 2026-07-09` (already the value — leave as is) and keep `status: current`.

- [ ] **Step 4: Write and run the smoke**

`scripts/spikes/smoke_substitute_integration.py` — CPU-forced (`CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` set before imports), real models. Build a throwaway index for `data/lattice_profiles/proposed/drug-health-procedure.proposed.json` under `/tmp`, then call `substitute()` end-to-end with hand-built `Span`s (no Detector) on 2-3 sentences containing variant surfaces (`diabetic`, `an aspirin`, `heart murmurs`), pointing `match_spans_batch` at the proposed artifact via a `functools.partial` monkeypatch or module-level path override — whichever is smallest; `tau=2.0` with a comment (`walk_risk` pools may be absent for fine types; the smoke validates plumbing, not privacy calibration). Print `doc_p` and each `R` entry with its `match` block; assert: at least one semantic match with `entry` + `nli` populated flows into a replacement, exact hits carry `{"kind":"exact"}`, and a nonsense surface abstains to placeholder. Save output to `scripts/spikes/smoke_substitute_integration.out.txt` and commit both.

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python -u scripts/spikes/smoke_substitute_integration.py | tee scripts/spikes/smoke_substitute_integration.out.txt`
Expected: assertions pass, output shows semantic replacement + provenance.

- [ ] **Step 5: Full-suite regression + parity spike re-run**

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python -m pytest src/cloak/tests -q`
Expected: only the 11 pre-existing failures from the 5 files named in Global Constraints; every other test passes.

Run: `PYTHONPATH=src /home/timo/repos/agent-cloak/.venv/bin/python -u scripts/spikes/validate_profile_match.py 2>&1 | tail -15`
Expected: summary unchanged from the committed evidence (recall 14/15, abstain 3/5) — proves the wrapper refactor kept `match_profile_entry` behavior identical.

- [ ] **Step 6: Commit**

```bash
git add .gitignore docs/specs/substitutor-profile-match-retrieve-verify.md \
        scripts/spikes/smoke_substitute_integration.py scripts/spikes/smoke_substitute_integration.out.txt
git commit -m "feat(profile-match): index build policy, spec efficiency contract, end-to-end smoke"
```
