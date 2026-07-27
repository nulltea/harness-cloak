---
type: plan
status: stale
created: 2026-07-09
updated: 2026-07-27
tags: [extraction, reconstructor, archived-track]
archive_reason: subject retired to branch archive/reconstructor-track in the
  2026-07-27 cleanup (docs/plans/2026-07-27-codebase-cleanup-refactor.md)
---

# Frozen zero-shot extractor — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Spec: `docs/specs/extractor-frozen-rl-reward.md` (the controlling design — read the section
> named in each task). Scope: extractor ladder + migration + verification-gate harnesses.
> The benchmark build (planted substitutions, gold audits, calibration) is a separate plan.

## Global Constraints

- **No `pip install`.** All models are locally cached and pinned: encoder
  `BAAI/bge-small-en-v1.5`, NLI `cross-encoder/nli-deberta-v3-small`, MLM `roberta-base`.
  Never `pip install torch` (ROCm host `.venv`).
- **Determinism:** greedy/argmax only, all ties broken by explicit sort keys
  `(-score, start, end, entry_idx)`, no sampling, module-level model singletons guarded by a
  `threading.Lock`. Every verification decision applies the **margin rule**: a score within
  `EPS_MARGIN` of its threshold resolves to abstain.
- **Do-no-harm:** only `R` original surfaces may enter `out_final`; a failed gate = abstain;
  spans resolved by tier 0 are never altered by later stages.
- **Pins are data, not literals:** every model id, threshold, and template lives in one
  `EXTRACTOR_PINS` dict in `src/cloak/frozen_extractor.py`; nothing reads a magic number.
  Initial threshold values below are placeholders pending benchmark calibration — code must not
  care what they are.
- **Unit tests never load real models.** Encoder/NLI/MLM are injected callables; tests use
  deterministic stubs. Real-model paths get `@pytest.mark.slow` (skipped by default).
- **Test commands:** `PYTHONPATH=src:scripts .venv/bin/python -m pytest <file> -v`.
- **Dirty tree:** the working tree carries unrelated uncommitted WIP. `git add` ONLY the files
  your task creates/modifies — never `git add -A` / `git add .`.
- **Naming:** descriptive identifiers only; no plan/phase numbers in code (repo rule).

Initial pin values (placeholders, benchmark will recalibrate):
`SIM_MIN=0.55, ASSIGN_MARGIN=0.05, PRIOR_WEIGHT=0.15, NLI_ENTAIL=0.80, TYPE_ENTAIL=0.70,
PLL_MIN_DELTA=-6.0 (mean per-token log-prob drop), EPS_MARGIN=0.02, CHUNK_MAX_WORDS=6,
LADDER_SEMVER="0.1.0"`.

## File Structure

- `src/cloak/frozen_extractor.py` — the frozen extractor: pins, version, ladder stages,
  `extract(doc_p, R, out_p, *, models=None)`.
- `scripts/train_ranker.py` — `assemble()` gains `fill_spans` offset bookkeeping (Task 1).
- `src/cloak/train/roundtrip.py` — opt-in frozen-extractor path (Task 8).
- `scripts/extractor_determinism_gate.py`, `scripts/extractor_microbench.py` — gate harnesses.
- Tests: `src/cloak/tests/test_assemble_fill_spans.py`, `src/cloak/tests/test_frozen_extractor.py`
  (+ per-stage additions to the latter).

---

### Task 1: `fill_spans` offset bookkeeping in `assemble()`

**Files:** Modify `scripts/train_ranker.py`. Create `src/cloak/tests/test_assemble_fill_spans.py`.
Spec section: "`R` offset schema (assemble-time bookkeeping)".

`assemble()` applies replacements right-to-left on `doc_orig` offsets, then `_cleanup()` regex
edits the result. Add exact final-offset tracking:

- During the right-to-left pass, after applying each replacement at `(e.start, e.end)` with
  string `rep`, record the span `(e.start, e.start + len(rep))` **in the current intermediate
  string** and shift every previously recorded span (which all lie to the RIGHT of the edit) by
  `len(rep) - (e.end - e.start)`. Wait — previously recorded spans lie right of the current edit
  point (right-to-left processing), so they shift by the delta. Implement and unit-test this
  shifting; do not trust the parenthetical.
- Replace `_cleanup()` usage with a tracked version: apply each of its regex substitutions via
  `re.finditer`, computing per-deletion offsets and shifting recorded spans; if a deletion
  overlaps a recorded span's interior, drop that span from `fill_spans` and continue (rare;
  count it in an assertion-friendly return, do not fail the build).
- Attach to each `R` entry: `"fill_spans": [[start, end], ...]` — one span per applied
  occurrence of that entry's `(surface, replacement)` pair, placeholders included.
- **Build invariant, asserted at the end of `assemble()`:** for every recorded span,
  `doc_p[start:end] == the applied replacement string` (exact — replacements are stored
  case-adjusted). Assertion failure = bug, must raise.

Tests (write first, watch fail, implement, pass):
1. Single replacement: span matches, invariant holds.
2. Multiple replacements with different length deltas: all spans correct.
3. Repeated surface (two occurrences of one surface): two spans on that entry.
4. Cleanup shift: construct text where `_cleanup` deletes a duplicate article BEFORE a recorded
   span ("the " + fill starting with "an ") — span shifts left, invariant still holds.
5. Mixed typing (same surface as placeholder + generalize, as in current `assemble`): each `R`
   entry carries only its own occurrences' spans.

Commit: `feat(extractor): fill_spans offset bookkeeping in assemble (tracked through cleanup)`

---

### Task 2: frozen-extractor module skeleton — pins, version, tier 0

**Files:** Create `src/cloak/frozen_extractor.py`, `src/cloak/tests/test_frozen_extractor.py`.
Spec sections: "Definitions" (Frozen), "Proposed architecture" stage 0, "Coupling hygiene".

- `EXTRACTOR_PINS`: dict with `models` (encoder/nli/mlm ids), `thresholds` (all values from
  Global Constraints), `type_hypotheses` (Task 5 fills it; empty dict now), `ladder_semver`.
- `extractor_version() -> str`: `"fx-" + sha256(canonical-json(EXTRACTOR_PINS))[:12]` — stable
  across processes, changes iff any pin changes.
- `extract(doc_p: str | None, R: list[dict], out_p: str, *, models: dict | None = None)
  -> tuple[str, dict]`:
  - runs tier 0 = `cloak.extract._rule_prepass(out_p, R, semantic=True)`;
  - with `models=None` (deterministic-only mode): every residue entry becomes outcome
    `abstained/no-models`; return `_finalize(prepass_text, stats)`.
  - stats gains `"entries"`: one `{"surface", "type", "outcome", "reason"}` per residue entry,
    and `"extractor_version"`.
  - later stages (Tasks 3–7) slot in between; structure the function so each stage is a small
    pure helper call.

Tests: deterministic-only `extract` equals `invert()` output on a placeholder case and an
exact-fill case; residue entries appear as abstained outcomes; `extractor_version()` stable
across two calls and changes when a pin is (temporarily monkeypatched) different.

Commit: `feat(extractor): frozen_extractor skeleton — pins, version hash, tier-0 + abstain`

---

### Task 3: alignment prior

**Files:** Modify `src/cloak/frozen_extractor.py`, extend `test_frozen_extractor.py`.
Spec sections: "Proposed architecture" stage 1, "Alignment-prior failure modes".

- `sentence_spans(text) -> list[tuple[int, int]]`: deterministic regex sentence splitter
  (split on `[.!?\n]` + whitespace; no model).
- `align_sentences(doc_vecs, out_vecs) -> list[int]`: monotonic alignment mapping each doc
  sentence index to an out sentence index — small DTW over cosine distance, ties resolved
  toward the diagonal, pure numpy, deterministic.
- `position_bonus(fill_span, doc_sent_spans, out_sent_spans, alignment) -> tuple[int, int] | None`:
  locate which doc sentence contains the fill span → aligned out sentence → return that out_p
  char window (± one sentence). `None` when the entry has no `fill_spans` or `doc_p is None`.
- Bonus application (used in Task 4's scoring): candidate chunks inside the window get
  `+PRIOR_WEIGHT`, decaying linearly to 0 one sentence outside. **Bonus only — never filters
  candidates.**
- Encoder is injected: `models["encoder"].encode(list[str]) -> np.ndarray` (rows L2-normalized).

Tests with a deterministic toy encoder (e.g., bag-of-words hashed to fixed dims, normalized):
identity docs align diagonally; a reordered pair aligns correctly; a deleted-sentence pair maps
around the gap; entry without `fill_spans` → `None`; window covers the expected sentence.

Commit: `feat(extractor): doc_p→out_p alignment prior (DTW over sentence embeddings, bonus-only)`

---

### Task 4: candidate generation + global one-to-one assignment

**Files:** Modify `src/cloak/frozen_extractor.py`, extend tests. Spec: stages 2–3.

- `candidate_chunks(out_p) -> list[tuple[int, int, str]]`: word n-grams (1..CHUNK_MAX_WORDS)
  over `out_p` tokens, word-boundary exact offsets, skipping chunks that are only
  stopwords/punctuation. Deduplicate identical `(start, end)`.
- `score_pairs(residue, chunks, encoder, windows) -> list[tuple[float, int, int]]`:
  score(entry i, chunk j) = `0.6·cos(fill_i, chunk_j) + 0.4·cos(surface_i, chunk_j) + bonus`
  where bonus is Task 3's window bonus (0 without a window). One batched `encode` call for all
  fills+surfaces+chunks.
- `assign(scores, n_entries, chunks) -> dict[int, int]`: greedy by
  `(-score, chunk_start, chunk_end, entry_idx)`; a pair is taken only if the entry is unassigned
  AND the chunk span overlaps no already-taken chunk span AND `score >= SIM_MIN`; after
  assignment, an entry whose best remaining alternative is within `ASSIGN_MARGIN` of its taken
  score for a DIFFERENT surface's chunk → demote to abstain `ambiguous` (repeated-generic-fill
  guard).
- Unassigned entries → outcome `abstained/no-candidate`.

Tests (toy encoder): two entries with near-identical generic fills + position windows →
disambiguated by the prior; overlapping chunk claims excluded; sub-SIM_MIN pairs abstain;
ambiguity demotion fires when two entries tie on one chunk.

Commit: `feat(extractor): candidate chunks + global greedy assignment with ambiguity abstain`

---

### Task 5: verification stack

**Files:** Modify `src/cloak/frozen_extractor.py`, extend tests. Spec: stage 4.

- NLI protocol (injected): `models["nli"](premise, hypothesis) -> tuple[str, float]` (label,
  prob of that label).
- `verify(entry, chunk_text, sentence, nli) -> tuple[bool, str]` — all must pass, first failure
  returns its reason:
  a. **scalar gate**: reuse `cloak.reconstruct._value_compatible(fill, chunk_text)`; `False` →
     reject `added-digit`; `None` → continue to NLI.
  b. **correspondence**: `nli(fill-in-its-sentence-template, chunk-in-sentence)` — require
     entailment with prob ≥ `NLI_ENTAIL`; margin rule (within EPS_MARGIN → abstain
     `margin-correspondence`).
  c. **type gate**: `TYPE_HYPOTHESES[runtime_type]` — a pinned template per runtime type
     (cover: PERSON, CODE, ORG, LOC, DATETIME, QUANTITY, MISC + the fine leaves from
     `docs/specs/lattice-substitutor.md`'s runtime-type table; e.g. `health-condition` →
     "This text mentions a disease, diagnosis, or health condition."). Require
     `nli(sentence, hypothesis)` entailment ≥ `TYPE_ENTAIL`, margin rule applies. Unknown type →
     skip the type gate (fail-open on the TYPE gate only; correspondence still mandatory).
  d. **added proper noun**: any capitalized token in `chunk_text` (not sentence-initial) absent
     from fill AND surface → reject `added-proper-noun`.
- Empty/whitespace fill or placeholder-pattern fill (`cloak.runtime_types.PLACEHOLDER_RE`) →
  reject `bad-fill` before anything else.

Tests (stub NLI returning scripted labels/probs): each reason path; margin-rule abstain at
threshold±EPS; spec's canonical cases — fill "some time ago" vs chunk "three years ago" rejects
(`added-digit`); fill "the early 1980s" vs chunk "Early 1980s" passes with entailing stub;
empty fill rejects.

Commit: `feat(extractor): verification stack — scalar gate, NLI correspondence, type gate, margin rule`

---

### Task 6: boundary-tight splice + MLM fluency gate

**Files:** Modify `src/cloak/frozen_extractor.py`, extend tests. Spec: stages 5–6.

- `splice(out_p, chunk_span, surface) -> str`: replace exactly the chunk span (already
  word-snapped by construction), then apply the article-agreement fixer — reuse the existing
  helper in `src/cloak/substitute.py` (added with the lattice-cache runtime wiring; grep for the
  article/`a/an` fixer and import it rather than re-implementing).
- MLM protocol (injected): `models["mlm"].pll(sentence) -> float` (mean per-token
  pseudo-log-likelihood). `pll_delta = pll(sentence_after) - pll(sentence_before)`; if
  `pll_delta < PLL_MIN_DELTA` → revert the splice, outcome `abstained/fluency`.
- Real loader (used only by Task 7's `load_models`): roberta-base masked-token scoring, batched
  over mask positions, `torch.no_grad`, greedy deterministic.
- Splices are applied right-to-left over assigned chunk spans so earlier splices don't shift
  later spans.

Tests: splice replaces exactly the chunk; article fixed ("a" vs "an" boundary case); stub MLM
crater → revert + outcome recorded; two splices in one doc applied right-to-left correctly.

Commit: `feat(extractor): boundary-tight splice + article fix + MLM fluency revert`

---

### Task 7: `extract()` composition + real model bundle

**Files:** Modify `src/cloak/frozen_extractor.py`, extend tests. Spec: "Proposed architecture"
(whole pipeline), "Requirements" (determinism).

- Wire stages 1–6 into `extract()` between tier 0 and `_finalize`; every residue entry ends in
  exactly one outcome: `spliced`, or `abstained/<reason>`; tier-0-resolved entries are
  `resolved_tier0` (do not enumerate them per-mention if `_rule_prepass` doesn't expose it —
  a count is enough; state this in stats docs).
- `load_models(device="cpu") -> dict`: module-singleton (double-checked with a
  `threading.Lock`) building `{"encoder", "nli", "mlm"}` from `EXTRACTOR_PINS["models"]`:
  sentence-transformers for the encoder; transformers pipeline for NLI wrapped to the
  `(label, prob)` protocol (handle dict|list return shapes — see `_load_nli` in
  `cloak.reconstruct` for the known pitfall); Task 6's MLM loader.
- End-to-end tests with toy models: (a) reworded fill recovered and spliced; (b) garbage fill
  ("an information") abstains at verification, `out_final` unchanged there; (c) repeated
  generic fills resolved to distinct mentions via windows; (d) `models=None` still equals
  Task 2 behavior; (e) `stats["extractor_version"]` present.
- One `@pytest.mark.slow` test loading real models on CPU for a single tiny doc (skipped by
  default; run manually before the gates).

Commit: `feat(extractor): compose ladder in extract() + pinned model bundle loader`

---

### Task 8: roundtrip opt-in integration + pin documentation

**Files:** Modify `src/cloak/train/roundtrip.py`,
`docs/specs/RL/roundtrip-ranker-infiller.md` (extractor-pin paragraph only), extend tests
(new `src/cloak/tests/test_roundtrip_extractor_optin.py`). Spec: "Integration contract" items
2–4.

- `roundtrip_batch(jobs, workers=6, extractor_models=None)`: default `None` → exactly today's
  `invert(op, j["R"])` path, byte-identical behavior. With a models dict → 
  `frozen_extractor.extract(j.get("doc_p"), j["R"], op, models=extractor_models)` and each
  result dict gains `"extractor_version"`.
- Do NOT change the reward pin defaults; add to the roundtrip module docstring: the extractor
  is part of the reward pin; frozen-extractor results are keyed by `extractor_version`.
- Update the RL spec's extractor-pin paragraph: legacy cascade remains the default pin; the
  frozen extractor is opt-in and carries `extractor_version`; cached rewards are valid only
  under the pin they were produced with.
- Tests: monkeypatch `_remote` and `invert`/`extract` — default path calls `invert` and has no
  `extractor_version` key; opt-in path calls `extract` with the job's `doc_p` and stamps the
  version.

Commit: `feat(extractor): roundtrip opt-in frozen-extractor path + extractor_version stamp`

---

### Task 9: verification-gate harnesses

**Files:** Create `scripts/extractor_determinism_gate.py`, `scripts/extractor_microbench.py`.
Extend tests minimally (arg-parsing/fixture-loading unit only). Spec: "Verification gates
before RL use".

- `extractor_determinism_gate.py`: loads fixture triples `(doc_p, R, out_p)` from a JSONL
  (`--fixtures`, default `data/extractor_gate_fixtures.jsonl`; also `--make-fixtures` mode that
  synthesizes a small set INCLUDING near-threshold cases by monkeypatching thresholds around a
  stub score); runs `extract` in `N=3` fresh subprocesses (`sys.executable -c` or a `--worker`
  entry mode) with real models (`--device`), byte-compares `out_final` + outcomes across
  processes; exit 0 iff identical. `--stub` flag swaps in the deterministic toy models so the
  harness itself is CI-testable without GPU.
- `extractor_microbench.py`: loads real models once, times `extract` cache-hot over the fixture
  set, reports p50/p95 wall per doc, peak `torch.cuda.max_memory_allocated()` (works on ROCm),
  and the residue-count distribution; writes `results/extractor_microbench.json`.
- Neither script is RUN in this plan (GPU permission gate); both must smoke in `--stub` mode:
  add one unit test invoking the determinism gate's main with `--stub` via `subprocess` and
  asserting exit 0.

Commit: `feat(extractor): determinism gate + microbench harnesses (stub-smokeable)`
