# Lattice Producer Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the next lattice-producer run emit generalization chains that (a) always carry ≥2 semantically-close levels and (b) carry coherent, correctly-defined counts, by fixing the deterministic-path bypass, redefining and shape-constraining model counts, ranking the vocabulary context by relevance instead of raw count, and running coherence reconciliation periodically.

**Architecture:** Ontology-first backbone (DOID `is_a` / ICD-10-PCS prefixes / openFDA pharm_class supply adjacent, close tiers by construction); the model only fills gaps and names tiers the ontology lacks, under a stricter prompt that (1) defines `count` as an anonymity-set size (distinct entities, NOT people affected), (2) is shown a relevance-ranked, count-annotated vocabulary slice, and (3) is required to produce ≥2 levels. A gate enforces the chain-length floor, an adjacent-step log-gap band (model segments only), and count-agreement with the vocabulary; certifying member-set counts are held fixed as anchors and never reshaped. Coherence normalization runs every N accepted items, not only at queue end.

**Tech Stack:** Python 3, LangGraph, OpenAI-compatible local llama-swap client, pytest. All heavy runs in the host `.venv` on GPU.

## Global Constraints

- **Empirical honesty (hard rule):** never invent or apply a per-model calibration knob to equalize a secondary quantity. The log-gap shape band applies **only to `model-proposed` level segments**; any `certifying` (real member-set) count is a fixed anchor and is never rewritten to satisfy shape. Report degeneracies as findings.
- **`level_count` semantics:** `level_count` is an **anonymity-set size** — the number of distinct entities that generalize to that level (`aset_count` / `|member_set|`), used for the k-anonymity monotone risk walk (`src/cloak/lattice_profiles.py:71`, `K_FLOORS`). It is NOT epidemiological prevalence, market size, or disease burden.
- **No plan/phase identifiers in code** (per project CLAUDE.md). Name after method/feature (e.g. `min_chain_length`, `log_gap_band`), never after this doc's task numbers.
- **Deterministic-first:** a real local dataset (openFDA / DOID / ICD-10-PCS) must be tried before any model call.
- **Naming:** new training/records/docs follow the project frontmatter + folder rules; this plan touches only `src/cloak/lattice_producer/**`, `scripts/run_lattice_producer.py`, and their tests.
- **Tests:** every code task ends with a runnable pytest. Run with `.venv/bin/python -m pytest`.
- **GPU/proxy:** the smoke re-run (final task) needs the GPU + local proxy — check `rocm-smi --showpidgpus` AND confirm with the user before launching (memory: gpu-occupancy-check, no-paid-models-without-permission).

---

### Task 1: Stop category-seeded queue items from bypassing the deterministic path (register #1)

**Files:**
- Modify: `src/cloak/lattice_producer/queue.py:104-125` (`_queue_from_profile_categories`)
- Test: `src/cloak/tests/test_lattice_producer_queue.py`

**Interfaces:**
- Consumes: `normalize_item(raw, index)` (unchanged).
- Produces: category-seeded items **without** the `force_model_proposal` key, so `route_selected` (`graph.py:152`) falls through to `deterministic_lookup`.

- [ ] **Step 1: Write the failing test**

```python
# in src/cloak/tests/test_lattice_producer_queue.py
import json
from cloak.lattice_producer.queue import _queue_from_profile_categories


def test_category_seeded_items_do_not_force_model_proposal(tmp_path):
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({
        "profiles": {"drug": {"aspirin": {"aliases": ["asa"], "levels": [], "level_counts": {}}}}
    }))
    items = _queue_from_profile_categories(profiles, ["drug"])
    assert items, "expected one drug item"
    assert all("force_model_proposal" not in item for item in items)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_queue.py::test_category_seeded_items_do_not_force_model_proposal -v`
Expected: FAIL — `force_model_proposal` present (currently hardcoded `True`).

- [ ] **Step 3: Remove the forced flag**

In `_queue_from_profile_categories`, delete the `"force_model_proposal": True,` line (`queue.py:120`) from the dict passed to `normalize_item`. Leave everything else identical.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_queue.py -v`
Expected: PASS (all queue tests).

- [ ] **Step 5: Commit**

```bash
git add src/cloak/lattice_producer/queue.py src/cloak/tests/test_lattice_producer_queue.py
git commit -m "fix(lattice-producer): stop category-seeded items bypassing deterministic lookup"
```

---

### Task 2: Add a retry/backoff wrapper and a bounded default thinking budget around the LLM call (register #7)

**Files:**
- Modify: `src/cloak/lattice_producer/propose.py:214-234` (`propose_with_llama_swap`, the `client.chat.completions.create` calls)
- Modify: `scripts/run_lattice_producer.py:33` (`--thinking-budget-tokens` default)
- Test: `src/cloak/tests/test_lattice_producer_propose_retry.py` (create)

**Interfaces:**
- Consumes: `openai.OpenAI` client; `openai.APITimeoutError`.
- Produces: `_create_with_retry(client, *, model, request_kwargs, attempts=3, base_timeout=600) -> response` — retries on `APITimeoutError`/`APIConnectionError` with escalating per-call `timeout`, re-raises after the last attempt.

- [ ] **Step 1: Write the failing test**

```python
# src/cloak/tests/test_lattice_producer_propose_retry.py
import pytest
from openai import APITimeoutError
from cloak.lattice_producer.propose import _create_with_retry


class _FlakyClient:
    def __init__(self, fail_times):
        self.calls = 0
        self.fail_times = fail_times
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise APITimeoutError(request=None)
        return {"ok": True, "timeout": kwargs.get("timeout")}


def test_retry_succeeds_after_transient_timeouts():
    client = _FlakyClient(fail_times=2)
    resp = _create_with_retry(client, model="m", request_kwargs={}, attempts=3, base_timeout=10)
    assert resp["ok"] and client.calls == 3


def test_retry_reraises_after_exhausting_attempts():
    client = _FlakyClient(fail_times=5)
    with pytest.raises(APITimeoutError):
        _create_with_retry(client, model="m", request_kwargs={}, attempts=3, base_timeout=10)
    assert client.calls == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_propose_retry.py -v`
Expected: FAIL — `_create_with_retry` not defined.

- [ ] **Step 3: Implement the retry helper and route both create calls through it**

Add to `propose.py` (near the top of the module body, after imports):

```python
from openai import APIConnectionError, APITimeoutError

_RETRYABLE = (APITimeoutError, APIConnectionError)


def _create_with_retry(client, *, model, request_kwargs, attempts=3, base_timeout=600):
    """Bounded retry around a single chat completion. Escalates the per-call timeout each
    attempt (600s, 1200s, 1800s by default) and re-raises the last error after `attempts`."""
    last = None
    for attempt in range(attempts):
        try:
            return client.chat.completions.create(
                model=model, timeout=base_timeout * (attempt + 1), **request_kwargs
            )
        except _RETRYABLE as exc:
            last = exc
    raise last
```

Then in `propose_with_llama_swap`, replace `response = client.chat.completions.create(model=model, **request_kwargs)` with `response = _create_with_retry(client, model=model, request_kwargs=request_kwargs)` and the escalation-model call `response = client.chat.completions.create(model=escalation_model, **request_kwargs)` with `response = _create_with_retry(client, model=escalation_model, request_kwargs=request_kwargs)`.

- [ ] **Step 4: Change the CLI thinking-budget default**

In `scripts/run_lattice_producer.py`, change `parser.add_argument("--thinking-budget-tokens", type=int, default=-1)` to `default=2048` (matches `QWEN36_THINKING_BUDGET_TOKENS` in `propose.py:19`; caps per-item latency instead of unbounded).

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_propose_retry.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cloak/lattice_producer/propose.py scripts/run_lattice_producer.py src/cloak/tests/test_lattice_producer_propose_retry.py
git commit -m "feat(lattice-producer): bounded LLM retry + non-unbounded thinking budget default"
```

---

### Task 3: Redefine `count` in the prompt and require ≥2 levels (findings A, B; register #9, #10)

**Files:**
- Modify: `src/cloak/lattice_producer/propose.py:64-90` (`assemble_context_packet`) and `:215-221` (the prompt string)
- Test: `src/cloak/tests/test_lattice_producer_propose_packet.py` (create)

**Interfaces:**
- Consumes: `assemble_context_packet(item, *, profiles_path, run_dir, prompt_version, max_context_rows, proposed_out)`.
- Produces: packet keys `count_semantics_instruction` (str), `min_levels` (int = 2); the prompt string additionally states the count definition and the ≥2-level requirement.

- [ ] **Step 1: Write the failing test**

```python
# src/cloak/tests/test_lattice_producer_propose_packet.py
from cloak.lattice_producer.propose import assemble_context_packet


def test_packet_defines_count_as_anonymity_set_and_requires_two_levels(tmp_path):
    profiles = tmp_path / "p.json"
    profiles.write_text('{"profiles": {}}')
    packet = assemble_context_packet(
        {"runtime_type": "health-condition", "surface": "eczema"},
        profiles_path=profiles, run_dir=tmp_path, prompt_version="v", max_context_rows=8,
    )
    text = packet["count_semantics_instruction"].lower()
    assert "distinct" in text and ("not" in text and ("people" in text or "prevalence" in text))
    assert packet["min_levels"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_propose_packet.py -v`
Expected: FAIL — key `count_semantics_instruction` missing.

- [ ] **Step 3: Add the count definition and level-floor to the packet**

In `assemble_context_packet`, inside the `packet = {...}` dict add:

```python
        "min_levels": 2,
        "count_semantics_instruction": (
            "proposed_count is an ANONYMITY-SET SIZE: the number of DISTINCT entities of this "
            "runtime_type that generalize to this level (e.g. how many distinct medical "
            "conditions fall under 'metabolic disorder'). It is NOT how many people are affected, "
            "NOT prevalence, NOT disease burden, NOT market size. Typical values are small: a "
            "specific class holds a handful to a few hundred distinct members, not millions."
        ),
```

And change `required_proposal_fields` / instructions to state at least two levels are required: append to the prompt string in `propose_with_llama_swap` (Step 4).

- [ ] **Step 4: Update the prompt string**

In `propose_with_llama_swap`, change the `prompt = (...)` text to include, before the `json.dumps(packet, ...)`:

```python
        "Provide AT LEAST TWO ordered levels: the nearest truthful generalization and at least "
        "one broader tier, each semantically close to its neighbor (no jump straight to a "
        "universal catch-all). proposed_count is an anonymity-set size (count of DISTINCT "
        "entities under the level), never a count of people or prevalence.\n\n"
```

(Insert this into the existing multi-line string; keep the strict-JSON and per-level-fields sentences.)

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_propose_packet.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cloak/lattice_producer/propose.py src/cloak/tests/test_lattice_producer_propose_packet.py
git commit -m "feat(lattice-producer): define count as anonymity-set size and require >=2 levels in prompt"
```

---

### Task 4: Rank the vocabulary context slice by relevance and annotate it with counts (findings C, D; register #6, #10, #11)

**Files:**
- Modify: `src/cloak/lattice_producer/vocabulary.py:59-113` (`_seed_from_run`, `context_slice`)
- Modify: `src/cloak/lattice_producer/propose.py:62-87` (packet uses the new slice shape)
- Modify: `scripts/run_lattice_producer.py:32` (`--max-context-rows` default)
- Test: `src/cloak/tests/test_lattice_producer_vocabulary.py`

**Interfaces:**
- Consumes: `CanonicalVocabulary(runtime_type, *, proposed_out)`.
- Produces:
  - `_seed_from_run` now records the **latest** count for a label (overwrites, not first-write-wins) and keeps the max-seen for de-dup stability.
  - `context_slice(n, *, surface=None) -> list[dict]` — returns `[{"label": str, "count": float}, ...]`. When `surface` is given, ranks by token-overlap with `surface` first (descending), then by count; when `surface` is None, falls back to count-desc (back-compat).

- [ ] **Step 1: Write the failing tests**

```python
# add to src/cloak/tests/test_lattice_producer_vocabulary.py
def test_context_slice_returns_label_count_pairs_ranked_by_surface_overlap(tmp_path):
    path = tmp_path / "proposed.json"
    _write_proposed(path, "health-condition", {
        "eczema": {"levels": ["skin disorder", "human medical condition"],
                    "level_counts": {"skin disorder": 40, "human medical condition": 900}},
    })
    vocab = CanonicalVocabulary("health-condition", proposed_out=path)
    slice_ = vocab.context_slice(n=5, surface="chronic skin rash")
    assert isinstance(slice_[0], dict) and {"label", "count"} <= set(slice_[0])
    # "skin disorder" shares 'skin' with the surface, so it must outrank the higher-count sink
    labels = [row["label"] for row in slice_]
    assert labels.index("skin disorder") < labels.index("human medical condition")


def test_seed_from_run_tracks_latest_count(tmp_path):
    path = tmp_path / "proposed.json"
    _write_proposed(path, "drug", {
        "a": {"levels": ["analgesic"], "level_counts": {"analgesic": 10}},
        "b": {"levels": ["analgesic"], "level_counts": {"analgesic": 25}},
    })
    vocab = CanonicalVocabulary("drug", proposed_out=path)
    # dict iteration is insertion order; "b" (25) is seen last and must win over "a" (10)
    assert vocab.context_slice(n=5)[0]["count"] == 25 or any(
        r["label"] == "analgesic" and r["count"] == 25 for r in vocab.context_slice(n=50)
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_vocabulary.py -v`
Expected: FAIL — `context_slice` returns strings, no `surface` kwarg; `_seed_from_run` keeps first count.

- [ ] **Step 3: Update `_seed_from_run` to overwrite with latest count**

In `_seed_from_run`, change the inner loop so a run-label's count is updated every time (still tracking membership in `_run_labels`):

```python
            for level in row.get("levels", []):
                key = _norm(level)
                count = float(level_counts.get(level, 1.0))
                if key not in self._labels or key in self._run_labels:
                    self._labels[key] = count
                    self._run_labels.add(key)
```

(Static seeds — not in `_run_labels` — are never overwritten by a run count; run labels always take the latest.)

- [ ] **Step 4: Rewrite `context_slice`**

```python
    def context_slice(self, n: int = 10, *, surface: str | None = None) -> list[dict]:
        """A bounded, representative slice for a context packet as {label, count} rows. When a
        surface is given, rank by token-overlap with the surface first (so the model sees the
        labels most likely to be the right reuse target), then by count; otherwise count-desc."""
        labels = list(self._labels)
        if surface:
            surface_tokens = _tokens(surface)
            def key(label):
                overlap = len(_tokens(label) & surface_tokens)
                return (-overlap, -self._labels[label])
            labels.sort(key=key)
        else:
            labels.sort(key=lambda label: -self._labels[label])
        return [{"label": label, "count": self._labels[label]} for label in labels[:n]]
```

- [ ] **Step 5: Wire the new slice into the packet**

In `assemble_context_packet` (`propose.py`), change:

```python
    vocabulary_slice = vocabulary.context_slice(n=max_context_rows, surface=surface) if vocabulary else []
```

and update `canonical_vocabulary_instruction` to reference the attached count:

```python
        "canonical_vocabulary_instruction": (
            "canonical_vocabulary_slice lists {label, count} rows this run already uses. If any "
            "label fits a proposed level, reuse it verbatim, set reused_canonical_label: true, "
            "and reuse its attached count. Only coin new phrasing when nothing fits."
        ),
```

- [ ] **Step 6: Raise the CLI default `--max-context-rows`**

In `scripts/run_lattice_producer.py`, change `--max-context-rows` default from `8` to `20` (register #11: 8 is too few against a 350–770-label vocabulary; relevance ranking makes a larger slice useful without dominating the prompt).

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_vocabulary.py src/cloak/tests/test_lattice_producer_propose_packet.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/cloak/lattice_producer/vocabulary.py src/cloak/lattice_producer/propose.py scripts/run_lattice_producer.py src/cloak/tests/test_lattice_producer_vocabulary.py
git commit -m "feat(lattice-producer): relevance-ranked, count-annotated vocabulary slice"
```

---

### Task 5: Gate — enforce minimum chain length and count-agreement for reused exact-match labels (findings B; register #2)

**Files:**
- Modify: `src/cloak/lattice_producer/vocabulary.py` (add `count_for(label)`)
- Modify: `src/cloak/lattice_producer/gates.py:109-214` (`gate_candidates`)
- Test: `src/cloak/tests/test_lattice_producer_gates.py`

**Interfaces:**
- Consumes: `CanonicalVocabulary.has_exact(label)`; adds `CanonicalVocabulary.count_for(label) -> float | None`.
- Produces: `gate_candidates` diagnoses `count_disagreement` when a `model-proposed` level exactly matches a vocabulary label but its `level_count` is off by more than a tolerance factor (default 4×) from the vocabulary's recorded count; and diagnoses the whole item `too_few_levels` when fewer than 2 candidates survive to `accepted` for an eligible item.

- [ ] **Step 1: Write the failing tests**

```python
# add to src/cloak/tests/test_lattice_producer_gates.py
import json
from cloak.lattice_producer.gates import gate_candidates


def _proposed(path, rt, entries):
    path.write_text(json.dumps({
        "artifact_role": "proposal", "proposal_scope": "producer-processed-only",
        "profiles": {rt: entries},
    }))


def _model_cand(level, count):
    return {
        "level": level, "source_family": "model-proposed", "level_count": count,
        "level_grounding": {"status": "model-proposed", "count_evidence": "e", "selector": "s"},
        "count_evidence": "e", "selector": "s", "rationale": "r",
    }


def test_gate_flags_count_disagreement_on_reused_exact_label(tmp_path):
    out = tmp_path / "proposed.json"
    _proposed(out, "drug", {"x": {"levels": ["analgesic"], "level_counts": {"analgesic": 20}}})
    item = {"runtime_type": "drug", "surface": "ibuprofen", "aliases": ["advil"]}
    # exact reuse of "analgesic" but count 5000 vs recorded 20 -> disagreement
    res = gate_candidates(item, [_model_cand("analgesic", 5000)], proposed_out=str(out))
    assert any(d.get("reason") == "count_disagreement" for d in res.diagnostics)


def test_gate_flags_item_with_single_level(tmp_path):
    out = tmp_path / "proposed.json"
    out.write_text('{"profiles": {"drug": {}}}')
    item = {"runtime_type": "drug", "surface": "ibuprofen", "aliases": ["advil"]}
    res = gate_candidates(item, [_model_cand("analgesic", 30)], proposed_out=str(out))
    # one accepted level is below the >=2 floor -> the surviving level is diverted to diagnostics
    assert any(d.get("reason") == "too_few_levels" for d in res.diagnostics)
    assert not res.accepted
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_gates.py::test_gate_flags_count_disagreement_on_reused_exact_label src/cloak/tests/test_lattice_producer_gates.py::test_gate_flags_item_with_single_level -v`
Expected: FAIL — reasons not produced.

- [ ] **Step 3: Add `count_for` to the vocabulary**

In `vocabulary.py`:

```python
    def count_for(self, label: str) -> float | None:
        return self._labels.get(_norm(label))
```

- [ ] **Step 4: Add the count-disagreement check in the model-proposed branch**

In `gate_candidates`, inside the `if _is_model_proposed(candidate):` block, after the existing `has_exact` near-duplicate check, add:

```python
            if vocabulary is not None and vocabulary.has_exact(level):
                recorded = vocabulary.count_for(level)
                proposed = float(candidate.get("level_count", 1.0))
                if recorded and recorded > 0 and proposed > 0:
                    ratio = max(proposed / recorded, recorded / proposed)
                    if ratio > 4.0:
                        diagnostics.append({**record, "reason": "count_disagreement",
                                            "recorded_count": recorded})
                        continue
```

- [ ] **Step 5: Add the chain-length floor after the per-candidate loop**

At the end of `gate_candidates`, before `return GateResult(...)`, add:

```python
    # >=2-level floor (findings B): an eligible item that yields a single accepted level has no
    # real generalization chain. Divert the survivors to diagnostics so the item is retried
    # (route_after_gate -> requeue) rather than persisted as a degenerate 1-level row.
    if item.get("eligible", True) and 0 < len(accepted) < 2:
        diagnostics.extend({**row, "reason": "too_few_levels"} for row in accepted)
        accepted = []
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_gates.py -v`
Expected: PASS (new + existing gate tests).

- [ ] **Step 7: Commit**

```bash
git add src/cloak/lattice_producer/gates.py src/cloak/lattice_producer/vocabulary.py src/cloak/tests/test_lattice_producer_gates.py
git commit -m "feat(lattice-producer): gate min-chain-length and reused-label count agreement"
```

---

### Task 6: Coherence — derive counts from corpus membership, keeping certifying counts fixed (findings A; supersedes the reverted log-gap band)

**Why the redesign:** the earlier log-gap band (reverted in `61be00e`) forced a per-step minimum log-gap over the *entire* global label order. On the real drug corpus this compounded into fabricated 1e66 counts and 586 non-monotone per-entry chains, because pinned anchors that sit closer than `min_decades` apart make the forced spacing incompatible with monotonicity. Per the empirical-honesty rule ("if a method degenerates at fixed settings, report it, don't engineer around it"), shaping a fabricated quantity is abandoned. Instead, each level's count becomes a **real corpus-membership anonymity-set size** — the number of distinct entries whose generalization chain contains that level — which is monotone up a chain by construction and fabricates nothing.

**Files:**
- Modify: `src/cloak/lattice_producer/coherence.py` (`normalize_runtime_type`: replace the median-of-model-counts `baseline`/`weight` with corpus-membership counts; update the non-anchored grounding evidence text)
- Test: `src/cloak/tests/test_lattice_producer_coherence.py`

**Interfaces:**
- Consumes: `normalize_runtime_type(entries, runtime_type)`; the already-built `canonical_by_entry` (each entry's deduped canonical chain), `real_certifying_value`, `anchored_labels`.
- Produces: no new public function. `baseline[canon]` = number of distinct entries carrying `canon`; certifying/anchored values still override and are pinned (unchanged). Non-anchored levels get `count_basis: "corpus-membership"`.

- [ ] **Step 1: Write the failing test**

```python
# add to src/cloak/tests/test_lattice_producer_coherence.py
from cloak.lattice_producer.coherence import normalize_runtime_type


def test_counts_are_corpus_membership_sizes():
    # 3 entries; "broad" is carried by all 3, "mid" by 2, each specific by 1.
    # model-reported counts are deliberately nonsense (prevalence-scale) and must be overwritten.
    entries = {
        "e1": {"levels": ["alpha", "mid", "broad"],
                "level_counts": {"alpha": 9e9, "mid": 5e8, "broad": 8e9},
                "level_grounding": {}},
        "e2": {"levels": ["beta", "mid", "broad"],
                "level_counts": {"beta": 1e6, "mid": 2e8, "broad": 7e9},
                "level_grounding": {}},
        "e3": {"levels": ["gamma", "broad"],
                "level_counts": {"gamma": 4e5, "broad": 6e9},
                "level_grounding": {}},
    }
    normalize_runtime_type(entries, "health-condition")
    # broad carried by e1,e2,e3 -> 3 ; mid by e1,e2 -> 2 ; each specific -> 1
    assert entries["e1"]["level_counts"]["broad"] == 3.0
    assert entries["e1"]["level_counts"]["mid"] == 2.0
    assert entries["e1"]["level_counts"]["alpha"] == 1.0
    # every chain stays monotone non-decreasing
    for row in entries.values():
        cs = [row["level_counts"][l] for l in row["levels"]]
        assert cs == sorted(cs)
    # the non-anchored grounding says corpus-membership, not certifying
    assert entries["e1"]["level_grounding"]["broad"]["status"] == "model-proposed"
    assert entries["e1"]["level_grounding"]["broad"]["count_basis"] == "corpus-membership"


def test_certifying_count_still_pinned_not_membership():
    # a level with a real certifying member-set count must keep that count, not the membership size
    entries = {
        "e1": {"levels": ["benzodiazepine", "broad"],
                "level_counts": {"benzodiazepine": 24.0, "broad": 5.0},
                "level_grounding": {"benzodiazepine": {"status": "certifying",
                                                        "member_set_ref": "openfda-ndc:pharm_class:X"}}},
    }
    normalize_runtime_type(entries, "drug")
    assert entries["e1"]["level_counts"]["benzodiazepine"] == 24.0  # pinned, not 1 (its membership)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_coherence.py::test_counts_are_corpus_membership_sizes -v`
Expected: FAIL — counts are the model's median values, not membership sizes.

- [ ] **Step 3: Replace the baseline/weight computation with corpus-membership counts**

In `normalize_runtime_type` (coherence.py), find:

```python
    baseline = {canon: statistics.median(vals) for canon, vals in raw_counts_by_canonical.items()}
    weight = {canon: float(len(vals)) for canon, vals in raw_counts_by_canonical.items()}
```

and replace with:

```python
    # Corpus-membership counts: each level's count is the number of DISTINCT entries whose
    # generalization chain contains it -- a real anonymity-set-within-corpus size, not the
    # model's fabricated per-item number. Monotone up a chain by construction (a broader level
    # is carried by a superset of the entries carrying any level that always rolls up into it),
    # so no forced log-spacing is needed. Certifying/anchored counts still override below.
    membership: dict[str, set[str]] = defaultdict(set)
    for entry_key, chain in canonical_by_entry.items():
        for canon in chain:
            membership[canon].add(entry_key)
    baseline = {canon: float(len(members)) for canon, members in membership.items()}
    weight = {canon: float(len(members)) for canon, members in membership.items()}
```

(`raw_counts_by_canonical` is still collected above for the `real_certifying_value` detection and the early-return guard — leave that loop untouched. `defaultdict` is already imported.)

- [ ] **Step 4: Update the non-anchored grounding evidence text**

In the grounding-annotation block of `normalize_runtime_type`, the non-anchored `else` branch currently sets `count_basis = "corpus-wide-rank-coherent"` with an evidence string about "average-depth ranking". Replace that branch's two assignments with:

```python
                    grounding["count_basis"] = "corpus-membership"
                    grounding["count_evidence"] = (
                        f"'{canon}' count is the number of distinct entries in this run whose "
                        f"generalization chain includes it (corpus-membership anonymity-set "
                        f"size); not certifying"
                    )
```

Leave the anchored (`real-world-reference-estimate`) and certifying branches exactly as they are.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_coherence.py -v`
Expected: PASS for the two new tests. **Existing coherence tests that asserted specific median-derived count values will now fail** because counts are membership sizes, not model-count medians. Reconcile each such assertion inside `test_lattice_producer_coherence.py` ONLY, to the new membership semantics (recompute the expected count as the number of distinct entries carrying the label in that test's fixture). Do not weaken a test to pass — the expected value must be the correct membership count. If a test's intent was specifically to check median/PAVA behavior on model counts and can no longer be expressed, note it in the report rather than deleting it. Do not touch any file outside the two allowed ones; if an existing test outside this file fails, STOP and report it.

- [ ] **Step 6: Commit**

```bash
git add src/cloak/lattice_producer/coherence.py src/cloak/tests/test_lattice_producer_coherence.py
git commit -m "feat(lattice-producer): derive level counts from corpus membership, certifying counts pinned"
```

---

### Task 7: Run coherence normalization periodically, not only at queue end (register #4)

**Files:**
- Modify: `src/cloak/lattice_producer/graph.py:376-423` (`record_item_result`) and `:426-447` (`should_continue`, `normalize_coherence_node`)
- Modify: `src/cloak/lattice_producer/state.py` (add `normalize_every: int` to state + initial state)
- Test: `src/cloak/tests/test_lattice_producer_graph.py`

**Interfaces:**
- Consumes: `should_continue(state)` routing.
- Produces: `should_continue` returns `"normalize_coherence"` when `processed % normalize_every == 0` (and `processed > 0`), then coherence must route back into the processing loop instead of straight to validation, until the queue is genuinely done.

- [ ] **Step 1: Write the failing test**

```python
# add to src/cloak/tests/test_lattice_producer_graph.py
from cloak.lattice_producer.graph import should_continue


def test_should_continue_triggers_periodic_normalization():
    state = {"processed": 50, "max_items": None, "normalize_every": 50}
    assert should_continue(state) == "normalize_coherence"
    state = {"processed": 49, "max_items": None, "normalize_every": 50}
    assert should_continue(state) == "select_next_item"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_graph.py::test_should_continue_triggers_periodic_normalization -v`
Expected: FAIL — periodic branch not implemented.

- [ ] **Step 3: Add `normalize_every` to state**

In `state.py`, add `normalize_every: int` to `ProducerState`, add a `normalize_every: int = 50` parameter to `make_initial_state`, and set `"normalize_every": normalize_every` in the returned dict.

- [ ] **Step 4: Update `should_continue` and the graph edge**

Change `should_continue`:

```python
def should_continue(state: ProducerState) -> Literal["select_next_item", "normalize_coherence"]:
    max_items = state.get("max_items")
    processed = int(state.get("processed", 0))
    if max_items is not None and processed >= int(max_items):
        return "normalize_coherence"
    every = int(state.get("normalize_every", 0) or 0)
    if every > 0 and processed > 0 and processed % every == 0:
        return "normalize_coherence"
    return "select_next_item"
```

Then make periodic normalization return to the loop. In `build_graph`, change the fixed edge `graph.add_edge("normalize_coherence", "validate_proposed_artifact")` to a conditional:

```python
    graph.add_conditional_edges("normalize_coherence", _route_after_normalize)
```

and add the router:

```python
def _route_after_normalize(state: ProducerState) -> Literal["select_next_item", "validate_proposed_artifact"]:
    max_items = state.get("max_items")
    processed = int(state.get("processed", 0))
    done = (max_items is not None and processed >= int(max_items))
    # queue-empty end also arrives here via route_selected -> normalize_coherence with no
    # current_item; validate only when the run is actually finished, else resume processing.
    return "validate_proposed_artifact" if done or state.get("current_item") is None and state.get("queue_exhausted") else "select_next_item"
```

Set `queue_exhausted` in `select_next_item`'s no-more-items return (`{"current_item": None, "queue_index": idx, "queue_exhausted": True}`) and add `queue_exhausted: bool` to `ProducerState`. When `route_selected` sends an empty queue to `normalize_coherence`, `queue_exhausted` is True so the router validates; a periodic trigger has `queue_exhausted` False so it resumes.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest src/cloak/tests/test_lattice_producer_graph.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cloak/lattice_producer/graph.py src/cloak/lattice_producer/state.py src/cloak/tests/test_lattice_producer_graph.py
git commit -m "feat(lattice-producer): run coherence normalization periodically, resume loop after"
```

---

### Task 8: Full suite + perf gate + supervised smoke re-run with fresh damage re-measurement (register #8 temporal check)

**Files:**
- Create: `scripts/spikes/measure_lattice_run_quality.py` (chunk-bucketed damage re-measurement, per register #8)
- No production code changes.

**Interfaces:**
- Consumes: a completed proposed artifact + the run's `accepted.jsonl`.
- Produces: prints the register's tables — per 5 chronological chunks: `fully_generic %`, `new_specific_labels`, top sinks; plus aggregate `count_disagreement` rate and chain-length histogram.

- [ ] **Step 1: Run the full producer test suite**

Run: `.venv/bin/python -m pytest src/cloak/tests/ -q`
Expected: all pass (the 255 baseline plus the new tests). Fix any regression before proceeding.

- [ ] **Step 2: Write the re-measurement spike**

Create `scripts/spikes/measure_lattice_run_quality.py` that: loads `accepted.jsonl` in file (=processing) order, splits into 5 equal chunks, and per chunk prints the % of items whose only levels are in a generic-sink set, the count of first-seen specific labels, and the top-3 most-frequent levels. Also prints the chain-length histogram and, from the proposed artifact, the fraction of reused labels whose counts disagree >4×. (Model this on the tables in `docs/issues/2026-07-09-lattice-producer-generation-quality-issue-register.md`.)

- [ ] **Step 3: Perf-gate the smoke run**

Read `scripts/harness/perf_gate.md`. The smoke run is `--max-items 60 --category health-condition --category medical-procedure` (small, answers "did #1 + #3 + #4 recover chunk-3+ diversity"). Estimate wall-time; with a 2048 thinking budget and ~1–3 min/item this is ~1–3h for 60 items — **confirm GPU saturation and get user go-ahead before launch** (memory: gpu-occupancy-check, no-paid-models-without-permission). If too slow, drop to `--max-items 30`.

- [ ] **Step 4: Launch the smoke run (after user OK + GPU check)**

Run (background, unbuffered):

```bash
rocm-smi --showpidgpus   # confirm no other GPU process first
.venv/bin/python -u scripts/run_lattice_producer.py \
  --run-dir data/lattice_runs/smoke-overhaul \
  --profiles data/lattice_profiles/lattice_profiles.json \
  --out data/lattice_profiles/proposed/smoke-overhaul.proposed.json \
  --category health-condition --category medical-procedure \
  --max-items 60 --normalize-every 30 \
  2>&1 | tee data/lattice_runs/smoke-overhaul/run.log
```

(Add `--normalize-every` to `scripts/run_lattice_producer.py` argparse + pass-through if not already wired; default 50.)

- [ ] **Step 5: Re-measure and compare to the register's baseline tables**

Run: `.venv/bin/python scripts/spikes/measure_lattice_run_quality.py data/lattice_profiles/proposed/smoke-overhaul.proposed.json data/lattice_runs/smoke-overhaul/accepted.jsonl`
Expected/success criteria: chunk-3-through-5 `new_specific_labels` rate is materially above the register's ~20/chunk collapse; `fully_generic %` does not ramp to 29%+; some `source_family` values are now `openfda`/`doid-is-a`/`icd10pcs-prefix` (register #1 fixed); `count_disagreement` rate near zero; chain-length histogram has no length-1 entries. Record the numbers; if the temporal collapse persists, that is the finding — report it (do not tune around it, empirical-honesty rule).

- [ ] **Step 6: Update the issue register and commit**

Append a "Post-overhaul re-measurement" section to `docs/issues/2026-07-09-lattice-producer-generation-quality-issue-register.md` with the measured tables and which issues are resolved vs. still open (esp. register #13 ICD-10-PCS coverage, which this overhaul does not add a new source for).

```bash
git add scripts/spikes/measure_lattice_run_quality.py docs/issues/2026-07-09-lattice-producer-generation-quality-issue-register.md
git commit -m "chore(lattice-producer): post-overhaul quality re-measurement spike + register update"
```

---

## Deferred / not in this plan

- **Register #5, #3 (auto-clustering of already-accepted synonyms):** the static `CLUSTERS` table is untouched; Task 4's relevance-ranked reuse + Task 5's count-agreement gate reduce synonym proliferation at the source, so a live auto-clustering pass is deferred until the smoke re-run shows whether it is still needed. `# ponytail:` — add Jaccard/embedding auto-clustering in `normalize_coherence` only if post-run synonym clusters persist.
- **Register #13 (medical-procedure reference coverage):** adding a second source (SNOMED/CPT/MeSH) is a separate, larger effort — the fork-2 answer kept the ontology backbone without a new source. Flagged for its own plan if the smoke run confirms ICD-10-PCS coverage is the binding constraint.
- **Register #12 (stateless per-item calls):** structural; not changed here.

## Self-Review

- **Spec coverage:** register #1→Task 1; #7→Task 2; #9,#10→Tasks 3,4; #6,#11→Task 4; #2→Task 5; #8(B)→Tasks 3,5; user count-shape request + finding A→Tasks 3,6; #4→Task 7; #8 temporal re-check→Task 8. Deferred: #3,#5,#12,#13 (documented above).
- **Type consistency:** `context_slice` returns `list[dict]` with `label`/`count` keys in Task 4 and is consumed as such by the packet; `count_for` added in Task 5 and used in the same task's gate; `enforce_log_gap_band` signature identical in Tasks 6 test and impl; `normalize_every`/`queue_exhausted` added to `ProducerState` in Task 7 where used.
- **Placeholders:** none — every code step shows the code.
