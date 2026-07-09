---
type: handoff
status: current
created: 2026-07-09
updated: 2026-07-09
tags: [lattice-producer, drug, hybrid-anchor, openfda, full-run, queue, k-floor]
companion: docs/superpowers/plans/2026-07-09-lattice-producer-overhaul.md
supersedes:
---

# Handoff — drug hybrid lattice path + running the full-run queue

## Where things stand

The lattice-producer overhaul (plan: `docs/superpowers/plans/2026-07-09-lattice-producer-overhaul.md`,
issue register with post-run results: `docs/issues/2026-07-09-lattice-producer-generation-quality-issue-register.md`)
is **implemented, reviewed, and merged to `main`** (commits `19b83aa`..`3285369`, all tagged
`lattice-producer` in `git log`). Validated end-to-end against OpenRouter
(`nvidia/nemotron-3-super-120b-a12b:free`):

- **health-condition** (60-item run, `data/lattice_runs/smoke-overhaul`): 215/215 accepted levels
  `certifying` via DOID, register #8 collapse eliminated, 0% count-disagreement, no length-1 chains.
- **medical-procedure** (30-item run, `data/lattice_runs/smoke-mp`): 22/26 model-proposed + 4
  certifying — confirms register #13 (ICD-10-PCS ~7% coverage, model carries the rest); the
  specificity prompt yields specific nearest tiers (`adenoidectomy → ENT surgical procedure → …`).

Current branch: **`drug-hybrid-lattice`** (base `3285369`), empty so far — created to implement the
task below. `main` is 39 commits ahead of `origin/main` (NOT pushed; push only if asked).

## The task for this session: drug hybrid anchor + model tiers

**Decision made (user-approved):** implement the hybrid path — *keep the deterministic certifying
anchor as the specific nearest rung, and have the model add broader tiers above it*. Make `drug`
producer-eligible. This also fixes a general bug affecting single-rung procedure/health hits.

### Why (the gap this closes — all confirmed in code)

- `drug` is a real domain type but openFDA (`data/lattice_sources/raw/drug/openfda_ndc.json.zip`,
  the only drug source — no ATC/hierarchy) returns a **single flat EPC tier**, count median ~7,
  **below the k-floor of 100** (`K_FLOORS`, `src/cloak/anonymity.py:40`). So a drug entry needs the
  model for broader tiers to become anonymizable — deterministic alone is insufficient.
- `drug` is currently gated `NEEDS_PROFILE` (`src/cloak/lattice_producer/coverage.py:81`), so the
  producer skips it entirely (`route_selected` returns early on `eligible: False`).
- If drug were made eligible as-is, a single anchor trips the `too_few_levels` gate →
  `route_after_gate` → `requeue_rejected_item` (`graph.py:351`) which **sets
  `current_candidates: []`**, discarding the certifying anchor; the model then regenerates the whole
  chain (loses grounding ≈ the old model-generated-drug bug `NEEDS_PROFILE` was added to stop).
- The same requeue-discards-anchor flaw hits ANY single-rung deterministic hit (the ~12 ICD
  procedure hits; any 1-hop DOID health hit) — so the fix is broadly valuable, not drug-only.

### Implementation plan (TDD, in order)

The current flow (verified): `select_next_item → route_selected → deterministic_lookup →
route_after_deterministic` (`graph.py:248`) — if candidates → `compile_level_counts → gate_candidates
→ route_after_gate`; a single-rung hit trips `too_few_levels`/`chain_below_floor` in the gate, then
`route_after_gate → requeue_rejected_item` (`graph.py:351`) which does `"current_candidates": []`,
**discarding the anchor**, then `→ propose_with_llama_swap`. The fix inserts an *augment* node between
deterministic lookup and compile for insufficient chains, so the anchor is kept and the model only
adds tiers.

Reference candidates from `reference_sources.reference_candidates_for` carry `member_set` (a
`frozenset`), `source_family` (`openfda-pharm-class`/`doid-is-a`/`icd10pcs-prefix`), `selector`,
`member_set_ref`. `compile_level_counts` (`counts.py`) turns a candidate with `member_set` into a
`certifying` level, count = `len(member_set)`, and pops the frozenset. Keep that intact.

**Task 1 — registry: make `drug` producer-eligible** (`src/cloak/lattice_producer/coverage.py:81`).
Change:
```python
*[_entry(label, CategoryOutcome.NEEDS_PROFILE, None) for label in ("medical process", "drug", "blood type")],
```
to:
```python
*[_entry(label, CategoryOutcome.NEEDS_PROFILE, None) for label in ("medical process", "blood type")],
_entry("drug", CategoryOutcome.RUNTIME_LATTICE, "drug"),
```
Test first (`test_lattice_producer_coverage.py` or `_queue.py`): `registry_outcome_for_runtime_type("drug")
== CategoryOutcome.RUNTIME_LATTICE`; `medical process`/`blood type` still `NEEDS_PROFILE`; a drug queue
item normalizes to `eligible: True` (currently `False`).

**Task 2 — pure merge helper** (new, unit-tested; put in `graph.py` or a small helper imported by it).
```python
import re
def _lvl_tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))

def merge_anchor_and_model(anchors: list[dict], model_cands: list[dict], *, near_dup=0.5) -> list[dict]:
    """Keep the deterministic certifying anchor(s) as the nearest rung(s); append the model's
    broader tiers, dropping any model tier that is a near-duplicate (token-Jaccard >= near_dup) of
    an anchor -- the certifying anchor wins over the model's ungrounded paraphrase of the same tier.
    Ordering/counts are fixed later by compile_level_counts + coherence, so this need not sort."""
    out = list(anchors)
    anchor_tok = [_lvl_tokens(a.get("level", "")) for a in anchors]
    for m in model_cands:
        mt = _lvl_tokens(m.get("level", ""))
        if not mt:
            continue
        if any(at and len(mt & at) / len(mt | at) >= near_dup for at in anchor_tok):
            continue
        out.append(m)
    return out
```
Test: anchor `[{level:"benzodiazepine", source_family:"openfda-pharm-class", member_set:frozenset(...)}]`
+ model `[{level:"benzodiazepine derivative"}(near-dup, dropped), {level:"cns depressant"}, {level:"medication"}]`
→ `["benzodiazepine","cns depressant","medication"]`, anchor dict unchanged (still has `member_set`).

**Task 3 — sufficiency check + augment node + routing** (`graph.py`).
Add `from cloak.anonymity import K_FLOORS` at the top. Then:
```python
def _deterministic_chain_sufficient(state: ProducerState) -> bool:
    """A deterministic chain is enough on its own iff it has >=2 rungs AND its broadest rung already
    reaches the k-floor (so the release-time walk has a legal target). Single-rung / below-floor hits
    (all drug openFDA EPC hits; the ~12 ICD procedure hits; any 1-hop DOID) need model broader tiers."""
    cands = state.get("current_candidates") or []
    rt = str((state.get("current_item") or {}).get("runtime_type") or "")
    floor = float(K_FLOORS.get(rt, 100.0))
    sizes = [len(c["member_set"]) for c in cands if c.get("member_set")]
    return len(cands) >= 2 and bool(sizes) and max(sizes) >= floor

def augment_with_model_node(state: ProducerState) -> ProducerState:
    """Insufficient deterministic hit: KEEP the certifying anchor(s) and call the model for broader
    tiers, then merge. Mirrors propose_with_llama_swap_node's model call, but merges instead of
    replacing current_candidates."""
    item = state["current_item"] or {}
    anchors = list(state.get("current_candidates") or [])
    model = state.get("model") or QWEN36_ESCALATION_MODEL
    proposal = propose_with_llama_swap(
        item, profiles_path=state["profiles_path"], run_dir=state["run_dir"],
        prompt_version=state["prompt_version"], max_context_rows=state["max_context_rows"],
        base_url=state["base_url"], model=model,
        escalation_model=state.get("escalation_model") or model,
        thinking_budget_tokens=int(state.get("thinking_budget_tokens", -1)),
        proposed_out=state["proposed_out"],
    )
    append_jsonl_unique(_jsonl_path(state, "proposals.jsonl"),
                        [{**proposal, "item_id": item.get("item_id"), "augment": True, "model_used": model}])
    model_cands = extract_candidate_levels(proposal)
    return {"current_candidates": merge_anchor_and_model(anchors, model_cands)}
```
Update `route_after_deterministic` (`graph.py:248`) to add the augment branch:
```python
def route_after_deterministic(state) -> Literal["compile_level_counts","augment_with_model","propose_with_llama_swap","record_item_result"]:
    if state.get("current_candidates"):
        if state.get("offline_only") or _deterministic_chain_sufficient(state):
            return "compile_level_counts"
        return "augment_with_model"
    if state.get("offline_only"):
        return "record_item_result"
    return "propose_with_llama_swap"
```
Wire the node in `build_graph` (near `graph.py:627`/`644`):
```python
graph.add_node("augment_with_model", augment_with_model_node)
graph.add_edge("augment_with_model", "compile_level_counts")
```
(The `add_conditional_edges("deterministic_lookup", route_after_deterministic)` at `:644` already routes
the new return value to the node by name — no other edge change needed.)

**Task 4 — graph e2e test** (`test_lattice_producer_graph.py`, mirror the existing
`test_dynamic_vocabulary_*` style; mock the model, do NOT hit the network). Monkeypatch
`graph.reference_candidates_for` to return one certifying anchor
`[{"level":"benzodiazepine","source_family":"openfda-pharm-class","member_set":frozenset({"a","b",...}),"member_set_ref":"..."}]`
(member_set size < 100 so it's insufficient), and monkeypatch `graph.propose_with_llama_swap` to return
broader tiers `{"candidates":[{"level":"central nervous system depressant",...},{"level":"medication",...}]}`.
Drive `deterministic_lookup → route_after_deterministic` (assert it returns `"augment_with_model"`) →
`augment_with_model_node → compile_level_counts_node → gate_candidates_node`; assert: the merged chain
is accepted, rung 0 is `benzodiazepine` with grounding `certifying`, and the chain now reaches the
k-floor via a model tier (so no `chain_below_floor`/`too_few_levels`). Also assert a *sufficient* DOID
chain (>=2 rungs, big top) routes straight to `compile_level_counts` (no augment).

**Task 5 — full suite green** (`PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/ -q`; was 274
on `main`). Then a real end-to-end smoke: a small drug batch via OpenRouter (see run command below with
`--category drug --max-items 20`) — confirm accepted drug entries have a `certifying` openFDA rung 0 +
model broader tiers, and inspect `data/lattice_profiles/proposed/<out>.json`.

### Known limitations / decisions (don't rathole)

- **Counts:** the model's `proposed_count` is unreliable (finding A) — `compile_level_counts` +
  coherence recompute model tiers from corpus membership, and pin the anchor's real member-set count.
  So `merge_anchor_and_model` deliberately does NOT sort by count; ordering is coherence's job.
- **Augment fallback:** if the model returns nothing usable and the merged chain still fails the gate,
  `route_after_gate` will `requeue_rejected_item` (one retry, `MAX_REJECTION_RETRIES=1`) which *does*
  clear candidates → last-resort model regen (anchor lost). Acceptable as a rare fallback; do not
  redesign requeue for it unless a run shows it matters.
- **"broader-only" prompt mode:** NOT needed — reuse the existing full-chain prompt and let the merge
  drop the model's near-dup of the anchor. Simpler and good enough; the specificity prompt already
  makes the model's own tiers sensible.

### Regenerate the queue to include drug (after Task 1 lands)

```python
PYTHONPATH=src .venv/bin/python - <<'PY'
import json, itertools
from pathlib import Path
art = json.load(open("data/lattice_profiles/proposed/drug-health-procedure.proposed.json"))
per = {}
for rt in ("drug", "health-condition", "medical-procedure"):   # drug now included
    per[rt] = [{"item_id": f"{rt}:{s}", "task_kind": "level-proposal", "runtime_type": rt,
                "detector_label_family": rt, "surface": s, "canonical_value": s,
                "aliases": list(r.get("aliases", []))}
               for s, r in sorted(art["profiles"].get(rt, {}).items())]
items = [x for x in itertools.chain.from_iterable(itertools.zip_longest(*per.values())) if x is not None]
Path("data/lattice_runs/full-run").mkdir(parents=True, exist_ok=True)
Path("data/lattice_runs/full-run/queue.jsonl").write_text("".join(json.dumps(i, sort_keys=True)+"\n" for i in items))
print(f"{len(items)} items:", {k: len(v) for k, v in per.items()})
PY
```
(Current queue on disk is 809 items = health+procedure only, drug excluded pending this feature; the
above rebuilds it to 1347 incl. drug.)

### Watch-outs (bit us this session)

- `member_set` is a `frozenset` — it must be dropped before persistence (`counts.py` already pops
  it; keep that path intact for the merged deterministic candidate).
- OpenRouter free endpoints can return `choices=None` and 429 — already handled by
  `_create_with_retry` (`propose.py`); don't regress it.
- Empirical-honesty rule (`CLAUDE.md`): the anchor's certifying count is pinned; never fabricate or
  shape-force counts. Verify any new run on the REAL artifact, not just toy tests (the reverted
  log-gap band `61be00e` passed 265 toy tests but degenerated on the 831-label corpus).
- Do not sweep unrelated pre-existing dirty files into commits (`CLAUDE.md`,
  `data/lattice_profiles/proposed/drug-health-procedure.proposed.json`, reconstructor specs, etc.).
  A parallel session commits to shared branches — use path-scoped `git add` and path-scoped review diffs.

## Running the prepared full-run queue

A ready queue for the producer-eligible types is at **`data/lattice_runs/full-run/queue.jsonl`**
(809 items: health-condition 646 + medical-procedure 163, interleaved round-robin, each carrying
aliases; drug deliberately excluded until the hybrid path lands). Rebuild it (e.g. to add drug once
eligible) by re-running the inline generator that produced it — see this handoff's git history / the
transcript; it reads surfaces+aliases from `drug-health-procedure.proposed.json` and discards old
levels.

Validate the queue before launching:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from cloak.lattice_producer.queue import build_or_load_queue
import collections, tempfile
items = build_or_load_queue(tempfile.mkdtemp(), "data/lattice_profiles/lattice_profiles.json",
                            explicit_queue="data/lattice_runs/full-run/queue.jsonl")
print("eligible:", dict(collections.Counter(i.get("eligible", True) for i in items)))
PY
```

Launch the full run (OpenRouter free model; set the key from `.env`):

```bash
export OPENROUTER_API_KEY=$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2- | tr -d '"'\''')
export PYTHONPATH=src
.venv/bin/python -u scripts/run_lattice_producer.py \
  --run-dir data/lattice_runs/full-run \
  --profiles data/lattice_profiles/lattice_profiles.json \
  --out data/lattice_profiles/proposed/full-run.proposed.json \
  --queue data/lattice_runs/full-run/queue.jsonl \
  --base-url https://openrouter.ai/api/v1 \
  --model nvidia/nemotron-3-super-120b-a12b:free \
  --normalize-every 50 > data/lattice_runs/full-run.log 2>&1
```

Notes:
- `--out` must be under `data/lattice_profiles/proposed/`. The run pauses at a review-interrupt at
  the end (`status=None`); per-item persistence + periodic coherence mean the artifact is complete
  before that — inspect it directly.
- **Resume, don't restart** on a crash: re-run with the SAME `--run-dir` (or `--resume`); the
  producer checkpoints per item and `select_next_item` skips `item_id`s already in
  `accepted/rejected/diagnostics.jsonl`. (Deleting the run dir loses that — a mistake made once this session.)
- Budget: ~1,100 model calls for a full run incl. drug tiers; free but multi-hour with free-tier 429
  throttling. `--max-items N` for a capped smoke; the interleaved queue samples all types.
- Re-measure quality with the spike:
  `PYTHONPATH=src .venv/bin/python scripts/spikes/measure_lattice_run_quality.py <proposed.json> <run-dir>/accepted.jsonl`
  and append results to the issue register (compare per-chunk fully_generic / new_specific vs the
  register #8 baseline; confirm source_family mix and chain-length histogram).

## Drug is separate from health/procedure

Drug profiles historically come from `scripts/build_lattice_profiles.py` (deterministic openFDA;
canonical `lattice_profiles.json` drug entries carry `openfda-ndc` source_ids). That builder yields
only the single EPC anchor. The hybrid path above is what lets drug also get model broader tiers
through the producer — after it lands, add drug to the queue and re-run.

## Model / infra facts

- Pinned model: `nvidia/nemotron-3-super-120b-a12b:free` (strongest free JSON+reasoning model that
  was actually available; Qwen3-Next-80B and Gemma-4-31B free endpoints 429'd). `openrouter/free`
  auto-router routes non-deterministically (to a weak 9B) — don't use it.
- OpenRouter wiring lives in `src/cloak/lattice_producer/propose.py` (base-url allowlist includes
  `openrouter.ai`, `OPENROUTER_API_KEY`, reasoning `extra_body`, model threaded from state).
- GPU/local path unchanged: `--model ""` defaults to `Qwen3.6-35B-A3B` on local llama-swap
  (`http://localhost:8060/v1`).

## Suggested skills

- `superpowers:writing-plans` then `superpowers:subagent-driven-development` — if you want the
  hybrid feature run task-by-task with review gates (matches how the overhaul was executed). For a
  focused single-feature change, `superpowers:test-driven-development` inline + one review pass is
  enough.
- `superpowers:requesting-code-review` / the `auto-review-loop` (codex gpt-5.5) — review the graph
  control-flow change before merging; core producer flow has repeatedly hidden subtle bugs.
- `superpowers:verification-before-completion` — verify any new run on the REAL artifact, not toy
  tests, before claiming success.
- Subagent model rule (memory): every `Agent` dispatch uses `model: "opus"` explicitly.
