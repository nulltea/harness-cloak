---
type: handoff
status: current
created: 2026-07-02
updated: 2026-07-02
tags: [d2, way-2, round-trip-reward, training-free-search, go-no-go, handoff]
companion: [../plans/2026-07-02-roundtrip-grpo-training.md, ../plans/2026-07-02-codesign-next-stage.md, ../plans/2026-07-02-d1-prototype-implementation.md]
---

# Handoff: round-trip search probe (D2 · Way 2)

## Task for the next session

Implement and run the **training-free per-document search probe**: for each document, search over
the substitutor's per-span lattice-level choices, score every candidate with the real round-trip
reward `r = α·(1−A(doc_p)) + (1−α)·U(out_final)`, keep the best, and compare the resulting Pareto
curve against D1's τ-walk at matched realized privacy.

**Why it exists:** it is the **go/no-go gate for GRPO training** (Way 1). Upper-bound argument: a
trained policy can only learn choices that score well under the reward — if direct per-document
search with the true reward in hand can't shift the frontier vs the τ-walk, RL won't either; kill
D2's training leg cheaply. If it does shift, the scored candidates double as SFT/DPO training data
for Way 1. Full mechanism, definitions, and the Way-1 counterpart:
[`2026-07-02-roundtrip-grpo-training.md`](../plans/2026-07-02-roundtrip-grpo-training.md).

## Context established this session (2026-07-02)

- **Search space:** D1's substitutor reduces each doc to structured discrete choices — one lattice
  level per quasi-identifier span (product space, L^S joint assignments; e.g. 6 spans × 4 levels =
  4096). The τ-walk is one heuristic point in that space: per-span, first level with MTI guess-back
  risk < τ, blind to the round-trip outcome.
- **Binding constraint:** each candidate scored = one real round trip (remote proxy call +
  extractor pass + attack-head pass). Budget ≈ 10–30 reward evaluations per document, not L^S.
- **NaPaRe reality check:** the parent plan's "port NaPaRe's tree search" was reviewed — NaPaRe's
  wiki page ([huang2025_tree_search_rewriting.md](../../research-wiki/papers/huang2025_tree_search_rewriting.md))
  records **no code release**, and its actions are free-form LLM-proposed edits, not structured
  lattice choices. A port = reimplementation onto a mismatched action space.

## Unresolved fork — decide first (grill the user)

Search scaffold, options as framed this session (user was mid-grilling when the session pivoted to
Way 1; **not yet decided**):

1. **Best-of-n sampling (was the session's recommendation):** per doc, score the τ-walk solution +
   n≈8–16 sampled joint assignments (guided sampling, e.g. softmax over MTI probe risks), one
   batched round-trip round, keep argmax. Laziest, fully parallel, directly yields DPO pairs.
2. **Greedy coordinate ascent:** seed at τ-walk; per span try adjacent levels, keep improvements,
   iterate. More sample-efficient in structured spaces; sequential (poor batching), local optima.
3. **NaPaRe tree-search port:** faithful to the parent plan's text, citable as "their method, our
   objective"; most work, mismatched action space, no code to port.

Also open (shared with Way 1, see its plan's "Open design forks"): local attack head `A` (MTI
probe exists; encoder re-id classifier and reusable heads from eth-sri/AgentStealth/SEAL to be
surveyed) and utility term `U` (decide by measurement on D1 tuples — which cheap metric tracks the
D1 headline utility; no literature pass, per grilling decision).

## Prerequisites (check before implementing)

- **D1 P4 does not exist yet** (repo state at handoff: P0–P3 + rung-A extractor done through the
  8-doc e2e smoke, commit `f8f60f3`; no attacker eval, no τ-sweep Pareto, no rung-B LoRA). The
  probe needs the τ-walk baseline curve and an extractor to invert with — **finish or scope-cut
  D1 P4 first** (attacker + τ sweep at v0.1 scale, 60 SynthPAI docs, is enough; rung B optional —
  rung A rules are a valid frozen extractor for the probe).
- **SEAL ([arXiv 2506.01420](https://arxiv.org/abs/2506.01420)) registration + overlap assessment**
  — mandated by the D1 plan before D2 work; not yet done.

## Where things live

- Substitutor cascade: `src/cloak/` (`substitute.py` = τ-walk + R emission, `probe.py` = MTI,
  `extract.py` = rung A, `tasks.py`, `detect.py`, `lattice.py`); scripts `scripts/d1_*.py`.
- Proxy + cache: `src/inferdpt/llm.py` (`LLMClient`, content-addressed cache, `pmap`). Roles/model
  ids: D1 plan "Role assignment" section. Corpora: `corpora/synthpai/`; tuples
  `data/latticecloak_tuples/`.
- Rules that bind: CLAUDE.md empirical honesty (no calibration knobs; compare only at matched
  realized privacy), perf gate before heavy runs, one GPU process, `-u` logging, no plan-ids in
  code/artifact names (name things `roundtrip_search_*`, not `d2_*`).

## Suggested skills for the next session

`/grill-me` (resolve the search-scaffold fork + attack-head/U choices before coding),
`/auto-review-loop` with `scripts/harness/perf_gate.md` (before the sweep run), `/html-report`
(after results land).
