---
type: research
status: current
created: 2026-07-06
updated: 2026-07-06
tags: [tech-debt, refactor, inferdpt, namespace, imports, rantext-retired, issue]
companion: [../../src/inferdpt/pipeline.py, ../../src/inferdpt/llm.py]
---

# Issue: live `cloak` pipeline still imports generic infra from the retired `inferdpt` namespace

## Summary

`inferdpt/` tangles two things and only one is retired. The **method** (RANTEXT/InferDPT) is
dead — `rantext.py`, `attacks/`, `embeddings.py`, `extraction.py`, `probes/`, the `InferDPT`
class in `pipeline.py`, and the `scripts/latticecloak_*` + `dp_sweep.py` sweeps. But the
**generic plumbing** in that namespace is load-bearing for the *current* pipeline and imported
everywhere: `inferdpt.llm.LLMClient` (+ `_cache_path`), `inferdpt.pipeline.pmap`, and the
`INFERDPT_LLM_CACHE` env var (round-trip reward determinism cache). So it's migration debt —
infra never moved out of a dead method's namespace — not a live dependency on dead logic.

## The real bug (not cosmetic)

`src/inferdpt/pipeline.py` has **module-level** imports of the retired stack:

```python
from inferdpt.embeddings import VocabEmbeddings
from inferdpt.extraction import extract
from inferdpt.rantext import Perturber
```

So every current `from inferdpt.pipeline import pmap` (reward path, roundtrip, probes,
reward_gate, tasks, lattice, many spikes) **transitively imports the whole RANTEXT machinery at
load time** — just to get a ~6-line `ThreadPoolExecutor` wrapper.

## Fix (clean end-state: `inferdpt/` = pure retired method, self-contained)

1. Lift `pmap` out of the RANTEXT-tainted `pipeline.py` into a tiny `cloak` module (it's ~6
   lines of stdlib — candidate to inline). **Highest value / lowest risk**: kills the transitive
   RANTEXT import on the hot path.
2. Move `LLMClient` + `_cache_path` → `cloak/llm.py`.
3. Rename `INFERDPT_LLM_CACHE` → `CLOAK_LLM_CACHE`.

## Cost / why deferred (2026-07-06)

~30 import sites, plus the env-var name in code **and every shell command in the
runbooks/handoffs**. A parallel session shares this git checkout (see the reader-perf handoff's
Gotchas) — a wide rename mid-flight collides with their uncommitted work. Deferred by user
decision while the reader-perf fix + RL pilot are in progress. Do it as a dedicated pass when the
checkout isn't shared; step 1 alone is safe to land early.

## Affected (survey 2026-07-06)

- Live infra importers: `src/cloak/train/reward.py`, `roundtrip.py`, `probes.py`,
  `src/cloak/tasks.py`, `src/cloak/lattice.py`, `scripts/train_ranker.py` (via roundtrip),
  `scripts/reward_gate.py`, `scripts/build_probes.py`, and most `scripts/spikes/*`.
- `INFERDPT_LLM_CACHE` string: ~28 files under `src/` + `scripts/`.
