---
type: dev-log
status: current
created: 2026-07-06
updated: 2026-07-06
tags: [qa-build, round-trip, corpus, wikibio, qmsum, probes, domain-diversity]
companion: ../../handoffs/2026-07-05-rl-pilot-runbook.md
---

# QA-build corpus expansion: +wikibio (viable), qmsum committee (desert)

Extended the round-trip / QA reward corpus beyond clinical+legal with two new domains for
diversity and balance, then rebuilt the QA probe set at full corpus scale. Pins unchanged:
probe teacher **Qwen3.6-35B-A3B non-thinking (prompt v3)**, reward model **gemma 4 (E4B)**
non-thinking, `:8060`. The QA build (`build_probes.py`) validates each teacher-generated
question against ceiling (`doc_orig`) and floor (all-placeholder) anchors; a corpus is only
useful if the round-trip **task output carries the sensitive facts** (so the reader can
recover them from `out_final`).

## Corpora added

Both registered in `cloak.corpora.FILES` + `cloak.tasks.TASK_TEMPLATE`; builders in
`scripts/build_task_corpora.py` (`--wikibio`, `--qmsum-src`).

- **wikibio** — Wikipedia biographies (Papadopoulou et al., LREC 2022,
  [arXiv 2205.06895](https://arxiv.org/abs/2205.06895)), already vendored at
  `corpora/wikipedia_bio/{train,test}.json`. Task = bio summarization. Filtered to the
  400–4000-char band → **220 docs** (553 total; median 318 chars, so most are shorter
  single-paragraph summaries). Span-dense (~19 quasi spans/doc). `gold_ref` = first sentence
  (proxy; the QA build does not consume `gold_ref`).
- **qmsum** — QMSum committee subset (parliamentary meetings, real participant names;
  [arXiv 2004.13822](https://arxiv.org/abs/2004.13822)). Unit = one `specific_query` excerpt
  (the transcript turns in `relevant_text_span`; whole meetings are ~60k chars, far past the
  round-trip window). Task = meeting-discussion summary. 800–4000-char band → **247 docs**.

## Smoke (n=8/corpus) — the checkpoint that decided scope

| corpus | kept_facts_mean | reader ceiling-F1 | ceiling-reject | floor-reject | verdict |
|---|---|---|---|---|---|
| wikibio | 2.62 | 0.82 | 0.40 | 0.34 | **viable** (in the healthy 0.83–0.85 reader band) |
| qmsum (committee) | 0.00 | — | **0.857** | 0.14 | **desert** — dropped |

- **wikibio viable.** Modest yield (~3/8 docs clear the ≥3-train-fact bar), but fact-bearing
  and span-dense. Scaling clears the ≥30-docs/corpus gate.
- **qmsum committee = QA desert.** 86% ceiling-reject, zero kept facts: the
  "summarize the discussion" round-trip is too lossy — the short summary drops the answerable
  specifics, so even from `doc_orig` the reader recovers nothing. **Same failure class as the
  cut enron+aeslc** (task output does not carry the facts), via a different mechanism (lossy
  summary vs pleasantry replies). Dropped per user decision; kept registered as a documented
  negative (as enron/aeslc are). A fact-preserving task template ("detailed minutes: who said
  what, all figures/dates/decisions") might rescue ceiling recall — **untested**; would be
  legitimate task design, not metric-gaming (clinical→note and lexsum→case-summary were chosen
  for exactly this restatement-forcing property).

## Full build (running at time of writing)

Corpus availability: clinical **267** (aci 67 + mts 200), lexsum **161**, wikibio **220**
(in-band). Full QA build = all three = **648 docs**, `--n-docs 267` (each corpus caps at its
available count). Rebuilds arms fresh at full scale (pilot was 80/corpus); teacher probe cache
hits on the old texts, gemma anchors mostly re-run (detection is process-nondeterministic).

- Artifacts: `data/task_arms_full.json`, `data/ranker_env_full.json`,
  `data/probes_validated.json`, `results/probe_health.json`. Log `results/qa_build_full.log`.
- Gate: ≥30 docs/corpus with ≥3 train facts.
- Perf: Qwen teacher `-np 1` (serialized server-side) is the bottleneck and the saturation
  ceiling; gemma anchors at workers=6 (measured saturation point). ~3–5 h wall, exclusive
  `:8060` lock.

**probe_health (full build) — PENDING.** Fill per-corpus `kept_facts_mean` / excluded counts
and the gate pass/fail here when `results/qa_build_full.log` shows `QA BUILD DONE`.
