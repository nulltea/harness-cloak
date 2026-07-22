---
type: reference
status: stale
created: 2026-07-08
updated: 2026-07-22
tags: [rl, reward-design, training-task, qa-construction, task-necessity, placeholder-gaming,
       granularity-ladder, schema-task, decision-probes, spec]
companion: [docs/specs/RL/roundtrip-ranker-infiller.md,
            docs/issues/2026-07-06-placeholder-gaming-reward-qa-necessity.md,
            docs/issues/2026-07-08-rl-env-and-lattice-count-issue-register.md]
superseded_by: docs/specs/RL/interactive-ranker-v2.md
---

# Training-task environment for round-trip RL — design alternatives

**Status: stale predecessor.** The ladder, decision, schema, and carrier components below remain
historical design inputs, but they are not the live ranker-v2 reward contract. The normative
contract is [interactive ranker v2](interactive-ranker-v2.md).

What task should the remote model execute during RL training, and how should the reward read
its output, so that **truthful generalization measurably beats placeholder exactly where it
should** — where the task needs the fact's semantics — and placeholder wins where copying
suffices? This report specifies the alternatives with prompt examples, pseudocode, trade-offs,
and a recommendation. It is the design answer to the
[placeholder-gaming issue](../../issues/2026-07-06-placeholder-gaming-reward-qa-necessity.md).

## Definitions

- **Copy-fact** — a fact the task output merely restates (HPI-style). Placeholder-echo +
  `invert()` recovers it at zero remote leakage; placeholder is *legitimately optimal* here.
- **Reasoning-fact** — a fact whose *semantic content* the remote model must use to produce a
  correct output (referral choice, med-class continuation, category-level statement). A
  placeholder destroys it; a sufficiently fine generalization preserves it.
- **Echo channel** — remote model copies the replacement token/phrase verbatim into `out_p`;
  `invert()` restores the original. Deployed utility for copy-facts (decision 2026-07-08:
  credited in the reward, not treated as gaming).
- **Semantic channel** — reward earned from `out_p`'s *content* before inversion; placeholders
  cannot earn on this channel.
- **Granularity ladder** — per-span probe set with one question per lattice level; the question
  at level ℓ is answerable iff the chosen action is at least as specific as ℓ.
- **Ceiling agreement** — scoring a probe against the answer the reader extracts from
  `out_hi = Remote(task(doc_orig))` instead of a teacher-written gold.
- **Collapse cliff** — gemma's degeneration to bracketed templates on heavily anonymized
  `doc_p` (measured, `research-wiki/experiments/context-injection-surface-ablation.md`
  follow-up); the floor-walk BC init sits on it.
- Everything else (R_rt, probes, anchors, floor-walk, ExIt/RLOO) as in the
  [round-trip RL spec](roundtrip-ranker-infiller.md).

## The design constraints (from measurements, 2026-07-06 → 07-08)

1. **Echo stance (decided):** echo is legitimate utility for copy-facts. The reward must credit
   it there and must NOT let it substitute for reasoning-facts. Floor-rejection's deletion of
   64% of candidate probes gets replaced by channel-separated scoring, not kept.
2. **Question shape is currently inverted.** Existing probes put the category in the question
   and demand the surface as the answer ("Which *endocrine disorder* is included in the
   history?" → "hypothyroidism", `data/probes_validated.json` mts/0). That is the shape
   echo+invert games. Task-necessity probes must flip it: surface-neutral question,
   category-level gold ("What class of condition is being managed?" → "an endocrine
   condition") — and the lattice already enumerates the valid category answers per level.
3. **Extractor asymmetry (plausible, unmeasured):** `invert()` restores placeholders via exact
   token echo but generalizations only via fuzzy-90/semantic-window over paraphrases — the
   current reward likely *penalizes* generalization relative to placeholder through inversion
   brittleness alone. Any design keeping surface probes must measure this gap
   (exact-vs-fuzzy inversion recall per action mode).
4. **Corpus reality (reviewed 2026-07-08):** ACI (67 docs, ~6.2k-char dialogues → sectioned
   notes with ASSESSMENT/PLAN) has real inference structure. MTS (200 docs, median 349-char
   snippet → 77-char gold) is nearly pure restatement — echo-dominant, thin task-necessity.
   lexsum/wikibio sit between. Reasoning-fact designs must not pretend MTS supports them.
5. **Collapse cliff:** the reward must have support at and around the BC init. Designs that
   only add reasoning-facts scored at heavy anonymization inherit zero-gradient regions.
6. **Credit:** per-span counterfactuals are the credit mechanism (deterministic cached reward);
   probes that target one span keep credit exact; multi-span decision probes smear it.
7. **Cost:** the serial QA reader is the measured wall (~1 rt/s at ~13 probes/doc); remote
   round trips are cached. Probe-count multipliers hit the reader, episode multipliers hit the
   remote.

## Approaches

Names are the identifiers (repo naming rule); no approach numbering is used outside this doc.

---

### A. Free-form generation + surface probes (status quo — reference point)

The current environment: corpus-native generation prompt (`src/cloak/tasks.py:23-40`), surface
probes validated by ceiling/floor anchors, reward = fact recall on `out_final`.

Prompt (clinical, unchanged):

```text
Write a clinical visit note for the following doctor-patient dialogue, using standard
note sections. Base it only on the dialogue.

Dialogue:
{doc_p}
```

**Why it fails the objective:** recall-only + surface-answers means every probe is a copy-fact
by construction; the only anti-placeholder force is floor-rejection (subtractive, deletes 64%);
generalization earns nothing placeholder doesn't, and inversion brittleness may make it earn
*less*.

**Verdict:** keep only as the measured baseline arm — every new design must beat it at matched
floors, per the empirical-honesty rule.

---

### B. Generation task + granularity-ladder probes (two-channel scoring)

Keep the deployed generation task as the reward carrier. Replace the per-span probe set with a
**ladder aligned to the span's lattice**: one question per level, gold answer at that level's
granularity. Score the exact tier on `out_final` (echo channel — copying is legitimate) and the
coarser tiers on `out_p` **before inversion** (semantic channel — placeholders cannot earn).

Ladder for span "hypothyroidism" (health-condition; lattice: hypothyroidism → endocrine
condition → chronic condition → `<HEALTH_CONDITION_n>`):

| tier | question | gold | scored on |
|------|----------|------|-----------|
| exact | "Which specific thyroid condition is the patient treated for?" | hypothyroidism | `out_final` |
| family | "What body-system category of condition is being managed with medication?" | an endocrine condition | `out_p` |
| coarse | "Does the patient have an ongoing chronic condition?" | yes — a chronic condition | `out_p` |

Reward (pseudocode):

```python
def R_ladder(doc, actions):
    doc_p, R = assemble(doc, actions)
    out_p  = Remote(task_prompt[corpus](doc_p))        # ONE round trip, cached — as today
    out_f  = invert(out_p, R)
    per_span = []
    for s in probed_spans(doc):
        exact  = mean(fact_score(reader(q, out_f), a) for q, a in ladder[s].exact_tier)
        coarse = mean(entail_score(reader(q, out_p), a) for q, a in ladder[s].coarse_tiers)
        per_span.append(w_exact * exact + w_coarse * coarse)     # w's fixed, pre-registered
    return mean(per_span)
```

`entail_score` accepts any answer the gold level entails (keep or finer fill answers coarser
tiers too — monotone by entailment): NLI or the existing `fact_score` canon over the lattice
level's alias set.

**What the policy's optimum becomes:** the deepest floor-legal level that keeps the needed
tiers answerable — placeholder where only the exact tier ever mattered (echo covers it), a
mid-lattice level where family/coarse tiers carry reward. This is exactly the stated training
objective, and the reward is **smooth and monotone in generalization depth by construction**
(each deeper level trades specific-tier loss against nothing — until the semantic tiers start
failing), which directly attacks reward quantization and the collapse cliff's zero-gradient
plateau.

**Probe construction** (teacher, one-time, cached): for each detected span with a lattice, the
teacher writes one question per level whose gold is that level's phrase; validation extends the
existing anchor check per tier — the tier is kept iff the reader answers it from `out_hi`
(ceiling) and NOT from `out_lo` (floor). Ladder tiers multiply probes/span by ~2–3×, which also
fixes the probe-density problem (1.4–2.7 kept facts/doc today) without new corpora.

- **Credit:** exact — each ladder targets one span; counterfactual span advantage unchanged.
- **Cost:** one remote round trip per rollout (unchanged); reader calls ×2–3 (the reader wall —
  the across-rollout reader parallelism fix becomes mandatory, not optional).
- **Risks:** (i) small-reader tier ability — Qwen3.5-0.8B must map "hypothyroidism" in `out_p`
  to "an endocrine condition"; tier validation at the ceiling catches most failures but a
  reader-knowledge ceiling becomes part of the environment pin; (ii) teacher tier-question
  leakage (question text must not contain the gold phrase — lint it); (iii) MTS contributes
  mostly exact tiers (fine — the design degrades gracefully to the copy channel there).

---

### C. Schema-constrained clinical task (structured note)

Change the *task prompt* so the output is a fixed schema whose fields force category-level
statements — the note format itself demands the semantics that make generalization pay.

Prompt (clinical):

```text
Write a clinical visit note for the following dialogue with EXACTLY these sections:

CHIEF COMPLAINT: one line.
HISTORY OF PRESENT ILLNESS: short paragraph.
ASSESSMENT: one line per active problem, formatted "problem — category — status".
PLAN: one line per active problem, formatted "problem — action — follow-up".

Base it only on the dialogue. Do not invent content for missing sections; write "none".

Dialogue:
{doc_p}
```

On doc_orig gemma writes `congestive heart failure — cardiovascular — stable — continue
lisinopril, start Lasix`. On a doc_p with `<HEALTH_CONDITION_1>` the category and action
fields have nothing to draw on; with "a cardiovascular condition" they survive. Scoring parses
the schema and grades fields:

```python
def R_schema(doc, actions):
    doc_p, R = assemble(doc, actions)
    out_p = Remote(schema_prompt(doc_p)); out_f = invert(out_p, R)
    rows  = parse_sections(out_f)                     # deterministic parser
    gold  = parse_sections(out_hi_cached[doc])        # ceiling agreement — no teacher gold
    return mean(field_match(rows[p][f], gold[p][f])   # canon + fact_score per field
                for p in gold for f in ("problem", "category", "action"))
```

- **Pros:** deployment-plausible (clinical notes ARE schema'd); deterministic parsing beats
  free-text reading (reader wall shrinks — most fields score without the QA reader); the
  provided template *absorbs the collapse cliff* (gemma's failure mode was inventing a
  template — here the template is given, so degradation shows up as empty/generic fields,
  which score 0 in a graded, non-cliff way); ceiling agreement removes teacher-gold
  circularity.
- **Cons:** per-*field* credit, not per-span — a span feeding several fields smears credit
  (counterfactuals still work, just coarser); schema design is per-corpus work (what is the
  schema for enron replies? aeslc subjects have none); grading "action" fields drifts toward
  whole-output similarity, which the spec deliberately keeps eval-only — the field grammar
  must stay tight to avoid smuggling ROUGE into training.
- **Corpus fit:** strong for ACI/clinical and lexsum (parties—claims—court—outcome is already
  a schema); weak for MTS (two fields), inapplicable to aeslc.

---

### D. Downstream-decision probes (task-necessity by construction)

Add probes that ask for a **decision** the note's consumer would make, where the correct
decision depends on a span's semantics at some granularity. The remote model still runs the
generation task; the reader answers decision questions from `out_final`; gold comes from
ceiling agreement.

Examples (ACI D2N001, Martha: CHF + hypertension + depression):

```text
Q: Which specialty should follow up on the reduced ejection fraction finding?
   gold(ceiling): cardiology                    # needs CHF at family granularity
Q: Should the blood-pressure medication be continued, adjusted, or stopped?
   gold(ceiling): adjusted (increase lisinopril) # needs drug-class + condition linkage
Q: Is weekly therapy an appropriate plan component for this patient?
   gold(ceiling): yes                            # needs depression at family granularity
```

A placeholder `<HEALTH_CONDITION_1>` supports none of these; "a cardiovascular condition"
supports the first. This is the purest operationalization of "the task must require the PII" —
the probe *is* a downstream inference.

- **Pros:** unimpeachable task-necessity semantics; ceiling agreement avoids teacher gold;
  decision answers are short and canon-scorable.
- **Cons:** **multi-span credit smearing** — a referral decision may draw on 2–3 spans, so the
  per-span counterfactual becomes the only credit mechanism (more counterfactual round trips);
  decision density is corpus-limited (ACI yes; MTS mostly no); risk that gemma-at-ceiling
  makes unstable decisions (gold noise) — mitigated by keeping only decisions stable across
  the two cached anchors.
- Best used **as the top tier of the ladder in B**, restricted to docs where the teacher finds
  a stable decision — not as the sole reward.

---

### E. QA-as-remote-task (replace generation)

The remote model's task becomes answering questions about `doc_p` directly (the user's
"replace it by QA pairs" variant):

```text
You are assisting with a patient case. Based only on the case description below, answer
the question in one short sentence.

Case:
{doc_p}

Question: {q}
```

One episode = one (doc_p, question) pair or a batched question set; reward = graded answer
score against ladder-tier gold.

- **Pros:** densest, cleanest per-span credit (a one-span question is a one-span episode — the
  credit problem dissolves without counterfactual machinery); short outputs → cheap round
  trips; directly matches "answer questions about a patient".
- **Cons:** **train/deploy skew** — the deployed layer anonymizes documents for arbitrary
  generation tasks, and a ranker tuned on QA episodes optimizes "keep facts answerable", not
  "keep the note writable" (unprobed coherence/format quality exits the reward entirely);
  it also dissolves the project's stated novelty (round-trip *task-execution* reward,
  strategy plan §1) and weakens what Phase-5 can claim about real tasks.
- **Honest role:** a *diagnostic environment* — because its reward is dense and per-span, it
  isolates whether the ranker can learn selection *at all* when credit is clean. If RL fails
  even here, the problem is not the task design. Run it as a cheap sanity arm, never the
  headline.

---

### F. Two-role round trip (note → downstream consumer)

Extend the round trip one hop: the remote model writes the note from `doc_p` (unchanged); a
**second, local** model consumes `out_final` and performs the end-user task (answer the
patient's follow-up question, extract the problem list for a registry, route the referral).
Reward = the consumer's task success.

```python
out_p  = Remote(task_prompt(doc_p)); out_f = invert(out_p, R)
reward = mean(score(local_consumer(task_q, out_f), gold_ceiling[task_q]) for task_q in suite)
```

This prices exactly what the pipeline promises — `out_final` is *useful to its consumer* — and
the consumer being local means no new privacy exposure and no new remote cost.

- **Pros:** the most deployment-faithful utility definition; unprobed-quality Goodhart shrinks
  (a template-collapsed note fails its consumer); composes with any probe design (the consumer
  tasks can BE the ladder/decision questions).
- **Cons:** it is mechanically the current design with the reader renamed "consumer" *unless*
  the consumer tasks are richer than span QA — the value is real only combined with
  ladder/decision suites; a second pinned model enters the re-gate surface.
- **Verdict:** not a separate reward — a framing that says the reader should be upgraded into
  a consumer running task-shaped suites (which B and D already provide).

---

### G. Mixture curriculum (generation-carrier + diagnostic arms)

Not a new reward — the run structure that combines the above honestly:

1. **Carrier (headline):** generation task + ladder probes (B), schema prompt where the corpus
   supports it (C for ACI/lexsum), decision tiers where stable (D). One reward, pre-registered
   weights, one Pareto claim.
2. **Diagnostic arm:** QA-as-task (E) at small scale to verify selection-learning under clean
   credit before burning compute on the carrier.
3. **Baseline arm:** status quo (A) at matched floors — the comparison the claim needs.

Gate order: ladder-probe health report (tiers kept per corpus, reader-tier error rate on
`out_hi`) → support scan on the ladder reward → E-arm sanity → carrier pilot.

## Comparison

| approach | placeholder beaten where it should be? | per-span credit | collapse-cliff exposure | reader/remote cost | train/deploy skew | corpus fit |
|---|---|---|---|---|---|---|
| A status quo | no (echo ties or wins) | exact | full | baseline | none | all |
| B ladder | yes — semantic tiers | exact | reduced (smooth reward) | reader ×2–3 | none | all (degrades to copy-tier on MTS) |
| C schema | yes — category/action fields | field-level | absorbed by template | reader ↓ (parser) | low | ACI, lexsum |
| D decision | yes — strongest semantics | smeared (2–3 spans) | moderate | + counterfactual RTs | none | ACI mainly |
| E QA-as-task | yes | cleanest | avoided | remote ×n_q (cheap each) | **high** | all |
| F two-role | yes (via consumer) | inherits B/D | reduced | + local consumer | none | all |

## Recommendation

**Primary: B (granularity-ladder probes, two-channel scoring) on the unchanged generation
task, with C's schema prompt adopted for clinical/lexsum and D's decision questions as the
ladder's top tier where the teacher finds ceiling-stable decisions.** Rationale: it is the
smallest change that makes generalization *pay on the merits* (semantic channel), keeps echo
credit where echo is honest (exact tier on `out_final`), keeps per-span credit exact, multiplies
probe density ~2–3× for free, smooths the quantized reward, and preserves the task-execution
novelty claim. E runs once, small, as the clean-credit sanity arm; A stays as the baseline arm.

Prerequisites before any run (in order):
1. Measure the extractor asymmetry (exact-vs-fuzzy inversion recall, placeholder vs level
   fills) — it contaminates the exact tier of every design.
2. Ladder-probe builder + per-tier anchor validation + probe-health report (new gate numbers:
   tiers/span kept per corpus, reader tier-mapping error on `out_hi`).
3. Reader parallelism across rollouts (the ×2–3 reader load makes the measured wall binding).
4. Support scan re-run on the ladder reward (the old PASS does not transfer — reward changed).

**Rejected as headline:** E (skew + novelty loss), pure D (credit smearing, ACI-only), pure C
(per-corpus schema authoring never covers enron/aeslc).

## Probe generation (approved design, 2026-07-08)

Carrier approved: ladder probes on the generation task, schema prompt for clinical/lexsum,
decision probes as the ladder's top tier. This section pins how the questions are made.

Decided forks: multiple-choice decision probes (deterministic scoring); existing validated
probes reused as rung 0 (already anchor-validated); per-span teacher calls with the doc as a
shared prefix (llama-swap prompt-caches it); template questions for rule types
(DATETIME/QUANTITY/age) with teacher fallback.

### The acceptance set (scoring rule shared by everything below)

The lattice supplies gold *and* grading. For span `s` with rungs
`[surface, level_1, …, level_k]`, the acceptance set of rung ℓ is

```python
accept(s, l) = {canon(s.surface)} | {canon(x) for x in s.levels[:l]} | aliases(s, l)
entail_score(answer, s, l) = max(fact_score(answer, a) for a in accept(s, l))
```

— an answer finer than the rung entails the rung and must count (the reader answering
"hypothyroidism" satisfies the "endocrine condition" rung). Deterministic; no NLI model on
the reward path.

### Ladder probes — teacher prompt (`LADDER_PROMPT`, pv 1)

Teacher Qwen3.6-35B-A3B, non-thinking, temp 0, JSON out; one call per span, doc as shared
prefix; cache `data/ladder_probes.json` tagged `{teacher, pv}` (other-teacher/pv entries
auto-retired, as `cloak/train/probes.py` does today).

```text
You write probe questions used to grade how well a {output_kind} preserves facts at
different levels of detail.

A {output_kind} will be written from the document below. Some facts may appear in it only
in a generalized form. For the target fact you are given its generalization ladder: the
exact value first, then successively broader truthful descriptions.

For EACH rung, write ONE question that:
1. has exactly that rung's phrase as its best answer when the {output_kind} states the
   fact at that rung — ask about the PROPERTY the rung expresses, not the specific value;
2. does not contain the exact value, any finer rung's phrase, or close synonyms of them;
3. identifies which fact it asks about through surrounding circumstances (what it is
   treated with, its role in the document, who raised it) — never through the fact itself;
4. is a wh- question with a short-phrase answer; no yes/no questions;
5. is answerable from the {output_kind} alone, by a reader who never saw the document.

Document:
{doc}

Target fact: "{surface}"   (type: {type}; appears in: "{sentence}")
Ladder rungs, exact -> broad:
{rungs}

Reply ONLY with a JSON list: [{{"rung": 0, "q": "...", "a": "<that rung's phrase>"}}, ...]
```

`output_kind` comes from the corpus (clinical note / case summary / biography summary …).
Rung 0 questions are taken from the existing validated probe set where present; the teacher
still writes one (used only if the doc has no validated rung-0 probe for that fact).

**Leakage lint** (pure code, applied before any validation): drop a rung question if its
canon tokens intersect the gold phrase, the surface, or any finer rung's phrase (stopwords
excluded); drop non-`?`-terminated, yes/no-shaped (`is/are/does/did/has/can/should …`
openers), or >200-char questions.

**Anchor validation** (extends `scripts/build_probes.py`, same two cached round trips): keep
rung ℓ iff `entail_score(reader(q, out_hi), s, l) ≥ TH` AND
`entail_score(reader(q, out_lo), s, l) < TH`. The ceiling check prices the reader's
specific→category mapping ability; the floor check kills prior-guessable category questions.
Per-corpus rung-rejection rates land in the probe-health report (`reader_rung_reject_rate`).

### Decision probes — teacher prompt (`DECISION_PROMPT`, pv 1; ACI + lexsum docs only)

One call per doc; teacher sees the document AND the ceiling output `out_hi` (the decision
must be answerable from the output alone); multiple-choice; cache
`data/decision_probes.json`, same teacher/pv tagging.

```text
You design decision checks that grade whether a {output_kind} supports the decisions its
readers must make.

Below are a document and the {output_kind} written from it. Write up to {k} decision
questions that a professional reading ONLY the {output_kind} would need to answer
({decision_kinds}). For each question:
1. the correct answer must be determinable from the {output_kind} alone;
2. give 3-5 plausible answer options, exactly one correct;
3. the decision must turn on the substantive content, never on names, dates, or other
   administrative details;
4. quote the exact document phrases the decision depends on.

Document:
{doc}

{output_kind_title}:
{out_hi}

Reply ONLY with a JSON list:
[{{"q": "...", "options": ["...", "..."], "gold": "...", "depends_on": ["...", "..."]}}]
```

`decision_kinds` per corpus — clinical: "referral routing, medication continue/adjust/stop,
follow-up interval, appropriateness of a plan element"; lexsum: "likely prevailing party,
remedy type, procedural posture, which court's rules govern".

**Validation:** keep a decision probe iff `reader(q, options, out_hi)` picks `gold`
(ceiling agreement — the pinned reader must recover it from the full output) AND
`reader(q, options, out_lo)` picks differently or abstains. Options are order-shuffled at
every scoring call (seeded per call site). `depends_on` quotes are matched to detected spans
(canon substring); the probe is tagged with those span ids — the counterfactual scheduler
targets exactly those spans. A decision probe whose `depends_on` matches no detected span is
kept but tagged span-free (scores utility, drives no per-span credit).

### Schema task — no generated questions

Gold is `parse_sections(out_hi)` (ceiling agreement). Problem rows align on canon'd problem
names (both sides post-inversion); the category field scores with `entail_score` against the
problem span's acceptance set. Teacher involvement: none at runtime; offline audit of parse
failures only.

### Reward assembly (carrier)

```python
def R_carrier(doc, actions):
    doc_p, R = assemble(doc, actions)
    out_p  = Remote(task_prompt[corpus](doc_p))          # schema prompt on clinical/lexsum
    out_f  = invert(out_p, R)
    parts = []
    for s in probed_spans(doc):
        exact  = mean(fact_score(reader(q, out_f), a) for q, a in rung0[s])      # echo channel
        sem    = mean(entail_score(reader(q, out_p), s, l) for l, q in rungs[s]) # semantic channel
        parts.append(w_exact * exact + w_sem * sem)
    dec = mean(mc_score(reader(q, opts, out_f), gold) for q, opts, gold in decisions(doc))
    sch = schema_field_score(out_f, out_hi[doc]) if corpus in SCHEMA_CORPORA else None
    return combine(parts, dec, sch)     # weights fixed + pre-registered before the pilot
```

`w_exact`, `w_sem`, and the decision/schema weights are pre-registered constants, never
tuned per model or per corpus (honesty rule).

## Interactions with the open issue register

- Un-deletes the floor-rejected 64%: echo probes return as exact-tier copy-facts
  (register issue: reward cannot express the objective).
- Ladder gold answers come from lattice levels — the lattice-count fixes (per-level counts on
  the runtime path) must land first or the menus the ladder scores against are the ones the
  mask mis-prices.
- The collapse cliff argues for also revisiting the BC init operating point (init from a
  mid-depth walk rather than floor-walk) — environment-side, orthogonal to this spec, noted
  for the pilot design.
- PERSON/CODE stay outside the learned loop (no lattice); unchanged by all approaches.

## Sources

Data reviewed 2026-07-08: `corpora/clinical/aci.jsonl` (67 docs), `corpora/clinical/mts.jsonl`
(200), `data/probes_validated.json` (492 docs; meta pins gemma-4-E4B / Qwen3.6 teacher /
Qwen3.5-0.8B reader). Specs: [round-trip RL](roundtrip-ranker-infiller.md) ·
[placeholder-gaming issue](../../issues/2026-07-06-placeholder-gaming-reward-qa-necessity.md) ·
[issue register](../../issues/2026-07-08-rl-env-and-lattice-count-issue-register.md) ·
[context-injection ablation](../../../research-wiki/experiments/context-injection-surface-ablation.md)
(collapse-cliff evidence) · strategy plan
[2026-07-05-roundtrip-rl-strategy](../../plans/2026-07-05-roundtrip-rl-strategy.md).
