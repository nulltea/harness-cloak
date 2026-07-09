---
type: reference
status: current
created: 2026-07-09
updated: 2026-07-09
tags: [extractor, reconstructor, rl-reward, zero-shot, frozen, alignment, do-no-harm, benchmark]
companion: [docs/specs/reconstructor-extractive-locate-splice.md, docs/specs/RL/surrogate-ranker-infiller.md, docs/plans/2026-07-05-survived-recovery-extractor.md]
---

# Frozen zero-shot extractor — the RL reward anchor

Design-from-scratch of the extractor (`out_p` → `out_final`) under a new controlling requirement:
the extractor sits **inside the RL reward** that trains the substitutor (ranker + infiller), so it
must be a **frozen, well-performing, substitutor-independent** component built entirely from
existing pretrained models and rules — no training on anything the current substitutor produced.
Any learned extractor distilled from today's flawed substitutor would bake that substitutor's
defects into the reward the next substitutor is trained against.

## Why independence is the controlling requirement (reward-bias analysis)

The RL reward is `R_rt = fact recall on out_final`, where
`out_final = extract(Remote(doc_p), R)`. The policy explores the **whole action space** — every
lattice depth, every type, placeholders, and (during exploration) fills of any quality. Two
failure shapes:

- **Uniform weakness is survivable.** If extractor recall is roughly constant across actions, the
  reward is scaled but the *ranking* of actions is preserved; the policy still learns the right
  ordering.
- **Non-uniform weakness is fatal.** If the extractor recovers shallow/specific fills better than
  deep/generic ones (or placeholders better than text levels), the policy is rewarded for
  *extractor-friendliness*, not utility — it will converge to whatever the extractor happens to
  invert well. A model distilled from the current pipeline is the extreme case: its competence is
  literally the current substitutor's output distribution.

Therefore the acceptance bar below includes **flatness across the action space**, not just average
recovery — and the extractor must behave sanely on garbage fills (`"an information"`,
`"between 4.25 and 17 $illion"`), because exploration will produce garbage-adjacent actions even
after upstream fixes.

## Definitions

- **Extractor** — the client-side inverse map `(doc_p, R, out_p) → out_final`; recovers original
  surfaces at their mentions in `out_p`, or abstains. Synonym in older docs: reconstructor.
- **Frozen** — version-pinned: models, thresholds, and code hash recorded; any change re-gates RL
  runs that consumed it (same discipline as the `RT_MODEL` reward pin in `cloak.train.roundtrip`).
- **Mention** — the (possibly reworded) occurrence in `out_p` of one `R` entry's fill content.
- **Alignment prior** — a positional mapping `doc_p` region → `out_p` region derived by comparing
  the two documents; predicts *where* each fill's mention should be before any lexical search.
- **Assignment** — the global one-to-one matching between residue `R` entries and candidate
  mentions (one mention serves at most one entry).
- **Do-no-harm** — only `R` originals may enter `out_final`, only at verified mentions; a false
  substitution is worse than a miss; spans already resolved deterministically are never altered.
- **Flatness** — max spread of per-stratum recovery across (type × lattice-depth × fill-quality)
  benchmark strata; the anti-reward-bias metric.

## Requirements (named, all hard unless marked)

- **zero-coupling** — no component trained, tuned, or distilled on the current substitutor's or
  judge's outputs; pretrained models + deterministic rules only. (The judge may build the
  *benchmark* once, hand-audited — it may not be a runtime component of the reward path.)
- **action-space flatness** — recovery may not systematically degrade with lattice depth or
  action mode; measured on the frozen benchmark, spread bounded (bar set at benchmark build).
- **do-no-harm** — wrong-insert = 0 and cascade-resolved-span harm = 0 on the benchmark.
- **garbage-tolerance** — corrupted/nonsense fills must yield abstain, never a false restore.
- **determinism** — same `(doc_p, R, out_p)` → same `out_final`, across processes: local models,
  greedy/argmax decoding only, pinned checkpoints, no sampling. (Reward memoization and ExIt-pool
  reuse depend on this.)
- **throughput** (soft) — the reward loop calls the extractor once per rollout; the budget is
  "small local models, batched" — embedding + cross-encoder forwards per doc, no 35B-class model
  in the inner loop.
- **drop-in** — keep `invert(out_p, R)`-compatible signature with `doc_p` as a new optional
  argument; existing callers keep working (without `doc_p` the alignment prior is skipped).

## Design space — what existing models can do which job

The task decomposes into **localize** (find each residue entry's mention or decide absent),
**verify** (the mention is that fill's restatement, not a re-derived specific), **splice**
(boundary-tight replacement + fluency). Candidates per job:

| # | Component (existing model) | Job | Cost | Verdict |
|---|---|---|---|---|
| 1 | Deterministic cascade: placeholder / exact / fuzzy / acronym / morphology (`_rule_prepass` + survived-recovery proposers) | localize+splice | ~0 | **Keep as tier 0** — already resolves the bulk; deterministic and trivially flat |
| 2 | **`doc_p`→`out_p` document alignment** (sentence-embed both, monotonic soft alignment, e.g. DTW over bge-small sentence vectors) | localize (prior) | 1 embed batch/doc | **Adopt** — the strongest untapped signal; substitutor-independent by construction (uses only positions, not fill quality); disambiguates repeated generic fills ("the organization" ×3) that defeat per-fill search |
| 3 | Bi-encoder phrase retrieval (bge-small-en-v1.5, same model/infra as the profile-match embindex): embed residue fills + candidate phrases (noun chunks / sliding windows) in the prior's region | localize (candidates) | 1 embed batch/doc | **Adopt** — recall workhorse for rewordings below fuzzy threshold |
| 4 | Cross-encoder NLI (nli-deberta-v3-small, already in repo): correspondence `fill ⊨ mention` + digit gate; type check via entailment against a type-description hypothesis instead of word lists | verify | k small forwards/entry | **Adopt** — replaces both `_corresponds` NLI and the broken `_type_sane` word lists |
| 5 | Zero-shot extractive QA (SQuAD2-style null-aware reader, e.g. deberta-v3-large-squad2): "Which phrase refers to {fill}?" with built-in no-answer | localize (alt) | 1 forward/entry | **Benchmark candidate** — elegant single-model localizer with native abstention, but alignment-style questions are out-of-distribution for QA training; keep as a challenger, let the benchmark pick |
| 6 | MLM pseudo-log-likelihood delta (small roberta/deberta): PLL(sentence after splice) − PLL(before); reject splices that crater fluency | splice gate | 1–2 forwards/splice | **Adopt (cheap insurance)** — catches wrong-location splices that produce gibberish; deterministic |
| 7 | Small causal LM fit scoring (pythia-410m, already loaded for `walk_risk`) | splice gate (alt) | similar | Redundant with #6; keep only if #6 underperforms on the benchmark |
| 8 | Local instruct LLM grounded-quote judge (Qwen3.6-35B, llama-swap) | localize+verify ceiling | 1 big call/doc, `-np 1` serial | **Offline only** — builds/adjudicates the benchmark and reports the quality ceiling; excluded from the RL inner loop (throughput; and keeping the reward path free of any LLM judgment keeps it auditable) |
| 9 | Trained seq2seq editor / distilled pointer (the dead design-3 path) | all | — | **Rejected until the substitutor is trained** — violates zero-coupling; revisit as a *successor* trained on the trained substitutor's distribution, gated against this frozen extractor as the regression baseline |

## Proposed architecture

```
extract(doc_p, R, out_p):
  0. tier-0 deterministic cascade (unchanged): placeholders, exact/fuzzy fill,
     exact-surface leak lock, acronym/alias, morphology  -> resolved + residue
  1. alignment prior: sentence-embed doc_p & out_p once; monotonic soft alignment;
     each residue fill's doc_p offset (recorded by assemble()) -> predicted out_p window
  2. candidate generation: noun-phrase/window chunks inside (prior window ∪ global),
     scored  s = cos(fill, chunk) ⊕ cos(surface, chunk) ⊕ position-prior
  3. global assignment: one-to-one entry↔chunk matching (greedy by score margin, or
     Hungarian), so repeated fills cannot double-claim one mention
  4. verification per assigned pair (all must pass, fail -> abstain):
     a. correspondence: NLI fill ⊨ mention-in-context; digit gate (_value_compatible)
     b. type: NLI mention ⊨ "<type description>" (no word lists)
     c. no added specifics: mention introduces no digit-run/proper-noun absent from fill
  5. boundary-tight splice: replace only the aligned sub-span, word-snapped;
     article/agreement fixer
  6. fluency gate: MLM PLL delta over the spliced sentence; revert splice on crater
  7. stats: per-entry outcome {resolved_tier0, spliced, abstained(reason)} -> _finalize
```

Every stage is deterministic, local, batched per document (one embed batch, one NLI wave, one MLM
wave — same batching pattern as the profile-match pre-pass). Per-mention acceptance; no
all-or-nothing rejection; no text generation anywhere.

**Assemble-time bookkeeping (tiny upstream change, substitutor-independent):** `assemble()`
records each fill's character offset in `doc_p` on its `R` entry. Mechanical position tracking —
no dependence on fill quality — and it is what makes stage 1 cheap and exact.

## The frozen benchmark (built once, substitutor-independent)

The benchmark is how candidates are compared, how the ladder's thresholds are set **once**, and
how the freeze is certified. It must span the *action space*, not the current substitutor's
output distribution:

1. **Planted substitutions.** Sample surfaces per runtime type from external gazetteers (O*NET,
   Mondo, GeoNames, Wikidata — ground truth independent of our lattice artifacts). For each
   surface, construct fills at controlled strata: lattice depth 1 / mid / coarsest, typed
   placeholder, **deliberately corrupted fills** (garbled units, wrong-entity, nonsense phrases)
   for the garbage-tolerance stratum.
2. **Real documents, pinned roundtrip.** Plant the substitutions in real corpus docs (clinical,
   lexsum, enron), send through the pinned remote (gemma E4B, temp 0, cached) to obtain natural
   rewordings — the same distribution shift the deployed extractor faces.
3. **Gold alignment.** The Qwen judge proposes grounded quotes; a one-time hand audit corrects
   boundaries and false grounding (the Jul-7/Jul-9 audits showed judge quotes need boundary
   correction ~1/3 of the time). Frozen as JSONL with per-mention gold spans, including explicit
   `absent` gold for fills the remote dropped.
4. **Metrics & acceptance.** Per (type × depth × fill-quality) stratum: recovery, wrong-insert,
   deletion, abstain. Acceptance: wrong-insert = 0, harm = 0, garbage-stratum false-restore = 0,
   recovery flatness across depth strata within a bound set at build time, and per-stratum
   recovery ≥ the deterministic tier-0 baseline (the ladder must never lose to its own tier 0).
5. **Selection happens here, zero-shot.** Configurations compared on the benchmark: ladder with
   embedding+cross-encoder localizer (#2–#4) vs zero-shot QA localizer (#5) vs both; judge (#8)
   run once as the ceiling reference. The winner is frozen: model ids, thresholds, code hash →
   `extractor_version`, recorded in every RL env artifact and result that consumes it.

## RL integration contract

- `roundtrip_batch` calls the frozen extractor exactly where `invert()` runs today; signature
  gains optional `doc_p` (jobs already carry it).
- The extractor version is part of the reward pin, alongside `RT_MODEL`: changing extractor
  version re-gates every RL comparison, same as changing the remote model.
- Determinism + the content-addressed LLM cache keep reward memoization valid; all extractor
  models are local and greedy, so no extractor-side cache is needed for determinism (an
  embedding/NLI memo is a pure speed optimization).
- Cost per rollout: one bge embed batch + one NLI wave + one MLM wave on the residue only —
  small against the gemma generation + reader-scoring calls already in the loop.

## After the substitutor is trained

Only then does a *learned* extractor become admissible: train on the **trained** substitutor's
output distribution (pointer-style span prediction, per the locate-splice spec's deferred
component), evaluate on this same frozen benchmark plus a fresh judge-audited set, and gate
against the frozen extractor as the regression baseline. The frozen extractor never disappears:
it remains the reward anchor for subsequent substitutor iterations, or is replaced only by a
successor that dominates it on the benchmark at equal flatness.

## Tradeoffs

- **Ceiling below a trained model's.** A zero-shot ladder will miss some heavy paraphrases a
  trained localizer could catch. Accepted: the benchmark's judge-ceiling row quantifies exactly
  how much is left on the table, and the flatness requirement outranks peak recall for reward
  use.
- **Alignment prior degrades on reordering tasks.** Summaries reorder content; the prior is a
  soft boost, not a hard window — stage 2 always includes global candidates. Measured on the
  benchmark per corpus.
- **Type-description NLI is weaker than curated word lists on their home turf** but has no
  zero-recall failure mode on fine types and needs no per-type maintenance; the digit/proper-noun
  gate carries the hard safety anyway.
- **Benchmark construction cost** (~judge calls + one hand-audit session) is paid once; it also
  becomes the regression suite for every future extractor and substitutor change.

## Open questions

1. Assignment algorithm: greedy-by-margin is simpler and deterministic; Hungarian is optimal but
   O(n³) on rare dense docs — pick on the benchmark, freeze the choice.
2. QA localizer (#5): worth carrying as a second opinion in the ladder, or benchmark-and-drop?
3. Flatness bound: set after the first benchmark run (empirical-honesty: it is an acceptance
   bar chosen once, not a per-run tuning knob).
4. Whether `enron`/`aeslc` email tasks need their own benchmark stratum (short docs, heavy
   templating) — decide when building the planted set.
