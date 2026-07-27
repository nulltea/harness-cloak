---
type: reference
status: stale
created: 2026-07-09
updated: 2026-07-27
tags: [extractor, reconstructor, rl-reward, zero-shot, frozen, alignment, do-no-harm, benchmark]
companion: [docs/specs/reconstructor-extractive-locate-splice.md, docs/specs/RL/surrogate-ranker-infiller.md, docs/specs/RL/roundtrip-ranker-infiller.md, docs/plans/2026-07-05-survived-recovery-extractor.md]
archive_reason: subject retired in 2026-07-27 cleanup (see docs/plans/2026-07-27-codebase-cleanup-refactor.md); reconstructor/frozen-extractor track
  preserved on branch archive/reconstructor-track
---

# Frozen zero-shot extractor — the RL reward anchor

Design-from-scratch of the extractor (`out_p` → `out_final`) under a new controlling requirement:
the extractor sits **inside the RL reward** that trains the substitutor (ranker + infiller), so it
must be a **frozen, well-performing, substitutor-independent** component built entirely from
existing pretrained models and rules — no training on anything the current substitutor produced.
Any learned extractor distilled from today's flawed substitutor would bake that substitutor's
defects into the reward the next substitutor is trained against.

## Why independence is the controlling requirement (reward-bias model)

The RL reward is `R_rt = mean fact recall on out_final`, where
`out_final = extract(Remote(doc_p), R)`. Write the extractor's effect as a per-mention recovery
probability `q(a, m)` for action `a` at mention `m`. The reward the policy sees is the oracle
reward filtered through `q`.

- If `q` were a constant, the reward is rescaled and every action ranking is preserved.
- In reality `q` varies by type, lattice depth, action mode, and *combinations within one
  document* — documents carry **mixed action sets**, and recall is aggregated per doc, so bias can
  appear at the document level even when marginal per-stratum averages look flat.

Therefore flatness of per-stratum recovery is **necessary but not sufficient**, and the acceptance
gates live **at the level the optimizer actually consumes rewards** — group-relative advantages and
group winners (ExIt pooling / RLOO-style baselines), not global rank correlation, which is
dominated by ties and irrelevant actions:

- **Oracle reward** `R_or`: fact recall computed on gold-restored `out_final` (constructive
  definition below).
- **Extractor reward** `R_ex`: fact recall on the extractor's `out_final`, same docs, same probes.
- **Action groups sampled by the training sampler.** Candidate action sets are generated through
  the actual RL environment machinery (legality masks, floor regimes, group sampling as in the
  env's rollout construction), at several policy stages as visitation proxies — uniform-random,
  behavior-clone, and a greedy/perturbed stage — and results are reported visitation-weighted per
  stage.
- **Optimizer-level gates** (acceptance): per sampled group,
  (a) **winner agreement** — `argmax R_ex == argmax R_or` rate;
  (b) **oracle regret** — `regret = R_or(winner_or) − R_or(winner_ex) ≥ 0` per group;
  bars on both the mean and the p95 over groups;
  (c) **advantage sign-agreement weighted by |oracle advantage|**, so near-ties cannot inflate
  the score.
- Kendall-τ/Spearman across action sets and per-stratum recovery flatness remain *diagnostics*
  that localize which strata break the gates.

Acceptance bars are frozen **before** the sealed evaluation (see Benchmark) — pre-registered
acceptance criteria, not post-hoc tuning. Per-env re-gates and checkpoint audits use a fixed
sample floor (≥ 30 groups per policy stage, ≥ 20 audited gold mentions per major type; recorded
with the env version). **Failure consequence**: a failed gate blocks RL consumption of that env
under that extractor version — either the extractor is revised (version bump, full re-gate) or
the env change is rolled back; there is no silent fallback. Two scope notes: the optimizer-level gate is
**evaluation-only** — no extractor threshold is ever calibrated against env-sampled groups
(calibration uses only the substitutor-independent mention benchmark), so evaluating on the env's
action space does not violate zero-coupling; and because the gate samples through the env, it is
**re-run against each new env version** (sample floor below), while the mention-level benchmark
stays fixed. Because the three policy-stage proxies cannot anticipate where ExIt/RLOO drives the
policy once it climbs the *extractor-shaped* reward, RL training additionally runs
**checkpoint-triggered oracle audits**: at each reward-climbing checkpoint, sample groups from
the current policy, judge-propose + hand-audit their gold, and re-check the optimizer gates —
evaluation-only, never a threshold recalibration.

### Oracle reward — constructive definition

Gold restoration is a deterministic rule over the audited gold alignment, no models involved:

```
oracle_restore(out_p, gold):            # gold: per R-entry, span(s) in out_p or ABSENT
  spans = all gold mention spans        # audit invariant: pairwise non-overlapping —
                                        # an overlap is an audit ERROR, fixed before freeze
  for (start, end, surface) in spans sorted by start DESC:
      out_p = out_p[:start] + surface + out_p[end:]
  return out_p                          # no fluency repair — the oracle measures
                                        # information recovery; the reader-based fact
                                        # scorer is tolerant of local grammar damage
```

Edge cases, decided here: repeated mentions each carry their own gold span and are each restored;
ABSENT entries make no edit; a mention smeared non-contiguously across a clause is labeled
`absent-noncontiguous` at audit time — excluded from the restorable denominator and reported as
its own miss class (no extractor is credited or penalized for it). **Validity audit before
freeze:** on a hand-checked sample, reader-scored fact recall on gold-restored docs must
dominate every candidate extractor's recall in every stratum (the oracle must actually be a
ceiling), and gold spans must satisfy the non-overlap and `doc-grounded` invariants. The
no-fluency-repair assumption is itself audited: where the oracle underperforms *because of
splice grammar* (reader drops a fact whose content is present), the gold span/abstention is
manually repaired before freeze — the oracle may not carry a grammar handicap into the gates.

The extractor must also behave sanely on garbage fills (`"an information"`,
`"between 4.25 and 17 $illion"`): RL exploration produces garbage-adjacent actions even after
upstream fixes, and a false restore on garbage poisons the counterfactual signal.

## Definitions

- **Extractor** — the client-side inverse map `(doc_p, R, out_p) → out_final`; recovers original
  surfaces at their mentions in `out_p`, or abstains. Synonym in older docs: reconstructor.
- **Frozen** — version-pinned: model ids + revisions, thresholds, prompts/hypothesis templates,
  and code hash recorded as `extractor_version`; any change re-gates RL runs that consumed it
  (same discipline as the `RT_MODEL` reward pin in `cloak.train.roundtrip`).
- **Mention** — the (possibly reworded) occurrence in `out_p` of one `R` entry's fill content.
- **Alignment prior** — a positional mapping `doc_p` region → `out_p` region derived by comparing
  the two documents; a *soft score bonus* on candidates, never a gate (see failure modes).
- **Assignment** — the global one-to-one matching between residue `R` entries and candidate
  mentions (one mention serves at most one entry).
- **Do-no-harm** — only `R` originals may enter `out_final`, only at verified mentions; a false
  substitution is worse than a miss; spans already resolved deterministically are never altered.
- **Oracle reward / optimizer-fidelity gates** — defined constructively in the reward-bias model above (winner agreement, oracle regret, weighted advantage sign-agreement).

## Requirements (named, all hard unless marked)

- **zero-coupling** — no component trained, tuned, or *threshold-calibrated* on the current
  substitutor's or judge's outputs. Pretrained public weights are allowed; **artifacts and
  thresholds derived from the lattice/profile pipeline are not** (see Coupling hygiene). The
  judge may build the *benchmark* once, hand-audited — it may not be a runtime component of the
  reward path.
- **optimizer-fidelity** — winner-agreement, oracle-regret, and weighted advantage
  sign-agreement bars met on the sealed split, visitation-weighted across policy stages.
- **do-no-harm** — wrong-insert = 0 and cascade-resolved-span harm = 0 on the sealed split.
- **garbage-tolerance** — corrupted/nonsense fills yield abstain, never a false restore.
- **determinism** — same `(doc_p, R, out_p)` → byte-identical `out_final` across fresh processes:
  local models, greedy/argmax only, pinned checkpoints, fixed dtype/device policy, and a
  **margin rule**: any verification score within ε of its threshold resolves to abstain
  (fail-closed), so float jitter cannot flip a decision. Verified by a cross-process test, not
  asserted.
- **throughput** (soft, measured) — small local models batched per doc; no 35B-class model in the
  inner loop; a cache-hot microbenchmark reports wall time per rollout and VRAM before RL use.
- **migration, not drop-in** — integration requires coordinated changes (signature, `R` offsets,
  reward-cache key, companion-spec updates) listed in the Integration contract; the extractor is
  not claimed compatible until that checklist lands.

## Design space — what existing models can do which job

The task decomposes into **localize** (find each residue entry's mention or decide absent),
**verify** (the mention is that fill's restatement, not a re-derived specific), **splice**
(boundary-tight replacement + fluency). Candidates per job:

| # | Component (existing model) | Job | Cost | Verdict |
|---|---|---|---|---|
| 1 | Deterministic cascade (`_rule_prepass` today: placeholder / exact / fuzzy / semantic-window; acronym + morphology proposers are planned additions from the survived-recovery plan, not yet in code) | localize+splice | ~0 | **Keep as tier 0** — resolves the bulk; deterministic and trivially flat |
| 2 | **`doc_p`→`out_p` document alignment** (sentence-embed both, monotonic soft alignment, e.g. DTW over sentence vectors) | localize (prior) | 1 embed batch/doc | **Adopt as soft prior** — fill-agnostic *mechanism*; usefulness varies with remote behavior, so it only re-scores candidates and its independence is *measured*, not assumed (see failure modes) |
| 3 | Bi-encoder phrase retrieval (bge-small-en-v1.5 public weights): embed residue fills/surfaces + candidate phrases (noun chunks / sliding windows) from `out_p` itself — **no profile embindex, no lattice artifacts** | localize (candidates) | 1 embed batch/doc | **Adopt** — recall workhorse for rewordings below fuzzy threshold |
| 4 | Cross-encoder NLI (nli-deberta-v3-small, public weights): correspondence `fill ⊨ mention-in-context` + digit gate; type check via entailment against a pinned type-description hypothesis template instead of word lists | verify | k small forwards/entry | **Adopt** — replaces both `_corresponds` NLI and the broken `_type_sane` word lists; thresholds calibrated on the benchmark calibration split only |
| 5 | Zero-shot extractive QA (SQuAD2-style null-aware reader): "Which phrase refers to {fill}?" with built-in no-answer | localize (alt) | 1 forward/entry | **Benchmark candidate** — native abstention, but alignment-style questions are out-of-distribution for QA training; challenger, benchmark decides |
| 6 | MLM pseudo-log-likelihood delta (small roberta/deberta): PLL(sentence after splice) − PLL(before); reject splices that crater fluency | splice gate | 1–2 forwards/splice | **Adopt (cheap insurance)** — catches wrong-location splices producing gibberish |
| 7 | Small causal LM fit scoring (pythia-410m, already local) | splice gate (alt) | similar | Redundant with #6; keep only if #6 underperforms on the benchmark |
| 8 | Local instruct LLM grounded-quote judge (Qwen3.6-35B, llama-swap) | localize+verify ceiling | 1 big call/doc, `-np 1` serial | **Offline only** — builds/adjudicates the benchmark and reports the quality ceiling; excluded from the reward path (throughput + auditability) |
| 9 | Trained seq2seq editor / distilled pointer (the dead generative-editor path) | all | — | **Rejected until the substitutor is trained** — violates zero-coupling; revisit as a successor trained on the *trained* substitutor's distribution, gated against this frozen extractor |

## Proposed architecture

```
extract(doc_p, R, out_p):
  0. tier-0 deterministic cascade (today's _rule_prepass, plus the survived-recovery
     acronym/alias + morphology proposers once landed): placeholders, exact/fuzzy fill,
     exact-surface leak lock                      -> resolved + residue
  1. alignment prior: sentence-embed doc_p & out_p once; monotonic soft alignment;
     each residue fill's doc_p offset (from R, see offset schema) -> predicted out_p
     window; contributes a SCORE BONUS only
  2. candidate generation: noun-phrase/window chunks over ALL of out_p,
     scored  s = cos(fill, chunk) ⊕ cos(surface, chunk) ⊕ position-prior-bonus
  3. global assignment: one-to-one entry↔chunk matching (greedy by score margin, or
     Hungarian), so repeated fills cannot double-claim one mention
  4. verification per assigned pair (all must pass, fail -> abstain; margin rule applies):
     a. correspondence: NLI fill ⊨ mention-in-context; digit gate (_value_compatible)
     b. type: NLI mention ⊨ pinned type-description hypothesis (no word lists)
     c. no added specifics: mention introduces no digit-run/proper-noun absent from fill
  5. boundary-tight splice: replace only the aligned sub-span, word-snapped;
     article/agreement fixer
  6. fluency gate: MLM PLL delta over the spliced sentence; revert splice on crater
  7. stats: per-entry outcome {resolved_tier0, spliced, abstained(reason)} -> _finalize
```

Every stage is deterministic, local, batched per document (one embed batch, one NLI wave, one MLM
wave). Per-mention acceptance; no all-or-nothing rejection; no text generation anywhere.

### Alignment-prior failure modes (why it is a bonus, never a gate)

The *mechanism* uses only positions and public sentence embeddings — nothing from the lattice.
But whether a mention exists, where it lands, and whether sections reorder all depend on how the
remote model reacted to the chosen fills, so the prior's *usefulness* can correlate with action
choice. Mitigations, all mandatory:

- candidates are always generated globally (stage 2); the prior only adds score;
- the benchmark includes reordering-heavy (summarization), omission, hallucinated-mention, and
  repeated-generic-fill strata specifically to stress the prior;
- the **prior-influence diagnostic**, measured at reward level, not assignment level: run the
  optimizer-level gates (winner agreement, oracle regret) with the prior on vs off. The prior
  keeps its weight only if it does not degrade any gate in any stratum; assignment-flip counts
  are reported as a secondary localizer. Worst case the weight calibrates to zero (calibration
  split only) and the ladder runs without it.

### `R` offset schema (assemble-time bookkeeping)

`assemble()` (scripts/train_ranker.py) currently applies replacements right-to-left on `doc_orig`
offsets and then `_cleanup()` regex-mutates the text, so offsets must be *tracked, not assumed*:

- during the right-to-left pass, record each applied replacement's `(start, end)` in the
  intermediate string and shift previously recorded (right-of-edit) offsets by each edit's length
  delta;
- `_cleanup()` (and any future mutation) runs as tracked edits: every regex substitution shifts
  recorded offsets the same way;
- each `R` entry gains `fill_spans: [[start, end], …]` — one per applied occurrence (repeated
  surfaces and mixed-typing occurrences carry their own spans);
- **build invariant, asserted**: `doc_p[start:end] == entry.replacement` (modulo the recorded
  case adjustment) for every recorded span; a violated invariant fails the build, never ships.

The deployed `substitute()` gets the same bookkeeping when the extractor is wired for deployment;
until then the extractor treats `fill_spans` as optional and skips stage 1 without it.

## Coupling hygiene (pins and calibration discipline)

- **Pin table** (part of `extractor_version`): model ids + HF revisions for the sentence-embedding
  model, cross-encoder NLI, MLM, and optional QA reader; every threshold (`SIM`, NLI, PLL-delta,
  assignment margin, prior weight, margin-rule ε); the type-description hypothesis template per
  runtime type; the chunker configuration.
- **No inherited thresholds.** The profile-match spec's `SIM_FLOOR`/`NLI_THRESH` were calibrated
  for lattice-entry matching and have known generic-level certification failures; the extractor
  calibrates its own thresholds from scratch for localization/verification.
- **No lattice artifacts at runtime.** The extractor embeds `out_p` phrases and `R` strings only;
  it never touches `lattice_profiles.json`, the profile embindex, gazetteers, or probe pools.
- **Calibration vs sealed split.** The benchmark is split once (doc-level, seeded). All threshold
  calibration and configuration selection happen on the calibration split; the sealed split is
  evaluated once per frozen configuration and reports the headline optimizer-fidelity, do-no-harm,
  and flatness numbers. Re-touching the sealed split requires a version bump and a fresh sealed
  sample.

## The frozen benchmark (built once, substitutor-independent)

1. **Planted substitutions, external-ontology fills.** Sample surfaces per runtime type from
   external gazetteers (O*NET, Mondo, GeoNames, Wikidata). Generalization **depth is defined by
   hops up the external ontology's own hierarchy** (Mondo parents, SOC major groups, GeoNames
   admin chains) — shallow / mid / top-level — so the depth strata are independent of our lattice
   generator; a *labeled* in-distribution stratum built from the current lattice artifact may be
   added for realism but is excluded from the sealed acceptance bars. Also planted: typed
   placeholders and **deliberately corrupted fills** (garbled units, wrong-entity, nonsense) for
   the garbage-tolerance stratum, and **mixed action-set documents** (several spans,
   heterogeneous depths/modes). The optimizer-level gate's groups and counterfactual pairs are
   sampled separately, **through the training sampler** (see reward-bias model), on top of this
   document pool.
2. **Real documents, pinned roundtrip.** Plant the substitutions in real corpus docs (clinical,
   lexsum, enron/aeslc pending the short-doc stratum decision), send through the pinned remote
   (gemma E4B, temp 0, cached) to obtain natural rewordings.
3. **Gold alignment, double-audited.** The Qwen judge proposes grounded quotes; a hand audit
   corrects boundaries and false grounding (Jul-7/Jul-9 audits: judge quote boundaries need
   correction roughly a third of the time); a **second pass** (second annotator or a
   disagreement-focused re-audit) covers every `absent` gold and every boundary the first pass
   changed. Frozen as JSONL with per-mention gold spans and explicit `absent` labels.
4. **Pre-registration.** Strata and minimum cell counts are fixed before generation (target
   ≥ 50 gold mentions per (type × depth) cell and ≥ 30 per fill-quality cell; final numbers set
   and recorded at build, then frozen), along with the acceptance bars (winner agreement,
   nonnegative oracle regret mean/p95, |advantage|-weighted sign-agreement, wrong-insert = 0,
   harm = 0, garbage false-restore = 0, per-stratum recovery ≥ tier-0 baseline). Bars are
   chosen before the sealed evaluation runs.
5. **Selection happens here, zero-shot.** Configurations (embedding+cross-encoder ladder vs QA
   localizer vs both; prior weight; assignment algorithm) are compared on the calibration split;
   the judge runs once as the ceiling reference; the winner is evaluated once on the sealed split
   and frozen as `extractor_version`.

**Status of existing evidence:** the Jul-9 fresh probe (40 grounded quotes, 7 admitted, 6/7
spliced) is evidence *against the generative editor* and for the general locate-splice shape —
it is far too small to certify a reward anchor and is superseded by this benchmark for that
purpose.

## Verification gates before RL use (measured, not asserted)

- **Cross-process determinism test**: identical `(doc_p, R, out_p)` triples → byte-identical
  `out_final` across ≥ 3 fresh processes (fresh CUDA/ROCm context each), including
  borderline-score cases constructed near each threshold (the margin rule must absorb them).
- **Cache-hot microbenchmark**: wall time per rollout and peak VRAM over representative residue
  counts (p50/p95 of the benchmark), reported alongside the gemma generation + reader-scoring
  cost it adds to. Includes the concurrency reality: `roundtrip_batch` runs jobs across worker
  threads, so extractor models are process-level singletons with batched, lock-guarded calls —
  contention is part of the measured number, and the extractor shares the GPU with the served
  llama-swap models.
- **Reward-cache key**: `extractor_version` joins the reward pin (with `RT_MODEL`); any cached
  reward is invalid under a different extractor version.

## Integration contract (migration checklist, in order)

1. `assemble()` (and later `substitute()`) records `fill_spans` per the offset schema, with the
   build invariant asserted.
2. `invert()` keeps its `(out_p, R)` signature for legacy callers; the frozen extractor lands as
   `extract(doc_p, R, out_p)` beside it, and `roundtrip_batch` switches its call site
   (`src/cloak/train/roundtrip.py`) to pass the job's `doc_p`.
3. `docs/specs/RL/roundtrip-ranker-infiller.md` updates its extractor pin (currently the rule
   exact/fuzzy path and `invert(out_p, R)` pseudocode) to reference `extractor_version`.
4. Reward-cache key gains `extractor_version`; existing cached rewards remain valid only for runs
   that pin the legacy cascade.
5. Only after 1–4 and the verification gates pass does an RL run consume the frozen extractor.

## After the substitutor is trained

Only then does a *learned* extractor become admissible: train on the **trained** substitutor's
output distribution (pointer-style span prediction, per the locate-splice spec's deferred
component), evaluate on this same frozen benchmark (sealed split, fresh sample if the old one was
consumed) and gate against the frozen extractor as the regression baseline. The frozen extractor
never disappears: it remains the reward anchor for subsequent substitutor iterations, or is
replaced only by a successor that dominates it at equal optimizer fidelity.

## Tradeoffs

- **Ceiling below a trained model's.** A zero-shot ladder will miss some heavy paraphrases a
  trained localizer could catch. Accepted: the benchmark's judge-ceiling row quantifies what is
  left on the table, and optimizer fidelity outranks peak recall for reward use.
- **Alignment prior degrades on reordering tasks** — handled by bonus-only design + the
  prior-influence diagnostic; worst case its weight calibrates to zero and the ladder still works.
- **Type-description NLI is weaker than curated word lists on their home turf** but has no
  zero-recall failure mode on fine types and needs no per-type maintenance; the
  digit/proper-noun gate carries the hard safety anyway.
- **Benchmark construction cost** (judge calls + two audit passes) is paid once; it becomes the
  regression suite for every future extractor and substitutor change.

## Open questions

1. Assignment algorithm: greedy-by-margin vs Hungarian — pick on the calibration split, freeze.
2. QA localizer (#5): carry as a second opinion in the ladder, or benchmark-and-drop?
3. Whether `enron`/`aeslc` email tasks need their own benchmark stratum (short docs, heavy
   templating) — decide at pre-registration.
4. Exact optimizer-fidelity bars — proposed starting points: winner agreement ≥ 0.9,
   oracle regret ≤ 0.02 recall, weighted sign-agreement ≥ 0.9; frozen at pre-registration,
   before any sealed run.
