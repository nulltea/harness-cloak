---
type: reference
status: current
created: 2026-07-09
updated: 2026-07-09
tags: [substitution, lattice, profile-matching, entity-linking, constrained-decoding, training, privacy]
companion: [docs/specs/lattice-substitutor.md, docs/specs/substitutor-profile-match-retrieve-verify.md]
---

# Substitutor profile matching — learned canonicalizer (constrained entity linking)

## Purpose

The retrieve-then-verify matcher
([substitutor-profile-match-retrieve-verify.md](substitutor-profile-match-retrieve-verify.md))
recovers surface variance (plural, typos, morphology, synonyms) by embedding the **bare span**.
Its structural ceiling is context: a bare-surface query cannot resolve context-dependent forms —
abbreviations (`MI` → `myocardial infarction`), ellipsis (`the procedure` → the colonoscopy named
two sentences earlier is out of scope, but `the scan` → `MRI scan` is not), and sense ambiguity
within a type.

This spec defines the trained successor: a small local seq2seq model that maps
`(span, context, runtime_type)` to a **canonical profile key**, decoding under a trie constraint
built from the loaded profile artifact so the output is guaranteed to be a real entry (or an
explicit abstain). This is entity linking against the profile vocabulary, GENRE-style
autoregressive linking with constrained decoding, adapted to the substitutor's fail-closed
contract.

Adopt only after retrieve-then-verify is deployed and its measured recall plateaus; the
graduation criterion is defined under Evaluation.

## Definitions

- **Canonical key** — the exact canonical-surface string of a profile entry under one runtime
  type; the target vocabulary of the linker.
- **Trie constraint** — a prefix trie over the tokenized canonical keys of the span's runtime
  type; during decoding only token continuations present in the trie are allowed, so every
  completed output is a valid key.
- **Abstain token** — a reserved output (`<abstain>`) meaning "no profile entry applies"; maps to
  the existing fail-closed placeholder path.
- **Proposer / certifier** — as in the retrieve-then-verify spec: the canonicalizer only
  *proposes* an entry; the NLI gate in the span's sentence plus
  `aset_count >= K_FLOORS[type]` remain the sole legality authority.
- **Teacher harvest** — the offline pattern already used for teacher lattices: a large model is
  called at artifact-build time only, its outputs cached; deployed runtime never calls it.

## Architectural invariant — unchanged from retrieve-then-verify

The canonicalizer replaces the *proposer*, not the certifier. Every canonicalizer hit is
non-deterministic: its entry's levels must pass `nli_gate` in the span's sentence, and counts
resolve through the entry's `level_counts` verbatim. Abstain and certification failure both land
on the typed placeholder. The matcher therefore cannot widen the privacy surface regardless of
model quality; a bad model only costs utility.

## Model

- **Task format** — text-to-text. Input linearization:

  ```
  type: health-condition | span: MI | context: He was admitted after an <span>MI</span> last spring.
  ```

  Output: a canonical key (`myocardial infarction`) or `<abstain>`.

- **Init checkpoint** — a small local seq2seq; candidates, decided in the training record, not
  here: `google/byt5-small` (byte-level, inherently typo-robust, no tokenizer/trie mismatch) or
  `google/flan-t5-small`/`-base` (faster, needs token-level trie). An alternative architecture —
  a canonicalization head on the existing fine-tuned detector, emitting the canonical form next
  to the span — is admissible but couples detector and matcher release cycles; default is the
  standalone seq2seq.
- The model is trained for the **task** (variant → canonical, given a vocabulary), not for a
  frozen vocabulary: the trie is rebuilt from whatever profile artifact is loaded, so new or
  user-defined runtime types get linking without retraining (zero-shot over new keys; measured,
  see Evaluation).

## Runtime algorithm

```python
# --- artifacts ---
# model: local seq2seq checkpoint (versioned; recorded in run config)
# trie[runtime_type]: prefix trie over tokenized canonical keys + "<abstain>",
#                     rebuilt whenever the profile artifact (identified by hash) changes

ABSTAIN = "<abstain>"
MIN_SEQ_LOGPROB = ...   # calibrated once on the eval set; below -> treat as abstain

def build_tries(profiles) -> dict[str, Trie]:
    tries = {}
    for runtime_type, entries in profiles.items():
        keys = [ABSTAIN] + list(entries.keys())          # canonical surfaces only;
        tries[runtime_type] = Trie(tokenize(k) for k in keys)  # aliases are training data,
    return tries                                              # not decode targets

def canonicalize(span_text, runtime_type, context) -> MatchResult | None:
    levels = lookup_levels(span_text, runtime_type)      # exact fast path, unchanged
    if levels:
        return MatchResult(levels=levels, kind="exact", deterministic=True)

    trie = tries.get(runtime_type)
    if trie is None or model is None or not context:
        return None                                      # degrade to exact-only / fall through

    x = linearize(runtime_type, span_text, mark_span(context, span_text))
    # constrained beam search: at each step, allowed tokens = trie continuations of the
    # current hypothesis prefix; every finished beam is a valid key or <abstain>
    beams = model.generate(x, num_beams=4,
                           prefix_allowed_tokens_fn=lambda _, prefix: trie.next_tokens(prefix))
    best = beams[0]
    if best.text == ABSTAIN or best.seq_logprob < MIN_SEQ_LOGPROB:
        return None                                      # fail closed -> placeholder path

    entry = profiles[runtime_type][best.text]
    approved = nli_gate(span_text, context, entry["levels"])   # certifier, mandatory
    if not approved:
        # optional: try the next distinct-key beam once, then give up
        return None
    return MatchResult(levels=approved, kind="linked", deterministic=False,
                       entry=best.text, seq_logprob=best.seq_logprob)
```

Integration is identical to retrieve-then-verify: `lattice_for()` calls the matcher where it now
calls `lookup_levels`; on `None` the existing fallback chain and placeholder terminal apply. The
two matchers are alternatives behind the same call site — deployment selects one (plus exact fast
path), never both stacked, so method comparisons stay clean.

## Training data

Two offline sources, both cached under the existing teacher-harvest pattern (build-time only,
never a deployed call):

1. **Teacher-labeled runtime misses.** Collect `(surface, runtime_type, context)` triples that
   missed the exact path — from fine-mode detector output over training corpora and from deployed
   `R` match-diagnostic logs. Prompt the teacher with the triple plus the candidate key slice for
   that type (top-k by embedding similarity, to keep the prompt small); the teacher returns a key
   or abstain. Cache keyed by `(type, surface, context-hash)`.
2. **Synthetic variants of profile entries.** Generate variant surfaces from canonical
   surfaces/aliases — inflection, articles, typo injection, casing, abbreviation expansion pairs
   where the alias list already carries them. Rule-generated *training data* is cheap and
   legitimate here; the runtime stays rule-free, the model absorbs the variance.

Mix and ratios are experiment parameters recorded in the training record. Abstain examples are
mandatory (surfaces of the right type with no valid entry, and surfaces of the *wrong* type), or
the model will hallucinate links; target roughly the abstain rate observed in runtime logs.

## Training record

Runs live under `research-wiki/training/` per the repo schema, on their own track:
`YYYY-MM-DD-FT-canonicalizer-v<N>-<slug>.md` — spec before the run, results after, with the
standard sections (objective, data mix + ratios, config, selection, evaluation, results,
ablations, cost, risks, artifacts). The companion research report links back to this spec.

## Evaluation

- **Matcher metrics** on the held-out `(variant, context, type) → gold key` set shared with
  retrieve-then-verify: linked-match precision, recall, abstain rate; plus a **new-vocabulary
  split** (types/keys absent from training) to measure the zero-shot claim.
- **Graduation criterion** — adopt over retrieve-then-verify only if it wins on certified-match
  recall at equal-or-better precision on the shared eval set, *and* the gain survives the
  end-to-end comparison: utility on `out_final` at matched attacker-measured privacy, identical
  settings, both matchers behind the identical certifier. No per-matcher threshold tuning to
  equalize secondary quantities.
- `MIN_SEQ_LOGPROB` is calibrated once on the eval set and then frozen for all comparisons.

## Failure-mode coverage beyond retrieve-then-verify

| Case | retrieve-then-verify | learned canonicalizer |
|---|---|---|
| Plural/articles/typos/morphology/synonyms | yes | yes |
| Context-dependent abbreviation (`MI`, `PT`) | no | yes — context is model input |
| Within-type sense ambiguity | weak | yes |
| Generic anaphora (`the scan` → `MRI scan` when stated earlier in the sentence) | no | partial — only if the referent is inside the context window |
| Cross-sentence anaphora / coreference | no | no — out of scope, detector-side problem |

## Non-goals

- No runtime remote calls; the linker is a small local checkpoint.
- No runtime normalization rules; all variance handling is learned or certified.
- No replacement of the certifier: NLI gate and anonymity floors are untouched.
- No coreference resolution.
- No claim of improved privacy — finer matching is a utility mechanism; privacy claims require the
  attacker-measured end-to-end comparison.

## Open questions

1. Byte-level (ByT5) vs subword (T5) trade-off: typo robustness vs decode speed with a token
   trie; decide in FT-canonicalizer v1 ablations.
2. Whether aliases should also be decode targets mapped to their canonical entry (larger trie,
   possibly easier decoding) vs training-data-only; v1 ablation.
3. Prompt-slice size for teacher harvest (candidate keys shown to the teacher): quality vs cost.
4. Whether the detector-head variant beats the standalone seq2seq enough to justify coupling the
   two models' release cycles.
