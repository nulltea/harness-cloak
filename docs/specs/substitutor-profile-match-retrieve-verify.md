---
type: reference
status: current
created: 2026-07-09
updated: 2026-07-09
tags: [substitution, lattice, profile-matching, embeddings, nli, privacy]
companion: [docs/specs/lattice-substitutor.md, docs/specs/substitutor-profile-match-learned-canonicalizer.md]
---

# Substitutor profile matching — retrieve-then-verify (embedding retrieval + NLI certification)

## Purpose

The lattice substitutor currently matches a detected span to a lattice profile entry by **exact
string equality** on `(runtime_type, _norm(surface))` against the entry's canonical surface and
producer-enumerated aliases (`src/cloak/lattice_profiles.py::lookup_levels`). Any surface variance
the producer did not enumerate — plural, articles, typos, punctuation, morphology
(`diabetic`→`diabetes`), modifiers (`severe asthma`→`asthma`), synonyms
(`heart attack`→`myocardial infarction`) — misses the profile and silently degrades to the
uncertified WordNet last-word fallback or to the typed placeholder. These are utility failures
(placeholder where a certified generalization exists) and one fidelity failure (last-word WordNet
levels entering lattices without certified counts).

This spec replaces the miss path with **semantic retrieval**: match by meaning via a local
sentence-embedding model, then certify the match with the existing NLI gate. It is deliberately
training-free; the trained successor is the learned canonicalizer
([substitutor-profile-match-learned-canonicalizer.md](substitutor-profile-match-learned-canonicalizer.md)).

## Definitions

- **Profile entry** — one row of `lattice_profiles.json`: a canonical surface with `aliases`,
  ordered `levels`, `count`, and optional `level_counts`, under one runtime type.
- **Proposer** — a component that suggests a profile entry for a span. Proposers may be fuzzy and
  wrong; they carry no privacy authority.
- **Certifier** — the components that make a proposed replacement legal: the NLI truthfulness gate
  in the span's sentence (`cloak.lattice.nli_gate`) and the anonymity-floor check
  (`aset_count(fill, type, original, strict=True) >= K_FLOORS[type]`).
- **Exact hit** — the current `_norm` match (lowercase + whitespace collapse) against canonical
  surface or alias. Remains the trusted fast path.
- **Semantic hit** — a match produced by embedding retrieval and accepted by the certifier.
- **Embedding index** — a per-runtime-type matrix of embedded surfaces/aliases derived from the
  profile artifact at build time, plus the row → entry mapping.

## Architectural invariant — propose vs certify

**Matching only proposes; certification is unchanged and mandatory for every non-exact hit.**
A wrong semantic match must be caught by NLI-in-context or land on levels that are still truthfully
entailed; if certification fails, the span falls closed to the typed placeholder exactly as today.
Under this invariant the matcher can never widen the privacy surface — it can only recover utility —
and matcher quality reduces to proposer recall, measured as a diagnostic.

Consequences:

1. Exact hits keep their current deterministic treatment (`deterministic = True` in
   `lattice_for()`, NLI gate skipped).
2. Semantic hits are **never** deterministic: their levels must pass `nli_gate` in the span's
   sentence before use, even though the levels come from a certified profile row.
3. The WordNet last-word fallback becomes diagnostic-only for fine and domain types. It must no
   longer produce lattice levels that reach `aset_count` certification (this closes the existing
   spec violation in `cloak.lattice.wordnet_chain` usage).

## Embedding index artifact

Built by the profile build step (`scripts/build_lattice_profiles.py` or a sibling script), never at
runtime. For each runtime type in the profile artifact:

- Embed every canonical surface and every alias with the configured local model.
- Default model: `BAAI/bge-small-en-v1.5` (small, local, subword-tokenized so typos degrade
  gracefully). The model id is a build-time config value recorded in the artifact metadata, not a
  runtime choice.
- L2-normalize embeddings; similarity is cosine via dot product.

Artifact layout, stored next to the profile it indexes:

```
data/lattice_profiles/lattice_profiles.embindex.npz
  vectors        float32 [N, D]      # normalized
  meta.json (embedded or sidecar)
    model_id, dim, schema_version
    profile_hash                     # hash of the profiles JSON the index was built from
    rows: [{runtime_type, canonical, source_text}]   # row i ↔ vectors[i]
```

Runtime loading rules:

- If the index file is absent, or `profile_hash` does not match the loaded profile artifact, or the
  embedding model cannot be loaded, the matcher **degrades to exact-only** (current behavior) and
  logs the reason once. It never crashes and never rebuilds at runtime.
- User-defined runtime types get semantic matching for free: the index is rebuilt from whatever
  profile artifact is loaded, with no per-type code.

## Runtime matching algorithm

```python
# --- constants (initial; calibrated on the eval set, never per-method fudged) ---
TOP_K = 5
SIM_FLOOR = 0.70          # below: no proposal, fall through to placeholder path
NLI_THRESH = 0.6          # reuse the existing gate threshold

def match_profile_entry(span_text, runtime_type, context) -> MatchResult | None:
    # 1. exact fast path — unchanged current behavior
    levels = lookup_levels(span_text, runtime_type)          # _norm exact match
    if levels:
        return MatchResult(levels=levels, kind="exact", deterministic=True,
                           similarity=1.0, entry=canonical_of(span_text, runtime_type))

    # 2. semantic retrieval (proposer)
    index = load_embindex()                                   # cached; None if degraded
    if index is None or runtime_type not in index.types:
        return None                                           # exact-only degradation
    q = embed(span_text)                                      # same model as build time
    hits = top_k_cosine(q, index.vectors_for(runtime_type), k=TOP_K)
    hits = [h for h in hits if h.sim >= SIM_FLOOR]
    if not hits:
        return None

    # 3. certification (certifier) — try candidates best-first
    for h in hits:
        entry = index.entry(h.row)                            # (canonical, levels)
        if not context:
            # no sentence to certify in -> cannot certify a fuzzy match; fail closed
            return None
        approved = nli_gate(span_text, context, entry.levels, thresh=NLI_THRESH)
        if approved:
            return MatchResult(levels=approved, kind="semantic", deterministic=False,
                               similarity=h.sim, entry=entry.canonical)
    return None                                               # all candidates refused
```

Integration point — as implemented, `cloak.substitute.substitute()` runs one **document-level
pre-pass** (`match_spans_batch`) over every profile-backed span, then feeds each certified verdict
into `cloak.lattice.lattice_for()` as a `proposal`. `lattice_for` reads the pre-pass result via a
three-state `proposal` argument instead of matching per-span:

```python
# substitute(): one batched pre-pass for the whole document
items = [(s.text, s.type, sentence_around(text, s.start, s.end))
         for s in spans if s.type in PROFILE_BACKED_TYPES]
proposals = match_spans_batch(items)          # {span_key: MatchResult | None}

# per span:
prop = proposals.get(span_key(s.text, s.type), NO_PREPASS)
lattice = lattice_for(s.text, s.type, sent, proposal=prop)

# lattice_for(): proposal is one of three states
#   MatchResult -> got, deterministic = m.levels, True   # certified upstream, not re-gated
#   None        -> abstained: skip the matcher, fall through to curated/GeoNames/teacher cache
#   NO_PREPASS  -> no pre-pass ran: match_profile_entry(...) per-span here (single-span path)
# wordnet_chain is now diagnostic-only for fine/domain types — it never feeds `got`.
```

The single-span `match_profile_entry` remains as a thin wrapper over `match_spans_batch` (used by
the `NO_PREPASS` path and by callers that match one span at a time), so both paths share one
retrieval+certification implementation.

Notes:

- Step 3 iterates candidates: cosine-close **siblings** (`hypothyroidism` vs `hyperthyroidism`) are
  the known false-positive class; NLI-in-context refuses the harmful ones, and harmless ones share
  family-level levels anyway. A top-1/top-2 similarity margin is logged as a diagnostic but is not
  a legality condition.
- Modifier semantics are handled by the certifier, not string logic: `severe asthma` → entry
  `asthma` passes (levels remain entailed). Hedged/negated modifiers (`suspected X`) are refused by
  NLI **only when the entry's levels are specific enough to break entailment** — see "Measured
  limitation" below: the spike measured `suspected cancer` → entry `cancer` *certified* because that
  entry's levels are generic (`disease` / `medical condition`), which the hedged span still entails.
- A span with no context sentence cannot be certified, so semantic matching is disabled for it
  (fail closed). Exact hits do not need context, as today.

## Anonymity counts

`aset_count` / `lookup_count` are unchanged. Semantic hits return levels **verbatim** from the
matched entry, so `lookup_count(fill, type)` resolves by construction, with the entry's
`level_counts` semantics intact. The span variant belongs to the same anonymity set as the entry's
canonical surface; the count attaches to the generalization tier, not to the surface spelling.

## Efficiency

Retrieval and certification are batched per document, not per span:

- **Per-type brute-force GEMV.** Retrieval is a single matrix-vector product against the queried
  type's row block (`index.vectors[type_rows] @ q`), then a top-k over `SIM_FLOOR`. Per-type matrices
  are a few hundred to a few thousand rows; brute force suffices. ANN is out of scope until profiles
  grow past ~10^5 rows per type.
- **One embed batch per document.** `match_spans_batch` collects every uncached miss surface across
  the whole document and embeds them in a single `encode` call; exact hits and cache hits never touch
  the model.
- **Wave-batched best-first NLI certification.** `nli_gate_batch` certifies candidates in waves — wave
  `w` submits every still-unresolved span's `w`-th retrieval candidate as one NLI batch, stopping a span
  as soon as a candidate is approved. This keeps certification best-first per span while batching the
  work across spans.
- **In-process proposal cache** keyed `(index_path, runtime_type, norm_surface)` memoizes **retrieval
  only** — the candidate list. Certification is context-dependent (NLI in the span's sentence) and is
  **never** cached. The cache clears wholesale past a size cap before the uncached-embed step, so no key
  is left dangling.
- **Alias promotion** from logged `R` semantic matches (folding recovered surfaces back into the
  profile's alias lists so future runs hit the exact fast path) is a producer-side follow-up, not part
  of this runtime path.

## Substitution record `R` diagnostics

Every semantic hit records its provenance for offline analysis and eval-set harvesting:

As implemented in `cloak.substitute.substitute()`, a matched span carries a `match` block on its
`R` entry:

```json
{
  "surface": "heart attacks",
  "type": "health-condition",
  "action": "generalize",
  "replacement": "cardiovascular condition",
  "match": {"kind": "semantic", "entry": "myocardial infarction",
            "similarity": 0.83, "nli": 0.91}
}
```

Semantic hits record `{"kind": "semantic", "entry", "similarity", "nli"}` (`similarity` and `nli`
rounded to 3 places; `nli` is `None` when the certifier returns no per-level score). Exact hits
record `{"kind": "exact"}`. Spans with no pre-pass verdict (abstained or not profile-backed) carry
no `match` block. These logs are the training/eval seed for the learned canonicalizer.

## Failure-mode coverage

| Variation (current behavior: miss) | Covered | Mechanism |
|---|---|---|
| Singular/plural, morphology (`diabetic`) | yes | embedding proximity + NLI |
| Articles `a/an/the`, trailing punctuation | yes | embedding proximity |
| Typos | mostly | subword embeddings degrade gracefully; gross typos fall below `SIM_FLOOR` → placeholder |
| Modifiers (`severe asthma`) | entailed: yes; hedged/negated (`suspected X`): only when levels are non-generic | proximity proposes, NLI certifies entailment — but generic levels are weakly entailed even by hedged spans (measured; see below) |
| Synonyms/paraphrase (`heart attack`) | yes | the case no alias list ever covers |
| Context-dependent abbreviations (`MI`) | **no** | the query embeds the bare surface; needs the learned canonicalizer |
| Cross-type mismatch (`medical procedure` vs `medical-procedure` type key) | no | out of scope; type contract is the detector's |

## Measured limitation — generic levels certify too easily

The validation spike (`scripts/spikes/validate_profile_match.py`, evidence
`scripts/spikes/validate_profile_match.out.txt`) empirically falsified two abstain expectations:

- `suspected cancer` → entry `cancer` **certified** (sim 0.789), where it should have abstained.
- `malaria` → entry `influenza` **certified** (sim 0.741), a wrong-entry link that should have abstained.

Root cause: **the NLI certifier's protection is only as fine-grained as the entry's levels.** When
an entry's levels are generic (`disease`, `medical condition`, `illness`), they are weakly entailed
even by hedged (`suspected X`), negated, or merely-neighboring spans — so the gate approves. This is
not a property of the modifier: `possible fracture` correctly abstained in the same run, because the
`fracture` entry's levels are specific enough to break entailment. Hedged-modifier behavior therefore
depends on level granularity, not on the modifier word.

Mitigation is an open design question — e.g. a minimum level-specificity requirement for semantic
hits, or an NLI margin against a type-name hypothesis (span must entail the entry's levels *more* than
it entails the bare type name). No solution is chosen here; the limitation is recorded so a comparison
does not overstate the certifier's protection.

## Evaluation

- Build a held-out set of `(surface variant, context, runtime_type) → gold entry` pairs from
  (a) semantic-hit logs in `R`, human-spot-checked, and (b) teacher-labeled variants of profile
  surfaces. Report proposer recall@k, certified-match precision, and abstain (placeholder) rate.
- Calibrate `SIM_FLOOR` on this set once; thresholds are global, not per-type or per-method
  (empirical-honesty rule — no per-model calibration knobs).
- The claim-bearing metric remains end-to-end: utility on `out_final` at matched attacker-measured
  privacy, with matcher recall reported as a diagnostic only.

## Non-goals

- No runtime remote calls; embedding and NLI models are local.
- No rule/normalization ladder (article stripping, lemmatizers); variance absorption is semantic.
- No context-aware disambiguation of abbreviations — explicitly deferred to the learned
  canonicalizer spec.
- No change to placeholder-or-keep categorical leaves (`gender`, `marital-status`,
  `sexual-orientation`) or to forced-placeholder direct types.

## Open questions

1. Whether to embed `surface + short context window` instead of the bare surface as the query —
   improves sense disambiguation slightly, but pollutes similarity with context topic; measure on
   the eval set before adopting.
2. `SIM_FLOOR` per type family vs global: keep global unless the eval set shows a large,
   attacker-measured utility gap (and then it is a calibrated constant recorded in the artifact,
   not a tuning knob per experiment).
