---
type: plan
status: current
created: 2026-07-08
updated: 2026-07-27
tags:
  - lattice-producer
  - coherence
  - grounding
  - langgraph
  - fine-types
  - drug
  - health-condition
  - medical-procedure
  - doid
  - icd10pcs
companion:
  - docs/plans/2026-07-07-generalization-lattice-producer-agent.md
  - docs/specs/lattice-substitutor.md
  - docs/specs/generalization-lattice-cache.md
  - docs/specs/offline-k-anonimity-risk-walk.md
---

# Lattice Producer Coherence & Grounding Hardening

> For agentic workers: implement this plan task-by-task. Keep checkboxes updated. This plan
> amends the graph/gates/prompts defined in
> [2026-07-07-generalization-lattice-producer-agent.md](2026-07-07-generalization-lattice-producer-agent.md)
> (the "producer plan") -- read that doc first. It is not superseded; this plan only changes the
> node contracts and gates it lists where evidence below shows they don't hold in practice.

## Why this plan exists

The producer plan's `drug-health-procedure` run (`results/lattice-producer/drug-health-procedure/`,
proposed into `data/lattice_profiles/proposed/drug-health-procedure.proposed.json`) was manually
reviewed and cleaned up entry-by-entry over several passes (see
`scripts/spikes/clean_drug_health_lattice_coherence.py`,
`scripts/spikes/add_openfda_epc_levels.py`, and the merge into
`data/lattice_profiles/lattice_profiles.json`). That cleanup fixed the *data* for one run. This
plan fixes the *agent* so the next run doesn't need the same manual pass. Every fix below is
motivated by a specific, reproduced failure from that review -- no speculative hardening.

## Evidence: what actually broke, with file:line

1. **Count gate never really fails closed.** `compile_level_counts`
   (`src/cloak/lattice/producer/counts.py:68-89`) takes the model's self-reported
   `proposed_count` verbatim as `level_count` whenever the model also fills in non-empty
   `count_evidence`/`selector` text -- it only fails closed if those fields are *missing*, never
   if they're unverifiable. `gate_candidates`'s Count gate (`src/cloak/lattice/producer/gates.py:97-120`)
   then checks that self-reported number against `K_FLOORS`. Net effect: the model proposes a
   count, invents plausible-sounding evidence for it, and the gate compares the invented count to
   the floor -- grading its own homework. Result on the real run: **100% of 2,597 accepted levels**
   had `level_grounding.status == "model-proposed"`, `member_set_ref: null`, zero real
   `fail-closed` rejections, and some counts were physically absurd (`rizatriptan` ->
   "therapeutic compound" = 120,000,000).

2. **`drug` is `needs_profile` in the registry but eligible in the queue builder.**
   `coverage.py:81` correctly assigns `drug`/`medical process`/`blood type` to
   `CategoryOutcome.NEEDS_PROFILE`, matching the producer plan's own coverage table ("do not
   silently fold into health-condition unless a profile defines levels/count semantics"). But
   `queue.py:19-35` has its own separate, hardcoded `LATTICE_RUNTIME_TYPES` set that includes
   `"drug"` as eligible, and `_queue_from_profile_categories` (`queue.py:114-135`) resolves
   `runtime_type` directly from the base artifact's own top-level profile keys, **never calling
   `registry_entry_for_label`** -- so `normalize_item`'s registry-lookup branch (`queue.py:41`) is
   skipped and eligibility falls through to the wrong, locally-duplicated set. Result: a
   538-entry `drug` profile got generated and persisted despite the explicit policy gate.

3. **No domain-relevance check, so mined category bleed passes straight through.**
   `gate_candidates` has no check that a candidate's levels are even in the right domain. Real
   examples that passed every existing gate: `baseball` -> `["sport", "game", "human activity"]`
   filed under `health-condition`; `pizza burgers` filed under `drug`; medical *devices*
   (`ekg`, `pulse ox`, `pacemaker`, `stent`, `cast`) filed under `drug`/`health-condition` instead
   of the `medical process` bucket the producer plan's own coverage table specifies. Some of this
   traces to mis-tagged entries in the mined source (`data/lattice_profiles/lattice_profiles.json`)
   itself, but nothing downstream catches the mismatch either.

4. **No cross-entry context -> the same concept gets a different phrasing and a different count
   every time.** `assemble_context_packet` (producer plan, "Context Management") is scoped to one
   item with no visibility into how *other* items in the same run already phrased or counted a
   shared abstraction tier. Real effect: `pharmaceutical compound` / `pharmaceutical agent` /
   `pharmaceutical product` / `pharmacological substance` / `active pharmaceutical ingredient` all
   appeared as *distinct* levels for the identical concept, and the literal string `"medication"`
   carried 70 different self-reported counts across 186 occurrences (142 to 24,500,000). This is
   the single largest driver of the manual cleanup effort and has no owner in the current graph.

5. **Ambiguous/short surfaces get one confident, sometimes-wrong resolution with no
   disambiguation path.** Clinical shorthand collides with drug names in the mined data:
   `bun` -> resolved to "bunion" (real BUN = blood urea nitrogen, a lab test); `bph` ->
   resolved to "hypertension" (real BPH = benign prostatic hyperplasia); `ct` -> resolved to
   "chest/cardiac trauma" (real CT = CT scan); `cad` -> resolved to "cadmium derivative" (real CAD
   = coronary artery disease); `g5 p5` (an obstetric gravida/para notation, not a drug) ->
   resolved to a fabricated "compound g5p5". The model always answers; there is no
   `needs_disambiguation` outcome, and no gate flags a resolution that contradicts the surface's
   dominant real-world meaning.

6. **Hallucinated non-answer aliases pass the "has evidence" check.** When the model has no real
   referent, it emits templated filler (`pharmaceutical_compound`, `clinical_agent`,
   `therapeutic_substance`) as if they were aliases, satisfying `_has_model_evidence`
   (`gates.py:51-56`, which only checks the fields are *non-empty*, not that they're informative).
   12 confirmed cases in the reviewed run, including `mcnuggates` -> "medicinal compound"
   (a McDonald's menu item) and `camila` -> generic filler (a real birth-control brand the model
   didn't recognize).

7. **Real, free, deterministic classification data sits unused -- for all three of drug,
   health-condition, and medical-procedure, not just drug.** This is not a drug-specific gap;
   it's a general pattern in how the producer treats every runtime type, and the other two
   categories are currently worse off than drug was before this session's manual patch:
   - **drug**: `data/lattice_sources/raw/drug/openfda_ndc.json.zip` carries FDA's own
     Established Pharmacologic Class (`pharm_class` EPC tags) for 53.8% of NDC records
     (single-active-ingredient records: 59,761). The manual patch
     (`scripts/spikes/add_openfda_epc_levels.py`) recovered real, certifying levels for 143 of
     538 drug entries this way (e.g. `bupropion` -> `Aminoketone [EPC]`, count = 9 distinct FDA
     generic names).
   - **health-condition**: `data/lattice_sources/raw/health/doid.obo` is the Disease Ontology
     (14,735 `[Term]` stanzas, CC0 public domain) -- already the *mining* source for 213 of 771
     health-condition entries (`source_ids` prefixed `DOID:`), but its `is_a` parent hierarchy is
     never read for level generation. It is richer than the drug case: it gives a full multi-hop
     **chain**, not just one class tag. Confirmed by direct lookup this session: `chlamydia`
     (`DOID:11263`) has the real chain `chlamydia -> commensal bacterial infectious disease ->
     bacterial infectious disease -> disease by infectious agent -> disease`, straight from the
     ontology -- truthful and free, versus the LLM-generated chain it actually got
     (`Sexually transmitted infection -> Bacterial infection -> Human disease -> Pathological
     condition`, plausible-sounding but non-authoritative and, per Fix Area 6's casing bug,
     inconsistently cased against the rest of the corpus).
   - **medical-procedure**: `data/lattice_sources/raw/procedure/icd10pcs_order_2026.zip` contains
     the full ICD-10-PCS order file (80,029 codes, fixed-width text). Every code's 7 characters
     positionally encode Section / Body System / Root Operation / Body Part / Approach / Device /
     Qualifier -- code-prefix membership is a real, deterministic, computable hierarchy (e.g. all
     codes sharing the first 3 characters share Section+BodySystem+RootOperation). This category
     is the worst-served of the three right now: **427 of 488 medical-procedure entries (87.5%)
     have only the single bare level `"medical procedure"`**, with no intermediate tier at all --
     worse than drug's pre-patch state, where every entry at least had the
     `medication`/`pharmaceutical compound`/`chemical substance` ladder.

   `compile_level_counts` never attempts a real lookup against any of these before asking the
   model -- it goes straight from "no `member_set`" to "ask the model" to "trust the model's own
   number," for every runtime type equally.

## Non-Negotiable Constraints (additions to the producer plan's list)

- A canonical-vocabulary lookup or a real local reference dataset must be tried **before** any
  model call for level proposal or count compilation. The model is the fallback, not the first
  resort.
- A context packet for level-proposal or generated-universe tasks must include the current
  bounded canonical vocabulary slice for that runtime type (see Fix Area 3). This does not
  relax the producer plan's context-rot rules (still bounded, still hashed, still no raw
  history) -- it adds one more capped, typed slice alongside `nearby_profile_rows`.
- The model must have a legitimate way to say "I don't know" or "this surface is ambiguous."
  Forcing a confident answer out of every call is what produced the BUN/BPH/CAD/CT
  misresolutions.
- Every accepted level's count must be reproducible by rerunning the same deterministic
  compiler over the same source snapshot -- "model says X and provides prose" is never
  sufficient grounds for `status: certifying`, only for `status: model-proposed`.
- Coherence (one canonical spelling and one count per shared abstraction tier) is enforced
  **inside the graph, before persistence** -- not as an external cleanup script run after the
  fact.

## Fix Area 1 -- Registry Consistency (queue eligibility)

**Problem:** `queue.py`'s local `LATTICE_RUNTIME_TYPES` set duplicates and contradicts
`coverage.py`'s `CategoryOutcome` registry (Evidence #2).

**Files:**
- Modify `src/cloak/lattice/producer/queue.py`: delete the module-level `LATTICE_RUNTIME_TYPES`
  set (`queue.py:19-35`). `normalize_item` (`queue.py:38-63`) must derive eligibility purely from
  `registry_entry_for_label(...).outcome`, for every code path -- including
  `_queue_from_profile_categories` (`queue.py:114-135`), which currently sets
  `runtime_type`/`detector_label_family` directly from the base artifact's top-level keys and
  skips the registry lookup because `item.get("runtime_type")` is already truthy
  (`normalize_item`'s guard at `queue.py:41` is `if label and not item.get("runtime_type")`).
  Fix: `_queue_from_profile_categories` must NOT pre-set `runtime_type`; let `normalize_item`
  resolve it from `detector_label_family` via the registry every time, for both queue-building
  paths.
- Test: `src/cloak/tests/test_lattice_producer_queue.py` -- add a case building a queue from
  `_queue_from_profile_categories` over a fixture profile artifact that has a top-level `"drug"`
  key, and assert the resulting items are `eligible: False`, `skip_reason: "needs_profile"` (not
  silently re-labeled `runtime_type: "drug"` and marked eligible).

## Fix Area 2 -- Deterministic-First Generation (real ontology/registry lookups before the model)

**Problem:** `compile_level_counts` (and, for health-condition/medical-procedure, level
*generation* itself) never tries a real local dataset before falling back to the model
(Evidence #1, #7). This is the highest-priority fix area in this plan: medications already went
through a manual real-data patch this session, but health-condition and medical-procedure did
not, and medical-procedure in particular is currently worse off (87.5% flat single-level) than
drug ever was. All three loaders below are equally in scope for this plan -- do not implement
only the drug one because it's already proven; the point of this fix area is that the other two
need it more.

**Files:**
- Create `src/cloak/lattice/producer/reference_sources.py`: a registry of
  `runtime_type -> loader`, with three loaders:

  1. `openfda_pharm_class_index(raw_zip_path) -> dict[str, tuple[str, float]]` for `drug` (base
     drug name -> (EPC label, real distinct-generic-name count)), ported from
     `scripts/spikes/add_openfda_epc_levels.py`'s `load_epc_index`, **including the
     single-active-ingredient filter** (added after the real "acetaminophen -> antihistamine"
     combo-product contamination bug this session) **and the self-leak guard** (added after the
     real "progesterone -> progesterone" bug). Port both fixes, not just the happy path.

  2. `doid_hierarchy_index(obo_path) -> dict[str, DoidNode]` for `health-condition`, where
     `DoidNode` has `name`, `parents: list[str]` (DOID ids), and a lazily-computed
     `descendant_count` (real, deterministic: count of all DOID terms whose transitive `is_a`
     closure includes this node -- a genuine "how many named diseases fall under this
     generalization" number, not an estimate). Parse `[Term]` stanzas for `id:`, `name:`, and
     `is_a: DOID:xxxxx ! <parent name>` lines (obo format, confirmed this session: 14,735 terms,
     17,274 `is_a` edges). Given a matched surface, walk `is_a` from the leaf term upward,
     capping the number of hops emitted (e.g. 3-4, matching the depth other categories' lattices
     use) rather than always walking to the ontology root (`disease`, `DOID:4`) -- the root is
     too broad to be a useful mid-chain level; treat it as the runtime type's ceiling instead of
     an arbitrary rung. Each hop is `status: "certifying"`,
     `member_set_ref: "doid:is_a_descendants:<DOID id>"`.

  3. `icd10pcs_hierarchy_index(zip_path) -> dict[str, Icd10PcsCode]` for `medical-procedure`,
     parsing the fixed-width `icd10pcs_order_2026.txt` member of the zip: seq number at
     columns [0:5], 7-character code (space-padded for header rows) at [6:13], header flag at
     [14] (confirmed this session to be `"0"` for a header/prefix row and `"1"` for a fully
     specified code -- the reverse of a first guess), 61-character short description at
     [16:77], long description in the remainder; 80,029 data rows total, of which 914 are
     3-character header rows (Section+BodySystem+RootOperation, e.g. `001` -> "Central Nervous
     System and Cranial Nerves, Bypass"). The file has **no 1- or 4-character header rows**
     (confirmed this session) -- only the 3-character tier is real; do not synthesize a
     1-character Section tier from a hand-written table just to have one more rung. For a
     matched procedure, emit the one real intermediate level (3-character prefix) using the
     header row's own long description as the level text, never a synthesized label. Count per
     prefix = number of distinct full 7-character codes sharing it, computed once from the
     parsed table.

- Modify `src/cloak/lattice/producer/counts.py`: in `compile_level_counts`, before the
  `elif candidate.get("source_family") == "model-proposed"` branch (`counts.py:68`), add a new
  branch that calls the reference-source loader for `runtime_type` (if one is registered) and,
  on a hit, emits `status: "certifying"`, `source_family: "<source-name>-reference"`,
  `member_set_ref` pointing at the concrete source (e.g.
  `"openfda-ndc:pharm_class:Aminoketone [EPC]"`, `"doid:is_a_descendants:DOID:104"`,
  `"icd10pcs:prefix:001"`), skipping the model-proposed path entirely for that candidate.
- For `health-condition` and `medical-procedure`, this is not just a count backfill the way it
  was for drug -- the reference source supplies the **level text itself** (the real DOID parent
  name / real ICD-10-PCS prefix description), not only a number to attach to an
  already-model-proposed label. `deterministic_lookup` (producer plan, Node Contracts) should
  try the reference-source chain **before** `propose_with_llama_swap` runs at all for a matched
  surface, per the producer plan's existing "try local sources before model calls" ordering --
  when a DOID/ICD-10-PCS match exists, the model should not be generating a level chain for that
  item, only aliases and any level *below* what the ontology already covers (e.g. a specific
  brand/informal name the ontology doesn't list).
- Test: `src/cloak/tests/test_lattice_producer_reference_sources.py` -- three fixture-driven
  cases:
  - openFDA: a tiny NDC-shaped JSON fixture (one combo product, one single-ingredient, one
    molecule-name-equals-EPC-name case) asserting the two ported bug fixes still hold.
  - DOID: a tiny OBO-shaped text fixture with a 3-hop `is_a` chain, asserting the walked chain
    order, descendant counts, and the root-as-ceiling behavior (root itself is never emitted as
    a mid-chain rung).
  - ICD-10-PCS: a tiny fixed-width fixture with a handful of codes sharing prefixes at different
    lengths, asserting prefix-membership counts and that header rows (not synthesized text)
    supply the level label.

## Fix Area 3 -- Canonical Vocabulary (fixes paraphrase proliferation and count incoherence at the source)

**Problem:** No shared, bounded vocabulary of standard abstraction-tier labels exists, so the
model reinvents phrasing and a count for every occurrence independently (Evidence #4). This is
the fix that makes `scripts/spikes/clean_drug_health_lattice_coherence.py`'s post-hoc
rank+PAVA+anchor pass unnecessary for future runs.

**Scope note given Fix Area 2:** for `health-condition`, Fix Area 2's DOID chain covers most
entries most of the way up the ladder, so the hand-curated anchor table's job shrinks to (a) the
handful of ceiling-tier catch-all labels above DOID's own root (`disease`, `DOID:4`) that this
lattice still wants (e.g. a `medical condition` terminal broader than DOID covers), and (b) a
fallback for entries with no DOID match at all. It is not made redundant -- not every
health-condition surface will match a DOID term (`mined-clinical` entries with no `DOID:` source
id, 591 of 771, are exactly the ones most likely to miss). For `medical-procedure`, no
hand-curated anchor file is needed: the confirmed 2026 ICD-10-PCS order file carries 914 real
3-character header rows (Section+BodySystem+RootOperation, e.g. code `001` = "Central Nervous
System and Cranial Nerves, Bypass"), extracted directly by Fix Area 2's loader -- **not** a
1-character Section-only tier, since the source file has no header rows at that length (verified
this session; earlier drafts of this plan assumed a 31-section top tier that the actual file
does not support and this plan does not fabricate one to fill the gap). Those 914 header
descriptions are the natural top-tier vocabulary and should seed `CanonicalVocabulary` for that
runtime type automatically from Fix Area 2's loader output, not from a separate curated file.

**Files:**
- Create `data/lattice_sources/reference/drug_class_anchors.json` and
  `health_condition_class_anchors.json`: promote the curated `REFERENCE_COUNTS` table from
  `scripts/spikes/clean_drug_health_lattice_coherence.py` (with its sourced comments --
  CAS Registry, ChEMBL, FDA Orange Book, ICD-10-CM, SNOMED CT, DSM-5, Orphanet, NINDS) into a
  versioned data file. Same content, promoted from a spike script constant to a maintained
  source-of-truth artifact. No equivalent file for `medical-procedure` -- see scope note above.
- Create `src/cloak/lattice/producer/vocabulary.py`: a `CanonicalVocabulary` that, per runtime
  type, holds `{normalized_label: count}`, seeded from (drug, health-condition) the anchor file
  above, (medical-procedure) the 914 real header descriptions already parsed by Fix Area 2's
  `load_icd10pcs_index` loader, plus -- for every runtime type, via an optional `proposed_out`
  constructor argument -- every level accepted so far *this run*, read directly off the on-disk
  proposed artifact rather than threaded through LangGraph state. This works without a
  `ProducerState` schema change because the graph persists every accepted item
  (`persist_proposed_artifact_node`) before the next item's context packet or gate check ever
  runs -- items are processed strictly sequentially, never in parallel -- so item 500 genuinely
  sees item 50's already-committed labels. Exposes `nearest(candidate_label, k, *,
  min_overlap)` (token-Jaccard, no embeddings dependency) and `has_exact(label) -> bool`.
  **Without this dynamic half, the whole fix area only catches duplicates of the ~40 hand-curated
  anchor labels -- it does nothing for the far more common case of the model inventing two
  different paraphrases of the same concept across two different items, neither of which is a
  static anchor.** This is not an optional refinement; ship both halves together.
- Modify `assemble_context_packet` (producer plan) and `propose_with_llama_swap`: both take an
  optional `proposed_out` parameter, threaded through from `state["proposed_out"]` in
  `propose_with_llama_swap_node`. Add a capped field `canonical_vocabulary_slice: list[str]` --
  the top `--max-context-rows` vocabulary entries by count, hashed into the existing
  `context_packet_hash` (which already covers the whole packet, so no separate cache-key change
  needed -- the hash changes automatically as the run-grown vocabulary changes).
- Modify the `propose_with_llama_swap` prompt contract (producer plan, Node Contracts): require
  the model to pick from `canonical_vocabulary_slice` whenever an entry semantically fits one of
  those labels, and only propose new phrasing when none fit -- with a required
  `reused_canonical_label: bool` field per level so the gate can check it. This field lives on
  the raw model payload, not threaded through `extract_candidate_levels`'s several
  payload-schema branches -- attach it to each extracted candidate as a post-processing step in
  `propose_with_llama_swap_node` instead, a smaller and safer change.
- Modify `gate_candidates`: a new **Vocabulary gate**, and an optional `proposed_out` parameter
  threaded from `gate_candidates_node` -- if `reused_canonical_label` is false but
  `CanonicalVocabulary(runtime_type, proposed_out=...).nearest(level, k=3, min_overlap=...)`
  contains a near-duplicate of the proposed level, route to diagnostics with reason
  `"unreused_near_duplicate_label"` rather than accepting a fifth paraphrase of "pharmaceutical
  compound." The near-duplicate threshold is empirically calibrated at Jaccard > 0.3, not the
  originally-guessed 0.8 -- two genuine paraphrases of a 2-3 word concept typically only share
  one token out of a 3-4 token union (e.g. "pharmaceutical product" vs the anchored
  "pharmaceutical compound" scores ~0.33); 0.8 would never fire on real multi-word labels.
- Test: `src/cloak/tests/test_lattice_producer_vocabulary.py` and
  `test_lattice_producer_gates.py` -- static-anchor seeding and nearest-match tests, plus
  dedicated dynamic-vocabulary tests: a label accepted by one item in a fixture proposed
  artifact is visible to `CanonicalVocabulary(..., proposed_out=path)` for a later item, is
  distinguishable from a static anchor via `is_from_this_run()`, does not override a real
  anchor's value if they collide, and `gate_candidates(..., proposed_out=path)` actually catches
  a later item's near-duplicate of that run-discovered (not anchored) label -- with a sibling
  test confirming the same scenario is accepted when `proposed_out` is omitted, proving the
  dynamic behavior is additive and not silently always-on via some global cache.

## Fix Area 4 -- Domain-Relevance and Alias-Quality Gates

**Problem:** No check catches obvious category bleed or non-answer filler aliases
(Evidence #3, #6).

**Files:**
- Modify `src/cloak/lattice/producer/gates.py`: add a **Domain-relevance gate**. Cheapest
  viable version: reject to diagnostics if none of the candidate's proposed levels or the
  entry's own detector label family share a token with a small per-runtime-type keyword allowlist
  derived from the `CanonicalVocabulary` seed (Fix Area 3) -- e.g. a `drug` entry whose *entire*
  level chain is `["sport", "game", "human activity"]` shares zero tokens with any seeded drug
  vocabulary and gets flagged `reason: "no_domain_overlap"` instead of silently accepted. This
  reuses infrastructure from Fix Area 3, no new dependency.
- Add a **Generic-filler-alias gate**, porting the heuristic already validated ad hoc this
  session: reject to diagnostics if every alias's tokens are a subset of a fixed generic-filler
  vocabulary (`compound`, `agent`, `substance`, `reference`, `entry`, `record`, `formulation`,
  `preparation`, `product`, `variant`, `drug`, `medication`, `therapeutic`, `pharmaceutical`,
  `designated`, `indexed`, `clinical`, `chemical`, `medicinal`, `molecule`, `material`,
  `biological`, `code`, `fragment` -- the exact list that caught `mcnuggates`, `camila`,
  `abcdes`, and 9 other confirmed non-answers in the reviewed run) AND there is more than one
  alias (a single alias can't be checked against "all aliases are filler" meaningfully).
- Test: `src/cloak/tests/test_lattice_producer_gates.py` -- add cases for both gates using the
  exact real examples above (`baseball`/`["sport","game","human activity"]` for domain-relevance;
  `mcnuggates`/`["medicinal compound","pharmaceutical preparation","clinical agent"]` for
  generic-filler).

## Fix Area 5 -- Ambiguous-Surface Disambiguation Path

**Problem:** The model always resolves confidently, even for surfaces it cannot actually
identify (Evidence #5).

**Files:**
- Modify the `propose_with_llama_swap` prompt contract: require a new field
  `surface_confidence: "high" | "low" | "ambiguous"` per item, with instructions that a short
  (<=4 alphanumeric characters) or multi-referent clinical abbreviation must be marked `low` or
  `ambiguous` unless the surrounding `marked_context_sentence` disambiguates it, and that the
  model must not silently pick one referent among several equally plausible ones.
- Add a `needs_disambiguation` outcome to `gate_candidates`: `surface_confidence in {"low",
  "ambiguous"}` routes the candidate to `diagnostics.jsonl` with `reason:
  "low_confidence_surface"` regardless of how clean the rest of the candidate looks, rather than
  entering `accepted.jsonl`.
- Add a length/ambiguity heuristic as a backstop even when the model over-claims confidence:
  surfaces with `len(surface.replace(" ", "")) <= 4` and no exact match in
  `CanonicalVocabulary` or the deterministic sources get force-flagged
  `needs_disambiguation` regardless of the model's own `surface_confidence` field -- this is
  exactly the `bun`/`bph`/`cad`/`ct`/`g5 p5` class of failure, and the model's own confidence
  field cannot be fully trusted to self-report correctly (it didn't, in the reviewed run).
- Test: `src/cloak/tests/test_lattice_producer_gates.py` -- assert a 3-character surface with a
  clean-looking model proposal still routes to diagnostics, not accepted.

## Fix Area 6 -- In-Graph Coherence Normalization Pass

**Problem:** Coherence (one spelling, one count per shared tier, correct narrow->broad order)
was only achieved by an external script run after the fact
(`scripts/spikes/clean_drug_health_lattice_coherence.py`). Fix Areas 2-3 prevent most
incoherence from being introduced in the first place, but a normalization pass is still needed
as a closing guarantee -- future runs will still mix certifying, reference-anchored, and
model-proposed levels, and those need one final reconciliation before persistence.

**Files:**
- Create `src/cloak/lattice/producer/coherence.py`, porting (not reimplementing) the algorithm
  proven in `scripts/spikes/clean_drug_health_lattice_coherence.py`:
  `average_depth_rank`, `weighted_pava`, `rank_order` (with the anchor/empirical-rank hybrid
  ordering fix), and the reorder-chain-by-shared-rank step. Keep the same three bugs already
  found and fixed in the spike script fixed here too: (a) tie-breaking must be identical between
  the count-assignment sort and the chain-reorder sort (the `aldactone`/duplicate-level bug), (b)
  anchor-to-anchor order must come from the anchor's own value, not empirical chain-depth (the
  `genetic disorder`/`musculoskeletal disorder` false-collision bug), (c) a final per-entry
  monotonic safety clamp is still required as a backstop even after reordering.
- Add a graph node `normalize_coherence`, run once between `gate_candidates`'s accepted output
  and `validate_proposed_artifact` (producer plan, Graph Shape) -- not per-item, since it needs
  the whole run's accepted levels to compute corpus-wide ranks and anchor placements.
  Conditional edge: `persist_proposed_artifact -> normalize_coherence -> validate_proposed_artifact`.
- `normalize_coherence` must record which labels needed the safety clamp (the "ambiguous,
  needs review" set from the spike script) into `coverage.json`, not silently drop the
  information.
- Test: `src/cloak/tests/test_lattice_producer_coherence.py` -- port the spike script's
  `_selfcheck` assertions (monotone, deduped, one count per non-anchor-conflicted label
  corpus-wide) as real pytest assertions over a small synthetic multi-entry fixture, including a
  regression case for each of the three bugs listed above.

## Fix Area 7 -- Artifact Validation Hardening

**Problem:** `validate_proposed_artifact` doesn't check the invariants that
`data/lattice_profiles/lattice_profiles.json`'s schema now requires (added this session in
`src/cloak/lattice/profiles.py`'s `validate_profile_artifact`), and doesn't check the
casing-consistency invariant that broke the merge once already.

**Files:**
- Modify `src/cloak/lattice/producer/graph.py`'s `validate_proposed_artifact` node: call
  `cloak.lattice.profiles.validate_profile_artifact` (already checks `level_counts` keys are a
  subset of `levels` and values `>= 1`, added this session) in addition to the producer-specific
  invariants already listed in the producer plan.
  Add one more producer-specific check: every level in a row's `levels` list must appear in
  `level_counts` with the *exact same casing* -- normalize-compare, not literal-compare, and
  fail validation (not silently coerce) if a mismatch is found, since silent coercion is what
  masked the `chlamydia`/"Infectious disease" vs "infectious disease" bug until the merge script
  happened to raise on it.
- Test: extend whichever test file covers `validate_proposed_artifact` (producer plan lists
  `test_lattice_producer_graph.py`) with a casing-mismatch fixture.

## Explicitly Out of Scope for This Plan

- **`scripts/lattice_sources/drugs.py:40`** (the openFDA NDC -> `fine_lattice_profiles.json`
  builder hardcodes `levels=["medication"]` and never reads `pharm_class` at all) is a different
  pipeline -- a plain deterministic builder, not the LangGraph producer agent -- and was
  explicitly scoped out of the drug-health-procedure data patch earlier this session. Fixing it
  is a natural follow-up (it would give ~54% of the 30,284-entry `fine_lattice_profiles.json`
  drug cache a real EPC level for free, same mechanism as Fix Area 2) but is a separate,
  smaller, non-agent change and should be its own plan or a quick spike, not folded in here.
- Rebuilding the canonical vocabulary for every runtime type (profession, nationality, LOC,
  etc.), not just `drug`/`health-condition`. Start where the evidence is; extend once this run
  proves out.
- Any change to `K_FLOORS` values or the anonymity-floor policy itself -- out of scope per the
  producer plan's non-negotiable constraints, unaffected by anything in this plan.

## Implementation Tasks

- [ ] Fix Area 1: delete `LATTICE_RUNTIME_TYPES`, route all eligibility through the registry,
      add regression test.
- [ ] Fix Area 2: `reference_sources.py` with all three loaders -- ported+bug-fixed openFDA EPC
      loader (drug), new DOID `is_a` chain loader (health-condition), new ICD-10-PCS
      prefix-hierarchy loader (medical-procedure); `compile_level_counts` and
      `deterministic_lookup` try the matching loader before the model for all three runtime
      types; tests for all three.
- [ ] Fix Area 3: promote `REFERENCE_COUNTS` to `data/lattice_sources/reference/*.json` (drug,
      health-condition only -- medical-procedure seeds from Fix Area 2's ICD-10-PCS Section
      headers instead), `vocabulary.py`, context-packet slice, prompt contract update,
      Vocabulary gate, tests.
- [ ] Fix Area 4: Domain-relevance gate, Generic-filler-alias gate, tests with the real
      confirmed examples.
- [ ] Fix Area 5: `surface_confidence` prompt field, `needs_disambiguation` outcome, short-surface
      backstop heuristic, tests.
- [ ] Fix Area 6: `coherence.py` ported from the spike script (with its three bug fixes),
      `normalize_coherence` graph node, tests including the three regression cases.
- [ ] Fix Area 7: `validate_proposed_artifact` calls `validate_profile_artifact`, adds casing
      check, tests.
- [ ] Run full existing producer test suite plus all new tests; confirm nothing in the producer
      plan's existing behavior regresses.
- [ ] Re-run a live smoke restricted to `--category medical-procedure` (small `--max-items`,
      chosen deliberately because it's the category with zero prior enrichment -- 427/488 flat
      `"medical procedure"` entries per Evidence #7) and confirm: entries with an ICD-10-PCS
      match get a real, multi-level, `status: certifying` chain instead of the flat single level;
      entries without a match still get a sane model-proposed fallback, not a validation failure.
- [ ] Re-run a live smoke restricted to `--category health-condition` and confirm: entries with a
      DOID match get the real ontology chain (spot-check `chlamydia` specifically against the
      chain confirmed this session: `commensal bacterial infectious disease -> bacterial
      infectious disease -> disease by infectious agent`); entries without a DOID match
      (`mined-clinical`-sourced, no `DOID:` id) still get processed via the existing
      model-proposed + canonical-vocabulary path, not silently skipped.
- [ ] Re-run a fresh `drug-health-procedure`-shaped run at larger scale and diff its
      same-count-collision rate against the pre-fix baseline (321 drug / 146 health-condition
      collisions before any fix, 38 / 42 after the full manual cleanup) using the same counting
      method as `scripts/spikes/clean_drug_health_lattice_coherence.py`'s collision audit --
      the new run should need little to no post-hoc cleanup to reach a comparable or better rate
      for drug, and health-condition/medical-procedure (which had no prior manual pass to compare
      against) should land at a comparably low collision rate on their first real run.

## Verification Commands

Unit tests (existing + new):

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  src/cloak/tests/test_lattice_producer_queue.py \
  src/cloak/tests/test_lattice_producer_gates.py \
  src/cloak/tests/test_lattice_producer_merge.py \
  src/cloak/tests/test_lattice_producer_graph.py \
  src/cloak/tests/test_lattice_producer_reference_sources.py \
  src/cloak/tests/test_lattice_producer_vocabulary.py \
  src/cloak/tests/test_lattice_producer_coherence.py \
  -q
```

Full existing regression suite (must stay green):

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/cloak/tests/ -q
```

Offline smoke (no live model, deterministic sources + reference lookups only):

```bash
PYTHONPATH=src .venv/bin/python -u scripts/run_lattice_producer.py \
  --run-dir data/lattice_producer/runs/coherence-hardening-offline-smoke \
  --queue tests/fixtures/lattice_producer_queue.jsonl \
  --profiles data/lattice_profiles/fine_lattice_profiles.json \
  --out data/lattice_profiles/proposed/fine_lattice_profiles.coherence-hardening-smoke.json \
  --offline-only \
  --max-items 20
```

Category-scoped smokes proving Fix Area 2 covers health-condition and medical-procedure, not
just drug (offline-only exercises the new reference-source loaders without needing llama-swap up):

```bash
PYTHONPATH=src .venv/bin/python -u scripts/run_lattice_producer.py \
  --run-dir data/lattice_producer/runs/coherence-hardening-medical-procedure-smoke \
  --profiles data/lattice_profiles/lattice_profiles.json \
  --out data/lattice_profiles/proposed/lattice_profiles.medical-procedure-smoke.json \
  --category medical-procedure \
  --offline-only \
  --max-items 20

PYTHONPATH=src .venv/bin/python -u scripts/run_lattice_producer.py \
  --run-dir data/lattice_producer/runs/coherence-hardening-health-condition-smoke \
  --profiles data/lattice_profiles/lattice_profiles.json \
  --out data/lattice_profiles/proposed/lattice_profiles.health-condition-smoke.json \
  --category health-condition \
  --offline-only \
  --max-items 20
```

Do not claim completion unless every command above is run and its exact output reported, per the
producer plan's existing rule.

## Sources

- Producer plan: [2026-07-07-generalization-lattice-producer-agent.md](2026-07-07-generalization-lattice-producer-agent.md).
- Per-level count policy: [offline-k-anonimity-risk-walk.md](../specs/offline-k-anonimity-risk-walk.md).
- Profile schema: [generalization-lattice-cache.md](../specs/generalization-lattice-cache.md).
- Evidence for this plan: this session's review of
  `data/lattice_profiles/proposed/drug-health-procedure.proposed.json`,
  `results/lattice-producer/drug-health-procedure/{accepted,diagnostics,rejected}.jsonl`,
  `scripts/spikes/clean_drug_health_lattice_coherence.py`,
  `scripts/spikes/add_openfda_epc_levels.py`, and the schema/merge changes in
  `src/cloak/lattice/profiles.py` and `scripts/merge_lattice_profiles.py`.
- Raw reference sources confirmed present and parseable this session:
  `data/lattice_sources/raw/drug/openfda_ndc.json.zip` (openFDA NDC Directory, `pharm_class`
  field), `data/lattice_sources/raw/health/doid.obo` (Human Disease Ontology, CC0, 14,735 terms),
  `data/lattice_sources/raw/procedure/icd10pcs_order_2026.zip` (CMS ICD-10-PCS order file,
  80,029 codes).
