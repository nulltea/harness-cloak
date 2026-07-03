---
type: plan
status: current
created: 2026-07-03
updated: 2026-07-03
tags: [rl, surrogate-reward, gaps, extractor, echo-survival, candidate-sensitive-risk,
       injectivity, fact-recall, pre-rl-blockers, plan]
companion: [../specs/RL/surrogate-ranker-infiller.md, 2026-07-02-surrogate-grpo-training.md,
            ../specs/benchmarks.md, ../specs/attacks.md]
---

# Surrogate-RL gap analysis and pre-RL fixes

Companion to the RL spec ([surrogate-ranker-infiller.md](../specs/RL/surrogate-ranker-infiller.md)).
Records the 2026-07-03 component audit: what is broken in the environment/reward stack, which of
it blocks RL, and the ordered fix plan. Core conclusion: **as wired today, RL would be
utility-only training on a utility term blind to the dominant realized loss channel** — the gate
re-pass validated the ground truth, not the reward's gradient direction inside the feasible set.

## Definitions

- **Echo / echo factor** — whether the remote model reproduces a fill (verbatim or loosely) in
  out_p; the third factor of realized fact survival:
  `readable(doc_p) × invertible(R) × echoed(out_p)`.
- **Absorption** — the remote model incorporates a fill's information without any surface trace
  in out_p ("some time ago" → the note omits temporals entirely). Unrecoverable by any extractor.
- **Coincidence echo** — out_p contains a generic fill phrase ("an organization") produced
  independently by the model, not as an echo; inverting it splices a false fact into out_final.
- **Candidate-sensitive vs candidate-invariant risk** — whether the attack probe's score depends
  on the chosen replacement (attacker sees it) or only on the surrounding context (replacement
  masked away).
- **ŝ (echo-survival table)** — offline-measured `P(fill recoverable in out_p by the deployed
  extractor | fill mode, span type, task)`; a disclosed environment constant, not a tuning knob.
- Other terms (u_qa, fact recall, R, fill modes, E0/E1/E2): see the
  [RL spec definitions](../specs/RL/surrogate-ranker-infiller.md#definitions).

## Gap analysis (measured 2026-07-03)

### Gap 1 — privacy term A: currently null as a training signal

`cloak/probe.guess_back_risk` masks the replacement before probing, so its risk is
**candidate-invariant** — a slot property, not an action property (confirmed: every "Commission"
lattice level scores exactly 0.1466). Consequences:

1. **The τ-walk is degenerate**: accept-most-specific-if-slot-safe, else floor. The lattice
   ordering does nothing in between — the deep root cause of the measured τ-flatness
   (`at_most_specific` 583/586 on clinical).
2. **Zero privacy gradient in RL**: per-span risks don't depend on sampled actions → A is
   identical across all G rollouts → the group advantage sees utility only, for any α.
3. **Minted/descriptive placeholder labels and relational fills cannot be priced at all**
   (masking hides the label) — the E1 action spectrum is gated on fixing this.

A candidate-sensitive variant already exists — `mlm_guess_back` in
`scripts/latticecloak_tau_sweep.py` (masks the slot, keeps the replacement as an appositive) —
but needs promotion to `cloak/probe.py` and repair: single-token top-50 scoring structurally
underestimates multi-token secrets; the target filter is broken for hyphenated forms
("50-year-old" → targets {"year","old"}, not the 50); roberta-base is a weak attacker (the eval
attacker remains the honest measure); mean aggregation dilutes single catastrophic leaks.

Bonus measured defect, fixed by the same batch as injectivity: the walk's floor fallback ships
replacements that failed τ at every level ("Commission" shipped at risk 0.147 vs τ = 0.02).

### Gap 2 — utility term u_qa: blind to the echo factor

u_qa scores `readable × invertible` only. Inside the reward, `invert()` always runs its trivial
exact path (reader answers are doc_p substrings, where replacements are verbatim by
construction) — the hard inversion case (paraphrased echo in generated text) never occurs in
training, so **extractor upgrades change eval, not the reward**. u_qa is a valid
necessary-condition filter (approximately an upper bound on realized survival; it correctly
zeroes destructive candidates — that is what the gate validated), but it is **indifferent
along the anchor-vs-naturalistic axis where realized survival differs ~20×** (placeholders echo
9/9; naturalistic fills ~5%, `gen_absent` ≈ 95%).

Combined with Gap 1: total reward ≈ flat over the feasible action set → RL clones the τ-walk
init and learns nothing beyond avoiding self-destruction.

Unmeasured bias risk: the SQuAD reader on placeholder-laden doc_p may *penalize* placeholders —
the opposite of their realized advantage (smoke test M3 below).

### Gap 3 — extractor (E0): ground-truth integrity and headroom unknowns

- **Coincidence echo** corrupts out_final with false facts and **inflates fact recall** (a probe
  becomes spuriously answerable) — it contaminates the ground truth itself. Rate unmeasured (M2).
- The ambiguity branch (`extract.py:44-47`) pairs colliding replacements to occurrences in
  document order — arbitrary; removed for free by injectivity.
- Placeholder inversion is exact-literal; reformatted placeholders (`<PERSON 1>`, "Person 1")
  silently leak, and `ph_residue` counts only `PERSON|CODE` — wrong once typed fallbacks
  (`<DATETIME_n>`) land.
- **E1 headroom unknown**: `gen_absent` ≈ 95% conflates loosely-echoed (recoverable by better
  alignment) with absorbed (unrecoverable). This split (M1) gates the whole fork below.

### Other component gaps (context, not RL blockers)

- **Lattice shallowness**: most spans carry 1–3 levels (dates often exactly 1) → the E0 ranker
  has little to select among; the real E0 action axis is level-vs-placeholder.
- Detection recall (DEM 0.56, QUANTITY 0.25) caps the privacy ceiling and probe supply — known,
  reported per project rules.
- Probe supply: 3.19/1.0/0.38 per doc (clinical/enron/aeslc); probe-less docs are excluded from
  training; aeslc's thinness is corpus reality.
- Gate power: n = 5–12 probe-bearing docs per corpus; arms are coarse.

## The fork — how the echo factor enters the reward

Realized survival = `readable × invertible × echoed`. u_qa covers the first two; the fork is
about the third. **(a) and (b) are complements, not alternatives**; the decision is the mix.

**Route (a) — extractor upgrade (change the world, not the reward).** Make the deployed
extractor recover loose/paraphrased echoes so the echo factor flattens across fill modes and
u_qa's two factors become approximately sufficient. Ladder: E1-fuzzy (threshold ~70 **plus a
verification gate** — type compatibility / NLI on the aligned window; unverified threshold drops
make coincidence echo epidemic) → E1-semantic (MiniLM window alignment, injective assignment —
the `extract.py` ponytail upgrade path) → E2-learned reconstructor (highest ceiling; own
pathologies: hallucination inflates fact recall, parallel-data remote binding, memorization leak
channel). **Hard ceiling: absorption** — no extractor recovers a fact with no surface trace.
Residual risk: post-E1 survival may still be mode-dependent (a smaller version of the same
blindness).

**Route (b) — echo-survival prior ŝ (put the factor into the reward).** Measure ŝ(mode, type,
task) offline via a few hundred cached round trips under the deployed extractor; reward becomes
`u_expected = mean_j ŝ(mode_j, type_j, task) · f1_j`. The policy can then trade echo-ability vs
naturalness vs risk. Tensions: partial remote-model re-binding (mitigated by table coarseness;
sensitivity-checked under the second eval model); honesty boundary — ŝ is a pre-registered
measured constant, re-measured only on extractor/reward changes, never re-tuned on disappointing
results, each change re-gates; mode-level coarseness gives the stage-2 infiller no gradient for
fill-level echo-craft (that needs per-candidate feedback = round-trip reward, the documented
upgrade); distribution drift between measurement fills and policy fills (periodic re-measure;
large drift = overoptimization warning).

**Composition**: ŝ is extractor-relative, so (b) is only defined after the extractor version is
fixed; every (a) upgrade compresses the mode gap (b) carries. Pure-(a) is unreachable
(absorption exists); pure-(b) forever wastes recoverable utility. M1 sizes both sides before
either is built.

## Fix plan — ordered

**Phase M — measurements that gate decisions (local/cached, ~a day total).**

- **M1. `gen_absent` decomposition** (gates the fork): for every absent fill in the cached
  constructed-arms out_p, best fuzzy-alignment score + embedding max-similarity against out_p →
  histogram loose-echo vs absorbed. Output: E1 headroom %, residual mode gap for ŝ.
- **M2. Coincidence-echo false-positive rate** (ground-truth integrity): null control — invert
  each out_p against a *mismatched* document's R; every inversion that fires is a false
  positive. If material, fact recall needs a verification gate before it is trusted further.
- **M3. Reader-on-placeholder bias**: u_qa on placeholder-only vs naturalistic-only arms of the
  same docs (local). If the reader tanks on placeholders, u_qa carries an anti-placeholder bias
  to correct (probe phrasing or reader swap) before any reward is trusted.

**Phase 1 — candidate-sensitive risk probe (the RL unblocker, ~1–2 days).**
Promote appositive-style guess-back into `cloak/probe.py` as the single risk function (gate +
reward A): attacker sees the replacement, slot masked alongside it. Fix multi-token target
scoring (score against all content tokens incl. numerics; aggregate per-span, not
single-token-or-nothing), fix hyphenated-form targets. Aggregation: report mean and max; reward
uses mean, max is the diagnostic. Acceptance: per-span risk varies across lattice levels of the
same span (kills the 0.1466-everywhere degeneracy); τ-walk selects intermediate levels on a
measurable fraction of spans; re-run the τ sweep to observe (for the first time) a real τ axis.

**Phase 2 — injectivity + placeholder fallback + extractor hygiene (~half a day, one diff).**
Used-set constraint in the walk; generic typed placeholder on exhaustion (also closes the
measured τ-floor violation); extend `ph_residue`/inversion to all `<TYPE_n>`; drop the
order-matching ambiguity branch (dead after injectivity). Acceptance: self-check asserts
injective R; no over-τ replacement shipped; typed placeholders invert.

**Phase 3 — E1 extractor, scoped by M1 (~2–4 days).**
Semantic-window aligner with verification gate; FP rate re-measured (M2 harness) after. If M1
shows absorption dominates (>~70% of gen_absent), descope to E1-fuzzy+verify and shift weight to
Phase 4.

**Phase 4 — echo-survival table ŝ + reward composition (~1 day, cached remote calls).**
Measure ŝ under the Phase-3 extractor across fill modes × types × tasks (few hundred docs,
cached); pre-register the table; reward = ŝ-discounted u_qa.

**Phase 5 — re-gate, then RL.**
Re-run the constructed-arms gate on the new (reward, environment) pair — Phases 1–4 change both
sides. Go criterion unchanged (positive per-doc arm agreement where the ground truth is sane) +
new requirement: reward must separate placeholder-mode from naturalistic-mode arms in the
direction realized fact recall does. Then stage-1 ranker RL per the
[training plan](2026-07-02-surrogate-grpo-training.md).

**Deferred (tracked, not blockers):** lattice deepening (only if Phase-1's real τ axis shows
selection needs more levels); probe-supply expansion (number normalization, teacher fact
tuples); document-level attack head (E2, pre-registered escalation); spec §2.3-invariant-3 /
§2.2 corrections land with Phase 1–2 diffs.

## Sources

RL spec: [surrogate-ranker-infiller.md](../specs/RL/surrogate-ranker-infiller.md). Gate results
and ground-truth root cause: [2026-07-02-surrogate-grpo-training.md](2026-07-02-surrogate-grpo-training.md)
(STATUS 2026-07-03), `results/surrogate_validation.json`. Anti-extractor deletion lesson:
[NaPaRe](../../research-wiki/papers/huang2025_tree_search_rewriting.md)
([arXiv 2509.20838](https://arxiv.org/abs/2509.20838)). Overoptimization playbook:
[Gao et al. 2022](../../research-wiki/papers/gao2022_reward_overoptimization.md)
([arXiv 2210.10760](https://arxiv.org/abs/2210.10760)) via
[adverserial-RL.md](../research/adverserial-RL.md).
