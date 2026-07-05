---
type: research
status: current
created: 2026-07-04
updated: 2026-07-04
tags: [privacy, tau, walk-risk, inference-architecture, lattice, k-anonymity, ranker,
       per-type-tau]
companion: ../specs/RL/surrogate-ranker-infiller.md
---

# Removing walk_risk from the inference path while keeping per-type tau

**Question.** The deployed pipeline already carries a detector, a ranker, a planned infiller,
and a planned extractor. The tau mask adds a fifth inference-time model — `walk_risk`
(a Pythia-410m contrastive probe, `src/cloak/probe.py`) plus its distractor-pool artifact —
to score every (span, lattice-level) pair before the ranker may choose. Can risk enforcement
stay a hard, per-type guarantee while the risk *model* leaves the inference path?

**Answer (summary).** Yes. Risk scoring and risk *enforcement* are separable: enforcement at
inference only needs a legality decision per (span, level), and that decision can be a
precomputed, structural property of the lattice level itself. Recommended: **structural
lattice risk** (anonymity-set counting, k-anonymity transplanted to span substitution) as the
destination architecture, with the **infiller-served probe** as the bridge wherever a live
LM score is still needed during the transition. The distilled-risk-head option is analyzed
and rejected as a mask (its guarantee is the wrong kind), retained only as an optional
diagnostic.

## Definitions

- **doc_p / R** — the anonymized document sent to the remote LLM, and the substitution map
  (surface → replacement) used to invert its output.
- **Decision span** — a detected quasi-identifying span carrying a generalization lattice;
  **direct identifiers** are always replaced by chain tokens (`<PERSON_1>`) and are not
  decisions.
- **Lattice / level / depth** — per-span chain of increasingly coarse truthful
  generalizations ("Oslo" → "a Norwegian city" → "a European city" → "a city"); depth
  increases with coarseness. Equivalent to k-anonymity's domain generalization hierarchies
  ([sweeney2002_k_anonymity](../../research-wiki/papers/sweeney2002_k_anonymity.md),
  [DOI 10.1142/S0218488502001648](https://doi.org/10.1142/S0218488502001648)).
- **walk_risk** — contrastive re-identification probe: P(attacker picks the original out of
  {original} ∪ ≤15 same-type distractors | sentence with the fill visible), computed from
  length-normalized causal-LM log-probs (currently Pythia-410m). The spec calls this P4;
  this report uses the code name.
- **fill_proximity** — cos_MiniLM(fill, original), the training reward's privacy term
  (spec name P6). Training-only; not on the inference path.
- **tau / per-type tau (tau_T)** — the per-span risk ceiling: a level with walk_risk ≥ tau is
  illegal. Per-type tau assigns each span type its own ceiling (identity-bearing types
  strict; user-waived attribute types loose or unbounded).
- **legal[s] (the mask)** — the action menu the ranker samples from: levels under the
  ceiling, plus the always-legal generic placeholder. Enforcement, not preference: illegal
  actions have probability zero by construction.
- **Anonymity set** — the set of same-type candidates consistent with a released value;
  k-anonymity's guarantee is |set| ≥ k, giving re-identification probability ≲ 1/k under a
  uniform attacker prior.
- **Distractor pools** — per-type candidate lists (`data/probe_distractors.json`) the
  contrastive probe samples against; a build artifact the current mask depends on.
- **Conformal risk control** — distribution-free calibration wrapper guaranteeing
  E[loss] ≤ α on exchangeable data
  ([angelopoulos2024_conformal_risk_control](../../research-wiki/papers/angelopoulos2024_conformal_risk_control.md),
  [arXiv 2208.02814](https://arxiv.org/abs/2208.02814)).
- **Shootout** — this project's probe-promotion protocol: a candidate risk signal is adopted
  only on measured correlation with real LLM attackers (`docs/specs/attacks.md`).

## 1. Why walk_risk-at-inference is the right thing to attack

1. **Operational surface.** Census on the deployment path: detector (GLiNER fine-tune),
   ranker, infiller (planned), extractor (planned), NLI truthfulness gate, plus the probe LM
   and its pools snapshot. The probe is the only one whose *output feeds a hard guarantee*
   while being (a) a separate 410M LM, (b) dependent on a data artifact (pools), and
   (c) invoked per (span × level) — the widest latency and versioning surface per unit of
   value.
2. **The planned infiller entrenches it.** The spec's constrained-decode loop (sample fill →
   injectivity → tau check → NLI check) puts walk_risk *inside* generation at inference;
   deciding now avoids building on it.
3. **Measured trust deficit.** The probe LM is weak (Pythia-410m; shootout AUC .64/.63 —
   adopted for level-*ordering* .86/.71, not absolute calibration); its per-instance context
   sensitivity is the least-validated part of the signal; and the 2026-07-04 identity-only
   arm run showed the current risk semantics (span recovery) mis-pricing the best measured
   clinical operating point (`results/ranker_reward_gate.json`,
   `results/identity_attack.json`).

Non-negotiable constraints on any replacement: the mask stays **enforced, not learned**
(zero probability for illegal actions on every input, including out-of-distribution);
per-type tau remains expressible; the offline attacker-validation methodology (shootout)
remains the promotion gate; the design must survive the infiller upgrade (generated fills,
not just lattice strings).

## 2. Options

### 2.1 Structural lattice risk (anonymity-set floors) — recommended destination

**Mechanism.** Make risk a precomputed property of the lattice level, not a scored property
of the instance. For each level, attach the size of its anonymity set — how many same-type
candidates are consistent with it — computed offline from the same source that generates the
lattice: gazetteer subtree size for LOC/ORG, ontology subtree for conditions/medications,
interval width for DATETIME ("early 2019" ≈ 90 candidate dates) and QUANTITY ranges.
Per-type tau becomes a count floor `k_T ≈ 1/tau_T`; the inference-time mask is a comparison
of two integers already stored on the lattice node. Two calibration flavors, decided by one
shootout: (a) *structural* — risk = 1/count (population-weighted where priors are skewed),
validated once against the attacker; (b) *LM-calibrated floors* — if 1/count correlates worse
than walk_risk, keep the LM probe **offline** to calibrate per-(type, depth) floors on a
reference corpus. Either flavor yields the same inference architecture: a lookup.

**Rationale.** Deletes the probe LM *and* the distractor pools from inference; zero added
latency; deterministic and reproducible by construction (no more fp16/ROCm nondeterminism in
the guarantee path). The guarantee becomes user-legible ("every kept generalization is
consistent with ≥ 50 candidates of its type"), which is also the project's positioning:
user-specified sensitive types and lattices make per-type floors the natural contract, and
keep-original enters the action space for free as the depth-0 node (count 1 — legal exactly
when the user waives the type). Survives the infiller: legality attaches to the *node* the
infiller is asked to verbalize, so the tau check leaves the decode loop entirely (replaced by
a cheaper surface-form guard, §4).

**Tradeoffs.**
- *Context-blindness at instance level.* A count floor cannot see that this document's
  surrounding text pins the value — the measured "LJM2" famous-context recovery is exactly
  k-anonymity's non-uniform-prior failure. Mitigations: population-weighted counts; the
  evaluation attacker remains the honest meter for the residual (same pre-registered
  escalation logic as fill_proximity's blind spots). Note the context sensitivity being given
  up is the probe's least-validated property, and the 2026-07-04 doc-level attack measured
  the context-scored walk *leaking more* identity than context-blind placeholders.
- *Needs candidate universes per type.* Closed-ish for LOC/ORG/DATETIME/QUANTITY/conditions;
  open-vocabulary MISC fails closed (placeholder) or takes the LM-calibrated-floor flavor.
- *Attribute disclosure is out of scope of the count* (a homogeneous block leaks the
  attribute without re-identification) — in this design that is deliberate: attribute
  sensitivity is the user's type-waiver decision, not a risk-model output.

**Future-proof: high.** Independent of every model on the path (detector swaps, infiller
retrains, and extractor additions cannot move the guarantee); the artifact it depends on
(the lattice) is already load-bearing; it is the only option whose semantics improve as the
tailorability story matures.

### 2.2 Infiller-served probe (model consolidation) — recommended bridge

**Mechanism.** walk_risk is nothing but length-normalized log-probs softmaxed over a
candidate set; any causal LM can serve it. Compute it from the infiller's logits (one batched
extra forward per span) instead of Pythia. Nothing else changes: same contrastive semantics,
same pools, same tau comparison.

**Rationale.** Removes the extra model *family* immediately with zero methodology change; a
4B-class infiller is a strictly stronger probe LM than Pythia-410m, so the probe-bias
objection shrinks rather than grows. One shootout re-validation run promotes it (established
procedure). Natively fits the constrained-decode loop if a live LM score is still wanted
during the transition to §2.1.

**Tradeoffs.**
- *Keeps the pools artifact and the per-(span × level) latency* — the operational surface
  shrinks by one model, not to zero.
- *Coupling hazard (the serious one).* The probe must never run on the live training weights:
  at joint RL training the mask, and through it the reward, becomes a function of the
  optimization variable, and REINFORCE selection pressure alone favors logit drift that
  under-ranks originals in the probe's fixed prompt template — a wireheading channel, no
  gradient through the probe required. Even absent adversarial pressure, ordinary LoRA
  updates recalibrate the risk table every checkpoint, so constant tau stops meaning constant
  realized strictness. Mandatory discipline: the probe runs **adapter-off on the pinned base
  model** (RL touches only the LoRA delta, so the probe's parameters sit outside the
  optimization variable; same weights in memory, two passes); base-model upgrades — never
  LoRA updates — trigger re-shootout and re-gate, the extractor-pinning rule applied here.
  What this does *not* close is correlated-error mining: the trained generator is the frozen
  scorer plus a delta, so its proposals concentrate exactly where the scorer's blind spots
  are; the eval attacker audits accepted fills, and the standing containment burden is why
  this is a bridge, not the destination.

**Future-proof: medium.** Fine through the infiller build; the frozen-snapshot discipline is
a standing tax, and it does nothing for legibility or per-type tailorability.

### 2.3 Distilled risk head with conformal calibration — analyzed, not recommended as mask

**Mechanism.** Train a small frozen regression head (span-in-context features, later the
ranker's frozen-encoder features) to predict walk_risk, supervised on the offline probe
table; wrap its "predict-safe" threshold per type with conformal risk control so that the
expected admitted-violation rate ≤ δ on exchangeable data; mask on the calibrated prediction
at inference.

**Rationale.** Keeps per-instance context sensitivity with microsecond scoring and no LM;
handles arbitrary generated strings (scores any fill); conformal wrapping is the strongest
guarantee available for a learned mask, with finite-sample validity.

**Tradeoffs.**
- *Wrong guarantee type for a floor.* Conformal control is **marginal** (an average over
  draws) under **exchangeability**; a privacy ceiling needs per-instance worst-case on
  arbitrary user documents — precisely the regime (per-user domain shift) where the
  exchangeability assumption is least credible. It also controls the expectation, not the
  tail: individual documents may violate arbitrarily.
- *Student of a distrusted teacher.* It distills Pythia's biases — the original objection —
  into a smaller, less inspectable form, and adds a calibration artifact plus a retraining
  cadence to the version surface.
- *Self-grading risk* if the head ever shares training with the policy (must stay frozen,
  reviewed as a separate deployment — the same tax as §2.2 without its simplicity).

**Future-proof: low-medium as a mask** (its guarantee dilutes exactly as the product reaches
heterogeneous users); **useful as a diagnostic** — a cheap monitor flagging spans where
predicted risk and the structural floor disagree, feeding the attacker-eval queue.

### 2.4 Dismissed briefly

- **Post-assembly verifier audit** (an LM re-reads doc_p before sending): keeps a fifth model
  at inference, adds a reject-and-reassemble loop at the worst latency point, and its verdict
  is again a learned score — dominated by §2.2 on every axis.
- **Legality learned by the policy** (no mask, risk priced in reward): rejected earlier on
  the record — a guarantee cannot be a statistical tendency, and training against one's own
  surrogate invites Goodharting (session analysis of the stage-1 NULL; CLAUDE.md
  empirical-honesty rule).

## 3. Comparison

| | structural lattice risk (§2.1) | infiller-served probe (§2.2) | distilled head + conformal (§2.3) |
|---|---|---|---|
| inference-time risk models | **0** | 0 new (frozen infiller snapshot) | 0 LMs (1 small head) |
| artifacts on guarantee path | lattice w/ counts | pools + pinned snapshot | head + calibration set |
| guarantee semantics | worst-case per span, within anonymity-set model | worst-case per span, within probe validity | **marginal expectation**, exchangeability-bound |
| context sensitivity | none at instance level (priors only) | full (probe's) | learned approximation |
| per-type tau | native (k_T floors, user-legible) | native (tau_T thresholds) | native (per-type δ_T) |
| survives infiller upgrade | yes — legality attaches to the node | yes, with frozen-snapshot discipline | yes |
| latency added | ~0 | per (span × level) LM forwards | ~0 |
| main risk | non-uniform priors (famous contexts) | probe drifts with infiller versions | guarantee voids under domain shift |
| future-proof | **high** | medium | low-medium (as mask) |

## 4. Recommendation and decision cascade

**Adopt structural lattice risk (§2.1) as the destination; use the infiller-served probe
(§2.2) as the bridge and as the upgraded *offline* referee.** Concretely:

1. **Validation first (one shootout, reusing the existing 150-item protocol):** correlate
   1/anonymity-set-count against attacker hit@5 next to walk_risk. If counts match or beat
   the probe on level-ordering, take the structural flavor; if not, take LM-calibrated
   per-(type, depth) floors. Either outcome leaves inference a lookup — the architecture
   decision is robust to the calibration result.
2. **Pythia dies everywhere, not just at inference:** the offline calibration/validation
   referee moves to the infiller-class LM (§2.2's mechanism, applied offline where its
   coupling hazard is harmless).
3. **Per-type tau** ships as count floors `k_T` in user config; identity-bearing types
   (LOC/ORG/DATETIME + direct identifiers) get strict floors, attribute types are
   user-waivable — the policy validated by the identity-only arm measurements.
4. **Keep-original** becomes the lattice's depth-0 node (count 1): automatically legal only
   for waived types. No special-case machinery.
5. **The infiller's decode loop** drops the live tau check: the ranker selects a *node*
   (legality already settled by lookup), the infiller verbalizes it, and a cheap surface
   guard (the fill must instantiate the node: vocabulary/template check + the existing NLI
   truthfulness gate) replaces LM risk scoring at generation time.
6. **Training carries over unchanged:** tau-randomized, tau-conditioned episodes work
   identically with k-floors (sample k_T per episode; feed active floors as policy
   features); walk_risk remains available offline as a feature and as the reward-side
   diagnostic.
7. **Residual channels stay priced where they always were:** the evaluation frontier
   attacker adjudicates famous-context pinning and joint/relational leakage; if it defeats
   the floors, escalation is pre-registered (population-weighted counts, then per-surface
   overrides for high-frequency entities), not a return of the inference-time LM.

**Adopted (2026-07-04).** The decision cascade above executed. The pre-registered shootout
(step 1, 150 items, gemini referee, `results/lattice_count_shootout.json`) had the **structural
flavor win outright** — inv_aset level-ordering **0.786** vs walk_risk **0.714** at matched
labels (adoption rule was "structural if within 0.05"; it beat, not tied), so no LM-calibrated
floors were needed. `K_FLOORS` calibrated to a measured-moderate strictness
(LOC/ORG/DATETIME/DEM/QUANTITY = 100, MISC/OTHER = 1, user-waivable). The migration is done:
`cloak.anonymity.aset_count` + integer floor comparison is the mask on the inference and
training paths, walk_risk is offline-only (`src/cloak/probe.py`), and the RL spec was rewritten
around it — [structural-lattice-risk plan](../plans/2026-07-04-structural-lattice-risk.md),
`docs/specs/RL/surrogate-ranker-infiller.md`.

## Sources

- [sweeney2002_k_anonymity](../../research-wiki/papers/sweeney2002_k_anonymity.md)
  ([DOI 10.1142/S0218488502001648](https://doi.org/10.1142/S0218488502001648)) — structural
  option's foundation and failure modes.
- [angelopoulos2024_conformal_risk_control](../../research-wiki/papers/angelopoulos2024_conformal_risk_control.md)
  ([arXiv 2208.02814](https://arxiv.org/abs/2208.02814)) — the strongest available guarantee
  for a learned mask, and why it is insufficient here.
- `docs/specs/RL/surrogate-ranker-infiller.md` — current mask/probe normative spec (tau
  ceiling, probe shootout, constrained decode loop).
- `results/ranker_reward_gate.json`, `results/identity_attack.json`,
  `scripts/spikes/identity_attack.py` (2026-07-04) — identity-only arm measurements cited in
  §1 and §4.
- `research-wiki/training/2026-07-04-RL-ranker-v1-stage1-bandit.md` — the stage-1 NULL whose
  diagnosis motivated this analysis.
