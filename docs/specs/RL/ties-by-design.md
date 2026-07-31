---
type: reference
status: current
created: 2026-07-31
updated: 2026-07-31
tags: [rl, ranker, ties, reward-design, counterfactual, behavior-cloning, exit, generalization, deployment, lexicographic, spec]
companion: [docs/specs/RL/interactive-ranker-v2.md,
            docs/specs/RL/interactive-ranker-v2-decision-log.md,
            docs/specs/RL/interactive-ranker-v2-diagnostics.md,
            docs/research/reward-ties-and-controller-authority.md]
---

# Ties-by-design — reward-tie semantics, gradient flow, and the deployment-generalization gap

This document is the normative reference for how the interactive ranker's reward ties work: why they are deliberate, which training channels are silent on them and which are not, what the pre-training stages (behavior cloning and expert iteration) contribute to tied-action margins, exactly which learned quantities carry to held-out documents and which do not, and the design fork this opens with its candidate solutions. The tie-ownership fork entries in the [decision log](interactive-ranker-v2-decision-log.md) record the chronological adjudication; this document consolidates the settled understanding.

## Definitions

- **Tie (reward tie / utility tie)**: two actions of the same decision whose document utilities are equal — both preserve every QA-probed answer, so the reader scores them identically. **Exact tie**: measured |ΔU| ≤ 1e-9 (`TIE_EXACT_ATOL`). **Sub-noise pair**: 0 < |ΔU| ≤ 0.044, the independently measured reader-resolution floor (`TIE_EXIT_BOUND`).
- **u / tower**: the λ-blind utility logits produced by the policy's utility head ("the tower"). **p̂ / count score**: the frozen per-action profile count score (privacy preference). **Controller**: the additive combination z(a) = u(a) + α·g(λ)·p̂(a) that orders actions at positive λ.
- **Counterfactual probe**: a scored pair (selected vector, single-decision-flipped vector) whose utility difference ΔU is measured by the reader; the per-pair policy loss is −ΔU·(q−0.5).
- **Evidence ledger**: the per-(doc, decision, action-pair) record of measured ΔU values keyed by surrounding-context hash; the source of verified-tie labels (≥3 distinct contexts at |ΔU| ≤ 1e-9, permanently disqualified by any |ΔU| > 0.044).
- **BC (behavior cloning)**: supervised warm start on the deterministic teacher walk. **ExIt (expert iteration)**: post-BC cloning of sampled trajectories that are strict, reverified pure-utility improvements over the BC reference.
- **Deployment-generalization gap**: the difference between tie-breaking behavior on training documents (where the evidence ledger and projection constraints exist) and on held-out documents (where they do not).

## 1. Ties-by-design

The lattice and QA artifact were deliberately built so that multiple generalization levels of the same span can all preserve the QA-probed answers. When they do, the reader assigns them identical utility 1 and the reward is silent between them; only the count-based privacy preference p̂ distinguishes them. This is a design decision, not an artifact defect, and it is correct for two reasons:

- **It reflects true task semantics.** Levels that preserve every task-relevant answer *are* utility-equivalent for the downstream consumer of `out_final`. Utility measures task preservation; privacy preference measures generalization depth. These are different axes and the reward keeps them separate.
- **It protects the decomposition.** Grading partial utility credit by generalization depth would inject the count preference into u: the λ-blind tower would learn a privacy ordering, which is wrong by construction (at λ=0 tied actions must *not* be privacy-ordered), the count signal would be double-counted at positive λ, λ semantics would distort, and matched-realized-privacy comparisons across methods would be corrupted.

One qualification bounds the claim: passing the current QA probes means *equivalent for the measured task artifact*, not universally semantically equivalent. False equivalence from QA incompleteness is the residual risk, and richer partial credit is legitimate only if it measures actual answer degradation — never lattice depth as a proxy.

Prevalence (full-corpus tie-structure census, 2026-07-31, `scripts/spikes/tie_structure_census.py`): 67 documents with policy decisions; the tiny stratum (≤6 decisions) is 22 documents (33%); of decisions with measured ΔU spans corpus-wide, 18% are exactly flat and a further 21% are sub-noise — so roughly 39% of measured decisions sit at or below the reader floor. Ties are not a corner case; they are a first-class regime.

## 2. Role of counterfactuals — sensor, not tie-breaker

On an exactly tied pair the tower receives **no utility-ordering gradient from any channel**: leave-one-out advantages between tied rollouts are zero, the counterfactual pair loss −ΔU·(q−0.5) is identically zero at ΔU=0, and the count score is detached from u by construction. The precise statement is stronger than "no gradient": the tied margin has no utility-semantic force controlling it, while *other* gradients — other documents sharing features, KL and entropy regularizers, the sensitivity regularizer — remain free to move it. That is the mechanism by which cross-document drift fills the silence (link 2 of the measured causal chain in the [failure-taxonomy report](../../research/reward-ties-and-controller-authority.md)) and why tied margins churn arbitrarily unless something else owns them; cf. the policy-churn phenomenon ([Schaul et al. 2022](../../../research-wiki/papers/schaul2022_policy_churn.md), [arXiv 2206.00730](https://arxiv.org/abs/2206.00730)) and reward underspecification ([D'Amour et al. 2020](../../../research-wiki/papers/damour2020_underspecification_ml.md), [arXiv 2011.03395](https://arxiv.org/abs/2011.03395)).

What counterfactuals *do* contribute is exactly two things:

1. **Rescue dead groups.** When all sampled rollouts of a group tie, LOO advantages vanish for every decision; counterfactual pairs restore gradient on the decisions that do have utility differences. This is per-decision causal credit, and it is the reason small documents train at all.
2. **Measure tie structure.** Every probe emits an evidence row (doc, decision, pair, ΔU, surrounding-context hash) into the evidence ledger. Verified-tie labels distilled from the ledger drive the tie-ownership machinery: the controller-only hinge at max λ and the cycle-boundary projection of α (the evidence-supervised tie-ownership experiment, RL-ranker v14, validated that the policy then tracks the hard lexicographic oracle on training documents).

So counterfactuals convert reward silence from an ambient failure into an explicit, labeled fact — but the zero itself never orders anything. Zero multiplied into a policy-gradient loss is zero. Ordering among ties is deliberately the controller's job, because the desired order is λ-dependent and the tower is λ-blind. The missing operation, and the subject of the fork in section 6, is making the *measured equivalence* transferable across documents rather than a per-document ledger entry.

The exact-tie regime also has a favorable theoretical property: in lexicographic multi-objective RL, optimizing a secondary objective over the primary objective's *exact* argmax set costs the primary objective nothing ([Skalse et al. 2022](../../../research-wiki/papers/skalse2022_lexicographic_morl.md), [arXiv 2212.13769](https://arxiv.org/abs/2212.13769)); the hazard is scalarized value functions blurring near-ties into interference ([Vamplew et al. 2024](../../../research-wiki/papers/vamplew2024_value_function_interference.md), [arXiv 2402.06266](https://arxiv.org/abs/2402.06266)) — which is precisely what arbitrary tower margins racing a bounded additive shift produce.

## 3. The coarse side of the boundary — the utility cliff IS learned

Reward silence holds only *inside* the equivalence set. RL does expose the policy to levels coarser than the utility-preserving ones, and there the tower receives genuine ordering gradient through both channels:

- **Structural-neighbor probes.** The counterfactual scheduler (`_structural_alternatives`) probes the levels *adjacent* to the selected one plus the keep and placeholder endpoints. When the policy sits at the coarsest utility-preserving level, the adjacent-coarser probe measures the utility drop directly and the pair loss −ΔU·(q−0.5) is nonzero — a "one step further is bad" gradient at exactly the cliff edge. The placeholder endpoint measures the maximal-generalization cost regardless of where the policy sits.
- **On-policy sampling at high λ.** The controller shift drags sampling toward coarser actions; trajectories that cross the boundary and break QA answers lose document utility and are punished by LOO advantages. This is visible in every run as U(λmax) sitting systematically below U(λ0).

Measured (evidence-supervised tie-ownership screening, RL-ranker v14, cycle-projected arm): of 498 counterfactual probes over the run, 57% were live pairs (max |ΔU| 0.68) and 43% exact zeros — the majority of probe gradient is boundary/live supervision, not silence.

Three caveats bound the claim. Exposure is progressive and λ-driven, not exhaustive: probes walk outward from the current policy, so right after BC the adjacent-coarser probe lands deep inside the equivalence set, and the cliff only gets probed once the controller has pushed the policy to its edge — which is where knowing it matters. "Utility 0" overstates the penalty: document utility aggregates assertion scores, so a too-coarse level costs only its linked assertions (fractional ΔU, not collapse). And zero-linked decisions (~13% of the corpus) can never teach coarse-is-bad — even the placeholder is utility-free per the measured reward there, which is consistent with ties-by-design, not a defect.

The net division of knowledge: the tower learns **where utility ends** (the cliff, from above) but not **what is equivalent** (inside the set, margins stay arbitrary). The fork in section 6 targets only the inside of the set; the outside is already supervised.

## 4. Role of pre-training — BC and ExIt install the initial tie margins

Pre-training is where tied actions get their *initial* relative margins, and neither stage orders ties by utility — by construction:

- **BC** clones the deterministic teacher walk (`behavior_clone_trajectory`): at each decision it selects the level action with the minimum authored level index (the most specific, least generalized level), or the single placeholder when no level is legal. The teacher is λ-independent and support-preserving. Consequence for ties: BC installs a *canonical* margin pattern among utility-equivalent actions — the most specific level starts as the argmax everywhere. This is intentional at λ=0 (identity behavior, minimal transformation) but it means the initial tied margins encode authored order, not measured utility, and they are anti-correlated with the count preference (more specific ⇒ lower anonymity count).
- **ExIt** (`collect_exit_winners`) samples rollouts at λ=0 and clones only strict, serially reverified *pure-utility improvements* over the BC reference. A tied alternative can never qualify — strict improvement excludes ΔU=0 by definition — so ExIt refines the tower on live utility differences while leaving tied margins exactly where BC put them.

The net effect: pre-training hands RL a tower whose tied margins are arbitrary-but-BC-anchored (canonical keep/most-specific preference), and RL's utility channels then have no force holding them there (section 2). The subsequent drift is what the softcap bounds in scale and the tie-ownership machinery overrides on training documents. One implementation consequence is pinned in code: the controller gain head is created *after* BC, so gain parameters are absent from BC checkpoints and keep their init values on import (`_import_unconditioned_state` explicitly tolerates the missing keys).

## 5. The deployment-generalization gap

The tie-ownership result on training documents does not carry to held-out documents, because the quantities that own ties are per-document. The inventory:

**What trains and generalizes (available on an unseen document):**

- The tower's feature→utility mapping — ordering on *live* (non-tied) pairs, learned from LOO advantages, counterfactual pair losses, BC, and ExIt.
- The count score p̂ — computed directly from the document's lattice and the frozen profile targets; no learning involved, always available.
- The global α — calibrated at warm start and raised by cycle projection; a single scalar, document-independent.
- The gain head's feature→gain mapping — technically generalizes, but as of the learned-gain screening (RL-ranker v13) it is degenerate/uniform: no per-decision differentiation was ever learned, so it currently adds nothing beyond global α.

**What does not generalize (empty on an unseen document):**

- The evidence ledger — per-(doc, decision, pair) measurements; inference performs no probes.
- Verified-tie labels and the hinge/projection constraints distilled from them.
- The hard lexicographic oracle — defined only over qualified labels.
- Any per-document notion of *which* margins are tie margins versus live margins.

Consequence: on a held-out document, high-λ tie-breaking degenerates to the calibrated global α racing **arbitrary interpolated tower margins** — BC residue plus training-time drift, interpolated through shared features onto pairs no evidence ever touched. That is the original coin flip, reintroduced at deployment where it cannot be measured by the ledger. Its severity has a sharp and asymmetric profile:

- **Utility cost: zero.** Tied actions are utility-equivalent by definition, so arbitrary tie-breaking never damages `out_final`. Standard utility regret is blind to the defect — held-out utility can look perfect while λ-controllability fails.
- **Privacy cost: material and unreproducible.** With ~39% of measured decisions at or below the reader floor and 33% of documents in the tiny stratum, arbitrary ownership makes realized privacy seed- and checkpoint-dependent, breaks λ monotonicity expectations, and invalidates operating-point calibration — the project's headline comparison (privacy vs utility at matched realized privacy) rests exactly on realized privacy being a stable function of λ.

Accepting this is therefore not viable for the product (see option v below); held-out evaluation must include tie-specific metrics that utility regret cannot see: greedy privacy among utility-equivalent actions, tie-oracle agreement where evidence exists, λ monotonicity, and cross-checkpoint tie-choice stability.

## 6. The design fork — making tie ownership transferable

The fork: **tie ownership is solved on training documents by per-document evidence; what mechanism carries it to documents with no evidence?** Candidate solutions, ranked; the decision-log fork entry preregisters the adjudicating experiment.

### (i) Counterfactual utility-equivalence critic — recommended

Add a small auxiliary head q_U(s, a) beside the actor's utility head (same features, no count score, no λ), trained by Huber regression on *every* evidence row: predicted pairwise difference q_U(s, a_i) − q_U(s, a_j) regressed onto the measured ΔU. Exact ties supervise a difference of 0, sub-noise pairs supervise their measured small value, live pairs supervise sign and magnitude — the zero evidence that policy-gradient losses discard becomes supervised signal. Because q_U is in utility units, the independently measured 0.044 reader floor is the equivalence threshold; no new tuned knob. At inference: λ=0 uses raw actor logits (exact identity preserved); at λ>0, actions whose q_U lies within 0.044 of the menu maximum form the predicted equivalence set and their actor logits are canonicalized to a common value (distance-from-max set construction, not pairwise, so intransitivity cannot bite), after which the existing count controller orders inside the set deterministically.

Justification: it separates "what is utility-equivalent" (a calibrated, supervised, transferable prediction) from "what does the policy prefer" (actor logits, which are policy parameters and not calibrated utility estimates — the core defect of option ii). It generalizes through features instead of the ledger, uses all evidence strata, and turns held-out tie behavior into a measurable supervised property (precision/recall) rather than an emergent RL property. Tradeoffs: a new head and inference-time set construction (medium-high implementation cost); trains on the same sparse evidence, so held-out interpolation is the open empirical question; the ~39% near-zero population pushes the critic toward predicting zero everywhere unless batches are macro-balanced across exact-tie / sub-noise / live strata; false equivalence costs real utility, so inference must be conservative (canonicalize only when confidently inside ±0.044, abstain to the ordinary controller otherwise). Stack composition: keep softcap, no-gap controller, and sensitivity regularizer (retarget after canonicalization); freeze the gain residual at zero for the first spike so the mechanism is identifiable; keep cycle projection as the oracle/comparison arm.

### (ii) Tie-equality regularization on the tower — ablation arm only

Pull u(a_i) == u(a_j) toward zero margin for verified-tied pairs, so the controller's bounded shift wins wherever margins are near zero. Cheap (low-medium cost) and it attacks drift at the root, but it has three structural defects: actor logits are uncalibrated, so a small margin does not reliably mean a small ΔU and any deploy-time threshold τ over logit distances is a new pre-baked knob over meaningless units; equalized logits soften λ=0 (a mixed policy among tied actions where BC deliberately encodes the most specific level as canonical); and flattening tied pairs can flatten nearby live pairs through shared features. Retained as an ablation to quantify how much of the critic's benefit is available for free.

### (iii) Evidence-supervised gain-head generalization — comparison arm

Supervise the gain head's feature→gain map from ledger constraints (the bootstrap now yields dozens at epoch 0, versus the 2 constraints under which the learned-gain screening degenerated). Mostly implemented already, so cheap to run — but conceptually weaker: its target ("how much authority beats the current actor margin") is checkpoint-dependent, nonstationary, and entangled with the softcap and global α; it compensates for arbitrary margins rather than modeling equivalence. Run as a comparison arm against the critic, not the default.

### (iv) Inference-time counterfactual probing — audit/eval oracle only

Build a local ledger at deployment by probing the unseen document. Most accurate per-document answer, but the ordinary inference path cannot afford it: each probe is a full scored round trip, cost scales with tie density (~40% of decisions in the tiny stratum), and — decisive — sending multiple differently generalized variants of the same document to the remote provider *increases* privacy exposure through cross-query composition, working against the layer's whole purpose. Retained for audits, difficult-document fallback, offline improvement, and as the held-out evaluation oracle.

### (v) Accept and document — rejected

Held-out ties break arbitrarily but utility-free, so this is superficially defensible. Rejected because arbitrary ownership makes realized privacy non-reproducible and λ semantics unreliable, which invalidates the matched-realized-privacy comparison the project is built on. Acceptable only as a documented limitation of intermediate prototypes.

### Adjudication frame

Document-held-out split with the evidence ledger completely unavailable at evaluation (unseen *documents*, not merely unseen pairs from seen documents). Primary gates: false-equivalence precision ≥ 95% (predicted ties measure |ΔU| ≤ 0.044; precision is never traded for recall — report recall, don't optimize it), held-out tie-oracle agreement ≥ 90% on verified ties, identical λ-ordered tie choices across three consecutive checkpoints, exact λ=0 identity with the actor-only distribution, and greedy ΔP(λmax − λ0) ≥ 0.20 on tiny held-out documents. Behavioral gates: λ=0 utility non-inferiority within 0.044 and no held-out utility regression from false canonicalization. Sampled-separation metrics remain report-only.
