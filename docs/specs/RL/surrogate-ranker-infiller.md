---
type: reference
status: current
created: 2026-07-03
updated: 2026-07-03
tags: [rl, grpo, surrogate-reward, ranker, infiller, environment, probes, reward, injectivity,
       relational-fills, fact-recall, spec]
companion: [docs/plans/2026-07-02-surrogate-grpo-training.md, docs/specs/benchmarks.md,
            docs/specs/attacks.md]
---

# RL specification — surrogate-reward training of the substitutor (ranker + infiller)

Living spec. Defines the **environment → probes → reward** stack for training the substitutor
cascade with a local surrogate reward, structured for review and iteration. Decision history and
wall-time live in the plan
([2026-07-02-surrogate-grpo-training.md](../../plans/2026-07-02-surrogate-grpo-training.md));
this doc is the normative statement of *what the RL system is*.

## Definitions

- **doc_orig / doc_p / out_p / out_final** — original document; anonymized rewrite sent to the
  remote LLM; the remote output; the locally re-identified output returned to the user.
- **R (substitution record)** — client-side list of `{surface, replacement, type, action}`
  entries; the only bridge from doc_p-space back to original-space.
- **Ranker** — policy choosing, per detected quasi-identifier span, an abstraction level (or
  placeholder). Stage-1 trainable component.
- **Infiller** — generative model rendering the chosen level as a concrete replacement string in
  context (planned flan-t5-base + LoRA; not yet implemented). Stage-2 trainable component.
- **Extractor** — the local component mapping out_p → out_final by restoring R surfaces.
- **Injectivity (of R)** — per document, no two distinct surfaces share a replacement string
  (canonical form). Violation = information destroyed before the remote call, unrecoverable by
  any extractor.
- **Relational fill** — a replacement whose meaning is relative to another fill ("two days
  later", "at the following meeting"). Preserves cross-fact structure; creates joint leakage.
- **Restated-span probe** — a (surface, natural question) pair for an R span whose original
  surface the gold output restates; the unit of both the utility reward and the realized
  ground truth.
- **u_qa** — surrogate utility: extractive-reader answerability of probes from doc_p, scored
  after inversion through R (the "mini round trip").
- **Fact recall** — realized utility ground truth: the same probes answered from **out_final**
  (original space; no inversion), token-F1 vs the original surface.
- **A(doc_p)** — attack head: document-level privacy score in [0,1] (higher = more re-identifiable).
- **MTI guess-back** — masked-token-inference probe: can a masked LM recover the original surface
  given the replacement in context; the v0 per-span attack signal and the τ-walk's risk gate.
- **τ / α** — τ: inference-time per-span risk ceiling of the rule walk (baseline knob);
  α: training-time privacy weight in the scalar reward (this method's knob; one α = one
  operating point).
- **GRPO / group advantage** — G rollouts per document; advantage = per-group standardized
  reward; REINFORCE/PPO-clip update.
- **E0/E1/E2** — environment versions (below). A trained policy is valid only for the
  environment version it was trained in.

## 1. Objective

Train the substitutor (ranker, then ranker+infiller) to maximize round-trip task utility at a
target privacy level, using **no remote calls during training**:

```
r(doc_p, R) = α · (1 − A(doc_p)) + (1 − α) · U(doc_p, R)
```

Evaluation is always the **real round trip** (remote model on held-out docs), scored by **fact
recall on out_final** (headline utility) and the frontier-LLM re-identification attacker on
doc_p and out_final (privacy). The surrogate-vs-realized gap is a first-class result, not noise.

**Measured motivation (2026-07-03 gate re-run, `results/surrogate_validation.json`):** the
surrogate's utility term ranks constructed quality arms in agreement with realized fact recall —
per-doc Spearman 0.37 (clinical, n=12), 0.44 (aeslc, n=5), 0.775 (enron, n=8) — on the same run
where reference-overlap metrics (ROUGE-L/BERTScore) were noise (ρ ≈ 0, insane arm orderings).
The realized numbers also expose the true baseline: the current rule pipeline delivers almost no
probed facts to out_final (tau-walk fact recall 0.012–0.038 vs no-privacy 0.13–0.18), because
the remote model paraphrases naturalistic fills (`gen_absent` ≈ 95%) while placeholders survive
and invert (9/9, zero residue). Closing that gap is what the policy is being trained to do.

## 2. Environment

### 2.1 Episode structure

One episode = one document (contextual bandit; no multi-step dynamics in v0).

```
state      s = (doc_orig, detected spans with types/coref chains, task family)
action     a = per-span choice:  level ℓ ∈ lattice(span)  ∪  {PLACEHOLDER}
render     doc_p, R = assemble(doc_orig, a)        # infiller renders fills in E1+
reward     r(doc_p, R)                             # §4, fully local
```

- **Stage 1 (ranker bandit):** G level-assignments sampled from π_θ(ℓ | span, doc, task);
  fills rendered by the frozen infiller (E0: the lattice's literal level string). REINFORCE on
  group advantage, KL leash to the SFT/behavior-cloned init.
- **Stage 2 (joint GRPO):** additionally sample fill strings y ~ p_φ(· | doc, span, ℓ); shared
  scalar advantage on ranker log-probs + infiller token log-probs (PPO-clipped), LoRA update
  of φ.

### 2.2 Action space (per detected span)

- Quasi-identifier types: the span's abstraction levels (most-specific-first) **plus an explicit
  PLACEHOLDER action**. Placeholder is a legitimate policy choice, not only a fallback: it is
  the substitution mode with measured round-trip survival, and pricing it lets the policy trade
  naturalness vs survival vs risk per span (e.g. dates in a timeline may be worth relational
  fills; a one-off case number is not).
- **Placeholder has two modes with different privacy contracts:**
  - *Generic typed* (`<DATETIME_1>`; label from the detector's fixed type vocabulary): the label
    depends only on the detected type, never on the secret — **risk 0 by construction**, exempt
    from the τ gate. The only mode usable as the invariant fallback (§2.3).
  - *Descriptive / minted* (`<MEETING_DATE_1>`; E1, infiller-chosen label): the label is
    conditioned on the secret and context — a generalization in placeholder syntax. It carries
    role semantics to the remote model (most of a fill's task value) while keeping the
    echo-anchor property of the `<…>` syntax, but it **loses the τ-exemption**: minted labels
    pass the guess-back gate like any fill, and the used-set/indexing constraint applies to the
    label namespace. Echo survival of descriptive labels is assumed from the generic-placeholder
    measurement (9/9, n=9 — thin); smoke-measure before E1 relies on it.
  - The resulting action spectrum, by anchor-ness: generic placeholder (risk 0, echo ~certain,
    no semantics) → descriptive placeholder (τ-gated, role semantics survive) → relational fill
    (τ-gated + joint leakage) → naturalistic fill (τ-gated, echo ~5% measured in E0).
- Direct identifiers (PERSON, CODE): forced placeholder (not part of the action space).
- **No KEEP action.** Every detected span is substituted. Rationale: the v0 privacy term prices
  only substituted spans, so a KEEP action would be a reward-hacking channel (under-anonymize
  for free utility). Revisit only when A upgrades to a document-level head that prices raw
  surfaces (§4.1).
- The detector is frozen and outside the policy (its recall is the reported privacy ceiling;
  gradients would teach under-detection — decided in the plan).

### 2.3 Environment invariants (constraints, not learned behaviors)

These are enforced by the environment at assembly time. The policy cannot violate them, and the
reward is never asked to teach them — sparse probe reward (§3.4) cannot guarantee structural
invariants, and one violation is silently unrecoverable.

1. **Injectivity of R** — decode-time constraint. E0: a lattice level already claimed by a
   different surface is masked out of the action space; if a span's lattice is exhausted, the
   action collapses to PLACEHOLDER. E1: the infiller resamples (next beam) under a per-doc
   used-set; the *generic typed* placeholder is the terminal fallback. Repeat mentions / coref
   chains legitimately reuse their own replacement.
2. **τ as a hard ceiling** — a fill whose guess-back risk ≥ τ is rejected; exhaustion →
   *generic typed* placeholder (the only mode that is risk 0 by construction, §2.2). *This fixes a measured defect: the current floor fallback ships the coarsest level regardless of risk ("Commission" shipped at risk 0.147 against τ = 0.02).* In RL training, τ screening applies to the candidate set the ranker chooses from; α, not τ, is the operating-point knob.
3. **Reward–deployment extractor identity** — the `invert()` used inside the reward is the same
   implementation deployed at inference for that environment version (§2.4). Training against a
   better extractor than deployed (or vice versa) silently invalidates the learned policy.
4. **Determinism** — detection, lattice lookup, and E0 rendering are deterministic; all remote
   calls (none in training; gate/eval only) are content-cached.

### 2.4 Environment versions

The extractor and infiller define the physics of the environment: what survives the round trip
and what the reward can credit. Version them explicitly; **a policy is retrained when the
version changes** (an E0-trained policy is stale in E1).

|        | fills                                                                                     | extractor (`invert`)                                                                              | attack head A                                          | status                        |
| ------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ----------------------------- |
| **E0** | static lattice strings ∪ placeholder                                                      | rule: exact/fuzzy-85 match of replacements in out_p                                               | per-span MTI guess-back, aggregated                    | runnable now                  |
| **E1** | infiller (SFT flan-t5 + LoRA), injectivity-constrained decoding; relational fills allowed | rule + **paraphrase-tolerant alignment** (embedding/fuzzy match of out_p spans to R replacements) | same as E0                                             | infiller + alignment to build |
| **E2** | E1                                                                                        | **learned reconstructor** (denoise/edit model; the project's extraction direction)                | **document-level re-id head** (SynthPAI-style encoder) | future                        |

**The E0→E1 extractor upgrade is load-bearing for relational fills.** A relational fill the
remote model echoes loosely ("a couple of days after") is invisible to exact-match inversion, so
in E0 the reward would (correctly, for E0 physics) steer the policy away from relational fills
toward placeholder-like anchors. If the deployed product is meant to use a stronger extractor,
train in the environment that has it — otherwise the policy optimizes for the wrong physics.
This is the concrete form of "rule-based extraction is the current bottleneck".

## 3. Probes (Phase 0 — once per document, cached)

### 3.1 Construction

1. **Candidate surfaces** = unique R surfaces (dedup by canonical form; no other filtering —
   the previous role-noun filter was a measured bug that ate ~60% of surfaces).
2. **Gold restatement match**: surface must appear in the gold output — exact word-boundary
   match on canonicalized text (lowercase; "dr." → "doctor"; "milligrams" → "mg"), then
   rapidfuzz partial-ratio ≥ 85 for surfaces ≥ 5 chars. The matched gold sentence is kept as
   question context.
3. **Teacher question**: one natural question per surface whose exact answer is that surface
   (local llama-swap teacher, one-time, cached in `data/surrogate_probes.json`). Cloze phrasing
   is out-of-distribution for SQuAD2 readers (measured: readers abstain) — natural questions
   only. Validity: ends with "?", < 200 chars, does not contain the answer.
4. **Probe split (anti-Goodhart)**: per document, a held-out probe subset is never used in the
   training reward and is scored only at evaluation.

### 3.2 The same probes serve two roles

- **Training reward** (u_qa): answered from **doc_p**, question generalized through R, answer
  inverted through R, F1 vs original surface. Local, model-free per candidate (reader only).
- **Realized ground truth** (fact recall): answered from **out_final**, no generalization, no
  inversion. Used by the validation gate and the evaluation protocol — never by training.

### 3.3 Why this is the right utility axis (and reference overlap is not)

Measured 2026-07-03: detected-span words cover 6.4% (clinical) / 8.0% (enron) of gold tokens, so
whole-output similarity metrics cannot see substitution damage above generation noise
(shuffled-gold ROUGE-L floor within 1.4–2.9× of matched on clinical/enron). Probes concentrate
all metric mass on the perturbable facts — sensitive fraction ≈ 1 by construction.

### 3.4 Supply facts and their consequences

Current supply (16-doc slices): **3.19/doc clinical, 1.0/doc enron, 0.38/doc aeslc**; docs with
zero probes: 4/16 clinical, 8/16 enron, 11/16 aeslc.

- **Docs without probes are excluded from training** (no utility signal; training on them would
  optimize privacy alone). They remain in evaluation.
- u_qa is quantized to multiples of 1/n_probes — with ~3 probes the per-candidate reward takes
  ≤ 4 utility values. The group-relative advantage tolerates this (ranking within a group), but
  reward variance per doc is high; corpus mix and G compensate.
- Known unmatched classes (accepted ceilings, revisit if supply blocks training): spoken-vs-
  written numerals ("July thirty first" vs "07/31"), facts the gold restates but detection
  missed. aeslc's thin supply is corpus reality (subject lines restate little), not a bug.

## 4. Reward

```
r  =  α · (1 − A(doc_p))  +  (1 − α) · u_qa(doc_p, R, train-split probes)
```

### 4.1 Privacy term A

- **v0 (E0/E1):** aggregate of per-span MTI guess-back risks over substituted spans (mean; max
  as a reported diagnostic). Scores each replacement *in its doc_p context*, so neighboring
  fills partially condition the probe — but it is structurally per-span.
- **Known blind spot — joint leakage:** relational fills correlate facts ("two days later"
  resolves exactly once any anchor is pinned); a per-span head cannot price a *set* of fills
  whose combination identifies. The policy therefore faces the true relational-fill trade-off
  only partially during training in E0/E1. Consequences:
  - The frontier-LLM re-identification attacker on doc_p at **evaluation** is the real privacy
    measure (project law) and will catch joint leakage the reward missed — reported as an
    outcome, never patched with a per-model knob.
  - **Escalation trigger** (pre-registered): if eval shows re-identification driven by fill
    combinations that per-span MTI scored as safe, upgrade A to the document-level encoder head
    (E2) and retrain. Until then, "the model learns to balance joint information" is only as
    true as A's ability to see it — this is the sharpest known gap between the training reward
    and the deployed threat model.

### 4.2 Utility term u_qa (the mini round trip)

Per candidate (doc_p, R), per train-split probe: generalize the question through R (teacher
questions quote other spans' original surfaces) → extractive reader (SQuAD2 DeBERTa-class,
local) answers from doc_p → invert the answer through R with the **environment's extractor** →
token-F1 vs the original surface. Mean over probes.

- **R-inversion invariance:** legitimate invertible coarsening costs nothing ("Oslo" → "a
  Norwegian city" scores 1.0 after inversion); keeping the original earns no extra utility —
  under-anonymization is not rewarded. Utility falls exactly for destruction: unanswerable,
  unalignable, or non-invertible (collision) substitutions.
- **u_nli is dropped** from the reward (was ½(u_qa+u_nli)). Measured: mixing it degrades
  agreement with realized fact recall (clinical ρ 0.367 → 0.183); it is ~flat on register-
  shifted text and has a coarsening bias (weaker hypotheses entail more easily). It may return
  as a diagnostic, never as reward, unless redesigned and re-gated.

### 4.3 What the reward deliberately does not price (and where that surfaces)

| blind spot | surfaces at | pre-registered response |
|---|---|---|
| remote paraphrase of fills (`gen_absent`) | eval fact recall on out_final | E1 alignment / E2 reconstructor; also the round-trip-reward upgrade trigger |
| joint leakage of relational fills | eval attacker on doc_p | A upgrade to doc-level head (E2) |
| generation fluency damage from placeholder-heavy doc_p | eval fact recall + task metrics on out_final | report; policy re-balances only if the reward's utility term can see it (it can: unanswerable probes) — *the "placeholders read worse but score better" hypothesis is tested at eval, not assumed either way* |
| unprobed content damage | held-out probe split + eval attacker | probe diversity; report |

### 4.4 Anti-Goodhart / overoptimization controls

KL leash to init; low optimization pressure (small G, few epochs before re-gating); held-out
probe split; held-out corpora at eval; the Gao overoptimization playbook per
[adverserial-RL.md](../../research/adverserial-RL.md). If reward climbs while gate-style
realized checks fall, stop and report — that divergence is a finding.

## 5. Validation gate (before any training run)

Constructed arms per doc (no_privacy / tau_walk / all_floor / suppression) → per-doc Spearman
between u_qa's arm ordering and realized fact recall's. **Go:** clearly positive agreement where
the ground truth orders arms sanely. Re-run the gate whenever the environment version, probe
construction, or reward composition changes — the gate validates a (reward, environment) pair,
not the surrogate in the abstract. Current status: **passed 2026-07-03 for E0**
(0.37/0.44/0.775; `results/surrogate_validation.json`).

## 6. Key tensions (open, for review)

1. **Train-in-E0 vs build-E1-first.** E0 is runnable today, but its physics (exact-match
   inversion) reward placeholder-like anchors and punish relational fills regardless of their
   true value under a better extractor. If the product's extractor will be learned (project
   direction), heavy RL investment in E0 optimizes for the wrong physics; E0 stage-1 is still
   worth running as the cheap mechanism test (does selection learning move anything at all).
   The fork: how much E0 training is informative before the E1 extractor exists?
2. **Relational fills: utility vs joint leakage.** The environment permits them (E1); the
   reward prices their utility (probes) but only partially their risk (per-span A). Until the
   doc-level head exists, the eval attacker is the only honest scorekeeper — meaning the policy
   may learn relational fills that eval then reveals as leaky. That is the correct failure mode
   under this project's rules (measure outcomes, report), but it costs a retrain per A upgrade.
3. **Placeholder-heavy optima.** Given measured survival asymmetry (placeholders 9/9 vs
   naturalistic ~5%), the E0/E1 policy may converge on "placeholder almost everywhere" — a
   legitimate optimum of the stated reward, and possibly the honest answer for the rule
   extractor. If that offends the product goal (natural doc_p), the goal must enter the reward
   (a naturalness term) — a deliberate spec change, not a tuning knob.
4. **Probe sparsity vs corpus breadth.** Training signal concentrates on clinical (3.2/doc);
   enron trains on 8/16 docs, aeslc barely participates. Options if this binds: expand probe
   supply (number normalization, teacher-extracted fact tuples from gold), or accept
   clinical-heavy training and report per-corpus transfer.
5. **α as a retrain-expensive knob.** One α = one trained operating point; the Pareto curve
   costs ~3 retrains. Pre-registered fallback: if realized privacy clusters across α, α is a bad
   placement knob → lexicographic reward (plan, imported decision 4).

## Sources

Plan and decision history:
[2026-07-02-surrogate-grpo-training.md](../../plans/2026-07-02-surrogate-grpo-training.md)
(gate results, root-cause analysis, component initialization, wall-time);
[2026-07-02-roundtrip-grpo-training.md](../../plans/2026-07-02-roundtrip-grpo-training.md)
(GRPO mechanism, round-trip reward kept as upgrade). Eval corpora and metrics:
[benchmarks.md](../benchmarks.md); attack instruments: [attacks.md](../attacks.md).
Background: [adverserial-RL.md](../../research/adverserial-RL.md) with
[Gao et al. 2022](../../../research-wiki/papers/gao2022_reward_overoptimization.md)
([arXiv 2210.10760](https://arxiv.org/abs/2210.10760)); anti-extractor deletion lesson:
[NaPaRe](../../../research-wiki/papers/huang2025_tree_search_rewriting.md)
([arXiv 2509.20838](https://arxiv.org/abs/2509.20838)); surface-similarity pathology excluded
by R-inversion invariance:
[AgentStealth](../../../research-wiki/papers/shao2025_agentstealth.md)
([arXiv 2506.22508](https://arxiv.org/abs/2506.22508)).
