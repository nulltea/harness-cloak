---
type: reference
status: current
created: 2026-07-03
updated: 2026-07-04
tags: [rl, surrogate-reward, ranker, infiller, environment, probes, reward, anonymity-set,
       count-floors, keep-original, floor-randomized, grammar-constrained, fact-recall,
       injectivity, spec]
companion: [docs/research/inference-risk-enforcement.md,
            docs/plans/2026-07-04-structural-lattice-risk.md,
            docs/plans/2026-07-02-surrogate-grpo-training.md, docs/specs/benchmarks.md,
            docs/specs/attacks.md, docs/issues/remote-llm-echo-absorption.md]
supersedes: the 2026-07-03 revision of this file (walk_risk tau-mask design)
---

# RL specification — surrogate-reward training of the substitutor (ranker + infiller)

Normative statement of the RL system: **pipeline → environment → risk measure → probes →
reward → baseline → gate**. Design pinned 2026-07-04 (second revision): **structural lattice
risk** — per-action anonymity-set counts with per-type floors — replaces the walk_risk LM
probe as the legality mask everywhere on the inference and training paths; the probe LM
retires to offline calibration/validation. Rationale and the option analysis:
[inference-risk-enforcement](../../research/inference-risk-enforcement.md); migration:
[structural-lattice-risk plan](../../plans/2026-07-04-structural-lattice-risk.md). Decision
history and wall-time live in the
[training plan](../../plans/2026-07-02-surrogate-grpo-training.md).

## Definitions

- **doc_orig / doc_p / out_p / out_final** — original document; anonymized rewrite sent to the
  remote LLM; the remote output; the locally re-identified output returned to the user.
- **R (substitution record)** — client-side `{surface, replacement, type, action}` list; the
  only bridge from doc_p-space back to original-space. **Injective per document** by
  environment constraint.
- **Ranker** — the stage-1 policy. Sole function: per detected quasi-identifier span, pick one
  action from that span's action set (how coarse the substitution is).
- **Infiller** — stage-2 generative component rendering the chosen lattice node as surface
  text under **grammar-constrained decoding** (§3.3-6; planned flan-t5-base + LoRA; not yet
  implemented).
- **Lattice** — ordered replacement phrases for a span, most-specific → most-general; with the
  keep-original node at depth 0 and the generic placeholder terminal, the ranker's action
  space.
- **aset (anonymity-set count)** — the structural risk measure: how many candidate values *at
  the original's granularity* are consistent with a fill (`cloak.anonymity.aset_count`;
  GeoNames counts for LOC, WordNet hyponym-leaf closure for DEM/ORG/MISC, window/granularity
  for DATETIME, range/precision-step for QUANTITY). **strict mode** is the certifying mode: a
  parse miss fails closed to 1.0, never falls through to a broader lookup (the permissive
  last-word fallback fails *open* and exists for diagnostics only). Keep-original ≡ 1.0;
  TYPE_LABEL coarse fills ≡ GENERIC (1e9). Deterministic, model-free, ~µs.
- **k_T / K_FLOORS (per-type count floors)** — the privacy contract: a level action is legal
  iff `aset ≥ k_T` for its span's type. Calibrated on the count-vs-attacker shootout
  (`results/lattice_count_shootout.json`): all types default to 100 (default-deny; MISC/OTHER
  have no shootout data, so they inherit the same conservative floor). A waiver = the user
  setting a type's floor to 1, which is what legalizes keep-original for that type.
  k_T ≈ 1/tau_T under a uniform attacker prior. Floors are user-facing
  config; the supported per-type range is [k_T/10, 10·k_T] (§5.4).
- **floor-walk** — the rule baseline and behavior-clone teacher: per span, the *minimum-aset*
  legal level (ties broken by list index), else generic placeholder. Replaces the tau-walk;
  the tau-walk survives only as the frozen artifact reference for BC-reproduction checks.
- **keep-original** — a first-class level action `{fill = surface, aset = 1.0, p6 = 1.0}`,
  legal exactly when `k_T ≤ 1` (a user-waived type). Supersedes the previous revision's "No
  KEEP action" rule: legality now comes from the user's waiver contract, not from the reward's
  ability to price it, and the reward *does* price it (maximal proximity, p6 = 1.0). Measured
  motivation: the identity-only constructed arm (conditions kept, identities hidden) is the
  best realized clinical operating point (fact recall 0.300 vs tau-walk 0.133,
  `results/ranker_reward_gate.json`) at attack ≈ all-placeholder
  (`results/identity_attack.json`).
- **walk_risk (contrastive re-identification probe; legacy P4)** — P(attacker picks the
  original from a same-type candidate set | context + visible fill) via causal-LM log-probs.
  **Offline-only**: shootout calibration/validation and build-time diagnostics; never on a
  deployment or training legality path. Survives in action tables as a feature/diagnostic
  column.
- **A / fill_proximity (legacy P6)** — the reward's privacy term: `cos_MiniLM(fill, original)`
  per level-mode fill (keep-original included at 1.0; generic placeholders excluded), mean-
  aggregated. Candidate-sensitive, context-blind.
- **Restated-span probe / u_qa / fact recall** — QA probes on gold-restated surfaces; u_qa
  reads doc_p (training utility), fact recall reads out_final (realized ground truth).
- **Echo / absorption** — whether the remote model reproduces a fill in out_p (echo) or uses
  its information with no surface trace (absorption). Dominated by task relevance, not fill
  form — **unpriced by the training reward** (§5.2).
- **frontier_claim** — the experiment's verdict: does the trained policy's Pareto dominate the
  floor-walk's at matched *realized* privacy (frontier-LLM attacker) on held-out docs. Never a
  training signal.
- **E0/E1/E2** — environment versions (§3.4). A policy is valid only for its training version.

### The three knobs (division of labor)

- **k_T** — hard per-span, per-type ceiling on structural risk (the contract; defines what is
  legal at all). The only legitimate way to move a policy's privacy operating point.
- **α** — the training-time mixing weight, `r = α·(1 − A) + (1 − α)·u_qa`; soft preference
  over how the policy trades the remaining slack. One (α, floor-config) = one operating point;
  the α sweep {0.3, 0.5, 0.7} samples the training-time Pareto. Never "tuned to a best value".
- **the ranker** — the learned per-span allocation under both.

## 1. Objective and verdict

Train the substitutor to maximize round-trip task utility at a target privacy level, with
**zero remote calls during training**. The verdict (`frontier_claim`) is decided only in
Phase 3: trained policies vs the floor-walk control group at **matched realized privacy**
(frontier-LLM attacker on doc_p, leak-through on out_final), utility as fact recall on
out_final, on held-out documents and the held-out probe split. `frontier_claim` feeds
gradients to nothing — routing the eval attacker into training would Goodhart the evaluation.

## 2. Pipeline (normative pseudocode)

### Phase 0 — offline, once per corpus (environment + reward machinery)

```python
# --- 0a. environment (per document; NEVER re-detected — artifact-frozen, §3.3-5) ---
for doc in corpus:
    spans[doc] = detect(doc)                                # frozen detector (§3.2)
    for s in spans[doc]:
        levels[s] = lattice(s) + [KEEP(s)]                  # NLI-truthfulness-gated; keep at depth 0
        for l in levels[s]:
            aset[s, l] = aset_count(l, s.type, s.orig, strict=True)   # model-free, certifying
        # legality is DERIVED AT USE TIME from floors, never stored:
        legal[s | k] = [l for l in levels[s] if aset[s, l] >= k[s.type]] + [PLACEHOLDER(s.type)]
        # placeholder: always legal (risk 0 by construction) -> legal[] is never empty.
        # walk_risk[s, l] is still computed HERE (offline) as a feature/diagnostic column.

# --- 0b. floor calibration (once per lattice-source change) ---
# count-vs-attacker shootout on cached labels: promote/retain the structural measure only by
# measured attacker correlation (matched items + labels). 2026-07-04, gemini referee, n=150:
#   inv_aset level-ordering 0.786 / AUC 0.726/0.759  vs  walk_risk 0.714 / 0.660/0.601
# K_FLOORS = smallest count bucket per type with attacker hit@5 at/below the reference rate.

# --- 0c. reward machinery + validation gate (go/no-go BEFORE any training run) ---
qa_probes[doc] = teacher_questions(R_surfaces ∩ gold(doc))  # cached; train/held-out split
pi_0 = behavior_clone(floor-walk decisions)                 # never RL from random
assert constructed_arms_gate(reward, fact_recall_on_out_final) is positive   # §6 — re-run
# required whenever reward, probes, prompt, extractor, OR THE ENVIRONMENT changes (the
# keep-original + floor environment of 2026-07-04 requires a fresh gate run before training).
```

### Phase 1 — stage-1 ranker training (per α; local; no remote calls)

```python
for epoch, doc in training:
    if randomize_floors:                                    # §5.4 — tau-portability training
        k = {T: exp(U(ln(max(k_T/10, 1)), ln(10*k_T))) for T, k_T in K_FLOORS.items()}
        legal, teacher, feats = derive_spans(doc, k)        # menus, floor-walk, features re-derived
    else:
        k = K_FLOORS
    for g in 1..G:
        a_g = sample(pi, legal | dynamic_injectivity_mask)  # §3.3-1: claimed fills unsampleable
        doc_p_g, R_g = assemble(doc, a_g)
        A_g = mean(fill_proximity(fill(a_g[s]), s.orig)     # keep-original counts at 1.0;
                   for s if mode(a_g[s]) != generic_placeholder)
        U_g = mean(F1(invert(reader(q̃_j, doc_p_g), R_g), a_j) for j in train_probes)  # §5.2
        r_g = α*(1 - A_g) + (1 - α)*U_g
    adv = (r - mean(r)) / std(r)                            # group advantage, within one floor sample
    pi.update(REINFORCE(adv) + KL(pi ‖ pi_0))               # pi_0 BC'd under the SAME floor regime:
                                                            # randomized runs use floor-randomized BC
# features: [is_placeholder, walk_risk, p6, level_index/4, n_levels/4,
#            log10_aset/9, log10_active_floor/9, type-onehot(7), corpus-onehot(3)]  (N_FEAT 17)
# greedy read-out: at FIXED floors on a declared grid — one operating point per (α, floor
# config); randomization is train-time only and results are NEVER averaged across floors.
```

### Phase 2 — stage-2 joint training (infiller unfrozen; E1+ only)

```python
# grammar-constrained decode loop — proposer/verifier separation (§3.3-6):
for span s with ranker-chosen node l:
    while True:
        fill = infiller.sample(doc, s, l | GRAMMAR(l))      # only canonical templates +
                                                            # closed slot vocabularies producible
        if fill claimed by another surface:      continue   # injectivity (resample/next beam)
        if aset_count(fill, s.type, s.orig, strict=True) < k[s.type]:  continue
        #   ^ deterministic online certificate — exact by construction inside GRAMMAR(l);
        #     the infiller may pick a DIFFERENT slot than the ranker's default (recomputed live)
        if not NLI_entailed(sent(s.orig), sent(fill)):      continue   # truthfulness gate
        break  # accept; exhaustion -> generic placeholder terminal
# The infiller NEVER emits or certifies its own count. A predicted-count head, if ever added,
# is a training-time calibration aid only. LM count-estimation is a BUILD-TIME tool (authoring
# slot vocabularies / populating counts for out-of-universe lattice nodes), shootout-calibrated.
# Shared scalar advantage on ranker log-probs + infiller token log-probs (PPO-clipped), LoRA.
```

### Phase 3 — evaluation (once per sweep; the only remote-heavy stage)

```python
for policy in trained_policies + [floor_walk]:              # the floor-walk = control group
    for (α, floor_config) operating point, doc in HELD_OUT docs:
        doc_p, R  = policy(doc | floor_config)
        out_p     = RemoteLLM(task_prompt(doc_p))           # real round trip, cached
        out_final = extract(out_p, R)                       # deployed extractor (§3.3-4)
        utility  += fact_recall(out_final, held_out_probes)          # headline utility
        privacy  += 1 - frontier_attacker_success(doc_p)             # REAL attacker — never
        leakthrough += attacker_success(out_final)                   #   aset/walk_risk/p6
frontier_claim = pareto(policies) vs pareto(floor_walk)     # at matched REALIZED privacy → §1
```

**Honesty boundaries:** aset, walk_risk, and fill_proximity never appear in Phase 3 as privacy
measures (training's teacher must not grade its own student); held-out docs, held-out probe
split, held-out attacker; second-remote-model arm per the training plan; results reported per
(α, floor-config) operating point, never averaged across floors.

## 3. Environment

### 3.1 Episode and action space

One episode = one document (contextual bandit; no multi-step dynamics in v0). Per detected
quasi-identifier span, the action set is `legal[s | k]` — floor-legal lattice levels
(keep-original included, at depth 0) ∪ generic typed placeholder.

Placeholder is a first-class action, not only a fallback: pricing it lets the policy trade
risk against locally-verifiable utility per span. Measured properties: **inversion given echo
is perfect** for placeholders (ph_swapped 9/9, zero residue), but **echo itself is
task-relevance-dependent, not form-guaranteed**.

**Placeholder modes** (different privacy contracts):
- *Generic typed* (`<DATETIME_1>`): label depends only on the detected type — **risk 0 by
  construction**, exempt from floors, the only legal invariant-fallback.
- *Descriptive/minted* (`<MEETING_DATE_1>`; E1, infiller-chosen): a generalization in
  placeholder syntax — inherits clean inversion-given-echo, but **loses the floor exemption**
  (aset-scored like any fill; used-set/indexing applies to the label namespace).

**Keep-original** (supersedes the previous "No KEEP action" rule): legal exactly when the
user's floor for the type is ≤ 1 (a waiver). Three facts changed the ruling: (1) the reward
now prices keeps — they are level-mode actions with fill_proximity 1.0, the maximal privacy
penalty; (2) legality is a *contract* question, and a waived type is the user declaring the
attribute non-sensitive — the mask, not the reward, carries that; (3) measured: the
identity-only arm (keeps on attribute types, placeholders on identity types) is the best
realized clinical operating point at attack parity with all-placeholder. Known residual — A
mis-prices keep-heavy configurations relative to realized utility/privacy (§7-2).

Direct identifiers (PERSON, CODE): forced generic placeholder, outside the action space.

### 3.2 Frozen components

The **detector** stays outside the policy: a missed span costs the reward nothing, so
gradients would teach under-detection; detection recall is the reported privacy ceiling.
The dominant attacker-recovery channel in shootout examples is *retained context* —
undetected sibling mentions — recorded in
[detection-sibling-mention-leak](../../issues/detection-sibling-mention-leak.md); a leak
channel no amount of level-selection training can close.

### 3.3 Environment invariants (constraints, never learned behaviors)

Sparse probe reward cannot guarantee structural properties; one violation is silently
unrecoverable. Enforced at assembly/decoding time:

1. **Injectivity of R** — the environment *guarantees* injectivity; RL never learns the
   invariant and the reward never prices collisions. E0: dynamic sampling mask in rollouts (a
   level whose fill is claimed by another surface is unsampleable; exhaustion → generic
   placeholder). E1: the constrained-decode loop's injectivity check (§2 Phase 2). Repeat
   mentions/coref chains reuse their own replacement. Accepted mismatch: the *static*
   floor-walk teacher trajectory can be jointly non-injective at high floors (several spans
   share one coarse level); BC is per-span cross-entropy and rollouts are dynamically masked,
   so no runtime guarantee is affected — collision counts are reported, not asserted away.
2. **Per-type count floors as the hard ceiling** — `aset[s, l] < k_T` ⇒ `l ∉ legal[s]`;
   exhaustion → generic placeholder (risk 0). α, not k_T, is the preference knob. The
   exhaustion fallback is load-bearing in four places: (a) the training environment (legal
   never empty); (b) deployed inference (the trained ranker samples only from the mask — it
   replaces the walk's *choice*, never the legality boundary; an unfamiliar user floor
   degrades choice quality, never privacy); (c) the behavior-clone init (a teacher that ships
   floor violations teaches them); (d) the eval control group. Floors are enforced by integer
   comparison against artifact-stored counts — **zero risk models at inference**.
3. **Truthfulness as a constraint (generate-then-verify)** — the reward cannot price
   truthfulness (u_qa is inversion-invariant to fill semantics; A prices only proximity), so
   truthfulness lives in the environment as the NLI entailment gate (premise = original
   sentence, hypothesis = sentence with the fill; keep iff entailed). E0: all lattice sources
   pass the gate (shipped 2026-07-04). E1: the gate joins the decode loop (§2 Phase 2). Known
   limit: NLI is a cheap verifier, not a semantics oracle — residual tracked in
   [rule-lattice-nli-gate-bypass](../../issues/rule-lattice-nli-gate-bypass.md).
4. **Extractor scope** — the reward calls the *deployed* `invert()`; the extractor version is
   pinned for the whole (gate → training → eval) cycle — any extractor change re-gates and
   invalidates trained policies, same rule as a reward change.
5. **Determinism via artifacts** — detection is nondeterministic across processes; all
   consumers load the persisted arms artifact (`data/task_arms_tau0.02.json`, filename
   historical), never re-detect. Count annotation is a separate idempotent in-place pass
   (`scripts/annotate_lattice_counts.py`) that never re-detects and never recomputes
   walk_risk.
6. **Grammar constraint (E1+)** — generated fills are producible only from the chosen node's
   canonical templates + closed slot vocabularies, so every producible string parses under
   `aset_count(strict=True)` and the online count certificate is exact by construction.
   **Proposer/verifier separation is absolute**: the infiller proposes; the deterministic
   scorer + frozen NLI gate verify; nothing that certifies legality shares parameters,
   gradients, or selection pressure with the proposer. (The rejected alternative — the
   infiller serving its own risk score — is a wireheading channel at joint training and a
   correlated-error-mining target even frozen; analysis in
   [inference-risk-enforcement](../../research/inference-risk-enforcement.md) §2.2.)

### 3.4 Environment versions

| | fills | risk mask | extractor | status |
|---|---|---|---|---|
| **E0** | static lattice strings ∪ keep-original ∪ generic placeholder | per-type count floors (artifact-stored aset) | rule exact/fuzzy-90 | **live** (2026-07-04) |
| **E1** | infiller under grammar-constrained decoding; descriptive/relational fills | floors, online strict `aset_count` per instantiation | E0 + light fuzzy-verify | infiller to build |
| **E2** | E1 | E1 + document-level attack head (frozen encoder + heads on SynthPAI attributes) | learned reconstructor | escalation |

The E1 semantic aligner remains descoped: absorption dominates `gen_absent`; what the remote
model absorbs cannot be won back at extraction time
([remote-llm-echo-absorption](../../issues/remote-llm-echo-absorption.md)).

## 4. Risk measure and probes

### 4.1 The structural risk measure (promotion by matched attacker correlation)

A risk measure is promoted only by measured correlation with a real LLM attacker on identical
items and identical labels; it is a legality/training signal, **never reported privacy**.
Count-vs-attacker shootout (2026-07-04, 150 (span, level) items, gemini-3.1-pro labels, zero
new calls — `scripts/spikes/lattice_count_shootout.py` →
`results/lattice_count_shootout.json`):

| measure | level-ordering | AUC hit@1 / hit@5 | role |
|---|---|---|---|
| **inv_aset (1/count, strict)** | **0.786** | 0.726 / 0.759 | **legality mask + floor calibration** |
| walk_risk (contrastive LM; legacy P4) | 0.714 | 0.660 / 0.601 | retired to offline calibration/diagnostics |
| fill_proximity (legacy P6) | 0.643 | 0.761 / 0.760 | reward privacy term A (unchanged) |

The pre-registered adoption rule was "structural if within 0.05 of walk_risk"; it won
outright. (The previous revision's .86 figure for walk_risk was the Qwen-referee column of the
same protocol; under the gemini labels used here its matched value is 0.714 — reproduced
exactly.)

**Known gaps of the structural measure** (priced at eval, pre-registered responses):
non-uniform priors — a famous-context instance ("LJM2") is recoverable from world knowledge a
count cannot see (escalation: population-weighted counts, then per-surface overrides);
calibration cells are thin (several n ≤ 3); the fuzzy hit@5 label rewards lexical proximity,
inflating measured leakiness of coarse-but-lexically-close fills (dates, quantity ranges);
GeoNames reverse counts carry ≤2× name/asciiname inflation (within-type consistent);
open-vocabulary MISC has no universe — fails closed, user-waivable, offline LM
count-estimation is the build-time remedy. Per-span measures remain blind to joint/relational
leakage until E2's document head.

### 4.2 QA probes (utility axis)

Construction per document, once, cached: unique R surfaces (dedup only — no role filter) →
gold-restatement match (canonicalized exact, then fuzzy ≥ 85) → one teacher-written natural
question per surface (cloze is reader-OOD, measured). Anti-Goodhart: per-doc held-out probe
subset scored only at evaluation. Probe supply concentrates on clinical; expansion options
queued if it binds.

**Reward-support requirement (measured 2026-07-04, the stage-1 NULL diagnosis):** the reward
can only teach what some legal action can change. At the retired tau=0.02 mask, only 3/106
train probes could flip under any single-action counterfactual
(`scripts/spikes/probe_flip_scan.py`) — an optimization-proof reward desert. The floor
environment roughly doubles decision freedom (146/177 spans with ≥2 legal actions vs 72/177)
and legalizes keep/level trades on probe-bearing attribute spans; **re-measuring probe-flip
support on the floor environment is part of the §6 gate** for the next training run.

## 5. Reward

### 5.1 Privacy term A

`A = mean over level-mode fills of cos_MiniLM(fill, original)` — keep-original included at
1.0, generic placeholders excluded (constant, no gradient). Context-blind by construction;
the frontier attacker at eval prices what it misses. Known mis-pricing: A rates keep-heavy
configurations maximally unsafe regardless of the type's identifying power — measured against
realized outcomes in the identity-only arm (§7-2); if the eval attacker confirms A's ordering
is wrong where it matters, A escalates (walk_risk term or E2 document head), the policy
retrains.

### 5.2 Utility term — u_qa, uniform reader path, no echo prior

```
u_j = F1( invert( reader(q̃_j, doc_p), R ),  a_j )     for EVERY train-split probe j
U   = mean_j u_j
```

One path for all fill modes: question generalized through R → SQuAD2 reader on doc_p → answer
inverted through R → token-F1 vs the original surface. Properties:

- **Coarsening invariance**: invertible coarsening scores 1.0; keeping the original earns no
  *bonus* over an invertible coarse fill — but note keep-original IS findable by the reader
  where coarse fills sometimes are not, so u_qa retains a measured pull toward specificity
  (no_privacy arm U 0.366 vs anonymized arms 0.05–0.075). The floors, not the reward, are
  what make under-anonymization unreachable.
- **Placeholder-covered probes score through the same reader**: usually 0 (SQuAD2 abstains on
  `<TYPE_n>`), occasionally 1. Measured conservative bias, accepted as fail-closed; whatever
  placeholders deliver beyond local verifiability shows up in the surrogate-vs-realized gap.
- **The echo factor is deliberately NOT priced** (decision 2026-07-04, upheld): echo is
  dominated by task relevance, outside the policy's control; the reward stays fully local
  with zero remote-measured constants. If the surrogate-vs-realized gap at eval is dominated
  by echo effects the policy could have influenced, the documented trigger is the round-trip
  reward upgrade — never an echo-survival table.
- **u_nli is dropped** (measured: mixing degrades gate agreement 0.367 → 0.183); diagnostic
  only.

### 5.3 Anti-Goodhart controls

KL leash to the behavior-clone init (reference trained under the same floor regime as the run,
§5.4); low optimization pressure (small G, few epochs between re-gates); held-out probe split;
held-out corpora/attacker at eval; Gao overoptimization playbook
([adverserial-RL.md](../../research/adverserial-RL.md)). Reward climbing while realized checks
fall = stop and report.

### 5.4 Floor-randomized, floor-conditioned training (tau-portability)

One trained policy must serve any user floor configuration in the supported range without
retraining. Mechanism (shipped 2026-07-04, `--randomize-floors`):

- Per (epoch, doc): sample `k_T ~ log-uniform[k_T/10, 10·k_T]` (median = the deployment
  default; clamped ≥ 1) independently per type; re-derive menus, floor-walk teacher, and
  features from the sample. **[k_T/10, 10·k_T] is the declared supported config range** —
  outside it the mask still enforces safely, choice quality is untested.
- The policy is conditioned pointwise: each span sees `log10(aset)` per action and
  `log10(active k_T of its own type)` — cross-type floor interactions reach it only through
  the doc-level advantage.
- Group advantage compares rollouts within one floor sample only; **BC pretrain randomizes
  floors identically** so the KL reference is trained on the floor-feature dimension it is
  queried on.
- **Read-outs are greedy at fixed floors on a declared grid** — one operating point per
  (α, floor config). Averaging results across floors blends operating points and is
  forbidden (the matched-privacy rule).

**Pre-registered protocols for the first randomized run** (failure-mode register,
2026-07-04):
1. *Conditioning ablation*: floor-blind vs floor-conditioned policy at held-out floors — if
   they tie, the conditioning claim dies and only the fixed-floor result stands.
2. *Nonstationarity control*: fixed-floor run at the same update budget precedes the
   randomized run; a randomized NULL against a fixed-floor positive means "not enough updates
   per context", not "cannot learn".
3. *Teacher drift diagnostic*: report placeholder-rate drift vs the static teacher at high
   floors (non-injective-teacher bias, §3.3-1).

## 6. Baseline and validation gate

The **floor-walk** (min-aset legal level, else placeholder) is: the control group of
`frontier_claim`, the behavior-clone teacher of `pi_0`, and the constructed-arms source. Its
teacher decisions are re-derived from floors at use time; the frozen artifact's stored
tau-walk remains only as the BC-reproduction reference for the legacy fixed configuration.

**Gate (before any training run):** constructed arms (no_privacy / floor_walk / all_floor /
suppression / identity_only) per doc → per-doc Spearman between the utility term's ordering
(U = u_qa) and realized fact recall on out_final. The gate validates the utility axis; the
risk measure is validated separately by the matched attacker-correlation shootout (§4.1).
r~realized is context, never the criterion. The gate validates a **(reward, environment)
pair** — re-run on every change to reward composition, probes, prompt, extractor, or
environment. **Status: REQUIRED-NOT-RUN for the floor environment** (the 2026-07-04 keep +
floors migration changed the environment; the last PASSED gate — U~realized 0.354/0.511/0.760
— was the tau-mask environment). The next training run is blocked on: fresh gate run + the
probe-flip support re-measurement (§4.2).

## 7. Open tensions

1. **Stage-2 gameability of A** — an unfrozen infiller can craft embedding-far,
   information-close fills. Pre-registered guard: a walk_risk term joins A offline-computed,
   or the E2 head lands, before stage 2 unfreezes the infiller. The grammar constraint
   (§3.3-6) independently bounds the fill space.
2. **A mis-prices keep-original** — maximal proximity penalty regardless of identifying
   power, while realized measurements (identity-only arm) show keep-heavy attribute configs
   can dominate. The α equilibrium therefore under-uses waivers relative to realized optimum;
   quantifying that gap is a first-class evaluation question, and the pre-registered remedy
   is a type-aware A (never a per-model calibration).
3. **Opposing placeholder biases in the reward** — the privacy term favors placeholders
   (excluded from A) while the utility term under-credits them (fail-closed reader path).
   The policy's placeholder rate is an α-governed equilibrium between two known biases.
4. **The echo channel is entirely unpriced** (deliberate, §5.2).
5. **Correlated-error mining of the verifier stack (E1+)** — the decode loop's resample
   operator optimizes against the frozen NLI gate and the grammar; a generator can converge
   on their blind spots. The eval attacker audits accepted fills; escalations are
   pre-registered per verifier (§4.1 gaps, §3.3-3 NLI residual).
6. **Famous-context priors** — the structural count cannot see world-knowledge pinning
   (measured: "LJM2"); population-weighted counts, then per-surface overrides, are the
   escalation ladder; the eval attacker adjudicates when it binds.
7. **Probe sparsity and reward support** — training concentrates on clinical; the reward
   teaches only what legal actions can flip (§4.2). Probe expansion and the floor-environment
   support re-measurement are queued ahead of the next run.

## Artifacts

`src/cloak/anonymity.py` (aset_count, K_FLOORS) · `scripts/annotate_lattice_counts.py`
(idempotent artifact annotation) · `scripts/build_ranker_env.py` → `data/ranker_env.json`
(k_floors) · `scripts/train_ranker.py` (floor legal sets, floor-walk BC, `--floors`,
`--randomize-floors`) · `src/cloak/train/ranker.py` (N_FEAT 17) ·
`scripts/spikes/lattice_count_shootout.py` → `results/lattice_count_shootout.json` ·
`scripts/spikes/probe_flip_scan.py` · `scripts/spikes/identity_attack.py` →
`results/identity_attack.json` · gate: `scripts/reward_gate.py` →
`results/ranker_reward_gate.json`.
