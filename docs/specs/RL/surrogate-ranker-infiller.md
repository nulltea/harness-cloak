---
type: reference
status: current
created: 2026-07-03
updated: 2026-07-05
tags: [rl, surrogate-reward, ranker, infiller, environment, probes, reward, anonymity-set,
       count-floors, keep-original, floor-randomized, grammar-constrained, gold-conditional,
       utility-only, fact-recall, injectivity, spec]
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
[structural-lattice-risk plan](../../plans/2026-07-04-structural-lattice-risk.md).

**Third revision (2026-07-05): reward redesign** — the utility term becomes the graded
**gold-conditional fact likelihood (u_gold, §5.1)**; the fill-proximity privacy term is
**retired from the reward** (privacy is floors-only) and **alpha is retired as an operating
knob** (the Pareto curve comes from the floor grid). Motivation: the measured
reward-landscape account of the two stage-1 NULLs — the old utility term was flat above bare
findability, the old privacy term's optimum was the degenerate all-placeholder point
(+0.062…0.177 over the init), and the KL leash had to overprice the only climb by ~10× to
contain it ([2026-07-05 training
record](../../../research-wiki/training/2026-07-05-RL-ranker-stage1-floor-env.md), landscape
addendum). Decision history and wall-time live in the
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
  legal exactly when `k_T ≤ 1` (a user-waived type). Supersedes the first revision's "No
  KEEP action" rule: legality comes from the user's waiver contract, and under the
  utility-only reward keep is simply the maximally-informative legal action. Measured
  motivation: the identity-only constructed arm (conditions kept, identities hidden) is the
  best realized clinical operating point (fact recall 0.300 vs tau-walk 0.133,
  `results/ranker_reward_gate.json`) at attack ≈ all-placeholder
  (`results/identity_attack.json`).
- **walk_risk (contrastive re-identification probe; legacy P4)** — P(attacker picks the
  original from a same-type candidate set | context + visible fill) via causal-LM log-probs.
  **Offline-only**: shootout calibration/validation and build-time diagnostics; never on a
  deployment or training legality path. Survives in action tables as a feature/diagnostic
  column.
- **fill_proximity (legacy P6)** — `cos_MiniLM(fill, original)` per fill. **Retired from the
  reward (2026-07-05)**: survives only as an action-table feature/diagnostic column, like
  walk_risk. The reward has no privacy term — privacy is floors-only (§5.2).
- **u_gold (gold-conditional fact likelihood)** — the reward's utility term (§5.1): the
  teacher-forced mean log-probability, under a **pinned frozen local causal LM**, of the gold
  output's *fact tokens* (gold tokens that restate substituted surfaces) conditioned on the
  task prompt built from doc_p; anchored per doc between the all-placeholder floor and the
  doc_orig ceiling. Dense, monotone in retained task-relevant information, one batched
  forward per rollout.
- **Restated-span probe / u_qa / fact recall** — QA probes on gold-restated surfaces. u_qa
  (reader → invert → F1) is **demoted to a diagnostic (2026-07-05)** — measured flat above
  bare findability (0/96 upward moves around a min-aset init) — and no longer trains or
  gates. The probe *machinery* (gold-restatement matching, teacher questions, train/held-out
  split) is retained: it defines u_gold's fact masks and fact recall's questions. fact
  recall on out_final remains the realized ground truth.
- **extract / invert** — the same deployed re-identification path (`cloak.extract.invert`);
  "extract" is the pipeline-stage name, `invert()` the implementation.
- **Echo / absorption** — whether the remote model reproduces a fill in out_p (echo) or uses
  its information with no surface trace (absorption). Dominated by task relevance, not fill
  form — **unpriced by the training reward** (§5.2).
- **frontier_claim** — the experiment's verdict: does the trained policy's Pareto dominate the
  floor-walk's at matched *realized* privacy (frontier-LLM attacker) on held-out docs. Never a
  training signal.
- **E0/E1/E2** — environment versions (§3.4). A policy is valid only for its training version.

### The two knobs (division of labor; alpha retired 2026-07-05)

- **k_T** — hard per-span, per-type ceiling on structural risk (the contract; defines what is
  legal at all). The only privacy knob, and now also the **operating-point knob**: one floor
  config = one operating point; the declared floor grid samples the Pareto curve.
- **the ranker** — the learned utility-maximizing allocation inside the legal set. Its
  legitimate target is the **non-monotone residual** the floor-walk rule cannot express:
  injectivity trades (coarsen one span to free a fill for another), cross-span coherence,
  and spots where a fluent coarser fill outscores an awkward specific one under u_gold.
- *(retired)* **α** — the old privacy↔utility mixing weight. The landscape probe showed its
  privacy side rewarded only the degenerate all-placeholder direction, forcing the KL leash
  to pin the policy. With privacy enforced by floors and utility monotone in retained
  information, a mixing weight has no legitimate job. Historical records keep their α
  labels.

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
# K_FLOORS calibration rule (reproducible): count buckets [<10, 10-100, 100-10k, >=10k];
# reference rate = the tau-walk arm's measured doc-level attack hit@5 (0.317); per type,
# floor = lower edge of the smallest bucket whose hit@5 <= reference AND n >= 5; types with
# no qualifying cell (thin cells, or no shootout items at all: MISC/OTHER) DEFAULT-DENY to
# 100. Re-derive on any change to lattice sources, count universes, or the referee label set.

# --- 0c. reward machinery + validation gate (go/no-go BEFORE any training run) ---
qa_probes[doc] = teacher_questions(R_surfaces ∩ gold(doc))  # cached; train/held-out split
pi_0 = behavior_clone(floor-walk decisions)                 # never RL from random
assert constructed_arms_gate(u_gold, fact_recall_on_out_final) is positive   # §6 — re-run
# required whenever reward, scorer LM, probes, prompt, extractor, OR THE ENVIRONMENT changes.
# STATUS: the u_gold reward has never been gated — training is blocked on that run (§6).
```

### Phase 1 — stage-1 ranker training (per floor config; local; no remote calls)

```python
for floor_config in FLOOR_GRID:                             # operating points; alpha is retired
  for epoch, doc in training:
    if randomize_floors:                                    # §5.4 — floor-portability training
        k = sample_floors(floor_config)                     # waived types stay 1; others centered
    else:
        k = floor_config
    legal, teacher, feats = derive_spans(doc, k)            # menus, floor-walk, features
    for g in 1..G:
        a_g = sample(pi, legal | dynamic_injectivity_mask)  # §3.3-1: claimed fills unsampleable
        doc_p_g, R_g = assemble(doc, a_g)
        r_g = u_gold(doc_p_g)                               # §5.1 — utility-only; privacy is
                                                            # already enforced by the mask
    adv = (r - mean(r)) / (std(r) + eps)                    # group advantage, within one floor sample
    loss = -sum(adv_g * logp(a_g)) / G + kl_coef * KL(pi ‖ pi_0)   # MINIMIZED; leash, kl_coef
    pi.minimize(loss)                                       # 0.01 default (§5.3): the reward no
                                                            # longer has a degenerate direction to
                                                            # contain. pi_0 BC'd under the SAME
                                                            # floor regime as the run.
# FIRST-SMOKE MILESTONE (mandatory before any full run): the smoke must show movement off
# the BC init (KL > 0.01 or greedy != BC on some span) — the optimizer canary inherited from
# the NULL diagnosis. A motionless smoke halts the run and triggers the coarse-init bug hunt.
# canonical training defaults (G = 8, epochs, lr 3e-4, kl_coef, seeds) are the argparse
# defaults of scripts/train_ranker.py; each run's actual values live in its training record.
# features: [is_placeholder, walk_risk, p6, level_index/4, n_levels/4,
#            log10_aset/9, log10_active_floor/9, type-onehot(7), corpus-onehot(3)]  (N_FEAT 17;
#            level_index and n_levels clipped at 4; placeholder rows feature aset as 1e9)
# greedy read-out: at FIXED floors on the declared grid — one operating point per floor
# config; randomization is train-time only and results are NEVER averaged across floors.
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
    for floor_config operating point, doc in HELD_OUT docs:
        doc_p, R  = policy(doc | floor_config)
        out_p     = RemoteLLM(task_prompt(doc_p))           # real round trip, cached
        out_final = extract(out_p, R)                       # deployed extractor (§3.3-4)
        utility  += fact_recall(out_final, held_out_probes)          # headline utility
        privacy  += 1 - frontier_attacker_success(doc_p)             # REAL attacker — never
        leakthrough += attacker_success(out_final)                   #   aset/walk_risk/p6
frontier_claim = pareto(policies) vs pareto(floor_walk)     # at matched REALIZED privacy → §1
```

**Pre-registered dominance test (the operational form of `frontier_claim`):** operating
points are compared within privacy bins of realized attacker success (bin width chosen so
each bin holds ≥ 2 operating points; no cross-bin interpolation claims); within a bin,
utility difference is judged by a **document-level bootstrap** (resample held-out docs,
1000 draws) — dominance requires the 95% CI of the utility difference to exclude 0;
otherwise the result is *equivalence at that privacy level*, reported as such. A policy
Pareto-dominates only if it wins ≥ 1 bin and loses none. Seeds and doc lists are fixed and
recorded before the eval run.

**Honesty boundaries:** aset, walk_risk, and fill_proximity never appear in Phase 3 as privacy
measures (training's teacher must not grade its own student); held-out docs, held-out probe
split, held-out attacker; second-remote-model arm per the training plan; results reported per
floor-config operating point, never averaged across floors.

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

**Keep-original** (supersedes the first revision's "No KEEP action" rule): legal exactly
when the user's floor for the type is ≤ 1 (a waiver). Two facts changed the ruling:
(1) legality is a *contract* question — a waived type is the user declaring the attribute
non-sensitive, and the mask, not the reward, carries that; under the utility-only reward
keep is simply the maximally-informative legal action. (2) Measured: the identity-only arm
(keeps on attribute types, placeholders on identity types) is the best realized clinical
operating point at attack parity with all-placeholder.

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
   exhaustion → generic placeholder (risk 0). k_T is both the contract and the operating
   point (alpha retired, §Definitions). The
   exhaustion fallback is load-bearing in four places: (a) the training environment (legal
   never empty); (b) deployed inference (the trained ranker samples only from the mask — it
   replaces the walk's *choice*, never the legality boundary; an unfamiliar user floor
   degrades choice quality, never privacy); (c) the behavior-clone init (a teacher that ships
   floor violations teaches them); (d) the eval control group. Floors are enforced by integer
   comparison against artifact-stored counts — **zero risk models at inference**.
3. **Truthfulness as a constraint (generate-then-verify)** — the reward must not price
   truthfulness (u_gold rewards informativeness whether or not the fill is true — a false
   but pinpointing fill scores high), so
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
| **E1** | infiller under grammar-constrained decoding; descriptive/relational fills | floors, online strict `aset_count` per instantiation | E0 + light fuzzy-verify | **design-not-build-ready** (see below) |
| **E2** | E1 | E1 + document-level attack head (frozen encoder + heads on SynthPAI attributes) | learned reconstructor | escalation |

The E1 semantic aligner remains descoped: absorption dominates `gen_absent`; what the remote
model absorbs cannot be won back at extraction time
([remote-llm-echo-absorption](../../issues/remote-llm-echo-absorption.md)).

**E1 build contract (what "design-not-build-ready" requires before implementation):** a
grammar artifact per (type, node class) — schema: canonical templates + closed slot
vocabularies, each shipped with a **parser round-trip test** (every producible string must
parse under `aset_count(strict=True)` back to a count for its node); decode-loop constants:
max resamples/beams before the generic-placeholder terminal, and deterministic behavior on
NLI-gate failure (resample budget shared with injectivity, never silent acceptance). These
are build deliverables, not open design questions — the design itself (§2 Phase 2, §3.3-6)
is settled.

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
| fill_proximity (legacy P6) | 0.643 | 0.761 / 0.760 | feature/diagnostic only (retired from the reward, §5.2) |

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

**Reward-support history (the two NULL diagnoses):** the reward can only teach what some
legal action can change. At the retired tau=0.02 mask, 3/106 train probes could flip under
any single-action counterfactual; on the floor environment 60/106 became flippable but all
7 actual flips pointed downward — u_qa was flat above bare findability
(`scripts/spikes/probe_flip_scan.py`, `scripts/spikes/reward_landscape_probe.py`). u_gold
(§5.1) removes the cliff (every legal action moves the score), so the binding constraint
shifts from flip support to **fact-mask coverage**: the §6 gate reports fact tokens per doc
per corpus, and corpus imbalance there is the queued expansion trigger (§7-6).

## 5. Reward (utility-only — third revision, 2026-07-05)

### 5.1 Utility term — u_gold (gold-conditional fact likelihood)

```python
# Phase 0, per doc, once (cached):
prompt(x)  = TASK_TEMPLATE[corpus].format(doc=x)        # the SAME prompt the round trip uses
gold_ids   = tokenize(gold(doc))
fact_mask  = positions in gold that restate substituted surfaces   # probes' matcher, reused
U_hi       = score(doc_orig)                            # ceiling: nothing hidden
U_lo       = score(all_placeholder(doc))                # floor: everything hidden

# per rollout — ONE teacher-forced forward, no generation, no reader, no inversion:
def score(doc_p):
    lp = log_softmax(LM(concat(prompt(doc_p), gold_ids)))      # pinned frozen local causal LM
    return mean(lp[t] for t in fact_mask)
u_gold(doc_p) = clip((score(doc_p) - U_lo) / (U_hi - U_lo), 0, 1)
```

Why this term: it is **dense and monotone in retained task-relevant information** — a fill
consistent with ~k candidates leaves the fact tokens at ≈ log(1/k), so the utility scale and
the anonymity-set scale are the same object seen from opposite sides. It prices the
specificity axis that u_qa was measured blind to (0/96 upward moves around a min-aset init),
has no abstention cliff (every action moves the number → REINFORCE always has a slope), and
costs one batched forward per rollout — cheaper than the retired per-probe reader loop.

Normative rules:
- **Anti-leak scoring (per-fact, cleaned prefix)** — the naive form above leaks: the gold
  prefix can restate the same fact (repeat mentions) or cue it, letting the scorer recover
  fact tokens without doc_p's help. Normative protocol: score **each fact span
  independently**; in its gold prefix, all *other* fact surfaces are R-generalized (the same
  rule u_qa used for question generalization) and *earlier mentions of the fact itself* are
  replaced by their doc_p replacement. One forward per (rollout, fact span), still
  teacher-forced and batched; U = mean over fact spans. The gate (§6) validates exactly this
  scorer, not the naive single-pass form.
- **The scorer LM is pinned and frozen for the whole (gate → training → eval) cycle** — any
  scorer change re-gates and invalidates trained policies (the extractor-pinning rule). It
  shares no parameters, gradients, or selection pressure with anything trained (§3.3-6
  proposer/verifier separation applies to reward models too).
- Fact masks come from the existing probe machinery (gold-restatement matching, train/
  held-out split): training scores only train-split fact tokens; held-out tokens are
  evaluation-only. Identity-typed spans are always chain-tokenized by the mask, so their
  fact tokens contribute a per-doc constant that group advantage cancels.
- Anchors U_hi/U_lo are per-doc constants (cached); the clip keeps u_gold in [0, 1] without
  any cross-doc normalization knob. **Edge-case rule (normative)**: a doc with an empty fact
  mask, or with `|U_hi − U_lo| < 0.05` nats (anchor separation too small for a stable
  denominator), is **excluded from the RL reward and listed in the gate report** — never
  silently clipped after division. Exclusion counts per corpus are part of the §6 gate
  output.
- **Intended, not proven, monotonicity**: "dense and monotone in retained information" is
  the design intent, not an established fact — a causal LM can prefer a fluent coarse fill
  over an awkward specific one, and per-fact scoring may be locally non-monotone.
  **Pre-registered sanity check (runs with the gate, before any training)**: the u_gold
  landscape probe — single-swap directionality on the cached constructed arms (does u_gold
  rise with specificity on ≥ a clear majority of level→level swaps, and does
  floor-walk ≥ all-placeholder per doc). A failed sanity check blocks training the same as
  a failed gate.

### 5.2 What was retired, and where privacy lives now

- **The privacy term (fill_proximity A) is retired from the reward.** Measured basis
  (landscape probe, 2026-07-05): A's optimum region was the degenerate all-placeholder point
  (+0.062/+0.119/+0.177 over the BC init at α 0.3/0.5/0.7; 43–46 of 96 single swaps locally
  positive toward collapse), so the term's only gradient advice was the behavior the KL
  leash existed to forbid. **Privacy is enforced exclusively by the per-type count floors**
  (§3.3-2); the reward is free to prefer specificity because illegal specificity is
  unreachable, not undesirable. fill_proximity and walk_risk survive as features and
  offline diagnostics.
- **alpha is retired**; operating points and the Pareto curve come from the **floor grid**
  (a declared set of per-type floor configs, each yielding one trained read-out; §5.4).
- **u_qa is demoted to a diagnostic** (its inversion-invariance made it a penalty-only term
  around any max-findability init); its probe machinery is retained for fact masks and for
  realized fact recall.
- **The echo factor remains deliberately NOT priced** (decision 2026-07-04, carried): echo
  is dominated by task relevance, outside the policy's control; the reward stays fully
  local. If the surrogate-vs-realized gap at eval is dominated by echo effects the policy
  could have influenced, the documented trigger is the round-trip-anchored reward upgrade
  (a local utility model fitted to cached realized fact recall), never an echo-survival
  table.
- **u_nli stays dropped** (measured: mixing degraded gate agreement 0.367 → 0.183).

### 5.3 Anti-Goodhart controls

- **KL leash retained at kl_coef 0.01 default** (was 0.05): the reward no longer contains a
  degenerate direction the leash must overpower — its job shrinks to damping noise-chasing.
  Landscape-probe basis: the old coefficient priced the (then-only) improving direction at
  ~10× its worth; a leash that must do that is compensating for a reward defect.
- **The u_gold duality, stated plainly**: u_gold is mathematically a friendly inference
  attack on the facts — p(original fact | doc_p). That is by design (utility of disclosure
  IS information disclosed) and is safe only because the floors, not the reward, decide
  what may be narrowed. Any proposal to soften floors "because utility wants it" is a
  contract change, never a training-time adjustment.
- Low optimization pressure (small G, few epochs between re-gates); held-out probe/fact
  split; held-out corpora and attacker at eval; Gao overoptimization playbook
  ([adverserial-RL.md](../../research/adverserial-RL.md)). Reward climbing while realized
  checks fall = stop and report.
- **Pre-registered null outcome**: with a monotone utility and floors-only privacy, the
  floor-walk rule is near-optimal by construction wherever the residual (§Definitions, the
  two knobs) is small. A trained policy ≈ floor-walk under a *healthy* reward (first-smoke
  milestone passed, gate passed) is a legitimate finding — "stage-1 selection learning adds
  little; learned value lives in the infiller" — to be reported as such, not engineered
  around.

### 5.4 Floor-randomized, floor-conditioned training (floor-portability)

One trained policy must serve any user floor configuration in the supported range without
retraining. Mechanism (shipped 2026-07-04, `--randomize-floors`):

- Per (epoch, doc): sample `k_T ~ log-uniform[k_T/10, 10·k_T]` (median = the deployment
  default; clamped ≥ 1) independently per type; **waived types (k_T ≤ 1) are not randomized**
  — a waiver is a discrete contract, and sampling above 1 would delegalize keep-original
  exactly where the user legalized it. Re-derive menus, floor-walk teacher, and features from
  the sample. **[k_T/10, 10·k_T] is the declared supported config range** — outside it the
  mask still enforces safely, choice quality is untested.
- The policy is conditioned pointwise: each span sees `log10(aset)` per action and
  `log10(active k_T of its own type)` — cross-type floor interactions reach it only through
  the doc-level advantage.
- Group advantage compares rollouts within one floor sample only; **BC pretrain randomizes
  floors identically** so the KL reference is trained on the floor-feature dimension it is
  queried on.
- **Read-outs are greedy at fixed floors on the declared grid** — one operating point per
  floor config; the grid IS the Pareto sample (alpha retired). Averaging results across
  floors blends operating points and is forbidden (the matched-privacy rule).

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
(**U = u_gold**, §5.1) and realized fact recall on out_final. The gate validates the utility
axis; the risk measure is validated separately by the matched attacker-correlation shootout
(§4.1). Operational bar: mean per-doc Spearman positive on **every** corpus (a negative
corpus is a no-go regardless of the pooled mean). The gate validates a **(reward,
environment) pair** — re-run on every change to reward composition, the scorer LM, probes,
prompt, extractor, or environment. **Status: REQUIRED-NOT-RUN for the u_gold reward** (never
gated; the last PASSED gate — u_qa~realized 0.354/0.511/0.760 on the floor environment,
2026-07-05 — certified the now-retired u_qa term). The next training run is blocked on: the
u_gold implementation + scorer-LM pinning, its gate run, the u_gold landscape sanity check
(§5.1), and the fact-mask coverage report (§4.2).

**Pre-registered gate report format (u_gold gate):** per corpus — (1) per-doc Spearman
u_gold~realized (mean, n); (2) fact-token coverage: fact tokens per doc (mean/min), docs
with empty fact masks; (3) anchor health: docs excluded by the `|U_hi − U_lo| < 0.05` rule
(§5.1); (4) usable-doc count after exclusions; (5) the landscape sanity-check verdict.
Pass = Spearman positive on every corpus AND usable docs ≥ 80% per corpus AND sanity check
passed. Any failure names the failing clause; no partial credit.

## 7. Open tensions

1. **Ranker headroom under a monotone utility** — with floors-only privacy and u_gold
   monotone in retained information, the floor-walk is near-optimal wherever the
   non-monotone residual (injectivity trades, cross-span coherence, fluency-vs-specificity)
   is small. Whether stage-1 selection learning adds measurable value over the walk is now
   the experiment's first question; the null outcome is pre-registered as legitimate
   (§5.3).
2. **Scorer-LM fidelity** — u_gold is only as aligned as the pinned local LM's conditional
   distributions; local-vs-remote mismatch is the standing surrogate gap, measured at eval
   (surrogate-vs-realized per checkpoint) and escalated to the round-trip-anchored reward
   (§5.2) if it dominates.
3. **The echo channel is entirely unpriced** (deliberate, §5.2, carried).
4. **Correlated-error mining of the verifier stack (E1+)** — the decode loop's resample
   operator optimizes against the frozen NLI gate and the grammar; a generator can converge
   on their blind spots. At stage 2 the u_gold scorer joins the frozen stack and inherits
   the same audit: the eval attacker reviews accepted fills; escalations pre-registered per
   verifier (§4.1 gaps, §3.3-3 NLI residual).
5. **Famous-context priors** — the structural count cannot see world-knowledge pinning
   (measured: "LJM2"); population-weighted counts, then per-surface overrides, are the
   escalation ladder; the eval attacker adjudicates when it binds.
6. **Fact-mask sparsity** — training signal concentrates where gold restates substituted
   spans (clinical-heavy). u_gold softens the old flip-support cliff (every action moves the
   score) but cross-doc coverage still binds; probe/fact expansion queued if the gate shows
   corpus imbalance.
7. **Waiver-region training coverage** — waived types stay fixed at floor 1 during
   randomization (a discrete contract), so keep-vs-level trades are trained only in grid
   points that include waivers; the declared floor grid must include at least one
   waiver-bearing config or the policy is untested exactly where users waive.

## Artifacts

`src/cloak/anonymity.py` (aset_count, K_FLOORS) · `scripts/annotate_lattice_counts.py`
(idempotent artifact annotation) · `scripts/build_ranker_env.py` → `data/ranker_env.json`
(k_floors) · `scripts/train_ranker.py` (floor legal sets, floor-walk BC, `--floors`,
`--randomize-floors`) · `src/cloak/train/ranker.py` (N_FEAT 17) ·
`scripts/spikes/lattice_count_shootout.py` → `results/lattice_count_shootout.json` ·
`scripts/spikes/probe_flip_scan.py` · `scripts/spikes/reward_landscape_probe.py` (the
landscape numbers behind the third revision) · `scripts/spikes/identity_attack.py` →
`results/identity_attack.json` · gate: `scripts/reward_gate.py` →
`results/ranker_reward_gate.json` · u_gold implementation lands in
`src/cloak/train/reward.py` (scorer LM pinned there and recorded in the gate artifact +
training records).
