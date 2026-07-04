---
type: reference
status: current
created: 2026-07-03
updated: 2026-07-04
tags: [rl, grpo, surrogate-reward, ranker, infiller, environment, probes, reward, tau-mask,
       contrastive-reid, embedding-proximity, fact-recall, injectivity, spec]
companion: [docs/plans/2026-07-03-surrogate-rl-gaps-fixes.md,
            docs/plans/2026-07-02-surrogate-grpo-training.md, docs/specs/benchmarks.md,
            docs/specs/attacks.md, docs/issues/remote-llm-echo-absorption.md]
---

# RL specification — surrogate-reward training of the substitutor (ranker + infiller)

Normative statement of the RL system: **pipeline → environment → probes → reward → baseline →
gate**. Decision history and wall-time live in the
[training plan](../../plans/2026-07-02-surrogate-grpo-training.md); open defects and fix order in
the [gaps plan](../../plans/2026-07-03-surrogate-rl-gaps-fixes.md). Design pinned 2026-07-04:
**contrastive re-identification probe (P4) for the walk/action mask; embedding-proximity score
(P6) for the reward's privacy term** — the probe-per-job split measured in the
[shootout](../attacks.md) (§ Probes).

## Definitions

- **doc_orig / doc_p / out_p / out_final** — original document; anonymized rewrite sent to the
  remote LLM; the remote output; the locally re-identified output returned to the user.
- **R (substitution record)** — client-side `{surface, replacement, type, action}` list; the
  only bridge from doc_p-space back to original-space. **Injective per document** (no two
  surfaces share a replacement) by environment constraint.
- **Ranker** — the stage-1 policy. Sole function: **per detected quasi-identifier span, pick one
  action from that span's lattice levels ∪ {placeholder}** (how coarse the substitution is).
- **Infiller** — stage-2 generative component rendering the chosen level as surface text
  (planned flan-t5-base + LoRA; not yet implemented).
- **Lattice** — ordered replacement phrases for a span, most-specific → most-general; the
  ranker's action space.
- **τ-walk** — the rule baseline: accept the first (most specific) lattice level whose probe
  risk < τ. In RL it survives as three artifacts: behavior-clone init, evaluation control group,
  and the τ action mask.
- **walk_risk / P4 (contrastive re-identification probe)** — P(attacker picks the original out
  of a same-type anonymity set | context + visible fill), via length-normalized causal-LM
  log-probabilities (pythia-410m) softmaxed over {original} ∪ ≤15 same-type corpus distractors.
  Candidate-sensitive; ~50 ms/item; precomputable per (span, level).
- **A / P6 (embedding-proximity score)** — the reward's privacy term:
  `cos_MiniLM(fill, original)` per span, mean-aggregated. Candidate-sensitive, context-blind.
- **Restated-span probe / u_qa / fact recall** — QA probes on gold-restated surfaces; u_qa reads
  doc_p (training utility), fact recall reads out_final (realized ground truth).
- **Echo / absorption** — whether the remote model reproduces a fill in out_p (echo) or uses its
  information with no surface trace (absorption). Measured to be dominated by *task relevance*
  (does the output need the fact), not fill form — hence **unpriced by the training reward**
  (decision 2026-07-04, §5.2); the surrogate-vs-realized gap measures it at evaluation.
- **frontier_claim** — the experiment's verdict: does the trained policy's Pareto dominate the
  τ-walk's at matched *realized* privacy (frontier-LLM attacker) on held-out docs. Never a
  training signal.
- **τ / α** — τ: hard per-span risk ceiling (action-mask boundary, structural guarantee);
  α: the method's operating-point knob (training-time privacy weight; one α = one policy).
- **E0/E1/E2** — environment versions (§3.4). A policy is valid only for its training version.

## 1. Objective and verdict

Train the substitutor to maximize round-trip task utility at a target privacy level, with **zero
remote calls during training**:

```
r(doc_p, R) = α · (1 − A(doc_p, R)) + (1 − α) · U(doc_p, R)
```

The trained ranker is an **amortized, distilled probe + trade-off function** — it can only learn
the privacy notion the reward shows it (probe quality is the ceiling on policy quality; the
promotion rule for probes is measured correlation with a real attacker, § 4.2). Evaluation is
always the real round trip; the output of the whole experiment is `frontier_claim`:

```
if policies dominate walk at matched realized privacy:  METHOD WORKS → report; ceiling study next
elif policies ≈ walk:                                    disambiguate by error analysis:
    remote-phrasing residual dominates → upgrade to round-trip reward (documented trigger)
    else                               → selection learning isn't the lever; report the null
if realized privacy clusters across α:                   α is a bad knob → lexicographic fallback
if attacker succeeds via fills A scored safe:            A is the gap → doc-level head (E2), retrain
```

`frontier_claim` feeds gradients to nothing — routing the eval attacker into training would
Goodhart the evaluation.

## 2. Pipeline (normative pseudocode)

### Phase 0 — offline, once per corpus (environment + reward machinery)

```python
# --- 0a. environment (per document) ---
for doc in corpus:
    spans[doc] = detect(doc)                               # frozen detector (§3.2)
    for s in spans[doc]:
        levels[s] = lattice(s)                             # NLI-truthfulness-gated (gaps plan Phase 2)
        for l in levels[s] + [PLACEHOLDER(s.type)]:
            walk_risk[s, l] = P4(sent(s), s.orig, l)       # precomputed table, ~50 ms × ~20/doc
        legal[s] = [l for l in levels[s] if walk_risk[s, l] < tau] or [PLACEHOLDER(s.type)]
        #                                                          ^^^^^^^^^^^^^^^^^^^^^^^
        # τ-mask: structural floor — an over-τ level is unsampleable, not merely penalized.
        # The `or [PLACEHOLDER]` branch is NOT an eval-only detail: it is what keeps the
        # action space non-empty in TRAINING, the safety floor at DEPLOYED INFERENCE (the
        # trained ranker samples from this same mask), the thing that keeps pi_0's clone
        # teacher τ-clean, and the eval baseline's exhaustion rule. See §3.3-2.

# --- 0b. reward machinery (fully local; no remote calls anywhere in Phase 0b/1) ---
for doc in train_split:
    qa_probes[doc] = teacher_questions(R_surfaces ∩ gold(doc))   # cached; train/held-out split
pi_0 = behavior_clone(tau_walk decisions)                  # never RL from random
train_docs = [d for d in corpus if qa_probes[d]]           # probe-less docs: eval only

# --- 0c. validation gate (go/no-go BEFORE any training run) ---
assert constructed_arms_gate(reward, fact_recall_on_out_final) is positive   # §6
```

### Phase 1 — stage-1 ranker training (per α; minutes; no remote calls)

```python
for epoch, doc in training:
    for g in 1..G:
        a_g = sample(pi, legal)                            # one action per span, inside the mask
        doc_p_g, R_g = assemble(doc, a_g)                  # injectivity enforced here (§3.3)

        # privacy term — P6:
        A_g = mean( cos_MiniLM(fill(a_g[s]), s.orig)
                    for s in spans[doc] if mode(a_g[s]) != generic_placeholder )
        # generic placeholders: label carries no original-signal → excluded (constant)

        # utility term — u_qa, uniform over ALL train-split probes (§5.2; no echo prior,
        # no mode routing — decision 2026-07-04):
        U_g = mean( F1(invert(reader(q̃_j, doc_p_g), R_g), a_j)
                    for j in qa_probes[doc].train_split )
        # placeholder-covered probes score through the same reader path: usually 0
        # (reader abstains on <TYPE_n>), sometimes 1 (token extracted → inversion
        # restores the surface) — a measured conservative bias, accepted (§5.2)

        r_g = α * (1 - A_g) + (1 - α) * U_g
    adv = (r - mean(r)) / std(r)                           # group-relative advantage
    pi.update(REINFORCE(adv) + KL(pi ‖ pi_0))
# α sweep (~3 values) → 3 policies = 3 operating points
```

### Phase 2 — stage-2 joint GRPO (infiller unfrozen; E1+ only)

As Phase 1, plus fills `y ~ p_φ(· | doc, span, ℓ)` sampled under the **injectivity-constrained
decoder** (resample/next-beam on used-set collision; generic placeholder terminal); shared scalar
advantage on ranker log-probs + infiller token log-probs (PPO-clipped), LoRA update of φ.
P6 and P4 are then computed on the *generated* fill (no precompute; ~50 ms/span-fill, batched).

### Phase 3 — evaluation (once per α sweep; the only remote-heavy stage)

```python
for policy in trained_policies + [tau_walk]:               # the walk = control group
    for doc in HELD_OUT docs:
        doc_p, R  = policy(doc)
        out_p     = RemoteLLM(task_prompt(doc_p))          # real round trip, cached
        out_final = extract(out_p, R)                      # deployed extractor (§3.4)
        utility  += fact_recall(out_final, held_out_probes)         # headline utility
        privacy  += 1 - frontier_attacker_success(doc_p)            # REAL attacker — never P4/P6
        leakthrough += attacker_success(out_final)                  # mandatory second axis
frontier_claim = pareto(policies) vs pareto(tau_walk)      # at matched REALIZED privacy → §1 verdict
```

**Honesty boundaries:** P6/P4 never appear in Phase 3 (training's teacher must not grade its own
student); held-out docs, held-out probe split, held-out attacker; second-remote-model arm per the
training plan.

## 3. Environment

### 3.1 Episode and action space

One episode = one document (contextual bandit; no multi-step dynamics in v0). Per detected
quasi-identifier span, the action set is `legal[s]` — τ-legal lattice levels ∪ generic typed
placeholder. Placeholder is a first-class action, not only a fallback: pricing it lets the
policy trade risk against locally-verifiable utility per span. Measured properties, stated
precisely: **inversion given echo is perfect** for placeholders (`ph_swapped` 9/9, zero residue
— every token that appeared in out_p swapped back cleanly), but **echo itself is
task-relevance-dependent, not form-guaranteed** (sampled docs: 1/3 and 0/4 placeholder tokens
appeared in out_p; a prose fill the task needed, "the autumn", echoed while person placeholders
a reply had no reason to restate did not).

**Placeholder modes** (different privacy contracts):
- *Generic typed* (`<DATETIME_1>`): label depends only on the detected type — **risk 0 by
  construction**, exempt from τ, the only legal invariant-fallback.
- *Descriptive/minted* (`<MEETING_DATE_1>`; E1, infiller-chosen): a generalization in
  placeholder syntax — carries role semantics and inherits the clean inversion-given-echo
  property of the `<…>` syntax, but **loses the τ-exemption** (P4-scored like any fill;
  used-set/indexing applies to the label namespace).
- Action spectrum by disclosed semantics: generic placeholder (risk 0, no semantics) →
  descriptive placeholder (role semantics, τ-gated) → relational fill (cross-fact structure,
  joint leakage — §4.2) → naturalistic fill (full prose semantics, τ-gated).

Direct identifiers (PERSON, CODE): forced generic placeholder, outside the action space.
**No KEEP action** — the per-span privacy term cannot price kept spans, so KEEP would be a
reward-hacking channel; revisit only under a document-level A (E2).

### 3.2 Frozen components

The **detector** stays outside the policy: a missed span costs the reward nothing, so gradients
would teach under-detection; detection recall is the reported privacy ceiling. Measured reminder
of what this costs: the dominant attacker-recovery channel in the shootout examples is *retained
context* — undetected sibling mentions ("gastroenterologist" cleartext beside a substituted
"gastroenterology") — recorded in
[detection-sibling-mention-leak](../../issues/detection-sibling-mention-leak.md); a leak channel
no amount of level-selection training can close.

### 3.3 Environment invariants (constraints, never learned behaviors)

Sparse probe reward cannot guarantee structural properties; one violation is silently
unrecoverable. Enforced at assembly/decoding time:

1. **Injectivity of R** — the infiller *resolves* collisions (context-aware fill choice); the
   environment *guarantees* injectivity; RL never learns the invariant and the reward never
   prices collisions (they cannot occur). Per environment version:
   - **E0 (now, stage-1):** used-set in the walk — a level claimed by a different surface is
     masked out; exhaustion → generic placeholder. Required *before* stage-1 RL even though the
     infiller will later own resolution: E0 fills are static lattice strings (nothing resolves
     yet), and both the training environment and the τ-walk control group must be injective.
   - **E1 (infiller):** constrained-decoding wrapper (non-RL infrastructure, ships with the
     infiller build): sample fill → canonical form claimed by another surface → resample /
     next beam → generic placeholder terminal. RL optimizes *which legal fill*; the wrapper
     guarantees legality. Pre-registered smoke measurement: resample-convergence / fallback
     rate — a high fallback rate indicates an infiller diversity (SFT-data) problem, not an
     RL problem.
   - Repeat mentions/coref chains reuse their own replacement (still injective).
2. **τ as a hard ceiling** — `walk_risk[s, l] ≥ τ` ⇒ `l ∉ legal[s]`; exhaustion → generic
   placeholder (risk 0). *Fixes the measured floor violation* ("Commission" shipped at risk
   0.147 against τ = 0.02). α, not τ, is the operating-point knob. **The exhaustion fallback is
   load-bearing in four places, only one of which is evaluation**: (a) the *training
   environment* — `legal[s]` must be non-empty when no level passes τ, and the generic
   placeholder is that guaranteed-legal action (otherwise: empty action space, or re-admitting
   over-τ levels); (b) *deployed inference* — the trained ranker also samples only from
   `legal[s]`, so when `legal[s] = {placeholder}` the fallback operates through the mask (the
   ranker replaced the walk's *choice*, never the legality boundary); (c) the *behavior-clone
   init* — a walk that ships τ-violations teaches violations to `pi_0`; (d) the *eval control
   group*. τ-scale note: under the contrastive probe, risk is re-identification probability in
   a ~16-candidate set (chance ≈ 0.0625), so meaningful τ sweeps live in roughly
   [0.0625, 0.3] — a legitimate knob re-sweep, not a calibration.
3. **Truthfulness as a constraint (generate-then-verify)** — **the reward cannot price
   truthfulness**: u_qa is invariant to fill semantics (inversion restores originals
   regardless) and A prices only proximity-leak, so a false-but-safe-and-extractable fill is
   reward-optimal while feeding the remote model false premises. Truthfulness therefore lives
   in the environment as a verifier: the NLI entailment gate (premise = original sentence,
   hypothesis = sentence with the fill; keep iff entailed). E0: all lattice sources pass the
   gate (**shipped 2026-07-04** — previously rule-sourced lattices bypassed it; measured on the
   motivating examples: "vermont" → "a city in Australia" now rejected; "dragon" → "a mythical
   monster" and "washington" → "a city in DC" pass — the word-sense ceiling). E1: the same gate joins the
   infiller's decode loop — sample fill → injectivity check → τ check → NLI check →
   accept / resample / placeholder. The infiller *proposes* (context-aware), the gate
   *verifies*; RL optimizes within the verified-legal set. Known limit: NLI is a cheap
   verifier, not a semantics oracle (word-sense errors may pass as lexical hypernymy) — the
   residual and the proposed truthfulness-reward extension are tracked in
   [rule-lattice-nli-gate-bypass](../../issues/rule-lattice-nli-gate-bypass.md).
4. **Extractor scope** — the reward calls the *deployed* `invert()` (u_qa inverts the reader's
   answer through R with the same implementation used at eval). Because reader answers are
   doc_p substrings where replacements appear verbatim, the exact-match path dominates — but
   the fuzzy branch *can* fire on clipped/partial answer spans, so extractor changes have a
   small, nonzero effect on the training reward, not only on eval. Consequence: **the extractor
   version is pinned for the whole (gate → training → eval) cycle** — any extractor change
   re-gates and invalidates trained policies, same rule as a reward change.
5. **Determinism via artifacts** — detection is nondeterministic across processes (measured:
   3/6 clinical doc_p hashes differ between identical runs); all consumers load the persisted
   arms artifact (`scripts/build_arms_artifact.py` → `data/task_arms_tau0.02.json`), never
   re-detect.

### 3.4 Environment versions

| | fills | extractor (`invert`) | status |
|---|---|---|---|
| **E0** | static lattice strings ∪ generic placeholder | rule exact/fuzzy-90 | **live** (gaps-plan Phases 1–2 shipped 2026-07-04) |
| **E1** | infiller (SFT flan-t5 + LoRA), injectivity-constrained decoding; descriptive/relational fills | E0 + light fuzzy-verify | infiller to build |
| **E2** | E1 | learned reconstructor; **doc-level attack head** (frozen encoder + heads on SynthPAI attributes) | escalation |

The E1 *semantic aligner* was descoped from the critical path: absorption dominates `gen_absent`
(82–95% of prose fills leave no trace in out_p; loose-echo headroom < 10% and partly spurious) —
what the remote model absorbs cannot be won back at extraction time. See
[remote-llm-echo-absorption](../../issues/remote-llm-echo-absorption.md).

## 4. Probes

### 4.1 QA probes (utility axis)

Construction per document, once, cached: unique R surfaces (dedup only — no role filter; the old
one was a measured bug) → gold-restatement match (canonicalized exact, then fuzzy ≥ 85) → one
teacher-written natural question per surface (cloze is reader-OOD, measured). Anti-Goodhart:
per-doc held-out probe subset scored only at evaluation. Supply (16-doc slices): 3.19/doc
clinical, 1.0 enron, 0.38 aeslc; probe-less docs excluded from training. Why probes and not
reference overlap: detected spans cover 6–8% of gold tokens — whole-output similarity is noise
(shuffled-gold floor 1.4–2.9×); probes make the sensitive fraction ≈ 1.

### 4.2 Privacy probes (candidate-sensitive; promotion by attacker correlation)

A probe is promoted only by measured correlation with a real LLM attacker
([attacks.md § shootout](../attacks.md)); it is a training signal, **never reported privacy**.
Shootout verdict (150 (span, level) items, two referees — local Qwen3.6 and frontier
gemini-3.1-pro; ranking stable under both):

| | mechanism | AUC (Qwen / gemini-parsed) | level-ordering | role |
|---|---|---|---|---|
| mask-away MTI (legacy) | candidate-invariant | — | — | **disqualified**: identical score for every level (degenerate walk, zero RL gradient) |
| appositive MLM (P2) | slot masked, fill visible | .57 / .57 | .50 / .33 | **failed** — chance-level |
| multi-mask PLL (P3) | k masks + visible fill | .68 / .68 | .79 / .57 | reserve |
| **contrastive re-id (P4)** | anonymity-set softmax, pythia-410m | .64 / .63 | **.86 / .71** | **walk_risk + τ-mask** — the walk's decision is within-span level ordering, P4's measured win; risk semantics match the τ threshold (probability of re-identification) |
| **embedding proximity (P6)** | cos_MiniLM(fill, original) | **.83 / .77** | .79 / .64 | **reward A** — doc-level discrimination, P6's measured win; ~free |

Blending loses on both metrics under both referees (combo AUC ≤ P6, level-ordering ≤ both) —
probe-per-job, never mixed scores. Referee severity note: gemini hit@5 ≈ 0.49 on parsed items —
at τ-walk operating points a frontier attacker recovers ~half the probed fills it engages.

**Known gaps of the chosen pair** (priced at eval, pre-registered responses):
P6 is context-blind (a context that pins the value raises real risk but not P6) and
embedding-gameable by a trainable infiller — stage-2 requires either a P4 term joining A or the
E2 doc-level head; P4's distractor pools are corpus-empirical (thin for DEM/QUANTITY);
both are per-span — **joint leakage of relational fills is invisible until E2's document head**.

## 5. Reward

### 5.1 Privacy term A (P6)

`A = mean over substituted spans of cos_MiniLM(fill, original)`; generic-placeholder spans
excluded (constant, no gradient). Max reported as diagnostic. Blind spots and escalation
triggers: §4.2. The real privacy measure remains the frontier attacker (Phase 3) — if it
succeeds through fills A scored safe, A escalates (E2) and the policy retrains.

### 5.2 Utility term — u_qa, uniform reader path, no echo prior

```
u_j = F1( invert( reader(q̃_j, doc_p), R ),  a_j )     for EVERY train-split probe j
U   = mean_j u_j
```

One path for all fill modes: question generalized through R → SQuAD2 reader on doc_p → answer
inverted through R → token-F1 vs the original surface. Properties:

- **Coarsening invariance**: invertible coarsening scores 1.0 ("Oslo" → "a Norwegian city" →
  inverted → F1 1.0); keeping the original earns nothing — under-anonymization is not rewarded.
  Utility falls exactly for destruction: unanswerable, unalignable, or context-wrecking fills.
- **Placeholder-covered probes score through the same reader**: usually 0 (SQuAD2 null-answer
  abstains on `<TYPE_n>` tokens), occasionally 1 (the reader selects the token and inversion
  restores the surface). Net effect is a **measured conservative bias**: the all-placeholder arm
  scores u_qa 0.107 vs tau_walk 0.238 on clinical — placeholder carriage is under-credited
  because it is *locally unverifiable* (it depends on the remote model echoing the token).
  Accepted as fail-closed: the reward credits only locally-verifiable carriage; whatever
  placeholders deliver beyond that shows up in the surrogate-vs-realized gap at evaluation.

**Decision 2026-07-04 — the echo factor is deliberately NOT priced (ŝ dropped).** An earlier
design multiplied u by an offline-measured echo-survival table ŝ(mode, type, task). Dropped on
two grounds: (1) *re-binding* — ŝ is remote-model behavior conditioned on our prompts/tasks,
importing into the detached surrogate exactly the dependence it exists to avoid; (2) *the
evidence for action-dependence collapsed on inspection* — cached round trips show echo is
dominated by **task relevance**, not fill form (a reply restates the timeline, so prose "the
autumn" echoes; it has no reason to restate names, so `<PERSON_4>` does not; the earlier
"placeholders echo 9/9" statistic measured inversion-given-echo, not echo). An effect outside
the policy's control does not belong in its reward. Consequences: the reward is fully local and
remote-free with zero measured environment constants from the remote model; the echo channel is
measured only at evaluation (fact recall on out_final), and if the surrogate-vs-realized gap is
dominated by echo effects the policy *could* have influenced, that is the documented trigger
for the round-trip reward upgrade — not for reintroducing ŝ.

- **u_nli is dropped** (measured: mixing it degrades gate agreement 0.367 → 0.183); may return
  as a diagnostic only.

### 5.3 Anti-Goodhart controls

KL leash to the behavior-clone init; low optimization pressure (small G, few epochs between
re-gates); held-out probe split; held-out corpora/attacker at eval; Gao overoptimization playbook
([adverserial-RL.md](../../research/adverserial-RL.md)). Reward climbing while realized checks
fall = stop and report.

## 6. Baseline and validation gate

The τ-walk must be non-degenerate **before** RL because it is: the control group of
`frontier_claim` (a broken baseline inflates the learned method — forbidden), the
behavior-clone teacher of `pi_0`, and the source of the τ-mask and constructed arms. Its two
measured degeneracies (candidate-invariant probe → binary walk; floor fallback → τ violations)
were fixed by gaps-plan Phases 1–2 (shipped 2026-07-04: max shipped generalization risk 0.0182
< τ = 0.02 over all 618 gen entries of the 48-doc arms artifact; injectivity verified;
intermediate-level selections now occur — 21/63 multi-level spans on the acceptance sample vs
0 under the old probe).

**Gate (before any training run):** constructed arms (no_privacy / tau_walk / all_floor /
suppression) per doc → per-doc Spearman between the **utility term's** ordering (U = u_qa) and
realized **fact recall on out_final**. The gate validates the utility axis only: realized fact
recall measures no privacy, so correlating the *mixed* reward r against it is not meaningful
(the α-weighted privacy term legitimately reorders arms); the privacy term is validated
separately by the attacker-correlation shootout ([attacks.md](../attacks.md)). r~realized is
reported as context, never as the criterion. Go: clearly positive U~realized agreement where
the ground truth orders arms sanely.
Reported diagnostic (not go/no-go): the reward's placement of an all-placeholder arm vs realized
— expected to under-credit it (the fail-closed bias of §5.2); the size of that gap is the
standing estimate of what the echo channel costs the surrogate. Re-run on every change to
reward composition, probe choice, prompt, or extractor — the gate validates a (reward,
environment) pair. **Status: PASSED 2026-07-04** on the current pair
(`scripts/reward_gate.py` → `results/ranker_reward_gate.json`): U~realized per-doc Spearman
**0.600 clinical (n=12) / 0.725 enron (n=8) / 0.520 aeslc (n=5)** — positive on all corpora,
up from 0.37/0.78/0.44 in the pre-Phase-1/2 environment. Echo diagnostic (all-placeholder arm):
realized fact recall 0.189/0.161/0.308 — **the best protected arm on every corpus, above
no_privacy on 2 of 3** — while the reward's U under-credits it (fail-closed bias, quantified:
U 0.178/0.029/0.208). This is evidence that fill form does matter for *probed* (task-relevant)
facts, contra the raw-absorption reading; the ŝ decision stands (reward stays local), but the
diagnostic is the standing measure of what that costs, and the privacy term's placeholder
preference means the α equilibrium can still reach placeholder-heavy optima the utility term
alone would miss.

## 7. Open tensions

1. **Stage-2 gameability of P6** — an unfrozen infiller can craft embedding-far,
   information-close fills. Pre-registered guard: P4 joins A (or E2 head lands) before stage 2
   unfreezes the infiller.
2. **Opposing placeholder biases in the reward** — the privacy term favors placeholders
   (excluded from A → zero risk contribution) while the utility term under-credits them
   (fail-closed reader path, §5.2). The policy's placeholder rate is therefore an α-governed
   equilibrium between two known biases rather than a calibrated estimate; where it lands, and
   whether realized fact recall agrees, is a first-class evaluation question. If product goals
   additionally demand natural doc_p, naturalness enters the reward as an explicit term — a
   spec change, never a quiet knob.
3. **The echo channel is entirely unpriced** (deliberate, §5.2) — a policy cannot learn to
   prefer fills the remote model will restate, because the reward cannot see restatement.
   Measured evidence says most of that channel is task-relevance (outside policy control); the
   part that is form-dependent, if any, surfaces as surrogate-vs-realized gap at eval and
   triggers the round-trip upgrade, not an ŝ revival.
4. **Probe sparsity** — 3.2/1.0/0.4 probes per doc concentrates training on clinical; expansion
   options (number normalization, teacher fact tuples) queued if it binds.
5. **α as a retrain-expensive knob** — one α = one operating point; clustering of realized
   privacy across α triggers the lexicographic fallback (imported decision).
6. **Upstream leak channel** — detection/coref recall (sibling mentions) is the dominant
   measured recovery path and is out of this spec's scope; tracked as the detector workstream.

## Sources

Gaps and fix order: [2026-07-03-surrogate-rl-gaps-fixes.md](../../plans/2026-07-03-surrogate-rl-gaps-fixes.md).
Training-plan decisions, wall-time, kill criteria:
[2026-07-02-surrogate-grpo-training.md](../../plans/2026-07-02-surrogate-grpo-training.md);
round-trip upgrade path: [2026-07-02-roundtrip-grpo-training.md](../../plans/2026-07-02-roundtrip-grpo-training.md).
Probe shootout protocol + results: [attacks.md](../attacks.md);
echo absorption: [remote-llm-echo-absorption.md](../../issues/remote-llm-echo-absorption.md).
Background: [Gao et al. 2022](../../../research-wiki/papers/gao2022_reward_overoptimization.md)
([arXiv 2210.10760](https://arxiv.org/abs/2210.10760));
[NaPaRe](../../../research-wiki/papers/huang2025_tree_search_rewriting.md)
([arXiv 2509.20838](https://arxiv.org/abs/2509.20838));
[AgentStealth](../../../research-wiki/papers/shao2025_agentstealth.md)
([arXiv 2506.22508](https://arxiv.org/abs/2506.22508)).
