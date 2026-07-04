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
- **ŝ (echo-survival table)** — offline-measured
  `P(fill recoverable in out_p by the deployed extractor | fill mode, span type, task)`; a
  pre-registered environment constant.
- **Restated-span probe / u_qa / fact recall** — QA probes on gold-restated surfaces; u_qa reads
  doc_p (training utility), fact recall reads out_final (realized ground truth).
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
        # τ-mask: structural floor — an over-τ level is unsampleable, not merely penalized

# --- 0b. reward machinery ---
for doc in train_split:
    qa_probes[doc] = teacher_questions(R_surfaces ∩ gold(doc))   # cached; train/held-out split
s_hat = measure_echo_survival(mode, type, task)            # ~200 cached remote calls, one-time,
                                                           # pre-registered; re-measure ⇒ re-gate
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

        # utility term — mode-aware, ŝ-discounted (§5.2):
        U_g = mean( s_hat[mode(a_g[j]), type(j), task] * carry_j
                    for j in qa_probes[doc].train_split )
        # carry_j = 1                                    placeholder-syntax fills
        #         = F1(invert(reader(q̃_j, doc_p_g)), a_j)  prose fills

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
placeholder. Placeholder is a first-class action, not only a fallback: it is the substitution
mode with measured round-trip survival (`ph_swapped` 9/9, zero residue), and pricing it lets the
policy trade naturalness vs survival vs risk per span.

**Placeholder modes** (different privacy contracts):
- *Generic typed* (`<DATETIME_1>`): label depends only on the detected type — **risk 0 by
  construction**, exempt from τ, the only legal invariant-fallback.
- *Descriptive/minted* (`<MEETING_DATE_1>`; E1, infiller-chosen): a generalization in
  placeholder syntax — carries role semantics, keeps the echo-anchor property, **loses the
  τ-exemption** (P4-scored like any fill; used-set/indexing applies to the label namespace).
  Echo survival of descriptive labels is extrapolated from n=9 — smoke-measure before E1 relies
  on it.
- Action spectrum by anchor-ness: generic placeholder (risk 0, echo ~certain, no semantics) →
  descriptive placeholder → relational fill (joint leakage, §5.1) → naturalistic fill
  (echo ~5% measured in E0).

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

1. **Injectivity of R** — E0: a level claimed by a different surface is masked out; lattice
   exhausted → generic placeholder. E1: infiller resamples under a per-doc used-set; generic
   placeholder terminal. Repeat mentions/coref chains reuse their own replacement.
2. **τ as a hard ceiling** — `walk_risk[s, l] ≥ τ` ⇒ `l ∉ legal[s]`; exhaustion → generic
   placeholder (risk 0). *Fixes the measured floor violation* ("Commission" shipped at risk
   0.147 against τ = 0.02). α, not τ, is the operating-point knob.
3. **Lattice truthfulness** — all lattice sources pass the NLI gate; rule-sourced lattices
   (WordNet/GeoNames/buckets) currently bypass it — measured context-wrong fills ("dragon" →
   "a mythical monster", "vermont" → "a city in Australia"); recorded in
   [rule-lattice-nli-gate-bypass](../../issues/rule-lattice-nli-gate-bypass.md), fix scheduled
   with the gaps plan's Phase 2 batch.
4. **Reward–deployment extractor consistency** — the reward's view of the extractor is the ŝ
   table, measured under the extractor actually deployed at gate/eval. (The reward's own
   `invert()` only ever runs the trivial exact path — reader answers are doc_p substrings — so
   extractor upgrades change gate/eval, never u_qa; the consistency requirement lives entirely
   in ŝ.)
5. **Determinism via artifacts** — detection is nondeterministic across processes (measured:
   3/6 clinical doc_p hashes differ between identical runs); all consumers load the persisted
   arms artifact (`scripts/build_arms_artifact.py` → `data/task_arms_tau0.02.json`), never
   re-detect.

### 3.4 Environment versions

| | fills | extractor (`invert`) | status |
|---|---|---|---|
| **E0** | static lattice strings ∪ generic placeholder | rule exact/fuzzy-90 | runnable after gaps-plan Phases 1–2 |
| **E1** | infiller (SFT flan-t5 + LoRA), injectivity-constrained decoding; descriptive/relational fills | E0 + light fuzzy-verify | infiller to build |
| **E2** | E1 | learned reconstructor; **doc-level attack head** (frozen encoder + heads on SynthPAI attributes) | escalation |

The E1 *semantic aligner* was descoped from the critical path: absorption dominates `gen_absent`
(82–95% of prose fills leave no trace in out_p; loose-echo headroom < 10% and partly spurious) —
the echo factor is won at fill-choice time (anchor modes + ŝ), not extraction time. See
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

### 5.2 Utility term (mode-aware, ŝ-discounted)

```
u_j     = ŝ(mode_j, type_j, task) · carry_j
carry_j = 1                                     placeholder-syntax fills
        = F1(invert(reader(q̃_j, doc_p)), a_j)   prose fills
U       = mean_j u_j
```

- **Prose fills** take the reader path (question generalized through R → SQuAD2 reader on doc_p
  → answer inverted → token-F1): semantic readability + R-invertibility. Coarsening invariance
  holds: invertible coarsening scores 1.0; keeping the original earns nothing —
  under-anonymization is not rewarded.
- **Placeholder-syntax fills skip the reader** — measured SQuAD2 null-abstention on placeholder
  tokens under-scores the realizedly best-surviving mode ~2× (all-placeholder u_qa 0.107 vs
  tau_walk 0.238 on clinical) while inversion restores F1 = 1.0 whenever the reader does answer.
  Their carriage is mechanical (echo + swap-back) ⇒ `carry ≡ 1`, weight = ŝ.
- **ŝ multiplies both paths** — the reward's only view of the echo factor (absorption dominates
  realized loss and u_qa is structurally blind to it). Pre-registered; re-measured only on
  environment change (prompt, extractor, remote model); every re-measurement re-gates.
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
are fixed by gaps-plan Phases 1–2.

**Gate (before any training run):** constructed arms (no_privacy / tau_walk / all_floor /
suppression) per doc → per-doc Spearman between the reward's ordering and realized **fact recall
on out_final**. Go: clearly positive agreement where the ground truth orders arms sanely, **and**
the reward separates placeholder-mode from naturalistic-mode arms in the realized direction.
Re-run on every change to reward composition, probe choice, ŝ, prompt, or extractor — the gate
validates a (reward, environment) pair. Status: ground truth validated 2026-07-03
(factrecall~u_qa 0.37/0.44/0.78); the P6+ŝ reward composition is **not yet gated** — that re-run
is a Phase-4 exit criterion in the gaps plan.

## 7. Open tensions

1. **Stage-2 gameability of P6** — an unfrozen infiller can craft embedding-far,
   information-close fills. Pre-registered guard: P4 joins A (or E2 head lands) before stage 2
   unfreezes the infiller.
2. **Placeholder-heavy optima** — given measured survival asymmetry, the policy may converge on
   placeholder-almost-everywhere; a legitimate optimum of the stated reward. If product goals
   demand natural doc_p, naturalness enters the reward as an explicit term — a spec change,
   never a quiet knob.
3. **Probe sparsity** — 3.2/1.0/0.4 probes per doc concentrates training on clinical; expansion
   options (number normalization, teacher fact tuples) queued if it binds.
4. **α as a retrain-expensive knob** — one α = one operating point; clustering of realized
   privacy across α triggers the lexicographic fallback (imported decision).
5. **Upstream leak channel** — detection/coref recall (sibling mentions) is the dominant
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
