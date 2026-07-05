---
type: reference
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [rl, round-trip-reward, ranker, infiller, expert-iteration, rloo, probes,
       fact-recall, count-floors, gates, anti-goodhart, spec]
companion: [docs/plans/2026-07-05-roundtrip-rl-strategy.md,
            docs/specs/RL/surrogate-ranker-infiller.md,
            docs/specs/benchmarks.md,
            docs/handoffs/2026-07-05-roundtrip-rl-pivot.md]
supersedes: docs/specs/RL/surrogate-ranker-infiller.md §5–§6 (surrogate reward + constructed-arms
  gate) — that spec's §1–§4 (environment, floors, invariants, risk measure) remain normative and
  are incorporated here by reference.
---

# RL specification — round-trip training of the substitutor (ranker + infiller)

Round-trip RL = train the span-level **ranker**, then the generative **infiller**, against the
one real quantity — how many probed sensitive facts survive anonymize → remote task →
re-identify — with expert iteration as the workhorse, corrected policy gradient as the refiner,
and privacy enforced entirely by environment masks, never by the reward. Strategy rationale and
literature basis: [round-trip RL strategy plan](../../plans/2026-07-05-roundtrip-rl-strategy.md).

## Definitions

- **doc_orig / doc_p / out_p / out_final** — original document; anonymized rewrite; the remote
  model's task output on doc_p; the locally re-identified output (`invert(out_p, R)`).
- **R (substitution record)** — client-side `{surface → replacement}` map, injective per doc.
- **Round-trip reward `R_rt`** — realized fact recall on out_final over a doc's train-split
  probes (§ Phase 1). The only training reward; deterministic given doc_p (pinned temp-0
  remote model + cache).
- **Probe** — one QA pair `(q, a)` whose answer `a` is a detected sensitive span restated by
  the gold task output; validated action-sensitive by the ceiling/floor anchor checks
  (§ Phase 0 step 4). Train/held-out split per doc.
- **Anchors** — `out_hi = Remote(task(doc_orig))` (ceiling) and
  `out_lo = Remote(task(all_placeholder(doc)))` (floor); two cached round trips per doc.
- **Ranker π_rank / Infiller π_fill** — stage-1 per-span action selection over the legal
  lattice menu; stage-2 grammar-constrained rendering of the chosen node (E1+).
- **Floor-walk** — the rule baseline and behavior-clone teacher: per span the minimum-aset
  legal level, else generic placeholder (unchanged from the surrogate spec).
- **aset / k_T / legal[s|k]** — anonymity-set count, per-type count floors, and the derived
  legal action set — all unchanged and normative in the
  [surrogate spec](surrogate-ranker-infiller.md) §Definitions, §3, §4.1. Privacy is
  **floors-only**; the reward has no privacy term.
- **ExIt (expert iteration / ReST^EM)** — sample G rollouts, SFT on the realized-reward
  winners, iterate ([arXiv 2312.06585](https://arxiv.org/abs/2312.06585)).
- **RLOO** — REINFORCE with the leave-one-out group mean as baseline
  ([arXiv 2402.14740](https://arxiv.org/abs/2402.14740)); no per-group std division
  (Dr.GRPO correction, [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)); zero-variance
  groups dropped (DAPO tie-filter, [arXiv 2503.14476](https://arxiv.org/abs/2503.14476)).
- **Counterfactual span advantage** — `R_rt(doc, a[s→alt]) − R_rt(doc, a)`: one span's action
  flipped, all others held; exact per-span credit (COMA's baseline made exact by reward
  determinism, [arXiv 1705.08926](https://arxiv.org/abs/1705.08926)).
- **Support scan** — the pre-training gate: single-action counterfactuals from the floor-walk
  baseline must move realized per-probe recall in both directions (§ Gates).
- **frontier_claim** — the Phase-5 verdict: trained Pareto vs floor-walk Pareto at matched
  *realized* privacy on held-out everything. Feeds gradients to nothing.

## Pinned components (one table = the re-gate surface)

Every row is frozen for a whole gate → train → eval cycle; changing any row invalidates the
cache, the gate verdict, and all policies trained under it (§ The one subtlety).

Concrete bindings picked 2026-07-05 for the local hardware (Strix Halo gfx1151, llama-swap
proxy at :8060; family separation across grader/teacher/environment roles):

| component | role | binding |
|---|---|---|
| detector | spans, once, frozen into the arms artifact | GLiNER multidomain fine-tune (`pii_gliner_multidomain` ckpt-2479 @0.3) |
| lattice + NLI gate | generalization candidates, truthfulness-filtered | WordNet/GeoNames sources + frozen NLI entailment gate; hygiene-audited (perplexity screen) before any run |
| `aset_count` + floors k_T | legality mask; the only privacy knob and operating-point knob | deterministic (`cloak.anonymity`), surrogate spec §3.3-2 |
| π_rank | per-span choice from `legal[s|k]` | ModernBERT-base (~150M) doc encoder + per-span action head, autoregressive over spans; trains on the iGPU in `.venv`; 17-feature MLP retained as capacity-ablation arm |
| π_fill (E1) | grammar-constrained rendering | **TBD — decided at Stage-2 kickoff** (user decision 2026-07-05). Constraints that survive the deferral: torch-side in `.venv` (HF logits-processor grammar masks + LoRA cannot live behind llama-swap), ~1–3B trainable on the iGPU, mature HF/PEFT/TRL support. Candidate at time of writing: Qwen3-1.7B-Instruct |
| BC teacher | π_rank init | floor-walk |
| remote task model | executes the task on doc_p; the reward's LLM | **gemma 4 (E4B)**, non-thinking (`enable_thinking: false` — honored, measured: clean ~150-token notes, all probe facts restated; `results/thinking_mode_probe.json`), max_tokens 512, **temp 0**, cache key `hash(model, template, doc_p)`. User re-pin 2026-07-05: LFM2.5-8B-A1B was rejected for this role — it cannot disable thinking (every off-switch fails; the flag leaks `<think>` in-band) and pays ~700 reasoning tokens/call. Go/no-go: Phase-0 ceiling-anchor pass rate per corpus — fallback Qwen3.6-35B-A3B (re-pin ⇒ re-gate) |
| task prompt | the SAME per-corpus template everywhere | `TASK_TEMPLATE[corpus]`, `src/cloak/tasks.py` |
| extractor | `invert(out_p, R)` | rule exact/fuzzy-90, deployed path |
| QA reader | answers probes against out_final | existing frozen u_qa reader (local, batched, deterministic); shares no gradients with anything |
| probe teacher + atomic decomposition | writes probe questions | **LFM2.5-8B-A1B** (user re-pin 2026-07-05; different family than the reward model). Thinking is unconditional for this model — run WITHOUT the `enable_thinking` kwarg (passing `false` leaks `<think>` into content), max_tokens 1024; reasoning is separated server-side, content = the question. One-time, cached. The legacy gemma-authored question cache must be set aside before rebuild (gemma is now the reward model — teacher ≠ reward-model rule) |
| distilled RM | candidate screening ONLY, never gradients | `qwen3-embedding-0.6b` (already served) doc_p embeddings + ridge/GBM regressor on cached (doc_p → realized recall) pairs |
| eval-only | second remote model; frontier attacker | second remote = **Qwen3.6-35B-A3B** (third family, uninvolved in training or probes); frontier attacker = paid frontier model, **requires explicit user approval per standing rule** (local Qwen3.6 attacker for dev-side checks only) |

Operational note (llama-swap): model switches evict and reload — **phase-order the pipeline**
(all LFM2.5 teacher work first, then swap to gemma for anchors + all training; Qwen3.6 only at
eval) so training never alternates served models.

Thinking-mode ruling (measured 2026-07-05, `results/thinking_mode_probe.json`): the reward
model and probe teacher run **without thinking wherever the model allows it** — the tasks are
transduction, reasoning buys nothing measurable and costs 3–5× decode. gemma honors the off
flag; LFM2.5 cannot not-think (acceptable in the cached one-time teacher role only). The
thinking mode of the reward model is part of the pin: flipping it re-gates. The Phase-5
attacker is the explicit exception — it runs at full reasoning strength (a weak attacker
overstates privacy).

## Datasets

| corpus | task | role | access |
|---|---|---|---|
| ACI-Bench (207) + MTS-Dialog (1,701) | dialogue → visit note | clinical training core | in repo |
| Enron mined replies (200 → scale by mining) | email → reply | direct-identifier stress | in repo |
| Multi-LexSum (9,280) | legal case → summary | third domain, real names | open ([arXiv 2206.10883](https://arxiv.org/abs/2206.10883)) |
| MeetingBank / ECTSum | minutes / call bullets | optional fourth domain | open |
| Discharge Me! (68,785) / ProbSum (~1k) | discharge sections / problem list | bulk clinical, later | PhysioNet DUA; remote proxy must be inside the DUA boundary, else local-eval-only |
| SynthPAI · TAB · PersonalReddit · LLM-PBE | — | privacy/attacker axis ONLY | per benchmarks/attacks specs |

Selection criterion unchanged ([benchmarks spec](../benchmarks.md)): the gold output must
restate the quasi-identifiers, else neither inversion nor utility cost is exercised.

## Phase 0 — environment + probes (once per corpus)

1. Detect once, freeze: `spans[doc] = detect(doc)` → arms artifact; **never re-detected**
   (surrogate spec §3.3-5).
2. Menus: `levels[s] = lattice(s) + [KEEP(s)]`; annotate
   `aset[s,l] = aset_count(l, s.type, s.orig, strict=True)`. Legality derived at use time:
   `legal[s|k] = {l : aset[s,l] ≥ k[s.type]} ∪ {PLACEHOLDER(s.type)}` — never empty.
3. Anchors, 2 cached round trips per doc: `out_hi`, `out_lo` (Definitions).
4. Probes, 10–20 per doc:
   ```python
   atoms = atomic_facts(gold(doc))                      # FActScore-style decomposition
   cands = [teacher_question(t) for t in atoms if answer(t) in detected_spans(doc)]
   probes[doc] = [(q, a) for q, a in cands
                  if  f1(reader(q, out_hi), a) >= TH    # ceiling: task+model DO surface it
                  and f1(reader(q, out_lo), a) <  TH]   # floor: it flows through the spans
   train_probes, heldout_probes = split(probes[doc], seed=0)   # persisted
   ```
   - Docs with < 3 surviving train probes are excluded from the RL reward and listed in the
     gate report — never silently kept.
5. `π_rank ← behavior_clone(floor_walk)` under the run's floor regime (never RL from random).
6. Run the **support scan** (§ Gates). No pass → no training run.

## Phase 1 — the reward (per rollout)

```python
def R_rt(doc, actions):
    doc_p, Rmap = assemble(doc, actions)            # injectivity mask enforced here (env, not reward)
    out_p  = Remote(task_prompt[corpus](doc_p))     # pinned model, temp 0, CACHED on hash(doc_p)
    out_f  = invert(out_p, Rmap)                    # deployed extractor, pinned
    return mean(f1(reader(q, out_f), a)             # frozen reader; GRADED token-F1 (the
                for q, a in train_probes[doc])      # deployed fact_recall, v2-gate-certified)
```

The keep/drop threshold TH binarizes only probe *validation* (Phase 0 step 4), never the
reward — the reward stays the graded mean.

Normative rules: utility-only (privacy lives in the mask); no cross-floor averaging anywhere;
the cache is first-class infrastructure (reward memoization = ExIt candidate pool = RM data);
whole-output similarity (ROUGE/BERTScore) never trains — eval-side only (§ Phase 5).

## Phase 2 — Stage A: ranker on E0 (static fills)

```python
# ── workhorse: expert iteration (ReST^EM) ──
for round in 1..5:
    D = []
    for doc in corpus:                              # async workers 16-32, cache-hot
        k     = sample_floors() if randomize else fixed_k      # waived types stay 1
        group = [sample(pi_rank, legal[.|k]) for _ in range(G)]        # G = 16-32
        r     = [R_rt(doc, a) for a in group]
        if max(r) > R_rt(doc, floor_walk(doc, k)):
            D += [(doc, k, group[argmax(r)])]       # keep-the-best only
    pi_rank = SFT(pi_rank, D)

# ── refiner: RLOO from the ExIt checkpoint ──
for step, (doc, k) in batches:
    r = [R_rt(doc, a_g) for a_g in group]
    if len(set(r)) == 1: continue                   # tie-filter: dead group, refill batch
    adv = r - loo_mean(r)                           # NO std division
    for s in sample_spans(doc, frac=0.25):          # exact counterfactual credit (cached)
        adv_span[s] = R_rt(doc, a[s -> placeholder]) - R_rt(doc, a)
    loss = -(adv * logp(a)).mean() - 0.01 * entropy(pi_rank)
    # KL leash OFF at start; re-add a small BC anchor only on observed collapse.
```

Protocols (unchanged from the surrogate spec where named there): fixed-floor run precedes the
floor-randomized run at matched budget; conditioning ablation at held-out floors; first-smoke
movement milestone; G ≥ 2 guard; greedy read-outs at fixed floors on the declared grid (the
grid is the Pareto sample; ≥ 1 waiver-bearing config required). Pre-registered null outcome:
ExIt *and* RLOO both landing on the floor-walk = "selection adds little" — a finding, reported
as such.

## Phase 3 — Stage B: infiller on E1

Prereqs: E1 build contract (grammar artifacts, decode constants, parser round-trip tests —
surrogate spec §3.4) + the lattice hygiene audit.

```python
pi_fill = SFT(base_1to3B, pairs(node -> canonical_rendering))     # grammar-fluent init
for round in 1..K:                                  # rung 1: best-of-n rejection FT
    for doc:
        cands   = [decode(pi_fill | GRAMMAR, injectivity, aset >= k, NLI) for _ in range(n)]
        top     = RM_screen(cands, top=4)           # RM screens; round trip DECIDES
        winner  = argmax(R_rt(doc, ranker_frozen + c) for c in top)
    pi_fill = SFT(pi_fill, winners)
# rung 2: GRPO — token-level logprobs, sequence-level advantage, PPO-clip, LoRA;
#          same group rules as Phase 2 (tie-filter, no std division, KL off).
```

Grammar mask, injectivity, strict online `aset_count`, and the NLI gate stay **environment**,
never reward — proposer/verifier separation is absolute and doubles as the
instruction-injection firewall (§ Risks 4).

## Phase 4 — Stage C: joint (only if A or B moved)

Shared episodic advantage on `logp_rank(a) + Σ logp_fill(tokens)`, PPO-clipped, low lr,
re-gate cadence per § The one subtlety. Joint training of two components that were both flat
alone is compute theater — skip and report.

## Phase 5 — evaluation (the only verdict)

Unchanged machinery: held-out docs + held-out probes; frontier-LLM attacker on doc_p and
leak-through on out_final; per-floor operating points; document-bootstrap dominance test
(surrogate spec §2 Phase 3); second-remote-model arm reports the provider-transfer gap —
reported, never calibrated away. Additions (2026-07-05):

- **Whole-task-quality regression gate** (kill-argument P_2 fix,
  `review-stage/KILL_ARGUMENT.md`): at matched realized privacy, ROUGE-L / BERTScore /
  entity-F1 on out_final for the trained policy must not regress vs the floor-walk control
  (document-bootstrap, same test as dominance). Catches probe-reward Goodharting on unprobed
  quality (coherence, omissions, non-sensitive facts).
- **External baselines through the identical harness at matched realized privacy:**
  adversarial anonymization ([arXiv 2402.13846](https://arxiv.org/abs/2402.13846)),
  AgentStealth ([arXiv 2506.22508](https://arxiv.org/abs/2506.22508)), DP rewriting
  (DP-Prompt [arXiv 2310.16111](https://arxiv.org/abs/2310.16111), DP-BART
  [arXiv 2302.07636](https://arxiv.org/abs/2302.07636)), legacy InferDPT
  ([arXiv 2310.12214](https://arxiv.org/abs/2310.12214)).

## Gates (before any training run; report format pre-registered)

1. **Round-trip support scan** — THE training gate (handoff-mandated): from the floor-walk
   baseline, ~100 single-action counterfactuals → cached round trips → per-probe realized
   recall deltas. Pass = flips exist in BOTH directions and per-swap deltas exceed the
   quantization step on ≥ 1 corpus-representative subset. A support desert is a finding about
   the environment, reported, never worked around.
2. **Probe health report** — per corpus: probes/doc (mean/min), ceiling/floor rejection rates,
   docs excluded (< 3 train probes), reader spot-check error rate on `out_hi`.
3. Re-run both on ANY pinned-component change. Pass criteria name the failing clause; no
   partial credit.

## The one subtlety

Everything hangs on `R_rt` being **deterministic given doc_p**. That single property makes the
cache a shared substrate (reward memoization = ExIt pool = RM training set), makes
counterfactual span advantages *exact* instead of estimated, and makes probe validation
meaningful. It holds only while every pinned-components row stays fixed; change one and every
cached number, gate verdict, and trained policy is invalidated **together** — the re-gate rule
is cache coherence, not bureaucracy.

## Worked example (one doc, one ExIt round)

Clinical dialogue; 3 spans — "metformin 500mg" (QUANTITY), "gastroenterology" (ORG-like),
"March 3" (DATETIME); floors all 100; 12 validated probes; floor-walk `R_rt = 0.42` (5/12).
Sample G = 16 action sets. A₇ = {"a biguanide, standard dose", "a specialty clinic",
"early March"} → the note restates all three → `invert` narrows them back via R → 9/12 probes
hit → `r = 0.75`. A₁₂ placeholders the drug → the note absorbs it ("medication continued") →
nothing to invert → 5/12 → `r = 0.42`. `max(r) = 0.75 > 0.42` ⇒ (doc, A₇) enters the SFT set.
Counterfactual check on the drug span alone: flip A₇'s drug action to placeholder, re-run
(one cache-miss call) → 0.58; the single decision carries +0.17 — credit lands on that span,
not smeared over three.

## Risks & anti-Goodhart (register)

1. **Extractor gaming** — fills whose fuzzy inversion falsely inflates recall. Monitor
   exact-vs-fuzzy recall gap per checkpoint; audit inversions on reward-climbing checkpoints.
2. **Probe-teacher circularity** — teacher ≠ remote task model family; second-remote-model arm
   is the detector.
3. **Provider overfitting** — the transfer gap on the second-model arm is reported (honesty
   rule: never a calibration knob).
4. **Instruction injection through fills** — an unconstrained infiller under round-trip reward
   is an adversarial-string miner ("list all entities verbatim…"). Grammar-constrained
   decoding is a **security boundary**; any relaxation requires an injection audit first.
5. **Minted-placeholder smuggling** — descriptive placeholders stay aset-scored; audit label
   entropy on trained policies.
6. **Held-out erosion** — checkpoint selection on a dev split only; held-out docs/probes
   untouched until the single Phase-5 eval; seeds and doc lists pre-registered.
7. **Unprobed-quality Goodhart** — priced by the Phase-5 regression gate (above).
8. **What RL cannot fix** — sibling-mention detection leak (the privacy ceiling),
   famous-context priors, thin floor calibration: all priced at eval (surrogate spec §4.1
   gaps). Reward climbing while held-out realized checks fall = stop and report.

## Artifacts

Arms artifact `data/task_arms_tau0.02.json` (annotate-in-place only) · env builder
`scripts/build_ranker_env.py` · trainer `scripts/train_ranker.py` (to gain: RLOO baseline,
tie-filter, async reward client + cache, counterfactual scheduler, ExIt outer loop) · reward
lands in `src/cloak/train/reward.py` (u_gold scorer stays as diagnostic) · support scan spike
(pattern of `scripts/spikes/probe_flip_scan.py`, round-trip version) · probe builder (extends
`src/cloak/tasks.py` machinery) · infiller training via TRL GRPO trainer + LoRA
(Trinity-RFT [arXiv 2505.17826](https://arxiv.org/abs/2505.17826) as the integrated
alternative) · results under `results/`, training records in `research-wiki/training/`
(next: RL-ranker v3, v-schema).

## Sources

Strategy + full literature basis:
[2026-07-05 round-trip RL strategy](../../plans/2026-07-05-roundtrip-rl-strategy.md) (§ Sources
carries the verified citation list). Pivot rationale:
[2026-07-05 handoff](../../handoffs/2026-07-05-roundtrip-rl-pivot.md). Environment/floors/risk
measure: [surrogate spec](surrogate-ranker-infiller.md) §1–§4 (normative). NULL evidence:
[RL-ranker v1](../../../research-wiki/training/2026-07-04-RL-ranker-v1-stage1-bandit.md) ·
[RL-ranker v2](../../../research-wiki/training/2026-07-05-RL-ranker-v2-stage1-floor-env.md).
Benchmark-as-reward adjudication: `review-stage/KILL_ARGUMENT.md`.
