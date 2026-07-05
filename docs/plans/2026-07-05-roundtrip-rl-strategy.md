---
type: plan
status: current
created: 2026-07-05
updated: 2026-07-05
tags: [rl, round-trip-reward, ranker, infiller, grpo, rloo, expert-iteration,
       credit-assignment, data-scaling, anti-goodhart, strategy]
companion: [docs/specs/RL/surrogate-ranker-infiller.md,
            docs/handoffs/2026-07-05-roundtrip-rl-pivot.md,
            docs/specs/benchmarks.md]
---

# Round-trip RL strategy — optimal training of ranker + infiller for real-world tasks

Design exercise (user-mandated 2026-07-05): given everything measured so far — two stage-1
NULLs, the reward-landscape account, the surrogate-reward pivot — devise the best-practice RL
strategy for the ranker and infiller under an **ample-compute assumption**. This plan is the
strategy; the RL spec (`docs/specs/RL/surrogate-ranker-infiller.md`) remains the normative
contract for everything it pins (floors-only privacy, proposer/verifier separation, honesty
boundaries, gates). Where this plan proposes changes to spec-pinned machinery, it says so
explicitly.

## Definitions

- **Round-trip reward** — realized fact recall on `out_final` over train-split probes, where
  `out_final = invert(Remote(task_prompt(doc_p)))`; the reward adopted by the 2026-07-05 pivot.
- **RLVR (RL from verifiable rewards)** — the LLM-RL regime where the reward is a deterministic
  program over the model's output (math checkers, unit tests), not a learned preference model.
- **Contextual bandit** — single-decision episodic RL: context in, one (possibly structured)
  action out, one scalar reward, no state transitions.
- **GRPO / RLOO** — critic-free policy-gradient methods: advantage = a rollout's reward relative
  to its own sampling group (mean-normalized; RLOO uses the leave-one-out mean as baseline).
- **Expert iteration (ExIt) / rejection-sampling fine-tuning (RFT)** — sample k rollouts, keep
  the best under the real reward, supervised-fine-tune on the winners, iterate.
- **Counterfactual (per-span) advantage** — the reward delta from re-running the round trip with
  exactly one span's action flipped, all others held; exact factored-action credit.
- **Distilled reward model (RM)** — a local regressor fitted to cached (doc_p → realized recall)
  pairs; the spec §5.2 "round-trip-anchored reward upgrade".
- Everything else (floor-walk, aset, k_T, u_gold, echo/absorption, frontier_claim) as defined in
  the RL spec.

## 1. What kind of RL problem this is (and what that dictates)

One episode = one document; the policy emits a joint structured action — per-span lattice-level
choices (ranker) plus grammar-constrained fill strings (infiller, E1+) — and receives a single
scalar after the round trip. Formally:

1. **Contextual bandit, not an MDP.** No intermediate states, no transition dynamics, no
   bootstrapping target. ⇒ A learned critic/value network has no job; group-relative baselines
   (GRPO/RLOO) are the correct variance-reduction tool. This matches what we already run.
2. **Factored/combinatorial action space.** ~5–20 sub-actions per episode whose payoffs interact
   (injectivity trades, cross-span coherence, joint leakage). ⇒ A doc-level scalar dilutes credit
   over spans — the measured killer in RL-ranker v1 (rollout reward std ~0.024 spread over ~8
   decisions). The literature's answer is counterfactual per-component baselines (COMA-style),
   which our cacheable deterministic reward makes *exactly computable* rather than estimated.
3. **Expensive, black-box, but deterministic-given-action reward.** Pinned remote proxy at
   temp 0 + cache ⇒ the reward is a repeatable program over doc_p. That places us in the **RLVR
   regime** (DeepSeek-R1-style single-turn LLM RL against a computed reward), *not*
   preference-model RLHF. RLVR best practice: critic-free group PG, little or no KL (the
   environment's hard constraints carry safety), aggressive filtering of zero-signal prompts,
   large groups, and reward-hacking audits on the *program* (here: the extractor and probes).
4. **Quantized reward.** Fact recall moves in ~1/n_probes steps (~0.2 today). ⇒ The binding fix
   is *data* (more probes per doc), never reward smoothing by auxiliary terms — the surrogate
   post-mortems are the evidence.
5. **Logged-feedback goldmine.** Every round trip is cached and reusable ⇒ off-policy methods
   (ExIt/RFT, distilled-RM screening) come almost free and are the natively robust option for
   bandits.

So the closest well-studied instance is: **single-turn RLVR fine-tuning of an LM policy, with
combinatorial-bandit credit assignment underneath**. Every design choice below follows from that
classification. The 2025–26 RLVR literature's headline lesson matches our own NULLs exactly:
wins come from reward/group structure and sample selection, not from a fancier gradient
estimator.

**Nearest prior art (verified 2026-07-05):** RL-trained anonymizers exist — AgentStealth
(SFT + online RL on small local models, adversarial re-identification feedback as reward,
[arXiv 2506.22508](https://arxiv.org/abs/2506.22508)), composite-reward privacy rewriting
([arXiv 2508.19286](https://arxiv.org/abs/2508.19286)), and SEAL (adversarial distillation —
no RL — [arXiv 2506.01420](https://arxiv.org/abs/2506.01420)). All use an LLM
re-identification adversary as the privacy signal, consistent with our honesty rule. None
trains against a *round-trip task-execution* reward with hard structural privacy floors —
that combination is this project's contribution; these are comparators and design references,
not templates.

## 2. Where ample compute actually goes (ranked by measured binding constraint)

The v1/v2 NULLs were *not* optimizer starvation — the landscape probe proved the equilibria were
correct for their (reward, leash, init) triples. Ample compute must therefore buy signal and
capacity, in this order:

1. **Reward support & density → data scale-up (§4).** 23 trainable docs / 106 train probes /
   ~4.6 probes/doc is a toy environment; no algorithm rescues it. Target: **~2,000 probe-dense
   docs across ≥3 domains, 10–20 validated probes each** — reward quantization drops to
   0.05–0.1, and per-doc groups stop tying.
2. **Exact credit assignment → counterfactual round trips (§5).** The deterministic cached
   reward turns COMA's estimated counterfactual baseline into an exact measurement. This is the
   single largest algorithmic win ample compute enables.
3. **Policy capacity → doc-conditioned policies (§5, §6).** The 17-feature MLP cannot represent
   the cross-span residual that is the ranker's entire justification. Upgrade to an encoder
   policy (span menus scored in document context), autoregressive across spans so earlier
   choices condition later ones — injectivity trades become representable.
4. **Throughput → async rollout infrastructure (§7).** Remote-call latency is the wall-clock
   axis; workers + cache + off-policy tolerance, not a faster optimizer.

## 3. Architecture: separate first, joint last

**Keep the ranker/infiller decomposition; train in three stages.** A unified free-text rewriter
policy is rejected: it would (a) reopen risk certification (strict `aset_count` is exact only
inside the grammar), (b) dissolve the proposer/verifier separation the spec makes load-bearing,
and (c) open an instruction-injection channel (§8-4) that grammar-constrained decoding closes by
construction.

- **Stage A — ranker on E0** (static fills), round-trip reward. Cheap; answers the
  pre-registered headroom question ("does selection learning add anything over the floor-walk?")
  under a *healthy* reward for the first time. The pre-registered null remains a legitimate
  outcome.
- **Stage B — infiller on E1**, ranker frozen (best Stage-A checkpoint, else floor-walk). The
  spec's own null-hypothesis says learned value likely lives here: rendering quality is what the
  policy can actually do about echo-vs-absorption, and the round-trip reward — unlike every
  surrogate — prices the echo channel (handoff known-issue: absorption is now *in* the reward).
- **Stage C — joint fine-tune**, shared episodic advantage, per-component credit, PPO-clipped,
  low lr, frequent re-gates. Only if A or B shows a live policy; joint training of two dead
  components is compute theater.

Stage gating: the round-trip support scan (the handoff's mandated pre-flight) runs before every
stage's first training run, on that stage's action space.

## 4. Environment & data scale-up (the highest-leverage spend)

### 4.1 Corpora (restatement property required, per docs/specs/benchmarks.md)

Verified candidates (2026-07-05 literature pass; restatement = gold output legitimately
restates the sensitive/quasi spans):

| corpus | size | access | restatement | link |
|---|---|---|---|---|
| ACI-Bench + MTS-Dialog (in use) | 207 + 1,701 | open | yes | (registered, benchmarks spec) |
| Enron mined replies (in use) | 200 built, minable to thousands | open | yes | builder `build_task_corpora.py` |
| **Multi-LexSum** (legal summarization) | 9,280 | open (HF `allenai/multi_lexsum`) | yes — real party names/orgs/dates | [arXiv 2206.10883](https://arxiv.org/abs/2206.10883) |
| **PriMock57** (mock consultations → notes) | 57 | open (MIT) | yes — synthetic patients | [arXiv 2204.00333](https://arxiv.org/abs/2204.00333) |
| **MeetingBank** (council meetings → minutes) | ~1,366 | open | partial — named officials/motions | [arXiv 2305.17529](https://arxiv.org/abs/2305.17529) |
| **ECTSum** (earnings calls → bullets) | 2,425 | open | partial — org/financial PII | [arXiv 2210.12467](https://arxiv.org/abs/2210.12467) |
| **Discharge Me!** (MIMIC-IV discharge sections) | 68,785 train | PhysioNet DUA | yes — meds/diagnoses/dates (names pre-de-identified) | [physionet.org/content/discharge-me](https://physionet.org/content/discharge-me/1.3/) |
| **ProbSum** (progress note → problem list) | ~1k | PhysioNet DUA | yes | [arXiv 2306.05270](https://arxiv.org/abs/2306.05270) |

Adoption plan:
- **Now (no access lag):** full ACI + MTS + scaled Enron mining, plus **Multi-LexSum as the
  third domain** (open, real names, dense restatement — the best generality buy) and
  MeetingBank or ECTSum as the fourth if corpus balance needs it.
- **Later (DUA-gated bulk clinical):** Discharge Me!/ProbSum once PhysioNet credentialing
  clears — with a DUA-compliance check before any remote proxy sees a note; if the proxy can't
  be brought inside the DUA boundary, MIMIC-derived data is local-eval-only or excluded.
- **Not task corpora:** ai4privacy pii-masking / PANORAMA have no gold task output — detector
  training and memorization material only. SynthPAI, PersonalReddit ("Beyond Memorization",
  [arXiv 2310.07298](https://arxiv.org/abs/2310.07298)), LLM-PBE, and TAB stay on the
  privacy/attacker axis only.

### 4.2 Probe generation at scale, with support built in

Generate 10–20 probes per doc by **FActScore-style atomic decomposition of the gold output**
([arXiv 2305.14251](https://arxiv.org/abs/2305.14251)) — keep only atoms whose answer is a
detected quasi-identifier span — then phrase each as a question with an answerability filter
(QAFactEval lineage, [arXiv 2112.08542](https://arxiv.org/abs/2112.08542)); this replaces the
current 3-question teacher cap. Then **validate each probe with two cached round trips**:

- **Ceiling check:** the probe is answered correctly from `Remote(task_prompt(doc_orig))` —
  else the probe is unanswerable-by-task and measures nothing.
- **Floor check:** the probe is NOT answered from `Remote(task_prompt(all_placeholder(doc)))` —
  else it doesn't depend on substituted content and can never respond to any action.

Keep only probes passing both. This bakes reward support into the dataset — the exact property
whose absence produced both NULLs — and the two anchor round trips per doc are the same ones the
reward's exclusion rules already need. Report per-corpus probe coverage; the per-doc held-out
probe split and untouched held-out doc set are unchanged.

### 4.3 Environment mechanics at scale

- One new frozen arms artifact build (single detection pass → count annotation → freeze);
  determinism rules unchanged.
- **Lattice hygiene audit first** (handoff next-step 4): absurd fills ("female" → "an organism")
  now damage the *training reward* directly, not just eval. Perplexity-screen all lattice fills;
  prune or NLI-tighten. Cheap, helps every stage.
- Floors, grid, waiver rules, keep-original semantics: unchanged from spec.

## 5. Stage A — ranker (combinatorial bandit, critic-free PG + exact credit)

**Init:** behavior-clone from the floor-walk (never RL from random — kept).

**Policy:** doc-conditioned encoder (frozen small-LM encoder + per-span action-scoring head over
the legal menu), autoregressive across spans in document order so injectivity trades are
representable. The 17-feature MLP is retained as the cheap ablation arm — if the encoder policy
doesn't beat it, capacity wasn't the constraint.

**Primary optimizer — expert iteration (ReST^EM-style), not policy gradient.** Sample G
anonymizations per (doc, floor sample), keep the best under realized reward (ties broken
toward the teacher), SFT on the winners, iterate 3–5 rounds
(STaR [arXiv 2203.14465](https://arxiv.org/abs/2203.14465), ReST
[arXiv 2308.08998](https://arxiv.org/abs/2308.08998), ReST^EM
[arXiv 2312.06585](https://arxiv.org/abs/2312.06585) — measured to match or beat policy
gradient at small scale with markedly better stability). Rationale specific to us: no
advantage estimation, no leash arithmetic, robust to reward quantization (our failure mode
twice), natively off-policy (every cached round trip is reusable across iterations, amortizing
the expensive reward).

**Refiner — group-relative policy gradient, with the 2025 corrections**, layered on top of the
ExIt checkpoint once the reward is validated live:
- RLOO / GRPO with **leave-one-out mean baseline and NO per-group std division** — std
  normalization injects a spurious difficulty weighting that our quantized reward makes worse
  (Dr.GRPO, [arXiv 2503.20783](https://arxiv.org/abs/2503.20783); RLOO
  [arXiv 2402.14740](https://arxiv.org/abs/2402.14740)).
- **G = 16–32** per (doc, floor sample); **dynamic sampling / tie filtering**: groups whose
  rollouts all tie carry zero gradient — drop and refill the batch (DAPO,
  [arXiv 2503.14476](https://arxiv.org/abs/2503.14476)). With ample compute, G is the knob
  that finds live docs.
- **KL leash off at start + small entropy bonus.** RLVR practice drops KL; the v2 landscape
  showed our leash was overpricing the only gradient ~10×. Floors + grammar carry safety;
  collapse insurance = re-add a small BC anchor only on observed degeneration, plus early
  stopping on held-out-dev realized recall. (GSPO's sequence-level ratio,
  [arXiv 2507.18071](https://arxiv.org/abs/2507.18071), is the fallback if instability
  appears — largely moot for a single-decision episode.)
- **Exact per-span counterfactual credit** (COMA's baseline made exact by a deterministic
  cached reward, [arXiv 1705.08926](https://arxiv.org/abs/1705.08926)): for ~25% of
  (doc, span) pairs per batch, re-run the round trip with that one span flipped
  (→ placeholder / → teacher action), others held — a per-span advantage that bypasses
  doc-level dilution entirely. Flat broadcast of one scalar over ~8 spans is the prime
  credit-assignment suspect from v1; this kills it. Distilled-RM prioritization (§7) picks
  which spans get sweeps.
- Optimizer canaries kept: first-smoke movement milestone, G ≥ 2 NaN guard, greedy read-outs at
  fixed floors only.

Two independent optimizers double as evidence: if ExIt *and* corrected PG both land on the
floor-walk, the pre-registered null ("selection adds little") is confirmed with far more force
than one optimizer's flat line.

**Floor protocols unchanged:** fixed-floor run before floor-randomized at matched budget;
conditioning ablation at held-out floors; no cross-floor averaging; grid must include a
waiver-bearing config.

## 6. Stage B — infiller (where the value likely lives)

**Prereqs:** E1 build contract (grammar artifacts, decode constants, parser round-trip tests)
and the lattice hygiene audit.

**Init:** SFT on (lattice node → canonical-template rendering) pairs plus teacher paraphrases
filtered through the E1 verifier stack, so the starting policy is already grammar-fluent.

**Training ladder (robust first, PG second):**
1. **Best-of-n rejection-sampling FT:** n = 8–16 fill-sets per doc under the frozen ranker;
   keep winners under realized round-trip reward; SFT; iterate. For a generative component,
   this is the stable rung — most of the achievable gain usually lands here.
2. **Then GRPO** with token-level log-probs and sequence-level advantage, PPO-clipped, LoRA.
   Group = n fill-set rollouts of the same doc; same baseline/normalization rules as Stage A.
3. Model size: with ample compute, prefer a 1–3B-class decoder over flan-t5-base — fill fluency
   is plausibly the echo lever, and rendering is the one place capacity buys realized utility.
   (Spec names flan-t5-base as "planned", not pinned; this is a proposed build-time change,
   recorded here.)

**Environment, never reward:** grammar mask, injectivity, strict online `aset_count`, NLI gate —
the proposer/verifier separation is absolute and is also the injection firewall (§8-4).

## 7. Throughput engineering (the wall-clock axis)

- **Async rollout workers** (16–32 threads against the proxy), reward computed in a pipelined
  stage; the trainer tolerates one step of staleness (or use ExIt, which doesn't care).
- **Cache = first-class infrastructure:** key `hash(model, template, doc_p)`; near convergence
  the hit rate climbs and marginal epochs get cheap. The cache is also the RM training set and
  the ExIt candidate pool.
- **Distilled RM (spec §5.2 upgrade), used the safe way:** fit a local regressor on cached
  (doc_p → realized recall); use it ONLY to pre-screen ExIt/best-of-n candidates and to
  prioritize counterfactual sweeps. **Gradient-carrying rewards remain real round trips.** This
  keeps the honesty story trivial (no surrogate construct validity to re-litigate) while
  recovering most of the wall-time the surrogate era was chasing. If RM-in-the-loop training is
  ever wanted, it requires: on-policy refresh every iteration + overoptimization monitoring
  against the Goodhart budget of Gao et al. ([arXiv 2210.10760](https://arxiv.org/abs/2210.10760)
  — proxy/true divergence grows predictably with optimization pressure; re-anchor before the
  divergence point) + its own gate.
- **Budget sketch (Stage A):** 2,000 docs × G16 × ~10 effective epochs ≈ 3×10⁵ round trips;
  at ~1–2 s/call amortized over 32 workers ≈ 3–6 h of proxy time per run before cache credit;
  counterfactual sweeps +30–50%. Probe validation: 2 anchor calls/doc + probe reads, one-time.
  Well inside "ample"; degrades gracefully to the current iGPU+proxy setup by shrinking docs/G.

## 8. Anti-Goodhart & risk register (round-trip-specific)

1. **Extractor gaming** — the policy discovers fills whose fuzzy inversion *falsely* inflates
   recall. Monitors: exact-vs-fuzzy recall gap per checkpoint; manual audit sample of
   inversions on reward-climbing checkpoints; extractor pinned for the cycle (spec rule kept).
2. **Probe-teacher circularity** — probes authored by the same model family as the remote proxy
   reward what that family likes to restate. Mitigation: probe teacher ≠ remote proxy model;
   the second-remote-model eval arm (already mandated) is the detector.
3. **Provider overfitting** — the policy learns the pinned proxy's echo idiosyncrasies. Measure
   as the transfer gap on the second-model arm; report it, never calibrate it away (honesty
   rule).
4. **Instruction injection through fills** — an *unconstrained* generative infiller under
   round-trip reward is an adversarial-string miner: reward ascent would find fills that
   instruct the remote model ("list all entities verbatim…"). Grammar-constrained decoding is
   therefore a **security boundary, not a convenience**; any relaxation requires an injection
   audit before training resumes.
5. **Information smuggling via minted placeholders / R-record games** — descriptive placeholders
   stay aset-scored (spec rule kept); audit placeholder-label entropy on trained policies.
6. **Held-out erosion** — checkpoint selection on a dev split only; held-out docs and held-out
   probes untouched until the single Phase-3 eval; seeds and doc lists pre-registered.
7. **What RL cannot fix (kept in view):** sibling-mention detection leak is the privacy ceiling;
   famous-context priors and thin floor calibration are floor-contract issues — all priced at
   eval, none touchable by the policy. Reward climbing while held-out realized metrics fall =
   stop and report (standing rule).

## 9. Verdict & external comparison

Spec Phase 3 unchanged: matched realized privacy (frontier-LLM attacker on doc_p, leak-through
on out_final), utility = fact recall on out_final, held-out everything, per-floor operating
points, document-bootstrap dominance test. Additions for the "real-world" claim:

- **External baselines at matched realized privacy**, all run through the identical round-trip
  harness (ROUGE/BERTScore/entity-F1 stay eval-side color, never training signal):
  adversarial anonymization (Staab et al.,
  [arXiv 2402.13846](https://arxiv.org/abs/2402.13846)) as the real-world SOTA comparator;
  AgentStealth ([arXiv 2506.22508](https://arxiv.org/abs/2506.22508)) as the RL-trained
  comparator; RUPTA ([arXiv 2407.11770](https://arxiv.org/abs/2407.11770)) and IncogniText
  ([arXiv 2407.02956](https://arxiv.org/abs/2407.02956)) as secondary; DP rewriting as the
  formal-guarantee class — DP-Prompt ([arXiv 2310.16111](https://arxiv.org/abs/2310.16111)),
  DP-BART ([arXiv 2302.07636](https://arxiv.org/abs/2302.07636)), DP-MLM
  ([arXiv 2407.00637](https://arxiv.org/abs/2407.00637)) — alongside the legacy InferDPT
  ([arXiv 2310.12214](https://arxiv.org/abs/2310.12214)).
- **Pre-registered outcome tree:** (a) trained ≻ floor-walk in ≥1 privacy bin, no losses →
  frontier claim; (b) ranker null + infiller positive → "value lives in rendering" finding;
  (c) all null under a healthy gated reward with two optimizers → the floor-walk + grammar
  infiller IS the product; the RL chapter closes with evidence, not embarrassment.

## 10. Tooling

- **Ranker (Stage A):** extend `scripts/train_ranker.py` — custom loop is appropriate at this
  policy size; add RLOO baseline, tie filtering, async reward client + cache, counterfactual
  scheduler. No framework needed.
- **Infiller (Stage B/C):** TRL's GRPO trainer — accepts custom (sync or async) python reward
  functions, so the round trip drops in; LoRA + <1B models fully supported. Its training step
  is synchronous (no generation/training overlap), acceptable because reward latency dominates
  anyway. **Trinity-RFT** ([arXiv 2505.17826](https://arxiv.org/abs/2505.17826)) is the
  integrated alternative if the ExIt + replay + RM-refresh combination outgrows glue code —
  it ships RFT, replay, and reward-model-in-the-loop as first-class modes. verl / OpenRLHF
  ([arXiv 2405.11143](https://arxiv.org/abs/2405.11143)) only if multi-node scale is actually
  exercised. Constrained decoding via logits-processor grammar masks (the E1 artifacts define
  the automaton). ExGRPO-style correctness/entropy-bucketed replay
  ([arXiv 2510.02245](https://arxiv.org/abs/2510.02245)) is a lead for stabilizing the small
  policy if off-policy GRPO is attempted.
- Proxy client: existing `inferdpt.llm` + thread-pool workers; unbuffered logging; one-GPU rule
  applies to local models only (the reward is remote-proxy-bound).

## Sources

Verified 2026-07-05 (two web-research passes; every ID checked against its abstract page).
Papers to register in `research-wiki/papers/` if this plan graduates into a `docs/research/`
report.

**Algorithms / RLVR practice:** Dr.GRPO — Understanding R1-Zero-Like Training
([arXiv 2503.20783](https://arxiv.org/abs/2503.20783)) · DAPO
([arXiv 2503.14476](https://arxiv.org/abs/2503.14476)) · RLOO / Back to Basics
([arXiv 2402.14740](https://arxiv.org/abs/2402.14740)) · REINFORCE++
([arXiv 2501.03262](https://arxiv.org/abs/2501.03262)) · GSPO
([arXiv 2507.18071](https://arxiv.org/abs/2507.18071)) · STaR
([arXiv 2203.14465](https://arxiv.org/abs/2203.14465)) · ReST
([arXiv 2308.08998](https://arxiv.org/abs/2308.08998)) · ReST^EM / Beyond Human Data
([arXiv 2312.06585](https://arxiv.org/abs/2312.06585)) · COMA
([arXiv 1705.08926](https://arxiv.org/abs/1705.08926)) · RM overoptimization scaling laws
([arXiv 2210.10760](https://arxiv.org/abs/2210.10760)) · ExGRPO
([arXiv 2510.02245](https://arxiv.org/abs/2510.02245)).

**Anonymization prior art / baselines:** Adversarial Anonymization
([arXiv 2402.13846](https://arxiv.org/abs/2402.13846)) · AgentStealth
([arXiv 2506.22508](https://arxiv.org/abs/2506.22508)) · privacy-preserving RL rewriting
([arXiv 2508.19286](https://arxiv.org/abs/2508.19286)) · SEAL
([arXiv 2506.01420](https://arxiv.org/abs/2506.01420)) · RUPTA
([arXiv 2407.11770](https://arxiv.org/abs/2407.11770)) · IncogniText
([arXiv 2407.02956](https://arxiv.org/abs/2407.02956)) · DP-Prompt
([arXiv 2310.16111](https://arxiv.org/abs/2310.16111)) · DP-BART
([arXiv 2302.07636](https://arxiv.org/abs/2302.07636)) · DP-MLM
([arXiv 2407.00637](https://arxiv.org/abs/2407.00637)) · InferDPT
([arXiv 2310.12214](https://arxiv.org/abs/2310.12214)) · self-disclosure abstraction
([arXiv 2311.09538](https://arxiv.org/abs/2311.09538)).

**Attack/eval:** Beyond Memorization ([arXiv 2310.07298](https://arxiv.org/abs/2310.07298)) ·
SynthPAI ([arXiv 2406.07217](https://arxiv.org/abs/2406.07217)) · LLM-PBE
([arXiv 2408.12787](https://arxiv.org/abs/2408.12787)) · TAB
([arXiv 2202.00443](https://arxiv.org/abs/2202.00443)).

**Datasets:** §4.1 table · **Probe generation:** FActScore
([arXiv 2305.14251](https://arxiv.org/abs/2305.14251)) · QAFactEval
([arXiv 2112.08542](https://arxiv.org/abs/2112.08542)) · QuestEval
([arXiv 2103.12693](https://arxiv.org/abs/2103.12693)).

**Tooling:** OpenRLHF ([arXiv 2405.11143](https://arxiv.org/abs/2405.11143)) · Trinity-RFT
([arXiv 2505.17826](https://arxiv.org/abs/2505.17826)).
